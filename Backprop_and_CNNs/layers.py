import numpy as np

def affine_forward(x, w, b):
    """
    Computes the forward pass for an affine (fully-connected) layer.

    The input x has shape (N, d_1, ..., d_k) and contains a minibatch of N
    examples, where each example x[i] has shape (d_1, ..., d_k). We will
    reshape each input into a vector of dimension D = d_1 * ... * d_k, and
    then transform it to an output vector of dimension M.

    Inputs:
    - x: A numpy array containing input data, of shape (N, d_1, ..., d_k)
    - w: A numpy array of weights, of shape (D, M)
    - b: A numpy array of biases, of shape (M,)

    Returns a tuple of:
    - out: output, of shape (N, M)
    - cache: (x, w, b)
    """
    out = None
    #############################################################################
    # TODO: Implement the affine forward pass. Store the result in out. You     #
    # will need to reshape the input into rows.                                 #
    #############################################################################
    data = np.reshape(x, (len(x[:,0]),np.prod(x.shape[1:])))
    out = np.dot(data,w) + b
        
        
   # pass
    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################
    cache = (x, w, b)
    return out, cache

#%%
def affine_backward(dout, cache):
    """
    Computes the backward pass for an affine layer.

    Inputs:
    - dout: Upstream derivative, of shape (N, M)
    - cache: Tuple of:
      - x: Input data, of shape (N, d_1, ... d_k)
      - w: Weights, of shape (D, M)

    Returns a tuple of:
    - dx: Gradient with respect to x, of shape (N, d1, ..., d_k)
    - dw: Gradient with respect to w, of shape (D, M)
    - db: Gradient with respect to b, of shape (M,)
    """
    x, w, b = cache
    dx, dw, db = None, None, None
    #############################################################################
    # TODO: Implement the affine backward pass.                                 #
    #############################################################################
    dx = np.dot(dout, w.T).reshape(x.shape)
    dw = np.dot(x.reshape(x.shape[0], -1).T, dout)
    db = np.sum(dout, axis=0)
    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################
    return dx, dw, db


def relu_forward(x):
    """
    Computes the forward pass for a layer of rectified linear units (ReLUs).

    Input:
    - x: Inputs, of any shape

    Returns a tuple of:
    - out: Output, of the same shape as x
    - cache: x
    """
    out = None
    #############################################################################
    # TODO: Implement the ReLU forward pass.                                    #
    #############################################################################
    out = np.array([i if i > 0 else 0 for i in x.flatten()])
    out = out.reshape(x.shape)
    
 
    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################
    cache = x
    return out, cache


def relu_backward(dout, cache):
    """
    Computes the backward pass for a layer of rectified linear units (ReLUs).

    Input:
    - dout: Upstream derivatives, of any shape
    - cache: Input x, of same shape as dout

    Returns:
    - dx: Gradient with respect to x
    """
    dx, x = None, cache
    #############################################################################
    # TODO: Implement the ReLU backward pass.                                   #
    #############################################################################
    newx = np.array([1 if i > 0 else 0 for i in x.flatten()]).reshape(x.shape)
    dx = dout*newx
    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################
    return dx


def batchnorm_forward(x, gamma, beta, bn_param):
    """
    Forward pass for batch normalization.

    During training the sample mean and (uncorrected) sample variance are
    computed from minibatch statistics and used to normalize the incoming data.
    During training we also keep an exponentially decaying running mean of the mean
    and variance of each feature, and these averages are used to normalize data
    at test-time.

    At each timestep we update the running averages for mean and variance using
    an exponential decay based on the momentum parameter:

    running_mean = momentum * running_mean + (1 - momentum) * sample_mean
    running_var = momentum * running_var + (1 - momentum) * sample_var

    Note that the batch normalization paper suggests a different test-time
    behavior: they compute sample mean and variance for each feature using a
    large number of training images rather than using a running average. For
    this implementation we have chosen to use running averages instead since
    they do not require an additional estimation step; the torch7 implementation
    of batch normalization also uses running averages.

    Input:
    - x: Data of shape (N, D)
    - gamma: Scale parameter of shape (D,)
    - beta: Shift paremeter of shape (D,)
    - bn_param: Dictionary with the following keys:
      - mode: 'train' or 'test'; required
      - eps: Constant for numeric stability
      - momentum: Constant for running mean / variance.
      - running_mean: Array of shape (D,) giving running mean of features
      - running_var Array of shape (D,) giving running variance of features

    Returns a tuple of:
    - out: of shape (N, D)
    - cache: A tuple of values needed in the backward pass
    """
    mode = bn_param['mode']
    eps = bn_param.get('eps', 1e-5)
    momentum = bn_param.get('momentum', 0.9)

    N, D = x.shape
    running_mean = bn_param.get('running_mean', np.zeros(D, dtype=x.dtype))
    running_var = bn_param.get('running_var', np.zeros(D, dtype=x.dtype))

    out, cache = None, None
    if mode == 'train':
        #############################################################################
        # TODO: Implement the training-time forward pass for batch normalization.   #
        # Use minibatch statistics to compute the mean and variance, use these      #
        # statistics to normalize the incoming data, and scale and shift the        #
        # normalized data using gamma and beta.                                     #
        #                                                                           #
        # You should store the output in the variable out. Any intermediates that   #
        # you need for the backward pass should be stored in the cache variable.    #
        #                                                                           #
        # You should also use your computed sample mean and variance together with  #
        # the momentum variable to update the running mean and running variance,    #
        # storing your result in the running_mean and running_var variables.        #
        #############################################################################
        sample_mean = x.mean(axis=0)
        sample_var = x.var(axis=0)
        
        
        std = np.sqrt(sample_var+eps)
        x_adj = (x - sample_mean) / std
        x_minus_mu = (x - sample_mean)
        out = x_adj* gamma + beta
        
        running_mean = momentum * running_mean + (1 - momentum) * sample_mean
        running_var = momentum * running_var + (1 - momentum) * sample_var
        
        cache = (gamma, x_adj, x_minus_mu, std)
        #############################################################################
        #                             END OF YOUR CODE                              #
        #############################################################################
    elif mode == 'test':
        #############################################################################
        # TODO: Implement the test-time forward pass for batch normalization. Use   #
        # the running mean and variance to normalize the incoming data, then scale  #
        # and shift the normalized data using gamma and beta. Store the result in   #
        # the out variable.                                                         #
        #############################################################################
        std = (running_var+eps)**0.5
        x_adj = (x - running_mean) / std
        x_minus_mu = (x - running_mean)
        out = x_adj* gamma + beta
    #    cache_mean = running_mean
    #    cache_var = running_var
        #############################################################################
        #                             END OF YOUR CODE                              #
        #############################################################################
    else:
        raise ValueError('Invalid forward batchnorm mode "%s"' % mode)

    # Store the updated running means back into bn_param
    bn_param['running_mean'] = running_mean
    bn_param['running_var'] = running_var
    

    return out, cache


def batchnorm_backward(dout, cache):
    """
    Backward pass for batch normalization.

    For this implementation, you should write out a computation graph for
    batch normalization on paper and propagate gradients backward through
    intermediate nodes.

    Inputs:
    - dout: Upstream derivatives, of shape (N, D)
    - cache: Variable of intermediates from batchnorm_forward.

    Returns a tuple of:
    - dx: Gradient with respect to inputs x, of shape (N, D)
    - dgamma: Gradient with respect to scale parameter gamma, of shape (D,)
    - dbeta: Gradient with respect to shift parameter beta, of shape (D,)
    """
    dx, dgamma, dbeta = None, None, None
    gamma, x_adj, x_minus_mu, std = cache
    #print(std)
    #############################################################################
    # TODO: Implement the backward pass for batch normalization. Store the      #
    # results in the dx, dgamma, and dbeta variables.                           #
    #############################################################################
    n, d = (dout.shape)
    N = dout.shape[0]
    n_ones = np.ones(dout.shape)
    
    # d_xadj = (dout*gamma)
    # d_invvar = np.sum(d_xadj*x_minus_mu, axis=0)
    # dx_mu1 = d_xadj*1/np.sqrt(cache_var+epsilon)
    
    # d_sqrt_var = (-1/(cache_var + epsilon))*d_invvar
    
    # d_var = 0.5 * 1/(np.sqrt(cache_var + epsilon))*d_sqrt_var
    
    # d_sq = (1/n)*n_ones*d_var
    # dx_mu2 = 2*x_minus_mu*d_sq
      
    # dx1 = dx_mu1 + dx_mu2 
    # dmu = -1* np.sum(dx_mu1 + dx_mu2, axis=0)
    # dx2 = (1/n)*n_ones*dmu
    # dx = dx1 + dx2
    
 
    
    dgamma = np.sum(dout*x_adj, axis=0) 
    #gamma_deriv = np.ones(gamma.shape)
    #dgamma = np.dot(dout.T, np.dot(((x - cache_mean)/(cache_var**0.5 + epsilon)), gamma_deriv))
    dbeta = np.sum(dout, axis=0)
    #dx = (1/n)*gamma*(1/(np.sqrt(cache_var + epsilon)))*( (-dgamma*x_adj) + (n*dout) - (n_ones*dbeta)  )
    
    
    # dx_norm = dout * gamma
    # dx_centered = dx_norm / np.sqrt(cache_var+epsilon)
    # dmean = -(dx_centered.sum(axis=0) + 2/n * x_minus_mu.sum(axis=0))
    # dstd = (dx_norm * x_minus_mu * -(np.sqrt(cache_var+epsilon))**(-2)).sum(axis=0)
    # dvar = dstd / 2 / np.sqrt(cache_var+epsilon)
    # dx = dx_centered + (dmean + dvar * 2 * x_minus_mu) / n
    
    
    dx_norm = dout * gamma
    dx_centered = dx_norm / std
    dmean = -(dx_centered.sum(axis=0) + 2/N * x_minus_mu.sum(axis=0))
    dstd = (dx_norm * x_minus_mu * (-std**(-2))).sum(axis=0)
    dvar = dstd / 2 / std
    dx_final = dx_centered + (dmean + dvar * 2 * x_minus_mu) / N

    N = dout.shape[0]
    gamma, x_norm, x_centered, std = cache
    
    dgamma = (dout * x_norm).sum(axis=0)
    dbeta = dout.sum(axis=0)
    
    dx_norm = dout * gamma
    dx = 1/N / std * (N * dx_norm - 
                      dx_norm.sum(axis=0) - 
                      x_norm * (dx_norm * x_norm).sum(axis=0))    

    
    
    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################

    return dx, dgamma, dbeta


def batchnorm_backward_alt(dout, cache):
    """
    Alternative backward pass for batch normalization.

    For this implementation you should work out the derivatives for the batch
    normalizaton backward pass on paper and simplify as much as possible. You
    should be able to derive a simple expression for the backward pass.

    Note: This implementation should expect to receive the same cache variable
    as batchnorm_backward, but might not use all of the values in the cache.

    Inputs / outputs: Same as batchnorm_backward
    """
    dx, dgamma, dbeta = None, None, None
    #############################################################################
    # TODO: Implement the backward pass for batch normalization. Store the      #
    # results in the dx, dgamma, and dbeta variables.                           #
    #                                                                           #
    # After computing the gradient with respect to the centered inputs, you     #
    # should be able to compute gradients with respect to the inputs in a       #
    # single statement; our implementation fits on a single 80-character line.  #
    #############################################################################
 
    N = dout.shape[0]
    gamma, x_norm, x_centered, std = cache
    
    dgamma = (dout * x_norm).sum(axis=0)
    dbeta = dout.sum(axis=0)
    
    dx_norm = dout * gamma
    dx = 1/N / std * (N * dx_norm - 
                      dx_norm.sum(axis=0) - 
                      x_norm * (dx_norm * x_norm).sum(axis=0))    

    
    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################

    return dx, dgamma, dbeta


def dropout_forward(x, dropout_param):
    """
    Performs the forward pass for (inverted) dropout.

    Inputs:
    - x: Input data, of any shape
    - dropout_param: A dictionary with the following keys:
      - p: Dropout parameter. We drop each neuron output with probability p.
      - mode: 'test' or 'train'. If the mode is train, then perform dropout
        and rescale the outputs to have the same mean as at test time.
        if the mode is test, then just return the input.
      - seed: Seed for the random number generator. Passing seed makes this
        function deterministic, which is needed for gradient checking but not in
        real networks.

    Outputs:
    - out: Array of the same shape as x.
    - cache: A tuple (dropout_param, mask). In training mode, mask is the dropout
      mask that was used to multiply the input; in test mode, mask is None.
    """
    p, mode = dropout_param['p'], dropout_param['mode']
    if 'seed' in dropout_param:
        np.random.seed(dropout_param['seed'])

    mask = None
    out = None

    if mode == 'train':
        ###########################################################################
        # TODO: Implement the training phase forward pass for inverted dropout.   #
        # Store the dropout mask in the mask variable.                            #
        ###########################################################################
        mask = np.random.choice([0,1], size=x.shape, p=[p, (1-p)])
        out = (x * mask)/(1-p)
        #mask = mask * (np.mean(x)/np.mean(out))
       # out = out*(np.mean(x)/np.mean(out))
        
        ###########################################################################
        #                            END OF YOUR CODE                             #
        ###########################################################################
    elif mode == 'test':
        out = x

    cache = (dropout_param, mask)
    out = out.astype(x.dtype, copy=False)

    return out, cache


def dropout_backward(dout, cache):
    """
    Perform the backward pass for (inverted) dropout.

    Inputs:
    - dout: Upstream derivatives, of any shape
    - cache: (dropout_param, mask) from dropout_forward.
    """
    dropout_param, mask = cache
    mode = dropout_param['mode']

    dx = None
    if mode == 'train':
        ###########################################################################
        # TODO: Implement the training phase backward pass for inverted dropout.  #
        ###########################################################################
        dx = (mask *  dout) / ((1-dropout_param['p']))
        ###########################################################################
        #                            END OF YOUR CODE                             #
        ###########################################################################
    elif mode == 'test':
        dx = dout
    return dx


def conv_forward_naive(x, w, b, conv_param):
    """
    A naive implementation of the forward pass for a convolutional layer.

    The input consists of N data points, each with C channels, height H and width
    W. We convolve each input with F different filters, where each filter spans
    all C channels and has height HH and width HH.

    Input:
    - x: Input data of shape (N, C, H, W)
    - w: Filter weights of shape (F, C, HH, WW)
    - b: Biases, of shape (F,)
    - conv_param: A dictionary with the following keys:
      - 'stride': The number of pixels between adjacent receptive fields in the
        horizontal and vertical directions.
      - 'pad': The number of pixels that will be used to zero-pad the input.

    Returns a tuple of:
    - out: Output data, of shape (N, F, H', W') where H' and W' are given by
      H' = 1 + (H + 2 * pad - HH) / stride
      W' = 1 + (W + 2 * pad - WW) / stride
    - cache: (x, w, b, conv_param)
    """
    out = None
    #############################################################################
    # TODO: Implement the convolutional forward pass.                           #
    # Hint: you can use the function np.pad for padding.                        #
    #############################################################################
    
    
    N, C, H, W = x.shape
    F, C, HH, WW = w.shape
    
    H_prime = int(1 + (H + (2*conv_param['pad']) - HH)/conv_param['stride'])
    W_prime = int(1 + (W + (2*conv_param['pad']) - WW)/conv_param['stride'])
    

    out = np.zeros((N, F, H_prime, W_prime))
    

    pad = conv_param['pad']
    x_pad = np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)), 'constant', constant_values=0)
    
    #print(x_pad)
    
    for this_image in np.arange(N):
        
        image = x_pad[this_image, :, :, :]
  
        for i in np.arange(H_prime):     #move down rows of output matrix
            for j in np.arange(W_prime): #move across columns of output matrix
                start_row = i * conv_param['stride']
                end_row = start_row + HH
                start_column = j * conv_param['stride']
                end_column = start_column + WW

                for filternum in np.arange(F):

                    temp = np.multiply( w[filternum, :, :, :], image[:, start_row:end_row, start_column:end_column])
                    
                    z = np.sum(temp)
                    
                    out[this_image, filternum, int(i), int(j)] = z + b[filternum]
                    

    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################
    cache = (x, w, b, conv_param)

    return out, cache


def conv_backward_naive(dout, cache):
    """
    A naive implementation of the backward pass for a convolutional layer.

    Inputs:
    - dout: Upstream derivatives.
    - cache: A tuple of (x, w, b, conv_param) as in conv_forward_naive

    Returns a tuple of:
    - dx: Gradient with respect to x
    - dw: Gradient with respect to w
    - db: Gradient with respect to b
    """
    dx, dw, db = None, None, None
    #############################################################################
    # TODO: Implement the convolutional backward pass.                          #
    #############################################################################
    
    x, w, b, conv_param = cache #recover variables from forward pass
    pad = conv_param['pad'] #padding from fp
    stride = conv_param['stride'] #stride from fp
    
    x_pad = np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)), 'constant', constant_values=0)

    dx = np.zeros_like(x_pad)
    dw = np.zeros_like(w)
    db = np.zeros_like(b)
    

    N, C, H, W = x.shape
    F, C, HH, WW = w.shape
    
    # shape of dout = shape of out =  (N, F, H_prime, W_prime)
    N, F, H_prime, W_prime = dout.shape

    #db just sums across all other axes for each filter in F
    db = np.sum(dout, axis=(0, 2, 3))
    
    #for dw, take convolution between original input x and dout 
    for this_image in np.arange(N):
        for filternum in np.arange(F):
            for i in np.arange(H_prime):     #move down rows of output matrix
                for j in np.arange(W_prime): #move across columns of output matrix
                    start_row = i * conv_param['stride']
                    end_row = start_row + HH
                    start_column = j * conv_param['stride']
                    end_column = start_column + WW
                    
                    dw[filternum, :, :, :] += dout[this_image, filternum, int(i), int(j)] * x_pad[this_image, :, start_row:end_row, start_column:end_column]                    
    
    #dx 
    for this_image in np.arange(N):
        for filternum in np.arange(F):
            for i in np.arange(H_prime):     #move down rows of output matrix
                for j in np.arange(W_prime): #move across columns of output matrix
                    start_row = i * conv_param['stride']
                    end_row = start_row + HH
                    start_column = j * conv_param['stride']
                    end_column = start_column + WW
                    
                    #print(dout[this_image, filternum, int(i), int(j)])
                    
                   # print(w[filternum, :, :, :])
                    dx[this_image, :, start_row:end_row, start_column:end_column] += dout[this_image, filternum, int(i), int(j)] * w[filternum, :, :, :]

    
    dx = dx[:, :, pad:-pad, pad:-pad]
    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################
    return dx, dw, db


def max_pool_forward_naive(x, pool_param):
    """
    A naive implementation of the forward pass for a max pooling layer.

    Inputs:
    - x: Input data, of shape (N, C, H, W)
    - pool_param: dictionary with the following keys:
      - 'pool_height': The height of each pooling region
      - 'pool_width': The width of each pooling region
      - 'stride': The distance between adjacent pooling regions

    Returns a tuple of:
    - out: Output data
    - cache: (x, pool_param)
    """
    out = None
    #############################################################################
    # TODO: Implement the max pooling forward pass                              #
    #############################################################################
    PH = pool_param['pool_height']
    PW = pool_param['pool_width']
    stride = pool_param['stride']
    
    N, C, H, W = x.shape
    
    out_H = int(np.trunc((H-PH)/stride)+1)
    out_W = int(np.trunc((W-PW)/stride)+1)
    
    out = np.zeros((N, C, out_H, out_W))
    

    
    for image in np.arange(N):
        for row in np.arange(0, out_H):
            for column in np.arange(0, out_W):
                
                for c in np.arange(C):
                    
                    start_row = row * stride
                    start_column = column * stride
                    end_row = start_row + PH 
                    end_column = start_column  + PW  
                    
                    x_slice = x[image, c, start_row:end_row, start_column:end_column]
                   # print(image)
                   # print(c)
                   # print(row)
                   # print(column)
                   # print(start_column)
                   # print(x_slice)                    
                    out[image, c, row, column] = x_slice.max()
                

    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################
    cache = (x, pool_param)
    return out, cache


def max_pool_backward_naive(dout, cache):
    """
    A naive implementation of the backward pass for a max pooling layer.

    Inputs:
    - dout: Upstream derivatives
    - cache: A tuple of (x, pool_param) as in the forward pass.

    Returns:
    - dx: Gradient with respect to x
    """
    dx = None
    #############################################################################
    # TODO: Implement the max pooling backward pass                             #
    #############################################################################
    x, pool_param = cache
    
    N, C, out_H, out_W = dout.shape
    N, C, H, W   = x.shape
    
    dx = np.zeros_like(x)
    
    PH = pool_param['pool_height']
    PW = pool_param['pool_width']
    stride = pool_param['stride']
    
    for image in np.arange(N):
        for i in np.arange(out_H):
            for j in np.arange(out_W):
                for c in np.arange(C):
                    
                    start_row = i * stride
                    start_column = j * stride
                    end_row = start_row + PH 
                    end_column = start_column  + PW  
                    
                    x_slice = x[image, c, start_row:end_row, start_column:end_column]
                    mask = (x_slice == np.max(x_slice))
                    dx[image, c, start_row:end_row, start_column:end_column] += mask * dout[image, c, i, j]
    
    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################
    return dx


def spatial_batchnorm_forward(x, gamma, beta, bn_param):
    """
    Computes the forward pass for spatial batch normalization.

    Inputs:
    - x: Input data of shape (N, C, H, W)
    - gamma: Scale parameter, of shape (C,)
    - beta: Shift parameter, of shape (C,)
    - bn_param: Dictionary with the following keys:
      - mode: 'train' or 'test'; required
      - eps: Constant for numeric stability
      - momentum: Constant for running mean / variance. momentum=0 means that
        old information is discarded completely at every time step, while
        momentum=1 means that new information is never incorporated. The
        default of momentum=0.9 should work well in most situations.
      - running_mean: Array of shape (D,) giving running mean of features
      - running_var Array of shape (D,) giving running variance of features

    Returns a tuple of:
    - out: Output data, of shape (N, C, H, W)
    - cache: Values needed for the backward pass
    """
    out, cache = None, None

    #############################################################################
    # TODO: Implement the forward pass for spatial batch normalization.         #
    #                                                                           #
    # HINT: You can implement spatial batch normalization using the vanilla     #
    # version of batch normalization defined above. Your implementation should  #
    # be very short; ours is less than five lines.                              #
    #############################################################################
    N, C, H, W = x.shape
    new_x = x.transpose((0, 2, 3, 1)) 
    # now N, H, W, C = 0, 1, 2, 3
    
    #new_x = new_x.reshape((A, B, D))
    
    flattened_x = new_x.reshape(N*H*W, C)

    out, cache = batchnorm_forward(flattened_x, gamma, beta, bn_param)
    #print(out.shape)
    
    #out is N, H, W, C 
    
    out = out.reshape(N,H,W,C)
        
    out = out.transpose((0, 3, 1, 2)) 
    
    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################

    return out, cache


def spatial_batchnorm_backward(dout, cache):
    """
    Computes the backward pass for spatial batch normalization.

    Inputs:
    - dout: Upstream derivatives, of shape (N, C, H, W)
    - cache: Values from the forward pass

    Returns a tuple of:
    - dx: Gradient with respect to inputs, of shape (N, C, H, W)
    - dgamma: Gradient with respect to scale parameter, of shape (C,)
    - dbeta: Gradient with respect to shift parameter, of shape (C,)
    """
    dx, dgamma, dbeta = None, None, None

    #############################################################################
    # TODO: Implement the backward pass for spatial batch normalization.        #
    #                                                                           #
    # HINT: You can implement spatial batch normalization using the vanilla     #
    # version of batch normalization defined above. Your implementation should  #
    # be very short; ours is less than five lines.                              #
    #############################################################################
    N, C, H, W = dout.shape
    
    new_dout = dout.transpose((0, 2, 3, 1)) 
    
     # now N, H, W, C = 0, 1, 2, 3
    
    flattened_dout = new_dout.reshape(N*H*W, C)
    
    dx, dgamma, dbeta = batchnorm_backward(flattened_dout, cache)
    
    dx = dx.reshape(N,H,W,C)
        
    dx = dx.transpose((0, 3, 1, 2)) 
    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################

    return dx, dgamma, dbeta


def svm_loss(x, y):
    """
    Computes the loss and gradient using for multiclass SVM classification.

    Inputs:
    - x: Input data, of shape (N, C) where x[i, j] is the score for the jth class
      for the ith input.
    - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
      0 <= y[i] < C

    Returns a tuple of:
    - loss: Scalar giving the loss
    - dx: Gradient of the loss with respect to x
    """
    N = x.shape[0]
    correct_class_scores = x[np.arange(N), y]
    margins = np.maximum(0, x - correct_class_scores[:, np.newaxis] + 1.0)
    margins[np.arange(N), y] = 0
    loss = np.sum(margins) / N
    num_pos = np.sum(margins > 0, axis=1)
    dx = np.zeros_like(x)
    dx[margins > 0] = 1
    dx[np.arange(N), y] -= num_pos
    dx /= N
    return loss, dx


def softmax_loss(x, y):
    """
    Computes the loss and gradient for softmax classification.

    Inputs:
    - x: Input data, of shape (N, C) where x[i, j] is the score for the jth class
      for the ith input.
    - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
      0 <= y[i] < C

    Returns a tuple of:
    - loss: Scalar giving the loss
    - dx: Gradient of the loss with respect to x
    """
    probs = np.exp(x - np.max(x, axis=1, keepdims=True))
    probs /= np.sum(probs, axis=1, keepdims=True)
    N = x.shape[0]
    loss = -np.sum(np.log(probs[np.arange(N), y])) / N
    dx = probs.copy()
    dx[np.arange(N), y] -= 1
    dx /= N
    return loss, dx
