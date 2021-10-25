import numpy as np


"""
This file defines layer types that are commonly used for recurrent neural
networks.
"""


def rnn_step_forward(x, prev_h, Wx, Wh, b):
    """
    Run the forward pass for a single timestep of a vanilla RNN that uses a tanh
    activation function.

    The input data has dimension D, the hidden state has dimension H, and we use
    a minibatch size of N.

    Inputs:
    - x: Input data for this timestep, of shape (N, D).
    - prev_h: Hidden state from previous timestep, of shape (N, H)
    - Wx: Weight matrix for input-to-hidden connections, of shape (D, H)
    - Wh: Weight matrix for hidden-to-hidden connections, of shape (H, H)
    - b: Biases of shape (H,)

    Returns a tuple of:
    - next_h: Next hidden state, of shape (N, H)
    - cache: Tuple of values needed for the backward pass.
    """
    next_h, cache = None, None
    ##############################################################################
    # TODO: Implement a single forward step for the vanilla RNN. Store the next  #
    # hidden state and any values you need for the backward pass in the next_h   #
    # and cache variables respectively.                                          #
    ##############################################################################
  
 
    next_h = np.tanh(np.dot(x, Wx) + np.dot(prev_h, Wh) + b)
    
    
    cache = x, prev_h, next_h, Wx, Wh, b
    
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################
    return next_h, cache


def rnn_step_backward(dnext_h, cache):
    """
    Backward pass for a single timestep of a vanilla RNN.

    Inputs:
    - dnext_h: Gradient of loss with respect to next hidden state
    - cache: Cache object from the forward pass

    Returns a tuple of:
    - dx: Gradients of input data, of shape (N, D)
    - dprev_h: Gradients of previous hidden state, of shape (N, H)
    - dWx: Gradients of input-to-hidden weights, of shape (D, H)
    - dWh: Gradients of hidden-to-hidden weights, of shape (H, H)
    - db: Gradients of bias vector, of shape (H,)
    """
    dx, dprev_h, dWx, dWh, db = None, None, None, None, None
    ##############################################################################
    # TODO: Implement the backward pass for a single step of a vanilla RNN.      #
    #                                                                            #
    # HINT: For the tanh function, you can compute the local derivative in terms #
    # of the output value from tanh.                                             #
    ##############################################################################
    x, prev_h, next_h, Wx, Wh, b = cache

    
    dWh = np.zeros(Wh.shape)
    dWx = np.zeros(Wx.shape)
    db = np.zeros(b.shape)
    
    db = np.sum((1 - next_h**2)* dnext_h, axis=0)
    
    dWh = np.dot(prev_h.T, (1 - next_h**2)* dnext_h)
    dWx = np.dot(x.T, (1 - next_h**2)* dnext_h)
    
    dprev_h = np.dot((1 - next_h**2)*dnext_h, Wh.T)
    
    dx = np.dot((1 - next_h**2)*dnext_h, Wx.T)
    

    
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################
    return dx, dprev_h, dWx, dWh, db


def rnn_forward(x, h0, Wx, Wh, b):
    """
    Run a vanilla RNN forward on an entire sequence of data. We assume an input
    sequence composed of T vectors, each of dimension D. The RNN uses a hidden
    size of H, and we work over a minibatch containing N sequences. After running
    the RNN forward, we return the hidden states for all timesteps.

    Inputs:
    - x: Input data for the entire timeseries, of shape (N, T, D).
    - h0: Initial hidden state, of shape (N, H)
    - Wx: Weight matrix for input-to-hidden connections, of shape (D, H)
    - Wh: Weight matrix for hidden-to-hidden connections, of shape (H, H)
    - b: Biases of shape (H,)

    Returns a tuple of:
    - h: Hidden states for the entire timeseries, of shape (N, T, H).
    - cache: Values needed in the backward pass
    """
    h, cache = None, None
    ##############################################################################
    # TODO: Implement forward pass for a vanilla RNN running on a sequence of    #
    # input data. You should use the rnn_step_forward function that you defined  #
    # above. You can use a for loop to help compute the forward pass.            #
    ##############################################################################
    N, T, D = x.shape
    H, H = Wh.shape
    
    h_out = np.zeros((N,T,H))
  

    h_step = np.copy(h0)
    
    for time in np.arange(T):      
        x_step = x[:,time,:]    
        next_h, cache = rnn_step_forward(x_step, h_step, Wx, Wh, b)
        h_out[:,(time),:] = next_h
        h_step = next_h
    
    h = h_out
    
   # x, prev_h, next_h, Wx, Wh, b
    
    cache = x, h, Wx, Wh, b, h0 #, h0
    #print(h)
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################
    return h, cache


def rnn_backward(dh, cache):
    """
    Compute the backward pass for a vanilla RNN over an entire sequence of data.

    Inputs:
    - dh: Upstream gradients of all hidden states, of shape (N, T, H)

    Returns a tuple of:
    - dx: Gradient of inputs, of shape (N, T, D)
    - dh0: Gradient of initial hidden state, of shape (N, H)
    - dWx: Gradient of input-to-hidden weights, of shape (D, H)
    - dWh: Gradient of hidden-to-hidden weights, of shape (H, H)
    - db: Gradient of biases, of shape (H,)
    """
    dx, dh0, dWx, dWh, db = None, None, None, None, None
    ##############################################################################
    # TODO: Implement the backward pass for a vanilla RNN running an entire      #
    # sequence of data. You should use the rnn_step_backward function that you   #
    # defined above. You can use a for loop to help compute the backward pass.   #
    ##############################################################################
    N, T, H = dh.shape
   
    x, h, Wx, Wh, b, h0 = cache
    
    #x, prev_h, next_h, Wx, Wh, b = cache
    
    #print(x.shape)
    #N, D = x.shape
    
    dx = np.zeros(x.shape)
    dh0 = np.zeros((N,H))
    dWx = np.zeros(Wx.shape)
    dWh = np.zeros(Wh.shape)
    db = np.zeros(b.shape)
    
    dprev_h_t = np.zeros(h[0,1].shape)
    
    dh0 = 0
    
    for time in reversed(np.arange(T)):  
        
        dh_step = dh[:, time, :]
        x_step = x[:, time, :]
        h_step = h[:, time, :]
        if (time == 0):
            h_step_last = h0
        else:
            h_step_last = h[:, time-1, :]
        
        step_cache = x_step, h_step_last, h_step, Wx, Wh, b
        
        dx_t, dprev_h_t, dWx_t, dWh_t, db_t = rnn_step_backward(dh_step + dprev_h_t, step_cache)
        
        dx[:, time, :] = dx_t
        dWx +=  dWx_t
        
        dWh += dWh_t
        
        db += db_t
        dh0 = dprev_h_t
        
    
    
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################
    return dx, dh0, dWx, dWh, db


def word_embedding_forward(x, W):
    """
    Forward pass for word embeddings. We operate on minibatches of size N where
    each sequence has length T. We assume a vocabulary of V words, assigning each
    to a vector of dimension D.

    Inputs:
    - x: Integer array of shape (N, T) giving indices of words. Each element idx
      of x muxt be in the range 0 <= idx < V.
    - W: Weight matrix of shape (V, D) giving word vectors for all words.

    Returns a tuple of:
    - out: Array of shape (N, T, D) giving word vectors for all input words.
    - cache: Values needed for the backward pass
    """
    out, cache = None, None
    ##############################################################################
    # TODO: Implement the forward pass for word embeddings.                      #
    #                                                                            #
    # HINT: This can be done in one line using NumPy's array indexing.           #
    ##############################################################################
    out = W[x, :]
    cache = x, W
    #print(out)
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################
    return out, cache


def word_embedding_backward(dout, cache):
    """
    Backward pass for word embeddings. We cannot back-propagate into the words
    since they are integers, so we only return gradient for the word embedding
    matrix.

    HINT: Look up the function np.add.at

    Inputs:
    - dout: Upstream gradients of shape (N, T, D)
    - cache: Values from the forward pass

    Returns:
    - dW: Gradient of word embedding matrix, of shape (V, D).
    """
    dW = None
    ##############################################################################
    # TODO: Implement the backward pass for word embeddings.                     #
    #                                                                            #
    # Note that Words can appear more than once in a sequence.                   #
    # HINT: Look up the function np.add.at                                       #
    ##############################################################################
    x, W = cache
    dW = np.zeros_like(W)
    
    np.add.at(dW, x, dout)
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################
    return dW


def sigmoid(x):
    """
    A numerically stable version of the logistic sigmoid function.
    """
    pos_mask = (x >= 0)
    neg_mask = (x < 0)
    z = np.zeros_like(x)
    z[pos_mask] = np.exp(-x[pos_mask])
    z[neg_mask] = np.exp(x[neg_mask])
    top = np.ones_like(x)
    top[neg_mask] = z[neg_mask]
    return top / (1 + z)


def lstm_step_forward(x, prev_h, prev_c, Wx, Wh, b):
    """
    Forward pass for a single timestep of an LSTM.

    The input data has dimension D, the hidden state has dimension H, and we use
    a minibatch size of N.

    Inputs:
    - x: Input data, of shape (N, D)
    - prev_h: Previous hidden state, of shape (N, H)
    - prev_c: previous cell state, of shape (N, H)
    - Wx: Input-to-hidden weights, of shape (D, 4H)
    - Wh: Hidden-to-hidden weights, of shape (H, 4H)
    - b: Biases, of shape (4H,)

    Returns a tuple of:
    - next_h: Next hidden state, of shape (N, H)
    - next_c: Next cell state, of shape (N, H)
    - cache: Tuple of values needed for backward pass.
    """
    next_h, next_c, cache = None, None, None
    #############################################################################
    # TODO: Implement the forward pass for a single timestep of an LSTM.        #
    # You may want to use the numerically stable sigmoid implementation above.  #
    #############################################################################
    activation  = np.dot(x, Wx) + np.dot(prev_h, Wh) + b
    #print(activation.shape)

    H = int(b.shape[0]/4)

    
    a_input = activation[:, 0:H]
    a_forget = activation[:, H:(2*H)]
    #print(a_forget)
    a_output = activation[:, (2*H):(3*H),]
    a_block = activation[:,(3*H):(4*H),]
    
    input_gate = sigmoid(a_input)
    forget_gate = sigmoid(a_forget)
    output_gate = sigmoid(a_output)
    block_output_gate = np.tanh(a_block)
    
    #print(forget_gate.shape)
   # print(prev_c.shape)
    
    next_c = (forget_gate * prev_c) + (input_gate*block_output_gate)
    next_h = output_gate * np.tanh(next_c)
    
    
    cache =  x, prev_h, prev_c, Wx, Wh, b, next_c, next_h, a_input, a_forget, a_output, a_block, input_gate, forget_gate, output_gate, block_output_gate
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################

    return next_h, next_c, cache


def lstm_step_backward(dnext_h, dnext_c, cache):
    """
    Backward pass for a single timestep of an LSTM.

    Inputs:
    - dnext_h: Gradients of next hidden state, of shape (N, H)
    - dnext_c: Gradients of next cell state, of shape (N, H)
    - cache: Values from the forward pass

    Returns a tuple of:
    - dx: Gradient of input data, of shape (N, D)
    - dprev_h: Gradient of previous hidden state, of shape (N, H)
    - dprev_c: Gradient of previous cell state, of shape (N, H)
    - dWx: Gradient of input-to-hidden weights, of shape (D, 4H)
    - dWh: Gradient of hidden-to-hidden weights, of shape (H, 4H)
    - db: Gradient of biases, of shape (4H,)
    """
    dx, dh, dc, dWx, dWh, db = None, None, None, None, None, None
    dprev_h = None
    dprev_c = None
    #############################################################################
    # TODO: Implement the backward pass for a single timestep of an LSTM.       #
    #                                                                           #
    # HINT: For sigmoid and tanh you can compute local derivatives in terms of  #
    # the output value from the nonlinearity.                                   #
    #############################################################################
    x, prev_h, prev_c, Wx, Wh, b, next_c, next_h, a_input, a_forget, a_output, a_block, input_gate, forget_gate, output_gate, block_output_gate = cache
    
    H = int(b.shape[0]/4)
    
    dx = np.zeros(x.shape)
    dWx = np.zeros(Wx.shape)
    dWh = np.zeros(Wh.shape)
    dprev_h = np.zeros(prev_h.shape)
    dprev_c = np.zeros(prev_c.shape)
    
    
    dnext_c += dnext_h * output_gate * (1 - np.tanh(next_c) * np.tanh(next_c))
    dprev_c = dnext_c * forget_gate
   
    d_input = dnext_c * block_output_gate * input_gate * (1 - input_gate)
    d_forget = dnext_c * prev_c * forget_gate * (1 - forget_gate)
    d_output = dnext_h * np.tanh(next_c) * output_gate * (1 - output_gate)
    d_block = dnext_c * input_gate * (1 - (block_output_gate**2))

    db  = np.concatenate((np.sum(d_input, axis=0), np.sum(d_forget, axis=0), np.sum(d_output, axis=0),np.sum(d_block, axis=0)))
    da  = np.concatenate((d_input, d_forget, d_output, d_block))


    dWh[:, 0:H]= np.dot(prev_h.T, d_input)
    dWh[:, H:(2*H)] = np.dot(prev_h.T, d_forget)
    dWh[:, (2*H):(3*H)] = np.dot(prev_h.T, d_output)
    dWh[:, (3*H):(4*H)] = np.dot(prev_h.T, d_block)


    dWx[:, 0:H]= np.dot(x.T, d_input)
    dWx[:, H:(2*H)] = np.dot(x.T, d_forget)
    dWx[:, (2*H):(3*H)] = np.dot(x.T, d_output)
    dWx[:, (3*H):(4*H)] = np.dot(x.T, d_block)
    
    
    dx += np.dot(d_input, Wx[:, 0:H].T)
    dx += np.dot(d_forget, Wx[:, H:(2*H)].T)
    dx += np.dot(d_output, Wx[:, (2*H):(3*H)].T)
    dx += np.dot(d_block, Wx[:, (3*H):(4*H)].T)
    
    dprev_h += np.dot(d_input, Wh[:, 0:H].T)
    dprev_h += np.dot(d_forget, Wh[:, H:(2*H)].T)
    dprev_h += np.dot(d_output, Wh[:, (2*H):(3*H)].T)
    dprev_h += np.dot(d_block, Wh[:, (3*H):(4*H)].T)
    
    
    
    
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################

    return dx, dprev_h, dprev_c, dWx, dWh, db


def lstm_forward(x, h0, Wx, Wh, b):
    """
    Forward pass for an LSTM over an entire sequence of data. We assume an input
    sequence composed of T vectors, each of dimension D. The LSTM uses a hidden
    size of H, and we work over a minibatch containing N sequences. After running
    the LSTM forward, we return the hidden states for all timesteps.

    Note that the initial cell state is passed as input, but the initial cell
    state is set to zero. Also note that the cell state is not returned; it is
    an internal variable to the LSTM and is not accessed from outside.

    Inputs:
    - x: Input data of shape (N, T, D)
    - h0: Initial hidden state of shape (N, H)
    - Wx: Weights for input-to-hidden connections, of shape (D, 4H)
    - Wh: Weights for hidden-to-hidden connections, of shape (H, 4H)
    - b: Biases of shape (4H,)

    Returns a tuple of:
    - h: Hidden states for all timesteps of all sequences, of shape (N, T, H)
    - cache: Values needed for the backward pass.
    """
    h, cache = None, None
    #############################################################################
    # TODO: Implement the forward pass for an LSTM over an entire timeseries.   #
    # You should use the lstm_step_forward function that you just defined.      #
    #############################################################################
    N, T, D = x.shape
    H = int(b.shape[0]/4)
    
    h = np.zeros((N, T, H))
    c = np.zeros_like(h)
    
    #h[:,-1,:] = h0
    c0 = np.zeros_like(h0)
    
    cache = []
    
    
    for i in np.arange(T):
        if i == 0:
            next_h, next_c, cache_now = lstm_step_forward(x[:,i,:], h0, c0, Wx, Wh, b)   
            cache.append(cache_now)
        else:
            next_h, next_c, cache_now = lstm_step_forward(x[:,i,:], h[:,i-1,:], c[:,i-1,:], Wx, Wh, b)   
            cache.append(cache_now)
        #x, prev_h, prev_c, Wx, Wh, b, next_c, next_h, a_input, a_forget, a_output, a_block, input_gate, forget_gate, output_gate, block_output_gate = cache
        h[:,i,:] = next_h
        c[:,i,:] = next_c
        #cache_now =  x, prev_h, prev_c, Wx, Wh, b, next_c, next_h, a_input, a_forget, a_output, a_block, input_gate, forget_gate, output_gate, block_output_gate   
        
    
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################
    #print(h)
    return h, cache


def lstm_backward(dh, cache):
    """
    Backward pass for an LSTM over an entire sequence of data.]

    Inputs:
    - dh: Upstream gradients of hidden states, of shape (N, T, H)
    - cache: Values from the forward pass

    Returns a tuple of:
    - dx: Gradient of input data of shape (N, T, D)
    - dh0: Gradient of initial hidden state of shape (N, H)
    - dWx: Gradient of input-to-hidden weight matrix of shape (D, 4H)
    - dWh: Gradient of hidden-to-hidden weight matrix of shape (H, 4H)
    - db: Gradient of biases, of shape (4H,)
    """
    dx, dh0, dWx, dWh, db = None, None, None, None, None
    #############################################################################
    # TODO: Implement the backward pass for an LSTM over an entire timeseries.  #
    # You should use the lstm_step_backward function that you just defined.     #
    #############################################################################
    N, T,  H = dh.shape
    D = cache[0][0].shape[1]
    
    #x, prev_h, prev_c, Wx, Wh, b, next_c, next_h, a_input, a_forget, a_output, a_block, input_gate, forget_gate, output_gate, block_output_gate = cache
    
    dx = np.zeros((N,T,D))
    dh0 = np.zeros((N,H))
    dWx = np.zeros((T, D, 4*H))
    dWh = np.zeros((T,H, 4*H))
    db = np.zeros((T,4*H))
    dprev_h = np.zeros((N,H))
    dprev_c = np.zeros((N, H))
    
    
    for i in reversed(np.arange(T)):
        
        dnext_h = dprev_h + dh[:, i, :]
        dnext_c = dprev_c 
        
        dx[:,i,:], dprev_h, dprev_c, dWx[i], dWh[i], db[i] = lstm_step_backward(dnext_h, dnext_c, cache[i])
        
    dWx = np.sum(dWx, axis=0)
    dWh = np.sum(dWh, axis=0)
    db = np.sum(db, axis=0)
   
    dh0 = dprev_h
    
    
    #     dh_step = dh[:, time, :]
    #     x_step = x[:, time, :]
    #     h_step = h[:, time, :]
    #     if (time == 0):
    #         h_step_last = h0
    #     else:
    #         h_step_last = h[:, time-1, :]
        
    #     step_cache = x_step, h_step_last, h_step, Wx, Wh, b
        
    #     dx_t, dprev_h_t, dWx_t, dWh_t, db_t = rnn_step_backward(dh_step + dprev_h_t, step_cache)
        
    #     dx[:, time, :] = dx_t
    #     dWx +=  dWx_t
        
    #     dWh += dWh_t
        
    #     db += db_t
    #     dh0 = dprev_h_t
    
    
    # ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################

    return dx, dh0, dWx, dWh, db


def temporal_affine_forward(x, w, b):
    """
    Forward pass for a temporal affine layer. The input is a set of D-dimensional
    vectors arranged into a minibatch of N timeseries, each of length T. We use
    an affine function to transform each of those vectors into a new vector of
    dimension M.

    Inputs:
    - x: Input data of shape (N, T, D)
    - w: Weights of shape (D, M)
    - b: Biases of shape (M,)

    Returns a tuple of:
    - out: Output data of shape (N, T, M)
    - cache: Values needed for the backward pass
    """
    N, T, D = x.shape
    M = b.shape[0]
    out = x.reshape(N * T, D).dot(w).reshape(N, T, M) + b
    cache = x, w, b, out
    return out, cache


def temporal_affine_backward(dout, cache):
    """
    Backward pass for temporal affine layer.

    Input:
    - dout: Upstream gradients of shape (N, T, M)
    - cache: Values from forward pass

    Returns a tuple of:
    - dx: Gradient of input, of shape (N, T, D)
    - dw: Gradient of weights, of shape (D, M)
    - db: Gradient of biases, of shape (M,)
    """
    x, w, b, out = cache
    N, T, D = x.shape
    M = b.shape[0]

    dx = dout.reshape(N * T, M).dot(w.T).reshape(N, T, D)
    dw = dout.reshape(N * T, M).T.dot(x.reshape(N * T, D)).T
    db = dout.sum(axis=(0, 1))

    return dx, dw, db


def temporal_softmax_loss(x, y, mask, verbose=False):
    """
    A temporal version of softmax loss for use in RNNs. We assume that we are
    making predictions over a vocabulary of size V for each timestep of a
    timeseries of length T, over a minibatch of size N. The input x gives scores
    for all vocabulary elements at all timesteps, and y gives the indices of the
    ground-truth element at each timestep. We use a cross-entropy loss at each
    timestep, summing the loss over all timesteps and averaging across the
    minibatch.

    As an additional complication, we may want to ignore the model output at some
    timesteps, since sequences of different length may have been combined into a
    minibatch and padded with NULL tokens. The optional mask argument tells us
    which elements should contribute to the loss.

    Inputs:
    - x: Input scores, of shape (N, T, V)
    - y: Ground-truth indices, of shape (N, T) where each element is in the range
         0 <= y[i, t] < V
    - mask: Boolean array of shape (N, T) where mask[i, t] tells whether or not
      the scores at x[i, t] should contribute to the loss.

    Returns a tuple of:
    - loss: Scalar giving loss
    - dx: Gradient of loss with respect to scores x.
    """

    N, T, V = x.shape

    x_flat = x.reshape(N * T, V)
    y_flat = y.reshape(N * T)
    mask_flat = mask.reshape(N * T)

    probs = np.exp(x_flat - np.max(x_flat, axis=1, keepdims=True))
    probs /= np.sum(probs, axis=1, keepdims=True)
    loss = -np.sum(mask_flat * np.log(probs[np.arange(N * T), y_flat])) / N
    dx_flat = probs.copy()
    dx_flat[np.arange(N * T), y_flat] -= 1
    dx_flat /= N
    dx_flat *= mask_flat[:, None]

    if verbose: print('dx_flat: ', dx_flat.shape)

    dx = dx_flat.reshape(N, T, V)

    return loss, dx
