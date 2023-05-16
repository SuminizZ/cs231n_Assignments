from builtins import range
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
    N = x.shape[0]  
    # print(x.shape, w.shape, b.shape)
    out = (x.reshape(N, -1)).dot(w) + b
    
    cache = (x, w, b)
    return out, cache


def affine_backward(dout, cache):
    """
    Computes the backward pass for an affine layer.

    Inputs:
    - dout: Upstream derivative, of shape (N, M)
    - cache: Tuple of:
      - x: Input data, of shape (N, d_1, ... d_k)
      - w: Weights, of shape (D, M)
      - b: Biases, of shape (M,)

    Returns a tuple of:
    - dx: Gradient with respect to x, of shape (N, d1, ..., d_k)
    - dw: Gradient with respect to w, of shape (D, M)
    - db: Gradient with respect to b, of shape (M,)
    """
    x, w, b = cache
    N, D = x.shape[0], np.prod(x.shape[1:])
    x = x.reshape(N, D)
    
    dx = dout.dot(w.T).reshape(x.shape)
    dw = (x.T).dot(dout)
    db = dout.sum(axis=0)

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
    out = np.maximum(0, x)
  
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
    dx, x = dout, cache
    dx[x <= 0] = 0

    return dx


def svm_loss(x, y):
    """
    Computes the loss and gradient using for multiclass SVM classification.

    Inputs:
    - x: Input data, of shape (N, C) where x[i, j] is the score for the jth
      class for the ith input.
    - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
      0 <= y[i] < C

    Returns a tuple of:
    - loss: Scalar giving the loss
    - dx: Gradient of the loss with respect to x
    """
    num_train, num_class = x.shape[0], x.shape[1]

    correct_x = x[np.arange(num_train), y]
    margins = (x.T - correct_x + 1).T
    margins[margins <= 0] = 0
    margins[np.arange(num_train), y] = 0

    loss = margins.sum()/num_train

    margins[margins > 0] = 1
    valid_cnt = margins.sum(axis=1)
    margins[np.arange(num_train), y] = -valid_cnt

    dx = margins/num_train

    return loss, dx


def softmax_loss(x, y):
    """
    Computes the loss and gradient for softmax classification.

    Inputs:
    - x: Input data, of shape (N, C) where x[i, j] is the score for the jth
      class for the ith input.
    - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
      0 <= y[i] < C

    Returns a tuple of:
    - loss: Scalar giving the loss
    - dx: Gradient of the loss with respect to x
    """
    num_train, num_class = x.shape[0], x.shape[1]
    
    x = np.exp(x - x.max(axis=1, keepdims=1))
    x = x/x.sum(axis=1, keepdims=1)
    correct_x = x[np.arange(num_train), y] + 1e-8

    loss = -(np.log(correct_x).sum())/num_train

    x[np.arange(num_train), y] -= 1
    dx = x/num_train

    return loss, dx


def batchnorm_forward(x, gamma, beta, bn_param):
    """Forward pass for batch normalization.

    During training the sample mean and (uncorrected) sample variance are
    computed from minibatch statistics and used to normalize the incoming data.
    During training we also keep an exponentially decaying running mean of the
    mean and variance of each feature, and these averages are used to normalize
    data at test-time.

    At each timestep we update the running averages for mean and variance using
    an exponential decay based on the momentum parameter:

    running_mean = momentum * running_mean + (1 - momentum) * sample_mean
    running_var = momentum * running_var + (1 - momentum) * sample_var

    Note that the batch normalization paper suggests a different test-time
    behavior: they compute sample mean and variance for each feature using a
    large number of training images rather than using a running average. For
    this implementation we have chosen to use running averages instead since
    they do not require an additional estimation step; the torch7
    implementation of batch normalization also uses running averages.

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
    m = bn_param.get('momentum', 0.9)
    N, D = x.shape
    # set default 
    running_mean = bn_param.get("running_mean", np.zeros(D, dtype=x.dtype))
    running_var = bn_param.get("running_var", np.zeros(D, dtype=x.dtype))
    out, cache = None, None 

    if mode == "train":
        b_mean = x.mean(axis=0, keepdims=True)
        b_var = x.var(axis=0, keepdims=True) + eps
        x_norm = (x-b_mean)/np.sqrt(b_var)       #(N, D)
        out = gamma*x_norm + beta
        cache = (x_norm, b_mean, b_var, x, gamma, beta)

        # Store the updated running means back into bn_param
        bn_param['running_mean']  = m*(running_mean) + (1-m)*b_mean
        bn_param['running_var']  = m*(running_var) + (1-m)*b_var

    elif mode == "test":
        x_norm = (x-running_mean)/np.sqrt(running_var + eps) 
        out = x_norm*gamma + beta

    else:
        raise ValueError('Invalid forward batchnorm mode "%s"' % mode)

    return out, cache


def batchnorm_backward(dout, cache):
    """Backward pass for batch normalization.

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
    N, D = dout.shape
    x_norm, b_mean, b_var, x, gamma, beta = cache 

    b_std = np.sqrt(b_var)
    dgamma = (dout*x_norm).sum(axis=0)
    dbeta = dout.sum(axis=0)

    dz = dout*gamma
    dzdv = (-0.5)*(b_std**-3)*(x-b_mean)
    dzdm = -1/b_std
    dvdm = np.sum((-2/N)*(x-b_mean), axis=0)
    dzdx = 1/b_std
    dmdx = 1/N
    dvdx = (2/N)*(x-b_mean)

    dx = dz*dzdx + np.sum(dz*dzdm, axis=0)*dmdx + np.sum(dz*dzdv, axis=0)*(dvdx + dvdm*dmdx)

    return dx, dgamma, dbeta


def batchnorm_backward_alt(dout, cache):
    """Alternative backward pass for batch normalization.

    For this implementation you should work out the derivatives for the batch
    normalizaton backward pass on paper and simplify as much as possible. You
    should be able to derive a simple expression for the backward pass.
    See the jupyter notebook for more hints.

    Note: This implementation should expect to receive the same cache variable
    as batchnorm_backward, but might not use all of the values in the cache.

    Inputs / outputs: Same as batchnorm_backward
    """
    dx, dgamma, dbeta = None, None, None
    N, D = dout.shape
    x_norm, b_mean, b_var, x, gamma, beta = cache 

    b_std = np.sqrt(b_var)
    dgamma = (dout*x_norm).sum(axis=0)
    dbeta = dout.sum(axis=0)

    dz = dout*gamma
    dx = (dz + np.sum(-dz, axis=0)*(1/N) + np.sum(dz*(-1/N)*x_norm, axis=0)*x_norm)*(1/b_std)

    return dx, dgamma, dbeta


def layernorm_forward(x, gamma, beta, ln_param):
    """Forward pass for layer normalization.

    During both training and test-time, the incoming data is normalized per data-point,
    before being scaled by gamma and beta parameters identical to that of batch normalization.

    Note that in contrast to batch normalization, the behavior during train and test-time for
    layer normalization are identical, and we do not need to keep track of running averages
    of any sort.

    Input:
    - x: Data of shape (N, D)
    - gamma: Scale parameter of shape (D,)
    - beta: Shift paremeter of shape (D,)
    - ln_param: Dictionary with the following keys:
        - eps: Constant for numeric stability

    Returns a tuple of:
    - out: of shape (N, D)
    - cache: A tuple of values needed in the backward pass
    """
    out, cache = None, None
    eps = ln_param.get("eps", 1e-5)
    N, D = x.shape
    out, cache = None, None 

    l_mean = x.mean(axis=1, keepdims=True)
    l_var = x.var(axis=1, keepdims=True) + eps
    x_norm = (x-l_mean)/np.sqrt(l_var)       #(N, D)
    out = gamma*x_norm + beta
    cache = (x_norm, l_mean, l_var, x, gamma, beta)

    return out, cache


def layernorm_backward(dout, cache):
    """Backward pass for layer normalization.

    For this implementation, you can heavily rely on the work you've done already
    for batch normalization.

    Inputs:
    - dout: Upstream derivatives, of shape (N, D)
    - cache: Variable of intermediates from layernorm_forward.

    Returns a tuple of:
    - dx: Gradient with respect to inputs x, of shape (N, D)
    - dgamma: Gradient with respect to scale parameter gamma, of shape (D,)
    - dbeta: Gradient with respect to shift parameter beta, of shape (D,)
    """
    dx, dgamma, dbeta = None, None, None
    N, D = dout.shape
    x_norm, l_mean, l_var, x, gamma, beta = cache 

    l_std = np.sqrt(l_var)
    dgamma = (dout*x_norm).sum(axis=0)
    dbeta = dout.sum(axis=0)

    dz = dout*gamma
    dx = (np.sum(dz*x_norm, axis=1, keepdims=True)*(-1/D)*x_norm \
         + np.sum(dz*(-1/D), axis=1, keepdims=True) + dz)/l_std

    return dx, dgamma, dbeta


def dropout_forward(x, dropout_param):
    """Forward pass for inverted dropout.

    Note that this is different from the vanilla version of dropout.
    Here, p is the probability of keeping a neuron output, as opposed to
    the probability of dropping a neuron output.
    See http://cs231n.github.io/neural-networks-2/#reg for more details.

    Inputs:
    - x: Input data, of any shape
    - dropout_param: A dictionary with the following keys:
      - p: Dropout parameter. We keep each neuron output with probability p.
      - mode: 'test' or 'train'. If the mode is train, then perform dropout;
        if the mode is test, then just return the input.
      - seed: Seed for the random number generator. Passing seed makes this
        function deterministic, which is needed for gradient checking but not
        in real networks.

    Outputs:
    - out: Array of the same shape as x.
    - cache: tuple (dropout_param, mask). In training mode, mask is the dropout
      mask that was used to multiply the input; in test mode, mask is None.
    """
    p, mode = dropout_param["p"], dropout_param["mode"]
    if "seed" in dropout_param:
        np.random.seed(dropout_param["seed"])

    mask, out = None, x.copy()
    if mode == "train":
        mask = (np.random.rand(*x.shape) < p)/p
        out = out*mask

    cache = (dropout_param, mask)
    out = out.astype(x.dtype, copy=False)

    return out, cache


def dropout_backward(dout, cache):
    """Backward pass for inverted dropout.

    Inputs:
    - dout: Upstream derivatives, of any shape
    - cache: (dropout_param, mask) from dropout_forward.
    """
    dropout_param, mask = cache
    mode = dropout_param["mode"]
    p = dropout_param["p"]

    dx = dout
    if mode == "train":
        dx = dx*mask
        # dx /= p
    return dx


def conv_forward_naive(x, w, b, conv_param):
    """A naive implementation of the forward pass for a convolutional layer.

    The input consists of N data points, each with C channels, height H and
    width W. We convolve each input with F different filters, where each filter
    spans all C channels and has height HH and width WW.

    Input:
    - x: Input data of shape (N, C, H, W)
    - w: Filter weights of shape (F, C, HH, WW)
    - b: Biases, of shape (F,)
    - conv_param: A dictionary with the following keys:
      - 'stride': The number of pixels between adjacent receptive fields in the
        horizontal and vertical directions.
      - 'pad': The number of pixels that will be used to zero-pad the input.

    During padding, 'pad' zeros should be placed symmetrically (i.e equally on both sides)
    along the height and width axes of the input. Be careful not to modfiy the original
    input x directly.

    Returns a tuple of:
    - out: Output data, of shape (N, F, H', W') where H' and W' are given by
      H' = 1 + (H + 2 * pad - HH) / stride
      W' = 1 + (W + 2 * pad - WW) / stride
    - cache: (x, w, b, conv_param)
    """
    stride = conv_param.get("stride", 1)
    pad = conv_param.get("pad", 0)
    N, C, H, W = x.shape
    F, C, HH, WW = w.shape

    px = np.pad(x, ((0,0), (0,0), (pad,pad), (pad,pad)), 'constant', constant_values=0)
    assert (H + pad*2 - HH)%stride == 0
    assert (W + pad*2 - WW)%stride == 0
    conv_h = int((H + pad*2 - HH)/stride + 1)
    conv_w = int((W + pad*2 - WW)/stride + 1)
    out = np.zeros((N, F, conv_h, conv_w))

    # for f in range(F):
    #     for ch in range(conv_h):
    #         sidx_h = ch*stride
    #         eidx_h = sidx_h+HH
    #         for cw in range(conv_w):
    #             sidx_w = cw*stride
    #             eidx_w = sidx_w+WW
    #             out[:,f,ch,cw] = np.sum(px[:, :, sidx_h:eidx_h, sidx_w:eidx_w]*w[f], axis=(1,2,3)) + b[f]
    
    w = w.reshape(F, C*HH*WW)
    
    for ch in range(conv_h):
        sidx_h = ch*stride
        eidx_h = sidx_h+HH
        for cw in range(conv_w):
            sidx_w = cw*stride
            eidx_w = sidx_w+WW
            out[:, :, ch, cw] = px[:, :, sidx_h:eidx_h, sidx_w:eidx_w].reshape(N, C*HH*WW).dot(w.T) 

    out += b.reshape(1, F, 1, 1)
    w = w.reshape(F, C, HH, WW)
    cache = (px, x, w, b, conv_param)
    return out, cache


def conv_backward_naive(dout, cache):
    """A naive implementation of the backward pass for a convolutional layer.

    Inputs:
    - dout: Upstream derivatives.
    - cache: A tuple of (x, w, b, conv_param) as in conv_forward_naive

    Returns a tuple of:
    - dx: Gradient with respect to x
    - dw: Gradient with respect to w
    - db: Gradient with respect to b
    """
    px, x, w, b, conv_param = cache 
    N, F, conv_h, conv_w = dout.shape 
    dpx, dx, dw, db = np.zeros_like(px), np.zeros_like(x), np.zeros_like(w), np.zeros_like(b)
    N, C, H, W = x.shape
    F, C, HH, WW = w.shape
    stride, pad = conv_param['stride'], conv_param['pad']

    for f in range(F):
        for ch in range(conv_h):
            sidx_h = ch*stride
            eidx_h = sidx_h+HH
            for cw in range(conv_w):
                sidx_w = cw*stride
                eidx_w = sidx_w+WW
                dw[f] += (px[:, :, sidx_h:eidx_h, sidx_w:eidx_w].T.dot(dout[:, f, ch, cw])).T
                dpx[:, :, sidx_h:eidx_h, sidx_w:eidx_w] += dout[:, f, ch, cw].reshape(N,1,1,1)*w[f]
        
    db = np.sum(dout, axis=(0,2,3))        
    dx = dpx[:, :, 1:-1, 1:-1]
    return dx, dw, db


def max_pool_forward_naive(x, pool_param):
    """A naive implementation of the forward pass for a max-pooling layer.

    Inputs:
    - x: Input data, of shape (N, C, H, W)
    - pool_param: dictionary with the following keys:
      - 'pool_height': The height of each pooling region
      - 'pool_width': The width of each pooling region
      - 'stride': The distance between adjacent pooling regions

    No padding is necessary here, eg you can assume:
      - (H - pool_height) % stride == 0
      - (W - pool_width) % stride == 0

    Returns a tuple of:
    - out: Output data, of shape (N, C, H', W') where H' and W' are given by
      H' = 1 + (H - pool_height) / stride
      W' = 1 + (W - pool_width) / stride
    - cache: (x, pool_param)
    """
    N, C, H, W = x.shape
    PH, PW, stride = pool_param['pool_height'], pool_param['pool_width'], pool_param['stride']
    assert (H-PH)%stride == 0
    assert (W-PW)%stride == 0
    pool_h, pool_w = int((H-PH)/stride + 1), int((W-PW)/stride + 1)
    out = np.zeros((N, C, pool_h, pool_w))
    
    for ph in range(pool_h):
        sidx_h = ph*stride
        eidx_h = sidx_h+PH
        for pw in range(pool_w):
            sidx_w = pw*stride
            eidx_w = sidx_w+PW
            out[:, :, ph, pw] = x[:, :, sidx_h:eidx_h, sidx_w:eidx_w].max(axis=(2,3))

    cache = (x, pool_param)
    return out, cache


def max_pool_backward_naive(dout, cache):
    """A naive implementation of the backward pass for a max-pooling layer.

    Inputs:
    - dout: Upstream derivatives
    - cache: A tuple of (x, pool_param) as in the forward pass.

    Returns:
    - dx: Gradient with respect to x
    """
    x, pool_param = cache
    N, C, H, W = x.shape
    PH, PW, stride = pool_param['pool_height'], pool_param['pool_width'], pool_param['stride']
    pool_h, pool_w = int((H-PH)/stride + 1), int((W-PW)/stride + 1)
    dx = np.zeros((N, C, H, W))
    
    for ph in range(pool_h):
        sidx_h = ph*stride
        eidx_h = sidx_h+PH
        for pw in range(pool_w):
            sidx_w = pw*stride
            eidx_w = sidx_w+PW
            tmpx = x[:, :, sidx_h:eidx_h, sidx_w:eidx_w]
            tmpx = (tmpx == tmpx.max(axis=(2,3), keepdims=True)).astype(int)
            dx[:, :, sidx_h:eidx_h, sidx_w:eidx_w] += dout[:, :, ph, pw].reshape(N, C, 1, 1)*tmpx

    return dx


def spatial_batchnorm_forward(x, gamma, beta, bn_param):
    """Computes the forward pass for spatial batch normalization.

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
    N, C, H, W = x.shape

    x = x.transpose(0,2,3,1).reshape(-1, C)
    out, cache = batchnorm_forward(x, gamma, beta, bn_param)
    out = out.reshape(N, H, W, C).transpose(0, 3, 1, 2)
    return out, cache

    # for c in range(1, C):
    #     x_reshaped = np.concatenate([x_reshaped, np.expand_dims(x[:, c, :, :].reshape(-1), axis=1)], axis=1)
    # for c in range(1, C):
    #     x_recovered = np.concatenate([x_recovered, np.expand_dims(out[:,c].reshape(N,H,W), axis=1)], axis=1)   

    return x_recovered, cache


def spatial_batchnorm_backward(dout, cache):
    """Computes the backward pass for spatial batch normalization.

    Inputs:
    - dout: Upstream derivatives, of shape (N, C, H, W)
    - cache: Values from the forward pass

    Returns a tuple of:
    - dx: Gradient with respect to inputs, of shape (N, C, H, W)
    - dgamma: Gradient with respect to scale parameter, of shape (C,)
    - dbeta: Gradient with respect to shift parameter, of shape (C,)
    """
    dx, dgamma, dbeta = None, None, None
    N, C, H, W = dout.shape
    dout = dout.transpose(0,2,3,1).reshape(-1, C)
    dx, dgamma, dbeta = batchnorm_backward_alt(dout, cache)
    dx = dx.reshape(N, H, W, C).transpose(0, 3, 1, 2)

    return dx, dgamma, dbeta


def spatial_groupnorm_forward(x, gamma, beta, G, gn_param):
    """Computes the forward pass for spatial group normalization.
    
    In contrast to layer normalization, group normalization splits each entry in the data into G
    contiguous pieces, which it then normalizes independently. Per-feature shifting and scaling
    are then applied to the data, in a manner identical to that of batch normalization and layer
    normalization.

    Inputs:
    - x: Input data of shape (N, C, H, W)
    - gamma: Scale parameter, of shape (1, C, 1, 1)
    - beta: Shift parameter, of shape (1, C, 1, 1)
    - G: Integer mumber of groups to split into, should be a divisor of C
    - gn_param: Dictionary with the following keys:
      - eps: Constant for numeric stability

    Returns a tuple of:
    - out: Output data, of shape (N, C, H, W)
    - cache: Values needed for the backward pass
    """
    N, C, H, W = x.shape
    eps = gn_param.get("eps", 1e-5)
    x = x.reshape(N*G, -1)
    x_std = np.sqrt(x.var(axis=1, keepdims=True) + eps)
    x_norm = ((x-x.mean(axis=1, keepdims=True))/x_std).reshape(N, C, H, W)
    out = gamma*(x_norm) + beta
    cache = (G, x_norm, x_std, gamma, beta)

    return out, cache


def spatial_groupnorm_backward(dout, cache):
    """Computes the backward pass for spatial group normalization.

    Inputs:
    - dout: Upstream derivatives, of shape (N, C, H, W)
    - cache: Values from the forward pass

    Returns a tuple of:
    - dx: Gradient with respect to inputs, of shape (N, C, H, W)
    - dgamma: Gradient with respect to scale parameter, of shape (1, C, 1, 1)
    - dbeta: Gradient with respect to shift parameter, of shape (1, C, 1, 1)
    """
    dx, dgamma, dbeta = None, None, None
    N, C, H, W = dout.shape
    G, x_norm, x_std, gamma, beta = cache 
    D = C*H*W//G

    dgamma = (dout*x_norm).sum(axis=(0, 2, 3), keepdims=True)
    dbeta = dout.sum(axis=(0, 2, 3), keepdims=True)

    dz = dout*gamma
    # print(dz.shape, x_norm.shape, (dz*x_norm).shape, (dz*x_norm).reshape(N*G, -1).shape, dz*(-1/D).reshape(N*G, -1).shape)
    # print((np.sum((dz*x_norm).reshape(N*G, -1), axis=1, keepdims=True)).shape)
    dx = (np.sum((dz*x_norm).reshape(N*G, -1), axis=1, keepdims=True)*(-1/D)*x_norm.reshape(N*G, -1) \
         + np.sum(dz.reshape(N*G, -1), axis=1, keepdims=True)*(-1/D) + dz.reshape(N*G, -1))/x_std
    dx = dx.reshape(N, C, H, W)

    return dx, dgamma, dbeta


