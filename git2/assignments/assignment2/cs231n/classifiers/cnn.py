from builtins import object
import numpy as np

from ..layers import *
from ..fast_layers import *
from ..layer_utils import *


class ThreeLayerConvNet(object):
    """
    A three-layer convolutional network with the following architecture:

    conv - relu - 2x2 max pool - affine - relu - affine - softmax

    The network operates on minibatches of data that have shape (N, C, H, W)
    consisting of N images, each with height H and width W and with C input
    channels.
    """

    def __init__(
        self,
        input_dim=(3, 32, 32),
        num_filters=32,
        filter_size=7,
        hidden_dim=100,
        num_classes=10,
        weight_scale=1e-3,
        reg=0.0,
        dtype=np.float32,
    ):
        """
        Initialize a new network.

        Inputs:
        - input_dim: Tuple (C, H, W) giving size of input data
        - num_filters: Number of filters to use in the convolutional layer
        - filter_size: Width/height of filters to use in the convolutional layer
        - hidden_dim: Number of units to use in the fully-connected hidden layer
        - num_classes: Number of scores to produce from the final affine layer.
        - weight_scale: Scalar giving standard deviation for random initialization
          of weights.
        - reg: Scalar giving L2 regularization strength
        - dtype: numpy datatype to use for computation.
        """
        self.params = {}
        self.reg = reg
        self.dtype = dtype
        C, H, W = input_dim 
        F, size = num_filters, filter_size
        # height and width of input is preserved with appropriate stride and pad size 
        CH, CW = H, W                    
        PH, PW = (H-2)//2 + 1, (W-2)//2 + 1

        self.params["W1"], self.params["b1"] = np.random.randn(F, C, size, size)*weight_scale, np.zeros(F)     # conv layer
        self.params["W2"], self.params["b2"] = np.random.randn(F*PH*PW, hidden_dim)*weight_scale, np.zeros(hidden_dim)
        self.params["W3"], self.params["b3"] = np.random.randn(hidden_dim, hidden_dim)*weight_scale, np.zeros(hidden_dim)

        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)

    def loss(self, X, y=None):
        """
        Evaluate loss and gradient for the three-layer convolutional network.

        Input / output: Same API as TwoLayerNet in fc_net.py.
        """
        N = X.shape[0]
        W1, b1 = self.params["W1"], self.params["b1"]
        W2, b2 = self.params["W2"], self.params["b2"]
        W3, b3 = self.params["W3"], self.params["b3"]
        loss, grads = 0, {}

        # pass conv_param to the forward pass for the convolutional layer
        # Padding and stride chosen to preserve the input spatial size
        filter_size = W1.shape[2]
        conv_param = {"stride": 1, "pad": (filter_size - 1) // 2}
        # pass pool_param to the forward pass for the max-pooling layer
        pool_param = {"pool_height": 2, "pool_width": 2, "stride": 2}

        out, conv_relu_cache = conv_relu_forward(X, W1, b1, conv_param)
        out, pool_cache = max_pool_forward_naive(out, pool_param)
        pool_shape = out.shape
        out = out.reshape(N, -1)
        out, aff_relu_cache = affine_relu_forward(out, W2, b2)
        scores, aff_cache = affine_forward(out, W3, b3)

        if y is None:
            return scores
        loss, dout = softmax_loss(scores, y)

        dout, grads['W3'], grads['b3'] = affine_backward(dout, aff_cache)
        dout, grads['W2'], grads['b2'] = affine_relu_backward(dout, aff_relu_cache)
        dout = dout.reshape(pool_shape)
        dout = max_pool_backward_naive(dout, pool_cache)
        dout, grads['W1'], grads['b1'] = conv_relu_backward(dout, conv_relu_cache)

        for layer in range(1, 4):
            loss += np.sum(self.reg*0.5*self.params[f"W{layer}"]**2)
            grads[f"W{layer}"] += self.reg*self.params[f"W{layer}"]

        return loss, grads



