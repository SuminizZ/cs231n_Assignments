from builtins import range
from builtins import object
import numpy as np

from ..layers import *
from ..layer_utils import *


class TwoLayerNet(object):
    """
    A two-layer fully-connected neural network with ReLU nonlinearity and
    softmax loss that uses a modular layer design. We assume an input dimension
    of D, a hidden dimension of H, and perform classification over C classes.

    The architecure should be affine - relu - affine - softmax.

    Note that this class does not implement gradient descent; instead, it
    will interact with a separate Solver object that is responsible for running
    optimization.

    The learnable parameters of the model are stored in the dictionary
    self.params that maps parameter names to numpy arrays.
    """

    def __init__(
        self,
        input_dim=3 * 32 * 32,
        hidden_dim=100,
        num_classes=10,
        weight_scale=1e-3,
        reg=0.0,
    ):
        """
        Initialize a new network.

        Inputs:
        - input_dim: An integer giving the size of the input
        - hidden_dim: An integer giving the size of the hidden layer
        - num_classes: An integer giving the number of classes to classify
        - weight_scale: Scalar giving the standard deviation for random
          initialization of the weights.
        - reg: Scalar giving L2 regularization strength.
        """
        self.params = {}
        self.reg = reg

        W1 = np.random.normal(0, weight_scale, (input_dim, hidden_dim))
        W2 = np.random.normal(0, weight_scale, (hidden_dim, num_classes))
        b1 = np.zeros(hidden_dim)
        b2 = np.zeros(num_classes)

        self.params['W1'], self.params['W2'], self.params['b1'], self.params['b2'] = W1, W2, b1, b2
        

    def loss(self, X, y=None):
        """
        Compute loss and gradient for a minibatch of data.

        Inputs:
        - X: Array of input data of shape (N, d_1, ..., d_k)
        - y: Array of labels, of shape (N,). y[i] gives the label for X[i].

        Returns:
        If y is None, then run a test-time forward pass of the model and return:
        - scores: Array of shape (N, C) giving classification scores, where
          scores[i, c] is the classification score for X[i] and class c.

        If y is not None, then run a training-time forward and backward pass and
        return a tuple of:
        - loss: Scalar value giving the loss
        - grads: Dictionary with the same keys as self.params, mapping parameter
          names to gradients of the loss with respect to those parameters.
        """
        
        W1, W2, b1, b2 = self.params['W1'], self.params['W2'], self.params['b1'], self.params['b2']
        A1, cache = affine_relu_forward(X, W1, b1)
        A2, aff_cache = affine_forward(A1, W2, b2)

        loss, grads = 0, {}
        # If y is None then we are in test mode so just return scores
        if y is None:
            return A2

        loss, dA2 = softmax_loss(A2, y)
        dA1, dW2, db2 = affine_backward(dA2, aff_cache)
        dX, dW1, db1 = affine_relu_backward(dA1, cache)

        reg = self.reg        
        loss += reg * (np.sum(W2 * W2) + np.sum(W1 * W1)) 
        dW1 += 2*reg*W1
        dW2 += 2*reg*W2
        
        grads['W1'], grads['W2'], grads['b1'], grads['b2'] = dW1, dW2, db1, db2

        return loss, grads
