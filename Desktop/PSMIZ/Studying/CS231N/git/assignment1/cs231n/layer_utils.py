from .layers import *


def affine_relu_forward(x, w, b):
    """
    Convenience layer that perorms an affine transform followed by a ReLU

    Inputs:
    - x: Input to the affine layer
    - w, b: Weights for the affine layer

    Returns a tuple of:
    - out: Output from the ReLU
    - cache: Object to give to the backward pass
    """
    z, aff_cache = affine_forward(x, w, b)
    a, relu_cache = relu_forward(z)
    cache = (aff_cache, relu_cache)
    return a, cache


def affine_relu_backward(dout, cache):
    """
    Backward pass for the affine-relu convenience layer
    """
    aff_cache, relu_cache = cache 
    dz = relu_backward(dout, relu_cache)
    dx, dw, db = affine_backward(dz, aff_cache)
    return dx, dw, db

