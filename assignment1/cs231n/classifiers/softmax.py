from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange


def softmax_loss_naive(W, X, y, reg):
    """
    Softmax loss function, naive implementation (with loops)

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)
    num_train = X.shape[0]
    num_classes = W.shape[1]

    scores = X.dot(W)
    scores = np.exp(scores.T - np.max(scores, axis=1)).T
    softmax = (scores.T/scores.sum(axis=1)).T

    for i in range(num_train):
        for j in range(num_classes):
            dW[:, j] += X[i]*softmax[i, j]
        dW[:, y[i]] -= X[i]
        loss += -np.log(softmax[i, y[i]]) 
        
    loss /= num_train 
    loss += reg*np.sum((W*W))

    dW /= num_train
    dW += reg*2*W

    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)
    num_train = X.shape[0]
    num_classes = W.shape[1]   
    
    scores = X.dot(W)
    scores = np.exp(scores - scores.max(axis=1, keepdims=True))
    softmax = scores / scores.sum(axis=1, keepdims=True)

    loss = np.sum(-np.log(softmax[np.arange(num_train), y])) / num_train 
    loss += reg*np.sum((W*W))

    softmax[np.arange(num_train), y] -= 1
    dW = (X.T).dot(softmax)/num_train
    dW += reg*2*W

    return loss, dW


    
