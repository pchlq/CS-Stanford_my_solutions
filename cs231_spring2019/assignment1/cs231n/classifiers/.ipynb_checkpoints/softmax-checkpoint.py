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
    dW = np.zeros_like(W) # (3073, 10)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    for i in range(len(X)): # iteration by each sample [0-499]
        scores = X[i].dot(W) # get scores of sample
        true_score = scores[y[i]]
        max_score = np.max(scores)
        scores -= max_score # substraction of max score for numeric stability
        loss += max_score - true_score + np.log(np.exp(scores).sum())
        
        for j in range( len(set(y)) ): # compute dL/dW
            dW[:, j] = np.exp(scores[j]) / np.exp(scores).sum() * X[i, :]
        dW[:, y[i]] -= X[i, :]
    
    loss /= len(X)
    loss += reg * np.sum(W * W)
    
    dW /= len(X)
    dW += 2 * reg * W
    
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    scores = X.dot(W)
    true_scores = scores[range(len(X)), y]
    max_scores = np.max(scores, 1, keepdims=True)
    scores -= max_scores # substraction of max score for numeric stability
    loss = max_scores.sum() - true_scores.sum() + np.sum(np.log(np.exp(scores).sum(1)))
    loss /= len(X)
    loss += reg * np.sum(W * W)
    
    # compute dL/dZ
    dZ = np.exp(scores) / np.exp(scores).sum(1, keepdims=True)
    dZ[range(len(X)), y] -= 1
    # compute dZ/dW
    dW = np.transpose(X).dot(dZ)
    dW /= len(X)
    dW += 2 * reg * W

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
