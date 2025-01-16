"""This module contains loss functions with their gradient and hessian"""

import numpy as np

def compute_loss(beta, C, c, loss):
    """
    """
    x = c - np.matmul(C, beta)
    if loss == 'hinge':
        loss_val = np.sum(np.power(np.maximum(0.0, -x), 3.0))
        loss_grad = - 3.0 * np.square(np.maximum(0.0, -x))
        loss_hess = np.diag(6.0 * np.maximum(0, -x))
    if loss == 'logit':
        loss_val = np.sum(np.log(1.0 + np.exp(-x)))
        loss_grad = - np.exp(-x) / (1.0 + np.exp(-x))
        loss_hess = np.diag(np.exp(-x) / np.square(1.0 + np.exp(-x)))
    return (loss_val, loss_grad, loss_hess)

def compute_dist(beta, y, q, method):
    """
    """
    if method == 'chi2':
        dist_val = np.sum(np.square(beta - y) / (2.0 * q * y))
        dist_grad = (beta / y - 1.0) / q
        dist_hess = np.diag(1.0 / (q * y))
    return (dist_val, dist_grad, dist_hess)

