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
        loss_grad = - 1.0 / (1.0 + np.exp(x))
        loss_hess = np.diag(np.exp(x) / np.square(1.0 + np.exp(x)))
    return (loss_val, loss_grad, loss_hess)

def compute_dist(beta, y, q, method, l=None, h=None):
    """
    """
    if method == 'chi2':

        indices = ((q != 0) & (y != 0))
        dist_val = np.sum(np.square(beta[indices] - y[indices]) / (2.0 * q[indices] * y[indices]))

        dist_grad = np.zeros(len(beta))
        dist_grad[indices] = (beta[indices] / y[indices] - 1.0) / q[indices]

        dist_hess = np.zeros(len(beta))
        dist_hess[indices] = 1.0 / (q[indices] * y[indices])
        dist_hess = np.diag(dist_hess)

    elif method=='entropic':

        indices = ((q != 0) & (y != 0) & (beta / y > 0))
        dist_val = np.sum((1.0 / q[indices]) * (beta[indices] * np.log(beta[indices] / y[indices]) - beta[indices] + y[indices]))

        dist_grad = np.zeros(len(beta))
        dist_grad[indices] = np.log(beta[indices] / y[indices]) / q[indices]

        indices = ((q != 0) & (beta !=0))
        dist_hess = np.zeros(len(beta))
        dist_hess[indices] = 1.0 / (q[indices] * beta[indices])
        dist_hess = np.diag(dist_hess)

    elif method == 'logit':

        indices = ((q != 0) & (y != l) & (y != h) & ((beta - l) / (y - l) > 0) & ((h - beta) / (h - y) > 0))
        dist_val = np.sum((1.0 / q[indices]) * ( \
            (beta[indices] - l[indices]) * np.log((beta[indices] - l[indices]) / (y[indices] - l[indices])) + \
            (h[indices] - beta[indices]) * np.log((h[indices] - beta[indices]) / (h[indices] - y[indices]))))

        dist_grad = np.zeros(len(beta))
        dist_grad = (1.0 / q[indices]) * ( \
            np.log((beta[indices] - l[indices]) / (y[indices] - l[indices])) - \
            np.log((h[indices] - beta[indices]) / (h[indices] - y[indices])))

        indices = ((q != 0) & (beta != l) & (beta != h))
        dist_hess = np.zeros(len(beta))
        dist_hess = (1.0 / q[indices]) * (1.0 / (beta[indices] - l[indices]) + 1.0 / (h[indices] - beta[indices]))
        dist_hess = np.diag(dist_hess)

    return (dist_val, dist_grad, dist_hess)

