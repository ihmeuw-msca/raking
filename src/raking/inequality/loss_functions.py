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

        dist_val = np.sum(np.square( \
            beta[(q != 0) & (y != 0)] - y[(q != 0) & (y != 0)]) / \
            (2.0 * q[(q != 0) & (y != 0)] * y[(q != 0) & (y != 0)]))

        dist_grad = np.zeros(len(beta))
        dist_grad[(q != 0) & (y != 0)] = (beta[(q != 0) & (y != 0)] / \
            y[(q != 0) & (y != 0)] - 1.0) / q[(q != 0) & (y != 0)]

        dist_hess = np.zeros(len(beta))
        dist_hess[(q != 0) & (y != 0)] = 1.0 / \
            (q[(q != 0) & (y != 0)] * y[(q != 0) & (y != 0)])
        dist_hess = np.diag(dist_hess)

    elif method=='entropic':

        dist_val = np.sum((1.0 / q[(q != 0) & (y != 0)]) * ( \
            beta[(q != 0) & (y != 0)] * np.log(beta[(q != 0) & (y != 0)] / \
            y[(q != 0) & (y != 0)]) - beta[(q != 0) & (y != 0)] + y[(q != 0) & (y != 0)]))

        dist_grad = np.zeros(len(beta))
        dist_grad[(q != 0) & (y != 0)] = np.log(beta[(q != 0) & (y != 0)] / \
            y[(q != 0) & (y != 0)]) / q[(q != 0) & (y != 0)]

        dist_hess = np.zeros(len(beta))
        dist_hess[(q != 0) & (beta != 0)] = 1.0 / (q[(q != 0) & (beta != 0)] * \
            beta[(q != 0) & (beta != 0)])
        dist_hess = np.diag(dist_hess)

    elif method == 'logit':

        dist_val = np.sum((1.0 / q[(beta != l) & (beta != h) & (y != l) & (y != h)]) * \
            ((beta[(beta != l) & (beta != h) & (y != l) & (y != h)] - l[(beta != l) & (beta != h) & (y != l) & (y != h)]) * \
            np.log((beta[(beta != l) & (beta != h) & (y != l) & (y != h)] - l[(beta != l) & (beta != h) & (y != l) & (y != h)]) / \
            (y[(beta != l) & (beta != h) & (y != l) & (y != h)] - l[(beta != l) & (beta != h) & (y != l) & (y != h)])) + \
            (h[(beta != l) & (beta != h) & (y != l) & (y != h)] - beta[(beta != l) & (beta != h) & (y != l) & (y != h)]) * \
            np.log((h[(beta != l) & (beta != h) & (y != l) & (y != h)] - beta[(beta != l) & (beta != h) & (y != l) & (y != h)]) / \
            (h[(beta != l) & (beta != h) & (y != l) & (y != h)] - y[(beta != l) & (beta != h) & (y != l) & (y != h)]))))

        dist_grad = np.zeros(len(beta))
        dist_grad = (1.0 / q[(beta != l) & (beta != h) & (y != l) & (y != h)]) * ( \
            np.log((beta[(beta != l) & (beta != h) & (y != l) & (y != h)] - l[(beta != l) & (beta != h) & (y != l) & (y != h)]) / \
            (y[(beta != l) & (beta != h) & (y != l) & (y != h)] - l[(beta != l) & (beta != h) & (y != l) & (y != h)])) - \
            np.log((h[(beta != l) & (beta != h) & (y != l) & (y != h)] - beta[(beta != l) & (beta != h) & (y != l) & (y != h)]) / \
            (h[(beta != l) & (beta != h) & (y != l) & (y != h)] - y[(beta != l) & (beta != h) & (y != l) & (y != h)])))

        dist_hess = np.zeros(len(beta))
        dist_hess = (1.0 / q[(beta != l) & (beta != h)]) * ( \
            1.0 / (beta[(beta != l) & (beta != h)] - l[(beta != l) & (beta != h)]) + \
            1.0 / (h[(beta != l) & (beta != h)] - beta[(beta != l) & (beta != h)]))
        dist_hess = np.diag(dist_hess)

    return (dist_val, dist_grad, dist_hess)

