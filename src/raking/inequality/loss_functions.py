"""This module contains loss functions with their gradient and hessian"""

import numpy as np

def compute_loss(x, penalty, loss, order):
    """
    """
    if order == 'objective':
        if loss == 'hinge':
            return np.sum(np.power(np.maximum(0.0, - penalty * x), 3.0))
        if loss == 'logit':
            return np.sum(np.log(1.0 + np.exp(- penalty * x)))

    if order == 'gradient':
        if loss == 'hinge':
            return - 3.0 * penalty * np.square(np.maximum(0.0, - penalty * x))
        if loss == 'logit':
            return - penalty / (1.0 + np.exp(penalty * x))

    if order == 'hessian':
        if loss == 'hinge':
            return 6.0 * (penalty**2.0) * np.maximum(0, - penalty * x)
        if loss == 'logit':
            return (penalty**2.0) * np.exp(penalty * x) / np.square(1.0 + np.exp(penalty * x))

def compute_conjugate_loss(z, penalty, loss, order):
    """
    """
    if order == 'objective':
        if loss == 'hinge':
            return - 2.0 * np.power(- z / (3.0 * penalty), 3.0 / 2.0)
        if loss == 'logit':
            return (3.0 / penalty) * np.log(- (z + penalty) / z) - np.log(penalty / (penalty + z))

    if order == 'gradient':
        if loss == 'hinge':
            return np.sqrt(- z / (3.0 * penalty)) / penalty
        if loss == 'logit':
            return np.log(- (z + penalty) / z) / penalty

    if order == 'hessian':
        if loss == 'hinge':
            return - np.sqrt(- 3.0 * penalty / z) / (6.0 * penalty)
        if loss == 'logit':
            return - 1.0 / (z * (z + penalty))

def compute_dist(beta, y, q, method, order, l=None, h=None):
    """
    """
    if order == 'objective':
        if method == 'chi2':
            indices = ((q != 0) & (y != 0))
            return np.sum(np.square(beta[indices] - y[indices]) / (2.0 * q[indices] * y[indices]))
        if method=='entropic':
            indices = ((q != 0) & (y != 0) & (beta / y > 0))
            return np.sum((1.0 / q[indices]) * (beta[indices] * np.log(beta[indices] / y[indices]) - beta[indices] + y[indices]))
        if method == 'logit':
            indices = ((q != 0) & (y != l) & (y != h) & ((beta - l) / (y - l) > 0) & ((h - beta) / (h - y) > 0))
            return np.sum((1.0 / q[indices]) * ( \
                (beta[indices] - l[indices]) * np.log((beta[indices] - l[indices]) / (y[indices] - l[indices])) + \
                (h[indices] - beta[indices]) * np.log((h[indices] - beta[indices]) / (h[indices] - y[indices]))))

    if order == 'gradient':
        if method == 'chi2':
            indices = ((q != 0) & (y != 0))
            gradient = np.zeros(len(beta))
            gradient[indices] = (beta[indices] / y[indices] - 1.0) / q[indices]
            return gradient
        if method == 'entropic':
            indices = ((q != 0) & (y != 0) & (beta / y > 0))
            gradient = np.zeros(len(beta))
            gradient[indices] = np.log(beta[indices] / y[indices]) / q[indices]
            return gradient
        if method == 'logit':
            indices = ((q != 0) & (y != l) & (y != h) & ((beta - l) / (y - l) > 0) & ((h - beta) / (h - y) > 0))
            gradient = np.zeros(len(beta))
            gradient = (1.0 / q[indices]) * ( \
                np.log((beta[indices] - l[indices]) / (y[indices] - l[indices])) - \
                np.log((h[indices] - beta[indices]) / (h[indices] - y[indices])))
            return gradient
            
    if order == 'hessian':
        if method == 'chi2':
            indices = ((q != 0) & (y != 0))
            hessian = np.zeros(len(beta))
            hessian[indices] = 1.0 / (q[indices] * y[indices])
            return hessian
        if method == 'entropic':
            indices = ((q != 0) & (beta !=0))
            hessian = np.zeros(len(beta))
            hessian[indices] = 1.0 / (q[indices] * beta[indices])
            return hessian
        if method == 'logit':
            indices = ((q != 0) & (beta != l) & (beta != h))
            hessian = np.zeros(len(beta))
            hessian[indices] = (1.0 / q[indices]) * (1.0 / (beta[indices] - l[indices]) + 1.0 / (h[indices] - beta[indices]))
            return hessian

def compute_conjugate_dist(z, y, q, method, order, l=None, h=None):
    """
    """
    if order == 'objective':
        if method == 'chi2':
            return np.sum(z * y * (1.0 + q * z / 2.0))
        if method == 'entropic':
            indices = (q != 0)
            return np.sum(y[indices] * (np.exp(q[indices] * z[indices]) - 1.0) / q[indices])
        if method == 'logit':
            indices = ((q != 0) & ((h - y) + (y - l) * np.exp(q * z) != 0) & ((h - l) / ((h - y) + (y - l) * np.exp(q * z)) > 0))
            return np.sum(l[indices] * z[indices] - ((h[indices] - l[indices]) / q[indices]) * np.log( \
                (h[indices] - l[indices]) / ((h[indices] - y[indices]) + (y[indices] - l[indices]) * np.exp(q[indices] * z[indices]))))

    if order == 'gradient':
        if method == 'chi2':
            return y * (1.0 + q * z)
        if method == 'entropic':
            return y * np.exp(q * z)
        if method == 'logit':
            indices = ((h - y) + (y - l) * np.exp(q * z) != 0)
            gradient = np.zeros(z)
            gradient[indices] = (l[indices] * (h[indices] - y[indices]) + h[indices] * (y[indices] - l[indices]) * np.exp(q[indices] * z[indices])) / \
                ((h[indices] - y[indices]) + (y[indices] - l[indices]) * np.exp(q[indices] * z[indices]))
            return gradient

    if order == 'hessian':
        if method == 'chi2':
            return q * y
        if method == 'entropic':
            return q * y * np.exp(q * z)
        if method == 'logit':
            indices = ((h - y) + (y - l) * np.exp(q * z) != 0)
            hessian = np.zeros(z)
            hessian[indices] = (q[indices] * (y[indices] - l[indices]) * (h[indices] - y[indices]) * (h[indices] - l[indices]) * np.exp(q[indices] * z[indices])) / \
                np.square((h[indices] - y[indices]) + (y[indices] - l[indices]) * np.exp(q[indices] * z[indices]))
            return hessian

