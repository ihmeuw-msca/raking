"""Module to compute distances with their gradient and Hessian"""

import numpy as np


def compute_distance(beta, y, method, q, l, h, alpha):
    """ """
    if method == "chi2":
        distance = np.sum(np.square(beta - y) / (2.0 * q * y))
        gradient = beta / y - 1.0
        hessian_beta = 1.0 / y
        hessian_y = -beta / np.square(y)

    if method == "entropic":
        distance = np.sum(beta * np.log(beta / y) + beta - y)

    return (distance, gradient, hessian_beta, hessian_y)
