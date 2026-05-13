"""Compute objective, gradient and Hessian of the distance functions and their conjugate."""

import numpy as np
import scipy.sparse as sps

def get_objective(
    beta: np.ndarray,
    y: np.ndarray,
    q: np.ndarray,
    method: str = 'chi2',
    l: np.ndarray | None = None,
    h: np.ndarray | None = None
) -> float:
    """
    Compute the distance function (scalar).

    Parameters
    ----------
    beta : np.ndarray
        Raked values
    y : np.ndarray
        Initial observations
    q : np.ndarray
        Inverse of the weights
    method : string
        Distance function (chi2, entropic or logit)
    l : np.ndarray
        Lower bounds for the observations and raked values (if using logit)
    h : np.ndarray
        Upper bounds for the observations and raked values (if using logit)

    Returns
    -------
    float : Objective function
    """
    assert method in ['chi2', 'entropic', 'logit'], 'The distance function must be chi2, entropic or logit.'

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

def get_gradient(
    beta: np.ndarray,
    y: np.ndarray,
    q: np.ndarray,
    method: str = 'chi2',
    l: np.ndarray | None = None,
    h: np.ndarray | None = None
) -> np.ndarray:
    """
    Compute the gradient of the distance function (1D array).

    Parameters
    ----------
    beta : np.ndarray
        Raked values
    y : np.ndarray
        Initial observations
    q : np.ndarray
        Inverse of the weights
    method : string
        Distance function (chi2, entropic or logit)
    l : np.ndarray
        Lower bounds for the observations and raked values (if using logit)
    h : np.ndarray
        Upper bounds for the observations and raked values (if using logit)

    Returns
    -------
    gradient : np.ndarray
        Gradient of the objective function with respect to beta
    """
    assert method in ['chi2', 'entropic', 'logit'], 'The distance function must be chi2, entropic or logit.'

    gradient = np.zeros(len(beta))

    if method == 'chi2':
        indices = ((q != 0) & (y != 0))
        gradient[indices] = (beta[indices] / y[indices] - 1.0) / q[indices]

    if method == 'entropic':
        indices = ((q != 0) & (y != 0) & (beta / y > 0))
        gradient[indices] = np.log(beta[indices] / y[indices]) / q[indices]

    if method == 'logit':
        indices = ((q != 0) & (y != l) & (y != h) & ((beta - l) / (y - l) > 0) & ((h - beta) / (h - y) > 0))
        gradient = (1.0 / q[indices]) * ( \
            np.log((beta[indices] - l[indices]) / (y[indices] - l[indices])) - \
            np.log((h[indices] - beta[indices]) / (h[indices] - y[indices])))

    return gradient
            
def get_hessian(
    beta: np.ndarray,
    y: np.ndarray,
    q: np.ndarray,
    method: str = 'chi2',
    l: np.ndarray | None = None,
    h: np.ndarray | None = None
) -> sps.csc_matrix:
    """
    Compute the Hessian of the distance function (2D sparse array).

    Parameters
    ----------
    beta : np.ndarray
        Raked values
    y : np.ndarray
        Initial observations
    q : np.ndarray
        Inverse of the weights
    method : string
        Distance function (chi2, entropic or logit)
    l : np.ndarray
        Lower bounds for the observations and raked values (if using logit)
    h : np.ndarray
        Upper bounds for the observations and raked values (if using logit)

    Returns
    -------
    hessian : sps.csc_matrix
        Hessian of the objective function with respect to beta
    """
    assert method in ['chi2', 'entropic', 'logit'], 'The distance function must be chi2, entropic or logit.'

    p = len(q)

    if method == 'chi2':
        indices = ((q != 0) & (y != 0))
        row = np.arange(0, p)[indices]
        data = 1.0 / (q[indices] * y[indices])

    if method == 'entropic':
        indices = ((q != 0) & (beta !=0))
        row = np.arange(0, p)[indices]
        data = 1.0 / (q[indices] * beta[indices])

    if method == 'logit':
        indices = ((q != 0) & (beta != l) & (beta != h))
        row = np.arange(0, p)[indices]
        data = (1.0 / q[indices]) * (1.0 / (beta[indices] - l[indices]) + 1.0 / (h[indices] - beta[indices]))

    hessian = sps.csc_matrix((data, (row, row)), shape=(p, p))
    return hessian

def get_both(
    beta: np.ndarray,
    y: np.ndarray,
    q: np.ndarray,
    method: str = 'chi2',
    l: np.ndarray | None = None,
    h: np.ndarray | None = None
) -> sps.csc_matrix:
    """
    Compute the derivative of the distance function with respect to beta and y (2D sparse array).

    Parameters
    ----------
    beta : np.ndarray
        Raked values
    y : np.ndarray
        Initial observations
    q : np.ndarray
        Inverse of the weights
    method : string
        Distance function (chi2, entropic or logit)
    l : np.ndarray
        Lower bounds for the observations and raked values (if using logit)
    h : np.ndarray
        Upper bounds for the observations and raked values (if using logit)

    Returns
    -------
    hessian : sps.csc_matrix
        Hessian of the objective function with respect to beta and y
    """
    assert method in ['chi2', 'entropic', 'logit'], 'The distance function must be chi2, entropic or logit.'

    p = len(q)

    if method == 'chi2':
        indices = ((q != 0) & (y != 0))
        row = np.arange(0, p)[indices]
        data = - beta[indices] / (q[indices] * np.square(y[indices]))

    if method == 'entropic':
        indices = ((q != 0) & (y != 0))
        row = np.arange(0, p)[indices]
        data = - 1.0 / (q[indices] * y[indices])

    if method == 'logit':
        indices = ((q != 0) & (y != l) & (y != h))
        row = np.arange(0, p)[indices]
        data = (1.0 / q[indices]) * ( - 1.0 / (y[indices] - l[indices]) - 1.0 / (h[indices] - y[indices]))

    hessian = sps.csc_matrix((data, (row, row)), shape=(p, p))
    return hessian

def get_conjugate_objective(
    z: np.ndarray,
    y: np.ndarray,
    q: np.ndarray,
    method: str = 'chi2',
    l: np.ndarray | None = None,
    h: np.ndarray | None = None
) -> float:
    """
    Compute the conjugate of the distance function (scalar).

    Parameters
    ----------
    z : np.ndarray
        Conjugate of the raked values beta
    y : np.ndarray
        Initial observations
    q : np.ndarray
        Inverse of the weights
    method : string
        Distance function (chi2, entropic or logit)
    l : np.ndarray
        Lower bounds for the observations and raked values (if using logit)
    h : np.ndarray
        Upper bounds for the observations and raked values (if using logit)

    Returns
    -------
    float : Conjugate of the objective function
    """
    assert method in ['chi2', 'entropic', 'logit'], 'The distance function must be chi2, entropic or logit.'

    if method == 'chi2':
        return np.sum(z * y * (1.0 + q * z / 2.0))

    if method == 'entropic':
        indices = (q != 0)
        return np.sum(y[indices] * (np.exp(q[indices] * z[indices]) - 1.0) / q[indices])

    if method == 'logit':
        indices = ((q != 0) & ((h - y) + (y - l) * np.exp(q * z) != 0) & ((h - l) / ((h - y) + (y - l) * np.exp(q * z)) > 0))
        return np.sum(l[indices] * z[indices] - ((h[indices] - l[indices]) / q[indices]) * np.log( \
            (h[indices] - l[indices]) / ((h[indices] - y[indices]) + (y[indices] - l[indices]) * np.exp(q[indices] * z[indices]))))

def get_conjugate_gradient(
    z: np.ndarray,
    y: np.ndarray,
    q: np.ndarray,
    method: str = 'chi2',
    l: np.ndarray | None = None,
    h: np.ndarray | None = None
) -> np.ndarray:
    """
    Compute the gradient of the conjugate of the distance function (1D array).

    Parameters
    ----------
    z : np.ndarray
        Conjugate of the raked values beta
    y : np.ndarray
        Initial observations
    q : np.ndarray
        Inverse of the weights
    method : string
        Distance function (chi2, entropic or logit)
    l : np.ndarray
        Lower bounds for the observations and raked values (if using logit)
    h : np.ndarray
        Upper bounds for the observations and raked values (if using logit)

    Returns
    -------
    gradient : np.ndarray
        Gradient of the conjugate of the objective function with respect to z
    """
    assert method in ['chi2', 'entropic', 'logit'], 'The distance function must be chi2, entropic or logit.'

    if method == 'chi2':
        return y * (1.0 + q * z)

    if method == 'entropic':
        return y * np.exp(q * z)

    if method == 'logit':
        indices = ((h - y) + (y - l) * np.exp(q * z) != 0)
        gradient = np.zeros(len(z))
        gradient[indices] = (l[indices] * (h[indices] - y[indices]) + h[indices] * (y[indices] - l[indices]) * np.exp(q[indices] * z[indices])) / \
            ((h[indices] - y[indices]) + (y[indices] - l[indices]) * np.exp(q[indices] * z[indices]))
        return gradient

def get_conjugate_hessian(
    z: np.ndarray,
    y: np.ndarray,
    q: np.ndarray,
    method: str = 'chi2',
    l: np.ndarray | None = None,
    h: np.ndarray | None = None
) -> sps.csc_matrix:
    """
    Compute the Hessian of the conjugate of the distance function (2D sparse array).

    Parameters
    ----------
    z : np.ndarray
        Conjugate of the raked values beta
    y : np.ndarray
        Initial observations
    q : np.ndarray
        Inverse of the weights
    method : string
        Distance function (chi2, entropic or logit)
    l : np.ndarray
        Lower bounds for the observations and raked values (if using logit)
    h : np.ndarray
        Upper bounds for the observations and raked values (if using logit)

    Returns
    -------
    hessian : np.ndarray
        Hessian of the conjugate of the objective function with respect to z
    """
    assert method in ['chi2', 'entropic', 'logit'], 'The distance function must be chi2, entropic or logit.'

    p = len(q)

    if method == 'chi2':
        row = np.arange(0, p)
        data = q * y

    if method == 'entropic':
        row = np.arange(0, p)
        data = q * y * np.exp(q * z)

    if method == 'logit':
        indices = ((h - y) + (y - l) * np.exp(q * z) != 0)
        row = np.arange(0, p)[indices]
        data = (q[indices] * (y[indices] - l[indices]) * (h[indices] - y[indices]) * (h[indices] - l[indices]) * np.exp(q[indices] * z[indices])) / \
            np.square((h[indices] - y[indices]) + (y[indices] - l[indices]) * np.exp(q[indices] * z[indices]))

    hessian = sps.csc_matrix((data, (row, row)), shape=(p, p))
    return hessian

