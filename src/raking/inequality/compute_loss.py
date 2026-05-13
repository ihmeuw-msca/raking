"""Compute objective, gradient and Hessian of the loss functions and their conjugate."""

import numpy as np
import scipy.sparse as sps

from math import sqrt

def get_loss(
    x: np.ndarray,
    penalty: float,
    loss: str = 'logit'
) -> float:
    """
    Compute the loss function (scalar).

    Parameters
    ----------
    x : np.ndarray
        Terms that must be negative in the inequality constraints
    penalty : float
        Scaling parameter for the loss function
    loss : string
        Loss function (hinge or logit)

    Returns
    -------
    float : Loss function
    """
    assert loss in ['hinge', 'logit'], 'The loss function must be hinge or logit.'

    if loss == 'hinge':
        return np.sum(np.power(np.maximum(0.0, - penalty * x), 3.0))

    if loss == 'logit':
        return np.sum(np.log(1.0 + np.exp(- penalty * x)))

def get_loss_gradient(
    x: np.ndarray,
    penalty: float,
    loss: str = 'logit'
) -> np.ndarray:
    """
    Compute the gradient of the loss function (1D array).

    Parameters
    ----------
    x : np.ndarray
        Terms that must be negative in the inequality constraints
    penalty : float
        Scaling parameter for the loss function
    loss : string
        Loss function (hinge or logit)

    Returns
    -------
    np.ndarray : Gradient of the loss function with respect to x
    """
    assert loss in ['hinge', 'logit'], 'The loss function must be hinge or logit.'

    if loss == 'hinge':
        return - 3.0 * penalty * np.square(np.maximum(0.0, - penalty * x))

    if loss == 'logit':
        return - penalty / (1.0 + np.exp(penalty * x))

def get_loss_hessian(
    x: np.ndarray,
    penalty: float,
    loss: str = 'logit'
) -> sps.csc_matrix:
    """
    Compute the Hessian of the loss function (2D sparse array).

    Parameters
    ----------
    x : np.ndarray
        Terms that must be negative in the inequality constraints
    penalty : float
        Scaling parameter for the loss function
    loss : string
        Loss function (hinge or logit)

    Returns
    -------
    hessian : sps.csc_matrix
        Hessian of the loss function with respect to x
    """
    assert loss in ['hinge', 'logit'], 'The loss function must be hinge or logit.'

    m = len(x)
    row = np.arange(0, m)

    if loss == 'hinge':
        data = 6.0 * (penalty**2.0) * np.maximum(0, - penalty * x)

    if loss == 'logit':
        data = (penalty**2.0) * np.exp(penalty * x) / np.square(1.0 + np.exp(penalty * x))

    hessian = sps.csc_matrix((data, (row, row)), shape=(m, m))
    return hessian

def get_conjugate_loss(
    z: np.ndarray,
    penalty: float,
    loss: str = 'logit'
) -> float:
    """
    Compute the conjugate of the loss function (scalar).

    Parameters
    ----------
    z : np.ndarray
        Conjugate of the terms that must be negative in the inequality constraints
    penalty : float
        Scaling parameter for the loss function
    loss : string
        Loss function (hinge or logit)

    Returns
    -------
    float : Conjugate of the loss function
    """
    assert loss in ['hinge', 'logit'], 'The loss function must be hinge or logit.'

    if loss == 'hinge':
        assert np.all(z < 0.0), 'The conjugate must be negative.'
        return np.sum((2.0 / sqrt(3.0)) * np.power(- z / penalty, 3.0 / 2.0))

    if loss == 'logit':
        assert np.all((z < 0.0) & (z > - penalty)), 'The conjugate must be negative and higher than -penalty.'
        return np.sum((3.0 / penalty) * np.log(- (z + penalty) / z) - np.log(penalty / (penalty + z)))

def get_conjugate_loss_gradient(
    z: np.ndarray,
    penalty: float,
    loss: str = 'logit'
) -> np.ndarray:
    """
    Compute the gradient of the conjugate of the loss function (1D array).

    Parameters
    ----------
    z : np.ndarray
        Conjugate of the terms that must be negative in the inequality constraints
    penalty : float
        Scaling parameter for the loss function
    loss : string
        Loss function (hinge or logit)

    Returns
    -------
    np.ndarray : Gradient of the conjugate of the loss function with respect to z
    """
    assert loss in ['hinge', 'logit'], 'The loss function must be hinge or logit.'

    if loss == 'hinge':
        assert np.all(z < 0.0), 'The conjugate must be negative.'
        return - (sqrt(3.0) / penalty) * np.sqrt(- z / penalty)

    if loss == 'logit':
        assert np.all((z < 0.0) & (z > - penalty)), 'The conjugate must be negative and higher than -penalty.'
        return np.log(- (z + penalty) / z) / penalty

def get_conjugate_loss_hessian(
    z: np.ndarray,
    penalty: float,
    loss: str = 'logit'
) -> sps.csc_matrix:
    """
    Compute the Hessian of the conjugate of the loss function (2D sparse array).

    Parameters
    ----------
    z : np.ndarray
        Conjugate of the terms that must be negative in the inequality constraints
    penalty : float
        Scaling parameter for the loss function
    loss : string
        Loss function (hinge or logit)

    Returns
    -------
    hessian : sps.csc_matrix
        Hessian of the conjugate of the loss function with respect to z
    """
    assert loss in ['hinge', 'logit'], 'The loss function must be hinge or logit.'

    m = len(z)
    row = np.arange(0, m)

    if loss == 'hinge':
        assert np.all(z < 0.0), 'The conjugate must be negative.'
        data = (sqrt(3.0) / (2.0 * (penalty**2.0))) * np.sqrt(- penalty / z)

    if loss == 'logit':
        assert np.all((z < 0.0) & (z > - penalty)), 'The conjugate must be negative and higher than -penalty.'
        data = - 1.0 / (z * (z + penalty))

    hessian = sps.csc_matrix((data, (row, row)), shape=(m, m))
    return hessian

