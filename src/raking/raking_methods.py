"""Module with methods to solve the raking problem"""

import numpy as np
from scipy.sparse.linalg import cg

def raking_chi2(
    y: np.ndarray,
    A: np.ndarray,
    s: np.ndarray,
    q: np.ndarray = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Raking using the chi2 distance f(beta, y) = (beta - y)^2 / 2y.

    This will solve the problem:
        min_beta 1/q f(beta, y) s.t. A beta = s

    Parameters
    ----------
    y : np.ndarray
        Vector of observations
    A: np.ndarray
        Constraints matrix (output of a function from the compute_constraints module)
    s: np.ndarray
        Margin vector (output of a function from the compute_constraints module)
    q: np.ndarray
        Vector of weights (default to all 1)

    Returns
    -------
    beta: np.ndarray
        Vector of reaked values
    lambda_k: np.ndarray
        Dual (needed for th uncertainty computation)
    """
    assert isinstance(y, np.ndarray), \
        'The vector of observations should be a Numpy array.'
    assert len(y.shape) == 1, \
        'The vector of observations should be a 1D Numpy array.'
    if q is not None:
        assert isinstance(q, np.ndarray), \
            'The vector of weights should be a Numpy array.'
        assert len(y.shape) == 1, \
            'The vector of weights should be a 1D Numpy array.'
        assert len(y) == len(q), \
            'Observations and weights vectors should have the same length.'
    assert isinstance(A, np.ndarray), \
        'The constraint matrix should be a Numpy array.'
    assert len(A.shape) == 2, \
        'The constraints matrix should be a 2D Numpy array.'
    assert isinstance(s, np.ndarray), \
        'The margins vector should be a Numpy array.'
    assert len(s.shape) == 1, \
        'The margins vector should be a 1D Numpy array.'
    assert np.shape(A)[0] == len(s), \
        'The number of linear constraints should be equal to the number of margins.'
    assert np.shape(A)[1] == len(y), \
        'The number of coefficients for the linear constraints should be equal to the number of observations.'

    if q == None:
        q = np.ones(len(y))
    s_hat = np.matmul(A, y)
    Phi = np.matmul(A, np.transpose(A * y * q))
    lambda_k = cg(Phi, s_hat - s)[0]
    beta = y * (1 - q * np.matmul(np.transpose(A), lambda_k))
    return (beta, lambda_k)

def raking_entropic(
    y: np.ndarray,
    A: np.ndarray,
    s: np.ndarray,
    q: np.ndarray = None,
    gamma0: float = 1.0,
    max_iter: int = 500
) -> tuple[np.ndarray, np.ndarray, int]:
    """Raking using the entropic distance f(beta, y) = beta log(beta/y) + y - beta.

    This will solve the problem:
        min_beta 1/q f(beta, y) s.t. A beta = s

    Parameters
    ----------
    y : np.ndarray
        Vector of observations
    A: np.ndarray
        Constraints matrix (output of a function from the compute_constraints module)
    s: np.ndarray
        Margin vector (output of a function from the compute_constraints module)
    q: np.ndarray
        Vector of weights (default to all 1)
    gamma0: float
        Initial value for line search
    max_iter: int
        Number of iterations for Newton's root finding method

    Returns
    -------
    beta: np.ndarray
        Vector of reaked values
    lambda_k: np.ndarray
        Dual (needed for th uncertainty computation)
    iters_eps: int
        Number of iterations until convergence
    """
    assert isinstance(y, np.ndarray), \
        'The vector of observations should be a Numpy array.'
    assert len(y.shape) == 1, \
        'The vector of observations should be a 1D Numpy array.'
    if q is not None:
        assert isinstance(q, np.ndarray), \
            'The vector of weights should be a Numpy array.'
        assert len(y.shape) == 1, \
            'The vector of weights should be a 1D Numpy array.'
        assert len(y) == len(q), \
            'Observations and weights vectors should have the same length.'
    assert isinstance(A, np.ndarray), \
        'The constraint matrix should be a Numpy array.'
    assert len(A.shape) == 2, \
        'The constraints matrix should be a 2D Numpy array.'
    assert isinstance(s, np.ndarray), \
        'The margins vector should be a Numpy array.'
    assert len(s.shape) == 1, \
        'The margins vector should be a 1D Numpy array.'
    assert np.shape(A)[0] == len(s), \
        'The number of linear constraints should be equal to the number of margins.'
    assert np.shape(A)[1] == len(y), \
        'The number of coefficients for the linear constraints should be equal to the number of observations.'

    if q == None:
        q = np.ones(len(y))
    s_hat = np.matmul(A, y)
    lambda_k = np.zeros(A.shape[0])
    beta = np.copy(y)
    epsilon = 1.0
    iter_eps = 0
    while (epsilon > 1.0e-10) & (iter_eps < max_iter):
        Phi = np.matmul(A, y * (1.0 - np.exp(- q * np.matmul(np.transpose(A), lambda_k))))
        D = np.diag(y * q * np.exp(- q * np.matmul(np.transpose(A), lambda_k)))
        J = np.matmul(np.matmul(A, D), np.transpose(A))
        delta_lambda = cg(J, Phi - s_hat + s)[0]
        gamma = gamma0
        iter_gam = 0
        lambda_k = lambda_k - gamma * delta_lambda
        beta = y * np.exp(- q * np.matmul(np.transpose(A), lambda_k))
        if iter_eps > 0:
            while (np.mean(np.abs(s - np.matmul(A, beta))) > epsilon) & \
                    (iter_gam < max_iter):
                gamma = gamma / 2.0
                iter_gam = iter_gam + 1
                lambda_k = lambda_k - gamma * delta_lambda
                beta = y * np.exp(- q * np.matmul(np.transpose(A), lambda_k))
        epsilon = np.mean(np.abs(s - np.matmul(A, beta)))
        iter_eps = iter_eps + 1
    return (beta, lambda_k, iter_eps)

