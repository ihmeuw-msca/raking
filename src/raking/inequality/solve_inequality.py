"""Solve the raking problem with inequality constraints using an interior point method."""

import numpy as np
import scipy.sparse as sps

from scipy.sparse.linalg import cg

from raking.inequality.compute_distance import get_gradient, get_conjugate_gradient, get_conjugate_hessian
from raking.raking_methods import raking_chi2, raking_entropic, raking_logit

def centering_step(
    y: np.ndarray,
    A: sps.csc_matrix,
    s: np.ndarray,
    C: sps.csc_matrix,
    c: np.ndarray,
    q: np.ndarray,
    mu: float,
    lambda0: np.ndarray,
    nu0: np.ndarray,
    method: str = 'chi2',
    l: np.ndarray | None = None,
    h: np.ndarray | None = None,
    epsilon: float = 1.0e-11,
    N: int = 100,
    gamma_iter: float = 0.9,
    num_iter: int = 100
) -> tuple[np.ndarray, float, int]:
    """
    Implement Newton method to solve min lambda^T c + nu^T s + f(−C^T lambda − A^T nu ; y) − mu 1^T log \lambda.
    """
    n = 0
    x = np.concatenate((lambda0, nu0), axis=0)
    unit = np.concatenate((np.ones(len(lambda0)), np.zeros(len(nu0))), axis=0)
    b = np.concatenate((c, s), axis=0)
    B = sps.vstack([A, C])
    z = - B.transpose() @ x
    gradient = get_conjugate_gradient(z, y, q, method, l, h)
    F = b - B @ gradient - mu * unit / x
    res = np.sqrt(F.transpose() @ F)
    while (res > epsilon) & (n < N):
        hessian = get_conjugate_hessian(z, y, q, method, l, h)
        J = (B.multiply(hessian.diagonal())) @ B.transpose() + mu * unit / np.square(x)
        Delta_x = cg(J, F)[0]
        m = 0
        i = 0
        x_n = x - 2.0 ** (-m) * Delta_x
        lambda_n = x[0:len(lambda0)]
        z_n = - B.transpose() @ x_n
        gradient_n = get_conjugate_gradient(z_n, y, q, method, l, h)
        F_n = b - B @ gradient_n - mu * unit / x_n
        res_n = np.sqrt(F_n.transpose() @ F_n)
        while (i < num_iter) & ((res_n > (1.0 - gamma_iter * 2.0 ** (-m)) * res) | (np.any(lambda_n < 0))):
            m = m + 1
            x_n = x - 2.0 ** (-m) * Delta_x
            lambda_n = x[0:len(lambda0)]
            z_n = - B.transpose() @ x_n
            gradient_n = get_conjugate_gradient(z_n, y, q, method, l, h)
            F_n = b - B @ gradient_n - mu * unit / x_n
            res_n = np.sqrt(F_n.transpose() @ F_n)
            i = i + 1
        x = x - 2.0 ** (-m) * Delta_x
        z = - B.transpose() @ x
        gradient = get_conjugate_gradient(z, y, q, method, l, h)
        F = b - B @ gradient - mu * unit / x
        res = np.sqrt(F.transpose() @ F)
        n = n + 1
    return (x, res, n)

def barrier_method(
    y: np.ndarray,
    A: sps.csc_matrix,
    s: np.ndarray,
    C: sps.csc_matrix,
    c: np.ndarray,
    q: np.ndarray,
    lambda0: np.ndarray,
    nu0: np.ndarray,
    method: str = 'chi2',
    l: np.ndarray | None = None,
    h: np.ndarray | None = None,
    epsilon: float = 1.0e-11,
    mu: float = 1.0,
    gamma: float = 0.1,
    N: int = 100,
    gamma_iter: float = 0.9,
    num_iter: int = 100
) -> tuple[np.ndarray, float]:
    """
    Implement barrier method to solve min lambda^T c + nu^T s + f(−C^T lambda − A^T nu ; y) s.t. lambda > 0.
    """
    m = len(lambda0)
    k = len(nu0)
    x = np.concatenate((lambda0, nu0), axis=0)
    while len(x) * mu > epsilon:
        (x, res, n) = centering_step(y, A, s, C, c, q, mu, lambda0, nu0, method, l, h, epsilon, N, gamma_iter, num_iter)
        lambda0 = x[0:m]
        nu0 = x[m:(m + k)]
        mu = gamma * mu
    return (x, mu)

def solve(
    y: np.ndarray,
    A: np.ndarray,
    s: np.ndarray,
    C: np.ndarray,
    c: np.ndarray,
    q: np.ndarray,
    method: str = 'chi2',
    l: np.ndarray | None = None,
    h: np.ndarray | None = None,
    epsilon: float = 1.0e-11,
    mu: float = 1.0,
    gamma: float = 0.1,
    N: int = 100,
    gamma_iter: float = 0.9,
    num_iter: int = 100
) -> tuple[np.ndarray]:
    """
    Initialize and launch the barrier method with Newton centering step.
    """
    # Transform A and C into sps.csc_matrix
    k = A.shape()[0]
    m = C.shape()[0]
    A = sps.csc_matrix(A)
    C = sps.csc_matrix(C)
    # Get output of raking with equality constraints to start the algorithm
    if method == 'chi2':
        (beta0, nu0) = raking_chi2(y, A, s, q)
    if method == 'entropic':
        (beta0, nu0, iter_eps) = raking_entropic(y, A, s, q, gamma_iter, N)
    if method == 'logit':
        (beta0, nu0, iter_eps) = raking_logit(y, A, s, l, h, q, gamma_iter, N)
    gradient = get_gradient(beta0, y, q, method, l, h)
    B = sps.vstack([A, C])
    x0 = cg(B.transpose(), - gradient)[0]
    lambda0 = x0[0:m]
    nu0 = x0[m:(m + k)]
    # Run the barrier method and transform back into the primal
    (x, mu) = barrier_method(y, A, s, C, c, q, lambda0, nu0, method, l, h, epsilon, mu, gamma, N, gamma_iter, num_iter)
    z = - B.transpose() @ x
    beta = get_conjugate_gradient(z, y, q, method, l, h)
    return beta

