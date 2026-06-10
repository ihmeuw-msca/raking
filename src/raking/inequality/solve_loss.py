"""Solve the raking problem with penalty loss using an interior point method."""

import numpy as np
import scipy.sparse as sps

from scipy.sparse.linalg import cg

from raking.inequality.compute_distance import get_gradient, get_conjugate_gradient, get_conjugate_hessian
from raking.inequality.compute_loss import get_conjugate_loss_gradient, get_conjugate_loss_hessian
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
    loss: str = 'logit',
    penalty: float = 1.0,
    l: np.ndarray | None = None,
    h: np.ndarray | None = None,
    epsilon: float = 1.0e-11,
    N: int = 100,
    gamma_iter: float = 0.9,
    num_iter: int = 100
) -> tuple[np.ndarray, float, int]:
    """
    Implement Newton method to solve min lambda^T c + nu^T s + f(−C^T lambda − A^T nu ; y)
    + L(lambda) - mu 1^T log(-lambda) − mu 1^T log(lambda + penalty).
    """
    n = 0
    x = np.concatenate((lambda0, nu0), axis=0)
    lambda_ = x[0:len(lambda0)]
    unit = np.concatenate((- 1.0 / lambda0, np.zeros(len(nu0))), axis=0)
    if loss == 'logit':
        unit_penalty = np.concatenate((1.0 / (lambda0 + penalty), np.zeros(len(nu0))), axis=0)
    b = np.concatenate((c, s), axis=0)
    B = sps.vstack([C, A])
    z = - B.transpose() @ x
    gradient = get_conjugate_gradient(z, y, q, method, l, h)
    gradient_loss = get_conjugate_loss_gradient(lambda0, penalty, loss)
    gradient_loss = np.concatenate((gradient_loss, np.zeros(len(nu0))), axis=0)
    F = b - B @ gradient + gradient_loss - mu * unit
    if loss == 'logit':
        F = F - mu * unit_penalty
    res = np.sqrt(F.transpose() @ F)
    while (res > epsilon) & (n < N):
        hessian = get_conjugate_hessian(z, y, q, method, l, h)
        hessian_loss = get_conjugate_loss_hessian(lambda_, penalty, loss)
        hessian_loss = sps.vstack( \
            [sps.hstack([hessian_loss, sps.csr_matrix(np.zeros((len(lambda0), len(nu0))))]), \
            sps.csr_matrix(np.zeros((len(nu0), len(lambda0) + len(nu0))))])
        unit_square = np.concatenate((- 1.0 / np.square(lambda_), np.zeros(len(nu0))), axis=0)
        J = (B.multiply(hessian.diagonal())) @ B.transpose() + hessian_loss + mu * unit_square
        if loss == 'logit':
            unit_penalty_square = np.concatenate((1.0 / np.square(lambda0 + penalty), np.zeros(len(nu0))), axis=0)
            J = J + mu * unit_penalty_square
        Delta_x = cg(J, F)[0]
        m = 0
        i = 0
        x_n = x - 2.0 ** (-m) * Delta_x
        lambda_n = x[0:len(lambda0)]
        unit_n = np.concatenate((- 1.0 / lambda_n, np.zeros(len(nu0))), axis=0)
        if loss == 'logit':
            unit_penalty_n = np.concatenate((1.0 / (lambda_n + penalty), np.zeros(len(nu0))), axis=0)
        z_n = - B.transpose() @ x_n
        gradient_n = get_conjugate_gradient(z_n, y, q, method, l, h)
        gradient_loss_n = get_conjugate_loss_gradient(lambda_n, penalty, loss)
        gradient_loss_n = np.concatenate((gradient_loss_n, np.zeros(len(nu0))), axis=0)
        F_n = b - B @ gradient_n + gradient_loss_n - mu * unit_n
        if loss == 'logit':
            F_n = F_n - mu * unit_penalty_n
        res_n = np.sqrt(F_n.transpose() @ F_n)
        while (i < num_iter) & ((res_n > (1.0 - gamma_iter * 2.0 ** (-m)) * res) | (np.any(lambda_n < 0))):
            m = m + 1
            x_n = x - 2.0 ** (-m) * Delta_x
            lambda_n = x[0:len(lambda0)]
            unit_n = np.concatenate((- 1.0 / lambda_n, np.zeros(len(nu0))), axis=0)
            if loss == 'logit':
                unit_penalty_n = np.concatenate((1.0 / (lambda_n + penalty), np.zeros(len(nu0))), axis=0)
            z_n = - B.transpose() @ x_n
            gradient_n = get_conjugate_gradient(z_n, y, q, method, l, h)
            gradient_loss_n = get_conjugate_loss_gradient(lambda_n, penalty, loss)
            gradient_loss_n = np.concatenate((gradient_loss_n, np.zeros(len(nu0))), axis=0)
            F_n = b - B @ gradient_n + gradient_loss_n - mu * unit_n
            if loss == 'logit':
                F_n = F_n - mu * unit_penalty_n
            res_n = np.sqrt(F_n.transpose() @ F_n)
            i = i + 1
        x = x - 2.0 ** (-m) * Delta_x
        lambda_ = x[0:len(lambda0)]
        unit = np.concatenate((- 1.0 / lambda_, np.zeros(len(nu0))), axis=0)
        if loss == 'logit':
            unit_penalty = np.concatenate((1.0 / (lambda_ + penalty), np.zeros(len(nu0))), axis=0)
        z = - B.transpose() @ x
        gradient = get_conjugate_gradient(z, y, q, method, l, h)
        gradient_loss = get_conjugate_loss_gradient(lambda_, penalty, loss)
        gradient_loss = np.concatenate((gradient_loss, np.zeros(len(nu0))), axis=0)
        F = b - B @ gradient + gradient_loss - mu * unit
        if loss == 'logit':
            F = F - mu * unit_penalty
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
    loss: str = 'logit',
    penalty: float = 1.0,
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
    Implement barrier method to solve min lambda^T c + nu^T s + f(−C^T lambda − A^T nu ; y) + L(lambda) s.t. lambda in I.
    """
    m = len(lambda0)
    k = len(nu0)
    x = np.concatenate((lambda0, nu0), axis=0)
    while len(x) * mu > epsilon:
        (x, res, n) = centering_step(y, A, s, C, c, q, mu, lambda0, nu0, method, loss, penalty, l, h, epsilon, N, gamma_iter, num_iter)
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
    loss: str = 'logit',
    penalty: float = 1.0,
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
    # Get output of raking with equality constraints to start the algorithm
    if method == 'chi2':
        (beta0, nu0) = raking_chi2(y, A, s, q)
    if method == 'entropic':
        (beta0, nu0, iter_eps) = raking_entropic(y, A, s, q, gamma_iter, N)
    if method == 'logit':
        (beta0, nu0, iter_eps) = raking_logit(y, A, s, l, h, q, gamma_iter, N)
    # Transform A and C into sps.csc_matrix
    k = A.shape[0]
    m = C.shape[0]
    A = sps.csc_matrix(A)
    C = sps.csc_matrix(C)
    B = sps.vstack([C, A])
    # Initialize lambda with value in the interval (-penalty, 0)
    lambda0 = -0.5 * penalty * np.ones(m)
    # Run the barrier method and transform back into the primal
    (x, mu) = barrier_method(y, A, s, C, c, q, lambda0, nu0, method, loss, penalty, l, h, epsilon, mu, gamma, N, gamma_iter, num_iter)
    z = - B.transpose() @ x
    beta = get_conjugate_gradient(z, y, q, method, l, h)
    lambda_ = x[0:m]
    zeta = get_conjugate_loss_gradient(lambda_, penalty, loss)
    return (beta, zeta)

