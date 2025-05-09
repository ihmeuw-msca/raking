"""Module with methods to solve the raking problem with a penalty loss"""

import numpy as np

from scipy.sparse.linalg import cg

from raking.inequality.loss_functions import compute_loss, compute_dist

def raking_loss(
    y: np.ndarray,
    A: np.ndarray,
    s: np.ndarray,
    C: np.ndarray,
    c: np.ndarray,
    q: np.ndarray,
    method: str = 'chi2',
    loss: str = 'logit',
    penalty: float = 1.0,
    gamma0: float = 1.0,
    max_iter: int = 500,
):
    """
    """
    beta = np.copy(y)
    lambda_k = np.zeros(A.shape[0])
    sol_k = np.concatenate((beta, lambda_k))
    epsilon = 1.0
    iter_eps = 0
    while (epsilon > 1.0e-10) & (iter_eps < max_iter):
        (loss_val, loss_grad, loss_hess) = compute_loss(beta, C, c, loss)
        (dist_val, dist_grad, dist_hess) = compute_dist(beta, y, q, method)
        F1 = dist_grad + np.matmul(np.transpose(A), lambda_k) \
            - penalty * np.matmul(np.transpose(C), loss_grad)
        F2 = np.matmul(A, beta) - s
        F = np.concatenate((F1, F2))
        J = dist_hess + penalty * np.matmul(np.transpose(C), np.matmul(loss_hess, C))
        J = np.concatenate(
            (np.concatenate((J, np.transpose(A)), axis=1),
             np.concatenate((A, np.zeros((A.shape[0], A.shape[0]))), axis=1),
            ), axis=0,
        )
        delta_sol = cg(J, F)[0]
        sol_k = sol_k - delta_sol
        beta = sol_k[0:A.shape[1]]
        lambda_k = sol_k[A.shape[1]:(A.shape[0] + A.shape[1])]
        epsilon = np.mean(np.abs(s - np.matmul(A, beta)))
        iter_eps = iter_eps + 1
    return (beta, lambda_k, iter_eps)

