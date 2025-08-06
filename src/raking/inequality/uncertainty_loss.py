"""Module with methods to propagate the uncertainties through the raking process"""

import numpy as np

from scipy.linalg import lu_factor, lu_solve

from raking.inequality.loss_functions import compute_loss, compute_dist

def compute_covariance(
    Dphi_y: np.ndarray,
    Dphi_c: np.ndarray,
    Dphi_s: np.ndarray,
    y_var: np.ndarray,
    c_var: np.ndarray,
    s_var: np.ndarray,
) -> np.ndarray:
    """Compute the covariance matrix of the raked values.

    The covariance matrix of the raked values is phi' Sigma phi'T
    where phi' is the matrix of the partial derivatives of the raked values beta
    with respect to the observations y and margins s.

    Parameters
    ----------
    Dphi_y : np.ndarray
        Derivatives with respect to the observations
    Dphi_c : np.ndarray
        Derivatives with respect to the bounds
    Dphi_s : np.ndarray
        Derivatives with respect to the margins
    y_var : np.ndarray
        Variances of the observations
    c_var : np.ndarray
        Variances of the bounds
    s_var : np.ndarray
        Variances of the margins

    Returns
    -------
    covariance : np.ndarray
        Covariance matrix of the raked values
    """

    Dphi = np.concatenate((Dphi_y, Dphi_c, Dphi_s), axis=1)
    sigma = np.concatenate((y_var, c_var, s_var))
    covariance = np.matmul(Dphi * sigma, np.transpose(Dphi))
    return covariance
    
def compute_gradient(
    beta_0: np.ndarray,
    lambda_0: np.ndarray,
    y: np.ndarray,
    A: np.ndarray,
    s: np.ndarray,
    C: np.ndarray,
    c: np.ndarray,
    DyC: np.ndarray,
    q: np.ndarray,
    method: str = 'chi2',
    loss: str = 'logit',
    penalty: float = 1.0,
    l: np.ndarray = None,
    h: np.ndarray = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    """
    # Gradient with respect to beta and lambda
    dist_hessian = compute_dist(beta_0, y, q, method, 'hessian', l, h)
    loss_hessian = compute_loss(c - np.matmul(C, beta_0), penalty, loss, 'hessian')
    DF1_beta = np.diag(dist_hessian) + np.matmul(np.transpose(C) * loss_hessian, C)
    DF1_lambda = np.transpose(np.copy(A))
    DF2_beta = np.copy(A)
    DF2_lambda = np.zeros((np.shape(A)[0], np.shape(A)[0]))
    DF_beta_lambda = np.concatenate(
        (
            np.concatenate((DF1_beta, DF1_lambda), axis=1),
            np.concatenate((DF2_beta, DF2_lambda), axis=1),
        ),
        axis=0,
    )

    # Gradient with respect to y and s
    dist_both = compute_dist(beta_0, y, q, method, 'both', l, h)
    loss_gradient = compute_loss(c - np.matmul(C, beta_0), penalty, loss, 'gradient')
    DF1_y = np.diag(dist_both) - \
        (np.transpose(DyC, (1, 2, 0)) * loss_gradient).sum(axis=2) + \
        np.matmul(np.transpose(C) * loss_hessian, \
            (np.transpose(DyC, (0, 2, 1)) * beta_0).sum(axis=1))
    DF1_c = - np.transpose(C) * loss_hessian
    DF1_s = np.zeros((np.shape(A)[1], np.shape(A)[0]))
    DF2_y = np.zeros((np.shape(A)[0], np.shape(A)[1]))
    DF2_c = np.zeros((np.shape(A)[0], np.shape(C)[0]))
    DF2_s = - np.identity(np.shape(A)[0])
    DF_y_c_s = np.concatenate(
        (
            np.concatenate((DF1_y, DF1_c, DF1_s), axis=1),
            np.concatenate((DF2_y, DF2_c, DF2_s), axis=1),
        ),
        axis=0,
    )

    # Solve system DF_beta_lambda Dphi_y_s = - DF_y_s
    Dphi_y_c_s = np.zeros_like(DF_y_c_s)
    lu, piv = lu_factor(DF_beta_lambda)
    for i in range(0, np.shape(DF_y_c_s)[1]):
        Dphi_y_c_s[:, i] = -lu_solve((lu, piv), DF_y_c_s[:, i])

    # Return gradient of beta and lambda with respect to y and s
    Dphi_y = Dphi_y_c_s[0 : np.shape(A)[1], 0 : np.shape(A)[1]]
    Dphi_c = Dphi_y_c_s[0 : np.shape(A)[1], np.shape(A)[1] : (np.shape(A)[1] + np.shape(C)[0])]
    Dphi_s = Dphi_y_c_s[0 : np.shape(A)[1], (np.shape(A)[1] + np.shape(C)[0]) : (np.shape(A)[0] + np.shape(A)[1] + np.shape(C)[0])]
    return (Dphi_y, Dphi_c, Dphi_s)

