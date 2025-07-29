"""Module with methods to solve the raking problem with inequality constraints"""

import numpy as np

from scipy.optimize import Bounds
from scipy.optimize import LinearConstraint
from scipy.optimize import OptimizeResult
from scipy.optimize import minimize

from raking.inequality.loss_functions import compute_dist
from raking.raking_methods import raking_chi2, raking_entropic, raking_logit

def raking_inequality(
    y: np.ndarray,
    A: np.ndarray,
    s: np.ndarray,
    C: np.ndarray,
    c: np.ndarray,
    q: np.ndarray,
    method: str = 'chi2',
    with_bounds: bool = False,
    l: np.ndarray = None,
    h: np.ndarray = None
) -> OptimizeResult:

    equality = LinearConstraint(A, s, s)
    inequality = LinearConstraint(C, np.repeat(-np.inf, len(c)), c)
    if with_bounds:
        bounds = Bounds(l, h)

    def distance(beta):
        return compute_dist(beta, y, q, method, 'objective', l, h)
    def jacobian(beta):
        return compute_dist(beta, y, q, method, 'gradient', l, h)
    def hessian(beta):
        return np.diag(compute_dist(beta, y, q, method, 'hessian', l, h))

    if method == 'chi2':
        (beta, lambda_k) = raking_chi2(y, A, s, q)
    elif method=='entropic':
        (beta, lambda_k, iter_eps) = raking_entropic(y, A, s, q)
    elif method == 'logit':
        (beta, lambda_k, iter_eps) = raking_logit(y, A, s, l, h, q)

    if with_bounds:
        res = minimize(fun=distance, x0=beta, method='trust-constr', \
            jac=jacobian, hess=hessian, constraints=[equality, inequality], bounds=bounds)
    else:
        res = minimize(fun=distance, x0=beta, method='trust-constr', \
            jac=jacobian, hess=hessian, constraints=[equality, inequality])
    return res


def raking_dual_inequality(
    y: np.ndarray,
    A: np.ndarray,
    s: np.ndarray,
    C: np.ndarray,
    c: np.ndarray,
    q: np.ndarray,
    method: str = 'chi2',
    l: np.ndarray = None,
    h: np.ndarray = None
) -> OptimizeResult:

    bounds = Bounds(lb=np.concatenate(( \
        np.repeat(0, len(c)), np.repeat(-np.inf, len(s))), axis=0))

    def objective(mu_k):
        lambda_k = mu_k[0:len(c)]
        nu_k = mu_k[len(c):(len(c) + len(s))]
        z = - np.matmul(np.transpose(C), lambda_k) - np.matmul(np.transpose(A), nu_k)
        conjugate_dist = compute_conjugate_dist(z, y, q, method, 'objective', l, h)
        return (np.dot(lambda_k, c) + np.dot(nu_k, s) + conjugate_dist)[0]
    def jacobian(lambda_k):
        lambda_k = mu_k[0:len(c)]
        nu_k = mu_k[len(c):(len(c) + len(s))]
        z = - np.matmul(np.transpose(C), lambda_k) - np.matmul(np.transpose(A), nu_k)
        conjugate_dist = compute_conjugate_dist(z, y, q, method, 'gradient', l, h)
        return np.concatenate( \
            (c - np.matmul(C, conjugate_dist), \
             s - np.matmul(A, conjugate_dist)), axis=0)

    if method == 'chi2':
        (beta, nu_k) = raking_chi2(y, A, s, q)
    elif method=='entropic':
        (beta, nu_k, iter_eps) = raking_entropic(y, A, s, q)
    elif method == 'logit':
        (beta, nu_k, iter_eps) = raking_logit(y, A, s, l, h, q)
    lambda_k = np.repeat(1.0, len(c))
    mu_k = np.concatenate((lambda_k, nu_k), axis=0)

    result = minimize(fun=objective, x0=mu_k, method='L-BFGS-B', jac=jacobian, bounds=bounds)
    iter_eps = result.nit
    mu_k = result.x
    lambda_k = mu_k[0:len(c)]
    nu_k = mu_k[len(c):(len(c) + len(s))]
    z = - np.matmul(np.transpose(C), lambda_k) - np.matmul(np.transpose(A), nu_k)
    beta = compute_conjugate_dist(z, y, q, method, 'gradient', l, h)
    return (beta, mu_k, iter_eps)

