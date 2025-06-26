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

