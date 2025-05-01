"""Module with methods to solve the raking problem with inequality constraints"""

import numpy as np

from scipy.optimize import Bounds
from scipy.optimize import LinearConstraint
from scipy.optimize import OptimizeResult
from scipy.optimize import minimize

from raking.inequality.loss_functions import compute_dist

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
    else:
        bounds = None

    def distance(beta):
        return compute_dist(beta, y, q, method, l, h)[0]
    def jacobian(beta):
        return compute_dist(beta, y, q, method, l, h)[1]
    def hessian(beta):
        return compute_dist(beta, y, q, method, l, h)[2]

    if with_bounds:
        res = minimize(fun=distance, x0=y, method='trust-constr', \
            jac=jacobian, hess=hessian, constraints=[equality, inequality], bounds=bounds)
    else:
        res = minimize(fun=distance, x0=y, method='trust-constr', \
            jac=jacobian, hess=hessian, constraints=[equality, inequality])
    return res

