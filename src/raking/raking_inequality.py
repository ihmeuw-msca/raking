"""Module with methods to solve the raking problem with inequality constraints"""

import numpy as np

from scipy.optimize import Bounds
from scipy.optimize import LinearConstraint
from scipy.optimize import OptimizeResult
from scipy.optimize import minimize

def raking_chi2_inequality(
    y: np.ndarray,
    A: np.ndarray,
    s: np.ndarray,
    C: np.ndarray,
    c: np.ndarray,
    q: np.ndarray,
    l: np.ndarray = None,
    h: np.ndarray = None
) -> OptimizeResult:

    equality = LinearConstraint(A, s, s)
    inequality = LinearConstraint(C, np.repeat(-np.inf, len(c)), c)
    if (l is not None) and (h is not None):
        bounds = Bounds(l, h)
    else:
        bounds = None

    def distance(beta):
        return np.sum(np.square(beta - y) / (2.0 * q * y))
    def jacobian(beta):
        return (beta / y - 1.0) / q
    def hessian(beta):
        return np.diag(1.0 / (q * y))

    if (l is not None) and (h is not None):
        res = minimize(fun=distance, x0=y, method='trust-constr', \
            jac=jacobian, hess=hessian, constraints=[equality, inequality], bounds=bounds)
    else:
        res = minimize(fun=distance, x0=y, method='trust-constr', \
            jac=jacobian, hess=hessian, constraints=[equality, inequality])
    return res


def raking_entropic_inequality(
    y: np.ndarray,
    A: np.ndarray,
    s: np.ndarray,
    C: np.ndarray,
    c: np.ndarray,
    q: np.ndarray
) -> OptimizeResult:

    equality = LinearConstraint(A, s, s)
    inequality = LinearConstraint(C, np.repeat(-np.inf, len(c)), c)

    def distance(beta):
        return np.sum((beta * np.log(beta / y) - beta + y) / q)
    def jacobian(beta):
        return (np.log(beta / y)) / q
    def hessian(beta):
        return np.diag(1.0 / (q * beta))

    res = minimize(fun=distance, x0=y, method='trust-constr', \
        jac=jacobian, hess=hessian, constraints=[equality, inequality])
    return res


def raking_logit_inequality(
    y: np.ndarray,
    A: np.ndarray,
    s: np.ndarray,
    C: np.ndarray,
    c: np.ndarray,
    q: np.ndarray,
    l: np.ndarray,
    h: np.ndarray
) -> OptimizeResult:

    equality = LinearConstraint(A, s, s)
    inequality = LinearConstraint(C, np.repeat(-np.inf, len(c)), c)

    def distance(beta):
        return np.sum(((beta - l) * np.log((beta - l) / (y - l)) + \
                       (h - beta) * np.log((h - beta) / (h - y))) / q)
    def jacobian(beta):
        return (np.log((beta - l) / (y - l)) + np.log((h - beta) / (h - y))) / q
    def hessian(beta):
        return np.diag((1.0 / ( beta - l) - 1.0 / (h - beta)) / y)

    res = minimize(fun=distance, x0=y, method='trust-constr', \
        jac=jacobian, hess=hessian, constraints=[equality, inequality])
    return res

