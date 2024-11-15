"""Module with methods to solve the raking problem with inequality constraints"""

import numpy as np

from scipy.optimize import LinearConstraint
from scipy.optimize import minimize

def raking_chi2_inequality(
    y: np.ndarray,
    A: np.ndarray,
    s: np.ndarray,
    C: np.ndarray,
    c: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:

    equality = LinearConstraint(A, s, s)
    inequality = LinearConstraint(C, np.repeat(-np.inf, len(c)), c)

    def distance(x):
        return np.sum(np.square(x - y) / (2.0 * y))
    def jacobian(x):
        return x / y - 1.0
    def hessian(x):
        return np.diag(1.0 / y)

    res = minimize(distance, y, method='trust-constr', \
        jac=jacobian, hess=hessian, constraints=[equality, inequality])
    return res

