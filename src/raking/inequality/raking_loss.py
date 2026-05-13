"""Module with methods to solve the raking problem with a penalty loss"""

import numpy as np

from scipy.optimize import Bounds
from scipy.optimize import OptimizeResult
from scipy.optimize import minimize
from scipy.sparse.linalg import cg

from raking.inequality.loss_functions import (
    compute_loss,
    compute_dist,
    compute_conjugate_loss,
    compute_conjugate_dist,
)
from raking.raking_methods import (
    raking_chi2,
    raking_entropic,
    raking_logit,
)

def barrier_loss(
    c: np.ndarray,
    penalty: float = 1.0,
    mu: float = 1.0,
    gamma_barrier: float = 0.1,
    tol_barrier: float = 1.0e-11
):
    """

    Parameters
    ----------
    c : np.ndarray
        Bounds for the inequality constraints
    penalty : float
        Parameter in the definition of the loss function
    mu : float
        Parameter for the barrier method. Start high and decrease to get to 0
    gamma_barrier : float
        Between 0 and 1. Used to uo update mu: mu+ = gamma_barrier * mu
    tol_barrier : float
        Tolerance for the barrier method
    """
    assert mu > 0, "The parameter mu for the barrier method must be positive."
    assert (gamma_barrier > 0) & (gamma_barrier < 1), "The parameter to update mu must be between 0 and 1."
    assert tol_barrier > 0, "The tolerance for the barrier method must be positive."

    # Initialization
    lambda_k = - np.repeat(penalty / 2.0, len(c))

    # Check number of inequality constraints
    m = len(c)
    while (m * mu >= tol_barrier):
        # Centering step
        (lambda_k, nu_k) = centering_loss(lambda_k)
        # Update mu
        mu = gamma_barrier * mu
    return (lambda_k, nu_k, mu)
    
def centering_loss(
    y: np.ndarray,
    A: np.ndarray,
    s: np.ndarray,
    C: np.ndarray,
    c: np.ndarray,
    q: np.ndarray,
    lambda_k: np.ndarray,
    method: str = 'chi2',
    loss: str = 'logit',
    penalty: float = 1.0,
    l: np.ndarray = None,
    h: np.ndarray = None,
    tol_center: float = 1.0e-11,
    max_iter: int = 500,
):
    """

    Parameters
    ----------
    y : np.ndarray
        Vector of observations
    A : np.ndarray
        Constraints matrix (output of a function from the compute_constraints module)
    s : np.ndarray
        Margin vector (output of a function from the compute_constraints module)
    q : np.ndarray
        Vector of weights (default to all 1)
    lambda_k : np.ndarray
        Dual value obtained at the end of the last centering step
    method : string
        Name of the distance function used for the raking.
        Possible values are chi2, entropic, logit
    loss : string
        Name of the loss function used for the inequality constraints.
        Possible values are hinge, logit
    penalty : float
        Parameter in the definition of the loss function
    l : np.ndarray
        Lower bounds for the observations
    h : np.ndarray
        Upper bounds for the observations
    tol_center : float
        Tolerance for the centering step
    max_iter : int
        Maximum number of iterations 
    """
    assert tol_center > 0, "The tolerance for the centering step must be positive."

    # Initialization: use solution of raking with equality for the initial dual
    if method == 'chi2':
        (beta, nu_k) = raking_chi2(y, A, s, q)
    elif method=='entropic':
        (beta, nu_k, iter_eps) = raking_entropic(y, A, s, q)
    elif method == 'logit':
        (beta, nu_k, iter_eps) = raking_logit(y, A, s, l, h, q)

    num_iter = 0
    Phi = compute_gradient(y, A, s, C, c, q, method, loss, l, h)
    epsilon = np.sqrt(np.dot(Phi, Phi))
    while (epsilon >= tol_center) & (num_iter < max_iter):
        J = compute_jacobian(y, A, s, C, c, q, method, loss, l, h)
    
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
    l: np.ndarray = None,
    h: np.ndarray = None,
    tol: float = 1.0e-11,
    gamma: float = 1.0e-4,
    max_iter: int = 500,
):
    """
    """
    if method == 'chi2':
        (beta, lambda_k) = raking_chi2(y, A, s, q)
    elif method=='entropic':
        (beta, lambda_k, iter_eps) = raking_entropic(y, A, s, q)
    elif method == 'logit':
        (beta, lambda_k, iter_eps) = raking_logit(y, A, s, l, h, q)
    sol_k = np.concatenate((beta, lambda_k), axis=0)
    loss_gradient = compute_loss(c - np.matmul(C, beta), penalty, loss, 'gradient')
    dist_gradient = compute_dist(beta, y, q, method, 'gradient', l, h)
    F1 = dist_gradient + np.matmul(np.transpose(A), lambda_k) \
        - np.matmul(np.transpose(C), loss_gradient)
    F2 = np.matmul(A, beta) - s
    F = np.concatenate((F1, F2), axis=0)
    epsilon = np.sqrt(np.sum(np.square(F)))
    iter_eps = 0
    while (epsilon > tol) & (iter_eps < max_iter):
        loss_hessian = compute_loss(c - np.matmul(C, beta), penalty, loss, 'hessian')
        dist_hessian = compute_dist(beta, y, q, method, 'hessian', l, h)
        J = np.diag(dist_hessian) + np.matmul(np.transpose(C) * loss_hessian, C)
        J = np.concatenate(
            (np.concatenate((J, np.transpose(A)), axis=1),
             np.concatenate((A, np.zeros((A.shape[0], A.shape[0]))), axis=1),
            ), axis=0,
        )
        delta_sol = cg(J, F)[0]
        m = 0
        iter_armijo = 0
        sol_kn = sol_k - 2.0**(-m) * delta_sol
        beta_n = sol_kn[0:len(y)]
        lambda_kn = sol_kn[len(y):(len(y) + len(s))]
        loss_gradient = compute_loss(c - np.matmul(C, beta_n), penalty, loss, 'gradient')
        dist_gradient = compute_dist(beta_n, y, q, method, 'gradient', l, h)
        F1n = dist_gradient + np.matmul(np.transpose(A), lambda_kn) \
            - np.matmul(np.transpose(C), loss_gradient)
        F2n = np.matmul(A, beta_n) - s
        Fn = np.concatenate((F1n, F2n), axis=0)
        epsilon_n = np.sqrt(np.sum(np.square(Fn)))
        armijo_rule = (epsilon_n < (1.0 - gamma * 2.0**(-m)) * epsilon)
        while (armijo_rule==False) & (iter_armijo < 500):
            m = m + 1
            sol_kn = sol_k - 2.0**(-m) * delta_sol
            beta_n = sol_kn[0:len(y)]
            lambda_kn = sol_kn[len(y):(len(y) + len(s))]
            loss_gradient = compute_loss(c - np.matmul(C, beta_n), penalty, loss, 'gradient')
            dist_gradient = compute_dist(beta_n, y, q, method, 'gradient', l, h)
            F1n = dist_gradient + np.matmul(np.transpose(A), lambda_kn) \
                - np.matmul(np.transpose(C), loss_gradient)
            F2n = np.matmul(A, beta_n) - s
            Fn = np.concatenate((F1n, F2n), axis=0)
            epsilon_n = np.sqrt(np.sum(np.square(Fn)))
            armijo_rule = (epsilon_n < (1.0 - gamma * 2.0**(-m)) * epsilon)
            iter_armijo = iter_armijo + 1 
        sol_k = sol_k - 2.0**(-m) * delta_sol
        beta = sol_k[0:len(y)]
        lambda_k = sol_k[len(y):(len(y) + len(s))]
        loss_gradient = compute_loss(c - np.matmul(C, beta), penalty, loss, 'gradient')
        dist_gradient = compute_dist(beta, y, q, method, 'gradient', l, h)
        F1 = dist_gradient + np.matmul(np.transpose(A), lambda_k) \
            - np.matmul(np.transpose(C), loss_gradient)
        F2 = np.matmul(A, beta) - s
        F = np.concatenate((F1, F2), axis=0)
        epsilon = np.sqrt(np.sum(np.square(F)))
        iter_eps = iter_eps + 1
    return (beta, lambda_k, iter_eps)


def raking_dual_loss(
    y: np.ndarray,
    A: np.ndarray,
    s: np.ndarray,
    C: np.ndarray,
    c: np.ndarray,
    q: np.ndarray,
    method: str = 'chi2',
    loss: str = 'logit',
    penalty: float = 1.0,
    l: np.ndarray = None,
    h: np.ndarray = None,
    tol: float = 1.0e-11,
    gamma: float = 1.0e-4,
    max_iter: int = 500,
):
    if method == 'chi2':
        (beta, lambda_1) = raking_chi2(y, A, s, q)
    elif method=='entropic':
        (beta, lambda_1, iter_eps) = raking_entropic(y, A, s, q)
    elif method == 'logit':
        (beta, lambda_1, iter_eps) = raking_logit(y, A, s, l, h, q)
    lambda_2 = - np.repeat(penalty / 2.0, len(c))
    lambda_k = np.concatenate((lambda_1, lambda_2), axis=0)
    conjugate_dist_grad = compute_conjugate_dist( \
        - np.matmul(np.transpose(A), lambda_1) \
        + np.matmul(np.transpose(C), lambda_2), y, q, method, 'gradient', l, h)
    conjugate_loss_grad = compute_conjugate_loss(lambda_2, penalty, loss, 'gradient')
    F1 = s - np.matmul(A, conjugate_dist_grad)
    F2 = - c + np.matmul(C, conjugate_dist_grad) + conjugate_loss_grad
    F = np.concatenate((F1, F2), axis=0)
    epsilon = np.sqrt(np.sum(np.square(F)))
    iter_eps = 0
    while (epsilon > tol) & (iter_eps < max_iter):
        conjugate_dist_hess = compute_conjugate_dist( \
            - np.matmul(np.transpose(A), lambda_1) \
            + np.matmul(np.transpose(C), lambda_2), y, q, method, 'hessian', l, h)
        conjugate_loss_hess = compute_conjugate_loss(lambda_2, penalty, loss, 'hessian')
        J = np.concatenate( \
            (np.concatenate( \
                (np.matmul(A * conjugate_dist_hess, np.transpose(A)), \
                - np.matmul(A * conjugate_dist_hess, np.transpose(C))), axis=1), \
             np.concatenate( \
                 ( - np.matmul(C * conjugate_dist_hess, np.transpose(A)), \
                 np.matmul(C * conjugate_dist_hess, np.transpose(C)) + np.diag(conjugate_loss_hess)), axis=1) \
            ), axis=0)
        delta_lambda = cg(J, F)[0]
        m = 0
        iter_armijo = 0
        lambda_kn = lambda_k - 2.0**(-m) * delta_lambda
        lambda_1n = lambda_kn[0:len(s)]
        lambda_2n = lambda_kn[len(s):(len(s) + len(c))]
        conjugate_dist_grad = compute_conjugate_dist( \
            - np.matmul(np.transpose(A), lambda_1n) \
            + np.matmul(np.transpose(C), lambda_2n), y, q, method, 'gradient', l, h)
        conjugate_loss_grad = compute_conjugate_loss(lambda_2n, penalty, loss, 'gradient')
        F1n = s - np.matmul(A, conjugate_dist_grad)
        F2n = - c + np.matmul(C, conjugate_dist_grad) + conjugate_loss_grad
        Fn = np.concatenate((F1n, F2n), axis=0)
        epsilon_n = np.sqrt(np.sum(np.square(Fn)))
        armijo_rule = (epsilon_n < (1.0 - gamma * 2.0**(-m)) * epsilon)
        while (armijo_rule==False) & (iter_armijo < 500):
            m = m + 1
            lambda_kn = lambda_k - 2.0**(-m) * delta_lambda
            lambda_1n = lambda_kn[0:len(s)]
            lambda_2n = lambda_kn[len(s):(len(s) + len(c))]
            conjugate_dist_grad = compute_conjugate_dist( \
                - np.matmul(np.transpose(A), lambda_1n) \
                + np.matmul(np.transpose(C), lambda_2n), y, q, method, 'gradient', l, h)
            conjugate_loss_grad = compute_conjugate_loss(lambda_2n, penalty, loss, 'gradient')
            F1n = s - np.matmul(A, conjugate_dist_grad)
            F2n = - c + np.matmul(C, conjugate_dist_grad) + conjugate_loss_grad
            Fn = np.concatenate((F1n, F2n), axis=0)
            epsilon_n = np.sqrt(np.sum(np.square(Fn)))
            armijo_rule = (epsilon_n < (1.0 - gamma * 2.0**(-m)) * epsilon)
            iter_armijo = iter_armijo + 1 
        lambda_k = lambda_k - 2.0**(-m) * delta_lambda
        lambda_1 = lambda_k[0:len(s)]
        lambda_2 = lambda_k[len(s):(len(s) + len(c))]
        conjugate_dist_grad = compute_conjugate_dist( \
            - np.matmul(np.transpose(A), lambda_1) \
            + np.matmul(np.transpose(C), lambda_2), y, q, method, 'gradient', l, h)
        conjugate_loss_grad = compute_conjugate_loss(lambda_2, penalty, loss, 'gradient')
        F1 = s - np.matmul(A, conjugate_dist_grad)
        F2 = - c + np.matmul(C, conjugate_dist_grad) + conjugate_loss_grad
        F = np.concatenate((F1, F2), axis=0)
        epsilon = np.sqrt(np.sum(np.square(F)))
        iter_eps = iter_eps + 1
    z = - np.matmul(np.transpose(A), lambda_1) + np.matmul(np.transpose(C), lambda_2)
    beta = compute_conjugate_dist(z, y, q, method, 'gradient', l, h)
    gamma = compute_conjugate_loss(lambda_2, penalty, loss, 'gradient')
    return (beta, gamma, lambda_k, iter_eps)


def raking_dual_loss_scipy(
    y: np.ndarray,
    A: np.ndarray,
    s: np.ndarray,
    C: np.ndarray,
    c: np.ndarray,
    q: np.ndarray,
    method: str = 'chi2',
    loss: str = 'logit',
    penalty: float = 1.0,
    l: np.ndarray = None,
    h: np.ndarray = None,
    tol: float = 1.0e-11,
    max_iter: int = 500,
):
    """
    """
    if loss == 'hinge':
        bounds = Bounds(ub=np.concatenate(( \
            np.repeat(np.inf, len(s)), np.repeat(0, len(c))), axis=0))
    if loss == 'logit':
        bounds = Bounds(lb=np.concatenate(( \
            np.repeat(-np.inf, len(s)), np.repeat(- penalty, len(c))), axis=0), \
            ub=np.concatenate(( \
            np.repeat(np.inf, len(s)), np.repeat(0, len(c))), axis=0))

    def objective(lambda_k):
        lambda_1 = lambda_k[0:len(s)]
        lambda_2 = lambda_k[len(s):(len(s) + len(c))]
        z = - np.matmul(np.transpose(A), lambda_1) + np.matmul(np.transpose(C), lambda_2)
        conjugate_dist = compute_conjugate_dist(z, y, q, method, 'objective', l, h)
        conjugate_loss = compute_conjugate_loss(lambda_2, penalty, loss, 'objective')
        return (np.dot(lambda_1, s) - np.dot(lambda_2, c) + conjugate_dist + conjugate_loss)[0]
    def jacobian(lambda_k):
        lambda_1 = lambda_k[0:len(s)]
        lambda_2 = lambda_k[len(s):(len(s) + len(c))]
        z = - np.matmul(np.transpose(A), lambda_1) + np.matmul(np.transpose(C), lambda_2)
        conjugate_dist = compute_conjugate_dist(z, y, q, method, 'gradient', l, h)
        conjugate_loss = compute_conjugate_loss(lambda_2, penalty, loss, 'gradient')
        return np.concatenate( \
            (s - np.matmul(A, conjugate_dist), \
             - c + np.matmul(C, conjugate_dist) + conjugate_loss), axis=0)

    if method == 'chi2':
        (beta, lambda_1) = raking_chi2(y, A, s, q)
    elif method=='entropic':
        (beta, lambda_1, iter_eps) = raking_entropic(y, A, s, q)
    elif method == 'logit':
        (beta, lambda_1, iter_eps) = raking_logit(y, A, s, l, h, q)
    lambda_2 = - np.repeat(penalty / 2.0, len(c))
    lambda_k = np.concatenate((lambda_1, lambda_2), axis=0)

    result = minimize(fun=objective, x0=lambda_k, method='L-BFGS-B', jac=jacobian, bounds=bounds)
    iter_eps = result.nit
    lambda_k = result.x
    lambda_1 = lambda_k[0:len(s)]
    lambda_2 = lambda_k[len(s):(len(s) + len(c))]
    z = - np.matmul(np.transpose(A), lambda_1) + np.matmul(np.transpose(C), lambda_2)
    beta = compute_conjugate_dist(z, y, q, method, 'gradient', l, h)
    gamma = compute_conjugate_loss(lambda_2, penalty, loss, 'gradient')
    return (beta, gamma, lambda_k, iter_eps)

