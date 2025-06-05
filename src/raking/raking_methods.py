"""Module with methods to solve the raking problem"""

import numpy as np
from scipy.sparse.linalg import cg

from scipy.optimize import minimize

def raking_chi2(
    y: np.ndarray,
    A: np.ndarray,
    s: np.ndarray,
    q: np.ndarray = None
    tol: float = 1e-11
) -> tuple[np.ndarray, np.ndarray]:

    lambda_0 = np.zeros(A.shape[0])

    def conjugate_distance(lambda_k):
        z = - np.matmul(np.transpose(A), lambda_k)
        f_star = np.sum(y * z + q * y * np.square(z) / 2.0)
        return f_star + np.dot(lambda_k , s)

    def conjugate_jacobian(lambda_k):
        z = - np.matmul(np.transpose(A), lambda_k)
        fstar_grad = y * ( 1 + q * z)
        return - np.matmul(np.transpose(A), f_star_grad) + s

    def conjugate_hessian(lambda_k):
        z = - np.matmul(np.transpose(A), lambda_k)
        f_star_hess = np.diag(q * y)
        return np.matmul(np.matmul(np.transpose(A), f_star_hess), A)

    res = minimize( \
        fun=conjugate_distance, \
        x0=lambda_0, \
        method='Newton-CG', \
        jac=conjugate_jacobian, \
        hess=conjugate_hessian,
        tol=tol)
    lambda_k = res.x
    beta = y * (1 - q * np.matmul(np.transpose(A), lambda_k))
    return (beta, lambda_k)


def raking_entropic(
    y: np.ndarray,
    A: np.ndarray,
    s: np.ndarray,
    q: np.ndarray = None,
    tol: float = 1.0e-11,
    gamma: float = 1.0e-4,
    max_iter: int = 500,
) -> tuple[np.ndarray, np.ndarray, int]:

    lambda_0 = np.zeros(A.shape[0])

    def conjugate_distance(lambda_k):
        z = - np.matmul(np.transpose(A), lambda_k)
        f_star = np.sum(y * (np.exp(q * z) - 1) / q)
        return f_star + np.dot(lambda_k , s)

    def conjugate_jacobian(lambda_k):
        z = - np.matmul(np.transpose(A), lambda_k)
        fstar_grad = y * np.exp(q * z)
        return - np.matmul(np.transpose(A), f_star_grad) + s

    def conjugate_hessian(lambda_k):
        z = - np.matmul(np.transpose(A), lambda_k)
        f_star_hess = np.diag(q * np.exp(q * z))
        return np.matmul(np.matmul(np.transpose(A), f_star_hess), A)

    res = minimize( \
        fun=conjugate_distance, \
        x0=lambda_0, \
        method='Newton-CG', \
        jac=conjugate_jacobian, \
        hess=conjugate_hessian,
        tol=tol,
        options={'maxiter': max_iter})
    lambda_k = res.x
    iter_eps = res.nit
    beta = y * np.exp(-q * np.matmul(np.transpose(A), lambda_k))
    return (beta, lambda_k, iter_eps)


def raking_chi2_old(
    y: np.ndarray,
    A: np.ndarray,
    s: np.ndarray,
    q: np.ndarray = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Raking using the chi2 distance f(beta, y) = (beta - y)^2 / 2y.

    This will solve the problem:
        min_beta 1/q f(beta, y) s.t. A beta = s

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

    Returns
    -------
    beta : np.ndarray
        Vector of raked values
    lambda_k : np.ndarray
        Dual (needed for the uncertainty computation)
    """
    assert isinstance(
        y, np.ndarray
    ), "The vector of observations should be a Numpy array."
    assert (
        len(y.shape) == 1
    ), "The vector of observations should be a 1D Numpy array."
    if q is not None:
        assert isinstance(
            q, np.ndarray
        ), "The vector of weights should be a Numpy array."
        assert (
            len(y.shape) == 1
        ), "The vector of weights should be a 1D Numpy array."
        assert len(y) == len(
            q
        ), "Observations and weights vectors should have the same length."
    assert isinstance(
        A, np.ndarray
    ), "The constraint matrix should be a Numpy array."
    assert (
        len(A.shape) == 2
    ), "The constraints matrix should be a 2D Numpy array."
    assert isinstance(
        s, np.ndarray
    ), "The margins vector should be a Numpy array."
    assert len(s.shape) == 1, "The margins vector should be a 1D Numpy array."
    assert (
        np.shape(A)[0] == len(s)
    ), "The number of linear constraints should be equal to the number of margins."
    assert (
        np.shape(A)[1] == len(y)
    ), "The number of coefficients for the linear constraints should be equal to the number of observations."

    if q is None:
        q = np.ones(len(y))
    s_hat = np.matmul(A, y)
    Phi = np.matmul(A, np.transpose(A * y * q))
    lambda_k = cg(Phi, s_hat - s)[0]
    beta = y * (1 - q * np.matmul(np.transpose(A), lambda_k))
    return (beta, lambda_k)


def raking_entropic_old(
    y: np.ndarray,
    A: np.ndarray,
    s: np.ndarray,
    q: np.ndarray = None,
    tol: float = 1.0e-11,
    gamma: float = 1.0e-4,
    max_iter: int = 500,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Raking using the entropic distance f(beta, y) = beta log(beta/y) + y - beta.

    This will solve the problem:
        min_beta 1/q f(beta, y) s.t. A beta = s

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
    tol: float
        Tolerance for the convergence
    gamma : float
        Parameter for Armijo rule
    max_iter : int
        Number of iterations for Newton's root finding method

    Returns
    -------
    beta : np.ndarray
        Vector of reaked values
    lambda_k : np.ndarray
        Dual (needed for th uncertainty computation)
    iters_eps : int
        Number of iterations until convergence
    """
    assert isinstance(
        y, np.ndarray
    ), "The vector of observations should be a Numpy array."
    assert (
        len(y.shape) == 1
    ), "The vector of observations should be a 1D Numpy array."
    if q is not None:
        assert isinstance(
            q, np.ndarray
        ), "The vector of weights should be a Numpy array."
        assert (
            len(q.shape) == 1
        ), "The vector of weights should be a 1D Numpy array."
        assert len(y) == len(
            q
        ), "Observations and weights vectors should have the same length."
    assert isinstance(
        A, np.ndarray
    ), "The constraint matrix should be a Numpy array."
    assert (
        len(A.shape) == 2
    ), "The constraints matrix should be a 2D Numpy array."
    assert isinstance(
        s, np.ndarray
    ), "The margins vector should be a Numpy array."
    assert len(s.shape) == 1, "The margins vector should be a 1D Numpy array."
    assert (
        np.shape(A)[0] == len(s)
    ), "The number of linear constraints should be equal to the number of margins."
    assert (
        np.shape(A)[1] == len(y)
    ), "The number of coefficients for the linear constraints should be equal to the number of observations."

    if q is None:
        q = np.ones(len(y))
    s_hat = np.matmul(A, y)
    lambda_k = np.zeros(A.shape[0])
    beta = np.copy(y)
    epsilon = np.mean(np.abs(s - np.matmul(A, beta)))
    iter_eps = 0
    while (epsilon > tol) & (iter_eps < max_iter):
        Phi = np.matmul(
            A, y * (1.0 - np.exp(-q * np.matmul(np.transpose(A), lambda_k)))
        )
        D = np.diag(y * q * np.exp(-q * np.matmul(np.transpose(A), lambda_k)))
        J = np.matmul(np.matmul(A, D), np.transpose(A))
        delta_lambda = cg(J, Phi - s_hat + s)[0]
        m = 0
        F = np.sqrt(np.sum(np.square(Phi - s_hat + s)))
        nextF = np.sqrt(np.sum(np.square(np.matmul(
            A, y * (1.0 - np.exp(-q * np.matmul(np.transpose(A), lambda_k + 2.0**(-m) * delta_lambda)))
        ) - s_hat + s)))
        armijo_rule = (nextF < (1 - gamma * 2**(-m)) * F)
        iter_armijo = 0
        while (armijo_rule==False) & (iter_armijo < max_iter):
            m = m + 1
            nextF = np.sqrt(np.sum(np.square(np.matmul(
                A, y * (1.0 - np.exp(-q * np.matmul(np.transpose(A), lambda_k + 2.0**(-m) * delta_lambda)))
            ) - s_hat + s)))
            armijo_rule = (nextF < (1 - gamma * 2**(-m)) * F)
            iter_armijo = iter_armijo + 1 
        lambda_k = lambda_k - 2**(-m) * delta_lambda
        beta = y * np.exp(-q * np.matmul(np.transpose(A), lambda_k))
        epsilon = np.mean(np.abs(s - np.matmul(A, beta)))
        iter_eps = iter_eps + 1
    return (beta, lambda_k, iter_eps)


def raking_general(
    y: np.ndarray,
    A: np.ndarray,
    s: np.ndarray,
    alpha: float = 1,
    q: np.ndarray = None,
    tol: float = 1e-11,
    gamma0: float = 1.0,
    max_iter: int = 500,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Raking using the general distance f(beta, y) = 1/alpha (y/alpha+1 (beta/y)^alpha+1 - beta + c(y)).

    This will solve the problem:
        min_beta 1/q f(beta, y) s.t. A beta = s

    Parameters
    ----------
    y : np.ndarray
        Vector of observations
    A : np.ndarray
        Constraints matrix (output of a function from the compute_constraints module)
    s : np.ndarray
        Margin vector (output of a function from the compute_constraints module)
    alpha : float
        Parameter of the distance function, alpha=1 is the chi2 distance, alpha=0 is the entropic distance
    q : np.ndarray
        Vector of weights (default to all 1)
    gamma0 : float
        Initial value for line search
    max_iter : int
        Number of iterations for Newton's root finding method

    Returns
    -------
    beta : np.ndarray
        Vector of reaked values
    lambda_k : np.ndarray
        Dual (needed for th uncertainty computation)
    iters_eps : int
        Number of iterations until convergence
    """
    assert isinstance(
        y, np.ndarray
    ), "The vector of observations should be a Numpy array."
    assert (
        len(y.shape) == 1
    ), "The vector of observations should be a 1D Numpy array."
    if q is not None:
        assert isinstance(
            q, np.ndarray
        ), "The vector of weights should be a Numpy array."
        assert (
            len(y.shape) == 1
        ), "The vector of weights should be a 1D Numpy array."
        assert len(y) == len(
            q
        ), "Observations and weights vectors should have the same length."
    assert isinstance(
        A, np.ndarray
    ), "The constraint matrix should be a Numpy array."
    assert (
        len(A.shape) == 2
    ), "The constraints matrix should be a 2D Numpy array."
    assert isinstance(
        s, np.ndarray
    ), "The margins vector should be a Numpy array."
    assert len(s.shape) == 1, "The margins vector should be a 1D Numpy array."
    assert (
        np.shape(A)[0] == len(s)
    ), "The number of linear constraints should be equal to the number of margins."
    assert (
        np.shape(A)[1] == len(y)
    ), "The number of coefficients for the linear constraints should be equal to the number of observations."
    assert isinstance(
        alpha, (int, float)
    ), "The parameter of the distance function should be an integer or a float."

    if q is None:
        q = np.ones(len(y))

    if alpha == 1:
        (beta, lambda_k) = raking_chi2(y, A, s, q)
        return (beta, lambda_k, 0)

    if alpha == 0:
        (beta, lambda_k, iter_eps) = raking_entropic(
            y, A, s, q, gamma0, max_iter
        )
        return (beta, lambda_k, iter_eps)

    s_hat = np.matmul(A, y)
    lambda_k = np.zeros(A.shape[0])
    beta = np.copy(y)
    epsilon = 1.0
    iter_eps = 0
    while (epsilon > 1.0e-10) & (iter_eps < max_iter):
        Phi = np.matmul(
            A,
            y
            * (
                1.0
                - np.power(
                    1 - alpha * q * np.matmul(np.transpose(A), lambda_k),
                    1.0 / alpha,
                )
            ),
        )
        D = np.diag(
            y
            * q
            * np.power(
                1.0 - alpha * q * np.matmul(np.transpose(A), lambda_k),
                1.0 / alpha - 1,
            )
        )
        J = np.matmul(np.matmul(A, D), np.transpose(A))
        delta_lambda = cg(J, Phi - s_hat + s)[0]
        gamma = gamma0
        iter_gam = 0
        lambda_k = lambda_k - gamma * delta_lambda
        beta = y * np.power(
            1.0 - alpha * q * np.matmul(np.transpose(A), lambda_k), 1.0 / alpha
        )
        if (alpha > 0.5) or (alpha < -1.0):
            if iter_eps > 0:
                while (
                    (
                        np.any(
                            1
                            - alpha
                            * q
                            * np.matmul(
                                np.transpose(A), lambda_k - gamma * delta_lambda
                            )
                            <= 0.0
                        )
                    )
                    & (np.mean(np.abs(s - np.matmul(A, beta))) > epsilon)
                    & (iter_gam < max_iter)
                ):
                    gamma = gamma / 2.0
                    iter_gam = iter_gam + 1
                    lambda_k = lambda_k - gamma * delta_lambda
                    beta = y * np.power(
                        1.0 - alpha * q * np.matmul(np.transpose(A), lambda_k),
                        1.0 / alpha,
                    )
            else:
                while (
                    np.any(
                        1
                        - alpha
                        * q
                        * np.matmul(
                            np.transpose(A), lambda_k - gamma * delta_lambda
                        )
                        <= 0.0
                    )
                ) & (iter_gam < max_iter):
                    gamma = gamma / 2.0
                    iter_gam = iter_gam + 1
                    lambda_k = lambda_k - gamma * delta_lambda
                    beta = y * np.power(
                        1.0 - alpha * q * np.matmul(np.transpose(A), lambda_k),
                        1.0 / alpha,
                    )
        else:
            if iter_eps > 0:
                while (np.mean(np.abs(s - np.matmul(A, beta))) > epsilon) & (
                    iter_gam < max_iter
                ):
                    gamma = gamma / 2.0
                    iter_gam = iter_gam + 1
                    lambda_k = lambda_k - gamma * delta_lambda
                    beta = y * np.power(
                        1.0 - alpha * q * np.matmul(np.transpose(A), lambda_k),
                        1.0 / alpha,
                    )
        epsilon = np.mean(np.abs(s - np.matmul(A, beta)))
        iter_eps = iter_eps + 1
    return (beta, lambda_k, iter_eps)


def raking_logit(
    y: np.ndarray,
    A: np.ndarray,
    s: np.ndarray,
    l: np.ndarray = None,
    h: np.ndarray = None,
    q: np.ndarray = None,
    tol: float = 1e-11,
    gamma0: float = 1.0,
    max_iter: int = 500,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Logit raking ensuring that l < beta < h.

    This will solve the problem:
        min_beta 1/q f(beta, y) s.t. A beta = s

    Parameters
    ----------
    y : np.ndarray
        Vector of observations
    A : np.ndarray
        Constraints matrix (output of a function from the compute_constraints module)
    s : np.ndarray
        Margin vector (output of a function from the compute_constraints module)
    l : np.ndarray
        Lower bounds for the observations
    h : np.ndarray
        Upper bounds for the observations
    q : np.ndarray
        Vector of weights (default to all 1)
    gamma0 : float
        Initial value for line search
    max_iter : int
        Number of iterations for Newton's root finding method

    Returns
    -------
    beta : np.ndarray
        Vector of reaked values
    lambda_k : np.ndarray
        Dual (needed for th uncertainty computation)
    iters_eps : int
        Number of iterations until convergence
    """
    assert isinstance(
        y, np.ndarray
    ), "The vector of observations should be a Numpy array."
    assert (
        len(y.shape) == 1
    ), "The vector of observations should be a 1D Numpy array."
    if q is not None:
        assert isinstance(
            q, np.ndarray
        ), "The vector of weights should be a Numpy array."
        assert (
            len(y.shape) == 1
        ), "The vector of weights should be a 1D Numpy array."
        assert len(y) == len(
            q
        ), "Observations and weights vectors should have the same length."
    assert isinstance(
        A, np.ndarray
    ), "The constraint matrix should be a Numpy array."
    assert (
        len(A.shape) == 2
    ), "The constraints matrix should be a 2D Numpy array."
    assert isinstance(
        s, np.ndarray
    ), "The margins vector should be a Numpy array."
    assert len(s.shape) == 1, "The margins vector should be a 1D Numpy array."
    assert (
        np.shape(A)[0] == len(s)
    ), "The number of linear constraints should be equal to the number of margins."
    assert (
        np.shape(A)[1] == len(y)
    ), "The number of coefficients for the linear constraints should be equal to the number of observations."

    if l is None:
        l = np.zeros(len(y))
    assert isinstance(
        l, np.ndarray
    ), "The vector of lower bounds should be a Numpy array."
    assert (
        len(l.shape) == 1
    ), "The vector of lower bounds should be a 1D Numpy array."
    assert len(y) == len(
        l
    ), "Observations and lower bounds vectors should have the same length."
    assert np.all(l >= 0.0), "The lower bounds must be positive."
    assert np.all(
        l <= y
    ), "The observations must be superior or equal to the corresponding lower bounds."

    if h is None:
        h = np.ones(len(y))
    assert isinstance(
        h, np.ndarray
    ), "The vector of upper bounds should be a Numpy array."
    assert (
        len(h.shape) == 1
    ), "The vector of upper bounds should be a 1D Numpy array."
    assert len(y) == len(
        h
    ), "Observations and upper bounds vectors should have the same length."
    assert np.all(h > 0.0), "The upper bounds must be strictly positive."
    assert np.all(
        h >= y
    ), "The observations must be inferior or equal to the correspondings upper bounds."

    assert np.all(
        l < h
    ), "The lower bounds must be stricty inferior to the correspondings upper bounds."

    if q is None:
        q = np.ones(len(y))
    lambda_k = np.zeros(A.shape[0])
    beta = np.copy(y)
    epsilon = 1.0
    iter_eps = 0
    while (epsilon > 1.0e-10) & (iter_eps < max_iter):
        Phi = np.matmul(
            A,
            (
                l * (h - y)
                + h
                * (y - l)
                * np.exp(-q * np.matmul(np.transpose(A), lambda_k))
            )
            / (
                (h - y)
                + (y - l) * np.exp(-q * np.matmul(np.transpose(A), lambda_k))
            ),
        )
        D = np.diag(
            -q
            * ((y - l) * (h - y) * (h - l))
            / np.square(
                (h - y)
                + (y - l) * np.exp(-q * np.matmul(np.transpose(A), lambda_k))
            )
        )
        J = np.matmul(np.matmul(A, D), np.transpose(A))
        delta_lambda = cg(J, Phi - s)[0]
        gamma = gamma0
        iter_gam = 0
        lambda_k = lambda_k - gamma * delta_lambda
        beta = (
            l * (h - y)
            + h * (y - l) * np.exp(-q * np.matmul(np.transpose(A), lambda_k))
        ) / (
            (h - y)
            + (y - l) * np.exp(-q * np.matmul(np.transpose(A), lambda_k))
        )
        if iter_eps > 0:
            while (np.mean(np.abs(s - np.matmul(A, beta))) > epsilon) & (
                iter_gam < max_iter
            ):
                gamma = gamma / 2.0
                iter_gam = iter_gam + 1
                lambda_k = lambda_k - gamma * delta_lambda
                beta = (
                    l * (h - y)
                    + h
                    * (y - l)
                    * np.exp(-q * np.matmul(np.transpose(A), lambda_k))
                ) / (
                    (h - y)
                    + (y - l)
                    * np.exp(-q * np.matmul(np.transpose(A), lambda_k))
                )
        epsilon = np.mean(np.abs(s - np.matmul(A, beta)))
        iter_eps = iter_eps + 1
    return (beta, lambda_k, iter_eps)
