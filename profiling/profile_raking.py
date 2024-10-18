"""Script to profile the computation time of the raking.
See https://kernprof.readthedocs.io/en/latest/ for the documentation."""

import numpy as np

from line_profiler import profile
from numpy.linalg import solve, svd
from scipy.sparse.linalg import cg, minres


def create_test_matrix(I: int, J: int, K: int) -> tuple[np.ndarray, np.ndarray]:
    """Creates a balanced table, returns raked data vector and corresponding margins.

    Parameters
    ----------
    I : int
        Number of categories for the first variable
    J : int
        Number of categories for the second variable
    K : int
        Number of categories for the third variable

    Returns
    -------
    beta_flat : np.ndarray
        Observation vector
    s_cause : np.ndarray
        Margins vector
    """
    rng = np.random.default_rng(0)
    beta_ijk = rng.uniform(low=2.0, high=3.0, size=(I, J, K))
    beta_00k = np.sum(beta_ijk, axis=(0, 1))
    beta_i0k = np.sum(beta_ijk, axis=1)
    beta_0jk = np.sum(beta_ijk, axis=0)
    beta1 = np.concatenate(
        (beta_00k.reshape((1, 1, K)), beta_i0k.reshape(I, 1, K)), axis=0
    )
    beta2 = np.concatenate((beta_0jk.reshape((1, J, K)), beta_ijk), axis=0)
    beta = np.concatenate((beta1, beta2), axis=1)
    beta = beta.flatten("F")
    beta_i00 = np.sum(beta_ijk, axis=(1, 2))
    s_cause = np.array([np.sum(beta_i00)] + beta_i00.tolist())
    return (beta, s_cause)


def add_noise(beta: np.ndarray, sigma: float) -> np.ndarray:
    """Add noise to the initial raked data.

    Parameters
    ----------
    beta : np.ndarray
        Already raked data vector
    sigma: float
        Standard deviation of the noise

    Returns
    -------
    beta: np.ndarray
        Unbalanced vector with noise
    """
    rng = np.random.default_rng(0)
    beta = beta + rng.normal(0.0, sigma, size=len(beta))
    return beta


@profile
def constraints_USHD(
    s_cause: np.ndarray,
    I: int,
    J: int,
    K: int,
    rtol: float = 1e-05,
    atol: float = 1e-08,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute the constraints matrix A and the margins vector s for the USHD use case.

    This will define the raking optimization problem:
        min_beta f(beta,y) s.t. A beta = s
    The input margins are the 1 + I values:
        - beta_000 = Total number of deaths (all causes, all races, at the state level)
        - beta_i00 = Number of deaths for cause i (all races, at the state level)

    Parameters
    ----------
    s_cause : np.ndarray
        Total number of deaths (all causes, and each cause)
    I : int
        Number of causes of deaths
    J : int
        Number of races and ethnicities
    K : int
        Number of counties
    rtol : float
        Relative tolerance to check whether the margins are consistant. See numpy.allclose documentation for details.
    atol : float
        Absolute tolerance to check whether the margins are consistant. See numpy.allclose documentation for details.

    Returns
    -------
    A : np.ndarray
        (I + 2 * K + J * K + (I - 1) * K) * ((I + 1) * (J + 1) * K) constraints matrix
    s : np.ndarray
        length (I + 2 * K + J * K + (I - 1) * K) margins vector
    """
    assert isinstance(
        I, int
    ), "The number of causes of deaths must be an integer."
    assert I > 1, "The number of causes of deaths must be higher than 1."
    assert isinstance(
        J, int
    ), "The number of races and ethnicities must be an integer."
    assert J > 1, "The number of races and ethnicities must be higher than 1."
    assert isinstance(K, int), "The number of counties must be an integer."
    assert K > 1, "The number of counties must be higher than 1."

    assert isinstance(
        s_cause, np.ndarray
    ), "The margins vector for the causes of death must be a Numpy array."
    assert (
        len(s_cause.shape) == 1
    ), "The margins vector for the causes of death must be a 1D Numpy array."
    assert np.all(
        s_cause >= 0.0
    ), "The number of deaths for each cause must be positive or null."
    assert (
        len(s_cause) == I + 1
    ), "The length of the margins vector for the causes of death must be equal to 1 + number of causes."

    assert np.allclose(
        s_cause[0], np.sum(s_cause[1:]), rtol, atol
    ), "The all-causes number of deaths must be equal to the sum of the numbers of deaths per cause."

    A = np.zeros((I + 2 * K + J * K + (I - 1) * K, (I + 1) * (J + 1) * K))
    s = np.zeros(I + 2 * K + J * K + (I - 1) * K)
    # Constraint sum_k=0,...,K-1 beta_i,0,k = s_i for i=1,...,I
    for i in range(0, I):
        for k in range(0, K):
            A[i, k * (I + 1) * (J + 1) + i + 1] = 1
        s[i] = s_cause[i + 1]
    # Constraint sum_i=1,...,I beta_i,0,k - beta_0,0,k = 0 for k=0,...,K-1
    for k in range(0, K):
        for i in range(1, I + 1):
            A[I + k, k * (I + 1) * (J + 1) + i] = 1
        A[I + k, k * (I + 1) * (J + 1)] = -1
    # Constraint sum_j=1,...,J beta_0,j,k - beta_0,0,k = 0 for k=0,...,K-1
    for k in range(0, K):
        for j in range(1, J + 1):
            A[I + K + k, k * (I + 1) * (J + 1) + j * (I + 1)] = 1
        A[I + K + k, k * (I + 1) * (J + 1)] = -1
    # Constraint sum_i=1,...,I beta_i,j,k - beta_0,j,k = 0 for j=1,...,J and k=0,...,K-1
    for k in range(0, K):
        for j in range(1, J + 1):
            for i in range(1, I + 1):
                A[
                    I + 2 * K + k * J + j - 1,
                    k * (I + 1) * (J + 1) + j * (I + 1) + i,
                ] = 1
            A[
                I + 2 * K + k * J + j - 1, k * (I + 1) * (J + 1) + j * (I + 1)
            ] = -1
    # Constraint sum_j=1,...,J beta_i,j,k - beta_i,0,k = 0 for i=1,...,I and k=0,...,K-1
    for k in range(0, K):
        for i in range(1, I):
            for j in range(1, J + 1):
                A[
                    I + 2 * K + J * K + k * (I - 1) + i - 1,
                    k * (I + 1) * (J + 1) + j * (I + 1) + i,
                ] = 1
            A[
                I + 2 * K + J * K + k * (I - 1) + i - 1,
                k * (I + 1) * (J + 1) + i,
            ] = -1
    return (A, s)


@profile
def solve_system_linalg(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Solve A x = b using Numpy linalg.solve.

    Parameters
    ----------
    A : np.ndarray
    b : np.ndarray

    Returns
    -------
    x : np.ndarray
    """
    x = solve(A, b)
    return x


@profile
def solve_system_svd(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Solve A x = b by computing the SVD decomposition of A.

    Parameters
    ----------
    A : np.ndarray
    b : np.ndarray

    Returns
    -------
    x : np.ndarray
    """
    # Compute the Moore-Penrose pseudoinverse of A
    U, S, Vh = np.linalg.svd(A, full_matrices=True)
    V = np.transpose(Vh)
    # Invert diagonal matrix while dealing with 0 and near 0 values
    Sdiag = np.diag(S)
    Sdiag[np.abs(Sdiag) <= 1.0e-12] = 1.0e-12
    Sinv = 1.0 / Sdiag
    Sinv[np.abs(Sdiag) <= 1.0e-12] = 0.0
    A_plus = np.matmul(V, np.matmul(Sinv, np.transpose(U)))
    x = np.matmul(A_plus, b)
    return x


@profile
def solve_system_cg(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Solve A x = b using conjugate gradient.

    Parameters
    ----------
    A : np.ndarray
    b : np.ndarray

    Returns
    -------
    x : np.ndarray
    """
    x = cg(A, b)[0]
    return x


@profile
def solve_system_cg_rtol(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Solve A x = b using conjugate gradient.

    Set the tolerance to 1e-2.

    Parameters
    ----------
    A : np.ndarray
    b : np.ndarray

    Returns
    -------
    x : np.ndarray
    """
    x = cg(A, b, rtol=1e-2)[0]
    return x


@profile
def solve_system_cg_maxiter(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Solve A x = b using conjugate gradient.

    Set the number of iteration to 100.

    Parameters
    ----------
    A : np.ndarray
    b : np.ndarray

    Returns
    -------
    x : np.ndarray
    """
    x = cg(A, b, maxiter=100)[0]
    return x


@profile
def solve_system_cg_rtol_maxiter(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Solve A x = b using conjugate gradient.

    Set the tolerance to 1e-2 and the number of iteration to 100.

    Parameters
    ----------
    A : np.ndarray
    b : np.ndarray

    Returns
    -------
    x : np.ndarray
    """
    x = cg(A, b, rtol=1e-2, maxiter=100)[0]
    return x


@profile
def solve_system_minres(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Solve A x = b using minimum residual iteration.

    Parameters
    ----------
    A : np.ndarray
    b : np.ndarray

    Returns
    -------
    x : np.ndarray
    """
    x = minres(A, b)[0]
    return x


@profile
def solve_system_minres_rtol(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Solve A x = b using minimum residual iteration.

    Set the tolerance to 1e-2.

    Parameters
    ----------
    A : np.ndarray
    b : np.ndarray

    Returns
    -------
    x : np.ndarray
    """
    x = minres(A, b, rtol=1e-2)[0]
    return x


@profile
def solve_system_minres_maxiter(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Solve A x = b using minimum residual iteration.

    Set the number of iteration to 100.

    Parameters
    ----------
    A : np.ndarray
    b : np.ndarray

    Returns
    -------
    x : np.ndarray
    """
    x = minres(A, b, maxiter=100)[0]
    return x


@profile
def solve_system_minres_rtol_maxiter(
    A: np.ndarray, b: np.ndarray
) -> np.ndarray:
    """Solve A x = b using minimum residual iteration.

    Set the tolerance to 1e-2 and the number of iteration to 100.

    Parameters
    ----------
    A : np.ndarray
    b : np.ndarray

    Returns
    -------
    x : np.ndarray
    """
    x = minres(A, b, rtol=1e-2, maxiter=100)[0]
    return x


@profile
def raking_chi2(
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
    A: np.ndarray
        Constraints matrix (output of a function from the compute_constraints module)
    s: np.ndarray
        Margin vector (output of a function from the compute_constraints module)
    q: np.ndarray
        Vector of weights (default to all 1)

    Returns
    -------
    betas: list
        List of vectors of reaked values (np.ndarray)
    lambdas: list
        List of duals (needed for the uncertainty computation, np.ndarray)
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
    lambdas = []
    betas = []

    # linalg.solve
    lambda_k = solve_system_linalg(Phi, s_hat - s)
    beta = y * (1 - q * np.matmul(np.transpose(A), lambda_k))
    lambdas.append(lambda_k)
    betas.append(beta)
    # SVD
    lambda_k = solve_system_svd(Phi, s_hat - s)
    beta = y * (1 - q * np.matmul(np.transpose(A), lambda_k))
    lambdas.append(lambda_k)
    betas.append(beta)
    # CG 1
    lambda_k = solve_system_cg(Phi, s_hat - s)
    beta = y * (1 - q * np.matmul(np.transpose(A), lambda_k))
    lambdas.append(lambda_k)
    betas.append(beta)
    # CG 2
    lambda_k = solve_system_cg_rtol(Phi, s_hat - s)
    beta = y * (1 - q * np.matmul(np.transpose(A), lambda_k))
    lambdas.append(lambda_k)
    betas.append(beta)
    # CG 3
    lambda_k = solve_system_cg_maxiter(Phi, s_hat - s)
    beta = y * (1 - q * np.matmul(np.transpose(A), lambda_k))
    lambdas.append(lambda_k)
    betas.append(beta)
    # CG 4
    lambda_k = solve_system_cg_rtol_maxiter(Phi, s_hat - s)
    beta = y * (1 - q * np.matmul(np.transpose(A), lambda_k))
    lambdas.append(lambda_k)
    betas.append(beta)
    # MinRes 1
    lambda_k = solve_system_minres(Phi, s_hat - s)
    beta = y * (1 - q * np.matmul(np.transpose(A), lambda_k))
    lambdas.append(lambda_k)
    betas.append(beta)
    # MinRes 2
    lambda_k = solve_system_minres_rtol(Phi, s_hat - s)
    beta = y * (1 - q * np.matmul(np.transpose(A), lambda_k))
    lambdas.append(lambda_k)
    betas.append(beta)
    # MinRes 3
    lambda_k = solve_system_minres_maxiter(Phi, s_hat - s)
    beta = y * (1 - q * np.matmul(np.transpose(A), lambda_k))
    lambdas.append(lambda_k)
    betas.append(beta)
    # MinRes 4
    lambda_k = solve_system_minres_rtol_maxiter(Phi, s_hat - s)
    beta = y * (1 - q * np.matmul(np.transpose(A), lambda_k))
    lambdas.append(lambda_k)
    betas.append(beta)

    return (betas, lambdas)


@profile
def raking_entropic(
    y: np.ndarray,
    A: np.ndarray,
    s: np.ndarray,
    q: np.ndarray = None,
    gamma0: float = 1.0,
    max_iter: int = 500,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Raking using the entropic distance f(beta, y) = beta log(beta/y) + y - beta.

    This will solve the problem:
        min_beta 1/q f(beta, y) s.t. A beta = s

    Parameters
    ----------
    y : np.ndarray
        Vector of observations
    A: np.ndarray
        Constraints matrix (output of a function from the compute_constraints module)
    s: np.ndarray
        Margin vector (output of a function from the compute_constraints module)
    q: np.ndarray
        Vector of weights (default to all 1)
    gamma0: float
        Initial value for line search
    max_iter: int
        Number of iterations for Newton's root finding method

    Returns
    -------
    beta: np.ndarray
        Vector of reaked values
    lambda_k: np.ndarray
        Dual (needed for th uncertainty computation)
    iters_eps: int
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

    if q is None:
        q = np.ones(len(y))
    s_hat = np.matmul(A, y)
    lambdas = []
    betas = []
    iters_eps = []

    # linalg.solve
    lambda_k = np.zeros(A.shape[0])
    beta = np.copy(y)
    epsilon = 1.0
    iter_eps = 0
    while (epsilon > 1.0e-10) & (iter_eps < max_iter):
        Phi = np.matmul(
            A, y * (1.0 - np.exp(-q * np.matmul(np.transpose(A), lambda_k)))
        )
        D = np.diag(y * q * np.exp(-q * np.matmul(np.transpose(A), lambda_k)))
        J = np.matmul(np.matmul(A, D), np.transpose(A))
        delta_lambda = solve_system_linalg(J, Phi - s_hat + s)
        gamma = gamma0
        iter_gam = 0
        lambda_k = lambda_k - gamma * delta_lambda
        beta = y * np.exp(-q * np.matmul(np.transpose(A), lambda_k))
        if iter_eps > 0:
            while (np.mean(np.abs(s - np.matmul(A, beta))) > epsilon) & (
                iter_gam < max_iter
            ):
                gamma = gamma / 2.0
                iter_gam = iter_gam + 1
                lambda_k = lambda_k - gamma * delta_lambda
                beta = y * np.exp(-q * np.matmul(np.transpose(A), lambda_k))
        epsilon = np.mean(np.abs(s - np.matmul(A, beta)))
        iter_eps = iter_eps + 1
    lambdas.append(lambda_k)
    betas.append(beta)
    iters_eps.append(iter_eps)

    # SVD
    lambda_k = np.zeros(A.shape[0])
    beta = np.copy(y)
    epsilon = 1.0
    iter_eps = 0
    while (epsilon > 1.0e-10) & (iter_eps < max_iter):
        Phi = np.matmul(
            A, y * (1.0 - np.exp(-q * np.matmul(np.transpose(A), lambda_k)))
        )
        D = np.diag(y * q * np.exp(-q * np.matmul(np.transpose(A), lambda_k)))
        J = np.matmul(np.matmul(A, D), np.transpose(A))
        delta_lambda = solve_system_svd(J, Phi - s_hat + s)
        gamma = gamma0
        iter_gam = 0
        lambda_k = lambda_k - gamma * delta_lambda
        beta = y * np.exp(-q * np.matmul(np.transpose(A), lambda_k))
        if iter_eps > 0:
            while (np.mean(np.abs(s - np.matmul(A, beta))) > epsilon) & (
                iter_gam < max_iter
            ):
                gamma = gamma / 2.0
                iter_gam = iter_gam + 1
                lambda_k = lambda_k - gamma * delta_lambda
                beta = y * np.exp(-q * np.matmul(np.transpose(A), lambda_k))
        epsilon = np.mean(np.abs(s - np.matmul(A, beta)))
        iter_eps = iter_eps + 1
    lambdas.append(lambda_k)
    betas.append(beta)
    iters_eps.append(iter_eps)

    # CG 1
    lambda_k = np.zeros(A.shape[0])
    beta = np.copy(y)
    epsilon = 1.0
    iter_eps = 0
    while (epsilon > 1.0e-10) & (iter_eps < max_iter):
        Phi = np.matmul(
            A, y * (1.0 - np.exp(-q * np.matmul(np.transpose(A), lambda_k)))
        )
        D = np.diag(y * q * np.exp(-q * np.matmul(np.transpose(A), lambda_k)))
        J = np.matmul(np.matmul(A, D), np.transpose(A))
        delta_lambda = solve_system_cg(J, Phi - s_hat + s)
        gamma = gamma0
        iter_gam = 0
        lambda_k = lambda_k - gamma * delta_lambda
        beta = y * np.exp(-q * np.matmul(np.transpose(A), lambda_k))
        if iter_eps > 0:
            while (np.mean(np.abs(s - np.matmul(A, beta))) > epsilon) & (
                iter_gam < max_iter
            ):
                gamma = gamma / 2.0
                iter_gam = iter_gam + 1
                lambda_k = lambda_k - gamma * delta_lambda
                beta = y * np.exp(-q * np.matmul(np.transpose(A), lambda_k))
        epsilon = np.mean(np.abs(s - np.matmul(A, beta)))
        iter_eps = iter_eps + 1
    lambdas.append(lambda_k)
    betas.append(beta)
    iters_eps.append(iter_eps)

    # CG 2
    lambda_k = np.zeros(A.shape[0])
    beta = np.copy(y)
    epsilon = 1.0
    iter_eps = 0
    while (epsilon > 1.0e-10) & (iter_eps < max_iter):
        Phi = np.matmul(
            A, y * (1.0 - np.exp(-q * np.matmul(np.transpose(A), lambda_k)))
        )
        D = np.diag(y * q * np.exp(-q * np.matmul(np.transpose(A), lambda_k)))
        J = np.matmul(np.matmul(A, D), np.transpose(A))
        delta_lambda = solve_system_cg_rtol(J, Phi - s_hat + s)
        gamma = gamma0
        iter_gam = 0
        lambda_k = lambda_k - gamma * delta_lambda
        beta = y * np.exp(-q * np.matmul(np.transpose(A), lambda_k))
        if iter_eps > 0:
            while (np.mean(np.abs(s - np.matmul(A, beta))) > epsilon) & (
                iter_gam < max_iter
            ):
                gamma = gamma / 2.0
                iter_gam = iter_gam + 1
                lambda_k = lambda_k - gamma * delta_lambda
                beta = y * np.exp(-q * np.matmul(np.transpose(A), lambda_k))
        epsilon = np.mean(np.abs(s - np.matmul(A, beta)))
        iter_eps = iter_eps + 1
    lambdas.append(lambda_k)
    betas.append(beta)
    iters_eps.append(iter_eps)

    # CG 3
    lambda_k = np.zeros(A.shape[0])
    beta = np.copy(y)
    epsilon = 1.0
    iter_eps = 0
    while (epsilon > 1.0e-10) & (iter_eps < max_iter):
        Phi = np.matmul(
            A, y * (1.0 - np.exp(-q * np.matmul(np.transpose(A), lambda_k)))
        )
        D = np.diag(y * q * np.exp(-q * np.matmul(np.transpose(A), lambda_k)))
        J = np.matmul(np.matmul(A, D), np.transpose(A))
        delta_lambda = solve_system_cg_maxiter(J, Phi - s_hat + s)
        gamma = gamma0
        iter_gam = 0
        lambda_k = lambda_k - gamma * delta_lambda
        beta = y * np.exp(-q * np.matmul(np.transpose(A), lambda_k))
        if iter_eps > 0:
            while (np.mean(np.abs(s - np.matmul(A, beta))) > epsilon) & (
                iter_gam < max_iter
            ):
                gamma = gamma / 2.0
                iter_gam = iter_gam + 1
                lambda_k = lambda_k - gamma * delta_lambda
                beta = y * np.exp(-q * np.matmul(np.transpose(A), lambda_k))
        epsilon = np.mean(np.abs(s - np.matmul(A, beta)))
        iter_eps = iter_eps + 1
    lambdas.append(lambda_k)
    betas.append(beta)
    iters_eps.append(iter_eps)

    # CG 4
    lambda_k = np.zeros(A.shape[0])
    beta = np.copy(y)
    epsilon = 1.0
    iter_eps = 0
    while (epsilon > 1.0e-10) & (iter_eps < max_iter):
        Phi = np.matmul(
            A, y * (1.0 - np.exp(-q * np.matmul(np.transpose(A), lambda_k)))
        )
        D = np.diag(y * q * np.exp(-q * np.matmul(np.transpose(A), lambda_k)))
        J = np.matmul(np.matmul(A, D), np.transpose(A))
        delta_lambda = solve_system_cg_rtol_maxiter(J, Phi - s_hat + s)
        gamma = gamma0
        iter_gam = 0
        lambda_k = lambda_k - gamma * delta_lambda
        beta = y * np.exp(-q * np.matmul(np.transpose(A), lambda_k))
        if iter_eps > 0:
            while (np.mean(np.abs(s - np.matmul(A, beta))) > epsilon) & (
                iter_gam < max_iter
            ):
                gamma = gamma / 2.0
                iter_gam = iter_gam + 1
                lambda_k = lambda_k - gamma * delta_lambda
                beta = y * np.exp(-q * np.matmul(np.transpose(A), lambda_k))
        epsilon = np.mean(np.abs(s - np.matmul(A, beta)))
        iter_eps = iter_eps + 1
    lambdas.append(lambda_k)
    betas.append(beta)
    iters_eps.append(iter_eps)

    # MinRes 1
    lambda_k = np.zeros(A.shape[0])
    beta = np.copy(y)
    epsilon = 1.0
    iter_eps = 0
    while (epsilon > 1.0e-10) & (iter_eps < max_iter):
        Phi = np.matmul(
            A, y * (1.0 - np.exp(-q * np.matmul(np.transpose(A), lambda_k)))
        )
        D = np.diag(y * q * np.exp(-q * np.matmul(np.transpose(A), lambda_k)))
        J = np.matmul(np.matmul(A, D), np.transpose(A))
        delta_lambda = solve_system_minres(J, Phi - s_hat + s)
        gamma = gamma0
        iter_gam = 0
        lambda_k = lambda_k - gamma * delta_lambda
        beta = y * np.exp(-q * np.matmul(np.transpose(A), lambda_k))
        if iter_eps > 0:
            while (np.mean(np.abs(s - np.matmul(A, beta))) > epsilon) & (
                iter_gam < max_iter
            ):
                gamma = gamma / 2.0
                iter_gam = iter_gam + 1
                lambda_k = lambda_k - gamma * delta_lambda
                beta = y * np.exp(-q * np.matmul(np.transpose(A), lambda_k))
        epsilon = np.mean(np.abs(s - np.matmul(A, beta)))
        iter_eps = iter_eps + 1
    lambdas.append(lambda_k)
    betas.append(beta)
    iters_eps.append(iter_eps)

    # MinRes 2
    lambda_k = np.zeros(A.shape[0])
    beta = np.copy(y)
    epsilon = 1.0
    iter_eps = 0
    while (epsilon > 1.0e-10) & (iter_eps < max_iter):
        Phi = np.matmul(
            A, y * (1.0 - np.exp(-q * np.matmul(np.transpose(A), lambda_k)))
        )
        D = np.diag(y * q * np.exp(-q * np.matmul(np.transpose(A), lambda_k)))
        J = np.matmul(np.matmul(A, D), np.transpose(A))
        delta_lambda = solve_system_minres_rtol(J, Phi - s_hat + s)
        gamma = gamma0
        iter_gam = 0
        lambda_k = lambda_k - gamma * delta_lambda
        beta = y * np.exp(-q * np.matmul(np.transpose(A), lambda_k))
        if iter_eps > 0:
            while (np.mean(np.abs(s - np.matmul(A, beta))) > epsilon) & (
                iter_gam < max_iter
            ):
                gamma = gamma / 2.0
                iter_gam = iter_gam + 1
                lambda_k = lambda_k - gamma * delta_lambda
                beta = y * np.exp(-q * np.matmul(np.transpose(A), lambda_k))
        epsilon = np.mean(np.abs(s - np.matmul(A, beta)))
        iter_eps = iter_eps + 1
    lambdas.append(lambda_k)
    betas.append(beta)
    iters_eps.append(iter_eps)

    # MinRes 3
    lambda_k = np.zeros(A.shape[0])
    beta = np.copy(y)
    epsilon = 1.0
    iter_eps = 0
    while (epsilon > 1.0e-10) & (iter_eps < max_iter):
        Phi = np.matmul(
            A, y * (1.0 - np.exp(-q * np.matmul(np.transpose(A), lambda_k)))
        )
        D = np.diag(y * q * np.exp(-q * np.matmul(np.transpose(A), lambda_k)))
        J = np.matmul(np.matmul(A, D), np.transpose(A))
        delta_lambda = solve_system_minres_maxiter(J, Phi - s_hat + s)
        gamma = gamma0
        iter_gam = 0
        lambda_k = lambda_k - gamma * delta_lambda
        beta = y * np.exp(-q * np.matmul(np.transpose(A), lambda_k))
        if iter_eps > 0:
            while (np.mean(np.abs(s - np.matmul(A, beta))) > epsilon) & (
                iter_gam < max_iter
            ):
                gamma = gamma / 2.0
                iter_gam = iter_gam + 1
                lambda_k = lambda_k - gamma * delta_lambda
                beta = y * np.exp(-q * np.matmul(np.transpose(A), lambda_k))
        epsilon = np.mean(np.abs(s - np.matmul(A, beta)))
        iter_eps = iter_eps + 1
    lambdas.append(lambda_k)
    betas.append(beta)
    iters_eps.append(iter_eps)

    # MinRes 4
    lambda_k = np.zeros(A.shape[0])
    beta = np.copy(y)
    epsilon = 1.0
    iter_eps = 0
    while (epsilon > 1.0e-10) & (iter_eps < max_iter):
        Phi = np.matmul(
            A, y * (1.0 - np.exp(-q * np.matmul(np.transpose(A), lambda_k)))
        )
        D = np.diag(y * q * np.exp(-q * np.matmul(np.transpose(A), lambda_k)))
        J = np.matmul(np.matmul(A, D), np.transpose(A))
        delta_lambda = solve_system_minres_rtol_maxiter(J, Phi - s_hat + s)
        gamma = gamma0
        iter_gam = 0
        lambda_k = lambda_k - gamma * delta_lambda
        beta = y * np.exp(-q * np.matmul(np.transpose(A), lambda_k))
        if iter_eps > 0:
            while (np.mean(np.abs(s - np.matmul(A, beta))) > epsilon) & (
                iter_gam < max_iter
            ):
                gamma = gamma / 2.0
                iter_gam = iter_gam + 1
                lambda_k = lambda_k - gamma * delta_lambda
                beta = y * np.exp(-q * np.matmul(np.transpose(A), lambda_k))
        epsilon = np.mean(np.abs(s - np.matmul(A, beta)))
        iter_eps = iter_eps + 1
    lambdas.append(lambda_k)
    betas.append(beta)
    iters_eps.append(iter_eps)

    return (betas, lambdas, iters_eps)


def main():
    I = 3
    J = 5
    K = 254
    sigma = 0.1
    (beta, s_cause) = create_test_matrix(I, J, K)
    y = add_noise(beta, sigma)
    (A, s) = constraints_USHD(s_cause, I, J, K)
    (beta_chi2, lambda_chi2) = raking_chi2(y, A, s)
    beta_chi2_linalg = beta_chi2[0]
    beta_chi2_svd = beta_chi2[1]
    beta_chi2_cg1 = beta_chi2[2]
    beta_chi2_cg2 = beta_chi2[3]
    beta_chi2_cg3 = beta_chi2[4]
    beta_chi2_cg4 = beta_chi2[5]
    beta_chi2_minres1 = beta_chi2[6]
    beta_chi2_minres2 = beta_chi2[7]
    beta_chi2_minres3 = beta_chi2[8]
    beta_chi2_minres4 = beta_chi2[9]
    (beta_entropic, lambda_entropic, num_iters) = raking_entropic(y, A, s)
    beta_entropic_linalg = beta_entropic[0]
    beta_entropic_svd = beta_entropic[1]
    beta_entropic_cg1 = beta_entropic[2]
    beta_entropic_cg2 = beta_entropic[3]
    beta_entropic_cg3 = beta_entropic[4]
    beta_entropic_cg4 = beta_entropic[5]
    beta_entropic_minres1 = beta_entropic[6]
    beta_entropic_minres2 = beta_entropic[7]
    beta_entropic_minres3 = beta_entropic[8]
    beta_entropic_minres4 = beta_entropic[9]
    num_iters_linalg = num_iters[0]
    num_iters_svd = num_iters[1]
    num_iters_cg1 = num_iters[2]
    num_iters_cg2 = num_iters[3]
    num_iters_cg3 = num_iters[4]
    num_iters_cg4 = num_iters[5]
    num_iters_minres1 = num_iters[6]
    num_iters_minres2 = num_iters[7]
    num_iters_minres3 = num_iters[8]
    num_iters_minres4 = num_iters[9]

    print(
        "MAE of constraints for chi2 linalg:",
        np.mean(np.abs(s - np.matmul(A, beta_chi2_linalg))),
    )
    print(
        "MAE of constraints for chi2 svd:",
        np.mean(np.abs(s - np.matmul(A, beta_chi2_svd))),
    )
    print(
        "MAE of constraints for chi2 cg1:",
        np.mean(np.abs(s - np.matmul(A, beta_chi2_cg1))),
    )
    print(
        "MAE of constraints for chi2 cg2:",
        np.mean(np.abs(s - np.matmul(A, beta_chi2_cg2))),
    )
    print(
        "MAE of constraints for chi2 cg3:",
        np.mean(np.abs(s - np.matmul(A, beta_chi2_cg3))),
    )
    print(
        "MAE of constraints for chi2 cg4:",
        np.mean(np.abs(s - np.matmul(A, beta_chi2_cg4))),
    )
    print(
        "MAE of constraints for chi2 minres1:",
        np.mean(np.abs(s - np.matmul(A, beta_chi2_minres1))),
    )
    print(
        "MAE of constraints for chi2 minres2:",
        np.mean(np.abs(s - np.matmul(A, beta_chi2_minres2))),
    )
    print(
        "MAE of constraints for chi2 minres3:",
        np.mean(np.abs(s - np.matmul(A, beta_chi2_minres3))),
    )
    print(
        "MAE of constraints for chi2 minres4:",
        np.mean(np.abs(s - np.matmul(A, beta_chi2_minres4))),
    )
    print("\n")
    print(
        "MAE of constraints for entropic linalg:",
        np.mean(np.abs(s - np.matmul(A, beta_entropic_linalg))),
    )
    print(
        "MAE of constraints for entropic svd:",
        np.mean(np.abs(s - np.matmul(A, beta_entropic_svd))),
    )
    print(
        "MAE of constraints for entropic cg1:",
        np.mean(np.abs(s - np.matmul(A, beta_entropic_cg1))),
    )
    print(
        "MAE of constraints for entropic cg2:",
        np.mean(np.abs(s - np.matmul(A, beta_entropic_cg2))),
    )
    print(
        "MAE of constraints for entropic cg3:",
        np.mean(np.abs(s - np.matmul(A, beta_entropic_cg3))),
    )
    print(
        "MAE of constraints for entropic cg4:",
        np.mean(np.abs(s - np.matmul(A, beta_entropic_cg4))),
    )
    print(
        "MAE of constraints for entropic minres1:",
        np.mean(np.abs(s - np.matmul(A, beta_entropic_minres1))),
    )
    print(
        "MAE of constraints for entropic minres2:",
        np.mean(np.abs(s - np.matmul(A, beta_entropic_minres2))),
    )
    print(
        "MAE of constraints for entropic minres3:",
        np.mean(np.abs(s - np.matmul(A, beta_entropic_minres3))),
    )
    print(
        "MAE of constraints for entropic minres4:",
        np.mean(np.abs(s - np.matmul(A, beta_entropic_minres4))),
    )
    print("\n")
    print("Iterations for entropic linalg:", num_iters_linalg)
    print("Iterations for entropic svd:", num_iters_svd)
    print("Iterations for entropic cg1:", num_iters_cg1)
    print("Iterations for entropic cg2:", num_iters_cg2)
    print("Iterations for entropic cg3:", num_iters_cg3)
    print("Iterations for entropic cg4:", num_iters_cg4)
    print("Iterations for entropic minres1:", num_iters_minres1)
    print("Iterations for entropic minres2:", num_iters_minres2)
    print("Iterations for entropic minres3:", num_iters_minres3)
    print("Iterations for entropic minres4:", num_iters_minres4)
    print("\n")
    print(
        "MAPE between beta chi2 - linalg and svd:",
        np.mean(np.abs((beta_chi2_linalg - beta_chi2_svd) / beta_chi2_linalg)),
    )
    print(
        "MAPE between beta chi2 - linalg and cg1:",
        np.mean(np.abs((beta_chi2_linalg - beta_chi2_cg1) / beta_chi2_linalg)),
    )
    print(
        "MAPE between beta chi2 linalg and cg2:",
        np.mean(np.abs((beta_chi2_linalg - beta_chi2_cg2) / beta_chi2_linalg)),
    )
    print(
        "MAPE between beta chi2 linalg and cg3:",
        np.mean(np.abs((beta_chi2_linalg - beta_chi2_cg3) / beta_chi2_linalg)),
    )
    print(
        "MAPE between beta chi2 linalg and cg4:",
        np.mean(np.abs((beta_chi2_linalg - beta_chi2_cg4) / beta_chi2_linalg)),
    )
    print(
        "MAPE between beta chi2 - linalg and minres1:",
        np.mean(
            np.abs((beta_chi2_linalg - beta_chi2_minres1) / beta_chi2_linalg)
        ),
    )
    print(
        "MAPE between beta chi2 linalg and minres2:",
        np.mean(
            np.abs((beta_chi2_linalg - beta_chi2_minres2) / beta_chi2_linalg)
        ),
    )
    print(
        "MAPE between beta chi2 linalg and minres3:",
        np.mean(
            np.abs((beta_chi2_linalg - beta_chi2_minres3) / beta_chi2_linalg)
        ),
    )
    print(
        "MAPE between beta chi2 linalg and minres4:",
        np.mean(
            np.abs((beta_chi2_linalg - beta_chi2_minres4) / beta_chi2_linalg)
        ),
    )
    print("\n")
    print(
        "MAPE between beta entropic - linalg and svd:",
        np.mean(
            np.abs(
                (beta_entropic_linalg - beta_entropic_svd)
                / beta_entropic_linalg
            )
        ),
    )
    print(
        "MAPE between beta entropic - linalg and cg1:",
        np.mean(
            np.abs(
                (beta_entropic_linalg - beta_entropic_cg1)
                / beta_entropic_linalg
            )
        ),
    )
    print(
        "MAPE between beta entropic - linalg and cg2:",
        np.mean(
            np.abs(
                (beta_entropic_linalg - beta_entropic_cg2)
                / beta_entropic_linalg
            )
        ),
    )
    print(
        "MAPE between beta entropic - linalg and cg3:",
        np.mean(
            np.abs(
                (beta_entropic_linalg - beta_entropic_cg3)
                / beta_entropic_linalg
            )
        ),
    )
    print(
        "MAPE between beta entropic - linalg and cg4:",
        np.mean(
            np.abs(
                (beta_entropic_linalg - beta_entropic_cg4)
                / beta_entropic_linalg
            )
        ),
    )
    print(
        "MAPE between beta entropic - linalg and minres1:",
        np.mean(
            np.abs(
                (beta_entropic_linalg - beta_entropic_minres1)
                / beta_entropic_linalg
            )
        ),
    )
    print(
        "MAPE between beta entropic - linalg and minres2:",
        np.mean(
            np.abs(
                (beta_entropic_linalg - beta_entropic_minres2)
                / beta_entropic_linalg
            )
        ),
    )
    print(
        "MAPE between beta entropic - linalg and minres3:",
        np.mean(
            np.abs(
                (beta_entropic_linalg - beta_entropic_minres3)
                / beta_entropic_linalg
            )
        ),
    )
    print(
        "MAPE between beta entropic - linalg and minres4:",
        np.mean(
            np.abs(
                (beta_entropic_linalg - beta_entropic_minres4)
                / beta_entropic_linalg
            )
        ),
    )
    print("\n")
    print(
        "Difference between beta linalg - chi2 and entropic:",
        np.mean(
            np.abs((beta_chi2_linalg - beta_entropic_linalg) / beta_chi2_linalg)
        ),
    )


if __name__ == "__main__":
    main()
