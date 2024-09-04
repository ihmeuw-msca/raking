"""Script to profile the computation time of the uncertainty.
   See https://kernprof.readthedocs.io/en/latest/ for the documentation."""

import numpy as np

from line_profiler import profile
from scipy.linalg import lu_factor, lu_solve
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import cg, spsolve

def create_test_matrix(
    I: int,
    J: int,
    K: int
) -> tuple[np.ndarray, np.ndarray]:
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
    beta1 = np.concatenate((beta_00k.reshape((1, 1, K)), beta_i0k.reshape(I, 1, K)), axis=0)
    beta2 = np.concatenate((beta_0jk.reshape((1, J, K)), beta_ijk), axis=0)
    beta = np.concatenate((beta1, beta2), axis=1)
    beta = beta.flatten('F')
    beta_i00 = np.sum(beta_ijk, axis=(1, 2))
    s_cause = np.array([np.sum(beta_i00)] + beta_i00.tolist())
    return (beta, s_cause)

def add_noise(
    beta: np.ndarray,
    sigma: float
) -> np.ndarray:
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
    atol:float = 1e-08
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
    assert isinstance(I, int), \
        'The number of causes of deaths must be an integer.'
    assert I > 1, \
        'The number of causes of deaths must be higher than 1.'
    assert isinstance(J, int), \
        'The number of races and ethnicities must be an integer.'
    assert J > 1, \
        'The number of races and ethnicities must be higher than 1.'
    assert isinstance(K, int), \
        'The number of counties must be an integer.'
    assert K > 1, \
        'The number of counties must be higher than 1.'

    assert isinstance(s_cause, np.ndarray), \
        'The margins vector for the causes of death must be a Numpy array.'
    assert len(s_cause.shape) == 1, \
        'The margins vector for the causes of death must be a 1D Numpy array.'
    assert np.all(s_cause >= 0.0), \
        'The number of deaths for each cause must be positive or null.'
    assert len(s_cause) == I + 1, \
        'The length of the margins vector for the causes of death must be equal to 1 + number of causes.'
    
    assert np.allclose(s_cause[0], np.sum(s_cause[1:]), rtol, atol), \
        'The all-causes number of deaths must be equal to the sum of the numbers of deaths per cause.'

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
                A[I + 2 * K + k * J + j - 1, k * (I + 1) * (J + 1) + j * (I + 1) + i] = 1
            A[I + 2 * K + k * J + j - 1, k * (I + 1) * (J + 1) + j * (I + 1)] = -1
    # Constraint sum_j=1,...,J beta_i,j,k - beta_i,0,k = 0 for i=1,...,I and k=0,...,K-1
    for k in range(0, K):
        for i in range(1, I):
            for j in range(1, J + 1):
                A[I + 2 * K + J * K + k * (I - 1) + i - 1, k * (I + 1) * (J + 1) + j * (I + 1) + i] = 1
            A[I + 2 * K + J * K + k * (I - 1) + i - 1, k * (I + 1) * (J + 1) + i] = -1
    return (A, s)

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
    assert isinstance(y, np.ndarray), \
        'The vector of observations should be a Numpy array.'
    assert len(y.shape) == 1, \
        'The vector of observations should be a 1D Numpy array.'
    if q is not None:
        assert isinstance(q, np.ndarray), \
            'The vector of weights should be a Numpy array.'
        assert len(y.shape) == 1, \
            'The vector of weights should be a 1D Numpy array.'
        assert len(y) == len(q), \
            'Observations and weights vectors should have the same length.'
    assert isinstance(A, np.ndarray), \
        'The constraint matrix should be a Numpy array.'
    assert len(A.shape) == 2, \
        'The constraints matrix should be a 2D Numpy array.'
    assert isinstance(s, np.ndarray), \
        'The margins vector should be a Numpy array.'
    assert len(s.shape) == 1, \
        'The margins vector should be a 1D Numpy array.'
    assert np.shape(A)[0] == len(s), \
        'The number of linear constraints should be equal to the number of margins.'
    assert np.shape(A)[1] == len(y), \
        'The number of coefficients for the linear constraints should be equal to the number of observations.'

    if q is None:
        q = np.ones(len(y))
    s_hat = np.matmul(A, y)
    Phi = np.matmul(A, np.transpose(A * y * q))
    lambda_k = cg(Phi, s_hat - s)[0]
    beta = y * (1 - q * np.matmul(np.transpose(A), lambda_k))
    return (beta, lambda_k)

@profile
def raking_entropic(
    y: np.ndarray,
    A: np.ndarray,
    s: np.ndarray,
    q: np.ndarray = None,
    gamma0: float = 1.0,
    max_iter: int = 500
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
    assert isinstance(y, np.ndarray), \
        'The vector of observations should be a Numpy array.'
    assert len(y.shape) == 1, \
        'The vector of observations should be a 1D Numpy array.'
    if q is not None:
        assert isinstance(q, np.ndarray), \
            'The vector of weights should be a Numpy array.'
        assert len(q.shape) == 1, \
            'The vector of weights should be a 1D Numpy array.'
        assert len(y) == len(q), \
            'Observations and weights vectors should have the same length.'
    assert isinstance(A, np.ndarray), \
        'The constraint matrix should be a Numpy array.'
    assert len(A.shape) == 2, \
        'The constraints matrix should be a 2D Numpy array.'
    assert isinstance(s, np.ndarray), \
        'The margins vector should be a Numpy array.'
    assert len(s.shape) == 1, \
        'The margins vector should be a 1D Numpy array.'
    assert np.shape(A)[0] == len(s), \
        'The number of linear constraints should be equal to the number of margins.'
    assert np.shape(A)[1] == len(y), \
        'The number of coefficients for the linear constraints should be equal to the number of observations.'

    if q is None:
        q = np.ones(len(y))
    s_hat = np.matmul(A, y)
    lambda_k = np.zeros(A.shape[0])
    beta = np.copy(y)
    epsilon = 1.0
    iter_eps = 0
    while (epsilon > 1.0e-10) & (iter_eps < max_iter):
        Phi = np.matmul(A, y * (1.0 - np.exp(- q * np.matmul(np.transpose(A), lambda_k))))
        D = np.diag(y * q * np.exp(- q * np.matmul(np.transpose(A), lambda_k)))
        J = np.matmul(np.matmul(A, D), np.transpose(A))
        delta_lambda = cg(J, Phi - s_hat + s)[0]
        gamma = gamma0
        iter_gam = 0
        lambda_k = lambda_k - gamma * delta_lambda
        beta = y * np.exp(- q * np.matmul(np.transpose(A), lambda_k))
        if iter_eps > 0:
            while (np.mean(np.abs(s - np.matmul(A, beta))) > epsilon) & \
                  (iter_gam < max_iter):
                gamma = gamma / 2.0
                iter_gam = iter_gam + 1
                lambda_k = lambda_k - gamma * delta_lambda
                beta = y * np.exp(- q * np.matmul(np.transpose(A), lambda_k))
        epsilon = np.mean(np.abs(s - np.matmul(A, beta)))
        iter_eps = iter_eps + 1
    return (beta, lambda_k, iter_eps)

@profile
def solve_system_lu(A, B):
    """
    Solve system A X = B with X and B vectors instead of matrices
    Use LU factorization of A
    Input:
      A: N * N sparse square matrix
      B: N * M matrix
    Output:
      X: N * M matrix
    """
    assert np.shape(A)[0] == np.shape(A)[1], \
        'A should be a square matrix'
    assert np.shape(A)[1] == np.shape(B)[0], \
        'The numbers of columns in A should be equal to the number of rows in B'
    M = np.shape(B)[1]
    X = np.zeros_like(B)
    lu, piv = lu_factor(A)
    for i in range(0, M):
        X[:, i] = lu_solve((lu, piv), B[:, i])
    return X

@profile
def solve_system_spsolve(A, B):
    """
    Solve system A X = B with X and B vectors instead of matrices
    Use spares system linear solver
    Input:
      A: N * N sparse square matrix
      B: N * M matrix
    Output:
      X: N * M matrix
    """
    assert np.shape(A)[0] == np.shape(A)[1], \
        'A should be a square matrix'
    assert np.shape(A)[1] == np.shape(B)[0], \
        'The numbers of columns in A should be equal to the number of rows in B'
    A = csc_matrix(A)
    B = csc_matrix(B)
    X = spsolve(A, B)
    return X

@profile
def compute_gradient(
    beta_0: np.ndarray,
    lambda_0: np.ndarray,
    y: np.ndarray,
    A: np.ndarray,
    method: str,
    alpha: float = 1,
    l: np.ndarray = None,
    h: np.ndarray = None, 
    q: np.ndarray = None
) -> tuple[np.ndarray, np.ndarray]:
    """Compute the gradient dbeta/dy and dbeta/ds.

    The covariance matrix of the raked values is phi' Sigma phi'T
    where phi' is the matrix of the partial derivatives of the raked values beta
    with respect to the observations y and margins s. This function computes phi'

    Parameters
    ----------
    beta_0 : np.ndarray
        Vector of raked values
    lambda_0 : np.ndarray
        Corresponding dual
    y : np.ndarray
        Vector of observations
    A : np.ndarray
        Constraints matrix (output of a function from the compute_constraints module)
    method : string
        Raking method (one of chi2, entropic, general, logit)
    alpha : float
        Parameter of the distance function, alpha=1 is the chi2 distance, alpha=0 is the entropic distance
    l : np.ndarray
        Lower bounds for the observations
    h : np.ndarray
        Upper bounds for the observations
    q :  np.ndarray
        Vector of weights (default to all 1)

    Returns
    -------
    Dphi_y : np.ndarray
        Derivatives with respect to the observations
    Dphi_s: np.ndarray
        Derivatives with respect to the margins
    """
    assert isinstance(beta_0, np.ndarray), \
        'The vector of raked values should be a Numpy array.'
    assert len(beta_0.shape) == 1, \
        'The vector of raked values should be a 1D Numpy array.'
    assert isinstance(lambda_0, np.ndarray), \
        'The dual vector should be a Numpy array.'
    assert len(lambda_0.shape) == 1, \
        'The vdual vector should be a 1D Numpy array.'
    assert isinstance(y, np.ndarray), \
        'The vector of observations should be a Numpy array.'
    assert len(y.shape) == 1, \
        'The vector of observations should be a 1D Numpy array.'
    assert len(y) == len(beta_0), \
        'The vectors of observations and raked values should have the same length.'
    assert isinstance(A, np.ndarray), \
        'The constraint matrix should be a Numpy array.'
    assert len(A.shape) == 2, \
        'The constraints matrix should be a 2D Numpy array.'
    assert np.shape(A)[0] == len(lambda_0), \
        'The number of linear constraints should be equal to the length of the dual vector.'
    assert np.shape(A)[1] == len(y), \
        'The number of coefficients for the linear constraints should be equal to the number of observations.'
    assert method in ['chi2', 'entropic', 'general', 'logit'], \
        'The raking method must be "chi2", "entropic", "general", or "logit".'
    if method == 'general':
        assert isinstance(alpha, (int, float)), \
            'The parameter of the distance function should be an integer or a float.'
    if method == 'logit':
        if l is None:
            l = np.zeros(len(y))
        assert isinstance(l, np.ndarray), \
            'The vector of lower bounds should be a Numpy array.'
        assert len(l.shape) == 1, \
            'The vector of lower bounds should be a 1D Numpy array.'
        assert len(y) == len(l), \
            'Observations and lower bounds vectors should have the same length.'
        assert np.all(l >= 0.0), \
            'The lower bounds must be positive.'
        assert np.all(l <= y), \
            'The observations must be superior or equal to the corresponding lower bounds.'
        if h is None:
            h = np.ones(len(y))
        assert isinstance(h, np.ndarray), \
            'The vector of upper bounds should be a Numpy array.'
        assert len(h.shape) == 1, \
            'The vector of upper bounds should be a 1D Numpy array.'
        assert len(y) == len(h), \
            'Observations and upper bounds vectors should have the same length.'
        assert np.all(h > 0.0), \
            'The upper bounds must be strictly positive.'
        assert np.all(h >= y), \
            'The observations must be inferior or equal to the correspondings upper bounds.'   
        assert np.all(l < h), \
            'The lower bounds must be stricty inferior to the correspondings upper bounds.'
    if q is not None:
        assert isinstance(q, np.ndarray), \
            'The vector of weights should be a Numpy array.'
        assert len(q.shape) == 1, \
            'The vector of weights should be a 1D Numpy array.'
        assert len(y) == len(q), \
            'Observations and weights vectors should have the same length.'

    if q is None:
        q = np.ones(len(y))

    # Partial derivatives of the distance function with respect to raked values and observations
    if method == 'chi2':
        DF1_beta_diag = np.zeros(len(beta_0))
        DF1_beta_diag[y!=0] = 1.0 / (q[y!=0] * y[y!=0])
        DF1_beta_diag[y==0] = 0.0
        DF1_beta = np.diag(DF1_beta_diag)
        DF1_y_diag = np.zeros(len(y))
        DF1_y_diag[y!=0] = - beta_0[y!=0] / (q[y!=0] * np.square(y[y!=0]))
        DF1_y_diag[y==0] = 0.0
        DF1_y = np.diag(DF1_y_diag)
    elif method == 'entropic':
        DF1_beta_diag = np.zeros(len(beta_0))
        DF1_beta_diag[beta_0!=0] = 1.0 / (q[beta_0!=0] * beta_0[beta_0!=0])
        DF1_beta_diag[beta_0==0] = 0.0
        DF1_beta = np.diag(DF1_beta_diag)
        DF1_y_diag = np.zeros(len(y))
        DF1_y_diag[y!=0] = - 1.0 / (q[y!=0] * y[y!=0])
        DF1_y_diag[y==0] = 0.0
        DF1_y = np.diag(DF1_y_diag)
    elif method == 'general':
        DF1_beta_diag = np.zeros(len(beta_0))
        DF1_beta_diag[(y!=0)&(beta_0!=0)] = \
            np.power(beta_0[(y!=0)&(beta_0!=0)], alpha - 1.0) / \
            (q[(y!=0)&(beta_0!=0)] * np.power(y[(y!=0)&(beta_0!=0)], alpha))
        DF1_beta_diag[(y==0)|(beta_0==0)] = 0.0
        DF1_beta = np.diag(DF1_beta_diag)
        DF1_y_diag = np.zeros(len(y))
        DF1_y_diag[(y!=0)&(beta_0!=0)] = \
            - np.power(beta_0[(y!=0)&(beta_0!=0)], alpha) / \
            (q[(y!=0)&(beta_0!=0)] * np.power(y[(y!=0)&(beta_0!=0)], alpha + 1.0))
        DF1_y_diag[(y==0)|(beta_0==0)] = 0.0
        DF1_y = np.diag(DF1_y_diag)
    elif method == 'logit':
        DF1_beta_diag = np.zeros(len(beta_0))
        DF1_beta_diag[(beta_0!=l)&(beta_0!=h)] = \
            1.0 / (beta_0[(beta_0!=l)&(beta_0!=h)] - l[(beta_0!=l)&(beta_0!=h)]) + \
            1.0 / (h[(beta_0!=l)&(beta_0!=h)] - beta_0[(beta_0!=l)&(beta_0!=h)])
        DF1_beta_diag[(beta_0==l)|(beta_0==h)] = 0.0
        DF1_beta = np.diag(DF1_beta_diag)
        DF1_y_diag = np.zeros(len(y))
        DF1_y_diag[(y!=l)&(y!=h)] = \
            - 1.0 / (y[(y!=l)&(y!=h)] - l[(y!=l)&(y!=h)]) - \
            1.0 / (h[(y!=l)&(y!=h)] - y[(y!=l)&(y!=h)])
        DF1_y_diag[(y==l)|(y==h)] = 0.0
        DF1_y = np.diag(DF1_y_diag)

    # Gradient with respect to beta and lambda
    DF1_lambda = np.transpose(np.copy(A))
    DF2_beta = np.copy(A)
    DF2_lambda = np.zeros((np.shape(A)[0], np.shape(A)[0]))    
    DF_beta_lambda = np.concatenate(( \
        np.concatenate((DF1_beta, DF1_lambda), axis=1), \
        np.concatenate((DF2_beta, DF2_lambda), axis=1)), axis=0)

    # Gradient with respect to y and s
    DF1_s = np.zeros((np.shape(A)[1], np.shape(A)[0]))
    DF2_y = np.zeros((np.shape(A)[0], np.shape(A)[1]))
    DF2_s = - np.identity(np.shape(A)[0])    
    DF_y_s = np.concatenate(( \
        np.concatenate((DF1_y, DF1_s), axis=1), \
        np.concatenate((DF2_y, DF2_s), axis=1)), axis=0)

    # LU solver
    Dphi_lu = solve_system_lu(DF_beta_lambda, - DF_y_s)
    Dphi_y_lu = Dphi_lu[0:np.shape(A)[1], 0:np.shape(A)[1]]
    Dphi_s_lu = Dphi_lu[0:np.shape(A)[1], np.shape(A)[1]:(np.shape(A)[0] + np.shape(A)[1])]

    # Sparse solver
    Dphi_sp = solve_system_spsolve(DF_beta_lambda, - DF_y_s)
    Dphi_y_sp = Dphi_sp[0:np.shape(A)[1], 0:np.shape(A)[1]]
    Dphi_s_sp = Dphi_sp[0:np.shape(A)[1], np.shape(A)[1]:(np.shape(A)[0] + np.shape(A)[1])]

    return (Dphi_y_lu, Dphi_s_lu, Dphi_y_sp, Dphi_s_sp)

def main():
    I = 3
    J = 5
    K = 200
    sigma = 0.1
    (beta, s_cause) = create_test_matrix(I, J, K)
    y = add_noise(beta, sigma)
    (A, s) = constraints_USHD(s_cause, I, J, K)
    (beta_chi2, lambda_chi2) = raking_chi2(y, A, s)
    result_chi2 = compute_gradient(beta_chi2, lambda_chi2, y, A, 'chi2')
    (beta_entropic, lambda_entropic, num_iters) = raking_entropic(y, A, s)
    result_entropic = compute_gradient(beta_entropic, lambda_entropic, y, A, 'entropic')

if __name__ == '__main__':
    main()

