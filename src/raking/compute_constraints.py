"""Module with methods to compute the constraint matrix in 1D, 2D, 3D"""

import numpy as np


def constraints_1D(s: float, I: int) -> tuple[np.ndarray, np.ndarray]:
    """Compute the constraints matrix A and the margins vector s in 1D.

    This will define the raking optimization problem:
        min_beta f(beta,y) s.t. A beta = s

    Parameters
    ----------
    s : float
        Target sum of the observations s = sum_i y_i
    I : int
        Number of possible values for categorical variable 1
    Returns
    -------
    A : np.ndarray
        1 * I constraints matrix
    s : np.ndarray
        length 1 margins vector
    """
    assert isinstance(s, float), (
        "The target sum of the observations must be a float."
    )
    assert s >= 0.0, (
        "The target sum of the observations must be positive or null."
    )
    assert isinstance(I, int), (
        "The number of possible values taken by the categorical variable must be an integer."
    )
    assert I > 1, (
        "The number of possible values taken by the categorical variable must be higher than 1."
    )

    A = np.ones((1, I))
    s = np.array([s])
    return (A, s)


def constraints_2D(
    s1: np.ndarray,
    s2: np.ndarray,
    I: int,
    J: int,
    rtol: float = 1e-05,
    atol: float = 1e-08,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute the constraints matrix A and the margins vector s in 2D.

    This will define the raking optimization problem:
        min_beta f(beta,y) s.t. A beta = s

    Parameters
    ----------
    s1 : np.ndarray
        Target sums over rows of the observations table
    s2 : np.ndarray
        Target sums over columns of the observations table
    I : int
        Number of possible values for categorical variable 1
    J : int
        Number of possible values for categorical variable 2
    rtol : float
        Relative tolerance to check whether the margins are consistant. See numpy.allclose documentation for details.
    atol : float
        Absolute tolerance to check whether the margins are consistant. See numpy.allclose documentation for details.

    Returns
    -------
    A : np.ndarray
        (I + J - 1) * (I J) constraints matrix
    s : np.ndarray
        length (I + J) margins vector
    """
    assert isinstance(I, int), (
        "The number of possible values taken by the first categorical variable must be an integer."
    )
    assert I > 1, (
        "The number of possible values taken by the first categorical variable must be higher than 1."
    )
    assert isinstance(J, int), (
        "The number of possible values taken by the second categorical variable must be an integer."
    )
    assert J > 1, (
        "The number of possible values taken by the second categorical variable must be higher than 1."
    )
    assert isinstance(s1, np.ndarray), (
        "The target sums over rows of the observation table must be a Numpy array."
    )
    assert len(s1.shape) == 1, (
        "The target sums over rows of the observation table must be a 1D Numpy array."
    )
    assert isinstance(s2, np.ndarray), (
        "The target sums over columns of the observation table must be a Numpy array."
    )
    assert len(s2.shape) == 1, (
        "The target sums over rows of the observation table must be a 1D Numpy array."
    )
    assert np.all(s1 >= 0.0), (
        "The target sums over rows of the observation table must be positive or null."
    )
    assert np.all(s2 >= 0.0), (
        "The target sums over columns of the observation table must be positive or null."
    )
    assert len(s1) == J, (
        "The target sums over rows must be equal to the number of columns in the observation table."
    )
    assert len(s2) == I, (
        "The target sums over columns must be equal to the number of rows in the observation table."
    )
    assert np.allclose(np.sum(s1), np.sum(s2), rtol, atol), (
        "The sum of the row margins must be equal to the sum of the column margins."
    )

    A = np.zeros((J + I - 1, I * J))
    for j in range(0, J):
        for i in range(0, I - 1):
            A[J + i, j * I + i] = 1
            A[j, j * I + i] = 1
        A[j, j * I + I - 1] = 1
    s = np.concatenate([s1, s2[0 : (I - 1)]])
    return (A, s)


def constraints_3D(
    s1: np.ndarray,
    s2: np.ndarray,
    s3: np.ndarray,
    I: int,
    J: int,
    K: int,
    rtol: float = 1e-05,
    atol: float = 1e-08,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute the constraints matrix A and the margins vector s in 3D.

    This will define the raking optimization problem:
        min_beta f(beta,y) s.t. A beta = s
    The input margins are 3 matrices s1, s2 and s3 and we have:
        sum_i beta_ijk = s1_jk for all j,k
        sum_j beta_ijk = s2_ik for all i,k
        sum_k beta_ijk = s3_ij for all i,j

    Parameters
    ----------
    s1 : np.ndarray
        Target sums over dimension 1 of the observations array
    s2 : np.ndarray
        Target sums over dimension 2 of the observations array
    s3 : np.ndarray
        Target sums over dimension 3 of the observations array
    I : int
        Number of possible values for categorical variable 1
    J : int
        Number of possible values for categorical variable 2
    K : int
        Number of possible values for categorical variable 3
    rtol : float
        Relative tolerance to check whether the margins are consistant. See numpy.allclose documentation for details.
    atol : float
        Absolute tolerance to check whether the margins are consistant. See numpy.allclose documentation for details.

    Returns
    -------
    A : np.ndarray
        (I J + I K + J K - I - J - K + 1) * (I J K) constraints matrix
    s : np.ndarray
        length (I J + I K + J K - I - J - K + 1) margins vector
    """
    assert isinstance(I, int), (
        "The number of possible values taken by the first categorical variable must be an integer."
    )
    assert I > 1, (
        "The number of possible values taken by the first categorical variable must be higher than 1."
    )
    assert isinstance(J, int), (
        "The number of possible values taken by the second categorical variable must be an integer."
    )
    assert J > 1, (
        "The number of possible values taken by the second categorical variable must be higher than 1."
    )
    assert isinstance(K, int), (
        "The number of possible values taken by the third categorical variable must be an integer."
    )
    assert K > 1, (
        "The number of possible values taken by the third categorical variable must be higher than 1."
    )

    assert isinstance(s1, np.ndarray), (
        "The target sums over dimension 1 of the observation array must be a Numpy array."
    )
    assert len(s1.shape) == 2, (
        "The target sums over dimension 1 of the observation array must be a 2D Numpy array."
    )
    assert s1.shape[0] == J, (
        "The target sums over dimension 1 must have {} rows.".format(J)
    )
    assert s1.shape[1] == K, (
        "The target sums over dimension 1 must have {} columns.".format(K)
    )

    assert isinstance(s2, np.ndarray), (
        "The target sums over dimension 2 of the observation array must be a Numpy array."
    )
    assert len(s2.shape) == 2, (
        "The target sums over dimension 2 of the observation array must be a 2D Numpy array."
    )
    assert s2.shape[0] == I, (
        "The target sums over dimension 2 must have {} rows.".format(I)
    )
    assert s2.shape[1] == K, (
        "The target sums over dimension 2 must have {} columns.".format(K)
    )

    assert isinstance(s3, np.ndarray), (
        "The target sums over dimension 3 of the observation array must be a Numpy array."
    )
    assert len(s3.shape) == 2, (
        "The target sums over dimension 3 of the observation array must be a 2D Numpy array."
    )
    assert s3.shape[0] == I, (
        "The target sums over dimension 3 must have {} rows.".format(I)
    )
    assert s3.shape[1] == J, (
        "The target sums over dimension 3 must have {} columns.".format(J)
    )

    assert np.all(s1 >= 0.0), (
        "The target sums over dimension 1 of the observation array must be positive or null."
    )
    assert np.all(s2 >= 0.0), (
        "The target sums over dimension 2 of the observation array must be positive or null."
    )
    assert np.all(s3 >= 0.0), (
        "The target sums over dimension 3 of the observation array must be positive or null."
    )

    assert np.allclose(np.sum(s1, axis=0), np.sum(s2, axis=0), rtol, atol), (
        "The sums of the targets for dimension 1 and 2 must be equal."
    )
    assert np.allclose(np.sum(s2, axis=1), np.sum(s3, axis=1), rtol, atol), (
        "The sums of the targets for dimension 2 and 3 must be equal."
    )
    assert np.allclose(np.sum(s1, axis=1), np.sum(s3, axis=0), rtol, atol), (
        "The sums of the targets for dimension 1 and 3 must be equal."
    )

    A = np.zeros((I * J + I * K + J * K - I - J - K + 1, I * J * K))
    s = np.zeros(I * J + I * K + J * K - I - J - K + 1)
    for k in range(0, K):
        for j in range(0, J - 1):
            for i in range(0, I):
                A[(J - 1) * k + j, I * J * k + I * j + i] = 1
            s[(J - 1) * k + j] = s1[j, k]
    for i in range(0, I):
        A[(J - 1) * K, I * J * (K - 1) + I * (J - 1) + i] = 1
    s[(J - 1) * K] = s1[J - 1, K - 1]
    for i in range(0, I):
        for k in range(0, K - 1):
            for j in range(0, J):
                A[(J - 1) * K + 1 + (K - 1) * i + k, I * J * k + I * j + i] = 1
            s[(J - 1) * K + 1 + (K - 1) * i + k] = s2[i, k]
    for j in range(0, J):
        for i in range(0, I - 1):
            for k in range(0, K):
                A[
                    (J - 1) * K + 1 + (K - 1) * I + (I - 1) * j + i,
                    I * J * k + I * j + i,
                ] = 1
            s[(J - 1) * K + 1 + (K - 1) * I + (I - 1) * j + i] = s3[i, j]
    return (A, s)


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
    assert isinstance(I, int), (
        "The number of causes of deaths must be an integer."
    )
    assert I > 1, "The number of causes of deaths must be higher than 1."
    assert isinstance(J, int), (
        "The number of races and ethnicities must be an integer."
    )
    assert J > 1, "The number of races and ethnicities must be higher than 1."
    assert isinstance(K, int), "The number of counties must be an integer."
    assert K > 1, "The number of counties must be higher than 1."

    assert isinstance(s_cause, np.ndarray), (
        "The margins vector for the causes of death must be a Numpy array."
    )
    assert len(s_cause.shape) == 1, (
        "The margins vector for the causes of death must be a 1D Numpy array."
    )
    assert np.all(s_cause >= 0.0), (
        "The number of deaths for each cause must be positive or null."
    )
    assert len(s_cause) == I + 1, (
        "The length of the margins vector for the causes of death must be equal to 1 + number of causes."
    )

    assert np.allclose(s_cause[0], np.sum(s_cause[1:]), rtol, atol), (
        "The all-causes number of deaths must be equal to the sum of the numbers of deaths per cause."
    )

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

def constraints_USHD_lower(
    s_cause: np.ndarray,
    s_county: np.ndarray,
    s_all_causes: np.ndarray,
    I: int,
    J: int,
    K: int,
    rtol: float = 1e-05,
    atol: float = 1e-08,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute the constraints matrix A and the margins vector s for the USHD use case (lower levels).

    This will define the raking optimization problem:
        min_beta f(beta,y) s.t. A beta = s
    The first input margins are the I values:
        - beta_i00 = Number of deaths for cause i (all races, at the state level)
    The second input margins are the K values:
        - beta_00k = Total number of deaths (all causes, all races) for each county
    The third input margins are the J * K values:
        - beta_0jk = Number of deaths (all causes) for race j and county k

    Parameters
    ----------
    s_cause : np.ndarray
        Total number of deaths for each cause (all races, all counties)
    s_county: np.ndarray
        Number of deaths for each county (all causes, all races)
    s_all_causes: np.ndarray
        Number of deaths for each race and each county (all causes) 
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
        ( I + K + I * K + J * K - K - 1 ) * (I * (J + 1) * K) constraints matrix
    s : np.ndarray
        length ( I + K + I * K + J * K - K - 1 ) margins vector
    """
    assert isinstance(I, int), (
        "The number of causes of deaths must be an integer."
    )
    assert I > 1, "The number of causes of deaths must be higher than 1."
    assert isinstance(J, int), (
        "The number of races and ethnicities must be an integer."
    )
    assert J > 1, "The number of races and ethnicities must be higher than 1."
    assert isinstance(K, int), "The number of counties must be an integer."
    assert K > 1, "The number of counties must be higher than 1."

    assert isinstance(s_cause, np.ndarray), (
        "The margins vector for the causes of death must be a Numpy array."
    )
    assert len(s_cause.shape) == 1, (
        "The margins vector for the causes of death must be a 1D Numpy array."
    )
    assert np.all(s_cause >= 0.0), (
        "The number of deaths for each cause must be positive or null."
    )
    assert len(s_cause) == I, (
        "The length of the margins vector for the causes of death must be equal to the number of causes."
    )

    assert isinstance(s_county, np.ndarray), (
        "The margins vector for the counties must be a Numpy array."
    )
    assert len(s_county.shape) == 1, (
        "The margins vector for the counties must be a 1D Numpy array."
    )
    assert np.all(s_county >= 0.0), (
        "The number of deaths for each county must be positive or null."
    )
    assert len(s_county) == K, (
        "The length of the margins vector for the counties must be equal to the number of counties."
    )

    assert isinstance(s_all_causes, np.ndarray), (
        "The margins vector for the all causes deaths must be a Numpy array."
    )
    assert len(s_all_causes.shape) == 2, (
        "The margins vector for the all causes deaths must be a 2D Numpy array."
    )
    assert np.all(s_all_causes >= 0.0), (
        "The number of all causes deaths must be positive or null."
    )
    assert (s_all_causes.shape[0] == J) and (s_all_causes.shape[1] == K), (
        "The shape of the margins vector for the all causes deaths must be equal to the number of races multiplied by the number of counties."
    )

    assert np.allclose(np.sum(s_cause), np.sum(s_county), rtol, atol), (
        "The sum of the number of deaths per cause must be equal to the sum of the number of deaths per county."
    )
    assert np.allclose(np.sum(s_all_causes, axis=0), s_county, rtol, atol), (
        "For each county, the all-races number of deaths must be equal to the sum of the number of deaths per race."
    )

    A = np.zeros((I + K + I * K + J * K - K - 1, I * (J + 1) * K))
    s = np.zeros(I + K + I * K + J * K - K - 1)
    # Constraint sum_k=0,...,K-1 beta_i,0,k = s_cause_i for i=0,...,I-1
    for i in range(0, I):
        for k in range(0, K):
            A[i, k * I * (J + 1) + i] = 1
        s[i] = s_cause[i]
    # Constraint sum_i=0,...,I-1 beta_i,0,k = s_county_k = 0 for k=0,...,K-2
    for k in range(0, K - 1):
        for i in range(0, I):
            A[I + k, k * I * (J + 1) + i] = 1
        s[I + k] = s_county[k]
    # Constraint sum_i=0,...,I-1 beta_i,j,k = s_all_causes for j=1,...,J and k=0,...,K-1
    for k in range(0, K):
        for j in range(1, J + 1):
            for i in range(0, I):
                A[I + K - 1 + k * J + j - 1, k * I * (J + 1) + j * I + i] = 1
            s[I + K - 1 + k * J + j - 1] = s_all_causes[j - 1, k]
    # Constraint sum_j=1,...,J beta_i,j,k - beta_i,0,k = 0 for i=0,...,I-2 and k=0,...,K-1
    for k in range(0, K):
        for i in range(0, I - 1):
            for j in range(1, J + 1):
                A[I + K - 1 + J * K + k * (I - 1) + i, k * I * (J + 1) + j * I + i] = 1
            A[I + K - 1 + J * K + k * (I - 1) + i, k * I * (J + 1) + i] = -1
    return (A, s)
