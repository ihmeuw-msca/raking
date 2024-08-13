"""Module with methods to compute the constraint matrix in 1D, 2D, 3D"""

import numpy as np

def constraints_1D(
    s: float,
    I: int
) -> tuple[np.ndarray, np.ndarray]:
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
    assert isinstance(s, float), \
        'The target sum of the observations must be a float.'
    assert s >= 0.0, \
        'The target sum of the observations must be positive or null.'
    assert isinstance(I, int), \
        'The number of possible values taken by the categorical variable must be an integer.'
    assert I > 1, \
        'The number of possible values taken by the categorical variable must be higher than 1.'

    A = np.ones((1, I))
    s = np.array([s])
    return (A, s)

def constraints_2D(
    s1: np.ndarray,
    s2: np.ndarray,
    I: int,
    J: int
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
    Returns
    -------
    A : np.ndarray
        (I + J - 1) * (I J) constraints matrix
    s : np.ndarray
        length (I + J) margins vector
    """
    assert isinstance(I, int), \
        'The number of possible values taken by the first categorical variable must be an integer.'
    assert I > 1, \
        'The number of possible values taken by the first categorical variable must be higher than 1.'
    assert isinstance(J, int), \
        'The number of possible values taken by the second categorical variable must be an integer.'
    assert J > 1, \
        'The number of possible values taken by the second categorical variable must be higher than 1.'
    assert isinstance(s1, np.ndarray), \
        'The target sums over rows of the observation table must be a Numpy array.'
    assert len(s1.shape) == 1, \
        'The target sums over rows of the observation table must be a 1D Numpy array.'
    assert isinstance(s2, np.ndarray), \
        'The target sums over columns of the observation table must be a Numpy array.'
    assert len(s2.shape) == 1, \
        'The target sums over rows of the observation table must be a 1D Numpy array.'
    assert np.all(s1 >= 0.0), \
        'The target sums over rows of the observation table must be positive or null.'
    assert np.all(s2 >= 0.0), \
        'The target sums over columns of the observation table must be positive or null.'
    assert len(s1) == J, \
        'The target sums over rows must be equal to the number of columns in the observation table.'
    assert len(s2) == I, \
        'The target sums over columns must be equal to the number of rows in the observation table.'
    assert np.allclose(np.sum(s1), np.sum(s2)), \
        'The sum of the row margins must be equal to the sum of the column margins.'
    
    A = np.zeros((J + I - 1, I * J))
    for j in range(0, J):
        for i in range(0, I - 1):
            A[J + i, j * I + i] = 1
            A[j, j * I + i] = 1
        A[j, j * I + I - 1] = 1
    s = np.concatenate([s1, s2[0:(I - 1)]])
    return (A, s)

def constraints_3D(
    s1: np.ndarray,
    s2: np.ndarray,
    s3: np.ndarray,
    I: int,
    J: int,
    K: int
) -> tuple[np.ndarray, np.ndarray]:
    """Compute the constraints matrix A and the margins vector s in 3D.
        
    This will define the raking optimization problem:
        min_beta f(beta,y) s.t. A beta = s
    The imput margins are 3 matrices s1, s2 and s3 and we have:
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
    Returns
    -------
    A : np.ndarray
        (I J + I K + J K - I - J - K + 1) * (I J K) constraints matrix
    s : np.ndarray
        length (I J + I K + J K - I - J - K + 1) margins vector
    """
    assert isinstance(I, int), \
        'The number of possible values taken by the first categorical variable must be an integer.'
    assert I > 1, \
        'The number of possible values taken by the first categorical variable must be higher than 1.'
    assert isinstance(J, int), \
        'The number of possible values taken by the second categorical variable must be an integer.'
    assert J > 1, \
        'The number of possible values taken by the second categorical variable must be higher than 1.'
    assert isinstance(K, int), \
        'The number of possible values taken by the third categorical variable must be an integer.'
    assert K > 1, \
        'The number of possible values taken by the third categorical variable must be higher than 1.'

    assert isinstance(s1, np.ndarray), \
        'The target sums over dimension 1 of the observation array must be a Numpy array.'
    assert len(s1.shape) == 2, \
        'The target sums over dimension 1 of the observation array must be a 2D Numpy array.'
    assert s1.shape[0] == J, \
        'The target sums over dimension 1 must have {} rows.'.format(J)
    assert s1.shape[0] == K, \
        'The target sums over dimension 1 must have {} columns.'.format(K)

    assert isinstance(s2, np.ndarray), \
        'The target sums over dimension 2 of the observation array must be a Numpy array.'
    assert len(s2.shape) == 2, \
        'The target sums over dimension 2 of the observation array must be a 2D Numpy array.'
    assert s2.shape[0] == I, \
        'The target sums over dimension 2 must have {} rows.'.format(I)
    assert s2.shape[0] == K, \
        'The target sums over dimension 2 must have {} columns.'.format(K)

    assert isinstance(s3, np.ndarray), \
        'The target sums over dimension 3 of the observation array must be a Numpy array.'
    assert len(s3.shape) == 2, \
        'The target sums over dimension 3 of the observation array must be a 2D Numpy array.'
    assert s3.shape[0] == I, \
        'The target sums over dimension 3 must have {} rows.'.format(I)
    assert s3.shape[0] == J, \
        'The target sums over dimension 3 must have {} columns.'.format(J)

    assert np.all(s1 >= 0.0), \
        'The target sums over dimension 1 of the observation array must be positive or null.'
    assert np.all(s2 >= 0.0), \
        'The target sums over dimension 2 of the observation array must be positive or null.'
    assert np.all(s3 >= 0.0), \
        'The target sums over dimension 3 of the observation array must be positive or null.'
   
    assert np.allclose(np.sum(s1, axis=0), np.sum(s2, axis=0)), \
        'The sums of the targets for dimension 1 and 2 must be equal.'
    assert np.allclose(np.sum(s2, axis=1), np.sum(s3, axis=1)), \
        'The sums of the targets for dimension 2 and 3 must be equal.'
    assert np.allclose(np.sum(s1, axis=0), np.sum(s3, axis=1)), \
        'The sums of the targets for dimension 1 and 3 must be equal.'
    
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
                A[(J - 1) * K + 1 + (K - 1) * I + (I - 1) * j + i, I * J * k + I * j + i] = 1
            s[(J - 1) * K + 1 + (K - 1) * I + (I - 1) * j + i] = s3[i, j]
    return (A, s)

