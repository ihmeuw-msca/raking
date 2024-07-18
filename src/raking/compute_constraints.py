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
        1 * 1 margins vector
    """
    assert isinstance(s, float), \
        "The target sum of the observations must be a float."
    assert s >= 0.0, \
        "The target sum of the observations must be positive or null."
    assert isinstance(I, int), \
        "The number of possible values taken by the categorical variable must be an integer."
    assert I > 1, \
        "The number of possible values taken by the categorical variable must be higher than 1."

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
        (J + I - 1) * (I J) constraints matrix
    s : np.ndarray
        (J + I) * 1 margins vector
    """
    assert isinstance(s1, np.ndarray), \
        "The target sums over rows of the observation table must be a Numpy array."
    assert len(s1.shape) == 1, \
        "The target sums over rows of the observation table must be a 1D Numpy array."
    assert isinstance(s2, np.ndarray), \
        "The target sums over columns of the observation table must be a Numpy array."
    assert len(s2.shape) == 1, \
        "The target sums over rows of the observation table must be a 1D Numpy array."
    assert isinstance(I, int), \
        "The number of possible values taken by the first categorical variable must be an integer."
    assert I > 1, \
        "The number of possible values taken by the first categorical variable must be higher than 1."
    assert isinstance(J, int), \
        "The number of possible values taken by the second categorical variable must be an integer."
    assert J > 1, \
        "The number of possible values taken by the second categorical variable must be higher than 1."
    assert np.all(s1 >= 0.0), \
        "The target sums over rows of the observation table must be positive or null."
    assert np.all(s2 >= 0.0), \
        "The target sums over columns of the observation table must be positive or null."
    assert len(s1) == J, \
        "The target sums over rows must be equal to the number of columns in the observation table."
    assert len(s2) == I, \
        "The target sums over columns must be equal to the number of rows in the observation table."
    assert np.allclose(np.sum(s1), np.sum(s2)), \
        "The sum of the row margins must be equal to the sum of the column margins."
    
    A = np.zeros((J + I - 1, I * J))
    for j in range(0, J):
        for i in range(0, I - 1):
            A[J + i, j * I + i] = 1
            A[j, j * I + i] = 1
        A[j, j * I + I - 1] = 1
    s = np.concatenate([s1, s2[0:(I - 1)]]).reshape((1, J + I - 1))
    return (A, s)

