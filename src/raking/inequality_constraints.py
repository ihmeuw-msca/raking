"""Module with methods to compute the inequality constraint matrix"""

import numpy as np


def inequality_infant_mortality(
    n1: np.ndarray,
    n2: np.ndarray,
    t1: float,
    t2: float
) -> tuple[np.ndarray, np.ndarray]:
    """Compute the constraints matrix C and the inequality vector c for the infant mortality problem.

    We need to define the inequality constraints C beta < c
    for the infant mortality problem of the Population, Fertility, and Mortality team:
    Probability of death for 0-to-1-month-olds must be lower than probability of death for 0-to-1-year-olds.

    Parameters
    ----------
    n1: np.ndarray
        Population number for 0-to-1-month-olds
    n2: np.ndarray
        Population number for 0-to-1-year-olds
    t1: float
        Time interval between 0 and 1 month
    t2: float
        Time interval between 0 and 1 year

    Returns
    -------
    C : np.ndarray
        I * 2I constraints matrix
    c : np.ndarray
        length I inequality vector vector
    """
    assert isinstance(
        n1, np.ndarray
    ), "The population number for 0-to-1-month-olds must be a Numpy array."
    assert (
        len(n1.shape) == 1
    ), "The population number for 0-to-1-month-olds must be a 1D Numpy array."
    assert isinstance(
        n2, np.ndarray
    ), "The population number for 0-to-1-month-olds must be a Numpy array."
    assert (
        len(n2.shape) == 1
    ), "The population number for 0-to-1-month-olds must be a 1D Numpy array."
    assert (
        len(n1) == len(n2)
    ), "The two population number arrays must have the same length."
    assert isinstance(
        t1, float
    ), "The time interval between 0 and 1 month must be a float."
    assert isinstance(
        t2, float
    ), "The time interval between 0 and 1 year must be a float."
    assert (
        t2 > t1
    ), "The time interval between 0 and 1 year must be larger than the time interval between 0 and a month."

    I = len(n1)
    C = np.concatenate((np.diag(t1 / n1), np.diag(- t2 / n2)), axis=1)
    c = np.zeros(I)
    return (C, c)

