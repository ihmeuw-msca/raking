"""Module with methods to compute the inequality constraint matrix"""

import numpy as np


def inequality_bounds(
    l: np.ndarray,
    h: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute the constraints matrix C and the inequality vector c for the bounded problem.

    We need to define the inequality constraints C beta < c
    for the bounded problem: l < beta < h.

    Parameters
    ----------
    l: np.ndarray
       Lower bounds for the raked values
    h: np.ndarray
       Upper bounds for the raked values

    Returns
    -------
    C : np.ndarray
        2N * N inequality constraints matrix
    c : np.ndarray
        length 2N inequality constraints vector
    """
    assert isinstance(
        l, np.ndarray
    ), "The lower bounds for the raked values must be a Numpy array."
    assert (
        len(l.shape) == 1
    ), "The lower bounds for the raked values must be a 1D Numpy array."
    assert isinstance(
        h, np.ndarray
    ), "The upper bounds for the raked values must be a Numpy array."
    assert (
        len(h.shape) == 1
    ), "The upper bounds for the raked values must be a 1D Numpy array."
    assert (
        len(l) == len(h)
    ), "The lower bounds and upper bounds must have the same length."

    N = len(l)
    C = np.concatenate((- np.identity(N), np.identity(N)), axis=0)
    c = np.concatenate((-l, h))
    return (C, c)


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
        I * 2I inequality constraints matrix
    c : np.ndarray
        length I inequality constraints vector
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


def inequality_time_trend(
    y: list,
    pop: list
) -> tuple[np.ndarray, np.ndarray]:
    """Compute the constraints matrix C and the inequality vector c for the time trend problem.

    We need to define the inequality constraints C beta < c
    when conserving the time trend throught the raking:
    For years i and i+1 we have (beta_i - beta_i+1) (y_i - y_i+1) >= 0.

    Parameters
    ----------
    y: list of np.ndarray
       Observations (length p) for the n years of the dataset
    pop: list of np.ndarray
       Populations (length p) for the n years of the dataset

    Returns
    -------
    C : np.ndarray
        (n-1)p * np inequality constraints matrix
    c : np.ndarray
        length np inequality constraints vector
    """
    assert isinstance(y, list), \
        'The observations for all the years must be entered as a list.'
    assert len(y) >= 2, \
        'There must be at least 2 years of observations.'
    n = len(y)
    for i in range(0, n):
        assert isinstance(y[i], np.ndarray), \
            'The observations for year ' + str(i + 1) + ' must be a Numpy array.'
        assert len(y[i].shape) == 1, \
            'The observations for year ' + str(i + 1) + ' must be a 1D Numpy array.'
    p = len(y[0])
    for i in range(1, n):
        assert len(y[i]) == p, \
            'The observations for year ' + \
            str(i + 1) + \
            ' must have the same length as the observations for year 1.'
    C = np.zeros((p * (n - 1), p * n))
    for i in range(0, n - 1):
        C[(i * p):((i + 1) * p), (i * p):((i + 1) * p)] = \
            np.diag((y[i + 1] / pop[i + 1] - y[i] / pop[i]) / pop[i])
        C[(i * p):((i + 1) * p), ((i + 1) * p):((i + 2) * p)] = \
            np.diag((y[i] / pop[i] - y[i + 1] / pop[i +1]) / pop[i + 1])
    c = np.zeros(p * (n - 1))
    return (C, c)

