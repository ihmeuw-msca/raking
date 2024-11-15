"""Module with methods to set up the problems with enequality constraints"""

import numpy as np

from raking.compute_constraints import constraints_1D

from inequality_constraints import inequality_infant_mortality

def set_infant_mortality(
    n1: np.ndarray,
    n2: np.ndarray,
    t1: float,
    t2: float,
    y1: np.ndarray,
    y2: np.ndarray,
    s1: float,
    s2: float
) -> tuple[np.ndarray, np.ndarray]:
    """Set up the optimization problem for the infant mortality problem.

    We need to define the problem:
        min_beta f(beta, y) s.t. A beta = s and C beta <= c

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
    y
    A
    s
    C
    c
    """
    I = len(y1
    y = np.concatenate((y1, y2))
    (A1, s1) = constraints_1D(s1, I)
    (A2, s2) = constraints_1D(s2, I)
    A = np.concatenate(
        (
            np.concatenate((A1, np.zeros(I)), axis=1),
            np.concatenate((np.zeros(I), A2), axis=1),
        ),
        axis=0,
    )
    s = np.array([s1, s2])
    (C, c) = inequality_infant_mortality(n1, n2, t1, t2)
    return (y, A, s, C, c)

