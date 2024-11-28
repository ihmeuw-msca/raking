"""Module with methods to set up the problems with inequality constraints"""

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
    s2: float,
    q1: np.ndarray,
    q2: np.ndarray,
    l1: np.ndarray = None,
    l2: np.ndarray = None,
    h1: np.ndarray = None,
    h2: np.ndarray = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
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
    I = len(y1)
    y = np.concatenate((y1, y2))
    (A1, s1) = constraints_1D(s1, I)
    (A2, s2) = constraints_1D(s2, I)
    A = np.concatenate(
        (
            np.concatenate((A1, np.zeros((1, I))), axis=1),
            np.concatenate((np.zeros((1, I)), A2), axis=1),
        ),
        axis=0,
    )
    s = np.concatenate((s1, s2))
    (C, c) = inequality_infant_mortality(n1, n2, t1, t2)
    q = np.concatenate((q1, q2))
    if (l1 is not None) and (l2 is not None):
        l = np.concatenate((l1, l2))
    else:
        l = None
    if (h1 is not None) and (h2 is not None):
        h = np.concatenate((h1, h2))
    else:
        h = None
    return (y, A, s, C, c, q, l, h)

