"""Module with methods to set up the problems with inequality constraints"""

import numpy as np

from raking.compute_constraints import constraints_1D, constraints_2D, constraints_3D

from raking.inequality.inequality_constraints import inequality_bounds
from raking.inequality.inequality_constraints import inequality_infant_mortality
from raking.inequality.inequality_constraints import inequality_time_trend


def set_bounds(
    y: np.ndarray,
    s: list,
    q: np.ndarray,
    l: np.ndarray,
    h: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Set up the optimization problem for the bounded problem.

    We need to define the problem:
        min_beta f(beta, y) s.t. A beta = s and C beta <= c

    Parameters
    ----------

    Returns
    -------
    y
    A
    s
    C
    c
    q
    """
    dim = len(s)
    if dim == 1:
        (A, s) = constraints_1D(s[0], len(y))
    elif dim == 2:
        (A, s) = constraints_2D(s[0], s[1], len(s[1]), len(s[0]))
    elif dim == 3:
        (A, s) = constraints_3D(s[0], s[1], s[2], s3.shape[0], s1.shape[0], s2.shape[1])
    (C, c) = inequality_bounds(l, h)
    return (y, A, s, C, c, q)


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


def set_time_trend(
    y: list,
    pop: list,
    s: list,
    q: list,
    l: list = None,
    h: list = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Set up the optimization problem for the time trend problem.

    We need to define the problem:
        min_beta f(beta, y) s.t. A beta = s and C beta <= c

    Parameters
    ----------
    y: list of np.ndarray
       Observations (length p) for the n years of the dataset

    Returns
    -------
    y
    A
    s
    C
    c
    """
    n = len(y)
    p = len(y[0])
    (A0, s0) = constraints_1D(s[0][0], p)
    A = np.zeros((n * A0.shape[0], n * A0.shape[1]))
    for i in range(0, n):
        A[(i * A0.shape[0]):((i + 1) * A0.shape[0]), (i * A0.shape[1]):((i + 1) * A0.shape[1])] = A0
    s = np.concatenate(s)
    (C, c) = inequality_time_trend(y, pop)
    y = np.concatenate(y)
    q = np.concatenate(q)
    if l is not None:
        l = np.concatenate(l)
    if h is not None:
        h = np.concatenate(h)
    else:
        h = None
    return (y, A, s, C, c, q, l, h)

