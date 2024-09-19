import pytest
import numpy as np
import pandas as pd
from raking.compute_constraints import constraints_1D, constraints_2D, constraints_3D
from raking.formatting_methods import format_data_1D, format_data_2D, format_data_3D

def test_format_data_1D():
    # Generate balanced vector
    I = 3
    rng = np.random.default_rng(0)
    y = rng.uniform(low=2.0, high=3.0, size=I)
    s = np.sum(y)
    # Generate the data frames
    var1 = np.arange(0, I)
    df_obs = pd.DataFrame({'value': y, 'var1': var1})
    df_margins = pd.DataFrame({'value_agg_over_var1': [s]})
    # Get the formatted data
    (y, s, I, q, l, h) = format_data_1D(df_obs, df_margins, 'var1')
    # Generate the constraints
    (A, s) = constraints_1D(s, I)
    # Verify that the constraint A beta = s is respected
    assert np.allclose(np.matmul(A, y), s), \
        'For the constraints_1D function, the constraint A beta = s is not respected.'

def test_format_data_2D():
    # Generate balanced matrix
    I = 3
    J = 5
    rng = np.random.default_rng(0)
    y = rng.uniform(low=2.0, high=3.0, size=(I, J))
    s1 = np.sum(y, axis=0)
    s2 = np.sum(y, axis=1)
    y = y.flatten(order='F')
    # Generate the data frames
    var1 = np.tile(np.arange(0, I), J)
    var2 = np.repeat(np.arange(0, J), I)
    df_obs = pd.DataFrame({'value': y, 'var1': var1, 'var2': var2})
    # Generate the constraints
    (A, s) = constraints_2D(s1, s2, I, J)
    # Verify that the constraint A beta = s is respected
    assert np.allclose(np.matmul(A, beta), s), \
        'For the constraints_2D function, the constraint A beta = s is not respected.'
    # Verify that the matrix A has rank I + J - 1
    assert np.linalg.matrix_rank(A) == I + J - 1, \
        'The constraint matrix should have rank {}.'.format(I + J - 1)
