import pytest
import numpy as np
from raking.compute_constraints import constraints_1D, constraints_2D

@pytest.fixture
def test_constraints_1D():
    # Generate balanced vector
    I = 3
    rng = np.random.default_rng(0)
    beta = rng.uniform(low=2.0, high=3.0, size=(I, 1))
    s = np.sum(beta)
    # Generate the constraints
    (A, s) = constraints_1D(s, I)
    # Verify that the constraint A beta = s is respected
    assert np.allclose(np.matmul(A, beta), s), \
        'For the constraints_1D function, the constraint A beta = s is not respected.'
    # Verify that the matrix A has rank 1
    assert np.linalg.matrix_rank(A) == 1, \
        'The constraint matrix should have rank 1.'

@pytest.fixture
def test_constraints_2D():
    # Generate balanced matrix
    I = 3
    J = 5
    rng = np.random.default_rng(0)
    beta = rng.uniform(low=2.0, high=3.0, size=(I, J))
    s1 = np.sum(beta, axis=0)
    s2 = np.sum(beta, axis=1)
    beta = beta.flatten(order='F')
    # Generate the constraints
    (A, s) = constraints_2D(s1, s2, I, J)
    # Verify that the constraint A beta = s is respected
    assert np.allclose(np.matmul(A, beta), s), \
        'For the constraints_2D function, the constraint A beta = s is not respected.'
    # Verify that the matrix A has rank I + J - 1
    assert np.linalg.matrix_rank(A) == I + J - 1, \
        'The constraint matrix should have rank {}.'.format(I + J - 1)

@pytest.fixture
def test_constraints_3D():
    # Generate balanced matrix
    I = 3
    J = 4
    K = 5
    rng = np.random.default_rng(0)
    beta = rng.uniform(low=2.0, high=3.0, size=(I, J, K))
    s1 = np.sum(beta, axis=0)
    s2 = np.sum(beta, axis=1)
    s3 = np.sum(beta, axis=2)
    beta = beta.flatten(order='F')
    # Generate the constraints
    (A, s) = constraints_3D(s1, s2, s3, I, J, K)
    # Verify that the constraint A beta = s is respected
    assert np.allclose(np.matmul(A, beta), s), \
        'For the constraints_2D function, the constraint A beta = s is not respected.'
    # Verify that the matrix A has rank I * J + I * K + J * K - I - J - K + 1
    assert np.linalg.matrix_rank(A) == I * J + I * K + J * K - I - J - K + 1, \
        'The constraint matrix should have rank {}.'.format(I * J + I * K + J * K - I - J - K + 1)

