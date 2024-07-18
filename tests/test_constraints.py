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
        "For the constraints_1D function, the constraint A beta = s is not respected."

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
        "For the constraints_2D function, the constraint A beta = s is not respected."