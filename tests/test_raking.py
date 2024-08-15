import pytest
import numpy as np
from raking.compute_constraints import constraints_1D, constraints_2D, constraints_3D
from raking.raking_methods import raking_chi2, raking_entropic

@pytest.fixture
def test_chi2_raking_1D():
    # Generate balanced vector
    I = 3
    rng = np.random.default_rng(0)
    beta = rng.uniform(low=2.0, high=3.0, size=I)
    s = np.sum(beta)
    # Add noise
    y = beta + rng.normal(0.0, 0.1, size=len(beta))
    # Generate the constraints
    (A, s) = constraints_1D(s, I)
    # Rake using chi2 distance
    (beta_star, lambda_star) = raking_chi2(y, A, s)
    # Verify that the constraint A beta_star = s is respected
    assert np.allclose(np.matmul(A, beta_star), s), \
        'For the raking in 1D with the chi2 distance, the constraint A beta_star = s is not respected.'

@pytest.fixture
def test_chi2_raking_2D():
    # Generate balanced matrix
    I = 3
    J = 5
    rng = np.random.default_rng(0)
    beta = rng.uniform(low=2.0, high=3.0, size=(I, J))
    s1 = np.sum(beta, axis=0)
    s2 = np.sum(beta, axis=1)
    # Add noise
    y = beta + rng.normal(0.0, 0.1, size=beta.shape)
    y = y.flatten(order='F')
    # Generate the constraints
    (A, s) = constraints_2D(s1, s2, I, J)
    # Rake using chi2 distance
    (beta_star, lambda_star) = raking_chi2(y, A, s)
   # Verify that the constraint A beta_star = s is respected
    assert np.allclose(np.matmul(A, beta_star), s), \
        'For the raking in 2D with the chi2 distance, the constraint A beta_star = s is not respected.'

@pytest.fixture
def test_chi2_raking_3D():
    # Generate balanced matrix
    I = 3
    J = 4
    K = 5
    rng = np.random.default_rng(0)
    beta = rng.uniform(low=2.0, high=3.0, size=(I, J, K))
    s1 = np.sum(beta, axis=0)
    s2 = np.sum(beta, axis=1)
    s3 = np.sum(beta, axis=2)
    # Add noise
    y = beta + rng.normal(0.0, 0.1, size=beta.shape)
    y = y.flatten(order='F')
    # Generate the constraints
    (A, s) = constraints_3D(s1, s2, s3, I, J, K)
    # Rake using chi2 distance
    (beta_star, lambda_star) = raking_chi2(y, A, s)
   # Verify that the constraint A beta_star = s is respected
    assert np.allclose(np.matmul(A, beta_star), s), \
        'For the raking in 3D with the chi2 distance, the constraint A beta_star = s is not respected.'

@pytest.fixture
def test_entropic_raking_1D():
    # Generate balanced vector
    I = 3
    rng = np.random.default_rng(0)
    beta = rng.uniform(low=2.0, high=3.0, size=I)
    s = np.sum(beta)
    # Add noise
    y = beta + rng.normal(0.0, 0.1, size=len(beta))
    # Generate the constraints
    (A, s) = constraints_1D(s, I)
    # Rake using entropic distance
    (beta_star, lambda_star, iter_eps) = raking_entropic(y, A, s)
    # Verify that the constraint A beta_star = s is respected
    assert np.allclose(np.matmul(A, beta_star), s), \
        'For the raking in 1D with the entropic distance, the constraint A beta_star = s is not respected.'

@pytest.fixture
def test_entropic_raking_2D():
    # Generate balanced matrix
    I = 3
    J = 5
    rng = np.random.default_rng(0)
    beta = rng.uniform(low=2.0, high=3.0, size=(I, J))
    s1 = np.sum(beta, axis=0)
    s2 = np.sum(beta, axis=1)
    # Add noise
    y = beta + rng.normal(0.0, 0.1, size=beta.shape)
    y = y.flatten(order='F')
    # Generate the constraints
    (A, s) = constraints_2D(s1, s2, I, J)
    # Rake using entropic distance
    (beta_star, lambda_star, iter_eps) = raking_entropic(y, A, s)
   # Verify that the constraint A beta_star = s is respected
    assert np.allclose(np.matmul(A, beta_star), s), \
        'For the raking in 2D with the entropic distance, the constraint A beta_star = s is not respected.'

@pytest.fixture
def test_entropic_raking_3D():
    # Generate balanced matrix
    I = 3
    J = 4
    K = 5
    rng = np.random.default_rng(0)
    beta = rng.uniform(low=2.0, high=3.0, size=(I, J, K))
    s1 = np.sum(beta, axis=0)
    s2 = np.sum(beta, axis=1)
    s3 = np.sum(beta, axis=2)
    # Add noise
    y = beta + rng.normal(0.0, 0.1, size=beta.shape)
    y = y.flatten(order='F')
    # Generate the constraints
    (A, s) = constraints_3D(s1, s2, s3, I, J, K)
    # Rake using entropic distance
    (beta_star, lambda_star, iter_eps) = raking_entropic(y, A, s)
   # Verify that the constraint A beta_star = s is respected
    assert np.allclose(np.matmul(A, beta_star), s), \
        'For the raking in 3D with the entropic distance, the constraint A beta_star = s is not respected.'

