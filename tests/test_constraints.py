import pytest
import numpy as np
from raking.compute_constraints import (
    constraints_1D,
    constraints_2D,
    constraints_3D,
    constraints_USHD,
)


def test_constraints_1D():
    # Generate balanced vector
    I = 3
    rng = np.random.default_rng(0)
    beta = rng.uniform(low=2.0, high=3.0, size=I)
    s = np.sum(beta)
    # Generate the constraints
    (A, s) = constraints_1D(s, I)
    # Verify that the constraint A beta = s is respected
    assert np.allclose(np.matmul(A, beta), s), (
        "For the constraints_1D function, the constraint A beta = s is not respected."
    )
    # Verify that the matrix A has rank 1
    assert np.linalg.matrix_rank(A) == 1, (
        "The constraint matrix should have rank 1."
    )


def test_constraints_2D():
    # Generate balanced matrix
    I = 3
    J = 5
    rng = np.random.default_rng(0)
    beta = rng.uniform(low=2.0, high=3.0, size=(I, J))
    s1 = np.sum(beta, axis=0)
    s2 = np.sum(beta, axis=1)
    beta = beta.flatten(order="F")
    # Generate the constraints
    (A, s) = constraints_2D(s1, s2, I, J)
    # Verify that the constraint A beta = s is respected
    assert np.allclose(np.matmul(A, beta), s), (
        "For the constraints_2D function, the constraint A beta = s is not respected."
    )
    # Verify that the matrix A has rank I + J - 1
    assert np.linalg.matrix_rank(A) == I + J - 1, (
        "The constraint matrix should have rank {}.".format(I + J - 1)
    )


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
    beta = beta.flatten(order="F")
    # Generate the constraints
    (A, s) = constraints_3D(s1, s2, s3, I, J, K)
    # Verify that the constraint A beta = s is respected
    assert np.allclose(np.matmul(A, beta), s), (
        "For the constraints_3D function, the constraint A beta = s is not respected."
    )
    # Verify that the matrix A has rank I * J + I * K + J * K - I - J - K + 1
    assert np.linalg.matrix_rank(A) == I * J + I * K + J * K - I - J - K + 1, (
        "The constraint matrix should have rank {}.".format(
            I * J + I * K + J * K - I - J - K + 1
        )
    )


def test_constraints_USHD():
    # Generate balanced array
    I = 3
    J = 4
    K = 5
    rng = np.random.default_rng(0)
    beta_ijk = rng.uniform(low=2.0, high=3.0, size=(I, J, K))
    beta_00k = np.sum(beta_ijk, axis=(0, 1))
    beta_i0k = np.sum(beta_ijk, axis=1)
    beta_0jk = np.sum(beta_ijk, axis=0)
    beta1 = np.concatenate(
        (beta_00k.reshape((1, 1, K)), beta_i0k.reshape(I, 1, K)), axis=0
    )
    beta2 = np.concatenate((beta_0jk.reshape((1, J, K)), beta_ijk), axis=0)
    beta = np.concatenate((beta1, beta2), axis=1)
    beta = beta.flatten("F")
    beta_i = np.sum(beta_ijk, axis=(1, 2))
    beta_0 = np.sum(beta_i)
    s_cause = np.array([beta_0] + beta_i.tolist())
    # Generate the constraints
    (A, s) = constraints_USHD(s_cause, I, J, K)
    # Verify that the constraint A beta = s is respected
    assert np.allclose(np.matmul(A, beta), s), (
        "For the constraints_USHD function, the constraint A beta = s is not respected."
    )
    # Verify that the matrix A has rank I + 2 * K + J * K + (I - 1) * K
    assert np.linalg.matrix_rank(A) == I + 2 * K + J * K + (I - 1) * K, (
        "The constraint matrix should have rank {}.".format(
            I + 2 * K + J * K + (I - 1) * K
        )
    )

def test_constraints_USHD_lower():
    # Generate balanced array
    I = 3
    J = 4
    K = 5
    rng = np.random.default_rng(0)
    beta_ijk = rng.uniform(low=2.0, high=3.0, size=(I, J, K))
    s_cause = np.sum(beta_ijk, axis=(1, 2))
    s_county = np.sum(beta_ijk, axis=(0, 1))
    s_all_causes = np.sum(beta_ijk, axis=0)
    beta_i0k = np.sum(beta_ijk, axis=1)
    beta = np.concatenate((beta_i0k.reshape((I, 1, K)), beta_ijk), axis=1)
    beta = beta.flatten("F")
        # Generate the constraints
    (A, s) = constraints_USHD_lower(s_cause, s_county, s_all_causes, I, J, K)
    # Verify that the constraint A beta = s is respected
    assert np.allclose(np.matmul(A, beta), s), (
        "For the constraints_USHD_lower function, the constraint A beta = s is not respected."
    )
    # Verify that the matrix A has rank I + K + I * K + J * K - K - 1
    assert np.linalg.matrix_rank(A) == I + K + I * K + J * K - K - 1, (
        "The constraint matrix should have rank {}.".format(
            I + K + I * K + J * K - K - 1
        )
    )

