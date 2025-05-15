import pytest
import numpy as np
from raking.compute_constraints import (
    constraints_1D,
    constraints_2D,
    constraints_3D,
    constraints_USHD,
    constraints_USHD_lower,
)
from raking.raking_methods import (
    raking_chi2,
    raking_entropic,
    raking_general,
    raking_logit,
)


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
    assert np.allclose(np.matmul(A, beta_star), s), (
        "For the raking in 1D with the chi2 distance, the constraint A beta_star = s is not respected."
    )


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
    y = y.flatten(order="F")
    # Generate the constraints
    (A, s) = constraints_2D(s1, s2, I, J)
    # Rake using chi2 distance
    (beta_star, lambda_star) = raking_chi2(y, A, s)
    # Verify that the constraint A beta_star = s is respected
    assert np.allclose(np.matmul(A, beta_star), s), (
        "For the raking in 2D with the chi2 distance, the constraint A beta_star = s is not respected."
    )


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
    y = y.flatten(order="F")
    # Generate the constraints
    (A, s) = constraints_3D(s1, s2, s3, I, J, K)
    # Rake using chi2 distance
    (beta_star, lambda_star) = raking_chi2(y, A, s)
    # Verify that the constraint A beta_star = s is respected
    assert np.allclose(np.matmul(A, beta_star), s), (
        "For the raking in 3D with the chi2 distance, the constraint A beta_star = s is not respected."
    )


def test_chi2_raking_USHD():
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
    beta_i = np.sum(beta_ijk, axis=(1, 2))
    beta_0 = np.sum(beta_i)
    s_cause = np.array([beta_0] + beta_i.tolist())
    # Add noise
    y = beta + rng.normal(0.0, 0.1, size=beta.shape)
    y = y.flatten(order="F")
    # Generate the constraints
    (A, s) = constraints_USHD(s_cause, I, J, K)
    # Rake using chi2 distance
    (beta_star, lambda_star) = raking_chi2(y, A, s)
    # Verify that the constraint A beta_star = s is respected
    assert np.allclose(np.matmul(A, beta_star), s, atol=1.0e-5), (
        "For the USHD raking with the chi2 distance, the constraint A beta_star = s is not respected."
    )


def test_chi2_raking_USHD_lower():
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
    # Add noise
    y = beta + rng.normal(0.0, 0.1, size=beta.shape)
    y = y.flatten(order="F")
    # Generate the constraints
    (A, s) = constraints_USHD_lower(s_cause, s_county, s_all_causes, I, J, K)
    # Rake using chi2 distance
    (beta_star, lambda_star) = raking_chi2(y, A, s)
    # Verify that the constraint A beta_star = s is respected
    assert np.allclose(np.matmul(A, beta_star), s, atol=1.0e-5), (
        "For the USHD_lower raking with the chi2 distance, the constraint A beta_star = s is not respected."
    )


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
    assert np.allclose(np.matmul(A, beta_star), s), (
        "For the raking in 1D with the entropic distance, the constraint A beta_star = s is not respected."
    )


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
    y = y.flatten(order="F")
    # Generate the constraints
    (A, s) = constraints_2D(s1, s2, I, J)
    # Rake using entropic distance
    (beta_star, lambda_star, iter_eps) = raking_entropic(y, A, s)
    # Verify that the constraint A beta_star = s is respected
    assert np.allclose(np.matmul(A, beta_star), s), (
        "For the raking in 2D with the entropic distance, the constraint A beta_star = s is not respected."
    )


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
    y = y.flatten(order="F")
    # Generate the constraints
    (A, s) = constraints_3D(s1, s2, s3, I, J, K)
    # Rake using entropic distance
    (beta_star, lambda_star, iter_eps) = raking_entropic(y, A, s)
    # Verify that the constraint A beta_star = s is respected
    assert np.allclose(np.matmul(A, beta_star), s), (
        "For the raking in 3D with the entropic distance, the constraint A beta_star = s is not respected."
    )


def test_entropic_raking_USHD():
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
    beta_i = np.sum(beta_ijk, axis=(1, 2))
    beta_0 = np.sum(beta_i)
    s_cause = np.array([beta_0] + beta_i.tolist())
    # Add noise
    y = beta + rng.normal(0.0, 0.1, size=beta.shape)
    y = y.flatten(order="F")
    # Generate the constraints
    (A, s) = constraints_USHD(s_cause, I, J, K)
    # Rake using entropic distance
    (beta_star, lambda_star, iter_eps) = raking_entropic(y, A, s)
    # Verify that the constraint A beta_star = s is respected
    assert np.allclose(np.matmul(A, beta_star), s, atol=1.0e-6), (
        "For the USHD raking with the entropic distance, the constraint A beta_star = s is not respected."
    )


def test_entropic_raking_USHD_lower():
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
    # Add noise
    y = beta + rng.normal(0.0, 0.1, size=beta.shape)
    y = y.flatten(order="F")
    # Generate the constraints
    (A, s) = constraints_USHD_lower(s_cause, s_county, s_all_causes, I, J, K)
    # Rake using entropic distance
    (beta_star, lambda_star, iter_eps) = raking_entropic(y, A, s)
    # Verify that the constraint A beta_star = s is respected
    assert np.allclose(np.matmul(A, beta_star), s, atol=1.0e-5), (
        "For the USHD_lower raking with the entropic distance, the constraint A beta_star = s is not respected."
    )


def test_general_raking_1D():
    # Generate balanced vector
    I = 3
    rng = np.random.default_rng(0)
    beta = rng.uniform(low=2.0, high=3.0, size=I)
    s = np.sum(beta)
    # Add noise
    y = beta + rng.normal(0.0, 0.1, size=len(beta))
    # Generate the constraints
    (A, s) = constraints_1D(s, I)
    # Rake using general distance
    (beta_star, lambda_star, iter_eps) = raking_general(y, A, s, -2.0)
    # Verify that the constraint A beta_star = s is respected
    assert np.allclose(np.matmul(A, beta_star), s), (
        "For the raking in 1D with the general distance, the constraint A beta_star = s is not respected."
    )


def test_general_raking_2D():
    # Generate balanced matrix
    I = 3
    J = 5
    rng = np.random.default_rng(0)
    beta = rng.uniform(low=2.0, high=3.0, size=(I, J))
    s1 = np.sum(beta, axis=0)
    s2 = np.sum(beta, axis=1)
    # Add noise
    y = beta + rng.normal(0.0, 0.1, size=beta.shape)
    y = y.flatten(order="F")
    # Generate the constraints
    (A, s) = constraints_2D(s1, s2, I, J)
    # Rake using general distance
    (beta_star, lambda_star, iter_eps) = raking_general(y, A, s, -2.0)
    # Verify that the constraint A beta_star = s is respected
    assert np.allclose(np.matmul(A, beta_star), s), (
        "For the raking in 2D with the general distance, the constraint A beta_star = s is not respected."
    )


def test_general_raking_3D():
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
    y = y.flatten(order="F")
    # Generate the constraints
    (A, s) = constraints_3D(s1, s2, s3, I, J, K)
    # Rake using general distance
    (beta_star, lambda_star, iter_eps) = raking_general(y, A, s, -2.0)
    # Verify that the constraint A beta_star = s is respected
    assert np.allclose(np.matmul(A, beta_star), s), (
        "For the raking in 3D with the general distance, the constraint A beta_star = s is not respected."
    )


def test_general_raking_USHD():
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
    beta_i = np.sum(beta_ijk, axis=(1, 2))
    beta_0 = np.sum(beta_i)
    s_cause = np.array([beta_0] + beta_i.tolist())
    # Add noise
    y = beta + rng.normal(0.0, 0.1, size=beta.shape)
    y = y.flatten(order="F")
    # Generate the constraints
    (A, s) = constraints_USHD(s_cause, I, J, K)
    # Rake using general distance
    (beta_star, lambda_star, iter_eps) = raking_general(y, A, s, -2.0)
    # Verify that the constraint A beta_star = s is respected
    assert np.allclose(np.matmul(A, beta_star), s, atol=1.0e-6), (
        "For the USHD raking with the general distance, the constraint A beta_star = s is not respected."
    )


def test_general_raking_USHD_lower():
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
    # Add noise
    y = beta + rng.normal(0.0, 0.1, size=beta.shape)
    y = y.flatten(order="F")
    # Generate the constraints
    (A, s) = constraints_USHD_lower(s_cause, s_county, s_all_causes, I, J, K)
    # Rake using general distance
    (beta_star, lambda_star, iter_eps) = raking_general(y, A, s, -2.0)
    # Verify that the constraint A beta_star = s is respected
    assert np.allclose(np.matmul(A, beta_star), s, atol=1.0e-5), (
        "For the USHD_lower raking with the general distance, the constraint A beta_star = s is not respected."
    )


def test_logit_raking_1D():
    # Generate balanced vector
    I = 3
    rng = np.random.default_rng(0)
    beta = rng.uniform(low=2.0, high=3.0, size=I)
    s = np.sum(beta)
    # Add noise
    y = beta + rng.normal(0.0, 0.1, size=len(beta))
    l = np.repeat(np.min(y), len(y))
    h = np.repeat(np.max(y), len(y))
    # Generate the constraints
    (A, s) = constraints_1D(s, I)
    # Rake using logit distance
    (beta_star, lambda_star, iter_eps) = raking_logit(y, A, s, l, h)
    # Verify that the constraint A beta_star = s is respected
    assert np.allclose(np.matmul(A, beta_star), s), (
        "For the raking in 1D with the logit distance, the constraint A beta_star = s is not respected."
    )
    # Verify that the lower bound is respected
    assert np.all(beta_star - l > -1.0e-5), (
        "For the raking in 1D with the logit distance, some raked values are lower than the lower bound."
    )
    # Verify that the upper bound is respected
    assert np.all(h - beta_star > -1.0e-5), (
        "For the raking in 1D with the logit distance, some raked values are higher than the upper bound."
    )


def test_logit_raking_2D():
    # Generate balanced matrix
    I = 3
    J = 5
    rng = np.random.default_rng(0)
    beta = rng.uniform(low=2.0, high=3.0, size=(I, J))
    s1 = np.sum(beta, axis=0)
    s2 = np.sum(beta, axis=1)
    # Add noise
    y = beta + rng.normal(0.0, 0.1, size=beta.shape)
    y = y.flatten(order="F")
    l = np.repeat(np.min(y), len(y))
    h = np.repeat(np.max(y), len(y))
    # Generate the constraints
    (A, s) = constraints_2D(s1, s2, I, J)
    # Rake using logit distance
    (beta_star, lambda_star, iter_eps) = raking_logit(y, A, s, l, h)
    # Verify that the constraint A beta_star = s is respected
    assert np.allclose(np.matmul(A, beta_star), s), (
        "For the raking in 2D with the logit distance, the constraint A beta_star = s is not respected."
    )
    # Verify that the lower bound is respected
    assert np.all(beta_star - l > -1.0e-5), (
        "For the raking in 2D with the logit distance, some raked values are lower than the lower bound."
    )
    # Verify that the upper bound is respected
    assert np.all(h - beta_star > -1.0e-5), (
        "For the raking in 2D with the logit distance, some raked values are higher than the upper bound."
    )


def test_logit_raking_3D():
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
    y = y.flatten(order="F")
    l = np.repeat(np.min(y), len(y))
    h = np.repeat(np.max(y), len(y))
    # Generate the constraints
    (A, s) = constraints_3D(s1, s2, s3, I, J, K)
    # Rake using logitdistance
    (beta_star, lambda_star, iter_eps) = raking_logit(y, A, s, l, h)
    # Verify that the constraint A beta_star = s is respected
    assert np.allclose(np.matmul(A, beta_star), s), (
        "For the raking in 3D with the logit distance, the constraint A beta_star = s is not respected."
    )
    # Verify that the lower bound is respected
    assert np.all(beta_star - l > -1.0e-5), (
        "For the raking in 3D with the logit distance, some raked values are lower than the lower bound."
    )
    # Verify that the upper bound is respected
    assert np.all(h - beta_star > -1.0e-5), (
        "For the raking in 3D with the logit distance, some raked values are higher than the upper bound."
    )


def test_logit_raking_USHD():
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
    beta_i = np.sum(beta_ijk, axis=(1, 2))
    beta_0 = np.sum(beta_i)
    s_cause = np.array([beta_0] + beta_i.tolist())
    # Add noise
    y = beta + rng.normal(0.0, 0.1, size=beta.shape)
    y = y.flatten(order="F")
    l = np.repeat(np.min(y), len(y))
    h = np.repeat(np.max(y), len(y))
    # Generate the constraints
    (A, s) = constraints_USHD(s_cause, I, J, K)
    # Rake using logit distance
    (beta_star, lambda_star, iter_eps) = raking_logit(y, A, s, l, h)
    # Verify that the constraint A beta_star = s is respected
    assert np.allclose(np.matmul(A, beta_star), s, atol=1.0e-6), (
        "For the USHD raking with the logit distance, the constraint A beta_star = s is not respected."
    )
    # Verify that the lower bound is respected
    assert np.all(beta_star - l > -1.0e-5), (
        "For the USHD raking with the logit distance, some raked values are lower than the lower bound."
    )
    # Verify that the upper bound is respected
    assert np.all(h - beta_star > -1.0e-5), (
        "For the USHD raking with the logit distance, some raked values are higher than the upper bound."
    )


def test_logit_raking_USHD_lower():
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
    # Add noise
    y = beta + rng.normal(0.0, 0.1, size=beta.shape)
    y = y.flatten(order="F")
    l = np.repeat(np.min(y), len(y))
    h = np.repeat(np.max(y), len(y))
    # Generate the constraints
    (A, s) = constraints_USHD_lower(s_cause, s_county, s_all_causes, I, J, K)
    # Rake using logit distance
    (beta_star, lambda_star, iter_eps) = raking_logit(y, A, s, l, h)
    # Verify that the constraint A beta_star = s is respected
    assert np.allclose(np.matmul(A, beta_star), s, atol=1.0e-5), (
        "For the USHD_lower raking with the logit distance, the constraint A beta_star = s is not respected."
    )
    # Verify that the lower bound is respected
    assert np.all(beta_star - l > -1.0e-5), (
        "For the USHD_lower raking with the logit distance, some raked values are lower than the lower bound."
    )
    # Verify that the upper bound is respected
    assert np.all(h - beta_star > -1.0e-5), (
        "For the USHD_lower raking with the logit distance, some raked values are higher than the upper bound."
    )

