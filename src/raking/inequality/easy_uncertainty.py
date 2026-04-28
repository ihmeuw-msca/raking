import copy
import numpy as np
import pandas as pd
from scipy.sparse.linalg import cg
import torch


def centering_step(A, b, x0, mu, epsilon, N, gamma, num_iter):
    """
    Implement Newton method to solve
    """
    x = copy.deepcopy(x0)
    n = 0
    F = np.matmul(np.matmul(np.transpose(A), A), x) - np.matmul(np.transpose(A), b) - mu * (1.0 / x)
    res = np.sqrt(np.dot(F, F))
    while (res > epsilon) & (n < N):
        J = np.matmul(np.transpose(A), A) + mu * np.diag(1.0 / np.square(x))
        Delta_x = cg(J, F)[0]
        m = 0
        i = 0
        x_n = x - 2.0 ** (-m) * Delta_x
        F_n = np.matmul(np.matmul(np.transpose(A), A), x_n) - np.matmul(np.transpose(A), b) - mu * (1.0 / x_n)
        res_n = np.sqrt(np.dot(F_n, F_n))
        while (i < num_iter) & ((res_n > (1.0 - gamma * 2.0 ** (-m)) * res) | (np.any(x_n < 0))):
            m = m + 1
            x_n = x - 2.0 ** (-m) * Delta_x
            F_n = np.matmul(np.matmul(np.transpose(A), A), x_n) - np.matmul(np.transpose(A), b) - mu * (1.0 / x_n)
            res_n = np.sqrt(np.dot(F_n, F_n))
            i = i + 1
        x = x - 2.0 ** (-m) * Delta_x
        F = np.matmul(np.matmul(np.transpose(A), A), x) - np.matmul(np.transpose(A), b) - mu * (1.0 / x)
        res = np.sqrt(np.dot(F, F))
        n = n + 1
    return x


def barrier_method(A, b, x0, mu, gamma, epsilon, N, gamma_iter, num_iter):
    """
    Implement barrier method to solve min_x |Ax - b|^2 s.t. x >= 0
    """
    x = copy.deepcopy(x0)
    while len(x) * mu > epsilon:
        x = centering_step(A, b, x, mu, epsilon, N, gamma_iter, num_iter)
        mu = gamma * mu
    return (x, mu)


def step_conjugate_gradient(b_in, A, x_star, mu_star):
    At = A.t()
    AtA = At @ A
    
    d_0 = -AtA @ x_star + At @ b_in + mu_star * (1.0 / x_star)
    F = AtA @ x_star - At @ b_in - mu_star * (1.0 / x_star)
    J = AtA + torch.diag(mu_star / (x_star**2))
    alpha = - (F @ d_0) / (d_0.t() @ J @ d_0)

    return x_star + alpha * d_0


def compute_jacobian(A, b, x_star, mu_star):
    """
    First step of conjugate gradient.
    """
    A = torch.tensor(A, dtype=torch.float32)
    b = torch.tensor(b, dtype=torch.float32)
    b.requires_grad = True
    x_star = torch.tensor(x_star, dtype=torch.float32)

    jacobian = torch.autograd.functional.jacobian(
        lambda b_val: step_conjugate_gradient(b_val, A, x_star, mu_star), b
    )
    return(jacobian)


def example_minimization():
    """
    Apply minimization algorithm on a simple example.
    """
    A = np.array([[13, 18], [18, 40]])
    b = np.array([2, -8])
    x0 = np.array([1, 1])
    mu = 1.0
    gamma = 0.1
    epsilon = 1.0e-5
    N = 100
    gamma_iter = 0.9
    num_iter = 100
    (x_star, mu_star) = barrier_method(A, b, x0, mu, gamma, epsilon, N, gamma_iter, num_iter)





