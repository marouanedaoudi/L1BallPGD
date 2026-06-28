import numpy as np


def compute_loss(X: np.ndarray, y: np.ndarray, beta: np.ndarray) -> float:
    """Compute least squares loss: 0.5 * ||y - X*beta||_2^2"""
    residual = y - X @ beta
    return float(0.5 * np.sum(residual**2))


def compute_sparsity(beta: np.ndarray, tol: float = 1e-8) -> int:
    """Count number of non-zero coefficients (L0 norm)"""
    return int(np.sum(np.abs(beta) > tol))


def compute_error(beta: np.ndarray, beta_true: np.ndarray) -> float:
    """Compute L2 error: ||beta - beta_true||_2"""
    return float(np.linalg.norm(beta - beta_true))
