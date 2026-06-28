import numpy as np


def generate_synthetic_data(
    n: int, p: int, sparsity: int, noise_std: float = 0.1, seed: int = 42
) -> tuple:
    """
    Generate synthetic regression data with sparse ground truth.

    Parameters
    ----------
    n : int
        Number of samples
    p : int
        Number of features
    sparsity : int
        Number of non-zero coefficients in true beta
    noise_std : float
        Standard deviation of Gaussian noise
    seed : int
        Random seed

    Returns
    -------
    X : np.ndarray, shape (n, p)
        Design matrix
    y : np.ndarray, shape (n,)
        Response vector
    beta_true : np.ndarray, shape (p,)
        True sparse coefficients
    """
    rng = np.random.default_rng(seed)

    # Generate design matrix (standardized)
    X = rng.standard_normal((n, p))
    X = (X - X.mean(axis=0)) / X.std(axis=0)

    # Generate sparse beta
    beta_true = np.zeros(p)
    support = rng.choice(p, size=sparsity, replace=False)
    beta_true[support] = rng.uniform(-2, 2, size=sparsity)

    # Generate response with noise
    y = X @ beta_true + noise_std * rng.standard_normal(n)

    return X, y, beta_true
