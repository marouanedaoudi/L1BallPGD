import numpy as np

from .l1_projection import project_l1_ball
from .metrics import compute_loss, compute_sparsity


def pgd_l1_constrained(
    X: np.ndarray,
    y: np.ndarray,
    t: float,
    max_iter: int = 1000,
    tol: float = 1e-6,
    verbose: bool = True,
) -> dict:
    """
    Projected Gradient Descent for constrained Lasso.

    Solves: min 0.5*||y - X*beta||^2  s.t. ||beta||_1 <= t

    Parameters
    ----------
    X : np.ndarray, shape (n, p)
        Design matrix
    y : np.ndarray, shape (n,)
        Response vector
    t : float
        L1 ball radius
    max_iter : int
        Maximum number of iterations
    tol : float
        Convergence tolerance (relative change in beta)
    verbose : bool
        Print progress

    Returns
    -------
    results : dict
        - beta: final estimate
        - losses: loss at each iteration
        - sparsities: sparsity at each iteration
        - converged: whether algorithm converged
    """
    n, p = X.shape

    # Initialize
    beta = np.zeros(p)

    # Compute Lipschitz constant (step size = 1/L)
    L = np.linalg.norm(X, ord=2) ** 2
    eta = 1.0 / L

    # Storage
    losses = []
    sparsities = []

    for k in range(max_iter):
        # Compute gradient
        residual = X @ beta - y
        grad = X.T @ residual

        # Gradient step
        beta_new = beta - eta * grad

        # Project onto L1 ball
        beta_new = project_l1_ball(beta_new, t)

        # Check convergence
        rel_change = np.linalg.norm(beta_new - beta) / (np.linalg.norm(beta) + 1e-10)

        beta = beta_new

        # Track metrics
        loss = compute_loss(X, y, beta)
        sparsity = compute_sparsity(beta)
        losses.append(loss)
        sparsities.append(sparsity)

        if verbose and k % 100 == 0:
            print(
                f"Iter {k:4d} | Loss: {loss:.4e} | "
                f"Sparsity: {sparsity:3d} | Rel change: {rel_change:.4e}"
            )

        if rel_change < tol:
            if verbose:
                print(f"Converged at iteration {k}")
            break

    return {"beta": beta, "losses": losses, "sparsities": sparsities, "converged": k < max_iter - 1}
