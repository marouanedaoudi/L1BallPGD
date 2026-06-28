import numpy as np

from .l1_projection import project_l1_ball
from .metrics import compute_loss, compute_sparsity


def fista_l1_constrained(
    X: np.ndarray,
    y: np.ndarray,
    t: float,
    max_iter: int = 1000,
    tol: float = 1e-6,
    verbose: bool = True,
) -> dict:
    """Accelerated Projected Gradient Descent (FISTA) for the constrained Lasso.

    Solves: min 0.5 * ||y - X @ beta||^2  s.t.  ||beta||_1 <= t

    FISTA adds a Nesterov momentum term on top of vanilla PGD, improving the
    worst-case rate of the objective gap from O(1/k) to O(1/k^2) at essentially
    the same per-iteration cost.

    Parameters
    ----------
    X : np.ndarray, shape (n, p)
        Design matrix.
    y : np.ndarray, shape (n,)
        Response vector.
    t : float
        L1-ball radius.
    max_iter : int
        Maximum number of iterations.
    tol : float
        Convergence tolerance on the relative change of the iterate.
    verbose : bool
        Print progress every 100 iterations.

    Returns
    -------
    results : dict
        - beta: final estimate
        - losses: loss at each iteration
        - sparsities: sparsity at each iteration
        - converged: whether the tolerance was reached before max_iter
    """
    n, p = X.shape

    # Step size eta = 1 / L with L the Lipschitz constant of grad f.
    L = np.linalg.norm(X, ord=2) ** 2
    eta = 1.0 / L

    beta = np.zeros(p)
    z = beta.copy()
    theta = 1.0

    losses: list[float] = []
    sparsities: list[int] = []
    converged = False

    for k in range(max_iter):
        # Gradient step at the momentum point z, then projection.
        grad = X.T @ (X @ z - y)
        beta_new = project_l1_ball(z - eta * grad, t)

        # Nesterov momentum update.
        theta_new = 0.5 * (1.0 + np.sqrt(1.0 + 4.0 * theta**2))
        z = beta_new + ((theta - 1.0) / theta_new) * (beta_new - beta)

        rel_change = np.linalg.norm(beta_new - beta) / (np.linalg.norm(beta) + 1e-10)

        beta = beta_new
        theta = theta_new

        losses.append(compute_loss(X, y, beta))
        sparsities.append(compute_sparsity(beta))

        if verbose and k % 100 == 0:
            print(
                f"Iter {k:4d} | Loss: {losses[-1]:.4e} | "
                f"Sparsity: {sparsities[-1]:3d} | Rel change: {rel_change:.4e}"
            )

        if rel_change < tol:
            if verbose:
                print(f"Converged at iteration {k}")
            converged = True
            break

    return {
        "beta": beta,
        "losses": losses,
        "sparsities": sparsities,
        "converged": converged,
    }
