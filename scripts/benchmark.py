"""Validate and benchmark the constrained-Lasso solvers.

Two things are produced:

1. A correctness check against scikit-learn. The penalized Lasso and the
   L1-constrained Lasso are equivalent: if beta* solves the penalized problem,
   then beta* also solves the constrained problem with radius t = ||beta*||_1.
   We fit ``sklearn.linear_model.Lasso``, read off t, then check that our PGD
   and FISTA solvers recover the same coefficients.

2. A convergence comparison of PGD vs FISTA on the objective gap f(beta_k) - f*,
   illustrating the O(1/k) vs O(1/k^2) rates.
"""

import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import Lasso

from src.data import generate_synthetic_data
from src.fista_constrained_lasso import fista_l1_constrained
from src.metrics import compute_loss
from src.pgd_constrained_lasso import pgd_l1_constrained

OUT_DIR = "outputs"


def main() -> None:
    n, p, sparsity = 100, 200, 10
    X, y, beta_true = generate_synthetic_data(n, p, sparsity, noise_std=0.5)

    # --- 1. Correctness against scikit-learn -----------------------------------
    # sklearn solves (1 / (2n)) ||y - X beta||^2 + alpha ||beta||_1.
    alpha = 0.05
    lasso = Lasso(alpha=alpha, fit_intercept=False, max_iter=100_000, tol=1e-10)
    lasso.fit(X, y)
    beta_sklearn = lasso.coef_
    t = float(np.abs(beta_sklearn).sum())

    pgd = pgd_l1_constrained(X, y, t, max_iter=5000, tol=1e-10, verbose=False)
    fista = fista_l1_constrained(X, y, t, max_iter=5000, tol=1e-10, verbose=False)

    err_pgd = np.linalg.norm(pgd["beta"] - beta_sklearn)
    err_fista = np.linalg.norm(fista["beta"] - beta_sklearn)

    print("=== Validation against scikit-learn ===")
    print(f"sklearn alpha = {alpha} -> constrained radius t = ||beta*||_1 = {t:.4f}")
    print(f"||beta_PGD   - beta_sklearn||_2 = {err_pgd:.3e}")
    print(f"||beta_FISTA - beta_sklearn||_2 = {err_fista:.3e}")

    # --- 2. Convergence comparison: PGD vs FISTA -------------------------------
    f_star = min(min(pgd["losses"]), min(fista["losses"]), compute_loss(X, y, beta_sklearn))
    gap_pgd = np.array(pgd["losses"]) - f_star
    gap_fista = np.array(fista["losses"]) - f_star

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    ax = axes[0]
    ax.semilogy(gap_pgd, label="PGD", linewidth=2)
    ax.semilogy(gap_fista, label="FISTA", linewidth=2)
    kk = np.arange(1, len(gap_pgd) + 1)
    ax.semilogy(gap_pgd[0] / kk, "--", color="gray", alpha=0.7, label=r"$O(1/k)$")
    ax.semilogy(gap_fista[0] / kk**2, ":", color="black", alpha=0.7, label=r"$O(1/k^2)$")
    ax.set_xlabel("Iteration")
    ax.set_ylabel(r"Objective gap $f(\beta_k) - f^\star$")
    ax.set_title("Convergence: PGD vs FISTA")
    ax.legend()
    ax.grid(alpha=0.3)

    ax = axes[1]
    idx = np.arange(p)
    ax.stem(idx, beta_true, linefmt="C7-", markerfmt="C7o", basefmt=" ", label="true")
    ax.plot(idx, fista["beta"], "C1.", markersize=6, label="FISTA estimate")
    ax.set_xlabel("Coefficient index")
    ax.set_ylabel("Value")
    ax.set_title("Recovered support")
    ax.legend()
    ax.grid(alpha=0.3)

    fig.tight_layout()
    out = f"{OUT_DIR}/benchmark.png"
    fig.savefig(out, dpi=150)
    print(f"\nFigure saved to {out}")


if __name__ == "__main__":
    main()
