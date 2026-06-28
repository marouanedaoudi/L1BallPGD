import numpy as np

from src.data import generate_synthetic_data
from src.fista_constrained_lasso import fista_l1_constrained
from src.pgd_constrained_lasso import pgd_l1_constrained


def test_fista_decreases_loss():
    """FISTA should drive the loss well below its starting value."""
    X, y, _ = generate_synthetic_data(50, 100, 5, noise_std=0.1)
    res = fista_l1_constrained(X, y, t=5.0, max_iter=200, verbose=False)
    assert res["losses"][-1] < res["losses"][0]
    print("✓ test_fista_decreases_loss passed")


def test_fista_satisfies_constraint():
    """The final iterate must lie inside the L1 ball."""
    X, y, _ = generate_synthetic_data(50, 100, 5)
    t = 8.0
    res = fista_l1_constrained(X, y, t=t, max_iter=300, verbose=False)
    assert np.abs(res["beta"]).sum() <= t + 1e-6
    print("✓ test_fista_satisfies_constraint passed")


def test_fista_matches_pgd_at_optimum():
    """PGD and FISTA solve the same problem, so they must agree at convergence."""
    X, y, _ = generate_synthetic_data(60, 120, 8, noise_std=0.3)
    t = 6.0
    pgd = pgd_l1_constrained(X, y, t=t, max_iter=5000, tol=1e-10, verbose=False)
    fista = fista_l1_constrained(X, y, t=t, max_iter=5000, tol=1e-10, verbose=False)
    assert np.linalg.norm(pgd["beta"] - fista["beta"]) < 1e-3
    print("✓ test_fista_matches_pgd_at_optimum passed")


def _iters_to_threshold(losses: list[float], threshold: float) -> int:
    """First iteration index whose loss drops at or below ``threshold``."""
    for k, loss in enumerate(losses):
        if loss <= threshold:
            return k
    return len(losses)


def test_fista_converges_faster():
    """FISTA reaches a target accuracy in no more iterations than PGD.

    FISTA is non-monotone, so we compare iterations-to-threshold rather than the
    gap at a fixed iteration.
    """
    X, y, _ = generate_synthetic_data(80, 150, 10, noise_std=0.3)
    t = 5.0
    pgd = pgd_l1_constrained(X, y, t=t, max_iter=2000, tol=0.0, verbose=False)
    fista = fista_l1_constrained(X, y, t=t, max_iter=2000, tol=0.0, verbose=False)

    f_star = min(min(pgd["losses"]), min(fista["losses"]))
    f0 = pgd["losses"][0]
    threshold = f_star + 1e-3 * (f0 - f_star)

    k_pgd = _iters_to_threshold(pgd["losses"], threshold)
    k_fista = _iters_to_threshold(fista["losses"], threshold)
    assert k_fista <= k_pgd
    print("✓ test_fista_converges_faster passed")


if __name__ == "__main__":
    test_fista_decreases_loss()
    test_fista_satisfies_constraint()
    test_fista_matches_pgd_at_optimum()
    test_fista_converges_faster()
    print("\n✅ All FISTA tests passed!")
