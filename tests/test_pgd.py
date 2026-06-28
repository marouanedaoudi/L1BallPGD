import numpy as np

from src.data import generate_synthetic_data
from src.pgd_constrained_lasso import pgd_l1_constrained


def test_pgd_convergence():
    """Test that PGD decreases loss"""
    X, y, beta_true = generate_synthetic_data(50, 100, 5, noise_std=0.1)

    results = pgd_l1_constrained(X, y, t=5.0, max_iter=100, verbose=False)

    losses = results["losses"]
    # Loss should decrease
    assert losses[-1] < losses[0]
    print("✓ test_pgd_convergence passed")


def test_pgd_constraint_satisfaction():
    """Test that final beta satisfies L1 constraint"""
    X, y, _ = generate_synthetic_data(50, 100, 5)

    t = 8.0
    results = pgd_l1_constrained(X, y, t=t, max_iter=200, verbose=False)

    beta = results["beta"]
    l1_norm = np.sum(np.abs(beta))

    # Should satisfy constraint (with numerical tolerance)
    assert l1_norm <= t + 1e-6
    print("✓ test_pgd_constraint_satisfaction passed")


def test_pgd_produces_sparsity():
    """Test that PGD produces sparse solutions"""
    X, y, beta_true = generate_synthetic_data(50, 100, 5, noise_std=0.5)

    # Small t should give sparse solution
    results = pgd_l1_constrained(X, y, t=3.0, max_iter=500, verbose=False)

    sparsity = results["sparsities"][-1]
    # Should be much sparser than dimension
    assert sparsity < 50  # less than half of features
    print("✓ test_pgd_produces_sparsity passed")


def test_pgd_zero_constraint():
    """Test PGD with t=0 gives zero solution"""
    X, y, _ = generate_synthetic_data(30, 50, 3)

    results = pgd_l1_constrained(X, y, t=0.0, max_iter=10, verbose=False)

    beta = results["beta"]
    np.testing.assert_allclose(beta, np.zeros_like(beta), atol=1e-10)
    print("✓ test_pgd_zero_constraint passed")


def test_pgd_large_constraint():
    """Test PGD with large t (close to unconstrained)"""
    X, y, beta_true = generate_synthetic_data(50, 30, 5, noise_std=0.1)

    # Very large t (essentially unconstrained)
    results = pgd_l1_constrained(X, y, t=1000.0, max_iter=500, verbose=False)

    # Should achieve low loss
    final_loss = results["losses"][-1]
    initial_loss = results["losses"][0]
    assert final_loss < 0.5 * initial_loss
    print("✓ test_pgd_large_constraint passed")


if __name__ == "__main__":
    test_pgd_convergence()
    test_pgd_constraint_satisfaction()
    test_pgd_produces_sparsity()
    test_pgd_zero_constraint()
    test_pgd_large_constraint()
    print("\n✅ All PGD tests passed!")
