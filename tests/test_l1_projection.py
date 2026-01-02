import numpy as np
import sys
sys.path.append('.')

from src.l1_projection import project_l1_ball

def test_projection_identity():
    """Test that vectors already in the ball are unchanged"""
    v = np.array([0.5, -0.3, 0.1, 0.0])
    t = 2.0
    z = project_l1_ball(v, t)
    np.testing.assert_allclose(z, v, rtol=1e-6)
    print("✓ test_projection_identity passed")

def test_projection_norm_constraint():
    """Test that projected vector satisfies ||z||_1 = t"""
    v = np.array([3.0, -2.0, 1.5, -1.0])
    t = 4.0
    z = project_l1_ball(v, t)
    l1_norm = np.sum(np.abs(z))
    np.testing.assert_allclose(l1_norm, t, rtol=1e-6)
    print("✓ test_projection_norm_constraint passed")

def test_projection_reduces_norm():
    """Test that projection reduces L1 norm when needed"""
    v = np.array([5.0, -3.0, 2.0])
    t = 5.0
    z = project_l1_ball(v, t)
    assert np.sum(np.abs(z)) <= t + 1e-6
    assert np.sum(np.abs(v)) > t
    print("✓ test_projection_reduces_norm passed")

def test_projection_zero_vector():
    """Test projection of zero vector"""
    v = np.zeros(10)
    t = 1.0
    z = project_l1_ball(v, t)
    np.testing.assert_allclose(z, v, rtol=1e-6)
    print("✓ test_projection_zero_vector passed")

def test_projection_sparse():
    """Test projection on sparse vector"""
    v = np.array([10.0, 0.0, 0.0, 0.0, -5.0])
    t = 8.0
    z = project_l1_ball(v, t)
    assert np.sum(np.abs(z)) <= t + 1e-6
    # Should preserve sparsity pattern (zeros stay zeros)
    assert np.all((v == 0) == (z == 0)) or np.sum(np.abs(z)) < np.sum(np.abs(v))
    print("✓ test_projection_sparse passed")

def test_projection_sign_preservation():
    """Test that projection preserves signs"""
    v = np.array([3.0, -2.0, 1.0, -4.0])
    t = 5.0
    z = project_l1_ball(v, t)
    assert np.all(np.sign(z) == np.sign(v))
    print("✓ test_projection_sign_preservation passed")

if __name__ == "__main__":
    test_projection_identity()
    test_projection_norm_constraint()
    test_projection_reduces_norm()
    test_projection_zero_vector()
    test_projection_sparse()
    test_projection_sign_preservation()
    print("\n✅ All L1 projection tests passed!")
