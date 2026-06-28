import matplotlib.pyplot as plt
import numpy as np

from src.data import generate_synthetic_data
from src.metrics import compute_error
from src.pgd_constrained_lasso import pgd_l1_constrained

# Generate data
n, p = 100, 200
sparsity = 10
X, y, beta_true = generate_synthetic_data(n, p, sparsity, noise_std=0.5)

print(f"Data: n={n}, p={p}, true sparsity={sparsity}")
print(f"True L1 norm: {np.sum(np.abs(beta_true)):.2f}")

# Run PGD with L1 constraint
t = 8.0  # L1 ball radius
results = pgd_l1_constrained(X, y, t, max_iter=1000, verbose=True)

# Compute final metrics
beta_pgd = results["beta"]
error = compute_error(beta_pgd, beta_true)
final_sparsity = results["sparsities"][-1]

print("\n=== Final Results ===")
print(f"L2 error: {error:.4f}")
print(f"Final sparsity: {final_sparsity}")
print(f"Final L1 norm: {np.sum(np.abs(beta_pgd)):.2f}")

# Plot results
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Loss curve
axes[0].plot(results["losses"], linewidth=2)
axes[0].set_xlabel("Iteration")
axes[0].set_ylabel("Loss")
axes[0].set_title("Loss vs Iteration")
axes[0].grid(alpha=0.3)

# Sparsity curve
axes[1].plot(results["sparsities"], linewidth=2, color="orange")
axes[1].axhline(sparsity, color="red", linestyle="--", label="True sparsity")
axes[1].set_xlabel("Iteration")
axes[1].set_ylabel("Number of non-zeros")
axes[1].set_title("Sparsity vs Iteration")
axes[1].legend()
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig("outputs/results.png", dpi=150)
print("\nPlot saved to outputs/results.png")
