# Noise Comparison for ε-DP vs (ε, δ)-DP
"""
Laplace noise will have a sharp peak and taper off slowly (heavy tails).

Gaussian noise will be wider (due to high σ) but more concentrated in the center.

Gaussian noise often has higher variance, especially for small δ, due to the stronger requirement on high-probability privacy protection.
"""
import numpy as np
import matplotlib.pyplot as plt

# --- Parameters ---
epsilon = 1.0
delta = 1e-5
sensitivity = 100  # Sensitivity of average salary
num_samples = 10000

# --- Laplace noise (strict ε-DP) ---
laplace_scale = sensitivity / epsilon
laplace_noise = np.random.laplace(loc=0, scale=laplace_scale, size=num_samples)

# --- Gaussian noise ((ε, δ)-DP) ---
sigma = (np.sqrt(2 * np.log(1.25 / delta)) * sensitivity) / epsilon
gaussian_noise = np.random.normal(loc=0, scale=sigma, size=num_samples)

# --- Plot ---
plt.figure(figsize=(12, 6))
plt.hist(laplace_noise, bins=100, density=True, alpha=0.6, label=f"Laplace Noise (ε={epsilon})")
plt.hist(gaussian_noise, bins=100, density=True, alpha=0.6, label=f"Gaussian Noise (ε={epsilon}, δ={delta})")
plt.title("Laplace vs. Gaussian Noise for Differential Privacy")
plt.xlabel("Noise Value")
plt.ylabel("Density")
plt.legend()
plt.grid(True)
plt.show()

