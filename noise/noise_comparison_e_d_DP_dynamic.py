import numpy as np
import matplotlib.pyplot as plt

# Function to generate and plot noise distributions
def compare_noise(epsilon_vals, delta_vals, sensitivity=100, num_samples=10000):
    for epsilon in epsilon_vals:
        for delta in delta_vals:
            # Laplace noise (ε-DP)
            laplace_scale = sensitivity / epsilon
            laplace_noise = np.random.laplace(loc=0, scale=laplace_scale, size=num_samples)
            
            # Gaussian noise ((ε, δ)-DP)
            sigma = (np.sqrt(2 * np.log(1.25 / delta)) * sensitivity) / epsilon
            gaussian_noise = np.random.normal(loc=0, scale=sigma, size=num_samples)

            # Plotting
            plt.figure(figsize=(10, 5))
            plt.hist(laplace_noise, bins=100, density=True, alpha=0.5, label=f"Laplace (ε={epsilon})")
            plt.hist(gaussian_noise, bins=100, density=True, alpha=0.5, label=f"Gaussian (ε={epsilon}, δ={delta})")
            plt.title(f"Noise Comparison - ε={epsilon}, δ={delta}")
            plt.xlabel("Noise Value")
            plt.ylabel("Density")
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.show()

# Run comparisons
epsilon_values = [0.1, 1.0, 5.0]     # Try low, medium, high privacy budgets
delta_values = [1e-3, 1e-5, 1e-8]    # Try less strict to very strict δ

compare_noise(epsilon_values, delta_values)

"""
🔍 Observe:
Smaller ε → more noise (stricter privacy).

Smaller δ (in Gaussian) → higher σ → broader Gaussian noise.

Larger ε → less noise → better utility, weaker privacy.

🧠 Insights:
Laplace noise directly shrinks with increasing ε.

Gaussian noise is very sensitive to δ; shrinking δ by orders of magnitude can blow up the required noise even with the same ε.

For tight δ (e.g., 1e-8), Gaussian noise can become very wide and destroy utility — Laplace might be preferred if δ is not allowed.


"""
