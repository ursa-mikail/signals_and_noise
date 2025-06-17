import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Set seed for reproducibility
np.random.seed(42)

data_length = 10000

# Create a sample dataset
data = np.random.normal(loc=50, scale=10, size=data_length)  # mean=50, std=10

# Define queries (metrics to compute)
def compute_metrics(data):
    return {
        "mean": np.mean(data),
        "sum": np.sum(data),
        "count_above_60": np.sum(data > 60)
    }

# Add Laplacian and Gaussian noise
def add_laplace_noise(value, scale=5.0):
    return value + np.random.laplace(loc=0.0, scale=scale)

def add_gaussian_noise(value, scale=5.0):
    return value + np.random.normal(loc=0.0, scale=scale)

# Compute original metrics
original = compute_metrics(data)

# Compute metrics with noise
laplace_noisy = {k: add_laplace_noise(v) for k, v in original.items()}
gaussian_noisy = {k: add_gaussian_noise(v) for k, v in original.items()}

# Create dataframe for comparison
df = pd.DataFrame([original, laplace_noisy, gaussian_noisy], index=["Original", "Laplace Noise", "Gaussian Noise"])
df = df.T.round(2)  # Round for readability

# Plot 1: Metric comparison
# Plot comparison
fig, ax = plt.subplots(figsize=(10, 5))
df.plot(kind="bar", ax=ax)
plt.title("Comparison of Metrics: Original vs Noisy (Laplace & Gaussian)")
plt.ylabel("Value")
plt.xticks(rotation=0)
plt.grid(True)
plt.tight_layout()

df  # Show table too


# --- Plot 2: Noise distribution curves ---
# Simulate noise
laplace_noise = np.random.laplace(loc=0.0, scale=5.0, size=data_length)
gaussian_noise = np.random.normal(loc=0.0, scale=5.0, size=data_length)

# Plot histograms
fig2, ax2 = plt.subplots(figsize=(10, 5))
ax2.hist(laplace_noise, bins=100, density=True, alpha=0.6, label="Laplace", color="yellow")
ax2.hist(gaussian_noise, bins=100, density=True, alpha=0.6, label="Gaussian", color="lightblue")

# Add legend and labels
plt.title("Noise Distribution Curves (Laplace vs Gaussian)")
plt.xlabel("Noise Value")
plt.ylabel("Density")
plt.legend()
plt.grid(True)
plt.tight_layout()

plt.show()

laplace_noisy_signal = add_laplace_noise(data)
gaussian_noisy_signal = add_gaussian_noise(data)

# Plot histograms
fig3, ax3 = plt.subplots(figsize=(10, 5))
ax3.hist(laplace_noisy_signal, bins=100, density=True, alpha=0.6, label="laplace_noisy_signal", color="red")
ax3.hist(gaussian_noisy_signal, bins=100, density=True, alpha=0.6, label="gaussian_noisy_signal", color="yellow")

# Add legend and labels
plt.title("Noisy Signals Distribution Curves (Laplace vs Gaussian)")
plt.xlabel("Noise Value")
plt.ylabel("Density")
plt.legend()
plt.grid(True)
plt.tight_layout()

plt.show()

# ---------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt

# Sample user ages
# data = np.array([52, 47, 51, 53, 50])

import pandas as pd

# Set seed for reproducibility
np.random.seed(42)

# Create a sample dataset
data = np.random.normal(loc=50, scale=10, size=data_length)  # mean=50, std=10


# True metrics (no noise)
true_sum = np.sum(data)
true_mean = np.mean(data)
true_count = len(data)

# Privacy settings
epsilon = 1.0
sensitivity = 1.0  # Assume 1 for simplicity

# Add Laplacian Noise
lap_noise_sum = np.random.laplace(loc=0.0, scale=sensitivity/epsilon)
lap_noise_count = np.random.laplace(loc=0.0, scale=sensitivity/epsilon)
lap_sum = true_sum + lap_noise_sum
lap_count = true_count + lap_noise_count
lap_mean = lap_sum / lap_count

# Add Gaussian Noise
delta = 1e-5
gauss_stddev = np.sqrt(2 * np.log(1.25 / delta)) * sensitivity / epsilon
gauss_noise_sum = np.random.normal(loc=0.0, scale=gauss_stddev)
gauss_noise_count = np.random.normal(loc=0.0, scale=gauss_stddev)
gauss_sum = true_sum + gauss_noise_sum
gauss_count = true_count + gauss_noise_count
gauss_mean = gauss_sum / gauss_count

# Plot mean estimates for illustration
labels = ['True Mean', 'Laplacian Mean', 'Gaussian Mean']
values = [true_mean, lap_mean, gauss_mean]

plt.bar(labels, values, color=['green', 'orange', 'blue'])
plt.ylabel('Estimated Mean Age')
plt.title('Comparison of Means With and Without Noise')
plt.show()

