import numpy as np
import matplotlib.pyplot as plt

# Define a function to calculate noise for Laplace and Gaussian mechanisms
def calculate_noise(epsilon, delta, sensitivity):
    # Laplace scale parameter
    laplace_scale = sensitivity / epsilon

    # Gaussian standard deviation
    gaussian_stddev = np.sqrt(2 * np.log(1.25 / delta)) * sensitivity / epsilon

    return laplace_scale, gaussian_stddev

# Example ranges for epsilon and sensitivity
epsilons = np.linspace(0.1, 2.0, 20)  # reasonable epsilon range
sensitivity = 1.0
delta = 1e-5

laplace_scales = []
gaussian_stddevs = []

for eps in epsilons:
    lap_scale, gauss_std = calculate_noise(eps, delta, sensitivity)
    laplace_scales.append(lap_scale)
    gaussian_stddevs.append(gauss_std)

# Plotting the noise scale vs epsilon
plt.figure(figsize=(10, 6))
plt.plot(epsilons, laplace_scales, label='Laplace Scale', marker='o')
plt.plot(epsilons, gaussian_stddevs, label='Gaussian StdDev', marker='x')
plt.xlabel('Epsilon (ε)')
plt.ylabel('Noise Magnitude')
plt.title('Noise vs Privacy Level (ε), Sensitivity=1, δ=1e-5')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

