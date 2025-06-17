import numpy as np
import matplotlib.pyplot as plt

# Parameters
n = 1000
true_mean = 50000
true_std = 15000
epsilon = 1.0
delta = 1e-5
sensitivity = 500  # example for mean query

# Generate synthetic data
np.random.seed(0)
data = np.random.normal(loc=true_mean, scale=true_std, size=n)
true_output = np.mean(data)

# Laplace noise
laplace_scale = sensitivity / epsilon
laplace_noise = np.random.laplace(loc=0, scale=laplace_scale, size=n)
laplace_noisy_data = data + laplace_noise
laplace_noisy_output = np.mean(laplace_noisy_data)

# Gaussian noise
gaussian_std = np.sqrt(2 * np.log(1.25 / delta)) * sensitivity / epsilon
gaussian_noise = np.random.normal(loc=0, scale=gaussian_std, size=n)
gaussian_noisy_data = data + gaussian_noise
gaussian_noisy_output = np.mean(gaussian_noisy_data)

# Compute SNRs
signal_var = np.var(data)
laplace_noise_var = np.var(laplace_noise)
gaussian_noise_var = np.var(gaussian_noise)

snr_laplace = signal_var / laplace_noise_var
snr_gaussian = signal_var / gaussian_noise_var

# Plotting distributions
plt.figure(figsize=(12, 6))
plt.hist(data, bins=50, alpha=0.5, label='Original Data')
plt.hist(laplace_noisy_data, bins=50, alpha=0.5, label='Laplace Noisy Data')
plt.hist(gaussian_noisy_data, bins=50, alpha=0.5, label='Gaussian Noisy Data')
plt.legend()
plt.title("Data Distributions with Laplace and Gaussian Noise")
plt.xlabel("Value")
plt.ylabel("Frequency")
plt.grid(True)
plt.show()

(signal_var, laplace_noise_var, gaussian_noise_var, snr_laplace, snr_gaussian)

"""
(np.float64(219202752.6702347),
 np.float64(529184.9695509374),
 np.float64(5399912.044085224),
 np.float64(414.2270950292647),
 np.float64(40.59376354293358))
"""