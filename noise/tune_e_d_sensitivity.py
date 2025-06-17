import numpy as np
import matplotlib.pyplot as plt
from ipywidgets import interact, FloatSlider

# Function to calculate noise parameters
def calculate_noise(epsilon, delta, sensitivity):
    laplace_scale = sensitivity / epsilon
    gaussian_stddev = np.sqrt(2 * np.log(1.25 / delta)) * sensitivity / epsilon
    return laplace_scale, gaussian_stddev

# Interactive plotting function
@interact(
    epsilon=FloatSlider(min=0.1, max=2.0, step=0.1, value=0.5, description='ε'),
    delta=FloatSlider(min=1e-6, max=1e-3, step=1e-6, value=1e-5, readout_format='.0e', description='δ'),
    sensitivity=FloatSlider(min=0.1, max=5.0, step=0.1, value=1.0, description='Sensitivity')
)
def plot_distributions_and_snr(epsilon, delta, sensitivity):
    lap_scale, gauss_std = calculate_noise(epsilon, delta, sensitivity)
    snr_laplace = 1.0 / lap_scale
    snr_gaussian = 1.0 / gauss_std

    x = np.linspace(-10, 10, 1000)
    laplace_pdf = (1 / (2 * lap_scale)) * np.exp(-np.abs(x) / lap_scale)
    gaussian_pdf = (1 / (gauss_std * np.sqrt(2 * np.pi))) * np.exp(-0.5 * (x / gauss_std)**2)

    fig, axs = plt.subplots(1, 2, figsize=(14, 5))

    # Plot the noise distributions
    axs[0].plot(x, laplace_pdf, label='Laplace')
    axs[0].plot(x, gaussian_pdf, label='Gaussian')
    axs[0].set_title(f'Distributions (ε={epsilon}, δ={delta:.0e}, Sens={sensitivity})')
    axs[0].set_xlabel('Noise Value')
    axs[0].set_ylabel('Probability Density')
    axs[0].legend()
    axs[0].grid(True)

    # Plot SNRs
    axs[1].bar(['Laplace', 'Gaussian'], [snr_laplace, snr_gaussian], color=['blue', 'orange'])
    axs[1].set_title('Signal-to-Noise Ratio (SNR)')
    axs[1].set_ylabel('SNR (Signal / Noise)')
    axs[1].grid(axis='y')

    plt.tight_layout()
    plt.show()

