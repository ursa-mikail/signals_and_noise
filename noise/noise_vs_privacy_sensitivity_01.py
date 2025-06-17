import numpy as np
import matplotlib.pyplot as plt
import ipywidgets as widgets
from IPython.display import display

# Noise calculation function
def calculate_noise(epsilon, delta, sensitivity):
    laplace_scale = sensitivity / epsilon
    gaussian_stddev = np.sqrt(2 * np.log(1.25 / delta)) * sensitivity / epsilon
    return laplace_scale, gaussian_stddev

# Interactive update function
def update_plot(epsilon, delta, sensitivity):
    laplace_scale, gaussian_stddev = calculate_noise(epsilon, delta, sensitivity)

    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.bar(['Laplace Noise Scale', 'Gaussian StdDev'],
                  [laplace_scale, gaussian_stddev],
                  color=['orange', 'blue'])
    ax.set_title(f'Noise for ε={epsilon}, δ={delta:.1e}, Sensitivity={sensitivity}')
    ax.set_ylabel('Noise Magnitude')
    ax.grid(axis='y')

    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.3f}', xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom')

    plt.tight_layout()
    plt.show()

# Sliders
widgets.interact(update_plot,
    epsilon=widgets.FloatSlider(value=1.0, min=0.1, max=5.0, step=0.1, description='Epsilon (ε):'),
    delta=widgets.FloatLogSlider(value=1e-5, base=10, min=-8, max=-1, step=0.1, description='Delta (δ):'),
    sensitivity=widgets.FloatSlider(value=1.0, min=0.1, max=10.0, step=0.1, description='Sensitivity:')
);

