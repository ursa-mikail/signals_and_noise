# Define ranges for epsilon, delta, and sensitivity
epsilon_values = [0.1, 0.5, 1.0, 2.0, 5.0]
delta_values = [1e-7, 1e-5, 1e-3]
sensitivity_values = [100, 300, 500, 700, 1000]

# Collect results for plotting
results = []

for eps in epsilon_values:
    for delt in delta_values:
        for sens in sensitivity_values:
            # Laplace noise SNR
            lap_var = 2 * (sens / eps) ** 2

            # Gaussian noise SNR
            gauss_std = np.sqrt(2 * np.log(1.25 / delt)) * sens / eps
            gauss_var = gauss_std ** 2

            snr_lap = signal_var / lap_var
            snr_gauss = signal_var / gauss_var

            results.append({
                "epsilon": eps,
                "delta": delt,
                "sensitivity": sens,
                "snr_laplace": snr_lap,
                "snr_gaussian": snr_gauss
            })

# Convert results to arrays for plotting
import pandas as pd
df = pd.DataFrame(results)

# Plot SNR heatmaps
import seaborn as sns
import matplotlib.pyplot as plt

fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

# Laplace SNR heatmap
pivot_lap = df[df["delta"] == 1e-5].pivot(index="sensitivity", columns="epsilon", values="snr_laplace")
sns.heatmap(pivot_lap, annot=True, fmt=".1f", cmap="YlGnBu", ax=axes[0])
axes[0].set_title("Laplace SNR (delta=1e-5)")
axes[0].set_ylabel("Sensitivity")
axes[0].set_xlabel("Epsilon")

# Gaussian SNR heatmap
pivot_gauss = df[df["delta"] == 1e-5].pivot(index="sensitivity", columns="epsilon", values="snr_gaussian")
sns.heatmap(pivot_gauss, annot=True, fmt=".1f", cmap="YlOrRd", ax=axes[1])
axes[1].set_title("Gaussian SNR (delta=1e-5)")
axes[1].set_ylabel("Sensitivity")
axes[1].set_xlabel("Epsilon")

plt.tight_layout()
plt.show()

