from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

# Prepare data for 3D plot
filtered_df = df[df["delta"] == 1e-5]
X, Y = np.meshgrid(sorted(filtered_df["epsilon"].unique()), sorted(filtered_df["sensitivity"].unique()))
Z_lap = filtered_df.pivot(index="sensitivity", columns="epsilon", values="snr_laplace").values
Z_gauss = filtered_df.pivot(index="sensitivity", columns="epsilon", values="snr_gaussian").values

fig = plt.figure(figsize=(16, 6))

# 3D plot for Laplace SNR
ax1 = fig.add_subplot(121, projection='3d')
ax1.plot_surface(X, Y, Z_lap, cmap=cm.viridis, edgecolor='k')
ax1.set_title("Laplace SNR (delta=1e-5)")
ax1.set_xlabel("Epsilon")
ax1.set_ylabel("Sensitivity")
ax1.set_zlabel("SNR")

# 3D plot for Gaussian SNR
ax2 = fig.add_subplot(122, projection='3d')
ax2.plot_surface(X, Y, Z_gauss, cmap=cm.plasma, edgecolor='k')
ax2.set_title("Gaussian SNR (delta=1e-5)")
ax2.set_xlabel("Epsilon")
ax2.set_ylabel("Sensitivity")
ax2.set_zlabel("SNR")

plt.tight_layout()
plt.show()
