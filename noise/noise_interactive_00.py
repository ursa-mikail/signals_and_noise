import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.widgets import Slider, Button, RadioButtons
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px

class PrivacyUtilityAnalyzer:
    def __init__(self):
        self.delta = 1e-5  # Fixed delta for Gaussian mechanism
        self.setup_data()
    
    def setup_data(self):
        """Generate synthetic differential privacy analysis data"""
        # Simulate real-world scenario: Website click-through rates
        # Privacy parameters
        epsilon_vals = np.logspace(-1, 0.5, 20)  # 0.1 to ~3.16
        sensitivity_vals = np.linspace(0.5, 5.0, 18)
        delta_vals = [1e-5, 1e-4, 1e-3]
        
        # Generate comprehensive dataset
        data = []
        for delta in delta_vals:
            for eps in epsilon_vals:
                for sens in sensitivity_vals:
                    # Laplace mechanism
                    noise_laplace = sens / eps
                    snr_laplace = 1.0 / noise_laplace if noise_laplace > 0 else np.inf
                    utility_laplace = 1 / (1 + noise_laplace)
                    
                    # Gaussian mechanism (requires delta)
                    c = np.sqrt(2 * np.log(1.25 / delta))
                    noise_gaussian = (sens * c) / eps
                    snr_gaussian = 1.0 / noise_gaussian if noise_gaussian > 0 else np.inf
                    utility_gaussian = 1 / (1 + noise_gaussian)
                    
                    # Real-world metrics simulation
                    base_ctr = 0.05  # 5% baseline click-through rate
                    noisy_ctr_laplace = max(0, base_ctr + np.random.normal(0, noise_laplace/10))
                    noisy_ctr_gaussian = max(0, base_ctr + np.random.normal(0, noise_gaussian/10))
                    
                    data.append({
                        'epsilon': eps,
                        'sensitivity': sens,
                        'delta': delta,
                        'noise_laplace': noise_laplace,
                        'noise_gaussian': noise_gaussian,
                        'snr_laplace': snr_laplace,
                        'snr_gaussian': snr_gaussian,
                        'utility_laplace': utility_laplace,
                        'utility_gaussian': utility_gaussian,
                        'ctr_true': base_ctr,
                        'ctr_laplace': noisy_ctr_laplace,
                        'ctr_gaussian': noisy_ctr_gaussian
                    })
        
        self.df = pd.DataFrame(data)
    
    def create_matplotlib_plot(self, delta_val=1e-5):
        """Create matplotlib 3D surface plots"""
        # Filter data for specified delta
        filtered_df = self.df[self.df["delta"] == delta_val]
        
        # Prepare meshgrid for 3D plotting
        eps_vals = sorted(filtered_df["epsilon"].unique())
        sens_vals = sorted(filtered_df["sensitivity"].unique())
        eps_grid, sens_grid = np.meshgrid(eps_vals, sens_vals)
        
        # Reshape data for surface plotting
        snr_laplace_grid = filtered_df.pivot(index="sensitivity", columns="epsilon", values="snr_laplace").values
        snr_gaussian_grid = filtered_df.pivot(index="sensitivity", columns="epsilon", values="snr_gaussian").values
        utility_laplace_grid = filtered_df.pivot(index="sensitivity", columns="epsilon", values="utility_laplace").values
        utility_gaussian_grid = filtered_df.pivot(index="sensitivity", columns="epsilon", values="utility_gaussian").values
        
        # Create comprehensive visualization
        fig = plt.figure(figsize=(20, 12))
        
        # SNR Comparison
        ax1 = fig.add_subplot(221, projection='3d')
        surf1 = ax1.plot_surface(eps_grid, sens_grid, snr_laplace_grid, 
                                cmap=cm.viridis, alpha=0.8, edgecolor='none')
        ax1.set_title(f"Laplace SNR (δ = {delta_val})", fontsize=14, fontweight='bold')
        ax1.set_xlabel("Privacy Budget (ε)", fontsize=12)
        ax1.set_ylabel("Sensitivity", fontsize=12)
        ax1.set_zlabel("Signal-to-Noise Ratio", fontsize=12)
        fig.colorbar(surf1, ax=ax1, shrink=0.5, aspect=10)
        
        ax2 = fig.add_subplot(222, projection='3d')
        surf2 = ax2.plot_surface(eps_grid, sens_grid, snr_gaussian_grid, 
                                cmap=cm.plasma, alpha=0.8, edgecolor='none')
        ax2.set_title(f"Gaussian SNR (δ = {delta_val})", fontsize=14, fontweight='bold')
        ax2.set_xlabel("Privacy Budget (ε)", fontsize=12)
        ax2.set_ylabel("Sensitivity", fontsize=12)
        ax2.set_zlabel("Signal-to-Noise Ratio", fontsize=12)
        fig.colorbar(surf2, ax=ax2, shrink=0.5, aspect=10)
        
        # Utility Comparison
        ax3 = fig.add_subplot(223, projection='3d')
        surf3 = ax3.plot_surface(eps_grid, sens_grid, utility_laplace_grid, 
                                cmap=cm.YlOrRd, alpha=0.8, edgecolor='none')
        ax3.set_title("Laplace Utility Score", fontsize=14, fontweight='bold')
        ax3.set_xlabel("Privacy Budget (ε)", fontsize=12)
        ax3.set_ylabel("Sensitivity", fontsize=12)
        ax3.set_zlabel("Utility Score", fontsize=12)
        fig.colorbar(surf3, ax=ax3, shrink=0.5, aspect=10)
        
        ax4 = fig.add_subplot(224, projection='3d')
        surf4 = ax4.plot_surface(eps_grid, sens_grid, utility_gaussian_grid, 
                                cmap=cm.coolwarm, alpha=0.8, edgecolor='none')
        ax4.set_title("Gaussian Utility Score", fontsize=14, fontweight='bold')
        ax4.set_xlabel("Privacy Budget (ε)", fontsize=12)
        ax4.set_ylabel("Sensitivity", fontsize=12)
        ax4.set_zlabel("Utility Score", fontsize=12)
        fig.colorbar(surf4, ax=ax4, shrink=0.5, aspect=10)
        
        plt.suptitle("Website Analytics: Privacy-Utility Tradeoff Analysis", 
                    fontsize=16, fontweight='bold', y=0.95)
        plt.tight_layout()
        plt.show()
    
    def create_interactive_plotly(self):
        """Create interactive Plotly visualization"""
        # Filter for main analysis
        main_df = self.df[self.df["delta"] == 1e-5]
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=("Laplace SNR", "Gaussian SNR", "Utility Comparison", "Noise Analysis"),
            specs=[[{"type": "surface"}, {"type": "surface"}],
                   [{"type": "scatter3d"}, {"type": "scatter3d"}]]
        )
        
        # Prepare data for surface plots
        eps_vals = sorted(main_df["epsilon"].unique())
        sens_vals = sorted(main_df["sensitivity"].unique())
        
        snr_laplace = main_df.pivot(index="sensitivity", columns="epsilon", values="snr_laplace").values
        snr_gaussian = main_df.pivot(index="sensitivity", columns="epsilon", values="snr_gaussian").values
        
        # Surface plot 1: Laplace SNR
        fig.add_trace(
            go.Surface(
                x=eps_vals, y=sens_vals, z=snr_laplace,
                colorscale='Viridis',
                name='Laplace SNR',
                showscale=False
            ),
            row=1, col=1
        )
        
        # Surface plot 2: Gaussian SNR
        fig.add_trace(
            go.Surface(
                x=eps_vals, y=sens_vals, z=snr_gaussian,
                colorscale='Plasma',
                name='Gaussian SNR',
                showscale=False
            ),
            row=1, col=2
        )
        
        # 3D Scatter: Utility comparison
        fig.add_trace(
            go.Scatter3d(
                x=main_df['epsilon'],
                y=main_df['sensitivity'],
                z=main_df['utility_laplace'],
                mode='markers',
                marker=dict(
                    size=3,
                    color=main_df['utility_laplace'],
                    colorscale='YlOrRd',
                    opacity=0.8
                ),
                name='Laplace Utility',
                showlegend=False
            ),
            row=2, col=1
        )
        
        # 3D Scatter: Noise analysis
        fig.add_trace(
            go.Scatter3d(
                x=main_df['epsilon'],
                y=main_df['sensitivity'],
                z=main_df['noise_gaussian'],
                mode='markers',
                marker=dict(
                    size=3,
                    color=main_df['noise_gaussian'],
                    colorscale='Reds',
                    opacity=0.8
                ),
                name='Gaussian Noise',
                showlegend=False
            ),
            row=2, col=2
        )
        
        # Update layout
        fig.update_layout(
            title="Interactive Privacy-Utility Analysis Dashboard",
            font=dict(size=12),
            height=800,
            scene=dict(
                xaxis_title="Privacy Budget (ε)",
                yaxis_title="Sensitivity",
                zaxis_title="SNR"
            ),
            scene2=dict(
                xaxis_title="Privacy Budget (ε)",
                yaxis_title="Sensitivity",
                zaxis_title="SNR"
            ),
            scene3=dict(
                xaxis_title="Privacy Budget (ε)",
                yaxis_title="Sensitivity",
                zaxis_title="Utility Score"
            ),
            scene4=dict(
                xaxis_title="Privacy Budget (ε)",
                yaxis_title="Sensitivity",
                zaxis_title="Noise Magnitude"
            )
        )
        
        fig.show()
    
    def analyze_real_world_scenario(self):
        """Analyze a specific real-world scenario"""
        print("=== REAL-WORLD SCENARIO ANALYSIS ===")
        print("Website Analytics: Click-Through Rate Measurement")
        print("=" * 50)
        
        # Scenario parameters
        true_ctr = 0.05  # 5% click-through rate
        daily_visitors = 10000
        epsilon_values = [0.1, 0.5, 1.0, 2.0]
        sensitivity = 1.0 / daily_visitors  # Sensitivity for adding/removing one user
        
        print(f"True CTR: {true_ctr:.1%}")
        print(f"Daily Visitors: {daily_visitors:,}")
        print(f"Sensitivity: {sensitivity:.2e}")
        print()
        
        results = []
        for eps in epsilon_values:
            # Laplace noise
            laplace_noise_std = sensitivity / eps
            laplace_noise = np.random.laplace(0, laplace_noise_std, 1000)
            noisy_ctr_laplace = true_ctr + laplace_noise
            
            # Gaussian noise
            c = np.sqrt(2 * np.log(1.25 / self.delta))
            gaussian_noise_std = (sensitivity * c) / eps
            gaussian_noise = np.random.normal(0, gaussian_noise_std, 1000)
            noisy_ctr_gaussian = true_ctr + gaussian_noise
            
            # Calculate metrics
            mse_laplace = np.mean((noisy_ctr_laplace - true_ctr) ** 2)
            mse_gaussian = np.mean((noisy_ctr_gaussian - true_ctr) ** 2)
            
            results.append({
                'epsilon': eps,
                'laplace_mse': mse_laplace,
                'gaussian_mse': mse_gaussian,
                'laplace_std': laplace_noise_std,
                'gaussian_std': gaussian_noise_std
            })
            
            print(f"ε = {eps:.1f}:")
            print(f"  Laplace - Noise Std: {laplace_noise_std:.2e}, MSE: {mse_laplace:.2e}")
            print(f"  Gaussian - Noise Std: {gaussian_noise_std:.2e}, MSE: {mse_gaussian:.2e}")
            print()
        
        # Visualization of scenario results
        results_df = pd.DataFrame(results)
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # MSE Comparison
        ax1.plot(results_df['epsilon'], results_df['laplace_mse'], 'o-', label='Laplace', linewidth=2)
        ax1.plot(results_df['epsilon'], results_df['gaussian_mse'], 's-', label='Gaussian', linewidth=2)
        ax1.set_xlabel('Privacy Budget (ε)')
        ax1.set_ylabel('Mean Squared Error')
        ax1.set_title('Accuracy vs Privacy Budget')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_yscale('log')
        
        # Noise Standard Deviation
        ax2.plot(results_df['epsilon'], results_df['laplace_std'], 'o-', label='Laplace', linewidth=2)
        ax2.plot(results_df['epsilon'], results_df['gaussian_std'], 's-', label='Gaussian', linewidth=2)
        ax2.set_xlabel('Privacy Budget (ε)')
        ax2.set_ylabel('Noise Standard Deviation')
        ax2.set_title('Noise Level vs Privacy Budget')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_yscale('log')
        
        # Privacy-Utility Tradeoff
        utility_laplace = 1 / (1 + results_df['laplace_std'])
        utility_gaussian = 1 / (1 + results_df['gaussian_std'])
        
        ax3.plot(results_df['epsilon'], utility_laplace, 'o-', label='Laplace', linewidth=2)
        ax3.plot(results_df['epsilon'], utility_gaussian, 's-', label='Gaussian', linewidth=2)
        ax3.set_xlabel('Privacy Budget (ε)')
        ax3.set_ylabel('Utility Score')
        ax3.set_title('Utility vs Privacy Budget')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Combined Privacy-Utility Score
        privacy_scores = 1 / np.array(results_df['epsilon'])  # Higher ε = less privacy
        combined_laplace = utility_laplace * privacy_scores
        combined_gaussian = utility_gaussian * privacy_scores
        
        ax4.plot(results_df['epsilon'], combined_laplace, 'o-', label='Laplace', linewidth=2)
        ax4.plot(results_df['epsilon'], combined_gaussian, 's-', label='Gaussian', linewidth=2)
        ax4.set_xlabel('Privacy Budget (ε)')
        ax4.set_ylabel('Privacy-Utility Score')
        ax4.set_title('Combined Privacy-Utility Score')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.suptitle('Website Analytics: Privacy-Preserving CTR Measurement', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.show()

def main():
    """Main execution function"""
    print("Initializing Privacy-Utility Analyzer...")
    analyzer = PrivacyUtilityAnalyzer()
    
    print("1. Creating matplotlib 3D surface plots...")
    analyzer.create_matplotlib_plot()
    
    print("2. Creating interactive Plotly dashboard...")
    analyzer.create_interactive_plotly()
    
    print("3. Analyzing real-world scenario...")
    analyzer.analyze_real_world_scenario()
    
    print("\nAnalysis complete! Check the generated plots.")

if __name__ == "__main__":
    main()

"""
3. Analyzing real-world scenario...
=== REAL-WORLD SCENARIO ANALYSIS ===
Website Analytics: Click-Through Rate Measurement
==================================================
True CTR: 5.0%
Daily Visitors: 10,000
Sensitivity: 1.00e-04

ε = 0.1:
  Laplace - Noise Std: 1.00e-03, MSE: 2.04e-06
  Gaussian - Noise Std: 4.84e-03, MSE: 2.42e-05

ε = 0.5:
  Laplace - Noise Std: 2.00e-04, MSE: 8.15e-08
  Gaussian - Noise Std: 9.69e-04, MSE: 8.85e-07

ε = 1.0:
  Laplace - Noise Std: 1.00e-04, MSE: 2.08e-08
  Gaussian - Noise Std: 4.84e-04, MSE: 2.32e-07

ε = 2.0:
  Laplace - Noise Std: 5.00e-05, MSE: 4.71e-09
  Gaussian - Noise Std: 2.42e-04, MSE: 6.07e-08
"""