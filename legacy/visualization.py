"""
Comprehensive visualization module for multivariate sunspot data analysis.
Creates publication-quality plots for the updated research paper.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from pathlib import Path
from typing import List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')


class SolarDataVisualizer:
    """Comprehensive visualization tools for multivariate solar data."""
    
    def __init__(self, figsize: Tuple[int, int] = (15, 10)):
        self.figsize = figsize
        self.setup_style()
        
    def setup_style(self):
        """Set up publication-quality plotting style."""
        plt.style.use('seaborn-v0_8-whitegrid')
        sns.set_palette("husl")
        
        # Configure matplotlib for better plots
        plt.rcParams.update({
            'figure.figsize': self.figsize,
            'font.size': 12,
            'axes.labelsize': 14,
            'axes.titlesize': 16,
            'xtick.labelsize': 11,
            'ytick.labelsize': 11,
            'legend.fontsize': 12,
            'figure.titlesize': 18
        })
    
    def plot_time_series_overview(self, df: pd.DataFrame, save_path: Optional[str] = None):
        """Create comprehensive time series overview of all major variables."""
        fig, axes = plt.subplots(4, 1, figsize=(16, 20), sharex=True)
        
        # Convert date column
        df['date'] = pd.to_datetime(df['date'])
        
        # Plot 1: Sunspot Numbers (actual, non-smoothed)
        axes[0].plot(df['date'], df['sunspot_number'], linewidth=1.2, alpha=0.8, color='red', label='Daily Sunspot Number')
        if 'sunspot_number_roll_mean_30' in df.columns:
            axes[0].plot(df['date'], df['sunspot_number_roll_mean_30'], linewidth=2, color='darkred', label='30-day Rolling Mean')
        axes[0].set_ylabel('Sunspot Number')
        axes[0].set_title('Daily Sunspot Numbers (Non-Smoothed) - Solar Cycle Activity', fontsize=16, pad=20)
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Plot 2: F10.7 Solar Flux
        if 'f107_F10.7_ADJ' in df.columns:
            axes[1].plot(df['date'], df['f107_F10.7_ADJ'], linewidth=1.2, alpha=0.8, color='blue', label='F10.7 Solar Flux (Adjusted)')
            if 'f107_roll_mean_30' in df.columns:
                axes[1].plot(df['date'], df['f107_roll_mean_30'], linewidth=2, color='darkblue', label='30-day Rolling Mean')
        axes[1].set_ylabel('F10.7 Flux (SFU)')
        axes[1].set_title('Solar Radio Flux at 10.7 cm', fontsize=16, pad=20)
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        # Plot 3: Geomagnetic Indices
        if 'ap_avg' in df.columns:
            ax3_twin = axes[2].twinx()
            line1 = axes[2].plot(df['date'], df['ap_avg'], linewidth=1.2, alpha=0.8, color='green', label='Ap Index')
            if 'kp_sum' in df.columns:
                line2 = ax3_twin.plot(df['date'], df['kp_sum'], linewidth=1.2, alpha=0.8, color='orange', label='Kp Sum')
            
            axes[2].set_ylabel('Ap Index', color='green')
            ax3_twin.set_ylabel('Kp Sum', color='orange')
            axes[2].set_title('Geomagnetic Activity Indices', fontsize=16, pad=20)
            
            # Combine legends
            lines1, labels1 = axes[2].get_legend_handles_labels()
            lines2, labels2 = ax3_twin.get_legend_handles_labels()
            axes[2].legend(lines1 + lines2, labels1 + labels2, loc='upper right')
            axes[2].grid(True, alpha=0.3)
        
        # Plot 4: Solar Cycle Phases and Activity Regimes
        if 'solar_cycle_phase' in df.columns:
            axes[3].plot(df['date'], df['solar_cycle_phase'], linewidth=1.5, color='purple', label='Solar Cycle Phase')
            if any(col.endswith('_regime_high') for col in df.columns):
                regime_col = [col for col in df.columns if col.endswith('_regime_high')][0]
                high_activity = df[df[regime_col] == 1]
                axes[3].scatter(high_activity['date'], high_activity['solar_cycle_phase'], 
                              color='red', alpha=0.6, s=20, label='High Activity Periods')
        
        axes[3].set_ylabel('Solar Cycle Phase')
        axes[3].set_xlabel('Date')
        axes[3].set_title('Solar Cycle Phase and Activity Regimes', fontsize=16, pad=20)
        axes[3].legend()
        axes[3].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Time series overview saved to: {save_path}")
        
        plt.show()
    
    def plot_correlation_matrix(self, df: pd.DataFrame, save_path: Optional[str] = None):
        """Create correlation matrix heatmap for key variables."""
        # Select key variables for correlation analysis
        key_vars = ['sunspot_number']
        
        # Add F10.7 variables
        f107_vars = [col for col in df.columns if 'f107' in col.lower() and 'adj' in col.lower()]
        key_vars.extend(f107_vars[:2])  # Take first 2 F10.7 variables
        
        # Add geomagnetic variables
        geomag_vars = [col for col in df.columns if any(x in col.lower() for x in ['ap_avg', 'kp_sum'])]
        key_vars.extend(geomag_vars)
        
        # Add some engineered features
        rolling_vars = [col for col in df.columns if 'roll_mean_30' in col and 'sunspot' in col]
        key_vars.extend(rolling_vars[:1])
        
        volatility_vars = [col for col in df.columns if 'volatility' in col and 'sunspot' in col]
        key_vars.extend(volatility_vars[:2])
        
        # Filter to existing columns
        key_vars = [var for var in key_vars if var in df.columns]
        
        if len(key_vars) < 2:
            print("Not enough variables for correlation matrix")
            return
        
        # Calculate correlation matrix
        corr_matrix = df[key_vars].corr()
        
        # Create heatmap
        plt.figure(figsize=(12, 10))
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        
        sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='RdBu_r', center=0,
                   square=True, linewidths=0.5, cbar_kws={"shrink": 0.5}, fmt='.2f')
        
        plt.title('Correlation Matrix: Key Solar and Geomagnetic Variables', fontsize=16, pad=20)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Correlation matrix saved to: {save_path}")
        
        plt.show()
    
    def plot_solar_cycle_analysis(self, df: pd.DataFrame, save_path: Optional[str] = None):
        """Analyze sunspot activity by solar cycle phases."""
        if 'solar_cycle_phase' not in df.columns:
            print("Solar cycle phase data not available")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Plot 1: Sunspot activity vs solar cycle phase
        axes[0, 0].scatter(df['solar_cycle_phase'], df['sunspot_number'], alpha=0.6, s=20)
        axes[0, 0].set_xlabel('Solar Cycle Phase')
        axes[0, 0].set_ylabel('Sunspot Number')
        axes[0, 0].set_title('Sunspot Activity vs Solar Cycle Phase')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Box plot of sunspot activity by cycle phase bins
        df['cycle_phase_bin'] = pd.cut(df['solar_cycle_phase'], bins=10, labels=False)
        df.boxplot(column='sunspot_number', by='cycle_phase_bin', ax=axes[0, 1])
        axes[0, 1].set_xlabel('Solar Cycle Phase Bin')
        axes[0, 1].set_ylabel('Sunspot Number')
        axes[0, 1].set_title('Sunspot Distribution by Cycle Phase')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Seasonal patterns
        df['month'] = pd.to_datetime(df['date']).dt.month
        monthly_avg = df.groupby('month')['sunspot_number'].mean()
        axes[1, 0].plot(monthly_avg.index, monthly_avg.values, marker='o', linewidth=2)
        axes[1, 0].set_xlabel('Month')
        axes[1, 0].set_ylabel('Average Sunspot Number')
        axes[1, 0].set_title('Seasonal Patterns in Sunspot Activity')
        axes[1, 0].set_xticks(range(1, 13))
        axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 4: Activity regime transitions
        if any(col.endswith('_regime_duration') for col in df.columns):
            regime_dur_col = [col for col in df.columns if col.endswith('_regime_duration')][0]
            axes[1, 1].hist(df[regime_dur_col], bins=50, alpha=0.7, edgecolor='black')
            axes[1, 1].set_xlabel('Regime Duration (days)')
            axes[1, 1].set_ylabel('Frequency')
            axes[1, 1].set_title('Distribution of Activity Regime Durations')
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Solar cycle analysis saved to: {save_path}")
        
        plt.show()
    
    def plot_feature_importance_preview(self, df: pd.DataFrame, save_path: Optional[str] = None):
        """Create preview of feature distributions and relationships."""
        # Select diverse feature types for visualization
        feature_categories = {
            'Original': ['sunspot_number', 'f107_F10.7_ADJ', 'ap_avg'],
            'Rolling Stats': [col for col in df.columns if 'roll_mean' in col][:3],
            'Volatility': [col for col in df.columns if 'volatility' in col][:3],
            'Temporal': [col for col in df.columns if any(x in col for x in ['month_sin', 'solar_cycle_sin'])][:2]
        }
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        axes = axes.flatten()
        
        for i, (category, features) in enumerate(feature_categories.items()):
            if i >= 4:
                break
                
            # Filter to existing features
            existing_features = [f for f in features if f in df.columns]
            if not existing_features:
                continue
            
            # Plot distributions
            for j, feature in enumerate(existing_features[:3]):  # Max 3 features per subplot
                if df[feature].dtype in ['float64', 'int64']:
                    axes[i].hist(df[feature].dropna(), bins=50, alpha=0.6, 
                               label=feature.replace('sunspot_number_', '').replace('f107_', ''),
                               density=True)
            
            axes[i].set_title(f'{category} Features Distribution')
            axes[i].set_xlabel('Feature Value')
            axes[i].set_ylabel('Density')
            axes[i].legend()
            axes[i].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Feature importance preview saved to: {save_path}")
        
        plt.show()
    
    def create_interactive_dashboard(self, df: pd.DataFrame, save_path: Optional[str] = None):
        """Create interactive Plotly dashboard for data exploration."""
        # Create subplot figure
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=['Sunspot Numbers Over Time', 'F10.7 Solar Flux',
                          'Geomagnetic Indices', 'Solar Cycle Analysis',
                          'Correlation Heatmap', 'Activity Regimes'],
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": True}, {"secondary_y": False}],
                   [{"type": "heatmap"}, {"secondary_y": False}]]
        )
        
        df['date'] = pd.to_datetime(df['date'])
        
        # Sunspot numbers
        fig.add_trace(go.Scatter(x=df['date'], y=df['sunspot_number'],
                                mode='lines', name='Daily Sunspot Number',
                                line=dict(width=1, color='red')), row=1, col=1)
        
        if 'sunspot_number_roll_mean_30' in df.columns:
            fig.add_trace(go.Scatter(x=df['date'], y=df['sunspot_number_roll_mean_30'],
                                    mode='lines', name='30-day Average',
                                    line=dict(width=2, color='darkred')), row=1, col=1)
        
        # F10.7 flux
        if 'f107_F10.7_ADJ' in df.columns:
            fig.add_trace(go.Scatter(x=df['date'], y=df['f107_F10.7_ADJ'],
                                    mode='lines', name='F10.7 Flux',
                                    line=dict(width=1, color='blue')), row=1, col=2)
        
        # Geomagnetic indices
        if 'ap_avg' in df.columns:
            fig.add_trace(go.Scatter(x=df['date'], y=df['ap_avg'],
                                    mode='lines', name='Ap Index',
                                    line=dict(width=1, color='green')), row=2, col=1)
        
        if 'kp_sum' in df.columns:
            fig.add_trace(go.Scatter(x=df['date'], y=df['kp_sum'],
                                    mode='lines', name='Kp Sum',
                                    line=dict(width=1, color='orange')), row=2, col=1, secondary_y=True)
        
        # Solar cycle phase
        if 'solar_cycle_phase' in df.columns:
            fig.add_trace(go.Scatter(x=df['date'], y=df['solar_cycle_phase'],
                                    mode='lines', name='Solar Cycle Phase',
                                    line=dict(width=2, color='purple')), row=2, col=2)
        
        # Correlation heatmap
        key_vars = ['sunspot_number']
        if 'f107_F10.7_ADJ' in df.columns:
            key_vars.append('f107_F10.7_ADJ')
        if 'ap_avg' in df.columns:
            key_vars.append('ap_avg')
        
        if len(key_vars) > 1:
            corr_matrix = df[key_vars].corr()
            fig.add_trace(go.Heatmap(z=corr_matrix.values,
                                    x=corr_matrix.columns,
                                    y=corr_matrix.index,
                                    colorscale='RdBu',
                                    zmid=0), row=3, col=1)
        
        # Activity regimes
        if any(col.endswith('_regime_high') for col in df.columns):
            regime_col = [col for col in df.columns if col.endswith('_regime_high')][0]
            high_activity = df[df[regime_col] == 1]
            fig.add_trace(go.Scatter(x=high_activity['date'], y=high_activity['sunspot_number'],
                                    mode='markers', name='High Activity',
                                    marker=dict(color='red', size=4)), row=3, col=2)
        
        # Update layout
        fig.update_layout(height=1200, showlegend=True,
                         title_text="Interactive Multivariate Solar Data Dashboard")
        
        if save_path:
            fig.write_html(save_path)
            print(f"Interactive dashboard saved to: {save_path}")
        
        fig.show()
    
    def generate_all_visualizations(self, df: pd.DataFrame, output_dir: str = "visualizations"):
        """Generate all visualization types and save to output directory."""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        print("Generating comprehensive multivariate visualizations...")
        
        # Time series overview
        print("1. Creating time series overview...")
        self.plot_time_series_overview(df, save_path=str(output_path / "01_time_series_overview.png"))
        
        # Correlation matrix
        print("2. Creating correlation matrix...")
        self.plot_correlation_matrix(df, save_path=str(output_path / "02_correlation_matrix.png"))
        
        # Solar cycle analysis
        print("3. Creating solar cycle analysis...")
        self.plot_solar_cycle_analysis(df, save_path=str(output_path / "03_solar_cycle_analysis.png"))
        
        # Feature importance preview
        print("4. Creating feature distributions...")
        self.plot_feature_importance_preview(df, save_path=str(output_path / "04_feature_distributions.png"))
        
        # Interactive dashboard
        print("5. Creating interactive dashboard...")
        self.create_interactive_dashboard(df, save_path=str(output_path / "05_interactive_dashboard.html"))
        
        print(f"\nAll visualizations saved to: {output_path}")
        
        # Generate summary statistics
        self.generate_data_summary(df, str(output_path / "data_summary.txt"))
    
    def generate_data_summary(self, df: pd.DataFrame, save_path: str):
        """Generate comprehensive data summary statistics."""
        with open(save_path, 'w') as f:
            f.write("MULTIVARIATE SOLAR DATA SUMMARY\n")
            f.write("=" * 50 + "\n\n")
            
            f.write(f"Dataset Shape: {df.shape}\n")
            f.write(f"Date Range: {df['date'].min()} to {df['date'].max()}\n")
            f.write(f"Total Days: {len(df)}\n\n")
            
            # Key variable statistics
            f.write("KEY VARIABLE STATISTICS:\n")
            f.write("-" * 30 + "\n")
            
            key_vars = ['sunspot_number', 'f107_F10.7_ADJ', 'ap_avg', 'kp_sum']
            for var in key_vars:
                if var in df.columns:
                    stats = df[var].describe()
                    f.write(f"\n{var}:\n")
                    f.write(f"  Mean: {stats['mean']:.2f}\n")
                    f.write(f"  Std:  {stats['std']:.2f}\n")
                    f.write(f"  Min:  {stats['min']:.2f}\n")
                    f.write(f"  Max:  {stats['max']:.2f}\n")
                    f.write(f"  Missing: {df[var].isna().sum()} ({df[var].isna().mean()*100:.1f}%)\n")
            
            f.write(f"\nFeature Categories:\n")
            f.write("-" * 20 + "\n")
            
            categories = {
                'Original': len([c for c in df.columns if not any(x in c for x in ['roll', 'lag', 'diff', 'volatility', 'regime'])]),
                'Rolling Stats': len([c for c in df.columns if 'roll' in c]),
                'Lag Features': len([c for c in df.columns if 'lag' in c]),
                'Difference': len([c for c in df.columns if 'diff' in c or 'pct_change' in c]),
                'Volatility': len([c for c in df.columns if 'volatility' in c]),
                'Regime': len([c for c in df.columns if 'regime' in c])
            }
            
            for cat, count in categories.items():
                f.write(f"{cat}: {count} features\n")
        
        print(f"Data summary saved to: {save_path}")


def main():
    """Example usage of visualization tools."""
    # Load engineered data
    data_path = "data/engineered_multivariate_data.csv"
    if not Path(data_path).exists():
        print(f"Data file not found: {data_path}")
        print("Run data collection and preprocessing first!")
        return
    
    print("Loading engineered multivariate data...")
    df = pd.read_csv(data_path)
    
    print(f"Loaded dataset with shape: {df.shape}")
    print(f"Features: {len(df.columns)} total")
    
    # Initialize visualizer
    visualizer = SolarDataVisualizer()
    
    # Generate all visualizations
    visualizer.generate_all_visualizations(df, output_dir="visualizations")


if __name__ == "__main__":
    main()