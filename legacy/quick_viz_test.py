"""Quick test of visualization functionality."""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def quick_multivariate_plot():
    """Create a simple multivariate plot to test the data."""
    # Load the data
    data_path = "data/engineered_multivariate_data.csv"
    if not Path(data_path).exists():
        print(f"Data file not found: {data_path}")
        return
    
    print("Loading data...")
    df = pd.read_csv(data_path)
    df['date'] = pd.to_datetime(df['date'])
    
    print(f"Dataset shape: {df.shape}")
    print(f"Columns: {list(df.columns[:10])}...")  # Show first 10 columns
    
    # Create a simple multivariate plot
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot 1: Sunspot numbers
    axes[0, 0].plot(df['date'], df['sunspot_number'], linewidth=0.8, alpha=0.7)
    axes[0, 0].set_title('Daily Sunspot Numbers (Non-Smoothed)')
    axes[0, 0].set_ylabel('Sunspot Number')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: F10.7 flux
    if 'f107_F10.7_ADJ' in df.columns:
        axes[0, 1].plot(df['date'], df['f107_F10.7_ADJ'], linewidth=0.8, alpha=0.7, color='blue')
        axes[0, 1].set_title('F10.7 Solar Flux')
        axes[0, 1].set_ylabel('F10.7 (SFU)')
        axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Geomagnetic activity
    if 'ap_avg' in df.columns:
        axes[1, 0].plot(df['date'], df['ap_avg'], linewidth=0.8, alpha=0.7, color='green')
        axes[1, 0].set_title('Geomagnetic Activity (Ap Index)')
        axes[1, 0].set_ylabel('Ap Index')
        axes[1, 0].set_xlabel('Date')
        axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Correlation scatter
    if 'f107_F10.7_ADJ' in df.columns:
        axes[1, 1].scatter(df['sunspot_number'], df['f107_F10.7_ADJ'], alpha=0.5, s=10)
        axes[1, 1].set_title('Sunspot vs F10.7 Correlation')
        axes[1, 1].set_xlabel('Sunspot Number')
        axes[1, 1].set_ylabel('F10.7 Flux')
        axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save the plot
    output_dir = Path("data/visualizations")
    output_dir.mkdir(exist_ok=True, parents=True)
    plt.savefig(output_dir / "quick_multivariate_overview.png", dpi=150, bbox_inches='tight')
    
    print(f"Quick visualization saved to: {output_dir / 'quick_multivariate_overview.png'}")
    
    # Show basic statistics
    print("\n=== DATA OVERVIEW ===")
    print(f"Total observations: {len(df)}")
    print(f"Date range: {df['date'].min()} to {df['date'].max()}")
    print(f"Total features: {len(df.columns)}")
    
    print("\n=== KEY VARIABLE STATISTICS ===")
    key_vars = ['sunspot_number', 'f107_F10.7_ADJ', 'ap_avg', 'kp_sum']
    for var in key_vars:
        if var in df.columns:
            stats = df[var].describe()
            print(f"\n{var}:")
            print(f"  Mean: {stats['mean']:.2f}, Std: {stats['std']:.2f}")
            print(f"  Min: {stats['min']:.2f}, Max: {stats['max']:.2f}")
            print(f"  Missing: {df[var].isna().sum()} ({df[var].isna().mean()*100:.1f}%)")
    
    print("\n=== FEATURE CATEGORIES ===")
    original_features = [c for c in df.columns if not any(x in c for x in ['roll', 'lag', 'diff', 'volatility', 'regime', 'sin', 'cos'])]
    rolling_features = [c for c in df.columns if 'roll' in c]
    lag_features = [c for c in df.columns if 'lag' in c]
    temporal_features = [c for c in df.columns if any(x in c for x in ['sin', 'cos', 'cycle', 'month', 'year'])]
    
    print(f"Original features: {len(original_features)}")
    print(f"Rolling statistics: {len(rolling_features)}")
    print(f"Lag features: {len(lag_features)}")
    print(f"Temporal features: {len(temporal_features)}")
    
    # Show recent data (last 30 days)
    print(f"\n=== RECENT DATA (Last 30 observations) ===")
    recent = df.tail(30)
    print(f"Recent sunspot activity: {recent['sunspot_number'].mean():.1f} ± {recent['sunspot_number'].std():.1f}")
    if 'f107_F10.7_ADJ' in df.columns:
        print(f"Recent F10.7 flux: {recent['f107_F10.7_ADJ'].mean():.1f} ± {recent['f107_F10.7_ADJ'].std():.1f}")
    

if __name__ == "__main__":
    quick_multivariate_plot()