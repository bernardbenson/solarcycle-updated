"""
Efficient visualization for large historical dataset.
Optimized for 275+ years of solar data.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

def load_and_sample_data(data_path, sample_size=10000):
    """Load data and create efficient samples for visualization."""
    print("Loading full historical dataset...")
    df = pd.read_csv(data_path, low_memory=False)
    df['date'] = pd.to_datetime(df['date'])
    
    print(f"Full dataset: {df.shape}")
    print(f"Date range: {df['date'].min()} to {df['date'].max()}")
    
    # Create different sampling strategies
    datasets = {}
    
    # 1. Recent data (last 10 years) - full resolution
    recent_cutoff = df['date'].max() - pd.DateOffset(years=10)
    datasets['recent'] = df[df['date'] >= recent_cutoff].copy()
    
    # 2. Full history - monthly averages for overview (numeric columns only)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    df_numeric = df[['date'] + numeric_cols].copy()
    df_monthly = df_numeric.set_index('date').resample('M').mean().reset_index()
    datasets['monthly'] = df_monthly
    
    # 3. Random sample for correlations
    sample_indices = np.random.choice(len(df), min(sample_size, len(df)), replace=False)
    datasets['sample'] = df.iloc[sample_indices].copy()
    
    # 4. Solar cycle peaks/minimums (high/low activity periods)
    high_activity = df[df['sunspot_number'] > df['sunspot_number'].quantile(0.9)]
    low_activity = df[df['sunspot_number'] < df['sunspot_number'].quantile(0.1)]
    datasets['extremes'] = pd.concat([high_activity, low_activity])
    
    print(f"Created efficient datasets:")
    for name, data in datasets.items():
        print(f"  {name}: {len(data)} observations")
    
    return df, datasets

def plot_historical_overview(df, datasets, save_dir):
    """Create comprehensive historical overview."""
    fig, axes = plt.subplots(4, 1, figsize=(20, 16))
    
    monthly_df = datasets['monthly']
    
    # Plot 1: Full sunspot history (monthly averages)
    axes[0].plot(monthly_df['date'], monthly_df['sunspot_number'], 
                linewidth=1, alpha=0.8, color='red', label='Monthly Average')
    axes[0].fill_between(monthly_df['date'], monthly_df['sunspot_number'], 
                        alpha=0.3, color='red')
    axes[0].set_ylabel('Sunspot Number')
    axes[0].set_title('Complete Sunspot History: 1818-2025 (275+ Years)', fontsize=16, pad=20)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Highlight major solar cycles
    cycle_years = range(1820, 2030, 11)  # Approximate 11-year cycles
    for year in cycle_years[::2]:  # Every other cycle
        axes[0].axvline(pd.to_datetime(f'{year}-01-01'), color='gray', alpha=0.3, linestyle='--')
    
    # Plot 2: Solar cycle envelope (running max/min)
    window = 365 * 5  # 5-year window
    if 'sunspot_number_roll_max_1825' in monthly_df.columns:
        roll_max_col = [c for c in monthly_df.columns if 'roll_max' in c and 'sunspot' in c]
        roll_min_col = [c for c in monthly_df.columns if 'roll_min' in c and 'sunspot' in c]
        if roll_max_col and roll_min_col:
            axes[1].fill_between(monthly_df['date'], 
                               monthly_df[roll_min_col[0]], 
                               monthly_df[roll_max_col[0]], 
                               alpha=0.4, color='blue', label='5-Year Envelope')
    
    axes[1].plot(monthly_df['date'], monthly_df['sunspot_number'], 
                linewidth=1.5, color='darkblue', label='Monthly Average')
    axes[1].set_ylabel('Sunspot Number')
    axes[1].set_title('Solar Cycle Envelope and Long-term Modulation', fontsize=16, pad=20)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # Plot 3: Modern era with multivariate data (F10.7 available from ~1957)
    modern_df = monthly_df[monthly_df['date'] >= '1950-01-01']
    if not modern_df.empty:
        ax3_twin = axes[2].twinx()
        
        line1 = axes[2].plot(modern_df['date'], modern_df['sunspot_number'], 
                           linewidth=2, color='red', label='Sunspot Number')
        
        if 'f107_F10.7_ADJ' in modern_df.columns:
            line2 = ax3_twin.plot(modern_df['date'], modern_df['f107_F10.7_ADJ'], 
                                linewidth=2, color='blue', alpha=0.7, label='F10.7 Flux')
            ax3_twin.set_ylabel('F10.7 Solar Flux (SFU)', color='blue')
        
        axes[2].set_ylabel('Sunspot Number', color='red')
        axes[2].set_title('Space Age: Sunspot Numbers vs F10.7 Solar Flux (1950-2025)', fontsize=16, pad=20)
        
        # Combine legends
        lines1, labels1 = axes[2].get_legend_handles_labels()
        if 'f107_F10.7_ADJ' in modern_df.columns:
            lines2, labels2 = ax3_twin.get_legend_handles_labels()
            axes[2].legend(lines1 + lines2, labels1 + labels2, loc='upper left')
        else:
            axes[2].legend()
        axes[2].grid(True, alpha=0.3)
    
    # Plot 4: Recent detailed activity (last 10 years)
    recent_df = datasets['recent']
    if not recent_df.empty:
        axes[3].plot(recent_df['date'], recent_df['sunspot_number'], 
                    linewidth=0.8, alpha=0.7, color='red', label='Daily Values')
        
        # Add 30-day rolling average if available
        if 'sunspot_number_roll_mean_30' in recent_df.columns:
            axes[3].plot(recent_df['date'], recent_df['sunspot_number_roll_mean_30'], 
                        linewidth=2, color='darkred', label='30-day Average')
        
        axes[3].set_ylabel('Sunspot Number')
        axes[3].set_xlabel('Date')
        axes[3].set_title('Recent Solar Activity: Daily Resolution (Last 10 Years)', fontsize=16, pad=20)
        axes[3].legend()
        axes[3].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_dir / "01_historical_overview.png", dpi=300, bbox_inches='tight')
    plt.show()
    print("Historical overview saved!")

def plot_solar_cycles_analysis(df, datasets, save_dir):
    """Analyze solar cycle patterns across centuries."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    monthly_df = datasets['monthly']
    
    # Plot 1: Solar cycle phase distribution
    if 'solar_cycle_phase' in monthly_df.columns:
        axes[0, 0].hist(monthly_df['solar_cycle_phase'], bins=50, alpha=0.7, color='purple', edgecolor='black')
        axes[0, 0].set_xlabel('Solar Cycle Phase')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].set_title('Distribution of Solar Cycle Phases (275+ Years)')
        axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Sunspot activity by decade
    monthly_df['decade'] = (monthly_df['date'].dt.year // 10) * 10
    decade_stats = monthly_df.groupby('decade')['sunspot_number'].agg(['mean', 'std', 'max']).reset_index()
    
    axes[0, 1].errorbar(decade_stats['decade'], decade_stats['mean'], 
                       yerr=decade_stats['std'], capsize=3, marker='o', linewidth=1.5)
    axes[0, 1].set_xlabel('Decade')
    axes[0, 1].set_ylabel('Average Sunspot Number')
    axes[0, 1].set_title('Solar Activity by Decade (with Standard Deviation)')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].tick_params(axis='x', rotation=45)
    
    # Plot 3: Seasonal patterns
    monthly_df['month'] = monthly_df['date'].dt.month
    seasonal = monthly_df.groupby('month')['sunspot_number'].mean()
    axes[1, 0].plot(seasonal.index, seasonal.values, marker='o', linewidth=2, markersize=8)
    axes[1, 0].set_xlabel('Month')
    axes[1, 0].set_ylabel('Average Sunspot Number')
    axes[1, 0].set_title('Seasonal Patterns in Solar Activity')
    axes[1, 0].set_xticks(range(1, 13))
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Activity regime durations
    if any(col.endswith('_regime_duration') for col in monthly_df.columns):
        regime_cols = [col for col in monthly_df.columns if col.endswith('_regime_duration')]
        if regime_cols:
            axes[1, 1].hist(monthly_df[regime_cols[0]].dropna(), bins=50, alpha=0.7, color='green', edgecolor='black')
            axes[1, 1].set_xlabel('Regime Duration (months)')
            axes[1, 1].set_ylabel('Frequency')
            axes[1, 1].set_title('Solar Activity Regime Persistence')
            axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_dir / "02_solar_cycles_analysis.png", dpi=300, bbox_inches='tight')
    plt.show()
    print("Solar cycles analysis saved!")

def plot_multivariate_correlations(datasets, save_dir):
    """Create correlation analysis for multivariate features."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    sample_df = datasets['sample']
    
    # Select key variables for correlation
    key_vars = ['sunspot_number']
    
    # Add available multivariate features
    for col in sample_df.columns:
        if any(x in col.lower() for x in ['f107_adj', 'ap_avg', 'kp_sum']):
            key_vars.append(col)
            if len(key_vars) >= 6:  # Limit for readability
                break
    
    # Add some engineered features
    eng_vars = [col for col in sample_df.columns if any(x in col for x in ['roll_mean_30', 'volatility_30'])]
    key_vars.extend(eng_vars[:3])
    
    # Filter to existing columns and numeric data
    available_vars = [var for var in key_vars if var in sample_df.columns]
    if len(available_vars) > 1:
        corr_data = sample_df[available_vars].select_dtypes(include=[np.number])
        
        if len(corr_data.columns) > 1:
            # Correlation heatmap
            corr_matrix = corr_data.corr()
            mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
            
            sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='RdBu_r', center=0,
                       square=True, linewidths=0.5, ax=axes[0], fmt='.2f')
            axes[0].set_title('Correlation Matrix: Key Solar Variables')
    
    # Scatter plot: Sunspot vs F10.7 (if available)
    if 'f107_F10.7_ADJ' in sample_df.columns:
        scatter_data = sample_df[['sunspot_number', 'f107_F10.7_ADJ']].dropna()
        if len(scatter_data) > 100:
            axes[1].scatter(scatter_data['sunspot_number'], scatter_data['f107_F10.7_ADJ'], 
                          alpha=0.6, s=20, color='blue')
            axes[1].set_xlabel('Sunspot Number')
            axes[1].set_ylabel('F10.7 Solar Flux')
            axes[1].set_title('Sunspot-F10.7 Relationship')
            axes[1].grid(True, alpha=0.3)
            
            # Add correlation coefficient
            corr_coef = scatter_data['sunspot_number'].corr(scatter_data['f107_F10.7_ADJ'])
            axes[1].text(0.05, 0.95, f'r = {corr_coef:.3f}', transform=axes[1].transAxes,
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(save_dir / "03_multivariate_correlations.png", dpi=300, bbox_inches='tight')
    plt.show()
    print("Multivariate correlations saved!")

def generate_data_summary(df, datasets, save_dir):
    """Generate comprehensive data summary."""
    summary_path = save_dir / "data_summary_full_history.txt"
    
    with open(summary_path, 'w') as f:
        f.write("COMPREHENSIVE MULTIVARIATE SOLAR DATA SUMMARY\n")
        f.write("=" * 60 + "\n\n")
        
        f.write(f"FULL DATASET:\n")
        f.write(f"Shape: {df.shape}\n")
        f.write(f"Date range: {df['date'].min()} to {df['date'].max()}\n")
        f.write(f"Time span: {(df['date'].max() - df['date'].min()).days / 365.25:.1f} years\n")
        f.write(f"Total features: {len(df.columns)}\n\n")
        
        # Historical periods
        f.write("HISTORICAL PERIODS:\n")
        f.write("-" * 30 + "\n")
        periods = [
            ("Pre-Space Age", "1818-1956", df[df['date'] < '1957-01-01']),
            ("Space Age", "1957-1999", df[(df['date'] >= '1957-01-01') & (df['date'] < '2000-01-01')]),
            ("Modern Era", "2000-2025", df[df['date'] >= '2000-01-01'])
        ]
        
        for period_name, period_range, period_data in periods:
            if not period_data.empty:
                f.write(f"\n{period_name} ({period_range}):\n")
                f.write(f"  Observations: {len(period_data)}\n")
                f.write(f"  Mean sunspot number: {period_data['sunspot_number'].mean():.1f}\n")
                f.write(f"  Max sunspot number: {period_data['sunspot_number'].max():.0f}\n")
                f.write(f"  Solar cycles: ~{len(period_data) / (365.25 * 11):.1f}\n")
        
        # Key statistics
        f.write(f"\n\nKEY VARIABLE STATISTICS (Full History):\n")
        f.write("-" * 40 + "\n")
        
        key_vars = ['sunspot_number', 'f107_F10.7_ADJ', 'ap_avg', 'kp_sum']
        for var in key_vars:
            if var in df.columns:
                stats = df[var].describe()
                f.write(f"\n{var}:\n")
                f.write(f"  Count: {stats['count']:.0f}\n")
                f.write(f"  Mean: {stats['mean']:.2f}\n")
                f.write(f"  Std: {stats['std']:.2f}\n")
                f.write(f"  Min: {stats['min']:.2f}\n")
                f.write(f"  Max: {stats['max']:.2f}\n")
                f.write(f"  Missing: {df[var].isna().sum()} ({df[var].isna().mean()*100:.1f}%)\n")
        
        # Feature engineering summary
        f.write(f"\n\nFEATURE ENGINEERING SUMMARY:\n")
        f.write("-" * 35 + "\n")
        
        categories = {
            'Original Data': len([c for c in df.columns if not any(x in c for x in ['roll', 'lag', 'diff', 'volatility', 'regime', 'sin', 'cos', 'pct'])]),
            'Rolling Statistics': len([c for c in df.columns if 'roll' in c]),
            'Lag Features': len([c for c in df.columns if 'lag' in c]),
            'Temporal Features': len([c for c in df.columns if any(x in c for x in ['sin', 'cos', 'cycle', 'month', 'year'])]),
            'Volatility Features': len([c for c in df.columns if 'volatility' in c]),
            'Regime Features': len([c for c in df.columns if 'regime' in c]),
            'Difference Features': len([c for c in df.columns if any(x in c for x in ['diff', 'pct_change', 'roc'])])
        }
        
        for category, count in categories.items():
            f.write(f"{category}: {count} features\n")
        
        f.write(f"\nTotal engineered features: {sum(categories.values())}\n")
        
        # Solar cycle information
        f.write(f"\n\nSOLAR CYCLE INFORMATION:\n")
        f.write("-" * 30 + "\n")
        f.write(f"Estimated solar cycles covered: ~{len(df) / (365.25 * 11):.1f}\n")
        f.write(f"Average cycle length: 11 years (4017 days)\n")
        f.write(f"Data density: {len(df) / ((df['date'].max() - df['date'].min()).days):.3f} obs/day\n")
        
    print(f"Comprehensive data summary saved to: {summary_path}")

def main():
    """Generate efficient visualizations for the full historical dataset.""" 
    # Setup
    data_path = "data/engineered_multivariate_data.csv"
    save_dir = Path("data/visualizations")
    save_dir.mkdir(exist_ok=True, parents=True)
    
    if not Path(data_path).exists():
        print(f"Data file not found: {data_path}")
        print("Run data collection and preprocessing first!")
        return
    
    # Load and sample data efficiently
    df, datasets = load_and_sample_data(data_path, sample_size=15000)
    
    print("\n" + "="*60)
    print("GENERATING COMPREHENSIVE HISTORICAL VISUALIZATIONS")
    print("="*60)
    
    # Generate visualizations
    print("\n1. Creating historical overview...")
    plot_historical_overview(df, datasets, save_dir)
    
    print("\n2. Creating solar cycles analysis...")
    plot_solar_cycles_analysis(df, datasets, save_dir)
    
    print("\n3. Creating multivariate correlations...")
    plot_multivariate_correlations(datasets, save_dir)
    
    print("\n4. Generating comprehensive data summary...")
    generate_data_summary(df, datasets, save_dir)
    
    print(f"\n" + "="*60)
    print(f"ALL VISUALIZATIONS COMPLETE!")
    print(f"Saved to: {save_dir}")
    print(f"Dataset: {df.shape[0]:,} observations, {df.shape[1]} features")
    print(f"Time span: {(df['date'].max() - df['date'].min()).days / 365.25:.1f} years")
    print("="*60)

if __name__ == "__main__":
    main()