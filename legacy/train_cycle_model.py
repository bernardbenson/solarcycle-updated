"""
Train cycle-based WaveNet+LSTM model matching the paper methodology.
Uses 4 solar cycles as input to predict 1 solar cycle as output.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from cycle_based_model import CycleBasedTrainer
import matplotlib.pyplot as plt
import json


def load_and_prepare_data():
    """Load and prepare the solar data for cycle-based training."""
    
    data_dir = Path("data")
    
    engineered_path = data_dir / "engineered_multivariate_data.csv"
    raw_path = data_dir / "raw_multivariate_data.csv"
    
    if engineered_path.exists():
        print("Loading engineered multivariate data...")
        df = pd.read_csv(engineered_path)
    elif raw_path.exists():
        print("Loading raw multivariate data...")
        df = pd.read_csv(raw_path)
    else:
        raise FileNotFoundError("No solar data found. Please run data collection first.")
    
    print(f"Loaded data shape: {df.shape}")
    print(f"Date range: {df['date'].min()} to {df['date'].max()}")
    
    return df


def prepare_monthly_data(df: pd.DataFrame, target_col: str = 'sunspot_number',
                        start_year: int = 1749) -> np.ndarray:
    """Prepare monthly averaged data as in the paper."""
    
    print("Preparing monthly averaged sunspot data...")
    
    # Convert to monthly averages
    df['date'] = pd.to_datetime(df['date'])
    df = df.set_index('date')
    
    # Filter from start year
    df = df[df.index.year >= start_year]
    
    # Resample to monthly averages
    monthly_data = df[target_col].resample('M').mean()
    
    # Remove NaN values
    monthly_data = monthly_data.dropna()
    
    print(f"Monthly data shape: {monthly_data.shape}")
    print(f"Date range: {monthly_data.index[0]} to {monthly_data.index[-1]}")
    print(f"Years of data: {len(monthly_data) / 12:.1f}")
    
    return monthly_data.values


def main():
    """Train cycle-based WaveNet+LSTM model on full solar dataset."""
    
    print("=" * 80)
    print("CYCLE-BASED WAVENET+LSTM MODEL FOR SOLAR CYCLE 25 PREDICTION")
    print("Paper Methodology: 4 Solar Cycles → 1 Solar Cycle Prediction")
    print("=" * 80)
    
    # Load data
    df = load_and_prepare_data()
    
    # Prepare monthly averaged data
    monthly_data = prepare_monthly_data(df, target_col='sunspot_number', start_year=1749)
    
    print(f"\nMonthly data shape: {monthly_data.shape}")
    print(f"Years of data: {len(monthly_data) / 12:.1f}")
    print(f"Data range: {monthly_data.min():.1f} to {monthly_data.max():.1f}")
    
    # Initialize cycle-based trainer
    trainer = CycleBasedTrainer()
    
    # Train model with cycle-based methodology
    print("\n" + "=" * 60)
    print("TRAINING CYCLE-BASED MODEL")
    print("=" * 60)
    
    try:
        results = trainer.train(
            monthly_data=monthly_data,
            epochs=100,              # Fewer epochs due to limited cycle data
            batch_size=4,            # Small batch size for cycle-based training
            lr=1e-3,                 
            patience=20              
        )
        
        print(f"\nTraining Summary:")
        print(f"  Total cycles available: {results['cycle_info']['n_cycles']}")
        print(f"  Training samples: {results['dataset_info']['train_samples']}")
        print(f"  Test samples: {results['dataset_info']['test_samples']}")
        print(f"  Best test loss: {results['best_test_loss']:.6f}")
        
    except Exception as e:
        print(f"Training failed: {e}")
        print("This might be due to insufficient solar cycle data.")
        print("The cycle-based approach requires at least 5 complete solar cycles.")
        return
    
    # Generate Solar Cycle 25 predictions
    print("\n" + "=" * 60) 
    print("GENERATING SOLAR CYCLE 25 PREDICTIONS")
    print("=" * 60)
    
    try:
        cycle_25_predictions = trainer.predict_solar_cycle_25(monthly_data)
        
        # Print results in paper format
        print(f"\nSOLAR CYCLE 25 FORECAST RESULTS:")
        print("-" * 50)
        print(f"Maximum sunspot number: {cycle_25_predictions['max_sunspot_number']:.1f}")
        print(f"Peak occurs at month: {cycle_25_predictions['max_sunspot_month']}")
        print(f"Peak occurs at year: {cycle_25_predictions['max_sunspot_month']/12:.1f} into cycle")
        print(f"Forecast duration: {cycle_25_predictions['prediction_years']:.1f} years")
        print(f"Input cycles used: {cycle_25_predictions['input_cycles']}")
        
        # Compare with paper results
        paper_max = 106  # From paper
        paper_uncertainty = 19.75  # From paper
        
        print(f"\nCOMPARISON WITH PAPER:")
        print("-" * 30)
        print(f"Paper prediction: {paper_max} ± {paper_uncertainty}")
        print(f"Our prediction:   {cycle_25_predictions['max_sunspot_number']:.1f}")
        print(f"Difference: {abs(cycle_25_predictions['max_sunspot_number'] - paper_max):.1f}")
        
        # Calculate relative difference
        rel_diff = abs(cycle_25_predictions['max_sunspot_number'] - paper_max) / paper_max * 100
        print(f"Relative difference: {rel_diff:.1f}%")
        
        if rel_diff <= 20:
            print("✅ Good agreement with paper results!")
        elif rel_diff <= 40:
            print("⚠️  Moderate agreement with paper results")
        else:
            print("❌ Significant difference from paper results")
        
    except Exception as e:
        print(f"Prediction failed: {e}")
        return
    
    # Save results
    results_dir = Path("data/cycle_based_results")
    results_dir.mkdir(exist_ok=True, parents=True)
    
    # Save comprehensive results
    results_summary = {
        'model_type': 'Cycle-Based WaveNet+LSTM',
        'methodology': '4 solar cycles → 1 solar cycle prediction',
        'training_data_years': len(monthly_data) / 12,
        'total_cycles_available': results['cycle_info']['n_cycles'],
        'training_samples': results['dataset_info']['train_samples'],
        'test_samples': results['dataset_info']['test_samples'],
        'best_test_loss': float(results['best_test_loss']),
        'solar_cycle_25_prediction': {
            'max_sunspot_number': float(cycle_25_predictions['max_sunspot_number']),
            'max_sunspot_month': int(cycle_25_predictions['max_sunspot_month']),
            'prediction_years': float(cycle_25_predictions['prediction_years']),
            'input_cycles_used': int(cycle_25_predictions['input_cycles'])
        },
        'paper_comparison': {
            'paper_max': paper_max,
            'paper_uncertainty': paper_uncertainty,
            'our_prediction': float(cycle_25_predictions['max_sunspot_number']),
            'absolute_difference': float(abs(cycle_25_predictions['max_sunspot_number'] - paper_max)),
            'relative_difference_percent': float(rel_diff)
        }
    }
    
    with open(results_dir / "cycle_based_predictions.json", 'w') as f:
        json.dump(results_summary, f, indent=2)
    
    # Save predictions array
    np.save(results_dir / "cycle_25_predictions.npy", cycle_25_predictions['prediction'])
    
    print(f"\nResults saved to: {results_dir}")
    
    # Create visualizations
    print("\nGenerating visualizations...")
    
    # Plot training history
    trainer.plot_training_history(
        save_path=results_dir / "training_history.png"
    )
    
    # Plot Solar Cycle 25 predictions
    trainer.plot_cycle_prediction(
        cycle_25_predictions,
        save_path=results_dir / "solar_cycle_25_forecast.png"
    )
    
    # Create cycle comparison plot
    plt.figure(figsize=(15, 10))
    
    # Try to visualize the segmented cycles
    try:
        cycles, cycle_info = trainer.prepare_cycle_data(monthly_data)
        
        # Plot the last few cycles and prediction
        n_cycles_to_show = min(6, len(cycles))
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        # Show last cycles used for training
        for i in range(n_cycles_to_show - 1):
            cycle_idx = len(cycles) - n_cycles_to_show + i
            if cycle_idx >= 0:
                cycle_data = trainer.scaler.inverse_transform(
                    cycles[cycle_idx].reshape(-1, 1)
                ).ravel()
                
                months = np.arange(len(cycle_data))
                axes[i].plot(months, cycle_data, 'b-', linewidth=2)
                axes[i].set_title(f'Solar Cycle {20 + cycle_idx} (Historical)')
                axes[i].set_xlabel('Months into Cycle')
                axes[i].set_ylabel('Sunspot Number')
                axes[i].grid(True, alpha=0.3)
        
        # Show prediction
        pred_months = np.arange(len(cycle_25_predictions['prediction']))
        axes[-1].plot(pred_months, cycle_25_predictions['prediction'], 'r-', linewidth=2)
        axes[-1].axhline(y=cycle_25_predictions['max_sunspot_number'], 
                        color='orange', linestyle='--', alpha=0.7)
        axes[-1].set_title('Solar Cycle 25 (Predicted)')
        axes[-1].set_xlabel('Months into Cycle')
        axes[-1].set_ylabel('Sunspot Number')
        axes[-1].grid(True, alpha=0.3)
        
        plt.suptitle('Solar Cycles: Historical Context and Cycle 25 Prediction', fontsize=16)
        plt.tight_layout()
        plt.savefig(results_dir / "cycle_context_comparison.png", dpi=300, bbox_inches='tight')
        plt.show()
        
    except Exception as e:
        print(f"Could not create cycle comparison plot: {e}")
    
    print(f"\n🌞 TRAINING COMPLETE!")
    print(f"Solar Cycle 25 maximum predicted: {cycle_25_predictions['max_sunspot_number']:.1f}")
    print(f"Agreement with paper: {100-rel_diff:.1f}% similarity")
    print(f"All results saved to: {results_dir}")


if __name__ == "__main__":
    main()