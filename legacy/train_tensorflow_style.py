"""
Train TensorFlow-style WaveNet+LSTM model with overlapping prediction plots.
Recreates the exact methodology from the original TensorFlow implementation.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from tensorflow_style_model import TensorFlowStyleTrainer
import matplotlib.pyplot as plt
import json
import datetime
import uuid


def load_and_prepare_data():
    """Load and prepare the solar data."""
    
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
    """Prepare monthly averaged data."""
    
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


def generate_run_id():
    """Generate a unique run ID with timestamp and short UUID."""
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    short_uuid = str(uuid.uuid4())[:8]
    return f"run_{timestamp}_{short_uuid}"


def main():
    """Train TensorFlow-style model and create overlapping predictions."""
    
    # Generate unique run ID
    run_id = generate_run_id()
    
    print("=" * 80)
    print("TENSORFLOW-STYLE WAVENET+LSTM FOR SOLAR CYCLE PREDICTION")
    print("Methodology: 528 months input → 132 months output (sliding window)")
    print("Recreates exact overlapping prediction plots from TensorFlow version")
    print(f"Run ID: {run_id}")
    print("=" * 80)
    
    # Load data
    df = load_and_prepare_data()
    
    # Prepare monthly data
    monthly_data = prepare_monthly_data(df, target_col='sunspot_number', start_year=1749)
    
    print(f"\nMonthly data shape: {monthly_data.shape}")
    print(f"Years of data: {len(monthly_data) / 12:.1f}")
    print(f"Data range: {monthly_data.min():.1f} to {monthly_data.max():.1f}")
    
    # Check if we have enough data
    required_months = 924  # Need at least this much for cycle 22 prediction
    if len(monthly_data) < required_months:
        print(f"Warning: Need at least {required_months} months of data, have {len(monthly_data)}")
        print("Some historical cycle predictions may not be available.")
    
    # Initialize trainer
    trainer = TensorFlowStyleTrainer()
    
    # Train model
    print("\n" + "=" * 60)
    print("TRAINING TENSORFLOW-STYLE MODEL")
    print("=" * 60)
    
    try:
        results = trainer.train(
            monthly_data=monthly_data,
            epochs=150,              # Moderate epochs
            batch_size=32,           # Standard batch size
            lr=1e-3,                 # Standard learning rate
            patience=20              # Standard patience
        )
        
        print(f"\nTraining Summary:")
        print(f"  Training samples: {len(results['data_info']['train_data'])}")
        print(f"  Test samples: {len(results['data_info']['test_data'])}")
        print(f"  Final training loss: {results['history']['train_loss'][-1]:.6f}")
        
    except Exception as e:
        print(f"Training failed: {e}")
        return
    
    # Create overlapping predictions (matching TensorFlow plots)
    print("\n" + "=" * 60)
    print("CREATING OVERLAPPING PREDICTIONS")
    print("=" * 60)
    
    try:
        prediction_results = trainer.create_overlapping_predictions(monthly_data)
        
        # Print results
        for cycle_name, result in prediction_results.items():
            if result['actual'] is not None:
                print(f"{cycle_name}: RMSE={result.get('rmse', 0):.1f}, MAE={result.get('mae', 0):.1f}")
            else:
                max_val = np.max(result['prediction'])
                max_month = np.argmax(result['prediction']) + 1
                print(f"{cycle_name}: Max={max_val:.1f} at month {max_month}")
        
        # Compare Cycle 25 with paper
        if 'cycle_25' in prediction_results:
            cycle_25_max = np.max(prediction_results['cycle_25']['prediction'])
            paper_max = 106  # From paper
            
            print(f"\nSOLAR CYCLE 25 COMPARISON:")
            print("-" * 40)
            print(f"Paper prediction: {paper_max} ± 19.75")
            print(f"Our prediction:   {cycle_25_max:.1f}")
            print(f"Difference: {abs(cycle_25_max - paper_max):.1f}")
            
            rel_diff = abs(cycle_25_max - paper_max) / paper_max * 100
            print(f"Relative difference: {rel_diff:.1f}%")
            
            if rel_diff <= 15:
                print("✅ Excellent agreement with paper!")
            elif rel_diff <= 30:
                print("✅ Good agreement with paper")
            else:
                print("⚠️  Moderate agreement with paper")
        
    except Exception as e:
        print(f"Prediction failed: {e}")
        return
    
    # Save results with unique run ID
    results_dir = Path("data/tensorflow_style_results") / run_id
    results_dir.mkdir(exist_ok=True, parents=True)
    
    # Save comprehensive results
    results_summary = {
        'model_type': 'TensorFlow-Style WaveNet+LSTM',
        'methodology': '528 months input → 132 months output (sliding window)',
        'training_data_years': len(monthly_data) / 12,
        'training_samples': len(results['data_info']['train_data']),
        'test_samples': len(results['data_info']['test_data']),
        'final_training_loss': float(results['history']['train_loss'][-1]),
        'predictions': {}
    }
    
    # Add prediction results
    for cycle_name, result in prediction_results.items():
        pred_summary = {
            'max_sunspot_number': float(np.max(result['prediction'])),
            'max_month': int(np.argmax(result['prediction']) + 1),
            'description': result['description']
        }
        
        if result['actual'] is not None and 'rmse' in result:
            pred_summary['rmse'] = float(result['rmse'])
            pred_summary['mae'] = float(result['mae'])
        
        results_summary['predictions'][cycle_name] = pred_summary
    
    # Paper comparison
    if 'cycle_25' in prediction_results:
        cycle_25_max = np.max(prediction_results['cycle_25']['prediction'])
        results_summary['paper_comparison'] = {
            'paper_max': 106,
            'paper_uncertainty': 19.75,
            'our_prediction': float(cycle_25_max),
            'absolute_difference': float(abs(cycle_25_max - 106)),
            'relative_difference_percent': float(abs(cycle_25_max - 106) / 106 * 100)
        }
    
    with open(results_dir / "tensorflow_style_results.json", 'w') as f:
        json.dump(results_summary, f, indent=2)
    
    # Save individual predictions
    for cycle_name, result in prediction_results.items():
        np.save(results_dir / f"{cycle_name}_prediction.npy", result['prediction'])
        if result['actual'] is not None:
            np.save(results_dir / f"{cycle_name}_actual.npy", result['actual'])
    
    print(f"\nResults saved to: {results_dir}")
    
    # Create visualizations
    print("\nGenerating TensorFlow-style visualizations...")
    
    # Individual plots matching TensorFlow implementation
    trainer.plot_individual_predictions(
        prediction_results,
        save_dir=results_dir
    )
    
    # Training history plot
    plt.figure(figsize=(10, 6))
    plt.plot(results['history']['train_loss'], 'b-', linewidth=2, label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.title('Training History - TensorFlow Style WaveNet+LSTM')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(results_dir / "training_history.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    # Create summary comparison plot
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    
    cycle_names = ['cycle_22', 'cycle_23', 'cycle_24', 'cycle_25']
    
    for i, cycle_name in enumerate(cycle_names):
        if cycle_name not in prediction_results:
            axes[i].set_visible(False)
            continue
        
        result = prediction_results[cycle_name]
        
        # Plot prediction
        pred_months = np.arange(len(result['prediction']))
        axes[i].plot(pred_months, result['prediction'], 'r-', linewidth=2, label='Prediction')
        
        # Plot actual if available
        if result['actual'] is not None:
            actual_months = np.arange(len(result['actual']))
            axes[i].plot(actual_months, result['actual'], 'b-', linewidth=2, label='Actual')
            
            if 'rmse' in result:
                title = f"{result['description']}\nRMSE: {result['rmse']:.1f}"
            else:
                title = result['description']
        else:
            title = result['description']
            max_val = np.max(result['prediction'])
            max_month = np.argmax(result['prediction']) + 1
            title += f"\nMax: {max_val:.1f} at month {max_month}"
        
        axes[i].set_title(title, fontsize=12)
        axes[i].set_xlabel('Months')
        axes[i].set_ylabel('Sunspot Number')
        axes[i].legend()
        axes[i].grid(True, alpha=0.3)
    
    plt.suptitle('TensorFlow-Style Solar Cycle Predictions - All Cycles', fontsize=16)
    plt.tight_layout()
    plt.savefig(results_dir / "all_cycles_summary.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"\n🌞 TENSORFLOW-STYLE TRAINING COMPLETE!")
    print(f"Solar Cycle 25 maximum predicted: {cycle_25_max:.1f}")
    print(f"Agreement with paper: {100-rel_diff:.1f}% similarity")
    print(f"All results and plots saved to: {results_dir}")
    print("\nNow you have overlapping prediction plots matching the TensorFlow implementation!")


if __name__ == "__main__":
    main()