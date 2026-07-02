"""
Train WaveNet+LSTM model on real solar data to recreate paper results.
Uses actual sunspot data from 1749-2019 to predict Solar Cycle 25 (11 years).
"""

import pandas as pd
import numpy as np
from pathlib import Path
from wavenet_lstm_model import SolarCycleTrainer
import matplotlib.pyplot as plt
import json


def load_and_prepare_data():
    """Load and prepare the solar data for training."""
    
    # Try to load the engineered data first, fallback to raw data
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


def create_solar_cycle_info():
    """Create solar cycle timing information for backtesting."""
    
    # Solar cycle start dates (approximate months from 1749)
    # These are rough estimates for backtesting purposes
    cycle_info = {
        "20": {"start_month": 2640, "end_month": 2772},  # 1964-1976
        "21": {"start_month": 2772, "end_month": 2904},  # 1976-1986  
        "22": {"start_month": 2904, "end_month": 3036},  # 1986-1996
        "23": {"start_month": 3036, "end_month": 3180},  # 1996-2008
        "24": {"start_month": 3180, "end_month": 3324},  # 2008-2020
        "25": {"start_month": 3324, "end_month": 3456}   # 2020-2032 (predicted)
    }
    
    return cycle_info


def main():
    """Train WaveNet+LSTM model on full solar dataset with backtesting."""
    
    print("=" * 70)
    print("TRAINING WAVENET+LSTM MODEL FOR SOLAR CYCLE 25 PREDICTION")
    print("Based on: Forecasting Solar Cycle 25 using Deep Neural Networks")
    print("With Historical Cycle Validation")
    print("=" * 70)
    
    # Load data
    df = load_and_prepare_data()
    
    # Initialize trainer
    trainer = SolarCycleTrainer()
    
    # Create solar cycle information for backtesting
    cycle_info = create_solar_cycle_info()
    
    # Prepare monthly averaged data (as in paper)
    print("\nPreparing monthly averaged sunspot data...")
    monthly_data = trainer.prepare_monthly_data(
        df, 
        target_col='sunspot_number',
        start_year=1749  # Full historical record as in paper
    )
    
    print(f"Monthly data shape: {monthly_data.shape}")
    print(f"Years of data: {len(monthly_data) / 12:.1f}")
    print(f"Data range: {monthly_data.min():.1f} to {monthly_data.max():.1f}")
    
    # Train model with paper parameters
    print("\n" + "=" * 50)
    print("TRAINING MODEL")
    print("=" * 50)
    
    results = trainer.train(
        data=monthly_data,
        epochs=50,               # Reduced for faster initial training
        batch_size=16,           # Smaller batch size
        lr=1e-3,                 # Learning rate from paper
        patience=15              # Early stopping patience
    )
    
    # Generate Solar Cycle 25 predictions
    print("\n" + "=" * 50) 
    print("GENERATING SOLAR CYCLE 25 PREDICTIONS")
    print("=" * 50)
    
    cycle_25_predictions = trainer.predict_solar_cycle_25(monthly_data)
    
    # Print results in paper format
    print(f"\nSOLAR CYCLE 25 FORECAST RESULTS:")
    print("-" * 40)
    print(f"Maximum sunspot number: {cycle_25_predictions['max_sunspot_number']:.1f}")
    print(f"Peak occurs at month: {cycle_25_predictions['max_sunspot_month']}")
    print(f"Peak occurs at year: {cycle_25_predictions['max_sunspot_month']/12:.1f} into cycle")
    print(f"Forecast duration: {cycle_25_predictions['prediction_years']:.1f} years")
    
    # Calculate uncertainty (simple std of recent predictions)
    recent_std = np.std(monthly_data[-120:])  # Last 10 years
    uncertainty = recent_std * 0.2  # Rough uncertainty estimate
    print(f"Estimated uncertainty: ± {uncertainty:.1f}")
    
    # Compare with paper results
    paper_max = 106  # From paper
    paper_uncertainty = 19.75  # From paper
    
    print(f"\nCOMPARISON WITH PAPER:")
    print("-" * 25)
    print(f"Paper prediction: {paper_max} ± {paper_uncertainty}")
    print(f"Our prediction:   {cycle_25_predictions['max_sunspot_number']:.1f} ± {uncertainty:.1f}")
    print(f"Difference: {abs(cycle_25_predictions['max_sunspot_number'] - paper_max):.1f}")
    
    # Run historical backtesting
    print("\n" + "=" * 50)
    print("HISTORICAL CYCLE BACKTESTING")
    print("=" * 50)
    
    backtest_results = trainer.backtest_historical_cycles(monthly_data, cycle_info)
    
    # Calculate average backtesting performance
    if backtest_results:
        avg_rmse = np.mean([r['rmse'] for r in backtest_results.values()])
        avg_peak_error = np.mean([r['peak_error'] for r in backtest_results.values()])
        
        print(f"\nBacktesting Summary:")
        print(f"  Average RMSE: {avg_rmse:.2f}")
        print(f"  Average Peak Error: {avg_peak_error:.2f}")
        print(f"  Cycles tested: {list(backtest_results.keys())}")
    
    # Evaluate model performance on test set
    print("\n" + "=" * 50)
    print("MODEL EVALUATION ON TEST SET")
    print("=" * 50)
    
    evaluation = trainer.evaluate(results['data_info'])
    
    print(f"Test Set Performance Metrics:")
    print(f"  RMSE: {evaluation['rmse']:.2f}")
    print(f"  MAE:  {evaluation['mae']:.2f}")
    print(f"  R²:   {evaluation['r2']:.4f}")
    
    # Save results
    results_dir = Path("data/paper_results")
    results_dir.mkdir(exist_ok=True, parents=True)
    
    # Save model predictions
    results_summary = {
        'model': 'WaveNet+LSTM',
        'training_data_years': len(monthly_data) / 12,
        'prediction_horizon_years': cycle_25_predictions['prediction_years'],
        'max_sunspot_prediction': float(cycle_25_predictions['max_sunspot_number']),
        'max_sunspot_month': int(cycle_25_predictions['max_sunspot_month']),
        'uncertainty_estimate': float(uncertainty),
        'paper_comparison': {
            'paper_max': paper_max,
            'paper_uncertainty': paper_uncertainty,
            'difference': float(abs(cycle_25_predictions['max_sunspot_number'] - paper_max))
        },
        'model_metrics': {
            'test_rmse': float(evaluation['rmse']),
            'test_mae': float(evaluation['mae']),
            'test_r2': float(evaluation['r2'])
        },
        'backtesting_metrics': {
            'avg_rmse': float(avg_rmse) if backtest_results else None,
            'avg_peak_error': float(avg_peak_error) if backtest_results else None,
            'cycles_tested': list(backtest_results.keys()) if backtest_results else []
        }
    }
    
    with open(results_dir / "solar_cycle_25_predictions.json", 'w') as f:
        json.dump(results_summary, f, indent=2)
    
    # Save full predictions array
    np.save(results_dir / "cycle_25_monthly_predictions.npy", cycle_25_predictions['prediction'])
    
    print(f"\nResults saved to: {results_dir}")
    
    # Create visualizations
    print("\nGenerating visualizations...")
    
    # Plot training history
    trainer.plot_training_history(
        save_path=results_dir / "training_history.png"
    )
    
    # Plot historical backtesting results
    if backtest_results:
        trainer.plot_backtest_results(
            backtest_results,
            save_path=results_dir / "historical_cycle_backtesting.png"
        )
    
    # Plot Solar Cycle 25 predictions
    trainer.plot_predictions(
        cycle_25_predictions,
        save_path=results_dir / "solar_cycle_25_forecast.png"
    )
    
    # Plot historical data with predictions
    plt.figure(figsize=(15, 8))
    
    # Plot last 20 years of historical data
    recent_years = 20
    recent_months = recent_years * 12
    recent_data = monthly_data[-recent_months:]
    recent_time = np.arange(-recent_months, 0) / 12
    
    # Plot predictions
    pred_time = np.arange(0, len(cycle_25_predictions['prediction'])) / 12
    
    plt.plot(recent_time, recent_data, 'b-', linewidth=2, label='Historical Data')
    plt.plot(pred_time, cycle_25_predictions['prediction'], 'r-', linewidth=2, 
             label='Solar Cycle 25 Prediction')
    
    # Mark the maximum
    max_idx = cycle_25_predictions['max_sunspot_month'] - 1
    plt.axvline(x=max_idx/12, color='orange', linestyle='--', alpha=0.7)
    plt.scatter(max_idx/12, cycle_25_predictions['max_sunspot_number'], 
                color='red', s=100, zorder=5)
    
    plt.axvline(x=0, color='black', linestyle='-', alpha=0.5, label='Prediction Start')
    plt.xlabel('Years (relative to prediction start)')
    plt.ylabel('Sunspot Number')
    plt.title('Solar Cycle 25 Prediction - WaveNet+LSTM Model\n(Historical Context)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(results_dir / "cycle_25_with_context.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"\nTraining complete!")
    print(f"Solar Cycle 25 maximum predicted: {cycle_25_predictions['max_sunspot_number']:.1f}")
    print(f"All results saved to: {results_dir}")


if __name__ == "__main__":
    main()