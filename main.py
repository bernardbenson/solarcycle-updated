"""
Main entry point for the multivariate sunspot prediction project.
This updated version uses actual (non-smoothed) sunspot numbers and PyTorch.
"""

import argparse
import sys
import pandas as pd
import numpy as np
from pathlib import Path
import json

from data_collection import SolarDataCollector


def collect_data(args):
    """Collect and prepare multivariate solar data."""
    print("=== Data Collection Phase ===")
    
    collector = SolarDataCollector()
    dataset = collector.create_multivariate_dataset(
        start_year=args.start_year,
        end_year=args.end_year
    )
    
    # Save raw dataset
    output_path = Path(args.output_dir) / "raw_multivariate_data.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_csv(output_path, index=False)
    
    print(f"\nRaw dataset saved to: {output_path}")
    print(f"Shape: {dataset.shape}")
    print(f"Date range: {dataset['date'].min()} to {dataset['date'].max()}")
    
    return dataset


def preprocess_data(args):
    """Preprocess and engineer features from raw data."""
    print("=== Data Preprocessing Phase ===")
    
    # Load raw data
    raw_data_path = Path(args.output_dir) / "raw_multivariate_data.csv"
    if not raw_data_path.exists():
        print("Raw data not found. Run data collection first.")
        return None
    
    from feature_engineering import SolarFeatureEngineer
    
    # Load and preprocess data
    print("Loading raw multivariate data...")
    df = pd.read_csv(raw_data_path)
    df['date'] = pd.to_datetime(df['date'])
    
    print(f"Raw data shape: {df.shape}")
    print(f"Date range: {df['date'].min()} to {df['date'].max()}")
    
    # Initialize feature engineer
    engineer = SolarFeatureEngineer()
    
    # Apply comprehensive feature engineering
    print("\nApplying feature engineering...")
    df_engineered = engineer.engineer_all_features(df, target_col='sunspot_number')
    
    # Remove rows with too many missing values (from lag features)
    print("Cleaning data and handling missing values...")
    initial_rows = len(df_engineered)
    df_engineered = df_engineered.dropna(subset=['sunspot_number'])  # Keep target intact
    
    # For other features, use forward fill followed by backward fill
    numeric_cols = df_engineered.select_dtypes(include=[np.number]).columns
    df_engineered[numeric_cols] = df_engineered[numeric_cols].ffill().bfill()
    
    final_rows = len(df_engineered)
    print(f"Removed {initial_rows - final_rows} rows with missing target values")
    
    # Save engineered dataset
    engineered_path = Path(args.output_dir) / "engineered_multivariate_data.csv"
    df_engineered.to_csv(engineered_path, index=False)
    
    print(f"\nFeature engineering complete!")
    print(f"Original features: {df.shape[1]}")
    print(f"Engineered features: {df_engineered.shape[1]}")
    print(f"Added {df_engineered.shape[1] - df.shape[1]} new features")
    print(f"Final dataset shape: {df_engineered.shape}")
    print(f"Engineered dataset saved to: {engineered_path}")
    
    return df_engineered


def train_models(args):
    """Train PyTorch models on processed data."""
    print("=== Model Training Phase ===")
    
    # Check for engineered data
    engineered_path = Path(args.output_dir) / "engineered_multivariate_data.csv"
    if not engineered_path.exists():
        print("Engineered data not found. Run preprocessing first.")
        return None
    
    from pytorch_models import (
        ModelTrainer, SolarTransformerModel, AttentionLSTMModel, 
        AttentionGRUModel, EnsembleModel
    )
    
    # Load engineered data
    print("Loading engineered multivariate data...")
    df = pd.read_csv(engineered_path)
    df['date'] = pd.to_datetime(df['date'])
    
    print(f"Training data shape: {df.shape}")
    print(f"Date range: {df['date'].min()} to {df['date'].max()}")
    
    # Initialize trainer
    trainer = ModelTrainer(device='auto')
    
    # Prepare data for PyTorch training
    print("\nPreparing data for PyTorch models...")
    data_info = trainer.prepare_data(
        df, 
        target_col='sunspot_number',
        sequence_length=365,  # 1 year of context
        prediction_horizon=30,  # Predict 30 days ahead
        train_ratio=0.7,
        val_ratio=0.15
        # test_ratio will be 0.15
    )
    
    n_features = data_info['n_features']
    print(f"Number of features: {n_features}")
    
    # Training configuration
    training_config = getattr(args, 'training_config', {
        'batch_size': 32,
        'epochs': 100,
        'lr': 1e-3,
        'patience': 15
    })
    
    # Initialize models
    models_to_train = {}
    
    print(f"\n{'='*60}")
    print("INITIALIZING PYTORCH MODELS")
    print(f"{'='*60}")
    
    # 1. Transformer Model (PatchTST-style)
    print("1. Initializing Solar Transformer Model...")
    transformer_model = SolarTransformerModel(
        n_features=n_features,
        d_model=128,
        n_heads=8,
        n_layers=6,
        patch_size=16,
        prediction_horizon=30,
        dropout=0.1
    )
    models_to_train['SolarTransformer'] = transformer_model
    
    # 2. LSTM with Attention
    print("2. Initializing LSTM with Attention...")
    lstm_model = AttentionLSTMModel(
        n_features=n_features,
        hidden_size=128,
        n_layers=3,
        prediction_horizon=30,
        dropout=0.2,
        bidirectional=True
    )
    models_to_train['AttentionLSTM'] = lstm_model
    
    # 3. GRU with Attention
    print("3. Initializing GRU with Attention...")
    gru_model = AttentionGRUModel(
        n_features=n_features,
        hidden_size=128,
        n_layers=3,
        prediction_horizon=30,
        dropout=0.2,
        bidirectional=True
    )
    models_to_train['AttentionGRU'] = gru_model
    
    # Train each model
    training_results = {}
    
    for model_name, model in models_to_train.items():
        print(f"\n{'='*60}")
        print(f"TRAINING {model_name.upper()}")
        print(f"{'='*60}")
        
        try:
            # Train the model
            history = trainer.train_model(
                model=model,
                data_info=data_info,
                model_name=model_name,
                batch_size=training_config['batch_size'],
                epochs=training_config['epochs'],
                lr=training_config['lr'],
                patience=training_config['patience']
            )
            
            training_results[model_name] = history
            print(f"✅ {model_name} training completed successfully!")
            
        except Exception as e:
            print(f"❌ Error training {model_name}: {e}")
            continue
    
    # Create ensemble model if multiple models trained successfully
    if len(trainer.models) >= 2:
        print(f"\n{'='*60}")
        print("CREATING ENSEMBLE MODEL")
        print(f"{'='*60}")
        
        trained_models = list(trainer.models.values())
        ensemble_model = EnsembleModel(trained_models)
        
        # Train ensemble (fine-tune the weights)
        try:
            ensemble_history = trainer.train_model(
                model=ensemble_model,
                data_info=data_info,
                model_name='Ensemble',
                batch_size=training_config['batch_size'],
                epochs=training_config['epochs'] // 2,  # Fewer epochs for ensemble
                lr=training_config['lr'] / 10,  # Lower learning rate
                patience=training_config['patience'] // 2
            )
            training_results['Ensemble'] = ensemble_history
            print("✅ Ensemble model created and trained successfully!")
            
        except Exception as e:
            print(f"❌ Error training ensemble: {e}")
    
    # Save training results and model summaries
    results_dir = Path(args.output_dir) / "training_results"
    results_dir.mkdir(exist_ok=True, parents=True)
    
    # Save model information
    model_info = {
        'models_trained': list(trainer.models.keys()),
        'n_features': n_features,
        'feature_names': data_info['feature_names'],
        'sequence_length': data_info['sequence_length'],
        'prediction_horizon': data_info['prediction_horizon'],
        'training_config': training_config,
        'data_splits': {
            'train_samples': len(data_info['train_dataset']),
            'val_samples': len(data_info['val_dataset']),
            'test_samples': len(data_info['test_dataset'])
        }
    }
    
    import json
    with open(results_dir / "model_info.json", 'w') as f:
        json.dump(model_info, f, indent=2)
    
    print(f"\n{'='*60}")
    print("TRAINING PHASE COMPLETE")
    print(f"{'='*60}")
    print(f"Models trained: {len(trainer.models)}")
    print(f"✅ {', '.join(trainer.models.keys())}")
    print(f"Results saved to: {results_dir}")
    
    return {
        'trainer': trainer,
        'data_info': data_info,
        'training_results': training_results,
        'model_info': model_info
    }


def visualize_data(args):
    """Create comprehensive visualizations of multivariate data."""
    print("=== Data Visualization Phase ===")
    
    # Check for engineered data first
    engineered_path = Path(args.output_dir) / "engineered_multivariate_data.csv"
    if engineered_path.exists():
        data_path = engineered_path
        print("Using engineered multivariate data for visualization...")
    else:
        # Fall back to raw data
        raw_path = Path(args.output_dir) / "raw_multivariate_data.csv"
        if raw_path.exists():
            data_path = raw_path
            print("Using raw multivariate data for visualization...")
        else:
            print("No data found. Run data collection first.")
            return None
    
    from visualization import SolarDataVisualizer
    
    # Load data
    print(f"Loading data from: {data_path}")
    df = pd.read_csv(data_path)
    
    print(f"Dataset shape: {df.shape}")
    print(f"Date range: {df['date'].min()} to {df['date'].max()}")
    
    # Initialize visualizer
    visualizer = SolarDataVisualizer()
    
    # Create visualization output directory
    viz_dir = Path(args.output_dir) / "visualizations"
    
    # Generate all visualizations
    visualizer.generate_all_visualizations(df, output_dir=str(viz_dir))
    
    return df


def evaluate_models(args):
    """Evaluate trained models and generate results."""
    print("=== Model Evaluation Phase ===")
    print("Model evaluation will be implemented next...")
    return None


def main():
    parser = argparse.ArgumentParser(description="Multivariate Sunspot Prediction with PyTorch")
    parser.add_argument("--phase", choices=["collect", "preprocess", "visualize", "train", "evaluate", "all"],
                       default="all", help="Which phase to run")
    parser.add_argument("--start-year", type=int, default=1749,
                       help="Starting year for data collection (1749 = full sunspot history)")
    parser.add_argument("--end-year", type=int, default=None,
                       help="Ending year for data collection (None for current)")
    parser.add_argument("--output-dir", type=str, default="./data",
                       help="Output directory for data and results")
    
    args = parser.parse_args()
    
    print("🌞 Multivariate Sunspot Prediction with Actual (Non-Smoothed) Data")
    print("=" * 60)
    
    try:
        if args.phase in ["collect", "all"]:
            collect_data(args)
        
        if args.phase in ["preprocess", "all"]:
            preprocess_data(args)
        
        if args.phase in ["visualize", "all"]:
            visualize_data(args)
        
        if args.phase in ["train", "all"]:
            train_models(args)
        
        if args.phase in ["evaluate", "all"]:
            evaluate_models(args)
            
    except KeyboardInterrupt:
        print("\n\nProcess interrupted by user.")
    except Exception as e:
        print(f"\nError: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
