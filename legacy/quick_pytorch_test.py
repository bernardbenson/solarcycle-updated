"""
Quick test of PyTorch models on a subset of data to verify functionality.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from pytorch_models import (
    ModelTrainer, SolarTransformerModel, AttentionLSTMModel, AttentionGRUModel
)

def test_pytorch_models():
    """Test PyTorch models on a smaller subset of data."""
    print("🧪 Testing PyTorch Models on Subset of Data")
    print("=" * 50)
    
    # Load data
    data_path = "data/engineered_multivariate_data.csv"
    if not Path(data_path).exists():
        print(f"Data file not found: {data_path}")
        return
    
    # Load and sample data
    print("Loading data...")
    df = pd.read_csv(data_path, low_memory=False)
    df['date'] = pd.to_datetime(df['date'])
    
    # Use recent data only for testing (last 5 years = ~1825 days)
    recent_data = df.tail(5000).copy().reset_index(drop=True)
    print(f"Using subset: {recent_data.shape} (last 5000 observations)")
    print(f"Date range: {recent_data['date'].min()} to {recent_data['date'].max()}")
    
    # Initialize trainer
    trainer = ModelTrainer(device='cpu')  # Force CPU for testing
    
    # Prepare data
    print("\nPreparing data...")
    data_info = trainer.prepare_data(
        recent_data,
        target_col='sunspot_number',
        sequence_length=90,  # 3 months context
        prediction_horizon=7,  # Predict 1 week ahead
        train_ratio=0.7,
        val_ratio=0.2
    )
    
    n_features = data_info['n_features']
    print(f"Features: {n_features}")
    print(f"Training samples: {len(data_info['train_dataset'])}")
    print(f"Validation samples: {len(data_info['val_dataset'])}")
    print(f"Test samples: {len(data_info['test_dataset'])}")
    
    # Test each model
    models_to_test = {
        'Transformer': SolarTransformerModel(
            n_features=n_features,
            d_model=64,  # Smaller for testing
            n_heads=4,
            n_layers=2,
            patch_size=8,
            prediction_horizon=7,
            dropout=0.1
        ),
        'LSTM': AttentionLSTMModel(
            n_features=n_features,
            hidden_size=64,  # Smaller for testing
            n_layers=2,
            prediction_horizon=7,
            dropout=0.2,
            bidirectional=True
        ),
        'GRU': AttentionGRUModel(
            n_features=n_features,
            hidden_size=64,  # Smaller for testing
            n_layers=2,
            prediction_horizon=7,
            dropout=0.2,
            bidirectional=True
        )
    }
    
    results = {}
    
    for model_name, model in models_to_test.items():
        print(f"\n{'='*40}")
        print(f"Testing {model_name}")
        print(f"{'='*40}")
        
        try:
            # Quick training (few epochs)
            history = trainer.train_model(
                model=model,
                data_info=data_info,
                model_name=model_name,
                batch_size=16,  # Small batch for testing
                epochs=5,  # Just a few epochs
                lr=1e-3,
                patience=3
            )
            
            print(f"✅ {model_name} training successful!")
            
            # Quick evaluation
            eval_results = trainer.evaluate_model(model_name, data_info, batch_size=16)
            print(f"Test RMSE: {eval_results['metrics']['overall']['rmse']:.2f}")
            print(f"Test R²: {eval_results['metrics']['overall']['r2']:.4f}")
            
            results[model_name] = {
                'training_history': history,
                'evaluation': eval_results
            }
            
        except Exception as e:
            print(f"❌ {model_name} failed: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\n{'='*50}")
    print("PYTORCH MODELS TEST COMPLETE")
    print(f"{'='*50}")
    print(f"Models tested: {len(results)}")
    print(f"Successful: {', '.join(results.keys())}")
    
    if results:
        print("\nModel Performance Summary:")
        for model_name, result in results.items():
            rmse = result['evaluation']['metrics']['overall']['rmse']
            r2 = result['evaluation']['metrics']['overall']['r2']
            print(f"  {model_name}: RMSE={rmse:.2f}, R²={r2:.4f}")
    
    return results

if __name__ == "__main__":
    test_pytorch_models()