#!/usr/bin/env python3
"""
Training script for the enhanced WaveNet attention seq2seq model.
Supports CPU, CUDA, and Apple Silicon MPS devices.
"""

import sys
import argparse
from pathlib import Path
import torch

# Add the project root (for `solar.*`) and the scripts dir (for `backtest`) to the path
sys.path.append(str(Path(__file__).parent.parent))
sys.path.append(str(Path(__file__).parent))

from solar.utils.config import load_config, ExperimentConfig
from solar.trainers.seq2seq_trainer import Seq2SeqTrainer
from solar.utils.plotting import create_summary_report
from solar.data import load_solar_data


def check_device_compatibility():
    """Check and report available devices."""
    print("Device Compatibility Check:")
    print(f"- PyTorch version: {torch.__version__}")
    print(f"- CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"- CUDA device: {torch.cuda.get_device_name()}")
    
    if hasattr(torch.backends, 'mps'):
        print(f"- MPS available: {torch.backends.mps.is_available()}")
        if torch.backends.mps.is_available():
            print("- Apple Silicon GPU detected")
    
    # Test device selection
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    
    print(f"- Selected device: {device}")
    
    # Test basic tensor operations
    try:
        x = torch.randn(10, 10).to(device)
        y = torch.mm(x, x.t())
        print(f"- Device test passed: {y.shape}")
        return device
    except Exception as e:
        print(f"- Device test failed: {e}")
        print("- Falling back to CPU")
        return torch.device('cpu')


def main():
    parser = argparse.ArgumentParser(description="Train WaveNet Attention Seq2Seq model")
    parser.add_argument('--config', type=str, default='solar/configs/seq2seq_quantile.yaml',
                       help='Path to configuration file')
    parser.add_argument('--device', type=str, default='auto',
                       choices=['auto', 'cpu', 'cuda', 'mps'],
                       help='Device to use for training')
    parser.add_argument('--test-device', action='store_true',
                       help='Test device compatibility and exit')
    parser.add_argument('--epochs', type=int, default=None,
                       help='Override number of training epochs')
    parser.add_argument('--batch-size', type=int, default=None,
                       help='Override batch size')
    parser.add_argument('--no-backtest', action='store_true',
                       help='Skip the hindcast backtest that runs automatically after training')
    parser.add_argument('--backtest-panels', type=int, default=4,
                       help='Number of past cycles to validate in the post-training backtest')

    args = parser.parse_args()
    
    # Test device compatibility
    detected_device = check_device_compatibility()
    
    if args.test_device:
        print("Device test completed.")
        return
    
    print(f"\n{'='*60}")
    print("WAVENET ATTENTION SEQ2SEQ TRAINING")
    print(f"{'='*60}")
    
    # Load configuration
    try:
        config = load_config(args.config)
        print(f"Loaded config: {config.experiment_name}")
    except FileNotFoundError:
        print(f"Config file not found: {args.config}")
        print("Using default configuration...")
        config = ExperimentConfig()
    
    # Override config parameters
    if args.epochs is not None:
        config.training.epochs = args.epochs
        print(f"Overriding epochs: {args.epochs}")
    
    if args.batch_size is not None:
        config.training.batch_size = args.batch_size
        print(f"Overriding batch size: {args.batch_size}")
    
    # Use detected device if auto
    device = args.device if args.device != 'auto' else str(detected_device)
    
    # Disable AMP for non-CUDA devices
    if device != 'cuda':
        config.training.amp = False
        print(f"Disabled AMP for device: {device}")
    
    # Load data
    print("\nLoading solar data...")
    df = load_solar_data(dataset=config.data.dataset,
                         need_precursors=bool(config.data.precursor_cols))
    print(f"Data shape: {df.shape}")
    print(f"Date range: {df['date'].min()} to {df['date'].max()}")
    
    # Initialize trainer
    print(f"\nInitializing trainer with device: {device}")
    trainer = Seq2SeqTrainer(config, device=device)
    
    # Train model
    print("\nStarting training...")
    try:
        training_results = trainer.train(df)
        print(f"\nTraining completed successfully!")
        print(f"Best validation loss: {training_results['best_val_loss']:.4f}")
        print(f"Results saved to: {training_results['output_dir']}")

        # The trainer already generates uncertainty plots during train(); here we
        # just write the summary report from the returned training results.
        output_dir = Path(training_results['output_dir'])
        create_summary_report(
            {'training_results': training_results, 'config': config.dict()},
            output_dir,
            config.experiment_name,
        )
        
        # Hindcast backtest against past cycles (writes plots/cycle_backtest.png).
        # Reuses the already-loaded dataframe; reloads the best checkpoint from disk
        # so it validates exactly what a standalone `scripts/backtest.py` run would.
        if not args.no_backtest:
            print("\nRunning hindcast backtest against past cycles...")
            try:
                from backtest import run_backtest
                run_backtest(config, output_dir, device=device,
                             n_panels=args.backtest_panels, df=df)
            except Exception as e:
                print(f"⚠️  Backtest step failed (training results are unaffected): {e}")

        print(f"\n✅ Training pipeline completed successfully!")
        print(f"All results saved to: {output_dir}")

    except Exception as e:
        print(f"\n❌ Training failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())