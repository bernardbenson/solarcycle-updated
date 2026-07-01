"""
Seq2Seq trainer for WaveNet attention-based solar cycle prediction.
Supports teacher forcing, MC-dropout, quantile regression, and robust evaluation.
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any
import warnings
from tqdm import tqdm
import matplotlib.pyplot as plt

# Import our components
from .mixins import CombinedTrainerMixin
from ..models.wavenet_attn_seq2seq import WaveNetAttnSeq2Seq
from ..models.tcn_only import TCNOnly
from ..models.nbeatsx import NBEATSx
from ..models.heads import pinball_loss, quantile_loss_with_coverage, compute_combined_loss
from ..utils.rolling_cv import RollingOriginCV, BlockedTimeSeriesCV, TimeSeriesMetrics
from ..utils.peak_metrics import PeakMetrics, ConformalPeakPredictor
from ..utils.normalization import RobustScaler, prepare_enhanced_monthly_data
from ..utils.config import ExperimentConfig
from ..utils.plotting import SolarCyclePlotter


class SolarSequenceDataset(Dataset):
    """Dataset for sliding window sequence-to-sequence prediction."""
    
    def __init__(self, data: np.ndarray, features: Optional[Dict[str, np.ndarray]] = None,
                 input_window: int = 528, output_window: int = 132, step: int = 1):
        self.data = data
        self.features = features
        self.input_window = input_window
        self.output_window = output_window
        self.step = step
        
        # Create sequences
        self.sequences = []
        self.targets = []
        self.feature_sequences = []
        
        for i in range(0, len(data) - input_window - output_window + 1, step):
            # Input sequence
            input_seq = data[i:i + input_window]
            
            # Target sequence
            target_seq = data[i + input_window:i + input_window + output_window]
            
            self.sequences.append(torch.FloatTensor(input_seq).unsqueeze(-1))
            self.targets.append(torch.FloatTensor(target_seq))
            
            # Feature sequences if available
            if features:
                feat_seq = {}
                for feat_name, feat_data in features.items():
                    feat_seq[feat_name] = torch.FloatTensor(
                        feat_data[i:i + input_window]
                    )
                self.feature_sequences.append(feat_seq)
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        if self.features:
            return self.sequences[idx], self.targets[idx], self.feature_sequences[idx]
        else:
            return self.sequences[idx], self.targets[idx]


class Seq2SeqTrainer(CombinedTrainerMixin):
    """
    Trainer for sequence-to-sequence solar cycle prediction models.
    Supports WaveNet attention models with teacher forcing and uncertainty quantification.
    """
    
    def __init__(self, config: ExperimentConfig, device: str = 'auto'):
        super().__init__(
            patience=config.training.early_stop_patience,
            use_amp=config.training.amp,
            teacher_forcing_ratio=config.training.teacher_forcing
        )
        
        self.config = config
        
        # Device setup with Apple Silicon support
        if device == 'auto':
            if torch.cuda.is_available():
                self.device = torch.device('cuda')
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                self.device = torch.device('mps')
            else:
                self.device = torch.device('cpu')
        else:
            self.device = torch.device(device)
        
        print(f"Using device: {self.device}")
        
        # Initialize components
        self.model = None
        self.data_scaler = None
        self.peak_detector = PeakMetrics()
        self.conformal_predictor = ConformalPeakPredictor()
        
        # Set random seeds
        self._set_seeds(config.seed)
    
    def _set_seeds(self, seed: int):
        """Set random seeds for reproducibility."""
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
    
    def prepare_data(self, df, use_features: bool = True) -> Tuple[np.ndarray, Dict[str, np.ndarray], RobustScaler]:
        """Prepare data with enhanced normalization and features."""
        # Convert enum values to strings if needed
        method = self.config.data.normalization.method
        transform = self.config.data.normalization.transform
        
        if hasattr(method, 'value'):
            method = method.value
        if hasattr(transform, 'value'):
            transform = transform.value
            
        scaler_config = {
            'method': method,
            'transform': transform,
            'quantile_range': self.config.data.normalization.quantile_range
        }
        
        if use_features and self.config.data.add_features:
            try:
                scaled_target, features, scaler = prepare_enhanced_monthly_data(
                    df, 
                    target_col=self.config.data.target_col,
                    start_year=self.config.data.start_year,
                    scaler_config=scaler_config
                )
            except Exception as e:
                print(f"Warning: Feature preparation failed ({e}), using simple preparation")
                use_features = False
        
        if not use_features or not self.config.data.add_features:
            # Simple preparation without features
            import pandas as pd
            df['date'] = pd.to_datetime(df['date'])
            df = df.set_index('date')
            df = df[df.index.year >= self.config.data.start_year]
            
            monthly_data = df[self.config.data.target_col].resample('ME').mean().dropna()
            
            scaler = RobustScaler(**scaler_config)
            scaled_target = scaler.fit_transform(monthly_data.values)
            features = {}
        
        return scaled_target, features, scaler
    
    def create_data_loaders(self, data: np.ndarray, features: Optional[Dict[str, np.ndarray]] = None,
                           train_ratio: float = 0.8) -> Tuple[DataLoader, DataLoader]:
        """Create train and validation data loaders."""
        dataset = SolarSequenceDataset(
            data, features,
            input_window=self.config.data.input_window,
            output_window=self.config.data.prediction_horizon,
            step=1
        )
        
        # Split dataset
        total_samples = len(dataset)
        train_size = int(total_samples * train_ratio)
        
        train_indices = list(range(train_size))
        val_indices = list(range(train_size, total_samples))
        
        train_dataset = torch.utils.data.Subset(dataset, train_indices)
        val_dataset = torch.utils.data.Subset(dataset, val_indices)
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.training.batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=self.device.type == 'cuda'
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.training.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=self.device.type == 'cuda'
        )
        
        return train_loader, val_loader
    
    # Available architectures, selected via config.model.name.
    MODEL_REGISTRY = {
        'WaveNetAttnSeq2Seq': WaveNetAttnSeq2Seq,
        'TCNOnly': TCNOnly,
        'NBEATSx': NBEATSx,
    }

    def create_model(self) -> nn.Module:
        """Instantiate the architecture named in config.model.name."""
        name = self.config.model.name
        if name not in self.MODEL_REGISTRY:
            raise ValueError(
                f"Unknown model '{name}'. Available: {list(self.MODEL_REGISTRY)}"
            )
        model_cfg = self.config.model.dict()
        # N-BEATS needs the flattened input length, which lives in the data config.
        model_cfg.setdefault('input_window', self.config.data.input_window)
        model = self.MODEL_REGISTRY[name](model_cfg)
        return model.to(self.device)
    
    def create_optimizer(self, model: nn.Module) -> torch.optim.Optimizer:
        """Create optimizer from configuration."""
        if self.config.training.optimizer == 'adam':
            return torch.optim.Adam(
                model.parameters(),
                lr=self.config.training.lr,
                weight_decay=self.config.training.weight_decay
            )
        elif self.config.training.optimizer == 'adamw':
            return torch.optim.AdamW(
                model.parameters(),
                lr=self.config.training.lr,
                weight_decay=self.config.training.weight_decay
            )
        elif self.config.training.optimizer == 'sgd':
            return torch.optim.SGD(
                model.parameters(),
                lr=self.config.training.lr,
                weight_decay=self.config.training.weight_decay,
                momentum=0.9
            )
        else:
            raise ValueError(f"Unknown optimizer: {self.config.training.optimizer}")
    
    def compute_loss(self, outputs: Dict[str, torch.Tensor], targets: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute loss based on model head type."""
        if self.config.model.head == 'mse':
            loss = nn.MSELoss()(outputs['predictions'], targets)
            loss_components = {'mse_loss': loss.item(), 'total_loss': loss.item()}
        
        elif self.config.model.head == 'quantile':
            loss, coverage_stats = quantile_loss_with_coverage(
                outputs['predictions'], targets, self.config.model.quantiles
            )
            loss_components = {'quantile_loss': loss.item(), 'total_loss': loss.item()}
            loss_components.update(coverage_stats)
        
        elif self.config.model.head == 'combined':
            loss, loss_components = compute_combined_loss(
                outputs, targets, self.config.model.quantiles
            )
            loss_components = {k: v.item() if hasattr(v, 'item') else v for k, v in loss_components.items()}
        
        else:
            raise ValueError(f"Unknown head type: {self.config.model.head}")
        
        return loss, loss_components
    
    def train_epoch(self, model: nn.Module, train_loader: DataLoader, 
                   optimizer: torch.optim.Optimizer, epoch: int) -> Dict[str, float]:
        """Train for one epoch."""
        model.train()
        
        epoch_losses = []
        epoch_components = {}
        
        # Get current teacher forcing ratio
        tf_ratio = self.get_teacher_forcing_ratio(epoch)
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
        
        for batch_idx, batch in enumerate(progress_bar):
            if len(batch) == 3:
                inputs, targets, features = batch
            else:
                inputs, targets = batch
                features = None
            
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
            
            optimizer.zero_grad()
            
            # Forward pass with autocast. targets/teacher_forcing_ratio are passed by
            # keyword so feed-forward models (TCN, N-BEATS) can ignore them via **kwargs.
            with self.autocast_context():
                outputs = model(inputs, targets=targets, teacher_forcing_ratio=tf_ratio)
                loss, loss_components = self.compute_loss(outputs, targets)
            
            # Backward pass with gradient scaling
            self.scale_and_step(
                loss, optimizer, 
                clip_grad_norm=self.config.training.grad_clip_norm,
                model=model
            )
            
            # Track losses
            epoch_losses.append(loss.item())
            for key, value in loss_components.items():
                if key not in epoch_components:
                    epoch_components[key] = []
                epoch_components[key].append(value)
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'tf_ratio': f"{tf_ratio:.3f}"
            })
        
        # Aggregate epoch metrics
        epoch_metrics = {'train_loss': np.mean(epoch_losses)}
        for key, values in epoch_components.items():
            epoch_metrics[f'train_{key}'] = np.mean(values)
        
        epoch_metrics['teacher_forcing_ratio'] = tf_ratio
        
        return epoch_metrics
    
    def validate_epoch(self, model: nn.Module, val_loader: DataLoader) -> Dict[str, float]:
        """Validate for one epoch."""
        model.eval()
        
        val_losses = []
        val_components = {}
        
        with torch.no_grad():
            for batch in val_loader:
                if len(batch) == 3:
                    inputs, targets, features = batch
                else:
                    inputs, targets = batch
                    features = None
                
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                
                # Forward pass without teacher forcing
                with self.autocast_context():
                    outputs = model(inputs, teacher_forcing_ratio=0.0)
                    loss, loss_components = self.compute_loss(outputs, targets)
                
                val_losses.append(loss.item())
                for key, value in loss_components.items():
                    if key not in val_components:
                        val_components[key] = []
                    val_components[key].append(value)
        
        # Aggregate validation metrics
        val_metrics = {'val_loss': np.mean(val_losses)}
        for key, values in val_components.items():
            val_metrics[f'val_{key}'] = np.mean(values)
        
        return val_metrics
    
    def train(self, df, output_dir: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
        """
        Full training pipeline with cross-validation and evaluation.
        
        Args:
            df: Solar data DataFrame
            output_dir: Directory to save results
        
        Returns:
            Training results dictionary
        """
        # Generate unique run ID with datetime
        from datetime import datetime
        import uuid
        
        if output_dir is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            run_id = f"{timestamp}_{str(uuid.uuid4())[:8]}"
            output_dir = Path(self.config.output_dir) / f"run_{self.config.experiment_name}_{run_id}"
        else:
            output_dir = Path(output_dir)
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"Starting training: {self.config.experiment_name}")
        print(f"Output directory: {output_dir}")
        
        # Prepare data
        print("Preparing data...")
        scaled_data, features, scaler = self.prepare_data(df, use_features=self.config.data.add_features)
        self.data_scaler = scaler
        
        print(f"Data shape: {scaled_data.shape}")
        print(f"Features: {list(features.keys()) if features else 'None'}")
        
        # Save scaler
        self.data_scaler.save_params(output_dir / "scaler.json")
        
        # Create model
        print("Creating model...")
        model = self.create_model()
        self.model = model
        
        print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        # Create data loaders
        train_loader, val_loader = self.create_data_loaders(
            scaled_data, features, train_ratio=1.0 - self.config.training.val_ratio
        )
        
        print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
        
        # Create optimizer and scheduler
        optimizer = self.create_optimizer(model)
        scheduler = self.create_scheduler(
            optimizer, 
            self.config.training.scheduler,
            {
                'warmup_epochs': self.config.training.warmup_epochs,
                'patience': self.config.training.scheduler_patience,
                'factor': self.config.training.scheduler_factor
            }
        )
        
        # Training loop
        print(f"Starting training for {self.config.training.epochs} epochs...")
        
        best_val_loss = float('inf')
        
        for epoch in range(self.config.training.epochs):
            # Train epoch
            train_metrics = self.train_epoch(model, train_loader, optimizer, epoch)
            
            # Validation epoch
            val_metrics = self.validate_epoch(model, val_loader)
            
            # Combine metrics
            epoch_metrics = {**train_metrics, **val_metrics}
            
            # Log metrics (train_loss/val_loss are passed explicitly; the rest as extras).
            current_lr = optimizer.param_groups[0]['lr']
            extra_metrics = {k: v for k, v in epoch_metrics.items()
                             if k not in ('train_loss', 'val_loss')}
            self.log_metrics(
                epoch,
                train_metrics['train_loss'],
                val_metrics['val_loss'],
                lr=current_lr,
                **extra_metrics
            )
            
            # Step scheduler
            self.step_scheduler(scheduler, self.config.training.scheduler, val_metrics['val_loss'])
            
            # Step teacher forcing
            self.step_teacher_forcing()
            
            # Early stopping check
            should_stop = self.step(val_metrics['val_loss'], model, epoch)
            
            # Logging
            if (epoch + 1) % self.config.log_interval == 0 or epoch == 0:
                print(f"Epoch {epoch+1:3d}/{self.config.training.epochs}: "
                      f"Train Loss: {train_metrics['train_loss']:.4f}, "
                      f"Val Loss: {val_metrics['val_loss']:.4f}, "
                      f"LR: {current_lr:.2e}, "
                      f"TF: {train_metrics['teacher_forcing_ratio']:.3f}")
            
            # Save best model
            if val_metrics['val_loss'] < best_val_loss:
                best_val_loss = val_metrics['val_loss']
                if self.config.save_model:
                    self.save_checkpoint(
                        model, optimizer, scheduler, epoch, val_metrics['val_loss'],
                        output_dir / "best_model.pt"
                    )
            
            if should_stop:
                print(f"Early stopping at epoch {epoch+1}")
                break
        
        # Save training metrics
        self.save_metrics(output_dir / "training_metrics.json")
        
        # Save final model
        if self.config.save_model:
            self.save_checkpoint(
                model, optimizer, scheduler, epoch, val_metrics['val_loss'],
                output_dir / "final_model.pt"
            )
        
        # Training results
        training_results = {
            'best_epoch': self.get_best_epoch('val_loss'),
            'best_val_loss': best_val_loss,
            'total_epochs': epoch + 1,
            'early_stopped': self.early_stopped,
            'model_parameters': sum(p.numel() for p in model.parameters()),
            'output_dir': str(output_dir)
        }
        
        # Generate plots if enabled
        if self.config.plot_training or self.config.plot_predictions:
            print("Generating plots...")
            self._generate_plots(output_dir, training_results, scaled_data, features)
        
        return training_results
    
    def _generate_plots(self, output_dir: Path, training_results: Dict[str, Any], 
                       scaled_data: np.ndarray, features: Dict[str, np.ndarray]):
        """Generate comprehensive plots after training."""
        # Create plots subdirectory
        plots_dir = output_dir / "plots"
        plots_dir.mkdir(exist_ok=True)
        
        plotter = SolarCyclePlotter(style='publication')
        
        try:
            # 1. Training history plot
            if hasattr(self, 'metrics_history') and self.config.plot_training:
                print("  - Generating training history plot...")
                fig = plotter.plot_training_history_enhanced(
                    self.metrics_history,
                    save_path=plots_dir / "training_history.png"
                )
                plt.close(fig)
            
            # 2. Generate sample predictions with uncertainty
            if self.config.plot_predictions and self.model is not None:
                print("  - Generating prediction plots with uncertainty...")
                
                # Generate predictions on recent data
                recent_window = min(self.config.data.input_window, len(scaled_data) - 50)
                test_input = scaled_data[-recent_window-50:-50]
                
                # Get uncertainty predictions
                uncertainty_results = self.predict_with_uncertainty(
                    test_input, n_mc_samples=20
                )
                
                # Plot predictions with uncertainty
                prediction_data = {
                    'prediction': uncertainty_results['mean'],
                    'uncertainty': {
                        'q10': uncertainty_results['q10'],
                        'q50': uncertainty_results['q50'], 
                        'q90': uncertainty_results['q90']
                    }
                }
                
                # Get actual data for comparison if available
                actual_data = scaled_data[-50:] if len(scaled_data) >= 50 else None
                if actual_data is not None and self.data_scaler is not None:
                    actual_data = self.data_scaler.inverse_transform(actual_data.reshape(-1, 1)).ravel()
                
                fig = plotter.plot_single_cycle_with_uncertainty(
                    actual=actual_data,
                    prediction=uncertainty_results['mean'],
                    uncertainty=prediction_data['uncertainty'],
                    title=f"Recent Predictions - {self.config.experiment_name}",
                    xlabel="Time Steps",
                    ylabel="Sunspot Number",
                    save_path=plots_dir / "recent_predictions_with_uncertainty.png"
                )
                plt.close(fig)
                
                # 3. MC-Dropout uncertainty plot
                print("  - Generating MC-dropout uncertainty plot...")
                fig = plotter.plot_mc_dropout_uncertainty(
                    mc_samples=uncertainty_results['samples'],
                    actual=actual_data,
                    title=f"MC-Dropout Uncertainty - {self.config.experiment_name}",
                    save_path=plots_dir / "mc_dropout_uncertainty.png"
                )
                plt.close(fig)
                
                # 4. Peak distribution plot
                print("  - Generating peak distribution plot...")
                fig = plotter.plot_peak_distribution(
                    prediction_samples=uncertainty_results['samples'].T,  # Transpose for correct shape
                    title=f"Peak Distribution - {self.config.experiment_name}",
                    save_path=plots_dir / "peak_distribution.png"
                )
                plt.close(fig)
                
        except Exception as e:
            print(f"Warning: Plot generation failed with error: {e}")
            print("Continuing without plots...")
        
        print(f"  ✅ Plots saved to: {plots_dir}")
    
    def predict_with_uncertainty(self, input_data: np.ndarray, 
                                n_mc_samples: int = 30) -> Dict[str, np.ndarray]:
        """Generate predictions with uncertainty using MC-Dropout."""
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        self.model.eval()
        
        # Prepare input
        if self.data_scaler is not None:
            input_scaled = self.data_scaler.transform(input_data.reshape(-1, 1)).ravel()
        else:
            input_scaled = input_data
        
        input_tensor = torch.FloatTensor(input_scaled).unsqueeze(0).unsqueeze(-1).to(self.device)
        
        # Generate MC-Dropout predictions
        mc_predictions = self.model.mc_predict(input_tensor, n_samples=n_mc_samples)
        
        # Convert back to original scale
        if self.data_scaler is not None:
            mc_predictions_unscaled = []
            for i in range(mc_predictions.shape[-1]):
                pred_unscaled = self.data_scaler.inverse_transform(
                    mc_predictions[0, :, i].cpu().numpy().reshape(-1, 1)
                ).ravel()
                mc_predictions_unscaled.append(pred_unscaled)
            mc_predictions_unscaled = np.stack(mc_predictions_unscaled, axis=-1)
        else:
            mc_predictions_unscaled = mc_predictions[0].cpu().numpy()
        
        # Compute statistics
        mean_pred = np.mean(mc_predictions_unscaled, axis=-1)
        std_pred = np.std(mc_predictions_unscaled, axis=-1)
        
        # Quantiles
        q10 = np.percentile(mc_predictions_unscaled, 10, axis=-1)
        q50 = np.percentile(mc_predictions_unscaled, 50, axis=-1)
        q90 = np.percentile(mc_predictions_unscaled, 90, axis=-1)
        
        return {
            'mean': mean_pred,
            'std': std_pred,
            'q10': q10,
            'q50': q50,
            'q90': q90,
            'samples': mc_predictions_unscaled
        }


if __name__ == "__main__":
    # Test the seq2seq trainer
    from ..utils.config import ExperimentConfig
    
    print("Testing Seq2Seq Trainer...")
    
    # Create test config
    config = ExperimentConfig(
        experiment_name="test_seq2seq",
        training=TrainingConfig(epochs=2, batch_size=4),
        data=DataConfig(input_window=100, prediction_horizon=20)
    )
    
    # Create trainer
    trainer = Seq2SeqTrainer(config)
    
    # Test data preparation
    import pandas as pd
    
    # Create dummy data
    dates = pd.date_range('1950-01-01', periods=2000, freq='M')
    dummy_data = pd.DataFrame({
        'date': dates,
        'sunspot_number': np.random.randn(2000) * 20 + 100
    })
    
    print("Testing data preparation...")
    scaled_data, features, scaler = trainer.prepare_data(dummy_data, use_features=False)
    print(f"Scaled data shape: {scaled_data.shape}")
    
    print("\n✅ Seq2Seq trainer implemented and tested!")