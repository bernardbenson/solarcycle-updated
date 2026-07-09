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
from ..models.wavenet_lstm_direct import WaveNetLSTMDirect
from ..models.tcn_only import TCNOnly
from ..models.nbeatsx import NBEATSx
from ..models.heads import pinball_loss, quantile_loss_with_coverage, compute_combined_loss
from ..data.monthly import build_monthly_series
from ..eval.metrics import forecast_metrics, aggregate_metrics
from ..utils.peak_metrics import PeakMetrics
from ..utils.normalization import (
    RobustScaler, MultiChannelScaler, prepare_multivariate_monthly_data,
)
from ..utils.precursors import detect_cycle_minima, cycle_length_series
from ..utils.splits import time_axis_split, assert_no_target_overlap
from ..utils.config import ExperimentConfig
from ..utils.plotting import SolarCyclePlotter


class SolarSequenceDataset(Dataset):
    """Sliding-window dataset emitting ``(input, target, cond)``.

    ``data`` is the scaled series: 1-D ``(T,)`` (univariate) or 2-D ``(T, C)``
    (multivariate, channel 0 = target). The input is the window of all channels; the
    target is the following horizon of channel 0 only. ``cond`` is the per-window
    precursor conditioning vector (empty when conditioning is disabled).
    """

    def __init__(self, data: np.ndarray, cond: Optional[np.ndarray] = None,
                 input_window: int = 528, output_window: int = 132, step: int = 1,
                 origins: Optional[List[int]] = None):
        data = np.asarray(data, dtype=np.float32)
        multivariate = data.ndim == 2
        cond_dim = 0 if cond is None else 1

        # A window is identified by its forecast origin o: input [o-W, o), target
        # [o, o+H). Explicit `origins` lets callers build leak-free splits; the
        # default enumerates every valid origin at the given stride.
        if origins is None:
            origins = list(range(input_window, len(data) - output_window + 1, step))

        self.sequences, self.targets, self.conds = [], [], []
        for o in origins:
            if o < input_window or o + output_window > len(data):
                raise ValueError(f"Origin {o} out of range for series of length {len(data)}")
            window = data[o - input_window:o]
            input_seq = torch.from_numpy(window if multivariate else window[:, None])

            target_col = data[o:o + output_window]
            target_seq = torch.from_numpy(target_col[:, 0] if multivariate else target_col)

            self.sequences.append(input_seq)
            self.targets.append(target_seq)
            if cond_dim:
                self.conds.append(torch.tensor([cond[o - 1]], dtype=torch.float32))
            else:
                self.conds.append(torch.zeros(0, dtype=torch.float32))

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx], self.targets[idx], self.conds[idx]


class Seq2SeqTrainer(CombinedTrainerMixin):
    """
    Trainer for sequence-to-sequence solar cycle prediction models.
    Supports WaveNet attention models with teacher forcing and uncertainty quantification.
    """
    
    def __init__(self, config: ExperimentConfig, device: str = 'auto'):
        super().__init__(
            patience=config.training.early_stop_patience,
            use_amp=config.training.amp,
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
        self.data_scaler = None          # target (sunspot) scaler; used for inverse-transform
        self.multi_scaler = None         # per-channel input scaler (multivariate runs only)
        self.cond_scaler = None          # normaliser for the terminator conditioning scalar
        self._train_df = None            # df used for training (for plot rebuilding)
        self.peak_detector = PeakMetrics()
        
        # Set random seeds
        self._set_seeds(config.seed)
    
    def _set_seeds(self, seed: int):
        """Set random seeds for reproducibility."""
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
    
    def _scaler_config(self) -> Dict[str, Any]:
        """Normalization settings for the target channel (accept str or enum)."""
        method = self.config.data.normalization.method
        transform = self.config.data.normalization.transform
        return {
            'method': getattr(method, 'value', method),
            'transform': getattr(transform, 'value', transform),
            'quantile_range': self.config.data.normalization.quantile_range,
        }

    def _train_boundary(self, n_months: int) -> int:
        """Time-axis train/val boundary: everything at/after it is validation era."""
        horizon = self.config.data.prediction_horizon
        val_months = max(horizon, int(round(self.config.training.val_ratio * n_months)))
        return n_months - val_months

    def prepare_data(self, df) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Prepare scaled model inputs and optional precursor conditioning.

        Returns ``(scaled_input, cond_series)``: ``scaled_input`` is ``(T,)`` univariate
        or ``(T, C)`` multivariate (channel 0 = target); ``cond_series`` is a normalized
        ``(T,)`` terminator series or None. Sets ``self.data_scaler`` (target),
        ``self.multi_scaler`` and ``self.cond_scaler``.

        All scalers are fit on the training era only (months before the time-axis
        split boundary, ``self.t_split``) and then applied to the full series.
        """
        scaler_config = self._scaler_config()
        precursor_cols = list(self.config.data.precursor_cols)

        if precursor_cols:
            # Boundary must be known before fitting: compute the monthly length first.
            monthly = build_monthly_series(df, target_col=self.config.data.target_col,
                                           start_year=self.config.data.start_year)
            self.t_split = self._train_boundary(len(monthly))
            X_scaled, X_raw, multi_scaler, dates = prepare_multivariate_monthly_data(
                df,
                target_col=self.config.data.target_col,
                precursor_cols=precursor_cols,
                geomag_mask=self.config.data.geomag_mask,
                start_year=self.config.data.start_year,
                scaler_config=scaler_config,
                fit_end_idx=self.t_split,
            )
            self.multi_scaler = multi_scaler
            self.data_scaler = multi_scaler.target_scaler
            scaled_input = X_scaled
            raw_target = X_raw[:, 0]
        else:
            monthly = build_monthly_series(df, target_col=self.config.data.target_col,
                                           start_year=self.config.data.start_year)
            self.t_split = self._train_boundary(len(monthly))
            self.data_scaler = RobustScaler(**scaler_config)
            self.data_scaler.fit(monthly.values[:self.t_split])
            scaled_input = self.data_scaler.transform(monthly.values)
            self.multi_scaler = None
            raw_target = monthly.values
            dates = monthly.index

        self._train_dates = dates

        cond_series = None
        self.cond_scaler = None
        if self.config.data.use_terminator:
            # cycle_length_series is causal (confirmed minima only), so the full
            # series can be built once; only the scaler must be train-era fit.
            cond_raw = cycle_length_series(raw_target, self.config.data.prediction_horizon)
            self.cond_scaler = RobustScaler(method='standard', transform='identity')
            self.cond_scaler.fit(cond_raw[:self.t_split])
            if self.cond_scaler.scaler is not None and hasattr(self.cond_scaler.scaler, 'scale_'):
                # Guard against a degenerate (constant) training slice.
                self.cond_scaler.scaler.scale_ = np.maximum(self.cond_scaler.scaler.scale_, 1e-6)
            cond_series = self.cond_scaler.transform(cond_raw.reshape(-1, 1)).ravel()

        return scaled_input, cond_series

    def create_data_loaders(self, data: np.ndarray, cond: Optional[np.ndarray] = None
                            ) -> Tuple[DataLoader, DataLoader]:
        """Create leak-free train and validation data loaders.

        The split is on the TIME axis (see solar.utils.splits): train targets end
        before ``self.t_split``; validation forecast origins start at it. Train
        and validation target months are disjoint (asserted).
        """
        input_window = self.config.data.input_window
        horizon = self.config.data.prediction_horizon

        train_origins, val_origins, _ = time_axis_split(
            len(data), input_window, horizon,
            val_ratio=self.config.training.val_ratio,
            val_stride=self.config.training.val_stride,
        )
        assert_no_target_overlap(train_origins, val_origins, horizon)

        train_dataset = SolarSequenceDataset(
            data, cond, input_window=input_window, output_window=horizon,
            origins=train_origins)
        val_dataset = SolarSequenceDataset(
            data, cond, input_window=input_window, output_window=horizon,
            origins=val_origins)

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.training.batch_size,
            shuffle=True,
            drop_last=len(train_dataset) > self.config.training.batch_size,
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
        'WaveNetLSTM': WaveNetLSTMDirect,
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
    
    def _batch_cond(self, cond: torch.Tensor) -> Optional[torch.Tensor]:
        """Move a batch's conditioning tensor to device, or None when disabled."""
        if cond is None or cond.shape[-1] == 0:
            return None
        return cond.to(self.device)

    def train_epoch(self, model: nn.Module, train_loader: DataLoader,
                   optimizer: torch.optim.Optimizer, epoch: int) -> Dict[str, float]:
        """Train for one epoch."""
        model.train()

        epoch_losses = []
        epoch_components = {}

        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}")

        for batch_idx, batch in enumerate(progress_bar):
            inputs, targets, cond = batch
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
            cond = self._batch_cond(cond)

            optimizer.zero_grad()

            # Forward pass with autocast. All registered models are direct
            # (non-autoregressive) forecasters; cond is passed by keyword so
            # models without conditioning ignore it via **kwargs.
            with self.autocast_context():
                outputs = model(inputs, cond=cond)
                loss, loss_components = self.compute_loss(outputs, targets)

            # Backward pass with gradient scaling
            self.scale_and_step(
                loss, optimizer,
                clip_grad_norm=self.config.training.grad_clip_norm,
                model=model
            )

            # Update the EMA shadow weights after each optimizer step.
            self.update_ema(model)

            # Track losses
            epoch_losses.append(loss.item())
            for key, value in loss_components.items():
                if key not in epoch_components:
                    epoch_components[key] = []
                epoch_components[key].append(value)
            
            # Update progress bar
            progress_bar.set_postfix({'loss': f"{loss.item():.4f}"})

        # Aggregate epoch metrics
        epoch_metrics = {'train_loss': np.mean(epoch_losses)}
        for key, values in epoch_components.items():
            epoch_metrics[f'train_{key}'] = np.mean(values)

        return epoch_metrics
    
    def _inverse_target(self, scaled: np.ndarray) -> np.ndarray:
        """Inverse-transform scaled target values to raw units, clipped at 0."""
        raw = self.data_scaler.inverse_transform(
            np.asarray(scaled, dtype=float).reshape(-1, 1)).ravel()
        return np.clip(raw, 0.0, None).reshape(np.asarray(scaled).shape)

    def _split_quantile_outputs(self, preds: torch.Tensor
                                ) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
        """(B,H) or (B,H,nq) scaled predictions -> raw-unit (point, q10, q90)."""
        preds = preds.detach().cpu().numpy()
        if preds.ndim == 3:
            quantiles = self.config.model.quantiles
            median_idx = quantiles.index(0.5) if 0.5 in quantiles else len(quantiles) // 2
            point = self._inverse_target(preds[:, :, median_idx])
            q10 = self._inverse_target(preds[:, :, 0])
            q90 = self._inverse_target(preds[:, :, -1])
            return point, q10, q90
        return self._inverse_target(preds), None, None

    def validate_epoch(self, model: nn.Module, val_loader: DataLoader) -> Dict[str, float]:
        """Validate for one epoch.

        Reports both the scaled training loss (val_loss) and raw-unit metrics
        (val_rmse_raw, val_mae_raw, peak errors, 80% coverage) computed after
        inverse-transforming predictions back to physical target units.
        """
        model.eval()

        val_losses = []
        val_components = {}
        window_metrics = []

        with torch.no_grad():
            for batch in val_loader:
                inputs, targets, cond = batch
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                cond = self._batch_cond(cond)

                with self.autocast_context():
                    outputs = model(inputs, cond=cond)
                    loss, loss_components = self.compute_loss(outputs, targets)

                val_losses.append(loss.item())
                for key, value in loss_components.items():
                    if key not in val_components:
                        val_components[key] = []
                    val_components[key].append(value)

                # Raw-unit metrics per validation window.
                preds = outputs['quantile'] if 'quantile' in outputs else outputs['predictions']
                point, q10, q90 = self._split_quantile_outputs(preds)
                actual = self._inverse_target(targets.detach().cpu().numpy())
                for i in range(point.shape[0]):
                    window_metrics.append(forecast_metrics(
                        actual[i], point[i],
                        q10=q10[i] if q10 is not None else None,
                        q90=q90[i] if q90 is not None else None))

        # Aggregate validation metrics
        val_metrics = {'val_loss': np.mean(val_losses)}
        for key, values in val_components.items():
            val_metrics[f'val_{key}'] = np.mean(values)

        raw = aggregate_metrics(window_metrics)
        val_metrics['val_rmse_raw'] = raw.get('rmse', float('nan'))
        val_metrics['val_mae_raw'] = raw.get('mae', float('nan'))
        val_metrics['val_peak_amp_mae'] = raw.get('peak_amp_mae', float('nan'))
        val_metrics['val_peak_timing_mae'] = raw.get('peak_timing_mae', float('nan'))
        if 'coverage_80' in raw:
            val_metrics['val_coverage_80_raw'] = raw['coverage_80']

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
        self._train_df = df
        scaled_data, cond_series = self.prepare_data(df)

        # Keep model input/cond dims consistent with the assembled channels.
        self.config.model.input_dim = scaled_data.shape[1] if scaled_data.ndim == 2 else 1
        self.config.model.cond_dim = 1 if self.config.data.use_terminator else 0

        print(f"Data shape: {scaled_data.shape}, input_dim={self.config.model.input_dim}, "
              f"cond_dim={self.config.model.cond_dim}")

        # Save scalers (target scaler.json is unchanged; extras only when multivariate/conditioned).
        self.data_scaler.save_params(output_dir / "scaler.json")
        if self.multi_scaler is not None:
            self.multi_scaler.save_params(output_dir / "feature_scalers.json")
        if self.cond_scaler is not None:
            self.cond_scaler.save_params(output_dir / "cond_scaler.json")

        # Create model + EMA
        print("Creating model...")
        model = self.create_model()
        self.model = model
        self.init_ema(model, decay=self.config.training.ema_decay,
                      enabled=self.config.training.use_ema)

        print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

        # Create data loaders (leak-free time-axis split at self.t_split).
        train_loader, val_loader = self.create_data_loaders(scaled_data, cond_series)

        train_end_date = str(self._train_dates[self.t_split - 1].date()) \
            if getattr(self, '_train_dates', None) is not None else None
        print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)} "
              f"(train era ends {train_end_date})")

        # Persist run metadata so backtests can distinguish in-sample panels.
        import json as _json
        with open(output_dir / "run_meta.json", 'w') as f:
            _json.dump({
                't_split': int(self.t_split),
                'train_end_date': train_end_date,
                'n_months': int(len(scaled_data)),
                'model_name': self.config.model.name,
            }, f, indent=2)
        
        # Create optimizer and scheduler
        optimizer = self.create_optimizer(model)
        scheduler = self.create_scheduler(
            optimizer,
            self.config.training.scheduler,
            {
                'warmup_epochs': self.config.training.warmup_epochs,
                'total_epochs': self.config.training.epochs,
                'patience': self.config.training.scheduler_patience,
                'factor': self.config.training.scheduler_factor
            }
        )
        
        # Training loop. Early stopping, LR plateau, and best-checkpoint selection
        # all monitor config.training.early_stop_metric (e.g. val_rmse_raw, the
        # RMSE in physical target units) - never the scaled loss unless asked.
        monitor_key = self.config.training.early_stop_metric
        print(f"Starting training for {self.config.training.epochs} epochs "
              f"(monitoring {monitor_key})...")

        best_monitor = float('inf')

        for epoch in range(self.config.training.epochs):
            # Train epoch
            train_metrics = self.train_epoch(model, train_loader, optimizer, epoch)

            # Evaluate / select / checkpoint using the EMA weights when enabled; the raw
            # weights are swapped back after the epoch (unless early stopping restores the
            # EMA-best weights into the model, in which case we keep them).
            ema_backup = self.copy_to(model) if self.ema_enabled else None

            # Validation epoch
            val_metrics = self.validate_epoch(model, val_loader)
            monitored = val_metrics.get(monitor_key, val_metrics['val_loss'])
            if not np.isfinite(monitored):
                monitored = val_metrics['val_loss']

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
            self.step_scheduler(scheduler, self.config.training.scheduler, monitored)

            # Early stopping check
            should_stop = self.step(monitored, model, epoch)

            # Logging
            if (epoch + 1) % self.config.log_interval == 0 or epoch == 0:
                print(f"Epoch {epoch+1:3d}/{self.config.training.epochs}: "
                      f"Train Loss: {train_metrics['train_loss']:.4f}, "
                      f"Val Loss: {val_metrics['val_loss']:.4f}, "
                      f"Val RMSE (raw): {val_metrics.get('val_rmse_raw', float('nan')):.2f}, "
                      f"LR: {current_lr:.2e}")

            # Save best model (EMA weights when enabled).
            if monitored < best_monitor:
                best_monitor = monitored
                if self.config.save_model:
                    self.save_checkpoint(
                        model, optimizer, scheduler, epoch, monitored,
                        output_dir / "best_model.pt"
                    )

            # Swap raw weights back for the next epoch's training (skip on stop so the
            # EMA-best weights restored by early stopping are preserved).
            if ema_backup is not None and not should_stop:
                self.restore(model, ema_backup)

            if should_stop:
                print(f"Early stopping at epoch {epoch+1}")
                break
        
        # Save training metrics
        self.save_metrics(output_dir / "training_metrics.json")
        
        # Save final model (last-epoch weights)
        if self.config.save_model:
            self.save_checkpoint(
                model, optimizer, scheduler, epoch, monitored,
                output_dir / "final_model.pt"
            )

        # Load the best (early-stopping-tracked) weights so plots use the best model.
        if getattr(self, 'best_weights', None) is not None:
            device = next(model.parameters()).device
            model.load_state_dict({k: v.to(device) for k, v in self.best_weights.items()})

        # Training results
        best_epoch = self.get_best_epoch(monitor_key if monitor_key in self.metrics_history
                                         else 'val_loss')
        training_results = {
            'best_epoch': best_epoch,
            'monitor_metric': monitor_key,
            'best_val_loss': best_monitor,
            'best_val_rmse_raw': float(np.min(self.metrics_history['val_rmse_raw']))
                if self.metrics_history.get('val_rmse_raw') else None,
            'total_epochs': epoch + 1,
            'early_stopped': self.early_stopped,
            'model_parameters': sum(p.numel() for p in model.parameters()),
            'train_end_date': train_end_date,
            'output_dir': str(output_dir)
        }
        
        # Generate plots if enabled
        if self.config.plot_training or self.config.plot_predictions:
            print("Generating plots...")
            self._generate_plots(output_dir, training_results)

        return training_results
    
    def _generate_plots(self, output_dir: Path, training_results: Dict[str, Any]):
        """Generate comprehensive plots after training."""
        plots_dir = output_dir / "plots"
        plots_dir.mkdir(exist_ok=True)

        plotter = SolarCyclePlotter(style='publication')
        name = self.config.experiment_name

        try:
            # 1. Training history plot
            if hasattr(self, 'metrics_history') and self.config.plot_training:
                print("  - Generating training history plot...")
                fig = plotter.plot_training_history_enhanced(
                    self.metrics_history, save_path=plots_dir / "training_history.png"
                )
                plt.close(fig)

            if not (self.config.plot_predictions and self.model is not None):
                print(f"  ✅ Plots saved to: {plots_dir}")
                return

            raw_target, X_raw, cond_raw, dates = self._monthly_arrays(self._train_df)
            window = self.config.data.input_window
            n = len(raw_target)

            # 2-4. Recent predictions (anchored 50 months before the end for overlap).
            if n - 50 - window >= 0:
                print("  - Generating recent predictions + uncertainty plots...")
                unc = self._forecast_from(raw_target, X_raw, cond_raw, n - 50, n_mc=20)
                actual = raw_target[-50:]
                fig = plotter.plot_single_cycle_with_uncertainty(
                    actual=actual, prediction=unc['mean'],
                    uncertainty={'q10': unc['q10'], 'q50': unc['q50'], 'q90': unc['q90']},
                    title=f"Recent Predictions - {name}", xlabel="Months", ylabel="Sunspot Number",
                    save_path=plots_dir / "recent_predictions_with_uncertainty.png")
                plt.close(fig)
                fig = plotter.plot_mc_dropout_uncertainty(
                    mc_samples=unc['samples'], actual=actual,
                    title=f"MC-Dropout Uncertainty - {name}",
                    save_path=plots_dir / "mc_dropout_uncertainty.png")
                plt.close(fig)
                fig = plotter.plot_peak_distribution(
                    prediction_samples=unc['samples'].T,
                    title=f"Peak Distribution - {name}",
                    save_path=plots_dir / "peak_distribution.png")
                plt.close(fig)

            # 5. Genuine next-cycle forecast anchored on the most recent window.
            if n >= window:
                print("  - Generating next-cycle forecast...")
                fc = self._forecast_from(raw_target, X_raw, cond_raw, n,
                                         n_mc=self.config.model.mc_dropout_samples)
                fig = plotter.plot_forecast_continuation(
                    history=raw_target, forecast_mean=fc['mean'],
                    forecast_lower=fc['q10'], forecast_upper=fc['q90'],
                    title=f"Next Solar Cycle Forecast - {name}",
                    save_path=plots_dir / "next_cycle_forecast.png")
                plt.close(fig)

        except Exception as e:
            import traceback
            print(f"Warning: Plot generation failed with error: {e}")
            traceback.print_exc()
            print("Continuing without plots...")

        print(f"  ✅ Plots saved to: {plots_dir}")
    
    def load_trained(self, run_dir: Union[str, Path]) -> 'Seq2SeqTrainer':
        """Load a previously trained model + scaler from an experiment directory."""
        run_dir = Path(run_dir)
        model = self.create_model()
        ckpt_path = run_dir / "best_model.pt"
        if not ckpt_path.exists():
            ckpt_path = run_dir / "final_model.pt"
        checkpoint = torch.load(ckpt_path, map_location=self.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        self.model = model

        self.data_scaler = RobustScaler.load_params(run_dir / "scaler.json")
        feature_scalers = run_dir / "feature_scalers.json"
        if feature_scalers.exists():
            self.multi_scaler = MultiChannelScaler.load_params(feature_scalers)
            self.data_scaler = self.multi_scaler.target_scaler
        cond_scaler = run_dir / "cond_scaler.json"
        if cond_scaler.exists():
            self.cond_scaler = RobustScaler.load_params(cond_scaler)

        # Train-era boundary (for labeling in-sample backtest panels honestly).
        self.train_end_date = None
        meta_path = run_dir / "run_meta.json"
        if meta_path.exists():
            import json as _json
            with open(meta_path) as f:
                self.train_end_date = _json.load(f).get('train_end_date')
        return self

    def _monthly_series(self, df) -> Tuple[np.ndarray, 'Any']:
        """Return the raw monthly target series and its DatetimeIndex (sentinel-safe)."""
        monthly = build_monthly_series(df, target_col=self.config.data.target_col,
                                       start_year=self.config.data.start_year)
        return monthly.values.astype(float), monthly.index

    def _monthly_arrays(self, df):
        """Return ``(raw_target (T,), X_raw (T,C) or None, cond_raw (T,) or None, dates)``.

        Rebuilds the raw monthly arrays (target, precursor channels, conditioning) for
        plotting/backtesting without needing prior state — works after load_trained().
        """
        precursor_cols = list(self.config.data.precursor_cols)
        if precursor_cols:
            _, X_raw, _, dates = prepare_multivariate_monthly_data(
                df, target_col=self.config.data.target_col,
                precursor_cols=precursor_cols, geomag_mask=self.config.data.geomag_mask,
                start_year=self.config.data.start_year, scaler_config=self._scaler_config())
            raw_target = X_raw[:, 0]
        else:
            raw_target, dates = self._monthly_series(df)
            X_raw = None
        cond_raw = (cycle_length_series(raw_target, self.config.data.prediction_horizon)
                    if self.config.data.use_terminator else None)
        return raw_target, X_raw, cond_raw, dates

    def _forecast_from(self, raw_target, X_raw, cond_raw, end_idx: int,
                       n_mc: int) -> Dict[str, np.ndarray]:
        """Forecast the horizon starting at ``end_idx`` from the preceding window."""
        window = self.config.data.input_window
        start = end_idx - window
        input_data = X_raw[start:end_idx] if X_raw is not None else raw_target[start:end_idx]
        cond = cond_raw[end_idx - 1] if cond_raw is not None else None
        return self.predict_with_uncertainty(input_data, cond=cond, n_mc_samples=n_mc)

    def backtest_cycles(self, df, n_panels: int = 4,
                        history_months: int = 660) -> List[Dict[str, Any]]:
        """Hindcast past solar cycles for validation.

        Detects recent cycle minima, and for each one forecasts the following
        horizon from history ending at that minimum, pairing the forecast (with
        MC-Dropout interval) against the actual observed cycle.
        """
        if self.model is None or self.data_scaler is None:
            raise ValueError("No trained model. Call train() or load_trained() first.")

        raw, X_raw, cond_raw, dates = self._monthly_arrays(df)
        window = self.config.data.input_window
        horizon = self.config.data.prediction_horizon

        # Cycle minima as forecast origins (shared with the terminator precursor logic).
        troughs = detect_cycle_minima(raw, horizon)
        origins = [t for t in troughs if t >= window and t + horizon <= len(raw)]
        origins = origins[-n_panels:]
        if not origins:
            raise ValueError("Not enough data to build backtest panels.")

        import pandas as pd
        train_end = getattr(self, 'train_end_date', None)
        train_end_ts = pd.Timestamp(train_end) if train_end else None

        panels = []
        for origin in origins:
            forecast = self._forecast_from(raw, X_raw, cond_raw, origin,
                                           n_mc=self.config.model.mc_dropout_samples)
            start = max(0, origin - history_months)
            # A panel is genuinely out-of-sample only if its whole target horizon
            # lies after the model's training era; anything else is (partly)
            # in-sample and must be labeled as such.
            in_sample = (train_end_ts is None
                         or dates[origin] <= train_end_ts)
            suffix = "" if not in_sample else "  [IN-SAMPLE]"
            panels.append({
                'history_dates': dates[start:origin],
                'history_values': raw[start:origin],
                'forecast_dates': dates[origin:origin + horizon],
                'pred_mean': forecast['mean'],
                'pred_lower': forecast['q10'],
                'pred_upper': forecast['q90'],
                'actual': raw[origin:origin + horizon],
                'origin_date': dates[origin],
                'in_sample': in_sample,
                'label': f"Forecast from {dates[origin].strftime('%Y-%m')}{suffix}",
            })
        return panels

    def predict_with_uncertainty(self, input_data: np.ndarray, cond=None,
                                n_mc_samples: int = 30) -> Dict[str, np.ndarray]:
        """Forecast from a raw input window with uncertainty.

        ``input_data`` is a raw window: ``(W,)`` univariate or ``(W, C)`` multivariate.
        ``cond`` is the raw (unscaled) conditioning scalar or None.

        Interval semantics: for quantile-head models, q10/q50/q90 come from the
        TRAINED quantile head (the pinball-calibrated aleatoric spread), which is
        what conformal calibration operates on. MC-Dropout samples of the median
        provide 'mean'/'std'/'samples' as an epistemic diagnostic only. For pure
        MSE models everything falls back to MC-Dropout statistics. All outputs
        are inverse-transformed to raw units and clipped at 0.
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")

        self.model.eval()
        input_data = np.asarray(input_data, dtype=float)

        # Scale the input window.
        if self.multi_scaler is not None:
            input_scaled = self.multi_scaler.transform(input_data)          # (W, C)
            input_tensor = torch.FloatTensor(input_scaled).unsqueeze(0).to(self.device)
        else:
            input_scaled = self.data_scaler.transform(input_data.reshape(-1, 1)).ravel()
            input_tensor = torch.FloatTensor(input_scaled).unsqueeze(0).unsqueeze(-1).to(self.device)

        # Normalize the conditioning scalar.
        cond_tensor = None
        if cond is not None and self.cond_scaler is not None:
            c = self.cond_scaler.transform(np.array([[float(cond)]])).ravel()
            cond_tensor = torch.FloatTensor(c).unsqueeze(0).to(self.device)

        # Deterministic forward pass: the trained quantile band (if any).
        with torch.no_grad():
            outputs = self.model(input_tensor, cond=cond_tensor)
        preds = outputs['quantile'] if 'quantile' in outputs else outputs['predictions']
        point, q10, q90 = self._split_quantile_outputs(preds)
        point, = point  # batch of 1

        # MC-Dropout samples of the median/point path (epistemic diagnostic).
        mc = self.model.mc_predict(input_tensor, cond=cond_tensor, n_samples=n_mc_samples)
        samples = np.stack([
            self._inverse_target(mc[0, :, i].cpu().numpy()) for i in range(mc.shape[-1])
        ], axis=-1)

        result = {
            'mean': np.mean(samples, axis=-1),
            'std': np.std(samples, axis=-1),
            'q50': point,
            'samples': samples,
        }
        if q10 is not None:
            result['q10'], result['q90'] = q10[0], q90[0]
        else:
            result['q10'] = np.percentile(samples, 10, axis=-1)
            result['q90'] = np.percentile(samples, 90, axis=-1)
        return result


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
    scaled_data, cond_series = trainer.prepare_data(dummy_data)
    print(f"Scaled data shape: {scaled_data.shape}, cond: {cond_series}")

    print("\n✅ Seq2Seq trainer implemented and tested!")