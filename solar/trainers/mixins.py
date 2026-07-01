"""
Training mixins for common functionality across different trainers.
Provides early stopping, scheduling, AMP, and checkpointing capabilities.
"""

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, ReduceLROnPlateau, StepLR
from pathlib import Path
from typing import Dict, Optional, Any, Union
import numpy as np
import json
import warnings


class EarlyStoppingMixin:
    """Early stopping functionality for training loops."""
    
    def __init__(self, patience: int = 15, min_delta: float = 1e-6, 
                 restore_best_weights: bool = True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        
        self.best_score = None
        self.best_epoch = 0
        self.patience_counter = 0
        self.best_weights = None
        self.early_stopped = False
    
    def step(self, current_score: float, model: nn.Module, epoch: int) -> bool:
        """
        Check if training should stop early.
        
        Args:
            current_score: Current validation score (lower is better)
            model: Model to save weights from
            epoch: Current epoch number
        
        Returns:
            True if training should stop, False otherwise
        """
        if self.best_score is None:
            self.best_score = current_score
            self.best_epoch = epoch
            if self.restore_best_weights:
                self.best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        
        elif current_score < self.best_score - self.min_delta:
            self.best_score = current_score
            self.best_epoch = epoch
            self.patience_counter = 0
            if self.restore_best_weights:
                self.best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        
        else:
            self.patience_counter += 1
        
        if self.patience_counter >= self.patience:
            self.early_stopped = True
            if self.restore_best_weights and self.best_weights is not None:
                # Move weights back to model's device
                device = next(model.parameters()).device
                best_weights_on_device = {k: v.to(device) for k, v in self.best_weights.items()}
                model.load_state_dict(best_weights_on_device)
            return True
        
        return False
    
    def state_dict(self) -> Dict:
        """Get early stopping state for checkpointing."""
        return {
            'best_score': self.best_score,
            'best_epoch': self.best_epoch,
            'patience_counter': self.patience_counter,
            'early_stopped': self.early_stopped
        }
    
    def load_state_dict(self, state_dict: Dict):
        """Load early stopping state from checkpoint."""
        self.best_score = state_dict.get('best_score')
        self.best_epoch = state_dict.get('best_epoch', 0)
        self.patience_counter = state_dict.get('patience_counter', 0)
        self.early_stopped = state_dict.get('early_stopped', False)


class SchedulerMixin:
    """Learning rate scheduling functionality."""
    
    def create_scheduler(self, optimizer: torch.optim.Optimizer, 
                        scheduler_type: str, scheduler_config: Dict) -> Optional[Any]:
        """Create learning rate scheduler."""
        if scheduler_type == "cosine_with_warmup":
            # CosineAnnealingWarmRestarts approximates cosine with warmup
            T_0 = scheduler_config.get('warmup_epochs', 5)
            T_mult = scheduler_config.get('T_mult', 2)
            return CosineAnnealingWarmRestarts(optimizer, T_0=T_0, T_mult=T_mult)
        
        elif scheduler_type == "reduce_on_plateau":
            patience = scheduler_config.get('patience', 5)
            factor = scheduler_config.get('factor', 0.5)
            return ReduceLROnPlateau(optimizer, mode='min', patience=patience, factor=factor)
        
        elif scheduler_type == "step":
            step_size = scheduler_config.get('step_size', 30)
            gamma = scheduler_config.get('gamma', 0.1)
            return StepLR(optimizer, step_size=step_size, gamma=gamma)
        
        elif scheduler_type == "none":
            return None
        
        else:
            warnings.warn(f"Unknown scheduler type: {scheduler_type}")
            return None
    
    def step_scheduler(self, scheduler: Any, scheduler_type: str, 
                      val_loss: Optional[float] = None):
        """Step the scheduler with appropriate arguments."""
        if scheduler is None:
            return
        
        if scheduler_type == "reduce_on_plateau":
            if val_loss is not None:
                scheduler.step(val_loss)
        else:
            scheduler.step()


class AMPMixin:
    """Automatic Mixed Precision training functionality."""
    
    def __init__(self, use_amp: bool = True):
        # AMP is only supported on CUDA, not on MPS or CPU
        self.use_amp = use_amp and torch.cuda.is_available()
        self.scaler = GradScaler() if self.use_amp else None
    
    def autocast_context(self):
        """Get autocast context manager."""
        if self.use_amp:
            return autocast()
        else:
            # No-op context manager
            from contextlib import nullcontext
            return nullcontext()
    
    def scale_and_step(self, loss: torch.Tensor, optimizer: torch.optim.Optimizer,
                      clip_grad_norm: Optional[float] = None, model: Optional[nn.Module] = None):
        """Scale loss and step optimizer with optional gradient clipping."""
        if self.scaler is not None:
            # AMP scaling
            self.scaler.scale(loss).backward()
            
            if clip_grad_norm is not None and model is not None:
                self.scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)
            
            self.scaler.step(optimizer)
            self.scaler.update()
        else:
            # Regular training
            loss.backward()
            
            if clip_grad_norm is not None and model is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)
            
            optimizer.step()


class CheckpointMixin:
    """Model checkpointing functionality."""
    
    def save_checkpoint(self, model: nn.Module, optimizer: torch.optim.Optimizer,
                       scheduler: Any, epoch: int, loss: float, 
                       checkpoint_path: Union[str, Path], 
                       additional_state: Optional[Dict] = None):
        """Save training checkpoint."""
        checkpoint_path = Path(checkpoint_path)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        
        checkpoint = {
            'epoch': int(epoch),
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            # Cast to a plain float so the checkpoint stays loadable under
            # torch.load's default weights_only=True (no numpy scalar globals).
            'loss': float(loss),
        }
        
        if scheduler is not None:
            checkpoint['scheduler_state_dict'] = scheduler.state_dict()
        
        if hasattr(self, 'scaler') and self.scaler is not None:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()
        
        if additional_state:
            checkpoint.update(additional_state)
        
        torch.save(checkpoint, checkpoint_path)
    
    def load_checkpoint(self, checkpoint_path: Union[str, Path],
                       model: nn.Module, optimizer: torch.optim.Optimizer,
                       scheduler: Any = None) -> Dict:
        """Load training checkpoint."""
        checkpoint_path = Path(checkpoint_path)
        
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if scheduler is not None and 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        if hasattr(self, 'scaler') and self.scaler is not None and 'scaler_state_dict' in checkpoint:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
        return checkpoint


class TeacherForcingMixin:
    """Teacher forcing ratio scheduling for sequence-to-sequence models."""
    
    def __init__(self, initial_ratio: float = 0.5, decay_rate: float = 0.95, 
                 min_ratio: float = 0.1):
        self.initial_ratio = initial_ratio
        self.current_ratio = initial_ratio
        self.decay_rate = decay_rate
        self.min_ratio = min_ratio
    
    def get_teacher_forcing_ratio(self, epoch: int) -> float:
        """Get current teacher forcing ratio."""
        return self.current_ratio
    
    def step_teacher_forcing(self):
        """Decay teacher forcing ratio."""
        self.current_ratio = max(
            self.min_ratio,
            self.current_ratio * self.decay_rate
        )
    
    def reset_teacher_forcing(self):
        """Reset teacher forcing ratio to initial value."""
        self.current_ratio = self.initial_ratio


class MetricsTrackingMixin:
    """Training metrics tracking and logging."""
    
    def __init__(self):
        self.metrics_history = {
            'train_loss': [],
            'val_loss': [],
            'lr': [],
            'teacher_forcing_ratio': []
        }
    
    def log_metrics(self, epoch: int, train_loss: float, val_loss: Optional[float] = None,
                   lr: float = None, **kwargs):
        """Log training metrics."""
        self.metrics_history['train_loss'].append(train_loss)
        
        if val_loss is not None:
            self.metrics_history['val_loss'].append(val_loss)
        
        if lr is not None:
            self.metrics_history['lr'].append(lr)
        
        # Log additional metrics
        for key, value in kwargs.items():
            if key not in self.metrics_history:
                self.metrics_history[key] = []
            self.metrics_history[key].append(value)
    
    def save_metrics(self, filepath: Union[str, Path]):
        """Save metrics history to JSON file."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert numpy types to Python types for JSON serialization
        serializable_metrics = {}
        for key, values in self.metrics_history.items():
            serializable_metrics[key] = [
                float(v) if isinstance(v, (np.floating, np.integer)) else v 
                for v in values
            ]
        
        with open(filepath, 'w') as f:
            json.dump(serializable_metrics, f, indent=2)
    
    def get_best_epoch(self, metric: str = 'val_loss', mode: str = 'min') -> int:
        """Get epoch with best metric value."""
        if metric not in self.metrics_history:
            return 0
        
        values = self.metrics_history[metric]
        if not values:
            return 0
        
        if mode == 'min':
            return int(np.argmin(values))
        else:
            return int(np.argmax(values))


class CombinedTrainerMixin(EarlyStoppingMixin, SchedulerMixin, AMPMixin, 
                          CheckpointMixin, TeacherForcingMixin, MetricsTrackingMixin):
    """Combined mixin with all training utilities."""
    
    def __init__(self, patience: int = 15, use_amp: bool = True,
                 teacher_forcing_ratio: float = 0.5, **kwargs):
        EarlyStoppingMixin.__init__(self, patience=patience)
        AMPMixin.__init__(self, use_amp=use_amp)
        TeacherForcingMixin.__init__(self, initial_ratio=teacher_forcing_ratio)
        MetricsTrackingMixin.__init__(self)


if __name__ == "__main__":
    # Test the mixins
    print("Testing training mixins...")
    
    # Test early stopping
    early_stopping = EarlyStoppingMixin(patience=3)
    
    # Simulate training with improving then deteriorating loss
    losses = [1.0, 0.8, 0.6, 0.7, 0.8, 0.9]
    model = nn.Linear(10, 1)  # Dummy model
    
    for epoch, loss in enumerate(losses):
        should_stop = early_stopping.step(loss, model, epoch)
        print(f"Epoch {epoch}: Loss {loss:.1f}, Should stop: {should_stop}")
        if should_stop:
            break
    
    print(f"Best epoch: {early_stopping.best_epoch}, Best score: {early_stopping.best_score}")
    
    # Test teacher forcing
    tf_scheduler = TeacherForcingMixin(initial_ratio=0.8, decay_rate=0.9, min_ratio=0.1)
    
    print("\nTeacher forcing schedule:")
    for epoch in range(10):
        ratio = tf_scheduler.get_teacher_forcing_ratio(epoch)
        print(f"Epoch {epoch}: TF ratio {ratio:.3f}")
        tf_scheduler.step_teacher_forcing()
    
    # Test metrics tracking
    metrics_tracker = MetricsTrackingMixin()
    
    for epoch in range(5):
        train_loss = 1.0 - epoch * 0.1
        val_loss = 1.1 - epoch * 0.08
        lr = 0.001 * (0.9 ** epoch)
        
        metrics_tracker.log_metrics(epoch, train_loss, val_loss, lr)
    
    print(f"\nBest validation epoch: {metrics_tracker.get_best_epoch('val_loss')}")
    
    print("\n✅ Training mixins implemented and tested!")