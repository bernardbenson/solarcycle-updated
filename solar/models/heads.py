"""
Prediction heads for probabilistic and deterministic forecasting.
Supports MSE and quantile regression with pinball loss.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Optional, Tuple


class MSEHead(nn.Module):
    """Standard MSE prediction head for deterministic forecasting."""
    
    def __init__(self, input_dim: int, output_dim: int, dropout: float = 0.1):
        super().__init__()
        self.output_dim = output_dim
        
        self.head = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(input_dim // 2, output_dim)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch_size, input_dim)
        
        Returns:
            Predictions of shape (batch_size, output_dim)
        """
        return self.head(x)


class QuantileHead(nn.Module):
    """Quantile regression head for probabilistic forecasting."""
    
    def __init__(self, input_dim: int, output_dim: int, 
                 quantiles: List[float] = [0.1, 0.5, 0.9], 
                 dropout: float = 0.1):
        super().__init__()
        self.output_dim = output_dim
        self.quantiles = sorted(quantiles)
        self.n_quantiles = len(quantiles)
        
        # Shared feature extraction
        self.shared_layers = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        
        # Separate heads for each quantile
        self.quantile_heads = nn.ModuleList([
            nn.Linear(input_dim // 2, output_dim) 
            for _ in range(self.n_quantiles)
        ])
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch_size, input_dim)
        
        Returns:
            Predictions of shape (batch_size, output_dim, n_quantiles)
        """
        # Shared feature extraction
        shared_features = self.shared_layers(x)
        
        # Generate predictions for each quantile
        quantile_outputs = []
        for head in self.quantile_heads:
            quantile_outputs.append(head(shared_features))
        
        # Stack quantile predictions: (batch_size, output_dim, n_quantiles)
        outputs = torch.stack(quantile_outputs, dim=-1)
        
        # Ensure quantile ordering (monotonicity constraint)
        # Sort along quantile dimension to maintain q1 <= q2 <= q3
        outputs, _ = torch.sort(outputs, dim=-1)
        
        return outputs


def pinball_loss(predictions: torch.Tensor, targets: torch.Tensor, 
                quantiles: List[float], reduction: str = 'mean') -> torch.Tensor:
    """
    Compute pinball loss for quantile regression.
    
    Args:
        predictions: Predicted quantiles of shape (batch_size, seq_len, n_quantiles)
        targets: Ground truth targets of shape (batch_size, seq_len)
        quantiles: List of quantile levels
        reduction: 'mean', 'sum', or 'none'
    
    Returns:
        Pinball loss tensor
    """
    # Expand targets to match predictions shape
    # Handle case where targets already have extra dimension
    if targets.dim() == 3 and targets.shape[-1] == 1:
        # Remove the extra dimension and then add back with correct size
        targets = targets.squeeze(-1).unsqueeze(-1).expand_as(predictions)
    elif targets.dim() == 2:
        targets = targets.unsqueeze(-1).expand_as(predictions)
    else:
        # Already the right shape
        targets = targets.expand_as(predictions)
    
    # Convert quantiles to tensor
    quantiles_tensor = torch.tensor(quantiles, device=predictions.device, dtype=predictions.dtype)
    quantiles_tensor = quantiles_tensor.view(1, 1, -1).expand_as(predictions)
    
    # Compute pinball loss
    errors = targets - predictions
    loss = torch.where(
        errors >= 0,
        quantiles_tensor * errors,
        (quantiles_tensor - 1) * errors
    )
    
    if reduction == 'mean':
        return loss.mean()
    elif reduction == 'sum':
        return loss.sum()
    else:
        return loss


def quantile_loss_with_coverage(predictions: torch.Tensor, targets: torch.Tensor, 
                              quantiles: List[float]) -> Tuple[torch.Tensor, dict]:
    """
    Compute quantile loss with coverage statistics.
    
    Args:
        predictions: Predicted quantiles of shape (batch_size, seq_len, n_quantiles)
        targets: Ground truth targets of shape (batch_size, seq_len)
        quantiles: List of quantile levels
    
    Returns:
        Tuple of (loss, coverage_stats)
    """
    # Pinball loss
    loss = pinball_loss(predictions, targets, quantiles)
    
    # Coverage statistics
    coverage_stats = {}
    # Handle case where targets already have extra dimension
    if targets.dim() == 3 and targets.shape[-1] == 1:
        # Remove the extra dimension and then add back with correct size
        targets_expanded = targets.squeeze(-1).unsqueeze(-1).expand_as(predictions)
    elif targets.dim() == 2:
        targets_expanded = targets.unsqueeze(-1).expand_as(predictions)
    else:
        # Already the right shape
        targets_expanded = targets.expand_as(predictions)
    
    for i, q in enumerate(quantiles):
        # Empirical coverage (should be close to q)
        coverage = (targets_expanded[:, :, i] <= predictions[:, :, i]).float().mean()
        coverage_stats[f'coverage_q{q}'] = coverage.item()
    
    # Interval coverage for prediction intervals
    if len(quantiles) >= 2:
        # Assume first and last quantiles form prediction interval
        lower_q, upper_q = quantiles[0], quantiles[-1]
        interval_coverage = (
            (targets >= predictions[:, :, 0]) & 
            (targets <= predictions[:, :, -1])
        ).float().mean()
        
        nominal_coverage = upper_q - lower_q
        coverage_stats['interval_coverage'] = interval_coverage.item()
        coverage_stats['nominal_coverage'] = nominal_coverage
        coverage_stats['coverage_error'] = abs(interval_coverage.item() - nominal_coverage)
    
    return loss, coverage_stats


class CombinedHead(nn.Module):
    """Combined head that can output both MSE and quantile predictions."""
    
    def __init__(self, input_dim: int, output_dim: int, 
                 quantiles: Optional[List[float]] = None,
                 dropout: float = 0.1, mode: str = 'both'):
        super().__init__()
        self.mode = mode  # 'mse', 'quantile', or 'both'
        self.output_dim = output_dim
        
        if mode in ['mse', 'both']:
            self.mse_head = MSEHead(input_dim, output_dim, dropout)
        
        if mode in ['quantile', 'both'] and quantiles is not None:
            self.quantile_head = QuantileHead(input_dim, output_dim, quantiles, dropout)
            self.quantiles = quantiles
    
    def forward(self, x: torch.Tensor) -> dict:
        """
        Args:
            x: Input tensor of shape (batch_size, input_dim)
        
        Returns:
            Dictionary with 'mse' and/or 'quantile' predictions
        """
        outputs = {}
        
        if hasattr(self, 'mse_head'):
            outputs['mse'] = self.mse_head(x)
        
        if hasattr(self, 'quantile_head'):
            outputs['quantile'] = self.quantile_head(x)
        
        return outputs


def compute_combined_loss(outputs: dict, targets: torch.Tensor, 
                         quantiles: Optional[List[float]] = None,
                         mse_weight: float = 1.0, quantile_weight: float = 1.0) -> Tuple[torch.Tensor, dict]:
    """
    Compute combined loss for models with both MSE and quantile outputs.
    
    Args:
        outputs: Dictionary with 'mse' and/or 'quantile' predictions
        targets: Ground truth targets
        quantiles: List of quantile levels (if using quantile head)
        mse_weight: Weight for MSE loss
        quantile_weight: Weight for quantile loss
    
    Returns:
        Tuple of (total_loss, loss_components)
    """
    loss_components = {}
    total_loss = 0.0
    
    # MSE loss
    if 'mse' in outputs:
        mse_loss = F.mse_loss(outputs['mse'], targets)
        loss_components['mse_loss'] = mse_loss
        total_loss += mse_weight * mse_loss
    
    # Quantile loss
    if 'quantile' in outputs and quantiles is not None:
        quantile_loss, coverage_stats = quantile_loss_with_coverage(
            outputs['quantile'], targets, quantiles
        )
        loss_components['quantile_loss'] = quantile_loss
        loss_components.update(coverage_stats)
        total_loss += quantile_weight * quantile_loss
    
    loss_components['total_loss'] = total_loss
    return total_loss, loss_components


def extract_point_predictions(outputs: dict, quantiles: Optional[List[float]] = None) -> torch.Tensor:
    """
    Extract point predictions from model outputs.
    
    Args:
        outputs: Dictionary with 'mse' and/or 'quantile' predictions
        quantiles: List of quantile levels
    
    Returns:
        Point predictions tensor
    """
    if 'mse' in outputs:
        return outputs['mse']
    elif 'quantile' in outputs and quantiles is not None:
        # Use median (0.5 quantile) as point prediction
        if 0.5 in quantiles:
            median_idx = quantiles.index(0.5)
            return outputs['quantile'][:, :, median_idx]
        else:
            # Use middle quantile as approximation
            middle_idx = len(quantiles) // 2
            return outputs['quantile'][:, :, middle_idx]
    else:
        raise ValueError("No valid predictions found in outputs")


if __name__ == "__main__":
    # Test the prediction heads
    torch.manual_seed(42)
    
    batch_size = 16
    input_dim = 128
    output_dim = 132
    quantiles = [0.1, 0.5, 0.9]
    
    # Generate test data
    x = torch.randn(batch_size, input_dim)
    targets = torch.randn(batch_size, output_dim)
    
    print("Testing MSE Head...")
    mse_head = MSEHead(input_dim, output_dim)
    mse_predictions = mse_head(x)
    mse_loss = F.mse_loss(mse_predictions, targets)
    print(f"MSE predictions shape: {mse_predictions.shape}")
    print(f"MSE loss: {mse_loss.item():.4f}")
    
    print("\nTesting Quantile Head...")
    quantile_head = QuantileHead(input_dim, output_dim, quantiles)
    quantile_predictions = quantile_head(x)
    pinball = pinball_loss(quantile_predictions, targets, quantiles)
    print(f"Quantile predictions shape: {quantile_predictions.shape}")
    print(f"Pinball loss: {pinball.item():.4f}")
    
    print("\nTesting Combined Head...")
    combined_head = CombinedHead(input_dim, output_dim, quantiles, mode='both')
    combined_outputs = combined_head(x)
    combined_loss, loss_components = compute_combined_loss(
        combined_outputs, targets, quantiles
    )
    print(f"Combined outputs keys: {list(combined_outputs.keys())}")
    print(f"Combined loss: {combined_loss.item():.4f}")
    print(f"Loss components: {[f'{k}: {v.item():.4f}' if hasattr(v, 'item') else f'{k}: {v}' for k, v in loss_components.items()]}")
    
    print("\n✅ Prediction heads implemented and tested!")