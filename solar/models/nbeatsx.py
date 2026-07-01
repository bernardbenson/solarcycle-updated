"""
N-BEATS inspired model for solar cycle prediction.
Implements basis expansion and interpretable decomposition for cyclical data.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from typing import Dict, List, Optional, Tuple
from .heads import MSEHead, QuantileHead, CombinedHead


class LinearBasis(nn.Module):
    """Linear trend basis functions."""
    
    def __init__(self, backcast_length: int, forecast_length: int, basis_size: int = 3):
        super().__init__()
        self.backcast_length = backcast_length
        self.forecast_length = forecast_length
        self.basis_size = basis_size
        
        # Polynomial basis (1, t, t^2, ...)
        self.register_buffer('backcast_basis', self._polynomial_basis(backcast_length, basis_size))
        self.register_buffer('forecast_basis', self._polynomial_basis(forecast_length, basis_size))
    
    def _polynomial_basis(self, length: int, degree: int) -> torch.Tensor:
        """Create polynomial basis functions."""
        t = torch.linspace(0, 1, length).unsqueeze(0)  # (1, length)
        basis = []
        for i in range(degree):
            basis.append(t ** i)
        return torch.cat(basis, dim=0)  # (degree, length)
    
    def forward(self, theta: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            theta: Basis coefficients (batch_size, basis_size)
        
        Returns:
            Tuple of (backcast, forecast)
        """
        # Expand theta to match basis dimensions
        backcast = torch.matmul(theta, self.backcast_basis)  # (batch_size, backcast_length)
        forecast = torch.matmul(theta, self.forecast_basis)  # (batch_size, forecast_length)
        
        return backcast, forecast


class SeasonalBasis(nn.Module):
    """Seasonal/cyclical basis functions for solar cycles."""
    
    def __init__(self, backcast_length: int, forecast_length: int, 
                 basis_size: int = 6, cycle_periods: List[float] = None):
        super().__init__()
        self.backcast_length = backcast_length
        self.forecast_length = forecast_length
        self.basis_size = basis_size
        
        # Default solar cycle periods (in months)
        if cycle_periods is None:
            cycle_periods = [132, 66, 44]  # 11-year, 5.5-year, and 3.7-year cycles
        
        self.cycle_periods = cycle_periods
        
        # Create basis functions
        self.register_buffer('backcast_basis', self._seasonal_basis(backcast_length))
        self.register_buffer('forecast_basis', self._seasonal_basis(forecast_length))
    
    def _seasonal_basis(self, length: int) -> torch.Tensor:
        """Create seasonal basis functions."""
        t = torch.arange(length, dtype=torch.float32)
        basis = []
        
        for period in self.cycle_periods:
            # Sine and cosine components for each period
            basis.append(torch.sin(2 * math.pi * t / period))
            basis.append(torch.cos(2 * math.pi * t / period))
        
        # Truncate or pad to desired basis size
        basis_tensor = torch.stack(basis, dim=0)  # (n_harmonics, length)
        
        if len(basis) > self.basis_size:
            basis_tensor = basis_tensor[:self.basis_size]
        elif len(basis) < self.basis_size:
            # Pad with additional harmonics
            remaining = self.basis_size - len(basis)
            for i in range(remaining):
                # Add higher frequency components
                freq = 2 ** (i + 1)
                if i % 2 == 0:
                    basis_tensor = torch.cat([basis_tensor, torch.sin(2 * math.pi * t / (132 / freq)).unsqueeze(0)])
                else:
                    basis_tensor = torch.cat([basis_tensor, torch.cos(2 * math.pi * t / (132 / freq)).unsqueeze(0)])
        
        return basis_tensor[:self.basis_size]
    
    def forward(self, theta: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            theta: Basis coefficients (batch_size, basis_size)
        
        Returns:
            Tuple of (backcast, forecast)
        """
        backcast = torch.matmul(theta, self.backcast_basis)
        forecast = torch.matmul(theta, self.forecast_basis)
        
        return backcast, forecast


class GenericBasis(nn.Module):
    """Generic learnable basis functions."""
    
    def __init__(self, backcast_length: int, forecast_length: int, basis_size: int = 10):
        super().__init__()
        self.basis_size = basis_size
        
        # Learnable basis functions
        self.backcast_basis = nn.Linear(basis_size, backcast_length, bias=False)
        self.forecast_basis = nn.Linear(basis_size, forecast_length, bias=False)
    
    def forward(self, theta: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            theta: Basis coefficients (batch_size, basis_size)
        
        Returns:
            Tuple of (backcast, forecast)
        """
        backcast = self.backcast_basis(theta)
        forecast = self.forecast_basis(theta)
        
        return backcast, forecast


class NBeatsBlock(nn.Module):
    """N-BEATS block with basis expansion."""
    
    def __init__(self, input_size: int, hidden_size: int, basis_module: nn.Module,
                 num_layers: int = 4, dropout: float = 0.1):
        super().__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.basis_module = basis_module
        
        # Fully connected stack
        layers = []
        layers.append(nn.Linear(input_size, hidden_size))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout))
        
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
        
        self.fc_stack = nn.Sequential(*layers)
        
        # Theta projection (coefficients for basis functions)
        self.theta_layer = nn.Linear(hidden_size, basis_module.basis_size)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Input time series (batch_size, input_size)
        
        Returns:
            Tuple of (backcast, forecast)
        """
        # Process through FC stack
        h = self.fc_stack(x)
        
        # Generate basis coefficients
        theta = self.theta_layer(h)
        
        # Generate backcast and forecast using basis functions
        backcast, forecast = self.basis_module(theta)
        
        return backcast, forecast


class NBEATSx(nn.Module):
    """
    N-BEATS inspired model for solar cycle prediction.
    
    Features:
    - Trend and seasonal decomposition
    - Interpretable basis functions
    - Hierarchical residual learning
    """
    
    def __init__(self, config: Dict):
        super().__init__()
        
        # Extract config
        self.input_size = config.get('input_window', 528)
        self.output_size = config.get('output_size', 132)
        self.hidden_size = config.get('d_model', 128)
        
        # N-BEATS config
        nbeats_config = config.get('nbeats', {})
        self.num_trend_blocks = nbeats_config.get('trend_blocks', 2)
        self.num_seasonal_blocks = nbeats_config.get('seasonal_blocks', 2)
        self.num_generic_blocks = nbeats_config.get('generic_blocks', 2)
        self.basis_size = nbeats_config.get('basis_size', 8)
        self.dropout = nbeats_config.get('dropout', 0.1)
        
        # Create blocks
        self.blocks = nn.ModuleList()
        
        # Trend blocks
        for _ in range(self.num_trend_blocks):
            basis = LinearBasis(self.input_size, self.output_size, self.basis_size)
            block = NBeatsBlock(self.input_size, self.hidden_size, basis, dropout=self.dropout)
            self.blocks.append(block)
        
        # Seasonal blocks
        for _ in range(self.num_seasonal_blocks):
            basis = SeasonalBasis(self.input_size, self.output_size, self.basis_size)
            block = NBeatsBlock(self.input_size, self.hidden_size, basis, dropout=self.dropout)
            self.blocks.append(block)
        
        # Generic blocks
        for _ in range(self.num_generic_blocks):
            basis = GenericBasis(self.input_size, self.output_size, self.basis_size)
            block = NBeatsBlock(self.input_size, self.hidden_size, basis, dropout=self.dropout)
            self.blocks.append(block)
        
        # Optional prediction head for additional processing
        head_type = config.get('head', 'mse')
        quantiles = config.get('quantiles', [0.1, 0.5, 0.9])
        
        # A prediction head is required for quantile/combined outputs (the raw
        # basis-expansion forecast is point-only), so enable it automatically.
        self.use_head = config.get('use_prediction_head', False) or head_type in ('quantile', 'combined')
        if self.use_head:
            if head_type == 'mse':
                self.head = MSEHead(self.output_size, self.output_size, self.dropout)
            elif head_type == 'quantile':
                self.head = QuantileHead(self.output_size, self.output_size, quantiles, self.dropout)
            elif head_type == 'combined':
                self.head = CombinedHead(self.output_size, self.output_size, quantiles, self.dropout, 'both')
            else:
                self.use_head = False
        
        self.head_type = head_type if self.use_head else 'mse'
        self.quantiles = quantiles if head_type in ['quantile', 'combined'] and self.use_head else None
    
    def forward(self, x: torch.Tensor, **kwargs) -> Dict[str, torch.Tensor]:
        """
        Args:
            x: Input sequences (batch_size, seq_len, input_dim)
        
        Returns:
            Dictionary containing predictions and decomposition
        """
        # Flatten input for N-BEATS processing
        batch_size = x.size(0)
        x_flat = x.view(batch_size, -1)  # (batch_size, input_size)
        
        # Initialize residual and forecast
        residual = x_flat
        total_forecast = torch.zeros(batch_size, self.output_size, device=x.device)
        
        # Store components for interpretability
        trend_forecast = torch.zeros(batch_size, self.output_size, device=x.device)
        seasonal_forecast = torch.zeros(batch_size, self.output_size, device=x.device)
        generic_forecast = torch.zeros(batch_size, self.output_size, device=x.device)
        
        # Process through blocks
        for i, block in enumerate(self.blocks):
            backcast, forecast = block(residual)
            
            # Update residual (doubly residual)
            residual = residual - backcast
            
            # Accumulate forecast
            total_forecast = total_forecast + forecast
            
            # Track components
            if i < self.num_trend_blocks:
                trend_forecast = trend_forecast + forecast
            elif i < self.num_trend_blocks + self.num_seasonal_blocks:
                seasonal_forecast = seasonal_forecast + forecast
            else:
                generic_forecast = generic_forecast + forecast
        
        # Optional prediction head
        if self.use_head:
            if self.head_type == 'combined':
                head_outputs = self.head(total_forecast)
                return {
                    **head_outputs,
                    'trend': trend_forecast,
                    'seasonal': seasonal_forecast,
                    'generic': generic_forecast,
                    'residual': residual
                }
            else:
                final_predictions = self.head(total_forecast)
                return {
                    'predictions': final_predictions,
                    'trend': trend_forecast,
                    'seasonal': seasonal_forecast,
                    'generic': generic_forecast,
                    'residual': residual
                }
        else:
            return {
                'predictions': total_forecast,
                'trend': trend_forecast,
                'seasonal': seasonal_forecast,
                'generic': generic_forecast,
                'residual': residual
            }
    
    def enable_mc_dropout(self):
        """Enable MC-Dropout for uncertainty estimation."""
        for module in self.modules():
            if isinstance(module, nn.Dropout):
                module.train()
    
    def mc_predict(self, x: torch.Tensor, n_samples: int = 30) -> torch.Tensor:
        """Generate MC-Dropout predictions."""
        self.eval()
        self.enable_mc_dropout()
        
        mc_predictions = []
        with torch.no_grad():
            for _ in range(n_samples):
                outputs = self.forward(x)
                if self.head_type == 'quantile' and self.quantiles:
                    median_idx = self.quantiles.index(0.5) if 0.5 in self.quantiles else len(self.quantiles) // 2
                    pred = outputs['predictions'][:, :, median_idx]
                elif self.head_type == 'combined':
                    pred = outputs.get('mse', outputs['predictions'])
                else:
                    pred = outputs['predictions']
                mc_predictions.append(pred)
        
        return torch.stack(mc_predictions, dim=-1)
    
    def get_decomposition(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Get interpretable decomposition of the forecast."""
        with torch.no_grad():
            outputs = self.forward(x)
            return {
                'trend': outputs['trend'],
                'seasonal': outputs['seasonal'], 
                'generic': outputs['generic'],
                'total': outputs['predictions']
            }


if __name__ == "__main__":
    # Test N-BEATS model
    torch.manual_seed(42)
    
    # Model configuration
    config = {
        'input_window': 528,
        'output_size': 132,
        'd_model': 128,
        'nbeats': {
            'trend_blocks': 2,
            'seasonal_blocks': 2,
            'generic_blocks': 1,
            'basis_size': 8,
            'dropout': 0.1
        },
        'head': 'mse',
        'use_prediction_head': False
    }
    
    # Test data
    batch_size = 4
    seq_len = 528
    
    x = torch.randn(batch_size, seq_len, 1)
    
    print("Testing N-BEATS model...")
    
    # Initialize model
    model = NBEATSx(config)
    print(f"N-BEATS parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Forward pass
    outputs = model(x)
    print(f"Output keys: {list(outputs.keys())}")
    print(f"Predictions shape: {outputs['predictions'].shape}")
    print(f"Trend shape: {outputs['trend'].shape}")
    print(f"Seasonal shape: {outputs['seasonal'].shape}")
    
    # Test decomposition
    decomposition = model.get_decomposition(x)
    print(f"Decomposition keys: {list(decomposition.keys())}")
    
    # Test MC-Dropout
    mc_predictions = model.mc_predict(x, n_samples=10)
    print(f"MC predictions shape: {mc_predictions.shape}")
    
    # Test individual basis functions
    print("\nTesting basis functions...")
    
    linear_basis = LinearBasis(100, 50, 3)
    theta = torch.randn(2, 3)
    backcast, forecast = linear_basis(theta)
    print(f"Linear basis - Backcast: {backcast.shape}, Forecast: {forecast.shape}")
    
    seasonal_basis = SeasonalBasis(100, 50, 6)
    theta = torch.randn(2, 6)
    backcast, forecast = seasonal_basis(theta)
    print(f"Seasonal basis - Backcast: {backcast.shape}, Forecast: {forecast.shape}")
    
    print("\n✅ N-BEATS model implemented and tested!")