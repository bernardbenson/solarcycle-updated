"""
Temporal Convolutional Network (TCN) baseline for solar cycle prediction.
Pure TCN implementation without LSTM or attention for ablation studies.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict
from .heads import MSEHead, QuantileHead, CombinedHead


class TemporalConvolution(nn.Module):
    """Single temporal convolution layer with dilated causal convolution."""
    
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3,
                 dilation: int = 1, dropout: float = 0.1):
        super().__init__()
        
        self.kernel_size = kernel_size
        self.dilation = dilation
        
        # Causal convolution
        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size=kernel_size,
            dilation=dilation, padding=0  # No padding, handle causality manually
        )
        
        # Normalization and activation
        self.batch_norm = nn.BatchNorm1d(out_channels)
        self.dropout = nn.Dropout(dropout)
        
        # Residual connection
        self.residual_conv = nn.Conv1d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else None
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch_size, in_channels, seq_len)
        
        Returns:
            Output tensor of shape (batch_size, out_channels, seq_len)
        """
        # Causal padding
        pad_size = (self.kernel_size - 1) * self.dilation
        x_padded = F.pad(x, (pad_size, 0), mode='constant', value=0)
        
        # Convolution
        out = self.conv(x_padded)
        
        # Ensure output length matches input
        if out.size(-1) > x.size(-1):
            out = out[:, :, :x.size(-1)]
        
        # Normalization and activation
        out = F.relu(self.batch_norm(out))
        out = self.dropout(out)
        
        # Residual connection
        if self.residual_conv is not None:
            residual = self.residual_conv(x)
        else:
            residual = x
        
        return out + residual


class TCNBlock(nn.Module):
    """TCN block with multiple dilated convolutions."""
    
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3,
                 num_layers: int = 4, dropout: float = 0.1):
        super().__init__()
        
        self.num_layers = num_layers
        
        # Create layers with exponentially increasing dilation
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            dilation = 2 ** i
            layer_in_channels = in_channels if i == 0 else out_channels
            
            self.layers.append(
                TemporalConvolution(
                    layer_in_channels, out_channels, kernel_size, dilation, dropout
                )
            )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply all layers in sequence."""
        for layer in self.layers:
            x = layer(x)
        return x


class TCNOnly(nn.Module):
    """
    Pure Temporal Convolutional Network for solar cycle prediction.
    
    Architecture:
    - Input projection
    - Multiple TCN blocks with increasing dilation
    - Global pooling or last timestep
    - Prediction head (MSE, Quantile, or Combined)
    """
    
    def __init__(self, config: Dict):
        super().__init__()
        
        # Extract config parameters
        self.input_dim = config.get('input_dim', 1)
        self.d_model = config.get('d_model', 128)
        self.output_size = config.get('output_size', 132)
        
        # TCN configuration
        tcn_config = config.get('tcn', {})
        self.num_blocks = tcn_config.get('num_blocks', 3)
        self.block_layers = tcn_config.get('layers_per_block', 4)
        self.kernel_size = tcn_config.get('kernel_size', 3)
        self.dropout = tcn_config.get('dropout', 0.1)
        
        # Input projection
        self.input_conv = nn.Conv1d(self.input_dim, self.d_model, kernel_size=1)
        
        # TCN blocks
        self.tcn_blocks = nn.ModuleList()
        for i in range(self.num_blocks):
            self.tcn_blocks.append(
                TCNBlock(
                    in_channels=self.d_model,
                    out_channels=self.d_model,
                    kernel_size=self.kernel_size,
                    num_layers=self.block_layers,
                    dropout=self.dropout
                )
            )
        
        # Global features extraction
        self.pooling_type = config.get('pooling', 'last')  # 'last', 'mean', 'max', 'attention'
        
        if self.pooling_type == 'attention':
            self.attention_weights = nn.Linear(self.d_model, 1)
        
        # Prediction head
        head_type = config.get('head', 'mse')
        quantiles = config.get('quantiles', [0.1, 0.5, 0.9])
        
        if head_type == 'mse':
            self.head = MSEHead(self.d_model, self.output_size, self.dropout)
        elif head_type == 'quantile':
            self.head = QuantileHead(self.d_model, self.output_size, quantiles, self.dropout)
        elif head_type == 'combined':
            self.head = CombinedHead(self.d_model, self.output_size, quantiles, self.dropout, 'both')
        else:
            raise ValueError(f"Unknown head type: {head_type}")
        
        self.head_type = head_type
        self.quantiles = quantiles if head_type in ['quantile', 'combined'] else None
    
    def forward(self, x: torch.Tensor, **kwargs) -> Dict[str, torch.Tensor]:
        """
        Args:
            x: Input sequences (batch_size, seq_len, input_dim)
        
        Returns:
            Dictionary containing predictions
        """
        # Convert to conv format: (batch_size, input_dim, seq_len)
        x = x.transpose(1, 2)
        
        # Input projection
        x = self.input_conv(x)
        
        # Apply TCN blocks
        for block in self.tcn_blocks:
            x = block(x)
        
        # Global feature extraction
        if self.pooling_type == 'last':
            # Use last timestep
            global_features = x[:, :, -1]  # (batch_size, d_model)
        
        elif self.pooling_type == 'mean':
            # Global average pooling
            global_features = torch.mean(x, dim=2)
        
        elif self.pooling_type == 'max':
            # Global max pooling
            global_features, _ = torch.max(x, dim=2)
        
        elif self.pooling_type == 'attention':
            # Attention-weighted pooling
            x_transposed = x.transpose(1, 2)  # (batch_size, seq_len, d_model)
            attention_scores = self.attention_weights(x_transposed)  # (batch_size, seq_len, 1)
            attention_weights = F.softmax(attention_scores, dim=1)
            global_features = torch.sum(attention_weights * x_transposed, dim=1)  # (batch_size, d_model)
        
        else:
            raise ValueError(f"Unknown pooling type: {self.pooling_type}")
        
        # Apply prediction head
        if self.head_type == 'combined':
            head_outputs = self.head(global_features)
            return head_outputs
        else:
            predictions = self.head(global_features)
            return {'predictions': predictions}
    
    def enable_mc_dropout(self):
        """Enable MC-Dropout for uncertainty estimation."""
        for module in self.modules():
            if isinstance(module, nn.Dropout):
                module.train()
    
    def mc_predict(self, x: torch.Tensor, cond=None, n_samples: int = 30) -> torch.Tensor:
        """
        Generate MC-Dropout predictions for uncertainty estimation.

        Args:
            x: Input sequences (batch_size, seq_len, input_dim)
            cond: Unused (accepted for a uniform trainer interface)
            n_samples: Number of MC samples
        
        Returns:
            MC predictions (batch_size, output_size, n_samples)
        """
        self.eval()
        self.enable_mc_dropout()
        
        mc_predictions = []
        with torch.no_grad():
            for _ in range(n_samples):
                outputs = self.forward(x)
                if self.head_type == 'quantile':
                    # Use median quantile for MC-Dropout
                    median_idx = self.quantiles.index(0.5) if 0.5 in self.quantiles else len(self.quantiles) // 2
                    pred = outputs['predictions'][:, :, median_idx]
                elif self.head_type == 'combined':
                    pred = outputs['mse'] if 'mse' in outputs else outputs['quantile'][:, :, len(self.quantiles)//2]
                else:
                    pred = outputs['predictions']
                mc_predictions.append(pred)
        
        return torch.stack(mc_predictions, dim=-1)


if __name__ == "__main__":
    # Test TCN model
    torch.manual_seed(42)
    
    # Model configuration
    config = {
        'input_dim': 1,
        'd_model': 128,
        'output_size': 132,
        'tcn': {
            'num_blocks': 3,
            'layers_per_block': 4,
            'kernel_size': 3,
            'dropout': 0.1
        },
        'pooling': 'attention',
        'head': 'quantile',
        'quantiles': [0.1, 0.5, 0.9]
    }
    
    # Test data
    batch_size = 4
    seq_len = 528
    
    x = torch.randn(batch_size, seq_len, 1)
    
    print("Testing TCN model...")
    
    # Initialize model
    model = TCNOnly(config)
    print(f"TCN parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Forward pass
    outputs = model(x)
    print(f"Output keys: {list(outputs.keys())}")
    print(f"Predictions shape: {outputs['predictions'].shape}")
    
    # Test MC-Dropout
    mc_predictions = model.mc_predict(x, n_samples=10)
    print(f"MC predictions shape: {mc_predictions.shape}")

    print("\n✅ TCN baseline model implemented and tested!")