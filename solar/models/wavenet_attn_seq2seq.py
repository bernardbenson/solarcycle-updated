"""
WaveNet Attention-based Encoder-Decoder for Solar Cycle Prediction.
Combines WaveNet feature extraction with bidirectional LSTM encoding 
and attention-based LSTM decoding for probabilistic forecasting.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, List, Dict, Tuple


class WaveNetBlock(nn.Module):
    """Single WaveNet residual block with dilated convolutions."""
    
    def __init__(self, channels: int, kernel_size: int = 3, dilation: int = 1, dropout: float = 0.1):
        super().__init__()
        
        self.channels = channels
        self.kernel_size = kernel_size
        self.dilation = dilation
        
        # Dilated convolution
        self.conv = nn.Conv1d(
            channels, 2 * channels, kernel_size=kernel_size,
            dilation=dilation, padding=0  # No padding, we'll handle causality manually
        )
        
        # Gated activation
        self.gate_conv = nn.Conv1d(channels, channels, kernel_size=1)
        self.filter_conv = nn.Conv1d(channels, channels, kernel_size=1)
        
        # Residual and skip connections
        self.residual_conv = nn.Conv1d(channels, channels, kernel_size=1)
        self.skip_conv = nn.Conv1d(channels, channels, kernel_size=1)
        
        self.dropout = nn.Dropout(dropout)
        self.batch_norm = nn.BatchNorm1d(channels)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Input tensor of shape (batch_size, channels, seq_len)
        
        Returns:
            Tuple of (residual_output, skip_output)
        """
        # Causal padding for dilated convolution
        pad_size = (self.kernel_size - 1) * self.dilation
        x_padded = F.pad(x, (pad_size, 0), mode='constant', value=0)
        
        # Dilated convolution
        conv_out = self.conv(x_padded)
        
        # Ensure output length matches input
        if conv_out.size(-1) > x.size(-1):
            conv_out = conv_out[:, :, :x.size(-1)]
        
        # Split for gated activation
        filter_out = conv_out[:, :self.channels, :]
        gate_out = conv_out[:, self.channels:, :]
        
        # Gated activation
        gated = torch.tanh(filter_out) * torch.sigmoid(gate_out)
        gated = self.dropout(gated)
        
        # Residual connection
        residual = self.residual_conv(gated)
        residual = self.batch_norm(residual + x)
        
        # Skip connection
        skip = self.skip_conv(gated)
        
        return residual, skip


class WaveNetEncoder(nn.Module):
    """WaveNet encoder with multiple stacks of dilated convolutions."""
    
    def __init__(self, input_dim: int = 1, channels: int = 128, 
                 stacks: int = 3, layers_per_stack: int = 4,
                 kernel_size: int = 3, dropout: float = 0.2):
        super().__init__()
        
        self.input_dim = input_dim
        self.channels = channels
        self.stacks = stacks
        self.layers_per_stack = layers_per_stack
        
        # Input projection
        self.input_conv = nn.Conv1d(input_dim, channels, kernel_size=1)
        
        # WaveNet blocks
        self.blocks = nn.ModuleList()
        for stack in range(stacks):
            for layer in range(layers_per_stack):
                dilation = 2 ** layer
                self.blocks.append(
                    WaveNetBlock(channels, kernel_size, dilation, dropout)
                )
        
        # Output processing for skip connections
        self.skip_conv = nn.Conv1d(channels, channels, kernel_size=1)
        self.output_conv = nn.Conv1d(channels, channels, kernel_size=1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch_size, seq_len, input_dim)
        
        Returns:
            Encoded features of shape (batch_size, seq_len, channels)
        """
        # Convert to conv format: (batch_size, input_dim, seq_len)
        x = x.transpose(1, 2)
        
        # Input projection
        x = self.input_conv(x)
        
        # Apply WaveNet blocks and collect skip connections
        skip_connections = []
        for block in self.blocks:
            x, skip = block(x)
            skip_connections.append(skip)
        
        # Combine skip connections
        skip_sum = torch.stack(skip_connections, dim=0).sum(dim=0)
        output = F.relu(self.skip_conv(skip_sum))
        output = self.output_conv(output)
        
        # Convert back to sequence format: (batch_size, seq_len, channels)
        output = output.transpose(1, 2)
        
        return output


class ScaledDotProductAttention(nn.Module):
    """Scaled dot-product attention mechanism."""
    
    def __init__(self, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            query: Query tensor (batch_size, seq_len_q, d_model)
            key: Key tensor (batch_size, seq_len_k, d_model)
            value: Value tensor (batch_size, seq_len_v, d_model)
            mask: Optional attention mask
        
        Returns:
            Tuple of (attended_output, attention_weights)
        """
        # Compute attention scores
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.d_model)
        
        # Apply mask if provided
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # Softmax attention weights
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention to values
        attended = torch.matmul(attention_weights, value)
        
        return attended, attention_weights


class BahdanauAttention(nn.Module):
    """Bahdanau (additive) attention mechanism."""
    
    def __init__(self, hidden_size: int, encoder_size: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.encoder_size = encoder_size
        
        self.W_h = nn.Linear(hidden_size, hidden_size, bias=False)
        self.W_s = nn.Linear(encoder_size, hidden_size, bias=False)
        self.v = nn.Linear(hidden_size, 1, bias=False)
        
    def forward(self, hidden: torch.Tensor, encoder_outputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            hidden: Decoder hidden state (batch_size, hidden_size)
            encoder_outputs: Encoder outputs (batch_size, seq_len, encoder_size)
        
        Returns:
            Tuple of (context_vector, attention_weights)
        """
        seq_len = encoder_outputs.size(1)
        
        # Expand hidden state to match encoder sequence length
        hidden_expanded = hidden.unsqueeze(1).expand(-1, seq_len, -1)
        
        # Compute attention energies
        energy = torch.tanh(self.W_h(hidden_expanded) + self.W_s(encoder_outputs))
        attention_scores = self.v(energy).squeeze(-1)
        
        # Softmax to get attention weights
        attention_weights = F.softmax(attention_scores, dim=1)
        
        # Apply attention to encoder outputs
        context_vector = torch.sum(attention_weights.unsqueeze(-1) * encoder_outputs, dim=1)
        
        return context_vector, attention_weights


class AttentionDecoder(nn.Module):
    """LSTM decoder with attention mechanism."""
    
    def __init__(self, encoder_size: int, hidden_size: int, output_size: int,
                 attention_type: str = "scaled_dot", dropout: float = 0.2):
        super().__init__()
        
        self.encoder_size = encoder_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.attention_type = attention_type
        
        # LSTM decoder
        self.lstm = nn.LSTM(
            input_size=1 + encoder_size,  # Previous prediction + context
            hidden_size=hidden_size,
            batch_first=True
        )
        
        # Attention mechanism
        if attention_type == "scaled_dot":
            self.attention = ScaledDotProductAttention(encoder_size, dropout)
            self.query_proj = nn.Linear(hidden_size, encoder_size)
        elif attention_type == "bahdanau":
            self.attention = BahdanauAttention(hidden_size, encoder_size)
        else:
            raise ValueError(f"Unknown attention type: {attention_type}")
        
        # Output projection
        self.output_proj = nn.Linear(hidden_size + encoder_size, 1)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, encoder_outputs: torch.Tensor, target_length: int,
                targets: Optional[torch.Tensor] = None, teacher_forcing_ratio: float = 0.5) -> Tuple[torch.Tensor, List[torch.Tensor], torch.Tensor]:
        """
        Args:
            encoder_outputs: Encoder outputs (batch_size, seq_len, encoder_size)
            target_length: Length of sequence to generate
            targets: Ground truth targets for teacher forcing (batch_size, target_length)
            teacher_forcing_ratio: Probability of using teacher forcing
        
        Returns:
            Tuple of (predictions, attention_weights_list, hidden_features)
        """
        batch_size = encoder_outputs.size(0)
        device = encoder_outputs.device
        
        # Initialize hidden state
        hidden = torch.zeros(1, batch_size, self.hidden_size, device=device)
        cell = torch.zeros(1, batch_size, self.hidden_size, device=device)
        
        # Initialize first input (zeros)
        decoder_input = torch.zeros(batch_size, 1, 1, device=device)
        
        predictions = []
        attention_weights_list = []
        hidden_features = []
        
        for t in range(target_length):
            # Attention mechanism
            current_hidden = hidden.squeeze(0)  # Remove batch dimension for attention
            
            if self.attention_type == "scaled_dot":
                query = self.query_proj(current_hidden).unsqueeze(1)
                context, attention_weights = self.attention(
                    query, encoder_outputs, encoder_outputs
                )
                context = context.squeeze(1)
            else:  # bahdanau
                context, attention_weights = self.attention(current_hidden, encoder_outputs)
            
            attention_weights_list.append(attention_weights)
            
            # Concatenate input with context
            lstm_input = torch.cat([decoder_input, context.unsqueeze(1)], dim=-1)
            
            # LSTM step
            output, (hidden, cell) = self.lstm(lstm_input, (hidden, cell))
            
            # Generate prediction
            combined = torch.cat([output.squeeze(1), context], dim=-1)
            prediction = self.output_proj(self.dropout(combined))
            predictions.append(prediction)
            
            # Store hidden features for heads
            hidden_features.append(combined)
            
            # Prepare next input (teacher forcing or previous prediction)
            if targets is not None and torch.rand(1).item() < teacher_forcing_ratio:
                decoder_input = targets[:, t:t+1].unsqueeze(-1)
            else:
                decoder_input = prediction.unsqueeze(1)
        
        # Stack predictions and hidden features
        predictions = torch.stack(predictions, dim=1)  # (batch_size, target_length, 1)
        predictions = predictions.squeeze(-1)  # (batch_size, target_length)
        
        hidden_features = torch.stack(hidden_features, dim=1)  # (batch_size, target_length, hidden_size)
        
        return predictions, attention_weights_list, hidden_features


class WaveNetAttnSeq2Seq(nn.Module):
    """
    WaveNet Attention-based Encoder-Decoder for Solar Cycle Prediction.
    
    Architecture:
    - Encoder: WaveNet feature extractor + Bidirectional LSTM
    - Decoder: LSTM with attention mechanism
    - Head: MSE, Quantile, or Combined prediction head
    """
    
    def __init__(self, config: Dict):
        super().__init__()
        
        # Extract config parameters
        self.d_model = config.get('d_model', 128)
        self.output_size = config.get('output_size', 132)
        
        # WaveNet encoder config
        wavenet_config = config.get('wavenet', {})
        self.encoder = WaveNetEncoder(
            input_dim=config.get('input_dim', 1),
            channels=self.d_model,
            stacks=wavenet_config.get('stacks', 3),
            layers_per_stack=wavenet_config.get('layers_per_stack', 4),
            kernel_size=wavenet_config.get('kernel_size', 3),
            dropout=wavenet_config.get('dropout', 0.2)
        )
        
        # Bidirectional LSTM encoder
        encoder_hidden = config.get('encoder_bilstm_hidden', 128)
        self.encoder_lstm = nn.LSTM(
            input_size=self.d_model,
            hidden_size=encoder_hidden,
            bidirectional=True,
            batch_first=True,
            dropout=config.get('dropout', 0.2)
        )
        
        # Project bidirectional LSTM output
        self.encoder_proj = nn.Linear(2 * encoder_hidden, self.d_model)
        
        # Attention decoder
        decoder_hidden = config.get('decoder_lstm_hidden', 128)
        attention_type = config.get('attention', 'scaled_dot')
        self.decoder = AttentionDecoder(
            encoder_size=self.d_model,
            hidden_size=decoder_hidden,
            output_size=1,
            attention_type=attention_type,
            dropout=config.get('dropout', 0.2)
        )
        
        # Prediction head.
        # This decoder is autoregressive: it already emits one point prediction per
        # output timestep. Quantile/combined heads therefore need a per-timestep
        # projection from the decoder's hidden features (decoder hidden state +
        # attention context), not the summary->horizon heads in heads.py that the
        # feed-forward models (TCN, N-BEATS) use.
        head_type = config.get('head', 'mse')
        if head_type not in ('mse', 'quantile', 'combined'):
            raise ValueError(f"Unknown head type: {head_type}")
        quantiles = sorted(config.get('quantiles', [0.1, 0.5, 0.9]))

        self.head_type = head_type
        self.quantiles = quantiles if head_type in ('quantile', 'combined') else None
        if self.quantiles is not None:
            # hidden_features have dim (decoder_hidden + encoder context = d_model).
            self.quantile_proj = nn.Linear(decoder_hidden + self.d_model, len(quantiles))
        
    def forward(self, x: torch.Tensor, targets: Optional[torch.Tensor] = None,
                teacher_forcing_ratio: float = 0.5) -> Dict[str, torch.Tensor]:
        """
        Args:
            x: Input sequences (batch_size, seq_len, input_dim)
            targets: Target sequences for teacher forcing (batch_size, target_length)
            teacher_forcing_ratio: Probability of using teacher forcing
        
        Returns:
            Dictionary containing predictions and attention weights
        """
        # Encode input sequence
        wavenet_features = self.encoder(x)
        encoder_outputs, _ = self.encoder_lstm(wavenet_features)
        encoder_outputs = self.encoder_proj(encoder_outputs)
        
        # Decode with attention
        predictions, attention_weights, hidden_features = self.decoder(
            encoder_outputs, 
            self.output_size,
            targets, 
            teacher_forcing_ratio
        )
        
        # The decoder already produced one point prediction per output timestep.
        result = {'attention_weights': attention_weights, 'seq_predictions': predictions}

        if self.head_type == 'mse':
            result['predictions'] = predictions
            return result

        # Per-timestep quantile projection with a monotonicity (sort) constraint
        # so that q_low <= q_mid <= q_high at every step.
        quantile_preds = self.quantile_proj(hidden_features)  # (batch, seq_len, n_quantiles)
        quantile_preds, _ = torch.sort(quantile_preds, dim=-1)

        if self.head_type == 'quantile':
            result['predictions'] = quantile_preds
        else:  # combined: expose both point and quantile predictions
            result['mse'] = predictions
            result['quantile'] = quantile_preds
            result['predictions'] = predictions
        return result
    
    def enable_mc_dropout(self):
        """Enable MC-Dropout for uncertainty estimation."""
        for module in self.modules():
            if isinstance(module, nn.Dropout):
                module.train()
    
    def mc_predict(self, x: torch.Tensor, n_samples: int = 30) -> torch.Tensor:
        """
        Generate MC-Dropout predictions for uncertainty estimation.
        
        Args:
            x: Input sequences (batch_size, seq_len, input_dim)
            n_samples: Number of MC samples
        
        Returns:
            MC predictions (batch_size, target_length, n_samples)
        """
        self.eval()
        self.enable_mc_dropout()
        
        mc_predictions = []
        with torch.no_grad():
            for _ in range(n_samples):
                outputs = self.forward(x, teacher_forcing_ratio=0.0)
                if self.head_type == 'quantile':
                    # Use median quantile for MC-Dropout
                    median_idx = self.quantiles.index(0.5) if 0.5 in self.quantiles else len(self.quantiles) // 2
                    pred = outputs['predictions'][:, :, median_idx]
                else:
                    pred = outputs['predictions']
                mc_predictions.append(pred)
        
        return torch.stack(mc_predictions, dim=-1)


if __name__ == "__main__":
    # Test the WaveNet attention seq2seq model
    torch.manual_seed(42)
    
    # Model configuration
    config = {
        'input_dim': 1,
        'd_model': 128,
        'output_size': 132,
        'wavenet': {
            'stacks': 3,
            'layers_per_stack': 4,
            'kernel_size': 3,
            'channels': 128,
            'dropout': 0.2
        },
        'encoder_bilstm_hidden': 128,
        'decoder_lstm_hidden': 128,
        'attention': 'scaled_dot',
        'head': 'quantile',
        'quantiles': [0.1, 0.5, 0.9],
        'dropout': 0.2
    }
    
    # Test data
    batch_size = 4
    seq_len = 528
    target_len = 132
    
    x = torch.randn(batch_size, seq_len, 1)
    targets = torch.randn(batch_size, target_len)
    
    print("Testing WaveNet Attention Seq2Seq model...")
    
    # Initialize model
    model = WaveNetAttnSeq2Seq(config)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Forward pass
    outputs = model(x, targets, teacher_forcing_ratio=0.5)
    print(f"Predictions shape: {outputs['predictions'].shape}")
    print(f"Number of attention layers: {len(outputs['attention_weights'])}")
    
    # Test MC-Dropout
    mc_predictions = model.mc_predict(x, n_samples=10)
    print(f"MC predictions shape: {mc_predictions.shape}")
    
    print("\n✅ WaveNet Attention Seq2Seq model implemented and tested!")