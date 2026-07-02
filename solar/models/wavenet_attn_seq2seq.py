"""
WaveNet attention encoder-decoder for solar cycle prediction.

A WaveNet + bidirectional-LSTM encoder builds a long-range representation of the
input history; a non-autoregressive attention decoder then produces the entire
forecast horizon in parallel. Learned per-month "horizon queries" (with sinusoidal
positional encoding for cycle phase) cross-attend to the encoder memory, so every
output month is generated jointly. This avoids the mean-collapse and error
accumulation that a step-by-step autoregressive decoder suffers from on long,
high-amplitude cyclical signals.
"""

import math
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class WaveNetBlock(nn.Module):
    """Single WaveNet residual block with a dilated causal convolution."""

    def __init__(self, channels: int, kernel_size: int = 3, dilation: int = 1, dropout: float = 0.1):
        super().__init__()
        self.channels = channels
        self.kernel_size = kernel_size
        self.dilation = dilation

        self.conv = nn.Conv1d(channels, 2 * channels, kernel_size=kernel_size,
                              dilation=dilation, padding=0)  # causality handled manually
        self.residual_conv = nn.Conv1d(channels, channels, kernel_size=1)
        self.skip_conv = nn.Conv1d(channels, channels, kernel_size=1)
        self.dropout = nn.Dropout(dropout)
        self.batch_norm = nn.BatchNorm1d(channels)

    def forward(self, x: torch.Tensor):
        """x: (batch, channels, seq_len) -> (residual, skip)."""
        pad_size = (self.kernel_size - 1) * self.dilation
        conv_out = self.conv(F.pad(x, (pad_size, 0)))
        if conv_out.size(-1) > x.size(-1):
            conv_out = conv_out[:, :, :x.size(-1)]

        # Gated activation unit.
        filter_out = conv_out[:, :self.channels, :]
        gate_out = conv_out[:, self.channels:, :]
        gated = self.dropout(torch.tanh(filter_out) * torch.sigmoid(gate_out))

        residual = self.batch_norm(self.residual_conv(gated) + x)
        skip = self.skip_conv(gated)
        return residual, skip


class WaveNetEncoder(nn.Module):
    """WaveNet encoder: stacks of dilated causal convolutions with skip connections."""

    def __init__(self, input_dim: int = 1, channels: int = 128,
                 stacks: int = 3, layers_per_stack: int = 4,
                 kernel_size: int = 3, dropout: float = 0.2):
        super().__init__()
        self.input_conv = nn.Conv1d(input_dim, channels, kernel_size=1)
        self.blocks = nn.ModuleList([
            WaveNetBlock(channels, kernel_size, dilation=2 ** layer, dropout=dropout)
            for _ in range(stacks) for layer in range(layers_per_stack)
        ])
        self.skip_conv = nn.Conv1d(channels, channels, kernel_size=1)
        self.output_conv = nn.Conv1d(channels, channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, seq_len, input_dim) -> (batch, seq_len, channels)."""
        x = self.input_conv(x.transpose(1, 2))

        skip_connections = []
        for block in self.blocks:
            x, skip = block(x)
            skip_connections.append(skip)

        out = torch.stack(skip_connections, dim=0).sum(dim=0)
        out = self.output_conv(F.relu(self.skip_conv(out)))
        return out.transpose(1, 2)


def sinusoidal_positional_encoding(length: int, d_model: int) -> torch.Tensor:
    """Standard sinusoidal positional encoding of shape (1, length, d_model)."""
    pe = torch.zeros(length, d_model)
    position = torch.arange(length, dtype=torch.float32).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe.unsqueeze(0)


class HorizonAttentionDecoder(nn.Module):
    """Non-autoregressive decoder.

    A learned query per forecast month (plus positional encoding for cycle phase)
    cross-attends to the encoder memory through Transformer decoder layers, emitting
    features for all horizon months at once — no feedback loop, no teacher forcing.
    """

    def __init__(self, d_model: int, horizon: int, n_heads: int = 4,
                 n_layers: int = 2, dropout: float = 0.2, cond_dim: int = 0):
        super().__init__()
        self.horizon = horizon
        self.cond_dim = cond_dim

        # One learnable query per output month, initialised small.
        self.step_query = nn.Parameter(torch.randn(horizon, d_model) * 0.02)
        self.register_buffer('pos_enc', sinusoidal_positional_encoding(horizon, d_model))

        # Precursor conditioning: project the conditioning vector into d_model and add
        # it to every horizon query, so the whole forecast responds to the precursor.
        if cond_dim > 0:
            self.cond_proj = nn.Linear(cond_dim, d_model)

        layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=4 * d_model,
            dropout=dropout, batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(layer, num_layers=n_layers)

    def forward(self, memory: torch.Tensor, cond: Optional[torch.Tensor] = None) -> torch.Tensor:
        """memory: (batch, src_len, d_model) -> (batch, horizon, d_model)."""
        batch_size = memory.size(0)
        queries = self.step_query.unsqueeze(0).expand(batch_size, -1, -1) + self.pos_enc
        if cond is not None and self.cond_dim > 0:
            # (batch, d_model) -> (batch, 1, d_model), broadcast across the horizon.
            queries = queries + self.cond_proj(cond).unsqueeze(1)
        return self.decoder(queries, memory)


class WaveNetAttnSeq2Seq(nn.Module):
    """WaveNet + BiLSTM encoder with a parallel attention decoder.

    Heads:
    - ``mse``: a point prediction per month.
    - ``quantile``: one value per configured quantile per month (monotone via sort).
    - ``combined``: both of the above.
    """

    def __init__(self, config: Dict):
        super().__init__()

        self.d_model = config.get('d_model', 128)
        self.output_size = config.get('output_size', 132)
        dropout = config.get('dropout', 0.2)

        # WaveNet feature extractor.
        wavenet_config = config.get('wavenet', {})
        self.encoder = WaveNetEncoder(
            input_dim=config.get('input_dim', 1),
            channels=self.d_model,
            stacks=wavenet_config.get('stacks', 3),
            layers_per_stack=wavenet_config.get('layers_per_stack', 4),
            kernel_size=wavenet_config.get('kernel_size', 3),
            dropout=wavenet_config.get('dropout', dropout),
        )

        # Bidirectional LSTM over the WaveNet features (single layer -> no inter-layer dropout).
        encoder_hidden = config.get('encoder_bilstm_hidden', 128)
        self.encoder_lstm = nn.LSTM(
            input_size=self.d_model, hidden_size=encoder_hidden,
            bidirectional=True, batch_first=True,
        )
        self.encoder_proj = nn.Linear(2 * encoder_hidden, self.d_model)

        # Parallel horizon decoder (optionally precursor-conditioned).
        n_heads = _valid_head_count(self.d_model, config.get('decoder_heads', 4))
        self.cond_dim = config.get('cond_dim', 0)
        self.decoder = HorizonAttentionDecoder(
            d_model=self.d_model, horizon=self.output_size,
            n_heads=n_heads, n_layers=config.get('decoder_layers', 2), dropout=dropout,
            cond_dim=self.cond_dim,
        )

        # Prediction heads (applied per output month).
        head_type = config.get('head', 'mse')
        if head_type not in ('mse', 'quantile', 'combined'):
            raise ValueError(f"Unknown head type: {head_type}")
        self.head_type = head_type
        self.quantiles = sorted(config.get('quantiles', [0.1, 0.5, 0.9])) \
            if head_type in ('quantile', 'combined') else None

        self.point_proj = nn.Linear(self.d_model, 1)
        if self.quantiles is not None:
            self.quantile_proj = nn.Linear(self.d_model, len(self.quantiles))

    def _encode(self, x: torch.Tensor) -> torch.Tensor:
        features = self.encoder(x)
        encoder_outputs, _ = self.encoder_lstm(features)
        return self.encoder_proj(encoder_outputs)

    def forward(self, x: torch.Tensor, cond: Optional[torch.Tensor] = None,
                targets: Optional[torch.Tensor] = None,
                teacher_forcing_ratio: float = 0.0) -> Dict[str, torch.Tensor]:
        """x: (batch, seq_len, input_dim); cond: (batch, cond_dim) or None.

        ``targets``/``teacher_forcing_ratio`` are accepted for a uniform trainer
        interface but unused: the decoder is non-autoregressive.
        """
        memory = self._encode(x)
        decoded = self.decoder(memory, cond)  # (batch, horizon, d_model)

        point = self.point_proj(decoded).squeeze(-1)  # (batch, horizon)

        if self.head_type == 'mse':
            return {'predictions': point}

        # Per-month quantile projection, sorted to enforce q_low <= q_mid <= q_high.
        quantile_preds, _ = torch.sort(self.quantile_proj(decoded), dim=-1)

        if self.head_type == 'quantile':
            return {'predictions': quantile_preds}
        return {'predictions': point, 'mse': point, 'quantile': quantile_preds}

    def enable_mc_dropout(self):
        """Put dropout layers into train mode for MC-Dropout sampling."""
        for module in self.modules():
            if isinstance(module, nn.Dropout):
                module.train()

    def mc_predict(self, x: torch.Tensor, cond: Optional[torch.Tensor] = None,
                   n_samples: int = 30) -> torch.Tensor:
        """Return point-prediction MC-Dropout samples: (batch, horizon, n_samples)."""
        self.eval()
        self.enable_mc_dropout()

        samples = []
        with torch.no_grad():
            for _ in range(n_samples):
                outputs = self.forward(x, cond=cond)
                if self.head_type == 'quantile':
                    median_idx = (self.quantiles.index(0.5) if 0.5 in self.quantiles
                                  else len(self.quantiles) // 2)
                    pred = outputs['predictions'][:, :, median_idx]
                else:
                    pred = outputs['predictions']
                samples.append(pred)

        return torch.stack(samples, dim=-1)


def _valid_head_count(d_model: int, requested: int) -> int:
    """Largest attention-head count <= requested that divides d_model evenly."""
    heads = min(max(1, requested), d_model)
    while d_model % heads != 0:
        heads -= 1
    return heads


if __name__ == "__main__":
    torch.manual_seed(42)

    x = torch.randn(4, 528, 1)
    targets = torch.randn(4, 132)

    for head in ('mse', 'quantile', 'combined'):
        config = {
            'input_dim': 1, 'd_model': 128, 'output_size': 132,
            'wavenet': {'stacks': 3, 'layers_per_stack': 4, 'kernel_size': 3, 'dropout': 0.2},
            'encoder_bilstm_hidden': 128, 'decoder_layers': 2, 'decoder_heads': 4,
            'head': head, 'quantiles': [0.1, 0.5, 0.9], 'dropout': 0.2,
        }
        model = WaveNetAttnSeq2Seq(config)
        out = model(x, targets=targets)
        params = sum(p.numel() for p in model.parameters())
        print(f"head={head:9s} params={params:,} keys={list(out)} "
              f"pred={tuple(out['predictions'].shape)}")

    mc = model.mc_predict(x, n_samples=5)
    print(f"MC predictions shape: {tuple(mc.shape)}")
    print("\n✅ WaveNet attention seq2seq model implemented and tested!")
