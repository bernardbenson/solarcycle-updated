"""
WaveNet + LSTM direct multi-step forecaster (Benson et al. 2020, corrected).

The architecture that won in "Forecasting Solar Cycle 25 Using Deep Neural
Networks" (Solar Physics 295:65): a WaveNet stem whose dilations double
CONTINUOUSLY (1, 2, 4, ..., 512 - receptive field 1 + (k-1)*(2^L - 1), i.e.
1024 months for 10 layers at kernel 2, covering the full 528-month / 4-cycle
input window), followed by a single unidirectional LSTM summary and a dense
one-shot emission of all 132 horizon months. Non-autoregressive: no feedback
loop, no teacher forcing.

Deliberate differences from the sibling WaveNetAttnSeq2Seq:
- Dilations never reset per stack (that variant's receptive field is only 91
  months - less than one solar cycle).
- No BatchNorm on the residual stream (it strips the absolute amplitude the
  model must predict, and is noisy at batch 32); weight-norm on the convs
  instead (config ``wavenet.norm``: weight | group | none).
- No transformer-query decoder: at d_model=128 it alone costs ~790K parameters,
  more than 3x this entire model, and with only ~19-25 independent solar cycles
  of training data it cannot earn them. The LSTM summary + linear heads emit
  the same one-shot multi-horizon forecast for ~150K.

Total: ~230K parameters at 32 channels / LSTM-128 (vs 2.46M for the seq2seq).
"""

from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.parametrizations import weight_norm


def _make_conv(in_ch: int, out_ch: int, kernel_size: int, dilation: int,
               norm: str) -> nn.Module:
    conv = nn.Conv1d(in_ch, out_ch, kernel_size=kernel_size, dilation=dilation)
    if norm == 'weight':
        conv = weight_norm(conv)
    return conv


class GatedResidualBlock(nn.Module):
    """WaveNet residual block: dilated causal conv, gated activation, skip out."""

    def __init__(self, channels: int, kernel_size: int, dilation: int,
                 dropout: float, norm: str = 'weight'):
        super().__init__()
        self.channels = channels
        self.pad = (kernel_size - 1) * dilation

        self.conv = _make_conv(channels, 2 * channels, kernel_size, dilation, norm)
        self.residual_conv = _make_conv(channels, channels, 1, 1, norm)
        self.skip_conv = _make_conv(channels, channels, 1, 1, norm)
        self.dropout = nn.Dropout(dropout)
        # GroupNorm normalizes per sample (no batch statistics, amplitude-safe
        # relative to BatchNorm); 'weight'/'none' leave the residual stream raw.
        self.group_norm = nn.GroupNorm(4, channels) if norm == 'group' else None

    def forward(self, x: torch.Tensor):
        """x: (batch, channels, seq_len) -> (residual, skip)."""
        conv_out = self.conv(F.pad(x, (self.pad, 0)))  # left-pad => causal
        filt, gate = conv_out[:, :self.channels], conv_out[:, self.channels:]
        gated = self.dropout(torch.tanh(filt) * torch.sigmoid(gate))

        residual = self.residual_conv(gated) + x
        if self.group_norm is not None:
            residual = self.group_norm(residual)
        skip = self.skip_conv(gated)
        return residual, skip


class WaveNetLSTMDirect(nn.Module):
    """WaveNet stem (continuously doubling dilations) + LSTM summary + dense heads.

    Heads (config ``head``):
    - ``mse``: point prediction per month -> predictions (batch, horizon)
    - ``quantile``: (batch, horizon, n_quantiles), monotone via sort
    - ``combined``: both
    """

    def __init__(self, config: Dict):
        super().__init__()

        wavenet_cfg = config.get('wavenet', {})
        input_dim = config.get('input_dim', 1)
        channels = wavenet_cfg.get('channels', 32)
        kernel_size = wavenet_cfg.get('kernel_size', 2)
        num_layers = wavenet_cfg.get('num_layers', 10)
        norm = wavenet_cfg.get('norm', 'weight')
        conv_dropout = wavenet_cfg.get('dropout', 0.25)
        dropout = config.get('dropout', 0.3)

        self.horizon = config.get('output_size', 132)
        self.receptive_field = 1 + (kernel_size - 1) * (2 ** num_layers - 1)

        self.input_conv = _make_conv(input_dim, channels, 1, 1, norm)
        self.blocks = nn.ModuleList([
            GatedResidualBlock(channels, kernel_size, dilation=2 ** i,
                               dropout=conv_dropout, norm=norm)
            for i in range(num_layers)
        ])
        self.skip_head = nn.Sequential(
            nn.ReLU(),
            _make_conv(channels, channels, 1, 1, norm),
            nn.ReLU(),
            _make_conv(channels, channels, 1, 1, norm),
        )

        hidden = config.get('decoder_lstm_hidden', 128)
        self.lstm = nn.LSTM(input_size=channels, hidden_size=hidden, batch_first=True)
        self.summary_dropout = nn.Dropout(dropout)

        # Optional precursor conditioning, added to the summary vector - a far
        # stronger coupling point than a uniform offset on attention queries.
        self.cond_dim = config.get('cond_dim', 0)
        if self.cond_dim > 0:
            self.cond_proj = nn.Linear(self.cond_dim, hidden)

        head_type = config.get('head', 'quantile')
        if head_type not in ('mse', 'quantile', 'combined'):
            raise ValueError(f"Unknown head type: {head_type}")
        self.head_type = head_type
        self.quantiles: Optional[List[float]] = (
            sorted(config.get('quantiles', [0.1, 0.5, 0.9]))
            if head_type in ('quantile', 'combined') else None
        )

        self.point_proj = nn.Linear(hidden, self.horizon)
        if self.quantiles is not None:
            self.quantile_proj = nn.Linear(hidden, self.horizon * len(self.quantiles))

    def _summarize(self, x: torch.Tensor, cond: Optional[torch.Tensor]) -> torch.Tensor:
        """x: (batch, seq_len, input_dim) -> summary (batch, hidden)."""
        h = self.input_conv(x.transpose(1, 2))
        skips = 0
        for block in self.blocks:
            h, skip = block(h)
            skips = skips + skip
        features = self.skip_head(skips).transpose(1, 2)  # (batch, seq_len, channels)

        _, (h_n, _) = self.lstm(features)
        summary = h_n[-1]  # (batch, hidden)
        if cond is not None and self.cond_dim > 0:
            summary = summary + self.cond_proj(cond)
        return self.summary_dropout(summary)

    def forward(self, x: torch.Tensor, cond: Optional[torch.Tensor] = None,
                **kwargs) -> Dict[str, torch.Tensor]:
        """x: (batch, seq_len, input_dim); cond: (batch, cond_dim) or None.

        Extra kwargs (targets, teacher_forcing_ratio) are absorbed for trainer
        interface compatibility; the model is non-autoregressive.
        """
        summary = self._summarize(x, cond)
        point = self.point_proj(summary)  # (batch, horizon)

        if self.head_type == 'mse':
            return {'predictions': point}

        q = self.quantile_proj(summary).view(-1, self.horizon, len(self.quantiles))
        quantile_preds, _ = torch.sort(q, dim=-1)  # q_low <= q_mid <= q_high

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
        """Median/point MC-Dropout samples: (batch, horizon, n_samples).

        Diagnostic epistemic spread only - calibrated intervals come from the
        quantile head plus conformal calibration, not from these samples.
        """
        self.eval()
        self.enable_mc_dropout()

        samples = []
        with torch.no_grad():
            for _ in range(n_samples):
                out = self.forward(x, cond=cond)
                if self.head_type == 'quantile':
                    median_idx = (self.quantiles.index(0.5) if 0.5 in self.quantiles
                                  else len(self.quantiles) // 2)
                    samples.append(out['predictions'][:, :, median_idx])
                else:
                    pred = out['predictions'] if self.head_type == 'mse' else out['mse']
                    samples.append(pred)
        return torch.stack(samples, dim=-1)


if __name__ == "__main__":
    torch.manual_seed(42)
    x = torch.randn(4, 528, 1)

    for head in ('mse', 'quantile', 'combined'):
        config = {
            'input_dim': 1, 'output_size': 132, 'head': head,
            'quantiles': [0.1, 0.5, 0.9], 'dropout': 0.3,
            'decoder_lstm_hidden': 128,
            'wavenet': {'channels': 32, 'kernel_size': 2, 'num_layers': 10,
                        'norm': 'weight', 'dropout': 0.25},
        }
        model = WaveNetLSTMDirect(config)
        out = model(x)
        params = sum(p.numel() for p in model.parameters())
        print(f"head={head:9s} params={params:,} RF={model.receptive_field} "
              f"pred={tuple(out['predictions'].shape)}")

    mc = model.mc_predict(x, n_samples=5)
    print(f"MC predictions shape: {tuple(mc.shape)}")
