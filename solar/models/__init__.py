"""
Solar cycle prediction models.

Available models:
- WaveNetAttnSeq2Seq: Attention-based encoder-decoder 
- TCNOnly: Temporal Convolutional Network baseline
- NBEATSx: N-BEATS with interpretable decomposition
"""

from .wavenet_attn_seq2seq import WaveNetAttnSeq2Seq
from .tcn_only import TCNOnly
from .nbeatsx import NBEATSx
from .heads import MSEHead, QuantileHead, CombinedHead
from . import baselines
from .precursor_ensemble import PrecursorEnsembleForecaster

# Register the learned precursor ensemble alongside the non-neural baselines so
# the LOCO harness (scripts/run_cv.py, scripts/hindcast_sc25.py) evaluates it via
# ALL_BASELINES. Done here to keep baselines.py free of a circular dependency.
if PrecursorEnsembleForecaster not in baselines.ALL_BASELINES:
    baselines.ALL_BASELINES.append(PrecursorEnsembleForecaster)

__all__ = [
    "WaveNetAttnSeq2Seq",
    "TCNOnly",
    "NBEATSx",
    "MSEHead",
    "QuantileHead",
    "CombinedHead",
    "PrecursorEnsembleForecaster",
]