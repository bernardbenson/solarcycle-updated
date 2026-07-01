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

__all__ = [
    "WaveNetAttnSeq2Seq", 
    "TCNOnly", 
    "NBEATSx",
    "MSEHead", 
    "QuantileHead", 
    "CombinedHead"
]