"""
Solar Cycle Prediction Package

Enhanced WaveNet attention-based models for solar cycle forecasting with uncertainty quantification.
"""

__version__ = "2.0.0"
__author__ = "Solar Cycle Prediction Team"

from . import data, models, trainers, utils

__all__ = ["data", "models", "trainers", "utils"]