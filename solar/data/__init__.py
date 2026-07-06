"""Data collection and loading for solar cycle prediction."""

from .collection import SolarDataCollector
from .loading import load_solar_data

__all__ = ["SolarDataCollector", "load_solar_data"]
