"""
Load the solar time series the trainer consumes.

Prefers a locally cached CSV (produced by ``solar.data.collection``); if none is
present it synthesises a realistic sunspot-like series so the pipeline is runnable
end-to-end without network access.
"""

from __future__ import annotations

from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd


def load_solar_data(data_dir: Union[str, Path] = "data") -> pd.DataFrame:
    """Return a DataFrame with at least ``date`` and ``sunspot_number`` columns.

    Resolution order: raw_multivariate_data.csv, then engineered_multivariate_data.csv,
    then a synthetic fallback.
    """
    data_dir = Path(data_dir)
    for name in ("raw_multivariate_data.csv", "engineered_multivariate_data.csv"):
        path = data_dir / name
        if path.exists():
            print(f"Loading {path}...")
            return pd.read_csv(path)

    print("No cached data found; generating synthetic solar data.")
    return _synthetic_solar_data(data_dir)


def _synthetic_solar_data(data_dir: Path, n_months: int = 3300) -> pd.DataFrame:
    """Generate a synthetic sunspot series (~275 years) for smoke tests."""
    rng = np.random.default_rng(42)
    dates = pd.date_range("1749-01-01", periods=n_months, freq="ME")
    t = np.arange(n_months)

    cycle = 50 * np.sin(2 * np.pi * t / 132) + 25 * np.sin(2 * np.pi * t / 66)
    trend = 0.01 * t
    noise = rng.normal(0, 15, n_months)
    sunspot_number = np.maximum(0, 80 + cycle + trend + noise)

    df = pd.DataFrame({"date": dates, "sunspot_number": sunspot_number})
    data_dir.mkdir(parents=True, exist_ok=True)
    out_path = data_dir / "synthetic_solar_data.csv"
    df.to_csv(out_path, index=False)
    print(f"Synthetic data saved to {out_path}")
    return df
