"""
Load the solar time series the trainer consumes.

Prefers locally cached CSVs (produced by ``solar.data.collection``); if none is
present it synthesises a realistic sunspot-like series so the pipeline is runnable
end-to-end without network access.
"""

from __future__ import annotations

from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd


def load_solar_data(data_dir: Union[str, Path] = "data",
                    dataset: str = "ssn",
                    need_precursors: bool = False) -> pd.DataFrame:
    """Return a DataFrame with a ``date`` column plus the dataset's target column.

    dataset='ssn' (target ``sunspot_number``): prefers the curated SILSO monthly
    series (silso_monthly.csv, 1749-present); falls back to the daily
    raw_multivariate_data.csv (1818-present). Runs that need exogenous precursor
    channels (``need_precursors=True``) always use the daily multivariate CSV.

    dataset='area' (target ``sunspot_area``): the monthly total sunspot area
    series (sunspot_area_monthly.csv, 1874-present).
    """
    data_dir = Path(data_dir)

    if dataset == "area":
        path = data_dir / "sunspot_area_monthly.csv"
        if path.exists():
            print(f"Loading {path}...")
            return pd.read_csv(path)
        raise FileNotFoundError(
            f"{path} not found. Fetch it first: uv run python -m solar.data.collection"
        )

    candidates = ["raw_multivariate_data.csv", "engineered_multivariate_data.csv"] \
        if need_precursors else \
        ["silso_monthly.csv", "raw_multivariate_data.csv", "engineered_multivariate_data.csv"]

    for name in candidates:
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
