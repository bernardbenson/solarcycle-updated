"""
Leak-free chronological splits for windowed multi-step forecasting.

Windows are identified by their *forecast origin* ``o``: the input is
``series[o - input_window : o]`` and the target ``series[o : o + horizon]``.

The train/validation boundary is a point ``t_split`` on the TIME axis (not the
window index): train windows must have their target entirely before ``t_split``
(``o + horizon <= t_split``); validation windows start at or after it
(``o >= t_split``). Train and validation targets are therefore disjoint by
construction. Validation *inputs* may reach back into the training era - that
is legitimate operational forecasting, not leakage.
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np


def time_axis_split(n_months: int, input_window: int, horizon: int,
                    val_ratio: float = 0.2, val_stride: int = 12
                    ) -> Tuple[List[int], List[int], int]:
    """Return ``(train_origins, val_origins, t_split)``.

    ``t_split`` reserves the last ``val_ratio`` fraction of the time axis (at
    least one horizon) for validation. Train origins are stride 1; validation
    origins are thinned to ``val_stride`` so the val metric isn't averaged over
    hundreds of nearly identical windows.
    """
    val_months = max(horizon, int(round(val_ratio * n_months)))
    t_split = n_months - val_months
    if t_split < input_window + horizon:
        raise ValueError(
            f"Series too short for input_window={input_window}, horizon={horizon}, "
            f"val_ratio={val_ratio}: n_months={n_months}"
        )

    train_origins = list(range(input_window, t_split - horizon + 1))
    val_origins = list(range(t_split, n_months - horizon + 1, val_stride))
    if not val_origins:
        raise ValueError("No validation windows fit; reduce val_ratio or horizon.")
    return train_origins, val_origins, t_split


def assert_no_target_overlap(train_origins: List[int], val_origins: List[int],
                             horizon: int) -> None:
    """Raise if any train target month is also a validation target month."""
    train_months = set()
    for o in train_origins:
        train_months.update(range(o, o + horizon))
    for o in val_origins:
        overlap = train_months.intersection(range(o, o + horizon))
        if overlap:
            raise AssertionError(
                f"Leakage: validation origin {o} shares {len(overlap)} target "
                f"months with training windows."
            )


def expanding_window_folds(n_months: int, input_window: int, horizon: int,
                           n_folds: int = 5, fold_stride: int = 0
                           ) -> List[Tuple[int, int]]:
    """Paper-style TimeSeriesSplit folds with a built-in embargo.

    Returns ``(t_split, val_origin)`` pairs: fold k trains on windows whose
    targets end before ``t_split`` and forecasts the horizon starting at
    ``t_split``. Folds are spaced evenly across the usable range (or every
    ``fold_stride`` months if given), most recent fold last.
    """
    first = input_window + horizon          # earliest t_split with >=1 train window
    last = n_months - horizon               # latest origin with a full actual horizon
    if last <= first:
        raise ValueError("Series too short for expanding-window folds.")
    if fold_stride:
        splits = list(range(last, first, -fold_stride))[:n_folds][::-1]
    else:
        splits = list(np.linspace(first, last, n_folds + 1).astype(int)[1:])
    return [(int(s), int(s)) for s in splits]
