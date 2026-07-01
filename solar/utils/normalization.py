"""
Robust normalization utilities for solar cycle data.
Provides variance-stabilizing transforms and robust scaling.
"""

import numpy as np
import json
from typing import Dict, Optional, Tuple, Any
from pathlib import Path
from sklearn.preprocessing import StandardScaler, RobustScaler as SklearnRobustScaler


class RobustScaler:
    """Enhanced robust scaler with configurable transforms and persistence."""
    
    def __init__(self, method: str = "robust", transform: str = "identity", 
                 quantile_range: Tuple[float, float] = (25.0, 75.0)):
        """
        Args:
            method: "robust", "standard", or "none"
            transform: "identity", "log1p", or "sqrt" 
            quantile_range: IQR range for robust scaling
        """
        self.method = method
        self.transform_type = transform  # Renamed to avoid conflict with transform() method
        self.quantile_range = quantile_range
        
        # Initialize scalers (accept either a plain string or a str-Enum).
        method_str = str(getattr(method, 'value', method)).lower()

        if method_str == "robust":
            self.scaler = SklearnRobustScaler(quantile_range=quantile_range)
        elif method_str == "standard":
            self.scaler = StandardScaler()
        else:
            self.scaler = None
            
        # Transform parameters
        self.transform_params = {}
        self.fitted = False
    
    def _apply_transform(self, data: np.ndarray) -> np.ndarray:
        """Apply variance-stabilizing transform."""
        transform_str = str(getattr(self.transform_type, 'value', self.transform_type)).lower()

        if transform_str == "log1p":
            # Ensure positive values for log transform
            if np.any(data < 0):
                # Shift to make all values positive
                shift = abs(np.min(data)) + 1
                self.transform_params['log1p_shift'] = shift
                data = data + shift
            return np.log1p(data)
        elif transform_str == "sqrt":
            # Ensure non-negative values for sqrt
            if np.any(data < 0):
                shift = abs(np.min(data))
                self.transform_params['sqrt_shift'] = shift
                data = data + shift
            return np.sqrt(data)
        else:  # identity
            return data
    
    def _inverse_transform(self, data: np.ndarray) -> np.ndarray:
        """Inverse variance-stabilizing transform."""
        transform_str = str(getattr(self.transform_type, 'value', self.transform_type)).lower()

        if transform_str == "log1p":
            data = np.expm1(data)
            if 'log1p_shift' in self.transform_params:
                data = data - self.transform_params['log1p_shift']
        elif transform_str == "sqrt":
            data = np.square(data)
            if 'sqrt_shift' in self.transform_params:
                data = data - self.transform_params['sqrt_shift']
        return data
    
    def fit(self, data: np.ndarray) -> 'RobustScaler':
        """Fit the scaler to data."""
        data = data.reshape(-1, 1) if data.ndim == 1 else data
        
        # Apply variance-stabilizing transform
        transformed_data = self._apply_transform(data)
        
        # Fit scaler if not "none"
        if self.scaler is not None:
            self.scaler.fit(transformed_data)
        
        self.fitted = True
        return self
    
    def transform(self, data: np.ndarray) -> np.ndarray:
        """Transform data using fitted parameters."""
        if not self.fitted:
            raise ValueError("Scaler not fitted. Call fit() first.")
        
        original_shape = data.shape
        data = data.reshape(-1, 1) if data.ndim == 1 else data
        
        # Apply variance-stabilizing transform
        transformed_data = self._apply_transform(data)
        
        # Apply scaling if not "none"
        if self.scaler is not None:
            scaled_data = self.scaler.transform(transformed_data)
        else:
            scaled_data = transformed_data
        
        return scaled_data.reshape(original_shape)
    
    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        """Inverse transform data back to original scale."""
        if not self.fitted:
            raise ValueError("Scaler not fitted. Call fit() first.")
        
        original_shape = data.shape
        data = data.reshape(-1, 1) if data.ndim == 1 else data
        
        # Inverse scaling if not "none"
        if self.scaler is not None:
            unscaled_data = self.scaler.inverse_transform(data)
        else:
            unscaled_data = data
        
        # Inverse variance-stabilizing transform
        original_data = self._inverse_transform(unscaled_data)
        
        return original_data.reshape(original_shape)
    
    def fit_transform(self, data: np.ndarray) -> np.ndarray:
        """Fit and transform data in one step."""
        self.fit(data)
        return self.transform(data)
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize fitted parameters to a plain dict."""
        if not self.fitted:
            raise ValueError("Cannot serialize an unfitted scaler.")

        params = {
            'method': self.method,
            'transform': self.transform_type,
            'quantile_range': self.quantile_range,
            'transform_params': self.transform_params,
            'fitted': self.fitted,
        }
        if self.scaler is not None:
            for attr in ('center_', 'scale_', 'mean_', 'var_'):
                if hasattr(self.scaler, attr):
                    params[attr] = getattr(self.scaler, attr).tolist()
        return params

    @classmethod
    def from_dict(cls, params: Dict[str, Any]) -> 'RobustScaler':
        """Reconstruct a fitted scaler from a dict produced by ``to_dict``."""
        scaler = cls(
            method=params['method'],
            transform=params['transform'],
            quantile_range=tuple(params['quantile_range']),
        )
        scaler.transform_params = params['transform_params']
        scaler.fitted = params['fitted']
        if scaler.scaler is not None and params['fitted']:
            for attr in ('center_', 'scale_', 'mean_', 'var_'):
                if attr in params:
                    setattr(scaler.scaler, attr, np.array(params[attr]))
        return scaler

    def save_params(self, filepath: Path) -> None:
        """Save scaler parameters to JSON file."""
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load_params(cls, filepath: Path) -> 'RobustScaler':
        """Load scaler parameters from JSON file."""
        with open(filepath, 'r') as f:
            return cls.from_dict(json.load(f))


class MultiChannelScaler:
    """Per-channel scaling for multivariate model inputs.

    Channel 0 is the target (sunspot); its scaler is the same ``RobustScaler`` used to
    inverse-transform predictions (``target_scaler``). Precursor channels are scaled
    independently and fit only on rows where the availability mask is 1, so zero-filled
    gaps do not corrupt the statistics. A mask channel (``is_mask=True``) passes through
    unchanged, and masked-out precursor rows are re-zeroed after scaling.
    """

    def __init__(self, channel_specs: list):
        self.names = [c['name'] for c in channel_specs]
        self.is_mask = [bool(c.get('is_mask', False)) for c in channel_specs]
        self.scalers = [
            None if c.get('is_mask') else RobustScaler(
                method=c.get('method', 'robust'),
                transform=c.get('transform', 'identity'),
                quantile_range=tuple(c.get('quantile_range', (25.0, 75.0))),
            )
            for c in channel_specs
        ]
        self._mask_idx = self.is_mask.index(True) if any(self.is_mask) else None

    @property
    def target_scaler(self) -> 'RobustScaler':
        return self.scalers[0]

    def _mask_col(self, X: np.ndarray) -> np.ndarray:
        if self._mask_idx is None:
            return np.ones(len(X))
        return X[:, self._mask_idx]

    def fit(self, X: np.ndarray) -> 'MultiChannelScaler':
        """Fit target on all rows; precursor channels on available (mask==1) rows only."""
        X = np.asarray(X, dtype=float)
        mask_col = self._mask_col(X)
        for c, scaler in enumerate(self.scalers):
            if scaler is None:
                continue
            if c == 0:
                scaler.fit(X[:, c])
            else:
                rows = mask_col > 0.5
                scaler.fit(X[rows, c] if rows.any() else X[:, c])
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Scale each channel; pass the mask through; zero masked-out precursor rows."""
        X = np.asarray(X, dtype=float)
        mask_col = self._mask_col(X)
        out = np.empty_like(X)
        for c, scaler in enumerate(self.scalers):
            if scaler is None:  # mask channel (0/1) passes through
                out[:, c] = X[:, c]
                continue
            scaled = scaler.transform(X[:, c])
            if c != 0:  # precursor: zero where unavailable
                scaled = np.where(mask_col > 0.5, scaled, 0.0)
            out[:, c] = scaled
        return out

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)

    def save_params(self, filepath: Path) -> None:
        channels = []
        for name, is_mask, scaler in zip(self.names, self.is_mask, self.scalers):
            if scaler is None:
                channels.append({'name': name, 'is_mask': True})
            else:
                channels.append({'name': name, 'is_mask': False, 'scaler': scaler.to_dict()})
        with open(filepath, 'w') as f:
            json.dump({'channels': channels}, f, indent=2)

    @classmethod
    def load_params(cls, filepath: Path) -> 'MultiChannelScaler':
        with open(filepath, 'r') as f:
            data = json.load(f)
        obj = cls.__new__(cls)
        obj.names, obj.is_mask, obj.scalers = [], [], []
        for ch in data['channels']:
            obj.names.append(ch['name'])
            if ch.get('is_mask'):
                obj.is_mask.append(True)
                obj.scalers.append(None)
            else:
                obj.is_mask.append(False)
                obj.scalers.append(RobustScaler.from_dict(ch['scaler']))
        obj._mask_idx = obj.is_mask.index(True) if any(obj.is_mask) else None
        return obj


def prepare_multivariate_monthly_data(df, target_col: str = 'sunspot_number',
                                      precursor_cols: Optional[list] = None,
                                      geomag_mask: bool = True,
                                      start_year: int = 1818,
                                      scaler_config: Optional[Dict[str, Any]] = None
                                      ) -> Tuple[np.ndarray, np.ndarray, MultiChannelScaler, Any]:
    """Build a scaled multivariate monthly input array with precursor channels.

    Returns ``(X_scaled (T,C), X_raw (T,C), MultiChannelScaler, DatetimeIndex)`` where
    channel 0 is the sunspot target. Precursors that start later than the sunspot record
    (e.g. geomagnetic from 1932) are zero-filled before that, with a binary availability
    mask appended as the last channel. The target (channel-0) scaler is identical to the
    univariate path, so prediction inverse-transform is unchanged.
    """
    import pandas as pd

    precursor_cols = precursor_cols or []
    scaler_config = scaler_config or {}

    d = df.copy()
    d['date'] = pd.to_datetime(d['date'])
    d = d.set_index('date')
    d = d[d.index.year >= start_year]

    # Target: monthly mean; -1 sentinels -> NaN; interpolate the few gaps for continuity.
    target = d[target_col].resample('ME').mean().replace(-1.0, np.nan)
    target = target.interpolate('linear').bfill().ffill()
    index = target.index

    columns = [target.values.astype(float)]
    specs = [{
        'name': target_col,
        'method': scaler_config.get('method', 'robust'),
        'transform': scaler_config.get('transform', 'sqrt'),
        'quantile_range': scaler_config.get('quantile_range', (25.0, 75.0)),
    }]

    available = np.ones(len(index), dtype=bool)
    for col in precursor_cols:
        series = d[col].resample('ME').mean().reindex(index)
        available &= series.notna().values
        columns.append(series.fillna(0.0).values.astype(float))
        specs.append({'name': col, 'method': 'robust', 'transform': 'log1p',
                      'quantile_range': (25.0, 75.0)})

    if geomag_mask and precursor_cols:
        columns.append(available.astype(float))
        specs.append({'name': 'precursor_mask', 'is_mask': True})

    X_raw = np.column_stack(columns)
    scaler = MultiChannelScaler(specs)
    X_scaled = scaler.fit_transform(X_raw)
    return X_scaled, X_raw, scaler, index


def create_solar_features(monthly_data: np.ndarray, window: int = 13) -> Dict[str, np.ndarray]:
    """
    Create additional features for solar cycle prediction.
    
    Args:
        monthly_data: Monthly sunspot numbers
        window: Window size for moving average (default: 13 months)
    
    Returns:
        Dictionary of feature arrays
    """
    n_months = len(monthly_data)
    features = {}
    
    # Month indices
    month_indices = np.arange(n_months)
    
    # Sine/cosine encoding for cycle position (approximate 11-year cycle)
    cycle_period = 132  # 11 years * 12 months
    features['cycle_sin'] = np.sin(2 * np.pi * month_indices / cycle_period)
    features['cycle_cos'] = np.cos(2 * np.pi * month_indices / cycle_period)
    
    # Calendar month sine/cosine (seasonal effects)
    calendar_period = 12
    features['calendar_sin'] = np.sin(2 * np.pi * (month_indices % 12) / calendar_period)
    features['calendar_cos'] = np.cos(2 * np.pi * (month_indices % 12) / calendar_period)
    
    # Running mean (smoothed trend)
    from scipy.ndimage import uniform_filter1d
    features['running_mean'] = uniform_filter1d(
        monthly_data.astype(float), size=window, mode='nearest'
    )
    
    # Rising/declining phase indicator
    # Use derivative of smoothed data
    smoothed = uniform_filter1d(monthly_data.astype(float), size=5, mode='nearest')
    derivative = np.gradient(smoothed)
    features['rising_phase'] = (derivative > 0).astype(float)
    
    # Lag features (previous values)
    for lag in [1, 3, 6, 12]:
        lag_feature = np.zeros_like(monthly_data, dtype=float)
        lag_feature[lag:] = monthly_data[:-lag]
        features[f'lag_{lag}'] = lag_feature
    
    return features


def prepare_enhanced_monthly_data(df, target_col: str = 'sunspot_number', 
                                start_year: int = 1749, 
                                scaler_config: Optional[Dict[str, Any]] = None) -> Tuple[np.ndarray, Dict[str, np.ndarray], RobustScaler]:
    """
    Enhanced data preparation with features and configurable scaling.
    
    Args:
        df: DataFrame with solar data
        target_col: Target column name
        start_year: Starting year for data
        scaler_config: Scaler configuration dict
    
    Returns:
        Tuple of (scaled_target, features_dict, fitted_scaler)
    """
    # Default scaler config
    if scaler_config is None:
        scaler_config = {
            'method': 'robust',
            'transform': 'sqrt',  # Variance-stabilizing for sunspot data
            'quantile_range': (25.0, 75.0)
        }
    
    # Prepare monthly data
    import pandas as pd
    df['date'] = pd.to_datetime(df['date'])
    df = df.set_index('date')
    df = df[df.index.year >= start_year]
    
    monthly_data = df[target_col].resample('ME').mean().dropna()
    monthly_values = monthly_data.values
    
    # Create features
    features = create_solar_features(monthly_values)
    
    # Fit and transform target
    scaler = RobustScaler(**scaler_config)
    scaled_target = scaler.fit_transform(monthly_values)
    
    return scaled_target, features, scaler


if __name__ == "__main__":
    # Test the robust scaler
    import matplotlib.pyplot as plt
    
    # Generate test data with outliers
    np.random.seed(42)
    data = np.random.normal(100, 20, 1000)
    data[50:60] = 300  # Add outliers
    
    # Test different scalers
    scalers = {
        'Standard': RobustScaler(method='standard', transform='identity'),
        'Robust': RobustScaler(method='robust', transform='identity'),
        'Robust + Log': RobustScaler(method='robust', transform='log1p'),
        'Robust + Sqrt': RobustScaler(method='robust', transform='sqrt')
    }
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()
    
    for i, (name, scaler) in enumerate(scalers.items()):
        scaled_data = scaler.fit_transform(data)
        reconstructed = scaler.inverse_transform(scaled_data)
        
        axes[i].hist(scaled_data, bins=50, alpha=0.7, label='Scaled')
        axes[i].set_title(f'{name}\nMean: {np.mean(scaled_data):.2f}, Std: {np.std(scaled_data):.2f}')
        axes[i].legend()
    
    plt.tight_layout()
    plt.savefig('scaler_comparison.png', dpi=150)
    plt.show()
    
    print("✅ Normalization utilities implemented and tested!")