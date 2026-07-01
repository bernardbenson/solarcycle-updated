"""
Peak detection and conformal prediction intervals for solar cycle analysis.
Provides metrics for peak timing and magnitude accuracy with uncertainty quantification.
"""

import numpy as np
from typing import Tuple, Dict, List, Optional, Union
from scipy.signal import find_peaks, savgol_filter
from scipy.ndimage import uniform_filter1d
import warnings


class SolarCyclePeakDetector:
    """Robust peak detection for solar cycle data."""
    
    def __init__(self, smoothing_window: int = 13, prominence_factor: float = 0.3):
        """
        Args:
            smoothing_window: Window size for smoothing (default: 13 months)
            prominence_factor: Minimum prominence as fraction of signal range
        """
        self.smoothing_window = smoothing_window
        self.prominence_factor = prominence_factor
    
    def smooth_signal(self, data: np.ndarray, method: str = "uniform") -> np.ndarray:
        """Apply smoothing to reduce noise in peak detection."""
        if method == "uniform":
            return uniform_filter1d(data, size=self.smoothing_window, mode='nearest')
        elif method == "savgol":
            # Ensure odd window size
            window = self.smoothing_window if self.smoothing_window % 2 == 1 else self.smoothing_window + 1
            window = min(window, len(data) - 1)
            if window >= 3:
                return savgol_filter(data, window, polyorder=2, mode='nearest')
        return data
    
    def detect_peak(self, data: np.ndarray, smooth: bool = True) -> Tuple[int, float]:
        """
        Detect the primary peak in a solar cycle.
        
        Args:
            data: Time series data (monthly sunspot numbers)
            smooth: Whether to apply smoothing
        
        Returns:
            Tuple of (peak_month_index, peak_value)
        """
        if len(data) == 0:
            return 0, 0.0
        
        # Apply smoothing if requested
        if smooth:
            smoothed_data = self.smooth_signal(data)
        else:
            smoothed_data = data.copy()
        
        # Calculate prominence threshold
        data_range = np.max(smoothed_data) - np.min(smoothed_data)
        min_prominence = self.prominence_factor * data_range
        
        # Find peaks with prominence constraint
        peaks, properties = find_peaks(
            smoothed_data, 
            prominence=min_prominence,
            distance=self.smoothing_window  # Minimum distance between peaks
        )
        
        if len(peaks) == 0:
            # No peaks found, use global maximum
            peak_idx = np.argmax(smoothed_data)
            peak_value = data[peak_idx]  # Use original data value
        else:
            # Select highest peak
            peak_prominences = properties['prominences']
            highest_peak_idx = peaks[np.argmax(peak_prominences)]
            peak_idx = highest_peak_idx
            peak_value = data[peak_idx]  # Use original data value
        
        return int(peak_idx), float(peak_value)
    
    def detect_multiple_peaks(self, data: np.ndarray, max_peaks: int = 3) -> List[Tuple[int, float]]:
        """Detect multiple peaks in order of prominence."""
        smoothed_data = self.smooth_signal(data)
        data_range = np.max(smoothed_data) - np.min(smoothed_data)
        min_prominence = self.prominence_factor * data_range
        
        peaks, properties = find_peaks(
            smoothed_data,
            prominence=min_prominence,
            distance=self.smoothing_window
        )
        
        if len(peaks) == 0:
            peak_idx = np.argmax(smoothed_data)
            return [(int(peak_idx), float(data[peak_idx]))]
        
        # Sort by prominence
        peak_prominences = properties['prominences']
        sorted_indices = np.argsort(peak_prominences)[::-1]
        
        result_peaks = []
        for i in range(min(max_peaks, len(sorted_indices))):
            peak_idx = peaks[sorted_indices[i]]
            peak_value = data[peak_idx]
            result_peaks.append((int(peak_idx), float(peak_value)))
        
        return result_peaks


class PeakMetrics:
    """Compute peak-specific metrics for solar cycle forecasting."""
    
    def __init__(self, detector: Optional[SolarCyclePeakDetector] = None):
        self.detector = detector or SolarCyclePeakDetector()
    
    def compute_peak_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """
        Compute peak-specific metrics between actual and predicted cycles.
        
        Args:
            y_true: Actual solar cycle data
            y_pred: Predicted solar cycle data
        
        Returns:
            Dictionary of peak metrics
        """
        # Detect peaks
        true_peak_month, true_peak_value = self.detector.detect_peak(y_true)
        pred_peak_month, pred_peak_value = self.detector.detect_peak(y_pred)
        
        # Peak timing error (months)
        peak_month_error = abs(pred_peak_month - true_peak_month)
        
        # Peak magnitude error
        peak_height_error = abs(pred_peak_value - true_peak_value)
        peak_height_error_rel = peak_height_error / true_peak_value if true_peak_value > 0 else np.inf
        
        # Phase error (percentage of cycle)
        cycle_length = len(y_true)
        phase_error = (peak_month_error / cycle_length) * 100 if cycle_length > 0 else np.inf
        
        return {
            'peak_month_error': peak_month_error,
            'peak_height_error': peak_height_error,
            'peak_height_error_rel': peak_height_error_rel,
            'phase_error_percent': phase_error,
            'true_peak_month': true_peak_month,
            'true_peak_value': true_peak_value,
            'pred_peak_month': pred_peak_month,
            'pred_peak_value': pred_peak_value
        }
    
    def compute_cycle_shape_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Compute metrics related to overall cycle shape."""
        # Ensure same length
        min_len = min(len(y_true), len(y_pred))
        y_true = y_true[:min_len]
        y_pred = y_pred[:min_len]
        
        metrics = {}
        
        # Correlation between shapes
        if len(y_true) > 1:
            correlation = np.corrcoef(y_true, y_pred)[0, 1]
            metrics['shape_correlation'] = correlation if not np.isnan(correlation) else 0.0
        
        # Rising vs declining phase analysis
        true_peak_idx, _ = self.detector.detect_peak(y_true)
        pred_peak_idx, _ = self.detector.detect_peak(y_pred)
        
        # Rising phase accuracy
        if true_peak_idx > 0 and pred_peak_idx > 0:
            true_rising = y_true[:true_peak_idx]
            pred_rising = y_pred[:min(pred_peak_idx, len(y_pred))]
            
            if len(true_rising) > 0 and len(pred_rising) > 0:
                min_rising_len = min(len(true_rising), len(pred_rising))
                rising_corr = np.corrcoef(
                    true_rising[:min_rising_len], 
                    pred_rising[:min_rising_len]
                )[0, 1]
                metrics['rising_phase_correlation'] = rising_corr if not np.isnan(rising_corr) else 0.0
        
        # Declining phase accuracy
        if true_peak_idx < len(y_true) - 1 and pred_peak_idx < len(y_pred) - 1:
            true_declining = y_true[true_peak_idx:]
            pred_declining = y_pred[pred_peak_idx:]
            
            if len(true_declining) > 0 and len(pred_declining) > 0:
                min_declining_len = min(len(true_declining), len(pred_declining))
                declining_corr = np.corrcoef(
                    true_declining[:min_declining_len],
                    pred_declining[:min_declining_len]
                )[0, 1]
                metrics['declining_phase_correlation'] = declining_corr if not np.isnan(declining_corr) else 0.0
        
        return metrics


class ConformalPeakPredictor:
    """Conformal prediction for peak timing and magnitude uncertainty."""
    
    def __init__(self, alpha: float = 0.1):
        """
        Args:
            alpha: Miscoverage level (e.g., 0.1 for 90% coverage)
        """
        self.alpha = alpha
        self.calibration_residuals = {'month': [], 'height': []}
        self.is_calibrated = False
    
    def calibrate(self, true_cycles: List[np.ndarray], predicted_cycles: List[np.ndarray],
                  detector: Optional[SolarCyclePeakDetector] = None):
        """
        Calibrate conformal predictor using historical cycle predictions.
        
        Args:
            true_cycles: List of actual cycle data
            predicted_cycles: List of predicted cycle data
            detector: Peak detector instance
        """
        if len(true_cycles) != len(predicted_cycles):
            raise ValueError("Number of true and predicted cycles must match")
        
        detector = detector or SolarCyclePeakDetector()
        
        month_residuals = []
        height_residuals = []
        
        for true_cycle, pred_cycle in zip(true_cycles, predicted_cycles):
            if len(true_cycle) == 0 or len(pred_cycle) == 0:
                continue
            
            # Detect peaks
            true_month, true_height = detector.detect_peak(true_cycle)
            pred_month, pred_height = detector.detect_peak(pred_cycle)
            
            # Compute residuals
            month_residual = abs(true_month - pred_month)
            height_residual = abs(true_height - pred_height)
            
            month_residuals.append(month_residual)
            height_residuals.append(height_residual)
        
        if len(month_residuals) == 0:
            warnings.warn("No valid calibration data found")
            return
        
        self.calibration_residuals = {
            'month': np.array(month_residuals),
            'height': np.array(height_residuals)
        }
        self.is_calibrated = True
    
    def predict_intervals(self, predicted_cycle: np.ndarray,
                         detector: Optional[SolarCyclePeakDetector] = None) -> Dict[str, Tuple[float, float]]:
        """
        Generate conformal prediction intervals for peak month and height.
        
        Args:
            predicted_cycle: Predicted cycle data
            detector: Peak detector instance
        
        Returns:
            Dictionary with 'month' and 'height' intervals
        """
        if not self.is_calibrated:
            raise ValueError("Predictor not calibrated. Call calibrate() first.")
        
        detector = detector or SolarCyclePeakDetector()
        
        # Get point prediction
        pred_month, pred_height = detector.detect_peak(predicted_cycle)
        
        # Compute conformal quantiles
        month_quantile = np.quantile(self.calibration_residuals['month'], 1 - self.alpha)
        height_quantile = np.quantile(self.calibration_residuals['height'], 1 - self.alpha)
        
        # Generate intervals
        month_interval = (
            max(0, pred_month - month_quantile),
            min(len(predicted_cycle) - 1, pred_month + month_quantile)
        )
        
        height_interval = (
            max(0, pred_height - height_quantile),
            pred_height + height_quantile
        )
        
        return {
            'month': month_interval,
            'height': height_interval,
            'point_month': pred_month,
            'point_height': pred_height,
            'month_quantile': month_quantile,
            'height_quantile': height_quantile
        }
    
    def evaluate_coverage(self, true_cycles: List[np.ndarray], 
                         predicted_cycles: List[np.ndarray],
                         detector: Optional[SolarCyclePeakDetector] = None) -> Dict[str, float]:
        """Evaluate conformal prediction coverage on test data."""
        if not self.is_calibrated:
            raise ValueError("Predictor not calibrated")
        
        detector = detector or SolarCyclePeakDetector()
        
        month_covered = 0
        height_covered = 0
        total_valid = 0
        
        for true_cycle, pred_cycle in zip(true_cycles, predicted_cycles):
            if len(true_cycle) == 0 or len(pred_cycle) == 0:
                continue
            
            # Get true peaks
            true_month, true_height = detector.detect_peak(true_cycle)
            
            # Get prediction intervals
            intervals = self.predict_intervals(pred_cycle, detector)
            
            # Check coverage
            month_in_interval = (
                intervals['month'][0] <= true_month <= intervals['month'][1]
            )
            height_in_interval = (
                intervals['height'][0] <= true_height <= intervals['height'][1]
            )
            
            if month_in_interval:
                month_covered += 1
            if height_in_interval:
                height_covered += 1
            
            total_valid += 1
        
        if total_valid == 0:
            return {'month_coverage': 0.0, 'height_coverage': 0.0, 'total_samples': 0}
        
        return {
            'month_coverage': month_covered / total_valid,
            'height_coverage': height_covered / total_valid,
            'target_coverage': 1 - self.alpha,
            'total_samples': total_valid
        }


def create_synthetic_cycles(n_cycles: int = 10, base_length: int = 132,
                           noise_level: float = 5.0) -> List[np.ndarray]:
    """Generate synthetic solar cycles for testing."""
    np.random.seed(42)
    cycles = []
    
    for i in range(n_cycles):
        # Vary cycle length slightly
        length = base_length + np.random.randint(-12, 13)
        
        # Create cycle shape (asymmetric with faster rise than decline)
        t = np.linspace(0, 2*np.pi, length)
        base_cycle = 50 + 40 * np.sin(t) + 20 * np.sin(2*t) + 10 * np.sin(3*t)
        
        # Add peak variation
        peak_shift = np.random.randint(-6, 7)
        peak_magnitude = np.random.uniform(0.8, 1.2)
        
        # Shift peak and scale
        base_cycle = np.roll(base_cycle, peak_shift)
        base_cycle *= peak_magnitude
        
        # Add noise
        noise = np.random.normal(0, noise_level, length)
        cycle = base_cycle + noise
        cycle = np.maximum(cycle, 0)  # Ensure non-negative
        
        cycles.append(cycle)
    
    return cycles


if __name__ == "__main__":
    # Test peak detection and conformal prediction
    print("Testing Solar Cycle Peak Detection and Conformal Prediction...")
    
    # Generate synthetic cycles
    true_cycles = create_synthetic_cycles(n_cycles=8, noise_level=3.0)
    predicted_cycles = create_synthetic_cycles(n_cycles=8, noise_level=5.0)
    
    # Test peak detector
    detector = SolarCyclePeakDetector()
    
    print("\nTesting peak detection on synthetic cycles:")
    for i, cycle in enumerate(true_cycles[:3]):
        peak_month, peak_value = detector.detect_peak(cycle)
        print(f"Cycle {i}: Peak at month {peak_month}, value {peak_value:.1f}")
    
    # Test peak metrics
    print("\nTesting peak metrics:")
    metrics_calculator = PeakMetrics(detector)
    
    for i in range(min(3, len(true_cycles))):
        metrics = metrics_calculator.compute_peak_metrics(
            true_cycles[i], predicted_cycles[i]
        )
        print(f"Cycle {i} metrics:")
        for name, value in metrics.items():
            print(f"  {name}: {value:.2f}")
    
    # Test conformal prediction
    print("\nTesting conformal peak prediction:")
    conformal_predictor = ConformalPeakPredictor(alpha=0.1)
    
    # Use first 5 cycles for calibration
    calib_true = true_cycles[:5]
    calib_pred = predicted_cycles[:5]
    conformal_predictor.calibrate(calib_true, calib_pred, detector)
    
    # Test on remaining cycles
    test_true = true_cycles[5:]
    test_pred = predicted_cycles[5:]
    
    if len(test_pred) > 0:
        intervals = conformal_predictor.predict_intervals(test_pred[0], detector)
        print(f"Peak prediction intervals for test cycle:")
        print(f"  Month: [{intervals['month'][0]:.1f}, {intervals['month'][1]:.1f}] (point: {intervals['point_month']})")
        print(f"  Height: [{intervals['height'][0]:.1f}, {intervals['height'][1]:.1f}] (point: {intervals['point_height']:.1f})")
        
        # Evaluate coverage
        coverage = conformal_predictor.evaluate_coverage(test_true, test_pred, detector)
        print(f"\nConformal prediction coverage:")
        print(f"  Month coverage: {coverage['month_coverage']:.2f} (target: {coverage['target_coverage']:.2f})")
        print(f"  Height coverage: {coverage['height_coverage']:.2f} (target: {coverage['target_coverage']:.2f})")
    
    print("\n✅ Peak detection and conformal prediction implemented and tested!")