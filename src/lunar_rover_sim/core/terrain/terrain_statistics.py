"""Surface statistics used for calibration and reporting: plane detrend, structure function, RMS slope, spectra."""
from __future__ import annotations

import numpy as np
from scipy import fft as scipy_fft


def plane_detrend(heights: np.ndarray, spacing_m: float) -> np.ndarray:
    """Subtract the least-squares plane z = a + b*east + c*north from a [north, east] array."""
    count_north, count_east = heights.shape
    north, east = np.mgrid[0:count_north, 0:count_east].astype(np.float64) * spacing_m
    design = np.column_stack([np.ones(heights.size), east.ravel(), north.ravel()])
    coefficients, *_ = np.linalg.lstsq(design, heights.ravel().astype(np.float64), rcond=None)
    return heights - (design @ coefficients).reshape(heights.shape)


def structure_function(heights: np.ndarray, spacing_m: float, lag_m: float) -> float:
    """Root-mean-square height difference (m) at horizontal lag lag_m, averaged over the east and north directions."""
    lag_posts = lag_m / spacing_m
    if abs(lag_posts - round(lag_posts)) > 1e-6 or round(lag_posts) < 1:
        raise ValueError(f"lag {lag_m} m is not a positive integer multiple of the spacing {spacing_m} m")
    lag_posts = int(round(lag_posts))
    heights = np.asarray(heights, dtype=np.float64)
    difference_east = heights[:, lag_posts:] - heights[:, :-lag_posts]
    difference_north = heights[lag_posts:, :] - heights[:-lag_posts, :]
    return float(np.sqrt(0.5 * (np.mean(difference_east ** 2) + np.mean(difference_north ** 2))))


def root_mean_square_slope_deg(heights: np.ndarray, spacing_m: float, baseline_m: float) -> float:
    """Root-mean-square slope angle (degrees) over a horizontal baseline: atan(structure_function / baseline)."""
    return float(np.degrees(np.arctan(structure_function(heights, spacing_m, baseline_m) / baseline_m)))


def radial_power_spectrum(heights: np.ndarray, spacing_m: float, bin_count: int = 40) -> tuple[np.ndarray, np.ndarray]:
    """Azimuthally averaged power spectrum of the plane-detrended, Hann-windowed field: (wavelength_m, power)."""
    detrended = plane_detrend(heights, spacing_m)
    window = np.outer(np.hanning(heights.shape[0]), np.hanning(heights.shape[1]))
    power = np.abs(scipy_fft.fft2(detrended * window, workers=1)) ** 2
    frequency = np.hypot(scipy_fft.fftfreq(heights.shape[0], spacing_m)[:, None],
                         scipy_fft.fftfreq(heights.shape[1], spacing_m)[None, :])
    valid = frequency > 0
    edges = np.logspace(np.log10(frequency[valid].min()), np.log10(frequency.max()), bin_count + 1)
    index = np.digitize(frequency[valid], edges) - 1
    mean_power = np.array([power[valid][index == b].mean() if np.any(index == b) else np.nan for b in range(bin_count)])
    centers = np.sqrt(edges[:-1] * edges[1:])
    return 1.0 / centers, mean_power


def fit_hurst_exponent(heights: np.ndarray, spacing_m: float, lags_m: list[float]) -> float:
    """Hurst exponent from the slope of log structure function versus log lag."""
    values = [structure_function(heights, spacing_m, lag) for lag in lags_m]
    slope, _ = np.polyfit(np.log(lags_m), np.log(values), 1)
    return float(slope)
