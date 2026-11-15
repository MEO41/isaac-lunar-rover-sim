"""Band-limited fractal (self-affine) height fields with absolute spectral normalisation (SYNTHETIC).

Target isotropic power spectrum S(k) = C * k^-(2H+2) (k in rad/m), with C chosen so that the structure function
(root-mean-square height difference at lag r) of the unlimited surface is sigma_anchor * (r / lag_anchor)^H:
    C = A * H * 2^(2H-1) * Gamma(1+H) / (pi * Gamma(1-H)),   A = sigma_anchor^2 / lag_anchor^(2H).
A band gain defined on its square, G^2(k) = smoothstep rising edge * (1 - smoothstep falling edge), limits the field
to [lambda_min, lambda_max]; adjacent bands sum to exactly one in power and the field has exactly zero power at
wavelengths longer than lambda_max / (1 - w), so it cannot alter what the DEM resolves.
"""
from __future__ import annotations

import math

import numpy as np
from scipy import fft as scipy_fft


def smoothstep(values: np.ndarray) -> np.ndarray:
    """Cubic smoothstep 3t^2 - 2t^3 of values clipped to [0, 1]."""
    clipped = np.clip(values, 0.0, 1.0)
    return clipped * clipped * (3.0 - 2.0 * clipped)


def spectrum_constant(hurst_exponent: float, anchor_sigma_m: float, anchor_lag_m: float) -> float:
    """C of S(k) = C k^-(2H+2) for a surface whose structure function is anchor_sigma_m at anchor_lag_m."""
    amplitude = anchor_sigma_m ** 2 / anchor_lag_m ** (2 * hurst_exponent)
    return amplitude * hurst_exponent * 2 ** (2 * hurst_exponent - 1) * math.gamma(1 + hurst_exponent) / (
        math.pi * math.gamma(1 - hurst_exponent))


def band_gain_squared(wavenumber: np.ndarray, wavelength_minimum_m: float, wavelength_maximum_m: float,
                      relative_edge_width: float) -> np.ndarray:
    """G^2(k) of the band [wavelength_minimum_m, wavelength_maximum_m] with smooth edges of relative width w."""
    k_low = 2 * math.pi / wavelength_maximum_m
    k_high = 2 * math.pi / wavelength_minimum_m
    width = relative_edge_width
    rising = smoothstep((wavenumber - (1 - width) * k_low) / (2 * width * k_low))
    falling = 1.0 - smoothstep((wavenumber - (1 - width) * k_high) / (2 * width * k_high))
    return rising * falling


def wavenumber_grid(shape: tuple[int, int], spacing_m: float) -> tuple[np.ndarray, float, float]:
    """|k| (rad/m) on the FFT grid of an array of ``shape`` = (north, east), plus delta k north and east."""
    count_north, count_east = shape
    k_north = 2 * math.pi * scipy_fft.fftfreq(count_north, spacing_m)
    k_east = 2 * math.pi * scipy_fft.fftfreq(count_east, spacing_m)
    magnitude = np.hypot(k_north[:, None], k_east[None, :])
    return magnitude, 2 * math.pi / (count_north * spacing_m), 2 * math.pi / (count_east * spacing_m)


def expected_band_variance_m2(shape: tuple[int, int], spacing_m: float, hurst_exponent: float, anchor_sigma_m: float,
                              anchor_lag_m: float, wavelength_minimum_m: float, wavelength_maximum_m: float,
                              relative_edge_width: float) -> float:
    """Exact expected variance (m^2) of ``synthesize_band_limited_fractal`` for these parameters: sum S G^2 dk^2."""
    magnitude, delta_north, delta_east = wavenumber_grid(shape, spacing_m)
    magnitude[0, 0] = np.inf
    constant = spectrum_constant(hurst_exponent, anchor_sigma_m, anchor_lag_m)
    spectrum = constant * magnitude ** -(2 * hurst_exponent + 2)
    gain = band_gain_squared(magnitude, wavelength_minimum_m, wavelength_maximum_m, relative_edge_width)
    return float(np.sum(spectrum * gain) * delta_north * delta_east)


def synthesize_band_limited_fractal(shape: tuple[int, int], spacing_m: float, hurst_exponent: float,
                                    anchor_sigma_m: float, anchor_lag_m: float, wavelength_minimum_m: float,
                                    wavelength_maximum_m: float, relative_edge_width: float,
                                    random_generator: np.random.Generator) -> np.ndarray:
    """Periodic band-limited fractal field (m, float64) of ``shape`` = (north, east) with absolute amplitude."""
    if wavelength_minimum_m < 2.5 * spacing_m:
        raise ValueError(f"band lower edge {wavelength_minimum_m} m is below 2.5 posts of {spacing_m} m")
    magnitude, delta_north, delta_east = wavenumber_grid(shape, spacing_m)
    magnitude[0, 0] = np.inf
    constant = spectrum_constant(hurst_exponent, anchor_sigma_m, anchor_lag_m)
    spectrum = constant * magnitude ** -(2 * hurst_exponent + 2)
    gain = band_gain_squared(magnitude, wavelength_minimum_m, wavelength_maximum_m, relative_edge_width)
    filter_amplitude = np.sqrt(spectrum * gain * delta_north * delta_east * shape[0] * shape[1])
    filter_amplitude[0, 0] = 0.0
    white_noise = random_generator.standard_normal(shape)
    field = scipy_fft.ifft2(scipy_fft.fft2(white_noise, workers=1) * filter_amplitude, workers=1).real
    return field


def gaussian_lowpass_fft(field: np.ndarray, sigma_m: float, spacing_m: float) -> np.ndarray:
    """Periodic Gaussian low-pass (standard deviation sigma_m in space) through the FFT."""
    magnitude, _, _ = wavenumber_grid(field.shape, spacing_m)
    transfer = np.exp(-0.5 * (magnitude * sigma_m) ** 2)
    return scipy_fft.ifft2(scipy_fft.fft2(field, workers=1) * transfer, workers=1).real


def periodic_noise_texture(size_px: int, tile_size_m: float, hurst_exponent: float,
                           random_generator: np.random.Generator) -> np.ndarray:
    """Seamlessly tiling zero-mean unit-variance noise with amplitude ~ f^-(H+1), from 2 cycles per tile to Nyquist."""
    pixel_m = tile_size_m / size_px
    magnitude, _, _ = wavenumber_grid((size_px, size_px), pixel_m)
    frequency = magnitude / (2 * math.pi)
    band = (frequency > 2 / tile_size_m) & (frequency < 1 / (2 * pixel_m))
    amplitude = np.where(band, np.where(frequency > 0, frequency, 1.0) ** -(hurst_exponent + 1), 0.0)
    noise = scipy_fft.ifft2(scipy_fft.fft2(random_generator.standard_normal((size_px, size_px)), workers=1)
                            * amplitude, workers=1).real
    return (noise - noise.mean()) / noise.std()
