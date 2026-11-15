"""Stamp craters into a height-detail lattice (SYNTHETIC; docs/DESIGN.md section 9.2 step 4).

Fresh profile (r' = normalised elliptical radius with rim harmonics, d = depth, h = rim height, E = apron extent):
    bowl  z = h - d (1 - r'^2)                    r' <= 1
    apron z = h (r'^-3 - E^-3) / (1 - E^-3)       1 < r' < E
Degradation is exact linear diffusion: a Gaussian filter with sigma = R sqrt(2u), never below a resolution floor so
the collider resolves the rim. Overprint is scale-aware: before a younger crater is added, relief accumulated so far
is blended towards its own low-pass at sigma = R inside the bowl, so the young crater erases only smaller relief.
Large filters are evaluated on a decimated window and interpolated back (the filtered fields are smooth there).
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage

from lunar_rover_sim.core.data_model.regular_grid import RegularGrid
from lunar_rover_sim.core.terrain.crater_population import HARMONIC_ORDERS, CraterTable
from lunar_rover_sim.core.terrain.spectral_fields import smoothstep


def normalised_crater_radius(east_m: np.ndarray, north_m: np.ndarray, table: CraterTable, index: int) -> np.ndarray:
    """r' = elliptical radius / (R * (1 + sum a_m cos(m theta + phi_m))) of crater ``index`` at points."""
    radius = table.diameter_m[index] / 2.0
    cos_orientation, sin_orientation = np.cos(table.orientation_rad[index]), np.sin(table.orientation_rad[index])
    offset_east = east_m - table.center_east_m[index]
    offset_north = north_m - table.center_north_m[index]
    along = offset_east * cos_orientation + offset_north * sin_orientation
    across = (-offset_east * sin_orientation + offset_north * cos_orientation) / table.ellipticity[index]
    rho = np.hypot(along, across)
    theta = np.arctan2(across, along)
    modulation = np.ones_like(rho)
    for order_index, order in enumerate(HARMONIC_ORDERS):
        modulation += table.harmonic_amplitudes[index, order_index] * np.cos(
            order * theta + table.harmonic_phases_rad[index, order_index])
    return rho / (radius * modulation)


def fresh_crater_profile(normalised_radius: np.ndarray, depth_m: float, rim_height_m: float,
                         apron_extent_radii: float) -> np.ndarray:
    """Fresh simple-crater profile z(r') in metres (zero beyond the apron extent)."""
    extent_term = apron_extent_radii ** -3
    bowl = rim_height_m - depth_m * (1.0 - normalised_radius ** 2)
    safe_radius = np.maximum(normalised_radius, 1.0)
    apron = rim_height_m * (safe_radius ** -3 - extent_term) / (1.0 - extent_term)
    return np.where(normalised_radius <= 1.0, bowl, np.where(normalised_radius < apron_extent_radii, apron, 0.0))


def smooth_lowpass(values: np.ndarray, sigma_cells: float) -> np.ndarray:
    """Gaussian low-pass; for sigma > 6 cells it is computed on a block-averaged copy and interpolated back."""
    if sigma_cells <= 6.0:
        return ndimage.gaussian_filter(values, sigma_cells, mode="nearest")
    factor = int(sigma_cells // 3)
    rows, columns = values.shape
    padded_rows = -(-rows // factor) * factor
    padded_columns = -(-columns // factor) * factor
    padded = np.pad(values, ((0, padded_rows - rows), (0, padded_columns - columns)), mode="edge")
    reduced = padded.reshape(padded_rows // factor, factor, padded_columns // factor, factor).mean(axis=(1, 3))
    filtered = ndimage.gaussian_filter(reduced, sigma_cells / factor, mode="nearest")
    row_coordinates = (np.arange(rows) - (factor - 1) / 2.0) / factor
    column_coordinates = (np.arange(columns) - (factor - 1) / 2.0) / factor
    mesh_rows, mesh_columns = np.meshgrid(row_coordinates, column_coordinates, indexing="ij")
    return ndimage.map_coordinates(filtered, [mesh_rows, mesh_columns], order=1, mode="nearest")


def stamp_craters(detail_m: np.ndarray, grid: RegularGrid, table: CraterTable, indices: np.ndarray,
                  apron_extent_radii: float, rim_height_to_diameter: float, minimum_filter_sigma_m: float,
                  record_realised: bool) -> None:
    """Add the selected craters (oldest first) to ``detail_m`` in place, with diffusion filter and overprint.

    When ``record_realised`` is set, the table's realised depth and rim height are overwritten from this lattice."""
    order = indices[np.argsort(table.age_rank[indices], kind="stable")]
    spacing = grid.spacing_m
    for index in order:
        radius = table.diameter_m[index] / 2.0
        sigma_m = max(radius * np.sqrt(2.0 * table.degradation_parameter[index]), minimum_filter_sigma_m)
        half_window = (apron_extent_radii + 0.5) * radius * 1.15 + 3.0 * sigma_m
        row_low, column_low = grid.fractional_index(table.center_east_m[index] - half_window,
                                                    table.center_north_m[index] - half_window)
        row_high, column_high = grid.fractional_index(table.center_east_m[index] + half_window,
                                                      table.center_north_m[index] + half_window)
        row_start = max(int(np.floor(row_low)), 0)
        row_stop = min(int(np.ceil(row_high)) + 1, grid.count_north)
        column_start = max(int(np.floor(column_low)), 0)
        column_stop = min(int(np.ceil(column_high)) + 1, grid.count_east)
        if row_start >= row_stop - 2 or column_start >= column_stop - 2:
            continue
        east = grid.origin_east_m + np.arange(column_start, column_stop) * spacing
        north = grid.origin_north_m + np.arange(row_start, row_stop) * spacing
        east_mesh, north_mesh = np.meshgrid(east, north)
        normalised = normalised_crater_radius(east_mesh, north_mesh, table, index)
        depth = table.fresh_depth_to_diameter[index] * table.diameter_m[index]
        rim_height = rim_height_to_diameter * table.diameter_m[index]
        stamp = smooth_lowpass(fresh_crater_profile(normalised, depth, rim_height, apron_extent_radii),
                               sigma_m / spacing)
        window = detail_m[row_start:row_stop, column_start:column_stop]
        lowpass = smooth_lowpass(window, radius / spacing)
        blend = np.where(normalised < 1.0, smoothstep((normalised - 0.6) / 0.4), 1.0)
        window[...] = lowpass + blend * (window - lowpass) + stamp
        if record_realised:
            inside_rim = normalised <= 1.15
            if inside_rim.any():
                table.realised_rim_height_m[index] = float(stamp[inside_rim].max())
                table.realised_depth_m[index] = float(stamp[inside_rim].max() - stamp[inside_rim].min())
