"""Analysis rasters on the workspace collision lattice: slope, roughness, crater id (none depends on the rover)."""
from __future__ import annotations

import numpy as np
from scipy import ndimage

from lunar_rover_sim.core.data_model.regular_grid import RegularGrid
from lunar_rover_sim.core.terrain.crater_population import CraterTable


def slope_over_baseline_deg(heights: np.ndarray, spacing_m: float, baseline_m: float = 0.5) -> np.ndarray:
    """Slope angle (deg) from height differences across a baseline centred on each post (edges replicated)."""
    half = max(1, int(round(baseline_m / spacing_m / 2)))
    padded = np.pad(np.asarray(heights, dtype=np.float64), half, mode="edge")
    slope_east = (padded[half:-half, 2 * half:] - padded[half:-half, :-2 * half]) / (2 * half * spacing_m)
    slope_north = (padded[2 * half:, half:-half] - padded[:-2 * half, half:-half]) / (2 * half * spacing_m)
    return np.degrees(np.arctan(np.hypot(slope_east, slope_north)))


def local_roughness_m(heights: np.ndarray, spacing_m: float, window_m: float = 0.5) -> np.ndarray:
    """Standard deviation of height about its local Gaussian trend (sigma = window_m) within a window_m box."""
    heights = np.asarray(heights, dtype=np.float64)
    residual = heights - ndimage.gaussian_filter(heights, window_m / spacing_m, mode="nearest")
    size = max(3, int(round(window_m / spacing_m)) | 1)
    mean = ndimage.uniform_filter(residual, size, mode="nearest")
    mean_square = ndimage.uniform_filter(residual ** 2, size, mode="nearest")
    return np.sqrt(np.maximum(mean_square - mean ** 2, 0.0))


def crater_id_raster(grid: RegularGrid, table: CraterTable) -> np.ndarray:
    """Crater index + 1 of the crater whose rim contains each post (0 = none; smaller craters win on overlap)."""
    east, north = grid.coordinate_mesh()
    raster = np.zeros(grid.shape, dtype=np.int32)
    for index in np.argsort(-table.diameter_m):
        radius = table.diameter_m[index] / 2
        row_low, column_low = grid.fractional_index(table.center_east_m[index] - radius, table.center_north_m[index] - radius)
        row_high, column_high = grid.fractional_index(table.center_east_m[index] + radius, table.center_north_m[index] + radius)
        rows = slice(max(int(row_low), 0), min(int(np.ceil(row_high)) + 1, grid.count_north))
        columns = slice(max(int(column_low), 0), min(int(np.ceil(column_high)) + 1, grid.count_east))
        if rows.start >= rows.stop or columns.start >= columns.stop:
            continue
        inside = (east[rows, columns] - table.center_east_m[index]) ** 2 + \
                 (north[rows, columns] - table.center_north_m[index]) ** 2 <= radius ** 2
        raster[rows, columns][inside] = index + 1
    return raster
