"""Sample LROC NAC DTM heights onto local true-metre lattices (docs/DESIGN.md section 9.1).

One float64 crop of the DTM is read around the area of interest plus a 16-post margin, NoData inside the crop is
nearest-filled (and remembered as invalid), and cubic B-spline coefficients are computed once. Every lattice is then
sampled from the same coefficients (``map_coordinates(order=3, prefilter=False)``), so overlapping tiers get
identical values at shared points. Bilinear interpolation is never used: it leaves 2 m creases under a low Sun.

Heights returned are relative to ``height_datum_m`` (the bicubic DTM height at the site origin), float64.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage

from lunar_rover_sim.core.data_model.regular_grid import RegularGrid
from lunar_rover_sim.core.dem.geotiff_memmap_reader import GeoTiffHeightRaster
from lunar_rover_sim.core.frames.site_frame import EquirectangularSiteFrame

SPLINE_MARGIN_POSTS = 16


@dataclass
class SplineDemCrop:
    """Cubic B-spline coefficients of one DTM crop plus its validity mask."""

    raster: GeoTiffHeightRaster
    row_start: int
    column_start: int
    coefficients: np.ndarray
    valid_fraction: np.ndarray

    def sample_absolute(self, frame: EquirectangularSiteFrame, east_m: np.ndarray, north_m: np.ndarray):
        """Absolute DTM heights (m, float64) and a validity mask at local points (any shape)."""
        map_x, map_y = frame.map_from_local(east_m, north_m)
        row, column = self.raster.fractional_pixel_from_map(map_x, map_y)
        local_row = row - self.row_start
        local_column = column - self.column_start
        rows, columns = self.coefficients.shape
        inside = (local_row >= 1) & (local_row <= rows - 2) & (local_column >= 1) & (local_column <= columns - 2)
        if not np.all(inside):
            raise ValueError("lattice reaches outside the DTM crop; the crop extent is too small")
        coordinates = np.stack([np.ravel(local_row), np.ravel(local_column)])
        heights = ndimage.map_coordinates(self.coefficients, coordinates, order=3, prefilter=False, mode="nearest")
        validity = ndimage.map_coordinates(self.valid_fraction, coordinates, order=1, mode="nearest")
        shape = np.shape(east_m)
        return heights.reshape(shape), (validity > 0.999).reshape(shape)


def build_spline_dem_crop(raster: GeoTiffHeightRaster, frame: EquirectangularSiteFrame, half_size_m: float,
                          block_average_factor: int = 1) -> SplineDemCrop:
    """Read the DTM around the site origin (square of half side ``half_size_m`` plus margin) and prefilter it.

    ``block_average_factor`` > 1 averages blocks of factor x factor pixels first (NaN-aware), for coarse backdrops.
    Raises ValueError if the requested crop does not fit inside the raster."""
    corner_east = np.array([-half_size_m, half_size_m, half_size_m, -half_size_m])
    corner_north = np.array([-half_size_m, -half_size_m, half_size_m, half_size_m])
    map_x, map_y = frame.map_from_local(corner_east, corner_north)
    rows, columns = raster.fractional_pixel_from_map(map_x, map_y)
    margin = SPLINE_MARGIN_POSTS * block_average_factor
    row_start = int(np.floor(rows.min())) - margin
    row_stop = int(np.ceil(rows.max())) + margin + 1
    column_start = int(np.floor(columns.min())) - margin
    column_stop = int(np.ceil(columns.max())) + margin + 1
    if block_average_factor > 1:
        row_stop = row_start + ((row_stop - row_start) // block_average_factor + 1) * block_average_factor
        column_stop = column_start + ((column_stop - column_start) // block_average_factor + 1) * block_average_factor
    window = raster.read_window(row_start, row_stop, column_start, column_stop)
    valid = np.isfinite(window)
    if block_average_factor > 1:
        factor = block_average_factor
        shape = (window.shape[0] // factor, factor, window.shape[1] // factor, factor)
        summed = np.where(valid, window, 0.0).reshape(shape).sum(axis=(1, 3))
        counts = valid.reshape(shape).sum(axis=(1, 3))
        window = np.where(counts > 0, summed / np.maximum(counts, 1), np.nan)
        valid = counts >= 0.5 * factor * factor
        window[~valid] = np.nan
    if not valid.any():
        raise ValueError(f"the DTM crop around the site holds no valid heights ({raster.path.name})")
    filled = nearest_fill(window, valid)
    coefficients = ndimage.spline_filter(filled, order=3, mode="mirror")
    if block_average_factor > 1:
        return _BlockAveragedCrop(raster, row_start, column_start, coefficients, valid.astype(np.float64),
                                  block_average_factor)
    return SplineDemCrop(raster, row_start, column_start, coefficients, valid.astype(np.float64))


class _BlockAveragedCrop(SplineDemCrop):
    """A spline crop whose samples are block averages of ``factor`` x ``factor`` DTM pixels."""

    def __init__(self, raster, row_start, column_start, coefficients, valid_fraction, factor):
        """Store the crop; pixel (r, c) of the averaged grid covers DTM rows row_start + r*factor ... + factor - 1."""
        super().__init__(raster, row_start, column_start, coefficients, valid_fraction)
        self.factor = factor

    def sample_absolute(self, frame, east_m, north_m):
        """Absolute heights and validity at local points from the block-averaged coefficients."""
        map_x, map_y = frame.map_from_local(east_m, north_m)
        row, column = self.raster.fractional_pixel_from_map(map_x, map_y)
        local_row = (row - self.row_start - (self.factor - 1) / 2.0) / self.factor
        local_column = (column - self.column_start - (self.factor - 1) / 2.0) / self.factor
        coordinates = np.stack([np.ravel(local_row), np.ravel(local_column)])
        rows, columns = self.coefficients.shape
        if coordinates[0].min() < 0 or coordinates[0].max() > rows - 1 or coordinates[1].min() < 0 \
                or coordinates[1].max() > columns - 1:
            raise ValueError("lattice reaches outside the block-averaged DTM crop")
        heights = ndimage.map_coordinates(self.coefficients, coordinates, order=3, prefilter=False, mode="nearest")
        validity = ndimage.map_coordinates(self.valid_fraction, coordinates, order=1, mode="nearest")
        shape = np.shape(east_m)
        return heights.reshape(shape), (validity > 0.999).reshape(shape)


def nearest_fill(values: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Replace invalid cells by the value of the nearest valid cell (Euclidean)."""
    if valid.all():
        return values.copy()
    _, (nearest_row, nearest_column) = ndimage.distance_transform_edt(~valid, return_indices=True)
    return values[nearest_row, nearest_column]


def fill_remaining_holes(heights: np.ndarray, valid: np.ndarray, feather_sigma_cells: float = 3.0) -> np.ndarray:
    """Nearest-fill holes, then blend the filled area towards a Gaussian-smoothed version so no terraces remain."""
    if valid.all():
        return heights.copy()
    filled = nearest_fill(heights, valid)
    smoothed = ndimage.gaussian_filter(filled, feather_sigma_cells)
    distance = ndimage.distance_transform_edt(~valid)
    weight = np.clip(distance / (2.0 * feather_sigma_cells), 0.0, 1.0)
    return np.where(valid, heights, (1 - weight) * filled + weight * smoothed)


def binomial_decimate(values: np.ndarray) -> np.ndarray:
    """Apply a [1, 2, 1]/4 filter along both axes and keep every second sample (anti-aliased factor-2 reduction)."""
    kernel = np.array([0.25, 0.5, 0.25])
    filtered = ndimage.convolve1d(ndimage.convolve1d(values, kernel, axis=0, mode="nearest"), kernel, axis=1, mode="nearest")
    return filtered[::2, ::2]


@dataclass
class DemSampler:
    """Samples the fine DTM (with coarse-mosaic hole filling) onto local lattices, relative to the site datum."""

    frame: EquirectangularSiteFrame
    fine_crop: SplineDemCrop
    coarse_crop: SplineDemCrop | None
    height_datum_m: float
    fine_pixel_size_m: float

    def sample(self, grid: RegularGrid, require_complete: bool = False) -> tuple[np.ndarray, dict]:
        """Heights (float64, relative to the datum) on a lattice, and a report of hole filling.

        Lattices coarser than the DTM posting are sampled at half spacing and binomially decimated (no aliasing).
        With ``require_complete`` any NoData raises ValueError (used for the workspace and near ring)."""
        oversample = grid.spacing_m > self.fine_pixel_size_m * 1.01
        sample_grid = grid
        if oversample:
            sample_grid = RegularGrid(grid.origin_east_m, grid.origin_north_m, grid.spacing_m / 2,
                                      (grid.count_east - 1) * 2 + 1, (grid.count_north - 1) * 2 + 1)
        east, north = sample_grid.coordinate_mesh()
        heights, valid = self.fine_crop.sample_absolute(self.frame, east, north)
        report = {"fine_invalid_fraction": float(1 - valid.mean()), "coarse_filled_fraction": 0.0,
                  "nearest_filled_fraction": 0.0}
        if not valid.all():
            if require_complete:
                raise ValueError(f"the fine DTM has NoData inside a lattice that must be complete "
                                 f"({100 * (1 - valid.mean()):.2f} % invalid)")
            if self.coarse_crop is not None:
                coarse_heights, coarse_valid = self.coarse_crop.sample_absolute(self.frame, east, north)
                use_coarse = ~valid & coarse_valid
                heights = np.where(use_coarse, coarse_heights, heights)
                report["coarse_filled_fraction"] = float(use_coarse.mean())
                valid = valid | coarse_valid
            if not valid.all():
                report["nearest_filled_fraction"] = float(1 - valid.mean())
                heights = fill_remaining_holes(heights, valid)
        relative = heights - self.height_datum_m
        if oversample:
            relative = binomial_decimate(relative)
        return relative, report


def build_dem_sampler(fine_raster: GeoTiffHeightRaster, coarse_raster: GeoTiffHeightRaster | None,
                      frame: EquirectangularSiteFrame, half_size_m: float) -> DemSampler:
    """Build the shared fine (and coarse) spline crops covering a square of half side ``half_size_m``."""
    fine_crop = build_spline_dem_crop(fine_raster, frame, half_size_m)
    coarse_crop = build_spline_dem_crop(coarse_raster, frame, half_size_m) if coarse_raster is not None else None
    datum_heights, datum_valid = fine_crop.sample_absolute(frame, np.array([0.0]), np.array([0.0]))
    if not datum_valid[0]:
        raise ValueError("the fine DTM has no valid height at the site origin")
    pixel_size = fine_raster.pixel_width_m * frame.east_scale
    return DemSampler(frame, fine_crop, coarse_crop, float(datum_heights[0]), float(pixel_size))


def sample_backdrop(coarse_raster: GeoTiffHeightRaster, frame: EquirectangularSiteFrame, grid: RegularGrid,
                    height_datum_m: float) -> tuple[np.ndarray, dict]:
    """Backdrop heights from a NaN-aware block average of the coarse mosaic (block about half the backdrop spacing)."""
    factor = max(1, int(grid.spacing_m / 2 / coarse_raster.pixel_width_m))
    half_size = max(grid.extent_east_m, grid.extent_north_m) / 2
    crop = build_spline_dem_crop(coarse_raster, frame, half_size + 2 * grid.spacing_m, block_average_factor=factor)
    east, north = grid.coordinate_mesh()
    heights, valid = crop.sample_absolute(frame, east, north)
    report = {"block_average_factor": factor, "nearest_filled_fraction": float(1 - valid.mean())}
    if not valid.all():
        heights = fill_remaining_holes(heights, valid)
    return heights - height_datum_m, report
