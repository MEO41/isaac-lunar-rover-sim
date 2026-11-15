"""Wheel tracks: world-anchored deformation of the workspace visual lattice (docs/DESIGN.md section 9.5).

Each wheel contact sample carves the field with a MINIMUM ENVELOPE in height-change space (relative to the rest
surface, so ruts follow slopes): idempotent and order-independent, so overlapping samples and repeated wheels never
double count. Target height change in the wheel frame (u along travel, v along the axle):
    target(u, v) = -z + (r - sqrt(r^2 - u^2)) - g * max(0, cos(N (theta + u / r)))^2 [|v| <= b/2]
                   + flank_slope * max(0, |v| - b/2)
with sinkage z, wheel radius r, grouser height g, grouser count N, true wheel angle theta (the imprint smears when the
wheel slips). Only newly carved volume is deposited as berms on the shoulders (berm_volume_fraction; the rest is
compaction). The PhysX collider is never modified: ruts are visual (depth capped by sinkage.maximum_m).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from lunar_rover_sim.core.data_model.regular_grid import RegularGrid, TileSlice


@dataclass(frozen=True)
class WheelContactSample:
    """One gated wheel-ground contact (world frame, metres, radians)."""

    time_s: float
    wheel_index: int
    contact_east_m: float
    contact_north_m: float
    heading_rad: float
    sinkage_m: float
    wheel_angle_rad: float
    normal_force_n: float
    slip_ratio: float


@dataclass(frozen=True)
class TrackStampGeometry:
    """Wheel and soil constants used by the stamp kernel."""

    wheel_radius_m: float
    wheel_width_m: float
    grouser_count: int
    grouser_height_m: float
    flank_slope: float
    berm_width_m: float
    berm_volume_fraction: float
    pass_separation_time_s: float


class TrackField:
    """Track state on the workspace visual lattice plus the RGBA mask image (R tread, G shoulder, B passes)."""

    def __init__(self, grid: RegularGrid, rest_height_m: np.ndarray, tile_cells: int, geometry: TrackStampGeometry,
                 exclusion_mask: np.ndarray | None = None):
        """Zeroed field; ``exclusion_mask`` marks cells under rocks (never carved)."""
        self.grid = grid
        self.rest_height_m = np.asarray(rest_height_m, dtype=np.float32)
        self.tile_cells = tile_cells
        self.geometry = geometry
        self.exclusion_mask = np.zeros(grid.shape, dtype=bool) if exclusion_mask is None else exclusion_mask.astype(bool)
        self.height_change_m = np.zeros(grid.shape, dtype=np.float32)
        self.compaction = np.zeros(grid.shape, dtype=np.uint8)
        self.disturbance = np.zeros(grid.shape, dtype=np.uint8)
        self.pass_count = np.zeros(grid.shape, dtype=np.uint8)
        self.last_visit_time_s = np.full(grid.shape, np.nan, dtype=np.float32)
        self.wheel_index = np.full(grid.shape, 255, dtype=np.uint8)
        self.mask_size_px = 1 << int(math.ceil(math.log2(max(grid.count_east, grid.count_north))))
        self._mask = np.zeros((self.mask_size_px, self.mask_size_px, 4), dtype=np.uint8)
        self.dirty_tiles: set[tuple[int, int]] = set()
        self.mask_dirty = False
        self.touched_box: list[int] | None = None
        self.sample_count = 0

    # ------------------------------------------------------------------ geometry of the mask texture
    @property
    def mask_size_m(self) -> float:
        """World size covered by the (power-of-two) mask texture."""
        return self.mask_size_px * self.grid.spacing_m

    @property
    def mask_origin_m(self) -> tuple[float, float]:
        """World (east, north) of the mask texture's south-west corner (texel centres sit on vertices)."""
        half = self.grid.spacing_m / 2
        return self.grid.origin_east_m - half, self.grid.origin_north_m - half

    # ------------------------------------------------------------------ stamping
    def stamp(self, sample: WheelContactSample) -> tuple[int, int, int, int] | None:
        """Carve one sample; returns the touched window (row_start, row_stop, column_start, column_stop) or None."""
        geometry = self.geometry
        radius, half_width = geometry.wheel_radius_m, geometry.wheel_width_m / 2
        sinkage, grouser = float(sample.sinkage_m), geometry.grouser_height_m
        if sinkage <= 0:
            return None
        depth = min(sinkage + grouser, radius)
        half_length = math.sqrt(max(radius ** 2 - (radius - depth) ** 2, 0.0))
        flank_width = sinkage / geometry.flank_slope
        berm_width = geometry.berm_width_m
        reach_across = half_width + flank_width + berm_width + self.grid.spacing_m
        reach = math.hypot(half_length + self.grid.spacing_m, reach_across)
        row_low, column_low = self.grid.fractional_index(sample.contact_east_m - reach, sample.contact_north_m - reach)
        row_high, column_high = self.grid.fractional_index(sample.contact_east_m + reach, sample.contact_north_m + reach)
        row_start, row_stop = max(int(math.floor(row_low)), 0), min(int(math.ceil(row_high)) + 1, self.grid.count_north)
        column_start = max(int(math.floor(column_low)), 0)
        column_stop = min(int(math.ceil(column_high)) + 1, self.grid.count_east)
        if row_start >= row_stop or column_start >= column_stop:
            return None
        window = (slice(row_start, row_stop), slice(column_start, column_stop))
        east = self.grid.origin_east_m + np.arange(column_start, column_stop) * self.grid.spacing_m - sample.contact_east_m
        north = self.grid.origin_north_m + np.arange(row_start, row_stop) * self.grid.spacing_m - sample.contact_north_m
        east_mesh, north_mesh = np.meshgrid(east, north)
        cos_heading, sin_heading = math.cos(sample.heading_rad), math.sin(sample.heading_rad)
        along = east_mesh * cos_heading + north_mesh * sin_heading
        across = np.abs(-east_mesh * sin_heading + north_mesh * cos_heading)
        in_length = np.abs(along) <= half_length
        tread = in_length & (across <= half_width)
        carved_zone = in_length & (across <= half_width + flank_width) & ~self.exclusion_mask[window]
        shoulder = in_length & (across > half_width + flank_width) & (across <= half_width + flank_width + berm_width) & \
            ~self.exclusion_mask[window]
        curvature = radius - np.sqrt(np.maximum(radius ** 2 - along ** 2, 0.0))
        grouser_term = grouser * np.maximum(0.0, np.cos(geometry.grouser_count * (sample.wheel_angle_rad + along / radius))) ** 2
        target = -sinkage + curvature - np.where(across <= half_width, grouser_term, 0.0) + \
            geometry.flank_slope * np.maximum(0.0, across - half_width)
        current = self.height_change_m[window]
        carved = np.where(carved_zone, np.minimum(current, target), current)
        removed_volume = float(np.sum(current - carved)) * self.grid.spacing_m ** 2
        if shoulder.any() and removed_volume > 0:
            weight = np.where(shoulder, 1.0 - (across - half_width - flank_width) / berm_width, 0.0)
            carved = carved + geometry.berm_volume_fraction * removed_volume * weight / \
                (weight.sum() * self.grid.spacing_m ** 2)
        self.height_change_m[window] = carved.astype(np.float32)

        last_time = self.last_visit_time_s[window]
        new_pass = tread & (np.isnan(last_time) | (self.wheel_index[window] != sample.wheel_index) |
                            (sample.time_s - np.nan_to_num(last_time, nan=-1e9) > geometry.pass_separation_time_s))
        passes = self.pass_count[window]
        passes[new_pass] = np.minimum(passes[new_pass].astype(np.int32) + 1, 255).astype(np.uint8)
        last_time[tread] = sample.time_s
        self.wheel_index[window][tread] = sample.wheel_index
        compaction_value = np.uint8(round(255 * min(max(sinkage / 0.010, 0.3), 1.0)))
        compaction = self.compaction[window]
        compaction[tread] = np.maximum(compaction[tread], compaction_value)
        disturbance = self.disturbance[window]
        shoulder_weight = np.where(shoulder, np.round(255 * (1.0 - (across - half_width - flank_width) / berm_width)), 0)
        disturbance[...] = np.maximum(disturbance, shoulder_weight.astype(np.uint8))
        self._update_mask(row_start, row_stop, column_start, column_stop)
        self._mark_dirty(row_start - 1, row_stop + 1, column_start - 1, column_stop + 1)
        self.sample_count += 1
        return row_start, row_stop, column_start, column_stop

    def _update_mask(self, row_start: int, row_stop: int, column_start: int, column_stop: int) -> None:
        """Refresh the RGBA mask inside a window (mask row 0 = NORTH, like the file textures)."""
        window = (slice(row_start, row_stop), slice(column_start, column_stop))
        touched = (self.compaction[window] > 0) | (self.disturbance[window] > 0)
        rgba = np.stack([self.compaction[window], self.disturbance[window],
                         np.minimum(60 * self.pass_count[window].astype(np.int32), 255).astype(np.uint8),
                         np.where(touched, 255, 0).astype(np.uint8)], axis=-1)
        top = self.mask_size_px - row_stop
        self._mask[top:self.mask_size_px - row_start, column_start:column_stop] = rgba[::-1]
        self.mask_dirty = True

    def _mark_dirty(self, row_start: int, row_stop: int, column_start: int, column_stop: int) -> None:
        """Record the visual tiles whose vertices intersect a window, and grow the touched box."""
        row_start, column_start = max(row_start, 0), max(column_start, 0)
        row_stop = min(row_stop, self.grid.count_north)
        column_stop = min(column_stop, self.grid.count_east)
        cells = self.tile_cells
        for tile_row in range(max(row_start - 1, 0) // cells, min((row_stop - 1) // cells, (self.grid.count_north - 2) // cells) + 1):
            for tile_column in range(max(column_start - 1, 0) // cells,
                                     min((column_stop - 1) // cells, (self.grid.count_east - 2) // cells) + 1):
                self.dirty_tiles.add((tile_row, tile_column))
        if self.touched_box is None:
            self.touched_box = [row_start, row_stop, column_start, column_stop]
        else:
            box = self.touched_box
            self.touched_box = [min(box[0], row_start), max(box[1], row_stop), min(box[2], column_start),
                                max(box[3], column_stop)]

    # ------------------------------------------------------------------ outputs
    def take_dirty_tiles(self) -> list[tuple[int, int]]:
        """Return and forget the dirty tiles."""
        tiles = sorted(self.dirty_tiles)
        self.dirty_tiles.clear()
        return tiles

    def tile_slice(self, tile_key: tuple[int, int]) -> TileSlice:
        """Vertex block of a visual tile."""
        row, column = tile_key
        cells = self.tile_cells
        return TileSlice(row * cells, row * cells + cells + 1, column * cells, column * cells + cells + 1)

    def deformed_tile(self, tile: TileSlice) -> tuple[np.ndarray, np.ndarray]:
        """(heights, normals) of a tile after deformation; normals from central differences with a one-post halo."""
        row_start, column_start = max(tile.row_start - 1, 0), max(tile.column_start - 1, 0)
        row_stop = min(tile.row_stop + 1, self.grid.count_north)
        column_stop = min(tile.column_stop + 1, self.grid.count_east)
        heights = self.rest_height_m[row_start:row_stop, column_start:column_stop].astype(np.float64) + \
            self.height_change_m[row_start:row_stop, column_start:column_stop]
        slope_north, slope_east = np.gradient(heights, self.grid.spacing_m)
        normals = np.stack([-slope_east, -slope_north, np.ones_like(heights)], axis=-1)
        normals /= np.linalg.norm(normals, axis=-1, keepdims=True)
        inner = (slice(tile.row_start - row_start, tile.row_start - row_start + tile.shape[0]),
                 slice(tile.column_start - column_start, tile.column_start - column_start + tile.shape[1]))
        return heights[inner], normals[inner].astype(np.float32)

    def mask_image(self) -> np.ndarray:
        """The RGBA8 mask (mask_size x mask_size x 4, row 0 = north); clears the dirty flag."""
        self.mask_dirty = False
        return self._mask

    def track_mask(self) -> np.ndarray:
        """Boolean raster on the visual lattice: True where a tread or shoulder touched the ground."""
        return (self.compaction > 0) | (self.disturbance > 0)

    def rut_depth_statistics(self) -> dict:
        """Depth and berm statistics of everything carved so far (metres)."""
        tread = self.compaction > 0
        return {"tread_cells": int(tread.sum()), "samples": self.sample_count,
                "median_tread_height_change_m": float(np.median(self.height_change_m[tread])) if tread.any() else 0.0,
                "minimum_height_change_m": float(self.height_change_m.min()),
                "maximum_berm_m": float(self.height_change_m.max())}

    def clear(self) -> list[tuple[int, int]]:
        """Zero the state; returns the tiles that must be rewritten from the rest surface."""
        if self.touched_box is None:
            return []
        row_start, row_stop, column_start, column_stop = self.touched_box
        self._mark_dirty(row_start, row_stop, column_start, column_stop)
        tiles = self.take_dirty_tiles()
        self.height_change_m[:] = 0
        self.compaction[:] = 0
        self.disturbance[:] = 0
        self.pass_count[:] = 0
        self.last_visit_time_s[:] = np.nan
        self.wheel_index[:] = 255
        self._mask[:] = 0
        self.mask_dirty = True
        self.touched_box = None
        self.sample_count = 0
        return tiles

    def save(self, path) -> None:
        """Write the field state to an npz file."""
        np.savez_compressed(path, height_change_m=self.height_change_m, compaction=self.compaction,
                            disturbance=self.disturbance, pass_count=self.pass_count,
                            last_visit_time_s=self.last_visit_time_s, wheel_index=self.wheel_index)
