"""Regular vertex lattices and the one triangulation rule used everywhere (render mesh, collider, queries).

Conventions (docs/DESIGN.md section 4.1): arrays are indexed [north_index, east_index]; vertex [i, j] is at
(origin_east_m + j * spacing_m, origin_north_m + i * spacing_m); index [0, 0] is the south-west vertex; the flat
vertex index is i * count_east + j. Cell with corners a=[i,j], b=[i,j+1], c=[i+1,j], d=[i+1,j+1] is split into the
triangles (a, b, d) and (a, d, c) (diagonal a-d, counter-clockwise seen from above).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class TileSlice:
    """A rectangular block of vertices [row_start, row_stop) x [column_start, column_stop); neighbours share an edge row."""

    row_start: int
    row_stop: int
    column_start: int
    column_stop: int

    @property
    def shape(self) -> tuple[int, int]:
        """(rows, columns) of vertices in the tile."""
        return self.row_stop - self.row_start, self.column_stop - self.column_start

    def array_slice(self) -> tuple[slice, slice]:
        """Numpy index for a [north, east] array of the parent lattice."""
        return slice(self.row_start, self.row_stop), slice(self.column_start, self.column_stop)


@dataclass(frozen=True)
class RegularGrid:
    """A regular vertex lattice in the local east-north frame (metres)."""

    origin_east_m: float
    origin_north_m: float
    spacing_m: float
    count_east: int
    count_north: int

    @classmethod
    def centered(cls, size_m: float, spacing_m: float, center_east_m: float = 0.0,
                 center_north_m: float = 0.0) -> "RegularGrid":
        """Square lattice of side size_m centred on a point; size_m must be an integer multiple of spacing_m."""
        cells = size_m / spacing_m
        if abs(cells - round(cells)) > 1e-6:
            raise ValueError(f"size {size_m} m is not an integer multiple of spacing {spacing_m} m")
        cells = int(round(cells))
        return cls(center_east_m - size_m / 2.0, center_north_m - size_m / 2.0, float(spacing_m), cells + 1, cells + 1)

    @property
    def shape(self) -> tuple[int, int]:
        """Array shape (count_north, count_east)."""
        return self.count_north, self.count_east

    @property
    def extent_east_m(self) -> float:
        """East-west extent from the first to the last vertex."""
        return (self.count_east - 1) * self.spacing_m

    @property
    def extent_north_m(self) -> float:
        """North-south extent from the first to the last vertex."""
        return (self.count_north - 1) * self.spacing_m

    @property
    def maximum_east_m(self) -> float:
        """East coordinate of the last vertex column."""
        return self.origin_east_m + self.extent_east_m

    @property
    def maximum_north_m(self) -> float:
        """North coordinate of the last vertex row."""
        return self.origin_north_m + self.extent_north_m

    def east_coordinates(self) -> np.ndarray:
        """East coordinate of each vertex column, float64, shape (count_east,)."""
        return self.origin_east_m + np.arange(self.count_east, dtype=np.float64) * self.spacing_m

    def north_coordinates(self) -> np.ndarray:
        """North coordinate of each vertex row, float64, shape (count_north,)."""
        return self.origin_north_m + np.arange(self.count_north, dtype=np.float64) * self.spacing_m

    def coordinate_mesh(self) -> tuple[np.ndarray, np.ndarray]:
        """(east, north) float64 arrays of shape (count_north, count_east)."""
        east, north = np.meshgrid(self.east_coordinates(), self.north_coordinates())
        return east, north

    def subsampled(self, decimation: int) -> "RegularGrid":
        """The lattice made of every ``decimation``-th vertex; cell counts must divide exactly."""
        if (self.count_east - 1) % decimation or (self.count_north - 1) % decimation:
            raise ValueError(f"cell counts {self.count_east - 1}x{self.count_north - 1} not divisible by {decimation}")
        return RegularGrid(self.origin_east_m, self.origin_north_m, self.spacing_m * decimation,
                           (self.count_east - 1) // decimation + 1, (self.count_north - 1) // decimation + 1)

    def fractional_index(self, east_m, north_m) -> tuple[np.ndarray, np.ndarray]:
        """Fractional (row, column) of points; not clipped."""
        column = (np.asarray(east_m, dtype=np.float64) - self.origin_east_m) / self.spacing_m
        row = (np.asarray(north_m, dtype=np.float64) - self.origin_north_m) / self.spacing_m
        return row, column

    def contains(self, east_m, north_m) -> np.ndarray:
        """Boolean mask: points inside or on the border of the lattice."""
        row, column = self.fractional_index(east_m, north_m)
        tolerance = 1e-9
        return (column >= -tolerance) & (column <= self.count_east - 1 + tolerance) & \
               (row >= -tolerance) & (row <= self.count_north - 1 + tolerance)

    def index_of(self, east_m: float, north_m: float) -> tuple[int, int]:
        """(row, column) of the vertex exactly at a point; raises if the point is not on a vertex."""
        row, column = self.fractional_index(east_m, north_m)
        row_index, column_index = int(round(float(row))), int(round(float(column)))
        if abs(row - row_index) > 1e-6 or abs(column - column_index) > 1e-6:
            raise ValueError(f"point ({east_m}, {north_m}) is not on a vertex of this lattice")
        return row_index, column_index

    def triangle_vertex_indices(self) -> np.ndarray:
        """Triangle list (M, 3) int32 of the whole lattice, diagonal a-d, counter-clockwise from above."""
        return lattice_triangle_indices(self.count_north, self.count_east)

    def triangulated_height(self, heights: np.ndarray, east_m, north_m) -> tuple[np.ndarray, np.ndarray]:
        """Exact height of the triangulated surface at points. Returns (height float64, valid mask)."""
        return triangulated_height(heights, self, east_m, north_m)

    def vertex_positions(self, heights: np.ndarray, tile: TileSlice | None = None) -> np.ndarray:
        """(N, 3) float32 vertex positions of the whole lattice or one tile, flat order row-major."""
        east, north = self.east_coordinates(), self.north_coordinates()
        if tile is not None:
            east = east[tile.column_start:tile.column_stop]
            north = north[tile.row_start:tile.row_stop]
            heights = heights[tile.array_slice()]
        east_mesh, north_mesh = np.meshgrid(east, north)
        return np.column_stack([east_mesh.ravel(), north_mesh.ravel(),
                                np.asarray(heights, dtype=np.float64).ravel()]).astype(np.float32)

    def split_into_tiles(self, tile_cells: int) -> list[TileSlice]:
        """Split into tiles of ``tile_cells`` x ``tile_cells`` cells (vertex blocks overlap by one row/column)."""
        if (self.count_east - 1) % tile_cells or (self.count_north - 1) % tile_cells:
            raise ValueError(f"{self.count_east - 1}x{self.count_north - 1} cells are not a multiple of {tile_cells}")
        tiles = []
        for row_start in range(0, self.count_north - 1, tile_cells):
            for column_start in range(0, self.count_east - 1, tile_cells):
                tiles.append(TileSlice(row_start, row_start + tile_cells + 1, column_start, column_start + tile_cells + 1))
        return tiles


def lattice_triangle_indices(count_north: int, count_east: int) -> np.ndarray:
    """Triangle list (M, 3) int32 for a count_north x count_east vertex block (a, b, d) and (a, d, c)."""
    a = (np.arange(count_east - 1)[None, :] + count_east * np.arange(count_north - 1)[:, None]).ravel()
    b, c = a + 1, a + count_east
    d = c + 1
    return np.column_stack([a, b, d, a, d, c]).reshape(-1, 3).astype(np.int32)


def triangulated_height(heights: np.ndarray, grid: RegularGrid, east_m, north_m) -> tuple[np.ndarray, np.ndarray]:
    """Height of the (a,b,d)/(a,d,c) triangulated surface at points; points outside get NaN and valid=False."""
    row, column = grid.fractional_index(east_m, north_m)
    row = np.atleast_1d(row)
    column = np.atleast_1d(column)
    valid = (column >= -1e-9) & (column <= grid.count_east - 1 + 1e-9) & (row >= -1e-9) & (row <= grid.count_north - 1 + 1e-9)
    clipped_column = np.clip(column, 0.0, grid.count_east - 1 - 1e-9)
    clipped_row = np.clip(row, 0.0, grid.count_north - 1 - 1e-9)
    j = np.floor(clipped_column).astype(np.int64)
    i = np.floor(clipped_row).astype(np.int64)
    u = clipped_column - j
    v = clipped_row - i
    height_a = heights[i, j].astype(np.float64)
    height_b = heights[i, j + 1].astype(np.float64)
    height_c = heights[i + 1, j].astype(np.float64)
    height_d = heights[i + 1, j + 1].astype(np.float64)
    upper = u >= v
    result = np.where(upper, height_a + u * (height_b - height_a) + v * (height_d - height_b),
                      height_a + u * (height_d - height_c) + v * (height_c - height_a))
    result = np.where(valid, result, np.nan)
    return result, valid


def triangulated_normal(heights: np.ndarray, grid: RegularGrid, east_m, north_m) -> np.ndarray:
    """Unit normal (N, 3) of the triangle containing each point (NaN outside the lattice)."""
    row, column = grid.fractional_index(east_m, north_m)
    row = np.atleast_1d(row)
    column = np.atleast_1d(column)
    valid = grid.contains(column * grid.spacing_m + grid.origin_east_m, row * grid.spacing_m + grid.origin_north_m)
    j = np.floor(np.clip(column, 0.0, grid.count_east - 1 - 1e-9)).astype(np.int64)
    i = np.floor(np.clip(row, 0.0, grid.count_north - 1 - 1e-9)).astype(np.int64)
    u = np.clip(column, 0.0, grid.count_east - 1 - 1e-9) - j
    v = np.clip(row, 0.0, grid.count_north - 1 - 1e-9) - i
    spacing = grid.spacing_m
    height_a, height_b = heights[i, j].astype(np.float64), heights[i, j + 1].astype(np.float64)
    height_c, height_d = heights[i + 1, j].astype(np.float64), heights[i + 1, j + 1].astype(np.float64)
    upper = u >= v
    slope_east = np.where(upper, height_b - height_a, height_d - height_c) / spacing
    slope_north = np.where(upper, height_d - height_b, height_c - height_a) / spacing
    normal = np.column_stack([-slope_east, -slope_north, np.ones_like(slope_east)])
    normal /= np.linalg.norm(normal, axis=1, keepdims=True)
    normal[~valid] = np.nan
    return normal


def vertex_normals(heights: np.ndarray, spacing_m: float) -> np.ndarray:
    """Per-vertex unit normals (count_north, count_east, 3) float32 from central differences (one-sided at borders)."""
    heights = np.asarray(heights, dtype=np.float64)
    slope_north, slope_east = np.gradient(heights, spacing_m)
    normal = np.stack([-slope_east, -slope_north, np.ones_like(heights)], axis=-1)
    normal /= np.linalg.norm(normal, axis=-1, keepdims=True)
    return normal.astype(np.float32)


def slope_degrees(heights: np.ndarray, spacing_m: float) -> np.ndarray:
    """Slope angle (degrees) at each vertex from central differences."""
    slope_north, slope_east = np.gradient(np.asarray(heights, dtype=np.float64), spacing_m)
    return np.degrees(np.arctan(np.hypot(slope_east, slope_north)))
