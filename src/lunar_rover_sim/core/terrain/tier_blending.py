"""Blend nested tiers into one watertight surface and cut the holes of outer tiers."""
from __future__ import annotations

import numpy as np

from lunar_rover_sim.core.data_model.regular_grid import RegularGrid, triangulated_height
from lunar_rover_sim.core.terrain.spectral_fields import smoothstep


def edge_weight(grid: RegularGrid, skirt_m: float) -> np.ndarray:
    """Weight (count_north, count_east): 0 on the lattice border rising smoothly to 1 at ``skirt_m`` inside."""
    east, north = grid.coordinate_mesh()
    distance = np.minimum.reduce([east - grid.origin_east_m, grid.maximum_east_m - east,
                                  north - grid.origin_north_m, grid.maximum_north_m - north])
    return smoothstep(distance / skirt_m)


def blend_into_outer(inner_grid: RegularGrid, inner_full_m: np.ndarray, outer_grid: RegularGrid,
                     outer_final_m: np.ndarray, skirt_m: float) -> np.ndarray:
    """inner = outer_triangulated + edge_weight * (inner_full - outer_triangulated): exact match on the border."""
    east, north = inner_grid.coordinate_mesh()
    outer_at_inner, valid = triangulated_height(outer_final_m, outer_grid, east.ravel(), north.ravel())
    if not valid.all():
        raise ValueError("inner tier reaches outside its outer tier")
    outer_at_inner = outer_at_inner.reshape(inner_grid.shape)
    return outer_at_inner + edge_weight(inner_grid, skirt_m) * (inner_full_m - outer_at_inner)


def cells_outside_inner(outer_grid: RegularGrid, inner_grid: RegularGrid) -> np.ndarray:
    """Cell mask (count_north-1, count_east-1) of the outer tier: True where the cell is NOT covered by the inner tier."""
    tolerance = 1e-6 * outer_grid.spacing_m
    east = outer_grid.east_coordinates()
    north = outer_grid.north_coordinates()
    inside_east = (east[:-1] >= inner_grid.origin_east_m - tolerance) & (east[1:] <= inner_grid.maximum_east_m + tolerance)
    inside_north = (north[:-1] >= inner_grid.origin_north_m - tolerance) & \
                   (north[1:] <= inner_grid.maximum_north_m + tolerance)
    return ~(inside_north[:, None] & inside_east[None, :])
