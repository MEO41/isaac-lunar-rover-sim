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


def vertical_offset_to_inner(outer_grid: RegularGrid, outer_m: np.ndarray, inner_grid: RegularGrid,
                             inner_m: np.ndarray, inner_margin_m: float) -> dict:
    """Median height difference inner - outer at the outer lattice points lying inside the inner tier.

    Points closer than ``inner_margin_m`` to the inner border (its blending skirt) are left out. Used to put a backdrop
    from another DEM (SLDEM2015, ~59 m) on the regional tier's NAC DTM heights. Returns offset_m (add to outer),
    the median absolute deviation of the differences after the shift, and the number of points."""
    east, north = outer_grid.coordinate_mesh()
    inside = ((east >= inner_grid.origin_east_m + inner_margin_m) & (east <= inner_grid.maximum_east_m - inner_margin_m)
              & (north >= inner_grid.origin_north_m + inner_margin_m)
              & (north <= inner_grid.maximum_north_m - inner_margin_m))
    if not inside.any():
        raise ValueError("no outer lattice point lies inside the inner tier; cannot measure the vertical offset")
    inner_at_outer, valid = triangulated_height(inner_m, inner_grid, east[inside], north[inside])
    differences = inner_at_outer[valid] - outer_m[inside][valid]
    offset = float(np.median(differences))
    return {"offset_m": offset, "median_absolute_deviation_m": float(np.median(np.abs(differences - offset))),
            "point_count": int(differences.size)}


def cells_outside_inner(outer_grid: RegularGrid, inner_grid: RegularGrid) -> np.ndarray:
    """Cell mask (count_north-1, count_east-1) of the outer tier: True where the cell is NOT covered by the inner tier."""
    tolerance = 1e-6 * outer_grid.spacing_m
    east = outer_grid.east_coordinates()
    north = outer_grid.north_coordinates()
    inside_east = (east[:-1] >= inner_grid.origin_east_m - tolerance) & (east[1:] <= inner_grid.maximum_east_m + tolerance)
    inside_north = (north[:-1] >= inner_grid.origin_north_m - tolerance) & \
                   (north[1:] <= inner_grid.maximum_north_m + tolerance)
    return ~(inside_north[:, None] & inside_east[None, :])
