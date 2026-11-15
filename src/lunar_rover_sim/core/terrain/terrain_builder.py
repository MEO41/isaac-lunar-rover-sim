"""``build_terrain``: the whole terrain recipe of docs/DESIGN.md section 9.2 (DEM + SYNTHETIC craters + fractal).

The DEM array is never modified: synthetic detail is a separate layer added on top, and the bundle keeps both the
DEM-only workspace heights and the final heights so the two can always be compared.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import ndimage

from lunar_rover_sim.core.configuration.configuration_models import SimulatorConfiguration
from lunar_rover_sim.core.data_model.regular_grid import RegularGrid
from lunar_rover_sim.core.data_model.tier_layout import TierLayout, build_tier_layout
from lunar_rover_sim.core.dem.dem_window import build_dem_sampler, sample_backdrop
from lunar_rover_sim.core.dem.geotiff_memmap_reader import open_height_raster
from lunar_rover_sim.core.frames.site_frame import EquirectangularSiteFrame, planetary_curvature_drop_m
from lunar_rover_sim.core.seeding import named_generator
from lunar_rover_sim.core.terrain.crater_population import BAND_A, CraterTable, sample_crater_population
from lunar_rover_sim.core.terrain.crater_stamping import stamp_craters
from lunar_rover_sim.core.terrain.spectral_fields import gaussian_lowpass_fft, synthesize_band_limited_fractal
from lunar_rover_sim.core.terrain.terrain_statistics import (plane_detrend, root_mean_square_slope_deg,
                                                             structure_function)
from lunar_rover_sim.core.terrain.tier_blending import blend_into_outer, cells_outside_inner


@dataclass
class TerrainBundle:
    """All terrain tiers of one scene, heights in metres relative to the site datum (z = 0 at the origin)."""

    layout: TierLayout
    height_datum_m: float
    workspace_height_m: np.ndarray
    workspace_dem_height_m: np.ndarray
    near_ring_height_m: np.ndarray
    regional_height_m: np.ndarray | None
    backdrop_height_m: np.ndarray | None
    near_ring_cells_kept: np.ndarray
    regional_cells_kept: np.ndarray | None
    backdrop_cells_kept: np.ndarray | None
    crater_table: CraterTable
    calibration: dict = field(default_factory=dict)
    statistics: dict = field(default_factory=dict)
    dem_report: dict = field(default_factory=dict)

    @property
    def workspace_collision_height_m(self) -> np.ndarray:
        """Collision lattice heights: an exact subsample of the visual workspace lattice."""
        decimation = self.layout.collision_decimation
        return self.workspace_height_m[::decimation, ::decimation]


def resample_bicubic(source: np.ndarray, source_grid: RegularGrid, target_grid: RegularGrid) -> np.ndarray:
    """Cubic-spline resampling of a field from one lattice onto another lattice inside it."""
    east, north = target_grid.coordinate_mesh()
    row, column = source_grid.fractional_index(east, north)
    return ndimage.map_coordinates(source, [row, column], order=3, mode="nearest")


def interior(values: np.ndarray, grid: RegularGrid, margin_m: float) -> np.ndarray:
    """The part of a lattice array more than ``margin_m`` inside its border."""
    margin = int(np.ceil(margin_m / grid.spacing_m))
    return values[margin:values.shape[0] - margin, margin:values.shape[1] - margin]


def build_terrain(configuration: SimulatorConfiguration, log=print) -> TerrainBundle:
    """Build every terrain tier for one configuration (deterministic for a given scene_seed)."""
    terrain, site, seed = configuration.terrain, configuration.site, configuration.scene_seed
    craters, fractal = terrain.craters, terrain.fractal
    layout = build_tier_layout(terrain)
    workspace_grid, near_grid = layout.workspace_visual_grid, layout.near_ring_grid
    collision_spacing = layout.workspace_collision_grid.spacing_m
    cutoff = site.dem_cutoff_wavelength_m

    parameter_generator = named_generator(seed, "terrain.parameters")
    equilibrium_coefficient = (parameter_generator.uniform(*craters.equilibrium_coefficient_range)
                               if craters.randomize_equilibrium_coefficient else craters.equilibrium_coefficient)
    roughness_multiplier = (parameter_generator.uniform(*fractal.roughness_multiplier_range)
                            if fractal.randomize_roughness_multiplier else fractal.roughness_multiplier)

    frame = EquirectangularSiteFrame(site.origin_longitude_deg, site.origin_latitude_deg, site.standard_parallel_deg,
                                     site.central_longitude_deg, site.body_radius_m)
    if site.origin_offset_east_m or site.origin_offset_north_m:
        raise ValueError("origin offsets are fixed at 0 in version 1")
    fine_raster = open_height_raster(site.dem_fine_path)
    coarse_raster = open_height_raster(site.dem_coarse_path)
    outermost_fine = layout.regional_grid or near_grid
    sampler = build_dem_sampler(fine_raster, coarse_raster, frame, outermost_fine.extent_east_m / 2 + 8.0)
    dem_workspace, report_workspace = sampler.sample(workspace_grid, require_complete=True)
    dem_near, report_near = sampler.sample(near_grid, require_complete=True)
    regional_height = backdrop_height = None
    dem_report = {"workspace": report_workspace, "near_ring": report_near}
    if layout.regional_grid is not None:
        regional_height, dem_report["regional"] = sampler.sample(layout.regional_grid)
    if layout.backdrop_grid is not None:
        backdrop_height, dem_report["backdrop"] = sample_backdrop(coarse_raster, frame, layout.backdrop_grid,
                                                                  sampler.height_datum_m)
    log(f"[terrain] DEM sampled; datum {sampler.height_datum_m:.3f} m", flush=True)

    if fractal.structure_function_anchor_source == "measured_from_dem":
        anchor_grid = RegularGrid.centered(near_grid.extent_east_m, 2.0)
        anchor_dem, _ = sampler.sample(anchor_grid, require_complete=True)
        anchor_sigma = structure_function(plane_detrend(anchor_dem, 2.0), 2.0, fractal.structure_function_anchor_lag_m)
        low, high = fractal.structure_function_anchor_valid_range_m
        if not low <= anchor_sigma <= high:
            raise ValueError(f"measured DEM structure function {anchor_sigma:.3f} m at "
                             f"{fractal.structure_function_anchor_lag_m} m is outside [{low}, {high}]")
    else:
        anchor_sigma = fractal.structure_function_anchor_m

    table = sample_crater_population(
        craters, equilibrium_coefficient, cutoff, near_ring_half_size_m=near_grid.extent_east_m / 2,
        workspace_half_size_m=workspace_grid.extent_east_m / 2, drivable_half_size_m=layout.drivable_half_size_m,
        minimum_band_b_diameter_m=8 * collision_spacing,
        population_generator=named_generator(seed, "terrain.crater_population"),
        shape_generator=named_generator(seed, "terrain.crater_shape"))
    craters_near = np.zeros(near_grid.shape)
    band_a = np.flatnonzero(table.band == BAND_A)
    stamp_craters(craters_near, near_grid, table, band_a, craters.apron_extent_radii, craters.rim_height_to_diameter,
                  minimum_filter_sigma_m=near_grid.spacing_m / 2, record_realised=True)
    reach = workspace_grid.extent_east_m / 2 + (craters.apron_extent_radii + 1) * table.diameter_m / 2
    near_workspace = np.flatnonzero((np.abs(table.center_east_m) < reach) & (np.abs(table.center_north_m) < reach))
    craters_workspace = np.zeros(workspace_grid.shape)
    stamp_craters(craters_workspace, workspace_grid, table, near_workspace, craters.apron_extent_radii,
                  craters.rim_height_to_diameter, minimum_filter_sigma_m=collision_spacing, record_realised=True)
    log(f"[terrain] {len(table)} craters ({len(near_workspace)} touch the workspace), c = {equilibrium_coefficient:.3f}",
        flush=True)

    macro_low = fractal.macro_band_wavelengths_m[0]
    meso_low = max(fractal.meso_band_wavelengths_m[0], 4 * collision_spacing)
    width = fractal.band_edge_relative_width
    if macro_low / (1 + width) < 2.5 * near_grid.spacing_m:
        raise ValueError("macro band lower edge is not supported by the near-ring lattice")
    macro = synthesize_band_limited_fractal(near_grid.shape, near_grid.spacing_m, fractal.hurst_exponent, anchor_sigma,
                                            fractal.structure_function_anchor_lag_m, macro_low, cutoff, width,
                                            named_generator(seed, "terrain.fractal.macro"))
    meso = synthesize_band_limited_fractal(workspace_grid.shape, workspace_grid.spacing_m, fractal.hurst_exponent,
                                           anchor_sigma, fractal.structure_function_anchor_lag_m, meso_low, macro_low,
                                           width, named_generator(seed, "terrain.fractal.meso"))
    fractal_workspace = resample_bicubic(macro, near_grid, workspace_grid) + meso

    skirt = layout.workspace_skirt_m
    lag = 1.0
    target = roughness_multiplier * anchor_sigma * (lag / fractal.structure_function_anchor_lag_m) ** fractal.hurst_exponent
    spacing = workspace_grid.spacing_m
    dem_sf = structure_function(plane_detrend(interior(dem_workspace, workspace_grid, skirt), spacing), spacing, lag)
    crater_sf = structure_function(plane_detrend(interior(craters_workspace, workspace_grid, skirt), spacing), spacing, lag)
    fractal_sf = structure_function(plane_detrend(interior(fractal_workspace, workspace_grid, skirt), spacing), spacing, lag)
    remaining = max(target ** 2 - dem_sf ** 2 - crater_sf ** 2, 0.0)
    unclipped_gain = np.sqrt(remaining) / fractal_sf
    gain = float(np.clip(unclipped_gain, fractal.minimum_fractal_gain, fractal.maximum_fractal_gain))
    calibration = {
        "equilibrium_coefficient": float(equilibrium_coefficient), "roughness_multiplier": float(roughness_multiplier),
        "anchor_structure_function_m": float(anchor_sigma), "anchor_lag_m": fractal.structure_function_anchor_lag_m,
        "target_structure_function_1m": float(target), "dem_structure_function_1m": float(dem_sf),
        "crater_structure_function_1m": float(crater_sf), "fractal_structure_function_1m": float(fractal_sf),
        "fractal_gain": gain, "fractal_gain_unclipped": float(unclipped_gain),
        "roughness_budget_exceeded": bool(unclipped_gain < fractal.minimum_fractal_gain),
        "macro_band_m": [macro_low, cutoff], "meso_band_m": [meso_low, macro_low],
    }

    near_full = dem_near + craters_near + gain * macro
    workspace_full = dem_workspace + craters_workspace + gain * fractal_workspace

    if terrain.apply_planetary_curvature:
        for grid, heights in ((layout.regional_grid, regional_height), (layout.backdrop_grid, backdrop_height)):
            if grid is not None:
                east, north = grid.coordinate_mesh()
                heights -= planetary_curvature_drop_m(east, north, site.body_radius_m)
    regional_cells = backdrop_cells = None
    if layout.backdrop_grid is not None and layout.regional_grid is not None:
        regional_height = blend_into_outer(layout.regional_grid, regional_height, layout.backdrop_grid,
                                           backdrop_height, layout.regional_skirt_m)
        backdrop_cells = cells_outside_inner(layout.backdrop_grid, layout.regional_grid)
    outer_name, outer_grid, outer_height = ("regional", layout.regional_grid, regional_height) \
        if layout.regional_grid is not None else ("backdrop", layout.backdrop_grid, backdrop_height)
    if outer_grid is not None:
        near_final = blend_into_outer(near_grid, near_full, outer_grid, outer_height, layout.near_ring_skirt_m)
        cells = cells_outside_inner(outer_grid, near_grid)
        if outer_name == "regional":
            regional_cells = cells
        else:
            backdrop_cells = cells
    else:
        near_final = near_full
    workspace_final = blend_into_outer(workspace_grid, workspace_full, near_grid, near_final, skirt)
    near_cells = cells_outside_inner(near_grid, workspace_grid)

    detail = workspace_final - dem_workspace
    lowpass_residual = gaussian_lowpass_fft(gain * fractal_workspace, cutoff / 2, spacing)
    block = int(round(2.0 / spacing))
    fractal_interior = interior(gain * fractal_workspace, workspace_grid, skirt)
    trimmed = fractal_interior[:fractal_interior.shape[0] // block * block, :fractal_interior.shape[1] // block * block]
    block_means = trimmed.reshape(trimmed.shape[0] // block, block, trimmed.shape[1] // block, block).mean(axis=(1, 3))
    final_interior = plane_detrend(interior(workspace_final, workspace_grid, skirt), spacing)
    statistics = {
        "full_surface_structure_function_m": {str(lag_m): structure_function(final_interior, spacing, lag_m)
                                              for lag_m in (0.5, 1.0, 2.0, 4.0)},
        "synthetic_detail_structure_function_m": {
            str(lag_m): structure_function(interior(detail, workspace_grid, skirt), spacing, lag_m)
            for lag_m in (0.5, 1.0, 2.0, 4.0)},
        "rms_slope_deg": {"0.1": root_mean_square_slope_deg(final_interior, spacing, 0.1),
                          "1.0": root_mean_square_slope_deg(final_interior, spacing, 1.0)},
        "dem_band_residual_rms_m": float(np.sqrt(np.mean(interior(lowpass_residual, workspace_grid, skirt) ** 2))),
        "two_metre_block_residual_rms_m": float(np.sqrt(np.mean(block_means ** 2))),
        "crater_count_total": int(len(table)),
        "crater_count_by_diameter_m": {f"{low}-{high}": int(np.sum((table.diameter_m >= low) & (table.diameter_m < high)))
                                       for low, high in ((0.4, 1), (1, 2), (2, 4), (4, 8))},
        "workspace_height_range_m": [float(workspace_final.min()), float(workspace_final.max())],
    }
    east, north = workspace_grid.coordinate_mesh()
    del east, north
    table.center_height_m = _crater_center_heights(table, workspace_grid, workspace_final, near_grid, near_final)
    log(f"[terrain] fractal gain {gain:.2f} (target SF(1 m) {target * 1000:.1f} mm, DEM {dem_sf * 1000:.1f} mm, "
        f"craters {crater_sf * 1000:.1f} mm)", flush=True)
    return TerrainBundle(
        layout=layout, height_datum_m=sampler.height_datum_m,
        workspace_height_m=workspace_final, workspace_dem_height_m=dem_workspace,
        near_ring_height_m=near_final, regional_height_m=regional_height, backdrop_height_m=backdrop_height,
        near_ring_cells_kept=near_cells, regional_cells_kept=regional_cells, backdrop_cells_kept=backdrop_cells,
        crater_table=table, calibration=calibration, statistics=statistics, dem_report=dem_report)


def _crater_center_heights(table: CraterTable, workspace_grid: RegularGrid, workspace_height: np.ndarray,
                           near_grid: RegularGrid, near_height: np.ndarray) -> np.ndarray:
    """Final surface height at each crater centre (workspace lattice where possible, else near ring, else NaN)."""
    from lunar_rover_sim.core.data_model.regular_grid import triangulated_height
    heights, valid = triangulated_height(workspace_height, workspace_grid, table.center_east_m, table.center_north_m)
    near_heights, near_valid = triangulated_height(near_height, near_grid, table.center_east_m, table.center_north_m)
    return np.where(valid, heights, np.where(near_valid, near_heights, np.nan))
