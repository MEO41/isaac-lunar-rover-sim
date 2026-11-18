"""``bake_scene``: configuration -> scene directory of arrays + manifest (docs/DESIGN.md sections 7.1 and 9.6).

Isaac-free and deterministic. Writes into ``scenes/<scene_id>.partial`` and renames at the end, so a crashed bake
never leaves a directory the runtime would accept. Everything synthetic is labelled so in the manifest.
"""
from __future__ import annotations

import hashlib
import json
import shutil
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import yaml

from lunar_rover_sim.core.configuration.configuration_loader import scene_identifier
from lunar_rover_sim.core.configuration.configuration_models import SimulatorConfiguration
from lunar_rover_sim.core.data_model.regular_grid import RegularGrid
from lunar_rover_sim.core.regolith.regolith_synthesis import (synthesize_soil_property_maps, tile_mean_static_friction,
                                                              write_regolith_textures)
from lunar_rover_sim.core.rocks.rock_placement import (SIZE_CLASS_NAMES, place_rocks, rasterize_rock_footprints,
                                                       rock_fillet_height_m)
from lunar_rover_sim.core.rocks.rock_shape_generator import build_rock_prototype_library, prototype_library_arrays
from lunar_rover_sim.core.scene.scene_rasters import crater_id_raster, local_roughness_m, slope_over_baseline_deg
from lunar_rover_sim.core.scene.spawn_pose import find_spawn_pose
from lunar_rover_sim.core.storage.data_root import DataRoot
from lunar_rover_sim.core.terrain.terrain_builder import TerrainBundle, build_terrain
from lunar_rover_sim.core.terrain.tier_blending import edge_weight

MANIFEST_SCHEMA_VERSION = 1
SYNTHETIC_NOTICE = ("All sub-DEM relief (craters, fractal detail, rock fillets), all rocks and all soil properties are "
                    "SYNTHETIC and uncalibrated. Only the DEM band (wavelengths >= dem_cutoff_wavelength_m) comes from "
                    "LROC NAC measurements.")


def grid_to_dict(grid: RegularGrid | None) -> dict | None:
    """JSON-serialisable description of a lattice."""
    return None if grid is None else asdict(grid)


def file_sha256(path: Path) -> str:
    """SHA-256 hex digest of a file."""
    digest = hashlib.sha256()
    with open(path, "rb") as file_handle:
        for chunk in iter(lambda: file_handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def compute_scene_rasters(terrain: TerrainBundle, rock_id: np.ndarray | None = None) -> dict[str, np.ndarray]:
    """Analysis rasters on the workspace collision lattice."""
    grid = terrain.layout.workspace_collision_grid
    heights = terrain.workspace_collision_height_m
    rasters = {
        "height_m": heights.astype(np.float32),
        "slope_deg": slope_over_baseline_deg(heights, grid.spacing_m).astype(np.float32),
        "roughness_m": local_roughness_m(heights, grid.spacing_m).astype(np.float32),
        "crater_id": crater_id_raster(grid, terrain.crater_table),
    }
    rasters["rock_id"] = rock_id if rock_id is not None else np.zeros(grid.shape, dtype=np.int32)
    return rasters


def write_preview_png(path: Path, terrain: TerrainBundle, sun_azimuth_deg: float, sun_elevation_deg: float,
                      spawn_east_m: float, spawn_north_m: float) -> None:
    """Hillshade of the workspace (row 0 = north) with crater rims and the spawn point marked."""
    from PIL import Image, ImageDraw
    grid = terrain.layout.workspace_visual_grid
    heights = terrain.workspace_height_m
    slope_north, slope_east = np.gradient(heights, grid.spacing_m)
    normal = np.stack([-slope_east, -slope_north, np.ones_like(heights)], axis=-1)
    normal /= np.linalg.norm(normal, axis=-1, keepdims=True)
    azimuth, elevation = np.radians(sun_azimuth_deg), np.radians(sun_elevation_deg)
    sun = np.array([np.sin(azimuth) * np.cos(elevation), np.cos(azimuth) * np.cos(elevation), np.sin(elevation)])
    shade = np.clip(normal @ sun, 0.0, 1.0)
    image = Image.fromarray(np.flipud(np.round(255 * shade ** 0.8)).astype(np.uint8), mode="L").convert("RGB")
    draw = ImageDraw.Draw(image)
    table = terrain.crater_table

    def to_pixel(east: float, north: float) -> tuple[float, float]:
        """Image pixel of a local point (row 0 = north)."""
        return (east - grid.origin_east_m) / grid.spacing_m, (grid.maximum_north_m - north) / grid.spacing_m

    for index in np.flatnonzero(table.diameter_m >= 1.0):
        x, y = to_pixel(table.center_east_m[index], table.center_north_m[index])
        radius = table.diameter_m[index] / 2 / grid.spacing_m
        colour = (255, 80, 80) if table.is_feature_crater[index] else (90, 160, 255)
        draw.ellipse([x - radius, y - radius, x + radius, y + radius], outline=colour)
    x, y = to_pixel(spawn_east_m, spawn_north_m)
    draw.ellipse([x - 6, y - 6, x + 6, y + 6], outline=(60, 255, 60), width=2)
    image.save(path, optimize=True)


def bake_scene(configuration: SimulatorConfiguration, data_root: DataRoot, overwrite: bool = False,
               log=print) -> Path:
    """Bake one scene and return its directory (raises FileExistsError unless ``overwrite``)."""
    started = time.time()
    scene_id = scene_identifier(configuration)
    final_directory = data_root.scenes_directory / scene_id
    if final_directory.exists():
        if not overwrite:
            raise FileExistsError(f"scene {final_directory} exists; pass overwrite to rebake")
        shutil.rmtree(final_directory)
    partial_directory = data_root.scenes_directory / f"{scene_id}.partial"
    if partial_directory.exists():
        shutil.rmtree(partial_directory)
    partial_directory.mkdir(parents=True)

    terrain = build_terrain(configuration, log=log)
    layout = terrain.layout
    table = terrain.crater_table
    seed = configuration.scene_seed

    soil_grid = RegularGrid.centered(layout.near_ring_grid.extent_east_m, configuration.regolith.property_map_spacing_m)
    soil = synthesize_soil_property_maps(soil_grid, configuration.regolith, table, seed)
    library = build_rock_prototype_library(configuration.rocks.prototypes_per_shape_class, seed)
    rock_table, rock_report = place_rocks(configuration.rocks, library, layout.workspace_visual_grid,
                                          terrain.workspace_height_m, layout.workspace_skirt_m, layout.near_ring_grid,
                                          terrain.near_ring_height_m, seed)
    fillet = rock_fillet_height_m(rock_table, library, layout.workspace_visual_grid, configuration.rocks, seed)
    terrain.workspace_height_m = terrain.workspace_height_m + \
        edge_weight(layout.workspace_visual_grid, layout.workspace_skirt_m) * fillet
    log(f"[bake] {len(rock_table)} rocks placed {rock_report['counts_by_size_class']}, "
        f"{rock_report['near_ring_rock_count']} in the near ring; fillet max {fillet.max() * 1000:.1f} mm", flush=True)
    rock_id = rasterize_rock_footprints(rock_table, library, layout.workspace_collision_grid,
                                        configuration.rocks.cobble_minimum_diameter_m, use_instance_id=False)
    rasters = compute_scene_rasters(terrain, rock_id)
    tile_friction = tile_mean_static_friction(soil, layout.workspace_collision_grid, layout.collision_tile_cells)
    regolith_parameters = write_regolith_textures(partial_directory / "textures", layout.near_ring_grid,
                                                  configuration.regolith, soil, table, seed)
    np.savez(partial_directory / "soil.npz", **soil.to_arrays(), collision_tile_static_friction=tile_friction,
             mean_static_friction=np.float64(soil.static_friction.mean()))
    np.savez(partial_directory / "rocks.npz", **rock_table.to_arrays(), **prototype_library_arrays(library))
    spawn_pose, spawn_report = find_spawn_pose(
        terrain.workspace_collision_height_m, layout.workspace_collision_grid, rasters["slope_deg"],
        rasters["roughness_m"], rasters["rock_id"], table.center_east_m, table.center_north_m, table.diameter_m,
        layout.drivable_half_size_m, configuration.spawn, configuration.rover)

    terrain_arrays = {
        "workspace_height_m": terrain.workspace_height_m.astype(np.float32),
        "workspace_dem_height_m": terrain.workspace_dem_height_m.astype(np.float32),
        "near_ring_height_m": terrain.near_ring_height_m.astype(np.float32),
        "near_ring_cells_kept": terrain.near_ring_cells_kept,
    }
    for name in ("regional", "backdrop"):
        heights = getattr(terrain, f"{name}_height_m")
        if heights is not None:
            terrain_arrays[f"{name}_height_m"] = heights.astype(np.float32)
            terrain_arrays[f"{name}_cells_kept"] = getattr(terrain, f"{name}_cells_kept")
    np.savez(partial_directory / "terrain.npz", **terrain_arrays)
    np.savez(partial_directory / "craters.npz", **table.to_arrays())
    np.savez(partial_directory / "rasters.npz", **rasters)
    (partial_directory / "configuration_resolved.yaml").write_text(
        yaml.safe_dump(configuration.model_dump(mode="json"), sort_keys=False))
    write_preview_png(partial_directory / "preview.png", terrain, configuration.lighting.sun_azimuth_deg,
                      max(configuration.lighting.sun_elevation_deg, 20.0), spawn_pose.position_m[0], spawn_pose.position_m[1])

    manifest = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "scene_id": scene_id,
        "generator_version": configuration.generator_version,
        "scene_seed": configuration.scene_seed,
        "preset": configuration.profile.preset,
        "site": {"name": configuration.site.name, "origin_longitude_deg": configuration.site.origin_longitude_deg,
                 "origin_latitude_deg": configuration.site.origin_latitude_deg,
                 "height_datum_m": terrain.height_datum_m,
                 "dem_fine_file": Path(configuration.site.dem_fine_path).name,
                 "dem_coarse_file": Path(configuration.site.dem_coarse_path).name,
                 "dem_backdrop_file": (Path(configuration.site.dem_backdrop_path).name
                                       if configuration.site.dem_backdrop_path else None)},
        "frames": "x east, y north, z up, metres; z = 0 at the bicubic DEM height of the origin; quaternions xyzw",
        "synthetic_notice": SYNTHETIC_NOTICE,
        "layout": {
            "workspace_visual_grid": grid_to_dict(layout.workspace_visual_grid),
            "workspace_collision_grid": grid_to_dict(layout.workspace_collision_grid),
            "near_ring_grid": grid_to_dict(layout.near_ring_grid),
            "regional_grid": grid_to_dict(layout.regional_grid),
            "backdrop_grid": grid_to_dict(layout.backdrop_grid),
            "drivable_half_size_m": layout.drivable_half_size_m,
        },
        "spawn_pose": {"position_m": spawn_pose.position_m, "orientation_xyzw": spawn_pose.orientation_xyzw,
                       "heading_rad": spawn_pose.heading_rad, **spawn_report},
        "calibration": terrain.calibration,
        "statistics": terrain.statistics,
        "dem_report": terrain.dem_report,
        "craters": [
            {"crater_id": int(index + 1), "center_m": [float(table.center_east_m[index]), float(table.center_north_m[index]),
                                                       float(table.center_height_m[index])],
             "diameter_m": float(table.diameter_m[index]), "realised_depth_m": float(table.realised_depth_m[index]),
             "freshness_class": int(table.freshness_class[index]), "is_feature_crater": bool(table.is_feature_crater[index]),
             "semantic_class": "crater", "synthetic": True}
            for index in np.flatnonzero((np.abs(table.center_east_m) <= layout.workspace_visual_grid.maximum_east_m) &
                                        (np.abs(table.center_north_m) <= layout.workspace_visual_grid.maximum_north_m) &
                                        (table.diameter_m >= 0.5))
        ],
        "rocks": [
            {"instance_id": int(rock_table.instance_id[index]), "semantic_class": "rock",
             "size_class": SIZE_CLASS_NAMES[rock_table.size_class[index]],
             "shape_class": library[rock_table.prototype_index[index]].shape_class,
             "albedo_class": library[rock_table.prototype_index[index]].albedo_class,
             "center_m": [float(v) for v in rock_table.position_m[index]],
             "orientation_xyzw": [float(v) for v in rock_table.orientation_xyzw[index]],
             "diameter_m": float(rock_table.diameter_m[index]),
             "height_above_ground_m": float(library[rock_table.prototype_index[index]].height_m
                                            * rock_table.diameter_m[index] * (1 - rock_table.burial_fraction[index])),
             "prim_path": f"/World/Rocks/Registered/rock_{int(rock_table.instance_id[index]):05d}", "synthetic": True}
            for index in np.argsort(rock_table.instance_id) if rock_table.instance_id[index] > 0
        ],
        "rock_report": rock_report,
        "regolith": regolith_parameters,
        "bake_seconds": None,
        "files_sha256": {},
    }
    manifest["files_sha256"] = {str(path.relative_to(partial_directory)): file_sha256(path)
                                for path in sorted(partial_directory.rglob("*")) if path.is_file()}
    manifest["bake_seconds"] = round(time.time() - started, 2)
    (partial_directory / "manifest.json").write_text(json.dumps(manifest, indent=1))
    partial_directory.rename(final_directory)
    log(f"[bake] scene {scene_id} written to {final_directory} in {manifest['bake_seconds']} s", flush=True)
    return final_directory
