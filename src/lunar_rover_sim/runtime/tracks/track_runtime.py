"""Wheel tracks in the running simulation: sample every physics step, stamp and flush once per control step.

Geometry: the deformed heights and normals of dirty workspace visual tiles are written to their USD ``points`` and
``normals`` (verified: picked up by RTX, about 1 ms per tile). Albedo/smoothness: the TrackField mask is uploaded to
``dynamic://wheel_track_mask`` (at most every ``mask_upload_minimum_interval_s`` of wall time) and read by the
lunar_surface MDL. The PhysX collider is not modified.
"""
from __future__ import annotations

import math
import time

import numpy as np
from pxr import Gf, Sdf, UsdGeom, UsdShade, Vt

from lunar_rover_sim.core.data_model.regular_grid import RegularGrid
from lunar_rover_sim.core.regolith.regolith_synthesis import sample_soil_map
from lunar_rover_sim.core.rocks.rock_placement import RockTable, rasterize_rock_footprints
from lunar_rover_sim.core.rocks.rock_shape_generator import (ALBEDO_CLASSES, RockPrototype, SHAPE_CLASSES,
                                                             prototype_mesh)
from lunar_rover_sim.core.soil_mechanics.bekker_sinkage import (REFERENCE_COHESIVE_MODULUS_N_PER_M2,
                                                                REFERENCE_FRICTIONAL_MODULUS_N_PER_M3)
from lunar_rover_sim.core.tracks.track_field import TrackField, TrackStampGeometry
from lunar_rover_sim.core.tracks.wheel_contact_sampling import WheelContactSampleBuilder

MASK_TEXTURE_NAME = "wheel_track_mask"


def _library_from_arrays(arrays) -> list[RockPrototype]:
    """Rebuild RockPrototype objects (detailed level only is needed for footprints)."""
    library = []
    for index in range(len(arrays["shape_class"])):
        vertices, faces = prototype_mesh(arrays, index, "detailed")
        library.append(RockPrototype(SHAPE_CLASSES[int(arrays["shape_class"][index])],
                                     ALBEDO_CLASSES[int(arrays["albedo_class"][index])], vertices, faces, vertices, faces))
    return library


class TrackRuntime:
    """Owns the TrackField of one session and pushes it to the stage."""

    def __init__(self, session):
        """Build the field, the sampler, the mask texture and bind the mask to the surface material."""
        self.session = session
        scene, configuration = session.scene, session.configuration
        self.configuration = configuration
        tracks, rover = configuration.tracks, configuration.rover
        layout = scene.layout
        self.grid: RegularGrid = layout.workspace_visual_grid
        exclusion = None
        if "rocks" in scene.extra_arrays:
            arrays = scene.extra_arrays["rocks"]
            exclusion = rasterize_rock_footprints(RockTable.from_arrays(arrays), _library_from_arrays(arrays), self.grid,
                                                  configuration.rocks.cobble_minimum_diameter_m, use_instance_id=False) > 0
        geometry = TrackStampGeometry(
            wheel_radius_m=rover.wheel_radius_m, wheel_width_m=rover.wheel_width_m, grouser_count=rover.grouser_count,
            grouser_height_m=rover.grouser_height_m, flank_slope=math.tan(math.radians(tracks.flank_repose_angle_deg)),
            berm_width_m=max(3 * self.grid.spacing_m, tracks.berm_width_wheel_fraction * rover.wheel_width_m),
            berm_volume_fraction=tracks.berm_volume_fraction, pass_separation_time_s=tracks.pass_separation_time_s)
        self.field = TrackField(self.grid, scene.workspace_height_m, layout.visual_tile_cells, geometry, exclusion)
        soil = scene.extra_arrays.get("soil")
        if soil is not None:
            soil_grid_values = soil["soil_grid"]
            soil_grid = RegularGrid(float(soil_grid_values[0]), float(soil_grid_values[1]), float(soil_grid_values[2]),
                                    int(soil_grid_values[3]), int(soil_grid_values[4]))

            def soil_lookup(east, north):
                """SYNTHETIC Bekker moduli at points."""
                return (sample_soil_map(soil["cohesive_modulus_n_per_m2"], soil_grid, east, north),
                        sample_soil_map(soil["frictional_modulus_n_per_m3"], soil_grid, east, north))
        else:
            def soil_lookup(east, north):
                """Reference lunar soil everywhere."""
                count = len(np.atleast_1d(east))
                return np.full(count, REFERENCE_COHESIVE_MODULUS_N_PER_M2), np.full(count, REFERENCE_FRICTIONAL_MODULUS_N_PER_M3)
        self.sample_builder = WheelContactSampleBuilder(rover, tracks, configuration.sinkage, layout.workspace_collision_grid,
                                                        scene.workspace_collision_height_m.astype(np.float64), soil_lookup,
                                                        self.field)
        self.sampling_enabled = False
        self.queue = []
        self.contact_log = []
        self.geometry_enabled = tracks.visual_mode in ("geometry_and_mask", "geometry_only")
        self.mask_enabled = tracks.visual_mode in ("geometry_and_mask", "mask_only")
        self.last_mask_upload_wall_s = -1e9
        self.timings_ms = {"sampling": 0.0, "stamping": 0.0, "tile_writes": 0.0, "mask_upload": 0.0}
        self.tile_attributes = {}
        stage = session.stage
        for key, path in session.terrain_handles.visual_tile_paths.items():
            mesh = UsdGeom.Mesh(stage.GetPrimAtPath(path))
            self.tile_attributes[key] = (mesh.GetPointsAttr(), mesh.GetNormalsAttr())
        self.mask_provider = None
        if self.mask_enabled:
            import omni.ui
            self.mask_provider = omni.ui.DynamicTextureProvider(MASK_TEXTURE_NAME)
            self.force_mask_upload()
            self._bind_mask_to_material()

    def _bind_mask_to_material(self) -> None:
        """Point the lunar_surface MDL track-mask inputs at the dynamic texture."""
        shader = UsdShade.Shader(self.session.stage.GetPrimAtPath(f"{self.session.terrain_handles.surface_material_path}/LunarSurface"))
        texture_input = shader.CreateInput("track_mask_texture", Sdf.ValueTypeNames.Asset)
        texture_input.Set(Sdf.AssetPath(f"dynamic://{MASK_TEXTURE_NAME}"))
        texture_input.GetAttr().SetMetadata("colorSpace", "raw")
        origin = self.field.mask_origin_m
        shader.CreateInput("track_mask_origin_m", Sdf.ValueTypeNames.Float2).Set(Gf.Vec2f(float(origin[0]), float(origin[1])))
        shader.CreateInput("track_mask_size_m", Sdf.ValueTypeNames.Float).Set(float(self.field.mask_size_m))
        shader.CreateInput("track_albedo_factor", Sdf.ValueTypeNames.Float).Set(float(self.configuration.tracks.track_albedo_factor))
        shader.CreateInput("tread_bump_reduction", Sdf.ValueTypeNames.Float).Set(float(self.configuration.tracks.tread_bump_reduction))

    # ------------------------------------------------------------------ per physics step
    def on_physics_step(self, step_dt_s: float, time_s: float) -> None:
        """Sample the four wheels (registered as a stepper listener)."""
        if not self.sampling_enabled:
            return
        started = time.perf_counter()
        state = self.session.rover.read_state()
        samples = self.sample_builder.process(time_s, step_dt_s, state.wheel_center_m, state.wheel_axle_direction,
                                              state.wheel_center_velocity_m_s, state.wheel_angle_rad,
                                              state.wheel_angular_velocity_rad_s, state.wheel_contact_force_n)
        self.queue.extend(samples)
        self.contact_log.extend(samples)
        self.timings_ms["sampling"] += 1000 * (time.perf_counter() - started)

    def reset_sampling(self) -> None:
        """Restart the sampler (accumulators full: each loaded wheel stamps where it stands)."""
        self.sample_builder.reset()
        self.queue.clear()

    # ------------------------------------------------------------------ per control step
    def flush(self) -> None:
        """Stamp queued samples, write dirty tiles, upload the mask when due."""
        started = time.perf_counter()
        for sample in self.queue:
            self.field.stamp(sample)
        self.queue.clear()
        stamped = time.perf_counter()
        self.timings_ms["stamping"] += 1000 * (stamped - started)
        if self.geometry_enabled:
            self._write_tiles(self.field.take_dirty_tiles())
        else:
            self.field.take_dirty_tiles()
        self.timings_ms["tile_writes"] += 1000 * (time.perf_counter() - stamped)
        if self.mask_enabled and self.field.mask_dirty and \
                time.perf_counter() - self.last_mask_upload_wall_s >= self.configuration.tracks.mask_upload_minimum_interval_s:
            self.force_mask_upload()

    def _write_tiles(self, tiles) -> None:
        """Write points (and normals) of tiles from the deformed field."""
        east_all, north_all = self.grid.east_coordinates(), self.grid.north_coordinates()
        for key in tiles:
            if key not in self.tile_attributes:
                continue
            tile = self.field.tile_slice(key)
            heights, normals = self.field.deformed_tile(tile)
            east_mesh, north_mesh = np.meshgrid(east_all[tile.column_start:tile.column_stop],
                                                north_all[tile.row_start:tile.row_stop])
            points = np.column_stack([east_mesh.ravel(), north_mesh.ravel(), heights.ravel()]).astype(np.float32)
            points_attribute, normals_attribute = self.tile_attributes[key]
            points_attribute.Set(Vt.Vec3fArray.FromNumpy(points))
            if self.configuration.tracks.author_normals:
                normals_attribute.Set(Vt.Vec3fArray.FromNumpy(normals.reshape(-1, 3)))

    def force_mask_upload(self) -> None:
        """Upload the mask texture now (before synchronous camera renders)."""
        if self.mask_provider is None:
            return
        started = time.perf_counter()
        image = np.ascontiguousarray(self.field.mask_image())
        self.mask_provider.set_data_array(image, [image.shape[1], image.shape[0]])
        self.last_mask_upload_wall_s = time.perf_counter()
        self.timings_ms["mask_upload"] += 1000 * (self.last_mask_upload_wall_s - started)

    def clear(self) -> None:
        """Erase all tracks and restore the rest surface on the touched tiles."""
        tiles = self.field.clear()
        if self.geometry_enabled:
            self._write_tiles(tiles)
        self.contact_log.clear()
        self.force_mask_upload()
