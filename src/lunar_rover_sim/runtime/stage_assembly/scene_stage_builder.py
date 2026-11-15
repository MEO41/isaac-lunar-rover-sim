"""``build_scene_stage``: a baked scene -> USD stage (physics scene, terrain tiers, materials, Sun).

Prim layout:
  /World/PhysicsScene
  /World/Looks/{LunarSurface, LunarSurfaceFar, Physics/...}
  /World/Terrain/Workspace/Visual/tile_<row>_<column>     rendered workspace tiles (no collider)
  /World/Terrain/Workspace/Collision/tile_<row>_<column>  invisible exact-triangle colliders (visual[::d, ::d])
  /World/Terrain/NearRing                                 rendered AND collided, hole under the workspace
  /World/Terrain/Regional, /World/Terrain/Backdrop        rendered only, holes under inner tiers
  /World/Sun
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from pxr import Gf, PhysxSchema, Sdf, UsdGeom, UsdLux, UsdPhysics, UsdShade

from lunar_rover_sim.core.data_model.regular_grid import (RegularGrid, TileSlice, lattice_triangle_indices,
                                                          vertex_normals)
from lunar_rover_sim.core.scene.scene_loader import LoadedScene
from lunar_rover_sim.core.storage.data_root import PROJECT_ROOT
from lunar_rover_sim.runtime.stage_assembly.usd_helpers import (bind_render_material, define_physics_material,
                                                                define_triangle_mesh, make_static_triangle_collider,
                                                                set_semantic_class)

MATERIAL_FILE = PROJECT_ROOT / "assets" / "materials" / "lunar_surface.mdl"
REGOLITH_PHYSICS_MATERIAL = "/World/Looks/Physics/Regolith"
WHEEL_PHYSICS_MATERIAL = "/World/Looks/Physics/Wheel"
ROCK_PHYSICS_MATERIAL = "/World/Looks/Physics/Rock"


@dataclass
class TerrainStageHandles:
    """Paths and tiles of the authored terrain, needed later by the track runtime and the verification code."""

    visual_tile_paths: dict[tuple[int, int], str] = field(default_factory=dict)
    visual_tiles: dict[tuple[int, int], TileSlice] = field(default_factory=dict)
    collision_tile_paths: list[str] = field(default_factory=list)
    surface_material_path: str = "/World/Looks/LunarSurface"
    triangle_counts: dict[str, int] = field(default_factory=dict)


def create_stage():
    """New in-memory stage: Z up, metres, /World as default prim."""
    import omni.usd
    context = omni.usd.get_context()
    context.new_stage()
    stage = context.get_stage()
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    world = UsdGeom.Xform.Define(stage, "/World")
    stage.SetDefaultPrim(world.GetPrim())
    return stage


def build_physics_scene(stage, scene: LoadedScene) -> None:
    """PhysicsScene with lunar gravity, TGS, CPU dynamics, and the physics materials."""
    physics = scene.configuration.physics
    physics_scene = UsdPhysics.Scene.Define(stage, "/World/PhysicsScene")
    physics_scene.CreateGravityDirectionAttr(Gf.Vec3f(0.0, 0.0, -1.0))
    physics_scene.CreateGravityMagnitudeAttr(float(physics.gravity_m_s2))
    physx_scene = PhysxSchema.PhysxSceneAPI.Apply(physics_scene.GetPrim())
    physx_scene.CreateSolverTypeAttr(physics.solver_type)
    physx_scene.CreateTimeStepsPerSecondAttr(int(physics.physics_rate_hz))
    physx_scene.CreateEnableGPUDynamicsAttr(bool(physics.gpu_dynamics_enabled))
    physx_scene.CreateBroadphaseTypeAttr("GPU" if physics.gpu_dynamics_enabled else physics.broadphase_type)
    physx_scene.CreateEnableCCDAttr(False)
    physx_scene.CreateEnableStabilizationAttr(False)
    soil_friction = scene.extra_arrays.get("soil", {}).get("mean_static_friction")
    static = float(soil_friction) if soil_friction is not None else 0.7
    define_physics_material(stage, REGOLITH_PHYSICS_MATERIAL, static, 0.875 * static)
    define_physics_material(stage, WHEEL_PHYSICS_MATERIAL, physics.wheel_friction, physics.wheel_friction,
                            combine_mode="multiply")
    define_physics_material(stage, ROCK_PHYSICS_MATERIAL, physics.rock_friction_static, physics.rock_friction_dynamic)


def build_surface_materials(stage, scene: LoadedScene) -> None:
    """MDL lunar_surface materials (near: textured; far: constant albedo) with a UsdPreviewSurface fallback."""
    textures = scene.directory / "textures"
    regolith = scene.configuration.regolith
    albedo_constant = float(scene.manifest.get("regolith", {}).get("base_normal_albedo", regolith.base_normal_albedo))
    near_grid = scene.layout.near_ring_grid
    for path, textured in (("/World/Looks/LunarSurface", True), ("/World/Looks/LunarSurfaceFar", False)):
        material = UsdShade.Material.Define(stage, path)
        preview = UsdShade.Shader.Define(stage, f"{path}/PreviewSurface")
        preview.CreateIdAttr("UsdPreviewSurface")
        preview.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(albedo_constant * 1.1))
        preview.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.95)
        material.CreateSurfaceOutput().ConnectToSource(preview.ConnectableAPI(), "surface")
        shader = UsdShade.Shader.Define(stage, f"{path}/LunarSurface")
        shader.SetSourceAsset(Sdf.AssetPath(str(MATERIAL_FILE)), "mdl")
        shader.SetSourceAssetSubIdentifier("lunar_surface", "mdl")
        shader.CreateInput("albedo_constant", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(albedo_constant))
        if textured:
            albedo_file = textures / "macro_albedo.png"
            if albedo_file.exists():
                shader.CreateInput("albedo_texture", Sdf.ValueTypeNames.Asset).Set(Sdf.AssetPath(str(albedo_file)))
                shader.GetInput("albedo_texture").GetAttr().SetMetadata("colorSpace", "sRGB")
                shader.CreateInput("albedo_origin_m", Sdf.ValueTypeNames.Float2).Set(
                    Gf.Vec2f(near_grid.origin_east_m, near_grid.origin_north_m))
                shader.CreateInput("albedo_size_m", Sdf.ValueTypeNames.Float).Set(float(near_grid.extent_east_m))
            for name, octave in (("coarse", regolith.detail_texture_coarse), ("fine", regolith.detail_texture_fine)):
                detail_file = textures / f"detail_{name}.png"
                if detail_file.exists():
                    detail_input = shader.CreateInput(f"detail_{name}_texture", Sdf.ValueTypeNames.Asset)
                    detail_input.Set(Sdf.AssetPath(str(detail_file)))
                    detail_input.GetAttr().SetMetadata("colorSpace", "raw")
                    shader.CreateInput(f"detail_{name}_tile_m", Sdf.ValueTypeNames.Float).Set(float(octave.tile_size_m))
        material.CreateSurfaceOutput("mdl").ConnectToSource(shader.ConnectableAPI(), "out")


def _mesh_arrays_for_cells(grid: RegularGrid, heights: np.ndarray, cells_kept: np.ndarray | None):
    """Points, triangles (only kept cells), normals and world-metre st for a whole tier lattice."""
    points = grid.vertex_positions(heights)
    triangles = lattice_triangle_indices(grid.count_north, grid.count_east)
    if cells_kept is not None:
        triangles = triangles.reshape(-1, 2, 3)[cells_kept.ravel()].reshape(-1, 3)
    normals = vertex_normals(heights, grid.spacing_m).reshape(-1, 3)
    return points, triangles, normals, points[:, :2].copy()


def build_terrain_prims(stage, scene: LoadedScene) -> TerrainStageHandles:
    """Author every terrain tier; returns handles of the workspace tiles for later updates."""
    layout = scene.layout
    handles = TerrainStageHandles()
    terrain_root = UsdGeom.Xform.Define(stage, "/World/Terrain")
    set_semantic_class(terrain_root.GetPrim(), "regolith")
    UsdGeom.Scope.Define(stage, "/World/Terrain/Workspace")

    visual_grid = layout.workspace_visual_grid
    heights = scene.workspace_height_m.astype(np.float64)
    normals = vertex_normals(heights, visual_grid.spacing_m)
    tile_cells = layout.visual_tile_cells
    tile_triangles = lattice_triangle_indices(tile_cells + 1, tile_cells + 1)
    triangle_total = 0
    for tile in visual_grid.split_into_tiles(tile_cells):
        key = (tile.row_start // tile_cells, tile.column_start // tile_cells)
        path = f"/World/Terrain/Workspace/Visual/tile_{key[0]:02d}_{key[1]:02d}"
        points = visual_grid.vertex_positions(heights, tile)
        mesh = define_triangle_mesh(stage, path, points, tile_triangles,
                                    normals=normals[tile.array_slice()].reshape(-1, 3),
                                    texture_coordinates=points[:, :2])
        bind_render_material(mesh.GetPrim(), handles.surface_material_path)
        handles.visual_tile_paths[key] = path
        handles.visual_tiles[key] = tile
        triangle_total += len(tile_triangles)
    handles.triangle_counts["workspace_visual"] = triangle_total

    collision_grid = layout.workspace_collision_grid
    collision_heights = scene.workspace_collision_height_m.astype(np.float64)
    collision_cells = layout.collision_tile_cells
    collision_triangles = lattice_triangle_indices(collision_cells + 1, collision_cells + 1)
    for tile in collision_grid.split_into_tiles(collision_cells):
        key = (tile.row_start // collision_cells, tile.column_start // collision_cells)
        path = f"/World/Terrain/Workspace/Collision/tile_{key[0]:02d}_{key[1]:02d}"
        mesh = define_triangle_mesh(stage, path, collision_grid.vertex_positions(collision_heights, tile),
                                    collision_triangles)
        make_static_triangle_collider(mesh.GetPrim(), REGOLITH_PHYSICS_MATERIAL)
        UsdGeom.Imageable(mesh.GetPrim()).MakeInvisible()
        handles.collision_tile_paths.append(path)
    handles.triangle_counts["workspace_collision"] = len(handles.collision_tile_paths) * len(collision_triangles)

    arrays = scene.terrain_arrays
    tiers = [("NearRing", layout.near_ring_grid, "near_ring", True, "/World/Looks/LunarSurface")]
    if layout.regional_grid is not None:
        tiers.append(("Regional", layout.regional_grid, "regional", False, "/World/Looks/LunarSurface"))
    if layout.backdrop_grid is not None:
        tiers.append(("Backdrop", layout.backdrop_grid, "backdrop", False, "/World/Looks/LunarSurfaceFar"))
    for prim_name, grid, key, collided, material_path in tiers:
        points, triangles, tier_normals, texture_coordinates = _mesh_arrays_for_cells(
            grid, arrays[f"{key}_height_m"].astype(np.float64), arrays.get(f"{key}_cells_kept"))
        mesh = define_triangle_mesh(stage, f"/World/Terrain/{prim_name}", points, triangles, tier_normals,
                                    texture_coordinates)
        bind_render_material(mesh.GetPrim(), material_path)
        if collided:
            make_static_triangle_collider(mesh.GetPrim(), REGOLITH_PHYSICS_MATERIAL)
        handles.triangle_counts[key] = len(triangles)
    return handles


def build_lighting(stage, scene: LoadedScene, azimuth_deg: float | None = None, elevation_deg: float | None = None) -> None:
    """One distant Sun: rotateZ(90 - azimuth) * rotateY(90 - elevation) turns its local +Z towards the Sun."""
    lighting = scene.configuration.lighting
    azimuth = lighting.sun_azimuth_deg if azimuth_deg is None else azimuth_deg
    elevation = lighting.sun_elevation_deg if elevation_deg is None else elevation_deg
    sun = UsdLux.DistantLight.Define(stage, "/World/Sun")
    sun.CreateIntensityAttr(float(lighting.sun_base_intensity * lighting.exposure_scale))
    sun.CreateAngleAttr(float(lighting.sun_angular_diameter_deg))
    sun.CreateColorAttr(Gf.Vec3f(*map(float, lighting.sun_color)))
    xformable = UsdGeom.Xformable(sun)
    xformable.ClearXformOpOrder()
    xformable.AddRotateZOp().Set(float(90.0 - azimuth))
    xformable.AddRotateYOp().Set(float(90.0 - elevation))


def build_scene_stage(scene: LoadedScene, include_rocks: bool = True) -> tuple[object, TerrainStageHandles]:
    """Create the stage and author physics, materials, terrain, rocks (if baked) and the Sun."""
    stage = create_stage()
    build_physics_scene(stage, scene)
    build_surface_materials(stage, scene)
    handles = build_terrain_prims(stage, scene)
    if include_rocks and "rocks" in scene.extra_arrays:
        from lunar_rover_sim.runtime.stage_assembly.rock_stage_builder import build_rock_prims
        build_rock_prims(stage, scene)
    build_lighting(stage, scene)
    return stage, handles


def material_file_path() -> Path:
    """Absolute path of the project's MDL file."""
    return MATERIAL_FILE
