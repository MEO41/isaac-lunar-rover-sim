"""Rocks on the stage (docs/DESIGN.md sections 7.4 and 9.3).

  /World/RockPrototypes (parked 1000 m below the surface; children are referenced, parents are not inherited)
      proto_XX/geom          detailed mesh, convex-hull collider, rock physics material, lunar_rock MDL (albedo class)
  /World/Rocks/Registered/rock_NNNNN   instanceable reference, label 'rock', collider (>= registration diameter)
  /World/Rocks/Cobbles/cobble_NNNNN    instanceable reference, label 'rock', collider
  /World/Rocks/Pebbles                 PointInstancer (coarse meshes), no collider, unlabelled
  /World/Rocks/NearRing                PointInstancer (coarse meshes), no collider
Instances are uniformly scaled by their diameter (prototypes have a 1 m longest axis).
"""
from __future__ import annotations

import numpy as np
from pxr import Gf, PhysxSchema, Sdf, UsdGeom, UsdPhysics, UsdShade, Vt

from lunar_rover_sim.core.rocks.rock_placement import REGION_NEAR_RING, REGION_WORKSPACE, RockTable
from lunar_rover_sim.core.rocks.rock_shape_generator import ALBEDO_CLASSES, prototype_mesh
from lunar_rover_sim.core.scene.scene_loader import LoadedScene
from lunar_rover_sim.runtime.stage_assembly.scene_stage_builder import MATERIAL_FILE, ROCK_PHYSICS_MATERIAL
from lunar_rover_sim.runtime.stage_assembly.usd_helpers import bind_physics_material, bind_render_material, define_triangle_mesh

PROTOTYPE_ROOT = "/World/RockPrototypes"


def _cube_projection_coordinates(vertices: np.ndarray) -> np.ndarray:
    """Texture coordinates from the two axes orthogonal to each vertex's dominant direction (metres)."""
    dominant = np.argmax(np.abs(vertices), axis=1)
    first = np.where(dominant == 0, vertices[:, 1], vertices[:, 0])
    second = np.where(dominant == 2, vertices[:, 1], vertices[:, 2])
    return np.column_stack([first, second]).astype(np.float32)


def _vertex_normals(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Area-weighted vertex normals."""
    face_normals = np.cross(vertices[faces[:, 1]] - vertices[faces[:, 0]], vertices[faces[:, 2]] - vertices[faces[:, 0]])
    normals = np.zeros_like(vertices, dtype=np.float64)
    for corner in range(3):
        np.add.at(normals, faces[:, corner], face_normals)
    return (normals / np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-12)).astype(np.float32)


def _rock_materials(stage, scene: LoadedScene) -> dict[str, str]:
    """One lunar_rock MDL material per albedo class; returns {albedo_class: material path}."""
    values = scene.configuration.rocks.albedo_values
    detail = scene.directory / "textures" / "detail_coarse.png"
    paths = {}
    for albedo_class in ALBEDO_CLASSES:
        path = f"/World/Looks/Rock_{albedo_class}"
        albedo = float(getattr(values, albedo_class))
        material = UsdShade.Material.Define(stage, path)
        preview = UsdShade.Shader.Define(stage, f"{path}/PreviewSurface")
        preview.CreateIdAttr("UsdPreviewSurface")
        preview.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(albedo))
        preview.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.9)
        material.CreateSurfaceOutput().ConnectToSource(preview.ConnectableAPI(), "surface")
        shader = UsdShade.Shader.Define(stage, f"{path}/LunarRock")
        shader.SetSourceAsset(Sdf.AssetPath(str(MATERIAL_FILE)), "mdl")
        shader.SetSourceAssetSubIdentifier("lunar_rock", "mdl")
        shader.CreateInput("albedo_constant", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(albedo, albedo * 0.98, albedo * 0.95))
        if detail.exists():
            detail_input = shader.CreateInput("detail_texture", Sdf.ValueTypeNames.Asset)
            detail_input.Set(Sdf.AssetPath(str(detail)))
            detail_input.GetAttr().SetMetadata("colorSpace", "raw")
            shader.CreateInput("detail_tile_m", Sdf.ValueTypeNames.Float).Set(0.35)
        material.CreateSurfaceOutput("mdl").ConnectToSource(shader.ConnectableAPI(), "out")
        paths[albedo_class] = path
    return paths


def _instancer(stage, path: str, table: RockTable, indices: np.ndarray, prototype_paths: list[str],
               prototype_lookup: dict[int, int]) -> None:
    """PointInstancer of the selected rocks (orientations as half-precision quaternions w-first in Gf)."""
    instancer = UsdGeom.PointInstancer.Define(stage, path)
    instancer.CreatePrototypesRel().SetTargets([Sdf.Path(p) for p in prototype_paths])
    instancer.CreatePositionsAttr(Vt.Vec3fArray.FromNumpy(table.position_m[indices].astype(np.float32)))
    quaternions = table.orientation_xyzw[indices]
    instancer.CreateOrientationsAttr(Vt.QuathArray([Gf.Quath(float(q[3]), float(q[0]), float(q[1]), float(q[2]))
                                                    for q in quaternions]))
    scales = np.repeat(table.diameter_m[indices, None], 3, axis=1).astype(np.float32)
    instancer.CreateScalesAttr(Vt.Vec3fArray.FromNumpy(scales))
    instancer.CreateProtoIndicesAttr(Vt.IntArray.FromNumpy(
        np.array([prototype_lookup[int(p)] for p in table.prototype_index[indices]], dtype=np.int32)))


def build_rock_prims(stage, scene: LoadedScene) -> dict:
    """Author prototypes, collided rock references and pebble instancers; returns counts."""
    arrays = scene.extra_arrays["rocks"]
    table = RockTable.from_arrays(arrays)
    rocks = scene.configuration.rocks
    materials = _rock_materials(stage, scene)
    albedo_classes = arrays["albedo_class"]
    prototype_count = len(arrays["shape_class"])
    root = UsdGeom.Xform.Define(stage, PROTOTYPE_ROOT)
    UsdGeom.Xformable(root).AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, -1000.0))
    UsdGeom.Xform.Define(stage, "/World/Rocks")
    collided = (table.region == REGION_WORKSPACE) & (table.diameter_m >= rocks.cobble_minimum_diameter_m)
    used = sorted(set(int(p) for p in table.prototype_index[collided]))
    for index in used:
        vertices, faces = prototype_mesh(arrays, index, "detailed")
        UsdGeom.Xform.Define(stage, f"{PROTOTYPE_ROOT}/proto_{index:02d}")
        mesh = define_triangle_mesh(stage, f"{PROTOTYPE_ROOT}/proto_{index:02d}/geom", vertices, faces,
                                    normals=_vertex_normals(vertices, faces),
                                    texture_coordinates=_cube_projection_coordinates(vertices))
        bind_render_material(mesh.GetPrim(), materials[ALBEDO_CLASSES[int(albedo_classes[index])]])
        UsdPhysics.CollisionAPI.Apply(mesh.GetPrim())
        UsdPhysics.MeshCollisionAPI.Apply(mesh.GetPrim()).CreateApproximationAttr(UsdPhysics.Tokens.convexHull)
        PhysxSchema.PhysxConvexHullCollisionAPI.Apply(mesh.GetPrim()).CreateHullVertexLimitAttr(255)
        bind_physics_material(mesh.GetPrim(), ROCK_PHYSICS_MATERIAL)

    layer = stage.GetRootLayer()
    counts = {"registered": 0, "cobbles": 0}
    with Sdf.ChangeBlock():
        for group in ("Registered", "Cobbles"):
            spec = Sdf.CreatePrimInLayer(layer, f"/World/Rocks/{group}")
            spec.specifier = Sdf.SpecifierDef
            spec.typeName = "Xform"
        for index in np.flatnonzero(collided):
            registered = table.instance_id[index] > 0
            path = f"/World/Rocks/Registered/rock_{int(table.instance_id[index]):05d}" if registered \
                else f"/World/Rocks/Cobbles/cobble_{index:05d}"
            spec = Sdf.CreatePrimInLayer(layer, path)
            spec.specifier = Sdf.SpecifierDef
            spec.typeName = "Xform"
            spec.instanceable = True
            spec.referenceList.Prepend(Sdf.Reference(primPath=f"{PROTOTYPE_ROOT}/proto_{int(table.prototype_index[index]):02d}"))
            x, y, z = map(float, table.position_m[index])
            qx, qy, qz, qw = map(float, table.orientation_xyzw[index])
            Sdf.AttributeSpec(spec, "xformOp:translate", Sdf.ValueTypeNames.Double3).default = Gf.Vec3d(x, y, z)
            Sdf.AttributeSpec(spec, "xformOp:orient", Sdf.ValueTypeNames.Quatd).default = Gf.Quatd(qw, qx, qy, qz)
            diameter = float(table.diameter_m[index])
            Sdf.AttributeSpec(spec, "xformOp:scale", Sdf.ValueTypeNames.Double3).default = Gf.Vec3d(diameter, diameter, diameter)
            Sdf.AttributeSpec(spec, "xformOpOrder", Sdf.ValueTypeNames.TokenArray, Sdf.VariabilityUniform).default = \
                Vt.TokenArray(["xformOp:translate", "xformOp:orient", "xformOp:scale"])
            spec.SetInfo("apiSchemas", Sdf.TokenListOp.CreateExplicit(["SemanticsLabelsAPI:class"]))
            Sdf.AttributeSpec(spec, "semantics:labels:class", Sdf.ValueTypeNames.TokenArray).default = Vt.TokenArray(["rock"])
            counts["registered" if registered else "cobbles"] += 1

    visual_only = {"Pebbles": (table.region == REGION_WORKSPACE) & ~collided, "NearRing": table.region == REGION_NEAR_RING}
    for name, mask in visual_only.items():
        indices = np.flatnonzero(mask)
        counts[name.lower()] = int(len(indices))
        if not len(indices):
            continue
        prototypes_used = sorted(set(int(p) for p in table.prototype_index[indices]))
        prototype_paths = []
        for prototype in prototypes_used:
            vertices, faces = prototype_mesh(arrays, prototype, "coarse")
            path = f"/World/Rocks/{name}/Prototypes/proto_{prototype:02d}"
            mesh = define_triangle_mesh(stage, path, vertices, faces, normals=_vertex_normals(vertices, faces),
                                        texture_coordinates=_cube_projection_coordinates(vertices))
            bind_render_material(mesh.GetPrim(), materials[ALBEDO_CLASSES[int(albedo_classes[prototype])]])
            prototype_paths.append(path)
        _instancer(stage, f"/World/Rocks/{name}", table, indices, prototype_paths,
                   {prototype: position for position, prototype in enumerate(prototypes_used)})
    counts["prototypes"] = len(used)
    del prototype_count
    return counts
