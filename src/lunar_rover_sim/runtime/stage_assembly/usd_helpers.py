"""Small USD authoring helpers shared by the stage builders (import only after SimulationApp exists)."""
from __future__ import annotations

import numpy as np
from pxr import Gf, Sdf, UsdGeom, UsdPhysics, UsdSemantics, UsdShade, Vt


def define_triangle_mesh(stage, path: str, points: np.ndarray, triangles: np.ndarray, normals: np.ndarray | None = None,
                         texture_coordinates: np.ndarray | None = None, display_color=(0.3, 0.3, 0.3)) -> UsdGeom.Mesh:
    """Author a triangle mesh: points (N,3) float32, triangles (M,3) int32, optional vertex normals and st (N,2)."""
    mesh = UsdGeom.Mesh.Define(stage, path)
    mesh.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(np.ascontiguousarray(points, dtype=np.float32)))
    mesh.CreateFaceVertexCountsAttr(Vt.IntArray.FromNumpy(np.full(len(triangles), 3, dtype=np.int32)))
    mesh.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(np.ascontiguousarray(triangles, dtype=np.int32).ravel()))
    mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
    mesh.CreateOrientationAttr(UsdGeom.Tokens.rightHanded)
    minimum, maximum = points.min(axis=0), points.max(axis=0)
    mesh.CreateExtentAttr(Vt.Vec3fArray([Gf.Vec3f(*map(float, minimum)), Gf.Vec3f(*map(float, maximum))]))
    if normals is not None:
        mesh.CreateNormalsAttr(Vt.Vec3fArray.FromNumpy(np.ascontiguousarray(normals, dtype=np.float32)))
        mesh.SetNormalsInterpolation(UsdGeom.Tokens.vertex)
    if texture_coordinates is not None:
        primvar = UsdGeom.PrimvarsAPI(mesh).CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.vertex)
        primvar.Set(Vt.Vec2fArray.FromNumpy(np.ascontiguousarray(texture_coordinates, dtype=np.float32)))
    mesh.CreateDisplayColorAttr([Gf.Vec3f(*map(float, display_color))])
    return mesh


def make_static_triangle_collider(prim, physics_material_path: str | None = None) -> None:
    """Exact triangle-mesh collider (approximation 'none') with an optional bound physics material."""
    UsdPhysics.CollisionAPI.Apply(prim)
    UsdPhysics.MeshCollisionAPI.Apply(prim).CreateApproximationAttr(UsdPhysics.Tokens.none)
    if physics_material_path:
        bind_physics_material(prim, physics_material_path)


def bind_physics_material(prim, material_path: str) -> None:
    """Bind a physics material (purpose 'physics') to a prim."""
    material = UsdShade.Material.Get(prim.GetStage(), material_path)
    UsdShade.MaterialBindingAPI.Apply(prim).Bind(material, UsdShade.Tokens.weakerThanDescendants, "physics")


def bind_render_material(prim, material_path: str) -> None:
    """Bind a render material (all purposes) to a prim."""
    material = UsdShade.Material.Get(prim.GetStage(), material_path)
    UsdShade.MaterialBindingAPI.Apply(prim).Bind(material)


def define_physics_material(stage, path: str, static_friction: float, dynamic_friction: float,
                            restitution: float = 0.0, combine_mode: str | None = None) -> None:
    """Author a physics material prim."""
    material = UsdShade.Material.Define(stage, path)
    api = UsdPhysics.MaterialAPI.Apply(material.GetPrim())
    api.CreateStaticFrictionAttr(float(static_friction))
    api.CreateDynamicFrictionAttr(float(dynamic_friction))
    api.CreateRestitutionAttr(float(restitution))
    if combine_mode:
        from pxr import PhysxSchema
        physx_material = PhysxSchema.PhysxMaterialAPI.Apply(material.GetPrim())
        physx_material.CreateFrictionCombineModeAttr(combine_mode)


def set_semantic_class(prim, class_name: str) -> None:
    """Label a prim with a semantic class (inherited by its descendants)."""
    UsdSemantics.LabelsAPI.Apply(prim, "class").CreateLabelsAttr([class_name])


def set_pose(prim, position_m, orientation_xyzw) -> None:
    """Author translate + orient ops (double precision) on an Xformable prim, replacing existing ops."""
    xformable = UsdGeom.Xformable(prim)
    xformable.ClearXformOpOrder()
    xformable.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Vec3d(*map(float, position_m)))
    x, y, z, w = map(float, orientation_xyzw)
    xformable.AddOrientOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Quatd(w, Gf.Vec3d(x, y, z)))
