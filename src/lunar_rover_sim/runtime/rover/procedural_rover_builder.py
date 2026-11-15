"""Procedural four-wheel skid-steer rover articulation (docs/DESIGN.md section 10.3).

Body frame: x forward, y left, z up; origin = centre of the wheel rectangle at axle height.
  /World/Rover                       Xform, ArticulationRootAPI, pose = spawn pose
    chassis                          rigid body (unscaled Xform); box collider child; mast and camera head visuals
    front_axle                       rigid body without collider, revolute pivot about x (passive, limited, damped)
    wheel_front_left ... rear_right  rigid bodies; capsule collider (axis y); visual tyre with grousers (no collider)
    joints/...                       revolute joints about y with angular velocity drives
Rigid bodies are unscaled Xforms; scaled geometry are children (a scale on a body would distort joint frames).
Mass, centre of mass and inertia are always authored.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from pxr import Gf, PhysxSchema, UsdGeom, UsdPhysics

from lunar_rover_sim.core.configuration.configuration_models import CameraMountConfiguration, RoverConfiguration
from lunar_rover_sim.core.data_model.regular_grid import lattice_triangle_indices
from lunar_rover_sim.core.frames.pose_math import focal_length_mm, quaternion_from_rotation_matrix
from lunar_rover_sim.core.rover.drive_models import WHEEL_NAMES
from lunar_rover_sim.core.scene.spawn_pose import wheel_offsets_body_m
from lunar_rover_sim.runtime.stage_assembly.scene_stage_builder import WHEEL_PHYSICS_MATERIAL
from lunar_rover_sim.runtime.stage_assembly.usd_helpers import (bind_physics_material, define_triangle_mesh, set_pose,
                                                                set_semantic_class)

ROVER_PATH = "/World/Rover"


@dataclass(frozen=True)
class RoverPrimPaths:
    """Prim paths of the built rover."""

    root: str
    chassis: str
    wheels: tuple[str, str, str, str]
    wheel_joints: tuple[str, str, str, str]
    pivot_joint: str | None
    cameras: dict[str, str]


def _apply_mass(prim, mass_kg: float, center_of_mass_m, diagonal_inertia_kg_m2) -> None:
    """Author mass, centre of mass and principal inertia on a rigid body."""
    mass_api = UsdPhysics.MassAPI.Apply(prim)
    mass_api.CreateMassAttr(float(mass_kg))
    mass_api.CreateCenterOfMassAttr(Gf.Vec3f(*map(float, center_of_mass_m)))
    mass_api.CreateDiagonalInertiaAttr(Gf.Vec3f(*map(float, diagonal_inertia_kg_m2)))
    mass_api.CreatePrincipalAxesAttr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))


def _rigid_body(stage, path: str, local_position_m) -> UsdGeom.Xform:
    """Unscaled Xform rigid body at a body-frame position."""
    body = UsdGeom.Xform.Define(stage, path)
    UsdGeom.Xformable(body).AddTranslateOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Vec3d(*map(float, local_position_m)))
    UsdPhysics.RigidBodyAPI.Apply(body.GetPrim())
    physx_body = PhysxSchema.PhysxRigidBodyAPI.Apply(body.GetPrim())
    physx_body.CreateSleepThresholdAttr(0.0)
    return body


def _box(stage, path: str, size_m, center_m, color, collider: bool) -> UsdGeom.Cube:
    """Unit cube scaled to a box (child geometry), optionally a collider."""
    cube = UsdGeom.Cube.Define(stage, path)
    cube.CreateSizeAttr(1.0)
    xformable = UsdGeom.Xformable(cube)
    xformable.AddTranslateOp().Set(Gf.Vec3d(*map(float, center_m)))
    xformable.AddScaleOp().Set(Gf.Vec3f(*map(float, size_m)))
    cube.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    if collider:
        UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
    return cube


def wheel_visual_mesh(radius_m: float, width_m: float, grouser_count: int, grouser_height_m: float,
                      segments: int = 96) -> tuple[np.ndarray, np.ndarray]:
    """Closed tyre mesh about the y axis with ``grouser_count`` raised bars (cos^2 profile, as the track imprint)."""
    angle = np.linspace(0.0, 2 * math.pi, segments, endpoint=False)
    profile = np.maximum(0.0, np.cos(grouser_count * angle / 2)) ** 8
    ring_radius = radius_m + grouser_height_m * profile
    ring = np.column_stack([ring_radius * np.cos(angle), np.zeros(segments), ring_radius * np.sin(angle)])
    left = ring + [0.0, width_m / 2, 0.0]
    right = ring + [0.0, -width_m / 2, 0.0]
    points = np.vstack([left, right, [[0.0, width_m / 2, 0.0]], [[0.0, -width_m / 2, 0.0]]]).astype(np.float32)
    triangles = []
    for index in range(segments):
        following = (index + 1) % segments
        triangles += [[index, segments + index, segments + following], [index, segments + following, following],
                      [2 * segments, index, following], [2 * segments + 1, segments + following, segments + index]]
    return points, np.asarray(triangles, dtype=np.int32)


def _camera(stage, path: str, mount: CameraMountConfiguration, resolution_px, horizontal_aperture_mm: float,
            near_clip_m: float, far_clip_m: float) -> None:
    """Body-mounted USD camera (looks along -Z, +Y up) with the mount's pitch and yaw."""
    camera = UsdGeom.Camera.Define(stage, path)
    width, height = resolution_px
    camera.CreateHorizontalApertureAttr(float(horizontal_aperture_mm))
    camera.CreateVerticalApertureAttr(float(horizontal_aperture_mm * height / width))
    camera.CreateFocalLengthAttr(float(focal_length_mm(mount.horizontal_field_of_view_deg, horizontal_aperture_mm)))
    camera.CreateClippingRangeAttr(Gf.Vec2f(float(near_clip_m), float(far_clip_m)))
    pitch, yaw = math.radians(mount.pitch_down_deg), math.radians(mount.yaw_deg)
    forward = np.array([math.cos(pitch) * math.cos(yaw), math.cos(pitch) * math.sin(yaw), -math.sin(pitch)])
    right = np.array([math.sin(yaw), -math.cos(yaw), 0.0])
    up = np.cross(right, forward)
    rotation = np.column_stack([right, up, -forward])
    set_pose(camera.GetPrim(), mount.position_body_m, quaternion_from_rotation_matrix(rotation))


def build_procedural_rover(stage, rover: RoverConfiguration, position_m, orientation_xyzw, render_resolution_px,
                           horizontal_aperture_mm: float, near_clip_m: float, far_clip_m: float,
                           camera_names: tuple[str, ...] = ("front", "rear", "follow")) -> RoverPrimPaths:
    """Author the rover articulation at a world pose; returns its prim paths."""
    root = UsdGeom.Xform.Define(stage, ROVER_PATH)
    set_pose(root.GetPrim(), position_m, orientation_xyzw)
    UsdPhysics.ArticulationRootAPI.Apply(root.GetPrim())
    articulation = PhysxSchema.PhysxArticulationAPI.Apply(root.GetPrim())
    articulation.CreateSolverPositionIterationCountAttr(16)
    articulation.CreateSolverVelocityIterationCountAttr(4)
    articulation.CreateEnabledSelfCollisionsAttr(False)
    articulation.CreateSleepThresholdAttr(0.0)
    set_semantic_class(root.GetPrim(), "rover")

    length, width, height = rover.chassis_size_m
    chassis_bottom = rover.ground_clearance_m - rover.wheel_radius_m
    chassis_center = (0.0, 0.0, chassis_bottom + height / 2)
    chassis = _rigid_body(stage, f"{ROVER_PATH}/chassis", (0.0, 0.0, 0.0))
    _box(stage, f"{ROVER_PATH}/chassis/body", rover.chassis_size_m, chassis_center, (0.75, 0.72, 0.62), collider=True)
    _box(stage, f"{ROVER_PATH}/chassis/mast", (0.04, 0.04, 0.42), (0.28, 0.0, chassis_bottom + height + 0.21),
         (0.6, 0.6, 0.6), collider=False)
    _box(stage, f"{ROVER_PATH}/chassis/camera_head", (0.08, 0.22, 0.07), (0.30, 0.0, 0.75), (0.15, 0.15, 0.15),
         collider=False)
    _box(stage, f"{ROVER_PATH}/chassis/solar_panel", (0.62, 0.30, 0.012), (-0.08, 0.0, chassis_bottom + height + 0.01),
         (0.08, 0.10, 0.25), collider=False)
    mass = rover.chassis_mass_kg
    _apply_mass(chassis.GetPrim(), mass, chassis_center,
                (mass / 12 * (width ** 2 + height ** 2), mass / 12 * (length ** 2 + height ** 2),
                 mass / 12 * (length ** 2 + width ** 2)))

    offsets = wheel_offsets_body_m(rover)
    half_base = rover.wheelbase_m / 2
    pivot_joint_path = None
    front_parent = f"{ROVER_PATH}/chassis"
    front_parent_offset = np.zeros(2)
    if rover.front_axle_pivot_enabled:
        axle = _rigid_body(stage, f"{ROVER_PATH}/front_axle", (half_base, 0.0, 0.0))
        _box(stage, f"{ROVER_PATH}/front_axle/beam", (0.05, rover.track_width_m - 0.25, 0.04), (0.0, 0.0, 0.0),
             (0.35, 0.35, 0.35), collider=False)
        axle_mass = rover.front_axle_mass_kg
        _apply_mass(axle.GetPrim(), axle_mass, (0.0, 0.0, 0.0),
                    (axle_mass * rover.track_width_m ** 2 / 12, 0.002, axle_mass * rover.track_width_m ** 2 / 12))
        pivot = UsdPhysics.RevoluteJoint.Define(stage, f"{ROVER_PATH}/joints/front_axle_pivot")
        pivot.CreateBody0Rel().SetTargets([f"{ROVER_PATH}/chassis"])
        pivot.CreateBody1Rel().SetTargets([f"{ROVER_PATH}/front_axle"])
        pivot.CreateAxisAttr("X")
        pivot.CreateLocalPos0Attr(Gf.Vec3f(float(half_base), 0.0, 0.0))
        pivot.CreateLocalRot0Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
        pivot.CreateLocalPos1Attr(Gf.Vec3f(0.0, 0.0, 0.0))
        pivot.CreateLocalRot1Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
        pivot.CreateLowerLimitAttr(-float(rover.front_axle_pivot_limit_deg))
        pivot.CreateUpperLimitAttr(float(rover.front_axle_pivot_limit_deg))
        pivot_drive = UsdPhysics.DriveAPI.Apply(pivot.GetPrim(), "angular")
        pivot_drive.CreateTypeAttr("force")
        pivot_drive.CreateStiffnessAttr(0.0)
        pivot_drive.CreateDampingAttr(float(rover.front_axle_pivot_damping_n_m_s_per_rad * math.pi / 180.0))
        pivot_drive.CreateTargetVelocityAttr(0.0)
        pivot_joint_path = str(pivot.GetPath())
        front_parent = f"{ROVER_PATH}/front_axle"
        front_parent_offset = np.array([half_base, 0.0])

    visual_points, visual_triangles = wheel_visual_mesh(rover.wheel_radius_m + rover.nominal_sinkage_m,
                                                        rover.wheel_width_m, rover.grouser_count, rover.grouser_height_m)
    wheel_paths, joint_paths = [], []
    for index, name in enumerate(WHEEL_NAMES):
        wheel_path = f"{ROVER_PATH}/wheel_{name}"
        offset = offsets[index]
        wheel = _rigid_body(stage, wheel_path, (offset[0], offset[1], 0.0))
        capsule = UsdGeom.Capsule.Define(stage, f"{wheel_path}/collider")
        capsule.CreateRadiusAttr(float(rover.wheel_radius_m))
        capsule.CreateHeightAttr(float(rover.wheel_collider_capsule_height_m))
        capsule.CreateAxisAttr(UsdGeom.Tokens.y)
        UsdPhysics.CollisionAPI.Apply(capsule.GetPrim())
        bind_physics_material(capsule.GetPrim(), WHEEL_PHYSICS_MATERIAL)
        UsdGeom.Imageable(capsule.GetPrim()).MakeInvisible()
        tyre = define_triangle_mesh(stage, f"{wheel_path}/tyre", visual_points, visual_triangles,
                                    display_color=(0.22, 0.22, 0.22))
        tyre.CreateDoubleSidedAttr(True)
        hub_side = 1.0 if offset[1] > 0 else -1.0
        _box(stage, f"{wheel_path}/hub", (0.10, 0.012, 0.10), (0.0, hub_side * (rover.wheel_width_m / 2 + 0.006), 0.0),
             (0.7, 0.7, 0.7), collider=False)
        wheel_mass, radius, wheel_width = rover.wheel_mass_kg, rover.wheel_radius_m, rover.wheel_width_m
        _apply_mass(wheel.GetPrim(), wheel_mass, (0.0, 0.0, 0.0),
                    (wheel_mass * (3 * radius ** 2 + wheel_width ** 2) / 12, wheel_mass * radius ** 2 / 2,
                     wheel_mass * (3 * radius ** 2 + wheel_width ** 2) / 12))
        PhysxSchema.PhysxContactReportAPI.Apply(wheel.GetPrim()).CreateThresholdAttr(0.0)

        is_front = name.startswith("front")
        parent = front_parent if is_front else f"{ROVER_PATH}/chassis"
        parent_offset = front_parent_offset if is_front else np.zeros(2)
        joint = UsdPhysics.RevoluteJoint.Define(stage, f"{ROVER_PATH}/joints/{name}_wheel")
        joint.CreateBody0Rel().SetTargets([parent])
        joint.CreateBody1Rel().SetTargets([wheel_path])
        joint.CreateAxisAttr("Y")
        joint.CreateLocalPos0Attr(Gf.Vec3f(float(offset[0] - parent_offset[0]), float(offset[1] - parent_offset[1]), 0.0))
        joint.CreateLocalRot0Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
        joint.CreateLocalPos1Attr(Gf.Vec3f(0.0, 0.0, 0.0))
        joint.CreateLocalRot1Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
        drive = UsdPhysics.DriveAPI.Apply(joint.GetPrim(), "angular")
        drive.CreateTypeAttr("force")
        drive.CreateStiffnessAttr(0.0)
        drive.CreateDampingAttr(float(rover.wheel_drive_damping_n_m_s_per_deg))
        drive.CreateMaxForceAttr(float(rover.wheel_drive_maximum_torque_n_m))
        drive.CreateTargetVelocityAttr(0.0)
        PhysxSchema.JointStateAPI.Apply(joint.GetPrim(), "angular")
        wheel_paths.append(wheel_path)
        joint_paths.append(str(joint.GetPath()))

    cameras = {}
    UsdGeom.Scope.Define(stage, f"{ROVER_PATH}/chassis/cameras")
    for camera_name in camera_names:
        mount = getattr(rover.cameras, camera_name)
        path = f"{ROVER_PATH}/chassis/cameras/{camera_name}"
        _camera(stage, path, mount, render_resolution_px, horizontal_aperture_mm, near_clip_m, far_clip_m)
        cameras[camera_name] = path
    return RoverPrimPaths(ROVER_PATH, f"{ROVER_PATH}/chassis", tuple(wheel_paths), tuple(joint_paths), pivot_joint_path,
                          cameras)


__all__ = ["build_procedural_rover", "RoverPrimPaths", "ROVER_PATH", "wheel_visual_mesh", "lattice_triangle_indices"]
