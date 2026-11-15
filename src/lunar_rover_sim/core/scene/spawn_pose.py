"""Deterministic safe start pose (docs/DESIGN.md section 9.7) and the four-wheel support pose on a height field."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from lunar_rover_sim.core.configuration.configuration_models import RoverConfiguration, SpawnConfiguration
from lunar_rover_sim.core.data_model.regular_grid import RegularGrid, triangulated_height
from lunar_rover_sim.core.frames.pose_math import quaternion_from_rotation_matrix


@dataclass(frozen=True)
class SupportPose:
    """Chassis pose resting on four wheel contact points: position (m) and orientation quaternion (xyzw)."""

    position_m: tuple[float, float, float]
    orientation_xyzw: tuple[float, float, float, float]
    heading_rad: float
    wheel_ground_heights_m: tuple[float, float, float, float]


def wheel_offsets_body_m(rover: RoverConfiguration) -> np.ndarray:
    """(4, 2) wheel centre offsets in the body frame, order front_left, front_right, rear_left, rear_right."""
    half_base, half_track = rover.wheelbase_m / 2, rover.track_width_m / 2
    return np.array([[half_base, half_track], [half_base, -half_track], [-half_base, half_track],
                     [-half_base, -half_track]])


def wheel_support_pose(heights: np.ndarray, grid: RegularGrid, east_m: float, north_m: float, heading_rad: float,
                       rover: RoverConfiguration) -> SupportPose:
    """Pose with the chassis origin one wheel radius above the least-squares plane through the ground under the wheels.

    Roll and pitch follow that plane, yaw is ``heading_rad`` (counter-clockwise from east)."""
    offsets = wheel_offsets_body_m(rover)
    cos_heading, sin_heading = np.cos(heading_rad), np.sin(heading_rad)
    wheel_east = east_m + offsets[:, 0] * cos_heading - offsets[:, 1] * sin_heading
    wheel_north = north_m + offsets[:, 0] * sin_heading + offsets[:, 1] * cos_heading
    ground, valid = triangulated_height(heights, grid, wheel_east, wheel_north)
    if not valid.all():
        raise ValueError(f"wheel footprint at ({east_m:.2f}, {north_m:.2f}) leaves the height field")
    design = np.column_stack([np.ones(4), wheel_east - east_m, wheel_north - north_m])
    (plane_height, slope_east, slope_north), *_ = np.linalg.lstsq(design, ground, rcond=None)
    up = np.array([-slope_east, -slope_north, 1.0])
    up /= np.linalg.norm(up)
    forward_flat = np.array([cos_heading, sin_heading, 0.0])
    forward = forward_flat - np.dot(forward_flat, up) * up
    forward /= np.linalg.norm(forward)
    left = np.cross(up, forward)
    rotation = np.column_stack([forward, left, up])
    center = np.array([east_m, north_m, plane_height]) + rover.wheel_radius_m * up
    return SupportPose(tuple(float(v) for v in center), tuple(float(v) for v in quaternion_from_rotation_matrix(rotation)),
                       float(heading_rad), tuple(float(v) for v in ground))


def find_spawn_pose(heights: np.ndarray, grid: RegularGrid, slope_deg: np.ndarray, roughness_m: np.ndarray,
                    rock_id: np.ndarray | None, crater_east_m: np.ndarray, crater_north_m: np.ndarray,
                    crater_diameter_m: np.ndarray, drivable_half_size_m: float, spawn: SpawnConfiguration,
                    rover: RoverConfiguration) -> tuple[SupportPose, dict]:
    """Lowest-cost safe start pose near the preferred point; raises ValueError with rejection counts if none is safe.

    ``slope_deg``, ``roughness_m`` and ``rock_id`` are rasters on ``grid`` (the collision lattice)."""
    step = 0.25
    offsets = np.arange(-spawn.search_radius_m, spawn.search_radius_m + 1e-9, step)
    candidate_east, candidate_north = np.meshgrid(spawn.preferred_east_m + offsets, spawn.preferred_north_m + offsets)
    candidate_east, candidate_north = candidate_east.ravel(), candidate_north.ravel()
    distance = np.hypot(candidate_east - spawn.preferred_east_m, candidate_north - spawn.preferred_north_m)
    keep = (distance <= spawn.search_radius_m + 1e-9) & (np.abs(candidate_east) <= drivable_half_size_m) & \
           (np.abs(candidate_north) <= drivable_half_size_m)
    counts = {"outside_drivable_zone": int((~keep).sum()), "rock": 0, "slope": 0, "roughness": 0, "crater": 0}
    east_mesh, north_mesh = grid.coordinate_mesh()
    large = crater_diameter_m >= spawn.crater_minimum_diameter_m
    best = None
    for index in np.flatnonzero(keep):
        east, north = candidate_east[index], candidate_north[index]
        near_crater = np.hypot(crater_east_m[large] - east, crater_north_m[large] - north) < \
            crater_diameter_m[large] / 2 + spawn.crater_clearance_m
        if near_crater.any():
            counts["crater"] += 1
            continue
        footprint = (east_mesh - east) ** 2 + (north_mesh - north) ** 2 <= spawn.footprint_radius_m ** 2
        if rock_id is not None and np.any(rock_id[footprint] > 0):
            counts["rock"] += 1
            continue
        maximum_slope = float(slope_deg[footprint].max())
        if maximum_slope > spawn.maximum_slope_deg:
            counts["slope"] += 1
            continue
        maximum_roughness = float(roughness_m[footprint].max())
        if maximum_roughness > spawn.maximum_roughness_m:
            counts["roughness"] += 1
            continue
        cost = maximum_slope / spawn.maximum_slope_deg + maximum_roughness / spawn.maximum_roughness_m + \
            0.1 * distance[index] / spawn.search_radius_m
        if best is None or cost < best[0]:
            best = (cost, east, north)
    if best is None:
        raise ValueError(f"no safe spawn pose within {spawn.search_radius_m} m of the preferred point: {counts}")
    pose = wheel_support_pose(heights, grid, best[1], best[2], np.radians(spawn.heading_deg), rover)
    return pose, {"cost": float(best[0]), "rejections": counts}
