"""Tensor-API access to the rover: wheel velocity targets, state, wheel kinematics, contact forces, teleport.

Views are created after ``timeline.play()`` plus one ``app.update()`` and stay valid as long as the timeline is not
stopped (resets teleport instead of stopping). Joint and link indices are resolved by name, never by assumed order.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from lunar_rover_sim.core.frames.pose_math import rotation_matrix_from_quaternion
from lunar_rover_sim.core.rover.drive_models import WHEEL_NAMES
from lunar_rover_sim.runtime.rover.procedural_rover_builder import RoverPrimPaths


@dataclass
class RoverState:
    """Rover state at one instant (world frame; quaternions xyzw)."""

    position_m: np.ndarray
    orientation_xyzw: np.ndarray
    linear_velocity_m_s: np.ndarray
    angular_velocity_rad_s: np.ndarray
    wheel_angle_rad: np.ndarray
    wheel_angular_velocity_rad_s: np.ndarray
    wheel_center_m: np.ndarray
    wheel_axle_direction: np.ndarray
    wheel_center_velocity_m_s: np.ndarray
    wheel_contact_force_n: np.ndarray
    pivot_angle_rad: float


class RoverInterface:
    """Wraps the articulation view and per-wheel contact views of one rover."""

    def __init__(self, stage, paths: RoverPrimPaths, physics_dt_s: float):
        """Store paths; call ``initialize`` after the timeline plays."""
        self.stage = stage
        self.paths = paths
        self.physics_dt_s = physics_dt_s
        self.simulation_view = None
        self.articulation = None
        self.contact_views = []
        self.wheel_dof_indices = None
        self.pivot_dof_index = None
        self.wheel_link_indices = None
        self.dof_count = 0

    def initialize(self) -> None:
        """Create the tensor views (requires a playing timeline and one app.update since play)."""
        import omni.physics.tensors as tensors
        from pxr import UsdUtils
        stage_id = UsdUtils.StageCache.Get().GetId(self.stage).ToLongInt()
        self.simulation_view = tensors.create_simulation_view("numpy", stage_id=stage_id)
        self.articulation = self.simulation_view.create_articulation_view(self.paths.root)
        if self.articulation is None or self.articulation.count != 1:
            raise RuntimeError(f"articulation view for {self.paths.root} not created (count "
                               f"{getattr(self.articulation, 'count', None)})")
        metatype = self.articulation.shared_metatype
        dof_names = list(metatype.dof_names)
        link_names = list(metatype.link_names)
        self.dof_count = len(dof_names)
        self.wheel_dof_indices = np.array([dof_names.index(f"{name}_wheel") for name in WHEEL_NAMES], dtype=np.int64)
        self.pivot_dof_index = dof_names.index("front_axle_pivot") if "front_axle_pivot" in dof_names else None
        self.wheel_link_indices = np.array([link_names.index(f"wheel_{name}") for name in WHEEL_NAMES], dtype=np.int64)
        self.contact_views = [self.simulation_view.create_rigid_contact_view(path) for path in self.paths.wheels]
        if any(view is None for view in self.contact_views):
            raise RuntimeError("wheel contact views not created (PhysxContactReportAPI missing?)")

    def set_wheel_velocity_targets(self, wheel_angular_velocity_rad_s: np.ndarray) -> None:
        """Set the four wheel drive velocity targets (rad/s, positive = rolling forward)."""
        targets = np.zeros((1, self.dof_count), dtype=np.float32)
        targets[0, self.wheel_dof_indices] = np.asarray(wheel_angular_velocity_rad_s, dtype=np.float32)
        self.articulation.set_dof_velocity_targets(targets, np.array([0], dtype=np.int32))

    def read_state(self) -> RoverState:
        """Read pose, velocities, joint states, wheel kinematics and net wheel contact forces."""
        root = np.asarray(self.articulation.get_root_transforms())[0].astype(np.float64)
        root_velocity = np.asarray(self.articulation.get_root_velocities())[0].astype(np.float64)
        dof_positions = np.asarray(self.articulation.get_dof_positions())[0].astype(np.float64)
        dof_velocities = np.asarray(self.articulation.get_dof_velocities())[0].astype(np.float64)
        links = np.asarray(self.articulation.get_link_transforms())[0].astype(np.float64)
        link_velocities = np.asarray(self.articulation.get_link_velocities())[0].astype(np.float64)
        wheel_links = links[self.wheel_link_indices]
        axles = np.array([rotation_matrix_from_quaternion(link[3:7])[:, 1] for link in wheel_links])
        forces = np.vstack([np.asarray(view.get_net_contact_forces(self.physics_dt_s)).reshape(-1, 3)[0]
                            for view in self.contact_views]).astype(np.float64)
        return RoverState(
            position_m=root[:3], orientation_xyzw=root[3:7],
            linear_velocity_m_s=root_velocity[:3], angular_velocity_rad_s=root_velocity[3:6],
            wheel_angle_rad=dof_positions[self.wheel_dof_indices],
            wheel_angular_velocity_rad_s=dof_velocities[self.wheel_dof_indices],
            wheel_center_m=wheel_links[:, :3], wheel_axle_direction=axles,
            wheel_center_velocity_m_s=link_velocities[self.wheel_link_indices, :3],
            wheel_contact_force_n=forces,
            pivot_angle_rad=float(dof_positions[self.pivot_dof_index]) if self.pivot_dof_index is not None else 0.0)

    def read_link_poses(self) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        """World pose of every articulation link: {link name: (position_m (3,), orientation_xyzw (4,))}."""
        links = np.asarray(self.articulation.get_link_transforms())[0].astype(np.float64)
        names = list(self.articulation.shared_metatype.link_names)
        return {name: (links[index, :3].copy(), links[index, 3:7].copy()) for index, name in enumerate(names)}

    def teleport(self, position_m, orientation_xyzw) -> None:
        """Move the rover to a pose with zero velocities (does not stop the timeline)."""
        pose = np.array([[*position_m, *orientation_xyzw]], dtype=np.float32)
        indices = np.array([0], dtype=np.int32)
        self.articulation.set_root_transforms(pose, indices)
        self.articulation.set_root_velocities(np.zeros((1, 6), dtype=np.float32), indices)
        self.articulation.set_dof_velocities(np.zeros((1, self.dof_count), dtype=np.float32), indices)
        self.set_wheel_velocity_targets(np.zeros(4))
