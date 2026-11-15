"""``RoverDriveSession``: the one control loop of version 1 (docs/DESIGN.md sections 10.2 and 10.5).

Smoke tests, the headless drive demo and the keyboard window all drive through this object; they differ only in the
stepper. A later gymnasium environment wraps exactly this object (reset / advance_control_step / step listeners).
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from lunar_rover_sim.core.data_model.regular_grid import triangulated_height
from lunar_rover_sim.core.frames.pose_math import quaternion_from_yaw, yaw_from_quaternion
from lunar_rover_sim.core.rover.drive_models import SkidSteerDriveModel, TwistLimiter, YawRateController
from lunar_rover_sim.core.scene.scene_loader import LoadedScene
from lunar_rover_sim.core.scene.spawn_pose import SupportPose, wheel_support_pose
from lunar_rover_sim.runtime.rover.procedural_rover_builder import build_procedural_rover
from lunar_rover_sim.runtime.rover.rover_interface import RoverInterface, RoverState
from lunar_rover_sim.runtime.stage_assembly.scene_stage_builder import build_scene_stage
from lunar_rover_sim.runtime.stepping import (ApplicationUpdateStepper, ExplicitPhysicsStepper, PhysicsStepper,
                                              render_synchronously)


class StartCheckError(RuntimeError):
    """Raised when the rover does not come to rest on the terrain as expected after release."""


class FallThroughError(RuntimeError):
    """Raised (policy 'abort') when the rover ends up below the collider surface or outside the collided area."""


@dataclass
class DriveStepResult:
    """What one control step produced."""

    time_s: float
    commanded_twist: np.ndarray
    applied_twist: np.ndarray
    state: RoverState
    inside_drivable_zone: bool
    stopped_at_boundary: bool


@dataclass
class StartReport:
    """Numbers measured while starting (collider readiness, settle checks)."""

    collider_wait_updates: int = 0
    collider_wait_seconds: float = 0.0
    raycast_count: int = 0
    raycast_maximum_error_m: float = 0.0
    spawn_pose: dict = field(default_factory=dict)
    settled_height_error_m: float = 0.0
    wheel_normal_force_sum_n: float = 0.0
    rover_weight_n: float = 0.0
    wheel_normal_forces_n: list = field(default_factory=list)
    build_seconds: dict = field(default_factory=dict)


class RoverDriveSession:
    """Builds the stage and rover for one baked scene and runs the control loop."""

    def __init__(self, application, scene: LoadedScene, stepper_mode: str = "explicit", include_rocks: bool = True,
                 enable_tracks: bool | None = None, camera_names: tuple[str, ...] = (), log=print):
        """Configure; nothing is built until ``start``. stepper_mode: 'explicit' (headless) | 'application_update'."""
        self.application = application
        self.scene = scene
        self.configuration = scene.configuration
        self.stepper_mode = stepper_mode
        self.include_rocks = include_rocks
        self.enable_tracks = self.configuration.tracks.enabled if enable_tracks is None else enable_tracks
        self.camera_names = camera_names
        self.log = log
        self.stage = None
        self.terrain_handles = None
        self.rover_paths = None
        self.rover: RoverInterface | None = None
        self.stepper: PhysicsStepper | None = None
        self.track_runtime = None
        self.drive_model = SkidSteerDriveModel.from_rover(self.configuration.rover)
        self.limiter = TwistLimiter.from_rover(self.configuration.rover)
        self.yaw_controller = YawRateController.from_rover(self.configuration.rover)
        self.spawn_pose: SupportPose | None = None
        self.last_safe_pose = None
        self.pose_log: list[tuple[float, float, float, float]] = []
        self.step_listeners: list[Callable[[DriveStepResult], None]] = []
        self.start_report = StartReport()
        self._outside_message_printed = False
        self.rover_weight_n = (self.configuration.rover.chassis_mass_kg + self.configuration.rover.front_axle_mass_kg
                               + 4 * self.configuration.rover.wheel_mass_kg) * self.configuration.physics.gravity_m_s2

    # ------------------------------------------------------------------ geometry helpers
    def collider_height_m(self, east_m, north_m) -> np.ndarray:
        """Height of the collider surface (workspace collision lattice, else near ring) at points; NaN outside."""
        layout = self.scene.layout
        heights, valid = triangulated_height(self.scene.workspace_collision_height_m, layout.workspace_collision_grid,
                                             east_m, north_m)
        near, near_valid = triangulated_height(self.scene.terrain_arrays["near_ring_height_m"], layout.near_ring_grid,
                                               east_m, north_m)
        return np.where(valid, heights, np.where(near_valid, near, np.nan))

    def support_pose(self, east_m: float, north_m: float, heading_rad: float) -> SupportPose:
        """Rest pose on the collision lattice at a point and heading."""
        return wheel_support_pose(self.scene.workspace_collision_height_m, self.scene.layout.workspace_collision_grid,
                                  east_m, north_m, heading_rad, self.configuration.rover)

    def inside_drivable_zone(self, east_m: float, north_m: float) -> bool:
        """True when the rover centre is inside the drivable square."""
        half = self.scene.layout.drivable_half_size_m
        return abs(east_m) <= half and abs(north_m) <= half

    # ------------------------------------------------------------------ start
    def start(self, spawn_east_m: float | None = None, spawn_north_m: float | None = None,
              heading_rad: float | None = None) -> StartReport:
        """Build everything, wait for colliders while holding the rover, release, settle and check."""
        timer = time.perf_counter()
        self.stage, self.terrain_handles = build_scene_stage(self.scene, include_rocks=self.include_rocks)
        self.start_report.build_seconds["stage"] = round(time.perf_counter() - timer, 2)
        manifest_spawn = self.scene.manifest["spawn_pose"]
        east = manifest_spawn["position_m"][0] if spawn_east_m is None else spawn_east_m
        north = manifest_spawn["position_m"][1] if spawn_north_m is None else spawn_north_m
        heading = manifest_spawn["heading_rad"] if heading_rad is None else heading_rad
        self.spawn_pose = self.support_pose(east, north, heading)
        render = self.configuration.render
        position = np.array(self.spawn_pose.position_m) + [0.0, 0.0, 0.02]
        timer = time.perf_counter()
        self.rover_paths = build_procedural_rover(
            self.stage, self.configuration.rover, position, self.spawn_pose.orientation_xyzw,
            render.camera_resolution_px, render.camera_horizontal_aperture_mm, render.camera_near_clip_m,
            render.camera_far_clip_m)
        self.start_report.build_seconds["rover"] = round(time.perf_counter() - timer, 2)
        physics_rate = self.configuration.physics.physics_rate_hz
        if self.stepper_mode == "explicit":
            self.stepper = ExplicitPhysicsStepper(self.application, physics_rate, self.configuration.drive.control_rate_hz)
        else:
            self.stepper = ApplicationUpdateStepper(self.application, physics_rate)
        timer = time.perf_counter()
        self.stepper.start()
        self.rover = RoverInterface(self.stage, self.rover_paths, self.stepper.physics_dt_s)
        self.rover.initialize()
        self._wait_for_colliders()
        self.start_report.build_seconds["collider_wait"] = round(time.perf_counter() - timer, 2)
        if self.enable_tracks:
            from lunar_rover_sim.runtime.tracks.track_runtime import TrackRuntime
            self.track_runtime = TrackRuntime(self)
            self.stepper.add_listener(self.track_runtime.on_physics_step)
        self.start_report.spawn_pose = {"position_m": list(self.spawn_pose.position_m),
                                        "orientation_xyzw": list(self.spawn_pose.orientation_xyzw),
                                        "heading_rad": self.spawn_pose.heading_rad}
        self.reset(clear_tracks=False)
        return self.start_report

    def _wait_for_colliders(self) -> None:
        """Update until raycasts at every collision tile centre and near-ring probe points hit the expected surface.

        The rover is teleported back to the spawn pose before every update, so it cannot fall through terrain that is
        still cooking."""
        import carb
        import omni.physx
        query = omni.physx.get_physx_scene_query_interface()
        layout = self.scene.layout
        grid = layout.workspace_collision_grid
        tile_cells = layout.collision_tile_cells
        points = []
        for tile in grid.split_into_tiles(tile_cells):
            row = (tile.row_start + tile.row_stop - 1) / 2
            column = (tile.column_start + tile.column_stop - 1) / 2
            points.append((grid.origin_east_m + (column + 0.37) * grid.spacing_m,
                           grid.origin_north_m + (row + 0.61) * grid.spacing_m))
        near_half = layout.near_ring_grid.extent_east_m / 2
        workspace_half = grid.extent_east_m / 2
        ring_radius = (near_half + workspace_half) / 2
        for angle in np.linspace(0, 2 * math.pi, 8, endpoint=False):
            points.append((ring_radius * math.cos(angle), ring_radius * math.sin(angle)))
        points = np.array(points)
        expected = self.collider_height_m(points[:, 0], points[:, 1])
        timeout = self.configuration.drive.collider_wait_timeout_s
        started = time.perf_counter()
        updates = 0
        while True:
            errors = []
            for (east, north), expected_height in zip(points, expected):
                hit = query.raycast_closest(carb.Float3(float(east), float(north), float(expected_height) + 20.0),
                                            carb.Float3(0.0, 0.0, -1.0), 100.0)
                if not hit["hit"] or str(hit.get("collision", "")).startswith("/World/Rover"):
                    errors.append(np.inf if not hit["hit"] else 0.0)
                else:
                    errors.append(abs(hit["position"][2] - expected_height))
            maximum_error = float(np.max(errors))
            if maximum_error <= 1e-3:
                break
            if time.perf_counter() - started > timeout:
                raise StartCheckError(f"colliders not ready after {timeout} s (max raycast error {maximum_error})")
            self.rover.teleport(np.array(self.spawn_pose.position_m) + [0, 0, 0.02], self.spawn_pose.orientation_xyzw)
            self.application.update()
            updates += 1
        self.start_report.collider_wait_updates = updates
        self.start_report.collider_wait_seconds = round(time.perf_counter() - started, 2)
        self.start_report.raycast_count = len(points)
        self.start_report.raycast_maximum_error_m = maximum_error
        self.log(f"[session] colliders ready after {updates} updates; {len(points)} raycasts, max error "
                 f"{maximum_error * 1000:.3f} mm", flush=True)

    def reset(self, pose: SupportPose | None = None, clear_tracks: bool = True) -> None:
        """Teleport to a rest pose (default: spawn), settle, and check that the rover rests on the terrain."""
        pose = pose or self.spawn_pose
        if self.track_runtime is not None:
            self.track_runtime.sampling_enabled = False
            if clear_tracks:
                self.track_runtime.clear()
        self.limiter.reset()
        self.yaw_controller.reset()
        self.rover.teleport(np.array(pose.position_m) + [0.0, 0.0, 0.01], pose.orientation_xyzw)
        settle_steps = self.configuration.drive.settle_physics_steps
        self.stepper.advance_physics_steps(settle_steps)
        state = self.rover.read_state()
        height_error = float(state.position_m[2] - pose.position_m[2])
        forces = state.wheel_contact_force_n
        normal_forces = np.linalg.norm(forces, axis=1)
        self.start_report.settled_height_error_m = height_error
        self.start_report.wheel_normal_forces_n = [round(float(v), 2) for v in normal_forces]
        self.start_report.wheel_normal_force_sum_n = float(normal_forces.sum())
        self.start_report.rover_weight_n = self.rover_weight_n
        self.log(f"[session] settled: chassis height error {height_error * 1000:.1f} mm, wheel forces "
                 f"{np.round(normal_forces, 1).tolist()} N (sum {normal_forces.sum():.1f}, weight "
                 f"{self.rover_weight_n:.1f})", flush=True)
        if abs(height_error) > 0.05:
            raise StartCheckError(f"chassis rests {height_error * 1000:.0f} mm from the support plane (limit 50 mm)")
        if abs(normal_forces.sum() - self.rover_weight_n) > 0.1 * self.rover_weight_n:
            raise StartCheckError(f"wheel force sum {normal_forces.sum():.1f} N differs from the weight "
                                  f"{self.rover_weight_n:.1f} N by more than 10 %")
        self.last_safe_pose = (state.position_m.copy(), state.orientation_xyzw.copy())
        self.pose_log.clear()
        if self.track_runtime is not None:
            self.track_runtime.reset_sampling()
            self.track_runtime.sampling_enabled = True

    # ------------------------------------------------------------------ control loop
    def ensure_simulation_running(self) -> str | None:
        """Window mode guard: the GUI (Stop/Play buttons, Kit's Space play/pause hotkey) can stop or pause the timeline.

        Paused -> resume. Stopped -> play again, rebuild the tensor views (a stop invalidates them) and reset the rover
        to its last safe pose (tracks are kept). Returns a description of what was done, or None."""
        import omni.timeline
        timeline = omni.timeline.get_timeline_interface()
        if timeline.is_playing():
            return None
        if not timeline.is_stopped():
            timeline.play()
            return "timeline was paused: resumed"
        timeline.play()
        self.application.update()
        self.rover.initialize()
        position, orientation = self.last_safe_pose
        self.reset(self.support_pose(float(position[0]), float(position[1]), yaw_from_quaternion(orientation)),
                   clear_tracks=False)
        return "timeline was stopped: restarted physics and put the rover back at its last safe pose"

    def add_step_listener(self, listener: Callable[[DriveStepResult], None]) -> None:
        """Call ``listener(result)`` after every control step."""
        self.step_listeners.append(listener)

    def advance_control_step(self, forward_speed_m_s: float, yaw_rate_rad_s: float,
                             render_cameras: bool = False) -> DriveStepResult:
        """Apply a twist for one control period and return the resulting state."""
        commanded = np.array([forward_speed_m_s, yaw_rate_rad_s], dtype=np.float64)
        state = self.rover.read_state()
        east, north = float(state.position_m[0]), float(state.position_m[1])
        inside = self.inside_drivable_zone(east, north)
        stopped = False
        if not inside and self.configuration.drive.boundary_policy == "stop_outward":
            heading = yaw_from_quaternion(state.orientation_xyzw)
            outward = east * math.cos(heading) + north * math.sin(heading)
            if outward * forward_speed_m_s > 0:
                commanded = np.array([0.0, yaw_rate_rad_s])
                stopped = True
                if not self._outside_message_printed:
                    self.log("[session] rover at the edge of the drivable zone: forward motion outward stopped "
                             "(turn back towards the centre)", flush=True)
                    self._outside_message_printed = True
        else:
            self._outside_message_printed = False
        applied = self.limiter.limit(commanded[0], commanded[1], self.stepper.control_period_s)
        yaw_command = self.yaw_controller.command(float(applied[1]), float(state.angular_velocity_rad_s[2]),
                                                  self.stepper.control_period_s)
        self.rover.set_wheel_velocity_targets(self.drive_model.wheel_velocity_targets(applied[0], yaw_command))
        self.stepper.advance_control_period()
        if self.track_runtime is not None:
            self.track_runtime.flush()
        state = self.rover.read_state()
        self._fall_through_guard(state)
        if render_cameras:
            if self.track_runtime is not None:
                self.track_runtime.force_mask_upload()
            render_synchronously(self.configuration.render.subframes_per_camera_frame)
        self.pose_log.append((self.stepper.time_s, float(state.position_m[0]), float(state.position_m[1]),
                              yaw_from_quaternion(state.orientation_xyzw)))
        result = DriveStepResult(self.stepper.time_s, np.array([forward_speed_m_s, yaw_rate_rad_s]), applied, state,
                                 inside, stopped)
        for listener in self.step_listeners:
            listener(result)
        return result

    def _fall_through_guard(self, state: RoverState) -> None:
        """Abort or recover when the chassis is far below the collider surface or off the collided area."""
        surface = float(self.collider_height_m(np.array([state.position_m[0]]), np.array([state.position_m[1]]))[0])
        fallen = math.isnan(surface) or state.position_m[2] < surface - self.configuration.drive.fall_through_depth_m
        if not fallen:
            if self.inside_drivable_zone(state.position_m[0], state.position_m[1]):
                self.last_safe_pose = (state.position_m.copy(), state.orientation_xyzw.copy())
            return
        message = (f"rover fell through or left the collided area at {np.round(state.position_m, 3).tolist()} "
                   f"(surface {surface:.3f} m)")
        if self.configuration.drive.fall_through_policy == "abort":
            raise FallThroughError(message)
        self.log(f"[session] {message}; recovering to the last safe pose", flush=True)
        position, orientation = self.last_safe_pose
        heading = yaw_from_quaternion(orientation)
        self.reset(self.support_pose(float(position[0]), float(position[1]), heading), clear_tracks=False)

    def heading_quaternion(self, heading_rad: float) -> np.ndarray:
        """Quaternion of a level pose with this heading (helper for callers)."""
        return quaternion_from_yaw(heading_rad)
