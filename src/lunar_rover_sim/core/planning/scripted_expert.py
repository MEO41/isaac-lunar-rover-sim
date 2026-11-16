"""The scripted expert: a privileged driver that takes the rover to a goal around rocks, craters and steep ground.

Mission rules per control step (after MARE's ``mission.py``, without the lidar part: obstacles are known):
 1. no goal -> idle (zero twist);
 2. body tilt above ``maximum_tilt_deg`` -> aborted;
 3. within ``stop_tolerance_m`` of the goal -> reached (zero twist);
 4. no progress (neither ``stall_minimum_motion_m`` moved nor ``stall_minimum_turn_deg`` turned) for
    ``stall_window_s``, or farther than ``replan_deviation_m`` from the route -> plan again from the current pose;
    after ``maximum_stalls`` stalls in a row -> aborted;
 5. otherwise -> drive with the dynamic-window command along the route.
The expert sees the true scene (rock and crater tables, terrain), never the camera; it is the demonstrator, not a
perception baseline. Isaac-free: the caller supplies the pose and applies the twist.
"""
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field

import numpy as np

from lunar_rover_sim.core.configuration.configuration_models import ExpertConfiguration, RoverConfiguration
from lunar_rover_sim.core.planning.dynamic_window_planner import DynamicWindowPlanner, wrap_angle
from lunar_rover_sim.core.planning.global_planner import GlobalPlanner, Route
from lunar_rover_sim.core.planning.planner_map import PlannerMap, RoverFootprint

EXPERT_STATES = ("idle", "driving", "reached", "aborted")


@dataclass
class ExpertCommand:
    """What the expert decided at one control step."""

    forward_speed_m_s: float
    yaw_rate_rad_s: float
    state: str
    reason: str = ""
    distance_to_goal_m: float = float("nan")
    local_goal_m: np.ndarray | None = None
    replanned: bool = False
    blocked: bool = False


@dataclass
class ExpertStatistics:
    """Counters of one goal (for logs and dataset metadata)."""

    plans: int = 0
    stalls_in_a_row: int = 0
    total_stalls: int = 0
    blocked_steps: int = 0
    route_lengths_m: list = field(default_factory=list)


class ScriptedExpert:
    """Privileged global + local planner with the mission state machine."""

    def __init__(self, planner_map: PlannerMap, configuration: ExpertConfiguration, rover: RoverConfiguration,
                 control_period_s: float):
        """Check that the expert's limits fit the rover's, build both planners."""
        for expert_value, rover_value, name in (
                (configuration.maximum_forward_speed_m_s, rover.maximum_forward_speed_m_s, "forward speed"),
                (configuration.maximum_yaw_rate_rad_s, rover.maximum_yaw_rate_rad_s, "yaw rate"),
                (configuration.maximum_forward_acceleration_m_s2, rover.maximum_forward_acceleration_m_s2, "acceleration"),
                (configuration.maximum_yaw_acceleration_rad_s2, rover.maximum_yaw_acceleration_rad_s2, "yaw acceleration")):
            if expert_value > rover_value:
                raise ValueError(f"expert maximum {name} {expert_value} exceeds the rover's {rover_value}")
        self.map = planner_map
        self.configuration = configuration
        self.control_period_s = control_period_s
        self.footprint = RoverFootprint.from_rover(rover)
        self.global_planner = GlobalPlanner(planner_map, configuration)
        self.local_planner = DynamicWindowPlanner(configuration, self.footprint, control_period_s)
        self.state = "idle"
        self.reason = ""
        self.goal_m: np.ndarray | None = None
        self.route: Route | None = None
        self.last_command = np.zeros(2)
        self.statistics = ExpertStatistics()
        self._progress: deque = deque()
        self._last_stall_position: np.ndarray | None = None

    # ------------------------------------------------------------------ goals
    def set_goal(self, pose: np.ndarray, goal_m) -> Route:
        """Plan from ``pose`` (east, north, yaw) to ``goal_m``; the state becomes driving, or aborted without a route."""
        self.goal_m = np.asarray(goal_m, dtype=np.float64)
        self.statistics = ExpertStatistics()
        self.last_command = np.zeros(2)
        self._progress.clear()
        self._last_stall_position = None
        route = self._plan(np.asarray(pose, dtype=np.float64))
        if route.reachable:
            self.goal_m = route.goal_m.copy()
        return route

    def cancel(self) -> None:
        """Drop the goal (idle)."""
        self.state, self.reason, self.goal_m, self.route = "idle", "", None, None

    def _plan(self, pose: np.ndarray) -> Route:
        """Plan from the current pose to the goal and update the state."""
        route = self.global_planner.plan(pose[:2], self.goal_m)
        self.statistics.plans += 1
        if route.reachable:
            self.route, self.state, self.reason = route, "driving", ""
            self.statistics.route_lengths_m.append(round(route.length_m, 3))
        else:
            self.route, self.state, self.reason = None, "aborted", route.reason
        return route

    # ------------------------------------------------------------------ control step
    def _stalled(self, time_s: float, pose: np.ndarray) -> bool:
        """True when the pose changed less than the minima over the last stall window (window then restarts)."""
        self._progress.append((time_s, pose.copy()))
        while self._progress and time_s - self._progress[0][0] > self.configuration.stall_window_s:
            self._progress.popleft()
        oldest_time, oldest_pose = self._progress[0]
        if time_s - oldest_time < self.configuration.stall_window_s - 1.5 * self.control_period_s:
            return False
        moved = math.hypot(*(pose[:2] - oldest_pose[:2]))
        turned = abs(float(wrap_angle(pose[2] - oldest_pose[2])))
        return moved < self.configuration.stall_minimum_motion_m and \
            turned < math.radians(self.configuration.stall_minimum_turn_deg)

    def _route_deviation_m(self, position_m: np.ndarray) -> float:
        """Distance from the rover to the nearest route vertex (m)."""
        path = self.route.path_m
        return float(np.hypot(path[:, 0] - position_m[0], path[:, 1] - position_m[1]).min())

    def command(self, time_s: float, pose: np.ndarray, tilt_rad: float) -> ExpertCommand:
        """Twist for this control step at ``pose`` (east, north, yaw) with body tilt ``tilt_rad`` from vertical."""
        pose = np.asarray(pose, dtype=np.float64)
        if self.state in ("idle", "reached", "aborted"):
            self.last_command = np.zeros(2)
            return ExpertCommand(0.0, 0.0, self.state, self.reason)
        distance = float(math.hypot(*(self.goal_m - pose[:2])))
        if tilt_rad > math.radians(self.configuration.maximum_tilt_deg):
            self.state, self.reason = "aborted", f"tilt {math.degrees(tilt_rad):.1f} deg"
            self.last_command = np.zeros(2)
            return ExpertCommand(0.0, 0.0, self.state, self.reason, distance)
        if distance <= self.configuration.stop_tolerance_m:
            self.state, self.reason = "reached", ""
            self.last_command = np.zeros(2)
            return ExpertCommand(0.0, 0.0, self.state, "", distance)
        replanned = False
        stalled = self._stalled(time_s, pose)
        if stalled or self._route_deviation_m(pose[:2]) > self.configuration.replan_deviation_m:
            if stalled:
                self.statistics.stalls_in_a_row += 1
                self.statistics.total_stalls += 1
                self._progress.clear()
                self._last_stall_position = pose[:2].copy()
                if self.statistics.stalls_in_a_row > self.configuration.maximum_stalls:
                    self.state, self.reason = "aborted", f"stalled {self.statistics.stalls_in_a_row} times in a row"
                    self.last_command = np.zeros(2)
                    return ExpertCommand(0.0, 0.0, self.state, self.reason, distance)
            if not self._plan(pose).reachable:
                self.last_command = np.zeros(2)
                return ExpertCommand(0.0, 0.0, self.state, self.reason, distance)
            replanned = True
        local = self.local_planner.command(pose, tuple(self.last_command), self.route.path_m, self.goal_m,
                                           self.map.rock_obstacles)
        if local.blocked:
            self.statistics.blocked_steps += 1
        if self._last_stall_position is not None and \
                math.hypot(*(pose[:2] - self._last_stall_position)) >= self.configuration.stall_minimum_motion_m:
            self.statistics.stalls_in_a_row, self._last_stall_position = 0, None   # progress resumed after a stall
        self.last_command = np.array([local.forward_speed_m_s, local.yaw_rate_rad_s])
        return ExpertCommand(local.forward_speed_m_s, local.yaw_rate_rad_s, "driving", "", distance,
                             local.local_goal_m, replanned, local.blocked)


def tilt_from_quaternion(quaternion_xyzw) -> float:
    """Angle (rad) between the body z axis and the world vertical."""
    x, y, _, _ = np.asarray(quaternion_xyzw, dtype=np.float64) / np.linalg.norm(quaternion_xyzw)
    body_z_up = 1.0 - 2.0 * (x * x + y * y)
    return float(math.acos(max(-1.0, min(1.0, body_z_up))))
