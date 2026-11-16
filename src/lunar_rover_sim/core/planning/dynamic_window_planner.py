"""Local planner of the scripted expert: the twist command that follows the route (Dynamic Window Approach).

Same rules as MARE's ``local_planner.py`` (itself after the PythonRobotics DWA sample of A. Sakai and G. Karakasli,
MIT licence); this is a re-implementation, vectorised over all sampled commands:
 1. The local goal is the first route point at least ``lookahead_m`` from the route point nearest the rover.
 2. Every (forward speed, yaw rate) reachable within one control period is simulated as a constant-twist arc for the
    prediction horizon. A trajectory that brings the footprint rectangle within ``obstacle_safety_m`` of a rock disc
    is refused; of the rest, the one heading best at the local goal at the highest speed wins
    (cost = heading_gain * |heading error| + speed_gain * |max_speed * exp(-heading cost) - v|).
 3. Within the braking distance the speed limit falls linearly (not below the approach minimum), and the yaw rate is
    damped while the rover already heads at the goal, so it parks instead of circling it.
Keeping a distance from rocks is the route's job; the DWA only refuses to hit them.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from lunar_rover_sim.core.configuration.configuration_models import ExpertConfiguration
from lunar_rover_sim.core.planning.planner_map import RockObstacles, RoverFootprint


@dataclass
class LocalCommand:
    """One control-step decision of the local planner."""

    forward_speed_m_s: float
    yaw_rate_rad_s: float
    local_goal_m: np.ndarray
    predicted_trajectory: np.ndarray
    blocked: bool
    candidate_count: int
    refused_count: int


def wrap_angle(angle_rad):
    """Angle(s) wrapped to (-pi, pi]."""
    return np.arctan2(np.sin(angle_rad), np.cos(angle_rad))


def lookahead_point(position_m: np.ndarray, route_m: np.ndarray, lookahead_m: float) -> tuple[np.ndarray, int]:
    """(point, index): from the route point nearest the rover, the first at least ``lookahead_m`` away (else the last)."""
    distances = np.hypot(route_m[:, 0] - position_m[0], route_m[:, 1] - position_m[1])
    index = int(np.argmin(distances))
    while index < len(route_m) - 1 and distances[index] < lookahead_m:
        index += 1
    return route_m[index], index


def predict_arcs(pose: np.ndarray, forward_speed_m_s: np.ndarray, yaw_rate_rad_s: np.ndarray, horizon_s: float,
                 step_s: float) -> np.ndarray:
    """Poses (K, S + 1, 3: east, north, yaw) of constant-twist arcs from ``pose`` (exact unicycle integration)."""
    times = np.arange(0.0, horizon_s + 1e-9, step_s)
    speed = np.asarray(forward_speed_m_s, dtype=np.float64)[:, None]
    yaw_rate = np.asarray(yaw_rate_rad_s, dtype=np.float64)[:, None]
    yaw = pose[2] + yaw_rate * times[None, :]
    straight = np.abs(yaw_rate) < 1e-6
    safe_rate = np.where(straight, 1.0, yaw_rate)
    east = np.where(straight, pose[0] + speed * np.cos(pose[2]) * times,
                    pose[0] + speed / safe_rate * (np.sin(yaw) - math.sin(pose[2])))
    north = np.where(straight, pose[1] + speed * np.sin(pose[2]) * times,
                     pose[1] - speed / safe_rate * (np.cos(yaw) - math.cos(pose[2])))
    return np.stack([east, north, yaw], axis=-1)


def footprint_collides(trajectories: np.ndarray, obstacles: RockObstacles, footprint: RoverFootprint,
                       safety_m: float) -> np.ndarray:
    """(K,) True where any pose of a trajectory brings the footprint rectangle within ``safety_m`` of a rock disc."""
    count = trajectories.shape[0]
    if len(obstacles) == 0:
        return np.zeros(count, dtype=bool)
    half_length, half_width = footprint.length_m / 2, footprint.width_m / 2
    reach = math.hypot(half_length, half_width) + safety_m
    centers = obstacles.centers_m()
    positions = trajectories[:, :, :2].reshape(-1, 2)
    near = np.hypot(positions[:, None, 0] - centers[None, :, 0], positions[:, None, 1] - centers[None, :, 1]).min(axis=0) \
        <= reach + obstacles.radius_m
    if not near.any():
        return np.zeros(count, dtype=bool)
    centers, radii = centers[near], obstacles.radius_m[near]
    relative = centers[None, None, :, :] - trajectories[:, :, None, :2]
    cosine, sine = np.cos(trajectories[:, :, 2])[..., None], np.sin(trajectories[:, :, 2])[..., None]
    along = relative[..., 0] * cosine + relative[..., 1] * sine
    across = -relative[..., 0] * sine + relative[..., 1] * cosine
    outside_length = np.maximum(np.abs(along) - half_length, 0.0)
    outside_width = np.maximum(np.abs(across) - half_width, 0.0)
    gap = np.hypot(outside_length, outside_width) - radii[None, None, :]
    return (gap <= safety_m).any(axis=(1, 2))


def footprint_gap_m(pose: np.ndarray, obstacles: RockObstacles, footprint: RoverFootprint) -> float:
    """Smallest distance (m) from the footprint rectangle at ``pose`` to any rock disc (negative = overlap; inf if none)."""
    if len(obstacles) == 0:
        return float("inf")
    relative = obstacles.centers_m() - np.asarray(pose[:2], dtype=np.float64)
    cosine, sine = math.cos(pose[2]), math.sin(pose[2])
    along = relative[:, 0] * cosine + relative[:, 1] * sine
    across = -relative[:, 0] * sine + relative[:, 1] * cosine
    outside = np.hypot(np.maximum(np.abs(along) - footprint.length_m / 2, 0.0),
                       np.maximum(np.abs(across) - footprint.width_m / 2, 0.0))
    return float((outside - obstacles.radius_m).min())


def approach_speed_limit(maximum_speed_m_s: float, distance_to_goal_m: float, configuration: ExpertConfiguration) -> float:
    """Speed limit near the goal: linear over the braking distance, not below the approach minimum."""
    if distance_to_goal_m >= configuration.braking_distance_m:
        return maximum_speed_m_s
    return max(maximum_speed_m_s * distance_to_goal_m / configuration.braking_distance_m,
               configuration.approach_minimum_speed_m_s)


class DynamicWindowPlanner:
    """DWA with the expert configuration and the rover footprint."""

    def __init__(self, configuration: ExpertConfiguration, footprint: RoverFootprint, control_period_s: float):
        """Check that the speed resolution can reach a non-zero speed from rest in one period."""
        if configuration.speed_resolution_m_s >= configuration.maximum_forward_acceleration_m_s2 * control_period_s:
            raise ValueError("speed_resolution_m_s must be below maximum_forward_acceleration_m_s2 x control period, "
                             "otherwise only zero speed is sampled from rest")
        self.configuration = configuration
        self.footprint = footprint
        self.control_period_s = control_period_s

    def dynamic_window(self, forward_speed_m_s: float, yaw_rate_rad_s: float,
                       speed_limit_m_s: float) -> tuple[np.ndarray, np.ndarray]:
        """Sampled speeds and yaw rates reachable within one control period (endpoints included)."""
        configuration, period = self.configuration, self.control_period_s
        speed_low = max(0.0, forward_speed_m_s - configuration.maximum_forward_acceleration_m_s2 * period)
        speed_high = max(speed_low, min(speed_limit_m_s, forward_speed_m_s + configuration.maximum_forward_acceleration_m_s2 * period))
        rate_step = configuration.maximum_yaw_acceleration_rad_s2 * period
        rate_low = max(-configuration.maximum_yaw_rate_rad_s, yaw_rate_rad_s - rate_step)
        rate_high = min(configuration.maximum_yaw_rate_rad_s, yaw_rate_rad_s + rate_step)
        speeds = np.unique(np.append(np.arange(speed_low, speed_high, configuration.speed_resolution_m_s), speed_high))
        rates = np.unique(np.append(np.arange(rate_low, rate_high, configuration.yaw_rate_resolution_rad_s), rate_high))
        return speeds, rates

    def command(self, pose: np.ndarray, velocity: tuple[float, float], route_m: np.ndarray, goal_m: np.ndarray,
                obstacles: RockObstacles) -> LocalCommand:
        """Twist at ``pose`` (east, north, yaw) moving at ``velocity`` (forward m/s, yaw rad/s) along ``route_m``."""
        configuration = self.configuration
        pose = np.asarray(pose, dtype=np.float64)
        distance_to_goal = float(math.hypot(goal_m[0] - pose[0], goal_m[1] - pose[1]))
        local_goal, _ = lookahead_point(pose[:2], route_m, configuration.lookahead_m)
        speed_limit = approach_speed_limit(configuration.maximum_forward_speed_m_s, distance_to_goal, configuration)
        speeds, rates = self.dynamic_window(float(velocity[0]), float(velocity[1]), speed_limit)
        speed_grid, rate_grid = (array.ravel() for array in np.meshgrid(speeds, rates, indexing="ij"))
        trajectories = predict_arcs(pose, speed_grid, rate_grid, configuration.prediction_horizon_s,
                                    configuration.prediction_step_s)
        end = trajectories[:, -1, :]
        heading_error = np.abs(wrap_angle(np.arctan2(local_goal[1] - end[:, 1], local_goal[0] - end[:, 0]) - end[:, 2]))
        heading_cost = configuration.heading_cost_gain * heading_error
        speed_cost = configuration.speed_cost_gain * np.abs(speed_limit * np.exp(-heading_cost) - speed_grid)
        refused = footprint_collides(trajectories, obstacles, self.footprint, configuration.obstacle_safety_m)
        total = np.where(refused, np.inf, heading_cost + speed_cost)
        if not np.isfinite(total).any():
            return LocalCommand(0.0, 0.0, local_goal, trajectories[:1, :1, :][0], True, len(total), int(refused.sum()))
        best = int(np.argmin(total))
        speed, rate = float(speed_grid[best]), float(rate_grid[best])
        goal_heading_error = float(wrap_angle(math.atan2(goal_m[1] - pose[1], goal_m[0] - pose[0]) - pose[2]))
        trajectory = trajectories[best]
        if distance_to_goal < configuration.braking_distance_m and \
                abs(goal_heading_error) <= math.radians(configuration.approach_aligned_heading_deg):
            damped = 0.0 if distance_to_goal < configuration.stop_tolerance_m + 0.5 else \
                rate * distance_to_goal / configuration.braking_distance_m
            damped_trajectory = predict_arcs(pose, np.array([speed]), np.array([damped]),
                                             configuration.prediction_horizon_s, configuration.prediction_step_s)
            if not footprint_collides(damped_trajectory, obstacles, self.footprint, configuration.obstacle_safety_m)[0]:
                rate, trajectory = damped, damped_trajectory[0]      # the damped command is kept only when it is safe
        return LocalCommand(speed, rate, local_goal, trajectory, False, len(total), int(refused.sum()))
