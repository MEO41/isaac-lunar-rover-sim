"""Scripted expert planning (Isaac-free): fast marching, global route, dynamic window, closed-loop kinematic drive."""
from __future__ import annotations

import math

import numpy as np
import pytest

from lunar_rover_sim.core.data_model.regular_grid import RegularGrid
from lunar_rover_sim.core.planning.dynamic_window_planner import DynamicWindowPlanner, footprint_collides, predict_arcs
from lunar_rover_sim.core.planning.fast_marching import trace_route, travel_time
from lunar_rover_sim.core.planning.global_planner import GlobalPlanner, sample_reachable_goal
from lunar_rover_sim.core.planning.planner_map import (RockObstacles, RoverFootprint, build_planner_map,
                                                       distance_to_rock_discs, slope_cost)
from lunar_rover_sim.core.planning.scripted_expert import ScriptedExpert, tilt_from_quaternion


def flat_world(rocks: list[tuple[float, float, float]], expert_configuration, rover_configuration, size_m: float = 16.0):
    """Planner map of flat ground (collision spacing 0.05 m) with rock discs (east, north, radius)."""
    grid = RegularGrid.centered(size_m, 0.05)
    heights = np.zeros(grid.shape, dtype=np.float32)
    obstacles = RockObstacles(np.array([r[0] for r in rocks], float), np.array([r[1] for r in rocks], float),
                              np.array([r[2] for r in rocks], float), np.full(len(rocks), 0.3),
                              np.arange(len(rocks))) if rocks else RockObstacles.empty()
    return build_planner_map(heights, grid, size_m / 2 - 1.0, obstacles, None, expert_configuration,
                             RoverFootprint.from_rover(rover_configuration))


def test_fast_marching_uniform_cost_matches_euclidean_distance():
    """Uniform cost: T is the distance (exact on axes, first-order scheme within 8 % on the diagonal)."""
    cost = np.ones((81, 81))
    times = travel_time(cost, np.zeros_like(cost, dtype=bool), (40, 40))
    assert times[40, 80] == pytest.approx(40.0, rel=1e-9)
    assert times[0, 40] == pytest.approx(40.0, rel=1e-9)
    diagonal = 40 * math.sqrt(2)
    assert diagonal <= times[0, 0] <= 1.08 * diagonal


def test_fast_marching_route_goes_around_a_wall():
    """A wall with a gap: the traced route reaches the goal through the gap and never enters the wall."""
    impassable = np.zeros((60, 60), dtype=bool)
    impassable[10:60, 30] = True
    times = travel_time(np.ones((60, 60)), impassable, (50, 50))
    route = trace_route(times, impassable, (50, 10), (50, 50))
    assert route is not None
    cells = np.rint(route).astype(int)
    assert not impassable[cells[:, 0], cells[:, 1]].any()
    assert cells[:, 0].min() < 10                      # passed through the gap at the top
    assert tuple(cells[-1]) == (50, 50)


def test_slope_cost_table(expert_configuration):
    """DyMu table: 10 on flat ground, linear between breakpoints, maximum at and beyond 20 deg."""
    cost = slope_cost(np.array([0.0, 2.5, 15.0, 20.0, 35.0]), expert_configuration)
    assert cost.tolist() == pytest.approx([10.0, 11.0, 20.0, 50.0, 50.0])


def test_global_route_keeps_clearance_from_a_rock(expert_configuration, rover_configuration):
    """A rock on the straight line: the route goes around it, keeping hard + preferred clearance."""
    planner_map = flat_world([(0.0, 0.0, 0.5)], expert_configuration, rover_configuration)
    route = GlobalPlanner(planner_map, expert_configuration).plan((-5.0, 0.0), (5.0, 0.0))
    assert route.reachable and route.kept_preferred_clearance
    gap = distance_to_rock_discs(route.path_m[:, 0], route.path_m[:, 1], planner_map.rock_obstacles)
    assert gap.min() >= planner_map.hard_clearance_m + expert_configuration.preferred_clearance_m - 0.15
    assert route.length_m < 10.0 * 1.25


def test_global_route_drops_preferred_clearance_in_a_narrow_gap(expert_configuration, rover_configuration):
    """Two rocks leave a gap wide enough for the rover but not for the preferred margin: the fallback is used."""
    footprint = RoverFootprint.from_rover(rover_configuration)
    half_gap = footprint.width_m / 2 + expert_configuration.route_side_clearance_m + 0.15
    rocks = [(0.0, sign * (half_gap + 3.0 + 6.0 * k), 3.0) for sign in (1, -1) for k in (0, 1)]   # wall to the border
    planner_map = flat_world(rocks, expert_configuration, rover_configuration, size_m=24.0)
    route = GlobalPlanner(planner_map, expert_configuration).plan((-6.0, 0.0), (6.0, 0.0))
    assert route.reachable and not route.kept_preferred_clearance
    gap = distance_to_rock_discs(route.path_m[:, 0], route.path_m[:, 1], planner_map.rock_obstacles)
    assert gap.min() > planner_map.hard_clearance_m - 0.15


def test_goal_inside_a_rock_is_snapped_or_refused(expert_configuration, rover_configuration):
    """A goal on a rock is moved to drivable ground within the snap radius, or the plan is refused."""
    planner_map = flat_world([(3.0, 0.0, 0.3)], expert_configuration, rover_configuration)
    route = GlobalPlanner(planner_map, expert_configuration).plan((-4.0, 0.0), (3.0, 0.0))
    if route.reachable:
        assert route.details["goal_snapped"]
        assert math.hypot(*(route.goal_m - np.array([3.0, 0.0]))) <= expert_configuration.goal_snap_radius_m + 0.2
    else:
        assert route.reason == "goal not reachable"


def test_sampled_goal_has_a_route(expert_configuration, rover_configuration, random_generator):
    """Goal sampling returns a goal in the distance band with a route that keeps the preferred clearance."""
    planner_map = flat_world([(1.0, 1.0, 0.4), (-2.0, 3.0, 0.6)], expert_configuration, rover_configuration)
    goal, route = sample_reachable_goal(GlobalPlanner(planner_map, expert_configuration), (0.0, -2.0), 3.0, 5.0,
                                        random_generator)
    assert 3.0 <= math.hypot(goal[0], goal[1] + 2.0) <= 5.0
    assert route.reachable and route.kept_preferred_clearance


def test_footprint_collision_test():
    """A disc touching the footprint's side is a collision; one beyond the safety margin is not."""
    footprint = RoverFootprint(1.0, 0.8)
    pose = predict_arcs(np.zeros(3), np.zeros(1), np.zeros(1), 0.0, 0.1)
    touching = RockObstacles(np.array([0.0]), np.array([0.6]), np.array([0.25]), np.array([0.3]), np.array([0]))
    clear = RockObstacles(np.array([0.0]), np.array([0.8]), np.array([0.25]), np.array([0.3]), np.array([0]))
    assert footprint_collides(pose, touching, footprint, 0.1)[0]
    assert not footprint_collides(pose, clear, footprint, 0.1)[0]


def test_dynamic_window_refuses_driving_into_a_rock(expert_configuration, rover_configuration):
    """Moving at speed towards a close rock with the route through it: no chosen trajectory hits the rock."""
    footprint = RoverFootprint.from_rover(rover_configuration)
    planner = DynamicWindowPlanner(expert_configuration, footprint, 0.1)
    rock = RockObstacles(np.array([1.2]), np.array([0.0]), np.array([0.3]), np.array([0.3]), np.array([0]))
    route = np.column_stack([np.linspace(0, 6, 61), np.zeros(61)])
    command = planner.command(np.zeros(3), (0.3, 0.0), route, np.array([6.0, 0.0]), rock)
    trajectory = predict_arcs(np.zeros(3), np.array([command.forward_speed_m_s]), np.array([command.yaw_rate_rad_s]),
                              expert_configuration.prediction_horizon_s, expert_configuration.prediction_step_s)
    assert command.blocked or not footprint_collides(trajectory, rock, footprint, expert_configuration.obstacle_safety_m)[0]
    assert command.refused_count > 0


def test_tilt_from_quaternion():
    """Level pose = 0; 10 deg roll = 10 deg tilt."""
    assert tilt_from_quaternion([0, 0, 0, 1]) == pytest.approx(0.0, abs=1e-12)
    angle = math.radians(10)
    assert tilt_from_quaternion([math.sin(angle / 2), 0, 0, math.cos(angle / 2)]) == pytest.approx(angle)


def test_expert_drives_a_kinematic_rover_through_rocks(expert_configuration, rover_configuration):
    """Closed loop with a lagged unicycle: the goal is reached and the footprint never touches a rock."""
    rocks = [(0.0, 0.2, 0.5), (3.0, -1.5, 0.4), (-3.0, 1.0, 0.45), (2.0, 2.2, 0.35)]
    planner_map = flat_world(rocks, expert_configuration, rover_configuration, size_m=20.0)
    period = 0.1
    expert = ScriptedExpert(planner_map, expert_configuration, rover_configuration, period)
    pose = np.array([-6.0, 0.0, 0.0])
    route = expert.set_goal(pose, (6.0, 0.5))
    assert route.reachable
    velocity = np.zeros(2)
    footprint = RoverFootprint.from_rover(rover_configuration)
    minimum_gap = np.inf
    for step in range(1200):
        command = expert.command(step * period, pose, 0.0)
        if command.state != "driving":
            break
        velocity += 0.5 * (np.array([command.forward_speed_m_s, command.yaw_rate_rad_s]) - velocity)   # actuator lag
        pose = predict_arcs(pose, velocity[:1], velocity[1:], period, period)[0, -1]
        poses = pose[None, None, :]
        for safety in (0.0,):
            assert not footprint_collides(poses, planner_map.rock_obstacles, footprint, safety)[0]
        local = np.hypot(planner_map.rock_obstacles.center_east_m - pose[0],
                         planner_map.rock_obstacles.center_north_m - pose[1]) - planner_map.rock_obstacles.radius_m
        minimum_gap = min(minimum_gap, float(local.min()))
    assert expert.state == "reached", (expert.state, expert.reason, pose)
    assert math.hypot(*(pose[:2] - expert.goal_m)) <= expert_configuration.stop_tolerance_m
    assert minimum_gap > footprint.width_m / 2
