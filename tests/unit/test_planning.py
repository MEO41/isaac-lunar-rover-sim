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


# ---------------------------------------------------------------- obstacle craters (v1.0.5)

def crater_table(rows: list[tuple[float, float, float]], harmonic_amplitude: float = 0.0, ellipticity: float = 1.0):
    """CraterTable of craters (east, north, diameter), optionally with one rim harmonic and an ellipticity."""
    from lunar_rover_sim.core.terrain.crater_population import HARMONIC_ORDERS, CraterTable
    count, orders = len(rows), len(HARMONIC_ORDERS)
    amplitudes = np.zeros((count, orders))
    amplitudes[:, 0] = harmonic_amplitude
    return CraterTable(np.array([r[0] for r in rows], float), np.array([r[1] for r in rows], float),
                       np.array([r[2] for r in rows], float), np.full(count, 0.15), np.zeros(count),
                       np.full(count, ellipticity), np.full(count, 0.4), amplitudes, np.full((count, orders), 0.3),
                       np.zeros(count, dtype=np.int64), np.zeros(count, dtype=bool))


def crater_world(craters, expert_configuration, rover_configuration, size_m: float = 16.0, **table_options):
    """Planner map of flat ground with obstacle craters (east, north, diameter) and no rocks."""
    grid = RegularGrid.centered(size_m, 0.05)
    return build_planner_map(np.zeros(grid.shape, dtype=np.float32), grid, size_m / 2 - 1.0, RockObstacles.empty(),
                             crater_table(craters, **table_options), expert_configuration,
                             RoverFootprint.from_rover(rover_configuration))


def test_bowl_edge_distance_agrees_with_the_crater_formula():
    """Distance is zero exactly where the stamping formula has r' = 1, negative inside, (r' - 1) R everywhere."""
    from lunar_rover_sim.core.planning.planner_map import CraterObstacles
    from lunar_rover_sim.core.terrain.crater_stamping import normalised_crater_radius
    table = crater_table([(1.0, -2.0, 3.0)], harmonic_amplitude=0.08, ellipticity=1.2)
    craters = CraterObstacles.from_table(table, 1.5)
    east, north = np.meshgrid(np.linspace(-4, 6, 101), np.linspace(-7, 3, 101))
    expected = (normalised_crater_radius(east, north, table, 0) - 1.0) * 1.5
    assert np.allclose(craters.bowl_edge_distance_m(east, north), expected, atol=1e-12)
    small = CraterObstacles.from_table(crater_table([(0.0, 0.0, 1.0)]), 1.5)
    assert len(small) == 0 and np.isinf(small.bowl_edge_distance_m(east, north)).all()


def test_route_keeps_hard_clearance_from_a_crater_bowl(expert_configuration, rover_configuration):
    """A crater on the straight line: the route keeps the rover centre the hard clearance away from the bowl."""
    planner_map = crater_world([(0.0, 0.0, 3.0)], expert_configuration, rover_configuration)
    assert len(planner_map.crater_obstacles) == 1
    route = GlobalPlanner(planner_map, expert_configuration).plan((-5.0, 0.0), (5.0, 0.0))
    assert route.reachable
    distance = planner_map.crater_obstacles.bowl_edge_distance_m(route.path_m[:, 0], route.path_m[:, 1])
    assert distance.min() >= planner_map.hard_clearance_m - 0.15          # 0.15: planner cell diagonal and smoothing


def test_dynamic_window_refuses_driving_into_a_crater_bowl(expert_configuration, rover_configuration):
    """Heading straight at a close bowl with the route through it: the chosen arc never enters the bowl."""
    from lunar_rover_sim.core.planning.dynamic_window_planner import crater_bowl_entered
    from lunar_rover_sim.core.planning.planner_map import CraterObstacles
    planner = DynamicWindowPlanner(expert_configuration, RoverFootprint.from_rover(rover_configuration), 0.1)
    craters = CraterObstacles.from_table(crater_table([(1.6, 0.0, 2.0)]), 1.5)
    route = np.column_stack([np.linspace(0, 6, 61), np.zeros(61)])
    command = planner.command(np.zeros(3), (0.3, 0.0), route, np.array([6.0, 0.0]), RockObstacles.empty(), craters)
    trajectory = predict_arcs(np.zeros(3), np.array([command.forward_speed_m_s]), np.array([command.yaw_rate_rad_s]),
                              expert_configuration.prediction_horizon_s, expert_configuration.prediction_step_s)
    assert command.refused_count > 0
    assert command.blocked or not crater_bowl_entered(trajectory, craters, expert_configuration.obstacle_safety_m)[0]
    unaware = planner.command(np.zeros(3), (0.3, 0.0), route, np.array([6.0, 0.0]), RockObstacles.empty())
    assert unaware.refused_count == 0                                      # without craters nothing is refused


def test_expert_drives_a_kinematic_rover_past_craters(expert_configuration, rover_configuration):
    """Closed loop with a lagged unicycle through a crater field: goal reached, the centre never enters a bowl."""
    craters = [(0.0, 0.3, 3.0), (3.2, -1.8, 2.0), (-3.0, 1.6, 2.4), (2.5, 2.4, 1.6)]
    planner_map = crater_world(craters, expert_configuration, rover_configuration, size_m=20.0)
    period = 0.1
    expert = ScriptedExpert(planner_map, expert_configuration, rover_configuration, period)
    pose = np.array([-6.0, 0.0, 0.0])
    assert expert.set_goal(pose, (6.0, 0.5)).reachable
    velocity, minimum_distance = np.zeros(2), np.inf
    for step in range(1200):
        command = expert.command(step * period, pose, 0.0)
        if command.state != "driving":
            break
        velocity += 0.5 * (np.array([command.forward_speed_m_s, command.yaw_rate_rad_s]) - velocity)   # actuator lag
        pose = predict_arcs(pose, velocity[:1], velocity[1:], period, period)[0, -1]
        minimum_distance = min(minimum_distance,
                               float(planner_map.crater_obstacles.bowl_edge_distance_m(pose[:1], pose[1:2])[0]))
    assert expert.state == "reached", (expert.state, expert.reason, pose)
    assert minimum_distance > expert_configuration.obstacle_safety_m
