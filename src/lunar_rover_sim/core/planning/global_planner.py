"""Global planner of the scripted expert: a route from the rover to a goal over the planner map.

Plan (after MARE's ``global_planner.py``):
 1. A goal on impassable ground is moved to the nearest drivable cell of the rover's region within the snap radius,
    or refused.
 2. Cells near the start are released (the rover may stand closer to a rock than the clearance), except cells inside
    a rock disc itself.
 3. The preferred clearance ring around rocks is closed; the fast marching search runs from the goal to the rover and
    the route is traced back. If no route keeps the ring, the search is repeated without it.

Positions are (east, north) metres; routes are (N, 2) metres, rover first, goal last.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import ndimage

from lunar_rover_sim.core.configuration.configuration_models import ExpertConfiguration
from lunar_rover_sim.core.planning.fast_marching import trace_route, travel_time
from lunar_rover_sim.core.planning.planner_map import FOUR_CONNECTED, PlannerMap, distance_to_rock_discs


@dataclass
class Route:
    """Outcome of one plan."""

    reachable: bool
    path_m: np.ndarray | None = None
    goal_m: np.ndarray | None = None
    length_m: float = 0.0
    travel_cost: float = 0.0
    kept_preferred_clearance: bool = False
    reason: str = ""
    details: dict = field(default_factory=dict)


def route_length_m(path_m: np.ndarray) -> float:
    """Polyline length (m)."""
    return float(np.hypot(*np.diff(path_m, axis=0).T).sum()) if len(path_m) > 1 else 0.0


class GlobalPlanner:
    """Fast-marching routes over one planner map."""

    def __init__(self, planner_map: PlannerMap, configuration: ExpertConfiguration):
        """Keep the map and precompute the cells inside rock discs (never released)."""
        self.map = planner_map
        self.configuration = configuration
        east, north = planner_map.grid.coordinate_mesh()
        self.east, self.north = east, north
        self.inside_rock = distance_to_rock_discs(east, north, planner_map.rock_obstacles) <= 0.0

    def _release_disc(self, impassable: np.ndarray, point_m: np.ndarray) -> np.ndarray:
        """Copy of ``impassable`` with clearance cells within the release radius of a point opened (rocks stay)."""
        near = np.hypot(self.east - point_m[0], self.north - point_m[1]) <= self.configuration.start_release_radius_m
        outside = (np.abs(self.east) > self.map.drivable_half_size_m) | (np.abs(self.north) > self.map.drivable_half_size_m)
        released = impassable.copy()
        released[near & ~self.inside_rock & ~outside & (self.map.slope_deg <= self.configuration.slope_threshold_deg)] = False
        return released

    def drivable_goal(self, impassable: np.ndarray, goal_cell: tuple[int, int], start_cell: tuple[int, int],
                      regions: np.ndarray) -> tuple[int, int] | None:
        """``goal_cell`` if drivable, else the nearest drivable cell of the start's region within the snap radius."""
        if not impassable[goal_cell] and regions[goal_cell] == regions[start_cell]:
            return goal_cell
        radius_cells = self.configuration.goal_snap_radius_m / self.map.grid.spacing_m
        rows, columns = np.nonzero(~impassable & (regions == regions[start_cell]) & (regions > 0))
        if not len(rows):
            return None
        squared = (rows - goal_cell[0]) ** 2 + (columns - goal_cell[1]) ** 2
        best = int(np.argmin(squared))
        if squared[best] > radius_cells ** 2:
            return None
        return int(rows[best]), int(columns[best])

    def plan(self, start_m, goal_m) -> Route:
        """Route from ``start_m`` to ``goal_m`` (east, north metres)."""
        start_m, goal_m = np.asarray(start_m, dtype=np.float64), np.asarray(goal_m, dtype=np.float64)
        try:
            start_cell = self.map.cell_of(*start_m)
            goal_cell = self.map.cell_of(*goal_m)
        except ValueError as error:
            return Route(False, reason="outside the planner map", details={"error": str(error)})
        base = self._release_disc(self.map.impassable, start_m)
        if base[start_cell]:
            return Route(False, reason="start on impassable ground")
        attempts = (("with preferred clearance", True), ("without preferred clearance", False))
        last_reason = ""
        for label, keep_ring in attempts:
            opened = self._release_mask(start_m) | self._release_mask(goal_m)
            impassable = base | (self.map.clearance_ring & ~opened) if keep_ring else base
            regions, _ = ndimage.label(~impassable, structure=FOUR_CONNECTED)
            goal = self.drivable_goal(impassable, goal_cell, start_cell, regions)
            if goal is None:
                last_reason = "goal not reachable"
                continue
            times = travel_time(self.map.cost, impassable, goal, start_cell)
            if not np.isfinite(times[start_cell]):
                last_reason = "goal not reachable"
                continue
            cells = trace_route(times, impassable, start_cell, goal)
            if cells is None or len(cells) < 2:
                last_reason = "no route could be traced"
                continue
            path = np.column_stack([self.map.grid.origin_east_m + cells[:, 1] * self.map.grid.spacing_m,
                                    self.map.grid.origin_north_m + cells[:, 0] * self.map.grid.spacing_m])
            path[0] = start_m
            snapped = goal != goal_cell
            if not snapped:
                path[-1] = goal_m
            return Route(True, path, path[-1].copy(), route_length_m(path), float(times[start_cell]) * self.map.grid.spacing_m,
                         keep_ring, details={"attempt": label, "goal_snapped": bool(snapped),
                                             "goal_cell": list(goal), "start_cell": list(start_cell)})
        return Route(False, reason=last_reason or "goal not reachable")

    def _release_mask(self, point_m: np.ndarray) -> np.ndarray:
        """Cells within the release radius of a point."""
        return np.hypot(self.east - point_m[0], self.north - point_m[1]) <= self.configuration.start_release_radius_m


def sample_reachable_goal(planner: GlobalPlanner, start_m, minimum_distance_m: float, maximum_distance_m: float,
                          random_generator: np.random.Generator, attempts: int = 20) -> tuple[np.ndarray, Route]:
    """A random drivable goal at a straight-line distance in [minimum, maximum] from the start, with its route.

    Candidates are drivable cells of the start's region; the first one with a route (keeping the preferred clearance)
    is returned. Raises RuntimeError when no candidate in ``attempts`` has one."""
    start_m = np.asarray(start_m, dtype=np.float64)
    planner_map = planner.map
    distance = np.hypot(planner.east - start_m[0], planner.north - start_m[1])
    start_cell = planner_map.cell_of(*start_m)
    candidates = np.flatnonzero((~(planner_map.impassable | planner_map.clearance_ring)).ravel() &
                                (planner_map.regions.ravel() == planner_map.regions[start_cell]) &
                                (planner_map.regions.ravel() > 0) &
                                (distance.ravel() >= minimum_distance_m) & (distance.ravel() <= maximum_distance_m))
    if not len(candidates):
        raise RuntimeError(f"no drivable goal {minimum_distance_m}-{maximum_distance_m} m from {start_m.round(2).tolist()}")
    for flat_index in random_generator.choice(candidates, size=min(attempts, len(candidates)), replace=False):
        row, column = np.unravel_index(int(flat_index), planner_map.grid.shape)
        goal = planner_map.position_of(row, column)
        route = planner.plan(start_m, goal)
        if route.reachable and route.kept_preferred_clearance:
            return goal, route
    raise RuntimeError(f"none of {attempts} sampled goals had a route keeping the preferred clearance")


__all__ = ["GlobalPlanner", "Route", "route_length_m", "sample_reachable_goal"]
