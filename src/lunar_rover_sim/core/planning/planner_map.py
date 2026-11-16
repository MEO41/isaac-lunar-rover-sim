"""Planner maps for the scripted expert: slope cost, impassable cells and drivable regions on a planner lattice.

Design after MARE's ``planner_maps.py`` (GNC-Moon-Simulator): travel cost per cell rises with slope along the DyMu
table, cells steeper than a threshold are impassable, the cost is smoothed over 3 x 3 cells, drivable regions are
4-connected components. Privileged additions of this project: rocks standing higher than ``rock_obstacle_height_m``
(their exposed height and footprint measured from the placed collider meshes) and the bowls of large craters are
impassable, both grown by the same hard clearance (rover half width + side clearance) and surrounded by the same
preferred-clearance ring, and so is everything outside the drivable square of the session.

Units: metres, degrees. Arrays are indexed [row = north, column = east] like every lattice of the project.
"""
from __future__ import annotations

import math

from dataclasses import dataclass, field

import numpy as np
from scipy import ndimage

from lunar_rover_sim.core.configuration.configuration_models import ExpertConfiguration, RoverConfiguration
from lunar_rover_sim.core.data_model.regular_grid import RegularGrid, triangulated_height
from lunar_rover_sim.core.frames.pose_math import rotation_matrix_from_quaternion
from lunar_rover_sim.core.scene.scene_rasters import slope_over_baseline_deg
from lunar_rover_sim.core.terrain.crater_population import HARMONIC_ORDERS, CraterTable

FOUR_CONNECTED = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool)


@dataclass
class RockObstacles:
    """Rocks the expert never drives over, as discs on the ground plane (one row per rock)."""

    center_east_m: np.ndarray
    center_north_m: np.ndarray
    radius_m: np.ndarray
    exposed_height_m: np.ndarray
    rock_index: np.ndarray

    def __len__(self) -> int:
        """Number of obstacle rocks."""
        return len(self.radius_m)

    @classmethod
    def empty(cls) -> "RockObstacles":
        """No obstacles."""
        return cls(*(np.zeros(0) for _ in range(4)), np.zeros(0, dtype=np.int64))

    def centers_m(self) -> np.ndarray:
        """(N, 2) east, north."""
        return np.column_stack([self.center_east_m, self.center_north_m])


@dataclass(frozen=True)
class CraterObstacles:
    """Craters whose bowl (inside the rim crest) the expert never enters, one row per crater (table rows kept)."""

    table_row: np.ndarray
    center_east_m: np.ndarray
    center_north_m: np.ndarray
    radius_m: np.ndarray
    ellipticity: np.ndarray
    orientation_rad: np.ndarray
    harmonic_amplitudes: np.ndarray
    harmonic_phases_rad: np.ndarray

    def __len__(self) -> int:
        """Number of obstacle craters."""
        return len(self.table_row)

    @classmethod
    def empty(cls) -> "CraterObstacles":
        """No obstacle craters."""
        orders = len(HARMONIC_ORDERS)
        return cls(np.zeros(0, dtype=np.int64), *(np.zeros(0) for _ in range(5)), np.zeros((0, orders)),
                   np.zeros((0, orders)))

    @classmethod
    def from_table(cls, table: CraterTable | None, minimum_diameter_m: float) -> "CraterObstacles":
        """Craters of the table at least ``minimum_diameter_m`` wide."""
        if table is None or not len(table):
            return cls.empty()
        rows = np.flatnonzero(np.asarray(table.diameter_m) >= minimum_diameter_m)
        return cls(rows.astype(np.int64), np.asarray(table.center_east_m, dtype=np.float64)[rows],
                   np.asarray(table.center_north_m, dtype=np.float64)[rows],
                   np.asarray(table.diameter_m, dtype=np.float64)[rows] / 2.0,
                   np.asarray(table.ellipticity, dtype=np.float64)[rows],
                   np.asarray(table.orientation_rad, dtype=np.float64)[rows],
                   np.asarray(table.harmonic_amplitudes, dtype=np.float64)[rows],
                   np.asarray(table.harmonic_phases_rad, dtype=np.float64)[rows])

    def normalised_radius(self, index: int, east_m: np.ndarray, north_m: np.ndarray) -> np.ndarray:
        """r' of crater ``index`` at points (formula of ``core.terrain.crater_stamping.normalised_crater_radius``);
        r' <= 1 is inside the bowl."""
        offset_east = np.asarray(east_m, dtype=np.float64) - self.center_east_m[index]
        offset_north = np.asarray(north_m, dtype=np.float64) - self.center_north_m[index]
        cosine, sine = math.cos(self.orientation_rad[index]), math.sin(self.orientation_rad[index])
        along = offset_east * cosine + offset_north * sine
        across = (-offset_east * sine + offset_north * cosine) / self.ellipticity[index]
        rho = np.hypot(along, across)
        theta = np.arctan2(across, along)
        modulation = np.ones_like(rho)
        for order_index, order in enumerate(HARMONIC_ORDERS):
            modulation += self.harmonic_amplitudes[index, order_index] * np.cos(
                order * theta + self.harmonic_phases_rad[index, order_index])
        return rho / (self.radius_m[index] * modulation)

    def bowl_edge_distance_m(self, east_m: np.ndarray, north_m: np.ndarray) -> np.ndarray:
        """Distance (m) from each point to the nearest bowl edge, negative inside; inf without craters.

        APPROXIMATION: (r' - 1) x nominal radius; exact for a circular crater without rim harmonics, and zero exactly
        on the rim crest (r' = 1) for every crater. Craters whose bowl cannot come within 2 m of any point are skipped
        (bounding radius: nominal radius x max(1, ellipticity) x (1 + sum of harmonic amplitudes))."""
        east_m = np.asarray(east_m, dtype=np.float64)
        north_m = np.asarray(north_m, dtype=np.float64)
        distance = np.full(east_m.shape, np.inf)
        if len(self) == 0 or east_m.size == 0:
            return distance
        bounding = self.radius_m * np.maximum(1.0, self.ellipticity) * (1.0 + np.abs(self.harmonic_amplitudes).sum(axis=1))
        low_east, high_east = float(east_m.min()), float(east_m.max())
        low_north, high_north = float(north_m.min()), float(north_m.max())
        outside_east = np.maximum(np.maximum(low_east - self.center_east_m, self.center_east_m - high_east), 0.0)
        outside_north = np.maximum(np.maximum(low_north - self.center_north_m, self.center_north_m - high_north), 0.0)
        for index in np.flatnonzero(np.hypot(outside_east, outside_north) <= bounding + 2.0):
            distance = np.minimum(distance, (self.normalised_radius(int(index), east_m, north_m) - 1.0)
                                  * self.radius_m[index])
        return distance


@dataclass(frozen=True)
class RoverFootprint:
    """Rectangle around the rover centre (wheel contact rectangle plus tyres), metres."""

    length_m: float
    width_m: float

    @classmethod
    def from_rover(cls, rover: RoverConfiguration) -> "RoverFootprint":
        """Length = wheelbase + wheel diameter, width = track width + wheel width."""
        return cls(rover.wheelbase_m + 2 * rover.wheel_radius_m, rover.track_width_m + rover.wheel_width_m)


@dataclass
class PlannerMap:
    """Everything the global planner searches over (planner lattice posts are the cells)."""

    grid: RegularGrid
    slope_deg: np.ndarray
    cost: np.ndarray
    impassable: np.ndarray
    clearance_ring: np.ndarray
    regions: np.ndarray
    rock_obstacles: RockObstacles
    hard_clearance_m: float
    drivable_half_size_m: float
    crater_obstacles: CraterObstacles = field(default_factory=CraterObstacles.empty)

    def cell_of(self, east_m: float, north_m: float) -> tuple[int, int]:
        """(row, column) of the nearest planner post; raises ValueError outside the lattice."""
        row, column = self.grid.fractional_index(east_m, north_m)
        row_index, column_index = int(round(float(row))), int(round(float(column)))
        if not (0 <= row_index < self.grid.count_north and 0 <= column_index < self.grid.count_east):
            raise ValueError(f"point ({east_m:.2f}, {north_m:.2f}) is outside the planner lattice")
        return row_index, column_index

    def position_of(self, row: float, column: float) -> np.ndarray:
        """(east, north) of a (fractional) planner cell."""
        return np.array([self.grid.origin_east_m + column * self.grid.spacing_m,
                         self.grid.origin_north_m + row * self.grid.spacing_m])


def slope_cost(slope_deg: np.ndarray, configuration: ExpertConfiguration) -> np.ndarray:
    """Travel cost per cell: linear between the DyMu breakpoints, ``slope_cost_maximum`` at and beyond the last."""
    breakpoints = np.asarray(configuration.slope_cost_breakpoints_deg, dtype=np.float64)
    values = np.asarray(configuration.slope_cost_values, dtype=np.float64)
    if len(breakpoints) != len(values) or np.any(np.diff(breakpoints) <= 0):
        raise ValueError("slope_cost_breakpoints_deg must be increasing and as long as slope_cost_values")
    cost = np.interp(slope_deg, breakpoints, values)
    return np.where(slope_deg >= breakpoints[-1], configuration.slope_cost_maximum, cost)


def rock_obstacles_from_table(rock_arrays: dict, collision_height_m: np.ndarray, collision_grid: RegularGrid,
                              collided_minimum_diameter_m: float, obstacle_height_m: float) -> RockObstacles:
    """Rocks with a collider whose exposed height exceeds ``obstacle_height_m``.

    Exposed height = highest vertex of the placed detailed mesh (the convex-hull collider has the same top) minus the
    collider surface under the rock centre; radius = largest horizontal distance of the mesh vertices from the centre."""
    from lunar_rover_sim.core.rocks.rock_placement import REGION_WORKSPACE, RockTable
    from lunar_rover_sim.core.rocks.rock_shape_generator import prototype_mesh
    table = RockTable.from_arrays(rock_arrays)
    candidates = np.flatnonzero((table.region == REGION_WORKSPACE) & (table.diameter_m >= collided_minimum_diameter_m))
    rows = []
    for index in candidates:
        vertices, _ = prototype_mesh(rock_arrays, int(table.prototype_index[index]), "detailed")
        rotation = rotation_matrix_from_quaternion(table.orientation_xyzw[index])
        placed = (np.asarray(vertices, dtype=np.float64) * float(table.diameter_m[index])) @ rotation.T + \
            table.position_m[index]
        ground, valid = triangulated_height(collision_height_m, collision_grid, table.position_m[index, 0:1],
                                            table.position_m[index, 1:2])
        if not valid[0]:
            continue
        exposed = float(placed[:, 2].max() - ground[0])
        if exposed <= obstacle_height_m:
            continue
        radius = float(np.hypot(placed[:, 0] - table.position_m[index, 0], placed[:, 1] - table.position_m[index, 1]).max())
        rows.append((table.position_m[index, 0], table.position_m[index, 1], radius, exposed, index))
    if not rows:
        return RockObstacles.empty()
    columns = list(zip(*rows))
    return RockObstacles(*(np.asarray(column, dtype=np.float64) for column in columns[:4]),
                         np.asarray(columns[4], dtype=np.int64))


def distance_to_rock_discs(east: np.ndarray, north: np.ndarray, obstacles: RockObstacles) -> np.ndarray:
    """Distance (m) from each point to the nearest obstacle disc edge (negative inside), inf without obstacles."""
    distance = np.full(east.shape, np.inf)
    for center_east, center_north, radius in zip(obstacles.center_east_m, obstacles.center_north_m, obstacles.radius_m):
        distance = np.minimum(distance, np.hypot(east - center_east, north - center_north) - radius)
    return distance


def build_planner_map(collision_height_m: np.ndarray, collision_grid: RegularGrid, drivable_half_size_m: float,
                      rock_obstacles: RockObstacles, crater_table: CraterTable | None,
                      configuration: ExpertConfiguration, footprint: RoverFootprint) -> PlannerMap:
    """Cost, impassable cells and regions on the planner lattice (the collision lattice subsampled).

    Impassable: slope (maximum over the planner cell) above the threshold, rock discs and the bowls of craters at
    least ``crater_obstacle_minimum_diameter_m`` wide, both grown by half the rover width plus the side clearance
    (v1.0.5; before, bowls were not grown and the rover centre could graze a rim crest), and cells outside the
    drivable square. ``clearance_ring`` marks the preferred extra margin around rocks and bowls (closed when possible)."""
    decimation = int(round(configuration.planner_spacing_m / collision_grid.spacing_m))
    if decimation < 1 or abs(decimation * collision_grid.spacing_m - configuration.planner_spacing_m) > 1e-9:
        raise ValueError(f"planner_spacing_m {configuration.planner_spacing_m} is not a multiple of the collision "
                         f"spacing {collision_grid.spacing_m}")
    grid = collision_grid.subsampled(decimation)
    fine_slope = slope_over_baseline_deg(collision_height_m, collision_grid.spacing_m, configuration.slope_baseline_m)
    slope = ndimage.maximum_filter(fine_slope, size=decimation + 1, mode="nearest")[::decimation, ::decimation]
    east, north = grid.coordinate_mesh()

    impassable = slope > configuration.slope_threshold_deg
    impassable |= (np.abs(east) > drivable_half_size_m) | (np.abs(north) > drivable_half_size_m)
    hard_clearance = footprint.width_m / 2 + configuration.route_side_clearance_m
    rock_distance = distance_to_rock_discs(east, north, rock_obstacles)
    impassable |= rock_distance <= hard_clearance
    crater_obstacles = CraterObstacles.from_table(crater_table, configuration.crater_obstacle_minimum_diameter_m)
    crater_distance = crater_obstacles.bowl_edge_distance_m(east, north)
    impassable |= crater_distance <= hard_clearance
    obstacle_distance = np.minimum(rock_distance, crater_distance)
    clearance_ring = (obstacle_distance > hard_clearance) & \
        (obstacle_distance <= hard_clearance + configuration.preferred_clearance_m)

    cost = slope_cost(slope, configuration)
    cost[impassable] = 2.0 * configuration.slope_threshold_deg
    cost = ndimage.uniform_filter(cost, size=3, mode="nearest")
    regions, _ = ndimage.label(~impassable, structure=FOUR_CONNECTED)
    return PlannerMap(grid, slope.astype(np.float32), cost, impassable, clearance_ring & ~impassable, regions,
                      rock_obstacles, hard_clearance, drivable_half_size_m, crater_obstacles)


def planner_map_from_scene(scene, configuration: ExpertConfiguration | None = None) -> PlannerMap:
    """Planner map of a loaded scene (``LoadedScene``), with its rocks and craters."""
    configuration = configuration or scene.configuration.expert
    rocks = scene.configuration.rocks
    obstacles = RockObstacles.empty()
    if "rocks" in scene.extra_arrays:
        obstacles = rock_obstacles_from_table(scene.extra_arrays["rocks"], scene.workspace_collision_height_m,
                                              scene.layout.workspace_collision_grid, rocks.cobble_minimum_diameter_m,
                                              configuration.rock_obstacle_height_m)
    return build_planner_map(scene.workspace_collision_height_m, scene.layout.workspace_collision_grid,
                             scene.layout.drivable_half_size_m, obstacles, scene.crater_table, configuration,
                             RoverFootprint.from_rover(scene.configuration.rover))
