"""Rock size-frequency law, placement, poses, burial and regolith fillets (SYNTHETIC; DESIGN section 9.3).

Size law, cumulative areal density of rocks with longest axis > D:
  background  N(>D) = N_pivot (D / 0.1 m)^-gamma          (N_pivot, gamma drawn per scene)
  boulder tail N(>D) = K_b D^-gamma_b,  K_b = (4/pi) RA (gamma_b - 2) / gamma_b   (RA = Diviner rock abundance)
cross-faded linearly in DIFFERENTIAL density between 0.5 m and 1 m, so >= 1 m rocks come from the tail only.
Positions: thinning against a log-normal intensity field (patchy rock cover) plus Thomas clusters for part of the
5-30 cm rocks; larger rocks first, overlap rejection, keep-out zones, slope limit for large rocks.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from lunar_rover_sim.core.configuration.configuration_models import RockConfiguration
from lunar_rover_sim.core.data_model.regular_grid import RegularGrid, triangulated_height, triangulated_normal
from lunar_rover_sim.core.frames.pose_math import quaternion_from_rotation_matrix
from lunar_rover_sim.core.regolith.regolith_synthesis import gaussian_random_field
from lunar_rover_sim.core.rocks.rock_shape_generator import SHAPE_CLASS_PROBABILITIES, SHAPE_CLASSES, RockPrototype
from lunar_rover_sim.core.seeding import named_generator

SIZE_CLASS_NAMES = ("pebble", "cobble", "small_rock", "medium_rock", "large_boulder")
REGION_WORKSPACE = 0
REGION_NEAR_RING = 1


@dataclass
class RockTable:
    """One row per placed rock."""

    position_m: np.ndarray
    orientation_xyzw: np.ndarray
    diameter_m: np.ndarray
    prototype_index: np.ndarray
    size_class: np.ndarray
    region: np.ndarray
    burial_fraction: np.ndarray
    cluster_id: np.ndarray
    instance_id: np.ndarray

    def __len__(self) -> int:
        """Number of rocks."""
        return len(self.diameter_m)

    def to_arrays(self) -> dict[str, np.ndarray]:
        """Arrays for np.savez (prefix ``rock_``)."""
        return {f"rock_{name}": value for name, value in self.__dict__.items()}

    @classmethod
    def from_arrays(cls, arrays) -> "RockTable":
        """Rebuild from ``to_arrays`` output."""
        return cls(**{name: np.asarray(arrays[f"rock_{name}"]) for name in cls.__dataclass_fields__})


def size_class_of(diameter_m: np.ndarray, rocks: RockConfiguration) -> np.ndarray:
    """0 pebble, 1 cobble, 2 small_rock, 3 medium_rock, 4 large_boulder."""
    edges = [rocks.cobble_minimum_diameter_m, rocks.registration_minimum_diameter_m, 0.3, 1.0]
    return np.digitize(diameter_m, edges).astype(np.int8)


@dataclass
class RockSizeLaw:
    """Blended background + boulder-tail size-frequency law (per m^2)."""

    density_at_pivot_per_m2: float
    exponent: float
    pivot_diameter_m: float
    boulder_coefficient: float
    boulder_exponent: float
    fade_start_m: float
    fade_end_m: float

    def differential_density(self, diameter_m: np.ndarray) -> np.ndarray:
        """dN/dD (rocks per m^2 per m of diameter)."""
        diameter = np.asarray(diameter_m, dtype=np.float64)
        background = self.exponent * self.density_at_pivot_per_m2 * self.pivot_diameter_m ** self.exponent * \
            diameter ** (-self.exponent - 1)
        tail = self.boulder_exponent * self.boulder_coefficient * diameter ** (-self.boulder_exponent - 1)
        weight = np.clip((self.fade_end_m - diameter) / (self.fade_end_m - self.fade_start_m), 0.0, 1.0)
        return weight * background + (1 - weight) * tail

    def expected_counts(self, edges_m: np.ndarray, area_m2: float) -> np.ndarray:
        """Expected number of rocks per diameter bin over an area (trapezoid in log space, 16 sub-steps per bin)."""
        counts = []
        for low, high in zip(edges_m[:-1], edges_m[1:]):
            samples = np.geomspace(low, high, 17)
            values = self.differential_density(samples)
            counts.append(np.trapezoid(values, samples) * area_m2)
        return np.array(counts)


def build_size_law(rocks: RockConfiguration, generator: np.random.Generator) -> RockSizeLaw:
    """Draw the scene's background density and exponent (jointly) and build the law."""
    density = generator.uniform(*rocks.background_density_at_pivot_range_per_m2) if rocks.randomize_background_density \
        else rocks.background_density_at_pivot_per_m2
    exponent = generator.uniform(*rocks.background_exponent_range) if rocks.randomize_background_exponent \
        else rocks.background_exponent
    gamma_b = rocks.boulder_exponent
    coefficient = (4 / math.pi) * rocks.boulder_rock_abundance * (gamma_b - 2) / gamma_b
    return RockSizeLaw(float(density), float(exponent), rocks.pivot_diameter_m, coefficient, gamma_b,
                       *rocks.background_fade_diameters_m)


class _OccupancyGrid:
    """Hash grid of placed rocks (>= 5 cm) for overlap rejection."""

    def __init__(self, cell_m: float):
        """Empty grid with square cells of ``cell_m``."""
        self.cell_m = cell_m
        self.cells: dict[tuple[int, int], list[tuple[float, float, float]]] = {}

    def overlaps(self, east: float, north: float, diameter: float, factor: float) -> bool:
        """True if a placed rock is closer than factor * (D1 + D2)."""
        reach = int(math.ceil((factor * (diameter + 3.0)) / self.cell_m))
        column, row = int(east // self.cell_m), int(north // self.cell_m)
        for d_row in range(-reach, reach + 1):
            for d_column in range(-reach, reach + 1):
                for other_east, other_north, other_diameter in self.cells.get((column + d_column, row + d_row), ()):
                    if math.hypot(east - other_east, north - other_north) < factor * (diameter + other_diameter):
                        return True
        return False

    def add(self, east: float, north: float, diameter: float) -> None:
        """Register a placed rock."""
        self.cells.setdefault((int(east // self.cell_m), int(north // self.cell_m)), []).append((east, north, diameter))


def _rock_pose(east: float, north: float, diameter: float, prototype: RockPrototype, heights: np.ndarray,
               grid: RegularGrid, rocks: RockConfiguration, generator: np.random.Generator):
    """Position (centre) and orientation of a rock resting partly buried on the surface."""
    yaw = generator.uniform(0, 2 * math.pi)
    normal = triangulated_normal(heights, grid, np.array([east]), np.array([north]))[0]
    up = np.array([0.0, 0.0, 1.0]) + normal
    up /= np.linalg.norm(up)
    axis = generator.standard_normal(2)
    axis = np.array([axis[0], axis[1], 0.0]) / max(np.linalg.norm(axis), 1e-9)
    tilt = math.radians(generator.normal(0.0, rocks.tilt_sigma_deg))
    cos_tilt, sin_tilt = math.cos(tilt), math.sin(tilt)
    up = up * cos_tilt + np.cross(axis, up) * sin_tilt + axis * np.dot(axis, up) * (1 - cos_tilt)
    forward = np.array([math.cos(yaw), math.sin(yaw), 0.0])
    forward -= np.dot(forward, up) * up
    forward /= np.linalg.norm(forward)
    rotation = np.column_stack([forward, np.cross(up, forward), up])
    footprint = 0.35 * diameter
    sample_east = east + footprint * np.array([0, 1, -1, 0, 0])
    sample_north = north + footprint * np.array([0, 0, 0, 1, -1])
    ground, valid = triangulated_height(heights, grid, sample_east, sample_north)
    ground_reference = float(np.nanmin(np.where(valid, ground, np.nan)))
    burial = generator.uniform(*rocks.burial_fraction_range)
    height = prototype.height_m * diameter
    center_z = ground_reference - burial * height - prototype.minimum_z_m * diameter
    return np.array([east, north, center_z]), quaternion_from_rotation_matrix(rotation), burial


def place_rocks(rocks: RockConfiguration, library: list[RockPrototype], workspace_grid: RegularGrid,
                workspace_heights: np.ndarray, workspace_skirt_m: float, near_grid: RegularGrid,
                near_heights: np.ndarray, seed: int) -> tuple[RockTable, dict]:
    """Place every rock of the scene. Returns the table and a report (law parameters, counts per class)."""
    law = build_size_law(rocks, named_generator(seed, "rocks.parameters"))
    size_generator = named_generator(seed, "rocks.sizes")
    position_generator = named_generator(seed, "rocks.positions")
    cluster_generator = named_generator(seed, "rocks.clusters")
    pose_generator = named_generator(seed, "rocks.poses")
    prototype_choice = named_generator(seed, "rocks.prototypes.choice")

    intensity_grid = RegularGrid.centered(near_grid.extent_east_m, 0.5)
    field = gaussian_random_field(intensity_grid.shape, 0.5, rocks.clustering_field_correlation_length_m,
                                  named_generator(seed, "rocks.intensity"))
    sigma = rocks.clustering_field_sigma
    intensity = np.exp(sigma * field - sigma ** 2 / 2)
    intensity_maximum = float(intensity.max())

    def intensity_at(east: np.ndarray, north: np.ndarray) -> np.ndarray:
        """Intensity at points (nearest lattice value)."""
        row, column = intensity_grid.fractional_index(east, north)
        row = np.clip(np.round(row).astype(int), 0, intensity_grid.count_north - 1)
        column = np.clip(np.round(column).astype(int), 0, intensity_grid.count_east - 1)
        return intensity[row, column]

    workspace_half = workspace_grid.extent_east_m / 2 - workspace_skirt_m
    near_half = near_grid.extent_east_m / 2 - 2.0
    regions = [(REGION_WORKSPACE, rocks.pebble_minimum_diameter_m, workspace_half, 0.0),
               (REGION_NEAR_RING, rocks.near_ring_minimum_diameter_m, near_half, workspace_grid.extent_east_m / 2)]
    edges = np.geomspace(rocks.pebble_minimum_diameter_m, rocks.maximum_diameter_m, 61)
    shapes_by_class = {name: [i for i, p in enumerate(library) if p.shape_class == name] for name in SHAPE_CLASSES}
    occupancy = _OccupancyGrid(1.0)
    rows = []
    parents = None
    for region, minimum_diameter, half, hole_half in regions:
        area = (2 * half) ** 2 - (2 * hole_half) ** 2
        counts = law.expected_counts(edges, area)
        for bin_index in range(len(edges) - 2, -1, -1):
            low, high = edges[bin_index], edges[bin_index + 1]
            if high <= minimum_diameter:
                continue
            count = int(size_generator.poisson(counts[bin_index]))
            diameters = np.exp(size_generator.uniform(math.log(max(low, minimum_diameter)), math.log(high), count))
            for diameter in diameters:
                clustered = region == REGION_WORKSPACE and 0.05 <= diameter <= 0.3 and \
                    cluster_generator.random() < rocks.thomas_cluster_fraction
                cluster_id = -1
                for _ in range(30):
                    if clustered:
                        if parents is None:
                            parent_count = max(1, int(cluster_generator.poisson(rocks.thomas_parent_density_per_m2 * area)))
                            parents = cluster_generator.uniform(-half, half, (parent_count, 2))
                        cluster_id = int(cluster_generator.integers(len(parents)))
                        east, north = parents[cluster_id] + cluster_generator.normal(0, rocks.thomas_child_sigma_m, 2)
                    else:
                        east, north = position_generator.uniform(-half, half, 2)
                        if position_generator.random() * intensity_maximum > intensity_at(np.array([east]), np.array([north]))[0]:
                            continue
                    if abs(east) > half or abs(north) > half or (abs(east) < hole_half and abs(north) < hole_half):
                        continue
                    if any(math.hypot(east - zone.center_east_m, north - zone.center_north_m) < zone.radius_m + diameter / 2
                           for zone in rocks.keep_out_zones):
                        continue
                    if diameter >= rocks.cobble_minimum_diameter_m and \
                            occupancy.overlaps(east, north, diameter, rocks.overlap_rejection_factor):
                        continue
                    grid, heights = (workspace_grid, workspace_heights) if region == REGION_WORKSPACE else (near_grid, near_heights)
                    if diameter >= 0.3:
                        normal = triangulated_normal(heights, grid, np.array([east]), np.array([north]))[0]
                        if math.degrees(math.acos(min(1.0, normal[2]))) > rocks.maximum_slope_for_large_rocks_deg:
                            continue
                    shape_class = SHAPE_CLASSES[int(prototype_choice.choice(4, p=SHAPE_CLASS_PROBABILITIES))]
                    prototype_index = int(prototype_choice.choice(shapes_by_class[shape_class]))
                    position, orientation, burial = _rock_pose(east, north, diameter, library[prototype_index],
                                                               heights, grid, rocks, pose_generator)
                    if diameter >= rocks.cobble_minimum_diameter_m:
                        occupancy.add(east, north, diameter)
                    rows.append((position, orientation, diameter, prototype_index, region, burial, cluster_id))
                    break
    position = np.array([row[0] for row in rows]).reshape(-1, 3)
    diameter = np.array([row[2] for row in rows])
    size_class = size_class_of(diameter, rocks)
    region_array = np.array([row[4] for row in rows], dtype=np.int8)
    registered = (diameter >= rocks.registration_minimum_diameter_m) & (region_array == REGION_WORKSPACE)
    instance_id = np.zeros(len(rows), dtype=np.int32)
    order = np.flatnonzero(registered)[np.argsort(-diameter[registered], kind="stable")]
    instance_id[order] = np.arange(1, len(order) + 1)
    table = RockTable(position, np.array([row[1] for row in rows]).reshape(-1, 4), diameter,
                      np.array([row[3] for row in rows], dtype=np.int32), size_class, region_array,
                      np.array([row[5] for row in rows]), np.array([row[6] for row in rows], dtype=np.int32), instance_id)
    report = {"background_density_at_pivot_per_m2": law.density_at_pivot_per_m2, "background_exponent": law.exponent,
              "boulder_coefficient": law.boulder_coefficient,
              "counts_by_size_class": {name: int(np.sum((size_class == index) & (region_array == REGION_WORKSPACE)))
                                       for index, name in enumerate(SIZE_CLASS_NAMES)},
              "near_ring_rock_count": int(np.sum(region_array == REGION_NEAR_RING)),
              "registered_rock_count": int(registered.sum()), "synthetic": True}
    return table, report


def rock_fillet_height_m(table: RockTable, library: list[RockPrototype], grid: RegularGrid,
                         rocks: RockConfiguration, seed: int) -> np.ndarray:
    """Ring-shaped regolith skirts around workspace rocks >= fillet_minimum_diameter_m (combined by maximum)."""
    generator = named_generator(seed, "rocks.fillets")
    fillet = np.zeros(grid.shape)
    east_coordinates, north_coordinates = grid.east_coordinates(), grid.north_coordinates()
    for index in np.flatnonzero((table.diameter_m >= rocks.fillet_minimum_diameter_m) & (table.region == REGION_WORKSPACE)):
        prototype = library[table.prototype_index[index]]
        diameter = table.diameter_m[index]
        footprint = 0.5 * prototype.footprint_radius_m * diameter + 0.25 * diameter
        sigma = 0.6 * footprint
        exposed = prototype.height_m * diameter * (1 - table.burial_fraction[index])
        amplitude = generator.uniform(*rocks.fillet_height_fraction_range) * exposed
        if prototype.shape_class == "rounded":
            amplitude *= 1.5
        east, north = table.position_m[index, 0], table.position_m[index, 1]
        reach = footprint + 3 * sigma
        columns = slice(max(0, int(np.searchsorted(east_coordinates, east - reach))),
                        int(np.searchsorted(east_coordinates, east + reach)))
        rows = slice(max(0, int(np.searchsorted(north_coordinates, north - reach))),
                     int(np.searchsorted(north_coordinates, north + reach)))
        if columns.start >= columns.stop or rows.start >= rows.stop:
            continue
        east_mesh, north_mesh = np.meshgrid(east_coordinates[columns], north_coordinates[rows])
        rho = np.hypot(east_mesh - east, north_mesh - north)
        skirt = amplitude * np.exp(-np.maximum(rho - footprint, 0.0) ** 2 / (2 * sigma ** 2))
        fillet[rows, columns] = np.maximum(fillet[rows, columns], skirt)
    return fillet


def rasterize_rock_footprints(table: RockTable, library: list[RockPrototype], grid: RegularGrid,
                              minimum_diameter_m: float, use_instance_id: bool) -> np.ndarray:
    """Raster on ``grid``: rock id (instance id, or row + 1) of workspace rocks >= minimum diameter covering a post."""
    raster = np.zeros(grid.shape, dtype=np.int32)
    east_coordinates, north_coordinates = grid.east_coordinates(), grid.north_coordinates()
    selected = np.flatnonzero((table.diameter_m >= minimum_diameter_m) & (table.region == REGION_WORKSPACE))
    for index in selected[np.argsort(-table.diameter_m[selected])]:
        radius = 0.5 * library[table.prototype_index[index]].footprint_radius_m * table.diameter_m[index] + 0.5 * table.diameter_m[index]
        radius *= 0.5
        east, north = table.position_m[index, 0], table.position_m[index, 1]
        columns = slice(int(np.searchsorted(east_coordinates, east - radius)), int(np.searchsorted(east_coordinates, east + radius)))
        rows = slice(int(np.searchsorted(north_coordinates, north - radius)), int(np.searchsorted(north_coordinates, north + radius)))
        if columns.start >= columns.stop or rows.start >= rows.stop:
            continue
        east_mesh, north_mesh = np.meshgrid(east_coordinates[columns], north_coordinates[rows])
        inside = np.hypot(east_mesh - east, north_mesh - north) <= radius
        raster[rows, columns][inside] = table.instance_id[index] if use_instance_id else index + 1
    return raster
