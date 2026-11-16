"""Triangle soups of the rendered scene, and ground-class rasters (crater, rim), for ray-cast ground truth.

The soups mirror exactly what ``scene_stage_builder`` and ``rock_stage_builder`` author for rendering:
  workspace   visual lattice (a, b, d)/(a, d, c) triangles; the live track deformation is applied by the caller
  near ring, regional, backdrop   their lattices with the inner hole removed (cells_kept)
  rocks       detailed prototype mesh for collided rocks (workspace, d >= cobble minimum), coarse mesh for pebbles and
              near-ring rocks; vertex = position + R(q) (diameter x prototype vertex) (translate, orient, scale)
Every face carries an object kind (terrain or rock) and an object index (tier number or rock table row).
In ``lunar_regolith_scene`` appearance mode the rendered workspace is the upstream render mesh, not this lattice:
masks there are an APPROXIMATION (the difference is the upstream micro-relief and imprints).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from lunar_rover_sim.core.data_model.regular_grid import RegularGrid, lattice_triangle_indices
from lunar_rover_sim.core.frames.pose_math import rotation_matrix_from_quaternion
from lunar_rover_sim.core.ground_truth.mask_classes import OBJECT_ID
from lunar_rover_sim.core.terrain.crater_population import CraterTable
from lunar_rover_sim.core.terrain.crater_stamping import normalised_crater_radius

OBJECT_KIND_NONE = 0
OBJECT_KIND_TERRAIN = 1
OBJECT_KIND_ROCK = 2
OBJECT_KIND_ROVER = 3
TERRAIN_TIER_NAMES = ("workspace", "near_ring", "regional", "backdrop")


@dataclass
class TriangleSoup:
    """Triangles with a per-face object label."""

    vertices: np.ndarray        # (V, 3) float32, world metres
    faces: np.ndarray           # (F, 3) int32
    face_kind: np.ndarray       # (F,) int8, OBJECT_KIND_*
    face_index: np.ndarray      # (F,) int32, tier number or rock table row

    @classmethod
    def empty(cls) -> "TriangleSoup":
        """No triangles."""
        return cls(np.zeros((0, 3), np.float32), np.zeros((0, 3), np.int32), np.zeros(0, np.int8), np.zeros(0, np.int32))

    @property
    def face_count(self) -> int:
        """Number of triangles."""
        return len(self.faces)


def concatenate_soups(soups: list[TriangleSoup]) -> TriangleSoup:
    """One soup from several (face indices offset)."""
    soups = [soup for soup in soups if soup.face_count]
    if not soups:
        return TriangleSoup.empty()
    offsets = np.cumsum([0] + [len(soup.vertices) for soup in soups[:-1]])
    return TriangleSoup(np.concatenate([soup.vertices for soup in soups]).astype(np.float32),
                        np.concatenate([soup.faces + offset for soup, offset in zip(soups, offsets)]).astype(np.int32),
                        np.concatenate([soup.face_kind for soup in soups]).astype(np.int8),
                        np.concatenate([soup.face_index for soup in soups]).astype(np.int32))


def lattice_soup(grid: RegularGrid, heights: np.ndarray, cells_kept: np.ndarray | None, tier_number: int) -> TriangleSoup:
    """Terrain triangles of one tier, exactly as rendered (cells outside ``cells_kept`` removed)."""
    vertices = grid.vertex_positions(heights)
    faces = lattice_triangle_indices(grid.count_north, grid.count_east)
    if cells_kept is not None:
        faces = faces.reshape(-1, 2, 3)[np.asarray(cells_kept, dtype=bool).ravel()].reshape(-1, 3)
    return TriangleSoup(vertices, faces, np.full(len(faces), OBJECT_KIND_TERRAIN, np.int8),
                        np.full(len(faces), tier_number, np.int32))


def outer_terrain_soup(scene) -> TriangleSoup:
    """Near ring, regional and backdrop tiers of a ``LoadedScene`` (the ones that exist)."""
    layout, arrays = scene.layout, scene.terrain_arrays
    soups = [lattice_soup(layout.near_ring_grid, arrays["near_ring_height_m"], arrays.get("near_ring_cells_kept"), 1)]
    for number, (key, grid) in ((2, ("regional", layout.regional_grid)), (3, ("backdrop", layout.backdrop_grid))):
        if grid is not None and f"{key}_height_m" in arrays:
            soups.append(lattice_soup(grid, arrays[f"{key}_height_m"], arrays.get(f"{key}_cells_kept"), number))
    return concatenate_soups(soups)


def rock_soup(rock_arrays: dict, cobble_minimum_diameter_m: float) -> TriangleSoup:
    """Every rendered rock with the mesh level the stage uses for it."""
    from lunar_rover_sim.core.rocks.rock_placement import REGION_WORKSPACE, RockTable
    from lunar_rover_sim.core.rocks.rock_shape_generator import prototype_mesh
    table = RockTable.from_arrays(rock_arrays)
    collided = (table.region == REGION_WORKSPACE) & (table.diameter_m >= cobble_minimum_diameter_m)
    soups = []
    for level, selected in (("detailed", collided), ("coarse", ~collided)):
        indices = np.flatnonzero(selected)
        for prototype in np.unique(table.prototype_index[indices]):
            members = indices[table.prototype_index[indices] == prototype]
            vertices, faces = prototype_mesh(rock_arrays, int(prototype), level)
            vertices = np.asarray(vertices, dtype=np.float64)
            rotations = np.stack([rotation_matrix_from_quaternion(q) for q in table.orientation_xyzw[members]])
            placed = np.einsum("kij,vj->kvi", rotations, vertices) * table.diameter_m[members, None, None] + \
                table.position_m[members, None, :]
            count = len(members)
            vertex_count = len(vertices)
            all_faces = (faces[None, :, :] + vertex_count * np.arange(count)[:, None, None]).reshape(-1, 3)
            soups.append(TriangleSoup(placed.reshape(-1, 3).astype(np.float32), all_faces.astype(np.int32),
                                      np.full(len(all_faces), OBJECT_KIND_ROCK, np.int8),
                                      np.repeat(members.astype(np.int32), len(faces))))
    return concatenate_soups(soups)


def static_scene_soup(scene, include_rocks: bool) -> TriangleSoup:
    """Everything rendered that does not move or deform: outer terrain tiers and rocks."""
    soups = [outer_terrain_soup(scene)]
    if include_rocks and "rocks" in scene.extra_arrays:
        soups.append(rock_soup(scene.extra_arrays["rocks"], scene.configuration.rocks.cobble_minimum_diameter_m))
    return concatenate_soups(soups)


def crater_class_raster(grid: RegularGrid, table: CraterTable, rim_width_fraction: float,
                        minimum_diameter_m: float) -> tuple[np.ndarray, np.ndarray]:
    """(class uint8, crater row + 1 int32) on a lattice: crater (r' <= 1), crater_rim (1 < r' <= 1 + w), else 0.

    r' is the stamping's normalised elliptical radius with rim harmonics, so the crest is where the bowl profile
    meets the apron. Craters are painted largest first, so smaller craters win where they overlap."""
    classes = np.zeros(grid.shape, dtype=np.uint8)
    crater_index = np.zeros(grid.shape, dtype=np.int32)
    order = np.argsort(-table.diameter_m)
    for index in order:
        if table.diameter_m[index] < minimum_diameter_m:
            continue
        reach = 1.5 * table.diameter_m[index] / 2 * (1 + rim_width_fraction) / max(float(table.ellipticity[index]), 0.5)
        row_low, column_low = grid.fractional_index(table.center_east_m[index] - reach, table.center_north_m[index] - reach)
        row_high, column_high = grid.fractional_index(table.center_east_m[index] + reach, table.center_north_m[index] + reach)
        rows = slice(max(int(np.floor(row_low)), 0), min(int(np.ceil(row_high)) + 1, grid.count_north))
        columns = slice(max(int(np.floor(column_low)), 0), min(int(np.ceil(column_high)) + 1, grid.count_east))
        if rows.start >= rows.stop or columns.start >= columns.stop:
            continue
        east = grid.origin_east_m + np.arange(columns.start, columns.stop) * grid.spacing_m
        north = grid.origin_north_m + np.arange(rows.start, rows.stop) * grid.spacing_m
        east_mesh, north_mesh = np.meshgrid(east, north)
        normalised = normalised_crater_radius(east_mesh, north_mesh, table, int(index))
        rim = (normalised > 1.0) & (normalised <= 1.0 + rim_width_fraction)
        bowl = normalised <= 1.0
        classes[rows, columns][rim] = OBJECT_ID["crater_rim"]
        classes[rows, columns][bowl] = OBJECT_ID["crater"]
        crater_index[rows, columns][rim | bowl] = index + 1
    return classes, crater_index


def nearest_lattice_value(raster: np.ndarray, grid: RegularGrid, east_m: np.ndarray, north_m: np.ndarray,
                          outside_value=0) -> tuple[np.ndarray, np.ndarray]:
    """(values, inside mask): nearest-post values of a lattice raster at points; ``outside_value`` off the lattice."""
    row, column = grid.fractional_index(east_m, north_m)
    row_index, column_index = np.rint(row).astype(np.int64), np.rint(column).astype(np.int64)
    inside = (row_index >= 0) & (row_index < grid.count_north) & (column_index >= 0) & (column_index < grid.count_east)
    values = np.full(np.shape(east_m), outside_value, dtype=raster.dtype)
    values[inside] = raster[row_index[inside], column_index[inside]]
    return values, inside
