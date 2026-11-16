"""Ground-truth masks (Isaac-free): crater rasters, composition rules, and the Warp ray kernels on a toy scene."""
from __future__ import annotations

import math

import numpy as np
import pytest

from lunar_rover_sim.core.data_model.regular_grid import RegularGrid
from lunar_rover_sim.core.frames.pose_math import sun_direction_w
from lunar_rover_sim.core.ground_truth.mask_classes import (OBJECT_CLASSES, OBJECT_ID, SHADOW_CAUSE_ID, SHADOW_CLASSES,
                                                            SHADOW_ID, colorize, legend)
from lunar_rover_sim.core.ground_truth.mask_composition import RayCastResult, compose_ground_truth, sky_classes
from lunar_rover_sim.core.ground_truth.scene_geometry import (OBJECT_KIND_ROCK, OBJECT_KIND_ROVER, OBJECT_KIND_TERRAIN,
                                                              crater_class_raster, lattice_soup, nearest_lattice_value)
from lunar_rover_sim.core.terrain.crater_population import HARMONIC_ORDERS, CraterTable


def round_craters(centers_and_diameters) -> CraterTable:
    """Circular craters without rim harmonics."""
    count = len(centers_and_diameters)
    columns = np.asarray(centers_and_diameters, dtype=np.float64)
    return CraterTable(center_east_m=columns[:, 0], center_north_m=columns[:, 1], diameter_m=columns[:, 2],
                       fresh_depth_to_diameter=np.full(count, 0.15), degradation_parameter=np.zeros(count),
                       ellipticity=np.ones(count), orientation_rad=np.zeros(count),
                       harmonic_amplitudes=np.zeros((count, len(HARMONIC_ORDERS))),
                       harmonic_phases_rad=np.zeros((count, len(HARMONIC_ORDERS))), band=np.zeros(count, dtype=np.int8),
                       is_feature_crater=np.zeros(count, dtype=bool))


def test_class_tables_follow_mare_ids():
    """MARE ids: rover 2, terrain (landscape) 5, crater 6, rim 7, track 8, boulder 9; shadow rover_shadow 5."""
    assert [OBJECT_ID[name] for name in ("sky", "rover", "terrain", "crater", "crater_rim", "wheel_track", "boulder",
                                         "small_rock")] == [1, 2, 5, 6, 7, 8, 9, 10]
    assert SHADOW_ID["rover_shadow"] == 5 and SHADOW_ID["shadowed_object"] == 4
    assert len(legend()["object_mask"]) == len(OBJECT_CLASSES)
    image = colorize(np.array([[5, 2]]), OBJECT_CLASSES)
    assert image[0, 0].tolist() == list(OBJECT_CLASSES[5].rgb)
    assert colorize(np.array([[5]]), SHADOW_CLASSES).shape == (1, 1, 3)


def test_crater_raster_bowl_rim_and_overlap():
    """Bowl inside the crest, rim out to (1 + w) R, nothing beyond; a small crater wins inside a large one."""
    grid = RegularGrid.centered(12.0, 0.05)
    table = round_craters([(0.0, 0.0, 6.0), (2.0, 0.0, 1.0)])
    classes, crater_index = crater_class_raster(grid, table, 0.2, 0.0)
    values, _ = nearest_lattice_value(classes, grid, np.array([-1.0, -3.3, -3.8, 2.0, 2.55]), np.zeros(5))
    assert values.tolist() == [OBJECT_ID["crater"], OBJECT_ID["crater_rim"], 0, OBJECT_ID["crater"], OBJECT_ID["crater_rim"]]
    index, _ = nearest_lattice_value(crater_index, grid, np.array([2.0]), np.array([0.0]))
    assert index[0] == 2
    small_skipped, _ = crater_class_raster(grid, table, 0.2, 2.0)
    assert nearest_lattice_value(small_skipped, grid, np.array([2.0]), np.array([0.0]))[0][0] == OBJECT_ID["crater"]


def test_sky_classes_sun_disc_and_glare():
    """Rays at 0.1, 0.8 and 5 deg from the Sun are sun, sun_glare and sky."""
    sun = np.array([1.0, 0.0, 0.0])
    angles = np.radians([0.1, 0.8, 5.0])
    rays = np.column_stack([np.cos(angles), np.sin(angles), np.zeros(3)])
    assert sky_classes(rays, sun, 0.265, 1.1).tolist() == [OBJECT_ID["sun"], OBJECT_ID["sun_glare"], OBJECT_ID["sky"]]


def test_composition_rules():
    """Ground lit / shadowed / rover shadow, rover never rover_shadow, rocks split by diameter, unlit facets shadowed."""
    sun = np.array([0.0, 0.0, 1.0])
    down = np.array([0.0, 0.0, -1.0])
    up = np.array([0.0, 0.0, 1.0])
    rows = [  # kind, index, normal, blocked world, blocked rover
        (OBJECT_KIND_TERRAIN, 0, up, False, False),
        (OBJECT_KIND_TERRAIN, 0, up, True, False),
        (OBJECT_KIND_TERRAIN, 0, up, True, True),
        (OBJECT_KIND_ROVER, -1, up, False, True),
        (OBJECT_KIND_ROCK, 0, up, False, False),
        (OBJECT_KIND_ROCK, 1, up, False, True),
        (OBJECT_KIND_TERRAIN, 0, np.array([1.0, 0.0, -0.2]) / math.hypot(1, 0.2), False, False),
        (0, -1, up, False, False),
    ]
    count = len(rows)
    rays = np.tile(down, (count, 1))
    rays[6] = np.array([-1.0, 0.0, 0.0])            # horizontal view onto a facet tilted away from the Sun
    rays[-1] = sun
    result = RayCastResult(np.array([r[0] for r in rows], np.int8), np.array([r[1] for r in rows], np.int32),
                           np.zeros((count, 3)), np.array([r[2] for r in rows]), rays, np.ones(count),
                           np.array([r[3] for r in rows]), np.array([r[4] for r in rows]))
    ground = np.full(count, OBJECT_ID["terrain"], np.uint8)
    ground[1] = OBJECT_ID["crater"]
    object_ids, shadow_ids, instance, cause = compose_ground_truth(result, ground, np.array([0.5, 0.1]), sun, 0.265, 1.1, 0.3)
    assert object_ids.tolist() == [5, 6, 5, 2, 9, 10, 5, 3]
    assert shadow_ids.tolist() == [SHADOW_ID[name] for name in (
        "sunlit_ground", "shadowed_ground", "rover_shadow", "shadowed_object", "sunlit_object", "rover_shadow",
        "shadowed_ground", "sky")]
    assert instance.tolist() == [0, 0, 0, 0, 1, 2, 0, 0]
    assert cause.tolist() == [SHADOW_CAUSE_ID[name] for name in (
        "sunlit", "cast_by_scene", "cast_by_rover", "cast_by_rover", "sunlit", "cast_by_rover", "faces_away", "sky")]


def test_lattice_soup_matches_stage_triangulation():
    """A 2 x 2 lattice gives 2 triangles a-b-d / a-d-c; cells_kept removes both triangles of a cell."""
    grid = RegularGrid(0.0, 0.0, 1.0, 3, 3)
    heights = np.arange(9, dtype=np.float32).reshape(3, 3)
    soup = lattice_soup(grid, heights, None, 0)
    assert soup.face_count == 8 and soup.faces[:2].tolist() == [[0, 1, 4], [0, 4, 3]]
    kept = np.array([[True, False], [True, True]])
    assert lattice_soup(grid, heights, kept, 1).face_count == 6


def test_warp_shadow_of_a_box_on_a_plane():
    """Toy scene in the Warp kernels: a 1 m cube on a plane, Sun at 45 deg elevation from the east.

    Looking straight down, ground just west of the cube is in its shadow (out to 1 m), ground east of it is lit."""
    warp = pytest.importorskip("warp")
    warp.init()
    if not warp.is_cuda_available():
        pytest.skip("no CUDA device")
    from lunar_rover_sim.runtime.sensors.ground_truth_masks import _camera_ray_kernel, _shadow_ray_kernel
    plane = np.array([[-10, -10, 0], [10, -10, 0], [10, 10, 0], [-10, 10, 0]], np.float32)
    plane_faces = np.array([[0, 1, 2], [0, 2, 3]], np.int32)
    cube = np.array([[x, y, z] for x in (-0.5, 0.5) for y in (-0.5, 0.5) for z in (0.0, 1.0)], np.float32)
    cube_faces = np.array([[0, 1, 3], [0, 3, 2], [4, 6, 7], [4, 7, 5], [0, 4, 5], [0, 5, 1],
                           [2, 3, 7], [2, 7, 6], [0, 2, 6], [0, 6, 4], [1, 5, 7], [1, 7, 3]], np.int32)

    def mesh(vertices, faces):
        """Warp mesh on the GPU."""
        return warp.Mesh(points=warp.array(vertices, dtype=warp.vec3, device="cuda:0"),
                         indices=warp.array(faces.ravel(), dtype=warp.int32, device="cuda:0"), bvh_constructor="sah")

    static, workspace, rover = mesh(plane, plane_faces), mesh(cube, cube_faces), \
        mesh(np.array([[0, 0, -50], [0.01, 0, -50], [0, 0.01, -50]], np.float32), np.array([[0, 1, 2]], np.int32))
    targets = np.array([[-0.7, 0.0, 0.0], [-1.7, 0.0, 0.0], [0.8, 0.0, 0.0]])
    origin = np.array([0.0, 0.0, 10.0])
    directions = (targets - origin) / np.linalg.norm(targets - origin, axis=1, keepdims=True)
    count = len(directions)
    device = "cuda:0"
    directions_gpu = warp.array(directions.astype(np.float32), dtype=warp.vec3, device=device)
    hit_mesh = warp.empty(count, dtype=warp.int32, device=device)
    hit_face = warp.empty(count, dtype=warp.int32, device=device)
    hit_distance = warp.empty(count, dtype=warp.float32, device=device)
    hit_normal = warp.empty(count, dtype=warp.vec3, device=device)
    warp.launch(_camera_ray_kernel, dim=count, device=device,
                inputs=[warp.vec3(*origin), directions_gpu, warp.vec3(0.0, 0.0, -1.0), 0.05, static.id, workspace.id,
                        rover.id, 1.0e6],
                outputs=[hit_mesh, hit_face, hit_distance, hit_normal])
    assert hit_mesh.numpy().tolist() == [0, 0, 0]
    sun = sun_direction_w(90.0, 45.0)
    blocked_world = warp.empty(count, dtype=warp.int32, device=device)
    blocked_rover = warp.empty(count, dtype=warp.int32, device=device)
    warp.launch(_shadow_ray_kernel, dim=count, device=device,
                inputs=[warp.vec3(*origin), directions_gpu, hit_mesh, hit_distance, hit_normal, warp.vec3(*sun),
                        0.002, static.id, workspace.id, rover.id, 1.0e6],
                outputs=[blocked_world, blocked_rover])
    assert blocked_world.numpy().tolist() == [1, 0, 0]      # the 45 deg shadow of the cube spans x in [-1.5, -0.5]
    assert hit_distance.numpy()[0] == pytest.approx(np.linalg.norm(targets[0] - origin), abs=1e-4)
