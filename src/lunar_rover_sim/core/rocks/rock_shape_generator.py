"""Procedural rock shapes (SYNTHETIC; DESIGN section 9.3): watertight by construction, no boolean operations.

A rock is a radial function on unit directions s:
    r(s) = min( r_ellipsoid(s), min_k h_k / max(s . n_k, eps) ) * (1 + sum_j a_j cos(f_j (s . d_j) + phase_j))
i.e. an ellipsoid (axis ratios from impact-fragment statistics, roughly 2 : sqrt(2) : 1) cut by K random fracture
planes (many for angular blocks, few for rounded ones) with a little multi-octave surface noise. Tessellated on an
icosphere and rescaled so the longest (x) extent is exactly 1 m; instances are scaled to their diameter.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from lunar_rover_sim.core.seeding import named_generator

SHAPE_CLASSES = ("angular", "subangular", "rounded", "slab")
SHAPE_CLASS_PROBABILITIES = (0.35, 0.35, 0.2, 0.1)
ALBEDO_CLASSES = ("dark", "medium", "bright")
ALBEDO_CLASS_PROBABILITIES = (0.25, 0.5, 0.25)
PLANE_COUNT_RANGE = {"angular": (6, 10), "subangular": (4, 6), "rounded": (2, 3), "slab": (3, 5)}
SMOOTHING_PASSES = {"angular": 0, "subangular": 1, "rounded": 3, "slab": 1}


@dataclass
class RockPrototype:
    """One rock shape at two levels of detail, longest axis 1 m, origin at the ellipsoid centre."""

    shape_class: str
    albedo_class: str
    vertices_detailed: np.ndarray
    faces_detailed: np.ndarray
    vertices_coarse: np.ndarray
    faces_coarse: np.ndarray

    @property
    def minimum_z_m(self) -> float:
        """Lowest vertex z of the unit-size shape."""
        return float(self.vertices_detailed[:, 2].min())

    @property
    def height_m(self) -> float:
        """Vertical extent of the unit-size shape."""
        return float(np.ptp(self.vertices_detailed[:, 2]))

    @property
    def footprint_radius_m(self) -> float:
        """Radius of the horizontal footprint of the unit-size shape (max horizontal vertex distance)."""
        return float(np.hypot(self.vertices_detailed[:, 0], self.vertices_detailed[:, 1]).max())


def icosphere(subdivisions: int) -> tuple[np.ndarray, np.ndarray]:
    """Unit icosphere: vertices (N,3) float64, faces (M,3) int32 counter-clockwise from outside."""
    golden = (1 + 5 ** 0.5) / 2
    vertices = [(-1, golden, 0), (1, golden, 0), (-1, -golden, 0), (1, -golden, 0), (0, -1, golden), (0, 1, golden),
                (0, -1, -golden), (0, 1, -golden), (golden, 0, -1), (golden, 0, 1), (-golden, 0, -1), (-golden, 0, 1)]
    faces = [(0, 11, 5), (0, 5, 1), (0, 1, 7), (0, 7, 10), (0, 10, 11), (1, 5, 9), (5, 11, 4), (11, 10, 2), (10, 7, 6),
             (7, 1, 8), (3, 9, 4), (3, 4, 2), (3, 2, 6), (3, 6, 8), (3, 8, 9), (4, 9, 5), (2, 4, 11), (6, 2, 10),
             (8, 6, 7), (9, 8, 1)]
    vertices = [np.array(v, dtype=np.float64) / np.linalg.norm(v) for v in vertices]
    for _ in range(subdivisions):
        cache: dict[tuple[int, int], int] = {}

        def midpoint(first: int, second: int) -> int:
            """Index of the (cached) normalised midpoint of an edge."""
            key = (min(first, second), max(first, second))
            if key not in cache:
                point = vertices[first] + vertices[second]
                vertices.append(point / np.linalg.norm(point))
                cache[key] = len(vertices) - 1
            return cache[key]

        new_faces = []
        for a, b, c in faces:
            ab, bc, ca = midpoint(a, b), midpoint(b, c), midpoint(c, a)
            new_faces += [(a, ab, ca), (b, bc, ab), (c, ca, bc), (ab, bc, ca)]
        faces = new_faces
    return np.array(vertices), np.array(faces, dtype=np.int32)


def sample_rock_shape(shape_class: str, generator: np.random.Generator) -> dict:
    """Random parameters of one shape: axis lengths, fracture planes, surface-noise octaves."""
    a = 1.0
    b = float(np.clip(generator.normal(0.71, 0.10), 0.5, 0.95))
    c = float(generator.uniform(0.25, 0.35)) if shape_class == "slab" else float(np.clip(generator.normal(0.50, 0.10), 0.3, 0.8))
    c = min(c, b)
    plane_low, plane_high = PLANE_COUNT_RANGE[shape_class]
    count = int(generator.integers(plane_low, plane_high + 1))
    normals = generator.standard_normal((count, 3))
    normals /= np.linalg.norm(normals, axis=1, keepdims=True)
    if shape_class == "slab":
        normals = np.vstack([normals, [0.0, 0.0, 1.0]])
    ellipsoid_radius = 1.0 / np.sqrt((normals[:, 0] / a) ** 2 + (normals[:, 1] / b) ** 2 + (normals[:, 2] / c) ** 2)
    offsets = ellipsoid_radius * generator.uniform(0.6, 0.9, len(normals))
    noise_directions = generator.standard_normal((3, 3))
    noise_directions /= np.linalg.norm(noise_directions, axis=1, keepdims=True)
    total_amplitude = generator.uniform(0.02, 0.08)
    return {"axes": (a, b, c), "plane_normals": normals, "plane_offsets": offsets,
            "noise_directions": noise_directions, "noise_frequencies": np.array([3.0, 7.0, 15.0]),
            "noise_amplitudes": total_amplitude * np.array([0.6, 0.3, 0.1]),
            "noise_phases": generator.uniform(0, 2 * np.pi, 3), "smoothing_passes": SMOOTHING_PASSES[shape_class]}


def _laplacian_smooth(vertices: np.ndarray, faces: np.ndarray, passes: int) -> np.ndarray:
    """Move each vertex half-way to the mean of its neighbours, ``passes`` times."""
    if passes == 0:
        return vertices
    edges = np.vstack([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    edges = np.vstack([edges, edges[:, ::-1]])
    counts = np.bincount(edges[:, 0], minlength=len(vertices))[:, None]
    for _ in range(passes):
        sums = np.zeros_like(vertices)
        np.add.at(sums, edges[:, 0], vertices[edges[:, 1]])
        vertices = 0.5 * vertices + 0.5 * sums / counts
    return vertices


def tessellate_rock_shape(parameters: dict, subdivisions: int) -> tuple[np.ndarray, np.ndarray]:
    """Vertices (float32, longest x extent = 1 m) and faces (int32) of a shape on an icosphere of ``subdivisions``."""
    directions, faces = icosphere(subdivisions)
    a, b, c = parameters["axes"]
    ellipsoid = 1.0 / np.sqrt((directions[:, 0] / a) ** 2 + (directions[:, 1] / b) ** 2 + (directions[:, 2] / c) ** 2)
    projection = directions @ parameters["plane_normals"].T
    plane_radius = np.where(projection > 1e-6, parameters["plane_offsets"][None, :] / np.maximum(projection, 1e-6), np.inf)
    radius = np.minimum(ellipsoid, plane_radius.min(axis=1))
    noise = np.ones(len(directions))
    for direction, frequency, amplitude, phase in zip(parameters["noise_directions"], parameters["noise_frequencies"],
                                                      parameters["noise_amplitudes"], parameters["noise_phases"]):
        noise += amplitude * np.cos(frequency * (directions @ direction) + phase)
    vertices = directions * (radius * noise)[:, None]
    vertices = _laplacian_smooth(vertices, faces, parameters["smoothing_passes"])
    vertices /= np.ptp(vertices[:, 0])
    return vertices.astype(np.float32), faces


def build_rock_prototype_library(prototypes_per_shape_class: int, seed: int) -> list[RockPrototype]:
    """``prototypes_per_shape_class`` prototypes for each of the four shape classes (deterministic per seed)."""
    shape_generator = named_generator(seed, "rocks.prototypes")
    albedo_generator = named_generator(seed, "rocks.albedo")
    library = []
    for shape_class in SHAPE_CLASSES:
        for _ in range(prototypes_per_shape_class):
            parameters = sample_rock_shape(shape_class, shape_generator)
            detailed_vertices, detailed_faces = tessellate_rock_shape(parameters, 3)
            coarse_vertices, coarse_faces = tessellate_rock_shape(parameters, 1)
            albedo = ALBEDO_CLASSES[int(albedo_generator.choice(3, p=ALBEDO_CLASS_PROBABILITIES))]
            library.append(RockPrototype(shape_class, albedo, detailed_vertices, detailed_faces, coarse_vertices,
                                         coarse_faces))
    return library


def prototype_library_arrays(library: list[RockPrototype]) -> dict[str, np.ndarray]:
    """Flatten a library into npz arrays (vertices/faces concatenated with offsets)."""
    arrays: dict[str, np.ndarray] = {
        "shape_class": np.array([SHAPE_CLASSES.index(p.shape_class) for p in library], dtype=np.int8),
        "albedo_class": np.array([ALBEDO_CLASSES.index(p.albedo_class) for p in library], dtype=np.int8)}
    for level in ("detailed", "coarse"):
        vertices = [getattr(p, f"vertices_{level}") for p in library]
        faces = [getattr(p, f"faces_{level}") for p in library]
        arrays[f"{level}_vertices"] = np.concatenate(vertices)
        arrays[f"{level}_faces"] = np.concatenate(faces)
        arrays[f"{level}_vertex_offsets"] = np.cumsum([0] + [len(v) for v in vertices])
        arrays[f"{level}_face_offsets"] = np.cumsum([0] + [len(f) for f in faces])
    return arrays


def prototype_mesh(arrays: dict, index: int, level: str = "detailed") -> tuple[np.ndarray, np.ndarray]:
    """(vertices, faces) of one prototype from ``prototype_library_arrays`` output."""
    vertex_offsets, face_offsets = arrays[f"{level}_vertex_offsets"], arrays[f"{level}_face_offsets"]
    vertices = arrays[f"{level}_vertices"][vertex_offsets[index]:vertex_offsets[index + 1]]
    faces = arrays[f"{level}_faces"][face_offsets[index]:face_offsets[index + 1]]
    return vertices, faces
