"""Compose object mask, shadow label and rock-instance image from per-pixel ray-cast results (MARE class rules).

Inputs come from a ray caster (runtime): for every pixel, the camera ray's closest hit (kind, object index, point,
geometric normal) and two shadow rays from the hit towards the Sun (blocked by terrain/rocks, blocked by the rover).
Rules (MARE masks.md):
  object   sky pixels are sun within the disc radius, sun_glare within the glare radius, else sky; rover pixels
           rover; rock pixels boulder or small_rock by diameter; ground pixels wheel_track > crater / crater_rim >
           terrain (ground class supplied by the caller from the track field and the crater rasters)
  shadow   a surface is shadowed when it faces away from the Sun or a shadow ray is blocked; a shadowed ground or rock
           pixel is rover_shadow when the rover blocks its shadow ray (the rover's shadow wins over any other); the
           rover itself is never rover_shadow (its unlit side is shadowed_object)
The geometric (flat-face) normal is used for "faces away"; the renderer shades with smooth vertex normals, so pixels
at grazing incidence (|n . sun| near 0) can differ.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from lunar_rover_sim.core.ground_truth.mask_classes import OBJECT_ID, SHADOW_ID
from lunar_rover_sim.core.ground_truth.scene_geometry import OBJECT_KIND_ROCK, OBJECT_KIND_ROVER, OBJECT_KIND_TERRAIN


@dataclass
class RayCastResult:
    """Per-pixel ray-cast data, flattened (N pixels)."""

    hit_kind: np.ndarray            # (N,) int8 OBJECT_KIND_*, 0 = no hit (sky)
    hit_index: np.ndarray           # (N,) int32 rock table row for rocks, tier for terrain, -1 otherwise
    hit_point_m: np.ndarray         # (N, 3) float64 world point (NaN for sky)
    hit_normal: np.ndarray          # (N, 3) float64 unit geometric normal (any orientation)
    ray_direction: np.ndarray       # (N, 3) float64 unit camera-ray direction
    hit_distance_m: np.ndarray      # (N,) float64 along the ray (inf for sky)
    shadow_by_world: np.ndarray     # (N,) bool shadow ray blocked by terrain or rocks
    shadow_by_rover: np.ndarray     # (N,) bool shadow ray blocked by the rover


@dataclass
class GroundTruthImages:
    """Label images of one camera frame (H, W)."""

    object_mask: np.ndarray         # uint8 OBJECT_CLASSES ids
    shadow_label: np.ndarray        # uint8 SHADOW_CLASSES ids
    rock_instance: np.ndarray       # int32 rock table row + 1, 0 elsewhere
    depth_m: np.ndarray             # float32 distance to the image plane (inf for sky)


def sky_classes(ray_direction: np.ndarray, sun_direction: np.ndarray, sun_disc_radius_deg: float,
                glare_radius_deg: float) -> np.ndarray:
    """Object ids of sky pixels: sun, sun_glare or sky by the angle between the ray and the Sun direction."""
    cosine = np.clip(ray_direction @ (np.asarray(sun_direction) / np.linalg.norm(sun_direction)), -1.0, 1.0)
    angle_deg = np.degrees(np.arccos(cosine))
    return np.where(angle_deg <= sun_disc_radius_deg, OBJECT_ID["sun"],
                    np.where(angle_deg <= glare_radius_deg, OBJECT_ID["sun_glare"], OBJECT_ID["sky"])).astype(np.uint8)


def compose_ground_truth(result: RayCastResult, ground_class: np.ndarray, rock_diameter_m: np.ndarray | None,
                         sun_direction: np.ndarray, sun_disc_radius_deg: float, glare_radius_deg: float,
                         boulder_minimum_diameter_m: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(object ids, shadow ids, rock instance) per pixel, flattened.

    ``ground_class``: object id per pixel for terrain hits (terrain, crater, crater_rim or wheel_track)."""
    count = len(result.hit_kind)
    object_ids = np.full(count, OBJECT_ID["unlabeled"], dtype=np.uint8)
    shadow_ids = np.full(count, SHADOW_ID["sky"], dtype=np.uint8)
    rock_instance = np.zeros(count, dtype=np.int32)

    sky = result.hit_kind == 0
    object_ids[sky] = sky_classes(result.ray_direction[sky], sun_direction, sun_disc_radius_deg, glare_radius_deg)

    terrain = result.hit_kind == OBJECT_KIND_TERRAIN
    rock = result.hit_kind == OBJECT_KIND_ROCK
    rover = result.hit_kind == OBJECT_KIND_ROVER
    object_ids[terrain] = ground_class[terrain]
    object_ids[rover] = OBJECT_ID["rover"]
    if rock.any():
        if rock_diameter_m is None:
            raise ValueError("rock hits without a rock diameter table")
        diameter = rock_diameter_m[result.hit_index[rock]]
        object_ids[rock] = np.where(diameter >= boulder_minimum_diameter_m, OBJECT_ID["boulder"], OBJECT_ID["small_rock"])
        rock_instance[rock] = result.hit_index[rock] + 1

    surface = ~sky
    normal = result.hit_normal.copy()
    towards_camera = np.einsum("ij,ij->i", normal, result.ray_direction) > 0
    normal[towards_camera] *= -1.0
    sun = np.asarray(sun_direction, dtype=np.float64) / np.linalg.norm(sun_direction)
    faces_away = normal @ sun <= 0.0
    shadowed = surface & (faces_away | result.shadow_by_world | result.shadow_by_rover)
    in_rover_shadow = shadowed & result.shadow_by_rover & ~rover
    is_object = rock | rover
    shadow_ids[surface & ~is_object] = SHADOW_ID["sunlit_ground"]
    shadow_ids[surface & ~is_object & shadowed] = SHADOW_ID["shadowed_ground"]
    shadow_ids[surface & is_object] = SHADOW_ID["sunlit_object"]
    shadow_ids[surface & is_object & shadowed] = SHADOW_ID["shadowed_object"]
    shadow_ids[in_rover_shadow] = SHADOW_ID["rover_shadow"]
    return object_ids, shadow_ids, rock_instance


__all__ = ["RayCastResult", "GroundTruthImages", "compose_ground_truth", "sky_classes", "OBJECT_KIND_TERRAIN"]
