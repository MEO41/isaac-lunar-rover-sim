"""Quaternion (xyzw, scalar last) and transform helpers, Sun direction, camera conventions."""
from __future__ import annotations

import math

import numpy as np


def quaternion_from_rotation_matrix(rotation: np.ndarray) -> np.ndarray:
    """Unit quaternion (x, y, z, w) of a 3x3 rotation matrix (Shepperd's method), w >= 0."""
    matrix = np.asarray(rotation, dtype=np.float64)
    trace = np.trace(matrix)
    if trace > 0:
        scale = math.sqrt(trace + 1.0) * 2
        w = 0.25 * scale
        x = (matrix[2, 1] - matrix[1, 2]) / scale
        y = (matrix[0, 2] - matrix[2, 0]) / scale
        z = (matrix[1, 0] - matrix[0, 1]) / scale
    elif matrix[0, 0] > matrix[1, 1] and matrix[0, 0] > matrix[2, 2]:
        scale = math.sqrt(1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2]) * 2
        w = (matrix[2, 1] - matrix[1, 2]) / scale
        x = 0.25 * scale
        y = (matrix[0, 1] + matrix[1, 0]) / scale
        z = (matrix[0, 2] + matrix[2, 0]) / scale
    elif matrix[1, 1] > matrix[2, 2]:
        scale = math.sqrt(1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2]) * 2
        w = (matrix[0, 2] - matrix[2, 0]) / scale
        x = (matrix[0, 1] + matrix[1, 0]) / scale
        y = 0.25 * scale
        z = (matrix[1, 2] + matrix[2, 1]) / scale
    else:
        scale = math.sqrt(1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1]) * 2
        w = (matrix[1, 0] - matrix[0, 1]) / scale
        x = (matrix[0, 2] + matrix[2, 0]) / scale
        y = (matrix[1, 2] + matrix[2, 1]) / scale
        z = 0.25 * scale
    quaternion = np.array([x, y, z, w])
    quaternion /= np.linalg.norm(quaternion)
    return quaternion if quaternion[3] >= 0 else -quaternion


def rotation_matrix_from_quaternion(quaternion_xyzw) -> np.ndarray:
    """3x3 rotation matrix of a quaternion (x, y, z, w)."""
    x, y, z, w = np.asarray(quaternion_xyzw, dtype=np.float64) / np.linalg.norm(quaternion_xyzw)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def quaternion_from_yaw(yaw_rad: float) -> np.ndarray:
    """Quaternion (x, y, z, w) of a rotation about +z by yaw_rad."""
    return np.array([0.0, 0.0, math.sin(yaw_rad / 2), math.cos(yaw_rad / 2)])


def yaw_from_quaternion(quaternion_xyzw) -> float:
    """Heading (rad, counter-clockwise from east) of the body x axis projected on the horizontal plane."""
    forward = rotation_matrix_from_quaternion(quaternion_xyzw)[:, 0]
    return math.atan2(forward[1], forward[0])


def roll_pitch_yaw_from_quaternion(quaternion_xyzw) -> tuple[float, float, float]:
    """(roll, pitch, yaw) in radians, ZYX convention."""
    x, y, z, w = quaternion_xyzw
    roll = math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    pitch = math.asin(max(-1.0, min(1.0, 2 * (w * y - z * x))))
    yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    return roll, pitch, yaw


def sun_direction_w(azimuth_deg: float, elevation_deg: float) -> np.ndarray:
    """Unit vector towards the Sun (x east, y north, z up); azimuth = compass bearing (0 north, 90 east)."""
    azimuth, elevation = math.radians(azimuth_deg), math.radians(elevation_deg)
    return np.array([math.sin(azimuth) * math.cos(elevation), math.cos(azimuth) * math.cos(elevation), math.sin(elevation)])


def focal_length_mm(horizontal_field_of_view_deg: float, horizontal_aperture_mm: float) -> float:
    """Pinhole focal length (mm) giving a horizontal field of view on a horizontal aperture."""
    return horizontal_aperture_mm / (2 * math.tan(math.radians(horizontal_field_of_view_deg) / 2))


def camera_intrinsic_matrix(width_px: int, height_px: int, horizontal_field_of_view_deg: float) -> np.ndarray:
    """3x3 pinhole intrinsics (square pixels, principal point at the image centre)."""
    focal_px = width_px / (2 * math.tan(math.radians(horizontal_field_of_view_deg) / 2))
    return np.array([[focal_px, 0.0, width_px / 2], [0.0, focal_px, height_px / 2], [0.0, 0.0, 1.0]])


def look_at_rotation(eye: np.ndarray, target: np.ndarray, up: np.ndarray = np.array([0.0, 0.0, 1.0])) -> np.ndarray:
    """Rotation (3x3, columns = USD camera x right, y up, z backward) of a camera at eye looking at target."""
    backward = np.asarray(eye, dtype=np.float64) - np.asarray(target, dtype=np.float64)
    backward /= np.linalg.norm(backward)
    right = np.cross(up, backward)
    right /= np.linalg.norm(right)
    camera_up = np.cross(backward, right)
    return np.column_stack([right, camera_up, backward])
