"""Turn per-physics-step wheel kinematics into gated ``WheelContactSample`` s (Isaac-free, vectorised over wheels).

Cheap part every step: low-pass the net wheel contact force; accumulate rolling distance. Full geometry only for
wheels whose accumulator reached ``sample_spacing_m``: contact point on the collider rest surface below the wheel
centre (perpendicular to the axle), heading, slip, normal load, gap gate, sinkage from the soil maps.
"""
from __future__ import annotations

import math

import numpy as np

from lunar_rover_sim.core.configuration.configuration_models import (RoverConfiguration, SinkageConfiguration,
                                                                     TrackConfiguration)
from lunar_rover_sim.core.data_model.regular_grid import RegularGrid, triangulated_height, triangulated_normal
from lunar_rover_sim.core.soil_mechanics.bekker_sinkage import wheel_rut_depth_m
from lunar_rover_sim.core.tracks.track_field import TrackField, WheelContactSample


class WheelContactSampleBuilder:
    """Stateful sampler for four wheels."""

    def __init__(self, rover: RoverConfiguration, tracks: TrackConfiguration, sinkage: SinkageConfiguration,
                 collision_grid: RegularGrid, collision_heights: np.ndarray, soil_lookup, field: TrackField):
        """``soil_lookup(east, north) -> (cohesive_modulus, frictional_modulus)`` arrays (SYNTHETIC soil maps)."""
        self.rover, self.tracks, self.sinkage = rover, tracks, sinkage
        self.collision_grid, self.collision_heights = collision_grid, collision_heights
        self.soil_lookup = soil_lookup
        self.field = field
        self.filtered_force = np.zeros((4, 3))
        self.rolling_distance = np.full(4, tracks.sample_spacing_m)
        self.latest_ground_truth = {"contact": np.zeros(4, bool), "slip_ratio": np.zeros(4), "sinkage_m": np.zeros(4),
                                    "normal_force_n": np.zeros(4)}

    def reset(self) -> None:
        """Forget filter state; accumulators start full so every loaded wheel stamps where it stands."""
        self.filtered_force[:] = 0.0
        self.rolling_distance[:] = self.tracks.sample_spacing_m

    def process(self, time_s: float, step_dt_s: float, wheel_center_m: np.ndarray, wheel_axle_direction: np.ndarray,
                wheel_center_velocity_m_s: np.ndarray, wheel_angle_rad: np.ndarray,
                wheel_angular_velocity_rad_s: np.ndarray, wheel_contact_force_n: np.ndarray) -> list[WheelContactSample]:
        """Feed one physics step; returns the samples emitted at this step."""
        alpha = 1.0 - math.exp(-step_dt_s / self.sinkage.force_filter_time_constant_s)
        self.filtered_force += alpha * (wheel_contact_force_n - self.filtered_force)
        radius = self.rover.wheel_radius_m
        speed = np.maximum(np.linalg.norm(wheel_center_velocity_m_s, axis=1), np.abs(wheel_angular_velocity_rad_s) * radius)
        self.rolling_distance += speed * step_dt_s
        due = np.flatnonzero(self.rolling_distance >= self.tracks.sample_spacing_m)
        if not len(due):
            return []
        centers = wheel_center_m[due]
        axles = wheel_axle_direction[due] / np.linalg.norm(wheel_axle_direction[due], axis=1, keepdims=True)
        normal = triangulated_normal(self.collision_heights, self.collision_grid, centers[:, 0], centers[:, 1])
        contact = centers.copy()
        for _ in range(2):
            perpendicular = normal - np.sum(normal * axles, axis=1, keepdims=True) * axles
            perpendicular /= np.linalg.norm(perpendicular, axis=1, keepdims=True)
            contact = centers - radius * perpendicular
            normal = triangulated_normal(self.collision_heights, self.collision_grid, contact[:, 0], contact[:, 1])
        forward = np.cross(axles, perpendicular)
        forward /= np.linalg.norm(forward, axis=1, keepdims=True)
        heading = np.arctan2(forward[:, 1], forward[:, 0])
        forward_speed = np.sum(wheel_center_velocity_m_s[due] * forward, axis=1)
        rim_speed = wheel_angular_velocity_rad_s[due] * radius
        slip = (rim_speed - forward_speed) / np.maximum.reduce([np.abs(rim_speed), np.abs(forward_speed),
                                                                 np.full(len(due), 0.02)])
        load = np.maximum(0.0, np.sum(self.filtered_force[due] * perpendicular, axis=1))
        surface, valid = triangulated_height(self.collision_heights, self.collision_grid, contact[:, 0], contact[:, 1])
        gap = contact[:, 2] - surface
        inside = self.field.grid.contains(contact[:, 0], contact[:, 1])
        gate = (load > self.tracks.contact_force_threshold_n) & valid & (np.abs(gap) <= self.tracks.contact_gap_threshold_m) & inside
        samples = []
        cohesive, frictional = self.soil_lookup(contact[:, 0], contact[:, 1])
        for position, wheel in enumerate(due):
            self.latest_ground_truth["contact"][wheel] = bool(gate[position])
            self.latest_ground_truth["slip_ratio"][wheel] = float(slip[position])
            self.latest_ground_truth["normal_force_n"][wheel] = float(load[position])
            if not gate[position]:
                self.latest_ground_truth["sinkage_m"][wheel] = 0.0
                continue
            row, column = self.field.grid.fractional_index(contact[position, 0], contact[position, 1])
            row, column = int(round(float(row))), int(round(float(column)))
            previous = int(self.field.pass_count[row, column])
            last = self.field.last_visit_time_s[row, column]
            if previous > 0 and self.field.wheel_index[row, column] == wheel and not np.isnan(last) and \
                    time_s - last <= self.tracks.pass_separation_time_s:
                previous -= 1
            depth = float(wheel_rut_depth_m(load[position], slip[position], previous, cohesive[position],
                                            frictional[position], self.rover.wheel_width_m, 2 * radius,
                                            self.sinkage.exponent, self.sinkage.slip_gain, self.sinkage.multipass_gains,
                                            self.sinkage.visual_floor_m, self.sinkage.maximum_m))
            self.latest_ground_truth["sinkage_m"][wheel] = depth
            samples.append(WheelContactSample(time_s, int(wheel), float(contact[position, 0]), float(contact[position, 1]),
                                              float(heading[position]), depth, float(wheel_angle_rad[wheel]),
                                              float(load[position]), float(slip[position])))
            self.rolling_distance[wheel] = 0.0
        return samples
