"""Drive models: twist command [forward_speed_m_s, yaw_rate_rad_s] -> wheel angular velocity targets (rad/s).

Wheel order everywhere: front_left, front_right, rear_left, rear_right. Yaw rate positive = counter-clockwise.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import numpy as np

from lunar_rover_sim.core.configuration.configuration_models import RoverConfiguration

WHEEL_NAMES = ("front_left", "front_right", "rear_left", "rear_right")
WHEEL_SIDE_SIGN = np.array([1.0, -1.0, 1.0, -1.0])   # +1 = left side (positive body y)


class DriveModel(Protocol):
    """Anything that maps a twist to wheel angular velocity targets."""

    def wheel_velocity_targets(self, forward_speed_m_s: float, yaw_rate_rad_s: float) -> np.ndarray:
        """Wheel angular velocities (rad/s), shape (4,), positive = rolling forward."""
        ...


@dataclass
class SkidSteerDriveModel:
    """Differential (skid) steering: left wheels at v - w*B/2, right wheels at v + w*B/2, divided by the radius."""

    wheel_radius_m: float
    effective_track_width_m: float

    @classmethod
    def from_rover(cls, rover: RoverConfiguration) -> "SkidSteerDriveModel":
        """Build from the rover specification (effective track width = track width x calibration factor)."""
        return cls(rover.wheel_radius_m, rover.track_width_m * rover.skid_steer_track_width_factor,
                   rover.maximum_wheel_rim_speed_m_s)

    maximum_rim_speed_m_s: float = float("inf")

    def wheel_velocity_targets(self, forward_speed_m_s: float, yaw_rate_rad_s: float) -> np.ndarray:
        """Wheel angular velocities (rad/s), order front_left, front_right, rear_left, rear_right.

        If a wheel would exceed the rim-speed limit, all four are scaled down together (keeps the turn ratio)."""
        side_speed = forward_speed_m_s - WHEEL_SIDE_SIGN * yaw_rate_rad_s * self.effective_track_width_m / 2
        peak = float(np.max(np.abs(side_speed)))
        if peak > self.maximum_rim_speed_m_s:
            side_speed = side_speed * (self.maximum_rim_speed_m_s / peak)
        return side_speed / self.wheel_radius_m

    def twist_from_wheel_velocities(self, wheel_angular_velocity_rad_s: np.ndarray) -> tuple[float, float]:
        """Inverse model: (forward_speed_m_s, yaw_rate_rad_s) implied by four wheel speeds (no slip)."""
        speeds = np.asarray(wheel_angular_velocity_rad_s, dtype=np.float64) * self.wheel_radius_m
        left = 0.5 * (speeds[0] + speeds[2])
        right = 0.5 * (speeds[1] + speeds[3])
        return 0.5 * (left + right), (right - left) / self.effective_track_width_m


@dataclass
class YawRateController:
    """PI correction of the yaw-rate command from the measured yaw rate.

    Skid steering scrubs the wheels sideways, so the yaw rate reached by open-loop wheel speeds is only a fraction of
    the command (measured on this rover: about 24 % spinning in place, 10 % in an arc). The controller returns the
    yaw rate to feed the drive model so that the MEASURED yaw rate tracks the reference."""

    proportional_gain: float
    integral_gain: float
    integral_limit_rad_s: float
    integral_rad_s: float = 0.0

    @classmethod
    def from_rover(cls, rover: RoverConfiguration) -> "YawRateController":
        """Build from the rover specification."""
        return cls(rover.yaw_rate_proportional_gain, rover.yaw_rate_integral_gain, rover.yaw_rate_integral_limit_rad_s)

    def reset(self) -> None:
        """Clear the integral state."""
        self.integral_rad_s = 0.0

    def command(self, yaw_rate_reference_rad_s: float, yaw_rate_measured_rad_s: float, control_period_s: float) -> float:
        """Yaw-rate command for the drive model (reference + PI correction)."""
        error = yaw_rate_reference_rad_s - yaw_rate_measured_rad_s
        if abs(yaw_rate_reference_rad_s) < 1e-3:
            self.integral_rad_s *= 0.5
        self.integral_rad_s = float(np.clip(self.integral_rad_s + self.integral_gain * error * control_period_s,
                                            -self.integral_limit_rad_s, self.integral_limit_rad_s))
        return yaw_rate_reference_rad_s + self.proportional_gain * error + self.integral_rad_s


@dataclass
class TwistLimiter:
    """Clamps a twist command to speed limits and rate-limits it to acceleration limits (state = last output)."""

    maximum_forward_speed_m_s: float
    maximum_yaw_rate_rad_s: float
    maximum_forward_acceleration_m_s2: float
    maximum_yaw_acceleration_rad_s2: float
    last_output: np.ndarray = field(default_factory=lambda: np.zeros(2))

    @classmethod
    def from_rover(cls, rover: RoverConfiguration) -> "TwistLimiter":
        """Build from the rover specification."""
        return cls(rover.maximum_forward_speed_m_s, rover.maximum_yaw_rate_rad_s,
                   rover.maximum_forward_acceleration_m_s2, rover.maximum_yaw_acceleration_rad_s2)

    def reset(self) -> None:
        """Forget the previous command (next output starts from rest)."""
        self.last_output = np.zeros(2)

    def limit(self, forward_speed_m_s: float, yaw_rate_rad_s: float, control_period_s: float) -> np.ndarray:
        """Limited [forward_speed_m_s, yaw_rate_rad_s] for one control period."""
        desired = np.array([np.clip(forward_speed_m_s, -self.maximum_forward_speed_m_s, self.maximum_forward_speed_m_s),
                            np.clip(yaw_rate_rad_s, -self.maximum_yaw_rate_rad_s, self.maximum_yaw_rate_rad_s)])
        maximum_change = np.array([self.maximum_forward_acceleration_m_s2,
                                   self.maximum_yaw_acceleration_rad_s2]) * control_period_s
        self.last_output = self.last_output + np.clip(desired - self.last_output, -maximum_change, maximum_change)
        return self.last_output.copy()
