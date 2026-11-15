"""Rigid-wheel static sinkage (Bekker / Wong) with slip and multipass factors. Used for rut DEPTH only (visual).

Pressure-sinkage p = (k_c / b + k_phi) z^n. For a rigid wheel of diameter D and width b carrying normal load W
(Wong, Theory of Ground Vehicles, eq. 2.47):  z0 = ( 3 W / ((3 - n) (k_c / b + k_phi) b sqrt(D)) )^(2 / (2n + 1)).
Reference lunar soil (Apollo-era nominal, Costes et al. 1972 / Wong): n = 1, k_c = 1400 N/m^2, k_phi = 820 kN/m^3.
Slip sinkage and multipass gains are ASSUMPTIONS (configuration fields), not calibrated against lunar data.
"""
from __future__ import annotations

import numpy as np

REFERENCE_COHESIVE_MODULUS_N_PER_M2 = 1400.0
REFERENCE_FRICTIONAL_MODULUS_N_PER_M3 = 820000.0
REFERENCE_SINKAGE_EXPONENT = 1.0


def static_wheel_sinkage_m(normal_load_n, cohesive_modulus_n_per_m2, frictional_modulus_n_per_m3, wheel_width_m: float,
                           wheel_diameter_m: float, exponent: float = REFERENCE_SINKAGE_EXPONENT):
    """Static sinkage z0 (m) of a rigid wheel; arrays broadcast. Zero load gives zero sinkage."""
    load = np.maximum(np.asarray(normal_load_n, dtype=np.float64), 0.0)
    modulus = np.asarray(cohesive_modulus_n_per_m2, dtype=np.float64) / wheel_width_m + \
        np.asarray(frictional_modulus_n_per_m3, dtype=np.float64)
    base = 3.0 * load / ((3.0 - exponent) * modulus * wheel_width_m * np.sqrt(wheel_diameter_m))
    return base ** (2.0 / (2.0 * exponent + 1.0))


def wheel_rut_depth_m(normal_load_n, slip_ratio, previous_pass_count, cohesive_modulus_n_per_m2,
                      frictional_modulus_n_per_m3, wheel_width_m: float, wheel_diameter_m: float, exponent: float,
                      slip_gain: float, multipass_gains, visual_floor_m: float, maximum_m: float):
    """Rut depth (m) = clip(z0 (1 + slip_gain |slip|) gain[min(passes, 3)], floor, maximum); zero where load is zero."""
    static = static_wheel_sinkage_m(normal_load_n, cohesive_modulus_n_per_m2, frictional_modulus_n_per_m3,
                                    wheel_width_m, wheel_diameter_m, exponent)
    passes = np.minimum(np.asarray(previous_pass_count, dtype=np.int64), len(multipass_gains) - 1)
    depth = static * (1.0 + slip_gain * np.abs(np.asarray(slip_ratio, dtype=np.float64))) * \
        np.asarray(multipass_gains, dtype=np.float64)[passes]
    loaded = np.asarray(normal_load_n, dtype=np.float64) > 0
    return np.where(loaded, np.clip(depth, visual_floor_m, maximum_m), 0.0)
