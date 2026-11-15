"""Small-crater population below the DEM resolution (SYNTHETIC; docs/DESIGN.md section 9.2 step 3).

Cumulative areal density N(>D) = c * D^-2 per m^2 (equilibrium, Trask 1966 / Minton 2019), drawn in two disjoint
diameter bands so no crater is drawn twice: band A [near_ring_minimum_diameter, cutoff] over the near ring, band B
[minimum_diameter, near_ring_minimum_diameter) over the workspace plus an apron. Craters larger than the taper start
are thinned linearly to zero at the DEM cutoff wavelength because the DEM already contains the larger ones.
Degradation u = K t / R^2 (linear topographic diffusion) is uniform in [0, u_max].
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from lunar_rover_sim.core.configuration.configuration_models import CraterConfiguration

HARMONIC_ORDERS = (2, 3, 4, 5)
BAND_A = 0
BAND_B = 1


@dataclass
class CraterTable:
    """Column arrays describing every synthetic crater (one row per crater)."""

    center_east_m: np.ndarray
    center_north_m: np.ndarray
    diameter_m: np.ndarray
    fresh_depth_to_diameter: np.ndarray
    degradation_parameter: np.ndarray
    ellipticity: np.ndarray
    orientation_rad: np.ndarray
    harmonic_amplitudes: np.ndarray
    harmonic_phases_rad: np.ndarray
    band: np.ndarray
    is_feature_crater: np.ndarray
    freshness_class: np.ndarray = field(default=None)
    age_rank: np.ndarray = field(default=None)
    realised_depth_m: np.ndarray = field(default=None)
    realised_rim_height_m: np.ndarray = field(default=None)
    center_height_m: np.ndarray = field(default=None)

    def __len__(self) -> int:
        """Number of craters."""
        return len(self.diameter_m)

    def to_arrays(self) -> dict[str, np.ndarray]:
        """All columns as a dict suitable for np.savez (prefix ``crater_``)."""
        return {f"crater_{name}": np.asarray(value) for name, value in self.__dict__.items() if value is not None}

    @classmethod
    def from_arrays(cls, arrays) -> "CraterTable":
        """Rebuild a table from the dict written by ``to_arrays`` (or an open npz)."""
        names = [name for name in cls.__dataclass_fields__]
        return cls(**{name: (np.asarray(arrays[f"crater_{name}"]) if f"crater_{name}" in arrays else None)
                      for name in names})


def sample_diameters_power_law(count: int, diameter_low_m: float, diameter_high_m: float,
                               random_generator: np.random.Generator) -> np.ndarray:
    """Diameters with cumulative N(>D) ~ D^-2 between low and high, by inverse CDF."""
    quantile = random_generator.random(count)
    return diameter_low_m * (1.0 - quantile * (1.0 - (diameter_low_m / diameter_high_m) ** 2)) ** -0.5


def sample_crater_population(craters: CraterConfiguration, equilibrium_coefficient: float, dem_cutoff_m: float,
                             near_ring_half_size_m: float, workspace_half_size_m: float, drivable_half_size_m: float,
                             minimum_band_b_diameter_m: float, population_generator: np.random.Generator,
                             shape_generator: np.random.Generator) -> CraterTable:
    """Draw the crater population (positions, diameters, shapes, degradation) for one scene."""
    centers_east, centers_north, diameters, bands = [], [], [], []
    band_a_low = craters.near_ring_minimum_diameter_m
    band_b_low = max(craters.minimum_diameter_m, minimum_band_b_diameter_m)
    band_b_half = workspace_half_size_m + craters.apron_extent_radii * band_a_low / 2
    for band, low, high, half_size in ((BAND_A, band_a_low, dem_cutoff_m, near_ring_half_size_m),
                                       (BAND_B, band_b_low, band_a_low, band_b_half)):
        if low >= high:
            continue
        area = (2 * half_size) ** 2
        mean_count = equilibrium_coefficient * (low ** -2 - high ** -2) * area
        count = int(population_generator.poisson(mean_count))
        band_diameters = sample_diameters_power_law(count, low, high, population_generator)
        band_east = population_generator.uniform(-half_size, half_size, count)
        band_north = population_generator.uniform(-half_size, half_size, count)
        taper_draw = population_generator.random(count)
        keep_probability = np.where(band_diameters > craters.taper_start_diameter_m,
                                    (dem_cutoff_m - band_diameters) / (dem_cutoff_m - craters.taper_start_diameter_m), 1.0)
        keep = taper_draw < keep_probability
        centers_east.append(band_east[keep])
        centers_north.append(band_north[keep])
        diameters.append(band_diameters[keep])
        bands.append(np.full(int(keep.sum()), band, dtype=np.int8))
    feature_count = craters.feature_crater_count
    feature_low, feature_high = craters.feature_crater_diameter_range_m
    feature_diameters = population_generator.uniform(feature_low, feature_high, feature_count)
    feature_east, feature_north = [], []
    for diameter in feature_diameters:
        # Rim inside the drivable zone, centre at least one diameter from the origin (the preferred spawn point);
        # if no draw satisfies that (small presets), keep the draw farthest from the origin.
        limit = max(drivable_half_size_m - diameter / 2, diameter / 4)
        best_candidate, best_distance = None, -1.0
        for _ in range(1000):
            candidate = population_generator.uniform(-limit, limit, 2)
            distance = float(np.hypot(*candidate))
            if distance > best_distance:
                best_candidate, best_distance = candidate, distance
            if distance >= diameter:
                break
        feature_east.append(best_candidate[0])
        feature_north.append(best_candidate[1])
    centers_east.append(np.asarray(feature_east, dtype=np.float64))
    centers_north.append(np.asarray(feature_north, dtype=np.float64))
    diameters.append(feature_diameters)
    bands.append(np.full(feature_count, BAND_A, dtype=np.int8))
    is_feature = np.concatenate([np.zeros(sum(len(d) for d in diameters[:-1]), bool), np.ones(feature_count, bool)])

    diameter_m = np.concatenate(diameters)
    count = len(diameter_m)
    depth_low, depth_high = craters.fresh_depth_to_diameter_range
    degradation = shape_generator.uniform(0.0, craters.degradation_parameter_maximum, count)
    degradation[is_feature] = 0.01
    harmonic_maximum = craters.rim_harmonic_amplitude_maximum / np.asarray(HARMONIC_ORDERS, dtype=np.float64)
    table = CraterTable(
        center_east_m=np.concatenate(centers_east), center_north_m=np.concatenate(centers_north),
        diameter_m=diameter_m,
        fresh_depth_to_diameter=shape_generator.uniform(depth_low, depth_high, count),
        degradation_parameter=degradation,
        ellipticity=shape_generator.uniform(craters.ellipticity_minimum, 1.0, count),
        orientation_rad=shape_generator.uniform(0.0, np.pi, count),
        harmonic_amplitudes=shape_generator.random((count, len(HARMONIC_ORDERS))) * harmonic_maximum,
        harmonic_phases_rad=shape_generator.uniform(0.0, 2 * np.pi, (count, len(HARMONIC_ORDERS))),
        band=np.concatenate(bands), is_feature_crater=is_feature,
    )
    table.freshness_class = np.where(degradation < craters.fresh_degradation_threshold, 0,
                                     np.where(degradation < craters.moderate_degradation_threshold, 1, 2)).astype(np.int8)
    order = np.argsort(-degradation, kind="stable")
    table.age_rank = np.empty(count, dtype=np.int32)
    table.age_rank[order] = np.arange(count, dtype=np.int32)
    table.realised_depth_m = np.zeros(count)
    table.realised_rim_height_m = np.zeros(count)
    table.center_height_m = np.zeros(count)
    return table
