"""Regolith: SYNTHETIC soil property maps, per-tile friction, macro albedo and detail textures (DESIGN section 9.4).

Everything here is SYNTHETIC and uncalibrated. Soil properties vary with a correlated latent "compaction" field,
three facies (firm / loose / soft) at configured areal fractions, and softer soil near fresh craters (Apollo 15
measured the deepest sinkage on fresh crater rims). Bekker moduli scale with compaction inside a contrast limit so
ruts never become invisible or absurdly deep. (The design's adapter to the sibling lunar_regolith package is
replaced by this self-contained model; see docs/decision_log.md.)
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy import fft as scipy_fft
from scipy import ndimage

from lunar_rover_sim.core.configuration.configuration_models import RegolithConfiguration
from lunar_rover_sim.core.data_model.regular_grid import RegularGrid
from lunar_rover_sim.core.seeding import named_generator
from lunar_rover_sim.core.soil_mechanics.bekker_sinkage import (REFERENCE_COHESIVE_MODULUS_N_PER_M2,
                                                                REFERENCE_FRICTIONAL_MODULUS_N_PER_M3)
from lunar_rover_sim.core.terrain.crater_population import CraterTable
from lunar_rover_sim.core.terrain.spectral_fields import periodic_noise_texture, wavenumber_grid

FACIES_NAMES = ("firm", "loose", "soft")


@dataclass
class SoilPropertyMaps:
    """SYNTHETIC soil properties on a regular lattice ([north, east] arrays)."""

    grid: RegularGrid
    facies_id: np.ndarray
    compaction_index: np.ndarray
    friction_angle_deg: np.ndarray
    cohesion_pa: np.ndarray
    bulk_density_kg_m3: np.ndarray
    cohesive_modulus_n_per_m2: np.ndarray
    frictional_modulus_n_per_m3: np.ndarray
    static_friction: np.ndarray

    def to_arrays(self) -> dict[str, np.ndarray]:
        """Arrays for np.savez (lattice stored as soil_grid = [origin_east, origin_north, spacing, count_east, count_north])."""
        grid = self.grid
        arrays = {name: value for name, value in self.__dict__.items() if name != "grid"}
        arrays["soil_grid"] = np.array([grid.origin_east_m, grid.origin_north_m, grid.spacing_m, grid.count_east,
                                        grid.count_north])
        return arrays


def gaussian_random_field(shape: tuple[int, int], spacing_m: float, correlation_length_m: float,
                          random_generator: np.random.Generator) -> np.ndarray:
    """Zero-mean unit-variance field with a Gaussian covariance of the given correlation length (FFT, padded)."""
    padding = int(np.ceil(2 * correlation_length_m / spacing_m))
    padded_shape = (shape[0] + 2 * padding, shape[1] + 2 * padding)
    magnitude, _, _ = wavenumber_grid(padded_shape, spacing_m)
    transfer = np.exp(-0.25 * (magnitude * correlation_length_m) ** 2)
    noise = random_generator.standard_normal(padded_shape)
    field = scipy_fft.ifft2(scipy_fft.fft2(noise, workers=1) * transfer, workers=1).real
    field = field[padding:padding + shape[0], padding:padding + shape[1]]
    return (field - field.mean()) / field.std()


def synthesize_soil_property_maps(grid: RegularGrid, regolith: RegolithConfiguration, craters: CraterTable,
                                  seed: int) -> SoilPropertyMaps:
    """Build the SYNTHETIC soil maps on ``grid`` (normally 0.5 m over the near ring)."""
    generator = named_generator(seed, "regolith.latent_fields")
    facies_field = gaussian_random_field(grid.shape, grid.spacing_m, regolith.correlation_length_m, generator)
    independent = gaussian_random_field(grid.shape, grid.spacing_m, regolith.correlation_length_m / 2, generator)
    compaction_latent = 0.5 * facies_field + np.sqrt(0.75) * independent
    strength_residual = gaussian_random_field(grid.shape, grid.spacing_m, regolith.correlation_length_m / 3, generator)
    fractions = regolith.facies_fractions
    thresholds = np.quantile(facies_field, [fractions.firm, fractions.firm + fractions.loose])
    facies = np.where(facies_field < thresholds[0], 0, np.where(facies_field < thresholds[1], 1, 2)).astype(np.uint8)
    offset = np.array([1.0, 0.0, -1.0])[facies]
    compaction = 1.0 / (1.0 + np.exp(-(1.2 * compaction_latent + offset)))
    east, north = grid.coordinate_mesh()
    for index in np.flatnonzero(craters.freshness_class == 0):
        radius = craters.diameter_m[index] / 2
        near = (east - craters.center_east_m[index]) ** 2 + (north - craters.center_north_m[index]) ** 2 <= (1.5 * radius) ** 2
        compaction[near] *= regolith.fresh_crater_compaction_factor
    low, high = regolith.friction_angle_range_deg
    friction_angle = np.clip(low + compaction * (high - low) + 1.0 * strength_residual, low, high)
    cohesion_low, cohesion_high = regolith.cohesion_range_pa
    cohesion = np.exp(np.log(cohesion_low) + compaction * (np.log(cohesion_high) - np.log(cohesion_low)))
    density_low, density_high = regolith.bulk_density_range_kg_m3
    density = density_low + compaction * (density_high - density_low)
    contrast_low, contrast_high = regolith.soil_contrast_limit
    scale = np.exp(np.log(contrast_low) + compaction * (np.log(contrast_high) - np.log(contrast_low)))
    friction_low, friction_high = regolith.interface_friction_clip
    static_friction = np.clip(np.tan(np.radians(regolith.interface_friction_ratio * friction_angle)), friction_low, friction_high)
    return SoilPropertyMaps(grid, facies, compaction.astype(np.float32), friction_angle.astype(np.float32),
                            cohesion.astype(np.float32), density.astype(np.float32),
                            (REFERENCE_COHESIVE_MODULUS_N_PER_M2 * scale).astype(np.float32),
                            (REFERENCE_FRICTIONAL_MODULUS_N_PER_M3 * scale).astype(np.float32),
                            static_friction.astype(np.float32))


def tile_mean_static_friction(soil: SoilPropertyMaps, collision_grid: RegularGrid, tile_cells: int) -> np.ndarray:
    """Mean static friction over each collision tile, shape (tiles_north, tiles_east)."""
    tiles = collision_grid.split_into_tiles(tile_cells)
    rows = (collision_grid.count_north - 1) // tile_cells
    columns = (collision_grid.count_east - 1) // tile_cells
    east, north = soil.grid.coordinate_mesh()
    result = np.zeros((rows, columns))
    for tile in tiles:
        east_low = collision_grid.origin_east_m + tile.column_start * collision_grid.spacing_m
        east_high = collision_grid.origin_east_m + (tile.column_stop - 1) * collision_grid.spacing_m
        north_low = collision_grid.origin_north_m + tile.row_start * collision_grid.spacing_m
        north_high = collision_grid.origin_north_m + (tile.row_stop - 1) * collision_grid.spacing_m
        inside = (east >= east_low) & (east <= east_high) & (north >= north_low) & (north <= north_high)
        result[tile.row_start // tile_cells, tile.column_start // tile_cells] = soil.static_friction[inside].mean()
    return result


def sample_soil_map(values: np.ndarray, soil_grid: RegularGrid, east_m, north_m) -> np.ndarray:
    """Bilinear lookup of a soil map at points (nearest edge value outside)."""
    row, column = soil_grid.fractional_index(east_m, north_m)
    return ndimage.map_coordinates(values.astype(np.float64), [np.atleast_1d(row), np.atleast_1d(column)], order=1,
                                   mode="nearest")


def macro_albedo_image(grid: RegularGrid, size_px: int, base_albedo: float, mottling_amplitude: float,
                       ejecta_brightening: float, soil: SoilPropertyMaps, craters: CraterTable,
                       seed: int) -> np.ndarray:
    """sRGB uint8 image (size_px x size_px x 3, row 0 = NORTH) of the macro normal albedo over ``grid``'s extent."""
    generator = named_generator(seed, "regolith.albedo")
    pixel_m = grid.extent_east_m / size_px
    shape = (size_px, size_px)
    mottling = 0.7 * gaussian_random_field(shape, pixel_m, 15.0, generator) + \
        0.5 * gaussian_random_field(shape, pixel_m, 3.0, generator)
    centers = grid.origin_east_m + (np.arange(size_px) + 0.5) * pixel_m
    east, north = np.meshgrid(centers, grid.origin_north_m + (np.arange(size_px) + 0.5) * pixel_m)
    loose = sample_soil_map((soil.facies_id == 1).astype(np.float64), soil.grid, east.ravel(), north.ravel()).reshape(shape)
    mottling -= 0.4 * loose
    ejecta = np.zeros(shape)
    for index in np.flatnonzero(craters.freshness_class == 0):
        radius = craters.diameter_m[index] / 2
        distance = np.hypot(east - craters.center_east_m[index], north - craters.center_north_m[index]) / radius
        ejecta = np.maximum(ejecta, np.clip(2.0 - distance, 0.0, 1.0))
    edge_distance = np.minimum.reduce([east - grid.origin_east_m, grid.maximum_east_m - east,
                                       north - grid.origin_north_m, grid.maximum_north_m - north])
    fade = np.clip(edge_distance / (0.1 * grid.extent_east_m), 0.0, 1.0)
    albedo = base_albedo * (1.0 + mottling_amplitude * mottling * fade) * (1.0 + ejecta_brightening * ejecta)
    tint = np.array([1.0, 0.975, 0.93])
    linear = np.clip(albedo[..., None] * tint, 0.0, 1.0)
    srgb = np.where(linear <= 0.0031308, 12.92 * linear, 1.055 * linear ** (1 / 2.4) - 0.055)
    return np.flipud(np.round(srgb * 255)).astype(np.uint8)


def detail_texture_image(size_px: int, tile_size_m: float, hurst_exponent: float, standard_deviation: float,
                         generator: np.random.Generator) -> np.ndarray:
    """Seamlessly tiling grey micro-relief texture, uint8, linear, mean 0.5 (128)."""
    noise = periodic_noise_texture(size_px, tile_size_m, hurst_exponent, generator)
    return np.round(np.clip(0.5 + standard_deviation * noise, 0.0, 1.0) * 255).astype(np.uint8)


def write_regolith_textures(directory: Path, near_grid: RegularGrid, regolith: RegolithConfiguration,
                            soil: SoilPropertyMaps, craters: CraterTable, seed: int) -> dict:
    """Write textures/macro_albedo.png, detail_coarse.png, detail_fine.png; return the scene albedo parameters."""
    from PIL import Image
    directory.mkdir(parents=True, exist_ok=True)
    generator = named_generator(seed, "regolith.parameters")
    base_albedo = float(generator.uniform(*regolith.base_normal_albedo_range)) if regolith.randomize_base_normal_albedo \
        else regolith.base_normal_albedo
    albedo = macro_albedo_image(near_grid, regolith.macro_albedo_size_px, base_albedo, regolith.albedo_mottling_amplitude,
                                regolith.fresh_ejecta_brightening, soil, craters, seed)
    Image.fromarray(albedo, mode="RGB").save(directory / "macro_albedo.png", optimize=True)
    for name, octave in (("coarse", regolith.detail_texture_coarse), ("fine", regolith.detail_texture_fine)):
        image = detail_texture_image(octave.size_px, octave.tile_size_m, regolith.detail_hurst_exponent,
                                     regolith.detail_texture_standard_deviation,
                                     named_generator(seed, f"regolith.detail_texture.{name}"))
        Image.fromarray(image, mode="L").save(directory / f"detail_{name}.png", optimize=True)
    return {"base_normal_albedo": base_albedo, "synthetic": True}
