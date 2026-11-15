"""``load_scene``: read a baked scene directory back into arrays + manifest (no DEM access, no rasterio)."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

from lunar_rover_sim.core.configuration.configuration_models import SimulatorConfiguration
from lunar_rover_sim.core.data_model.tier_layout import TierLayout, build_tier_layout
from lunar_rover_sim.core.terrain.crater_population import CraterTable


@dataclass
class LoadedScene:
    """A baked scene: its configuration, tier layout, height arrays, crater table, rasters and manifest."""

    directory: Path
    configuration: SimulatorConfiguration
    layout: TierLayout
    terrain_arrays: dict[str, np.ndarray]
    crater_table: CraterTable
    rasters: dict[str, np.ndarray]
    manifest: dict
    extra_arrays: dict[str, dict[str, np.ndarray]]

    @property
    def workspace_height_m(self) -> np.ndarray:
        """Final heights on the workspace visual lattice (m, relative to the datum)."""
        return self.terrain_arrays["workspace_height_m"]

    @property
    def workspace_collision_height_m(self) -> np.ndarray:
        """Heights on the workspace collision lattice (exact subsample of the visual lattice)."""
        decimation = self.layout.collision_decimation
        return self.workspace_height_m[::decimation, ::decimation]


BAKED_SECTIONS = ("generator_version", "scene_seed", "site", "terrain", "rocks", "regolith")


def load_scene(directory: str | Path, runtime_configuration: SimulatorConfiguration | None = None) -> LoadedScene:
    """Load a scene directory written by ``bake_scene``; raises FileNotFoundError for missing or partial scenes.

    Baked sections (site, terrain, rocks, regolith, seed) always come from the scene. Runtime-only sections (rover,
    physics, tracks, lighting, render, ...) come from ``runtime_configuration``, or from the current configuration
    files for the scene's preset, so tuning them never requires a re-bake."""
    directory = Path(directory)
    if directory.name.endswith(".partial") or not (directory / "manifest.json").exists():
        raise FileNotFoundError(f"{directory} is not a complete baked scene (no manifest.json). "
                                f"Bake it with scripts/bake_scene.py.")
    manifest = json.loads((directory / "manifest.json").read_text())
    baked = yaml.safe_load((directory / "configuration_resolved.yaml").read_text())
    if runtime_configuration is None:
        from lunar_rover_sim.core.configuration.configuration_loader import load_configuration
        runtime_configuration = load_configuration(baked["profile"]["preset"])
    merged = runtime_configuration.model_dump(mode="json")
    for section in BAKED_SECTIONS:
        merged[section] = baked[section]
    configuration = SimulatorConfiguration.model_validate(merged)
    layout = build_tier_layout(configuration.terrain)
    with np.load(directory / "terrain.npz") as archive:
        terrain_arrays = {name: archive[name] for name in archive.files}
    with np.load(directory / "craters.npz") as archive:
        crater_table = CraterTable.from_arrays({name: archive[name] for name in archive.files})
    with np.load(directory / "rasters.npz") as archive:
        rasters = {name: archive[name] for name in archive.files}
    extra_arrays = {}
    for name in ("rocks", "soil", "tracks"):
        path = directory / f"{name}.npz"
        if path.exists():
            with np.load(path, allow_pickle=False) as archive:
                extra_arrays[name] = {key: archive[key] for key in archive.files}
    expected = layout.workspace_visual_grid.shape
    if terrain_arrays["workspace_height_m"].shape != expected:
        raise ValueError(f"workspace height shape {terrain_arrays['workspace_height_m'].shape} != layout {expected}")
    return LoadedScene(directory, configuration, layout, terrain_arrays, crater_table, rasters, manifest, extra_arrays)
