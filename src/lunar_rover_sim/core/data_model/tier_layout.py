"""Nested terrain tiers on one datum: workspace (fine, collided), near ring (collided), regional, backdrop.

All tiers are centred on the site origin. Each inner tier sits exactly on vertex lines of the next outer tier, so
the inner border vertices lie on outer triangle edges and the composed surface is watertight after the outer tier's
hole is cut.
"""
from __future__ import annotations

from dataclasses import dataclass

from lunar_rover_sim.core.configuration.configuration_models import TerrainConfiguration
from lunar_rover_sim.core.data_model.regular_grid import RegularGrid


@dataclass(frozen=True)
class TierLayout:
    """The lattices of every terrain tier and the workspace tiling."""

    workspace_visual_grid: RegularGrid
    workspace_collision_grid: RegularGrid
    near_ring_grid: RegularGrid
    regional_grid: RegularGrid | None
    backdrop_grid: RegularGrid | None
    collision_decimation: int
    visual_tile_cells: int
    collision_tile_cells: int
    workspace_skirt_m: float
    near_ring_skirt_m: float
    regional_skirt_m: float
    drivable_half_size_m: float

    def outer_grids(self) -> list[tuple[str, RegularGrid]]:
        """(name, grid) of the tiers outside the workspace, inner to outer."""
        grids = [("near_ring", self.near_ring_grid)]
        if self.regional_grid is not None:
            grids.append(("regional", self.regional_grid))
        if self.backdrop_grid is not None:
            grids.append(("backdrop", self.backdrop_grid))
        return grids


def _require_multiple(value: float, step: float, description: str) -> None:
    """Raise ValueError unless value is an integer multiple of step (within 1e-6 relative)."""
    ratio = value / step
    if abs(ratio - round(ratio)) > 1e-6:
        raise ValueError(f"{description}: {value} is not an integer multiple of {step}")


def build_tier_layout(terrain: TerrainConfiguration) -> TierLayout:
    """Build and validate the tier lattices from the terrain configuration (raises ValueError on misalignment)."""
    workspace_cells = terrain.workspace_size_m / terrain.visual_spacing_m
    _require_multiple(terrain.workspace_size_m, terrain.visual_spacing_m, "workspace size / visual spacing")
    workspace_cells = int(round(workspace_cells))
    if workspace_cells % terrain.visual_tile_cells:
        raise ValueError(f"workspace cells {workspace_cells} not a multiple of visual_tile_cells {terrain.visual_tile_cells}")
    if workspace_cells % (terrain.collision_decimation * terrain.collision_tile_cells):
        raise ValueError(f"workspace cells {workspace_cells} not a multiple of collision_decimation x collision_tile_cells")
    workspace_visual = RegularGrid.centered(terrain.workspace_size_m, terrain.visual_spacing_m)
    workspace_collision = workspace_visual.subsampled(terrain.collision_decimation)

    near = terrain.near_ring
    _require_multiple(terrain.workspace_size_m, near.spacing_m, "workspace size / near-ring spacing")
    _require_multiple((near.size_m - terrain.workspace_size_m) / 2, near.spacing_m, "near-ring margin / spacing")
    near_grid = RegularGrid.centered(near.size_m, near.spacing_m)
    inner_size, regional_grid, backdrop_grid = near.size_m, None, None
    if terrain.regional.enabled:
        regional = terrain.regional
        _require_multiple(inner_size, regional.spacing_m, "near-ring size / regional spacing")
        _require_multiple((regional.size_m - inner_size) / 2, regional.spacing_m, "regional margin / spacing")
        regional_grid = RegularGrid.centered(regional.size_m, regional.spacing_m)
        inner_size = regional.size_m
    if terrain.backdrop.enabled:
        backdrop = terrain.backdrop
        _require_multiple(inner_size, backdrop.spacing_m, "inner size / backdrop spacing")
        _require_multiple((backdrop.size_m - inner_size) / 2, backdrop.spacing_m, "backdrop margin / spacing")
        backdrop_grid = RegularGrid.centered(backdrop.size_m, backdrop.spacing_m)
    drivable_half_size = terrain.workspace_size_m / 2 - terrain.drivable_margin_m
    if drivable_half_size <= 0:
        raise ValueError("drivable_margin_m leaves no drivable zone inside the workspace")
    return TierLayout(
        workspace_visual_grid=workspace_visual, workspace_collision_grid=workspace_collision,
        near_ring_grid=near_grid, regional_grid=regional_grid, backdrop_grid=backdrop_grid,
        collision_decimation=terrain.collision_decimation, visual_tile_cells=terrain.visual_tile_cells,
        collision_tile_cells=terrain.collision_tile_cells, workspace_skirt_m=terrain.workspace_skirt_m,
        near_ring_skirt_m=near.skirt_m, regional_skirt_m=terrain.regional.skirt_m,
        drivable_half_size_m=drivable_half_size,
    )
