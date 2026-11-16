"""Build the lunar_regolith appearance for a baked scene (run with REGOLITH_PYTHON via scripts/regolith_run.sh).

    scripts/regolith_run.sh scripts/build_regolith_appearance.py --preset gpu_8gb --mode scene --quality balanced \
        [--contact-log <run>/contact_log.npz] [--tag drive1]
    scripts/regolith_run.sh scripts/build_regolith_appearance.py --preset gpu_8gb --mode overlay --quality realtime

Modes
  scene    lunar_regolith render scene (scene.usd) over the workspace: field-driven material + render-only geometry
           (micro-relief, clods, imprint patch). The render surface is THIS simulator's workspace collision lattice
           (SceneSpec.surface), so render meshes refine exactly the surface PhysX collides with. Wheel ruts come from
           a drive's contact log as static SurfaceImprint plates (time-varying ruts are not implemented upstream).
  overlay  appearance_overlay.usda: the field-driven material only, bound to this simulator's own terrain meshes, so
           the live wheel tracks keep deforming those meshes while driving.
Project choices applied to the upstream quality preset (no file in the submodule is edited): render-only rocks OFF and
micro-craters OFF (this simulator has collided rocks and a physical crater population; render-only ones would be
ghosts the wheels pass through), dust OFF, no collider (physics stays in the simulator).
Output: <scene>/regolith_appearance/<mode>_<quality>[_<tag>]/ with provenance.json (both lunar_regolith commits).
Everything here is SYNTHETIC soil -> ASSUMED appearance mapping (lunar_regolith config/appearance.yaml).
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

PROJECT = Path(os.environ.get("LUNAR_ROVER_SIM_PROJECT", Path(__file__).resolve().parents[1]))
SUBMODULE = PROJECT / "external" / "lunar_regolith"
PROJECT_PRESET_OVERRIDES = {"rocks": False, "microcraters": False, "dust": False}


def git_state(directory: Path) -> dict:
    """Commit and dirty flag of a git working tree."""
    def run(*arguments):
        """Run git in ``directory`` and return stripped stdout ('' on failure)."""
        result = subprocess.run(["git", "-C", str(directory), *arguments], capture_output=True, text=True)
        return result.stdout.strip() if result.returncode == 0 else ""
    return {"path": str(directory), "commit": run("rev-parse", "HEAD"),
            "dirty": bool(run("status", "--porcelain", "--untracked-files=no"))}


def workspace_surface(scene):
    """The simulator's physics surface (workspace collision lattice) as a lunar_regolith GridSurface."""
    from lunar_regolith.appearance.geometry import GridSurface
    grid = scene.layout.workspace_collision_grid
    return GridSurface(east=grid.east_coordinates(), north=grid.north_coordinates(),
                       z=scene.workspace_collision_height_m.astype(np.float64))


def terrain_mesh_paths(scene) -> list[str]:
    """Render mesh paths authored by runtime/stage_assembly/scene_stage_builder.py (overlay targets)."""
    layout = scene.layout
    tiles = (layout.workspace_visual_grid.count_east - 1) // layout.visual_tile_cells
    paths = [f"/World/Terrain/Workspace/Visual/tile_{row:02d}_{column:02d}" for row in range(tiles) for column in range(tiles)]
    paths.append("/World/Terrain/NearRing")
    if layout.regional_grid is not None:
        paths.append("/World/Terrain/Regional")
    return paths


def imprints_from_contact_log(path: Path, wheel_width_m: float, minimum_spacing_m: float):
    """SurfaceImprint plates along each wheel's track (radius = half the tread width, depth = smoothed sinkage)."""
    from lunar_regolith.contact.deformation import DeformationField, SurfaceImprint
    log = np.load(path)
    radius = wheel_width_m / 2
    imprints = []
    for wheel in np.unique(log["wheel_index"]):
        rows = np.flatnonzero(log["wheel_index"] == wheel)
        # Plates are circles: neighbours with different depths leave ring-shaped steps, so the sampled sinkage is
        # smoothed along the track (running median over ~15 cm of travel) before it becomes plate depth.
        sinkage = np.asarray(log["sinkage_m"][rows], dtype=np.float64)
        window = 6
        padded = np.pad(sinkage, window, mode="edge")
        smoothed = np.array([np.median(padded[k:k + 2 * window + 1]) for k in range(len(sinkage))])
        last = None
        for position, row in enumerate(rows):
            point = np.array([log["contact_east_m"][row], log["contact_north_m"][row]])
            if last is not None and np.hypot(*(point - last)) < minimum_spacing_m:
                continue
            last = point
            imprints.append(SurfaceImprint(east_m=float(point[0]), north_m=float(point[1]), radius_m=float(radius),
                                           sinkage_m=float(smoothed[position]), time_s=float(log["time_s"][row]),
                                           source=f"lunar_rover_sim wheel {int(wheel)} (Bekker sinkage on SYNTHETIC soil)"))
    return DeformationField(imprints=imprints)


def main() -> int:
    """Parse arguments, build the requested appearance, write provenance."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--preset", default="gpu_8gb", help="simulator preset (selects the baked scene)")
    parser.add_argument("--scene-directory", default=None)
    parser.add_argument("--mode", choices=["scene", "overlay"], default="scene")
    parser.add_argument("--quality", choices=["realtime", "balanced", "offline"], default="balanced",
                        help="lunar_regolith quality preset")
    parser.add_argument("--contact-log", type=Path, default=None, help="contact_log.npz of a drive -> static ruts (scene mode)")
    parser.add_argument("--focus", type=float, nargs=2, default=None, help="refined patch centre E N (default: spawn)")
    parser.add_argument("--overlay-half-size-m", type=float, default=1000.0)
    parser.add_argument("--tag", default="")
    arguments = parser.parse_args()

    import lunar_regolith
    from lunar_regolith.appearance.fields import load_surface_fields
    from lunar_regolith.appearance.settings import load_appearance_config
    from lunar_regolith.config import load_config

    from lunar_rover_sim.core.configuration.configuration_loader import load_configuration, scene_identifier
    from lunar_rover_sim.core.scene.scene_loader import load_scene
    from lunar_rover_sim.core.storage.data_root import require_data_root

    module_file = Path(lunar_regolith.__file__).resolve()
    if SUBMODULE.resolve() not in module_file.parents:
        raise SystemExit(f"lunar_regolith imported from {module_file}, not the pinned submodule {SUBMODULE}")
    configuration = load_configuration(arguments.preset)
    data_root = require_data_root(None, configuration.storage.data_root, configuration.storage.allow_root_disk,
                                  configuration.storage.minimum_free_gib)
    scene_directory = Path(arguments.scene_directory) if arguments.scene_directory else \
        data_root.scenes_directory / scene_identifier(configuration)
    scene = load_scene(scene_directory, configuration)
    regolith_run = Path(os.environ["REGOLITH_RUN"])
    generator_configuration = load_config(SUBMODULE / "config" / "apollo17_lander_site.yaml")
    appearance = load_appearance_config(SUBMODULE / "config" / "appearance.yaml")
    preset = appearance.presets[arguments.quality].model_copy(update=PROJECT_PRESET_OVERRIDES)
    appearance = appearance.model_copy(update={"presets": {**appearance.presets, arguments.quality: preset}})
    longitude = generator_configuration.contact.site_lon_deg
    latitude = generator_configuration.contact.site_lat_deg
    site = scene.configuration.site
    if abs(longitude - site.origin_longitude_deg) > 1e-9 or abs(latitude - site.origin_latitude_deg) > 1e-9:
        raise SystemExit(f"site origin mismatch: lunar_regolith ({longitude}, {latitude}) vs simulator "
                         f"({site.origin_longitude_deg}, {site.origin_latitude_deg})")
    name = f"{arguments.mode}_{arguments.quality}" + (f"_{arguments.tag}" if arguments.tag else "")
    output = scene_directory / "regolith_appearance" / name
    output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    provenance = {
        "label": "SYNTHETIC soil fields -> ASSUMED appearance mapping (lunar_regolith config/appearance.yaml); render only",
        "lunar_regolith_code": {**git_state(SUBMODULE), "role": "pinned submodule (code + configs)"},
        "lunar_regolith_data": {**git_state(Path(os.environ["REGOLITH_REPOSITORY"])), "run_directory": str(regolith_run),
                                "role": "working copy (data)"},
        "lunar_regolith_version": lunar_regolith.__version__, "mode": arguments.mode, "quality": arguments.quality,
        "project_preset_overrides": PROJECT_PRESET_OVERRIDES, "simulator_scene_id": scene.manifest["scene_id"],
        "site_origin_lon_lat_deg": [longitude, latitude],
    }
    if arguments.mode == "scene":
        from lunar_regolith.appearance.pipeline import SceneSpec, build_scene
        half = scene.layout.workspace_collision_grid.extent_east_m / 2
        fields = load_surface_fields(regolith_run, generator_configuration, longitude, latitude, (0.0, 0.0), half + 4.0)
        fields = fields.crop((-half, half), (-half, half))
        deformation = None
        if arguments.contact_log is not None:
            deformation = imprints_from_contact_log(arguments.contact_log, scene.configuration.rover.wheel_width_m,
                                                    scene.configuration.rover.wheel_width_m / 6)
            provenance["contact_log"] = str(arguments.contact_log)
            provenance["imprint_count"] = len(deformation.imprints)
        focus = tuple(arguments.focus) if arguments.focus else tuple(scene.manifest["spawn_pose"]["position_m"][:2])
        lighting = scene.configuration.lighting
        specification = SceneSpec(name, fields, (float(focus[0]), float(focus[1])), half,
                                  f"lunar_rover_sim workspace of {scene.manifest['scene_id']}", deformation=deformation,
                                  sun=dict(azimuth_deg=lighting.sun_azimuth_deg, elevation_deg=lighting.sun_elevation_deg,
                                           intensity=lighting.sun_base_intensity),
                                  include_collider=False, surface=workspace_surface(scene))
        metadata = build_scene(specification, generator_configuration, appearance, arguments.quality, output)
        provenance["scene_usd"] = str(output / "scene.usd")
        provenance["geometry"] = metadata["geometry"]
        provenance["instances"] = metadata.get("instances")
        provenance["imprint_render_check"] = metadata.get("imprint_render_check")
        print(f"[regolith] scene: refined {metadata['geometry']['near_vertices']:,} vertices, imprint patch "
              f"{metadata['geometry']['imprint_vertices']:,}, clods {metadata.get('instances', {}).get('clods', 0)}", flush=True)
    else:
        from lunar_regolith.appearance.textures import bake_detail_textures, bake_property_maps
        from lunar_regolith.appearance.usd_appearance import write_binding_overlay
        layout = scene.layout
        outer = layout.regional_grid or layout.near_ring_grid
        half = min(outer.extent_east_m / 2, arguments.overlay_half_size_m)
        fields = load_surface_fields(regolith_run, generator_configuration, longitude, latitude, (0.0, 0.0), half)
        textures = {"property": bake_property_maps(fields, appearance, output / "textures", preset.property_texel_m),
                    "detail": bake_detail_textures(appearance, output / "textures", preset.detail_texture_px)}
        mesh_paths = terrain_mesh_paths(scene)
        path = write_binding_overlay(output / "appearance_overlay.usda", mesh_paths, textures, appearance, preset,
                                     {"run_dir": str(regolith_run), "provenance": fields.provenance,
                                      "preset": arguments.quality, "seed": appearance.seed, "synthetic": True})
        provenance["overlay_usda"] = str(path)
        provenance["material_path"] = "/World/Looks/RegolithAppearance"
        provenance["bound_mesh_paths"] = mesh_paths
        provenance["map_half_size_m"] = half
        print(f"[regolith] overlay bound to {len(mesh_paths)} meshes, maps over +-{half:.0f} m", flush=True)
    provenance["build_seconds"] = round(time.perf_counter() - started, 2)
    (output / "provenance.json").write_text(json.dumps(provenance, indent=1, default=str))
    print(f"[regolith] wrote {output}  (lunar_regolith code {provenance['lunar_regolith_code']['commit'][:7]}, "
          f"data {provenance['lunar_regolith_data']['commit'][:7]}{' dirty' if provenance['lunar_regolith_data']['dirty'] else ''})",
          flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
