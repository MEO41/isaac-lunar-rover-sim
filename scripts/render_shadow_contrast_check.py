"""Measure shadow contrast: linear (HDR) radiance of shadowed vs sunlit ground, and what the tone curve makes of it.

    scripts/isaac_run.sh scripts/render_shadow_contrast_check.py --scene change5_area --seed 107 \
        [--sun-elevations-deg 10,25,40] [--sun-azimuth-deg 246] [--set render.exposure.tonemap_operator=1] \
        [--label aces] [--without-hdr] [--output-directory DIR]

A camera at the rover's front-camera pose (0.30 m ahead of the spawn position, 0.75 m up, 15 deg down, 90 deg field
of view) looks in 4 headings (Sun to the left, right, behind and ahead) for every Sun elevation. For each view it
reads the 8-bit ``rgb`` and the linear ``HdrColor`` outputs and the simulator's ray-cast ground-truth labels, and
reports, over ground pixels (terrain, crater, rim, track) in the lower half of the frame: median luminance of cast
shadow (cause ``cast_by_scene``), of ground facing away from the Sun and of sunlit ground, and their ratios. A cast
shadow on open ground is lit only by light scattered from nearby sunlit terrain and rocks, so a low HDR ratio is
physical; an 8-bit value near 0 for an HDR ratio well above 0 is the tone curve. Measured 2026-11-18 (Isaac Sim
6.0.1): attaching ``HdrColor`` darkens the same render product's 8-bit ``rgb`` about 5x, so a run reports HDR
statistics (default) or, with ``--without-hdr``, 8-bit statistics of the dataset's own rendering (rgb only).
Writes <output>/<label>.json and <label>.png (8-bit views, one row per Sun elevation).
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))

from lunar_rover_sim.core.configuration.configuration_loader import load_configuration, scene_identifier  # noqa: E402

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--preset", default="gpu_8gb")
parser.add_argument("--scene", default="apollo17_valley_floor")
parser.add_argument("--seed", type=int, default=None)
parser.add_argument("--set", action="append", default=[], help="configuration override key=value (repeatable)")
parser.add_argument("--label", default="shadow_contrast")
parser.add_argument("--sun-elevations-deg", default="10,25,40")
parser.add_argument("--sun-azimuth-deg", type=float, default=90.0)
parser.add_argument("--resolution", default="640x480")
parser.add_argument("--output-directory", default=None)
parser.add_argument("--without-hdr", action="store_true", help="8-bit rgb statistics (no HdrColor annotator)")
arguments = parser.parse_args()
overrides = list(arguments.set)
if arguments.seed is not None:
    overrides.append(f"scene_seed={arguments.seed}")
configuration = load_configuration(arguments.preset, arguments.scene, overrides)
width, height = (int(value) for value in arguments.resolution.split("x"))

from lunar_rover_sim.runtime.application import start_simulation_application  # noqa: E402

application, data_root = start_simulation_application(configuration, headless=True)

import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

from lunar_rover_sim.core.frames.pose_math import look_at_rotation, quaternion_from_rotation_matrix  # noqa: E402
from lunar_rover_sim.core.ground_truth.mask_classes import GROUND_OBJECT_IDS, SHADOW_CAUSE_ID  # noqa: E402
from lunar_rover_sim.core.scene.scene_loader import load_scene  # noqa: E402
from lunar_rover_sim.runtime.driving.rover_drive_session import RoverDriveSession  # noqa: E402
from lunar_rover_sim.runtime.sensors.camera_rig import CameraRig, define_observer_camera  # noqa: E402
from lunar_rover_sim.runtime.sensors.ground_truth_masks import GroundTruthMaskRenderer  # noqa: E402
from lunar_rover_sim.runtime.stage_assembly.scene_stage_builder import set_sun_direction  # noqa: E402
from lunar_rover_sim.runtime.stage_assembly.usd_helpers import set_pose  # noqa: E402
from lunar_rover_sim.runtime.stepping import render_synchronously  # noqa: E402

LUMINANCE_WEIGHTS = np.array([0.2126, 0.7152, 0.0722])


def median_or_none(values: np.ndarray, minimum_count: int = 200):
    """Median of the values, or None when fewer than ``minimum_count`` pixels are available."""
    return float(np.median(values)) if values.size >= minimum_count else None


def view_statistics(rgb: np.ndarray, hdr: np.ndarray | None, object_mask: np.ndarray,
                    shadow_cause: np.ndarray) -> dict:
    """Median luminance of cast shadow, faces-away and sunlit ground (lower half), HDR if given, else 8-bit."""
    ground = np.isin(object_mask, GROUND_OBJECT_IDS)
    ground[: rgb.shape[0] // 2] = False
    kind = "8bit" if hdr is None else "hdr"
    luminance = (rgb if hdr is None else hdr[..., :3]).astype(np.float64) @ LUMINANCE_WEIGHTS
    result = {}
    for name, cause in (("cast_shadow", "cast_by_scene"), ("faces_away", "faces_away"), ("sunlit", "sunlit")):
        pixels = ground & (shadow_cause == SHADOW_CAUSE_ID[cause])
        result[f"{name}_pixels"] = int(pixels.sum())
        result[f"{name}_{kind}_median"] = median_or_none(luminance[pixels])
    lit = result[f"sunlit_{kind}_median"]
    for name in ("cast_shadow", "faces_away"):
        value = result[f"{name}_{kind}_median"]
        result[f"{name}_to_sunlit_{kind}_ratio"] = value / lit if value is not None and lit else None
    return result


exit_code = 1
try:
    identifier = scene_identifier(configuration)
    scene = load_scene(data_root.scenes_directory / identifier, configuration)
    session = RoverDriveSession(application, scene, enable_tracks=False)
    session.start()
    state = session.rover.read_state()
    camera_path = "/World/ShadowContrastCamera"
    define_observer_camera(session.stage, camera_path, state.position_m + np.array([0.0, 0.0, 0.75]),
                           state.position_m + np.array([0.0, 1.0, 0.75]), 90.0, (width, height))
    rig = CameraRig()
    rig.add_camera("view", camera_path, (width, height),
                   annotator_names=("rgb",) if arguments.without_hdr else ("rgb", "HdrColor"))
    masks = GroundTruthMaskRenderer(session)
    report, rows = [], []
    for elevation_deg in (float(value) for value in arguments.sun_elevations_deg.split(",")):
        set_sun_direction(session.stage, arguments.sun_azimuth_deg, elevation_deg)
        tiles = []
        for relative_deg, name in ((-90, "sun_left"), (90, "sun_right"), (0, "down_sun"), (180, "up_sun")):
            heading = math.radians(arguments.sun_azimuth_deg + 180.0 + relative_deg)   # 0 = facing away from the Sun
            forward = np.array([math.sin(heading), math.cos(heading), 0.0])
            eye = state.position_m + 0.30 * forward + np.array([0.0, 0.0, 0.75])
            view = forward * math.cos(math.radians(15.0)) - np.array([0.0, 0.0, math.sin(math.radians(15.0))])
            set_pose(session.stage.GetPrimAtPath(camera_path), eye,
                     quaternion_from_rotation_matrix(look_at_rotation(eye, eye + 10.0 * view)))
            render_synchronously(24)
            output = rig.read("view")
            hdr = None if arguments.without_hdr else np.asarray(output["HdrColor"], dtype=np.float32)
            labels = masks.render(camera_path, (width, height))
            entry = {"sun_elevation_deg": elevation_deg, "view": name,
                     **view_statistics(output["rgb"], hdr, labels.object_mask, labels.shadow_cause)}
            report.append(entry)
            print(f"[shadow] {json.dumps(entry)}", flush=True)
            tiles.append(np.asarray(Image.fromarray(output["rgb"]).resize((width // 2, height // 2), Image.LANCZOS)))
        rows.append(np.hstack(tiles))
    output_directory = Path(arguments.output_directory or data_root.runs_directory / "render_shadow_contrast_check")
    output_directory.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.vstack(rows)).save(output_directory / f"{arguments.label}.png", optimize=True)
    (output_directory / f"{arguments.label}.json").write_text(json.dumps(
        {"label": arguments.label, "scene": identifier, "overrides": overrides,
         "exposure": configuration.render.exposure.model_dump(), "views": report}, indent=1))
    print(f"[shadow] saved {output_directory / (arguments.label + '.json')}", flush=True)
    rig.destroy()
    exit_code = 0
except Exception:
    import traceback
    traceback.print_exc()
    sys.stdout.flush()
finally:
    application.close(exit_code=exit_code)
