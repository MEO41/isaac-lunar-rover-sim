"""Compare camera models (tone curve + exposure rule) on one view at several Sun elevations, with shadow statistics.

    scripts/isaac_run.sh scripts/render_camera_model_comparison.py [--scene change5_area --seed 107]
        [--sun-elevations-deg 10,25,40] [--sun-azimuth-deg 246] [--reference-elevation-deg 25] [--output-directory DIR]

Models (all settings changed at runtime in one Isaac process, front-camera pose, Sun to the left):
* ``aces_fixed``: the v1.0.8 camera, RTX tone-map operator 6 (ACES filmic), film ISO 100 at every Sun elevation;
* ``linear_fixed``: operator 1 (linear radiance, then the sRGB transfer curve, as a linear sensor), ISO 100;
* ``linear_metered``: operator 1, ISO 100 x sin(reference elevation) / sin(elevation): exposure follows the flat-ground
  irradiance so sunlit regolith keeps one brightness, as a camera with auto exposure would (deterministic per Sun).
Reports per model and elevation the median 8-bit luminance of sunlit ground, of cast shadow and of ground facing
away from the Sun (ray-cast labels, lower half of the frame) and the fraction of clipped (>= 254) ground pixels.
Writes <output>/camera_models.png (rows = models, columns = elevations) and camera_models.json.
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
parser.add_argument("--scene", default="change5_area")
parser.add_argument("--seed", type=int, default=107)
parser.add_argument("--sun-elevations-deg", default="10,25,40")
parser.add_argument("--sun-azimuth-deg", type=float, default=246.0)
parser.add_argument("--reference-elevation-deg", type=float, default=25.0)
parser.add_argument("--output-directory", default=None)
arguments = parser.parse_args()
configuration = load_configuration(arguments.preset, arguments.scene, [f"scene_seed={arguments.seed}"])
width, height = configuration.render.camera_resolution_px

from lunar_rover_sim.runtime.application import EXPOSURE_SETTINGS, start_simulation_application  # noqa: E402

application, data_root = start_simulation_application(configuration, headless=True)

import carb  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402

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


def model_settings(model: str, elevation_deg: float) -> dict:
    """Tone-map operator and film ISO of a camera model at one Sun elevation."""
    if model == "aces_fixed":
        return {"tonemap_operator": 6, "film_iso": 100.0}
    if model == "linear_fixed":
        return {"tonemap_operator": 1, "film_iso": 100.0}
    gain = math.sin(math.radians(arguments.reference_elevation_deg)) / math.sin(math.radians(elevation_deg))
    return {"tonemap_operator": 1, "film_iso": 100.0 * gain}


def median_or_none(values: np.ndarray):
    """Median, or None for fewer than 100 pixels."""
    return float(np.median(values)) if values.size >= 100 else None


exit_code = 1
try:
    scene = load_scene(data_root.scenes_directory / scene_identifier(configuration), configuration)
    session = RoverDriveSession(application, scene, enable_tracks=False)
    session.start()
    state = session.rover.read_state()
    heading = math.radians(arguments.sun_azimuth_deg + 90.0)                       # Sun on the left
    forward = np.array([math.sin(heading), math.cos(heading), 0.0])
    eye = state.position_m + 0.30 * forward + np.array([0.0, 0.0, 0.75])
    view = forward * math.cos(math.radians(15.0)) - np.array([0.0, 0.0, math.sin(math.radians(15.0))])
    camera_path = "/World/CameraModelCamera"
    define_observer_camera(session.stage, camera_path, eye, eye + 10.0 * view, 90.0, (width, height))
    set_pose(session.stage.GetPrimAtPath(camera_path), eye,
             quaternion_from_rotation_matrix(look_at_rotation(eye, eye + 10.0 * view)))
    rig = CameraRig()
    rig.add_camera("view", camera_path, (width, height))
    masks = GroundTruthMaskRenderer(session)
    settings = carb.settings.get_settings()
    elevations = [float(value) for value in arguments.sun_elevations_deg.split(",")]
    models = ("aces_fixed", "linear_fixed", "linear_metered")
    report, rows = [], {model: [] for model in models}
    for elevation_deg in elevations:
        set_sun_direction(session.stage, arguments.sun_azimuth_deg, elevation_deg)
        render_synchronously(8)
        labels = masks.render(camera_path, (width, height))
        ground = np.isin(labels.object_mask, GROUND_OBJECT_IDS)
        ground[: height // 2] = False
        classes = {name: ground & (labels.shadow_cause == SHADOW_CAUSE_ID[cause])
                   for name, cause in (("sunlit", "sunlit"), ("cast_shadow", "cast_by_scene"),
                                       ("faces_away", "faces_away"))}
        for model in models:
            for name, value in model_settings(model, elevation_deg).items():
                settings.set(EXPOSURE_SETTINGS[name], value)
            render_synchronously(24)
            rgb = rig.read("view")["rgb"]
            luminance = rgb.astype(np.float64) @ LUMINANCE_WEIGHTS
            entry = {"model": model, "sun_elevation_deg": elevation_deg, **model_settings(model, elevation_deg),
                     **{f"{name}_8bit_median": median_or_none(luminance[pixels]) for name, pixels in classes.items()},
                     "clipped_ground_fraction": float((luminance[ground] >= 254).mean())}
            report.append(entry)
            print(f"[camera] {json.dumps(entry)}", flush=True)
            image = Image.fromarray(rgb)
            ImageDraw.Draw(image).text((8, 8), f"{model}, Sun {elevation_deg:.0f} deg, ISO "
                                       f"{entry['film_iso']:.0f}", fill=(255, 220, 80))
            rows[model].append(np.asarray(image.resize((width // 2, height // 2), Image.LANCZOS)))
    from lunar_rover_sim.runtime.application import apply_exposure
    apply_exposure(configuration.render.exposure)
    output_directory = Path(arguments.output_directory or data_root.runs_directory / "render_camera_model_comparison")
    output_directory.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.vstack([np.hstack(rows[model]) for model in models])).save(
        output_directory / "camera_models.png", optimize=True)
    (output_directory / "camera_models.json").write_text(json.dumps(
        {"scene": scene_identifier(configuration), "reference_elevation_deg": arguments.reference_elevation_deg,
         "results": report}, indent=1))
    print(f"[camera] saved {output_directory}", flush=True)
    rig.destroy()
    exit_code = 0
except Exception:
    import traceback
    traceback.print_exc()
    sys.stdout.flush()
finally:
    application.close(exit_code=exit_code)
