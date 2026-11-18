"""Which tone / exposure settings change the 8-bit camera output? One view, settings changed at runtime.

    scripts/isaac_run.sh scripts/probe_tone_settings.py [--scene change5_area --seed 107] [--sun-elevation-deg 25]
        [--sun-azimuth-deg 246] [--extra /rtx/some/path=value ...] [--output-directory DIR]

Renders the front-camera view (as scripts/render_shadow_contrast_check.py, Sun to the left) once with the configured
settings, then once per variant: every ``/rtx/post/tonemap/op`` value 0-7, film ISO x4, exposure compensation +2 EV,
white scale / 4, plus any ``--extra`` carb setting. For each it reports the median 8-bit luminance of sunlit ground
and of cast shadow (ray-cast labels) and the mean absolute difference to the baseline image, restoring the configured
settings after every variant. A variant that changes nothing (difference < 0.5 of 255) does not reach the output.
Writes <output>/probe_tone_settings.json and a PNG strip of the variants.
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
parser.add_argument("--sun-elevation-deg", type=float, default=25.0)
parser.add_argument("--sun-azimuth-deg", type=float, default=246.0)
parser.add_argument("--extra", action="append", default=[], help="carb setting path=value to try (repeatable)")
parser.add_argument("--set", action="append", default=[], help="configuration override key=value (repeatable)")
parser.add_argument("--output-directory", default=None)
arguments = parser.parse_args()
configuration = load_configuration(arguments.preset, arguments.scene,
                                   list(arguments.set) + [f"scene_seed={arguments.seed}"])
width, height = configuration.render.camera_resolution_px

from lunar_rover_sim.runtime.application import (EXPOSURE_SETTINGS, exposure_snapshot,  # noqa: E402
                                                 start_simulation_application)

application, data_root = start_simulation_application(configuration, headless=True)

import carb  # noqa: E402
import numpy as np  # noqa: E402
import yaml  # noqa: E402
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
TONEMAP_PATH = EXPOSURE_SETTINGS["tonemap_operator"]


def variants(exposure) -> list[tuple[str, dict]]:
    """(label, {carb path: value}) of every variant to try."""
    found = [(f"tonemap op {operator}", {TONEMAP_PATH: operator}) for operator in range(8)
             if operator != exposure.tonemap_operator]
    found += [("film ISO x4", {EXPOSURE_SETTINGS["film_iso"]: exposure.film_iso * 4.0}),
              ("exposure compensation +2 EV", {EXPOSURE_SETTINGS["exposure_compensation"]: 2.0}),
              ("white scale / 4", {EXPOSURE_SETTINGS["white_scale"]: exposure.white_scale / 4.0})]
    for item in arguments.extra:
        path, raw = item.split("=", 1)
        found.append((f"extra {path}={raw}", {path: yaml.safe_load(raw)}))
    return found


exit_code = 1
try:
    scene = load_scene(data_root.scenes_directory / scene_identifier(configuration), configuration)
    session = RoverDriveSession(application, scene, enable_tracks=False)
    after_application_start = exposure_snapshot()
    session.start()
    render_synchronously(4)
    after_first_render = exposure_snapshot()
    session.reset()
    render_synchronously(4)
    after_episode_reset = exposure_snapshot()
    print(f"[tone] configured exposure {configuration.render.exposure.model_dump()}", flush=True)
    print(f"[tone] read back after application start {after_application_start}", flush=True)
    print(f"[tone] read back after session start and first render {after_first_render}", flush=True)
    print(f"[tone] read back after an episode reset {after_episode_reset}", flush=True)
    set_sun_direction(session.stage, arguments.sun_azimuth_deg, arguments.sun_elevation_deg)
    state = session.rover.read_state()
    heading = math.radians(arguments.sun_azimuth_deg + 90.0)                       # Sun on the left
    forward = np.array([math.sin(heading), math.cos(heading), 0.0])
    eye = state.position_m + 0.30 * forward + np.array([0.0, 0.0, 0.75])
    view = forward * math.cos(math.radians(15.0)) - np.array([0.0, 0.0, math.sin(math.radians(15.0))])
    camera_path = "/World/ToneProbeCamera"
    define_observer_camera(session.stage, camera_path, eye, eye + 10.0 * view, 90.0, (width, height))
    set_pose(session.stage.GetPrimAtPath(camera_path), eye,
             quaternion_from_rotation_matrix(look_at_rotation(eye, eye + 10.0 * view)))
    rig = CameraRig()
    rig.add_camera("view", camera_path, (width, height))
    labels = None
    settings = carb.settings.get_settings()

    def capture() -> np.ndarray:
        """8-bit frame after enough subframes for the settings and the path tracer to settle."""
        render_synchronously(24)
        return rig.read("view")["rgb"].astype(np.float64)

    baseline = capture()
    labels = GroundTruthMaskRenderer(session).render(camera_path, (width, height))
    ground = np.isin(labels.object_mask, GROUND_OBJECT_IDS)
    ground[: height // 2] = False
    sunlit = ground & (labels.shadow_cause == SHADOW_CAUSE_ID["sunlit"])
    shadow = ground & (labels.shadow_cause == SHADOW_CAUSE_ID["cast_by_scene"])

    def describe(label: str, image: np.ndarray, written: dict) -> dict:
        """Statistics of one rendered variant."""
        luminance = image @ LUMINANCE_WEIGHTS
        entry = {"label": label, "settings": {path: settings.get(path) for path in written},
                 "sunlit_8bit_median": float(np.median(luminance[sunlit])),
                 "cast_shadow_8bit_median": float(np.median(luminance[shadow])) if shadow.sum() >= 100 else None,
                 "mean_absolute_difference_to_baseline": float(np.abs(image - baseline).mean())}
        print(f"[tone] {json.dumps(entry)}", flush=True)
        return entry

    report = [describe("configured", baseline, {TONEMAP_PATH: None})]
    tiles = [baseline]
    for label, written in variants(configuration.render.exposure):
        previous = {path: settings.get(path) for path in written}
        for path, value in written.items():
            settings.set(path, value)
        image = capture()
        report.append(describe(label, image, written))
        tiles.append(image)
        for path, value in previous.items():
            settings.set(path, value)
    output_directory = Path(arguments.output_directory or data_root.runs_directory / "probe_tone_settings")
    output_directory.mkdir(parents=True, exist_ok=True)
    (output_directory / "probe_tone_settings.json").write_text(json.dumps(
        {"scene": scene_identifier(configuration), "sun_elevation_deg": arguments.sun_elevation_deg,
         "renderer": configuration.render.renderer, "variants": report}, indent=1))
    strip = np.hstack([np.asarray(Image.fromarray(tile.astype(np.uint8)).resize((width // 3, height // 3)))
                       for tile in tiles])
    Image.fromarray(strip).save(output_directory / "probe_tone_settings.png", optimize=True)
    print(f"[tone] saved {output_directory}", flush=True)
    rig.destroy()
    exit_code = 0
except Exception:
    import traceback
    traceback.print_exc()
    sys.stdout.flush()
finally:
    application.close(exit_code=exit_code)
