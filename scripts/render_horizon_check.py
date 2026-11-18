"""Render the horizon all around the spawn point at rover-camera height (backdrop / distant-terrain check).

    scripts/isaac_run.sh scripts/render_horizon_check.py --scene change5_area --seed 107 --label sldem \
        [--set terrain.backdrop.enabled=false --set site.dem_backdrop_path=null] [--sun-elevation-deg 25] \
        [--sun-azimuth-deg 90] [--pitch-down-deg 15] [--output-directory DIR]

A camera 0.75 m above the rover's spawn position (the front camera's mast height; the rover is hidden), 90 deg
horizontal field of view, looks in 8 headings (0 = north, clockwise) at ``--pitch-down-deg`` (15 = the front camera)
and level. Writes <output>/<label>.png (two rows of 8 views) and <label>.json: per view, the image row of the
terrain/sky boundary (median over columns of the first non-black row; black = luminance <= 2 of 255) and the sky
fraction. The scene must be baked with the same --set overrides (scripts/bake_scene.py); this script never bakes.
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
parser.add_argument("--label", required=True)
parser.add_argument("--sun-elevation-deg", type=float, default=25.0)
parser.add_argument("--sun-azimuth-deg", type=float, default=90.0)
parser.add_argument("--pitch-down-deg", type=float, default=15.0)
parser.add_argument("--resolution", default="640x480")
parser.add_argument("--output-directory", default=None)
arguments = parser.parse_args()
overrides = list(arguments.set) + [f"lighting.sun_elevation_deg={arguments.sun_elevation_deg}",
                                   f"lighting.sun_azimuth_deg={arguments.sun_azimuth_deg}"]
if arguments.seed is not None:
    overrides.append(f"scene_seed={arguments.seed}")
configuration = load_configuration(arguments.preset, arguments.scene, overrides)
width, height = (int(value) for value in arguments.resolution.split("x"))

from lunar_rover_sim.runtime.application import start_simulation_application  # noqa: E402

application, data_root = start_simulation_application(configuration, headless=True)

import numpy as np  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402

from lunar_rover_sim.core.frames.pose_math import look_at_rotation, quaternion_from_rotation_matrix  # noqa: E402
from lunar_rover_sim.core.scene.scene_loader import load_scene  # noqa: E402
from lunar_rover_sim.runtime.driving.rover_drive_session import RoverDriveSession  # noqa: E402
from lunar_rover_sim.runtime.sensors.camera_rig import CameraRig, define_observer_camera  # noqa: E402
from lunar_rover_sim.runtime.stage_assembly.usd_helpers import set_pose  # noqa: E402
from lunar_rover_sim.runtime.stepping import render_synchronously  # noqa: E402


def horizon_statistics(rgb: np.ndarray) -> dict:
    """Median first non-black row over the columns (terrain/sky boundary) and the black (sky) fraction."""
    luminance = rgb.astype(np.float64).mean(axis=2)
    lit = luminance > 2.0
    first_rows = np.where(lit.any(axis=0), lit.argmax(axis=0), rgb.shape[0])
    return {"horizon_row_median": float(np.median(first_rows)), "horizon_row_minimum": int(first_rows.min()),
            "sky_fraction": float(1.0 - lit.mean())}


exit_code = 1
try:
    identifier = scene_identifier(configuration)
    scene = load_scene(data_root.scenes_directory / identifier, configuration)
    session = RoverDriveSession(application, scene, enable_tracks=False)
    session.start()
    state = session.rover.read_state()
    eye = state.position_m + np.array([0.0, 0.0, 0.75])
    from pxr import UsdGeom
    UsdGeom.Imageable(session.stage.GetPrimAtPath("/World/Rover")).MakeInvisible()   # the camera sits inside the rover
    camera_path = "/World/HorizonCheckCamera"
    define_observer_camera(session.stage, camera_path, eye, eye + np.array([0.0, 1.0, 0.0]), 90.0, (width, height))
    rig = CameraRig()
    rig.add_camera("horizon", camera_path, (width, height))
    rows, report = [], []
    for pitch_down_deg in (arguments.pitch_down_deg, 0.0):
        tiles = []
        for heading_deg in range(0, 360, 45):
            heading = math.radians(heading_deg)
            pitch = math.radians(pitch_down_deg)
            view = np.array([math.sin(heading) * math.cos(pitch), math.cos(heading) * math.cos(pitch),
                             -math.sin(pitch)])
            set_pose(session.stage.GetPrimAtPath(camera_path), eye,
                     quaternion_from_rotation_matrix(look_at_rotation(eye, eye + 10.0 * view)))
            render_synchronously(16)
            rgb = rig.read("horizon")["rgb"]
            entry = {"heading_deg": heading_deg, "pitch_down_deg": pitch_down_deg, **horizon_statistics(rgb)}
            report.append(entry)
            image = Image.fromarray(rgb)
            ImageDraw.Draw(image).text((8, 8), f"heading {heading_deg} deg, pitch -{pitch_down_deg:.0f} deg",
                                       fill=(255, 220, 80))
            tiles.append(np.asarray(image.resize((width // 2, height // 2), Image.LANCZOS)))
        rows.append(np.hstack(tiles))
    output_directory = Path(arguments.output_directory or data_root.runs_directory / "render_horizon_check")
    output_directory.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.vstack(rows)).save(output_directory / f"{arguments.label}.png", optimize=True)
    summary = {"label": arguments.label, "scene": identifier, "overrides": overrides, "eye_m": eye.tolist(),
               "views": report}
    (output_directory / f"{arguments.label}.json").write_text(json.dumps(summary, indent=1))
    for entry in report:
        print(f"[horizon] {json.dumps(entry)}", flush=True)
    print(f"[horizon] saved {output_directory / (arguments.label + '.png')}", flush=True)
    rig.destroy()
    exit_code = 0
except Exception:
    import traceback
    traceback.print_exc()
    sys.stdout.flush()
finally:
    application.close(exit_code=exit_code)
