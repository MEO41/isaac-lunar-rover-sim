"""Validate render.sun_glare: render the rover's front camera into the low Sun with glare off and at several scales.

    scripts/isaac_run.sh scripts/render_sun_glare_check.py [--preset gpu_8gb] [--scales 0.1,0.3,1.0]
        [--output docs/images/sun_glare_check.png]

The output is one labelled PNG (off + each scale, plus a 4x crop around the Sun) and prints, per image, the number
of near-white pixels around the Sun and the mean brightness of the terrain (glare must not wash out the ground).
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
parser.add_argument("--variants", default="0.15:1000:0,0.3:1000:0,0.6:1000:0",
                    help="scale:cutoff:spectral_samples,... (glare off is always rendered first)")
parser.add_argument("--resolution", default="960x720")
parser.add_argument("--output", default=str(PROJECT / "docs" / "images" / "sun_glare_check.png"))
arguments = parser.parse_args()
configuration = load_configuration(arguments.preset)
width, height = (int(v) for v in arguments.resolution.split("x"))

from lunar_rover_sim.runtime.application import apply_sun_glare, start_simulation_application  # noqa: E402

application, data_root = start_simulation_application(configuration, headless=True)

import numpy as np  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402

from lunar_rover_sim.core.frames.pose_math import camera_intrinsic_matrix, look_at_rotation, sun_direction_w  # noqa: E402
from lunar_rover_sim.core.scene.scene_loader import load_scene  # noqa: E402
from lunar_rover_sim.runtime.driving.rover_drive_session import RoverDriveSession  # noqa: E402
from lunar_rover_sim.runtime.sensors.camera_rig import CameraRig, define_observer_camera  # noqa: E402
from lunar_rover_sim.runtime.stepping import render_synchronously  # noqa: E402

exit_code = 1
try:
    scene = load_scene(data_root.scenes_directory / scene_identifier(configuration), configuration)
    session = RoverDriveSession(application, scene, enable_tracks=False)
    session.start()
    # A camera at the rover's front-camera position, aimed so the Sun sits in the upper third of the frame.
    state = session.rover.read_state()
    lighting = configuration.lighting
    sun = sun_direction_w(lighting.sun_azimuth_deg, lighting.sun_elevation_deg)
    horizontal = np.array([sun[0], sun[1], 0.0]) / np.hypot(sun[0], sun[1])
    eye = state.position_m + 1.0 * horizontal + np.array([0.0, 0.0, 0.75])   # in front of the rover, mast height
    view = horizontal * math.cos(math.radians(5.0)) - np.array([0.0, 0.0, math.sin(math.radians(5.0))])
    target = eye + 10.0 * view
    field_of_view = 90.0
    define_observer_camera(session.stage, "/World/SunCheckCamera", eye, target, field_of_view, (width, height))
    rig = CameraRig()
    rig.add_camera("sun", "/World/SunCheckCamera", (width, height))
    rotation = look_at_rotation(eye, target)
    intrinsics = camera_intrinsic_matrix(width, height, field_of_view)
    sun_camera = rotation.T @ sun
    sun_u = intrinsics[0, 0] * sun_camera[0] / -sun_camera[2] + width / 2
    sun_v = height / 2 - intrinsics[1, 1] * sun_camera[1] / -sun_camera[2]
    print(f"[glare] Sun projects to pixel ({sun_u:.0f}, {sun_v:.0f}); disc diameter "
          f"{intrinsics[0, 0] * math.radians(lighting.sun_angular_diameter_deg):.1f} px", flush=True)

    glare = configuration.render.sun_glare
    variants = [("glare off", glare.model_copy(update={"enabled": False}))]
    for part in arguments.variants.split(","):
        scale, cutoff, samples = part.split(":")
        variants.append((f"glare scale {scale} cutoff {cutoff} spectral {samples}",
                         glare.model_copy(update={"enabled": True, "scale": float(scale), "cutoff": float(cutoff),
                                                  "spectral_samples": int(samples)})))
    tiles, report = [], []
    ground = (slice(int(height * 0.75), height), slice(0, width))
    for label, settings in variants:
        read_back = apply_sun_glare(settings)
        print(f"[glare] settings read back for '{label}': {read_back}", flush=True)
        render_synchronously(16)
        rgb = rig.read("sun")["rgb"]
        luminance = rgb.astype(np.float64).mean(axis=2)
        yy, xx = np.mgrid[0:height, 0:width]
        around_sun = np.hypot(xx - sun_u, yy - sun_v) < 0.25 * width
        entry = {"label": label, "bright_pixels_near_sun": int(np.sum((luminance > 230) & around_sun)),
                 "glow_pixels_near_sun": int(np.sum((luminance > 60) & around_sun & (yy < sun_v + 0.1 * height))),
                 "ground_mean": float(luminance[ground].mean())}
        report.append(entry)
        print(f"[glare] {json.dumps(entry)}", flush=True)
        image = Image.fromarray(rgb)
        crop_half = 60
        left, top = int(sun_u) - crop_half, int(sun_v) - crop_half
        crop = image.crop((left, top, left + 2 * crop_half, top + 2 * crop_half)).resize((240, 240), Image.NEAREST)
        image.paste(crop, (width - 250, 10))
        draw = ImageDraw.Draw(image)
        draw.rectangle([width - 251, 9, width - 9, 251], outline=(255, 80, 80), width=2)
        draw.text((12, 10), label, fill=(255, 220, 80))
        tiles.append(np.asarray(image))
    columns = 2
    rows = [np.hstack(tiles[index:index + columns] + [np.zeros_like(tiles[0])] * (columns - len(tiles[index:index + columns])))
            for index in range(0, len(tiles), columns)]
    output = Path(arguments.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.vstack(rows)).save(output, optimize=True)
    print(f"[glare] saved {output}", flush=True)

    # Facing away from the Sun, glare must change nothing (only the Sun may exceed the bloom cutoff).
    from lunar_rover_sim.core.frames.pose_math import quaternion_from_rotation_matrix
    from lunar_rover_sim.runtime.stage_assembly.usd_helpers import set_pose
    away_target = eye - 10.0 * view
    set_pose(session.stage.GetPrimAtPath("/World/SunCheckCamera"), eye,
             quaternion_from_rotation_matrix(look_at_rotation(eye, away_target)))
    away = {}
    for label, settings in (("off", glare.model_copy(update={"enabled": False})), ("configured", glare)):
        apply_sun_glare(settings)
        render_synchronously(16)
        away[label] = rig.read("sun")["rgb"].astype(np.float64)
    difference = float(np.abs(away["configured"] - away["off"]).mean())
    print(f"[glare] facing away from the Sun, configured glare vs off: mean |difference| {difference:.2f} "
          f"(of 255); image mean {away['off'].mean():.1f} -> {away['configured'].mean():.1f}", flush=True)
    apply_sun_glare(glare)
    rig.destroy()
    exit_code = 0
except Exception:
    import traceback
    traceback.print_exc()
    sys.stdout.flush()
finally:
    application.close(exit_code=exit_code)
