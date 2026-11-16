"""Render the same views of a driven scene under one regolith appearance (one appearance per Isaac process).

    scripts/isaac_run.sh scripts/render_appearance_views.py --label native [--set appearance.source=...] \
        [--twists "3:0.4:0,3:0.35:0.35"] [--save-contact-log] [--output-directory DIR]

Drives a deterministic route from the spawn pose (wheel ruts form live in native/overlay modes; scene mode shows the
static imprints the appearance was built with), parks the rover out of view, renders three views (oblique, close,
low grazing) and writes <label>_<view>.png plus <label>.json (appearance provenance incl. lunar_regolith commits,
timings, image statistics). --save-contact-log writes contact_log.npz for scripts/build_regolith_appearance.py.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lunar_rover_sim.core.configuration.configuration_loader import load_configuration, scene_identifier  # noqa: E402

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--preset", default="gpu_8gb")
parser.add_argument("--label", required=True)
parser.add_argument("--set", dest="overrides", action="append", default=[])
parser.add_argument("--twists", default="3:0.4:0.0,3:0.35:0.35")
parser.add_argument("--save-contact-log", action="store_true")
parser.add_argument("--output-directory", default=None)
parser.add_argument("--resolution", default="960x540")
parser.add_argument("--subframes", type=int, default=16)
arguments = parser.parse_args()
configuration = load_configuration(arguments.preset, overrides=arguments.overrides)
width, height = (int(v) for v in arguments.resolution.split("x"))

from lunar_rover_sim.runtime.application import process_video_memory_mib, start_simulation_application  # noqa: E402

application, data_root = start_simulation_application(configuration, headless=True)

import numpy as np  # noqa: E402

from lunar_rover_sim.core.scene.scene_loader import load_scene  # noqa: E402
from lunar_rover_sim.runtime.driving.rover_drive_session import RoverDriveSession  # noqa: E402
from lunar_rover_sim.runtime.sensors.camera_rig import CameraRig, define_observer_camera, save_png  # noqa: E402
from lunar_rover_sim.runtime.stepping import render_synchronously  # noqa: E402

exit_code = 1
try:
    output = Path(arguments.output_directory) if arguments.output_directory else \
        data_root.run_directory("appearance_views", time.strftime("%Y%m%d"))
    output.mkdir(parents=True, exist_ok=True)
    scene = load_scene(data_root.scenes_directory / scene_identifier(configuration), configuration)
    started = time.perf_counter()
    session = RoverDriveSession(application, scene, stepper_mode="explicit", enable_tracks=True)
    report = session.start()
    build_seconds = time.perf_counter() - started
    spawn = np.array(session.spawn_pose.position_m)
    for part in arguments.twists.split(","):
        duration, speed, yaw_rate = (float(v) for v in part.split(":"))
        for _ in range(int(round(duration * configuration.drive.control_rate_hz))):
            session.advance_control_step(speed, yaw_rate)
    if arguments.save_contact_log:
        session.track_runtime.save_contact_log(output / "contact_log.npz")
    session.track_runtime.sampling_enabled = False
    park = session.support_pose(float(spawn[0] - 3.0), float(spawn[1] + 3.5), 0.0)
    session.rover.teleport(np.array(park.position_m) + [0, 0, 0.01], park.orientation_xyzw)
    session.stepper.advance_physics_steps(60)

    views = {"oblique": (spawn + [-1.0, -3.2, 1.6], spawn + [1.6, 0.4, 0.0], 70.0),
             "close": (spawn + [0.9, -1.1, 0.55], spawn + [1.5, 0.05, 0.0], 60.0),
             "grazing": (spawn + [-1.5, -0.6, 0.35], spawn + [3.0, 0.6, 0.0], 65.0)}
    rig = CameraRig()
    for name, (eye, target, field_of_view) in views.items():
        eye = eye.copy()
        eye[2] = float(session.collider_height_m(np.array([eye[0]]), np.array([eye[1]]))[0]) + (eye[2] - spawn[2] + 0.15)
        define_observer_camera(session.stage, f"/World/View_{name}", eye, target, field_of_view, (width, height))
        rig.add_camera(name, f"/World/View_{name}", (width, height))
    if session.track_runtime is not None:
        session.track_runtime.force_mask_upload()
    render_started = time.perf_counter()
    render_synchronously(arguments.subframes)
    render_seconds = time.perf_counter() - render_started
    statistics = {}
    for name in views:
        rgb = rig.read(name)["rgb"]
        save_png(output / f"{arguments.label}_{name}.png", rgb)
        luminance = rgb.astype(np.float64).mean(axis=2)
        statistics[name] = {"mean": float(luminance.mean()), "standard_deviation": float(luminance.std())}
    result = {"label": arguments.label, "appearance": configuration.appearance.model_dump(),
              "appearance_provenance": session.terrain_handles.regolith_appearance,
              "stage_and_start_seconds": round(build_seconds, 1), "synchronous_render_seconds": round(render_seconds, 2),
              "video_memory_mib": process_video_memory_mib(), "image_statistics": statistics,
              "track_field": session.track_runtime.field.rut_depth_statistics(),
              "settled_height_error_m": report.settled_height_error_m}
    (output / f"{arguments.label}.json").write_text(json.dumps(result, indent=1, default=str))
    print(f"[views] {arguments.label}: {json.dumps({k: result[k] for k in ('stage_and_start_seconds', 'synchronous_render_seconds', 'video_memory_mib')})} -> {output}", flush=True)
    rig.destroy()
    exit_code = 0
except Exception:
    import traceback
    traceback.print_exc()
    sys.stdout.flush()
finally:
    application.close(exit_code=exit_code)
