"""Render a short drive as videos (Isaac, headless): a chase camera and the rover's front camera.

    scripts/isaac_run.sh scripts/render_drive_video.py [--preset gpu_8gb] [--twists "2:0.4:0,3:0.35:0.35,2:0.4:0"]
        [--fps 30] [--resolution 1280x720] [--sun-elevation-deg 15] [--sun-azimuth-deg 90]

Writes <data root>/runs/render_drive_video/<timestamp>/{chase.mp4, front.mp4, frames/, last_chase.png, result.json}.
Physics runs at 120 Hz; one frame is rendered every 1/fps s of simulated time (control rate = fps).
"""
from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lunar_rover_sim.core.configuration.configuration_loader import load_configuration, scene_identifier  # noqa: E402

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--preset", default="gpu_8gb")
parser.add_argument("--twists", default="2:0.4:0.0,3:0.35:0.35,2:0.4:0.0", help="duration_s:speed_m_s:yaw_rate_rad_s,...")
parser.add_argument("--fps", type=int, default=30)
parser.add_argument("--resolution", default="1280x720")
parser.add_argument("--sun-elevation-deg", type=float, default=None)
parser.add_argument("--sun-azimuth-deg", type=float, default=None)
parser.add_argument("--subframes", type=int, default=2)
arguments = parser.parse_args()
overrides = [f"drive.control_rate_hz={arguments.fps}"]
if arguments.sun_elevation_deg is not None:
    overrides.append(f"lighting.sun_elevation_deg={arguments.sun_elevation_deg}")
if arguments.sun_azimuth_deg is not None:
    overrides.append(f"lighting.sun_azimuth_deg={arguments.sun_azimuth_deg}")
configuration = load_configuration(arguments.preset, overrides=overrides)
width, height = (int(v) for v in arguments.resolution.split("x"))

from lunar_rover_sim.runtime.application import start_simulation_application  # noqa: E402

application, data_root = start_simulation_application(configuration, headless=True)

import numpy as np  # noqa: E402

from lunar_rover_sim.core.frames.pose_math import look_at_rotation, quaternion_from_rotation_matrix, yaw_from_quaternion  # noqa: E402
from lunar_rover_sim.core.scene.scene_loader import load_scene  # noqa: E402
from lunar_rover_sim.runtime.driving.rover_drive_session import RoverDriveSession  # noqa: E402
from lunar_rover_sim.runtime.sensors.camera_rig import CameraRig, define_observer_camera, save_png  # noqa: E402
from lunar_rover_sim.runtime.stage_assembly.usd_helpers import set_pose  # noqa: E402
from lunar_rover_sim.runtime.stepping import render_synchronously  # noqa: E402


def encode(frame_pattern: Path, output: Path, fps: int) -> bool:
    """Encode numbered PNG frames to H.264 MP4 with the system ffmpeg; returns success."""
    if not shutil.which("ffmpeg"):
        print("[video] ffmpeg not found; frames are kept as PNG", flush=True)
        return False
    command = ["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(fps), "-i", str(frame_pattern),
               "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", "-movflags", "+faststart", str(output)]
    return subprocess.run(command).returncode == 0


exit_code = 1
try:
    scene = load_scene(data_root.scenes_directory / scene_identifier(configuration), configuration)
    run_directory = data_root.run_directory("render_drive_video", time.strftime("%Y%m%d_%H%M%S"))
    frames_directory = run_directory / "frames"
    frames_directory.mkdir()
    print(f"[video] run directory {run_directory}", flush=True)
    session = RoverDriveSession(application, scene, stepper_mode="explicit", enable_tracks=True)
    session.start()
    state = session.rover.read_state()
    smoothed_heading = yaw_from_quaternion(state.orientation_xyzw)
    smoothed_position = state.position_m.copy()

    def chase_pose(position: np.ndarray, heading: float):
        """Eye and target of a chase camera behind-left of the rover."""
        forward = np.array([math.cos(heading), math.sin(heading), 0.0])
        left = np.array([-math.sin(heading), math.cos(heading), 0.0])
        eye = position - 2.6 * forward + 1.3 * left + np.array([0.0, 0.0, 1.25])
        return eye, position + 0.8 * forward + np.array([0.0, 0.0, 0.1])

    eye, target = chase_pose(smoothed_position, smoothed_heading)
    define_observer_camera(session.stage, "/World/ChaseCamera", eye, target, 65.0, (width, height))
    rig = CameraRig()
    rig.add_camera("chase", "/World/ChaseCamera", (width, height))
    front_resolution = tuple(configuration.render.camera_resolution_px)
    rig.add_camera("front", session.rover_paths.cameras["front"], front_resolution)
    chase_prim = session.stage.GetPrimAtPath("/World/ChaseCamera")
    render_synchronously(16)

    segments = [tuple(float(v) for v in part.split(":")) for part in arguments.twists.split(",")]
    frame = 0
    started = time.perf_counter()
    blend = 1.0 - math.exp(-1.0 / (arguments.fps * 0.4))
    for duration, speed, yaw_rate in segments:
        for _ in range(int(round(duration * arguments.fps))):
            result = session.advance_control_step(speed, yaw_rate)
            heading = yaw_from_quaternion(result.state.orientation_xyzw)
            smoothed_heading += blend * ((heading - smoothed_heading + math.pi) % (2 * math.pi) - math.pi)
            smoothed_position += blend * (result.state.position_m - smoothed_position)
            eye, target = chase_pose(smoothed_position, smoothed_heading)
            set_pose(chase_prim, eye, quaternion_from_rotation_matrix(look_at_rotation(eye, target)))
            session.track_runtime.force_mask_upload()
            render_synchronously(arguments.subframes)
            save_png(frames_directory / f"chase_{frame:05d}.png", rig.read("chase")["rgb"])
            save_png(frames_directory / f"front_{frame:05d}.png", rig.read("front")["rgb"])
            frame += 1
    elapsed = time.perf_counter() - started
    shutil.copy(frames_directory / f"chase_{frame - 1:05d}.png", run_directory / "last_chase.png")
    encoded = {name: encode(frames_directory / f"{name}_%05d.png", run_directory / f"{name}.mp4", arguments.fps)
               for name in ("chase", "front")}
    final = session.rover.read_state()
    result = {"frames": frame, "fps": arguments.fps, "simulated_seconds": frame / arguments.fps,
              "wall_seconds": round(elapsed, 1), "encoded": encoded, "resolution": [width, height],
              "front_resolution": list(front_resolution), "end_position_m": final.position_m.tolist(),
              "track_field": session.track_runtime.field.rut_depth_statistics(),
              "sun": {"azimuth_deg": configuration.lighting.sun_azimuth_deg,
                      "elevation_deg": configuration.lighting.sun_elevation_deg}}
    (run_directory / "result.json").write_text(json.dumps(result, indent=1))
    print(f"[video] {json.dumps(result)}", flush=True)
    rig.destroy()
    exit_code = 0 if all(encoded.values()) else 1
except Exception:
    import traceback
    traceback.print_exc()
    sys.stdout.flush()
finally:
    application.close(exit_code=exit_code)
