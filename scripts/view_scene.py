"""Drive the rover with the keyboard in an Isaac Sim window.

    scripts/isaac_run.sh --interactive scripts/view_scene.py [--preset gpu_8gb] [--sun-elevation-deg 15]

Keys (click into the viewport first): W/S or Up/Down drive, A/D or Left/Right turn, Shift = x2, X = stop,
R = reset to the spawn pose and clear tracks, T = clear tracks, C = cycle camera (follow -> rover front -> free),
Esc = quit. In the free camera, orbit with Alt + left mouse, pan with the middle mouse, zoom with the wheel.
The scene must be baked first (scripts/bake_scene.py --preset <preset>); this script never bakes.
"""
from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lunar_rover_sim.core.configuration.configuration_loader import load_configuration, scene_identifier  # noqa: E402

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--preset", default="gpu_8gb")
parser.add_argument("--sun-elevation-deg", type=float, default=None)
parser.add_argument("--sun-azimuth-deg", type=float, default=None)
parser.add_argument("--no-tracks", action="store_true")
parser.add_argument("--no-rocks", action="store_true")
arguments = parser.parse_args()
overrides = ["drive.fall_through_policy=recover"]
if arguments.sun_elevation_deg is not None:
    overrides.append(f"lighting.sun_elevation_deg={arguments.sun_elevation_deg}")
if arguments.sun_azimuth_deg is not None:
    overrides.append(f"lighting.sun_azimuth_deg={arguments.sun_azimuth_deg}")
configuration = load_configuration(arguments.preset, overrides=overrides)
scene_directory_name = scene_identifier(configuration)

from lunar_rover_sim.runtime.application import start_simulation_application  # noqa: E402

width, height = configuration.window.viewport_resolution_px
application, data_root = start_simulation_application(configuration, headless=False, width=width, height=height)
scene_directory = data_root.scenes_directory / scene_directory_name
if not (scene_directory / "manifest.json").exists():
    print(f"ERROR: scene {scene_directory} is not baked. Run:\n  $ISAAC_ENV/bin/python scripts/bake_scene.py "
          f"--preset {arguments.preset}", flush=True)
    application.close(exit_code=2)

import numpy as np  # noqa: E402

from lunar_rover_sim.core.frames.pose_math import look_at_rotation, quaternion_from_rotation_matrix, yaw_from_quaternion  # noqa: E402
from lunar_rover_sim.core.scene.scene_loader import load_scene  # noqa: E402
from lunar_rover_sim.runtime.driving.rover_drive_session import RoverDriveSession  # noqa: E402
from lunar_rover_sim.runtime.sensors.camera_rig import define_observer_camera  # noqa: E402
from lunar_rover_sim.runtime.stage_assembly.usd_helpers import set_pose  # noqa: E402
from lunar_rover_sim.runtime.teleoperation.keyboard_twist_device import KeyboardTwistDevice  # noqa: E402

FOLLOW_CAMERA = "/World/FollowCamera"
HELP_TEXT = ("W/S drive   A/D turn   Shift x2   X stop\nR reset + clear tracks   T clear tracks\n"
             "C camera (follow / rover front / free)   Esc quit")


def follow_pose(position: np.ndarray, heading: float):
    """Eye and target of the follow camera behind the rover."""
    forward = np.array([math.cos(heading), math.sin(heading), 0.0])
    eye = position - 3.0 * forward + np.array([0.0, 0.0, 1.5])
    return eye, position + 1.0 * forward + np.array([0.0, 0.0, 0.2])


class StatusPanel:
    """Small omni.ui window with the controls and live rover status."""

    def __init__(self):
        """Build the window (silently disabled if omni.ui is unavailable)."""
        self.label = None
        try:
            import omni.ui as ui
            self.window = ui.Window("Lunar rover", width=420, height=210, position_x=20, position_y=height - 160)
            with self.window.frame:
                with ui.VStack(spacing=6):
                    ui.Label(HELP_TEXT, word_wrap=True)
                    self.label = ui.Label("starting...", word_wrap=True)
        except Exception as error:
            print(f"[view] status panel unavailable: {error}", flush=True)

    def update(self, text: str) -> None:
        """Replace the status text."""
        if self.label is not None:
            self.label.text = text


exit_code = 0
try:
    scene = load_scene(scene_directory, configuration)
    session = RoverDriveSession(application, scene, stepper_mode="application_update",
                                include_rocks=not arguments.no_rocks, enable_tracks=not arguments.no_tracks)
    print("[view] building the scene (terrain colliders cook on first use; this can take a minute)...", flush=True)
    session.start()
    state = session.rover.read_state()
    smoothed_position = state.position_m.copy()
    smoothed_heading = yaw_from_quaternion(state.orientation_xyzw)
    eye, target = follow_pose(smoothed_position, smoothed_heading)
    define_observer_camera(session.stage, FOLLOW_CAMERA, eye, target, 70.0, (width, height))
    follow_prim = session.stage.GetPrimAtPath(FOLLOW_CAMERA)
    from omni.kit.viewport.utility import get_active_viewport
    viewport = get_active_viewport()
    camera_modes = [("follow", FOLLOW_CAMERA), ("rover front", session.rover_paths.cameras["front"]),
                    ("free", "/OmniverseKit_Persp")]
    camera_index = 0
    viewport.camera_path = FOLLOW_CAMERA
    keyboard = KeyboardTwistDevice(configuration.window.keyboard_forward_speed_m_s,
                                   configuration.window.keyboard_yaw_rate_rad_s)
    panel = StatusPanel()
    print("[view] ready. Click into the viewport, then drive with W/A/S/D.\n" + HELP_TEXT, flush=True)
    last_status = 0.0
    blend = 1.0 - math.exp(-1.0 / (60 * 0.35))
    while application.is_running():
        for action in keyboard.take_actions():
            if action == "quit":
                raise KeyboardInterrupt
            if action == "reset":
                session.reset(clear_tracks=True)
                print("[view] reset to the spawn pose, tracks cleared", flush=True)
            elif action == "clear_tracks" and session.track_runtime is not None:
                session.track_runtime.clear()
                print("[view] tracks cleared", flush=True)
            elif action == "camera":
                camera_index = (camera_index + 1) % len(camera_modes)
                viewport.camera_path = camera_modes[camera_index][1]
                print(f"[view] camera: {camera_modes[camera_index][0]}", flush=True)
        recovered = session.ensure_simulation_running()
        if recovered:
            print(f"[view] {recovered}", flush=True)
        forward_speed, yaw_rate = keyboard.twist()
        try:
            result = session.advance_control_step(forward_speed, yaw_rate)
        except Exception as error:
            if "invalidated" not in str(error) and "backend" not in str(error):
                raise
            print(f"[view] physics view lost ({error}); restarting physics", flush=True)
            import omni.timeline
            omni.timeline.get_timeline_interface().stop()
            application.update()
            continue
        heading = yaw_from_quaternion(result.state.orientation_xyzw)
        smoothed_heading += blend * ((heading - smoothed_heading + math.pi) % (2 * math.pi) - math.pi)
        smoothed_position += blend * (result.state.position_m - smoothed_position)
        if camera_modes[camera_index][0] == "follow":
            eye, target = follow_pose(smoothed_position, smoothed_heading)
            set_pose(follow_prim, eye, quaternion_from_rotation_matrix(look_at_rotation(eye, target)))
        now = time.perf_counter()
        if now - last_status > 0.25:
            last_status = now
            position = result.state.position_m
            speed = float(np.linalg.norm(result.state.linear_velocity_m_s[:2]))
            truth = session.track_runtime.sample_builder.latest_ground_truth if session.track_runtime else None
            sinkage = "" if truth is None else f"\nsinkage {np.round(truth['sinkage_m'] * 1000, 1).tolist()} mm"
            edge = "\nAT DRIVABLE-ZONE EDGE: turn back" if result.stopped_at_boundary else ""
            panel.update(f"position E {position[0]:.2f}  N {position[1]:.2f}  z {position[2]:.2f} m\n"
                         f"heading {math.degrees(heading):.0f} deg   speed {speed:.2f} m/s   "
                         f"yaw rate {result.state.angular_velocity_rad_s[2]:.2f} rad/s\n"
                         f"camera: {camera_modes[camera_index][0]}{sinkage}{edge}")
except KeyboardInterrupt:
    print("[view] quit", flush=True)
except Exception:
    import traceback
    traceback.print_exc()
    sys.stdout.flush()
    exit_code = 1
finally:
    application.close(exit_code=exit_code)
