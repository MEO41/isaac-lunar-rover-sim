"""Determinism check (acceptance test T1 of the VLA handoff): same start, same commands -> same trajectory?

    scripts/isaac_run.sh tests/isaac/check_determinism.py [--preset gpu_8gb] [--steps 150] [--frame-every 15]
        [--compare-with <trajectory.npz of an earlier process>]

In one Isaac process the rover drives a fixed command sequence (forward speed 0.35 m/s, yaw rate 0.4 sin(2 pi t / 8 s))
from the spawn pose twice, with a session reset (tracks cleared) in between. Recorded: root pose every control step,
wheel joint angles, the front camera image and the ray-cast object mask and depth every ``frame-every`` steps, and the
final wheel-track field. Also measured: the render noise floor (the same state rendered twice without stepping).
``--compare-with`` additionally compares this process's first run of each reset mode with the same mode in a file
written by an earlier process
(determinism across an Isaac restart). Writes trajectory.npz and result.json to
<data root>/runs/check_determinism/<timestamp>/.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT / "src"))

from lunar_rover_sim.core.configuration.configuration_loader import load_configuration, scene_identifier  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("--preset", default="gpu_8gb")
parser.add_argument("--steps", type=int, default=150)
parser.add_argument("--frame-every", type=int, default=15)
parser.add_argument("--compare-with", default=None)
parser.add_argument("--reset-modes", default="teleport,full_state,timeline_restart",
                    help="comma-separated reset modes; each is run twice in this process and compared")
arguments = parser.parse_args()
configuration = load_configuration(arguments.preset)
from lunar_rover_sim.runtime.application import start_simulation_application  # noqa: E402

application, data_root = start_simulation_application(configuration, headless=True)

import numpy as np  # noqa: E402

from lunar_rover_sim.core.frames.pose_math import yaw_from_quaternion  # noqa: E402
from lunar_rover_sim.core.scene.scene_loader import load_scene  # noqa: E402
from lunar_rover_sim.runtime.driving.rover_drive_session import RoverDriveSession  # noqa: E402
from lunar_rover_sim.runtime.sensors.camera_rig import CameraRig  # noqa: E402
from lunar_rover_sim.runtime.sensors.ground_truth_masks import GroundTruthMaskRenderer  # noqa: E402
from lunar_rover_sim.runtime.stepping import render_synchronously  # noqa: E402


def command(step: int, period_s: float) -> tuple[float, float]:
    """The fixed command of a control step: (forward speed m/s, yaw rate rad/s)."""
    return 0.35, 0.4 * math.sin(2 * math.pi * step * period_s / 8.0)


def drive_once(session, rig, mask_renderer, camera_path, resolution, steps: int, frame_every: int,
               reset_mode: str) -> dict:
    """Reset to the spawn pose (tracks cleared), drive the fixed sequence, record everything compared later."""
    session.reset(clear_tracks=True, mode=reset_mode)
    period = session.stepper.control_period_s
    start_wheel_angles = session.rover.read_state().wheel_angle_rad.copy()
    poses, wheel_angles, frames, object_masks, depths = [], [], [], [], []
    for step in range(steps):
        frame_step = step % frame_every == frame_every - 1
        speed, yaw_rate = command(step, period)
        result = session.advance_control_step(speed, yaw_rate, render_cameras=frame_step)
        state = result.state
        poses.append(np.concatenate([state.position_m, state.orientation_xyzw]))
        wheel_angles.append(state.wheel_angle_rad.copy())
        if frame_step:
            frames.append(rig.read("front")["rgb"])
            images = mask_renderer.render(camera_path, resolution)
            object_masks.append(images.object_mask)
            depths.append(images.depth_m)
    field = session.track_runtime.field if session.track_runtime is not None else None
    return {"pose": np.asarray(poses), "wheel_angle": np.asarray(wheel_angles), "start_wheel_angle": start_wheel_angles,
            "frames": np.asarray(frames), "object_masks": np.asarray(object_masks), "depths": np.asarray(depths),
            "track_height_change": field.height_change_m.copy() if field is not None else np.zeros(1)}


def compare(first: dict, second: dict) -> dict:
    """Largest differences between two recorded runs."""
    position = np.linalg.norm(first["pose"][:, :3] - second["pose"][:, :3], axis=1)
    yaw = np.array([abs(math.remainder(yaw_from_quaternion(a) - yaw_from_quaternion(b), 2 * math.pi))
                    for a, b in zip(first["pose"][:, 3:], second["pose"][:, 3:])])
    relative_wheel = (first["wheel_angle"] - first["start_wheel_angle"]) - (second["wheel_angle"] - second["start_wheel_angle"])
    frame_difference = np.abs(first["frames"].astype(np.int16) - second["frames"].astype(np.int16))
    first_divergence = np.flatnonzero(position > 1e-6)
    return {"maximum_position_difference_m": float(position.max()),
            "final_position_difference_m": float(position[-1]),
            "first_step_with_position_difference_over_1_micrometre": int(first_divergence[0]) if len(first_divergence) else None,
            "maximum_yaw_difference_deg": float(np.degrees(yaw.max())),
            "maximum_relative_wheel_angle_difference_rad": float(np.abs(relative_wheel).max()),
            "start_wheel_angle_difference_rad": float(np.abs(first["start_wheel_angle"] - second["start_wheel_angle"]).max()),
            "frame_mean_absolute_difference_8bit": float(frame_difference.mean()),
            "frame_maximum_absolute_difference_8bit": int(frame_difference.max()),
            "frame_share_of_pixels_changed": float((frame_difference.max(axis=-1) > 0).mean()),
            "object_mask_share_of_pixels_changed": float((first["object_masks"] != second["object_masks"]).mean()),
            "depth_maximum_difference_m": float(np.nanmax(np.abs(np.where(np.isfinite(first["depths"]) & np.isfinite(second["depths"]),
                                                                        first["depths"] - second["depths"], 0.0)))),
            "track_height_change_maximum_difference_m": float(np.abs(first["track_height_change"] -
                                                                     second["track_height_change"]).max())}


exit_code = 1
try:
    scene = load_scene(data_root.scenes_directory / scene_identifier(configuration), configuration)
    run_directory = data_root.run_directory("check_determinism", time.strftime("%Y%m%d_%H%M%S"))
    print(f"[determinism] run directory {run_directory}", flush=True)
    session = RoverDriveSession(application, scene, stepper_mode="explicit")
    session.start()
    resolution = tuple(configuration.render.camera_resolution_px)
    camera_path = session.rover_paths.cameras["front"]
    rig = CameraRig()
    rig.add_camera("front", camera_path, resolution)
    mask_renderer = GroundTruthMaskRenderer(session)
    render_synchronously(configuration.render.warmup_subframes_after_reset)
    render_synchronously(configuration.render.subframes_per_camera_frame)
    noise_first = rig.read("front")["rgb"].astype(np.int16)
    render_synchronously(configuration.render.subframes_per_camera_frame)
    noise_second = rig.read("front")["rgb"].astype(np.int16)
    noise = np.abs(noise_first - noise_second)
    noise_floor = {"frame_mean_absolute_difference_8bit": float(noise.mean()),
                   "frame_maximum_absolute_difference_8bit": int(noise.max()),
                   "frame_share_of_pixels_changed": float((noise.max(axis=-1) > 0).mean())}
    print(f"[determinism] render noise floor (same state rendered twice) {json.dumps(noise_floor)}", flush=True)
    within, reset_seconds, first_runs = {}, {}, {}
    for reset_mode in arguments.reset_modes.split(","):
        runs = []
        for run_number in range(2):
            started = time.perf_counter()
            runs.append(drive_once(session, rig, mask_renderer, camera_path, resolution, arguments.steps,
                                   arguments.frame_every, reset_mode))
            final = runs[-1]["pose"][-1, :3]
            print(f"[determinism] {reset_mode} run {run_number + 1}: {arguments.steps} steps in "
                  f"{time.perf_counter() - started:.1f} s, final position {np.round(final, 4).tolist()}", flush=True)
        started = time.perf_counter()
        session.reset(clear_tracks=True, mode=reset_mode)
        reset_seconds[reset_mode] = round(time.perf_counter() - started, 2)
        within[reset_mode] = compare(runs[0], runs[1])
        print(f"[determinism] {reset_mode}, two runs in one process: {json.dumps(within[reset_mode])}; one reset "
              f"takes {reset_seconds[reset_mode]} s", flush=True)
        first_runs[reset_mode] = runs[0]
    np.savez_compressed(run_directory / "trajectory.npz", **{f"{mode}__{key}": value for mode, run in first_runs.items()
                                                             for key, value in run.items()})
    across = None
    if arguments.compare_with:
        across = {}
        with np.load(arguments.compare_with) as archive:
            for mode in first_runs:
                keys = [key for key in archive.files if key.startswith(f"{mode}__")]
                if not keys:
                    continue
                earlier = {key[len(mode) + 2:]: archive[key] for key in keys}
                across[mode] = compare(earlier, first_runs[mode])
                print(f"[determinism] {mode} across processes (vs {arguments.compare_with}): {json.dumps(across[mode])}",
                      flush=True)
    modes = list(first_runs)
    between_modes = {f"{a} vs {b}": compare(first_runs[a], first_runs[b])["maximum_position_difference_m"]
                     for index, a in enumerate(modes) for b in modes[index + 1:]}
    print(f"[determinism] maximum position difference between modes (first runs): {json.dumps(between_modes)}", flush=True)
    summary = {"scene": scene.directory.name, "preset": arguments.preset, "steps": arguments.steps,
               "frame_every": arguments.frame_every, "physics": configuration.physics.model_dump(),
               "renderer": configuration.render.renderer, "render_noise_floor": noise_floor,
               "within_one_process": within, "reset_seconds": reset_seconds, "across_processes": across,
               "between_modes_maximum_position_difference_m": between_modes,
               "compared_with": arguments.compare_with}
    (run_directory / "result.json").write_text(json.dumps(summary, indent=1))
    print(f"[determinism] wrote {run_directory}", flush=True)
    rig.destroy()
    exit_code = 0
except Exception:
    import traceback
    traceback.print_exc()
    sys.stdout.flush()
finally:
    application.close(exit_code=exit_code)
