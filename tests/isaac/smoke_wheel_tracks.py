"""Smoke test S4: wheels leave visible marks in the regolith.

    scripts/isaac_run.sh tests/isaac/smoke_wheel_tracks.py [--preset gpu_8gb] [--visual-mode geometry_and_mask]

Drives a forward + arc + forward route, then moves the rover out of view and renders a static low-Sun observer
camera and a top-down camera before and after. Checks (result.json under runs/smoke_wheel_tracks/<timestamp>/):
  * samples were emitted and stamped; median rut depth between the visual floor and the cap
  * the rendered images change inside the projected track mask much more than outside it (contrast-to-noise)
  * the deformed visual tiles really moved (rendered depth inside ruts is deeper than before)
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT / "src"))

from lunar_rover_sim.core.configuration.configuration_loader import load_configuration, scene_identifier  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("--preset", default="gpu_8gb")
parser.add_argument("--visual-mode", default=None, choices=[None, "geometry_and_mask", "geometry_only", "mask_only"])
parser.add_argument("--segments", default="5:0.4:0.0,5:0.35:0.3,4:0.4:0.0")
parser.add_argument("--sun-elevation-deg", type=float, default=None)
arguments = parser.parse_args()
overrides = [f"tracks.visual_mode={arguments.visual_mode}"] if arguments.visual_mode else []
if arguments.sun_elevation_deg is not None:
    overrides.append(f"lighting.sun_elevation_deg={arguments.sun_elevation_deg}")
configuration = load_configuration(arguments.preset, overrides=overrides)

from lunar_rover_sim.runtime.application import process_video_memory_mib, start_simulation_application  # noqa: E402

application, data_root = start_simulation_application(configuration, headless=True)

import numpy as np  # noqa: E402

from lunar_rover_sim.core.frames.pose_math import camera_intrinsic_matrix, look_at_rotation  # noqa: E402
from lunar_rover_sim.core.scene.scene_loader import load_scene  # noqa: E402
from lunar_rover_sim.runtime.driving.rover_drive_session import RoverDriveSession  # noqa: E402
from lunar_rover_sim.runtime.sensors.camera_rig import CameraRig, define_observer_camera, save_png  # noqa: E402
from lunar_rover_sim.runtime.stage_assembly.scene_stage_builder import build_lighting  # noqa: E402
from lunar_rover_sim.runtime.stepping import render_synchronously  # noqa: E402

exit_code = 1
try:
    scene = load_scene(data_root.scenes_directory / scene_identifier(configuration), configuration)
    run_directory = data_root.run_directory("smoke_wheel_tracks", time.strftime("%Y%m%d_%H%M%S"))
    print(f"[tracks] run directory {run_directory}; visual mode {configuration.tracks.visual_mode}", flush=True)
    session = RoverDriveSession(application, scene, stepper_mode="explicit", enable_tracks=True)
    report = session.start()
    if arguments.sun_elevation_deg is not None:
        build_lighting(session.stage, scene)

    spawn = np.array(session.spawn_pose.position_m)
    heading = session.spawn_pose.heading_rad
    forward = np.array([np.cos(heading), np.sin(heading), 0.0])
    left = np.array([-np.sin(heading), np.cos(heading), 0.0])
    side_eye = spawn + 2.6 * forward - 3.2 * left + np.array([0.0, 0.0, 1.3])
    side_target = spawn + 2.6 * forward + 0.6 * left
    top_eye = spawn + 2.6 * forward + 1.0 * left + np.array([0.0, 0.0, 6.0])
    top_target = spawn + 2.6 * forward + 1.0 * left + np.array([0.0, 0.0, 0.0]) + 0.01 * forward
    resolutions = {"side": (960, 540), "top": (800, 800)}
    fields_of_view = {"side": 70.0, "top": 75.0}
    eyes = {"side": (side_eye, side_target), "top": (top_eye, top_target)}
    rig = CameraRig()
    for name, (eye, target) in eyes.items():
        define_observer_camera(session.stage, f"/World/Observer_{name}", eye, target, fields_of_view[name], resolutions[name])
        rig.add_camera(name, f"/World/Observer_{name}", resolutions[name], ("rgb", "distance_to_image_plane"))

    def park_rover_out_of_view() -> None:
        """Teleport the rover behind the side camera and settle it (tracks are not cleared)."""
        session.track_runtime.sampling_enabled = False
        park = spawn - 3.0 * forward - 3.5 * left
        pose = session.support_pose(float(park[0]), float(park[1]), heading)
        session.rover.teleport(np.array(pose.position_m) + [0, 0, 0.01], pose.orientation_xyzw)
        session.stepper.advance_physics_steps(60)

    def capture(tag: str) -> dict:
        """Render both observers and save PNGs."""
        session.track_runtime.force_mask_upload()
        render_synchronously(16)
        output = {}
        for name in eyes:
            data = rig.read(name)
            save_png(run_directory / f"{name}_{tag}.png", data["rgb"])
            output[name] = data
        return output

    session.track_runtime.sampling_enabled = False
    saved_pose = session.rover.read_state()
    park_rover_out_of_view()
    before = capture("before")
    session.rover.teleport(saved_pose.position_m, saved_pose.orientation_xyzw)
    session.stepper.advance_physics_steps(60)
    session.track_runtime.reset_sampling()
    session.track_runtime.sampling_enabled = True

    segments = [tuple(float(v) for v in part.split(":")) for part in arguments.segments.split(",")]
    step_times = []
    for duration, speed, yaw_rate in segments:
        for _ in range(int(round(duration * configuration.drive.control_rate_hz))):
            started = time.perf_counter()
            session.advance_control_step(speed, yaw_rate)
            step_times.append(time.perf_counter() - started)
    statistics = session.track_runtime.field.rut_depth_statistics()
    print(f"[tracks] {len(session.track_runtime.contact_log)} samples; field {statistics}", flush=True)
    save_png(run_directory / "side_with_rover.png", capture("with_rover")["side"]["rgb"])
    park_rover_out_of_view()
    after = capture("after")

    field = session.track_runtime.field
    mask = field.track_mask()
    grid = field.grid
    east, north = grid.coordinate_mesh()
    heights = scene.workspace_height_m.astype(np.float64)
    tread = field.compaction > 0
    result = {"preset": arguments.preset, "visual_mode": configuration.tracks.visual_mode,
              "sun_elevation_deg": configuration.lighting.sun_elevation_deg, "samples": len(session.track_runtime.contact_log),
              "field": statistics, "timings_ms_total": session.track_runtime.timings_ms,
              "control_step_ms_mean": 1000 * float(np.mean(step_times)), "video_memory_mib": process_video_memory_mib(),
              "start": {"settled_height_error_m": report.settled_height_error_m}}
    sinkages = np.array([sample.sinkage_m for sample in session.track_runtime.contact_log])
    if len(sinkages):
        result["sinkage_m"] = {"median": float(np.median(sinkages)), "minimum": float(sinkages.min()),
                               "maximum": float(sinkages.max())}
    image_checks = {}
    for name, (eye, target) in eyes.items():
        width, height = resolutions[name]
        rotation = look_at_rotation(eye, target)
        intrinsics = camera_intrinsic_matrix(width, height, fields_of_view[name])
        points = np.column_stack([east[mask], north[mask], heights[mask]])
        camera_frame = (rotation.T @ (points - eye).T).T
        in_front = camera_frame[:, 2] < -0.05
        u = intrinsics[0, 0] * camera_frame[in_front, 0] / -camera_frame[in_front, 2] + width / 2
        v = height / 2 - intrinsics[1, 1] * camera_frame[in_front, 1] / -camera_frame[in_front, 2]
        keep = (u >= 0) & (u < width) & (v >= 0) & (v < height)
        projected = np.zeros((height, width), bool)
        projected[v[keep].astype(int), u[keep].astype(int)] = True
        from scipy import ndimage
        track_pixels = ndimage.binary_closing(projected, iterations=2)
        background = ~ndimage.binary_dilation(track_pixels, iterations=12)
        difference = np.abs(after[name]["rgb"].astype(np.float64) - before[name]["rgb"].astype(np.float64)).mean(axis=2)
        inside = float(difference[track_pixels].mean()) if track_pixels.any() else 0.0
        outside_values = difference[background]
        contrast = (inside - float(outside_values.mean())) / max(float(outside_values.std()), 0.5)
        depth_change = after[name]["depth_m"] - before[name]["depth_m"]
        image_checks[name] = {"track_pixels": int(track_pixels.sum()), "mean_abs_change_inside": inside,
                              "mean_abs_change_outside": float(outside_values.mean()),
                              "contrast_to_noise": contrast,
                              "median_depth_change_inside_m": float(np.nanmedian(np.where(track_pixels, depth_change, np.nan)))
                              if track_pixels.any() else 0.0}
        overlay = after[name]["rgb"].copy()
        overlay[track_pixels & ~ndimage.binary_erosion(track_pixels)] = [255, 60, 60]
        save_png(run_directory / f"{name}_mask_outline.png", overlay)
    result["images"] = image_checks
    floor, cap = configuration.sinkage.visual_floor_m, configuration.sinkage.maximum_m
    checks = {
        "samples_emitted": result["samples"] > 100,
        "median_tread_depth_between_floor_and_cap": -cap - 0.009 <= statistics["median_tread_height_change_m"] <= -floor + 1e-4,
        "berms_raised": statistics["maximum_berm_m"] > 0.0005,
        "top_view_track_visible_contrast_over_3": image_checks["top"]["contrast_to_noise"] >= 3.0,
        "side_view_track_visible_contrast_over_3": image_checks["side"]["contrast_to_noise"] >= 3.0,
    }
    if configuration.tracks.visual_mode != "mask_only":
        checks["rendered_surface_moved_down_in_ruts"] = image_checks["top"]["median_depth_change_inside_m"] > 0.001
    result["checks"] = checks
    result["passed"] = all(checks.values())
    (run_directory / "result.json").write_text(json.dumps(result, indent=1, default=float))
    print(json.dumps({key: result[key] for key in ("samples", "sinkage_m", "images", "control_step_ms_mean",
                                                    "timings_ms_total", "checks", "passed") if key in result},
                     indent=1, default=float), flush=True)
    rig.destroy()
    exit_code = 0 if result["passed"] else 1
except Exception:
    import traceback
    traceback.print_exc()
    sys.stdout.flush()
finally:
    application.close(exit_code=exit_code)
