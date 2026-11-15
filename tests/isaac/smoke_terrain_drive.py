"""Smoke test: NAC DEM terrain + procedural detail with colliders; the rover's wheels follow the terrain.

    scripts/isaac_run.sh tests/isaac/smoke_terrain_drive.py [--preset smoke] [--scene-directory DIR] [--no-tracks]

Checks (numbers go to <data root>/runs/smoke_terrain_drive/<timestamp>/result.json):
  * colliders ready: raycasts at every collision tile and near-ring probe hit the expected surface within 1 mm
  * settle: chassis within 50 mm of the four-wheel support plane, wheel force sum within 10 % of the weight
  * drive: forward / arc / reverse segments; travelled distance plausible; wheel-to-collider gap stays small
    (wheels neither float nor sink); the rover never falls through
Images: front camera and an observer camera before and after the drive.
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
parser.add_argument("--preset", default="smoke")
parser.add_argument("--scene-directory", default=None)
parser.add_argument("--no-tracks", action="store_true")
parser.add_argument("--segments", default="4:0.5:0.0,4:0.4:0.35,3:-0.3:0.0",
                    help="duration_s:speed_m_s:yaw_rate_rad_s,...")
arguments = parser.parse_args()

configuration = load_configuration(arguments.preset)
from lunar_rover_sim.runtime.application import process_video_memory_mib, start_simulation_application  # noqa: E402

application, data_root = start_simulation_application(configuration, headless=True)

import numpy as np  # noqa: E402

from lunar_rover_sim.core.data_model.regular_grid import triangulated_normal  # noqa: E402
from lunar_rover_sim.core.frames.pose_math import yaw_from_quaternion  # noqa: E402
from lunar_rover_sim.core.scene.scene_loader import load_scene  # noqa: E402
from lunar_rover_sim.runtime.driving.rover_drive_session import RoverDriveSession  # noqa: E402
from lunar_rover_sim.runtime.sensors.camera_rig import CameraRig, define_observer_camera, save_png  # noqa: E402
from lunar_rover_sim.runtime.stepping import render_synchronously  # noqa: E402

exit_code = 1
try:
    scene_directory = Path(arguments.scene_directory) if arguments.scene_directory else \
        data_root.scenes_directory / scene_identifier(configuration)
    run_directory = data_root.run_directory("smoke_terrain_drive", time.strftime("%Y%m%d_%H%M%S"))
    print(f"[smoke] scene {scene_directory}\n[smoke] run directory {run_directory}", flush=True)
    scene = load_scene(scene_directory, configuration)
    session = RoverDriveSession(application, scene, stepper_mode="explicit",
                                enable_tracks=False if arguments.no_tracks else None)
    report = session.start()
    result = {"scene": str(scene_directory), "preset": arguments.preset, "start": report.__dict__,
              "video_memory_after_start_mib": process_video_memory_mib(), "checks": {}}

    spawn = np.array(session.spawn_pose.position_m)
    heading = session.spawn_pose.heading_rad
    forward = np.array([np.cos(heading), np.sin(heading), 0.0])
    left = np.array([-np.sin(heading), np.cos(heading), 0.0])
    observer_eye = spawn + 2.5 * forward - 3.5 * left + np.array([0.0, 0.0, 1.6])
    observer_target = spawn + 1.5 * forward
    define_observer_camera(session.stage, "/World/ObserverCamera", observer_eye, observer_target, 70.0, (960, 540))
    rig = CameraRig()
    rig.add_camera("observer", "/World/ObserverCamera", (960, 540))
    rig.add_camera("front", session.rover_paths.cameras["front"], tuple(configuration.render.camera_resolution_px),
                   ("rgb", "distance_to_image_plane", "semantic_segmentation"))
    render_synchronously(8)
    save_png(run_directory / "observer_before.png", rig.read("observer")["rgb"])
    front_before = rig.read("front")
    save_png(run_directory / "front_before.png", front_before["rgb"])
    labels = {int(key): str(value) for key, value in front_before["semantic_labels"].items()}
    result["front_semantic_pixel_fractions"] = {
        label: float(np.mean(front_before["semantic_id"] == key)) for key, label in labels.items()}

    rock_check = {"checked": 0, "hit_own_rock": 0, "top_errors_m": []}
    if "rocks" in scene.extra_arrays:
        import carb
        import omni.physx
        from lunar_rover_sim.core.frames.pose_math import rotation_matrix_from_quaternion
        from lunar_rover_sim.core.rocks.rock_placement import RockTable
        from lunar_rover_sim.core.rocks.rock_shape_generator import prototype_mesh
        rock_arrays = scene.extra_arrays["rocks"]
        rock_table = RockTable.from_arrays(rock_arrays)
        query = omni.physx.get_physx_scene_query_interface()
        for index in np.argsort(-rock_table.diameter_m)[:200]:
            if rock_table.instance_id[index] <= 0 or rock_check["checked"] >= 20:
                continue
            vertices, _ = prototype_mesh(rock_arrays, int(rock_table.prototype_index[index]), "detailed")
            world = (rotation_matrix_from_quaternion(rock_table.orientation_xyzw[index]) @
                     (vertices.astype(np.float64) * rock_table.diameter_m[index]).T).T + rock_table.position_m[index]
            center = world.mean(axis=0)
            near = np.hypot(world[:, 0] - center[0], world[:, 1] - center[1]) < 0.15 * rock_table.diameter_m[index]
            expected_top = float(world[near, 2].max()) if near.any() else float(world[:, 2].max())
            hit = query.raycast_closest(carb.Float3(float(center[0]), float(center[1]), float(world[:, 2].max()) + 2.0),
                                        carb.Float3(0.0, 0.0, -1.0), 10.0)
            rock_check["checked"] += 1
            own = f"rock_{int(rock_table.instance_id[index]):05d}"
            if hit["hit"] and own in str(hit.get("collision", "")):
                rock_check["hit_own_rock"] += 1
                rock_check["top_errors_m"].append(float(hit["position"][2]) - expected_top)
                rock_check.setdefault("diameters_m", []).append(float(rock_table.diameter_m[index]))
        errors = np.abs(rock_check["top_errors_m"]) if rock_check["top_errors_m"] else np.array([np.inf])
        rock_check["maximum_top_error_m"] = float(errors.max())
        diameters = np.array(rock_check.get("diameters_m", [1.0]))
        rock_check["maximum_relative_top_error"] = float((errors / diameters).max()) if len(errors) == len(diameters) else np.inf
        print(f"[smoke] rock colliders: {rock_check['hit_own_rock']}/{rock_check['checked']} raycasts hit their own "
              f"rock; max top error {rock_check['maximum_top_error_m'] * 1000:.1f} mm", flush=True)
    result["rock_collider_check"] = rock_check

    grid = scene.layout.workspace_collision_grid
    radius = configuration.rover.wheel_radius_m
    gaps = []
    segments = [tuple(float(value) for value in part.split(":")) for part in arguments.segments.split(",")]
    start_state = session.rover.read_state()
    path_length = 0.0
    previous_position = start_state.position_m.copy()
    step_times = []
    for duration, speed, yaw_rate in segments:
        for _ in range(int(round(duration * configuration.drive.control_rate_hz))):
            started = time.perf_counter()
            step = session.advance_control_step(speed, yaw_rate)
            step_times.append(time.perf_counter() - started)
            state = step.state
            centers = state.wheel_center_m
            surface = session.collider_height_m(centers[:, 0], centers[:, 1])
            normal = triangulated_normal(scene.workspace_collision_height_m, grid, centers[:, 0], centers[:, 1])
            gaps.append((centers[:, 2] - surface) * normal[:, 2] - radius)
            path_length += float(np.linalg.norm(state.position_m[:2] - previous_position[:2]))
            previous_position = state.position_m.copy()
        print(f"[smoke] segment {duration:.0f} s at {speed} m/s, {yaw_rate} rad/s done: position "
              f"{np.round(state.position_m, 3).tolist()} heading {np.degrees(yaw_from_quaternion(state.orientation_xyzw)):.1f} deg",
              flush=True)

    render_synchronously(8)
    save_png(run_directory / "observer_after.png", rig.read("observer")["rgb"])
    front = rig.read("front")
    save_png(run_directory / "front_after.png", front["rgb"])
    gaps = np.array(gaps)
    end_state = session.rover.read_state()
    expected_path = sum(abs(speed) * duration for duration, speed, _ in segments)
    checks = {
        "colliders_ready": report.raycast_maximum_error_m <= 1e-3,
        "settle_height_within_50_mm": abs(report.settled_height_error_m) <= 0.05,
        "wheel_force_sum_within_10_percent": abs(report.wheel_normal_force_sum_n - report.rover_weight_n) <= 0.1 * report.rover_weight_n,
        "path_length_at_least_70_percent_of_commanded": path_length >= 0.7 * expected_path,
        "mean_wheel_gap_within_5_mm": abs(float(np.mean(gaps))) <= 0.005,
        "wheels_never_sink_more_than_15_mm": float(np.min(gaps)) >= -0.015,
    }
    if rock_check["checked"]:
        checks["rock_raycasts_hit_their_own_rock"] = rock_check["hit_own_rock"] == rock_check["checked"]
        checks["rock_collider_top_within_3_percent_of_diameter"] = rock_check["maximum_relative_top_error"] <= 0.03
    result.update({
        "path_length_m": path_length, "commanded_path_length_m": expected_path,
        "end_position_m": end_state.position_m.tolist(),
        "wheel_gap_m": {"mean": float(gaps.mean()), "minimum": float(gaps.min()), "maximum": float(gaps.max()),
                        "percentile_95_abs": float(np.percentile(np.abs(gaps), 95)),
                        "fraction_airborne_over_20_mm": float(np.mean(gaps > 0.02))},
        "control_step_ms": {"mean": 1000 * float(np.mean(step_times)), "maximum": 1000 * float(np.max(step_times))},
        "front_depth_m": {"minimum": float(np.nanmin(np.where(np.isfinite(front["depth_m"]), front["depth_m"], np.nan))),
                          "median": float(np.nanmedian(np.where(np.isfinite(front["depth_m"]), front["depth_m"], np.nan)))},
        "video_memory_end_mib": process_video_memory_mib(),
        "checks": checks, "passed": all(checks.values()),
    })
    (run_directory / "result.json").write_text(json.dumps(result, indent=1, default=float))
    print(json.dumps({key: result[key] for key in ("path_length_m", "commanded_path_length_m", "wheel_gap_m",
                                                    "control_step_ms", "video_memory_end_mib", "checks", "passed")},
                     indent=1, default=float), flush=True)
    rig.destroy()
    exit_code = 0 if result["passed"] else 1
except Exception:
    import traceback
    traceback.print_exc()
    sys.stdout.flush()
finally:
    application.close(exit_code=exit_code)
sys.exit(exit_code)
