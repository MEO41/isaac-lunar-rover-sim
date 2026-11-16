"""Smoke test: the scripted expert drives the PhysX rover to sampled goals around rocks and craters.

    scripts/isaac_run.sh tests/isaac/smoke_scripted_expert.py [--preset gpu_8gb] [--goals 3] [--minimum-distance-m 8]
        [--maximum-distance-m 14] [--seed 0] [--timeout-per-goal-s 120] [--throughput-steps 60]

Checks (numbers go to <data root>/runs/smoke_scripted_expert/<timestamp>/result.json):
  * every goal is reached within the timeout (state 'reached', distance <= stop tolerance)
  * the footprint rectangle never overlaps an obstacle rock (geometric gap at every control step > 0)
  * the rover never tilts beyond the expert's abort limit and never falls through
Also measured: wall time per control step while driving (physics + tracks + expert), and with one front-camera
frame rendered per step (the cost of a rollout that a VLA policy would see).
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
parser.add_argument("--goals", type=int, default=3)
parser.add_argument("--minimum-distance-m", type=float, default=8.0)
parser.add_argument("--maximum-distance-m", type=float, default=14.0)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--timeout-per-goal-s", type=float, default=120.0)
parser.add_argument("--throughput-steps", type=int, default=60)
arguments = parser.parse_args()

configuration = load_configuration(arguments.preset)
from lunar_rover_sim.runtime.application import process_video_memory_mib, start_simulation_application  # noqa: E402

application, data_root = start_simulation_application(configuration, headless=True)

import numpy as np  # noqa: E402

from lunar_rover_sim.core.frames.pose_math import yaw_from_quaternion  # noqa: E402
from lunar_rover_sim.core.planning.dynamic_window_planner import footprint_gap_m  # noqa: E402
from lunar_rover_sim.core.planning.global_planner import GlobalPlanner, sample_reachable_goal  # noqa: E402
from lunar_rover_sim.core.planning.planner_map import planner_map_from_scene  # noqa: E402
from lunar_rover_sim.core.planning.scripted_expert import ScriptedExpert, tilt_from_quaternion  # noqa: E402
from lunar_rover_sim.core.scene.scene_loader import load_scene  # noqa: E402
from lunar_rover_sim.runtime.driving.rover_drive_session import RoverDriveSession  # noqa: E402
from lunar_rover_sim.runtime.sensors.camera_rig import CameraRig, save_png  # noqa: E402
from lunar_rover_sim.runtime.stepping import render_synchronously  # noqa: E402


def pose_of(state) -> np.ndarray:
    """(east, north, yaw) of a rover state."""
    return np.array([state.position_m[0], state.position_m[1], yaw_from_quaternion(state.orientation_xyzw)])


exit_code = 1
try:
    scene = load_scene(data_root.scenes_directory / scene_identifier(configuration), configuration)
    run_directory = data_root.run_directory("smoke_scripted_expert", time.strftime("%Y%m%d_%H%M%S"))
    print(f"[expert] scene {scene.directory}\n[expert] run directory {run_directory}", flush=True)
    session = RoverDriveSession(application, scene, stepper_mode="explicit")
    report = session.start()
    started = time.perf_counter()
    planner_map = planner_map_from_scene(scene)
    planner_map_seconds = time.perf_counter() - started
    period = session.stepper.control_period_s
    expert = ScriptedExpert(planner_map, configuration.expert, configuration.rover, period)
    planner = GlobalPlanner(planner_map, configuration.expert)
    random_generator = np.random.default_rng(arguments.seed)
    episodes = []
    for goal_number in range(arguments.goals):
        state = session.rover.read_state()
        start_pose = pose_of(state)
        goal, _ = sample_reachable_goal(planner, start_pose[:2], arguments.minimum_distance_m,
                                        arguments.maximum_distance_m, random_generator)
        route = expert.set_goal(start_pose, goal)
        print(f"[expert] goal {goal_number}: {np.round(start_pose[:2], 2).tolist()} -> {np.round(goal, 2).tolist()}, "
              f"route {route.length_m:.1f} m ({route.details.get('attempt')})", flush=True)
        minimum_gap, maximum_tilt, steps, path = np.inf, 0.0, 0, [start_pose[:2].tolist()]
        stopped_at_boundary = False
        wall_started = time.perf_counter()
        simulated_started = session.stepper.time_s
        while session.stepper.time_s - simulated_started < arguments.timeout_per_goal_s:
            pose = pose_of(state)
            tilt = tilt_from_quaternion(state.orientation_xyzw)
            command = expert.command(session.stepper.time_s, pose, tilt)
            if command.state != "driving":
                break
            result = session.advance_control_step(command.forward_speed_m_s, command.yaw_rate_rad_s)
            state = result.state
            stopped_at_boundary |= result.stopped_at_boundary
            steps += 1
            minimum_gap = min(minimum_gap, footprint_gap_m(pose_of(state), planner_map.rock_obstacles, expert.footprint))
            maximum_tilt = max(maximum_tilt, tilt_from_quaternion(state.orientation_xyzw))
            if steps % 10 == 0:
                path.append(np.round(state.position_m[:2], 3).tolist())
        wall = time.perf_counter() - wall_started
        final_pose = pose_of(session.rover.read_state())
        episode = {"goal_m": np.round(goal, 3).tolist(), "start_m": np.round(start_pose[:2], 3).tolist(),
                   "route_length_m": round(route.length_m, 2), "state": expert.state, "reason": expert.reason,
                   "final_distance_m": round(float(math.hypot(*(final_pose[:2] - expert.goal_m))), 3),
                   "simulated_s": round(session.stepper.time_s - simulated_started, 2), "control_steps": steps,
                   "wall_s": round(wall, 2), "wall_ms_per_control_step": round(1000 * wall / max(steps, 1), 1),
                   "real_time_factor": round((session.stepper.time_s - simulated_started) / max(wall, 1e-9), 2),
                   "minimum_footprint_gap_to_obstacle_rocks_m": round(minimum_gap, 3),
                   "maximum_tilt_deg": round(math.degrees(maximum_tilt), 1), "stopped_at_boundary": stopped_at_boundary,
                   "plans": expert.statistics.plans, "stalls": expert.statistics.total_stalls,
                   "blocked_steps": expert.statistics.blocked_steps, "path_every_second_m": path}
        episodes.append(episode)
        print(f"[expert] goal {goal_number}: {episode['state']} {episode['reason']} after {episode['simulated_s']} s "
              f"simulated / {episode['wall_s']} s wall ({episode['real_time_factor']}x real time); final distance "
              f"{episode['final_distance_m']} m; minimum gap to obstacle rocks {episode['minimum_footprint_gap_to_obstacle_rocks_m']} m; "
              f"max tilt {episode['maximum_tilt_deg']} deg; plans {episode['plans']}, stalls {episode['stalls']}", flush=True)
        if expert.state != "reached":
            break

    rig = CameraRig()
    rig.add_camera("front", session.rover_paths.cameras["front"], tuple(configuration.render.camera_resolution_px))
    render_synchronously(configuration.render.warmup_subframes_after_reset)
    state = session.rover.read_state()
    pose = pose_of(state)
    goal, _ = sample_reachable_goal(planner, pose[:2], 6.0, 12.0, random_generator)
    expert.set_goal(pose, goal)
    throughput_started = time.perf_counter()
    rendered_steps = 0
    for _ in range(arguments.throughput_steps):
        command = expert.command(session.stepper.time_s, pose_of(state), tilt_from_quaternion(state.orientation_xyzw))
        if command.state != "driving":
            break
        state = session.advance_control_step(command.forward_speed_m_s, command.yaw_rate_rad_s, render_cameras=True).state
        rig.read("front")
        rendered_steps += 1
    throughput_wall = time.perf_counter() - throughput_started
    save_png(run_directory / "front_last.png", rig.read("front")["rgb"])
    rig.destroy()

    reached = [episode["state"] == "reached" for episode in episodes]
    gaps = [episode["minimum_footprint_gap_to_obstacle_rocks_m"] for episode in episodes]
    checks = {
        "all_goals_reached": bool(len(episodes) == arguments.goals and all(reached)),
        "footprint_never_overlapped_an_obstacle_rock": bool(min(gaps) > 0.0),
        "tilt_below_abort_limit": bool(max(e["maximum_tilt_deg"] for e in episodes) < configuration.expert.maximum_tilt_deg),
        "never_stopped_at_boundary": bool(not any(e["stopped_at_boundary"] for e in episodes)),
    }
    result = {"scene": scene.directory.name, "preset": arguments.preset, "seed": arguments.seed,
              "planner_map_seconds": round(planner_map_seconds, 2), "obstacle_rocks": len(planner_map.rock_obstacles),
              "settle": {"wheel_normal_force_sum_n": round(report.wheel_normal_force_sum_n, 2),
                         "rover_weight_n": round(report.rover_weight_n, 2)},
              "episodes": episodes,
              "throughput_with_front_camera": {
                  "control_steps": rendered_steps, "wall_s": round(throughput_wall, 2),
                  "wall_ms_per_control_step": round(1000 * throughput_wall / max(rendered_steps, 1), 1),
                  "real_time_factor": round(rendered_steps * period / max(throughput_wall, 1e-9), 2),
                  "resolution_px": list(configuration.render.camera_resolution_px),
                  "subframes_per_camera_frame": configuration.render.subframes_per_camera_frame},
              "video_memory_mib": process_video_memory_mib(), "checks": checks}
    (run_directory / "result.json").write_text(json.dumps(result, indent=1))
    print(f"[expert] throughput with the front camera: {json.dumps(result['throughput_with_front_camera'])}", flush=True)
    print(f"[expert] checks {json.dumps(checks)}", flush=True)
    exit_code = 0 if all(checks.values()) else 1
except Exception:
    import traceback
    traceback.print_exc()
    sys.stdout.flush()
finally:
    application.close(exit_code=exit_code)
