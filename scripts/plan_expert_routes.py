"""Plan scripted-expert routes on a baked scene without Isaac, and drive them with a kinematic rover model.

    $ISAAC_ENV/bin/python scripts/plan_expert_routes.py [--preset gpu_8gb] [--goals 6] [--minimum-distance-m 8]
        [--maximum-distance-m 15] [--seed 0]

Writes <data root>/runs/expert_routes/<timestamp>/{routes.png, routes.json}: the cost map with impassable cells,
rock obstacles, planned routes and the kinematically driven paths. The kinematic model is a lagged unicycle, NOT the
PhysX rover: it checks the planner logic only (tests/isaac/smoke_scripted_expert.py drives the physical rover).
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lunar_rover_sim.core.configuration.configuration_loader import load_configuration, scene_identifier  # noqa: E402
from lunar_rover_sim.core.planning.dynamic_window_planner import footprint_collides, predict_arcs  # noqa: E402
from lunar_rover_sim.core.planning.global_planner import GlobalPlanner, sample_reachable_goal  # noqa: E402
from lunar_rover_sim.core.planning.planner_map import RoverFootprint, planner_map_from_scene  # noqa: E402
from lunar_rover_sim.core.planning.scripted_expert import ScriptedExpert  # noqa: E402
from lunar_rover_sim.core.scene.scene_loader import load_scene  # noqa: E402
from lunar_rover_sim.core.storage.data_root import require_data_root  # noqa: E402


def drive_kinematic(expert: ScriptedExpert, start_pose: np.ndarray, goal_m: np.ndarray, period_s: float,
                    timeout_s: float) -> dict:
    """Drive a lagged unicycle with the expert; returns the path, outcome and minimum footprint gap to rocks."""
    pose = start_pose.copy()
    route = expert.set_goal(pose, goal_m)
    velocity = np.zeros(2)
    path = [pose[:2].copy()]
    footprint = expert.footprint
    collided = False
    for step in range(int(timeout_s / period_s)):
        command = expert.command(step * period_s, pose, 0.0)
        if command.state != "driving":
            break
        velocity += 0.5 * (np.array([command.forward_speed_m_s, command.yaw_rate_rad_s]) - velocity)
        pose = predict_arcs(pose, velocity[:1], velocity[1:], period_s, period_s)[0, -1]
        path.append(pose[:2].copy())
        collided |= bool(footprint_collides(pose[None, None, :], expert.map.rock_obstacles, footprint, 0.0)[0])
    return {"route": route, "path": np.asarray(path), "state": expert.state, "reason": expert.reason,
            "simulated_s": round(step * period_s, 1), "footprint_touched_rock": collided,
            "plans": expert.statistics.plans, "stalls": expert.statistics.total_stalls}


def main() -> int:
    """Plan and drive; write the figure and the summary."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--preset", default="gpu_8gb")
    parser.add_argument("--goals", type=int, default=6)
    parser.add_argument("--minimum-distance-m", type=float, default=8.0)
    parser.add_argument("--maximum-distance-m", type=float, default=15.0)
    parser.add_argument("--seed", type=int, default=0)
    arguments = parser.parse_args()
    configuration = load_configuration(arguments.preset)
    data_root = require_data_root(None, configuration.storage.data_root, configuration.storage.allow_root_disk,
                                  configuration.storage.minimum_free_gib)
    scene = load_scene(data_root.scenes_directory / scene_identifier(configuration), configuration)
    started = time.perf_counter()
    planner_map = planner_map_from_scene(scene)
    map_seconds = time.perf_counter() - started
    planner = GlobalPlanner(planner_map, configuration.expert)
    period = 1.0 / configuration.drive.control_rate_hz
    spawn = scene.manifest["spawn_pose"]
    start_pose = np.array([spawn["position_m"][0], spawn["position_m"][1], spawn["heading_rad"]])
    random_generator = np.random.default_rng(arguments.seed)
    episodes = []
    for _ in range(arguments.goals):
        planning_started = time.perf_counter()
        goal, _ = sample_reachable_goal(planner, start_pose[:2], arguments.minimum_distance_m,
                                        arguments.maximum_distance_m, random_generator)
        plan_seconds = time.perf_counter() - planning_started
        expert = ScriptedExpert(planner_map, configuration.expert, configuration.rover, period)
        driving_started = time.perf_counter()
        outcome = drive_kinematic(expert, start_pose, goal, period, 180.0)
        outcome["wall_seconds_per_control_step"] = (time.perf_counter() - driving_started) / max(len(outcome["path"]), 1)
        outcome["goal"], outcome["plan_seconds"] = goal, plan_seconds
        episodes.append(outcome)
        print(f"[expert] goal {np.round(goal, 2).tolist()} route {outcome['route'].length_m:.1f} m: {outcome['state']} "
              f"{outcome['reason']} in {outcome['simulated_s']} s, plans {outcome['plans']}, stalls {outcome['stalls']}, "
              f"touched rock {outcome['footprint_touched_rock']}", flush=True)

    run_directory = data_root.run_directory("expert_routes", time.strftime("%Y%m%d_%H%M%S"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle
    grid = planner_map.grid
    extent = (grid.origin_east_m, grid.maximum_east_m, grid.origin_north_m, grid.maximum_north_m)
    figure, axis = plt.subplots(figsize=(9, 9), dpi=120)
    axis.imshow(planner_map.cost, origin="lower", extent=extent, cmap="Greys", vmin=10, vmax=50)
    overlay = np.zeros(planner_map.impassable.shape + (4,))
    overlay[planner_map.impassable] = (0.85, 0.2, 0.2, 0.45)
    overlay[planner_map.clearance_ring] = (0.95, 0.65, 0.2, 0.25)
    axis.imshow(overlay, origin="lower", extent=extent)
    obstacles = planner_map.rock_obstacles
    for east, north, radius in zip(obstacles.center_east_m, obstacles.center_north_m, obstacles.radius_m):
        axis.add_patch(Circle((east, north), radius, color="black"))
    for index, episode in enumerate(episodes):
        color = plt.cm.tab10(index % 10)
        axis.plot(*episode["route"].path_m.T, "--", color=color, linewidth=1)
        axis.plot(*episode["path"].T, "-", color=color, linewidth=1.6)
        axis.plot(*episode["goal"], "*", color=color, markersize=12, markeredgecolor="black")
    axis.plot(*start_pose[:2], "o", color="white", markeredgecolor="black", markersize=9)
    half = planner_map.drivable_half_size_m
    axis.set_xlim(-half - 1, half + 1)
    axis.set_ylim(-half - 1, half + 1)
    axis.set_xlabel("east (m)")
    axis.set_ylabel("north (m)")
    axis.set_title(f"{scene.directory.name}: cost (grey), impassable (red), clearance ring (orange), rocks > "
                   f"{configuration.expert.rock_obstacle_height_m} m (black)\nroutes dashed, kinematic drives solid "
                   f"(lagged unicycle, not PhysX)", fontsize=8)
    figure.tight_layout()
    figure.savefig(run_directory / "routes.png")
    summary = {"scene": scene.directory.name, "planner_map_seconds": round(map_seconds, 2),
               "planner_cells": list(grid.shape), "rock_obstacles": len(obstacles),
               "impassable_fraction": round(float(planner_map.impassable.mean()), 4),
               "episodes": [{"goal_m": np.round(e["goal"], 3).tolist(), "route_length_m": round(e["route"].length_m, 2),
                             "kept_preferred_clearance": e["route"].kept_preferred_clearance,
                             "plan_seconds": round(e["plan_seconds"], 3), "state": e["state"], "reason": e["reason"],
                             "simulated_s": e["simulated_s"], "plans": e["plans"], "stalls": e["stalls"],
                             "footprint_touched_rock": e["footprint_touched_rock"],
                             "wall_ms_per_control_step": round(1000 * e["wall_seconds_per_control_step"], 2)}
                            for e in episodes]}
    (run_directory / "routes.json").write_text(json.dumps(summary, indent=1))
    print(f"[expert] {json.dumps({k: v for k, v in summary.items() if k != 'episodes'})}", flush=True)
    print(f"[expert] wrote {run_directory}", flush=True)
    return 0 if all(e["state"] == "reached" and not e["footprint_touched_rock"] for e in episodes) else 1


if __name__ == "__main__":
    sys.exit(main())


__all__ = ["drive_kinematic", "math"]
