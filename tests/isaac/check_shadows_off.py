"""Check which renderer switch turns the Sun's cast shadows off without changing anything else (Isaac, headless).

    scripts/isaac_run.sh tests/isaac/check_shadows_off.py [--preset gpu_8gb] [--sun-elevation-deg 15]

Same scene, rover pose and cameras (front, rear, an observer that sees the rover). Candidates, each applied, rendered
and reverted in turn:
  light_shadow_enable       UsdLux ShadowAPI inputs:shadow:enable = false on /World/Sun
  primvar_do_not_cast       set_cast_shadows(stage, False): primvars:doNotCastShadows on every terrain, rock, rover mesh
  render_setting_shadows    carb setting /rtx/shadows/enabled = false
Judged with the ray-cast shadow_cause image (ground truth of why each pixel is dark):
  * cast pixels (cast_by_scene, cast_by_rover) must brighten: share whose luminance rises by more than 0.05
  * sunlit pixels must not change beyond the render noise floor (two renders of the unchanged scene)
  * faces_away pixels must stay dark (with the Sun as the only light they cannot be lit)
  * after reverting, the image must return to the baseline (within the noise floor)
Writes result.json and a comparison figure to <data root>/runs/check_shadows_off/<timestamp>/.
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
parser.add_argument("--sun-elevation-deg", type=float, default=None)
parser.add_argument("--warmup-subframes", type=int, default=24)
arguments = parser.parse_args()
overrides = [] if arguments.sun_elevation_deg is None else [f"lighting.sun_elevation_deg={arguments.sun_elevation_deg}"]
configuration = load_configuration(arguments.preset, overrides=overrides)
from lunar_rover_sim.runtime.application import exposure_snapshot, start_simulation_application  # noqa: E402

application, data_root = start_simulation_application(configuration, headless=True)

import carb  # noqa: E402
import numpy as np  # noqa: E402
from pxr import Sdf, Usd, UsdGeom, UsdLux  # noqa: E402

from lunar_rover_sim.core.frames.pose_math import yaw_from_quaternion  # noqa: E402
from lunar_rover_sim.core.ground_truth.mask_classes import SHADOW_CAUSE_ID  # noqa: E402
from lunar_rover_sim.core.scene.scene_loader import load_scene  # noqa: E402
from lunar_rover_sim.runtime.driving.rover_drive_session import RoverDriveSession  # noqa: E402
from lunar_rover_sim.runtime.sensors.camera_rig import CameraRig, define_observer_camera  # noqa: E402
from lunar_rover_sim.runtime.sensors.ground_truth_masks import GroundTruthMaskRenderer  # noqa: E402
from lunar_rover_sim.runtime.stage_assembly.scene_stage_builder import set_cast_shadows  # noqa: E402
from lunar_rover_sim.runtime.stepping import render_synchronously  # noqa: E402

SHADOW_CASTER_ROOTS = ("/World/Terrain", "/World/Rocks", "/World/RockPrototypes", "/World/Rover")
BRIGHTEN_THRESHOLD = 0.05


def luminance(rgb: np.ndarray) -> np.ndarray:
    """Relative luminance (0-1) of an sRGB uint8 image (on the encoded values)."""
    values = rgb.astype(np.float64) / 255.0
    return 0.2126 * values[..., 0] + 0.7152 * values[..., 1] + 0.0722 * values[..., 2]


def caster_gprims(stage) -> list:
    """Every geometric prim under the shadow-caster roots (instance proxies excluded; prototypes are edited)."""
    prims = []
    for root in SHADOW_CASTER_ROOTS:
        root_prim = stage.GetPrimAtPath(root)
        if not root_prim:
            continue
        for prim in Usd.PrimRange(root_prim):
            if prim.IsA(UsdGeom.Gprim) or prim.IsA(UsdGeom.PointInstancer):
                prims.append(prim)
    return prims


def make_candidates(stage):
    """{name: (apply, revert)} functions of the three shadow switches."""
    sun = stage.GetPrimAtPath("/World/Sun")
    shadow_api = UsdLux.ShadowAPI.Apply(sun)
    enable = shadow_api.CreateShadowEnableAttr(True)
    settings = carb.settings.get_settings()
    original_setting = settings.get("/rtx/shadows/enabled")
    prims = caster_gprims(stage)
    return {
        "light_shadow_enable": (lambda: enable.Set(False), lambda: enable.Set(True)),
        "primvar_do_not_cast": (lambda: set_cast_shadows(stage, False), lambda: set_cast_shadows(stage, True)),
        "render_setting_shadows": (lambda: settings.set("/rtx/shadows/enabled", False),
                                   lambda: settings.set("/rtx/shadows/enabled",
                                                        True if original_setting is None else original_setting)),
    }, {"caster_prims": len(prims), "rtx_shadows_enabled_setting_before": original_setting}


exit_code = 1
try:
    scene = load_scene(data_root.scenes_directory / scene_identifier(configuration), configuration)
    run_directory = data_root.run_directory("check_shadows_off", time.strftime("%Y%m%d_%H%M%S"))
    print(f"[shadows] run directory {run_directory}", flush=True)
    session = RoverDriveSession(application, scene, stepper_mode="explicit")
    session.start()
    for _ in range(4 * configuration.drive.control_rate_hz):
        session.advance_control_step(0.3, 0.0)
    for _ in range(configuration.drive.control_rate_hz):
        session.advance_control_step(0.0, 0.0)
    resolution = tuple(configuration.render.camera_resolution_px)
    camera_paths = {name: session.rover_paths.cameras[name] for name in ("front", "rear")
                    if getattr(configuration.rover.cameras, name).enabled}
    state = session.rover.read_state()
    heading = yaw_from_quaternion(state.orientation_xyzw)
    forward, left = np.array([np.cos(heading), np.sin(heading), 0.0]), np.array([-np.sin(heading), np.cos(heading), 0.0])
    eye = state.position_m - 2.6 * forward + 1.3 * left + np.array([0.0, 0.0, 1.25])
    camera_paths["observer"] = define_observer_camera(session.stage, "/World/ObserverCamera", eye,
                                                      state.position_m + np.array([0.0, 0.0, 0.1]), 65.0, resolution)
    rig = CameraRig()
    for name, path in camera_paths.items():
        rig.add_camera(name, path, resolution)
    if session.track_runtime is not None:
        session.track_runtime.force_mask_upload()
    mask_renderer = GroundTruthMaskRenderer(session)
    causes = {name: mask_renderer.render(path, resolution).shadow_cause for name, path in camera_paths.items()}

    def capture() -> dict:
        """Render all cameras after warm-up subframes; returns {camera: luminance}."""
        render_synchronously(arguments.warmup_subframes)
        return {name: luminance(rig.read(name)["rgb"]) for name in camera_paths}, \
            {name: rig.read(name)["rgb"] for name in camera_paths}

    baseline, baseline_rgb = capture()
    repeat, _ = capture()
    candidates, candidate_info = make_candidates(session.stage)
    results = {"noise_floor": {}, "candidates": {}}
    for name in camera_paths:
        sunlit = causes[name] == SHADOW_CAUSE_ID["sunlit"]
        results["noise_floor"][name] = {
            "mean_absolute_luminance_change_all": round(float(np.abs(repeat[name] - baseline[name]).mean()), 5),
            "mean_absolute_luminance_change_sunlit": round(float(np.abs(repeat[name] - baseline[name])[sunlit].mean()), 5)}
    print(f"[shadows] noise floor {json.dumps(results['noise_floor'])}", flush=True)
    images = {}
    for candidate, (apply, revert) in candidates.items():
        apply()
        after, after_rgb = capture()
        revert()
        restored, _ = capture()
        row = {}
        for name in camera_paths:
            cause = causes[name]
            cast = np.isin(cause, [SHADOW_CAUSE_ID["cast_by_scene"], SHADOW_CAUSE_ID["cast_by_rover"]])
            sunlit = cause == SHADOW_CAUSE_ID["sunlit"]
            away = cause == SHADOW_CAUSE_ID["faces_away"]
            change = after[name] - baseline[name]
            row[name] = {
                "cast_pixels": int(cast.sum()),
                "cast_share_brightened": round(float((change[cast] > BRIGHTEN_THRESHOLD).mean()), 4) if cast.any() else None,
                "cast_median_luminance_before_after": [round(float(np.median(baseline[name][cast])), 4),
                                                       round(float(np.median(after[name][cast])), 4)] if cast.any() else None,
                "sunlit_mean_absolute_change": round(float(np.abs(change[sunlit]).mean()), 5),
                "faces_away_median_luminance_after": round(float(np.median(after[name][away])), 4) if away.any() else None,
                "restored_mean_absolute_change_all": round(float(np.abs(restored[name] - baseline[name]).mean()), 5)}
        results["candidates"][candidate] = row
        images[candidate] = after_rgb
        print(f"[shadows] {candidate}: {json.dumps(row)}", flush=True)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    names = list(camera_paths)
    figure, axes = plt.subplots(len(names), 1 + len(candidates), figsize=(4.2 * (1 + len(candidates)), 3.3 * len(names)),
                                dpi=100, squeeze=False)
    for row_index, name in enumerate(names):
        axes[row_index, 0].imshow(baseline_rgb[name])
        axes[row_index, 0].set_title(f"{name}: shadows on (baseline)", fontsize=8)
        for column_index, candidate in enumerate(candidates, start=1):
            axes[row_index, column_index].imshow(images[candidate][name])
            numbers = results["candidates"][candidate][name]
            axes[row_index, column_index].set_title(f"{candidate}\ncast brightened {numbers['cast_share_brightened']}, "
                                                    f"sunlit change {numbers['sunlit_mean_absolute_change']}", fontsize=7)
        for axis in axes[row_index]:
            axis.axis("off")
    figure.tight_layout()
    figure.savefig(run_directory / "shadows_off_comparison.png")
    summary = {"scene": scene.directory.name, "sun": {"azimuth_deg": configuration.lighting.sun_azimuth_deg,
                                                      "elevation_deg": configuration.lighting.sun_elevation_deg},
               "renderer": configuration.render.renderer, "warmup_subframes": arguments.warmup_subframes,
               "brighten_threshold": BRIGHTEN_THRESHOLD, "exposure_read_back": exposure_snapshot(),
               **candidate_info, **results}
    (run_directory / "result.json").write_text(json.dumps(summary, indent=1))
    print(f"[shadows] wrote {run_directory}", flush=True)
    rig.destroy()
    exit_code = 0
except Exception:
    import traceback
    traceback.print_exc()
    sys.stdout.flush()
finally:
    application.close(exit_code=exit_code)
