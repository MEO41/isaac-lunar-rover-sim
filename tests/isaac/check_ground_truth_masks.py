"""Check the ray-cast ground-truth masks against the renderer (Isaac, headless).

    scripts/isaac_run.sh tests/isaac/check_ground_truth_masks.py [--preset gpu_8gb] [--drive-s 8]
        [--sun-azimuth-deg 90] [--sun-elevation-deg 15] [--copy-figure]

The rover drives forward (leaving tracks), then the front and rear cameras are rendered with the RTX annotators
(rgb, distance_to_image_plane, semantic_segmentation) and our masks are ray cast for the same instant. Measured per
camera (result.json under <data root>/runs/check_ground_truth_masks/<timestamp>/):
  * depth: |ours - renderer| on pixels where both see a surface (median, 95th percentile, share within 1 % + 5 mm)
  * classes: share of the renderer's rover / rock / regolith / sky pixels that our mask labels the same
    (the renderer has no label for pebbles, so those are not scored)
  * shadows: luminance of ground pixels we call shadowed vs sunlit (medians, and the probability that a random
    shadowed pixel is darker than a random sunlit one)
Writes the label images (object_mask / shadow_label / rock_instance PNG, legend.json) and a comparison figure;
--copy-figure also copies it to docs/images/ground_truth_masks.png.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT / "src"))

from lunar_rover_sim.core.configuration.configuration_loader import load_configuration, scene_identifier  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("--preset", default="gpu_8gb")
parser.add_argument("--drive-s", type=float, default=8.0)
parser.add_argument("--sun-azimuth-deg", type=float, default=None)
parser.add_argument("--sun-elevation-deg", type=float, default=None)
parser.add_argument("--copy-figure", action="store_true")
arguments = parser.parse_args()
overrides = []
if arguments.sun_azimuth_deg is not None:
    overrides.append(f"lighting.sun_azimuth_deg={arguments.sun_azimuth_deg}")
if arguments.sun_elevation_deg is not None:
    overrides.append(f"lighting.sun_elevation_deg={arguments.sun_elevation_deg}")
configuration = load_configuration(arguments.preset, overrides=overrides)
from lunar_rover_sim.runtime.application import process_video_memory_mib, start_simulation_application  # noqa: E402

application, data_root = start_simulation_application(configuration, headless=True)

import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

from lunar_rover_sim.core.ground_truth.mask_classes import (GROUND_OBJECT_IDS, OBJECT_CLASSES, OBJECT_ID,  # noqa: E402
                                                            SHADOW_CLASSES, SHADOW_ID, colorize, legend)
from lunar_rover_sim.core.scene.scene_loader import load_scene  # noqa: E402
from lunar_rover_sim.runtime.driving.rover_drive_session import RoverDriveSession  # noqa: E402
from lunar_rover_sim.runtime.sensors.camera_rig import CameraRig, save_png  # noqa: E402
from lunar_rover_sim.runtime.sensors.ground_truth_masks import GroundTruthMaskRenderer  # noqa: E402
from lunar_rover_sim.runtime.stepping import render_synchronously  # noqa: E402


def renderer_categories(semantic_id: np.ndarray, id_to_labels: dict) -> np.ndarray:
    """Per-pixel category from the semantic annotator: 'rover', 'rock', 'regolith' or '' (unlabelled/background)."""
    lookup = {}
    for key, value in id_to_labels.items():
        text = json.dumps(value).lower()
        lookup[int(key)] = next((name for name in ("rover", "rock", "regolith") if name in text), "")
    categories = np.full(semantic_id.shape, "", dtype=object)
    for key, name in lookup.items():
        if name:
            categories[semantic_id == key] = name
    return categories


def probability_darker(shadowed: np.ndarray, sunlit: np.ndarray, random_generator, samples: int = 200000) -> float:
    """P(random shadowed pixel luminance < random sunlit pixel luminance) + 0.5 P(equal) (Monte Carlo)."""
    if len(shadowed) == 0 or len(sunlit) == 0:
        return float("nan")
    first = random_generator.choice(shadowed, samples)
    second = random_generator.choice(sunlit, samples)
    return float(np.mean(first < second) + 0.5 * np.mean(first == second))


def compare(name: str, rendered: dict, images, random_generator) -> dict:
    """Depth, class and shadow agreement numbers of one camera."""
    depth_renderer = np.asarray(rendered["depth_m"], dtype=np.float64)
    depth_ours = images.depth_m.astype(np.float64)
    renderer_surface = np.isfinite(depth_renderer) & (depth_renderer > 0) & (depth_renderer < 1e5)
    ours_surface = np.isfinite(depth_ours)
    both = renderer_surface & ours_surface
    error = np.abs(depth_ours - depth_renderer)[both]
    tolerance = 0.01 * depth_renderer[both] + 0.005
    categories = renderer_categories(rendered["semantic_id"], rendered["semantic_labels"])
    object_mask = images.object_mask
    ours_rock = np.isin(object_mask, [OBJECT_ID["boulder"], OBJECT_ID["small_rock"]])
    ours_ground = np.isin(object_mask, GROUND_OBJECT_IDS)
    ours_sky = np.isin(object_mask, [OBJECT_ID["sky"], OBJECT_ID["sun"], OBJECT_ID["sun_glare"]])
    agreement = {}
    for category, ours in (("rover", object_mask == OBJECT_ID["rover"]), ("rock", ours_rock), ("regolith", ours_ground)):
        selected = categories == category
        agreement[category] = {"renderer_pixels": int(selected.sum()),
                               "share_labelled_the_same": round(float(ours[selected].mean()), 4) if selected.any() else None}
    renderer_sky = ~renderer_surface
    agreement["sky"] = {"renderer_pixels": int(renderer_sky.sum()),
                        "share_labelled_the_same": round(float(ours_sky[renderer_sky].mean()), 4) if renderer_sky.any() else None}
    scored = (categories != "") | renderer_sky
    same = np.zeros(object_mask.shape, dtype=bool)
    same |= (categories == "rover") & (object_mask == OBJECT_ID["rover"])
    same |= (categories == "rock") & ours_rock
    same |= (categories == "regolith") & ours_ground
    same |= renderer_sky & ours_sky
    rgb = rendered["rgb"].astype(np.float64) / 255.0
    luminance = 0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]
    shadow = images.shadow_label
    sunlit = luminance[shadow == SHADOW_ID["sunlit_ground"]]
    shadowed = luminance[np.isin(shadow, [SHADOW_ID["shadowed_ground"], SHADOW_ID["rover_shadow"]])]
    counts = {mask_class.name: int((object_mask == mask_class.id).sum()) for mask_class in OBJECT_CLASSES}
    shadow_counts = {mask_class.name: int((shadow == mask_class.id).sum()) for mask_class in SHADOW_CLASSES}
    return {"camera": name,
            "depth": {"pixels_compared": int(both.sum()),
                      "median_absolute_error_m": round(float(np.median(error)), 5) if len(error) else None,
                      "p95_absolute_error_m": round(float(np.percentile(error, 95)), 5) if len(error) else None,
                      "share_within_1_percent_plus_5_mm": round(float(np.mean(error <= tolerance)), 4) if len(error) else None,
                      "surface_disagreement_pixels": int((renderer_surface ^ ours_surface).sum())},
            "classes": {"overall_share_same": round(float(same[scored].mean()), 4), "pixels_scored": int(scored.sum()),
                        "per_category": agreement, "ours_counts": counts},
            "shadows": {"counts": shadow_counts,
                        "median_luminance_sunlit_ground": round(float(np.median(sunlit)), 4) if len(sunlit) else None,
                        "median_luminance_shadowed_ground": round(float(np.median(shadowed)), 4) if len(shadowed) else None,
                        "probability_shadowed_darker_than_sunlit": round(probability_darker(shadowed, sunlit,
                                                                                            random_generator), 4)}}


exit_code = 1
try:
    scene = load_scene(data_root.scenes_directory / scene_identifier(configuration), configuration)
    run_directory = data_root.run_directory("check_ground_truth_masks", time.strftime("%Y%m%d_%H%M%S"))
    print(f"[masks] scene {scene.directory}\n[masks] run directory {run_directory}", flush=True)
    session = RoverDriveSession(application, scene, stepper_mode="explicit")
    session.start()
    control_rate = configuration.drive.control_rate_hz
    for _ in range(int(arguments.drive_s * control_rate)):
        session.advance_control_step(0.3, 0.05)
    for _ in range(control_rate):
        session.advance_control_step(0.0, 0.0)
    resolution = tuple(configuration.render.camera_resolution_px)
    cameras = [name for name in ("front", "rear") if getattr(configuration.rover.cameras, name).enabled]
    rig = CameraRig()
    for name in cameras:
        rig.add_camera(name, session.rover_paths.cameras[name], resolution,
                       ("rgb", "distance_to_image_plane", "semantic_segmentation"))
    if session.track_runtime is not None:
        session.track_runtime.force_mask_upload()
    render_synchronously(configuration.render.warmup_subframes_after_reset + 8)
    started = time.perf_counter()
    mask_renderer = GroundTruthMaskRenderer(session)
    build_seconds = time.perf_counter() - started
    print(f"[masks] warp meshes built in {build_seconds:.1f} s: {mask_renderer.face_counts}", flush=True)
    random_generator = np.random.default_rng(0)
    results, rows = [], []
    for name in cameras:
        rendered = rig.read(name)
        mask_renderer.render(session.rover_paths.cameras[name], resolution)     # first call compiles the kernels
        started = time.perf_counter()
        images = mask_renderer.render(session.rover_paths.cameras[name], resolution)
        mask_seconds = time.perf_counter() - started
        numbers = compare(name, rendered, images, random_generator)
        numbers["mask_render_ms"] = round(1000 * mask_seconds, 1)
        results.append(numbers)
        print(f"[masks] {name}: {json.dumps({k: numbers[k] for k in ('depth', 'mask_render_ms')})}", flush=True)
        print(f"[masks] {name} classes: {json.dumps(numbers['classes']['per_category'])} overall "
              f"{numbers['classes']['overall_share_same']}", flush=True)
        print(f"[masks] {name} shadows: {json.dumps(numbers['shadows'])}", flush=True)
        Image.fromarray(images.object_mask).save(run_directory / f"{name}_object_mask.png")
        Image.fromarray(images.shadow_label).save(run_directory / f"{name}_shadow_label.png")
        Image.fromarray(images.rock_instance.astype(np.uint16)).save(run_directory / f"{name}_rock_instance.png")
        save_png(run_directory / f"{name}_rgb.png", rendered["rgb"])
        rows.append((name, rendered["rgb"], colorize(images.object_mask, OBJECT_CLASSES),
                     colorize(images.shadow_label, SHADOW_CLASSES)))
    (run_directory / "legend.json").write_text(json.dumps({**legend(), "metadata": mask_renderer.metadata()}, indent=1))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    figure, axes = plt.subplots(len(rows), 3, figsize=(13.5, 3.6 * len(rows)), dpi=110, squeeze=False)
    for row_index, (name, rgb, object_colour, shadow_colour) in enumerate(rows):
        for column_index, (image, title) in enumerate(((rgb, f"{name} camera (RTX)"), (object_colour, "object mask"),
                                                       (shadow_colour, "shadow label"))):
            axes[row_index, column_index].imshow(image)
            axes[row_index, column_index].set_title(title, fontsize=9)
            axes[row_index, column_index].axis("off")
    present_objects = {OBJECT_CLASSES[i] for i in range(len(OBJECT_CLASSES)) if any(
        (r["classes"]["ours_counts"][OBJECT_CLASSES[i].name] > 0) for r in results)}
    present_shadows = {SHADOW_CLASSES[i] for i in range(len(SHADOW_CLASSES)) if any(
        (r["shadows"]["counts"][SHADOW_CLASSES[i].name] > 0) for r in results)}
    axes[-1, 1].legend(handles=[Patch(color=np.array(c.rgb) / 255, label=c.name) for c in OBJECT_CLASSES if c in present_objects],
                       loc="upper center", bbox_to_anchor=(0.5, -0.02), ncol=4, fontsize=7, frameon=False)
    axes[-1, 2].legend(handles=[Patch(color=np.array(c.rgb) / 255, label=c.name) for c in SHADOW_CLASSES if c in present_shadows],
                       loc="upper center", bbox_to_anchor=(0.5, -0.02), ncol=3, fontsize=7, frameon=False)
    figure.suptitle(f"Ray-cast ground truth (MARE classes), Sun azimuth {configuration.lighting.sun_azimuth_deg} deg, "
                    f"elevation {configuration.lighting.sun_elevation_deg} deg", fontsize=10)
    figure.tight_layout()
    figure.savefig(run_directory / "ground_truth_masks.png")
    if arguments.copy_figure:
        shutil.copy(run_directory / "ground_truth_masks.png", PROJECT / "docs" / "images" / "ground_truth_masks.png")
    summary = {"scene": scene.directory.name, "preset": arguments.preset, "warp_build_seconds": round(build_seconds, 2),
               "face_counts": mask_renderer.face_counts, "metadata": mask_renderer.metadata(), "cameras": results,
               "sun": {"azimuth_deg": configuration.lighting.sun_azimuth_deg,
                       "elevation_deg": configuration.lighting.sun_elevation_deg},
               "video_memory_mib": process_video_memory_mib()}
    (run_directory / "result.json").write_text(json.dumps(summary, indent=1))
    print(f"[masks] video memory {summary['video_memory_mib']} MiB; wrote {run_directory}", flush=True)
    rig.destroy()
    exit_code = 0
except Exception:
    import traceback
    traceback.print_exc()
    sys.stdout.flush()
finally:
    application.close(exit_code=exit_code)
