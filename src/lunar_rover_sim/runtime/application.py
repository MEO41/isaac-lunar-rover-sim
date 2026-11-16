"""Start Isaac Sim (SimulationApp) with the project's guards: data root, free video memory, cache redirects.

Rule: ``start_simulation_application`` must run before any ``omni`` / ``pxr`` import (Isaac requirement).
"""
from __future__ import annotations

import os
import subprocess
import sys

from lunar_rover_sim.core.storage.data_root import DataRoot, DataRootError, require_data_root


def free_video_memory_mib() -> int:
    """Free memory of GPU 0 in MiB according to nvidia-smi (-1 if unavailable)."""
    try:
        output = subprocess.run(["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
                                capture_output=True, text=True, timeout=10).stdout
        return int(output.strip().splitlines()[0])
    except Exception:
        return -1


def process_video_memory_mib() -> int:
    """Video memory (MiB) used by this process according to nvidia-smi (0 if not listed, -1 on error)."""
    try:
        output = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"],
                                capture_output=True, text=True, timeout=10).stdout
        for line in output.strip().splitlines():
            process_id, memory = [part.strip() for part in line.split(",")]
            if int(process_id) == os.getpid():
                return int(memory)
        return 0
    except Exception:
        return -1


def redirect_caches(data_root: DataRoot) -> None:
    """Point the XDG, Warp and NVIDIA caches at the data root unless the caller already set them."""
    caches = data_root.caches_directory
    defaults = {
        "XDG_CACHE_HOME": caches / "xdg_cache",
        "WARP_CACHE_PATH": caches / "warp",
        "__GL_SHADER_DISK_CACHE_PATH": caches / "nvidia" / "GLCache",
        "CUDA_CACHE_PATH": caches / "nvidia" / "ComputeCache",
    }
    for name, path in defaults.items():
        if not os.environ.get(name):
            path.mkdir(parents=True, exist_ok=True)
            os.environ[name] = str(path)


def start_simulation_application(configuration, headless: bool = True, width: int = 1280, height: int = 720,
                                 data_root_argument: str | None = None):
    """Check guards, redirect caches, create SimulationApp and apply render settings. Returns (app, data_root)."""
    try:
        data_root = require_data_root(data_root_argument, configuration.storage.data_root,
                                      configuration.storage.allow_root_disk, configuration.storage.minimum_free_gib)
    except DataRootError as error:
        print(f"ERROR: {error}", file=sys.stderr, flush=True)
        raise SystemExit(4)
    required = configuration.launch.required_free_video_memory_mib
    free = free_video_memory_mib()
    if 0 <= free < required:
        print(f"ERROR: {free} MiB video memory free, preset '{configuration.profile.preset}' asks for {required} MiB. "
              "Close other GPU programs or use --preset smoke.", file=sys.stderr, flush=True)
        raise SystemExit(6)
    redirect_caches(data_root)
    from isaacsim import SimulationApp
    application = SimulationApp({"headless": headless, "width": width, "height": height,
                                 "renderer": configuration.render.renderer, "anti_aliasing": 0})
    import carb
    settings = carb.settings.get_settings()
    settings.set("/rtx/post/aa/op", int(configuration.render.anti_aliasing_mode))
    settings.set("/log/channels/omni.physx.plugin", "error")
    settings.set("/rtx/raytracing/fractionalCutoutOpacity", False)
    apply_sun_glare(configuration.render.sun_glare)
    apply_exposure(configuration.render.exposure)
    return application, data_root


EXPOSURE_SETTINGS = {
    "auto_exposure": "/rtx/post/histogram/enabled", "tonemap_operator": "/rtx/post/tonemap/op",
    "film_iso": "/rtx/post/tonemap/filmIso", "camera_shutter": "/rtx/post/tonemap/cameraShutter",
    "f_number": "/rtx/post/tonemap/fNumber", "exposure_compensation": "/rtx/post/tonemap/exposure/compensation",
    "white_scale": "/rtx/post/tonemap/whiteScale",
}


def apply_exposure(exposure) -> dict:
    """Write the pinned exposure / tone-mapping settings; returns the values read back (raises if one did not stick)."""
    import carb
    settings = carb.settings.get_settings()
    read_back = {}
    for name, path in EXPOSURE_SETTINGS.items():
        value = getattr(exposure, name)
        settings.set(path, value)
        read_back[name] = settings.get(path)
        if isinstance(value, bool) and bool(read_back[name]) != value or \
                not isinstance(value, bool) and abs(float(read_back[name]) - float(value)) > 1e-3 * max(1.0, abs(float(value))):
            raise RuntimeError(f"render setting {path} reads back {read_back[name]!r}, expected {value!r}")
    return read_back


def exposure_snapshot() -> dict:
    """Current exposure / tone-mapping settings (for run metadata)."""
    import carb
    settings = carb.settings.get_settings()
    return {name: settings.get(path) for name, path in EXPOSURE_SETTINGS.items()}


def apply_sun_glare(glare) -> None:
    """Apply the camera glare (bloom / lens flare) settings; can be called again at runtime to change them."""
    import carb
    settings = carb.settings.get_settings()
    settings.set("/rtx/post/lensFlares/enabled", bool(glare.enabled))
    settings.set("/rtx/post/lensFlares/flareScale", float(glare.scale))
    settings.set("/rtx/post/lensFlares/cutoffPoint", [float(glare.cutoff)] * 3)
    settings.set("/rtx/post/lensFlares/cutoffFuzziness", float(glare.cutoff_fuzziness))
    settings.set("/rtx/post/lensFlares/blades", int(glare.aperture_blades))
    settings.set("/rtx/post/lensFlares/fNumber", float(glare.f_stop))
    settings.set("/rtx/post/lensFlares/physicalSettings", 1)
    settings.set("/rtx/post/lensFlares/spectralBlurSamples", int(glare.spectral_samples))
    return {"cutoff_read_back": settings.get("/rtx/post/lensFlares/cutoffPoint"),
            "scale_read_back": settings.get("/rtx/post/lensFlares/flareScale")}
