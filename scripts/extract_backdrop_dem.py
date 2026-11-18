"""Cut a backdrop DEM for a site out of SLDEM2015 (512 pixels per degree, ~59 m) with HTTP range requests.

    python3 scripts/extract_backdrop_dem.py change5_site [more site names ... | --all] [--half-size-m 56000]
        [--workers 8] [--overwrite]

No Isaac. For each site config (configs/sites/<name>.yaml) that sets ``dem_backdrop_path`` (``--all``: every such
site), it reads the origin, fetches only the bytes of the SLDEM2015 float tiles that cover a local square of half side
``half-size-m`` (one range request per row and tile, about 10 KB each, instead of whole 1.4 GB tiles), converts km to
m and writes the file named by ``dem_backdrop_path`` (uncompressed float32, standard parallel 0 deg, central
longitude 180 deg: the config's ``dem_backdrop_*`` projection defaults) plus a JSON sidecar with the source files,
the window and the height difference to the site's fine NAC DTM at the origin (heights of both are relative to the
1737.4 km sphere; the bake removes the remaining offset, see terrain_builder). An existing file is kept unless
``--overwrite``. Default half size 56 km: the 102.4 km backdrop tier (51.2 km) + 2 spacings + 16 spline posts, with
margin.

Source: LOLA / SELENE TC SLDEM2015 (Barker et al. 2016, Icarus 273), PDS LRO-L-LOLA-4-GDR-V1.0,
pds-geosciences.wustl.edu/lro/lro-l-lola-3-rdr-v1/lrolol_1xxx/data/sldem2015/tiles/float_img/.
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import yaml

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))

from lunar_rover_sim.core.dem.geotiff_memmap_reader import open_height_raster  # noqa: E402
from lunar_rover_sim.core.dem.sldem_tiles import (BYTES_PER_SAMPLE, TILE_DIRECTORY_URL, pixel_size_m,  # noqa: E402
                                                  site_window, window_fetch_plan, write_float32_geotiff)
from lunar_rover_sim.core.frames.site_frame import EquirectangularSiteFrame  # noqa: E402


def fetch_run(run, attempts: int = 5) -> np.ndarray:
    """Download one byte run (HTTP range request) as float32 km; retries with back-off."""
    url = TILE_DIRECTORY_URL + run.tile_file_name
    last_byte = run.byte_start + run.sample_count * BYTES_PER_SAMPLE - 1
    request = urllib.request.Request(url, headers={"Range": f"bytes={run.byte_start}-{last_byte}",
                                                   "User-Agent": "isaac-lunar-rover-sim backdrop extractor"})
    for attempt in range(attempts):
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                payload = response.read()
            if response.status != 206 or len(payload) != run.sample_count * BYTES_PER_SAMPLE:
                raise IOError(f"unexpected reply {response.status}, {len(payload)} bytes")
            return np.frombuffer(payload, dtype="<f4")
        except Exception:                                          # network errors: retry, then give up loudly
            if attempt == attempts - 1:
                raise
            time.sleep(2.0 * (attempt + 1))
    raise RuntimeError("unreachable")


def load_site(name: str) -> dict:
    """The site section of configs/sites/<name>.yaml with ${NAC_DEM_DIRECTORY} expanded."""
    site = yaml.safe_load((PROJECT / "configs" / "sites" / f"{name}.yaml").read_text())["site"]
    return {key: os.path.expandvars(value) if isinstance(value, str) else value for key, value in site.items()}


def extract(site_name: str, half_size_m: float, workers: int, overwrite: bool) -> dict | None:
    """Fetch, write and check the backdrop DEM of one site; returns the sidecar (None when the file already exists)."""
    site = load_site(site_name)
    if not site.get("dem_backdrop_path"):
        raise SystemExit(f"{site_name}: the site config sets no dem_backdrop_path")
    output = Path(site["dem_backdrop_path"])
    if "$" in str(output):
        raise SystemExit(f"{site_name}: {output} has an unset environment variable (NAC_DEM_DIRECTORY?)")
    if output.exists() and not overwrite:
        return None
    window = site_window(site["origin_longitude_deg"], site["origin_latitude_deg"], half_size_m)
    runs = window_fetch_plan(window)
    heights_km = np.full(window.shape, np.nan, dtype=np.float32)
    started = time.time()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for run, samples in zip(runs, pool.map(fetch_run, runs)):
            heights_km[run.output_row, run.output_column:run.output_column + run.sample_count] = samples
    if not np.isfinite(heights_km).all():
        raise RuntimeError("window not completely filled")
    description = (f"SLDEM2015 512 ppd crop for {site['name']}; heights m rel. 1737.4 km sphere; simple cylindrical, "
                   f"standard parallel 0, central longitude 180")
    write_float32_geotiff(output, heights_km.astype(np.float64) * 1000.0, window.left_m, window.top_m,
                          pixel_size_m(), pixel_size_m(), description)

    raster = open_height_raster(output)                            # read back through the simulator's own reader
    backdrop_frame = EquirectangularSiteFrame(site["origin_longitude_deg"], site["origin_latitude_deg"], 0.0, 180.0)
    row, column = raster.fractional_pixel_from_map(*backdrop_frame.map_origin_m)
    backdrop_origin_m = float(raster.heights[int(round(float(row))), int(round(float(column)))])
    fine = open_height_raster(site["dem_fine_path"])
    fine_frame = EquirectangularSiteFrame(site["origin_longitude_deg"], site["origin_latitude_deg"],
                                          site["standard_parallel_deg"], site.get("central_longitude_deg", 180.0))
    fine_row, fine_column = fine.fractional_pixel_from_map(*fine_frame.map_origin_m)
    fine_window = fine.read_window(int(fine_row) - 8, int(fine_row) + 9, int(fine_column) - 8, int(fine_column) + 9)
    fine_origin_m = float(np.nanmedian(fine_window))
    sidecar = {
        "site": site["name"], "file": output.name, "source": "SLDEM2015 512 ppd float tiles (LOLA + SELENE TC), "
        "PDS LRO-L-LOLA-4-GDR-V1.0", "source_url": TILE_DIRECTORY_URL,
        "tiles": sorted({run.tile_file_name for run in runs}),
        "window_global_pixels": {"row_start": window.global_row_start, "row_stop": window.global_row_stop,
                                 "column_start": window.global_column_start, "column_stop": window.global_column_stop},
        "shape": list(window.shape), "pixel_size_m": pixel_size_m(), "half_size_m": half_size_m,
        "projection": {"standard_parallel_deg": 0.0, "central_longitude_deg": 180.0, "body_radius_m": 1737400.0},
        "height_reference": "m relative to the 1737.4 km sphere",
        "height_at_origin_m": backdrop_origin_m, "fine_dtm_height_at_origin_m": fine_origin_m,
        "difference_backdrop_minus_fine_m": backdrop_origin_m - fine_origin_m,
        "requests": len(runs), "seconds": round(time.time() - started, 1),
        "created": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
    }
    output.with_suffix(".json").write_text(json.dumps(sidecar, indent=1))
    return sidecar


def main() -> None:
    """Parse arguments and extract every requested site."""
    parser = argparse.ArgumentParser()
    parser.add_argument("sites", nargs="*")
    parser.add_argument("--all", action="store_true", help="every configs/sites/*.yaml that sets dem_backdrop_path")
    parser.add_argument("--half-size-m", type=float, default=56000.0)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--overwrite", action="store_true", help="fetch again even if the file exists")
    arguments = parser.parse_args()
    names = arguments.sites
    if arguments.all:
        names = sorted(path.stem for path in (PROJECT / "configs" / "sites").glob("*.yaml")
                       if "dem_backdrop_path" in (yaml.safe_load(path.read_text()).get("site") or {}))
    for name in names:
        sidecar = extract(name, arguments.half_size_m, arguments.workers, arguments.overwrite)
        if sidecar is None:
            print(f"[backdrop] {name}: {load_site(name)['dem_backdrop_path']} exists (kept)", flush=True)
            continue
        print(f"[backdrop] {name}: {sidecar['file']} {sidecar['shape']} from {len(sidecar['tiles'])} tile(s), "
              f"{sidecar['requests']} requests in {sidecar['seconds']} s; SLDEM - NAC at origin "
              f"{sidecar['difference_backdrop_minus_fine_m']:+.1f} m", flush=True)


if __name__ == "__main__":
    main()
