"""Find a drivable origin for a new site: the flat 40 m patch of an LROC NAC DTM closest to a target point.

    $ISAAC_ENV/bin/python scripts/find_flat_site_origin.py --dem /path/NAC_DTM_CHANGE3.TIF --standard-parallel-deg 44 \
        --target-longitude-deg -19.5116 --target-latitude-deg 44.1214 [--search-radius-m 600]

No Isaac. Reads the DTM through the simulator's own GeoTIFF reader (equirectangular DTMs only, as the simulator).
For every DTM post within ``search-radius-m`` of the target it measures, over a square box (default 40 m, the
drivable workspace), the mean and maximum slope from the DTM gradient (true metres, east rescaled as in
``EquirectangularSiteFrame``), and whether the near-ring square (208 m + 12.8 m skirt, which the bake requires to be
complete) holds valid DTM data everywhere, and whether the bake's DTM crop
(regional extent 2048 m, skirt excluded, / 2 + 8 m, plus 16 spline posts) lies inside the raster (holes are allowed
there, the raster edge is not). It prints the target's own statistics, the closest post that meets the
slope limits, and the regional-tier (2304 m) coverage there (the regional tier tolerates holes; the bake fills them).

The slope limits are ASSUMPTIONS: no rougher than the Apollo 17 lunar-module patch, where the scripted expert works
(measured with this script on NAC_DTM_APOLLO17_2M: mean 3.29, max 8.51 deg; Apollo 15's patch: mean 1.9, max 4.7).
Slopes are DTM-scale (2-5 m posts); the synthetic sub-DEM relief added by the bake makes the driven ground rougher.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from lunar_rover_sim.core.dem.geotiff_memmap_reader import open_height_raster  # noqa: E402
from lunar_rover_sim.core.frames.site_frame import EquirectangularSiteFrame  # noqa: E402

NEAR_RING_SQUARE_M = 208.0 + 2 * 12.8
REGIONAL_SQUARE_M = 2048.0 + 2 * 128.0
REGIONAL_EXTENT_M = 2048.0         # tier layout: regional grid extent (skirt excluded)
BAKE_CROP_MARGIN_M = 8.0          # build_terrain: sampler half size = outermost fine tier extent / 2 + 8 m
SPLINE_MARGIN_POSTS = 16          # dem_window.SPLINE_MARGIN_POSTS


def parse_arguments() -> argparse.Namespace:
    """Command-line arguments (see the module docstring)."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dem", required=True, help="LROC NAC DTM GeoTIFF (equirectangular)")
    parser.add_argument("--standard-parallel-deg", type=float, required=True, help="CENTER_LATITUDE of the PDS label")
    parser.add_argument("--central-longitude-deg", type=float, default=180.0)
    parser.add_argument("--target-longitude-deg", type=float, required=True, help="east-positive, -180..360")
    parser.add_argument("--target-latitude-deg", type=float, required=True)
    parser.add_argument("--search-radius-m", type=float, default=600.0)
    parser.add_argument("--box-size-m", type=float, default=40.0)
    parser.add_argument("--maximum-mean-slope-deg", type=float, default=3.5)
    parser.add_argument("--maximum-slope-deg", type=float, default=9.0)
    return parser.parse_args()


def square_valid_fraction(raster, row: float, column: float, size_m: float) -> float:
    """Fraction of valid DTM posts in a square of ``size_m`` centred on (row, column); posts outside the raster count
    as invalid."""
    half_rows = int(math.ceil(size_m / 2.0 / raster.pixel_height_m))
    half_columns = int(math.ceil(size_m / 2.0 / raster.pixel_width_m))
    row_start, row_stop = int(round(row)) - half_rows, int(round(row)) + half_rows + 1
    column_start, column_stop = int(round(column)) - half_columns, int(round(column)) + half_columns + 1
    total = (row_stop - row_start) * (column_stop - column_start)
    inside_rows = (max(row_start, 0), min(row_stop, raster.row_count))
    inside_columns = (max(column_start, 0), min(column_stop, raster.column_count))
    if inside_rows[0] >= inside_rows[1] or inside_columns[0] >= inside_columns[1]:
        return 0.0
    window = raster.read_window(inside_rows[0], inside_rows[1], inside_columns[0], inside_columns[1])
    return float(np.isfinite(window).sum()) / total


def main() -> int:
    """Measure slopes around the target and print the chosen origin as JSON."""
    arguments = parse_arguments()
    raster = open_height_raster(arguments.dem)
    target_frame = EquirectangularSiteFrame(arguments.target_longitude_deg % 360.0, arguments.target_latitude_deg,
                                            arguments.standard_parallel_deg, arguments.central_longitude_deg)
    target_row, target_column = raster.fractional_pixel_from_map(*target_frame.map_origin_m)
    east_post_m = raster.pixel_width_m * target_frame.east_scale
    north_post_m = raster.pixel_height_m
    margin_m = arguments.search_radius_m + NEAR_RING_SQUARE_M / 2.0
    half_rows, half_columns = int(margin_m / north_post_m) + 2, int(margin_m / east_post_m) + 2
    row_start = max(int(round(target_row)) - half_rows, 0)
    row_stop = min(int(round(target_row)) + half_rows + 1, raster.row_count)
    column_start = max(int(round(target_column)) - half_columns, 0)
    column_stop = min(int(round(target_column)) + half_columns + 1, raster.column_count)
    if row_start >= row_stop or column_start >= column_stop:
        print(json.dumps({"error": "target is outside the DTM", "target_row": float(target_row),
                          "target_column": float(target_column), "rows": raster.row_count,
                          "columns": raster.column_count}))
        return 1
    heights = raster.read_window(row_start, row_stop, column_start, column_stop)
    valid = np.isfinite(heights)
    filled = np.where(valid, heights, np.nanmean(heights))
    gradient_north, gradient_east = np.gradient(filled, north_post_m, east_post_m)
    slope_deg = np.degrees(np.arctan(np.hypot(gradient_north, gradient_east)))
    slope_deg[~ndimage.binary_erosion(valid, iterations=1)] = np.nan
    box_posts = (max(int(round(arguments.box_size_m / north_post_m)), 1),
                 max(int(round(arguments.box_size_m / east_post_m)), 1))
    slope_known = np.isfinite(slope_deg)
    mean_slope = (ndimage.uniform_filter(np.where(slope_known, slope_deg, 0.0), box_posts, mode="constant")
                  / np.maximum(ndimage.uniform_filter(slope_known.astype(float), box_posts, mode="constant"), 1e-9))
    maximum_slope = ndimage.maximum_filter(np.where(slope_known, slope_deg, 90.0), box_posts, mode="constant",
                                           cval=90.0)
    near_posts = (int(round(NEAR_RING_SQUARE_M / north_post_m)) | 1, int(round(NEAR_RING_SQUARE_M / east_post_m)) | 1)
    near_complete = ndimage.minimum_filter(valid.astype(np.uint8), near_posts, mode="constant", cval=0) == 1
    rows, columns = np.mgrid[row_start:row_stop, column_start:column_stop]
    distance_m = np.hypot((rows - target_row) * north_post_m, (columns - target_column) * east_post_m)
    crop_half_rows = int(math.ceil((REGIONAL_EXTENT_M / 2 + BAKE_CROP_MARGIN_M) / north_post_m)) + SPLINE_MARGIN_POSTS + 1
    crop_half_columns = int(math.ceil((REGIONAL_EXTENT_M / 2 + BAKE_CROP_MARGIN_M) / east_post_m)) + SPLINE_MARGIN_POSTS + 1
    crop_inside = ((rows - crop_half_rows >= 0) & (rows + crop_half_rows < raster.row_count)
                   & (columns - crop_half_columns >= 0) & (columns + crop_half_columns < raster.column_count))
    candidate = (near_complete & crop_inside & (distance_m <= arguments.search_radius_m)
                 & (mean_slope <= arguments.maximum_mean_slope_deg) & (maximum_slope <= arguments.maximum_slope_deg))

    def describe(local_row: int, local_column: int) -> dict:
        """Statistics of one post of the window: geographic position, distance, slopes, coverage."""
        row, column = local_row + row_start, local_column + column_start
        map_x, map_y = raster.map_from_pixel(row, column)
        longitude = (math.degrees(map_x / (target_frame.body_radius_m
                                           * math.cos(math.radians(arguments.standard_parallel_deg))))
                     + arguments.central_longitude_deg + 180.0) % 360.0 - 180.0
        latitude = math.degrees(map_y / target_frame.body_radius_m)
        return {"origin_longitude_deg": round(longitude, 5), "origin_latitude_deg": round(latitude, 5),
                "distance_from_target_m": round(float(distance_m[local_row, local_column]), 1),
                "box_mean_slope_deg": round(float(mean_slope[local_row, local_column]), 2),
                "box_maximum_slope_deg": round(float(maximum_slope[local_row, local_column]), 2),
                "near_ring_complete": bool(near_complete[local_row, local_column]),
                "bake_crop_inside_raster": bool(crop_inside[local_row, local_column]),
                "regional_valid_fraction": round(square_valid_fraction(raster, row, column, REGIONAL_SQUARE_M), 3)}

    target_local = (int(round(target_row)) - row_start, int(round(target_column)) - column_start)
    inside = 0 <= target_local[0] < heights.shape[0] and 0 <= target_local[1] < heights.shape[1]
    report = {"dem": Path(arguments.dem).name, "post_spacing_m": [round(east_post_m, 3), round(north_post_m, 3)],
              "limits": {"box_size_m": arguments.box_size_m, "maximum_mean_slope_deg": arguments.maximum_mean_slope_deg,
                         "maximum_slope_deg": arguments.maximum_slope_deg},
              "target": describe(*target_local) if inside else "outside the DTM window"}
    if candidate.any():
        flat_index = np.argmin(np.where(candidate, distance_m, np.inf))
        report["chosen"] = describe(*np.unravel_index(flat_index, candidate.shape))
    else:
        score = np.where(near_complete & crop_inside & (distance_m <= arguments.search_radius_m), mean_slope, np.inf)
        report["chosen"] = None
        if np.isfinite(score).any():
            report["flattest_within_radius"] = describe(*np.unravel_index(np.argmin(score), score.shape))
    print(json.dumps(report, indent=1))
    return 0 if candidate.any() else 2


if __name__ == "__main__":
    raise SystemExit(main())
