"""SLDEM2015 backdrop DEM (Isaac-free): tile naming, byte plans, GeoTIFF round trip, vertical alignment, bake hash."""
from __future__ import annotations

import math

import numpy as np
import pytest

from lunar_rover_sim.core.configuration.configuration_loader import load_configuration, scene_identifier
from lunar_rover_sim.core.data_model.regular_grid import RegularGrid
from lunar_rover_sim.core.dem.geotiff_memmap_reader import open_height_raster
from lunar_rover_sim.core.dem.sldem_tiles import (BYTES_PER_SAMPLE, PIXELS_PER_DEGREE, TILE_COLUMNS, TILE_ROWS,
                                                  pixel_size_m, site_window, tile_file_name, window_fetch_plan,
                                                  write_float32_geotiff)
from lunar_rover_sim.core.frames.site_frame import EquirectangularSiteFrame
from lunar_rover_sim.core.terrain.tier_blending import vertical_offset_to_inner


def test_tile_names_follow_the_pds_listing():
    """Global row 0 is 60 N; the equator is '00n' as a south bound and '00s' as a north bound."""
    assert tile_file_name(0, 0) == "sldem2015_512_30n_60n_000_045_float.img"
    assert tile_file_name(TILE_ROWS, TILE_COLUMNS) == "sldem2015_512_00n_30n_045_090_float.img"
    assert tile_file_name(2 * TILE_ROWS, 7 * TILE_COLUMNS) == "sldem2015_512_30s_00s_315_360_float.img"
    assert tile_file_name(4 * TILE_ROWS - 1, 3 * TILE_COLUMNS) == "sldem2015_512_60s_30s_135_180_float.img"


def test_window_straddling_the_equator_and_180_deg_splits_into_tiles():
    """A window across 0 deg latitude and 180 deg longitude reads from four tiles, every sample exactly once."""
    window = site_window(180.0, 0.0, 5000.0)
    runs = window_fetch_plan(window)
    assert {run.tile_file_name for run in runs} == {
        "sldem2015_512_00n_30n_135_180_float.img", "sldem2015_512_00n_30n_180_225_float.img",
        "sldem2015_512_30s_00s_135_180_float.img", "sldem2015_512_30s_00s_180_225_float.img"}
    filled = np.zeros(window.shape, dtype=int)
    for run in runs:
        filled[run.output_row, run.output_column:run.output_column + run.sample_count] += 1
        assert run.byte_start % BYTES_PER_SAMPLE == 0
    assert (filled == 1).all()


def test_window_georeferencing_puts_the_site_inside_and_near_the_centre():
    """The site's map coordinates (standard parallel 0, central longitude 180) fall in the middle of the window."""
    longitude, latitude = -51.91616, 43.05776
    window = site_window(longitude, latitude, 56000.0)
    frame = EquirectangularSiteFrame(longitude, latitude, 0.0, 180.0)
    map_x, map_y = frame.map_origin_m
    column = (map_x - window.left_m) / pixel_size_m()
    row = (window.top_m - map_y) / pixel_size_m()
    rows, columns = window.shape
    assert abs(row - rows / 2) < 2 and abs(column - columns / 2) < 2
    assert rows == pytest.approx(2 * 56000.0 / pixel_size_m(), abs=10)
    assert columns == pytest.approx(rows / math.cos(math.radians(latitude)), abs=10)


def test_window_across_the_map_seam_or_outside_coverage_is_refused():
    """Longitude 0/360 is the seam of the central-longitude-180 frame; SLDEM2015 stops at 60 deg."""
    with pytest.raises(ValueError):
        site_window(0.1, 10.0, 20000.0)
    with pytest.raises(ValueError):
        site_window(30.0, 59.9, 20000.0)


def test_geotiff_written_is_read_back_by_the_memmap_reader(tmp_path):
    """Heights, size and georeferencing survive the round trip through the simulator's own reader."""
    heights = np.arange(12 * 7, dtype=np.float64).reshape(12, 7) * 0.5 - 2500.0
    path = tmp_path / "backdrop.tif"
    write_float32_geotiff(path, heights, -1000.0, 2000.0, 59.2, 59.2, "test")
    raster = open_height_raster(path)
    assert raster.heights.shape == (12, 7)
    assert np.array_equal(np.asarray(raster.heights), heights.astype(np.float32))
    assert (raster.left_m, raster.top_m, raster.pixel_width_m) == (-1000.0, 2000.0, 59.2)
    assert raster.map_from_pixel(0, 0) == pytest.approx((-1000.0 + 29.6, 2000.0 - 29.6))


def test_vertical_offset_is_the_median_difference_inside_the_inner_tier():
    """A backdrop 7 m too low, with a few outliers, is shifted by +7 m; points in the skirt are ignored."""
    inner = RegularGrid.centered(2048.0, 4.0)
    outer = RegularGrid.centered(10240.0, 128.0)
    east, north = inner.coordinate_mesh()
    inner_heights = 0.01 * east
    outer_east, outer_north = outer.coordinate_mesh()
    outer_heights = 0.01 * outer_east - 7.0
    outer_heights[outer.shape[0] // 2, outer.shape[1] // 2] += 300.0       # an outlier
    result = vertical_offset_to_inner(outer, outer_heights, inner, inner_heights, 128.0)
    assert result["offset_m"] == pytest.approx(7.0, abs=1e-9)
    assert result["median_absolute_deviation_m"] == pytest.approx(0.0, abs=1e-9)
    assert result["point_count"] == 15 * 15                                # |e|, |n| <= 1024 - 128 = 7 x 128 m


def test_bake_hash_of_sites_without_a_backdrop_dem_is_unchanged():
    """Scenes recorded with v1.0.8 keep their identifiers (the new site fields are left out while unused)."""
    configuration = load_configuration("gpu_8gb", "apollo17_valley_floor", ["scene_seed=124"])
    assert configuration.site.dem_backdrop_path is None
    assert scene_identifier(configuration) == "apollo17_landing_site_gpu_8gb_s124_9408c8"


def test_pixel_size_is_about_59_metres():
    """512 pixels per degree on the 1737.4 km sphere (label MAP_SCALE 0.0592252938 km)."""
    assert pixel_size_m() == pytest.approx(59.2252938, abs=1e-6)
    assert PIXELS_PER_DEGREE == 512
