"""SLDEM2015 tiles (LOLA + SELENE TC merged DEM, 512 pixels per degree) and a minimal float32 GeoTIFF writer.

SLDEM2015_512 tiles (PDS3 IMG, pds-geosciences.wustl.edu/lro/lro-l-lola-3-rdr-v1/lrolol_1xxx/data/sldem2015/tiles/
float_img/): little-endian float32 (PC_REAL) heights in km relative to a 1737.4 km sphere, simple cylindrical (centre
longitude 180 deg, no standard parallel), pixel registered, 30 deg of latitude x 45 deg of longitude per tile
(15360 x 23040 pixels), latitudes 60 S to 60 N, longitudes 0 to 360 E, row 0 = north. ~59.2 m per pixel.

``window_fetch_plan`` lists, per output row, the contiguous byte runs of the tiles that hold a window around a site,
so a window can be read with HTTP range requests instead of downloading whole 1.4 GB tiles. ``write_float32_geotiff``
writes the window as an uncompressed single-strip float32 GeoTIFF that ``geotiff_memmap_reader`` opens. Isaac-free.
"""
from __future__ import annotations

import math
import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np

PIXELS_PER_DEGREE = 512
TILE_LATITUDE_SPAN_DEG = 30
TILE_LONGITUDE_SPAN_DEG = 45
MAXIMUM_LATITUDE_DEG = 60
TILE_ROWS = TILE_LATITUDE_SPAN_DEG * PIXELS_PER_DEGREE          # 15360
TILE_COLUMNS = TILE_LONGITUDE_SPAN_DEG * PIXELS_PER_DEGREE      # 23040
BYTES_PER_SAMPLE = 4
REFERENCE_RADIUS_M = 1737400.0
GLOBAL_COLUMNS = 360 * PIXELS_PER_DEGREE
GLOBAL_ROWS = 2 * MAXIMUM_LATITUDE_DEG * PIXELS_PER_DEGREE
TILE_DIRECTORY_URL = ("https://pds-geosciences.wustl.edu/lro/lro-l-lola-3-rdr-v1/lrolol_1xxx/data/sldem2015/tiles/"
                      "float_img/")


def pixel_size_m() -> float:
    """Map metres per pixel (both axes) of the 512 pixels-per-degree grid on the 1737.4 km sphere."""
    return REFERENCE_RADIUS_M * math.radians(1.0 / PIXELS_PER_DEGREE)


def _latitude_label(latitude_deg: int, is_north_bound: bool) -> str:
    """'30n', '00n', '00s', '60s' as in the tile names (equator: '00n' as a south bound, '00s' as a north bound)."""
    if latitude_deg == 0:
        return "00s" if is_north_bound else "00n"
    return f"{abs(latitude_deg):02d}{'n' if latitude_deg > 0 else 's'}"


def tile_file_name(global_row: int, global_column: int) -> str:
    """Name of the 512 pixels-per-degree float tile holding global pixel (row from 60 N, column from 0 E)."""
    band = global_row // TILE_ROWS
    south = MAXIMUM_LATITUDE_DEG - TILE_LATITUDE_SPAN_DEG * (band + 1)
    north = south + TILE_LATITUDE_SPAN_DEG
    west = TILE_LONGITUDE_SPAN_DEG * (global_column // TILE_COLUMNS)
    return (f"sldem2015_512_{_latitude_label(south, False)}_{_latitude_label(north, True)}_"
            f"{west:03d}_{west + TILE_LONGITUDE_SPAN_DEG:03d}_float.img")


@dataclass(frozen=True)
class ByteRun:
    """One contiguous run of samples in one tile file, and where it lands in the output window."""

    tile_file_name: str
    byte_start: int
    sample_count: int
    output_row: int
    output_column: int


@dataclass(frozen=True)
class SiteWindow:
    """A rectangle of global pixels around a site and its georeferencing in the simulator's equirectangular frame.

    Map frame: standard parallel 0 deg, central longitude 180 deg (``EquirectangularSiteFrame`` with those values),
    PixelIsArea tie point at the outer corner of pixel (0, 0)."""

    global_row_start: int
    global_row_stop: int
    global_column_start: int
    global_column_stop: int

    @property
    def shape(self) -> tuple[int, int]:
        """(rows, columns) of the window."""
        return (self.global_row_stop - self.global_row_start, self.global_column_stop - self.global_column_start)

    @property
    def left_m(self) -> float:
        """Map x of the window's west edge (central longitude 180 deg: map_x = R * (longitude - 180 deg))."""
        west_longitude_deg = self.global_column_start / PIXELS_PER_DEGREE
        return REFERENCE_RADIUS_M * math.radians(west_longitude_deg - 180.0)

    @property
    def top_m(self) -> float:
        """Map y of the window's north edge."""
        north_latitude_deg = MAXIMUM_LATITUDE_DEG - self.global_row_start / PIXELS_PER_DEGREE
        return REFERENCE_RADIUS_M * math.radians(north_latitude_deg)


def site_window(longitude_deg: float, latitude_deg: float, half_size_m: float, margin_pixels: int = 4) -> SiteWindow:
    """Global pixel window covering a local square of half side ``half_size_m`` centred on the site.

    The local square maps to a map rectangle of half widths half_size / cos(latitude) (east) and half_size (north) in
    the standard-parallel-0 frame (the simulator's ``EquirectangularSiteFrame`` east scale). Raises ValueError when
    the window leaves the 60 S..60 N coverage or crosses the 0/360 deg seam of the map frame."""
    east_longitude = longitude_deg % 360.0
    half_latitude_deg = math.degrees(half_size_m / REFERENCE_RADIUS_M)
    half_longitude_deg = math.degrees(half_size_m / (REFERENCE_RADIUS_M * math.cos(math.radians(latitude_deg))))
    north = latitude_deg + half_latitude_deg
    south = latitude_deg - half_latitude_deg
    if north > MAXIMUM_LATITUDE_DEG or south < -MAXIMUM_LATITUDE_DEG:
        raise ValueError(f"window {south:.2f}..{north:.2f} deg leaves SLDEM2015 coverage (60 S..60 N)")
    row_start = math.floor((MAXIMUM_LATITUDE_DEG - north) * PIXELS_PER_DEGREE) - margin_pixels
    row_stop = math.ceil((MAXIMUM_LATITUDE_DEG - south) * PIXELS_PER_DEGREE) + margin_pixels
    column_start = math.floor((east_longitude - half_longitude_deg) * PIXELS_PER_DEGREE) - margin_pixels
    column_stop = math.ceil((east_longitude + half_longitude_deg) * PIXELS_PER_DEGREE) + margin_pixels
    if row_start < 0 or row_stop > GLOBAL_ROWS:
        raise ValueError("window rows leave the SLDEM2015 tiles")
    if column_start < 0 or column_stop > GLOBAL_COLUMNS:
        raise ValueError("window crosses longitude 0/360 deg, the seam of the central-longitude-180 map frame")
    return SiteWindow(row_start, row_stop, column_start, column_stop)


def window_fetch_plan(window: SiteWindow) -> list[ByteRun]:
    """Contiguous byte runs (one per output row and tile) that together fill the window."""
    runs = []
    for output_row, global_row in enumerate(range(window.global_row_start, window.global_row_stop)):
        tile_row = global_row % TILE_ROWS
        column = window.global_column_start
        while column < window.global_column_stop:
            tile_stop = (column // TILE_COLUMNS + 1) * TILE_COLUMNS
            stop = min(window.global_column_stop, tile_stop)
            tile_column = column % TILE_COLUMNS
            runs.append(ByteRun(tile_file_name(global_row, column),
                                (tile_row * TILE_COLUMNS + tile_column) * BYTES_PER_SAMPLE, stop - column,
                                output_row, column - window.global_column_start))
            column = stop
    return runs


def write_float32_geotiff(path: Path, heights_m: np.ndarray, left_m: float, top_m: float, pixel_width_m: float,
                          pixel_height_m: float, description: str = "") -> None:
    """Uncompressed, single-strip, little-endian float32 GeoTIFF (PixelIsArea tie point at pixel (0, 0)'s corner).

    Writes only the tags ``geotiff_memmap_reader`` needs plus ImageDescription; no GeoKey directory (the projection
    is recorded in the description and in the caller's sidecar)."""
    heights = np.ascontiguousarray(heights_m, dtype="<f4")
    rows, columns = heights.shape
    description_bytes = description.encode("ascii", errors="replace") + b"\x00"
    pixel_scale = struct.pack("<3d", pixel_width_m, pixel_height_m, 0.0)
    tie_point = struct.pack("<6d", 0.0, 0.0, 0.0, left_m, top_m, 0.0)
    # (tag, type, count, payload bytes); SHORT = 3, LONG = 4, ASCII = 2, DOUBLE = 12
    entries = [(256, 4, 1, struct.pack("<I", columns)), (257, 4, 1, struct.pack("<I", rows)),
               (258, 3, 1, struct.pack("<H", 32)), (259, 3, 1, struct.pack("<H", 1)), (262, 3, 1, struct.pack("<H", 1)),
               (270, 2, len(description_bytes), description_bytes), (273, 4, 1, None),
               (277, 3, 1, struct.pack("<H", 1)), (278, 4, 1, struct.pack("<I", rows)),
               (279, 4, 1, struct.pack("<I", heights.nbytes)), (284, 3, 1, struct.pack("<H", 1)),
               (339, 3, 1, struct.pack("<H", 3)), (33550, 12, 3, pixel_scale), (33922, 12, 6, tie_point)]
    directory_offset = 8
    directory_size = 2 + 12 * len(entries) + 4
    extra_offset = directory_offset + directory_size
    extra = bytearray()
    payload_offsets = {}
    for tag, _, _, payload in entries:
        if payload is not None and len(payload) > 4:
            payload_offsets[tag] = extra_offset + len(extra)
            extra += payload
            if len(extra) % 2:
                extra += b"\x00"
    data_offset = extra_offset + len(extra)
    data_offset += (-data_offset) % 8
    directory = bytearray(struct.pack("<H", len(entries)))
    for tag, field_type, count, payload in entries:
        if tag == 273:
            payload = struct.pack("<I", data_offset)
        value = struct.pack("<I", payload_offsets[tag]) if tag in payload_offsets else payload.ljust(4, b"\x00")
        directory += struct.pack("<HHI", tag, field_type, count) + value
    directory += struct.pack("<I", 0)
    with open(path, "wb") as file_handle:
        file_handle.write(b"II" + struct.pack("<H", 42) + struct.pack("<I", directory_offset))
        file_handle.write(directory)
        file_handle.write(extra)
        file_handle.write(b"\x00" * (data_offset - extra_offset - len(extra)))
        file_handle.write(heights.tobytes())
