"""Minimal GeoTIFF reader for uncompressed, single-band float32 DEMs (LROC NAC DTM products), without rasterio.

It parses the TIFF tags it needs (size, strips, sample format, GeoTIFF pixel scale and tie point, GDAL NoData) and
exposes the raster as a read-only ``np.memmap``. Compressed, tiled or multi-band files are rejected with a clear
error. The file is never opened for writing.

Georeferencing convention (GDAL "PixelIsArea"): the tie point is the outer corner of pixel (0, 0), so the centre of
pixel (row, column) is at map_x = left + (column + 0.5) * pixel_width, map_y = top - (row + 0.5) * pixel_height.
"""
from __future__ import annotations

import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np

TAG_IMAGE_WIDTH = 256
TAG_IMAGE_LENGTH = 257
TAG_BITS_PER_SAMPLE = 258
TAG_COMPRESSION = 259
TAG_STRIP_OFFSETS = 273
TAG_SAMPLES_PER_PIXEL = 277
TAG_ROWS_PER_STRIP = 278
TAG_STRIP_BYTE_COUNTS = 279
TAG_PLANAR_CONFIGURATION = 284
TAG_TILE_WIDTH = 322
TAG_SAMPLE_FORMAT = 339
TAG_MODEL_PIXEL_SCALE = 33550
TAG_MODEL_TIEPOINT = 33922
TAG_GDAL_NODATA = 42113

TIFF_TYPE_FORMATS = {1: "B", 2: "s", 3: "H", 4: "I", 5: "II", 11: "f", 12: "d", 16: "Q"}
TIFF_TYPE_SIZES = {1: 1, 2: 1, 3: 2, 4: 4, 5: 8, 11: 4, 12: 8, 16: 8}


class UnsupportedTiffError(ValueError):
    """Raised for TIFF files this minimal reader does not handle (compressed, tiled, multi-band, not float32)."""


@dataclass(frozen=True)
class GeoTiffHeightRaster:
    """A read-only height raster with its georeferencing. ``heights`` is indexed [row, column], row 0 = north."""

    path: Path
    heights: np.ndarray
    left_m: float
    top_m: float
    pixel_width_m: float
    pixel_height_m: float
    nodata_value: float | None

    @property
    def row_count(self) -> int:
        """Number of rows (north-south pixels)."""
        return self.heights.shape[0]

    @property
    def column_count(self) -> int:
        """Number of columns (east-west pixels)."""
        return self.heights.shape[1]

    def fractional_pixel_from_map(self, map_x_m, map_y_m):
        """Fractional (row, column) of pixel centres for map coordinates; row/column 0.0 is the centre of pixel 0."""
        column = (np.asarray(map_x_m, dtype=np.float64) - self.left_m) / self.pixel_width_m - 0.5
        row = (self.top_m - np.asarray(map_y_m, dtype=np.float64)) / self.pixel_height_m - 0.5
        return row, column

    def map_from_pixel(self, row, column):
        """Map coordinates (map_x, map_y) of the centre of pixel (row, column)."""
        map_x = self.left_m + (np.asarray(column, dtype=np.float64) + 0.5) * self.pixel_width_m
        map_y = self.top_m - (np.asarray(row, dtype=np.float64) + 0.5) * self.pixel_height_m
        return map_x, map_y

    def read_window(self, row_start: int, row_stop: int, column_start: int, column_stop: int) -> np.ndarray:
        """Copy a window into float64 with NoData (and non-finite or < -1e30 values) replaced by NaN.

        Raises ValueError if the window is not fully inside the raster (no silent clamping)."""
        if row_start < 0 or column_start < 0 or row_stop > self.row_count or column_stop > self.column_count \
                or row_start >= row_stop or column_start >= column_stop:
            raise ValueError(f"window rows [{row_start}, {row_stop}) columns [{column_start}, {column_stop}) is outside "
                             f"the raster of {self.row_count} x {self.column_count} pixels ({self.path.name})")
        window = np.array(self.heights[row_start:row_stop, column_start:column_stop], dtype=np.float64)
        invalid = ~np.isfinite(window) | (window < -1.0e30)
        if self.nodata_value is not None:
            invalid |= window == np.float64(np.float32(self.nodata_value))
        window[invalid] = np.nan
        return window


def _read_tags(file_handle, byte_order: str) -> dict[int, tuple]:
    """Read all entries of the first image file directory into {tag: tuple of values}."""
    file_handle.seek(4)
    (first_directory_offset,) = struct.unpack(byte_order + "I", file_handle.read(4))
    file_handle.seek(first_directory_offset)
    (entry_count,) = struct.unpack(byte_order + "H", file_handle.read(2))
    tags: dict[int, tuple] = {}
    for _ in range(entry_count):
        tag, field_type, count, value_or_offset = struct.unpack(byte_order + "HHI4s", file_handle.read(12))
        if field_type not in TIFF_TYPE_FORMATS:
            continue
        size = TIFF_TYPE_SIZES[field_type] * count
        position = file_handle.tell()
        if size <= 4:
            raw = value_or_offset[:size]
        else:
            (offset,) = struct.unpack(byte_order + "I", value_or_offset)
            file_handle.seek(offset)
            raw = file_handle.read(size)
            file_handle.seek(position)
        if field_type == 2:
            tags[tag] = (raw.rstrip(b"\x00").decode("ascii", errors="replace"),)
        else:
            tags[tag] = struct.unpack(byte_order + TIFF_TYPE_FORMATS[field_type] * count, raw)
    return tags


def open_height_raster(path: str | Path) -> GeoTiffHeightRaster:
    """Open an uncompressed single-band float32 GeoTIFF as a read-only memmap. Raises UnsupportedTiffError otherwise."""
    path = Path(path)
    with open(path, "rb") as file_handle:
        header = file_handle.read(4)
        if header[:2] == b"II":
            byte_order = "<"
        elif header[:2] == b"MM":
            byte_order = ">"
        else:
            raise UnsupportedTiffError(f"{path.name} is not a TIFF file")
        if struct.unpack(byte_order + "H", header[2:4])[0] != 42:
            raise UnsupportedTiffError(f"{path.name} is a BigTIFF or not a classic TIFF; not supported")
        tags = _read_tags(file_handle, byte_order)

    def single(tag: int, default=None):
        """Return the first value of a tag, or the default when the tag is absent."""
        return tags[tag][0] if tag in tags else default

    if single(TAG_COMPRESSION, 1) != 1:
        raise UnsupportedTiffError(f"{path.name} is compressed (compression={single(TAG_COMPRESSION)}); not supported")
    if TAG_TILE_WIDTH in tags:
        raise UnsupportedTiffError(f"{path.name} is tiled; only striped files are supported")
    if single(TAG_SAMPLES_PER_PIXEL, 1) != 1:
        raise UnsupportedTiffError(f"{path.name} has more than one band; not supported")
    if single(TAG_BITS_PER_SAMPLE) != 32 or single(TAG_SAMPLE_FORMAT, 1) != 3:
        raise UnsupportedTiffError(f"{path.name} is not float32 (bits {single(TAG_BITS_PER_SAMPLE)}, "
                                   f"sample format {single(TAG_SAMPLE_FORMAT)})")
    width = int(single(TAG_IMAGE_WIDTH))
    height = int(single(TAG_IMAGE_LENGTH))
    strip_offsets = np.asarray(tags[TAG_STRIP_OFFSETS], dtype=np.int64)
    rows_per_strip = int(single(TAG_ROWS_PER_STRIP, height))
    row_bytes = width * 4
    expected = strip_offsets[0] + np.arange(len(strip_offsets), dtype=np.int64) * rows_per_strip * row_bytes
    if not np.array_equal(strip_offsets, expected):
        raise UnsupportedTiffError(f"{path.name}: strips are not contiguous; not supported by the memmap reader")
    if TAG_MODEL_PIXEL_SCALE not in tags or TAG_MODEL_TIEPOINT not in tags:
        raise UnsupportedTiffError(f"{path.name} has no GeoTIFF pixel scale / tie point")
    scale_x, scale_y = tags[TAG_MODEL_PIXEL_SCALE][:2]
    tie_i, tie_j, _, tie_x, tie_y, _ = tags[TAG_MODEL_TIEPOINT][:6]
    nodata_text = single(TAG_GDAL_NODATA)
    nodata_value = float(nodata_text) if nodata_text not in (None, "") else None
    dtype = np.dtype(byte_order + "f4")
    heights = np.memmap(path, dtype=dtype, mode="r", offset=int(strip_offsets[0]), shape=(height, width))
    return GeoTiffHeightRaster(
        path=path, heights=heights,
        left_m=float(tie_x - tie_i * scale_x), top_m=float(tie_y + tie_j * scale_y),
        pixel_width_m=float(scale_x), pixel_height_m=float(scale_y), nodata_value=nodata_value,
    )
