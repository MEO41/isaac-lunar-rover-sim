"""Site frame: equirectangular map metres of the LROC NAC DTM <-> local east-north-up metres.

The DTM is in the "Equirectangular Moon" projection with standard parallel 20 deg and central meridian 180 deg on a
1737.4 km sphere:  map_x = R * radians(longitude - 180) * cos(20 deg),  map_y = R * radians(latitude).
A map metre east is therefore not a true metre east away from 20 deg latitude; the local frame rescales x by
cos(latitude_origin) / cos(standard_parallel) so that one local unit is one true metre (a 2 m DTM pixel is 1.9976 true
metres east-west at the Apollo 17 site).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class EquirectangularSiteFrame:
    """Local tangent frame (x east, y north, metres) anchored at a site origin on an equirectangular map."""

    origin_longitude_deg: float
    origin_latitude_deg: float
    standard_parallel_deg: float = 20.0
    central_longitude_deg: float = 180.0
    body_radius_m: float = 1737400.0

    @property
    def map_origin_m(self) -> tuple[float, float]:
        """Map coordinates (map_x, map_y) of the origin, metres."""
        return self.map_from_geographic(self.origin_longitude_deg, self.origin_latitude_deg)

    @property
    def east_scale(self) -> float:
        """True metres east per map metre east at the origin latitude."""
        return math.cos(math.radians(self.origin_latitude_deg)) / math.cos(math.radians(self.standard_parallel_deg))

    def map_from_geographic(self, longitude_deg, latitude_deg):
        """Equirectangular map metres from longitude/latitude in degrees (scalars or arrays)."""
        longitude_difference = (np.asarray(longitude_deg, dtype=np.float64) - self.central_longitude_deg + 180.0) % 360.0 - 180.0
        map_x = self.body_radius_m * np.radians(longitude_difference) * math.cos(math.radians(self.standard_parallel_deg))
        map_y = self.body_radius_m * np.radians(np.asarray(latitude_deg, dtype=np.float64))
        if np.ndim(map_x) == 0:
            return float(map_x), float(map_y)
        return map_x, map_y

    def local_from_map(self, map_x_m, map_y_m):
        """Local (east_m, north_m) from map metres; float64."""
        origin_x, origin_y = self.map_origin_m
        east = (np.asarray(map_x_m, dtype=np.float64) - origin_x) * self.east_scale
        north = np.asarray(map_y_m, dtype=np.float64) - origin_y
        return east, north

    def map_from_local(self, east_m, north_m):
        """Map metres (map_x, map_y) from local (east_m, north_m); float64."""
        origin_x, origin_y = self.map_origin_m
        map_x = np.asarray(east_m, dtype=np.float64) / self.east_scale + origin_x
        map_y = np.asarray(north_m, dtype=np.float64) + origin_y
        return map_x, map_y


def planetary_curvature_drop_m(east_m, north_m, body_radius_m: float = 1737400.0):
    """Height drop of a sphere below its tangent plane at horizontal distance sqrt(e^2+n^2): d^2 / (2 R), metres."""
    east = np.asarray(east_m, dtype=np.float64)
    north = np.asarray(north_m, dtype=np.float64)
    return (east * east + north * north) / (2.0 * body_radius_m)
