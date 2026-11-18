"""Camera exposure rule: the film ISO the RTX tone mapper gets for a given Sun elevation (Isaac-free).

``fixed``: the configured ISO at every Sun elevation; frame brightness follows the irradiance (v1.0.8 behaviour).
``sun_elevation``: ISO = configured ISO x sin(reference elevation) / sin(Sun elevation), so sunlit level ground keeps
the brightness it has at the reference elevation, as a rover camera with auto exposure would. ASSUMPTION: real
auto exposure meters the frame histogram; this rule meters only the flat-ground irradiance (cos of the Sun's zenith
angle), which keeps every frame a deterministic function of the scenario. Below ``minimum_elevation_deg`` the gain
stops growing (a grazing Sun would otherwise ask for an unbounded ISO).
"""
from __future__ import annotations

import math

EXPOSURE_RULES = ("fixed", "sun_elevation")


def metered_film_iso(film_iso: float, rule: str, sun_elevation_deg: float | None, reference_elevation_deg: float,
                     minimum_elevation_deg: float) -> float:
    """Film ISO for the Sun elevation under ``rule``; the configured ISO when the elevation is unknown (None)."""
    if rule not in EXPOSURE_RULES:
        raise ValueError(f"unknown exposure rule {rule!r}; expected one of {EXPOSURE_RULES}")
    if rule == "fixed" or sun_elevation_deg is None:
        return float(film_iso)
    if not 0.0 < minimum_elevation_deg <= reference_elevation_deg <= 90.0:
        raise ValueError("need 0 < minimum elevation <= reference elevation <= 90 deg")
    elevation = max(float(sun_elevation_deg), minimum_elevation_deg)
    return float(film_iso) * math.sin(math.radians(reference_elevation_deg)) / math.sin(math.radians(elevation))
