"""Camera exposure rule (Isaac-free): metered film ISO and the v1.0.9 defaults."""
from __future__ import annotations

import math

import pytest

from lunar_rover_sim.core.configuration.configuration_loader import load_configuration
from lunar_rover_sim.core.perception.camera_exposure import metered_film_iso


def test_fixed_rule_and_unknown_sun_keep_the_configured_iso():
    """The fixed rule ignores the Sun; before any Sun is set the configured ISO is used."""
    assert metered_film_iso(100.0, "fixed", 10.0, 25.0, 5.0) == 100.0
    assert metered_film_iso(100.0, "sun_elevation", None, 25.0, 5.0) == 100.0


def test_sun_elevation_rule_keeps_flat_ground_irradiance_times_iso_constant():
    """ISO x sin(elevation) is the same at every elevation above the minimum; the reference elevation gives the base."""
    assert metered_film_iso(100.0, "sun_elevation", 25.0, 25.0, 5.0) == pytest.approx(100.0)
    for elevation in (10.0, 40.0, 70.0):
        iso = metered_film_iso(100.0, "sun_elevation", elevation, 25.0, 5.0)
        assert iso * math.sin(math.radians(elevation)) == pytest.approx(100.0 * math.sin(math.radians(25.0)))
    assert metered_film_iso(100.0, "sun_elevation", 10.0, 25.0, 5.0) == pytest.approx(243.4, abs=0.1)


def test_gain_stops_growing_below_the_minimum_elevation():
    """A grazing or set Sun gets the ISO of the minimum elevation, never an unbounded one."""
    at_minimum = metered_film_iso(100.0, "sun_elevation", 5.0, 25.0, 5.0)
    assert metered_film_iso(100.0, "sun_elevation", 1.0, 25.0, 5.0) == at_minimum
    assert metered_film_iso(100.0, "sun_elevation", -3.0, 25.0, 5.0) == at_minimum


def test_invalid_rule_or_limits_are_refused():
    """Unknown rules and inconsistent elevation limits raise instead of guessing."""
    with pytest.raises(ValueError):
        metered_film_iso(100.0, "histogram", 10.0, 25.0, 5.0)
    with pytest.raises(ValueError):
        metered_film_iso(100.0, "sun_elevation", 10.0, 25.0, 30.0)


def test_defaults_are_a_linear_sensor_metered_to_the_sun():
    """v1.0.9 defaults: tone-map operator 1 (linear + sRGB), ISO 100 at a 25 deg reference Sun, auto exposure off."""
    exposure = load_configuration("gpu_8gb").render.exposure
    assert (exposure.tonemap_operator, exposure.film_iso, exposure.rule) == (1, 100.0, "sun_elevation")
    assert (exposure.reference_elevation_deg, exposure.minimum_elevation_deg, exposure.auto_exposure) == (25.0, 5.0,
                                                                                                         False)
