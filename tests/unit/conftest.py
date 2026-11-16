"""Shared fixtures of the Isaac-free unit tests: configuration sections read from configs/defaults.yaml."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from lunar_rover_sim.core.configuration.configuration_models import (  # noqa: E402
    ExpertConfiguration, GroundTruthConfiguration, RoverConfiguration)


@pytest.fixture(scope="session")
def defaults() -> dict:
    """The parsed defaults file."""
    return yaml.safe_load((PROJECT_ROOT / "configs" / "defaults.yaml").read_text())


@pytest.fixture(scope="session")
def expert_configuration(defaults) -> ExpertConfiguration:
    """Default expert section."""
    return ExpertConfiguration(**defaults["expert"])


@pytest.fixture(scope="session")
def rover_configuration(defaults) -> RoverConfiguration:
    """Default rover section."""
    return RoverConfiguration(**defaults["rover"])


@pytest.fixture(scope="session")
def ground_truth_configuration(defaults) -> GroundTruthConfiguration:
    """Default ground-truth section."""
    return GroundTruthConfiguration(**defaults["ground_truth"])


@pytest.fixture
def random_generator() -> np.random.Generator:
    """Fixed-seed generator."""
    return np.random.default_rng(1234)
