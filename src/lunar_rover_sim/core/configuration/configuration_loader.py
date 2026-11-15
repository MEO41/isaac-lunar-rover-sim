"""Load and merge the YAML configuration files into a validated ``SimulatorConfiguration``.

Merge order (later wins): defaults <- sites/<site> <- scenes/<scene> <- rovers/<rover> <- presets/<preset> <- overrides.
The site and rover file names come from ``profile.site`` / ``profile.rover`` after the scene file is merged, so a
scene can pick its site and rover. ``${NAME}`` in strings is expanded from the environment and ``.env.local``.
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any

import yaml

from lunar_rover_sim.core.configuration.configuration_models import SimulatorConfiguration
from lunar_rover_sim.core.storage.data_root import PROJECT_ROOT, read_env_local

CONFIGURATION_DIRECTORY = PROJECT_ROOT / "configs"
BAKE_HASH_SECTIONS = ("site", "terrain", "rocks", "regolith")
ENVIRONMENT_REFERENCE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")


class ConfigurationError(ValueError):
    """Raised for missing files, unknown keys, bad overrides or unresolved ${NAME} references."""


def read_yaml_file(path: Path) -> dict[str, Any]:
    """Read one YAML mapping; an empty file gives an empty dict."""
    if not path.exists():
        raise ConfigurationError(f"configuration file not found: {path}")
    content = yaml.safe_load(path.read_text()) or {}
    if not isinstance(content, dict):
        raise ConfigurationError(f"{path} must contain a mapping at the top level")
    return content


def deep_merge(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    """Return base updated recursively by overlay (nested dicts merge, everything else is replaced)."""
    merged = copy.deepcopy(base)
    for key, value in overlay.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def expand_environment_references(value: Any, environment: dict[str, str]) -> Any:
    """Replace ${NAME} in every string of a nested structure; raise ConfigurationError if NAME is undefined."""
    if isinstance(value, dict):
        return {key: expand_environment_references(item, environment) for key, item in value.items()}
    if isinstance(value, list):
        return [expand_environment_references(item, environment) for item in value]
    if isinstance(value, str):
        def substitute(match: re.Match) -> str:
            """Return the value of one ${NAME} reference."""
            name = match.group(1)
            if name not in environment:
                raise ConfigurationError(f"${{{name}}} is not set in the environment or .env.local")
            return environment[name]
        return ENVIRONMENT_REFERENCE.sub(substitute, value)
    return value


def apply_dotted_override(configuration: dict[str, Any], override: str) -> None:
    """Apply one ``a.b.c=value`` override in place; the value is parsed as YAML. The key must already exist."""
    if "=" not in override:
        raise ConfigurationError(f"override '{override}' is not of the form key.path=value")
    dotted_key, raw_value = override.split("=", 1)
    keys = dotted_key.strip().split(".")
    node = configuration
    for key in keys[:-1]:
        if not isinstance(node.get(key), dict):
            raise ConfigurationError(f"override '{override}': '{key}' is not a configuration section")
        node = node[key]
    if keys[-1] not in node:
        raise ConfigurationError(f"override '{override}': unknown key '{keys[-1]}'")
    node[keys[-1]] = yaml.safe_load(raw_value)


def load_configuration_dictionary(preset_name: str | None = None, scene_name: str = "apollo17_valley_floor",
                                  overrides: list[str] | None = None,
                                  configuration_directory: Path = CONFIGURATION_DIRECTORY) -> dict[str, Any]:
    """Merge the YAML files and overrides into one plain dictionary (environment references expanded)."""
    merged = read_yaml_file(configuration_directory / "defaults.yaml")
    scene_overlay = read_yaml_file(configuration_directory / "scenes" / f"{scene_name}.yaml")
    profile = deep_merge(merged.get("profile", {}), scene_overlay.get("profile", {}))
    merged = deep_merge(merged, read_yaml_file(configuration_directory / "sites" / f"{profile['site']}.yaml"))
    merged = deep_merge(merged, scene_overlay)
    merged = deep_merge(merged, read_yaml_file(configuration_directory / "rovers" / f"{profile['rover']}.yaml"))
    preset_name = preset_name or profile.get("preset", "gpu_8gb")
    merged = deep_merge(merged, read_yaml_file(configuration_directory / "presets" / f"{preset_name}.yaml"))
    merged["profile"]["preset"] = preset_name
    for override in overrides or []:
        apply_dotted_override(merged, override)
    environment = {**read_env_local(), **os.environ}
    return expand_environment_references(merged, environment)


def load_configuration(preset_name: str | None = None, scene_name: str = "apollo17_valley_floor",
                       overrides: list[str] | None = None,
                       configuration_directory: Path = CONFIGURATION_DIRECTORY) -> SimulatorConfiguration:
    """Merge and validate the configuration. Raises ConfigurationError or pydantic.ValidationError."""
    dictionary = load_configuration_dictionary(preset_name, scene_name, overrides, configuration_directory)
    return SimulatorConfiguration.model_validate(dictionary)


def _strip_machine_paths(value: Any) -> Any:
    """Replace absolute file paths by their file names so the bake hash does not depend on the machine."""
    if isinstance(value, dict):
        return {key: _strip_machine_paths(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_strip_machine_paths(item) for item in value]
    if isinstance(value, str) and os.sep in value:
        return Path(value).name
    return value


def bake_hash(configuration: SimulatorConfiguration) -> str:
    """First 6 hex digits of the SHA-256 of everything a baked scene depends on (site, terrain, rocks, regolith, seed)."""
    dumped = configuration.model_dump(mode="json")
    hashed = {section: _strip_machine_paths(dumped[section]) for section in BAKE_HASH_SECTIONS}
    hashed["scene_seed"] = configuration.scene_seed
    hashed["generator_version"] = configuration.generator_version
    canonical = json.dumps(hashed, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:6]


def scene_identifier(configuration: SimulatorConfiguration) -> str:
    """Scene directory name: <site>_<preset>_s<seed>_<hash6>."""
    return (f"{configuration.site.name}_{configuration.profile.preset}_s{configuration.scene_seed}_"
            f"{bake_hash(configuration)}")
