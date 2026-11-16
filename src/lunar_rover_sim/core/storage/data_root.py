"""Data root resolution and the startup check.

All generated files live under one data root on the large disk (DISK02). That disk is auto-mounted at login and is
not in /etc/fstab, so it can be missing; every entry point calls ``require_data_root()`` first. The check never
creates the root and never falls back to the root disk.

Command line: ``python -m lunar_rover_sim.core.storage.data_root --require [--data-root PATH]`` prints the resolved
root and exits 0, or prints the reason and exits 4 (used by scripts/isaac_run.sh).
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
import uuid
from dataclasses import dataclass
from pathlib import Path

DEFAULT_DATA_ROOT = "/media/noron/DISK02/isaac-lunar-rover-sim-data"
DATA_ROOT_ENVIRONMENT_VARIABLE = "LUNAR_ROVER_SIM_DATA_ROOT"
PROJECT_ROOT = Path(__file__).resolve().parents[4]
SUBDIRECTORIES = ("scenes", "runs", "runs/logs", "runs/pytest_tmp", "caches")


class DataRootError(RuntimeError):
    """Raised when the data root is missing, unwritable, on the root disk or too full."""


@dataclass(frozen=True)
class DataRoot:
    """A checked data root and the standard locations below it."""

    path: Path

    @property
    def scenes_directory(self) -> Path:
        """Directory holding baked scenes (one subdirectory per scene_id)."""
        return self.path / "scenes"

    @property
    def runs_directory(self) -> Path:
        """Directory holding outputs of smoke tests, probes and drive scripts."""
        return self.path / "runs"

    @property
    def logs_directory(self) -> Path:
        """Directory holding Isaac process logs."""
        return self.path / "runs" / "logs"

    @property
    def caches_directory(self) -> Path:
        """Directory holding redirected caches."""
        return self.path / "caches"

    def run_directory(self, run_name: str, timestamp: str) -> Path:
        """Create and return ``runs/<run_name>/<timestamp>/``."""
        directory = self.runs_directory / run_name / timestamp
        directory.mkdir(parents=True, exist_ok=True)
        return directory


def read_env_local(project_root: Path = PROJECT_ROOT) -> dict[str, str]:
    """Parse ``KEY=VALUE`` lines of the project's .env.local (missing file gives an empty dict)."""
    values: dict[str, str] = {}
    env_file = project_root / ".env.local"
    if not env_file.exists():
        return values
    for line in env_file.read_text().splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        values[key.strip()] = value.strip().strip('"').strip("'")
    return values


def environment_value(name: str) -> str | None:
    """Return a setting from the process environment, falling back to .env.local."""
    if os.environ.get(name):
        return os.environ[name]
    return read_env_local().get(name)


def resolve_data_root(command_line_path: str | None = None, configured_path: str | None = None) -> Path:
    """Resolve the data root: command line > LUNAR_ROVER_SIM_DATA_ROOT (env or .env.local) > configuration > default."""
    for candidate in (command_line_path, environment_value(DATA_ROOT_ENVIRONMENT_VARIABLE), configured_path):
        if candidate:
            return Path(os.path.expanduser(candidate)).absolute()
    return Path(DEFAULT_DATA_ROOT)


def check_data_root(path: Path, allow_root_disk: bool = False, minimum_free_gib: float = 20.0) -> None:
    """Raise DataRootError unless ``path`` is an existing, writable directory off the root disk with enough free space."""
    hint = (
        "Is DISK02 mounted? It is auto-mounted at login and not in /etc/fstab. "
        "Create the data root once with scripts/setup_data_root.sh. Refusing to fall back to the root disk."
    )
    if not path.exists():
        raise DataRootError(f"Data root {path} does not exist. {hint}")
    if not path.is_dir():
        raise DataRootError(f"Data root {path} is not a directory. {hint}")
    probe_file = path / f".write_probe_{os.getpid()}_{uuid.uuid4().hex[:8]}"
    try:
        probe_file.write_bytes(b"probe")
        probe_file.unlink()
    except OSError as error:
        raise DataRootError(f"Data root {path} is not writable ({error}). {hint}") from error
    if not allow_root_disk and os.stat(path).st_dev == os.stat("/").st_dev:
        raise DataRootError(f"Data root {path} is on the root disk (/). {hint} "
                            "Set storage.allow_root_disk: true only on machines where that is intended.")
    free_gib = shutil.disk_usage(path).free / 2**30
    if free_gib < minimum_free_gib:
        raise DataRootError(f"Data root {path} has only {free_gib:.1f} GiB free (< {minimum_free_gib} GiB required).")


ALLOW_ROOT_DISK_ENVIRONMENT_VARIABLE = "LUNAR_ROVER_SIM_ALLOW_ROOT_DISK"


def root_disk_allowed_on_this_machine() -> bool:
    """True when this machine opts in to a data root on the root disk (LUNAR_ROVER_SIM_ALLOW_ROOT_DISK=1 in the
    environment or .env.local), for machines whose root disk is their only, large disk. Default: refuse."""
    return environment_value(ALLOW_ROOT_DISK_ENVIRONMENT_VARIABLE) == "1"


def require_data_root(command_line_path: str | None = None, configured_path: str | None = None,
                      allow_root_disk: bool = False, minimum_free_gib: float = 20.0) -> DataRoot:
    """Resolve and check the data root, create its standard subdirectories, and return it. Raises DataRootError.

    The root disk is accepted when ``allow_root_disk`` (configuration) or the machine opt-in is set."""
    path = resolve_data_root(command_line_path, configured_path)
    allow_root_disk = allow_root_disk or root_disk_allowed_on_this_machine()
    check_data_root(path, allow_root_disk=allow_root_disk, minimum_free_gib=minimum_free_gib)
    for subdirectory in SUBDIRECTORIES:
        (path / subdirectory).mkdir(parents=True, exist_ok=True)
    return DataRoot(path=path)


def main(argument_list: list[str] | None = None) -> int:
    """Command-line entry: ``--require`` checks the root and prints it; returns the process exit code (0 or 4)."""
    parser = argparse.ArgumentParser(description="Resolve and check the lunar_rover_sim data root.")
    parser.add_argument("--require", action="store_true", help="check the root and create its subdirectories")
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--allow-root-disk", action="store_true")
    parser.add_argument("--minimum-free-gib", type=float, default=20.0)
    arguments = parser.parse_args(argument_list)
    if not arguments.require:
        print(resolve_data_root(arguments.data_root))
        return 0
    try:
        data_root = require_data_root(arguments.data_root, allow_root_disk=arguments.allow_root_disk,
                                      minimum_free_gib=arguments.minimum_free_gib)
    except DataRootError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 4
    print(data_root.path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
