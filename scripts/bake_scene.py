"""Bake a scene (Isaac-free): configuration -> <data root>/scenes/<scene_id>/.

    $ISAAC_ENV/bin/python scripts/bake_scene.py --preset smoke [--scene apollo17_valley_floor] [--seed 7]
        [--set terrain.craters.equilibrium_coefficient=0.06] [--overwrite] [--data-root PATH]
Prints the scene directory on the last line.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lunar_rover_sim.core.configuration.configuration_loader import load_configuration  # noqa: E402
from lunar_rover_sim.core.scene.scene_bake_pipeline import bake_scene  # noqa: E402
from lunar_rover_sim.core.storage.data_root import DataRootError, require_data_root  # noqa: E402


def main() -> int:
    """Parse arguments, check the data root, bake, and print the scene directory."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--preset", default=None, help="smoke | gpu_8gb | gpu_32gb (default from the scene profile)")
    parser.add_argument("--scene", default="apollo17_valley_floor")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--set", dest="overrides", action="append", default=[], help="dotted override key=value")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--data-root", default=None)
    arguments = parser.parse_args()
    overrides = list(arguments.overrides) + ([f"scene_seed={arguments.seed}"] if arguments.seed is not None else [])
    configuration = load_configuration(arguments.preset, arguments.scene, overrides)
    try:
        data_root = require_data_root(arguments.data_root, configuration.storage.data_root,
                                      configuration.storage.allow_root_disk, configuration.storage.minimum_free_gib)
    except DataRootError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 4
    try:
        directory = bake_scene(configuration, data_root, overwrite=arguments.overwrite)
    except FileExistsError as error:
        print(f"{error}", file=sys.stderr)
        return 1
    print(directory)
    return 0


if __name__ == "__main__":
    sys.exit(main())
