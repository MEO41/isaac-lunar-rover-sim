#!/usr/bin/env bash
# Run a script with the lunar_regolith Python (REGOLITH_PYTHON: rasterio, pyproj, usd-core) - NOT Isaac's Python.
#   scripts/regolith_run.sh scripts/build_regolith_appearance.py [arguments]
# Code: the PINNED submodule external/lunar_regolith (its src goes first on PYTHONPATH, ahead of the working copy's
# editable install). Data: REGOLITH_RUN in the working copy. Working directory: REGOLITH_REPOSITORY, so the module's
# relative data paths resolve. Never edit files inside the submodule; changes go upstream.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
[ -f "$HERE/.env.local" ] && set -a && . "$HERE/.env.local" && set +a
: "${REGOLITH_PYTHON:?set REGOLITH_PYTHON in .env.local}"
: "${REGOLITH_REPOSITORY:?set REGOLITH_REPOSITORY in .env.local}"
: "${REGOLITH_RUN:?set REGOLITH_RUN in .env.local}"
REGOLITH_PYTHON="${REGOLITH_PYTHON/#\~/$HOME}"; REGOLITH_REPOSITORY="${REGOLITH_REPOSITORY/#\~/$HOME}"
export REGOLITH_RUN="${REGOLITH_RUN/#\~/$HOME}" REGOLITH_REPOSITORY
[ -f "$HERE/external/lunar_regolith/src/lunar_regolith/__init__.py" ] || {
  echo "ERROR: submodule missing; run: git submodule update --init external/lunar_regolith" >&2; exit 2; }
export LUNAR_ROVER_SIM_PROJECT="$HERE"
export PYTHONPATH="$HERE/external/lunar_regolith/src:$HERE/src${PYTHONPATH:+:$PYTHONPATH}"
script="$1"; shift
[ -f "$script" ] || script="$HERE/$script"
script="$(cd "$(dirname "$script")" && pwd)/$(basename "$script")"
cd "$REGOLITH_REPOSITORY"
exec "$REGOLITH_PYTHON" "$script" "$@"
