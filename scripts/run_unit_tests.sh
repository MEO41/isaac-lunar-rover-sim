#!/usr/bin/env bash
# Isaac-free unit tests with the Isaac environment's Python (no SimulationApp). pytest temporary files go to the data root.
#   scripts/run_unit_tests.sh [pytest arguments]
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
[ -f "$HERE/.env.local" ] && set -a && . "$HERE/.env.local" && set +a
: "${ISAAC_ENV:?set ISAAC_ENV in .env.local}"
export PYTHONPATH="$HERE/src${PYTHONPATH:+:$PYTHONPATH}"
ROOT="$("$ISAAC_ENV/bin/python" -m lunar_rover_sim.core.storage.data_root --require)" || exit 4
export WARP_CACHE_PATH="${WARP_CACHE_PATH:-$ROOT/caches/warp}"   # Warp kernel cache on the data root, not ~/.cache
mkdir -p "$WARP_CACHE_PATH"
cd "$HERE"
exec "$ISAAC_ENV/bin/python" -m pytest tests/unit --basetemp="$ROOT/runs/pytest_tmp" -p no:cacheprovider "$@"
