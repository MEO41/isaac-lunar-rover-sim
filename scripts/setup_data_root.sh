#!/usr/bin/env bash
# Create the data root once, after checking that its parent is on a mounted disk other than /.
#   scripts/setup_data_root.sh [--data-root PATH] [--seed-caches]
# --seed-caches copies ~/.cache/ov and ~/.local/share/ov into the redirected cache locations so the first Isaac
# launch with the redirect does not recompile every shader.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
[ -f "$HERE/.env.local" ] && set -a && . "$HERE/.env.local" && set +a
ROOT="${LUNAR_ROVER_SIM_DATA_ROOT:-/media/noron/DISK02/isaac-lunar-rover-sim-data}"
SEED_CACHES=0
while [ $# -gt 0 ]; do
  case "$1" in
    --data-root) ROOT="$2"; shift 2 ;;
    --seed-caches) SEED_CACHES=1; shift ;;
    *) echo "unknown argument $1" >&2; exit 2 ;;
  esac
done
PARENT="$(dirname "$ROOT")"
[ -d "$PARENT" ] || { echo "ERROR: $PARENT does not exist. Is DISK02 mounted? (auto-mounted at login, not in /etc/fstab)" >&2; exit 4; }
MOUNT_POINT="$(findmnt -n -o TARGET -T "$PARENT")"
if [ "$MOUNT_POINT" = "/" ]; then
  echo "ERROR: $PARENT is on the root disk (/). Mount DISK02 first; refusing to create the data root on /." >&2; exit 4
fi
mkdir -p "$ROOT"/{scenes,runs/logs,runs/pytest_tmp,caches/xdg_cache,caches/xdg_data,caches/warp,caches/nvidia/GLCache,caches/nvidia/ComputeCache}
echo "data root ready: $ROOT (mount point $MOUNT_POINT, $(df -h --output=avail "$ROOT" | tail -1 | tr -d ' ') free)"
if [ "$SEED_CACHES" = 1 ]; then
  [ -d "$HOME/.cache/ov" ] && cp -a "$HOME/.cache/ov" "$ROOT/caches/xdg_cache/" && echo "seeded ~/.cache/ov"
  [ -d "$HOME/.local/share/ov" ] && cp -a "$HOME/.local/share/ov" "$ROOT/caches/xdg_data/" && echo "seeded ~/.local/share/ov"
fi
