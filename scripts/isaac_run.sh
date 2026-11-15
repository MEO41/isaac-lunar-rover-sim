#!/usr/bin/env bash
# Launch one Isaac Sim script of this project with guards and a start-up watchdog.
#   scripts/isaac_run.sh [--interactive] [--timeout SECONDS] [--data-root PATH] <script.py> [script arguments]
# 1 data-root check (same Python module as every entry point)  2 one Isaac process of this project at a time (flock)
# 3 free video memory / root disk floors  4 cache redirects to the data root  5 stale /dev/shm cleanup
# 6 launch with stdin closed, log under <data root>/runs/logs  7 relaunch if the log stays empty for 75 s (3 attempts)
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
[ -f "$HERE/.env.local" ] && set -a && . "$HERE/.env.local" && set +a
: "${ISAAC_ENV:?set ISAAC_ENV in .env.local}"
INTERACTIVE=0; TIMEOUT_S=900; DATA_ROOT_ARGUMENT=()
while [ $# -gt 0 ]; do
  case "$1" in
    --interactive) INTERACTIVE=1; shift ;;
    --timeout) TIMEOUT_S="$2"; shift 2 ;;
    --data-root) DATA_ROOT_ARGUMENT=(--data-root "$2"); export LUNAR_ROVER_SIM_DATA_ROOT="$2"; shift 2 ;;
    *) break ;;
  esac
done
[ $# -ge 1 ] || { echo "usage: isaac_run.sh [--interactive] [--timeout S] [--data-root P] <script.py> [args]" >&2; exit 2; }
SCRIPT="$1"; shift
[ -f "$SCRIPT" ] || SCRIPT="$HERE/$SCRIPT"
[ -f "$SCRIPT" ] || { echo "no such script: $SCRIPT" >&2; exit 2; }
export PYTHONPATH="$HERE/src${PYTHONPATH:+:$PYTHONPATH}"

ROOT="$("$ISAAC_ENV/bin/python" -m lunar_rover_sim.core.storage.data_root --require "${DATA_ROOT_ARGUMENT[@]}")" || exit 4
export LUNAR_ROVER_SIM_DATA_ROOT="$ROOT"

exec 9>"$ROOT/runs/isaac.lock"
if ! flock -n 9; then echo "ERROR: another Isaac process of this project holds $ROOT/runs/isaac.lock" >&2; exit 5; fi

FREE_VRAM_MIB="$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1 | tr -d ' ')"
if [ "${FREE_VRAM_MIB:-0}" -lt "${MINIMUM_FREE_VIDEO_MEMORY_MIB:-2500}" ]; then
  echo "ERROR: only ${FREE_VRAM_MIB} MiB video memory free (< ${MINIMUM_FREE_VIDEO_MEMORY_MIB:-2500}). Close other GPU programs." >&2; exit 6
fi
FREE_ROOT_GIB="$(df --output=avail -BG / | tail -1 | tr -dc '0-9')"
if [ "${FREE_ROOT_GIB:-0}" -lt "${MINIMUM_FREE_ROOT_DISK_GIB:-3}" ]; then
  echo "ERROR: only ${FREE_ROOT_GIB} GiB free on / (< ${MINIMUM_FREE_ROOT_DISK_GIB:-3}); Isaac logs still write there." >&2; exit 6
fi

export XDG_CACHE_HOME="$ROOT/caches/xdg_cache"
export WARP_CACHE_PATH="$ROOT/caches/warp"
export __GL_SHADER_DISK_CACHE_PATH="$ROOT/caches/nvidia/GLCache"
export CUDA_CACHE_PATH="$ROOT/caches/nvidia/ComputeCache"
mkdir -p "$XDG_CACHE_HOME" "$WARP_CACHE_PATH" "$__GL_SHADER_DISK_CACHE_PATH" "$CUDA_CACHE_PATH"

for f in /dev/shm/carb-RStringInternals-*; do [ -e "$f" ] && ! kill -0 "${f##*-}" 2>/dev/null && rm -f "$f"; done

STAMP="$(date +%Y%m%d_%H%M%S)"
LOG="$ROOT/runs/logs/${STAMP}_$(basename "$SCRIPT" .py).log"
echo "log: $LOG"
if [ "$INTERACTIVE" = 1 ]; then
  "$ISAAC_ENV/bin/python" "$SCRIPT" "$@" < /dev/null 2>&1 | tee "$LOG"
  exit "${PIPESTATUS[0]}"
fi
for attempt in 1 2 3; do
  : > "$LOG"
  timeout -k 10 "$TIMEOUT_S" "$ISAAC_ENV/bin/python" "$SCRIPT" "$@" > "$LOG" 2>&1 < /dev/null &
  pid=$!
  for _ in $(seq 1 75); do [ -s "$LOG" ] && break; kill -0 "$pid" 2>/dev/null || break; sleep 1; done
  if [ -s "$LOG" ] || ! kill -0 "$pid" 2>/dev/null; then
    wait "$pid"; code=$?
    tail -n 40 "$LOG"
    echo "exit code $code (log: $LOG)"
    exit "$code"
  fi
  echo "isaac_run: start-up stalled (attempt $attempt/3), restarting" >&2
  kill "$pid" 2>/dev/null; sleep 2; kill -9 "$pid" 2>/dev/null
done
echo "isaac_run: Isaac did not start after 3 attempts" >&2; exit 3
