# CLAUDE.md — isaac-lunar-rover-sim

Lunar rover simulator for Isaac Sim 6.0.1: LROC NAC DEM terrain with collision, procedural sub-DEM craters and
fractal relief, rocks, regolith and wheel tracks. The design is `docs/DESIGN.md`; deviations made while building are
recorded in `docs/decision_log.md`.

## Scope (user instruction)

Build and prove the **simulation environment** first. The learning-facing layer (gymnasium wrapper, episode recorder,
tasks, language instructions, Isaac Lab adapter) is deferred until the user confirms the environment works. Keep the
cheap seams: named seeds, semantic labels on prims, crater/rock registry and scene manifest, track ground truth.

## Architecture rule

- `lunar_rover_sim.core` is Isaac-free (stdlib, numpy, scipy, pydantic, pyyaml, Pillow). It does all generation maths.
  `tests/unit/test_import_rules.py` enforces it.
- `lunar_rover_sim.runtime` is a thin Isaac layer: arrays -> USD/PhysX, stepping, reading sensors.
- Baked scenes are `.npy/.npz/.png/.json` only (the Isaac env has no rasterio).

## Storage (user instruction)

Code lives here. Every generated file goes under the data root, set per machine in `.env.local`
(`LUNAR_ROVER_SIM_DATA_ROOT`, or `--data-root`). Entry points call `require_data_root()`, which refuses the root disk
unless the machine opts in with `LUNAR_ROVER_SIM_ALLOW_ROOT_DISK=1`. Machines (2026-11-16):
- **minsky** (RTX 5090, current workstation): one 1.9 TB disk, opt-in set; data root `~/Desktop/MEO/isaac-lunar-rover-sim-data`
  next to the repos.
- old machine (RTX 5060): data root `/media/noron/DISK02/isaac-lunar-rover-sim-data` (DISK02 is auto-mounted at login
  and not in fstab; the root disk is ~97 % full, so no opt-in there).
DEM files are referenced read-only from `lunar_regolith/data/nac_dem`, never copied.

## lunar_regolith appearance (user instruction)

- `external/lunar_regolith` is a pinned git submodule. Never edit files inside it; changes go upstream in its own
  repository/session. Bump the pin deliberately when newer features are needed.
- Large data is not in git: `.env.local` sets REGOLITH_REPOSITORY, REGOLITH_PYTHON, REGOLITH_RUN (working copy).
- Build appearance with `scripts/regolith_run.sh scripts/build_regolith_appearance.py` (REGOLITH_PYTHON, submodule code
  first on PYTHONPATH), never with Isaac's Python. Record the lunar_regolith commit in every result (provenance.json).
- Pass this simulator's surface as `SceneSpec.surface` (the workspace collision lattice); render meshes stay
  render-only; colliders, rocks and the rover stay ours. Upstream render-only rocks/micro-craters stay off.
- Ruts for lunar_regolith scenes are `SurfaceImprint` plates from a contact log (static per build).
- One heavy scene per Isaac process; never PathTracing with totalSpp 0.
- Appearance is SYNTHETIC soil through an ASSUMED mapping; label it so.

## Commands

```bash
scripts/setup_data_root.sh                     # once: create the data root (path from .env.local)
scripts/run_unit_tests.sh                      # Isaac-free unit tests (uses $ISAAC_ENV python)
$ISAAC_ENV/bin/python scripts/bake_scene.py --preset smoke      # bake a scene (no Isaac)
scripts/isaac_run.sh tests/isaac/smoke_terrain_drive.py --preset smoke   # headless Isaac smoke test
scripts/isaac_run.sh --interactive scripts/view_scene.py --preset smoke  # drive with the keyboard
```

## Conventions

1. Fully descriptive names, no cryptic abbreviations (`configuration`, not `cfg`; `random_generator`, not `rng`).
   Allowed: `id`, `usd`, `dem`, `rgb`, `png`, `json`, `xyzw`, unit suffixes `_m`, `_s`, `_deg`, `_rad`, `_n`, `_pa`, `_kg`.
2. A docstring on every function, method and class (purpose, units, return shape).
3. Honest labelling: synthetic or uncalibrated data is marked SYNTHETIC / ASSUMPTION in code, metadata and docs.
4. Type hints on public signatures; pydantic models use `extra="forbid"`.
5. Every print in an Isaac script uses `flush=True`.
6. No silent clamping: functions that clamp to a grid raise or return a validity mask.
7. Report only tests that were actually run, with the numbers.
8. Git: `origin` = https://github.com/MEO41/isaac-lunar-rover-sim (branch `main`). Commit or push only when the
   user asks. Never commit `.env.local`, DEM data or anything from the data root.
9. Always launch Isaac through `scripts/isaac_run.sh`. One Isaac process at a time by default; on minsky (RTX 5090)
   up to 3 in parallel with `ISAAC_PARALLEL_SLOTS=3` (user decision 2026-11-17; each process needs ~5 GB of video
   memory and ~11 CPU cores, so 3 processes give ~1.4x the throughput of one). The old RTX 5060 stays at 1.
