# Isaac Lunar Rover Simulator — Design (version 1)

Status: design only. No project code exists yet and Isaac Sim was not launched while writing this document.
Target: Isaac Sim 6.0.1 (pip) in `ISAAC_ENV=/media/noron/DISK02/isaac-lab-native/env_isaaclab`, built and verified on an RTX 5060 (8 GB), scaled by preset to an RTX 5090 (32 GB).

Labels used throughout:

- **VERIFIED**: measured on this machine in the research probes (section 15 says where the probes live).
- **UNVERIFIED**: must be settled by a runtime probe in work package WP-09 before anything depends on it. Each has a named fallback.
- **ASSUMPTION**: a modelling value without a firm source. Each is a named configuration field and is listed in `docs/assumptions.md`.
- **ESTIMATE**: arithmetic from verified parts, to be replaced by a measurement.
- **SYNTHETIC**: generated data that does not come from a measurement of the Moon (all soil properties, all sub-DEM detail, all rocks).

---

## 1. Scope

### 1.1 What version 1 is

Version 1 is the **simulation environment**, built and proven correct on its own. Work on vision-language-action models comes later and only once the environment is trusted. Version 1 therefore contains exactly:

1. Terrain from the LROC NAC DEM with collision that the rover's wheels follow.
2. Procedural sub-DEM detail: craters and band-limited fractal relief.
3. Rocks, from pebbles to boulders, with natural shapes and placement.
4. Regolith: look (material, albedo, micro-relief textures) and SYNTHETIC soil property maps.
5. Wheel tracks that are visible in the rendered image.
6. The rover: articulation, drive controller, cameras, basic proprioception (pose, twist, wheel speeds, per-wheel contact, slip and sinkage).
7. Lighting and rendering.
8. A minimal way to drive: a scripted twist or waypoint drive script (headless) and a keyboard teleoperation launcher with a window.
9. Unit tests and headless smoke tests that prove items 1 to 8 with numbers.

Version 1 also keeps five structural things that are cheap now and hard to retrofit: reproducible named random streams, semantic class labels on prims, the crater and rock registry with a scene manifest, the ground-truth track mask and rover centreline, and render annotators as far as the smoke tests need them (`rgb`, depth, semantic segmentation).

### 1.2 What is deferred

The whole learning-facing layer is deferred to section 14: the `gymnasium.Env` wrapper, the episode recorder (parquet, MP4, LeRobot mapping), task specifications and success predicates, language instruction generation, path planning and a scripted expert beyond a trivial waypoint follower, randomisation curricula, and any Isaac Lab adapter. No version 1 work package builds them, no version 1 file imports them, and version 1 is not incomplete for lacking them. Section 14 names the seams where each one plugs in.

Also not in version 1 (section 14): force-based terramechanics, collider update under ruts, compliant-contact sinkage, dynamic rocks, streaming terrain, ejecta block fields, soil maps exported from the sibling project, random site offsets, props, polar sites, ROS.

### 1.3 Requirement trace

| User requirement | Delivered in version 1 by | Proven by |
|---|---|---|
| 1. A lunar simulator on which vision-language-action research is easy **later** | The seams only: named seeds (section 4.3), class labels on prims (7.4), crater and rock tables plus manifest (7.2, 7.3), track ground truth (9.5), cameras with `rgb`, depth and semantic segmentation (10.4), a non-gym drive session that a later environment wraps (8.10, 14) | Unit tests of seeding and manifest; smoke tests S4 and S5 |
| 2. NAC DEM terrain with correct collision; wheels follow it | DEM window in a true-metre local frame, four nested tiers, exact triangle-mesh colliders, rendered surface checked against the collider | Smoke tests S1 and S3 |
| 3. Procedural sub-DEM topography | Equilibrium craters with diffusion degradation, band-limited fractal residual calibrated on the full surface, DEM band preserved by construction | Unit tests (11.1), calibration numbers in the manifest |
| 4. Rock formations and regolith modules | Rock shape generator, size-frequency placement with clusters, fillets; soil property maps, albedo and detail textures, lunar material | Unit tests; smoke tests S2 (material) and S5 (rocks) |
| 5. Wheels leave visible marks | `TrackField` carve with grouser imprint and berms, visual tile updates, mask texture | Smoke test S4 |

### 1.4 Basis of the design

Three design studies were compared (highest fidelity, best data interface, lowest risk). The result is the low-risk skeleton: every primary mechanism was measured on this machine, and every unmeasured one has a probe and a fallback. From the fidelity study it takes absolute spectral normalisation, measured DEM statistics and the grouser imprint keyed on the true wheel angle. From the data-interface study it takes the rule that all generation mathematics is Isaac-free and the scene manifest. Force-based terramechanics and the learning-facing layer are the two large items left for later.

---

## 2. Architecture

### 2.1 The design rule

1. `lunar_rover_sim.core` is **Isaac-free**. It may import only the standard library, numpy, scipy, pydantic, pyyaml, Pillow, and (through one adapter file) numpy-only modules of `lunar_regolith`. It does **all** generation mathematics: DEM window, craters, fractal detail, rocks, soil maps, track deformation kernel, manifest, spawn pose.
2. `lunar_rover_sim.runtime` is a **thin Isaac layer**. It turns arrays into USD prims and PhysX objects, steps the simulation, reads tensors and annotators. It contains no generation mathematics.
3. A unit test (`tests/unit/test_import_rules.py`) parses every file under `core/` and fails if it imports `pxr`, `omni`, `isaacsim`, `carb`, `usdrt`, `warp`, `torch`, `rasterio`, `pyproj`, `gymnasium`, `pyarrow`, `cv2` or `numba` (numba is broken in the Isaac environment).
4. Baked scenes are stored as `.npy`, `.npz`, `.png`, `.json` and `.yaml` only. No GeoTIFF and no USD is written by the bake, so the runtime never needs rasterio.
5. Version 1 dependencies: numpy, scipy, pydantic, pyyaml, Pillow, pytest. All are already in `ISAAC_ENV`. Nothing is installed into it.

### 2.2 Stages and interpreters

```
NAC DTM .TIF (referenced, read-only memmap)        lunar_regolith/src at commit aae047b (numpy-only modules, one adapter)
        │                                                   │
        ▼                                                   ▼
BAKE   scripts/bake_scene.py      $ISAAC_ENV/bin/python, NO SimulationApp, CPU only, deterministic
        └─► <data root>/scenes/<scene_id>/                  (arrays + manifest.json, section 7.1)
        │
        ▼
RUNTIME scripts/isaac_run.sh <script>   $ISAAC_ENV/bin/python + SimulationApp, ONE Isaac process at a time
        load_scene() → RoverDriveSession (build_scene_stage → rover → tracks → cameras)
        └─► <data root>/runs/<run_name>/<timestamp>/        (result.json, PNG, run log, section 7.5)

LATER  (section 14)  gymnasium.Env wraps RoverDriveSession; recorder subscribes to its per-step output;
                     tasks, instructions and planner read manifest + rasters through an Isaac-free SceneQuery
```

| Stage | Interpreter | Output |
|---|---|---|
| Unit tests | `scripts/run_unit_tests.sh` (`$ISAAC_ENV/bin/python -m pytest tests/unit`, no `SimulationApp`) | pass or fail |
| Bake | `$ISAAC_ENV/bin/python scripts/bake_scene.py` (no `SimulationApp`) | scene directory |
| Runtime, smoke tests, driving | `scripts/isaac_run.sh` | PNG, JSON, run logs |

Why the runtime authors USD instead of the bake: mesh authoring from numpy plus cooking is VERIFIED (0.85 s for 0.52 M triangles, 3.1 s for 2.1 M, 11.7 s cold and 1.15 s warm for 8.39 M), the runtime needs the height arrays in memory for the tracks anyway, and it avoids mixing `usd-core` 26.11, the pip `pxr` and Kit's embedded USD.

One baked scene per Isaac process. A new scene means a new process, which also gives crash isolation and reuses the launch watchdog.

---

## 3. Storage, caches and launch

### 3.1 Two locations

| Location | Holds | Disk |
|---|---|---|
| `/home/noron/Desktop/MEO/isaac-lunar-rover-sim` | code, small configuration files, documents | root disk, 14 GB free (97 % full) |
| the **data root**, default `/media/noron/DISK02/isaac-lunar-rover-sim-data` | every generated file: baked scenes, textures, run outputs, logs, pytest temporary files, redirected caches | ext4 on `/dev/nvme1n1`, about 400 GB free |

Nothing generated is ever written into the project directory, into `/tmp`, or into the two sibling projects. Because the data disk has hundreds of GB free, bake resolution and extent are limited by video memory and time, not by disk.

### 3.2 Data root resolution and the startup check

The data root is one configurable path. Precedence, highest first:

1. command-line option `--data-root <path>` (every script accepts it);
2. environment variable `LUNAR_ROVER_SIM_DATA_ROOT` (also read from `.env.local`);
3. configuration key `storage.data_root` (default `/media/noron/DISK02/isaac-lunar-rover-sim-data`).

`/media/noron/DISK02` is auto-mounted at login and is **not** in `/etc/fstab`, so it can be missing. `core/storage/data_root.py` provides `require_data_root()`, and **every entry point calls it first**: `bake_scene.py`, `preview_scene.py`, `measure_dem_cutoff.py`, `run_unit_tests.sh`, `isaac_run.sh` (through the same Python module, before it takes its lock) and `start_simulation_application()` (for direct launches). The check stops the program with a clear message unless all of these hold:

- the path exists and is a directory;
- a probe file can be created and deleted in it;
- `os.stat(root).st_dev != os.stat("/").st_dev`, i.e. it is not on the root disk (opt-out for other machines: `storage.allow_root_disk: true`);
- free space is at least `storage.minimum_free_gib`.

The check never creates the root itself; it creates only subdirectories under an existing root. There is no fallback path. The message names the path and says: "Is DISK02 mounted? It is auto-mounted at login and not in /etc/fstab. Create the data root once with scripts/setup_data_root.sh. Refusing to fall back to the root disk."

`scripts/setup_data_root.sh` is the only thing that creates the root. It resolves the path the same way, requires `findmnt -T <parent>` to report a mount point other than `/`, then creates the root and its subdirectories. With `--seed-caches` it copies `~/.cache/ov` (1.9 GB) and `~/.local/share/ov` (284 MB) into the redirected cache locations once, so the first launch does not recompile every shader.

### 3.3 Layout of the data root

```
<data root>/
  scenes/<scene_id>/                 baked scenes (section 7.1); rock prototype libraries live inside each scene
  runs/<run_name>/<timestamp>/       outputs of smoke tests, probes and drive scripts (section 7.5)
  runs/logs/<timestamp>_<script>.log Isaac process logs written by isaac_run.sh
  runs/pytest_tmp/                   pytest --basetemp
  runs/isaac.lock                    flock file: one Isaac process of this project at a time
  caches/                            redirected caches (section 3.4)
```

### 3.4 Caches

Sizes were measured on this machine. Which launcher writes which directory was not established, so probe RP11 (section 12.2) maps them: it records the size of every directory below before and after one cold terrain cook, without and with the redirects, and resolves the Kit tokens `${omni_cache}`, `${omni_data}` and `${omni_logs}`. The probe fails if any token still resolves under `/home` after the redirect.

| Cache | Default location | Size now | Location in version 1 | Redirect | Status |
|---|---|---|---|---|---|
| Kit derived-data cache of the pip install (PhysX cooking) | `$ISAAC_ENV/lib/python3.12/site-packages/isaacsim/kit/cache` | 38 GB | unchanged, already on DISK02 | none needed | measured |
| Kit shader cache and logs of the pip install | `$ISAAC_ENV/.../isaacsim/kit/{cache/nv_shadercache, logs}` | 1.4 GB + 80 MB | unchanged, already on DISK02 | none needed | measured |
| Omniverse user cache (derived data, shaders, texture cache, Kit, generated nodes) | `~/.cache/ov` on `/` | 1.9 GB | `<data root>/caches/xdg_cache/ov` | `XDG_CACHE_HOME=<data root>/caches/xdg_cache`, plus the setting `--/UJITSO/datastore/localCachePath=<data root>/caches/xdg_cache/ov/DerivedDataCache` | UNVERIFIED (strings found in `libomni.kit.app.plugin.so` and the ujitso plugin) |
| Omniverse user data | `~/.local/share/ov` on `/` | 284 MB | `<data root>/caches/xdg_data/ov` | `XDG_DATA_HOME=<data root>/caches/xdg_data` | UNVERIFIED |
| Omniverse logs | `~/.nvidia-omniverse/logs` on `/` | 645 MB | follows `${omni_logs}` | none known; RP11 tests whether the XDG variables move it | UNVERIFIED |
| NVIDIA driver caches (GL shader cache, CUDA compute cache) | `~/.nv` on `/` | 212 MB | `<data root>/caches/nvidia/{GLCache, ComputeCache}` | `__GL_SHADER_DISK_CACHE_PATH`, `CUDA_CACHE_PATH` | UNVERIFIED (standard NVIDIA variables) |
| Warp kernel cache | `~/.cache/warp` on `/` | 73 MB | `<data root>/caches/warp` | `WARP_CACHE_PATH` (read in `warp/_src/build.py`) | source read, not run |
| Other XDG users inside the Isaac process (pip, fonts, torch) | `~/.cache` on `/` | small | follow `XDG_CACHE_HOME` | same variable | — |
| pytest temporary files | `/tmp` on `/` | — | `<data root>/runs/pytest_tmp` | `--basetemp` in `run_unit_tests.sh` | by construction |
| Isaac launch log | `/tmp` in the sibling script | — | `<data root>/runs/logs` | `isaac_run.sh` | by construction |

Rules that follow from the table:

- `isaac_run.sh` exports the five variables before Python starts. `start_simulation_application()` sets the same variables (when unset) and passes the UJITSO setting through `SimulationApp`'s `extra_args`, so a direct launch behaves the same.
- Fallback if the XDG variables are ignored: launch with `--portable-root <data root>/caches/kit_portable` (Kit's portable mode places data, cache and logs there).
- Any directory that RP11 shows cannot be moved is listed in `docs/runtime_probe_report.md`. For those, `isaac_run.sh` keeps a free-space guard on `/` (`MINIMUM_FREE_ROOT_DISK_GIB`).
- Cost of a new scene in the cooking cache: about 0.26 GB for the `gpu_8gb` preset (ESTIMATE: 3.27 M collision triangles at roughly 80 bytes each). On the root disk a handful of scenes would have used the remaining space; on the data root it is irrelevant.
- One-off cost of the redirect: the first launch recompiles shaders and re-cooks. Seed the caches with `setup_data_root.sh --seed-caches` and give the first run `--timeout 1800`.

### 3.5 Machine file `.env.local`

Git-ignored; `.env.example` documents it.

```
ISAAC_ENV=/media/noron/DISK02/isaac-lab-native/env_isaaclab
LUNAR_ROVER_SIM_DATA_ROOT=/media/noron/DISK02/isaac-lunar-rover-sim-data
NAC_DEM_DIRECTORY=/home/noron/Desktop/MEO/lunar_regolith/data/nac_dem
LUNAR_REGOLITH_SOURCE=/home/noron/Desktop/MEO/lunar_regolith/src
LUNAR_REGOLITH_EXPECTED_COMMIT=aae047b
MINIMUM_FREE_VIDEO_MEMORY_MIB=2500
MINIMUM_FREE_ROOT_DISK_GIB=3
```

Other path rules:

- DEM files are referenced by absolute path, opened with `np.memmap(mode="r")`, never copied and never opened for writing (`NAC_DTM_APOLLO17_2M.TIF` is hard-linked into three projects).
- The project is not pip-installed into the Isaac environment. Scripts and `isaac_run.sh` set `PYTHONPATH=<project>/src`.
- Nothing is written into `/home/noron/Desktop/MEO/lunar_regolith` or `/home/noron/Desktop/MEO/apollo_lander_modeling`.

### 3.6 `scripts/isaac_run.sh`

Adapted from `/home/noron/Desktop/MEO/apollo_lander_modeling/isaac_run.sh`. Usage: `isaac_run.sh [--interactive] [--timeout <seconds>] [--data-root <path>] <script.py> [script arguments]`. In order:

1. Source `.env.local`.
2. Resolve and check the data root by running `"$ISAAC_ENV/bin/python" -m lunar_rover_sim.core.storage.data_root --require` (pure Python, no `SimulationApp`). Stop with its message and exit code 4 on failure. This is the same code every Python entry point uses.
3. Take an exclusive `flock` on `<data root>/runs/isaac.lock` so only one Isaac process of this project runs.
4. Refuse to start if free video memory (from `nvidia-smi`) is below `MINIMUM_FREE_VIDEO_MEMORY_MIB`, or free space on `/` is below `MINIMUM_FREE_ROOT_DISK_GIB`. This is a coarse floor; the preset-specific video memory check runs in Python (step 7). Other sessions were seen holding 2.5 to 3.6 GB of this GPU.
5. Export the cache variables of section 3.4 and `PYTHONPATH`.
6. Remove stale `/dev/shm/carb-RStringInternals-<pid>` files whose process is dead.
7. Launch `"$ISAAC_ENV/bin/python" <script> ... > <data root>/runs/logs/<timestamp>_<script>.log 2>&1 < /dev/null`. Inside, `start_simulation_application()` compares free video memory with `launch.required_free_video_memory_mib` of the resolved preset and stops before creating `SimulationApp` if it is too low.
8. Watchdog: if the log is still empty after 75 s, kill and relaunch, up to 3 attempts; exit code 3 on failure. A hard `timeout` (default 900 s) bounds every run.

`--interactive` is for the window launcher: no hard timeout, the log is also copied to the terminal, the start-up watchdog and the lock stay active, and the script ends when the window is closed.

---

## 4. Conventions

### 4.1 Coordinate frames and units

| Item | Convention |
|---|---|
| Units | SI everywhere: metres, seconds, kilograms, newtons, pascals, radians in code. Degrees only in configuration files and in names ending `_deg`. Every variable holding a physical quantity carries its unit as a suffix (`_m`, `_m_s`, `_rad`, `_deg`, `_n`, `_pa`, `_kg`, `_s`). |
| World frame | Local east-north-up: x = east, y = north, z = up, metres. Origin = site longitude and latitude plus the configured offset. z = 0 is the bicubic DEM height at the origin (the site datum). USD stage: Z up, `metersPerUnit = 1.0`. |
| Map frame | Equirectangular metres of the DTM: `map_x = R·radians(longitude − 180°)·cos(20°)`, `map_y = R·radians(latitude)`, `R = 1 737 400 m`. |
| Map to local | `east = (map_x − map_x_origin)·cos(latitude_origin)/cos(20°)`, `north = map_y − map_y_origin`. VERIFIED against pyproj to 1e-6 m. A 2 m DTM pixel is 1.9976 true metres east-west at the Apollo 17 site, so the DEM is always resampled onto a true-metre lattice. |
| Datum | Subtracted in float64 before any cast to float32. The float32 unit in the last place at −2600 m is 0.24 mm, which would destroy millimetre ruts. |
| Planetary curvature | DEM heights are relative to the 1 737.4 km sphere and are laid on a flat plane. The workspace and near ring stay planar (the missing drop is 3 mm at 104 m), so collider and track mathematics are planar. The regional and backdrop tiers have `(east² + north²)/(2·R)` subtracted (0.30 m at 1 km, 106 m at 19.2 km), so the horizon of a camera 0.9 m above the ground (about 1.8 km away) hides what it should. Switch: `terrain.apply_planetary_curvature`. |
| Raster arrays | Indexed `[north_index, east_index]`. Index `[0, 0]` is the **south-west** vertex. PNG files are written with row 0 = north (image convention); the flip is explicit in the writer. |
| Lattices | Vertex lattices: `count_east` by `count_north` vertices, `spacing_m` apart. Vertex `[i, j]` is at `(origin_east_m + j·spacing_m, origin_north_m + i·spacing_m)`. Flat vertex index = `i·count_east + j`. |
| Triangulation | One rule everywhere. For the cell with corners a = `[i, j]`, b = `[i, j+1]`, c = `[i+1, j]`, d = `[i+1, j+1]`: triangles `(a, b, d)` and `(a, d, c)` (diagonal a–d, counter-clockwise seen from above). With `u` = fractional east and `v` = fractional north inside the cell: `z = za + u·(zb − za) + v·(zd − zb)` if `u ≥ v`, else `z = za + u·(zd − zc) + v·(zc − za)`. |
| Quaternions | `xyzw` (scalar last) in every array, file and manifest. Conversion happens only at the Isaac boundary (`omni.physics.tensors` root transforms are already `x, y, z, qx, qy, qz, qw`; `Vt.QuathArray.FromNumpy` also expects imaginary first). |
| Body frame | x forward, y left, z up. Rover pose = pose of the chassis body origin (geometric centre of the wheel contact rectangle, at wheel-axle height). |
| Wheel rotation | `wheel_rotation_angle_rad` increases when the wheel rolls forward. Wheel order everywhere: `front_left, front_right, rear_left, rear_right`. |
| Slip ratio | `(ω·r − v_forward) / max(abs(ω·r), abs(v_forward), 0.02 m/s)`; positive when the wheel spins faster than it travels. |
| Twist command | `[forward_speed_m_s, yaw_rate_rad_s]`, SI, unnormalised; yaw rate positive = counter-clockwise (left turn). |
| Camera frame | USD cameras look along −Z with +Y up. Camera poses and intrinsics handed to Isaac-free code use the ROS optical convention (z forward, x right, y down). `pose_math.py` holds the only conversion. |
| Sun | `azimuth_deg` is the compass bearing **towards** the Sun (0 = north, 90 = east), `elevation_deg` above the horizon. Unit vector towards the Sun: `(sin(az)·cos(el), cos(az)·cos(el), sin(el))`. |
| Time | `time_s` is simulation time since the last reset. Physics rate 120 Hz. Headless control rate 10 Hz (12 physics steps per control step). Window loop: one application update per control step, which is 1/60 s and 2 physics steps. |

### 4.2 Coding conventions of this repository

1. **Fully descriptive names, no cryptic abbreviations.** Variables, functions, classes, files and configuration keys use full words: `wheel_rotation_angle_rad`, not `theta_w`; `ground_truth`, not `gt`; `configuration`, not `cfg`; `random_generator`, not `rng`; `level_of_detail`, not `lod`. Allowed short forms are only universal ones: `id`, `usd`, `dem`, `rgb`, `png`, `json`, `xyzw`, and the unit suffixes `m`, `s`, `deg`, `rad`.
2. **A docstring on every function, method and class**, stating purpose, argument units and return shape.
3. **Honest labelling.** Everything synthetic or uncalibrated is labelled SYNTHETIC or ASSUMPTION in code comments, file metadata, the manifest and the documents. Every unsourced modelling constant is a named configuration field with a row in `docs/assumptions.md`.
4. Type hints on every public signature. Pydantic models use `extra="forbid"`.
5. Every print in an Isaac script uses `flush=True`.
6. No silent clamping: functions that clamp to a grid raise or return an explicit validity mask.
7. Report only tests that were actually run, with the numbers.
8. Git: the project is not a repository yet. Do not create commits or push unless the user asks.

`CLAUDE.md` in the project root repeats these eight rules and points at this document. `docs/decision_log.md` holds one row per design decision (decision, why, rejected alternatives).

### 4.3 Named random streams

`named_generator(seed, stream_name)` returns `np.random.default_rng(np.random.SeedSequence([seed, zlib.crc32(stream_name.encode("utf-8"))]))`. Adding a new stream never shifts existing draws. No module may create a generator any other way, and no generator is shared between modules.

Version 1 has one seed, `scene_seed`, consumed only by the bake. The runtime draws nothing random except test sample points.

| Seed | Stream names (fixed; add new ones, never rename) |
|---|---|
| `scene_seed` | `terrain.parameters`, `terrain.crater_population`, `terrain.crater_shape`, `terrain.fractal.macro`, `terrain.fractal.meso`, `rocks.prototypes`, `rocks.parameters`, `rocks.sizes`, `rocks.positions`, `rocks.clusters`, `rocks.poses`, `rocks.albedo`, `regolith.parameters`, `regolith.latent_fields`, `regolith.albedo`, `regolith.detail_texture.coarse`, `regolith.detail_texture.fine`, `diagnostics.sample_points` |
| reserved for later | every name starting with `episode.` (start pose, task, Sun, exposure) and `sensor.` (noise). No version 1 code uses them. |

---

## 5. Repository layout

Every file is listed with its responsibility and the work package that owns it (section 12). No file has two owners. WP-01 creates every `__init__.py` of the package tree (all empty).

```
isaac-lunar-rover-sim/
  pyproject.toml                         package "lunar_rover_sim" (src layout); pytest markers: unit, needs_dem         WP-01
  README.md  CLAUDE.md  .env.example  .gitignore                                                                       WP-01
  configs/
    defaults.yaml                        every configuration field with its default (section 6)                         WP-01
    presets/smoke.yaml  gpu_8gb.yaml  gpu_32gb.yaml                                                                     WP-01
    sites/apollo17_landing_site.yaml     DEM paths, projection constants, origin                                        WP-01
    scenes/apollo17_valley_floor.yaml    scene seed and scene-level switches                                            WP-01
    rovers/skid_steer_four_wheel.yaml    rover specification                                                            WP-01
    runtime_probe_results.yaml           fallback switches chosen by the runtime probes                                 WP-09
    measured_budgets.yaml                measured video memory per preset (sets the launch guard)                       WP-15
  assets/materials/lunar_surface.mdl     exports `lunar_surface` and `lunar_rock`, complete with track-mask inputs      WP-10
  scripts/
    setup_data_root.sh                   creates the data root once, after checking the mount (section 3.2)             WP-01
    isaac_run.sh                         launch guard and watchdog (section 3.6)                                        WP-01
    run_unit_tests.sh                    data-root check, then pytest with --basetemp under the data root               WP-01
    measure_dem_cutoff.py                Isaac-free: radial power spectrum of a DEM window → suggested cutoff           WP-08
    bake_scene.py                        Isaac-free command line: configuration → scene directory                       WP-08
    preview_scene.py                     Isaac-free: hillshade, rock and crater overlay, structure-function plot → PNG  WP-08
    run_isaac_smoke_tests.sh             runs tests/isaac/smoke_*.py serially through isaac_run.sh                      WP-09
    drive_demo.py                        Isaac, headless: scripted twists or waypoints, before/after PNG of the tracks  WP-14
    view_scene.py                        Isaac, window: load a scene, keyboard teleoperation                            WP-14
  src/lunar_rover_sim/
    core/                                                                    ───────── Isaac-free ─────────
      seeding.py                         named_generator, derive_seed                                                   WP-01
      storage/data_root.py               resolve_data_root, require_data_root, DataRoot, command-line entry             WP-01
      configuration/
        configuration_models.py          pydantic models for every section of the configuration                         WP-01
        configuration_loader.py          YAML overlay, dotted overrides, environment-variable expansion, bake hash      WP-01
      data_model/                        cross-module data classes with npz/json round trips (section 8.1)              WP-01
        regular_grid.py                  RegularGrid, TileSlice, triangulation, exact height and normal queries, slope raster
        tier_layout.py                   TierLayout, build_tier_layout with every alignment and band assertion
        terrain_data.py                  TerrainTier, TerrainBundle, HeightField, DemSource protocol
        crater_table.py                  CraterTable
        rock_table.py                    RockTable, RockPrototypeLibrary
        soil_property_maps.py            SoilPropertyMaps, FrictionZoneAssignment
        wheel_data.py                    WheelGeometry, WheelKinematicState, WheelContactSample
        rover_specification.py           RoverSpecification, CameraMount, collider overlap validator
        scene_manifest.py                pydantic schema of manifest.json
      frames/
        site_frame.py                    EquirectangularSiteFrame, planetary curvature drop                             WP-01
        pose_math.py                     quaternion (xyzw) and transform helpers, camera convention and intrinsics      WP-01
      soil_mechanics/bekker_sinkage.py   rigid-wheel sinkage with slip and multipass terms (used by regolith and tracks) WP-01
      external/lunar_regolith_adapter.py the ONLY file that imports lunar_regolith; commit check                        WP-01
      dem/
        geotiff_memmap_reader.py         minimal TIFF tag parser + read-only memmap window read                         WP-02
        dem_window.py                    shared spline coefficients, bicubic sampling onto local lattices, datum, block average, GeoTiffDemSource   WP-02
        hole_filling.py                  fill from the 5 m mosaic, then nearest + feathered Gaussian                    WP-02
      terrain/
        spectral_fields.py               band-limited fractal synthesis with absolute normalisation; Gaussian filters   WP-03
        crater_population.py             size-frequency sampling, degradation state                                     WP-03
        crater_stamping.py               analytic profile, diffusion, scale-aware overprint                             WP-03
        tier_blending.py                 edge weight, blend to the outer triangulated surface, hole cut                 WP-03
        terrain_statistics.py            structure function, radial power spectrum, plane detrend, roughness raster     WP-03
        terrain_builder.py               build_terrain(): the whole terrain recipe                                      WP-03
      rocks/
        rock_shape_generator.py          radial-function rock shapes, tessellation at three levels of detail            WP-04
        rock_size_distribution.py        blended power laws, expected counts, sampling                                  WP-04
        rock_placement.py                intensity field, clusters, overlap rejection, pose, burial                     WP-04
        rock_fillets.py                  Gaussian regolith skirts; footprint rasterisation                              WP-04
      regolith/
        soil_property_synthesis.py       in-process synthetic soil maps                                                 WP-05
        friction_zoning.py               friction zones per collision tile                                              WP-05
        albedo_maps.py                   macro albedo map (sRGB PNG)                                                    WP-05
        detail_textures.py               tileable micro-relief textures                                                 WP-05
      tracks/
        wheel_stamp_kernel.py            carve + grouser imprint + flank + berm deposit for one sample                  WP-06
        track_field.py                   world-anchored track state, dirty tiles, mask image, pressed pebbles, clear/save/load   WP-06
        wheel_contact_sampling.py        WheelKinematicState → gated WheelContactSample (vectorised numpy)              WP-06
        track_replay.py                  rebuild a TrackField from a contact log; contact log npz read and write        WP-06
        track_polyline.py                rover centreline, cross_track_error                                            WP-06
      rover/
        drive_models.py                  DriveModel protocol, SkidSteerDriveModel, TwistLimiter                         WP-07
        twist_sources.py                 TwistSource protocol, ScriptedTwistSequence, WaypointFollower                  WP-07
      perception/
        semantic_classes.py              the fixed class table                                                          WP-07
        camera_projection.py             depth unprojection, world-to-pixel projection, raster lookup                   WP-07
        image_metrics.py                 peak signal-to-noise ratio, intersection over union, contrast-to-noise ratio   WP-07
      scene/
        scene_identifier.py              scene_id from the resolved configuration                                       WP-08
        scene_bake_pipeline.py           bake_scene(): orchestrates terrain, rocks, regolith, rasters, manifest         WP-08
        scene_loader.py                  load_scene() → LoadedScene (arrays + manifest)                                 WP-08
        scene_rasters.py                 analysis rasters on the collision lattice                                      WP-08
        spawn_pose.py                    find_spawn_pose(): deterministic safe start pose                               WP-08
        scene_preview.py                 hillshade and overlays (numpy + Pillow)                                        WP-08
    runtime/                                                                 ───────── needs Isaac ─────────
      application.py                     data-root and video-memory checks, cache variables, SimulationApp start, carb settings   WP-10
      stage_assembly/
        physics_scene_builder.py         /World/PhysicsScene, physics materials                                         WP-10
        terrain_stage_builder.py         tier meshes, visual tiles, collision tiles                                     WP-10
        material_builder.py              MDL binding, textures, UsdPreviewSurface fallback                              WP-10
        lighting_builder.py              Sun and optional fill light                                                    WP-10
        scene_stage_builder.py           build_scene_stage(): orchestrates the builders (rocks through a lazy import)   WP-10
        rock_stage_builder.py            prototypes, instanceable references, pebble instancer, labels                  WP-13
      sensors/camera_rig.py              cameras, render products, annotators, class-id remapping, static cameras       WP-10
      diagnostics/
        collision_verification.py        raycasts against HeightFields, collider readiness                              WP-10
        rendered_surface_verification.py rendered depth against the visual lattice and the collider                     WP-10
        performance_probe.py             per-stage timers, resident memory                                              WP-10
        gpu_memory_monitor.py            per-process video memory from nvidia-smi                                       WP-10
      stepping.py                        ExplicitPhysicsStepper, ApplicationUpdateStepper, synchronous render           WP-11
      rover/
        procedural_rover_builder.py      articulation from a RoverSpecification                                         WP-11
        rover_interface.py               tensor views: targets, state, wheel kinematics, teleport                       WP-11
      driving/rover_drive_session.py     RoverDriveSession: the one control loop (tracks through a lazy import)         WP-11
      tracks/
        terrain_mesh_updater.py          UsdAttributeMeshUpdater                                                        WP-12
        track_mask_texture.py            dynamic texture provider for the mask                                          WP-12
        track_runtime.py                 per-step sampling, per-frame stamping and flush, pressed-pebble hiding         WP-12
      teleoperation/
        keyboard_twist_device.py         carb keyboard → twist                                                          WP-14
        drive_loop.py                    run_drive_loop(): the loop of view_scene.py, also run by smoke test S6         WP-14
  tests/
    conftest.py                          refuses to run with a temporary directory on the root disk                     WP-01
    unit/fixtures/                       synthetic_dem.py, synthetic_geotiff.py, synthetic_scene_parts.py               WP-01
    unit/test_*.py                       Isaac-free pytest, one file per module, owned by the module's package (section 12.1)
    isaac/probes/                        runtime probes RP1 to RP12 (file names in section 12.1)                        WP-09
    isaac/smoke_s1_terrain_load.py  smoke_s2_lunar_material.py                                                          WP-10
    isaac/smoke_s3_rover_drive.py                                                                                       WP-11
    isaac/smoke_s4_wheel_tracks.py                                                                                      WP-12
    isaac/smoke_s5_rocks_and_labels.py                                                                                  WP-13
    isaac/smoke_s6_window_loop_autodrive.py                                                                             WP-14
    isaac/smoke_s7_performance_ladder.py                                                                                WP-15
  docs/
    DESIGN.md
    decision_log.md  assumptions.md                                                                                     WP-01
    runtime_probe_report.md                                                                                             WP-09
    driving_guide.md                     how to bake, drive headless and drive with the keyboard                        WP-14
    verification_report.md               measured numbers of every smoke test                                           WP-15
```

Two lazy-import seams keep single ownership while the GPU chain is built in order. `build_scene_stage` imports `rock_stage_builder` only when `include_rocks` is true, and `RoverDriveSession` imports `runtime.tracks.track_runtime` only when `tracks.enabled` is true. Both callers are written against the interfaces of section 8.10, so the later packages add files and change none.

---

## 6. Configuration

### 6.1 Resolution order and hashing

`load_configuration(scene_file, preset_name, overrides)` merges, later wins:

`configs/defaults.yaml` ← `configs/runtime_probe_results.yaml` (if present) ← `configs/sites/<site>.yaml` ← `configs/scenes/<scene>.yaml` ← `configs/rovers/<rover>.yaml` ← `configs/presets/<preset>.yaml` ← the entry for this preset in `configs/measured_budgets.yaml` (if present) ← command-line dotted overrides (`terrain.craters.equilibrium_coefficient=0.06`).

Presets come after the rover file because presets choose cameras and resolutions. A unit test per preset asserts that the resolved configuration equals the table of section 6.3.

`${NAME}` in string values is expanded from the process environment and `.env.local`.

The **bake hash** is the first 6 hex digits of the SHA-256 of the canonical JSON of the sections `site`, `terrain`, `rocks`, `regolith`, plus `scene_seed` and `generator_version`, with machine-specific path prefixes replaced by the file name. `scene_id = <site.name>_<preset>_s<scene_seed>_<hash6>`. The bake reads no other section: nothing in a baked scene depends on the rover, the Sun, the renderer, the track settings or the sinkage law. A unit test passes a recording proxy of the configuration through `bake_scene` and fails if any field outside the hashed sections (or `storage`) is read.

### 6.2 Schema with defaults (`configs/defaults.yaml`)

The pydantic model tree is `SimulatorConfiguration` with one sub-model per top-level key, named `<Key>Configuration`.

```yaml
generator_version: "0.1.0"
scene_seed: 7

storage:
  data_root: /media/noron/DISK02/isaac-lunar-rover-sim-data   # overridden by LUNAR_ROVER_SIM_DATA_ROOT and by --data-root
  allow_root_disk: false                  # true only on machines where the data root is meant to be on the root disk
  minimum_free_gib: 20.0

launch:
  required_free_video_memory_mib: 6000    # per preset; replaced by measured peak + 500 MiB (configs/measured_budgets.yaml)

site:
  name: apollo17_landing_site
  dem_fine_path: ${NAC_DEM_DIRECTORY}/NAC_DTM_APOLLO17_2M.TIF     # 2 m product: workspace, near ring, regional
  dem_coarse_path: ${NAC_DEM_DIRECTORY}/NAC_DTM_APOLLO17.TIF      # 5 m mosaic: hole filling and backdrop
  origin_longitude_deg: 30.7717
  origin_latitude_deg: 20.1908
  origin_offset_east_m: 0.0               # fixed in version 1; the near ring must be free of NoData or the bake stops
  origin_offset_north_m: 0.0
  standard_parallel_deg: 20.0
  central_longitude_deg: 180.0
  body_radius_m: 1737400.0
  dem_cutoff_wavelength_m: 8.0            # ASSUMPTION until scripts/measure_dem_cutoff.py replaces it

terrain:
  workspace_size_m: 38.4
  visual_spacing_m: 0.025
  collision_decimation: 2                 # collision lattice = visual lattice[::2, ::2]
  visual_tile_cells: 128
  collision_tile_cells: 256               # in collision cells
  workspace_skirt_m: 3.2
  drivable_margin_m: 3.2                  # rover centre must stay this far inside the workspace edge
  apply_planetary_curvature: true         # regional and backdrop tiers only (section 4.1)
  near_ring: {size_m: 208.0, spacing_m: 0.2, skirt_m: 12.8, collision_enabled: true}
  regional:  {enabled: true, size_m: 2048.0, spacing_m: 4.0, skirt_m: 128.0}
  backdrop:  {enabled: true, size_m: 38400.0, spacing_m: 64.0}
  craters:
    equilibrium_coefficient: 0.05                       # N(>D) = c·D^-2 per m²
    randomize_equilibrium_coefficient: true
    equilibrium_coefficient_range: [0.034, 0.079]       # Minton 2019 to Trask 1966
    minimum_diameter_m: 0.4                             # raised automatically to 8 collision posts
    near_ring_minimum_diameter_m: 1.0
    taper_start_diameter_m: 4.0                         # acceptance falls linearly to 0 at dem_cutoff_wavelength_m
    fresh_depth_to_diameter_range: [0.12, 0.20]         # ASSUMPTION and a visibility choice: Stopar 2017 gives ≤ 0.12 below 100 m
    rim_height_to_diameter: 0.036
    apron_extent_radii: 3.0
    degradation_parameter_maximum: 0.17                 # u = K/r², uniform in [0, max]
    fresh_degradation_threshold: 0.03
    moderate_degradation_threshold: 0.10
    ellipticity_minimum: 0.85
    rim_harmonic_amplitude_maximum: 0.06
    feature_crater_count: 1                             # guaranteed clearly visible crater(s) in the drivable zone
    feature_crater_diameter_range_m: [4.0, 7.0]
  fractal:
    hurst_exponent: 0.9
    structure_function_anchor_source: measured_from_dem # measured_from_dem | configured
    structure_function_anchor_m: 0.20                   # σ(4 m); used only when the source is `configured`
    structure_function_anchor_lag_m: 4.0
    structure_function_anchor_valid_range_m: [0.10, 0.40]   # a measured anchor outside this range stops the bake
    roughness_multiplier: 1.0                           # full-surface target at 1 m = multiplier·σ_anchor·(1 m / 4 m)^H
    randomize_roughness_multiplier: true
    roughness_multiplier_range: [1.0, 1.4]              # ASSUMPTION: rougher mare patches; values below 1 are infeasible (the DEM alone carries most of the target)
    macro_band_wavelengths_m: [0.8, 8.0]                # upper edge replaced by dem_cutoff_wavelength_m
    meso_band_wavelengths_m: [0.2, 0.8]                 # lower edge raised automatically to 4 collision posts
    band_edge_relative_width: 0.3
    minimum_fractal_gain: 0.3                           # the fractal residual never drops below this share of its absolute amplitude
    maximum_fractal_gain: 1.5

rocks:
  pivot_diameter_m: 0.1
  background_density_at_pivot_per_m2: 0.18              # N(>0.1 m), ASSUMPTION (Surveyor-range fits)
  randomize_background_density: true
  background_density_at_pivot_range_per_m2: [0.15, 0.30]
  background_exponent: 2.3                              # N(>D) = N_pivot·(D/0.1 m)^-γ
  randomize_background_exponent: true
  background_exponent_range: [2.1, 2.6]
  background_fade_diameters_m: [0.5, 1.0]               # background law fades out, boulder tail fades in (linear in differential density)
  pebble_minimum_diameter_m: 0.02
  cobble_minimum_diameter_m: 0.05
  registration_minimum_diameter_m: 0.10
  maximum_diameter_m: 3.0
  boulder_rock_abundance: 0.004                         # Diviner global mean, areal fraction of rocks ≥ 1 m
  boulder_exponent: 3.0
  near_ring_minimum_diameter_m: 0.3
  clustering_field_correlation_length_m: 15.0
  clustering_field_sigma: 0.8
  thomas_cluster_fraction: 0.3                          # share of 5–30 cm rocks drawn from clusters
  thomas_parent_density_per_m2: 0.02
  thomas_child_sigma_m: 1.5
  burial_fraction_range: [0.1, 0.5]
  tilt_sigma_deg: 10.0
  maximum_slope_for_large_rocks_deg: 30.0
  overlap_rejection_factor: 0.4                         # reject if centre distance < factor·(D1 + D2)
  prototypes_per_shape_class: 6                         # 4 shape classes → 24 prototypes; each prototype has one albedo class
  fillet_minimum_diameter_m: 0.2
  fillet_height_fraction_range: [0.05, 0.15]
  albedo_values: {dark: 0.10, medium: 0.15, bright: 0.20}
  keep_out_zones: [{center_east_m: 0.0, center_north_m: 0.0, radius_m: 1.5}]   # the preferred spawn point stays clear

regolith:                                               # everything here is SYNTHETIC
  property_map_spacing_m: 0.5
  correlation_length_east_m: 30.0
  correlation_length_north_m: 15.0
  facies_fractions: {firm: 0.35, loose: 0.45, soft: 0.20}
  friction_angle_range_deg: [30.0, 42.0]
  cohesion_range_pa: [100.0, 1000.0]
  bulk_density_range_kg_m3: [1400.0, 1850.0]
  rock_abundance_firmness_correlation: -0.4
  fresh_crater_compaction_factor: 0.6                   # ASSUMPTION: softer soil inside 1.5 R of fresh craters
  soil_contrast_limit: [0.25, 4.0]                      # effective modulus clamp relative to the reference soil
  interface_friction_ratio: 0.92                        # μ_static = clip(tan(ratio·φ), clip range)
  interface_friction_clip: [0.2, 1.3]
  dynamic_to_static_friction_ratio: 0.875
  friction_zone_count: 8
  base_normal_albedo: 0.11
  randomize_base_normal_albedo: true
  base_normal_albedo_range: [0.07, 0.16]
  albedo_mottling_amplitude: 0.08
  fresh_ejecta_brightening: 0.15
  macro_albedo_size_px: 2048
  detail_texture_coarse: {size_px: 2048, tile_size_m: 4.0}
  detail_texture_fine:   {size_px: 1024, tile_size_m: 0.73}
  detail_hurst_exponent: 0.7
  detail_texture_standard_deviation: 0.18               # ASSUMPTION, uncalibrated (sibling value)

sinkage:                                                # runtime only: tuning rut depth never forces a re-bake
  exponent: 1.0
  slip_gain: 2.0                                        # ASSUMPTION: z = z_static·(1 + slip_gain·|slip|)
  multipass_gains: [1.0, 1.25, 1.35, 1.4]               # ASSUMPTION, indexed by min(pass_count, 3)
  visual_floor_m: 0.004                                 # non-physical: keeps tracks visible on firm soil
  maximum_m: 0.03                                       # the collider wheel does not sink, so deeper ruts would leave the wheel hovering
  force_filter_time_constant_s: 0.1

tracks:
  enabled: true
  visual_mode: geometry_and_mask                        # geometry_and_mask | geometry_only | mask_only
  mesh_update_backend: usd                              # the only implemented value; `fabric` is the named fallback of probe RP2a
  author_normals: true
  sample_spacing_m: 0.0125
  contact_force_threshold_n: 0.5
  contact_gap_threshold_m: 0.015
  flank_repose_angle_deg: 40.0
  berm_volume_fraction: 0.3                             # ASSUMPTION: the rest is compaction
  berm_width_wheel_fraction: 0.5                        # berm width = max(3 posts, fraction·wheel width)
  pass_separation_time_s: 1.0
  mask_decimation: 1                                    # one mask texel per mask_decimation visual vertices
  mask_upload_minimum_interval_s: 0.1                   # wall time; the upload is the most expensive track step
  track_albedo_factor: 0.90                             # ASSUMPTION
  tread_bump_reduction: 0.7                             # ASSUMPTION: compacted tread is smoother
  hide_pressed_pebbles: true                            # pebbles under a tread are pressed into the soil (hidden)
  collider_update: none                                 # the only value in version 1

rover:                                                  # configs/rovers/skid_steer_four_wheel.yaml
  name: skid_steer_four_wheel
  chassis_size_m: [0.80, 0.34, 0.20]
  chassis_mass_kg: 42.0
  front_axle_mass_kg: 3.0
  wheel_mass_kg: 2.5
  wheel_radius_m: 0.15                                  # collider radius
  wheel_width_m: 0.12                                   # visible wheel and tread width
  wheelbase_m: 0.70
  track_width_m: 0.72                                   # distance between left and right wheel centre planes
  ground_clearance_m: 0.22                              # chassis underside; the lowest rover collider apart from the wheels
  wheel_collider_capsule_height_m: 0.04                 # cylindrical part; total collider width = height + 2·radius = 0.34 m
  grouser_count: 6                                      # 15.7 cm pitch = 6.3 visual posts at 2.5 cm
  grouser_height_m: 0.008
  nominal_sinkage_m: 0.008                              # visual wheel radius = wheel_radius_m + nominal_sinkage_m
  front_axle_pivot_enabled: true
  front_axle_pivot_limit_deg: 15.0
  front_axle_pivot_damping_n_m_s_per_rad: 2.0           # converted to USD per-degree units when authored
  wheel_drive_damping_n_m_s_per_deg: 1000.0             # USD angular drive damping (VERIFIED value, USD per-degree units)
  wheel_drive_maximum_torque_n_m: 20.0
  maximum_forward_speed_m_s: 0.6
  maximum_yaw_rate_rad_s: 0.8
  maximum_forward_acceleration_m_s2: 1.0
  maximum_yaw_acceleration_rad_s2: 2.0
  skid_steer_track_width_factor: 1.0                    # calibrated by the spin test of smoke test S3
  cameras:
    front:  {enabled: true,  position_body_m: [0.30, 0.0, 0.75],  pitch_down_deg: 15.0, yaw_deg: 0.0,   horizontal_field_of_view_deg: 90.0}
    rear:   {enabled: true,  position_body_m: [-0.30, 0.0, 0.75], pitch_down_deg: 15.0, yaw_deg: 180.0, horizontal_field_of_view_deg: 90.0}
    follow: {enabled: false, position_body_m: [-2.5, 0.0, 1.5],   pitch_down_deg: 20.0, yaw_deg: 0.0,   horizontal_field_of_view_deg: 70.0}   # viewport camera of view_scene.py; never a render product

physics:
  gravity_m_s2: 1.62
  physics_rate_hz: 120
  solver_type: TGS
  gpu_dynamics_enabled: false                           # CPU dynamics: faster for one rover and 400 MB lighter (VERIFIED)
  broadphase_type: MBP
  articulation_position_iterations: 16
  articulation_velocity_iterations: 4
  articulation_self_collisions_enabled: false           # authored explicitly; the schema default is true
  contact_offset_m: null                                # null = PhysX default (the verified setting)
  rest_offset_m: null
  wheel_friction: 1.0
  rock_friction_static: 0.6
  rock_friction_dynamic: 0.5
  friction_combine_mode: multiply

spawn:
  preferred_east_m: 0.0
  preferred_north_m: 0.0
  heading_deg: 0.0                                      # counter-clockwise from east
  search_radius_m: 4.0
  maximum_slope_deg: 8.0
  maximum_roughness_m: 0.03
  crater_minimum_diameter_m: 0.5                        # stay at least one radius outside the rim of craters this large
  footprint_radius_m: 0.7

drive:
  control_rate_hz: 10                                   # headless loop; the window loop runs one application update (1/60 s) per control step
  settle_physics_steps: 120
  collider_wait_timeout_s: 60.0
  boundary_policy: stop_outward                         # stop_outward | none. Forward speed is zeroed while the rover centre is outside the drivable zone and heading further out
  fall_through_depth_m: 0.5                             # rover origin this far below the surface = fell through
  fall_through_policy: abort                            # abort (smoke tests) | recover (view_scene.py: teleport to the last safe pose)

lighting:
  sun_azimuth_deg: 90.0
  sun_elevation_deg: 15.0
  sun_angular_diameter_deg: 0.53
  sun_color: [1.0, 0.98, 0.95]
  sun_base_intensity: 6000.0                            # sibling value with default RTX tonemapping
  exposure_scale: 1.0                                   # multiplies sun_base_intensity (the one verified exposure lever)
  fill_light_fraction: 0.0                              # non-physical bounce stand-in, 0 = off

render:
  renderer: RealTimePathTracing                         # the only mode proven headless
  camera_resolution_px: [640, 480]
  anti_aliasing_mode: 4                                 # /rtx/post/aa/op; probe RP8 confirms; fallback 0
  subframes_per_camera_frame: 2                         # probe RP8 sets the smallest value above 30 dB
  warmup_subframes_after_reset: 8
  camera_near_clip_m: 0.05
  camera_far_clip_m: 100000.0
  camera_horizontal_aperture_mm: 20.955                 # vertical aperture = horizontal·height/width (square pixels)
  file_texture_flip_north: false                        # set by probe RP9
  dynamic_texture_flip_north: false                     # set by probe RP9

window:                                                 # view_scene.py only
  viewport_resolution_px: [1280, 720]
  onboard_cameras_enabled: false                        # no onboard render products unless asked: they cost video memory
  viewport_camera: follow                               # follow | front | free
  keyboard_forward_speed_m_s: 0.4
  keyboard_yaw_rate_rad_s: 0.6
```

### 6.3 Presets

Each preset is a small override file. Triangle counts are exact arithmetic from the lattice sizes.

| Parameter | `smoke` | `gpu_8gb` (default; verified on this machine by S1) | `gpu_32gb` (data only, **not run**; the first run on the 5090 is smoke test S7) |
|---|---|---|---|
| `terrain.workspace_size_m` | 12.8 | 38.4 | 76.8 |
| `terrain.visual_spacing_m` | 0.05 | 0.025 | 0.025 |
| Workspace visual lattice | 257² vertices | 1537² = 2.36 M vertices | 3073² = 9.44 M vertices |
| Workspace visual tiles (`visual_tile_cells` 128) | 2 × 2 of 6.4 m | 12 × 12 of 3.2 m (129² vertices each) | 24 × 24 of 3.2 m |
| Workspace visual triangles | 0.13 M | 4.72 M | 18.87 M |
| Collision lattice and tiles | 0.10 m, 2 × 2 tiles of 65² (`collision_tile_cells` 64), seams at east = 0 and north = 0 | 0.05 m, 3 × 3 tiles of 257², seams at ±6.4 m | 0.05 m, 6 × 6 tiles of 257² |
| Workspace collision triangles | 0.03 M | 1.18 M | 4.72 M |
| Near ring | 52 m at 0.2 m, skirt 6.4 m (0.13 M triangles) | 208 m at 0.2 m (2.09 M triangles, also the collider) | 408 m at 0.2 m (8.03 M) |
| Regional | 512 m at 2 m, skirt 64 m (0.13 M) | 2048 m at 4 m (0.52 M) | 2048 m at 2 m (2.01 M) |
| Backdrop | off | 38.4 km at 64 m (0.72 M) | 38.4 km at 32 m (2.87 M) |
| Total rendered terrain triangles | 0.39 M | 8.04 M | 31.8 M |
| Total collision triangles | 0.16 M | 3.27 M | 12.75 M |
| Drivable zone (rover centre) | 6.4 m square | 32 m square | 70.4 m square |
| Geometry minimum wavelength | 0.4 m | 0.2 m | 0.2 m |
| Crater minimum diameter (8 collision posts) | 0.8 m | 0.4 m | 0.4 m |
| Track mask texture (`tracks.mask_decimation`) | 512² (1) | 2048² (1: one texel per vertex) | 2048² (2: one texel per two vertices; a 4096² upload costs 27 ms) |
| Cameras | front only, 320 × 240 | front + rear, 640 × 480 | front + rear, 1280 × 720 |
| `render.subframes_per_camera_frame` | 1 | 2 | 4 |
| `rocks.pebble_minimum_diameter_m` | 0.05 | 0.02 | 0.015 |
| `rocks.prototypes_per_shape_class` | 2 | 6 | 12 |
| Macro albedo texture | 512² | 2048² | 4096² |
| `launch.required_free_video_memory_mib` until measured | 2500 | 6000 | 20000 |

Rules asserted by `build_tier_layout` (all presets above satisfy them):

- every tier is centred on the origin;
- `workspace_size_m / visual_spacing_m` is a multiple of `visual_tile_cells` and of `collision_decimation·collision_tile_cells`;
- each inner size is an integer multiple of the next outer spacing, and each `(outer_size − inner_size)/2` is an integer multiple of the outer spacing;
- band support: `macro_band lower edge / (1 + band_edge_relative_width) ≥ 2.5 × near_ring.spacing_m`, and the same for the meso band on the workspace visual lattice, so every band edge exists on the lattice that carries it (0.8 m / 1.3 = 0.62 m ≥ 0.5 m on the 0.2 m near ring).

Budget for `gpu_8gb`. The rendered terrain (8.04 M triangles) is below the largest count measured on this machine (8.39 M rendered and collided = +1.0 GB video memory). On top come 1.18 M invisible collision triangles, whose render-side cost is unknown until probe RP3, and the rocks. ESTIMATE: 3.0 to 3.5 GB of process video memory, resident memory 9 to 10 GB. Smoke test S1 measures it and fails above 5.5 GB. The launch guard asks for 6000 MiB free until `configs/measured_budgets.yaml` holds the measured peak plus 500 MiB.

Degradation ladder if S1 fails, changing configuration only: (1) `backdrop.spacing_m: 128`; (2) `visual_spacing_m: 0.05` with `collision_decimation: 1` (ruts keep depth and berms, lose the grouser imprint); (3) `workspace_size_m: 25.6`. If S1 shows more than 1.5 GB of headroom, `regional.spacing_m: 2.0` (2.08 M triangles) is the first upgrade.

---

## 7. Data formats

### 7.1 Scene directory (written by the bake, read by the runtime)

```
<data root>/scenes/<scene_id>/
  manifest.json                         section 7.3
  configuration_resolved.yaml           the full resolved configuration
  preview.png                           hillshade with rock and crater overlay
  terrain/
    workspace_height.npy                float32 [N, E] on the workspace VISUAL lattice, metres above the site datum (final surface, fillets included)
    workspace_dem_only_height.npy       float32, bicubic DEM only (the DEM itself is never modified)
    workspace_synthetic_detail.npy      float32, craters + fractal + fillets after blending  (SYNTHETIC)
    near_ring_height.npy                float32 [N, E] near-ring lattice (values inside the hole are valid but unused)
    regional_height.npy                 float32 (present if enabled)
    backdrop_height.npy                 float32 (present if enabled)
  craters/crater_table.npz              section 7.2
  rocks/rock_table.npz                  section 7.2
  rocks/rock_prototypes.npz             section 7.2
  rasters/                              all on the workspace COLLISION lattice unless stated
    height.npy                          float32 = workspace_height[::d, ::d]; equals the PhysX collider exactly
    slope_deg.npy                       float32, slope over a 0.5 m baseline
    roughness_m.npy                     float32, standard deviation of height minus its 1 m Gaussian mean
    crater_id.npy                       int32, 0 = none, else crater_id of the youngest crater whose rim circle covers the cell
    rock_id.npy                         int32, 0 = none, instance_id for registered rocks, −1 for unregistered cobbles
    soil_property_maps.npz              section 7.2 (own 0.5 m lattice over the near-ring extent)  (SYNTHETIC)
  textures/
    macro_albedo.png                    8-bit sRGB grey, covers the near-ring extent, row 0 = north
    detail_coarse.png  detail_fine.png  8-bit linear grey, mean 0.5, tileable
```

Sizes for `gpu_8gb`: about 50 MB of arrays plus about 10 MB of textures per scene.

Every `.npy` holding heights is written with `np.save` from a C-contiguous little-endian float32 array. The manifest stores the SHA-256 of every file listed above; `load_scene` verifies them.

### 7.2 `.npz` keys

All tables are column-oriented: every key is an array with one row per item unless noted. Strings are stored as fixed-width unicode arrays.

**`craters/crater_table.npz`** (`CraterTable`)

| Key | Type | Meaning |
|---|---|---|
| `crater_id` | int32 | 1-based, stable |
| `center_east_m`, `center_north_m` | float64 | world frame |
| `center_height_m` | float32 | final surface height at the centre |
| `diameter_m` | float32 | rim-crest diameter D |
| `fresh_depth_to_diameter` | float32 | drawn value before degradation |
| `depth_m`, `rim_height_m` | float32 | **realised** values measured from the finest stamp after the diffusion and resolution filters |
| `degradation_parameter` | float32 | u = K/r² |
| `freshness_class` | uint8 | 0 fresh, 1 moderate, 2 degraded |
| `ellipticity`, `orientation_rad` | float32 | minor/major axis ratio, major-axis direction counter-clockwise from east |
| `harmonic_amplitudes`, `harmonic_phases_rad` | float32 (n, 4) | rim harmonics m = 2..5 |
| `age_rank` | int32 | 0 = oldest; stamping order |
| `is_feature_crater`, `in_workspace` | bool | |

**`rocks/rock_table.npz`** (`RockTable`)

| Key | Type | Meaning |
|---|---|---|
| `rock_index` | int32 | 0-based row id |
| `representation` | uint8 | 0 pebble_instancer, 1 cobble_reference, 2 registered_reference, 3 near_ring_reference |
| `instance_id` | int32 | 1-based for registered rocks, −1 otherwise |
| `size_class` | uint8 | 0 pebble (< 0.05 m), 1 cobble (0.05–0.1), 2 small_rock (0.1–0.3), 3 medium_rock (0.3–1), 4 large_boulder (≥ 1) |
| `shape_class` | uint8 | 0 angular, 1 subangular, 2 rounded, 3 slab (copied from the prototype) |
| `albedo_class` | uint8 | 0 dark, 1 medium, 2 bright (copied from the prototype) |
| `prototype_index` | int16 | row of the prototype library |
| `level_of_detail` | uint8 | 0 (1280 triangles), 1 (320), 2 (80) |
| `position_w_m` | float64 (n, 3) | world position of the rock mesh origin (centre of its ellipsoid) |
| `quaternion_xyzw` | float32 (n, 4) | orientation |
| `diameter_m` | float32 | longest axis; the uniform scale applied to the unit prototype |
| `dimensions_m` | float32 (n, 3) | axis lengths a ≥ b ≥ c |
| `height_above_ground_m`, `burial_fraction` | float32 | |
| `cluster_id` | int32 | −1 = none |
| `has_collider` | bool | false for pebbles and near-ring rocks |

**`rocks/rock_prototypes.npz`** (`RockPrototypeLibrary`)

| Key | Type | Meaning |
|---|---|---|
| `prototype_names` | unicode (p,) | for example `angular_03` |
| `shape_class`, `albedo_class` | uint8 (p,) | |
| `axis_ratios` | float32 (p, 3) | 1, b/a, c/a |
| `vertices_<ppp>_level<L>` | float32 (v, 3) | unit rock (longest axis = 1 m), origin at the ellipsoid centre, `<ppp>` = zero-padded prototype index, `<L>` in 0..2 |
| `triangles_<ppp>_level<L>` | int32 (t, 3) | counter-clockwise seen from outside |
| `texture_coordinates_<ppp>_level<L>` | float32 (v, 2) | cube-projection coordinates in metres at unit scale |

**`rasters/soil_property_maps.npz`** (`SoilPropertyMaps`), every map float32 `[N, E]` unless noted, all **SYNTHETIC**

| Key | Meaning |
|---|---|
| `grid_origin_east_m`, `grid_origin_north_m`, `grid_spacing_m`, `grid_count_east`, `grid_count_north` | scalars describing the lattice |
| `friction_angle_deg`, `cohesion_pa`, `bulk_density_kg_m3`, `compaction_index` | surface properties |
| `facies_id` | uint8: 1 firm, 2 loose, 3 soft |
| `cohesive_modulus_pa`, `frictional_modulus_pa_per_m` | effective modulus is `cohesive + wheel_width·frictional`, contrast limit already applied |
| `interface_friction_static`, `interface_friction_dynamic` | for PhysX materials |
| `provenance_json` | 0-d unicode: source `synthetic_in_process`, seed, `lunar_regolith` commit, `calibration_status: illustrative_unvalidated` |

**Track field file** (`TrackField.save`): `row_start`, `column_start` (int, bounding box of touched cells on the visual lattice), `height_change_m` float32, `compaction` uint8, `disturbance` uint8, `pass_count` uint8, `last_visit_time_s` float32 (NaN = never), `wheel_index` uint8 (255 = none), `grid_json` (the `RegularGrid`).

**Wheel contact log** (`write_wheel_contact_log`): one array per name in `WHEEL_CONTACT_SAMPLE_COLUMNS`, one row per emitted sample, in emission order.

### 7.3 `manifest.json`

Validated by `core/data_model/scene_manifest.py`. Numeric values below are illustrative.

```json
{
  "schema_version": "1.0",
  "scene_id": "apollo17_landing_site_gpu_8gb_s7_9f3c1a",
  "generator": {"version": "0.1.0", "bake_hash": "9f3c1a", "scene_seed": 7,
                "numpy_version": "2.3.1", "scipy_version": "1.17.0", "lunar_regolith_commit": "aae047b"},
  "conventions": {"world": "east-north-up, z up, metres, origin at site datum", "quaternion": "xyzw",
                  "body": "x forward, y left, z up", "camera": "ros_optical",
                  "raster_indexing": "[north_index, east_index], [0,0] = south-west",
                  "planetary_curvature_applied_to": ["regional", "backdrop"]},
  "site": {
    "name": "apollo17_landing_site",
    "dem_fine_path": "/home/noron/Desktop/MEO/lunar_regolith/data/nac_dem/NAC_DTM_APOLLO17_2M.TIF",
    "dem_coarse_path": "/home/noron/Desktop/MEO/lunar_regolith/data/nac_dem/NAC_DTM_APOLLO17.TIF",
    "origin_longitude_deg": 30.7717, "origin_latitude_deg": 20.1908,
    "origin_offset_east_m": 0.0, "origin_offset_north_m": 0.0,
    "height_datum_m": -2626.99,
    "frame": {"standard_parallel_deg": 20.0, "central_longitude_deg": 180.0, "body_radius_m": 1737400.0}
  },
  "terrain": {
    "tiers": {
      "workspace": {"file": "terrain/workspace_height.npy",
                    "grid": {"origin_east_m": -19.2, "origin_north_m": -19.2, "spacing_m": 0.025,
                             "count_east": 1537, "count_north": 1537, "hole_cells": null}},
      "near_ring": {"file": "terrain/near_ring_height.npy",
                    "grid": {"origin_east_m": -104.0, "origin_north_m": -104.0, "spacing_m": 0.2,
                             "count_east": 1041, "count_north": 1041, "hole_cells": [424, 424, 192, 192]}},
      "regional":  {"file": "terrain/regional_height.npy",
                    "grid": {"origin_east_m": -1024.0, "origin_north_m": -1024.0, "spacing_m": 4.0,
                             "count_east": 513, "count_north": 513, "hole_cells": [230, 230, 52, 52]}},
      "backdrop":  {"file": "terrain/backdrop_height.npy", "grid": {"...": "..."}}
    },
    "collision": {"decimation": 2, "tile_cells": 256, "visual_tile_cells": 128},
    "drivable_extent_m": [-16.0, -16.0, 16.0, 16.0],
    "rasters": {"height": "rasters/height.npy", "slope_deg": "rasters/slope_deg.npy", "roughness_m": "rasters/roughness_m.npy",
                "crater_id": "rasters/crater_id.npy", "rock_id": "rasters/rock_id.npy",
                "soil_property_maps": "rasters/soil_property_maps.npz"},
    "synthetic_layers": ["craters", "fractal_macro", "fractal_meso", "rock_fillets"],
    "sampled_parameters": {"equilibrium_coefficient": 0.052, "roughness_multiplier": 1.12,
                           "dem_cutoff_wavelength_m": 8.0},
    "calibration": {"structure_function_anchor_m": 0.19, "anchor_source": "measured_from_dem",
                    "full_surface_target_at_one_metre_m": 0.061,
                    "dem_only_at_one_metre_m": 0.049, "craters_at_one_metre_m": 0.021, "fractal_at_one_metre_m": 0.029,
                    "fractal_gain": 1.02, "roughness_budget_exceeded": false},
    "statistics": {"full_surface_detrended_structure_function_m": {"0.5": 0.033, "1.0": 0.060, "2.0": 0.107, "4.0": 0.20},
                   "synthetic_detail_structure_function_m": {"0.5": 0.026, "1.0": 0.036, "2.0": 0.041, "4.0": 0.043},
                   "dem_band_residual_rms_m": 0.0003, "two_metre_block_residual_rms_m": 0.022,
                   "crater_count": 2500, "root_mean_square_slope_deg": {"0.1": 5.9, "1.0": 3.4}}
  },
  "regolith": {"synthetic": true, "calibration_status": "illustrative_unvalidated", "property_source": "synthetic_in_process",
               "base_normal_albedo": 0.11,
               "textures": {"macro_albedo": "textures/macro_albedo.png", "detail_coarse": "textures/detail_coarse.png",
                            "detail_fine": "textures/detail_fine.png"},
               "macro_albedo_extent_m": [-104.0, -104.0, 104.0, 104.0],
               "friction_zones": {"static": [0.58, 0.62, "..."], "dynamic": [0.51, "..."], "tile_zone_index": [[3, 4, 4], [2, 3, 5], [2, 2, 4]]}},
  "semantic_classes": {"0": "unlabelled", "1": "sky", "2": "regolith", "3": "rock", "4": "crater", "5": "wheel_track",
                       "6": "rover", "7": "lander", "8": "prop", "9": "distant_terrain", "10": "debris"},
  "objects": [{
    "id": "rock_00231", "instance_id": 231, "class": "rock", "size_class": "medium_rock",
    "shape_class": "angular", "albedo_class": "bright", "albedo": 0.20,
    "position_w_m": [14.2, 6.8, 0.31], "quaternion_xyzw": [0.0, 0.05, 0.38, 0.92],
    "dimensions_m": [0.62, 0.44, 0.30], "height_above_ground_m": 0.21, "burial_fraction": 0.3,
    "axis_aligned_bounds_w_m": [[13.9, 6.5, 0.1], [14.5, 7.1, 0.5]],
    "physics": "static", "prim_path": "/World/Rocks/Registered/rock_00231",
    "prototype": "angular_03", "cluster_id": 12
  }],
  "craters": [{
    "id": "crater_0007", "crater_id": 7, "source": "procedural", "center_w_m": [9.0, 4.0, -0.35],
    "diameter_m": 5.5, "depth_m": 0.61, "rim_height_m": 0.12, "freshness": "fresh", "is_feature_crater": true
  }],
  "counts": {"registered_rocks": 265, "cobbles": 1040, "pebbles": 9450, "near_ring_rocks": 650},
  "assumptions": ["crater equilibrium law extrapolated below 3 m", "all soil properties synthetic and uncalibrated", "..."],
  "files_sha256": {"terrain/workspace_height.npy": "…", "…": "…"}
}
```

Rules: only craters with `diameter_m ≥ 1.0` or `is_feature_crater` are listed in `craters` (the full table is in the npz). `objects` lists every registered rock. Semantic class ids 1, 4, 5, 7, 8 and 10 are reserved: version 1 authors only the labels `regolith`, `rock`, `rover` and `distant_terrain` on prims.

### 7.4 USD stage built by the runtime

Stage: Z up, `metersPerUnit = 1.0`, default prim `/World`.

| Prim path | Type and content |
|---|---|
| `/World/PhysicsScene` | `UsdPhysics.Scene`, gravity (0, 0, −1) × 1.62, `PhysxSceneAPI`: TGS, 120 steps per second, GPU dynamics off, broadphase MBP |
| `/World/Looks/LunarSurfaceNear` | MDL `lunar_surface` with macro albedo and track mask inputs; bound to workspace visual tiles and the near ring |
| `/World/Looks/LunarSurfaceFar` | MDL `lunar_surface`, constant albedo; bound to regional and backdrop |
| `/World/Looks/LunarRock_dark`, `_medium`, `_bright` | MDL `lunar_rock` with constant albedo |
| `/World/Looks/RoverBody`, `/World/Looks/RoverWheel` | `UsdPreviewSurface` |
| `/World/PhysicsMaterials/FrictionZone_00` … `_07` | `UsdPhysics.MaterialAPI` static/dynamic friction, restitution 0, `physxMaterial:frictionCombineMode = "multiply"` |
| `/World/PhysicsMaterials/Wheel`, `/World/PhysicsMaterials/Rock` | wheel 1.0/1.0, rock 0.6/0.5, both `multiply` |
| `/World/Terrain/Workspace/Visual/tile_<nn>_<ee>` | `UsdGeom.Mesh`, `subdivisionScheme = none`, points, authored vertex normals, `primvars:st` (vertex, world metres), label `class = regolith`. `<nn>` north tile index, `<ee>` east tile index, zero-padded to 2 digits. No physics schema. |
| `/World/Terrain/Workspace/Collision/tile_<nn>_<ee>` | `UsdGeom.Mesh`, invisible, `CollisionAPI` + `MeshCollisionAPI approximation = "none"`, friction-zone material bound with purpose `physics` |
| `/World/Terrain/NearRing/mesh` | visual mesh with the hole removed; also the collider when `near_ring.collision_enabled`; label `regolith` |
| `/World/Terrain/Regional/mesh`, `/World/Terrain/Backdrop/mesh` | visual only, **no collider**; label `class = distant_terrain` |
| `/World/RockPrototypes/<prototype>__level<L>__collider`, `…__visual` | Xform with child Mesh `geometry` (rock material of the prototype's albedo class bound). The `__collider` variant adds `CollisionAPI` + `MeshCollisionAPI approximation = "convexHull"`. Created only for combinations in use. Container handling: probe RP6. |
| `/World/Rocks/Registered/rock_<iiiii>` | instanceable Xform reference to a `__collider` prototype; own `xformOp:translate`, `xformOp:orient`, `xformOp:scale`; labels `class = rock`, `size = <size_class>`, `shape = <shape_class>`; `<iiiii>` = `instance_id` |
| `/World/Rocks/Cobbles/cobble_<iiiii>` | instanceable reference to a `__collider` prototype, label `class = rock`; `<iiiii>` = `rock_index` |
| `/World/Rocks/NearRing/rock_<iiiii>` | instanceable reference to a `__visual` prototype, label `class = rock` |
| `/World/Rocks/Pebbles` | one `UsdGeom.PointInstancer` (prototypes under `/World/Rocks/Pebbles/Prototypes`), no colliders, no label (instancer pixels are unlabelled, VERIFIED). `invisibleIds` hides pebbles pressed by a wheel. |
| `/World/Rover` | articulation root (`ArticulationRootAPI`, `PhysxArticulationAPI` 16/4 iterations, sleep threshold 0, `enabledSelfCollisions = false`), label `class = rover` |
| `/World/Rover/chassis`, `/front_axle`, `/wheel_front_left`, `/wheel_front_right`, `/wheel_rear_left`, `/wheel_rear_right` | unscaled Xform rigid bodies with authored mass, centre of mass and inertia; scaled geometry and colliders are children (colliders: section 10.3) |
| `/World/Rover/joints/front_axle_pivot`, `/wheel_front_left_joint`, … | revolute joints; wheel joints carry `DriveAPI("angular")` |
| `/World/Rover/chassis/cameras/front`, `/rear`, `/follow` | plain `UsdGeom.Camera` |
| `/World/Observers/<name>` | static `UsdGeom.Camera` prims created by smoke tests and `drive_demo.py` |
| `/World/Lights/Sun`, `/World/Lights/Fill` | `UsdLux.DistantLight` |

### 7.5 Run directory (written by smoke tests, probes and drive scripts)

```
<data root>/runs/<run_name>/<timestamp>/
  result.json                 numbers plus "passed"; scene_id, preset, spawn pose, configuration overrides
  *.png                       rendered images named by the script
  run_log.npz                 RoverDriveSession.save_run_log(): one row per control step
  wheel_contact_log.npz       every WheelContactSample of the run (section 7.2)
  track_field.npz             TrackField.save()
```

`run_log.npz` keys, one row per control step: `time_s`, `commanded_twist` (n, 2), `applied_twist` (n, 2), `base_pose_w` (n, 7), `base_twist_w` (n, 6), `wheel_angular_speeds_rad_s` (n, 4), `wheel_contact` (n, 4) bool, `wheel_slip_ratio`, `wheel_sinkage_m`, `wheel_normal_force_n` (n, 4), `in_drivable_zone` (n,) bool. The Isaac process log of the run is in `<data root>/runs/logs/`.

---

## 8. Public interfaces

These signatures are the contract between work packages. WP-01 creates `core/data_model/`, `core/frames/`, `core/soil_mechanics/` and `core/external/` in full, so every other pure package can be written and unit-tested against them without waiting for another package. Anything not listed here is private to its module. `ArrayLike` means a float or a numpy array; functions broadcast and return float64 arrays unless stated.

### 8.1 `core/data_model/` (WP-01)

```python
# regular_grid.py
@dataclass(frozen=True)
class RegularGrid:
    """Vertex lattice in the world frame. Arrays on it are indexed [north_index, east_index]."""
    origin_east_m: float                 # east coordinate of vertex [0, 0] (south-west corner)
    origin_north_m: float
    spacing_m: float
    count_east: int                      # number of vertices along east
    count_north: int
    hole_cells: tuple[int, int, int, int] | None = None   # (first_cell_north, first_cell_east, cell_count_north, cell_count_east) removed from the mesh
    @property
    def shape(self) -> tuple[int, int]: ...                 # (count_north, count_east)
    def east_coordinates_m(self) -> np.ndarray: ...         # float64 (count_east,)
    def north_coordinates_m(self) -> np.ndarray: ...        # float64 (count_north,)
    def extent_m(self) -> tuple[float, float, float, float]: ...   # (east_min, north_min, east_max, north_max)
    def contains(self, east_m: ArrayLike, north_m: ArrayLike) -> np.ndarray: ...   # bool
    def subsampled(self, decimation: int) -> "RegularGrid": ...
    def to_dictionary(self) -> dict: ...
    @classmethod
    def from_dictionary(cls, data: dict) -> "RegularGrid": ...

@dataclass(frozen=True)
class TileSlice:
    """One tile of a tiled lattice. Neighbouring tiles share their border vertices."""
    tile_north_index: int
    tile_east_index: int
    north_slice: slice                   # vertex rows, length tile_cells + 1
    east_slice: slice

def split_into_tiles(grid: RegularGrid, tile_cells: int) -> list[TileSlice]: ...
def tile_grid(grid: RegularGrid, tile: TileSlice) -> RegularGrid: ...
def triangle_vertex_indices(count_north: int, count_east: int,
                            hole_cells: tuple[int, int, int, int] | None = None) -> np.ndarray: ...
    # int32 (triangle_count, 3), a–d diagonal rule of section 4.1, cells inside the hole omitted
def vertex_positions(grid: RegularGrid, height_m: np.ndarray) -> np.ndarray: ...       # float32 (count_north·count_east, 3)
def vertex_normals(height_m: np.ndarray, spacing_m: float) -> np.ndarray: ...          # float32 (count_north, count_east, 3), central differences
def vertex_texture_coordinates_m(grid: RegularGrid) -> np.ndarray: ...                 # float32 (vertex_count, 2) = (east_m, north_m)
def triangulated_height(grid: RegularGrid, height_m: np.ndarray, east_m: ArrayLike, north_m: ArrayLike) -> np.ndarray: ...
    # exact height of the triangulated surface; raises ValueError for points outside the grid
def triangulated_normal(grid: RegularGrid, height_m: np.ndarray, east_m: ArrayLike, north_m: ArrayLike) -> np.ndarray: ...   # (..., 3) unit
def slope_deg_raster(height_m: np.ndarray, spacing_m: float, baseline_m: float) -> np.ndarray: ...   # float32 [N, E], slope over the given baseline

# tier_layout.py
@dataclass(frozen=True)
class TierLayout:
    workspace: RegularGrid; near_ring: RegularGrid; regional: RegularGrid | None; backdrop: RegularGrid | None
    collision_decimation: int; visual_tile_cells: int; collision_tile_cells: int
    drivable_extent_m: tuple[float, float, float, float]
    def workspace_collision_grid(self) -> RegularGrid: ...
def build_tier_layout(terrain: TerrainConfiguration) -> TierLayout: ...       # raises ValueError naming the violated rule of section 6.3

# terrain_data.py
@dataclass
class TerrainTier:
    name: str                            # "workspace" | "near_ring" | "regional" | "backdrop"
    grid: RegularGrid
    height_m: np.ndarray                 # float32 [N, E], metres above the site datum

class HeightField:
    """Exact query object for one triangulated surface (the workspace collision surface or the near ring)."""
    def __init__(self, grid: RegularGrid, height_m: np.ndarray) -> None: ...
    grid: RegularGrid
    height_m: np.ndarray
    def height_at(self, east_m: ArrayLike, north_m: ArrayLike) -> np.ndarray: ...
    def normal_at(self, east_m: ArrayLike, north_m: ArrayLike) -> np.ndarray: ...      # (..., 3)
    def slope_deg_at(self, east_m: ArrayLike, north_m: ArrayLike) -> np.ndarray: ...
    def contains(self, east_m: ArrayLike, north_m: ArrayLike) -> np.ndarray: ...       # False inside the hole of a holed grid

class DemSource(Protocol):
    """Where terrain synthesis gets DEM heights. Real implementation: core/dem (GeoTiffDemSource). Test implementation: SyntheticDemSource."""
    site_frame: "EquirectangularSiteFrame"
    def sample_fine(self, grid: RegularGrid) -> tuple[np.ndarray, np.ndarray]: ...                  # bicubic 2 m product: (height_m float64 relative to the datum, valid bool)
    def sample_coarse(self, grid: RegularGrid) -> tuple[np.ndarray, np.ndarray]: ...                # bicubic 5 m mosaic on the same datum
    def sample_coarse_block_average(self, grid: RegularGrid) -> tuple[np.ndarray, np.ndarray]: ...  # NaN-aware block mean (backdrop)

@dataclass
class TerrainBundle:
    site_frame: "EquirectangularSiteFrame"
    layout: TierLayout
    workspace: TerrainTier               # VISUAL lattice
    near_ring: TerrainTier
    regional: TerrainTier | None
    backdrop: TerrainTier | None
    workspace_dem_only_height_m: np.ndarray       # float32
    workspace_synthetic_detail_m: np.ndarray      # float32
    crater_table: "CraterTable"
    sampled_parameters: dict[str, float]
    calibration: dict[str, float | bool | str]    # the `calibration` block of the manifest
    statistics: dict
    def collision_height_field(self) -> HeightField: ...   # workspace lattice [::d, ::d]
    def near_ring_height_field(self) -> HeightField: ...   # near-ring lattice with its hole

# crater_table.py, rock_table.py, soil_property_maps.py
@dataclass
class CraterTable:                       # one numpy array attribute per key of section 7.2
    def __len__(self) -> int: ...
    def save(self, path: Path) -> None: ...
    @classmethod
    def load(cls, path: Path) -> "CraterTable": ...
    def select(self, mask: np.ndarray) -> "CraterTable": ...

@dataclass
class RockTable:                         # one numpy array attribute per key of section 7.2; same four methods as CraterTable
    ...
@dataclass
class RockPrototypeLibrary:
    prototype_names: list[str]
    shape_class: np.ndarray              # uint8 (p,)
    albedo_class: np.ndarray             # uint8 (p,)
    axis_ratios: np.ndarray              # float32 (p, 3)
    def mesh(self, prototype_index: int, level_of_detail: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]: ...
        # (vertices float32 (v,3) for a unit rock, triangles int32 (t,3), texture_coordinates float32 (v,2))
    def save(self, path: Path) -> None: ...
    @classmethod
    def load(cls, path: Path) -> "RockPrototypeLibrary": ...

@dataclass
class SoilPropertyMaps:                  # arrays per section 7.2 plus `grid: RegularGrid` and `provenance: dict`
    def sample(self, map_name: str, east_m: ArrayLike, north_m: ArrayLike) -> tuple[np.ndarray, np.ndarray]: ...
        # (values, inside) — bilinear, nearest for facies_id; points outside the lattice use the edge value and are flagged inside = False
    def effective_modulus_pa(self, east_m: ArrayLike, north_m: ArrayLike, wheel_width_m: float) -> np.ndarray: ...
    def save(self, path: Path) -> None: ...
    @classmethod
    def load(cls, path: Path) -> "SoilPropertyMaps": ...

@dataclass(frozen=True)
class FrictionZoneAssignment:
    zone_static_friction: np.ndarray     # float32 (zone_count,)
    zone_dynamic_friction: np.ndarray    # float32 (zone_count,)
    tile_zone_index: np.ndarray          # int32 (tile_count_north, tile_count_east), per workspace collision tile
    near_ring_zone_index: int

# wheel_data.py
@dataclass(frozen=True)
class WheelGeometry:
    radius_m: float                      # collider radius
    width_m: float
    grouser_count: int
    grouser_height_m: float

@dataclass(frozen=True)
class WheelKinematicState:
    """Raw state of all wheels after one physics step. Arrays have one row per wheel in the fixed wheel order."""
    time_s: float
    wheel_center_position_w_m: np.ndarray        # float64 (wheel_count, 3)
    wheel_axle_direction_w: np.ndarray           # float64 (wheel_count, 3), unit, points to the rover's left
    wheel_linear_velocity_w_m_s: np.ndarray      # float64 (wheel_count, 3)
    wheel_rotation_angle_rad: np.ndarray         # float64 (wheel_count,), increases rolling forward
    wheel_angular_speed_rad_s: np.ndarray        # float64 (wheel_count,)
    wheel_net_contact_force_w_n: np.ndarray      # float64 (wheel_count, 3)

@dataclass(frozen=True)
class WheelContactSample:
    wheel_index: int
    time_s: float
    contact_east_m: float
    contact_north_m: float
    heading_rad: float                   # direction of forward rolling, counter-clockwise from east
    wheel_rotation_angle_rad: float
    normal_force_n: float                # low-pass filtered
    slip_ratio: float
    sinkage_m: float                     # rut-floor depth below the rest surface, positive
WHEEL_CONTACT_SAMPLE_COLUMNS: tuple[str, ...]    # the nine field names, in this order; keys of wheel_contact_log.npz

# rover_specification.py
class CameraMount(BaseModel):
    name: str; enabled: bool; position_body_m: tuple[float, float, float]
    pitch_down_deg: float; yaw_deg: float; horizontal_field_of_view_deg: float
class RoverSpecification(BaseModel):     # exactly the fields of the `rover:` section in 6.2
    def wheel_geometry(self) -> WheelGeometry: ...
    def wheel_positions_body_m(self) -> np.ndarray: ...     # float64 (4, 3) in the fixed wheel order
    def total_mass_kg(self) -> float: ...
    def wheel_collider_lateral_half_extent_m(self) -> float: ...   # capsule height/2 + radius
    def minimum_ground_clearance_m(self) -> float: ...             # lowest point of all non-wheel colliders above flat ground
    def collider_overlaps_at_rest(self) -> list[str]: ...          # names of overlapping collider pairs; a model validator raises unless the list is empty
```

### 8.2 `core/seeding.py`, `core/storage/`, `core/configuration/`, `core/frames/`, `core/soil_mechanics/`, `core/external/` (WP-01)

```python
def named_generator(seed: int, stream_name: str) -> np.random.Generator: ...
def derive_seed(seed: int, stream_name: str) -> int: ...          # 63-bit child seed

# storage/data_root.py
class DataRootError(RuntimeError): ...
@dataclass(frozen=True)
class DataRoot:
    path: Path
    def scenes_directory(self) -> Path: ...
    def caches_directory(self) -> Path: ...
    def logs_directory(self) -> Path: ...
    def pytest_temporary_directory(self) -> Path: ...
    def new_run_directory(self, run_name: str) -> Path: ...        # runs/<run_name>/<timestamp>/, created
    def cache_environment(self) -> dict[str, str]: ...             # the five variables of section 3.4
def resolve_data_root(configured_path: str | None, command_line_path: str | None = None,
                      environment: Mapping[str, str] | None = None) -> Path: ...     # precedence of section 3.2; no file system access
def require_data_root(configured_path: str | None, command_line_path: str | None = None, allow_root_disk: bool = False,
                      minimum_free_gib: float = 20.0) -> DataRoot: ...
    # raises DataRootError with the message of section 3.2; creates only subdirectories; never creates the root
# `python -m lunar_rover_sim.core.storage.data_root --require [--data-root PATH]` prints the root or exits with code 4

# configuration/
def load_configuration(scene_file: Path, preset_name: str, overrides: Sequence[str] = ()) -> SimulatorConfiguration: ...
def bake_hash(configuration: SimulatorConfiguration) -> str: ...  # 6 hex digits
BAKE_HASHED_SECTIONS: tuple[str, ...] = ("site", "terrain", "rocks", "regolith")
def write_resolved_configuration(configuration: SimulatorConfiguration, path: Path) -> None: ...

# frames/site_frame.py
@dataclass(frozen=True)
class EquirectangularSiteFrame:
    origin_longitude_deg: float
    origin_latitude_deg: float
    origin_offset_east_m: float
    origin_offset_north_m: float
    height_datum_m: float
    standard_parallel_deg: float = 20.0
    central_longitude_deg: float = 180.0
    body_radius_m: float = 1_737_400.0
    def longitude_latitude_to_map(self, longitude_deg: ArrayLike, latitude_deg: ArrayLike) -> tuple[np.ndarray, np.ndarray]: ...
    def map_to_local(self, map_x_m: ArrayLike, map_y_m: ArrayLike) -> tuple[np.ndarray, np.ndarray]: ...     # (east_m, north_m)
    def local_to_map(self, east_m: ArrayLike, north_m: ArrayLike) -> tuple[np.ndarray, np.ndarray]: ...
    def to_dictionary(self) -> dict: ...
    @classmethod
    def from_dictionary(cls, data: dict) -> "EquirectangularSiteFrame": ...
def planetary_curvature_drop_m(east_m: ArrayLike, north_m: ArrayLike, body_radius_m: float) -> np.ndarray: ...   # (east² + north²)/(2·R)

# frames/pose_math.py — all quaternions xyzw
def quaternion_from_yaw(yaw_rad: float) -> np.ndarray: ...
def quaternion_from_roll_pitch_yaw(roll_rad: float, pitch_rad: float, yaw_rad: float) -> np.ndarray: ...
def roll_pitch_yaw_from_quaternion(quaternion_xyzw: np.ndarray) -> tuple[float, float, float]: ...
def quaternion_multiply(first_xyzw: np.ndarray, second_xyzw: np.ndarray) -> np.ndarray: ...
def quaternion_align_z_to(direction_w: np.ndarray) -> np.ndarray: ...
def rotate_vector(quaternion_xyzw: np.ndarray, vector: np.ndarray) -> np.ndarray: ...
def rotate_vector_inverse(quaternion_xyzw: np.ndarray, vector: np.ndarray) -> np.ndarray: ...
def transform_matrix(position_m: np.ndarray, quaternion_xyzw: np.ndarray) -> np.ndarray: ...                   # (4, 4)
def camera_usd_to_ros_optical_quaternion(quaternion_usd_xyzw: np.ndarray) -> np.ndarray: ...
def camera_look_at_quaternion_usd(position_w_m: np.ndarray, target_w_m: np.ndarray) -> np.ndarray: ...         # static observer cameras
def camera_mount_transform_body(mount: CameraMount) -> tuple[np.ndarray, np.ndarray]: ...                      # (position, USD-camera quaternion) in the body frame
def camera_intrinsic_matrix(width_px: int, height_px: int, horizontal_field_of_view_deg: float) -> np.ndarray: ...   # (3, 3), square pixels
def sun_direction_w(azimuth_deg: float, elevation_deg: float) -> np.ndarray: ...                               # unit vector towards the Sun

# soil_mechanics/bekker_sinkage.py
REFERENCE_SOIL: dict = {"friction_angle_deg": 35.0, "bulk_density_kg_m3": 1600.0, "cohesion_pa": 170.0}
def rigid_wheel_static_sinkage_m(normal_force_n: ArrayLike, effective_modulus_pa: ArrayLike, wheel_diameter_m: float, exponent: float = 1.0) -> np.ndarray: ...
    # z = (3·W / ((3 − n)·K·sqrt(D)))^(2/(2n + 1))
def wheel_sinkage_m(normal_force_n: ArrayLike, slip_ratio: ArrayLike, previous_pass_count: ArrayLike, effective_modulus_pa: ArrayLike,
                    wheel_diameter_m: float, sinkage: SinkageConfiguration) -> np.ndarray: ...
    # clip(z_static·(1 + slip_gain·|slip|)·multipass_gains[min(previous_pass_count, 3)], visual_floor_m, maximum_m)

# external/lunar_regolith_adapter.py — the only importer of lunar_regolith (numpy-only modules, via LUNAR_REGOLITH_SOURCE)
def lunar_regolith_commit() -> str: ...                                     # warns when it differs from LUNAR_REGOLITH_EXPECTED_COMMIT
def generate_gaussian_random_field(shape, spacing_m, correlation_length_east_m, correlation_length_north_m, random_generator) -> np.ndarray: ...   # wraps spatial_fields.generate_grf, standardised
def generate_correlated_latent_fields(shape, spacing_m, names: Sequence[str], correlation_matrix: np.ndarray,
                                      correlation_length_east_m, correlation_length_north_m, seed: int) -> dict[str, np.ndarray]: ...
def assign_facies_by_fraction(score: np.ndarray, fractions: Sequence[float]) -> np.ndarray: ...             # uint8, 1-based
def reece_effective_modulus_pa(friction_angle_deg, bulk_density_kg_m3, cohesion_pa, plate_width_m: float) -> np.ndarray: ...   # wraps contact.sinkage.reece_modulus with default SinkageParameters (lunar gravity)
def quantile_friction_zones(static_friction: np.ndarray, zone_count: int) -> tuple[np.ndarray, np.ndarray]: ...                # (zone index per element, zone static friction)
```

### 8.3 `core/dem/` (WP-02)

`core/dem` imports only `core/data_model` and `core/frames`. It never imports `core/terrain`.

```python
class GeoTiffHeightRaster:
    """Read-only single-band float32 GeoTIFF. Refuses compressed, tiled, big-endian or non-contiguous files with a clear error."""
    line_count: int; sample_count: int
    pixel_size_map_x_m: float; pixel_size_map_y_m: float
    upper_left_map_x_m: float; upper_left_map_y_m: float      # outer corner of pixel (row 0, column 0); AREA_OR_POINT = Area
    nodata_value: float
    @classmethod
    def open(cls, path: Path) -> "GeoTiffHeightRaster": ...
    def read_window(self, row_start: int, row_stop: int, column_start: int, column_stop: int) -> np.ndarray: ...   # float64, NaN at NoData; raises if outside
    def map_to_pixel(self, map_x_m: ArrayLike, map_y_m: ArrayLike) -> tuple[np.ndarray, np.ndarray]: ...        # fractional (row, column); integers are pixel centres

@dataclass
class SplineDemCrop:
    """Cubic B-spline coefficients of one DEM crop, computed once and shared by every lattice sampled from it."""
    coefficients: np.ndarray             # float64, scipy.ndimage.spline_filter(order=3) of the NoData-filled crop
    valid: np.ndarray                    # bool, False at NoData posts
    row_start: int; column_start: int
    margin_posts: int
def build_spline_dem_crop(raster: GeoTiffHeightRaster, site_frame: EquirectangularSiteFrame,
                          extent_m: tuple[float, float, float, float], margin_posts: int = 16) -> SplineDemCrop: ...
def sample_spline_dem_crop(crop: SplineDemCrop, raster: GeoTiffHeightRaster, site_frame: EquirectangularSiteFrame,
                           grid: RegularGrid) -> tuple[np.ndarray, np.ndarray]: ...
    # map_coordinates(order=3, prefilter=False); (height_m float64 [N, E] relative to the datum, valid bool [N, E]);
    # valid is False where any post of the 4 × 4 spline support is NoData; raises if the grid needs posts inside the margin
def block_average_dem_on_grid(raster: GeoTiffHeightRaster, site_frame: EquirectangularSiteFrame, grid: RegularGrid) -> tuple[np.ndarray, np.ndarray]: ...
    # NaN-aware mean of all posts within half a grid spacing of each vertex (backdrop, and any tier coarser than the DEM posting)
def site_frame_from_configuration(site: SiteConfiguration, fine_raster: GeoTiffHeightRaster) -> EquirectangularSiteFrame: ...
    # computes height_datum_m = bicubic DEM height at the origin, in float64
class GeoTiffDemSource:                  # implements DemSource; owns one SplineDemCrop per raster
    def __init__(self, fine_raster: GeoTiffHeightRaster, coarse_raster: GeoTiffHeightRaster, site_frame: EquirectangularSiteFrame,
                 fine_extent_m: tuple[float, float, float, float], coarse_extent_m: tuple[float, float, float, float]) -> None: ...
def open_dem_source(site: SiteConfiguration, fine_extent_m: tuple[float, float, float, float],
                    coarse_extent_m: tuple[float, float, float, float]) -> GeoTiffDemSource: ...
def fill_holes_from_secondary(height_m: np.ndarray, valid: np.ndarray, secondary_height_m: np.ndarray, secondary_valid: np.ndarray) -> tuple[np.ndarray, np.ndarray]: ...
def fill_remaining_holes(height_m: np.ndarray, valid: np.ndarray, smoothing_sigma_posts: float = 8.0) -> tuple[np.ndarray, np.ndarray]: ...
    # nearest valid value, then Gaussian smoothing feathered into the filled posts only; returns (filled, was_filled)
```

TIFF tags parsed with `struct`: ImageWidth 256, ImageLength 257, BitsPerSample 258, Compression 259 (must be 1), StripOffsets 273, RowsPerStrip 278, StripByteCounts 279, SampleFormat 339 (must be 3), ModelPixelScale 33550, ModelTiepoint 33922, GDAL_NODATA 42113. Strip offsets are read from the file, never hard-coded. Known facts for tests: first-strip offsets 115215 (`_1`), 158751 (`_2M`), 93471 (5 m); the lunar module is at row 8849, column 6854 of `_2M` with elevation −2626.99 m; NoData is −3.4028226550889045e38.

### 8.4 `core/terrain/` (WP-03)

```python
# spectral_fields.py
def band_gain_squared(wavenumber_rad_m: np.ndarray, minimum_wavelength_m: float, maximum_wavelength_m: float,
                      band_edge_relative_width: float) -> np.ndarray: ...
    # G²(k): smoothstep rising over [1 − w, 1 + w]·k_low, 1 − smoothstep over [1 − w, 1 + w]·k_high; exactly 0 outside
def synthesize_band_limited_fractal(shape: tuple[int, int], spacing_m: float, hurst_exponent: float,
                                    structure_function_anchor_m: float, structure_function_anchor_lag_m: float,
                                    minimum_wavelength_m: float, maximum_wavelength_m: float,
                                    band_edge_relative_width: float, random_generator: np.random.Generator) -> np.ndarray: ...
    # float64 [N, E] in METRES with absolute spectral normalisation (section 9.2 step 4); zero mean; periodic
def expected_band_variance_m2(shape: tuple[int, int], spacing_m: float, hurst_exponent: float, structure_function_anchor_m: float,
                              structure_function_anchor_lag_m: float, minimum_wavelength_m: float, maximum_wavelength_m: float,
                              band_edge_relative_width: float) -> float: ...     # exact discrete sum Σ S·G²·Δk² on this lattice
def gaussian_lowpass(field: np.ndarray, sigma_m: float, spacing_m: float) -> np.ndarray: ...
def resample_bicubic(field: np.ndarray, source_grid: RegularGrid, target_grid: RegularGrid) -> np.ndarray: ...

# crater_population.py
def sample_crater_population(layout: TierLayout, craters: CraterConfiguration, equilibrium_coefficient: float,
                             dem_cutoff_wavelength_m: float,
                             population_generator: np.random.Generator, shape_generator: np.random.Generator) -> CraterTable: ...
    # depth_m, rim_height_m and center_height_m are left as NaN and filled by build_terrain

# crater_stamping.py
def fresh_crater_profile(crater_table: CraterTable, crater_row: int, east_m: np.ndarray, north_m: np.ndarray) -> np.ndarray: ...
def stamp_craters(grid: RegularGrid, crater_table: CraterTable, minimum_diameter_m: float,
                  minimum_filter_sigma_m: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]: ...
    # (detail_m float64 [N, E], realised_depth_m (crater_count,), realised_rim_height_m (crater_count,)); NaN for craters not stamped on this grid

# tier_blending.py
def edge_weight(grid: RegularGrid, skirt_m: float) -> np.ndarray: ...          # float64 [N, E]: 0 on the border, smoothstep to 1 at skirt_m inside
def blend_inner_into_outer(inner_grid: RegularGrid, inner_height_m: np.ndarray,
                           outer_grid: RegularGrid, outer_height_m: np.ndarray, skirt_m: float) -> np.ndarray: ...
def with_hole_for(outer_grid: RegularGrid, inner_grid: RegularGrid) -> RegularGrid: ...

# terrain_statistics.py
def remove_best_fit_plane(height_m: np.ndarray) -> np.ndarray: ...
def structure_function_m(height_m: np.ndarray, spacing_m: float, lags_m: Sequence[float]) -> np.ndarray: ...   # sqrt(mean squared height difference), east and north lags pooled
def radial_power_spectrum(height_m: np.ndarray, spacing_m: float) -> tuple[np.ndarray, np.ndarray]: ...       # (wavenumber_rad_m, power)
def estimate_cutoff_wavelength_m(height_m: np.ndarray, spacing_m: float) -> float: ...
def roughness_m(height_m: np.ndarray, spacing_m: float, window_sigma_m: float = 1.0) -> np.ndarray: ...

# terrain_builder.py
def build_terrain(configuration: SimulatorConfiguration, dem_source: DemSource | None = None) -> TerrainBundle: ...
    # dem_source None → core.dem.open_dem_source(configuration.site, <regional or near-ring extent>, <backdrop extent>)
def add_height_layer_to_workspace(terrain: TerrainBundle, layer_m: np.ndarray, skirt_m: float) -> None: ...
    # adds layer·edge_weight to workspace.height_m and workspace_synthetic_detail_m (used for rock fillets)
```

### 8.5 `core/rocks/` (WP-04)

```python
# rock_shape_generator.py
SHAPE_CLASS_NAMES: tuple[str, ...] = ("angular", "subangular", "rounded", "slab")
@dataclass(frozen=True)
class RockShape:
    """Closed-form radial function r(direction) of one rock; evaluated at any tessellation level."""
    shape_class: str
    axis_ratios: tuple[float, float, float]
    cut_plane_normals: np.ndarray        # (k, 3)
    cut_plane_offsets: np.ndarray        # (k,)
    noise_directions: np.ndarray; noise_amplitudes: np.ndarray; noise_phases: np.ndarray
    smoothing_passes: int
def sample_rock_shape(shape_class: str, random_generator: np.random.Generator) -> RockShape: ...
def tessellate_rock_shape(shape: RockShape, subdivision_level: int) -> tuple[np.ndarray, np.ndarray]: ...   # vertices (unit rock), triangles
def build_rock_prototype_library(prototypes_per_shape_class: int, shape_generator: np.random.Generator,
                                 albedo_generator: np.random.Generator) -> RockPrototypeLibrary: ...

# rock_size_distribution.py
@dataclass(frozen=True)
class RockSizeLaw:
    """Blended size-frequency law of section 9.3: background power law, boulder tail, linear cross-fade between them."""
    background_density_at_pivot_per_m2: float; pivot_diameter_m: float; background_exponent: float
    boulder_coefficient_per_m2: float; boulder_exponent: float
    fade_start_diameter_m: float; fade_end_diameter_m: float; maximum_diameter_m: float
    def expected_count_per_m2(self, minimum_diameter_m: float, maximum_diameter_m: float) -> float: ...
    def expected_areal_fraction(self, minimum_diameter_m: float, maximum_diameter_m: float) -> float: ...   # circles of diameter D
    def sample_diameters(self, area_m2: float, minimum_diameter_m: float, maximum_diameter_m: float,
                         random_generator: np.random.Generator) -> np.ndarray: ...                           # Poisson count, then diameters
def boulder_coefficient_from_rock_abundance(rock_abundance: float, exponent: float) -> float: ...           # K = (4/π)·RA·(γ−2)/γ
def size_class_of(diameter_m: np.ndarray) -> np.ndarray: ...                                                # uint8 per section 7.2

# rock_placement.py
def place_rocks(terrain: TerrainBundle, soil_maps: SoilPropertyMaps, prototype_library: RockPrototypeLibrary,
                rocks: RockConfiguration, scene_seed: int) -> RockTable: ...

# rock_fillets.py
def rock_fillet_height_m(rock_table: RockTable, grid: RegularGrid, rocks: RockConfiguration) -> np.ndarray: ...   # float32 [N, E]
def rasterize_rock_footprints(rock_table: RockTable, grid: RegularGrid, minimum_diameter_m: float) -> np.ndarray: ...   # int32 rock_id raster of section 7.1
```

### 8.6 `core/regolith/` (WP-05)

```python
# soil_property_synthesis.py
def synthesize_soil_property_maps(extent_grid: RegularGrid, regolith: RegolithConfiguration, crater_table: CraterTable,
                                  scene_seed: int) -> SoilPropertyMaps: ...
    # reads only freshness_class, centre and diameter of the crater table (fresh-crater softening, section 9.4 step 6)

# friction_zoning.py
def assign_friction_zones(soil_maps: SoilPropertyMaps, layout: TierLayout, zone_count: int) -> FrictionZoneAssignment: ...

# albedo_maps.py, detail_textures.py
def build_macro_albedo_map(extent_grid: RegularGrid, size_px: int, soil_maps: SoilPropertyMaps, crater_table: CraterTable,
                           regolith: RegolithConfiguration, base_normal_albedo: float, random_generator: np.random.Generator) -> np.ndarray: ...
    # uint8 (size_px, size_px) sRGB-encoded, row 0 = north; mottling fades to 1 at the extent edge
def generate_detail_texture(size_px: int, tile_size_m: float, hurst_exponent: float, standard_deviation: float,
                            random_generator: np.random.Generator) -> np.ndarray: ...   # uint8, mean 127.5, periodic
def linear_to_srgb_8bit(linear: np.ndarray) -> np.ndarray: ...
```

### 8.7 `core/tracks/` (WP-06)

```python
# wheel_stamp_kernel.py
@dataclass(frozen=True)
class TrackParameters:                   # built from TracksConfiguration + WheelGeometry + grid spacing
    flank_slope: float                   # tan(flank_repose_angle)
    berm_volume_fraction: float
    berm_width_m: float
    pass_separation_time_s: float
@dataclass(frozen=True)
class StampWindow:
    north_slice: slice
    east_slice: slice
def stamp_wheel_sample(sample: WheelContactSample, wheel: WheelGeometry, parameters: TrackParameters, grid: RegularGrid,
                       height_change_m: np.ndarray, compaction: np.ndarray, disturbance: np.ndarray,
                       pass_count: np.ndarray, last_visit_time_s: np.ndarray, wheel_index: np.ndarray,
                       exclusion_mask: np.ndarray) -> StampWindow | None: ...
    # works in height-change space (no rest surface needed); modifies the state arrays in place inside the returned window;
    # None when the footprint lies outside the grid. TrackField.stamp refreshes the mask image for the returned window.

# track_field.py
@dataclass(frozen=True)
class TileUpdate:
    tile: TileSlice
    height_m: np.ndarray                 # float32 (tile_cells + 1, tile_cells + 1): rest + height_change
    normals: np.ndarray | None           # float32 (..., 3) from the global deformed lattice with a one-post halo
class TrackField:
    def __init__(self, grid: RegularGrid, rest_height_m: np.ndarray, wheel: WheelGeometry, parameters: TrackParameters,
                 visual_tile_cells: int, exclusion_mask: np.ndarray | None = None, mask_decimation: int = 1,
                 pebble_positions_w_m: np.ndarray | None = None) -> None: ...
    grid: RegularGrid                    # the workspace VISUAL lattice
    def stamp(self, samples: Sequence[WheelContactSample]) -> None: ...
    def previous_pass_count_at(self, east_m: float, north_m: float, wheel_index: int, time_s: float) -> int: ...
        # pass count at the nearest vertex, minus one when that vertex was last visited by this wheel within pass_separation_time_s
    def pop_dirty_tiles(self) -> list[TileSlice]: ...
    def tile_update(self, tile: TileSlice, with_normals: bool) -> TileUpdate: ...
    def mask_image(self) -> np.ndarray: ...            # uint8 (mask_size_px, mask_size_px, 4), updated incrementally; texel [i, j] = maximum over vertices [i·m : i·m + m, j·m : j·m + m]
    def mask_is_dirty(self) -> bool: ...
    def mark_mask_clean(self) -> None: ...
    def pop_pressed_pebble_indices(self) -> np.ndarray: ...   # int64 indices (rows of pebble_positions_w_m) newly covered by a tread cell
    def clear(self) -> list[TileSlice]: ...            # resets state (pressed pebbles included); returns the tiles whose meshes must be restored
    def save(self, path: Path) -> None: ...
    def load(self, path: Path) -> list[TileSlice]: ... # replaces state; returns the tiles to rewrite
    def track_mask(self) -> np.ndarray: ...            # bool [N, E]: height_change < −2 mm or compaction > 0   (ground truth)
    def collision_lattice_track_mask(self, decimation: int) -> np.ndarray: ...   # bool on the collision lattice (any-pooling)

# wheel_contact_sampling.py
class WheelContactSampleBuilder:
    def __init__(self, wheel: WheelGeometry, collision_rest_surface: HeightField, soil_maps: SoilPropertyMaps, track_field: TrackField,
                 sinkage: SinkageConfiguration, tracks: TracksConfiguration) -> None: ...
        # collision_rest_surface is the PhysX collider surface: the contact gap is measured against what the wheel rides on
    def reset(self, emit_first_sample_at_rest: bool = True) -> None: ...   # True: accumulators start full, so each wheel stamps once where it stands
    def process(self, state: WheelKinematicState, physics_time_step_s: float) -> list[WheelContactSample]: ...
    def latest_wheel_ground_truth(self) -> dict[str, np.ndarray]: ...
        # keys: contact (bool 4), slip_ratio, sinkage_m, normal_force_n (float32 4), contact_point_w (float32 (4, 3))

# track_replay.py
def write_wheel_contact_log(path: Path, samples: Sequence[WheelContactSample]) -> None: ...                # npz, section 7.2
def read_wheel_contact_log(path: Path) -> list[WheelContactSample]: ...
def replay_wheel_contacts(samples: Sequence[WheelContactSample], track_field: TrackField) -> None: ...     # bit-identical to online stamping

# track_polyline.py
def centreline_from_poses(positions_w_m: np.ndarray, arc_length_step_m: float = 0.05) -> np.ndarray: ...   # (n, 2)
def cross_track_error_m(position_east_m: float, position_north_m: float, polyline_w_m: np.ndarray) -> tuple[float, float]: ...   # (signed error, arc length of the closest point)
```

### 8.8 `core/rover/` and `core/perception/` (WP-07)

```python
# rover/drive_models.py
class DriveModel(Protocol):
    def wheel_velocity_targets_rad_s(self, forward_speed_m_s: float, yaw_rate_rad_s: float) -> np.ndarray: ...   # (4,) fixed wheel order
    def body_twist_from_wheels(self, wheel_angular_speeds_rad_s: np.ndarray) -> tuple[float, float]: ...
class SkidSteerDriveModel:               # ω_left = (v − ω·W_eff/2)/r, ω_right = (v + ω·W_eff/2)/r, W_eff = track_width·skid_steer_track_width_factor
    def __init__(self, rover: RoverSpecification) -> None: ...
class TwistLimiter:
    def __init__(self, rover: RoverSpecification) -> None: ...
    def reset(self) -> None: ...
    def limit(self, twist: np.ndarray, time_step_s: float) -> np.ndarray: ...          # clamps to speed limits, then to acceleration limits

# rover/twist_sources.py
class TwistSource(Protocol):
    def command(self, time_s: float, base_pose_w: np.ndarray) -> np.ndarray: ...       # [forward_speed_m_s, yaw_rate_rad_s]
    def is_finished(self) -> bool: ...
class ScriptedTwistSequence:             # a list of (duration_s, forward_speed_m_s, yaw_rate_rad_s) segments
    def __init__(self, segments: Sequence[tuple[float, float, float]]) -> None: ...
class WaypointFollower:                  # turn towards the next waypoint, drive to it, stop at the last one; no obstacle avoidance
    def __init__(self, waypoints_w_m: np.ndarray, cruise_speed_m_s: float, maximum_yaw_rate_rad_s: float,
                 arrival_radius_m: float = 0.25, heading_tolerance_rad: float = 0.2) -> None: ...

# perception/semantic_classes.py
SEMANTIC_CLASS_IDS: dict[str, int]                                   # the table of section 7.3

# perception/camera_projection.py
def unproject_depth_to_world(depth_m: np.ndarray, intrinsic_matrix: np.ndarray, camera_position_w_m: np.ndarray,
                             camera_quaternion_ros_optical_xyzw: np.ndarray) -> np.ndarray: ...       # float64 (H, W, 3), NaN where depth is not finite
def project_world_to_pixels(world_points_m: np.ndarray, intrinsic_matrix: np.ndarray, camera_position_w_m: np.ndarray,
                            camera_quaternion_ros_optical_xyzw: np.ndarray) -> tuple[np.ndarray, np.ndarray]: ...   # (pixel (n, 2) as column, row; in_front bool)
def sample_raster_at_pixels(world_points_m: np.ndarray, raster: np.ndarray, grid: RegularGrid, fill_value: float = 0) -> np.ndarray: ...   # nearest lookup, (H, W)

# perception/image_metrics.py
def peak_signal_to_noise_ratio_db(image: np.ndarray, reference: np.ndarray) -> float: ...
def intersection_over_union(first_mask: np.ndarray, second_mask: np.ndarray) -> float: ...
def contrast_to_noise_ratio(image: np.ndarray, region_mask: np.ndarray, background_mask: np.ndarray, noise_standard_deviation: float) -> float: ...
def changed_pixel_mask(image: np.ndarray, reference: np.ndarray, noise_standard_deviation: float, factor: float = 3.0) -> np.ndarray: ...
```

### 8.9 `core/scene/` (WP-08)

```python
def scene_identifier(configuration: SimulatorConfiguration, preset_name: str) -> str: ...
def bake_scene(configuration: SimulatorConfiguration, preset_name: str, data_root: DataRoot, overwrite: bool = False) -> Path: ...   # returns the scene directory

@dataclass
class LoadedScene:
    scene_directory: Path
    manifest: SceneManifest
    configuration: SimulatorConfiguration            # the resolved bake configuration
    layout: TierLayout
    tiers: dict[str, TerrainTier]                    # "workspace", "near_ring", optionally "regional", "backdrop"
    collision_height_field: HeightField
    near_ring_height_field: HeightField
    crater_table: CraterTable
    rock_table: RockTable
    rock_prototypes: RockPrototypeLibrary
    soil_maps: SoilPropertyMaps
    friction_zones: FrictionZoneAssignment
    rasters: dict[str, np.ndarray]                   # height, slope_deg, roughness_m, crater_id, rock_id
    texture_paths: dict[str, Path]                   # macro_albedo, detail_coarse, detail_fine
def load_scene(scene_directory: Path, verify_hashes: bool = True) -> LoadedScene: ...

def compute_scene_rasters(terrain: TerrainBundle, rock_table: RockTable) -> dict[str, np.ndarray]: ...   # rover-independent

def find_spawn_pose(scene: LoadedScene, spawn: SpawnConfiguration, rover: RoverSpecification) -> np.ndarray: ...
    # (7,) position + quaternion xyzw, chassis origin on the surface; deterministic (section 9.7); raises when no cell qualifies
def wheel_support_pose(collision_height_field: HeightField, rover: RoverSpecification, east_m: float, north_m: float, yaw_rad: float) -> np.ndarray: ...
    # (7,) pose of the chassis origin when the four wheels rest on the surface (least-squares plane through the four contact heights)
```

### 8.10 `runtime/` (WP-10 to WP-14)

```python
# application.py                                                                                   (WP-10)
def start_simulation_application(configuration: SimulatorConfiguration, headless: bool,
                                 command_line_data_root: str | None = None) -> tuple["SimulationApp", DataRoot]: ...
    # in order: require_data_root; preset video-memory check; cache variables of section 3.4 (when unset);
    # SimulationApp with the UJITSO setting in extra_args; carb settings (renderer, anti-aliasing, omni.physx log channel).
    # Must be called before any omni/pxr import of the runtime.

# stage_assembly                                                                                   (WP-10; rocks WP-13)
@dataclass
class TerrainPrimHandles:
    visual_tile_prim_paths: dict[tuple[int, int], str]     # (tile_north_index, tile_east_index) → prim path
    collision_tile_prim_paths: dict[tuple[int, int], str]
    near_ring_prim_path: str
@dataclass
class RockPrimHandles:
    registered_rock_prim_paths: dict[int, str]             # instance_id → prim path
    pebble_instancer_prim_path: str | None
    pebble_positions_w_m: np.ndarray                       # float64 (pebble_count, 3), instancer order
@dataclass
class SceneStageHandles:
    terrain: TerrainPrimHandles
    rocks: RockPrimHandles | None
    surface_material_path: str                             # /World/Looks/LunarSurfaceNear
    sun_prim_path: str
    fill_light_prim_path: str
def build_scene_stage(stage: "Usd.Stage", scene: LoadedScene, configuration: SimulatorConfiguration,
                      include_rocks: bool = True) -> SceneStageHandles: ...
def build_physics_scene(stage, physics: PhysicsConfiguration, friction_zones: FrictionZoneAssignment) -> None: ...
def build_terrain_prims(stage, scene: LoadedScene, configuration: SimulatorConfiguration) -> TerrainPrimHandles: ...
def build_surface_materials(stage, scene: LoadedScene, configuration: SimulatorConfiguration,
                            track_mask_asset_path: str | None = None) -> str: ...
    # returns the near-surface material path. track_mask_asset_path None: the mask input stays unauthored and the MDL
    # skips the lookup (tex::texture_isvalid). TrackMaskTexture authors the input after its first upload.
def build_lighting(stage, lighting: LightingConfiguration) -> tuple[str, str]: ...
def set_sun(stage, sun_prim_path: str, azimuth_deg: float, elevation_deg: float, intensity: float) -> None: ...
def build_rock_prims(stage, scene: LoadedScene, configuration: SimulatorConfiguration) -> RockPrimHandles: ...

# sensors/camera_rig.py                                                                            (WP-10)
def create_static_camera(stage, prim_path: str, position_w_m: np.ndarray, target_w_m: np.ndarray,
                         horizontal_field_of_view_deg: float, render: RenderConfiguration, resolution_px: tuple[int, int]) -> str: ...
class CameraRig:
    def __init__(self, stage, camera_prim_paths: dict[str, str], resolution_px: tuple[int, int],
                 annotator_names: Sequence[str] = ("rgb", "distance_to_image_plane", "semantic_segmentation")) -> None: ...
    camera_names: list[str]
    def intrinsic_matrix(self, camera_name: str) -> np.ndarray: ...
    def camera_pose_ros_optical_w(self, camera_name: str) -> tuple[np.ndarray, np.ndarray]: ...   # (position, quaternion xyzw) from the stage
    def read(self) -> dict[str, dict[str, np.ndarray]]: ...
        # after render_synchronously(): {camera: {"rgb": uint8 (H,W,3), "depth_m": float32 (H,W), inf for sky,
        #   "semantic_class_ids": uint8 (H,W) in the fixed class table, "hdr_color": float32 (H,W,3) when "HdrColor" is attached}}
    def close(self) -> None: ...

# diagnostics                                                                                      (WP-10)
def wait_for_colliders(height_fields: Sequence[HeightField], probe_points_w_m: np.ndarray, timeout_s: float,
                       before_each_update: Callable[[], None] | None = None) -> float: ...
    # loops app.update() until a raycast at every probe point hits within 1 mm of the height field that contains it;
    # calls before_each_update first in every iteration (used to hold the rover); returns the wait in seconds; raises on timeout
def verify_collision_surface(height_fields: Sequence[HeightField], sample_count: int, random_generator: np.random.Generator,
                             include_points_w_m: np.ndarray | None = None) -> dict: ...     # {"miss_count", "maximum_error_m", "mean_error_m"}; the first field that contains a point is its reference
def verify_rendered_surface(depth_m: np.ndarray, intrinsic_matrix: np.ndarray, camera_position_w_m: np.ndarray,
                            camera_quaternion_ros_optical_xyzw: np.ndarray, visual_grid: RegularGrid, visual_height_m: np.ndarray,
                            collision_height_field: HeightField, exclusion_mask: np.ndarray | None = None) -> dict: ...
    # {"maximum_error_to_visual_m", "percentile_99_error_to_collision_m", "pixel_count"}
class PerformanceProbe:                                    # named wall-clock timers plus resident memory of the process
    def measure(self, stage_name: str) -> "ContextManager[None]": ...
    def summary(self) -> dict[str, dict[str, float]]: ...  # per stage: count, mean_ms, percentile_95_ms; plus "resident_memory_mib"
def process_video_memory_mib(process_id: int | None = None) -> float: ...     # this process, from nvidia-smi --query-compute-apps
def free_video_memory_mib() -> float: ...

# stepping.py                                                                                      (WP-11)
class PhysicsStepper(Protocol):
    physics_time_step_s: float
    control_period_s: float
    def start(self) -> None: ...                               # timeline.play() + one app.update() so tensor views can be created
    def add_physics_step_callback(self, callback: Callable[[float], None]) -> None: ...
    def advance_control_period(self) -> None: ...
    def simulation_time_s(self) -> float: ...                  # sum of the physics steps seen by the callbacks
class ExplicitPhysicsStepper:    # headless: physics_steps_per_control_step × (psi.simulate(dt, t); psi.fetch_results(); callbacks). Never calls app.update() after start().
    def __init__(self, physics_rate_hz: int, control_rate_hz: int) -> None: ...
class ApplicationUpdateStepper:  # window loop: one app.update() per control period (1/60 s, 2 physics steps, renders the viewport); callbacks through omni.physx subscribe_physics_step_events
    def __init__(self, application, physics_rate_hz: int) -> None: ...
def render_synchronously(subframes: int) -> None: ...          # rep.orchestrator.step(rt_subframes=subframes, delta_time=0.0)

# rover                                                                                            (WP-11)
@dataclass
class RoverHandles:
    articulation_root_path: str
    chassis_path: str
    wheel_link_paths: list[str]                         # fixed wheel order
    wheel_joint_names: list[str]
    camera_prim_paths: dict[str, str]                   # camera name → prim path
def build_procedural_rover(stage, prim_path: str, rover: RoverSpecification, physics: PhysicsConfiguration,
                           render: RenderConfiguration, spawn_pose_w: np.ndarray) -> RoverHandles: ...
@dataclass
class RoverState:
    time_s: float
    base_pose_w: np.ndarray                             # (7,) position + quaternion xyzw
    base_twist_w: np.ndarray                            # (6,) linear + angular velocity in the world frame
    wheel_rotation_angles_rad: np.ndarray               # (4,)
    wheel_angular_speeds_rad_s: np.ndarray              # (4,)
    joint_positions: np.ndarray; joint_velocities: np.ndarray   # all degrees of freedom, articulation order
class RoverInterface:
    def __init__(self, stage, handles: RoverHandles, rover: RoverSpecification) -> None: ...
    def initialize(self) -> None: ...                   # creates tensor views; call after PhysicsStepper.start(); resolves joint indices BY NAME
    def set_wheel_velocity_targets(self, wheel_angular_speeds_rad_s: np.ndarray) -> None: ...
    def read_state(self, time_s: float) -> RoverState: ...
    def read_wheel_kinematic_state(self, time_s: float, physics_time_step_s: float) -> WheelKinematicState: ...
    def read_wheel_normal_forces_n(self, physics_time_step_s: float) -> np.ndarray: ...   # (4,)
    def teleport(self, base_pose_w: np.ndarray) -> None: ...     # root transform, zero velocities, zero joint state and targets; no timeline stop

# driving/rover_drive_session.py                                                                   (WP-11)
@dataclass
class DriveStepResult:
    time_s: float
    commanded_twist: np.ndarray                         # (2,) as requested
    applied_twist: np.ndarray                           # (2,) after the limiter and the boundary policy
    rover_state: RoverState
    wheel_ground_truth: dict[str, np.ndarray] | None    # WheelContactSampleBuilder.latest_wheel_ground_truth(); None when tracks are disabled
    in_drivable_zone: bool
    camera_frames: dict[str, dict[str, np.ndarray]] | None
class RoverDriveSession:
    """The one control loop of version 1. Smoke tests, drive_demo.py and view_scene.py all drive through it;
    the later gymnasium environment wraps exactly this object (section 14)."""
    def __init__(self, application, scene: LoadedScene, configuration: SimulatorConfiguration,
                 stepper_mode: str = "explicit",        # "explicit" | "application_update"
                 onboard_cameras_enabled: bool = True, include_rocks: bool = True) -> None: ...
    def start(self) -> RoverState: ...                  # section 10.2: build, play, wait for colliders while holding the rover, release, settle, check
    def reset(self, spawn_pose_w: np.ndarray | None = None, clear_tracks: bool = True) -> RoverState: ...
    def advance_control_step(self, twist: np.ndarray, render_cameras: bool = False) -> DriveStepResult: ...
    def add_step_listener(self, listener: Callable[[DriveStepResult], None]) -> None: ...
    def pose_log(self) -> np.ndarray: ...               # (n, 8): time_s + base_pose_w, one row per control step since the last reset
    def save_run_log(self, run_directory: Path) -> None: ...   # run_log.npz, wheel_contact_log.npz, track_field.npz (section 7.5)
    def close(self) -> None: ...
    stage: "Usd.Stage"; stage_handles: SceneStageHandles; rover_handles: RoverHandles
    rover_interface: RoverInterface; stepper: PhysicsStepper
    track_runtime: "TrackRuntime | None"; camera_rig: CameraRig | None
    spawn_pose_w: np.ndarray

# tracks                                                                                           (WP-12)
class TerrainMeshUpdater(Protocol):
    def write_tile(self, update: TileUpdate) -> None: ...
class UsdAttributeMeshUpdater:                          # mesh.GetPointsAttr().Set(Vt.Vec3fArray.FromNumpy(points)); normals likewise
    def __init__(self, stage, terrain_handles: TerrainPrimHandles, scene: LoadedScene) -> None: ...
class TrackMaskTexture:
    def __init__(self, stage, surface_material_path: str, size_px: int, texture_name: str = "wheel_track_mask") -> None: ...
    asset_path: str                                     # "dynamic://wheel_track_mask"
    def upload(self, mask_rgba: np.ndarray) -> None: ...   # the first upload also authors the material input
class TrackRuntime:
    def __init__(self, stage, scene: LoadedScene, configuration: SimulatorConfiguration, rover_interface: RoverInterface,
                 stage_handles: SceneStageHandles) -> None: ...
    track_field: TrackField
    sample_builder: WheelContactSampleBuilder
    def on_physics_step(self, physics_time_step_s: float) -> None: ...   # read wheel state, build samples, queue them
    def flush(self) -> list[WheelContactSample]: ...                    # stamp queued samples, write dirty tiles, hide pressed pebbles, upload the mask (rate-limited by wall time)
    def force_mask_upload(self) -> None: ...                            # used before a synchronous camera render
    def reset(self, track_field_file: Path | None = None) -> None: ...  # clear (or load) the field, restore tiles and pebbles, reset the sample builder
    def set_sampling_enabled(self, enabled: bool) -> None: ...
    def contact_log(self) -> list[WheelContactSample]: ...

# teleoperation                                                                                    (WP-14)
class KeyboardTwistDevice:                              # implements TwistSource; carb keyboard events through omni.appwindow; needs the window
    def __init__(self, window: WindowConfiguration) -> None: ...
    def pop_requests(self) -> set[str]: ...             # {"reset", "cycle_camera", "quit"} pressed since the last call
def run_drive_loop(session: RoverDriveSession, twist_source: TwistSource, application, maximum_duration_s: float | None = None,
                   request_source: KeyboardTwistDevice | None = None) -> dict: ...
    # drive_loop.py: until the source is finished, the window is closed, "quit" is requested or the duration is reached:
    # twist = twist_source.command(...); session.advance_control_step(twist); handle "reset" and "cycle_camera".
    # Returns {"frame_count", "simulated_time_s", "wall_time_s", "distance_travelled_m", "mask_upload_count"}.
```

---

## 9. Algorithms (all in `core/`, pure numpy and scipy)

Determinism rules for every algorithm here: float64 arithmetic until the final cast to float32; `scipy.fft` with `workers=1`; random draws only from the named streams of section 4.3; every field is generated once over its whole lattice and never per tile (the sibling's fields are not window-consistent). Same configuration and seed give byte-identical files on this environment (numpy 2.3, scipy 1.17). Across machines only 1e-5 m is promised; baked scene directories are the source of truth and are copied, not regenerated.

### 9.1 DEM window

1. Open the fine and coarse rasters with `GeoTiffHeightRaster.open`.
2. Site frame from the configured origin and offset (fixed in version 1). `height_datum_m` = bicubic fine-DEM height at the origin, float64.
3. **One shared spline crop.** `build_spline_dem_crop` reads one float64 crop of the fine DEM that covers the regional extent (the near-ring extent when the regional tier is off) plus a 16-post margin, nearest-fills NoData inside the crop, and computes the cubic B-spline coefficients once (`scipy.ndimage.spline_filter`, order 3). Every fine-DEM lattice is then sampled from these coefficients with `map_coordinates(order=3, prefilter=False)`. Reasons: the spline prefilter is global and its boundary influence decays by 0.268 per post, so 16 posts leave 7e-10 of any edge mismatch; and three tiers sampled from one set of coefficients get identical values for the same point. Bilinear interpolation is never used: it leaves 2 m creases under a low Sun and under the wheels.
4. Workspace and near ring: sampled bicubically. Any NoData inside the near-ring footprint stops the bake with a clear error (both 2 m products are hole-free within ±500 m of the lunar module).
5. Regional: sampled bicubically. When its spacing is coarser than the DEM posting (4 m in `gpu_8gb`), it is sampled at half its spacing and reduced with a `[1, 2, 1]/4` binomial filter along each axis, so it does not alias. Holes (they start 939 m from the lunar module) are filled from the coarse mosaic sampled on the same lattice with the same datum (measured agreement 0.09 m rms), then by `fill_remaining_holes`.
6. Backdrop: NaN-aware block average of the coarse mosaic, then `fill_remaining_holes`.
7. Planetary curvature: when `terrain.apply_planetary_curvature`, subtract `planetary_curvature_drop_m` from the regional and backdrop heights before any blending. The workspace and near ring are left planar.

### 9.2 Procedural terrain (`build_terrain`)

Symbols: `λc` = `site.dem_cutoff_wavelength_m`; `d_c` = collision spacing; `λmin = max(meso_band lower edge, 4·d_c)`; `H` = Hurst exponent; `w` = `band_edge_relative_width`.

1. **Lattices**: `build_tier_layout`. **Scene parameters** from stream `terrain.parameters`: equilibrium coefficient `c` and `roughness_multiplier` (uniform in their ranges when randomisation is on).
2. **Roughness anchor, measured from the DEM.** Sample the fine DEM on a 2 m lattice over the near-ring extent, remove the best-fit plane, and take `σ_anchor = structure_function(…, lag = 4 m)`. At this site the value is 0.17 to 0.21 m, inside the published 0.14 to 0.30 m for mare surfaces. A value outside `structure_function_anchor_valid_range_m` stops the bake (set `structure_function_anchor_source: configured` to override). Using the measured value makes the synthetic spectrum continue the DEM's own spectrum.
3. **Crater population** (`sample_crater_population`):
   - Cumulative density `N(>D) = c·D⁻²` per m², drawn in two disjoint diameter bands so no crater is drawn twice. **Band A**: `D ∈ [near_ring_minimum_diameter_m, λc]`, centres uniform over the whole near-ring footprint (this includes the workspace). **Band B**: `D ∈ [max(minimum_diameter_m, 8·d_c), near_ring_minimum_diameter_m)`, centres uniform over the workspace footprint grown by `apron_extent_radii·near_ring_minimum_diameter_m/2`.
   - Count per band is Poisson with mean `c·(D_low⁻² − D_high⁻²)·area`; diameters by inverse CDF `D = D_low·(1 − q·(1 − (D_low/D_high)²))^(−1/2)`.
   - Taper: a crater with `D > taper_start_diameter_m` is kept with probability `(λc − D)/(λc − taper_start_diameter_m)`, because the DEM already holds larger ones.
   - Per crater from stream `terrain.crater_shape`: `fresh_depth_to_diameter` uniform in its range; degradation `u` uniform in `[0, degradation_parameter_maximum]`; ellipticity uniform in `[ellipticity_minimum, 1]`; orientation uniform; harmonics `m = 2..5` with amplitude uniform in `[0, rim_harmonic_amplitude_maximum/m]` and uniform phase.
   - `freshness_class`: 0 if `u <` fresh threshold, 1 if `<` moderate threshold, else 2. `age_rank`: descending `u` (largest `u` = oldest = rank 0).
   - Feature craters: `feature_crater_count` craters with diameter uniform in `feature_crater_diameter_range_m`, `u = 0.01`, centre uniform in the drivable zone at least one diameter from the origin and from the drivable edge; exempt from the taper; flagged `is_feature_crater`.
   - Expected counts for `gpu_8gb`, `c = 0.05`: about 450 in band B on the workspace and about 2 100 in band A on the near ring (before the taper).
4. **Crater stamping** (`stamp_craters`, on the near-ring lattice with band A, on the workspace lattice with bands A and B), craters in ascending `age_rank` (oldest first), each on a window of ±`(apron_extent_radii + 0.5)·R` plus three filter sigmas:
   - Elliptical radius with harmonics: rotate into the crater frame, `ρ = sqrt(x² + (y/ellipticity)²)`, `θ = atan2(y/ellipticity, x)`, `r' = ρ / (R·(1 + Σ a_m·cos(m·θ + φ_m)))` with `R = D/2`.
   - Fresh profile with `d = fresh_depth_to_diameter·D`, `h = rim_height_to_diameter·D`, `E = apron_extent_radii`: bowl `z = h − d·(1 − r'²)` for `r' ≤ 1`; apron `z = h·(r'⁻³ − E⁻³)/(1 − E⁻³)` for `1 < r' < E`; `0` beyond.
   - **Filter, always applied**: Gaussian with `σ = max(R·sqrt(2u), minimum_filter_sigma_m)`, where `minimum_filter_sigma_m` is one workspace collision spacing (on the near-ring lattice, whose rendered mesh is its own collider, a sigma below half a post changes nothing). The first term is exact linear diffusion (degradation) and produces the equilibrium depth distribution; no separate depth scaling is applied (that would double count). The second term is a resolution limit: the fresh profile has a slope break of about 1 at the rim crest, and without the minimum filter the 5 cm collider would miss the crest by about a centimetre.
   - **Overprint, scale-aware**: a young crater erases only relief smaller than itself. Before adding the stamp, with `A` the crater detail accumulated so far on the window: `L = gaussian_lowpass(A, σ = R)`; `A ← L + blend·(A − L)` with `blend = smoothstep((r' − 0.6)/0.4)` inside `r' < 1` and 1 outside. A small young crater inside a large old bowl therefore keeps the old bowl beneath it and removes only the fine relief; it never ends up on a pedestal.
   - Record the realised depth (rim-crest maximum minus floor minimum of the filtered stamp) and rim height. The table keeps the values from the finest lattice the crater was stamped on.
5. **Fractal residual** (`synthesize_band_limited_fractal`), absolute normalisation:
   - Target isotropic spectrum `S(k) = C·k^−(2H+2)` (k in rad/m), `C = A·H·2^(2H−1)·Γ(1+H)/(π·Γ(1−H))`, `A = σ_anchor²/lag_anchor^(2H)`.
   - Band gain, defined on its **square**: `G²(k) = smoothstep((k − (1 − w)·k_low)/(2w·k_low)) · (1 − smoothstep((k − (1 − w)·k_high)/(2w·k_high)))`, with `k_low = 2π/λ_max`, `k_high = 2π/λ_min`. Because the square is the smoothstep, the falling edge of one band and the rising edge of the next sum to exactly one in power. `G²` is exactly zero for `k ≤ (1 − w)·k_low`.
   - Synthesis: draw real white noise `n` of the lattice shape, `field = real(ifft2(fft2(n)·sqrt(S(k)·G²(k)·Δk_east·Δk_north·N_east·N_north)))` with `Δk = 2π/(N·spacing)` and the `k = 0` term set to zero. The expected variance is `Σ S·G²·Δk²` over the discrete wavenumbers (`expected_band_variance_m2`), i.e. metres without any empirical rescale.
   - Macro band `[0.8 m, λc]` on the near-ring lattice (stream `terrain.fractal.macro`); meso band `[λmin, 0.8 m]` on the workspace lattice (stream `terrain.fractal.meso`).
   - Sanity value: `H = 0.9`, `σ(4 m) = 0.20 m`, band `[0.2 m, 8 m]` → standard deviation about 30 mm from the sharp-band closed form `π·C·(k_low^−2H − k_high^−2H)/H`. The smooth edges shift the variance by about 5 %, so tests compare against the discrete sum, not this closed form.
6. **DEM band preserved by construction.** The fractal layers have exactly zero power at wavelengths longer than `λc/(1 − w)` (11.4 m for the defaults), so they cannot change anything the DEM resolves. No extra filter is applied (a filter would break the absolute normalisation). Content between about 4 m and `λc` is deliberate: the DEM posts are 2 m apart but the DEM resolves features only from about `λc` upwards. Craters are bounded by `λc` and the taper. The DEM array itself is never modified. The manifest records two residuals: `dem_band_residual_rms_m` (Gaussian low-pass of the fractal at `σ = λc/2`; must be ≤ 1 cm, and is about 0.3 mm for the defaults) and `two_metre_block_residual_rms_m` (a statistic only; about 2.3 cm at gain 1 for the defaults, checked numerically).
7. **Calibration on the full surface** (closed form, no iteration), on the interior of the workspace lattice (inside the skirt). All structure functions are at 1 m lag, on plane-detrended fields:
   - target `T = roughness_multiplier · σ_anchor · (1 m / lag_anchor)^H` for the **full surface** (DEM + craters + fractal);
   - `D_dem = structure_function(workspace_dem_only)`. The interpolated DEM already carries most of the target (ESTIMATE from the DEM's measured 2 m and 4 m values: about 0.047 to 0.052 m of 0.055 m at this site);
   - `D_c = structure_function(craters_ws)`; `fractal_ws = resample_bicubic(macro) + meso`; `D_f = structure_function(fractal_ws)`;
   - `gain = clip( sqrt(max(T² − D_dem² − D_c², 0)) / D_f , minimum_fractal_gain, maximum_fractal_gain )`.
   - When the unclipped value is below `minimum_fractal_gain`, the DEM and the craters alone exceed the target; the manifest sets `roughness_budget_exceeded: true` and the surface is rougher than `T` by the recorded amount. `T`, `D_dem`, `D_c`, `D_f` and the gain are all written to the manifest `calibration` block.
8. **Compose**: `near_ring_full = dem_near_ring + craters_near_ring + gain·macro`; `workspace_full = dem_workspace + craters_workspace + gain·fractal_ws`.
9. **Blend from the outside in**, each time against the already final outer tier: `inner = outer_triangulated + edge_weight(inner_grid, skirt)·(inner_full − outer_triangulated)`, where `outer_triangulated` is `triangulated_height` of the outer tier at the inner vertices. Order: regional into backdrop (`regional.skirt_m`), near ring into regional (`near_ring.skirt_m`), workspace into near ring (`workspace_skirt_m`). Then cut the holes (`with_hole_for`). Inner border vertices lie exactly on outer triangles, so the surface is watertight. The blend also absorbs the few millimetres of curvature drop at the near-ring edge.
10. **Statistics** into `TerrainBundle.statistics`: structure function at 0.5, 1, 2 and 4 m of both the plane-detrended full surface and the synthetic detail alone; root-mean-square slope at 0.1 and 1 m; the two residuals of step 6; crater counts per diameter bin.
11. Geometry stops at `λmin` (4 collision posts) for the fractal and at 8 collision posts for craters, and every crater stamp is filtered at one collision spacing. The centimetre regime (`H = 0.7`) lives only in the detail textures. This is what makes the 5 cm collider and the 2.5 cm visual mesh agree to a few millimetres.

Rock fillets are added afterwards by the bake pipeline (section 9.6) through `add_height_layer_to_workspace`.

### 9.3 Rocks

**Shape** (`sample_rock_shape`, `tessellate_rock_shape`), a closed-form radial function, watertight by construction, no boolean operations:

1. Axis ratios `b/a ~ N(0.71, 0.10)` clipped to `[0.5, 0.95]`, `c/a ~ N(0.50, 0.10)` clipped to `[0.3, 0.8]` (impact-fragment statistics 2 : √2 : 1). `slab`: `c/a` uniform in `[0.25, 0.35]`.
2. `K` random cutting planes with unit normals `n_k` and offsets `h_k` uniform in `[0.6, 0.9]` of the ellipsoid radius along `n_k`: `angular` 6–10 planes, `subangular` 4–6, `rounded` 2–3, `slab` 3–5 plus one horizontal top plane.
3. Radial function on unit directions `s`: `r(s) = min(r_ellipsoid(s), min_k h_k / max(s·n_k, ε))`, multiplied by `1 + Σ_j amplitude_j·cos(frequency_j·(s·direction_j) + phase_j)` with three octaves, total amplitude 2–8 %.
4. Tessellation: icosphere at subdivision 3, 2, 1 (1280, 320, 80 triangles) for levels of detail 0, 1, 2; vertex = `r(s)·s`; then `smoothing_passes` Laplacian passes (`angular` 0, `subangular` 1, `rounded` 3, `slab` 1); rescale so the longest axis is exactly 1 m; origin at the ellipsoid centre.
5. Texture coordinates: cube projection (the two coordinates orthogonal to the dominant axis of the vertex normal).
6. Each prototype gets one shape class and one albedo class (probabilities 0.25, 0.5, 0.25 for dark, medium, bright; stream `rocks.albedo`). Albedo is a property of the prototype, so no material is ever bound on an instance.
7. The PhysX collider is `convexHull` of the prototype mesh. Convex-hull colliders on instanced prototypes are VERIFIED at unit scale with a translation only. Per-instance orientation and continuous uniform scale are UNVERIFIED and settled by probe RP6, with two named fallbacks (section 12.2).

**Size-frequency law** (`RockSizeLaw`):

- Background: `N(>D) = N_pivot·(D/0.1 m)^−γ` per m², with `N_pivot` and `γ` drawn **jointly** from stream `rocks.parameters`. Fixing the density at the 0.1 m pivot keeps pebble counts bounded when the exponent changes.
- Boulder tail: `N(>D) = K_b·D^−γ_b` with `K_b = (4/π)·RA·(γ_b − 2)/γ_b` (`RA` = Diviner rock abundance, the areal fraction of rocks ≥ 1 m; `K_b = 1.70e-3` per m² for the defaults), cut at `maximum_diameter_m`.
- Cross-fade in differential density between `background_fade_diameters_m` (0.5 m and 1 m): the background weight falls linearly from 1 to 0 and the tail weight rises from 0 to 1. Rocks of 1 m and larger therefore come from the tail only and are not counted twice. Their areal fraction is `RA·(1 − D_max^(2 − γ_b))` = 0.27 % for the defaults.
- "Diameter" is the longest axis. Areal fractions in the law and in the tests use circles of that diameter; the true footprint of a 2 : √2 : 1 fragment is about 0.71 of that circle.

**Representation** (expected counts on the `gpu_8gb` workspace, 1 475 m²; default `N_pivot = 0.18`, `γ = 2.3`; worst case `N_pivot = 0.30`, `γ = 2.6`):

| Class | Diameter | Count, default | Count, worst case | Representation | Level of detail | Collider | Ground truth |
|---|---|---|---|---|---|---|---|
| pebble | 2–5 cm | about 9 450 | about 26 400 | one `PointInstancer` | 2 | none | none (unlabelled) |
| cobble | 5–10 cm | about 1 040 | about 2 240 | instanceable reference | 2 | convex, static | label `rock`; row in the rock table |
| small_rock | 0.1–0.3 m | about 245 | about 420 | instanceable reference, registered | 1 | convex, static | manifest entry, own prim and `instance_id` |
| medium_rock | 0.3–1 m | about 20 | about 25 | instanceable reference, registered | 0 | convex, static | same |
| large_boulder | ≥ 1 m | 2–3 | 2–3 | instanceable reference, registered | 0 | convex, static | same |
| near-ring rocks | ≥ 0.3 m | about 650 | about 800 | instanceable reference | 1 | none | label `rock`, not registered |

All rocks are static in version 1. Probe RP10 and the prim-count budget of smoke test S5 are sized at the worst case. The total areal cover of rocks ≥ 2 cm is 1.1 % for the defaults and 2.3 % in the worst case.

**Placement** (`place_rocks`), streams `rocks.*`:

1. Scene parameters `N_pivot` and `γ` from stream `rocks.parameters`.
2. Intensity field on a 0.5 m lattice over the near ring: `λ(x) = exp(g·F − g²/2)` with `F` a Gaussian random field from the adapter (`clustering_field_correlation_length_m`, `g = clustering_field_sigma`), correlated with soil firmness at `rock_abundance_firmness_correlation` by mixing `F` with the standardised soil `compaction_index`.
3. Thomas clusters: `thomas_cluster_fraction` of 5–30 cm rocks are children of parents with density `thomas_parent_density_per_m2`, offsets `N(0, thomas_child_sigma_m)`; `cluster_id` recorded.
4. Sampling: per diameter bin, Poisson count from the law times area; positions by thinning against `λ`; processed from largest to smallest. Rocks ≥ 5 cm are rejected when `centre distance < overlap_rejection_factor·(D₁ + D₂)` against already placed rocks ≥ 5 cm (hash grid), when inside a keep-out zone, or (≥ 0.3 m) when the local slope exceeds `maximum_slope_for_large_rocks_deg`. Registered rocks are restricted to the workspace interior (inside the skirt).
5. Pose: yaw uniform; the short axis is rotated half-way towards the terrain normal, plus a tilt `N(0, tilt_sigma_deg)` about a random horizontal axis; burial fraction uniform in `burial_fraction_range`; ground reference = minimum of the surface height at the centre and four footprint points, so nothing floats on slopes; `position_w_m.z = ground + (0.5 − burial)·c_axis_length`.
6. Prototype: uniform among the prototypes of a shape class drawn with probabilities angular 0.35, subangular 0.35, rounded 0.2, slab 0.1.
7. `instance_id` = 1, 2, … in descending diameter over registered rocks.

**Fillets** (`rock_fillet_height_m`): for rocks ≥ `fillet_minimum_diameter_m`, a ring-shaped Gaussian skirt `amplitude·exp(−(max(ρ − ρ_f, 0))²/(2σ²))` with `ρ_f` = footprint radius, `σ = 0.6·ρ_f`, amplitude uniform in `fillet_height_fraction_range` times the exposed height (larger for rounded rocks). Combined over rocks by maximum, so overlapping skirts do not add up.

### 9.4 Regolith

**Soil property maps** (`synthesize_soil_property_maps`), SYNTHETIC and labelled so in every file:

1. Lattice: `property_map_spacing_m` over the near-ring extent.
2. Latent fields `facies`, `compaction`, `strength_residual` from `generate_correlated_latent_fields` (Matérn, the configured correlation lengths) with pairwise correlation 0.5, seeded from stream `regolith.latent_fields`.
3. `facies_id` from `assign_facies_by_fraction` (firm, loose, soft at the configured fractions).
4. `compaction_index Q = clip(logistic(1.2·latent_compaction + offset[facies]), 0, 1)` with offsets firm +1, loose 0, soft −1.
5. Fresh crater softening: inside `1.5·R` of craters with `freshness_class = 0`, `Q` is multiplied by `fresh_crater_compaction_factor` (Apollo 15 measured the deepest sinkage on soft-rimmed fresh craters). This is why the function takes the crater table.
6. `friction_angle_deg = lower + Q·(upper − lower) + 1.0·strength_residual`, clipped to the configured range; `cohesion_pa` log-interpolated across its range with `Q`; `bulk_density_kg_m3` linear with `Q`.
7. Moduli through the adapter: `cohesive_modulus_pa = reece(φ, ρ, c, plate_width = 0)`, `frictional_modulus_pa_per_m = reece(φ, ρ, c, 1) − reece(φ, ρ, c, 0)` (the Reece modulus is linear in the width). Both are scaled by a common factor so that `K(0.1 m)` stays within `soil_contrast_limit` times `reece(35°, 1600, 170, 0.1) = 83 400 Pa` (golden value for tests). Without the limit the Meyerhof scaling spans 340× and ruts would be invisible on a third of the map.
8. Interface friction: `μ_static = clip(tan(interface_friction_ratio·φ), interface_friction_clip)`, `μ_dynamic = dynamic_to_static_friction_ratio·μ_static`.

**Friction zones** (`assign_friction_zones`): mean `μ_static` over each workspace collision tile and over the near ring; `quantile_friction_zones` into `friction_zone_count` zones; one PhysX material per zone, one zone per tile. Version 1 does not use per-face subsets.

**Sinkage** (rut depth, not a force; `core/soil_mechanics`): `K = cohesive + wheel_width·frictional`; `z_static = (3·F_n/((3 − n)·K·sqrt(D)))^(2/(2n+1))`; `z = clip(z_static·(1 + slip_gain·|slip|)·multipass_gains[min(previous_pass_count, 3)], visual_floor_m, maximum_m)`. Check values: 20 N, `D = 0.30 m`, `b = 0.10 m` on reference soil → 7.6 mm; the default rover (22.3 N per wheel, `b = 0.12 m`) → 7.2 mm.

**Appearance**:
- Scene base albedo from stream `regolith.parameters` (uniform in `base_normal_albedo_range` when randomisation is on).
- `macro_albedo.png`: `base_normal_albedo·(1 + albedo_mottling_amplitude·M)·(1 + fresh_ejecta_brightening·E)`, with `M` a two-scale field (15 m and 3 m correlation, loose facies slightly darker; stream `regolith.albedo`) and `E` = 1 within `1·R` of fresh craters falling to 0 at `2·R`; mottling faded to 1 over the outer 10 % of the extent; sRGB-encoded.
- Detail textures: periodic FFT noise with amplitude `∝ f^−(H+1)`, `H = detail_hurst_exponent`, band from 2 cycles per tile to Nyquist, normalised to mean 0.5 and standard deviation `detail_texture_standard_deviation`. Tile sizes 4.0 m and 0.73 m are incommensurate to hide repetition. The texture amplitude and the material's bump strength are uncalibrated ASSUMPTIONS (the published anchor, 8° rms slope at 1 cm, is not yet tied to them).

### 9.5 Wheel tracks

**Principle.** A world-anchored `TrackField` lives on the workspace visual lattice. Each gated wheel sample carves the field with a **minimum envelope** in *height-change space* (relative to the local rest surface, so ruts follow slopes and micro-relief). The carve is idempotent and order-independent: sample density, overlapping samples and repeated wheels cannot double count. The PhysX collider is never touched; the wheel rides on the rest surface and the analytic sinkage sets the rut depth.

**State arrays** on the visual lattice (`gpu_8gb`: 1537², about 31 MB): `height_change_m` float32 (negative in ruts, positive on berms), `compaction` uint8, `disturbance` uint8, `pass_count` uint8, `last_visit_time_s` float32 (NaN = never), `wheel_index` uint8 (255 = none); plus the RGBA mask image, the read-only `exclusion_mask` (cells covered by rocks ≥ 5 cm, from `rasterize_rock_footprints` on the visual lattice) and, per pebble, the index of its nearest vertex.

**Step A — sampling** (`WheelContactSampleBuilder.process`, called every physics step at 120 Hz, vectorised over the four wheels).

Cheap part, every step, no surface query:

1. Low-pass filter the net contact force vector of each wheel with `force_filter_time_constant_s`.
2. Rolling distance accumulator `s += max(|wheel centre velocity|, |ω·r|)·dt`.

Full geometry, only for wheels whose accumulator has reached `sample_spacing_m`, one array call per query:

3. `n` = normal of the **collision** rest surface under the wheel centre; `n⊥ = normalise(n − (n·axle)·axle)`; contact point `p = centre − r·n⊥`; recompute `n` at `p` once and repeat.
4. `forward = normalise(axle × n⊥)`; `heading_rad = atan2(forward_north, forward_east)`; `v_forward = velocity·forward`; slip ratio per section 4.1.
5. Normal force `F = max(0, filtered_force·n⊥)`. Gap `g = p.z − collision_rest_height(p)`.
6. Gate: `F > contact_force_threshold_n`; `|g| ≤ contact_gap_threshold_m` (a wheel climbing a rock does not carve the ground beneath it); `p` is inside the visual lattice. If the gate holds, a sample is emitted and `s ← 0`; otherwise `s` stays full and the wheel is tried again at the next step.
7. `previous_pass_count` = pass count at the contact cell, minus one when that cell was last visited by this same wheel within `pass_separation_time_s`.
8. `sinkage_m = wheel_sinkage_m(F, slip, previous_pass_count, K(p), 2r, sinkage)`.

After a reset the accumulators start full, so every wheel stamps once where it stands as soon as it carries load. The visible wheel (which is 8 mm larger than the collider) therefore sits in its own imprint from the first frame instead of intersecting uncarved ground.

At 0.5 m/s this yields about 40 samples per second per wheel. Every emitted sample is kept in the contact log.

**Step B — stamping one sample** (`stamp_wheel_sample`). With `(e_c, n_c)` the contact point, `h` the heading, lattice points `(e, n)`:

- Wheel frame: `u = (e − e_c)·cos h + (n − n_c)·sin h` (along travel), `v = −(e − e_c)·sin h + (n − n_c)·cos h` (along the axle).
- Contact half-length `a = sqrt(r² − (r − z − g_h)²)` with `z` the sinkage and `g_h` the grouser height. Flank width `w_f = z/flank_slope`. Berm width `w_b = berm_width_m`.
- Window: `|u| ≤ a`, `|v| ≤ b/2 + w_f + w_b` (plus one post), clipped to the lattice.
- Target height change:
  `target(u, v) = −z + (r − sqrt(r² − u²)) − g_h·max(0, cos(N·(θ + u/r)))²·[|v| ≤ b/2] + flank_slope·max(0, |v| − b/2)`
  with `N` the grouser count and `θ` the wheel rotation angle. Because the true wheel angle is used, the imprint pitch is `2π·r·(1 − slip)/N` and the pattern smears when the wheel slips.
- The imprint is band-limited to the lattice on purpose. The `cos²` lobe is 25 % of the pitch wide at half height: 3.9 cm for 6 grousers on a 0.15 m wheel, which is 1.6 posts of the 2.5 cm visual lattice, so every bar is sampled at close to its full depth. A physically thin bar would fall between vertex rows and produce a beat pattern. On the `smoke` preset (5 cm posts) the imprint is not resolved and only depth and width are meaningful.
- **Carve**: `new = min(current, target)` on cells with `|v| ≤ b/2 + w_f` that are not excluded.
- **Berm**: `ΔV = Σ(current − new)·spacing²` over the carved cells. Deposit `berm_volume_fraction·ΔV` on the shoulder cells `b/2 + w_f < |v| ≤ b/2 + w_f + w_b`, `|u| ≤ a`, not excluded, with weights `weight = 1 − (|v| − b/2 − w_f)/w_b`: `height_change += fraction·ΔV·weight/(Σweight·spacing²)`. Only newly carved volume is deposited, so re-stamping the same sample changes nothing. For an 8 mm rut the berm peaks at about 0.5× the rut depth, inside the 0.2–0.5× range seen in Apollo imagery.
- **Bookkeeping** on tread cells (`|v| ≤ b/2`, `|u| ≤ a`): `pass_count += 1` (saturating) on cells that were never visited, or last visited by another wheel, or last visited more than `pass_separation_time_s` ago; then `last_visit_time_s = time`, `wheel_index = sample.wheel_index`, `compaction = max(old, round(255·clip(z/0.010, 0.3, 1)))`. On shoulder cells: `disturbance = max(old, round(255·weight))`.
- Returns the window. `TrackField.stamp` then refreshes the **mask image** inside it (R = compaction, G = disturbance, B = `min(60·pass_count, 255)`, A = 255 where anything was touched), marks every visual tile that the window grown by one post intersects as dirty, and adds every pebble whose nearest vertex became a tread cell to the pressed list.

Rut floor, as used by the tests: the height change at tread cells where the grouser term is zero. It equals `−z` at `u = 0`.

**Step C — flush, once per control step** (`TrackRuntime.flush`):

1. Stamp all queued samples.
2. For each dirty tile: `TileUpdate` with `height = rest + height_change` and, when `author_normals`, normals from central differences on the global deformed lattice with a one-post halo. Write `points` (only z differs) and `normals` with `UsdAttributeMeshUpdater` (VERIFIED: 0.3–0.8 ms per 129²–257² tile, picked up by RTX, about 1–2 ms extra on the next frame). Normally 1 or 2 tiles are dirty.
3. If `hide_pressed_pebbles` and new pebbles were pressed: write the cumulative index list to `invisibleIds` of the pebble instancer (probe RP10). Pebbles carry no collider, so without this the visible wheel would pass through them and they would sit untouched in the tread, about one per metre of track.
4. If the mask is dirty and at least `mask_upload_minimum_interval_s` of wall time has passed since the last upload: `TrackMaskTexture.upload(track_field.mask_image())` (VERIFIED headless: 8.6 ms for 2048² RGBA8, 27 ms for 4096², independent of mesh size). `force_mask_upload()` bypasses the interval before a synchronous camera render, so recorded images are never stale.

**Mask texture layout.** One world-anchored texture covers the whole workspace: `mask_size_px` = the next power of two ≥ `ceil(vertex count per side / mask_decimation)`; texel `[i, j]` covers the vertices `[i·m … i·m + m − 1, j·m … j·m + m − 1]` (maximum over the block); unused texels stay zero. The material computes `uv = (st_world − workspace_origin + m·spacing/2)/(mask_size_px·m·spacing)`; the constants are set once when the stage is built, so nothing is updated in the shader at runtime. Whether array row 0 lands at `v = 0` or `v = 1` is settled by probe RP9 and stored in `render.dynamic_texture_flip_north`, which sets the MDL input `track_mask_flip_north`.

**Shading from the mask** (`lunar_surface` MDL): tread (R): the shading normal is blended towards the geometric normal by `tread_bump_reduction·R` (compacted soil is smoother), albedo × `(1 − 0.03·R)`. Shoulders (G): albedo × `(1 − (1 − track_albedo_factor)·G)` (loosened soil is darker at moderate phase, after Kaydash 2011). Magnitudes are ASSUMPTIONS; smoke test S4 measures whether they make the track visible at high Sun.

**Wheel on the rut floor.** The collider wheel rides on the rest surface. The visual wheel mesh has radius `wheel_radius_m + nominal_sinkage_m` and its grousers add `grouser_height_m`, so on nominal soil it visibly stands on the rut floor. When the computed sinkage differs from the nominal value the wheel floats above or dips below the floor by the difference. `sinkage.maximum_m = 0.03` bounds this: the worst case (soft soil, spinning wheel, fourth pass) is a wheel hovering 2.2 cm above its trench. Deeper digging belongs with "wheel physically in the rut" in section 14.

**Reset, persistence, ground truth.**
- Tracks are world-anchored over the whole workspace, so they persist wherever the rover goes. There is no scrolling window. Tracks exist only inside the workspace, and the default boundary policy (section 10.5) keeps the rover inside the drivable zone.
- `clear()` zeroes the state inside the touched bounding box and returns the touched tiles; the runtime rewrites them from the rest surface, restores the pebbles and uploads the cleared mask.
- `save()` and `load()` round-trip the field.
- `replay_wheel_contacts(samples, field)` rebuilds a field from a contact log and is bit-identical to online stamping.
- Ground truth produced in version 1: `TrackField.track_mask()`, the saved track field, the wheel contact log, and the rover centreline from `centreline_from_poses` with `cross_track_error_m`. Tracks are not prims, so the semantic segmentation annotator cannot label them; the image-space track mask is obtained by projecting `track_mask()` into the camera with `core/perception/camera_projection.py` (used by smoke test S4).

**Fallback ladder** (configuration only; the `TrackField` and everything upstream are unchanged):

| Step | Trigger (found by probes or smoke test S4) | Change | What is lost |
|---|---|---|---|
| 0 | — | `visual_mode: geometry_and_mask`, USD writes, authored normals | nothing |
| 1 | Authored normals are not re-rendered after a runtime write (RP1) | `author_normals: false` on workspace tiles (renderer-computed normals after deformation are VERIFIED) | faint shading lines at tile borders |
| 2 | USD point writes are not seen by the synchronous render (RP2a) | implement `FabricMeshUpdater` (usdrt writes are VERIFIED at 0.03–0.10 ms per tile) and set `mesh_update_backend: fabric`; if that fails too, use the application-update stepper everywhere | Fabric writes do not flow back to USD |
| 3 | Dynamic texture fails inside the full MDL (RP9) | `visual_mode: geometry_only` | albedo and smoothness change; ruts still read under a low Sun |
| 4 | 2.5 cm visual lattice does not fit in memory (S1) | `visual_spacing_m: 0.05`, `collision_decimation: 1` | grouser imprint; depth and berms remain |
| 5 | Geometry updates are unusable | `visual_mode: mask_only` | geometric rut; albedo and bump change remain |
| 6 | Runtime `invisibleIds` writes are not rendered or are slow (RP10) | `hide_pressed_pebbles: false` | pebbles stay visible inside treads |

Smoke test S4 runs `geometry_and_mask`, `geometry_only` and `mask_only` separately, so each rung of the ladder is known to work on its own.

**Collider under the tracks: not updated in version 1.** Ruts are at most 3 cm, the multipass effect is carried by `pass_count`, and every PhysX re-cook leaks about 0.7 MiB that cannot be released (VERIFIED). See section 14.

### 9.6 Bake pipeline (`bake_scene`)

`scripts/bake_scene.py` calls `require_data_root` first and passes the `DataRoot` in.

1. Resolve the scene identifier; refuse to overwrite an existing directory unless asked; write into `scenes/<scene_id>.partial` and rename at the end.
2. `terrain = build_terrain(configuration)`.
3. `soil_maps = synthesize_soil_property_maps(<0.5 m lattice over the near ring>, regolith, terrain.crater_table, scene_seed)`.
4. `prototype_library = build_rock_prototype_library(...)`; `rock_table = place_rocks(terrain, soil_maps, prototype_library, ...)`.
5. Fillets: `add_height_layer_to_workspace(terrain, rock_fillet_height_m(rock_table, workspace_grid, rocks), workspace_skirt_m)`; fill `center_height_m` of the crater table and recompute rock `height_above_ground_m`.
6. `friction_zones = assign_friction_zones(...)`; macro albedo and the two detail textures.
7. `compute_scene_rasters(terrain, rock_table)` on the collision lattice: `height`, `slope_deg` (0.5 m baseline), `roughness_m`, `crater_id`, `rock_id`. None of them depends on the rover.
8. Manifest (objects, craters, calibration, statistics, hashes), `configuration_resolved.yaml`, `preview.png`.

Expected bake time for `gpu_8gb`: under one minute on CPU (ESTIMATE; largest FFT is 1537², crater stamping is about 2 500 small windows).

### 9.7 Spawn pose (`find_spawn_pose`)

Deterministic, no random draw, so a scene and a `spawn:` block always give the same pose.

1. Candidates: the cells of a 0.25 m lattice within `search_radius_m` of the preferred point and inside the drivable zone.
2. Reject a candidate when, within `footprint_radius_m` of it: any `rock_id` cell is non-zero; the maximum `slope_deg` exceeds `maximum_slope_deg`; the maximum `roughness_m` exceeds `maximum_roughness_m`; or it lies closer than one crater radius outside the rim (i.e. within `2·R` of the centre) of any crater with `D ≥ crater_minimum_diameter_m`.
3. Cost of the rest: `slope/maximum_slope + roughness/maximum_roughness + 0.1·distance/search_radius`. The lowest cost wins; ties go to the first in lattice order.
4. The pose is `wheel_support_pose` at that point with the configured heading: the chassis origin sits one wheel radius above the least-squares plane through the collision-surface heights under the four wheels, and roll and pitch follow that plane.
5. No candidate left: raise with the counts per rejection reason.

The chosen pose is written to every `result.json`. `rocks.keep_out_zones` keeps rocks away from the preferred point at bake time, so the search normally succeeds close to it.

---

## 10. Runtime layer (thin; `runtime/`)

The runtime touches only `pxr`, `omni.usd`, `omni.timeline`, `omni.physx`, `omni.physics.tensors`, `omni.replicator.core`, `omni.ui.DynamicTextureProvider`, `omni.appwindow` and `carb`. These sit underneath both the deprecated `isaacsim.core.api` generation and the experimental one, so nothing depends on either. Isaac Lab is not used.

### 10.1 Verified API facts every runtime module must follow

1. `from isaacsim import SimulationApp` and its construction come before any `omni` or `pxr` import. stdin must be closed (`isaac_run.sh` does it).
2. `tensors.create_simulation_view("numpy", stage_id=...)` needs the `stage_id` (`UsdUtils.StageCache.Get().GetId(stage).ToLongInt()`); without it the call fails.
3. Tensor views are created after `timeline.play()` plus one `app.update()`. A timeline stop invalidates them, so resets never stop the timeline; they teleport (0.2 ms).
4. `app.update()` with the timeline playing renders and advances physics by exactly 1/60 s (2 steps at 120 Hz). Explicit stepping with `psi.simulate(dt, t); psi.fetch_results()` advances one step without rendering.
5. `RigidContactView` requires `PhysxContactReportAPI` (threshold 0) on the wheel bodies.
6. Rigid bodies are unscaled Xforms; scaled geometry and colliders are children (a scale on the body puts joint frames in the scaled space).
7. `Gf.Vec3d(x, y, numpy_float32)` raises; cast to `float()`.
8. USD drive targets and drive damping are in degree units; tensor targets are in radians per second. Only tensor targets are used at runtime.
9. Annotator data read after a plain `app.update()` is one frame late; test images use only `rep.orchestrator.step(rt_subframes=n, delta_time=0.0)`, which is synchronous and does not advance time.
10. One annotator per name per render product.
11. `UsdSemantics.LabelsAPI.Apply(prim, "class").CreateLabelsAttr([...])`; labels inherit to descendants; a label on a parent of all rocks would merge them into one instance, so each rock prim is labelled itself.
12. `PointInstancer` instances are unlabelled in semantic segmentation.
13. PhysX does not re-cook when `points` change; that is relied upon (visual tiles are not colliders anyway).
14. Warp writes to Fabric on `cuda:0` are not rendered; no GPU stamping.
15. Collider cooking is asynchronous: a new collider is not ready after one update. Readiness is always proven with raycasts before the rover is released.
16. Mass, centre of mass and inertia are always authored (PhysX-derived values were wrong in the sibling project).
17. `physxArticulation:enabledSelfCollisions` defaults to true in the installed schema; the rover authors it explicitly.

### 10.2 Stage assembly and start sequence (`RoverDriveSession.start`)

1. New stage, Z up, `metersPerUnit = 1`, `/World` as default prim.
2. `build_scene_stage`:
   - `build_physics_scene`: scene prim and all physics materials (section 7.4).
   - `build_surface_materials`: MDL shaders with absolute asset paths (the MDL file in `assets/materials/`, textures in the scene directory); a `UsdPreviewSurface` (roughness 0.95) stays on the universal output as a fallback. The track mask input stays unauthored at this point.
   - `build_terrain_prims`: workspace visual tiles from `split_into_tiles(workspace_grid, visual_tile_cells)` with points from `vertex_positions`, indices from `triangle_vertex_indices`, normals sliced from `vertex_normals` of the **whole** workspace lattice (no shading seams between tiles) and `primvars:st` from `vertex_texture_coordinates_m`; workspace collision tiles from the subsampled lattice and `collision_tile_cells`, invisible, with the friction-zone material from `friction_zones.tile_zone_index`; near ring, regional and backdrop as one mesh each with the hole cells omitted.
   - `build_rock_prims` when `include_rocks`: prototypes actually used, then `Sdf`-level authoring of instanceable references inside one `Sdf.ChangeBlock` (VERIFIED: 2 000 rocks in 0.07 s), then the pebble instancer.
   - `build_lighting`.
3. Spawn pose from `find_spawn_pose` (or the pose passed by the caller, lifted onto the surface with `wheel_support_pose`).
4. `build_procedural_rover` at the spawn pose plus 2 cm, with the timeline stopped.
5. `stepper.start()` (play plus one update), then `RoverInterface.initialize()`.
6. **Hold and wait.** `wait_for_colliders` loops `app.update()` until a raycast at every collision tile centre, at eight near-ring points and under the spawn point hits within 1 mm of the `HeightField` that contains it. Each of those updates advances physics by 1/60 s, so before every update the rover is teleported back to the spawn pose with zero velocities (the verified 0.2 ms path). The rover therefore cannot fall through terrain that is still cooking, on a cold cache or a warm one. Timeout `drive.collider_wait_timeout_s` → abort with a clear error.
7. Create `TrackRuntime` (when `tracks.enabled`) and register its `on_physics_step` with the stepper; create the `CameraRig` for the enabled onboard cameras (when `onboard_cameras_enabled`).
8. **Release.** Teleport to the spawn pose once more, with track sampling disabled; advance `drive.settle_physics_steps` through the stepper; reset the sample builder and enable sampling.
9. **Checks**, each aborting with the measured numbers (the sibling "bodies did not fall" lesson): the chassis origin is within 5 cm of the height given by `wheel_support_pose` (the plane through the collision heights under the four wheels, not the height at the centre, which is wrong on rough ground); the sum of the four wheel normal forces is within 10 % of the rover weight (89.1 N).
10. `render_synchronously(warmup_subframes_after_reset)` when cameras exist.

After step 5 the explicit stepper never calls `app.update()` except inside step 6. All later physics goes through `PhysicsStepper.advance_control_period`, so the track callbacks and the simulation clock see every step.

`reset()` repeats steps 8 to 10 at a given pose and, by default, clears the tracks. It never stops the timeline.

### 10.3 Rover

- **Why procedural**: the Isaac asset bucket has no planetary rover; its wheeled robots use cylinder wheel colliders and Earth-gravity tuning and need the network at run time. The build pattern below is VERIFIED except where marked.
- **Bodies** (55 kg): `chassis` 42 kg box 0.80 × 0.34 × 0.20 m; `front_axle` 3 kg; four wheels of 2.5 kg. Inertia tensors are the analytic box and cylinder values.
- **Colliders, every body stated**:

  | Body | Collider | Extent |
  |---|---|---|
  | `chassis` | box 0.80 × 0.34 × 0.20 m | underside 0.22 m above flat ground (0.07 m above the axle line); side faces at y = ±0.17 m |
  | each wheel | `UsdGeom.Capsule`, radius 0.15 m, cylindrical height 0.04 m, axis Y, centred on the wheel centre plane at y = ±0.36 m | spans 0.19 m to 0.53 m from the centre line on its side |
  | `front_axle` | **none** | its visual is a beam recessed in the chassis nose plus two drop links directly inboard of the front wheels, so it lowers nothing below the chassis underside |

  The wheel capsules clear the chassis side faces by 2 cm at rest. `RoverSpecification` computes the true lateral extent of the capsule (`height/2 + radius`) and a validator refuses any specification in which two rover colliders overlap at rest. `physxArticulation:enabledSelfCollisions = false` is authored as well, because the front wheels are jointed to `front_axle`, not to the chassis, so the automatic parent-child filter does not cover that pair when the pivot moves. `minimum_ground_clearance_m()` returns 0.22 m.
- **Known cost of the capsule**: the collider is 0.34 m wide against a 0.12 m visible wheel. Its spherical caps reach about 0.10 m above the ground 0.10 m beside each visible wheel face, so the rover can touch a rock that the visible wheel appears to miss by up to about 11 cm on either side. The capsule is kept because it gave the smoothest ride in the measured comparison (slip 0.005, 2–3 contact points, 1.63 ms per step; a convex hull was six times rougher). Smoke test S5 measures the phantom contact. Named fallback if it is unacceptable: the native cylinder collider with the `omni.physx.plugin` log channel set to `error` (VERIFIED workaround for its warning flood, 1.56 ms per step, four times the vertical jitter of the capsule).
- **Joints**: `front_axle_pivot` (revolute about body x, limits ±15°) between chassis and front axle, with a `DriveAPI("angular")` of zero stiffness whose damping is `front_axle_pivot_damping_n_m_s_per_rad` converted to USD per-degree units; front wheels on the axle, rear wheels on the chassis, each a revolute joint about body y with `DriveAPI("angular")`, stiffness 0, damping `wheel_drive_damping_n_m_s_per_deg`, maximum torque 20 N·m. Five degrees of freedom. The pivot gives a statically determinate three-point support; a rigid four-wheel frame was measured to rest on a diagonal pair ([40, 2, 0, 42] N). The pivot variant itself is UNVERIFIED (probe RP7); fallback: rigid frame with documented diagonal loading.
- **Wheel visual**: a 32-sided cylinder of radius `wheel_radius_m + nominal_sinkage_m` and width `wheel_width_m` with `grouser_count` raised bars of `grouser_height_m`; no collider.
- **Materials**: wheel physics material 1.0/1.0 with `multiply` combine, so effective friction equals the terrain zone value.
- **Control**: twist → `TwistLimiter` → `SkidSteerDriveModel` → `RoverInterface.set_wheel_velocity_targets` (rad/s). Joint indices are resolved by name from the articulation view's metadata, never by assumed order.
- **Proprioception in version 1**: `RoverState` (pose, twist, wheel angles and speeds, joint states) plus the per-wheel contact flag, slip ratio, sinkage and normal force from `WheelContactSampleBuilder.latest_wheel_ground_truth`.
- **Wheel kinematics for tracks**: `get_link_transforms()` (wheel centres and axle directions), `get_dof_positions()` and `get_dof_velocities()`, link velocities, and `RigidContactView.get_net_contact_forces(dt)`. The reads are VERIFIED microsecond calls; that they are fresh after every explicit `simulate`/`fetch_results` is checked by probe RP2b.
- **Cameras**: plain `UsdGeom.Camera` prims under `/World/Rover/chassis/cameras/`, one per enabled `CameraMount`.

### 10.4 Material, lighting, rendering, cameras

**`assets/materials/lunar_surface.mdl`** starts from `/home/noron/Desktop/MEO/apollo_lander_modeling/materials/lunar_regolith.mdl` (VERIFIED to compile and render in Isaac 6.0.1) and keeps the measured back-scatter lobe: weight 0.1, roughness 1.0, normal-albedo gain 1.275. Oren-Nayar roughness is ignored by the real-time renderer, so the lobe is what makes it look lunar. The file is written once, complete; every optional texture is guarded by `tex::texture_isvalid`. Export `lunar_surface` inputs:

| Input | Type | Meaning |
|---|---|---|
| `albedo_texture`, `albedo_origin_m`, `albedo_size_m`, `albedo_flip_north` | texture_2d, float2, float, bool | macro albedo over the near-ring extent; `uv = (st − origin)/size`, v flipped when the flag is set (probe RP9, `render.file_texture_flip_north`) |
| `albedo_constant` | color | used when the texture is unset (far material) |
| `detail_coarse_texture`, `detail_coarse_tile_m` | texture_2d, float | `uv = st/tile` |
| `detail_fine_texture`, `detail_fine_tile_m` | texture_2d, float | second octave |
| `detail_albedo_strength`, `detail_bump_strength` | float, uniform float | 0.3 and 1.0 (sibling values, uncalibrated ASSUMPTIONS) |
| `backscatter_weight`, `backscatter_roughness`, `normal_albedo_gain` | float | 0.1, 1.0, 1.275 |
| `track_mask_texture` | texture_2d | `dynamic://wheel_track_mask`, colour space raw; unauthored until the first mask upload |
| `track_mask_origin_m`, `track_mask_size_m`, `track_mask_flip_north` | float2, float, bool | section 9.5 |
| `track_albedo_factor`, `tread_bump_reduction` | float | section 9.5 |

`st` is `state::texture_coordinate(0)` and is authored in world metres on every terrain mesh, so no tier or tile seams appear. The two bump normals are combined as `normalize(n_coarse + n_fine − state::normal())`. The tread smoothing is a blend of the final shading normal towards `state::normal()` (the bump factor of `base::file_bump_texture` is uniform and cannot be driven by a texture). Export `lunar_rock`: the same BSDF, constant albedo, coarse detail texture on the prototype's cube-projection coordinates.

Because the sibling MDL is changed substantially (world-metre coordinates, two detail octaves, combined bump normals, mask-driven blending), its photometric regression is re-run as smoke test S2.

**Lighting**: one `UsdLux.DistantLight` Sun, angle 0.53°, colour (1.0, 0.98, 0.95), xform ops `[rotateZ(90 − azimuth), rotateY(90 − elevation)]` (sibling pattern; a distant light shines along its local −Z). `core.frames.pose_math.sun_direction_w` is the reference; smoke test S1 checks that a test box's shadow points away from it. No dome light and no sky. An optional second distant light from near the zenith at `fill_light_fraction` of the Sun stands in for terrain bounce (non-physical, flagged, default off).

**Exposure**: version 1 uses the one lever already proven: Sun intensity = `sun_base_intensity·exposure_scale`. For a single-light scene this is equivalent to an exposure change. RTX tonemap keys are UNVERIFIED and not used.

**Renderer**: `RealTimePathTracing` only. `PathTracing` returned empty buffers headless in the sibling project and is ten times slower. At 640×480 the default DLSS mode renders internally at 320×240, so `/rtx/post/aa/op` is set explicitly (probe RP8 chooses 4 or 0).

**Cameras and annotators**: focal length from the horizontal field of view on a `camera_horizontal_aperture_mm` aperture; vertical aperture = horizontal × height / width, so pixels are square and `camera_intrinsic_matrix` matches the renderer (a smoke check compares it with the `camera_params` annotator). Clipping `[0.05 m, 100 km]`. One render product per camera. Version 1 attaches only `rgb`, `distance_to_image_plane` and `semantic_segmentation` (`colorize: False`); smoke test S2 additionally attaches `HdrColor` (linear, used by the sibling's photometric test). `CameraRig.read` remaps semantic ids through `info["idToLabels"]` every frame to the fixed class table by label name; background becomes `sky`, anything unlabelled (pebbles) becomes 0.

### 10.5 The control loop (`RoverDriveSession.advance_control_step`)

One method is the only control loop in version 1. Headless smoke tests, `drive_demo.py` and the window launcher all call it; they differ only in the stepper.

1. **Boundary policy** (`stop_outward`): when the rover centre is outside the drivable zone and the commanded forward motion would take it further out, the forward speed is set to zero (turning stays allowed, so the rover can turn back) and one console message is printed per excursion. Beyond the workspace there are no tracks, and beyond the near ring there is no collider at all.
2. `TwistLimiter.limit(twist, stepper.control_period_s)` → `SkidSteerDriveModel` → wheel targets.
3. `stepper.advance_control_period()`: explicit mode runs 12 physics steps (10 Hz control); application-update mode runs one `app.update()` (1/60 s, 2 physics steps, renders the viewport). In both modes every physics step calls `TrackRuntime.on_physics_step`.
4. `TrackRuntime.flush()`.
5. Read `RoverState`. **Fall-through guard**: if the chassis origin is more than `fall_through_depth_m` below the collider surface (or outside the near ring), policy `abort` raises with the pose (smoke tests), policy `recover` teleports to the last pose that was inside the drivable zone and prints a message (window launcher).
6. When `render_cameras`: `TrackRuntime.force_mask_upload()`, `render_synchronously(subframes_per_camera_frame)`, `CameraRig.read()`.
7. Append to the pose log and call the step listeners with the `DriveStepResult`.

In the window loop simulated time advances 1/60 s per rendered frame, so simulated speed is the frame rate divided by 60. The mask upload is limited by wall time (10 Hz), not by frame count.

### 10.6 Driving tools

**`scripts/drive_demo.py`** (headless). Arguments: scene directory or scene file plus preset, then either `--twists "5:0.4:0.0,4:0.3:0.3"` (segments of duration:speed:yaw rate, a `ScriptedTwistSequence`) or `--waypoints "3,0;3,3"` (a `WaypointFollower`: turn towards the waypoint, drive, stop; no obstacle avoidance). It renders a static observer camera before and after the drive and writes the PNG files, `result.json` and the run logs to `runs/drive_demo/<timestamp>/`.

**`scripts/view_scene.py`** (window; launched with `scripts/isaac_run.sh --interactive scripts/view_scene.py --scene <scene directory> [--preset smoke]`):

- Defaults chosen for the 8 GB card: preset `smoke`; `gpu_8gb` is accepted only when the video-memory check of the preset passes; viewport 1280 × 720; no onboard render products unless `--onboard-cameras` is given; `fall_through_policy: recover`.
- Stepper `application_update`; viewport camera `follow` (parented to the chassis).
- Keys: `W`/`S` drive forward/backward while held, `A`/`D` turn left/right while held, `Space` stop, `R` reset to the spawn pose and clear the tracks, `C` cycle the viewport camera (follow, front, free), `Esc` quit. The console prints speed, position and boundary messages.
- If the scene directory does not exist, the script says so and names the `bake_scene.py` command; it does not bake by itself.
- The loop is the function `run_drive_loop(session, twist_source, application, maximum_duration_s)` in `runtime/teleoperation/drive_loop.py`. Smoke test S6 runs this same function with a scripted twist source, so the path the user drives by hand is tested automatically.

### 10.7 Measured costs that size the loop (VERIFIED unless marked)

| Item | Cost |
|---|---|
| One physics step, 4-wheel rover on a 2.1 M-triangle terrain, CPU | 1.0–1.6 ms |
| Track sampling Step A, cheap part, per physics step | about 0.05 ms (ESTIMATE) |
| Track sampling Step A, full geometry, per batch of emitted samples | about 0.15 ms (ESTIMATE); about 4 batches per control step at 0.5 m/s |
| Stamping 16 samples | about 3 ms (ESTIMATE) |
| Visual tile `points` write (129²–257²) | 0.3–0.8 ms, +1–2 ms on the next frame |
| Dynamic texture upload 2048² / 4096² | 8.6 ms / 27 ms, at most 10 per wall second |
| `rep.orchestrator.step`, one 640×480 product, rgb | 30–39 ms |
| Two render products with four annotators | about 70 ms |
| Plain `app.update()` with a render product attached | 6.8 ms for a simple scene; 15–17 ms after a 1025² points write |
| Tensor teleport | 0.2 ms |
| Process video memory: after start / one 640×480 product / one 1080p product | 1.05–1.2 GB / 1.3–1.4 GB / 2.46 GB |

Per 10 Hz control step, headless on the RTX 5060 (ESTIMATE): 12 physics steps ≈ 18 ms + sampling ≈ 1.5 ms + stamping and tile writes ≈ 5 ms + mask upload ≈ 9 ms + camera render (when requested) ≈ 70 ms. Without cameras that is about 3× real time; with two cameras about 1× real time. The window loop and its video memory with a 1280 × 720 viewport are not measured yet (probe RP12, smoke test S6).

---

## 11. Test plan

### 11.1 Isaac-free unit tests (`scripts/run_unit_tests.sh`, seconds to run)

The wrapper checks the data root, then runs `$ISAAC_ENV/bin/python -m pytest tests/unit --basetemp=<data root>/runs/pytest_tmp`. `tests/conftest.py` refuses to run when pytest's temporary directory is on the root disk, so a bare `pytest` call fails with a message naming the wrapper.

Fixtures created in WP-01 (`tests/unit/fixtures/`): `synthetic_dem.py` (a `SyntheticDemSource`: a 2° tilted plane plus twelve fixed long sines with wavelengths from 9 to 60 m, scaled so that the plane-detrended `σ(4 m)` is 0.20 m; no files), `synthetic_geotiff.py` (`write_synthetic_geotiff(path, array)`: single-band uncompressed float32 strips with the same tags as the LROC products) and `synthetic_scene_parts.py` (a small `TerrainBundle`, `SoilPropertyMaps` and `RockTable` for packages that consume them).

| Area (package) | Acceptance checks |
|---|---|
| Import rule (WP-01) | No forbidden import under `core/` (section 2.1). |
| Data root (WP-01) | Precedence command line > environment > configuration; a missing path, a file instead of a directory, a read-only directory and a path on the root device are each refused with the message of section 3.2; the root is never created; `allow_root_disk` lifts only the device check. |
| Configuration (WP-01) | Defaults load; unknown keys are rejected; the resolved configuration of every preset equals the table of section 6.3 (camera list, resolutions, tile counts, triangle counts); the bake hash ignores machine path prefixes and ignores every non-hashed section (changing `sinkage`, `tracks`, `rover`, `render` leaves it unchanged). |
| Seeding (WP-01) | Same `(seed, name)` → identical draws; different names → different draws; value matches `SeedSequence([seed, crc32(name)])`. |
| Grid and layout (WP-01) | `triangulated_height` equals brute-force barycentric interpolation on 10⁴ random points (≤ 1e-9 m); triangles are counter-clockwise; tiles share border vertices; hole cells are omitted; every preset satisfies every rule of section 6.3 and a violated rule is named in the error (including the band-support rule with a 0.4 m near ring). |
| Site frame, pose mathematics (WP-01) | `map_x = −4252204.9897`, `map_y = 612252.7037` and east scale `1/1.0012190798` at the lunar module (1e-3 m); round trip `local → map → local` ≤ 1e-6 m; curvature drop 0.30 m at 1 024 m and 106 m at 19 200 m; quaternion round trips; intrinsic matrix has equal focal lengths in x and y. |
| Rover specification (WP-01) | Default rover: capsule lateral half-extent 0.17 m, no collider overlap, minimum ground clearance 0.22 m, total mass 55 kg; a specification with `track_width_m: 0.62` and a 0.44 m wide chassis is refused with the overlapping pair named. |
| Sinkage law, adapter (WP-01) | Golden value `reece(35°, 1600, 170, 0.1) = 83400 ± 1`; sinkage 7.6 ± 0.1 mm for 20 N, 0.30 m, 0.10 m and 7.2 ± 0.1 mm for the default rover; the cap and the floor apply; adapter fields are standardised and reproducible. |
| GeoTIFF reader (WP-02) | Round trip on the synthetic TIFF; rejects a compressed file; `needs_dem`: strip offsets 115215 / 158751 / 93471, value −2626.99 ± 0.01 m at row 8849, column 6854 of `_2M`; the file is opened read-only. |
| DEM window (WP-02) | Bicubic sampling reproduces a cubic polynomial surface to ≤ 1e-6 m at every point at least 16 posts from the crop edge; two lattices sampled from the same crop agree bit for bit at shared points; the validity mask is False within the spline support of a NoData post; datum subtraction leaves the origin at 0; a 4 m lattice built by the half-spacing rule shows no aliasing of a 5 m sine beyond 2 %. |
| Spectral fields (WP-03) | The synthesis filter reproduces `expected_band_variance_m2` exactly (deterministic, ≤ 1e-9 relative); the sample variance averaged over 8 seeds is within 5 % of it and within 10 % of the sharp-band closed form (30 mm case); fitted Hurst exponent 0.9 ± 0.05; `G²` of adjacent bands sums to 1 (≤ 1e-12); power at wavelengths longer than `λc/(1 − w)` is zero (≤ 1e-12 of the total). |
| Craters (WP-03) | Cumulative counts within Poisson 3σ of `c·D⁻²` in 5 diameter bins; no crater above `λc`; median realised depth/diameter between 0.045 and 0.10; feature craters lie in the drivable zone; a stamp has less than 1 % of its power at wavelengths below 4 collision posts. **Overprint**: a 0.4 m crater stamped inside a 6 m bowl changes the surface at its centre by less than twice its own depth; maximum slope of the crater layer over 8 seeds ≤ 45° for `c = 0.05` and ≤ 50° for `c = 0.079` (ESTIMATE of the limits; a reproduction of the recipe without overprint gave 38° and 47°). |
| Terrain build (WP-03) | Two builds with one seed → identical arrays; the DEM-only layer equals the bicubic DEM; `dem_band_residual_rms_m` ≤ 1 cm; with craters disabled the plane-detrended **full-surface** structure function at 1 m is within 5 % of the target, and with `c = 0.034` within 10 % whenever `roughness_budget_exceeded` is false; when the budget is exceeded the flag is set and the achieved value matches `sqrt(D_dem² + D_c² + (minimum_fractal_gain·D_f)²)` within 10 %; full-surface value at 4 m inside 0.14–0.30 m on the fixture; every workspace border vertex equals the near-ring triangulated surface (≤ 1e-4 m), same for the other seams; regional and backdrop carry the curvature drop, workspace and near ring do not; `abs(triangulated(collision) − triangulated(visual))` on 10⁶ random points: 99th percentile ≤ 3 mm, maximum ≤ 8 mm. |
| Rocks (WP-04) | Every prototype mesh is closed (each edge shared by exactly two triangles), outward-wound, positive volume, longest axis 1 m; axis-ratio statistics; cumulative counts within Poisson 3σ of the blended law in 6 bins; on a 1 km² diameter-only sample the areal fraction of rocks ≥ 1 m equals `RA·(1 − D_max^(2 − γ_b))` within 10 % and total cover of rocks ≥ 2 cm lies between 0.5 % and 4 % over the whole parameter range; pebble count at the worst-case parameters ≤ 30 000 on the `gpu_8gb` workspace; no overlap among rocks ≥ 5 cm under the rejection rule; no rock in a keep-out zone; base of every rock ≥ 5 cm at or below the ground reference; unique instance ids; fillet layer is zero farther than `3σ` from any rock. |
| Regolith (WP-05) | Effective modulus within the contrast limit everywhere; facies fractions within 1 % of the configuration; compaction inside a fresh crater footprint is lower than in the same seed without craters; friction zones: quantisation error below 0.05 rms; detail texture tiles (seam jump ≤ neighbour jump); albedo PNG row 0 is north. |
| Track kernel (WP-06) | Straight pass on a flat field with `grouser_height_m = 0`: rut floor depth = sinkage ± 0.5 mm, rut width = wheel width ± one post. With grousers: floor (tread cells where the grouser term is zero) = sinkage ± 0.5 mm; imprint period from an FFT peak = `2π·r·(1 − slip)/N` ± one post for slip 0 and 0.3 on a 2.5 cm lattice; bar depth varies by less than 30 % from bar to bar (the worst vertex offset gives 23 %). Re-stamping the same samples changes no array (idempotence); berm volume = `berm_volume_fraction` × carved volume ± 2 %; a second wheel on the same path deepens by the multipass gain; excluded cells never change; on a 15° tilted rest surface the rut depth relative to the rest surface is unchanged. |
| Track field (WP-06) | Dirty-tile bookkeeping (a stamp on a tile border marks both tiles); `clear`, `save`, `load` round trip; contact log written to npz and replayed equals the online arrays bit for bit; mask image equals the state arrays in the stamped window for `mask_decimation` 1 and 2; pebbles under a tread are reported exactly once and none outside a tread is reported. |
| Sample builder (WP-06) | No sample without contact force; no sample when the wheel is 5 cm above the rest surface; one sample per wheel at rest right after a reset; sample spacing 1.25 cm ± 10 % at constant speed; slip ratio and heading correct for a synthetic rolling wheel; the four-wheel vectorised result equals a per-wheel reference implementation. |
| Drive model, twist sources (WP-07) | Twist → wheel speeds → twist round trip; limiter respects acceleration limits for both control periods (1/10 s and 1/60 s); a scripted sequence finishes at the right time; the waypoint follower reaches three waypoints in a kinematic unicycle simulation and stops. |
| Perception helpers (WP-07) | `unproject_depth_to_world` and `project_world_to_pixels` are inverse for a synthetic camera; raster lookup; image metrics on constructed images (known PSNR, IoU, contrast-to-noise). |
| Scene bake (WP-08) | `smoke`-preset bake on the synthetic DEM: manifest validates; two bakes → identical `files_sha256`; `load_scene` reproduces every array; `rasters/height.npy` equals `workspace_height[::d, ::d]`; **hash coverage**: a recording proxy shows that `bake_scene` reads no configuration field outside the hashed sections and `storage`; the bake refuses a data root on the root disk. |
| Spawn pose (WP-08) | Deterministic; never on a rock, never within one radius of the rim of a crater ≥ 0.5 m, slope and roughness limits respected; `wheel_support_pose` puts all four wheel bottoms within 1 mm of a planar surface and reproduces the tilt of a 10° plane; a scene with no valid cell raises with the rejection counts. |

### 11.2 Headless Isaac smoke tests (`tests/isaac/`, strictly one process at a time through `isaac_run.sh`)

Each test is a standalone script that writes `result.json` (numbers plus `passed`) and PNG files to `<data root>/runs/<test>/<timestamp>/`. Every test that releases a rover aborts if the rover did not move after release. `run_isaac_smoke_tests.sh` finds the scripts by name, reads the `SMOKE_TEST_PRESETS` tuple at the top of each and runs one process per preset.

| Test | Scene | Numeric acceptance |
|---|---|---|
| S0 `runtime_probes` | synthetic meshes, `tests/isaac/probes/` | An answer and a number for every probe in section 12.2 |
| S1 `terrain_load` | `smoke`, then `gpu_8gb` | **Budget**: process video memory ≤ 5.5 GB; assembly and collider wait recorded (expect ≤ 30 s cold). **Collider**: 2 000 raycasts, including points on collision tile seams and on the workspace/near-ring boundary: 0 misses, maximum error ≤ 1 mm against the `HeightField` that contains each point. **Rendered surface against collider**: a downward-looking static camera at six places (two visual tile centres, a four-tile visual corner, a collision tile seam, the workspace/near-ring border, a near-ring point); `distance_to_image_plane` unprojected to world points; maximum error against the triangulated visual lattice ≤ 1 cm and 99th percentile against the collision `HeightField` ≤ 5 mm. A transposed, shifted or flipped visual tile fails this check. **Albedo orientation**: a top-down render under a zenith Sun, low-passed to 1 m, correlates with `macro_albedo` above 0.5, and the brightest ejecta patch lies within one crater radius of its crater. **Sun**: the shadow azimuth of a 1 m test box is within 5° of the anti-Sun direction. **Camera model**: `camera_params` equals `camera_intrinsic_matrix` within 0.5 px. Three low-Sun PNG files. |
| S2 `lunar_material` | spheres and a flat patch | Spheres with `lunar_surface` at phase 0°, 30°, 60°, 90°, linear `HdrColor`: phase ratios within 15 % of the sibling's measured 0.41 / 0.29 / 0.15 and the full-phase limb profile within 15 % of the sibling's 0.96 / 0.94 / 1.23, with the mask input unauthored and with a zero mask. **Tiling**: from a camera 0.9 m above a flat patch, the image autocorrelation peaks at the 4 m and 0.73 m tile periods are below 0.2. Beauty PNG files at 1 m and 5 m range. |
| S3 `rover_drive` | `smoke`, then `gpu_8gb`, rocks off, tracks off | Start poses: the default spawn; one whose straight drive crosses a collision tile seam (`gpu_8gb`: from (5.0, 2.0) heading east across east = 6.4 m; `smoke`: across east = 0); one crossing a four-tile corner diagonally. **Drive**: 5 s at 0.5 m/s travels 2.3–2.5 m. **Wheels follow the terrain**: wheel-bottom gap to the collider surface: mean ≤ 3 mm, maximum ≤ 20 mm, never below −5 mm; rover origin never more than 0.5 m below the surface. **Load**: sum of wheel normal forces at rest within 10 % of 89.1 N; minimum wheel force ≥ 0.5 × mean. **Seams**: standard deviation of frame-to-frame vertical-velocity change ≤ 0.01 while crossing. **Rendered contact**: from a static side camera, for each of the four wheels the lowest rendered wheel pixel and the rendered ground under it agree within 1.5 cm, at rest and after the drive. **Repeatability**: the same twist sequence twice from the same pose differs by ≤ 5 cm after 60 s; `reset()` takes ≤ 2 s. No NaN. The spin test records the skid factor. |
| S4 `wheel_tracks` | `smoke` (depth and width only), then `gpu_8gb` three times: `geometry_and_mask`, `geometry_only`, `mask_only` | Protocol: a static observer camera 2 m above the ground, 35° down, covering a 5 m straight and an arc. Image A before the drive; drive; teleport the rover out of view with sampling disabled; image B; `clear()`; image C. Noise floor = difference of two renders of the unchanged scene. Criteria: median `height_change` under the wheel paths within 20 % of the model sinkage; changed pixels (above 3 × noise) against the projected `TrackField.track_mask()`: intersection over union ≥ 0.5; track contrast-to-noise ratio in RGB ≥ 3 at Sun elevation 8° and at 40° (`geometry_only`: 8° only; `mask_only`: 40° only, and the asymmetric path pins `track_mask_flip_north`); depth-annotator difference along the track ≥ 3 mm (geometry modes); peak signal-to-noise ratio of C against A ≥ 35 dB; imprint period within one post of `2π·r·(1 − slip)/N` (`gpu_8gb` only); the npz contact log replays to the saved track field bit for bit; `TrackRuntime.flush` 95th percentile ≤ 20 ms; Step A mean ≤ 0.5 ms per physics step. Before/after PNG files at both Sun elevations. |
| S5 `rocks_and_labels` | `gpu_8gb` | Prim counts equal the manifest. **Labels** by USD traversal: every registered rock prim carries `class = rock` and its size and shape labels, and no common parent is labelled. **Semantic image**: in a top-down render the `rock` pixels and the projected footprints of rocks ≥ 0.1 m have intersection over union ≥ 0.5; `rover` pixels appear where the rover is. **Colliders at scale**: 200 raycasts from above onto registered rocks hit the manifest prim path within 1 cm of the scaled prototype top; a ball rests on a boulder. **Driving**: the rover is stopped by a 0.3 m rock and climbs a 5 cm cobble. **Phantom contact** (reported, not pass/fail): wheel lift while passing a 8 cm cobble 6 cm beside the visible wheel face. **Pebbles**: after a 3 m drive at least one pebble is hidden and every hidden pebble lies inside a tread. The S1 rendered-surface check is repeated with rock pixels masked. Video memory ≤ 5.5 GB. |
| S6 `window_loop_autodrive` | `smoke` (and `gpu_8gb` when the video-memory check passes) | `run_drive_loop` with a scripted twist source for 10 s of simulated time, run twice: headless with the application-update stepper, and with a window through `isaac_run.sh --interactive`. Criteria in both: distance travelled ≥ 3.5 m at 0.4 m/s; simulated time = frames / 60; a non-empty track field with four wheel paths; at most 11 mask uploads per wall second; a drive towards the boundary stops within 0.5 m outside the drivable zone; a rover placed 2 m below the surface is recovered to its last safe pose; video memory ≤ 5.5 GB; mean frame time recorded. If no display is available the windowed run is reported as "not run" and stays a manual check in `docs/driving_guide.md`. |
| S7 `performance_ladder` | `smoke`, `gpu_8gb`, degraded `gpu_8gb` (on the 5090 also `gpu_32gb`) | Per-stage milliseconds, video memory, resident memory growth ≤ 100 MiB over 2 000 control steps, growth of every cache directory of section 3.4; written to `docs/verification_report.md`; measured peaks written to `configs/measured_budgets.yaml`. |

---

## 12. Work packages

### 12.1 Ordered list

`[PURE]` packages need no GPU. After WP-01 they run in parallel with each other. `[GPU]` packages launch Isaac and are strictly serial (one Isaac process on the 8 GB card). No file is owned by two packages. A package never edits a file of another package; where a later package must change behaviour of an earlier one, the design provides a separate file (`configs/runtime_probe_results.yaml`, `configs/measured_budgets.yaml`) or a lazy-import seam (section 5).

**Pure-Python packages**

| Id | Package | Depends on | Files owned | Exit check |
|---|---|---|---|---|
| WP-01 | Foundation and contracts | — | `pyproject.toml`, `README.md`, `CLAUDE.md`, `.env.example`, `.gitignore`; all of `configs/defaults.yaml`, `configs/presets/*.yaml`, `configs/sites/*.yaml`, `configs/scenes/*.yaml`, `configs/rovers/*.yaml`; `scripts/setup_data_root.sh`, `scripts/isaac_run.sh`, `scripts/run_unit_tests.sh`; every `__init__.py`; `core/seeding.py`, `core/storage/data_root.py`, `core/configuration/*`, `core/data_model/*`, `core/frames/*`, `core/soil_mechanics/bekker_sinkage.py`, `core/external/lunar_regolith_adapter.py`; `tests/conftest.py`, `tests/unit/fixtures/*`; `tests/unit/test_import_rules.py`, `test_data_root.py`, `test_configuration.py`, `test_seeding.py`, `test_regular_grid.py`, `test_tier_layout.py`, `test_site_frame.py`, `test_pose_math.py`, `test_data_model_round_trips.py`, `test_rover_specification.py`, `test_bekker_sinkage.py`, `test_lunar_regolith_adapter.py`; `docs/decision_log.md`, `docs/assumptions.md` | Its unit tests pass; `setup_data_root.sh` has created the data root on DISK02; `isaac_run.sh --data-root /nonexistent` stops with the message of section 3.2 and exit code 4 |
| WP-02 | DEM reading | WP-01 | `core/dem/geotiff_memmap_reader.py`, `dem_window.py`, `hole_filling.py`; `tests/unit/test_geotiff_memmap_reader.py`, `test_dem_window.py`, `test_hole_filling.py` | Reader and DEM-window unit tests, including the `needs_dem` ones on the real files |
| WP-03 | Procedural terrain | WP-01 (develops against `SyntheticDemSource`) | `core/terrain/spectral_fields.py`, `crater_population.py`, `crater_stamping.py`, `tier_blending.py`, `terrain_statistics.py`, `terrain_builder.py`; `tests/unit/test_spectral_fields.py`, `test_crater_population.py`, `test_crater_stamping.py`, `test_tier_blending.py`, `test_terrain_statistics.py`, `test_terrain_builder.py` | Spectral, crater, overprint and terrain-build unit tests |
| WP-04 | Rocks | WP-01 | `core/rocks/rock_shape_generator.py`, `rock_size_distribution.py`, `rock_placement.py`, `rock_fillets.py`; `tests/unit/test_rock_shape_generator.py`, `test_rock_size_distribution.py`, `test_rock_placement.py`, `test_rock_fillets.py` | Rock unit tests |
| WP-05 | Regolith maps and textures | WP-01 | `core/regolith/soil_property_synthesis.py`, `friction_zoning.py`, `albedo_maps.py`, `detail_textures.py`; `tests/unit/test_soil_property_synthesis.py`, `test_friction_zoning.py`, `test_albedo_maps.py`, `test_detail_textures.py` | Regolith unit tests |
| WP-06 | Wheel-track core | WP-01 | `core/tracks/wheel_stamp_kernel.py`, `track_field.py`, `wheel_contact_sampling.py`, `track_replay.py`, `track_polyline.py`; `tests/unit/test_wheel_stamp_kernel.py`, `test_track_field.py`, `test_wheel_contact_sampling.py`, `test_track_replay.py`, `test_track_polyline.py` | Track kernel, track field and sample-builder unit tests |
| WP-07 | Drive and perception helpers | WP-01 | `core/rover/drive_models.py`, `twist_sources.py`; `core/perception/semantic_classes.py`, `camera_projection.py`, `image_metrics.py`; `tests/unit/test_drive_models.py`, `test_twist_sources.py`, `test_semantic_classes.py`, `test_camera_projection.py`, `test_image_metrics.py` | Their unit tests |
| WP-08 | Scene bake and integration | WP-02, WP-03, WP-04, WP-05 | `core/scene/scene_identifier.py`, `scene_bake_pipeline.py`, `scene_loader.py`, `scene_rasters.py`, `spawn_pose.py`, `scene_preview.py`; `scripts/bake_scene.py`, `preview_scene.py`, `measure_dem_cutoff.py`; `tests/unit/test_scene_bake_pipeline.py`, `test_scene_loader.py`, `test_scene_rasters.py`, `test_spawn_pose.py`, `test_bake_hash_coverage.py`, `test_real_dem_bake.py` | Scene-bake unit tests; a real `smoke` and a real `gpu_8gb` bake of the Apollo 17 site in the data root, each with a preview PNG; the measured DEM cutoff and roughness anchor reported |

**Serial GPU chain**

| Id | Package | Depends on | Files owned | Exit check |
|---|---|---|---|---|
| WP-09 | Runtime probes | WP-01 | `tests/isaac/probes/probe_common.py`, `probe_rp01_runtime_normals.py`, `probe_rp02_stepping_and_views.py`, `probe_rp03_invisible_colliders.py`, `probe_rp04_tile_assembly.py`, `probe_rp05_seam_crossing.py`, `probe_rp06_rock_instances.py`, `probe_rp07_rover_pivot.py`, `probe_rp08_render_settings.py`, `probe_rp09_material_textures.py`, `probe_rp09_material.mdl`, `probe_rp10_pebbles.py`, `probe_rp11_cache_map.py`, `probe_rp12_window.py`; `scripts/run_isaac_smoke_tests.sh`; `configs/runtime_probe_results.yaml`; `docs/runtime_probe_report.md` | S0: an answer and a number for every probe; fallback switches written to `configs/runtime_probe_results.yaml` |
| WP-10 | Application, terrain stage, material, cameras | WP-07, WP-08, WP-09 | `runtime/application.py`; `runtime/stage_assembly/physics_scene_builder.py`, `terrain_stage_builder.py`, `material_builder.py`, `lighting_builder.py`, `scene_stage_builder.py`; `runtime/sensors/camera_rig.py`; `runtime/diagnostics/collision_verification.py`, `rendered_surface_verification.py`, `performance_probe.py`, `gpu_memory_monitor.py`; `assets/materials/lunar_surface.mdl`; `tests/isaac/smoke_s1_terrain_load.py`, `smoke_s2_lunar_material.py` | S1 and S2 |
| WP-11 | Rover, stepping, drive session | WP-07, WP-10 | `runtime/stepping.py`; `runtime/rover/procedural_rover_builder.py`, `rover_interface.py`; `runtime/driving/rover_drive_session.py`; `tests/isaac/smoke_s3_rover_drive.py` | S3 — **user requirement 2 is met here** |
| WP-12 | Wheel tracks in the runtime | WP-06, WP-11 | `runtime/tracks/terrain_mesh_updater.py`, `track_mask_texture.py`, `track_runtime.py`; `tests/isaac/smoke_s4_wheel_tracks.py` | S4 — **user requirement 5** |
| WP-13 | Rocks in the runtime | WP-12 | `runtime/stage_assembly/rock_stage_builder.py`; `tests/isaac/smoke_s5_rocks_and_labels.py` | S5 — **user requirements 3 and 4 visible** |
| WP-14 | Driving tools and window | WP-13 | `runtime/teleoperation/keyboard_twist_device.py`, `drive_loop.py`; `scripts/drive_demo.py`, `view_scene.py`; `tests/isaac/smoke_s6_window_loop_autodrive.py`; `docs/driving_guide.md` | S6; one `drive_demo.py` run with before/after PNG; the manual keyboard checklist in `docs/driving_guide.md` |
| WP-15 | Performance and verification report | WP-14 | `tests/isaac/smoke_s7_performance_ladder.py`; `configs/measured_budgets.yaml`; `docs/verification_report.md` | S7; verification report with every measured number; checklist for the first run on the RTX 5090 |

Plan: WP-01 first. Then WP-02 to WP-07 at the same time (six packages, no shared files, all coding against `core/data_model/`), while WP-09 uses the GPU. WP-08 integrates the bake. The GPU chain is WP-09 → WP-10 → WP-11 → WP-12 → WP-13 → WP-14 → WP-15.

Deviations found while implementing a package are reported in that package's hand-over and collected into `docs/verification_report.md` by WP-15.

### 12.2 Runtime probes of WP-09 (each has a fallback, so no answer blocks the design)

Probes run on synthetic meshes through `isaac_run.sh`, write to `<data root>/runs/probes/`, and need nothing but WP-01.

| Probe | Question | Fallback if the answer is no |
|---|---|---|
| RP1 | Are `normals` written at runtime (USD) re-rendered by RTX? | `tracks.author_normals: false` |
| RP2a | Is a USD `points.Set` visible to `rep.orchestrator.step` with no `app.update()` in between? | Implement `FabricMeshUpdater`; if that fails too, use the application-update stepper everywhere |
| RP2b | Under pure `simulate`/`fetch_results`: do articulation and contact views return fresh data after every step (checked against `get_rigidbody_transformation`)? Is a commanded wheel speed reached within 0.5 s? Does the sum of wheel normal forces equal the rover weight within 5 %? | Step with `SimulationManager.step`; otherwise the application-update stepper everywhere |
| RP2c | Do `omni.physx` physics-step events fire under explicit `simulate`? (information only: decides whether both steppers can share one callback mechanism) | Keep the direct callback call in the explicit stepper |
| RP3 | Do invisible collision meshes collide, and do they cost render memory? (`visibility = invisible` versus `purpose = guide`) | Visual tiles double as colliders (`collision_decimation: 1` at 5 cm) |
| RP4 | Do 144 visual tiles + 9 collision tiles + the ring mesh assemble and cook in ≤ 30 s, with every tile raycast-ready? | Fewer, larger tiles (4 × 4 tiles of 513² are VERIFIED) |
| RP5 | Does a capsule wheel rolling across collision tile seams and the workspace/near-ring boundary show a bump? | A single workspace collision mesh (2.1 M triangles in one mesh is VERIFIED); widen the drivable margin |
| RP6 | Rock instances. (a) Can the prototype container be abstract (`class` specifier) so prototypes neither render nor collide, while instanceable references inherit mesh, material, collider and labels? (b) With 2 000 instanceable references carrying a random `xformOp:orient` and a uniform `xformOp:scale` between 0.05 and 3: play time and resident memory; 200 raycasts from above hit within 5 mm of the analytically scaled prototype top; a ball rests on a scaled rock; the raycast and the semantic label identify each rock prim. | (a) Concrete prototype scope translated to z = −10 000 m. (b) First: quantise scale into 12 logarithmic size bins per prototype and bake the scale into prototype variants, so references carry translate and orient only. Second: author the registered rocks (a few hundred) as plain non-instanced meshes with baked vertices (under 0.5 M triangles). |
| RP7 | Rover articulation: is the passive front-axle pivot stable, and is the minimum wheel force ≥ 0.5 × mean? With the rover lifted, do the front wheels spin freely? Are contact forces between rover links zero at rest, and wheel net contact forces zero in free fall, with self-collisions disabled? | Rigid frame with documented diagonal loading |
| RP8 | Anti-aliasing mode 4 versus 0 at 640×480: image quality, milliseconds, video memory. Is Sun-intensity exposure scaling monotonic and repeatable? After a 6 cm camera move, what is the peak signal-to-noise ratio of a 1-, 2- and 4-subframe frame against a 32-subframe reference? | Mode 0; fixed intensity; `subframes_per_camera_frame` = the smallest value above 30 dB |
| RP9 | Does the extended MDL compile with world-metre `st`, two detail textures and a `dynamic://` mask as a `uniform texture_2d`? Which way up are a dynamic texture and a file texture (a PNG with one bright corner)? | `visual_mode: geometry_only`; a single detail texture; the two flip switches are set from the answer |
| RP10 | Render cost and video memory of 30 000 instanced pebbles with the MDL material; is a runtime write of `invisibleIds` rendered in the next frame, and what does it cost? | Raise `pebble_minimum_diameter_m`; `hide_pressed_pebbles: false` |
| RP11 | Cache map: size of every directory of section 3.4 before and after one cold terrain cook, without and with the redirects; do `${omni_cache}`, `${omni_data}`, `${omni_logs}` resolve under the data root? Is `extra_args` honoured for the UJITSO setting? | `--portable-root <data root>/caches/kit_portable`; for any directory that still cannot be moved: the free-space guard on `/` and a documented manual clean-up |
| RP12 | Window: does `SimulationApp` with `headless=False` start through `isaac_run.sh --interactive` on this machine; do carb keyboard events arrive through `omni.appwindow`; frame time and process video memory with a 1280 × 720 viewport on a 2 M-triangle mesh | Lower the viewport resolution; `smoke` preset only for the window; if no display is reachable from the build session, the window is checked by hand by the user with `docs/driving_guide.md` |

---

## 13. Risks and mitigations

| Risk | Mitigation |
|---|---|
| The data disk is not mounted (auto-mount, not in `/etc/fstab`) | `require_data_root` in every entry point; no fallback path; clear message; the root is created only by `setup_data_root.sh` after a mount check |
| `/` fills up (14 GB free, 97 % full) | All outputs, logs and pytest files on the data root; caches redirected (section 3.4) and mapped by probe RP11; free-space guard on `/` for anything that cannot be moved |
| Video memory contention with other Isaac sessions on the same GPU (seen at 2.5–3.6 GB) | `flock`; per-preset free-memory check before `SimulationApp` starts; `gpu_8gb` sized below the largest measured triangle count; CPU dynamics; configuration-only degradation ladder (section 6.3) |
| An UNVERIFIED runtime mechanism fails | Every one is probed in WP-09 before dependent code is written, and has a named fallback (sections 9.5 and 12.2) |
| The window path cannot be tested from the build session (no display) | The window loop is one function that also runs headless in S6; probe RP12; a manual checklist for the user |
| Rendered surface and collider disagree (tile order, flips) while numeric tests pass | S1 compares rendered depth with the visual lattice and the collider; S3 compares rendered wheel bottoms with rendered ground |
| Tracks pass a numeric test but are not visible | S4 uses a static observer camera, a noise floor, contrast-to-noise at low and high Sun, and runs geometry and mask channels separately |
| Per-instance scale on instanced rock colliders does not behave | Probe RP6 with two named fallbacks |
| `gpu_32gb` cannot be validated on this machine | The preset is data only and labelled "not run"; smoke test S7 is the first thing to run on the 5090; no code path assumes the larger sizes |
| Synthetic, uncalibrated soil gives invisible or absurd ruts | Narrowed property ranges, soil contrast limit, 4 mm visual floor, 30 mm cap; everything tagged SYNTHETIC |
| The wheel does not physically sit in its rut (collider stays on the rest surface) | Visual wheel radius includes the nominal sinkage; rut depth capped at 3 cm; documented limit; force-based contact is a "Later" item |
| Skid-steer turning on μ ≈ 0.8 is jerky or under-rotates | Calibrated `skid_steer_track_width_factor`; applied and achieved twist are both in the run log |
| Capsule wheel caps touch rocks beside the visible wheel | Measured in S5; named cylinder fallback (section 10.3) |
| The DEM and the craters alone exceed the roughness target | `minimum_fractal_gain` floor, `roughness_budget_exceeded` flag and all terms recorded in the manifest |
| `lunar_regolith` changes under us | One adapter file, commit warning, golden-value tests; only six small functions are used |
| Isaac start-up stalls | Stale shared-memory clean-up, 75 s relaunch watchdog, stdin closed |
| Physics is not bit-reproducible | CPU PhysX and a tolerance test (≤ 5 cm after 60 s); the bake, which is the reproducible part, is byte-identical |
| Silent PhysX failure (bodies frozen in place, rover falling through) | Hold-and-wait before release, settle-height and wheel-force checks, "rover did not move" abort, fall-through guard |

---

## 14. Later (documented direction; not built in version 1)

Nothing in this section is built, imported or required by version 1. Each item names the seam it plugs into.

### 14.1 The learning-facing layer

1. **`LunarRoverEnvironment(gymnasium.Env)`** in a new `runtime/environment/`. It wraps `RoverDriveSession` and adds nothing to the physics path: `reset(seed, options)` calls `session.reset(...)`, `step(action)` calls `session.advance_control_step(action, render_cameras=True)`. Single environment, numpy in and out, no automatic reset, one scene per process.
   - Action: `Box(2)` = `[forward_speed_m_s, yaw_rate_rad_s]`, the twist convention of section 4.1, limits from the rover specification.
   - Observation: `observation.images.<camera>` (uint8 RGB), `observation.state` (float32, 20 values: wheel-odometry twist, gravity direction in the body frame, gyroscope, accelerometer, four wheel speeds, previous action, Sun direction in the body frame), `observation.odom`, `task` (the instruction string). Global pose stays privileged in `info["ground_truth"]` together with the per-wheel contact, slip and sinkage that version 1 already produces.
   - `gymnasium` is already in `ISAAC_ENV`; it becomes an allowed import of `runtime/environment/` only.
2. **Episode recorder** in a new Isaac-free `core/recording/`. It subscribes with `RoverDriveSession.add_step_listener` and receives every `DriveStepResult`. Planned layout per episode: `steps.parquet` (one row per control step), one MP4 per camera piped to `/usr/bin/ffmpeg`, PNG ground-truth streams (16-bit depth, 8-bit semantic, 16-bit instance), `episode.json` written last. The layout maps one-to-one onto LeRobot v2.1, so the exporter (in its own environment on the data disk, never inside `ISAAC_ENV`) is a rename plus metadata aggregation. `pyarrow` and ffmpeg become dependencies only then. The version 1 `run_log.npz` and `wheel_contact_log.npz` already hold the state needed for "record state, render later" (`replay_wheel_contacts` is bit-exact).
3. **Tasks, success predicates, language instructions** in a new Isaac-free `core/tasks/`, reading only the manifest and rasters through a new `core/scene/scene_query.py` built on `LoadedScene`. Planned families: go to an object, park on level ground, follow tracks, return on own tracks. The manifest then gains `tags`, viewpoint-independent `relations` and `regions` (schema version bump; the registered objects, craters and ids of version 1 stay as they are). Track tasks use the existing `TrackField.save`/`load`, `centreline_from_poses` and `cross_track_error_m`; pre-laid and distractor tracks need one new function that lays samples along a polyline without physics.
4. **Path planning and a scripted expert**: A* on a traversability cost computed **at load time** from `slope_deg`, the rock table and `RoverSpecification.minimum_ground_clearance_m()` (not baked, so scenes stay rover-independent), tracked by pure pursuit. Both implement the `TwistSource` protocol, so they drive through the same session as the waypoint follower.
5. **Randomisation curricula** on the reserved `episode.*` and `sensor.*` streams: start pose (the `spawn:` block with a randomised preferred point), Sun (`set_sun`), exposure (`lighting.exposure_scale`), track albedo, sensor noise, action delay. Random site offsets with redraws on NoData or slope belong here too.
6. **More ground truth**: `instance_segmentation_fast` with ids remapped to manifest `instance_id` (the prim paths are already in `RockPrimHandles`); a merged semantic image that adds `crater`, `wheel_track` and `debris` by unprojecting depth and sampling `crater_id.npy` and the track mask with `core/perception/camera_projection.py` (class ids are reserved); shadow rasters per Sun direction. `CameraRig` already takes the annotator list as an argument.
7. **Isaac Lab `DirectRLEnv` adapter** and multi-rover vectorised training; ROS 2 bridge. The core needs nothing from Isaac Lab.

### 14.2 Simulation fidelity and content

8. **Wheel physically in the rut**: (a) compliant contact (`physxMaterial:compliantContactStiffness` per friction zone, calibrated per wheel collider; verified on the sibling's lander only); (b) rate-limited re-cook of small collision tiles with the same-frame `RemoveAPI(CollisionAPI)` + `Apply` trick (13–19 ms per 33²–65² tile, leaks 0.7 MiB each); (c) force-based heightfield terramechanics (Reece envelope with elastic/plastic node state and Janosi shear applied through `omni.physics.tensors`). `WheelContactSampleBuilder` is the seam where a measured wheel underside replaces the analytic sinkage; deeper ruts than 3 cm come with this item.
9. Ejecta block fields around fresh craters (non-physical for craters this small; useful as referable content); registration of DEM-resolved craters.
10. Soil maps exported from a real `lunar_regolith` run (through that project's own environment) instead of the in-process synthetic maps; `SoilPropertyMaps` is the seam.
11. Prop library (flag, sample tube, instrument box, marker cone) and the Apollo 17 lander from the sibling project (class ids 7 and 8 are reserved).
12. Sensor model: linear `HdrColor` through a numpy camera response (exposure, shot and read noise, vignetting, monochrome); stereo partner; hazard camera.
13. USD-asset rover adapter (Jackal and Leatherback load) through `RoverHandles` + `RoverSpecification` + `DriveModel`; Ackermann and rocker-bogie drive models; gamepad teleoperation.
14. Per-face friction-zone subsets; chevron grousers; slip-dependent churned-soil shading; dynamic (pushable) small rocks.
15. Orthophoto-derived albedo from the 60 cm LROC mosaic; triplanar rock shading; bump strength calibrated to the published 8° rms slope at 1 cm.
16. `FabricMeshUpdater` as a second mesh backend; a stage export script for viewing; in-process scene swap; a multi-process launcher for the 5090.
17. Rover-centred streaming terrain for traverses longer than the workspace; polar sites (needs a polar-stereographic frame).

---

## 15. Honest limits and references

**Limits of version 1**

- The wheel-terrain contact is rigid PhysX contact on the undeformed surface. Sinkage is an analytic depth used to carve the visual rut; it does not feed back into forces. There is no rolling resistance from compaction, no slip-sinkage digging beyond the `slip_gain` factor, no bulldozing. Rut depth is capped at 3 cm; a spinning wheel on soft soil can be seen hovering up to about 2 cm above its trench.
- The wheel collider is a capsule 0.34 m wide for a 0.12 m visible wheel. It can touch rocks up to about 11 cm beside the visible wheel.
- Berms are a fixed-fraction volume heuristic; there is no soil flow and no dust. Pebbles under a tread are hidden, not pushed.
- All soil properties are SYNTHETIC and uncalibrated; the property ranges and the contrast limit were chosen so that tracks are visible, not from measurements.
- Sub-DEM terrain is statistically plausible, not real: craters below 8 m and all rocks are procedural. The crater law is extrapolated below its measured range, and the fresh depth-to-diameter range is deeper than the published trend for small craters, chosen so craters stay visible.
- The roughness target applies to the plane-detrended full surface at 1 m lag. Where the DEM and the craters alone exceed it, the surface is rougher than the target and the manifest says so.
- Geometry carries wavelengths down to 0.2 m; finer relief is shading only, with uncalibrated amplitude. The collider and the rendered surface differ by a few millimetres by design.
- Tracks exist only inside the workspace. The regional and backdrop tiers have no collider; the boundary policy and the fall-through guard keep the rover away from them.
- The workspace and near ring are planar; only the far tiers carry the Moon's curvature.
- Exposure is controlled through Sun intensity with default RTX tonemapping; images are not radiometrically calibrated. (v1.0.9: the RTX tone-map keys are verified and used: linear operator 1, ISO metered to the Sun elevation; see docs/decision_log.md 2026-11-18.)
- The `gpu_32gb` preset is extrapolated and has not been run.
- Throughput numbers for the full loop are estimates until smoke test S7 runs. In the window, simulated speed is the frame rate divided by 60.

**Source material for implementers**

- Sibling terrain and launch code: `/home/noron/Desktop/MEO/apollo_lander_modeling/{nac_dtm.py, build_apollo17_scene.py, surface_hazards.py, regolith_detail.py, materials/lunar_regolith.mdl, isaac_run.sh, check_collision.py, test_lunar_material.py}`. Rename the short identifiers when adapting.
- Sibling soil code (numpy-only modules): `/home/noron/Desktop/MEO/lunar_regolith/src/lunar_regolith/{spatial_fields.py, correlated_fields.py, facies.py, contact/sinkage.py, contact/friction_zones.py}`.
- Keyboard input pattern: `/media/noron/DISK02/isaac-lab-native/IsaacLab/source/isaaclab/isaaclab/devices/keyboard/se2_keyboard.py` (carb keyboard events through `omni.appwindow`; copy the pattern, do not import Isaac Lab).
- Verified Isaac 6.0.1 probe scripts (mesh from numpy, tensor rover, contact views, dynamic textures, instanceable rocks, annotators) and the numeric checks behind the terrain recipe: `/tmp/claude-1000/-home-noron-Desktop-MEO-isaac-lunar-rover-sim/e3750ef3-f630-42ce-9f26-d9d2e42f1e61/scratchpad/` (`isaac_probes/`, `review_feasibility/`, `review_probe/`). The scratch directory can be cleared by a reboot; WP-09 copies what it reuses into `tests/isaac/probes/`.
- OmniLRS reference clone (BSD-3, deformation kernel and rock clustering ideas): `/tmp/claude-1000/-home-noron-Desktop-MEO-isaac-lunar-rover-sim/e3750ef3-f630-42ce-9f26-d9d2e42f1e61/scratchpad/omnilrs_src/OmniLRS`.
- Literature anchors used for constants: Minton et al. 2019 and Trask 1966 (crater equilibrium), Stopar et al. 2017 and Mahanti et al. 2018 (small-crater depth), Fassett and Thomson 2014 (diffusion), Cai and Fa 2020 and Rosenburg et al. 2011 (roughness and Hurst exponent), Helfenstein and Shepard 1999 (centimetre roughness), Bandfield et al. 2011 (rock abundance), Costes et al. 1972 (Apollo soil values and rover sinkage), Kaydash et al. 2011 (track photometry).
