# Decision log

Deviations from `docs/DESIGN.md` and additions made while building, newest first.

## 2026-11-17: v1.0.7, parallel Isaac processes on minsky

User decision: the "one Isaac process at a time" rule came from the 8 GB RTX 5060; on minsky (RTX 5090, 32 GB) several
processes may run. `scripts/isaac_run.sh` takes the first free of `ISAAC_PARALLEL_SLOTS` lock files
(`runs/isaac.lock`, `runs/isaac.lock.1`, ...; default 1 = unchanged behaviour); the free-video-memory floor still
guards every start. Measured (minsky, VLA collection runner, scripted expert, 3 scenes at once): each process 7.3-7.6
control steps/s once all were running (one 3.9 while the others started), together about 19 control steps/s against
13.7 for one process alone, i.e. about 1.4x; each process keeps about 11 of the 32 CPU cores busy, so the CPU, not
the GPU (about 5 GB per process), limits further gains. All 6 test episodes reached the goal.

## 2026-11-16 (night, later): v1.0.6, three more landing sites

Requested for the VLA layer (test the rover beyond the Apollo 17 valley). Config only, no code change.
- **Sites** (`configs/sites/`): `apollo11_landing_site` (Mare Tranquillitatis, flat mare), `apollo15_landing_site`
  (Hadley mare plain), `apollo16_landing_site` (Descartes highlands). LROC NAC DTMs NAC_DTM_APOLLO11 / 15 / 16
  (2 m/post, equirectangular; standard parallels 1, 26, -9 deg from the PDS labels), downloaded from
  pds.lroc.im-ldi.com into `lunar_regolith/data/nac_dem` (not in git). The same DTM serves as fine and coarse source.
  Origins: the lunar-module positions (LROC-derived, ASSUMPTION: approximate to a few metres), except Apollo 15:
  130 m south-southwest of the LM, where the ground is flatter (40 m box: mean slope 1.9, max 4.7 deg; at the LM
  6.0 / 13.0 deg). Origin offsets are fixed at 0 in version 1, so the origin coordinates themselves were moved.
- **Scenes** (`configs/scenes/apolloNN_lunar_module_area.yaml`): backdrop tier off (a single NAC DTM, 4-25 km
  wide, cannot hold the 38.4 km backdrop; the 2 km regional tier reaches past the camera horizon, ~1.6 km); Apollo 15
  and 16 search 8 m for a spawn pose (seeds 7-10 at Apollo 15 and seed 31 at Apollo 16 have none within 4 m).
- **Checked** (minsky, gpu_8gb, scripted expert through the VLA environment, Sun 25 deg): goal reached on Apollo 11
  seed 23 (357 steps), Apollo 15 seeds 29 and 7 (415, 410), Apollo 16 seeds 31 and 7 (413, 335), nearest hazard
  0.11-1.16 m. Bake 30-34 s per scene.
- **Note**: the sub-DEM synthetic content (craters, rocks) depends on the scene seed only, so the same seed gives the
  same rock and crater layout at every site (seed 7: 6343 pebbles, 776 cobbles, 182 small, 24 medium rocks at all
  four sites). Use different seeds when sites must not share layouts.

## 2026-11-16 (night): v1.0.5, the expert keeps clear of crater bowls

Found by the VLA layer (lunar-rover-vla, `tests/isaac/drive_expert_loop.py`, gpu_8gb, minsky RTX 5090): in 1 of 12
PhysX episodes the expert's rover centre went 1 cm inside the rim crest of crater 2222 (2.6 m wide). Cause: bowls of
obstacle craters were impassable only at the 0.2 m planner posts and not grown, and the DWA checked rocks only, so a
corner cut could graze a bowl. Change:
- **Planner map**: `CraterObstacles` (rows of craters >= `crater_obstacle_minimum_diameter_m`, table rows kept,
  `bowl_edge_distance_m` = (r' - 1) x nominal radius, an APPROXIMATION that is exact on the rim crest). Bowls are grown
  by the same hard clearance as rock discs (rover half width + side clearance = 0.47 m) and get the same
  preferred-clearance ring. `PlannerMap.crater_obstacles` (new field, default empty).
- **DWA**: `crater_bowl_entered` refuses arcs that bring the rover centre within `obstacle_safety_m` of a bowl; the
  expert passes `PlannerMap.crater_obstacles`. `DynamicWindowPlanner.command(..., craters=None)` keeps the old
  behaviour when no craters are given.
- **Tests**: 4 new unit tests (22 total, all pass): distance equals the stamping formula with rim harmonics and
  ellipticity, route keeps the hard clearance from a bowl, DWA refuses a bowl, kinematic expert through a crater field
  never enters a bowl.
- **Measured** (scene s7, 100 sampled goals from the spawn, kinematic lagged unicycle, contact = centre inside a bowl,
  rock footprint overlap): v1.0.4 87 reached / 4 bowl contacts (all crater 2222) / 9 aborted, closest approach -0.002 m;
  v1.0.5 90 reached / 0 contacts / 10 aborted, closest approach 0.62 m. Every abort in v1.0.5 is "stalled 4 times in
  a row" 0.30-0.36 m from the goal (just outside `stop_tolerance_m` 0.3 m, final parking), not a hazard. PhysX through
  the VLA environment: 19 of 19 completed episodes reached the goal (1.0 m radius), 0 hazard contacts, nearest hazard
  0.31 m (a rock), nearest crater bowl 0.74 m. tests/isaac/smoke_scripted_expert.py: 3/3 goals reached, rock gap
  >= 1.05 m, tilt <= 9.6 deg, 64 ms per control step with the camera; its `never_stopped_at_boundary` check failed
  because the changed planner map changed the sampled goals and goal 1 was drawn exactly on the drivable edge
  (east = -16.0 m): turning from there towards goal 2 swung the centre to -16.1 m. Goals on the edge were always
  possible; the VLA layer samples goals at least 1.5 m inside the square.

## 2026-11-16 (evening): cast shadows switch, exact resets

- **`set_cast_shadows(stage, enabled)`** (`scene_stage_builder`): `primvars:doNotCastShadows` on every terrain, rock
  and rover prim. `tests/isaac/check_shadows_off.py` (gpu_8gb, Sun 90/15 deg, front/rear/observer), judged with the
  ray-cast shadow_cause image: 78.5 / 99.0 / 89.7 % of cast-shadow pixels brighten by > 0.05 luminance; sunlit pixels
  change 0.003-0.005 mean absolute luminance (render noise floor 0.0014-0.0016: bounce light from ground that is no
  longer shaded); faces-away pixels stay dark; after switching back the image is within 2-5x the noise floor. The
  Sun light's own switch (UsdLux `inputs:shadow:enable`) brightens 0.03-0.09 % and `/rtx/shadows/enabled` 3-16 % of
  the cast pixels with RealTimePathTracing, so neither is used.
- **Reset modes** (`drive.reset_mode`, `RoverDriveSession.reset(mode=...)`): the old reset (now `teleport`) moved the
  root only; wheel joint angles (5.5 rad apart) and the solver's contact state carried over, and two runs of the same
  150-step command sequence differed by up to 9.6 cm. `full_state` also restores every joint angle and velocity;
  `timeline_restart` stops and plays the timeline (PhysX rebuilds its scene from the stage), rebuilds the tensor
  views, waits for the colliders, then places the rover. `tests/isaac/check_determinism.py`, two processes with the
  modes in different orders: full_state and timeline_restart are bit-identical within a process, to each other and
  across an Isaac restart (pose, wheel angles, track field, masks, depth all 0.0 difference); teleport is
  reproducible across processes as a sequence but each episode depends on the previous ones. One earlier run of
  full_state after teleport episodes ended 0.2 mm off, so the default is `timeline_restart` (0.63-0.68 s per reset;
  full_state 0.21 s). Rendered RGB is not bit-identical even with identical physics (RTX sampling and temporal
  history): mean 1.2-1.8 / 255 between runs, 0.2 / 255 between two renders of the same state. The expert smoke test
  reproduced its earlier numbers exactly with the new default.

## 2026-11-16 (later): pinned exposure, shadow cause, runtime Sun, rover-class check

Requested by the VLA-layer handoff (`lunar-rover-vla/handoff_vla_layer/docs/11_reconciliation_with_simulator.md`).

- **Exposure pinned** (`render.exposure`, `application.apply_exposure`): auto exposure (`/rtx/post/histogram/enabled`)
  off, tonemapper op 6, ISO 100, shutter 1/50 s, f/5, compensation 0, white scale 40.2, written at start-up and read
  back (raises if a value does not stick); `exposure_snapshot()` for run metadata. These are the Isaac Sim 6.0.1
  defaults read on 2026-11-16, so images are unchanged. Probe (gpu_8gb, observer camera, Sun intensity 1 / 0.5 /
  0.25 x): mean frame luminance 0.469 / 0.333 / 0.193 with auto exposure off, 0.478 / 0.375 / 0.277 with it on, i.e.
  the default already follows the Sun intensity and auto exposure would partly compensate it.
- **Shadow cause image** (`SHADOW_CAUSE_CLASSES`, `GroundTruthImages.shadow_cause`; not in MARE): sky, sunlit,
  cast_by_scene, cast_by_rover, faces_away. faces_away wins (such pixels stay dark with cast shadows off), then
  cast_by_rover, then cast_by_scene; classes 2-3 are exactly the pixels a cast-shadows-off render would light. The
  MARE shadow label is unchanged. Measured at Sun 90 / 15 deg (front camera): 7,690 cast_by_scene and 19,181
  faces_away pixels, so about 70 % of the dark ground in that view is self-shading that a shadows-off render cannot
  remove.
- **Runtime Sun** (`scene_stage_builder.set_sun_direction`, `sun_direction_from_stage`): re-poses `/World/Sun`, raises
  outside [-90, 90] deg elevation; the mask renderer now reads the Sun direction from the stage by default. Check
  (`--runtime-sun 200 30`): read-back vs `sun_direction_w` 0.0 deg apart; masks follow it (depth p95 1.95-2.44 mm,
  classes 99.98-100 %, shadowed darker than sunlit with probability 0.976-0.994).
- **Rover class checked**: the mask check now adds an observer camera that sees the rover: 13,233 rover pixels in the
  RTX semantic annotator, 100 % labelled rover by the ray-cast mask (both Sun settings); depth p95 0.49 mm in that view.

## 2026-11-16: scripted expert and ground-truth masks, after MARE

Source: MARE lunar rover simulator (Castilla-Arquillo and Olivares-Mendez, arXiv:2610.28465, MIT licence). Designs
were read from github.com/raulcastar/GNC-Moon-Simulator (`component_path_planner`) and
github.com/raulcastar/Unreal-Moon-Terrain-Generation (`docs/masks.md`, `config/mask_classes.json`). The code here is a
re-implementation; no MARE source file is copied. MARE itself was not adopted as the simulator (Unreal editor + ROS
loop, Earth gravity, decal tracks, no Python reset/step API; see the comparison in the session notes).

### Scripted expert (`core/planning/`)

| Part | MARE design kept | Changed here, and why |
|---|---|---|
| Cost map | DyMu slope cost table (10/12/15/20/40 at 0/5/10/15/20 deg, 50 beyond), 15 deg impassable threshold, 3 x 3 smoothing, 4-connected regions | Slope over the wheelbase (0.7 m) on the 5 cm collision lattice, maximum per 0.2 m planner cell. Rocks and craters come from the scene tables (privileged), not lidar: a rock is an obstacle when its exposed height (top of the placed collider mesh above the ground) exceeds 0.10 m (ASSUMPTION, wheel radius 0.15 m); bowls of craters >= 1.5 m are impassable (ASSUMPTION). |
| Global route | Fast marching (Eikonal, first-order upwind, 4-connected) from the goal; gradient descent tracing with a cell-walk fallback; goal snapping to the nearest drivable cell of the rover's region; clearance margin around obstacles, dropped when no route keeps it | Binary heap with lazy deletion instead of a sorted list; 8-neighbour fallback walk without corner cutting; hard clearance = half the rover width + 5 cm, preferred margin 0.4 m. |
| Local planner | Dynamic Window Approach: sample the reachable (v, w) window, refuse trajectories bringing the footprint within a safety distance of an obstacle, heading + speed cost, braking-distance speed limit, yaw damping on final approach | Vectorised over all samples; obstacles are rock discs (not lidar points); the damped approach command is kept only if it is also collision-free; velocity window from the last command. |
| Mission | Tilt abort, stop tolerance, stall detection (motion or turn below minima over a window) with re-planning and abort after N stalls | No lidar re-planning (obstacles are known); re-plan when more than 1 m off the route. |

Validated: 12 planning unit tests (fast marching accuracy, routes around walls and rocks, clearance fallback, goal
snapping, DWA collision refusal, closed-loop drive of a lagged unicycle). On the gpu_8gb scene without Isaac
(`scripts/plan_expert_routes.py`): 6/6 sampled goals (8-17 m routes) reached by the kinematic model, no footprint
contact; planner map 1.3 s, a plan about 20 ms, 0.26 ms per control step. PhysX
(`tests/isaac/smoke_scripted_expert.py`, gpu_8gb, seed 0, 2026-11-16): 3/3 chained goals reached (10.4-12.8 m
routes, final distance 0.29-0.30 m, stop tolerance 0.3 m), minimum footprint gap to obstacle rocks 1.12 m, maximum
tilt 11.2 deg, no stall, one plan per goal. Throughput: 3.0-3.2x real time headless with tracks and no camera;
1.17x real time (86 ms per 0.1 s control step) with the 640 x 480 front camera rendered every step (2 subframes).

### Ground-truth masks (`core/ground_truth/`, `runtime/sensors/ground_truth_masks.py`)

| Part | MARE design kept | Changed here, and why |
|---|---|---|
| Object classes | Ids and colours of `mask_classes.json`: sky, rover, sun, sun_glare, landscape, crater, crater_rim, wheel_track, boulder, small_rock; crater = inside the rim crest, rim = ring 0.2 x radius wide outside it; tracks win over craters | `landscape` is called `terrain`. Boulder threshold 0.3 m diameter (ASSUMPTION, rover scale; MARE 1.0 m). Crater shape from the stamping's elliptical radius with rim harmonics. |
| Shadow label | MARE `shadow_label` ids: sky, sunlit/shadowed ground, sunlit/shadowed object, rover_shadow; surfaces facing away from the Sun are shadow; the rover's shadow wins over other shadows; the rover itself is never rover_shadow | Computed by ray casting (camera rays, then shadow rays to the Sun) with NVIDIA Warp against the exact render geometry, instead of a sun-only render pass and a rover depth map. Deterministic, no render noise, and no extra render. Uses flat-face normals (the renderer shades with smooth normals: grazing pixels can differ). |
| Extra | | Rock-instance image (rock table row + 1 per pixel) and image-plane depth, for per-rock attribution. |
| Glare radius | 1.1 deg (MARE navcam) | Kept as ASSUMPTION; not measured on this camera. |

In `lunar_regolith_scene` appearance mode the rendered workspace is the upstream render mesh, so the masks there are
an APPROXIMATION (recorded in the mask metadata). Validated: 6 ground-truth unit tests including the Warp kernels on a
toy scene. Renderer comparison (`tests/isaac/check_ground_truth_masks.py`, gpu_8gb, Sun azimuth 90 deg, elevation
15 deg, 2026-11-16; figure `docs/images/ground_truth_masks.png`), front and rear camera after an 8 s drive:
depth vs the RTX distance_to_image_plane annotator median error 0.0 mm, 95th percentile 1.95 mm, 100 % of pixels within
1 % + 5 mm, no pixel where one sees a surface and the other sky; classes vs the RTX semantic annotator 99.98-100 %
(rock, regolith, sky; the rover was in neither view, so the rover class is not yet checked against the renderer);
ground pixels labelled shadowed are darker than sunlit ones with probability 0.996 (front) and 0.997 (rear), median
luminance 0.00 vs 0.43-0.53. Cost: Warp meshes 9.9 M triangles built in 2.9 s; 28-30 ms per 640 x 480 frame.

Two fixes found by this comparison: (1) camera rays start at the camera's near-clip plane like the renderer's (the
front camera sits inside the camera-head box, which the 5 cm near clip removes); (2) the BVHs use the SAH builder:
with Warp's LBVH builder one frame took 23 s on this scene (triangle sizes from 64 m to 1 cm), with SAH 3 ms of kernel
time.
