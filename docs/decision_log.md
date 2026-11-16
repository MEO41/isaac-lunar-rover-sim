# Decision log

Deviations from `docs/DESIGN.md` and additions made while building, newest first.

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
