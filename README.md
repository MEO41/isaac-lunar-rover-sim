# isaac-lunar-rover-sim

A lunar rover simulator for NVIDIA Isaac Sim 6.0.1, built as the environment for later vision-language-action (VLA)
research. Version 1 is the simulation environment only; the learning-facing layer (gym wrapper, episode recorder,
tasks, language) comes once the environment is trusted. Design: [docs/DESIGN.md](docs/DESIGN.md).

## Demo

8 s drive on the Apollo 17 valley floor (preset `gpu_8gb`, Sun 15° above the eastern horizon), rendered headless with
`scripts/render_drive_video.py`. Click a preview for the full-quality MP4.

| Chase camera | Rover front camera (what a policy would see) |
|---|---|
| [![chase camera](docs/videos/rover_drive_chase.gif)](docs/videos/rover_drive_chase.mp4) | [![front camera](docs/videos/rover_drive_front_camera.gif)](docs/videos/rover_drive_front_camera.mp4) |

Watch for the wheel tracks with grouser imprints forming behind the rover during the left turn, rocks casting long
shadows, and the North and South Massifs (real DTM heights) on the horizon.

![sun glare check](docs/images/sun_glare_check.png)

*Camera glare (`render.sun_glare`): glare off, then bloom scale 0.15 / 0.3 / 0.6 (default). Only the Sun blooms; the
terrain brightness is unchanged.*

## Regolith appearance from lunar_regolith

The regolith look can come from the field-driven appearance module of
[lunar_regolith](https://github.com/MEO41/lunar_regolith) (pinned submodule `external/lunar_regolith`). There, the
rendered surface is a function of SYNTHETIC soil fields (density, compaction, cohesion, grain size, thickness, facies)
through an ASSUMED mapping (`config/appearance.yaml`); it is not calibrated against Apollo photographs.

![regolith appearance comparison](docs/images/regolith_appearance_comparison.png)

*Same drive, same views. Rows: native look, lunar_regolith overlay, lunar_regolith render scene. Columns: oblique,
close, grazing. Physics (colliders, rover, rocks) is identical in all three.*

| `appearance.source` | What renders | Wheel ruts |
|---|---|---|
| `native` (default) | this project's `lunar_surface.mdl` + baked textures | live: mesh deformation with grouser imprint + track-mask shading |
| `lunar_regolith_overlay` | lunar_regolith material bound to this project's terrain meshes | live mesh deformation (no mask shading: the upstream MDL has none) |
| `lunar_regolith_scene` | lunar_regolith render scene over the workspace: micro-relief, clods, imprint patch | static: a drive's contact log turned into `SurfaceImprint` plates |

```bash
# needs REGOLITH_REPOSITORY, REGOLITH_PYTHON, REGOLITH_RUN in .env.local (see .env.example); git submodule update --init
scripts/regolith_run.sh scripts/build_regolith_appearance.py --preset gpu_8gb --mode overlay --quality realtime
scripts/isaac_run.sh scripts/render_appearance_views.py --label native --save-contact-log --output-directory DIR
scripts/regolith_run.sh scripts/build_regolith_appearance.py --preset gpu_8gb --mode scene --quality balanced \
    --contact-log DIR/contact_log.npz --tag drive
scripts/isaac_run.sh --interactive scripts/view_scene.py --preset gpu_8gb --set appearance.source=lunar_regolith_overlay
```

How it is wired:
- The builder runs with `REGOLITH_PYTHON` (rasterio, pyproj), never Isaac's Python; the code comes from the pinned
  submodule, the data from the working copy's `REGOLITH_RUN`. Both commits go into `provenance.json`, which every
  result JSON copies.
- The render surface handed to lunar_regolith is this simulator's 5 cm workspace **collision lattice**
  (`SceneSpec.surface`), so its render meshes refine exactly the surface PhysX collides with. Render meshes carry no
  collision API; terrain colliders, rocks and the rover are unchanged.
- Upstream render-only rocks and micro-craters are switched off (this simulator has collided rocks and a physical
  crater population; render-only ones would be ghosts the wheels pass through); dust is off.
- Each appearance renders in its own Isaac process (Isaac Sim 6.0.1 runs out of RTX resource descriptors otherwise).
- Known limit of static imprints: ruts are chains of circular plates (faint rings where plate depths differ, no
  grouser bars) and appear all at once; time-varying ruts are not implemented upstream yet.

## What it does

- **Terrain from LROC NAC DTMs** (Apollo 17, 2 m + 5 m mosaic) in a true-metre local frame, nested tiers out to the
  horizon (2.5 cm workspace, 0.2 m near ring, 4 m regional, 64 m backdrop) with exact triangle-mesh colliders.
- **Procedural sub-DEM relief** (SYNTHETIC): equilibrium crater population with diffusion degradation and
  band-limited fractal relief, calibrated to the DEM's own roughness; the DEM's resolved band is left untouched.
- **Rocks** (SYNTHETIC): procedural fractured-ellipsoid shapes, blended background + boulder-tail size-frequency law,
  clustering, burial and regolith fillets; pebbles to boulders, convex-hull colliders, semantic labels, registry in
  the scene manifest.
- **Regolith**: back-scattering lunar MDL, macro albedo and micro-relief textures, SYNTHETIC soil property maps.
- **Wheel tracks**: ruts with grouser imprints and berms carved into the rendered mesh, depth from Bekker sinkage on
  the soil maps, plus a track mask that changes albedo and smoothness. Ruts are visual; the collider is not deformed.
- **Rover**: procedural 55 kg four-wheel skid-steer with a passive front-axle pivot, closed-loop yaw rate, cameras,
  lunar gravity, camera glare (bloom) for the Sun.

## Requirements

Linux, an NVIDIA RTX GPU (verified on an 8 GB RTX 5060), Isaac Sim 6.0.1 pip install, the LROC NAC Apollo 17 DTMs.

```bash
cp .env.example .env.local && $EDITOR .env.local      # ISAAC_ENV, data root on a large disk, DEM directory
scripts/setup_data_root.sh                            # creates the data root (refuses the root disk)
$ISAAC_ENV/bin/python scripts/bake_scene.py --preset gpu_8gb     # Isaac-free bake (~40 s) -> <data root>/scenes/
```

## Run

```bash
scripts/isaac_run.sh --interactive scripts/view_scene.py --preset gpu_8gb   # drive with W/A/S/D (X stop, R reset, C camera)
scripts/isaac_run.sh scripts/render_drive_video.py --preset gpu_8gb         # chase + front-camera MP4
scripts/isaac_run.sh tests/isaac/smoke_terrain_drive.py --preset gpu_8gb    # terrain, colliders, rocks, wheels follow ground
scripts/isaac_run.sh tests/isaac/smoke_wheel_tracks.py --preset gpu_8gb     # tracks visible in renders
scripts/isaac_run.sh scripts/render_sun_glare_check.py                      # camera glare comparison PNG
```

Presets: `smoke` (small, fast), `gpu_8gb` (default), `gpu_32gb` (RTX 5090; extrapolated, not yet run).

## Honest limits

All sub-DEM detail, rocks and soil properties are synthetic and uncalibrated. Wheels do not physically sink into
their ruts. Skid steering scrubs heavily, so yaw rate is closed-loop while forward speed in arcs is open-loop.
