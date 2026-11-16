"""Typed configuration models.

The models carry no defaults on purpose: ``configs/defaults.yaml`` is the single source of default values, and
``extra="forbid"`` rejects unknown keys, so the YAML and the models cannot drift apart silently.
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict


class StrictModel(BaseModel):
    """Base for every configuration section: unknown keys are errors, instances are immutable."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class ProfileConfiguration(StrictModel):
    """Which site, rover and preset files were merged into this configuration."""

    site: str
    rover: str
    preset: str


class StorageConfiguration(StrictModel):
    """Data root location and the startup-check thresholds."""

    data_root: str
    allow_root_disk: bool
    minimum_free_gib: float


class LaunchConfiguration(StrictModel):
    """Guards applied before SimulationApp is created."""

    required_free_video_memory_mib: int


class SiteConfiguration(StrictModel):
    """DEM files and the projection constants of one landing site."""

    name: str
    dem_fine_path: str
    dem_coarse_path: str
    origin_longitude_deg: float
    origin_latitude_deg: float
    origin_offset_east_m: float
    origin_offset_north_m: float
    standard_parallel_deg: float
    central_longitude_deg: float
    body_radius_m: float
    dem_cutoff_wavelength_m: float


class NearRingConfiguration(StrictModel):
    """Near-ring tier around the workspace (rendered and collided)."""

    size_m: float
    spacing_m: float
    skirt_m: float


class RegionalConfiguration(StrictModel):
    """Regional tier (rendered only)."""

    enabled: bool
    size_m: float
    spacing_m: float
    skirt_m: float


class BackdropConfiguration(StrictModel):
    """Backdrop tier out to the horizon (rendered only)."""

    enabled: bool
    size_m: float
    spacing_m: float


class CraterConfiguration(StrictModel):
    """Small-crater population (SYNTHETIC)."""

    equilibrium_coefficient: float
    randomize_equilibrium_coefficient: bool
    equilibrium_coefficient_range: tuple[float, float]
    minimum_diameter_m: float
    near_ring_minimum_diameter_m: float
    taper_start_diameter_m: float
    fresh_depth_to_diameter_range: tuple[float, float]
    rim_height_to_diameter: float
    apron_extent_radii: float
    degradation_parameter_maximum: float
    fresh_degradation_threshold: float
    moderate_degradation_threshold: float
    ellipticity_minimum: float
    rim_harmonic_amplitude_maximum: float
    feature_crater_count: int
    feature_crater_diameter_range_m: tuple[float, float]


class FractalConfiguration(StrictModel):
    """Band-limited fractal relief below DEM resolution (SYNTHETIC)."""

    hurst_exponent: float
    structure_function_anchor_source: Literal["measured_from_dem", "configured"]
    structure_function_anchor_m: float
    structure_function_anchor_lag_m: float
    structure_function_anchor_valid_range_m: tuple[float, float]
    roughness_multiplier: float
    randomize_roughness_multiplier: bool
    roughness_multiplier_range: tuple[float, float]
    macro_band_wavelengths_m: tuple[float, float]
    meso_band_wavelengths_m: tuple[float, float]
    band_edge_relative_width: float
    minimum_fractal_gain: float
    maximum_fractal_gain: float


class TerrainConfiguration(StrictModel):
    """Terrain tiers, lattices and procedural detail."""

    workspace_size_m: float
    visual_spacing_m: float
    collision_decimation: int
    visual_tile_cells: int
    collision_tile_cells: int
    workspace_skirt_m: float
    drivable_margin_m: float
    apply_planetary_curvature: bool
    near_ring: NearRingConfiguration
    regional: RegionalConfiguration
    backdrop: BackdropConfiguration
    craters: CraterConfiguration
    fractal: FractalConfiguration


class KeepOutZone(StrictModel):
    """A disc kept free of rocks (for example around the preferred spawn point)."""

    center_east_m: float
    center_north_m: float
    radius_m: float


class RockAlbedoValues(StrictModel):
    """Normal albedo of the three rock albedo classes."""

    dark: float
    medium: float
    bright: float


class RockConfiguration(StrictModel):
    """Rock size-frequency law, placement and shapes (SYNTHETIC)."""

    pivot_diameter_m: float
    background_density_at_pivot_per_m2: float
    randomize_background_density: bool
    background_density_at_pivot_range_per_m2: tuple[float, float]
    background_exponent: float
    randomize_background_exponent: bool
    background_exponent_range: tuple[float, float]
    background_fade_diameters_m: tuple[float, float]
    pebble_minimum_diameter_m: float
    cobble_minimum_diameter_m: float
    registration_minimum_diameter_m: float
    maximum_diameter_m: float
    boulder_rock_abundance: float
    boulder_exponent: float
    near_ring_minimum_diameter_m: float
    clustering_field_correlation_length_m: float
    clustering_field_sigma: float
    thomas_cluster_fraction: float
    thomas_parent_density_per_m2: float
    thomas_child_sigma_m: float
    burial_fraction_range: tuple[float, float]
    tilt_sigma_deg: float
    maximum_slope_for_large_rocks_deg: float
    overlap_rejection_factor: float
    prototypes_per_shape_class: int
    fillet_minimum_diameter_m: float
    fillet_height_fraction_range: tuple[float, float]
    albedo_values: RockAlbedoValues
    keep_out_zones: list[KeepOutZone]


class FaciesFractions(StrictModel):
    """Areal fractions of the three soil facies."""

    firm: float
    loose: float
    soft: float


class DetailTextureConfiguration(StrictModel):
    """One tileable micro-relief texture."""

    size_px: int
    tile_size_m: float


class RegolithConfiguration(StrictModel):
    """Soil property maps and regolith appearance (SYNTHETIC, uncalibrated)."""

    property_map_spacing_m: float
    correlation_length_m: float
    facies_fractions: FaciesFractions
    friction_angle_range_deg: tuple[float, float]
    cohesion_range_pa: tuple[float, float]
    bulk_density_range_kg_m3: tuple[float, float]
    fresh_crater_compaction_factor: float
    soil_contrast_limit: tuple[float, float]
    interface_friction_ratio: float
    interface_friction_clip: tuple[float, float]
    dynamic_to_static_friction_ratio: float
    friction_zone_count: int
    base_normal_albedo: float
    randomize_base_normal_albedo: bool
    base_normal_albedo_range: tuple[float, float]
    albedo_mottling_amplitude: float
    fresh_ejecta_brightening: float
    macro_albedo_size_px: int
    detail_texture_coarse: DetailTextureConfiguration
    detail_texture_fine: DetailTextureConfiguration
    detail_hurst_exponent: float
    detail_texture_standard_deviation: float


class SinkageConfiguration(StrictModel):
    """Analytic rut depth (visual only; the collider wheel does not sink)."""

    exponent: float
    slip_gain: float
    multipass_gains: tuple[float, float, float, float]
    visual_floor_m: float
    maximum_m: float
    force_filter_time_constant_s: float


class TrackConfiguration(StrictModel):
    """Wheel-track deformation and its rendering."""

    enabled: bool
    visual_mode: Literal["geometry_and_mask", "geometry_only", "mask_only"]
    author_normals: bool
    sample_spacing_m: float
    contact_force_threshold_n: float
    contact_gap_threshold_m: float
    flank_repose_angle_deg: float
    berm_volume_fraction: float
    berm_width_wheel_fraction: float
    pass_separation_time_s: float
    mask_upload_minimum_interval_s: float
    track_albedo_factor: float
    tread_bump_reduction: float


class CameraMountConfiguration(StrictModel):
    """One body-mounted camera; pitch positive = looking down, yaw counter-clockwise from body x."""

    enabled: bool
    position_body_m: tuple[float, float, float]
    pitch_down_deg: float
    yaw_deg: float
    horizontal_field_of_view_deg: float


class RoverCamerasConfiguration(StrictModel):
    """The rover's camera mounts."""

    front: CameraMountConfiguration
    rear: CameraMountConfiguration
    follow: CameraMountConfiguration


class RoverConfiguration(StrictModel):
    """Procedural skid-steer rover specification (body frame: x forward, y left, z up)."""

    name: str
    chassis_size_m: tuple[float, float, float]
    chassis_mass_kg: float
    front_axle_mass_kg: float
    wheel_mass_kg: float
    wheel_radius_m: float
    wheel_width_m: float
    wheelbase_m: float
    track_width_m: float
    ground_clearance_m: float
    wheel_collider_capsule_height_m: float
    grouser_count: int
    grouser_height_m: float
    nominal_sinkage_m: float
    front_axle_pivot_enabled: bool
    front_axle_pivot_limit_deg: float
    front_axle_pivot_damping_n_m_s_per_rad: float
    wheel_drive_damping_n_m_s_per_deg: float
    wheel_drive_maximum_torque_n_m: float
    maximum_forward_speed_m_s: float
    maximum_yaw_rate_rad_s: float
    maximum_forward_acceleration_m_s2: float
    maximum_yaw_acceleration_rad_s2: float
    skid_steer_track_width_factor: float
    maximum_wheel_rim_speed_m_s: float
    yaw_rate_proportional_gain: float
    yaw_rate_integral_gain: float
    yaw_rate_integral_limit_rad_s: float
    cameras: RoverCamerasConfiguration


class PhysicsConfiguration(StrictModel):
    """PhysX scene settings."""

    gravity_m_s2: float
    physics_rate_hz: int
    solver_type: Literal["TGS", "PGS"]
    gpu_dynamics_enabled: bool
    broadphase_type: str
    articulation_position_iterations: int
    articulation_velocity_iterations: int
    wheel_friction: float
    rock_friction_static: float
    rock_friction_dynamic: float


class SpawnConfiguration(StrictModel):
    """Deterministic safe start pose search."""

    preferred_east_m: float
    preferred_north_m: float
    heading_deg: float
    search_radius_m: float
    maximum_slope_deg: float
    maximum_roughness_m: float
    crater_minimum_diameter_m: float
    crater_clearance_m: float
    footprint_radius_m: float


class DriveConfiguration(StrictModel):
    """Control loop settings."""

    control_rate_hz: int
    settle_physics_steps: int
    collider_wait_timeout_s: float
    boundary_policy: Literal["stop_outward", "none"]
    fall_through_depth_m: float
    fall_through_policy: Literal["abort", "recover"]


class LightingConfiguration(StrictModel):
    """Sun (azimuth = compass bearing towards the Sun, 0 = north, 90 = east)."""

    sun_azimuth_deg: float
    sun_elevation_deg: float
    sun_angular_diameter_deg: float
    sun_color: tuple[float, float, float]
    sun_base_intensity: float
    exposure_scale: float


class SunGlareConfiguration(StrictModel):
    """Camera glare (RTX bloom / lens flare post-process): what a real lens does when the Sun is in or near the frame.

    Physically the Sun disc is only 0.53 deg wide; the large bright blob and star streaks in Apollo / Chang'e photos
    come from the camera (sensor saturation, diffraction at the aperture blades), not from the Sun or an atmosphere."""

    enabled: bool
    scale: float
    cutoff: float
    cutoff_fuzziness: float
    aperture_blades: int
    f_stop: float
    spectral_samples: int


class RenderConfiguration(StrictModel):
    """RTX render and camera settings."""

    sun_glare: SunGlareConfiguration
    renderer: Literal["RealTimePathTracing", "RayTracedLighting", "PathTracing"]
    camera_resolution_px: tuple[int, int]
    anti_aliasing_mode: int
    subframes_per_camera_frame: int
    warmup_subframes_after_reset: int
    camera_near_clip_m: float
    camera_far_clip_m: float
    camera_horizontal_aperture_mm: float


class WindowConfiguration(StrictModel):
    """Settings of the keyboard window launcher."""

    viewport_resolution_px: tuple[int, int]
    onboard_cameras_enabled: bool
    keyboard_forward_speed_m_s: float
    keyboard_yaw_rate_rad_s: float


class AppearanceConfiguration(StrictModel):
    """Which regolith look the renderer uses (runtime only; physics never changes).

    native                  this project's lunar_surface.mdl + baked textures; live wheel tracks with mask shading.
    lunar_regolith_overlay  lunar_regolith field-driven material bound to this project's terrain meshes; live wheel
                            ruts (geometry) still work, the track mask shading does not (the upstream MDL has none).
    lunar_regolith_scene    lunar_regolith render scene over the workspace (micro-relief, clods, static imprints);
                            this project's workspace render tiles are hidden, live ruts are not drawn.
    Both lunar_regolith modes need scripts/build_regolith_appearance.py first (SYNTHETIC soil, ASSUMED mapping)."""

    source: Literal["native", "lunar_regolith_overlay", "lunar_regolith_scene"]
    quality: Literal["realtime", "balanced", "offline"]
    tag: str


class SimulatorConfiguration(StrictModel):
    """The whole resolved configuration."""

    generator_version: str
    scene_seed: int
    profile: ProfileConfiguration
    storage: StorageConfiguration
    launch: LaunchConfiguration
    site: SiteConfiguration
    terrain: TerrainConfiguration
    rocks: RockConfiguration
    regolith: RegolithConfiguration
    sinkage: SinkageConfiguration
    tracks: TrackConfiguration
    rover: RoverConfiguration
    physics: PhysicsConfiguration
    spawn: SpawnConfiguration
    drive: DriveConfiguration
    lighting: LightingConfiguration
    render: RenderConfiguration
    window: WindowConfiguration
    appearance: AppearanceConfiguration
