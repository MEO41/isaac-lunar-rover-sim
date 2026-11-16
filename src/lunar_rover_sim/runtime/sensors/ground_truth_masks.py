"""Ground-truth masks of a camera by GPU ray casting (NVIDIA Warp) against the scene's render geometry.

For every pixel a camera ray finds the closest surface among three Warp meshes (static scene: outer terrain tiers +
rocks; workspace terrain, deformed by the live track field; the rover, re-posed from the articulation links every
frame). From the hit, two shadow rays go towards the Sun (terrain/rocks, rover). ``core.ground_truth`` turns the
result into MARE-compatible object and shadow labels plus a rock-instance image. No render pass is needed: masks are
deterministic and independent of the renderer's sampling noise; tests/isaac/check_ground_truth_masks.py compares them
with the renderer's own depth and semantic segmentation.

Camera model: the USD pinhole camera (focal length, apertures, no aperture offsets) through pixel centres.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import warp as wp

from lunar_rover_sim.core.frames.pose_math import rotation_matrix_from_quaternion, sun_direction_w
from lunar_rover_sim.core.ground_truth.mask_classes import OBJECT_ID
from lunar_rover_sim.core.ground_truth.mask_composition import GroundTruthImages, RayCastResult, compose_ground_truth
from lunar_rover_sim.core.ground_truth.scene_geometry import (OBJECT_KIND_ROVER, OBJECT_KIND_TERRAIN, TriangleSoup,
                                                              crater_class_raster, lattice_soup, nearest_lattice_value,
                                                              static_scene_soup)

MESH_STATIC, MESH_WORKSPACE, MESH_ROVER = 0, 1, 2
MAXIMUM_RAY_DISTANCE_M = 1.0e6


@wp.kernel
def _camera_ray_kernel(origin: wp.vec3, directions: wp.array(dtype=wp.vec3), forward: wp.vec3, near_clip: float,
                       static_mesh: wp.uint64, workspace_mesh: wp.uint64, rover_mesh: wp.uint64,
                       maximum_distance: float, hit_mesh: wp.array(dtype=wp.int32),
                       hit_face: wp.array(dtype=wp.int32), hit_distance: wp.array(dtype=wp.float32),
                       hit_normal: wp.array(dtype=wp.vec3)):
    """Closest hit of each camera ray beyond the near-clip plane among the three meshes (mesh -1 = no hit).

    Rays start on the near-clip plane like the renderer's, so geometry around the lens (the camera head box) is not
    seen; distances are still measured from the camera origin."""
    pixel = wp.tid()
    direction = directions[pixel]
    near_distance = near_clip / wp.max(wp.dot(direction, forward), 1.0e-3)
    start = origin + direction * near_distance
    best_distance = maximum_distance
    best_mesh = int(-1)
    best_face = int(-1)
    best_normal = wp.vec3(0.0, 0.0, 0.0)
    query = wp.mesh_query_ray(static_mesh, start, direction, best_distance)
    if query.result and query.t < best_distance:
        best_distance = query.t
        best_mesh = 0
        best_face = query.face
        best_normal = query.normal
    query = wp.mesh_query_ray(workspace_mesh, start, direction, best_distance)
    if query.result and query.t < best_distance:
        best_distance = query.t
        best_mesh = 1
        best_face = query.face
        best_normal = query.normal
    query = wp.mesh_query_ray(rover_mesh, start, direction, best_distance)
    if query.result and query.t < best_distance:
        best_distance = query.t
        best_mesh = 2
        best_face = query.face
        best_normal = query.normal
    hit_mesh[pixel] = best_mesh
    hit_face[pixel] = best_face
    hit_distance[pixel] = best_distance + near_distance
    if best_mesh >= 0:
        best_normal = wp.normalize(best_normal)
    hit_normal[pixel] = best_normal


@wp.kernel
def _shadow_ray_kernel(origin: wp.vec3, directions: wp.array(dtype=wp.vec3), hit_mesh: wp.array(dtype=wp.int32),
                       hit_distance: wp.array(dtype=wp.float32), hit_normal: wp.array(dtype=wp.vec3), sun: wp.vec3,
                       offset: float, static_mesh: wp.uint64, workspace_mesh: wp.uint64, rover_mesh: wp.uint64,
                       maximum_distance: float, blocked_world: wp.array(dtype=wp.int32),
                       blocked_rover: wp.array(dtype=wp.int32)):
    """From each hit point (lifted along the camera-facing normal), is the Sun hidden by terrain/rocks or the rover?"""
    pixel = wp.tid()
    blocked_world[pixel] = 0
    blocked_rover[pixel] = 0
    if hit_mesh[pixel] < 0:
        return
    direction = directions[pixel]
    normal = hit_normal[pixel]
    if wp.dot(normal, direction) > 0.0:
        normal = -normal
    start = origin + direction * hit_distance[pixel] + normal * offset
    query = wp.mesh_query_ray(static_mesh, start, sun, maximum_distance)
    if query.result:
        blocked_world[pixel] = 1
    else:
        query = wp.mesh_query_ray(workspace_mesh, start, sun, maximum_distance)
        if query.result:
            blocked_world[pixel] = 1
    query = wp.mesh_query_ray(rover_mesh, start, sun, maximum_distance)
    if query.result:
        blocked_rover[pixel] = 1


def _matrix_to_numpy(matrix) -> np.ndarray:
    """Gf.Matrix4d -> (4, 4) float64 (USD row-vector convention: p' = [p 1] @ M)."""
    return np.array([[matrix[row][column] for column in range(4)] for row in range(4)], dtype=np.float64)


def _prim_triangles(prim) -> tuple[np.ndarray, np.ndarray] | None:
    """(local vertices, triangles) of a Mesh (fan-triangulated) or Cube prim; None for other types."""
    from pxr import UsdGeom
    if prim.IsA(UsdGeom.Cube):
        half = 0.5 * float(UsdGeom.Cube(prim).GetSizeAttr().Get())
        corners = np.array([[x, y, z] for x in (-half, half) for y in (-half, half) for z in (-half, half)])
        faces = np.array([[0, 1, 3], [0, 3, 2], [4, 6, 7], [4, 7, 5], [0, 4, 5], [0, 5, 1],
                          [2, 3, 7], [2, 7, 6], [0, 2, 6], [0, 6, 4], [1, 5, 7], [1, 7, 3]], dtype=np.int32)
        return corners, faces
    if prim.IsA(UsdGeom.Mesh):
        mesh = UsdGeom.Mesh(prim)
        points = np.asarray(mesh.GetPointsAttr().Get(), dtype=np.float64)
        counts = np.asarray(mesh.GetFaceVertexCountsAttr().Get(), dtype=np.int64)
        indices = np.asarray(mesh.GetFaceVertexIndicesAttr().Get(), dtype=np.int64)
        triangles, cursor = [], 0
        for count in counts:
            polygon = indices[cursor:cursor + count]
            triangles.extend([polygon[0], polygon[k], polygon[k + 1]] for k in range(1, count - 1))
            cursor += count
        return points, np.asarray(triangles, dtype=np.int32)
    return None


@dataclass
class RoverBodyGeometry:
    """Visible rover triangles grouped by rigid body, in each body's frame."""

    body_names: list[str]
    vertex_body: np.ndarray        # (V,) index into body_names
    vertices_body_m: np.ndarray    # (V, 3) float64
    faces: np.ndarray              # (F, 3) int32


def rover_body_geometry(stage, root_path: str) -> RoverBodyGeometry:
    """Collect every visible Mesh/Cube under the rover root, expressed in its rigid-body ancestor's frame."""
    from pxr import Usd, UsdGeom, UsdPhysics
    cache = UsdGeom.XformCache(Usd.TimeCode.Default())
    names, vertex_body, vertices, faces, offset = [], [], [], [], 0
    for prim in Usd.PrimRange(stage.GetPrimAtPath(root_path)):
        if not prim.IsA(UsdGeom.Gprim) or UsdGeom.Imageable(prim).ComputeVisibility() == UsdGeom.Tokens.invisible:
            continue
        triangles = _prim_triangles(prim)
        if triangles is None:
            continue
        body = prim
        while body and body.GetPath() != stage.GetPrimAtPath(root_path).GetPath() and not body.HasAPI(UsdPhysics.RigidBodyAPI):
            body = body.GetParent()
        if not body.HasAPI(UsdPhysics.RigidBodyAPI):
            continue
        relative, _ = cache.ComputeRelativeTransform(prim, body)
        matrix = _matrix_to_numpy(relative)
        local, triangle_indices = triangles
        in_body = np.column_stack([local, np.ones(len(local))]) @ matrix
        name = body.GetName()
        if name not in names:
            names.append(name)
        vertex_body.append(np.full(len(local), names.index(name)))
        vertices.append(in_body[:, :3])
        faces.append(triangle_indices + offset)
        offset += len(local)
    return RoverBodyGeometry(names, np.concatenate(vertex_body), np.concatenate(vertices), np.concatenate(faces))


class GroundTruthMaskRenderer:
    """Builds the Warp meshes for one session and renders label images for any camera prim."""

    def __init__(self, session, include_rocks: bool = True):
        """Upload the static scene, the workspace lattice and the rover geometry; precompute crater rasters."""
        self.session = session
        self.scene = session.scene
        self.configuration = session.configuration
        ground_truth = self.configuration.ground_truth
        self.device = ground_truth.device
        wp.init()
        scene, layout = self.scene, self.scene.layout
        static_soup = static_scene_soup(scene, include_rocks and session.include_rocks)
        self.static_face_kind, self.static_face_index = static_soup.face_kind, static_soup.face_index
        self.static_mesh = self._mesh(static_soup.vertices, static_soup.faces)
        self.workspace_grid = layout.workspace_visual_grid
        workspace = lattice_soup(self.workspace_grid, scene.workspace_height_m, None, 0)
        self.workspace_vertices = workspace.vertices.copy()
        self.workspace_mesh = self._mesh(workspace.vertices, workspace.faces)
        self.workspace_revision = None
        rover_root = session.rover_paths.root
        self.rover_geometry = rover_body_geometry(session.stage, rover_root)
        self.rover_mesh = self._mesh(self.rover_geometry.vertices_body_m.astype(np.float32), self.rover_geometry.faces)
        self.rocks_diameter_m = None
        if "rocks" in scene.extra_arrays:
            self.rocks_diameter_m = np.asarray(scene.extra_arrays["rocks"]["rock_diameter_m"], dtype=np.float64)
        width = ground_truth.crater_rim_width_fraction
        minimum = ground_truth.crater_minimum_diameter_m
        self.workspace_crater_classes, _ = crater_class_raster(self.workspace_grid, scene.crater_table, width, minimum)
        self.near_ring_crater_classes, _ = crater_class_raster(layout.near_ring_grid, scene.crater_table, width, minimum)
        self.approximate_workspace = self.configuration.appearance.source == "lunar_regolith_scene"
        self.face_counts = {"static": static_soup.face_count, "workspace": workspace.face_count,
                            "rover": len(self.rover_geometry.faces)}
        self._camera_cache: dict[str, dict] = {}

    def _mesh(self, vertices: np.ndarray, faces: np.ndarray) -> wp.Mesh:
        """A Warp mesh (a single degenerate triangle far below the scene when empty, so queries stay valid)."""
        if len(faces) == 0:
            vertices = np.array([[0, 0, -1e5], [1e-3, 0, -1e5], [0, 1e-3, -1e5]], dtype=np.float32)
            faces = np.array([[0, 1, 2]], dtype=np.int32)
        points = wp.array(np.ascontiguousarray(vertices, dtype=np.float32), dtype=wp.vec3, device=self.device)
        indices = wp.array(np.ascontiguousarray(faces, dtype=np.int32).ravel(), dtype=wp.int32, device=self.device)
        return wp.Mesh(points=points, indices=indices, bvh_constructor="sah")   # lbvh: ~10^4 x slower on this scene

    # ------------------------------------------------------------------ dynamic geometry
    def update_workspace(self) -> None:
        """Apply the live track deformation to the workspace mesh when it changed (native/overlay modes)."""
        track_runtime = self.session.track_runtime
        if track_runtime is None or not track_runtime.geometry_enabled:
            return
        field = track_runtime.field
        revision = (field.sample_count, None if field.touched_box is None else tuple(field.touched_box))
        if revision == self.workspace_revision:
            return
        heights = field.rest_height_m.astype(np.float64) + field.height_change_m
        self.workspace_vertices[:, 2] = heights.ravel().astype(np.float32)
        self.workspace_mesh.points.assign(wp.array(self.workspace_vertices, dtype=wp.vec3, device=self.device))
        self.workspace_mesh.refit()
        self.workspace_revision = revision

    def update_rover(self) -> dict:
        """Re-pose the rover mesh from the articulation links; returns the link poses used."""
        poses = self.session.rover.read_link_poses()
        geometry = self.rover_geometry
        world = np.empty_like(geometry.vertices_body_m)
        for body_index, name in enumerate(geometry.body_names):
            selected = geometry.vertex_body == body_index
            position, orientation = poses[name]
            world[selected] = geometry.vertices_body_m[selected] @ rotation_matrix_from_quaternion(orientation).T + position
        self.rover_mesh.points.assign(wp.array(world.astype(np.float32), dtype=wp.vec3, device=self.device))
        self.rover_mesh.refit()
        return poses

    # ------------------------------------------------------------------ cameras
    def _camera_model(self, camera_path: str, resolution_px) -> dict:
        """Cached intrinsics, unit ray directions in the camera frame and the camera's parent body (if on the rover)."""
        key = f"{camera_path}@{resolution_px[0]}x{resolution_px[1]}"
        if key in self._camera_cache:
            return self._camera_cache[key]
        from pxr import Usd, UsdGeom, UsdPhysics
        stage = self.session.stage
        prim = stage.GetPrimAtPath(camera_path)
        camera = UsdGeom.Camera(prim)
        width, height = int(resolution_px[0]), int(resolution_px[1])
        focal = float(camera.GetFocalLengthAttr().Get())
        horizontal_aperture = float(camera.GetHorizontalApertureAttr().Get())
        vertical_aperture = float(camera.GetVerticalApertureAttr().Get())
        focal_x, focal_y = focal / horizontal_aperture * width, focal / vertical_aperture * height
        columns, rows = np.meshgrid(np.arange(width) + 0.5, np.arange(height) + 0.5)
        directions = np.stack([(columns - width / 2) / focal_x, -(rows - height / 2) / focal_y,
                               -np.ones_like(columns)], axis=-1).reshape(-1, 3)
        depth_factor = 1.0 / np.linalg.norm(directions, axis=1)          # distance along the ray -> image-plane depth
        directions *= depth_factor[:, None]
        body, body_name = prim, None
        while body and body.GetPath().pathString != "/":
            if body.HasAPI(UsdPhysics.RigidBodyAPI):
                body_name = body.GetName()
                break
            body = body.GetParent()
        relative = None
        if body_name is not None:
            cache = UsdGeom.XformCache(Usd.TimeCode.Default())
            relative = _matrix_to_numpy(cache.ComputeRelativeTransform(prim, body)[0])
        near_clip = float(camera.GetClippingRangeAttr().Get()[0])
        model = {"prim": prim, "width": width, "height": height, "directions_camera": directions, "near_clip_m": near_clip,
                 "depth_factor": depth_factor, "body_name": body_name, "relative": relative,
                 "focal_px": (focal_x, focal_y)}
        self._camera_cache[key] = model
        return model

    def camera_pose(self, model: dict, link_poses: dict | None) -> tuple[np.ndarray, np.ndarray]:
        """(rotation camera->world (3, 3), position (3,)) now."""
        if model["body_name"] is not None:
            position, orientation = (link_poses or self.session.rover.read_link_poses())[model["body_name"]]
            body_rotation = rotation_matrix_from_quaternion(orientation)
            relative = model["relative"]
            return body_rotation @ relative[:3, :3].T, body_rotation @ relative[3, :3] + position
        from pxr import Usd, UsdGeom
        matrix = _matrix_to_numpy(UsdGeom.XformCache(Usd.TimeCode.Default()).GetLocalToWorldTransform(model["prim"]))
        return matrix[:3, :3].T, matrix[3, :3]

    # ------------------------------------------------------------------ render
    def render(self, camera_path: str, resolution_px, sun_direction: np.ndarray | None = None) -> GroundTruthImages:
        """Label images of a camera at the current simulation state."""
        lighting, ground_truth = self.configuration.lighting, self.configuration.ground_truth
        sun = np.asarray(sun_direction if sun_direction is not None else
                         sun_direction_w(lighting.sun_azimuth_deg, lighting.sun_elevation_deg), dtype=np.float64)
        sun /= np.linalg.norm(sun)
        self.update_workspace()
        link_poses = self.update_rover()
        model = self._camera_model(camera_path, resolution_px)
        rotation, position = self.camera_pose(model, link_poses)
        directions = model["directions_camera"] @ rotation.T
        count = len(directions)
        device = self.device
        directions_gpu = wp.array(directions.astype(np.float32), dtype=wp.vec3, device=device)
        origin = wp.vec3(*map(float, position))
        hit_mesh = wp.empty(count, dtype=wp.int32, device=device)
        hit_face = wp.empty(count, dtype=wp.int32, device=device)
        hit_distance = wp.empty(count, dtype=wp.float32, device=device)
        hit_normal = wp.empty(count, dtype=wp.vec3, device=device)
        wp.launch(_camera_ray_kernel, dim=count, device=device,
                  inputs=[origin, directions_gpu, wp.vec3(*map(float, -rotation[:, 2])), model["near_clip_m"],
                          self.static_mesh.id, self.workspace_mesh.id, self.rover_mesh.id, float(MAXIMUM_RAY_DISTANCE_M)],
                  outputs=[hit_mesh, hit_face, hit_distance, hit_normal])
        blocked_world = wp.empty(count, dtype=wp.int32, device=device)
        blocked_rover = wp.empty(count, dtype=wp.int32, device=device)
        wp.launch(_shadow_ray_kernel, dim=count, device=device,
                  inputs=[origin, directions_gpu, hit_mesh, hit_distance, hit_normal, wp.vec3(*map(float, sun)),
                          float(ground_truth.shadow_ray_offset_m), self.static_mesh.id, self.workspace_mesh.id,
                          self.rover_mesh.id, float(MAXIMUM_RAY_DISTANCE_M)],
                  outputs=[blocked_world, blocked_rover])
        mesh = hit_mesh.numpy()
        face = hit_face.numpy()
        distance = hit_distance.numpy().astype(np.float64)
        normal = hit_normal.numpy().astype(np.float64)

        kind = np.zeros(count, dtype=np.int8)
        index = np.full(count, -1, dtype=np.int32)
        static = mesh == MESH_STATIC
        kind[static] = self.static_face_kind[face[static]]
        index[static] = self.static_face_index[face[static]]
        kind[mesh == MESH_WORKSPACE] = OBJECT_KIND_TERRAIN
        index[mesh == MESH_WORKSPACE] = 0
        kind[mesh == MESH_ROVER] = OBJECT_KIND_ROVER
        sky = mesh < 0
        distance[sky] = np.inf
        points = position + directions * np.where(sky, np.nan, distance)[:, None]

        ground_class = np.full(count, OBJECT_ID["terrain"], dtype=np.uint8)
        terrain = kind == OBJECT_KIND_TERRAIN
        if terrain.any():
            east, north = points[terrain, 0], points[terrain, 1]
            values, inside = nearest_lattice_value(self.workspace_crater_classes, self.workspace_grid, east, north)
            near_values, _ = nearest_lattice_value(self.near_ring_crater_classes, self.scene.layout.near_ring_grid,
                                                   east, north)
            classes = np.where(inside, values, near_values)
            classes = np.where(classes == 0, OBJECT_ID["terrain"], classes).astype(np.uint8)
            track_runtime = self.session.track_runtime
            if track_runtime is not None and track_runtime.geometry_enabled:
                tracks, track_inside = nearest_lattice_value(track_runtime.field.track_mask(), self.workspace_grid,
                                                             east, north, outside_value=False)
                classes[tracks & track_inside] = OBJECT_ID["wheel_track"]
            ground_class[terrain] = classes
        result = RayCastResult(kind, index, points, normal, directions, distance,
                               blocked_world.numpy().astype(bool), blocked_rover.numpy().astype(bool))
        object_ids, shadow_ids, rock_instance = compose_ground_truth(
            result, ground_class, self.rocks_diameter_m, sun, lighting.sun_angular_diameter_deg / 2,
            ground_truth.sun_glare_radius_deg, ground_truth.boulder_minimum_diameter_m)
        shape = (model["height"], model["width"])
        depth = (distance * model["depth_factor"]).astype(np.float32)
        return GroundTruthImages(object_ids.reshape(shape), shadow_ids.reshape(shape), rock_instance.reshape(shape),
                                 depth.reshape(shape))

    def metadata(self) -> dict:
        """Provenance of the masks (written next to saved frames)."""
        return {"method": "warp ray casting against the render geometry (camera rays through pixel centres, "
                          "shadow rays to the Sun)",
                "face_counts": self.face_counts,
                "workspace_geometry": ("APPROXIMATION: this project's visual lattice; the rendered workspace is the "
                                       "lunar_regolith render mesh") if self.approximate_workspace else "exact",
                "wheel_track_class": bool(self.session.track_runtime is not None and
                                          self.session.track_runtime.geometry_enabled),
                "configuration": self.configuration.ground_truth.model_dump(),
                "sun_glare_radius_note": "ASSUMPTION (MARE navcam value), not measured on this camera"}


__all__ = ["GroundTruthMaskRenderer", "rover_body_geometry", "TriangleSoup"]
