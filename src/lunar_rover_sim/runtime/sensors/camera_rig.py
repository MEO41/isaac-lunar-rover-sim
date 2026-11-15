"""Cameras, render products and annotators (rgb, depth, semantic segmentation), plus static observer cameras."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from lunar_rover_sim.core.frames.pose_math import focal_length_mm, look_at_rotation, quaternion_from_rotation_matrix


@dataclass
class CameraHandle:
    """One camera with its render product and annotators."""

    camera_path: str
    resolution_px: tuple[int, int]
    render_product: object
    annotators: dict = field(default_factory=dict)


def define_observer_camera(stage, path: str, eye_m, target_m, horizontal_field_of_view_deg: float,
                           resolution_px, horizontal_aperture_mm: float = 20.955) -> str:
    """Static world camera at ``eye_m`` looking at ``target_m``; returns its path."""
    from pxr import Gf, UsdGeom
    from lunar_rover_sim.runtime.stage_assembly.usd_helpers import set_pose
    camera = UsdGeom.Camera.Define(stage, path)
    width, height = resolution_px
    camera.CreateHorizontalApertureAttr(float(horizontal_aperture_mm))
    camera.CreateVerticalApertureAttr(float(horizontal_aperture_mm * height / width))
    camera.CreateFocalLengthAttr(float(focal_length_mm(horizontal_field_of_view_deg, horizontal_aperture_mm)))
    camera.CreateClippingRangeAttr(Gf.Vec2f(0.05, 100000.0))
    rotation = look_at_rotation(np.asarray(eye_m, float), np.asarray(target_m, float))
    set_pose(camera.GetPrim(), eye_m, quaternion_from_rotation_matrix(rotation))
    return path


class CameraRig:
    """Render products and annotators for a set of cameras; data is read after ``render_synchronously``."""

    def __init__(self):
        """Empty rig."""
        self.cameras: dict[str, CameraHandle] = {}

    def add_camera(self, name: str, camera_path: str, resolution_px, annotator_names=("rgb",)) -> CameraHandle:
        """Create a render product for ``camera_path`` and attach the named annotators."""
        import omni.replicator.core as replicator
        render_product = replicator.create.render_product(camera_path, tuple(int(v) for v in resolution_px))
        handle = CameraHandle(camera_path, tuple(resolution_px), render_product)
        for annotator_name in annotator_names:
            parameters = {"colorize": False} if annotator_name == "semantic_segmentation" else None
            annotator = replicator.AnnotatorRegistry.get_annotator(annotator_name, init_params=parameters)
            annotator.attach([render_product])
            handle.annotators[annotator_name] = annotator
        self.cameras[name] = handle
        return handle

    def read(self, name: str) -> dict[str, np.ndarray]:
        """Current annotator outputs of one camera: rgb (H,W,3 uint8), depth (H,W float32), semantic ids + labels."""
        output = {}
        for annotator_name, annotator in self.cameras[name].annotators.items():
            data = annotator.get_data()
            if annotator_name == "rgb":
                output["rgb"] = np.asarray(data)[..., :3].copy()
            elif annotator_name == "distance_to_image_plane":
                output["depth_m"] = np.asarray(data).copy()
            elif annotator_name == "semantic_segmentation":
                output["semantic_id"] = np.asarray(data["data"]).copy()
                output["semantic_labels"] = data["info"]["idToLabels"]
            else:
                output[annotator_name] = data
        return output

    def destroy(self) -> None:
        """Detach annotators and destroy render products (frees video memory)."""
        for handle in self.cameras.values():
            for annotator in handle.annotators.values():
                try:
                    annotator.detach()
                except Exception:
                    pass
            try:
                handle.render_product.destroy()
            except Exception:
                pass
        self.cameras.clear()


def save_png(path, rgb: np.ndarray) -> None:
    """Write an (H, W, 3) uint8 image."""
    from PIL import Image
    Image.fromarray(np.asarray(rgb, dtype=np.uint8)).save(path)
