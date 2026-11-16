"""Mask class tables: object mask and shadow label, with ids and colours of MARE.

Object ids and colours are those of MARE's ``config/mask_classes.json`` (github.com/raulcastar/Unreal-Moon-Terrain-
Generation), so datasets from both simulators share one legend; MARE's ``landscape`` is called ``terrain`` here.
Shadow ids are MARE's ``shadow_label`` ids. Colours are 8-bit and even (MARE keeps flags in the lowest bit).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True)
class MaskClass:
    """One class of a label image."""

    id: int
    name: str
    group: str
    rgb: tuple[int, int, int]
    description: str


OBJECT_CLASSES = (
    MaskClass(0, "unlabeled", "other", (0, 0, 0), "nothing hit and not sky (should not occur)"),
    MaskClass(1, "sky", "sky", (36, 40, 96), "no surface along the camera ray"),
    MaskClass(2, "rover", "objects", (250, 194, 16), "the rover's visual geometry"),
    MaskClass(3, "sun", "sky", (28, 128, 162), "sky within the Sun's disc"),
    MaskClass(4, "sun_glare", "sky", (230, 14, 118), "sky around the disc out to the configured glare radius"),
    MaskClass(5, "terrain", "ground", (10, 96, 216), "terrain surface (MARE: landscape)"),
    MaskClass(6, "crater", "ground", (124, 50, 208), "crater bowl, inside the rim crest"),
    MaskClass(7, "crater_rim", "ground", (86, 52, 82), "ring outside the rim crest, rim_width_fraction x radius wide"),
    MaskClass(8, "wheel_track", "ground", (64, 124, 162), "ground touched by a tread or shoulder (live track field)"),
    MaskClass(9, "boulder", "objects", (238, 244, 242), "rock at least boulder_minimum_diameter_m wide"),
    MaskClass(10, "small_rock", "objects", (210, 62, 6), "smaller rock (pebbles included)"),
)

SHADOW_CLASSES = (
    MaskClass(0, "sky", "sky", (0, 0, 0), "no surface"),
    MaskClass(1, "sunlit_ground", "ground", (40, 120, 230), "terrain, crater, rim or track in sunlight"),
    MaskClass(2, "shadowed_ground", "ground", (20, 20, 60), "terrain, crater, rim or track without sunlight"),
    MaskClass(3, "sunlit_object", "objects", (190, 190, 190), "rock or rover in sunlight"),
    MaskClass(4, "shadowed_object", "objects", (70, 70, 70), "rock or rover without sunlight, including its unlit side"),
    MaskClass(5, "rover_shadow", "ground", (250, 140, 20), "ground or rock in the rover's own shadow"),
)

OBJECT_ID = {mask_class.name: mask_class.id for mask_class in OBJECT_CLASSES}
SHADOW_ID = {mask_class.name: mask_class.id for mask_class in SHADOW_CLASSES}
GROUND_OBJECT_IDS = (OBJECT_ID["terrain"], OBJECT_ID["crater"], OBJECT_ID["crater_rim"], OBJECT_ID["wheel_track"])


def legend() -> dict:
    """JSON-ready legend of both label images (written next to every saved mask)."""
    return {"object_mask": [asdict(mask_class) for mask_class in OBJECT_CLASSES],
            "shadow_label": [asdict(mask_class) for mask_class in SHADOW_CLASSES],
            "source": "ids and colours after MARE mask_classes.json (landscape -> terrain); shadow ids after MARE "
                      "shadow_label; computed by ray casting against this simulator's render geometry"}


def colorize(label_image: np.ndarray, classes: tuple[MaskClass, ...]) -> np.ndarray:
    """(H, W, 3) uint8 colour image of a label image."""
    palette = np.zeros((256, 3), dtype=np.uint8)
    for mask_class in classes:
        palette[mask_class.id] = mask_class.rgb
    return palette[np.asarray(label_image, dtype=np.uint8)]
