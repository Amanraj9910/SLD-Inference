"""
Tiling helpers — direct port of the notebook Cell 4 logic.

tile_image()       : slice a PIL image into overlapping tiles
merge_detections() : re-offset tile boxes back to image coords + NMS
"""
from __future__ import annotations

import numpy as np
import supervision as sv
from PIL import Image


def tile_image(
    image: Image.Image,
    grid_size: int = 4,
    overlap: float = 0.20,
) -> list[tuple[Image.Image, int, int]]:
    """
    Slice *image* into grid_size × grid_size overlapping tiles.

    Returns a list of (crop, x_offset, y_offset) tuples where the
    offsets give the top-left corner of each tile in the original image.
    """
    W, H = image.size
    tile_w = W // grid_size
    tile_h = H // grid_size
    step_x = int(tile_w * (1 - overlap))
    step_y = int(tile_h * (1 - overlap))

    tiles: list[tuple[Image.Image, int, int]] = []
    for gy in range(grid_size):
        for gx in range(grid_size):
            x = int(gx * step_x)
            y = int(gy * step_y)
            x_end = min(x + tile_w, W)
            y_end = min(y + tile_h, H)
            tile = image.crop((x, y, x_end, y_end))
            tiles.append((tile, x, y))
    return tiles


def merge_detections(
    tile_results: list[tuple[sv.Detections, int, int]],
    orig_W: int,
    orig_H: int,
    iou_threshold: float = 0.50,
) -> sv.Detections:
    """
    Re-project tile-local boxes back into the original image coordinate
    space and apply NMS across all tiles.

    *tile_results* is a list of (sv.Detections, x_offset, y_offset).
    """
    all_boxes: list[np.ndarray] = []
    all_scores: list[np.ndarray] = []
    all_classes: list[np.ndarray] = []

    for det, x_off, y_off in tile_results:
        if len(det) == 0:
            continue
        boxes = det.xyxy.copy()
        boxes[:, [0, 2]] += x_off
        boxes[:, [1, 3]] += y_off
        boxes[:, 0] = np.clip(boxes[:, 0], 0, orig_W)
        boxes[:, 1] = np.clip(boxes[:, 1], 0, orig_H)
        boxes[:, 2] = np.clip(boxes[:, 2], 0, orig_W)
        boxes[:, 3] = np.clip(boxes[:, 3], 0, orig_H)
        all_boxes.append(boxes)
        all_scores.append(det.confidence)
        all_classes.append(det.class_id)

    if not all_boxes:
        return sv.Detections.empty()

    merged = sv.Detections(
        xyxy=np.vstack(all_boxes),
        confidence=np.concatenate(all_scores),
        class_id=np.concatenate(all_classes).astype(int),
    )
    return merged.with_nms(threshold=iou_threshold)
