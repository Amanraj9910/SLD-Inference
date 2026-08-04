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


def detect_white_margins(img_array: np.ndarray, white_thresh: int = 240, blank_frac: float = 0.985) -> tuple[int, int, int, int]:
    """Detect top, bottom, left, right white margin amounts in pixels."""
    h, w = img_array.shape[:2]
    is_white = np.all(img_array > white_thresh, axis=2) if img_array.ndim == 3 else img_array > white_thresh

    top = 0
    for r in range(h):
        if np.mean(is_white[r, :]) < blank_frac:
            break
        top = r + 1

    bottom = 0
    for r in range(h - 1, -1, -1):
        if np.mean(is_white[r, :]) < blank_frac:
            break
        bottom = h - r

    left = 0
    for c in range(w):
        if np.mean(is_white[:, c]) < blank_frac:
            break
        left = c + 1

    right = 0
    for c in range(w - 1, -1, -1):
        if np.mean(is_white[:, c]) < blank_frac:
            break
        right = w - c

    return top, bottom, left, right


def adaptive_tile_image(
    image: Image.Image,
    target_symbol_px: float = 48.0,
    estimated_symbol_px: float = 48.0,
    model_input_size: int = 640,
    overlap: float = 0.20,
    enable_auto_crop: bool = False,
    white_threshold: int = 240,
    blank_row_fraction: float = 0.985,
    enable_scale_norm: bool = False,
    target_reference_height: float = 60.0,
) -> list[tuple[Image.Image, int, int, int, int, int, int, float]]:
    """
    Adaptive Tiling pipeline matching dfine_train_best.py:
      1. Optional Scale Normalization
      2. Optional Safe Auto-Crop White Margins
      3. Dynamic Grid Calculation based on target symbol size vs image size
      4. Crop tiles & resize to uniform model_input_size (640x640)

    Returns a list of tuples:
      (tile_resized, tx1, ty1, tw, th, cx1, cy1, scale_factor)
    """
    orig_w, orig_h = image.size

    # 1. Scale Normalization
    if enable_scale_norm and estimated_symbol_px > 0:
        s = target_reference_height / estimated_symbol_px
        max_px = 50_000_000  # 50MP safety ceiling
        projected_px = (orig_w * s) * (orig_h * s)
        if projected_px > max_px:
            cap_scale = (max_px / (orig_w * orig_h)) ** 0.5
            s = min(s, cap_scale)
        new_w, new_h = max(32, round(orig_w * s)), max(32, round(orig_h * s))
        scaled_img = image.resize((new_w, new_h), Image.Resampling.BILINEAR)
    else:
        s = 1.0
        new_w, new_h = orig_w, orig_h
        scaled_img = image

    # 2. Auto Crop White Margins
    if enable_auto_crop:
        arr = np.array(scaled_img)
        top, bottom, left, right = detect_white_margins(arr, white_threshold, blank_row_fraction)
        del arr
        cx1 = max(0, left - 5)
        cy1 = max(0, top - 5)
        cx2 = min(new_w, new_w - right + 5)
        cy2 = min(new_h, new_h - bottom + 5)
        if cx2 - cx1 >= 50 and cy2 - cy1 >= 50:
            cropped_img = scaled_img.crop((cx1, cy1, cx2, cy2))
        else:
            cx1, cy1 = 0, 0
            cropped_img = scaled_img
    else:
        cx1, cy1 = 0, 0
        cropped_img = scaled_img

    cw, ch = cropped_img.size

    # 3. Adaptive Grid Size (g_x, g_y)
    effective_symbol_px = estimated_symbol_px if estimated_symbol_px > 0 else 48.0
    g_x = max(1, round(target_symbol_px * cw / (model_input_size * effective_symbol_px)))
    g_y = max(1, round(target_symbol_px * ch / (model_input_size * effective_symbol_px)))

    tile_w = cw / (g_x - (g_x - 1) * overlap) if g_x > 1 else cw
    tile_h = ch / (g_y - (g_y - 1) * overlap) if g_y > 1 else ch
    stride_x = tile_w * (1 - overlap) if g_x > 1 else cw
    stride_y = tile_h * (1 - overlap) if g_y > 1 else ch

    positions_x = [c * stride_x for c in range(g_x)]
    if g_x > 1:
        positions_x[-1] = max(0.0, cw - tile_w)
    positions_y = [r * stride_y for r in range(g_y)]
    if g_y > 1:
        positions_y[-1] = max(0.0, ch - tile_h)

    tiles_info: list[tuple[Image.Image, int, int, int, int, int, int, float]] = []

    for py in positions_y:
        for px in positions_x:
            tx1, ty1 = round(px), round(py)
            tx2, ty2 = min(cw, round(px + tile_w)), min(ch, round(py + tile_h))
            if tx2 <= tx1 or ty2 <= ty1:
                continue

            tile_crop = cropped_img.crop((tx1, ty1, tx2, ty2))
            tw, th = tile_crop.size
            if tw <= 0 or th <= 0:
                continue

            tile_resized = tile_crop.resize((model_input_size, model_input_size), Image.Resampling.BILINEAR)
            tiles_info.append((tile_resized, tx1, ty1, tw, th, cx1, cy1, s))

    return tiles_info


def merge_adaptive_detections(
    adaptive_results: list[tuple[sv.Detections, int, int, int, int, int, int, float]],
    orig_W: int,
    orig_H: int,
    model_input_size: int = 640,
    iou_threshold: float = 0.50,
) -> sv.Detections:
    """
    Re-project adaptive tile detections back through:
      Tile 640x640 -> Tile crop (tw, th) -> Crop offset (tx1, ty1) -> Auto-crop offset (cx1, cy1) -> Scale factor s -> Full image (orig_W, orig_H)
    and apply NMS.
    """
    all_boxes: list[np.ndarray] = []
    all_scores: list[np.ndarray] = []
    all_classes: list[np.ndarray] = []

    for det, tx1, ty1, tw, th, cx1, cy1, s in adaptive_results:
        if len(det) == 0:
            continue

        boxes = det.xyxy.copy().astype(float)
        # 1. Scale from model resolution (640) back to tile crop resolution (tw, th)
        boxes[:, [0, 2]] *= (tw / model_input_size)
        boxes[:, [1, 3]] *= (th / model_input_size)

        # 2. Add tile top-left offset inside cropped image
        boxes[:, [0, 2]] += tx1
        boxes[:, [1, 3]] += ty1

        # 3. Add auto-crop top-left offset inside scaled image
        boxes[:, [0, 2]] += cx1
        boxes[:, [1, 3]] += cy1

        # 4. Divide by scale-normalization factor s to get original unscaled image pixels
        if s > 0 and s != 1.0:
            boxes /= s

        # 5. Clip to original image boundaries
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

