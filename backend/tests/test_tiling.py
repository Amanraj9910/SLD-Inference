"""
Unit tests for tiling.py — no GPU required.
"""
import numpy as np
import pytest
import supervision as sv
from PIL import Image

from app.tiling import merge_detections, tile_image


def _solid_image(W: int = 800, H: int = 600, color=(200, 200, 200)) -> Image.Image:
    return Image.new("RGB", (W, H), color)


# ---------------------------------------------------------------------------
# tile_image
# ---------------------------------------------------------------------------

def test_tile_count_exact():
    img = _solid_image(800, 800)
    tiles = tile_image(img, grid_size=4, overlap=0.0)
    assert len(tiles) == 16, "4×4 grid should produce exactly 16 tiles"


def test_tile_count_with_overlap():
    img = _solid_image(800, 800)
    tiles = tile_image(img, grid_size=4, overlap=0.2)
    # tile_image uses a fixed grid_size × grid_size loop, so count is always grid²
    assert len(tiles) == 16


def test_tile_offsets_non_negative():
    img = _solid_image(1000, 800)
    for tile, x_off, y_off in tile_image(img, grid_size=4, overlap=0.2):
        assert x_off >= 0
        assert y_off >= 0


def test_tile_crops_within_image():
    W, H = 1000, 800
    img = _solid_image(W, H)
    for tile, x_off, y_off in tile_image(img, grid_size=4, overlap=0.2):
        tw, th = tile.size
        assert x_off + tw <= W
        assert y_off + th <= H


def test_tile_grid_size_1():
    """Grid 1×1 should produce exactly one tile equal to the original image."""
    img = _solid_image(640, 640)
    tiles = tile_image(img, grid_size=1, overlap=0.0)
    assert len(tiles) == 1
    tile, x, y = tiles[0]
    assert (x, y) == (0, 0)
    assert tile.size == img.size


# ---------------------------------------------------------------------------
# merge_detections
# ---------------------------------------------------------------------------

def _make_det(boxes, scores, class_ids) -> sv.Detections:
    return sv.Detections(
        xyxy=np.array(boxes, dtype=np.float32),
        confidence=np.array(scores, dtype=np.float32),
        class_id=np.array(class_ids, dtype=int),
    )


def test_merge_empty():
    result = merge_detections([], orig_W=800, orig_H=600)
    assert len(result) == 0


def test_merge_single_tile_offset():
    """Boxes should be shifted by the tile offset."""
    det = _make_det([[10, 10, 50, 50]], [0.9], [0])
    result = merge_detections([(det, 100, 200)], orig_W=800, orig_H=600)
    assert len(result) == 1
    box = result.xyxy[0]
    assert box[0] == pytest.approx(110.0)
    assert box[1] == pytest.approx(210.0)
    assert box[2] == pytest.approx(150.0)
    assert box[3] == pytest.approx(250.0)


def test_merge_clipped_to_image():
    """Boxes that extend beyond the image boundary should be clipped."""
    det = _make_det([[0, 0, 100, 100]], [0.9], [0])
    # Place tile at (750, 550) — box would go to (850, 650) exceeding 800×600
    result = merge_detections([(det, 750, 550)], orig_W=800, orig_H=600)
    assert len(result) == 1
    box = result.xyxy[0]
    assert box[2] <= 800
    assert box[3] <= 600


def test_merge_nms_removes_duplicate():
    """Two identical boxes from different tiles should be merged to one by NMS."""
    det1 = _make_det([[10, 10, 100, 100]], [0.9], [0])
    det2 = _make_det([[10, 10, 100, 100]], [0.85], [0])
    result = merge_detections(
        [(det1, 0, 0), (det2, 0, 0)],
        orig_W=800,
        orig_H=600,
        iou_threshold=0.5,
    )
    assert len(result) == 1


def test_merge_different_classes_kept():
    """Boxes with different class_ids should not be merged even if overlapping."""
    det1 = _make_det([[10, 10, 100, 100]], [0.9], [0])
    det2 = _make_det([[10, 10, 100, 100]], [0.85], [1])
    result = merge_detections(
        [(det1, 0, 0), (det2, 0, 0)],
        orig_W=800,
        orig_H=600,
        iou_threshold=0.5,
    )
    assert len(result) == 2
