"""Training-free panel and busbar detector for SLD raster images.

The detector uses only geometric evidence from the drawing:

* dashed horizontal/vertical edges are grouped into candidate panel sides;
* four compatible sides are assembled into a rectangle;
* horizontal strokes inside each rectangle are scored and the best one is kept.

It is deliberately independent of the Roboflow dataset.  The dataset is used
by ``evaluate_deterministic.py`` only for measurement and visual review.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np

from .graph_detector import (
    detect_panels_graph as _detect_panels_graph,
    detect_thick_horizontal_segments as _detect_thick_horizontal_segments,
    longest_busbar_for_panel as _longest_busbar_for_panel,
    preprocess_image as _graph_preprocess_image,
)


@dataclass
class LineCandidate:
    orientation: str
    const: float
    lo: float
    hi: float
    segment_count: int
    coverage: float
    score: float
    pattern: float = 0.0
    has_zigzag: bool = False


def _gray_binary(image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image.copy()
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    binary = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 25, 8
    )
    return gray, binary


def _lsd_segments(gray: np.ndarray) -> tuple[list[tuple[float, float, float, float]], list[tuple[float, float, float, float]]]:
    detector = cv2.createLineSegmentDetector(cv2.LSD_REFINE_STD)
    found = detector.detect(gray)[0]
    if found is None:
        return [], []

    h, w = gray.shape
    min_length = max(2.0, min(h, w) * 0.004)
    horizontal: list[tuple[float, float, float, float]] = []
    vertical: list[tuple[float, float, float, float]] = []
    for raw in found.reshape(-1, 4):
        x1, y1, x2, y2 = map(float, raw)
        length = float(np.hypot(x2 - x1, y2 - y1))
        if length < min_length:
            continue
        angle = abs(float(np.degrees(np.arctan2(y2 - y1, x2 - x1))))
        angle = min(angle, 180.0 - angle)
        if angle <= 6:
            horizontal.append((min(x1, x2), (y1 + y2) / 2, max(x1, x2), length))
        elif 84 <= angle <= 96:
            vertical.append((min(y1, y2), (x1 + x2) / 2, max(y1, y2), length))
    return horizontal, vertical


def _line_support(binary: np.ndarray, orientation: str, const: float, lo: float, hi: float) -> float:
    radius = max(1, int(round(min(binary.shape) * 0.004)))
    if orientation == "h":
        y = int(round(const))
        strip = binary[max(0, y - radius):min(binary.shape[0], y + radius + 1),
                       max(0, int(lo)):min(binary.shape[1], int(hi) + 1)]
    else:
        x = int(round(const))
        strip = binary[max(0, int(lo)):min(binary.shape[0], int(hi) + 1),
                       max(0, x - radius):min(binary.shape[1], x + radius + 1)]
    return float(np.count_nonzero(strip)) / float(strip.size) if strip.size else 0.0


def _line_pattern(binary: np.ndarray, orientation: str, const: float, lo: float, hi: float) -> float:
    """Estimate whether ink along a candidate is a repeated dash pattern."""
    radius = max(1, int(round(min(binary.shape) * 0.004)))
    if orientation == "h":
        y = int(round(const))
        strip = binary[max(0, y - radius):min(binary.shape[0], y + radius + 1), max(0, int(lo)):min(binary.shape[1], int(hi) + 1)]
    else:
        x = int(round(const))
        strip = binary[max(0, int(lo)):min(binary.shape[0], int(hi) + 1), max(0, x - radius):min(binary.shape[1], x + radius + 1)]
    if strip.size == 0:
        return 0.0
    profile = np.any(strip > 0, axis=0 if orientation == "h" else 1)
    padded = np.pad(profile.astype(np.uint8), (1, 1))
    starts = np.flatnonzero((padded[1:-1] == 1) & (padded[:-2] == 0))
    ends = np.flatnonzero((padded[1:-1] == 1) & (padded[2:] == 0))
    if len(starts) < 3:
        return 0.0
    runs = ends - starts + 1
    gaps = starts[1:] - ends[:-1] - 1
    if len(gaps) == 0:
        return 0.0
    gap_cv = float(np.std(gaps) / max(np.mean(gaps), 1.0))
    run_cv = float(np.std(runs) / max(np.mean(runs), 1.0))
    regularity = np.exp(-0.55 * min(gap_cv, 4.0)) * np.exp(-0.25 * min(run_cv, 4.0))
    return float(min(1.0, len(runs) / 8.0) * regularity)


def _is_uneven_dash_pattern(binary: np.ndarray, orientation: str, const: float, lo: float, hi: float) -> bool:
    """Return True only when the dash pattern has uneven/mixed run lengths.

    Real panel borders in SLD drawings use a distinctive repeating pattern
    with mixed dash sizes -- e.g. long-short-short-long-short.  Uniform
    patterns where every dash is roughly the same length (e.g. small, small,
    small) are typically dimension lines or hatching and should be rejected.

    Unevenness can show up in the dash lengths themselves, the gaps between
    dashes, or both -- so both are checked and either one qualifying is
    enough to accept the line.

    Zigzag/continuation sides are solid (not dashed) and are handled
    separately -- they are never passed through this filter.
    """
    radius = max(1, int(round(min(binary.shape) * 0.004)))
    if orientation == "h":
        y = int(round(const))
        strip = binary[max(0, y - radius):min(binary.shape[0], y + radius + 1),
                       max(0, int(lo)):min(binary.shape[1], int(hi) + 1)]
    else:
        x = int(round(const))
        strip = binary[max(0, int(lo)):min(binary.shape[0], int(hi) + 1),
                       max(0, x - radius):min(binary.shape[1], x + radius + 1)]
    if strip.size == 0:
        return False
    profile = np.any(strip > 0, axis=0 if orientation == "h" else 1)
    padded = np.pad(profile.astype(np.uint8), (1, 1))
    starts = np.flatnonzero((padded[1:-1] == 1) & (padded[:-2] == 0))
    ends = np.flatnonzero((padded[1:-1] == 1) & (padded[2:] == 0))
    if len(starts) < 4:
        # Too few dashes to judge the pattern -- accept it (could be a short
        # border fragment that's still legitimate).
        return True
    runs = (ends - starts + 1).astype(float)
    gaps = (starts[1:] - ends[:-1] - 1).astype(float) if len(starts) > 1 else np.array([])

    def _has_uneven_spread(values: np.ndarray) -> bool:
        if values.size == 0:
            return False
        mean_v = float(np.mean(values))
        if mean_v < 1.0:
            return False
        # Coefficient of variation.  Uniform patterns (all segments the same
        # size -- the "small, small, small, small" case) produce CV < 0.3.
        # Mixed big/small patterns produce CV > 0.3.
        cv = float(np.std(values) / mean_v)
        if cv > 0.30:
            return True
        # Additionally check for at least two distinct length clusters via
        # the ratio between the longest and shortest values.  Even if CV is
        # low, the presence of a few clearly longer segments is a signal
        # (e.g. "big, small, small, big, small").
        sorted_v = np.sort(values)
        q = max(1, len(sorted_v) // 4)
        short_median = float(np.median(sorted_v[:q]))
        long_median = float(np.median(sorted_v[-q:]))
        return short_median > 0 and long_median / short_median >= 1.8

    # Genuine panel borders ("big dash, small, small, big, small...") are
    # uneven in dash length, gap length, or both.  A pattern that is uniform
    # in both dashes and gaps ("small, small, small, small") is rejected.
    return _has_uneven_spread(runs) or _has_uneven_spread(gaps)


def _rect_sides_uneven(binary: np.ndarray, x1: float, y1: float, x2: float, y2: float) -> bool:
    """Return True unless one of the rectangle's four sides is a uniformly
    dashed line.

    This mirrors ``_is_uneven_dash_pattern`` but is applied directly to a
    finished rectangle so the native-resolution graph-cycle path (which
    builds panels from ``busbar_panel_detector_v4.detect_panels_graph`` and
    never checks dash regularity itself) also rejects evenly-dashed borders.
    A solid or too-short side already reads as "accept" from
    ``_is_uneven_dash_pattern``, so this only ever penalises sides that are
    genuinely dashed with a uniform rhythm.
    """
    sides = (
        ("h", float(y1), float(x1), float(x2)),
        ("h", float(y2), float(x1), float(x2)),
        ("v", float(x1), float(y1), float(y2)),
        ("v", float(x2), float(y1), float(y2)),
    )
    return all(_is_uneven_dash_pattern(binary, orientation, const, lo, hi) for orientation, const, lo, hi in sides)


def _suppress_overlapping_panels(panels: list[dict]) -> list[dict]:
    """Non-maximum suppression across panel candidates.

    No two panels may overlap in the final result. When two candidates
    overlap, the **larger** panel is kept; if two candidates have identical
    area, the one with the **higher confidence score** is kept instead.
    Overlap is measured as intersection area over the *smaller* panel's
    area, so a small panel that sits mostly inside a larger one is always
    treated as a duplicate and dropped -- not just near-identical boxes.

    The threshold is intentionally low (8%). Two genuinely adjacent panels
    that only share a border line produce ~0% true area overlap, so a low
    threshold does not affect them -- only real duplicate/nested detections
    cross this bar.
    """
    if not panels:
        return []
    ranked = sorted(panels, key=lambda p: (p["width"] * p["height"], p["score"]), reverse=True)
    kept: list[dict] = []
    for panel in ranked:
        dominated = False
        pa = panel["width"] * panel["height"]
        for old in kept:
            oa = old["width"] * old["height"]
            ix1 = max(panel["x"], old["x"])
            iy1 = max(panel["y"], old["y"])
            ix2 = min(panel["x"] + panel["width"], old["x"] + old["width"])
            iy2 = min(panel["y"] + panel["height"], old["y"] + old["height"])
            inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
            smaller_area = min(pa, oa)
            if smaller_area > 0 and inter / float(smaller_area) >= 0.08:
                dominated = True
                break
        if not dominated:
            kept.append(panel)
    return sorted(kept, key=lambda p: (p["y"], p["x"]))


def _has_internal_content(binary: np.ndarray, panel: dict) -> bool:
    """Return True when the panel contains a busbar or any significant
    horizontal/vertical line stroke in its interior.

    Panels that are empty dashed rectangles (no electrical content) are
    spurious detections and should be removed.
    """
    # Fast path: panel already has a busbar assigned.
    if panel.get("busbar"):
        return True
    x, y, w, h = panel["x"], panel["y"], panel["width"], panel["height"]
    inset_x = max(4, int(round(w * 0.06)))
    inset_y = max(4, int(round(h * 0.10)))
    x1, x2 = x + inset_x, x + w - inset_x
    y1, y2 = y + inset_y, y + h - inset_y
    if x2 <= x1 or y2 <= y1:
        return False
    roi = binary[y1:y2, x1:x2]
    if roi.size == 0:
        return False
    # Check for any horizontal line ≥ 10% of panel width
    min_h_len = max(8, int(round(w * 0.10)))
    h_opened = cv2.morphologyEx(roi, cv2.MORPH_OPEN,
                                cv2.getStructuringElement(cv2.MORPH_RECT, (min_h_len, 1)))
    if np.count_nonzero(h_opened) > 0:
        return True
    # Check for any vertical line ≥ 10% of panel height
    min_v_len = max(8, int(round(h * 0.10)))
    v_opened = cv2.morphologyEx(roi, cv2.MORPH_OPEN,
                                cv2.getStructuringElement(cv2.MORPH_RECT, (1, min_v_len)))
    if np.count_nonzero(v_opened) > 0:
        return True
    return False


def _has_zigzag(binary: np.ndarray, orientation: str, const: float, lo: float, hi: float) -> bool:
    """Return True when a short Z-shaped continuation break exists on the line."""
    h, w = binary.shape
    span = int(hi - lo)
    if span < 20:
        return False
    # Maximum zigzag extent: short relative to the full span
    max_break = max(8, int(span * 0.12))
    lateral_reach = max(4, int(min(h, w) * 0.025))
    radius = max(1, int(round(min(h, w) * 0.004)))

    # Build a 1-D ink profile along the candidate line
    if orientation == "h":
        y = int(round(const))
        y0, y1 = max(0, y - radius), min(h, y + radius + 1)
        x0, x1 = max(0, int(lo)), min(w, int(hi) + 1)
        strip = binary[y0:y1, x0:x1]
    else:
        x = int(round(const))
        x0, x1 = max(0, x - radius), min(w, x + radius + 1)
        y0, y1 = max(0, int(lo)), min(h, int(hi) + 1)
        strip = binary[y0:y1, x0:x1]
    if strip.size == 0:
        return False
    profile = np.any(strip > 0, axis=0 if orientation == "h" else 1).astype(np.uint8)

    # Find gaps (runs of zeros) in the profile
    padded = np.pad(profile, (1, 1))
    gap_starts = np.flatnonzero((padded[1:-1] == 0) & (padded[:-2] == 1))
    gap_ends = np.flatnonzero((padded[1:-1] == 0) & (padded[2:] == 1))
    if len(gap_starts) == 0 or len(gap_ends) == 0:
        return False

    for gs, ge in zip(gap_starts, gap_ends):
        gap_len = ge - gs + 1
        if gap_len < 3 or gap_len > max_break:
            continue
        # Look for displaced ink on both sides of the gap in the lateral direction
        if orientation == "h":
            gap_x0 = max(0, x0 + gs - 1)
            gap_x1 = min(w, x0 + ge + 2)
            above = binary[max(0, y - lateral_reach):max(0, y - radius), gap_x0:gap_x1]
            below = binary[min(h, y + radius + 1):min(h, y + lateral_reach + 1), gap_x0:gap_x1]
        else:
            gap_y0 = max(0, y0 + gs - 1)
            gap_y1 = min(h, y0 + ge + 2)
            above = binary[gap_y0:gap_y1, max(0, x - lateral_reach):max(0, x - radius)]
            below = binary[gap_y0:gap_y1, min(w, x + radius + 1):min(w, x + lateral_reach + 1)]
        if above.size > 0 and below.size > 0:
            above_ink = float(np.count_nonzero(above)) / above.size
            below_ink = float(np.count_nonzero(below)) / below.size
            if above_ink > 0.08 and below_ink > 0.08:
                return True
    return False


def _cluster_segments(
    segments: list[tuple[float, float, float, float]],
    orientation: str,
    binary: np.ndarray,
) -> list[LineCandidate]:
    if not segments:
        return []
    h, w = binary.shape
    const_tol = max(2.0, min(h, w) * 0.009)
    min_span = max(22.0, min(h, w) * 0.045)
    groups: list[list[tuple[float, float, float, float]]] = []
    for seg in sorted(segments, key=lambda s: s[1]):
        for group in groups:
            if abs(seg[1] - float(np.median([x[1] for x in group]))) <= const_tol:
                group.append(seg)
                break
        else:
            groups.append([seg])

    result: list[LineCandidate] = []
    axis_length = w if orientation == "h" else h
    for group in groups:
        intervals = sorted((s[0], s[2], s[3]) for s in group)
        merged_lo, merged_hi = intervals[0][0], intervals[0][1]
        union = 0.0
        for a, b, _ in intervals[1:]:
            if a <= merged_hi:
                merged_hi = max(merged_hi, b)
            else:
                union += merged_hi - merged_lo
                merged_lo, merged_hi = a, b
        union += merged_hi - merged_lo
        span_lo = min(s[0] for s in group)
        span_hi = max(s[2] for s in group)
        span = span_hi - span_lo
        if span < min_span or len(group) < 3:
            continue
        coverage = min(1.0, union / max(span, 1.0))
        const = float(np.median([s[1] for s in group]))
        zigzag = _has_zigzag(binary, orientation, const, span_lo, span_hi)
        # A panel side is dashed.  Solid lines are retained only when their
        # LSD representation is fragmented enough to be a plausible border.
        # Lines with a zigzag continuation break are always kept.
        if coverage > 0.92 and len(group) <= 5 and not zigzag:
            continue
        support = _line_support(binary, orientation, const, span_lo, span_hi)
        if support < 0.015:
            continue
        pattern = _line_pattern(binary, orientation, const, span_lo, span_hi)
        # Reject uniform dash patterns (small, small, small...) unless the
        # line has a zigzag continuation (which is solid, not dashed).
        if not zigzag and pattern > 0.1 and not _is_uneven_dash_pattern(binary, orientation, const, span_lo, span_hi):
            continue
        dash_score = min(1.0, len(group) / 8.0) * (1.0 - min(coverage, 0.9))
        score = 0.30 * dash_score + 0.20 * min(1.0, span / axis_length) + 0.20 * min(1.0, support * 4) + 0.30 * pattern
        if zigzag:
            score = min(1.0, score + 0.55)
        result.append(LineCandidate(orientation, const, span_lo, span_hi, len(group), coverage, score, pattern, zigzag))
    return result


def _morph_line_candidates(binary: np.ndarray, orientation: str) -> list[LineCandidate]:
    """Recover very short/faint dash runs which LSD may fragment too much."""
    h, w = binary.shape
    gap = max(3, int(round(min(h, w) * 0.012)))
    run = max(22, int(round((w if orientation == "h" else h) * 0.07)))
    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT, (gap, 3) if orientation == "h" else (3, gap)
    )
    bridged = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    opened = cv2.morphologyEx(
        bridged,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (run, 1) if orientation == "h" else (1, run)),
    )
    n, _, stats, _ = cv2.connectedComponentsWithStats(opened, 8)
    out: list[LineCandidate] = []
    for i in range(1, n):
        x, y, bw, bh, _ = stats[i]
        if orientation == "h":
            if bh > max(12, int(h * 0.025)) or bw < run:
                continue
            const, lo, hi = y + bh / 2.0, x, x + bw
        else:
            if bw > max(12, int(w * 0.025)) or bh < run:
                continue
            const, lo, hi = x + bw / 2.0, y, y + bh
        support = _line_support(binary, orientation, const, lo, hi)
        if 0.015 <= support <= 0.78:
            pattern = _line_pattern(binary, orientation, const, lo, hi)
            out.append(LineCandidate(orientation, float(const), float(lo), float(hi), 3, support, 0.20 + 0.45 * pattern + 0.35 * min(1.0, support * 3), pattern))
    return out


def _dedupe_lines(lines: Iterable[LineCandidate]) -> list[LineCandidate]:
    kept: list[LineCandidate] = []
    for line in sorted(lines, key=lambda x: x.score, reverse=True):
        duplicate = False
        for old in kept:
            overlap = max(0.0, min(line.hi, old.hi) - max(line.lo, old.lo))
            union = max(line.hi, old.hi) - min(line.lo, old.lo)
            if abs(line.const - old.const) <= 4 and overlap / max(union, 1.0) >= 0.70:
                duplicate = True
                break
        if not duplicate:
            kept.append(line)
    return kept


def _corner_cost(hline: LineCandidate, vline: LineCandidate, x: float, y: float, tol: float) -> float:
    # A line can extend past a corner when a neighbouring enclosure shares the
    # same scan row/column.  Distance to the span handles that case, while the
    # tolerance still covers a short bent corner connector.
    dx = max(hline.lo - x, 0.0, x - hline.hi)
    dy = max(vline.lo - y, 0.0, y - vline.hi)
    return min(1.0, (dx + dy) / max(2.0 * tol, 1.0))


def _panel_candidates(binary: np.ndarray, rows: list[LineCandidate], cols: list[LineCandidate]) -> list[dict]:
    h, w = binary.shape
    # Allow the short bent connector used at many panel corners.
    tol = max(5.0, min(h, w) * 0.060)
    min_width, min_height = max(28.0, w * 0.035), max(28.0, h * 0.035)
    candidates: list[dict] = []
    def endpoint_columns(line: LineCandidate) -> list[LineCandidate]:
        available = [c for c in cols if line.lo - tol <= c.const <= line.hi + tol]
        if len(available) <= 24:
            return available
        ranked = sorted(available, key=lambda c: c.score, reverse=True)
        selected = ranked[:12]
        selected += sorted(available, key=lambda c: abs(c.const - line.lo))[:6]
        selected += sorted(available, key=lambda c: abs(c.const - line.hi))[:6]
        unique = []
        for item in selected:
            if not any(abs(item.const - old.const) <= 2 and abs(item.lo - old.lo) <= 4 for old in unique):
                unique.append(item)
        return unique
    endpoint_map = {id(line): endpoint_columns(line) for line in rows}

    for top in rows:
        for bottom in rows:
            if bottom.const <= top.const + min_height:
                continue
            # A rectangle corner must be close to an endpoint of each side.
            # Restricting the search here prevents text-derived lines from
            # causing an O(rows^2 * cols^2) explosion.
            top_cols = endpoint_map[id(top)]
            bottom_cols = endpoint_map[id(bottom)]
            corner_cols = [c for c in top_cols if any(abs(c.const - d.const) <= tol for d in bottom_cols)]
            for left in corner_cols:
                for right in corner_cols:
                    if right.const <= left.const + min_width:
                        continue
                    x1, x2 = left.const, right.const
                    y1, y2 = top.const, bottom.const
                    if x1 < 0 or y1 < 0 or x2 > w or y2 > h:
                        continue
                    corners = [
                        _corner_cost(top, left, x1, y1, tol),
                        _corner_cost(top, right, x2, y1, tol),
                        _corner_cost(bottom, left, x1, y2, tol),
                        _corner_cost(bottom, right, x2, y2, tol),
                    ]
                    if max(corners) > 1.0:
                        continue
                    # Require each side to reach close to both corners.
                    span_checks = [
                        (top.lo <= x1 + tol and top.hi >= x2 - tol),
                        (bottom.lo <= x1 + tol and bottom.hi >= x2 - tol),
                        (left.lo <= y1 + tol and left.hi >= y2 - tol),
                        (right.lo <= y1 + tol and right.hi >= y2 - tol),
                    ]
                    if sum(span_checks) < 4:
                        continue
                    width, height = x2 - x1, y2 - y1
                    # The labelled terminal boxes in these SLDs are compact
                    # enclosures. Page frames and title blocks are much taller
                    # (or nearly full-page) and are not panels.
                    if height > 0.30 * h or width > 0.98 * w:
                        continue
                    score = (
                        0.25 * (top.score + bottom.score + left.score + right.score) / 4.0
                        + 0.35 * (1.0 - float(np.mean(corners)))
                        + 0.20 * sum(span_checks) / 4.0
                        + 0.20 * min(1.0, width / w) * min(1.0, height / h)
                    )
                    has_zz = any(s.has_zigzag for s in (top, bottom, left, right))
                    border = "zigzag" if has_zz else "dashed"
                    candidates.append({"x": int(round(x1)), "y": int(round(y1)), "width": int(round(width)), "height": int(round(height)), "score": float(score), "border_type": border, "has_continuation": has_zz})
    # Prefer the complete enclosure when several candidates are built from
    # nested fragments of the same four edges.
    candidates.sort(key=lambda p: (p["width"] * p["height"], p["score"]), reverse=True)
    kept: list[dict] = []
    for candidate in candidates:
        if any(_iou(candidate, old) >= 0.70 for old in kept):
            continue
        kept.append(candidate)
    # A large rectangle and a smaller rectangle produced from the same edge
    # fragments are duplicates, not separate panels.
    non_nested: list[dict] = []
    for candidate in sorted(kept, key=lambda p: p["width"] * p["height"], reverse=True):
        nested = False
        ca = candidate["width"] * candidate["height"]
        for old in non_nested:
            ix1 = max(candidate["x"], old["x"])
            iy1 = max(candidate["y"], old["y"])
            ix2 = min(candidate["x"] + candidate["width"], old["x"] + old["width"])
            iy2 = min(candidate["y"] + candidate["height"], old["y"] + old["height"])
            inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
            if inter / float(max(ca, 1)) >= 0.82:
                nested = True
                break
        if not nested:
            non_nested.append(candidate)
    return sorted(non_nested, key=lambda p: (p["y"], p["x"]))


def _iou(a: dict, b: dict) -> float:
    x1 = max(a["x"], b["x"])
    y1 = max(a["y"], b["y"])
    x2 = min(a["x"] + a["width"], b["x"] + b["width"])
    y2 = min(a["y"] + a["height"], b["y"] + b["height"])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    aa = a["width"] * a["height"]
    bb = b["width"] * b["height"]
    return inter / float(max(aa + bb - inter, 1))


def _horizontal_strokes(gray: np.ndarray, binary: np.ndarray, panel: dict) -> list[dict]:
    x, y, w, h = panel["x"], panel["y"], panel["width"], panel["height"]
    inset_x = max(3, int(round(w * 0.035)))
    inset_y = max(3, int(round(h * 0.08)))
    x1, x2 = x + inset_x, x + w - inset_x
    y1, y2 = y + inset_y, y + h - inset_y
    if x2 <= x1 or y2 <= y1:
        return []
    roi = binary[y1:y2, x1:x2]
    min_len = max(10, int(round(w * 0.10)))
    opened = cv2.morphologyEx(roi, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (min_len, 1)))
    n, _, stats, _ = cv2.connectedComponentsWithStats(opened, 8)
    distance = cv2.distanceTransform(binary, cv2.DIST_L2, 3)
    result: list[dict] = []
    for i in range(1, n):
        bx, by, bw, bh, area = stats[i]
        if bw < min_len or bh > max(12, int(round(h * 0.12))) or area < max(6, min_len):
            continue
        yy = y1 + by + bh // 2
        xx1, xx2 = x1 + bx, x1 + bx + bw
        samples = np.linspace(xx1, xx2 - 1, max(5, min(15, bw))).astype(int)
        thickness = float(np.median(np.maximum(1.0, distance[yy, samples] * 2.0)))
        local = binary[max(0, yy - 2):min(binary.shape[0], yy + 3), xx1:xx2]
        support = float(np.count_nonzero(local)) / float(max(local.size, 1))
        span_ratio = bw / float(max(w, 1))
        score = 0.60 * min(1.0, span_ratio) + 0.25 * min(1.0, thickness / 4.0) + 0.15 * min(1.0, support * 4.0)
        result.append({"x": int(xx1), "y": int(yy - max(1, round(thickness / 2))), "width": int(bw), "height": max(1, int(round(thickness))), "score": float(score), "thickness": thickness})
    return result


def _is_table_grid(binary: np.ndarray, panel: dict) -> bool:
    """Reject title blocks/legends without relying on fixed page coordinates."""
    x, y, w, h = panel["x"], panel["y"], panel["width"], panel["height"]
    inset = max(3, int(round(min(w, h) * 0.06)))
    roi = binary[y + inset:y + h - inset, x + inset:x + w - inset]
    if roi.size == 0:
        return False
    rh, rw = roi.shape
    hmask = cv2.morphologyEx(roi, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (max(12, int(rw * 0.55)), 1)))
    vmask = cv2.morphologyEx(roi, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(12, int(rh * 0.55)))))
    hn, _, hstats, _ = cv2.connectedComponentsWithStats(hmask, 8)
    vn, _, vstats, _ = cv2.connectedComponentsWithStats(vmask, 8)
    horizontal = sum(1 for i in range(1, hn) if hstats[i, cv2.CC_STAT_WIDTH] >= 0.55 * rw)
    vertical = sum(1 for i in range(1, vn) if vstats[i, cv2.CC_STAT_HEIGHT] >= 0.55 * rh)
    return horizontal >= 3 and vertical >= 3


def _detect_bracket_panels(binary: np.ndarray) -> list[dict]:
    """Detect panels whose borders are bracket-shaped dashed/solid lines.

    Many SLD panels use dashed horizontal top/bottom borders with short
    vertical returns at the corners (bracket shape).  The vertical sides
    may be solid lines with zigzag continuation breaks, dashed lines, or
    simply absent (open sides).  This detector finds horizontal border rows
    where multiple segments define individual panel x-extents, then pairs
    them with a bottom border row to produce panel rectangles.
    """
    h, w = binary.shape

    # --- find horizontal line segments, bridging dash gaps ---
    min_seg = max(8, int(w * 0.005))
    hkernel = cv2.getStructuringElement(cv2.MORPH_RECT, (min_seg, 1))
    hlines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, hkernel)
    bridge = max(5, int(w * 0.008))
    hbridge = cv2.morphologyEx(
        hlines, cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (bridge, 1)),
    )
    n_h, _, stats_h, _ = cv2.connectedComponentsWithStats(hbridge, 8)
    segments: list[tuple[float, float, float, float]] = []
    for i in range(1, n_h):
        x, y, bw, bh, _ = stats_h[i]
        if bw > w * 0.04 and bh < max(10, h * 0.015):
            segments.append((y + bh / 2.0, float(x), float(x + bw), float(bw)))

    # --- group segments into rows by y-coordinate ---
    segments.sort(key=lambda s: s[0])
    y_groups: list[list[tuple[float, float, float, float]]] = []
    for seg in segments:
        placed = False
        for grp in y_groups:
            if abs(seg[0] - float(np.mean([s[0] for s in grp]))) < max(6, h * 0.005):
                grp.append(seg)
                placed = True
                break
        if not placed:
            y_groups.append([seg])

    # --- identify candidate border rows ---
    class _Row:
        def __init__(self, y: float, segs: list[tuple[float, float, float, float]]):
            self.y = y
            self.segs = sorted(segs, key=lambda s: s[1])
            self.min_x = min(s[1] for s in segs)
            self.max_x = max(s[2] for s in segs)
            self.span = self.max_x - self.min_x
            self.n = len(segs)

    rows: list[_Row] = []
    for grp in y_groups:
        y_avg = float(np.mean([s[0] for s in grp]))
        r = _Row(y_avg, grp)
        if r.span > w * 0.25 and 0.03 * h < y_avg < 0.85 * h:
            rows.append(r)
    rows.sort(key=lambda r: r.y)

    # --- check for vertical return at a segment endpoint (corner bracket) ---
    def _has_vertical_return(x: float, y: float) -> bool:
        xi, yi = int(round(x)), int(round(y))
        radius = max(2, int(min(h, w) * 0.003))
        for dy_sign in (1, -1):
            strip_y0 = yi if dy_sign == 1 else max(0, yi - 40)
            strip_y1 = min(h, yi + 40) if dy_sign == 1 else yi + 1
            strip = binary[strip_y0:strip_y1,
                           max(0, xi - radius):min(w, xi + radius + 1)]
            if strip.size == 0:
                continue
            profile = np.any(strip > 0, axis=1).astype(np.uint8)
            run = 0
            for v in profile if dy_sign == 1 else reversed(profile):
                if v:
                    run += 1
                else:
                    break
            if run > max(6, h * 0.005):
                return True
        return False

    # --- check gap between two segments for zigzag / vertical divider ---
    def _gap_has_border(x1: float, x2: float, y_top: float, y_bot: float) -> bool:
        if x2 - x1 < 2:
            return True
        xi1, xi2 = int(round(x1)), int(round(x2))
        yi1, yi2 = int(round(y_top)), int(round(y_bot))
        roi = binary[yi1:yi2, max(0, xi1):min(w, xi2)]
        if roi.size == 0:
            return False
        return float(np.count_nonzero(roi)) / roi.size > 0.03

    # --- assemble panels from segmented top rows paired with bottom rows ---
    panels: list[dict] = []
    used_top_rows: set[int] = set()

    for ti, top_row in enumerate(rows):
        if top_row.n < 2:
            continue
        # Find the best bottom row below
        best_bot = None
        best_score = -1.0
        for bi, bot_row in enumerate(rows):
            if bot_row.y <= top_row.y + h * 0.04:
                continue
            panel_h = bot_row.y - top_row.y
            if panel_h < h * 0.05 or panel_h > h * 0.55:
                continue
            x_overlap = min(top_row.max_x, bot_row.max_x) - max(top_row.min_x, bot_row.min_x)
            if x_overlap < top_row.span * 0.5:
                continue
            score = x_overlap / max(top_row.span, 1.0) + min(1.0, panel_h / (h * 0.3))
            if score > best_score:
                best_score = score
                best_bot = bot_row
        if best_bot is None:
            continue

        panel_height = int(round(best_bot.y - top_row.y))
        py = int(round(top_row.y))

        for seg in top_row.segs:
            seg_w = seg[2] - seg[1]
            if seg_w < w * 0.03:
                continue
            has_bracket = (_has_vertical_return(seg[1], seg[0])
                          or _has_vertical_return(seg[2], seg[0]))
            if not has_bracket and seg_w < w * 0.10:
                continue
            # This detector's top/bottom borders are dashed by definition;
            # reject a uniform dash rhythm on either one the same way the
            # other panel-detection paths do.
            if not (_is_uneven_dash_pattern(binary, "h", seg[0], seg[1], seg[2])
                    and _is_uneven_dash_pattern(binary, "h", best_bot.y, seg[1], seg[2])):
                continue

            # Determine border type for left / right sides
            left_zz = False
            right_zz = False
            idx = top_row.segs.index(seg)
            if idx > 0:
                prev_seg = top_row.segs[idx - 1]
                left_zz = _gap_has_border(prev_seg[2], seg[1], top_row.y, best_bot.y)
            if idx < len(top_row.segs) - 1:
                next_seg = top_row.segs[idx + 1]
                right_zz = _gap_has_border(seg[2], next_seg[1], top_row.y, best_bot.y)

            has_cont = left_zz or right_zz
            border = "zigzag" if has_cont else "dashed"
            panels.append({
                "x": int(round(seg[1])),
                "y": py,
                "width": int(round(seg_w)),
                "height": panel_height,
                "score": 0.85 if has_bracket else 0.70,
                "border_type": border,
                "has_continuation": has_cont,
            })
        used_top_rows.add(ti)

    return panels


def _is_dashed_horizontal(binary: np.ndarray, x: int, y: int, width: int) -> bool:
    """Check that a horizontal candidate is made from repeated short dashes.

    The continuation enclosure uses dotted horizontal edges.  Closing those
    gaps is useful for finding the edge, but this check is deliberately made
    on the original binary image so a normal solid rule is not mistaken for a
    panel boundary.
    """
    h, w = binary.shape
    radius = max(1, int(round(min(h, w) * 0.0015)))
    x0, x1 = max(0, x), min(w, x + width)
    strip = binary[max(0, y - radius):min(h, y + radius + 1), x0:x1]
    if strip.size == 0:
        return False
    profile = np.any(strip > 0, axis=0).astype(np.uint8)
    padded = np.pad(profile, (1, 1))
    starts = np.flatnonzero((padded[1:-1] == 1) & (padded[:-2] == 0))
    ends = np.flatnonzero((padded[1:-1] == 1) & (padded[2:] == 0))
    if len(starts) < max(5, width // 120) or len(ends) != len(starts):
        return False
    runs = ends - starts + 1
    gaps = starts[1:] - ends[:-1] - 1
    if len(gaps) == 0:
        return False
    # A few long runs can occur where wiring crosses the border.  The median
    # still represents the underlying repeated dash pattern.
    basic_ok = bool(
        np.median(runs) <= max(12, width * 0.035)
        and 1 <= np.median(gaps) <= max(20, width * 0.035)
    )
    if not basic_ok:
        return False
    # Additionally reject uniform dash patterns (all dashes the same size).
    # Real panel borders use uneven patterns (big-small-small-big etc.).
    return _is_uneven_dash_pattern(binary, "h", float(y), float(x0), float(x1))


def _dashed_horizontal_edges(binary: np.ndarray) -> list[tuple[int, int, int]]:
    """Return long dotted horizontal edges as ``(x, y, width)`` tuples."""
    h, w = binary.shape
    scale = min(h, w)
    # The source drawings have small dash gaps.  Bridge them only to obtain a
    # line candidate; _is_dashed_horizontal verifies the unbridged pattern.
    bridge = max(3, int(round(scale * 0.007)))
    min_length = max(60, int(round(w * 0.08)))
    joined = cv2.morphologyEx(
        binary,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (bridge, 1)),
    )
    lines = cv2.morphologyEx(
        joined,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (min_length, 1)),
    )
    count, _, stats, _ = cv2.connectedComponentsWithStats(lines, 8)
    edges: list[tuple[int, int, int]] = []
    for index in range(1, count):
        x, y, bw, bh, _ = stats[index]
        if bw < min_length or bh > max(12, int(round(scale * 0.012))):
            continue
        center_y = int(round(y + bh / 2.0))
        if _is_dashed_horizontal(binary, int(x), center_y, int(bw)):
            edges.append((int(x), center_y, int(bw)))

    # A dotted rule can be two or three pixels high; retain one representative
    # when morphology returns overlapping copies of that same rule.
    deduped: list[tuple[int, int, int]] = []
    for edge in sorted(edges, key=lambda item: item[2], reverse=True):
        x, y, bw = edge
        if any(abs(y - old_y) <= 4 and _interval_iou(x, x + bw, old_x, old_x + old_bw) >= 0.85
               for old_x, old_y, old_bw in deduped):
            continue
        deduped.append(edge)
    return sorted(deduped, key=lambda item: item[1])


def _interval_iou(a1: int, a2: int, b1: int, b2: int) -> float:
    overlap = max(0, min(a2, b2) - max(a1, b1))
    return overlap / float(max((a2 - a1) + (b2 - b1) - overlap, 1))


def _continuation_side(
    binary: np.ndarray,
    side_x: int,
    top_y: int,
    bottom_y: int,
    vertical_strokes: list[tuple[int, int, int, int]],
) -> bool:
    """Recognize the vertical-line / zig-zag / vertical-line continuation mark.

    A continuation side has two long vertical legs with a short break between
    them.  The break contains one or more oblique strokes (the sideways Z).
    Looking for this topology, rather than a fixed bitmap template, supports
    the mirrored left/right symbols and different raster resolutions.
    """
    h, w = binary.shape
    panel_h = bottom_y - top_y
    scale = min(h, w)
    edge_tol = max(8, int(round(scale * 0.018)))
    min_leg = max(16, int(round(panel_h * 0.12)))
    min_coverage = max(12, int(round(panel_h * 0.10)))
    max_notch_h = max(18, int(round(min(panel_h * 0.30, scale * 0.10))))
    max_shift = max(14, int(round(scale * 0.050)))

    nearby = [
        stroke for stroke in vertical_strokes
        if abs((stroke[0] + stroke[2] / 2.0) - side_x) <= edge_tol
        and stroke[3] >= min_leg
        and stroke[1] < bottom_y - min_coverage
        and stroke[1] + stroke[3] > top_y + min_coverage
    ]
    uppers = [
        stroke for stroke in nearby
        if stroke[1] <= top_y + edge_tol
        and stroke[1] + stroke[3] >= top_y + min_coverage
    ]
    lowers = [
        stroke for stroke in nearby
        if stroke[1] <= bottom_y - min_coverage
        and stroke[1] + stroke[3] >= bottom_y - edge_tol
    ]

    for upper in uppers:
        upper_end = upper[1] + upper[3]
        for lower in lowers:
            lower_start = lower[1]
            gap = lower_start - upper_end
            if not 2 <= gap <= max_notch_h:
                continue
            upper_x = upper[0] + upper[2] / 2.0
            lower_x = lower[0] + lower[2] / 2.0
            if abs(upper_x - lower_x) > max_shift:
                continue

            # The legs alone could be ordinary interrupted wiring.  Require
            # an oblique stroke in the break to prove it is the Z marker.
            x0 = max(0, int(round(min(upper_x, lower_x) - max_shift)))
            x1 = min(w, int(round(max(upper_x, lower_x) + max_shift + 1)))
            y0 = max(0, upper_end - 4)
            y1 = min(h, lower_start + 5)
            roi = binary[y0:y1, x0:x1]
            if roi.size == 0:
                continue
            found = cv2.HoughLinesP(
                roi,
                rho=1,
                theta=np.pi / 180,
                threshold=5,
                minLineLength=max(5, int(round(min(max_shift, max_notch_h) * 0.25))),
                maxLineGap=3,
            )
            if found is None:
                continue
            for x_a, y_a, x_b, y_b in found.reshape(-1, 4):
                dx, dy = abs(int(x_b) - int(x_a)), abs(int(y_b) - int(y_a))
                if dx >= 3 and dy >= 3 and 0.20 <= dx / float(dy) <= 5.0:
                    return True
    return False


def _detect_continuation_panels(binary: np.ndarray) -> list[dict]:
    """Detect dotted-top/bottom panels closed by one or two Z continuation sides."""
    h, w = binary.shape
    scale = min(h, w)
    edges = _dashed_horizontal_edges(binary)
    if len(edges) < 2:
        return []

    leg_length = max(12, int(round(scale * 0.020)))
    vertical = cv2.morphologyEx(
        binary,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (1, leg_length)),
    )
    count, _, stats, _ = cv2.connectedComponentsWithStats(vertical, 8)
    vertical_strokes = [
        (int(stats[index, 0]), int(stats[index, 1]), int(stats[index, 2]), int(stats[index, 3]))
        for index in range(1, count)
        if stats[index, 3] >= leg_length
        and stats[index, 2] <= max(10, int(round(scale * 0.010)))
    ]

    panels: list[dict] = []
    for top_x, top_y, top_w in edges:
        for bottom_x, bottom_y, bottom_w in edges:
            panel_h = bottom_y - top_y
            if panel_h < max(40, int(round(h * 0.04))) or panel_h > int(round(h * 0.60)):
                continue
            x1 = max(top_x, bottom_x)
            x2 = min(top_x + top_w, bottom_x + bottom_w)
            if x2 - x1 < max(80, int(round(w * 0.08))):
                continue
            # The same panel border is expected on both rows, so most of each
            # dotted span should overlap.  This prevents unrelated rules from
            # being paired across a drawing.
            if _interval_iou(top_x, top_x + top_w, bottom_x, bottom_x + bottom_w) < 0.65:
                continue
            left = _continuation_side(binary, x1, top_y, bottom_y, vertical_strokes)
            right = _continuation_side(binary, x2, top_y, bottom_y, vertical_strokes)
            if not (left or right):
                continue
            sides = (["left"] if left else []) + (["right"] if right else [])
            panels.append({
                "x": x1,
                "y": top_y,
                "width": x2 - x1,
                "height": panel_h,
                "score": 0.95 if left and right else 0.88,
                "border_type": "dashed_with_continuation",
                "has_continuation": True,
                "continuation_sides": sides,
            })

    # There can be two almost-identical candidates from a multi-pixel dashed
    # edge.  Keep the best one while retaining separate adjacent panels.
    result: list[dict] = []
    for panel in sorted(panels, key=lambda item: item["score"], reverse=True):
        if any(_iou(panel, previous) >= 0.80 for previous in result):
            continue
        result.append(panel)
    return sorted(result, key=lambda item: (item["y"], item["x"]))


def _detect_native_resolution(image: np.ndarray) -> dict:
    """Structural path for source-resolution drawings.

    Dashes in the Roboflow export are frequently reduced to one pixel.  At
    native resolution they are intact, so the graph method can verify four
    real border sides rather than inferring rectangles from text and wiring.
    """
    gray, binary = _graph_preprocess_image(image)
    h, w = gray.shape
    scale = min(h, w)
    rects = _detect_panels_graph(
        gray,
        binary,
        w,
        h,
        corner_tol=max(10, int(round(scale * 0.014))),
        min_size=max(35, int(round(scale * 0.035))),
        max_size_ratio=0.90,
        min_area_ratio=0.0001,
    )
    # A separate panel is never nested inside another panel in this task;
    # nested cycles are normally switch/symbol detail inside an enclosure.
    outer_first = sorted(rects, key=lambda r: (r[2] - r[0]) * (r[3] - r[1]), reverse=True)
    def mostly_inside(inner, outer):
        ix1, iy1 = max(inner[0], outer[0]), max(inner[1], outer[1])
        ix2, iy2 = min(inner[2], outer[2]), min(inner[3], outer[3])
        overlap = max(0, ix2 - ix1) * max(0, iy2 - iy1)
        area = max(1, (inner[2] - inner[0]) * (inner[3] - inner[1]))
        return overlap / area >= 0.70
    rects = [r for index, r in enumerate(outer_first) if not any(mostly_inside(r, outer) for outer in outer_first[:index])]
    # Reject rectangles whose dashed sides use a uniform dash rhythm
    # (dimension lines / hatching); `detect_panels_graph` only checks that a
    # side looks broken-up at all, not whether the break pattern is uneven.
    rects = [r for r in rects if _rect_sides_uneven(binary, *r)]
    thick_segments = _detect_thick_horizontal_segments(
        gray,
        binary,
        # Source PDF/raster exports can render a visually bold busbar as only
        # two pixels high; relative thickness is evaluated within the panel.
        min_thickness=2.0,
        min_length=max(8, int(round(scale * 0.006))),
    )
    panels = []
    for x1, y1, x2, y2 in rects:
        found = _longest_busbar_for_panel(binary, thick_segments, (x1, y1, x2, y2))
        bar = None
        if found:
            bx1, by1, bx2, by2, _ = found[0]
            bar = {
                "x": int(bx1),
                "y": int(by1),
                "width": int(bx2 - bx1),
                "height": max(1, int(round(scale * 0.002))),
                "score": 1.0,
            }
        panels.append({
            "x": int(x1), "y": int(y1), "width": int(x2 - x1), "height": int(y2 - y1),
            "score": 1.0, "busbar_candidates": [bar] if bar else [], "busbar": bar,
            "table_like": False,
            "has_continuation": False, "continuation_sides": [],
        })

    # Continuation enclosures have dotted top/bottom borders and one or two
    # vertical Z-shaped sides, so they cannot form a four-straight-edge graph
    # cycle.  Detect them explicitly before considering the looser bracket
    # fallback below.
    continuation_panels = _detect_continuation_panels(binary)
    # Bracket panels are a compatibility fallback for drawings which have
    # dashed top/bottom borders but no continuation symbol.  Restrict it to
    # drawings for which the stricter structural methods found nothing: its
    # intentionally open-side definition is otherwise prone to enclosing
    # unrelated wiring between two long horizontal rules.
    bracket_panels = continuation_panels
    if not panels and not continuation_panels:
        bracket_panels += _detect_bracket_panels(binary)
    for bp in bracket_panels:
        if any(_iou(bp, p) >= 0.30 for p in panels):
            continue
        found = _longest_busbar_for_panel(binary, thick_segments,
                                          (bp["x"], bp["y"],
                                           bp["x"] + bp["width"],
                                           bp["y"] + bp["height"]))
        bar = None
        if found:
            bx1, by1, bx2, by2, _ = found[0]
            bar = {
                "x": int(bx1), "y": int(by1),
                "width": int(bx2 - bx1),
                "height": max(1, int(round(scale * 0.002))),
                "score": 1.0,
            }
        bp["busbar_candidates"] = [bar] if bar else []
        bp["busbar"] = bar
        bp["table_like"] = False
        # Ensure continuation_sides is set (already present from
        # _detect_continuation_panels; set default for bracket panels).
        bp.setdefault("has_continuation", False)
        bp.setdefault("continuation_sides", [])
        panels.append(bp)

    # --- Post-processing filters ---
    # 1. Remove panels with no internal content (no busbar or line).
    panels = [p for p in panels if _has_internal_content(binary, p)]
    # 2. Suppress overlapping panels (keep larger/higher-confidence).
    panels = _suppress_overlapping_panels(panels)
    # 3. Normalise continuation_sides into a human-readable "continuation"
    #    field: "left", "right", "both", or None.
    for p in panels:
        sides = p.get("continuation_sides", [])
        if "left" in sides and "right" in sides:
            p["continuation"] = "both"
        elif "left" in sides:
            p["continuation"] = "left"
        elif "right" in sides:
            p["continuation"] = "right"
        else:
            p["continuation"] = None

    return {"panels": panels, "all_panels": panels, "rows": [], "columns": []}


def detect(image: np.ndarray) -> dict:
    if min(image.shape[:2]) >= 900:
        return _detect_native_resolution(image)
    gray, binary = _gray_binary(image)
    horizontal, vertical = _lsd_segments(gray)
    rows = _dedupe_lines(_cluster_segments(horizontal, "h", binary) + _morph_line_candidates(binary, "h"))
    cols = _dedupe_lines(_cluster_segments(vertical, "v", binary) + _morph_line_candidates(binary, "v"))
    # Keep the strongest structural candidates. Shorter lines are retained by
    # the span term, so genuine small boxes are not lost to page furniture.
    row_ranked = sorted(rows, key=lambda x: (x.score, x.hi - x.lo), reverse=True)
    col_ranked = sorted(cols, key=lambda x: (x.score, x.hi - x.lo), reverse=True)
    rows = row_ranked[:60] + [x for x in row_ranked[60:] if x.hi - x.lo >= 0.40 * image.shape[1]][:20]
    cols = col_ranked[:70] + [x for x in col_ranked[70:] if x.hi - x.lo >= 0.40 * image.shape[0]][:20]
    panels = _panel_candidates(binary, rows, cols)
    for panel in panels:
        panel["table_like"] = _is_table_grid(binary, panel)

    h, w = binary.shape
    global_bars = _horizontal_strokes(gray, binary, {"x": 0, "y": 0, "width": w, "height": h})
    global_bars = [b for b in global_bars if b["width"] >= max(30, int(w * 0.07)) and b["score"] >= 0.45]
    for panel in panels:
        if panel["table_like"]:
            panel["busbar_candidates"] = []
            panel["busbar"] = None
            continue
        tolerance = max(4, int(round(min(panel["width"], panel["height"]) * 0.04)))
        bars = [
            b for b in global_bars
            if panel["x"] - tolerance <= b["x"]
            and b["x"] + b["width"] <= panel["x"] + panel["width"] + tolerance
            and panel["y"] + tolerance <= b["y"] <= panel["y"] + panel["height"] - tolerance
        ]
        panel["busbar_candidates"] = bars
        panel["busbar"] = max(bars, key=lambda b: b["score"]) if bars else None
    # Ensure every panel has continuation_sides defaulted.
    for panel in panels:
        panel.setdefault("has_continuation", False)
        panel.setdefault("continuation_sides", [])

    # --- Post-processing filters ---
    # 1. Remove panels with no internal content (no busbar or line).
    panels = [p for p in panels if _has_internal_content(binary, p)]
    # 2. Suppress overlapping panels (keep larger/higher-confidence).
    panels = _suppress_overlapping_panels(panels)
    # 3. Normalise continuation_sides into a human-readable "continuation" field.
    for p in panels:
        sides = p.get("continuation_sides", [])
        if "left" in sides and "right" in sides:
            p["continuation"] = "both"
        elif "left" in sides:
            p["continuation"] = "left"
        elif "right" in sides:
            p["continuation"] = "right"
        else:
            p["continuation"] = None

    # `panels` has already passed the internal-content filter (busbar OR a
    # qualifying line) and panel-vs-panel overlap suppression above, so no
    # panel is dropped here for lacking a busbar -- a panel with only a
    # qualifying line is still valid content and stays in the result.
    # The only thing left to dedupe is two different rectangle hypotheses
    # that both happen to enclose the *same physical busbar*; keep the
    # higher-scoring one of those.
    paired = []
    for panel in sorted(panels, key=lambda p: p["score"], reverse=True):
        bar = panel.get("busbar")
        if bar:
            cx = bar["x"] + bar["width"] / 2.0
            cy = bar["y"] + bar["height"] / 2.0
            duplicate_bar = any(
                old.get("busbar") is not None
                and old["x"] <= cx <= old["x"] + old["width"]
                and old["y"] <= cy <= old["y"] + old["height"]
                for old in paired
            )
            if duplicate_bar:
                continue
        paired.append(panel)
    return {
        "panels": paired,
        "all_panels": panels,
        "rows": [asdict(x) for x in rows],
        "columns": [asdict(x) for x in cols],
    }


def annotate(image: np.ndarray, result: dict, draw_candidates: bool = False) -> np.ndarray:
    out = image.copy()
    for idx, panel in enumerate(result["panels"], 1):
        p1 = (panel["x"], panel["y"])
        p2 = (panel["x"] + panel["width"], panel["y"] + panel["height"])
        cv2.rectangle(out, p1, p2, (255, 0, 0), 2)
        # Build label: P1, P1[L], P1[R], P1[L+R]
        label = f"P{idx}"
        cont = panel.get("continuation")
        if cont == "left":
            label += "[L]"
        elif cont == "right":
            label += "[R]"
        elif cont == "both":
            label += "[L+R]"
        cv2.putText(out, label, (p1[0], max(15, p1[1] - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 0, 0), 1, cv2.LINE_AA)
        bars = panel.get("busbar_candidates", []) if draw_candidates else ([panel["busbar"]] if panel.get("busbar") else [])
        for bar in bars:
            cv2.rectangle(out, (bar["x"], bar["y"]), (bar["x"] + bar["width"], bar["y"] + bar["height"]), (0, 170, 0), 2)
    return out


def detect_file(path: str | Path) -> tuple[np.ndarray, dict]:
    image = cv2.imread(str(path))
    if image is None:
        raise FileNotFoundError(path)
    return image, detect(image)
