"""
Busbar and Panel Detector v4
=============================
Two logic changes from v3, per updated spec:

  1. BUSBAR SELECTION: exactly ONE busbar per panel -- the longest
     horizontal thick solid stroke inside it (small gaps from crossing
     component legs are bridged first so a busbar broken by breakers
     still measures as one continuous run).

  2. PANEL DETECTION: rebuilt as a graph-theory problem.
       - Candidate dash pieces (short LSD segments) are grouped into
         "dashed lines": rows (near-horizontal) and columns
         (near-vertical), each requiring several small, regularly-spread
         pieces -- a structural test that a solid stroke can't satisfy.
       - Each dashed line becomes a NODE in a graph. An EDGE connects a
         row-node and a column-node when they meet near a common point
         (the column's x lines up with one end of the row's span, and
         the row's y lines up with one end of the column's span) --
         i.e. they could be two adjacent sides of the same rectangle,
         meeting at a corner.
       - A panel is exactly a length-4 CYCLE in this graph: row - col -
         row - col - (back to the first row). Cycle-finding (via
         networkx) replaces the old approach of morphologically bridging
         dash gaps and hoping contour-detection forms one closed blob --
         which either over-merged adjacent panels (dashes bridged across
         the gap between two side-by-side panels) or missed panels whose
         border was interrupted by a crossing symbol. Expressing "closed
         rectangle" as a 4-cycle constraint is a direct structural
         definition of a panel rather than an image-blob heuristic, so it
         is robust to both failure modes: two independent rectangles
         never share the exact corner-meeting-points needed to form a
         cycle across each other, and a interrupted border still closes
         the cycle as long as its 4 sides each survive as one graph node
         (they don't need every dash bridged, just enough dashes to be
         recognized as a "row"/"column" in the first place).
"""

import argparse
import os
import numpy as np
import cv2
import networkx as nx


# --------------------------------------------------------------------------
# Preprocessing
# --------------------------------------------------------------------------

def preprocess_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    denoised = cv2.fastNlMeansDenoising(gray, h=7, templateWindowSize=7, searchWindowSize=21)
    binary = cv2.adaptiveThreshold(
        denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,
        blockSize=25, C=10
    )
    return gray, binary


def _run_lsd(gray):
    lsd = cv2.createLineSegmentDetector(cv2.LSD_REFINE_STD)
    lines = lsd.detect(gray)[0]
    if lines is None:
        return np.zeros((0, 4))
    return lines.reshape(-1, 4)


def _seg_angle(x1, y1, x2, y2):
    return np.degrees(np.arctan2(y2 - y1, x2 - x1)) % 180


# --------------------------------------------------------------------------
# PANEL DETECTION -- dashed-line grouping + graph cycle search
# --------------------------------------------------------------------------

def _dashed_lines(segs, pos_tol, short_seg_max, min_dashes, min_span_factor=0.4):
    """
    Group same-row/column segments within pos_tol, keep the group only if
    it's made of several SHORT pieces (a dash pattern) spread over a
    reasonable extent -- rejects both solid strokes (too few pieces) and
    tiny unrelated tick-mark clusters (too little spread).
    Returns list of (const_coord, span_lo, span_hi).
    """
    if not segs:
        return []
    segs = sorted(segs, key=lambda s: s[1])
    groups = []
    for s in segs:
        placed = False
        for g in groups:
            if abs(s[1] - g[0][1]) <= pos_tol:
                g.append(s)
                placed = True
                break
        if not placed:
            groups.append([s])

    lines = []
    for g in groups:
        short_pieces = [s for s in g if s[3] <= short_seg_max]
        if len(short_pieces) < min_dashes:
            continue
        const = float(np.median([s[1] for s in short_pieces]))
        lo = min(s[0] for s in short_pieces)
        hi = max(s[2] for s in short_pieces)
        if hi - lo < short_seg_max * min_dashes * min_span_factor:
            continue
        lines.append((const, lo, hi))
    return lines


def _directional_opening(binary_img, length, direction):
    length = max(3, int(length))
    if direction == 'h':
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (length, 1))
    else:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, length))
    return cv2.morphologyEx(binary_img, cv2.MORPH_OPEN, kernel)


def _find_dashed_lines_morph(binary_img, direction, min_run_ratio=0.01,
                              dash_gap_px=None, min_coverage=0.03, max_coverage=0.6):
    """
    Robust, scale-adaptive dashed-LINE finder (not yet a rectangle) used
    to generate graph nodes:
      1. Bridge small dash-gaps with a modest closing kernel.
      2. Directionally open with a much longer kernel to keep only
         pixels that are part of a substantial bridged run -- this finds
         BOTH busbars and panel borders (anything long), so:
      3. Check the ORIGINAL (unbridged) ink coverage along each run: near
         1.0 -> solid stroke (busbar/wire), reject; well below 1.0 ->
         actually made of dashes, keep as a panel-border candidate.
    Returns list of (const_coord, span_lo, span_hi) exactly like the old
    LSD-based `_dashed_lines`, but far less sensitive to how a specific
    drawing's LSD segments happen to fragment.
    """
    h, w = binary_img.shape
    if dash_gap_px is None:
        dash_gap_px = max(6, int(round(min(h, w) * 0.005)))
    bridge_kernel_size = dash_gap_px
    if direction == 'h':
        bridge_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (bridge_kernel_size, 3))
        min_run = max(30, int(w * min_run_ratio))
    else:
        bridge_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, bridge_kernel_size))
        min_run = max(30, int(h * min_run_ratio))

    bridged = cv2.morphologyEx(binary_img, cv2.MORPH_CLOSE, bridge_kernel, iterations=1)
    opened = _directional_opening(bridged, min_run, direction)

    num, labels, stats, _ = cv2.connectedComponentsWithStats(opened, connectivity=8)
    lines = []
    for i in range(1, num):
        x, y, bw, bh, area = stats[i]
        if direction == 'h':
            if bh > 25:
                continue
            const = y + bh / 2.0
            lo, hi = x, x + bw
            strip = binary_img[max(0, y - 2):y + bh + 2, lo:hi]
        else:
            if bw > 25:
                continue
            const = x + bw / 2.0
            lo, hi = y, y + bh
            strip = binary_img[lo:hi, max(0, x - 2):x + bw + 2]

        if strip.size == 0:
            continue
        coverage = float(np.count_nonzero(strip)) / strip.size
        if min_coverage <= coverage <= max_coverage:
            lines.append((float(const), int(lo), int(hi)))
    return lines


def _extract_dash_candidates(gray):
    lines = _run_lsd(gray)
    h_segs, v_segs = [], []
    for (x1, y1, x2, y2) in lines:
        length = np.hypot(x2 - x1, y2 - y1)
        if length < 4:
            continue
        angle = _seg_angle(x1, y1, x2, y2)
        if angle <= 3 or angle >= 177:
            lo, hi = (x1, x2) if x1 <= x2 else (x2, x1)
            h_segs.append((lo, (y1 + y2) / 2.0, hi, length))
        elif 87 <= angle <= 93:
            lo, hi = (y1, y2) if y1 <= y2 else (y2, y1)
            v_segs.append((lo, (x1 + x2) / 2.0, hi, length))
    return h_segs, v_segs


def _build_panel_graph(rows, cols, corner_tol):
    """
    Nodes: 'R{i}' for each dashed row (y, x_lo, x_hi), 'C{j}' for each
    dashed column (x, y_lo, y_hi). Edge R{i}-C{j} iff they plausibly meet
    at a shared corner: the column's x sits near one end of the row's
    x-span AND the row's y sits near one end of the column's y-span.
    """
    G = nx.Graph()
    for i, r in enumerate(rows):
        G.add_node(('R', i), data=r)
    for j, c in enumerate(cols):
        G.add_node(('C', j), data=c)

    for i, (ry, rlo, rhi) in enumerate(rows):
        for j, (cx, clo, chi) in enumerate(cols):
            x_at_end = (abs(cx - rlo) <= corner_tol) or (abs(cx - rhi) <= corner_tol)
            y_at_end = (abs(ry - clo) <= corner_tol) or (abs(ry - chi) <= corner_tol)
            if x_at_end and y_at_end:
                G.add_edge(('R', i), ('C', j))
    return G


def _rectangles_from_cycles(G, min_size, max_size_ratio, w_img, h_img):
    """Every length-4 cycle row-col-row-col in G is one panel candidate."""
    rects = []
    try:
        cycles = nx.simple_cycles(G, length_bound=4)
    except TypeError:
        cycles = (c for c in nx.simple_cycles(G) if len(c) == 4)

    for cycle in cycles:
        if len(cycle) != 4:
            continue
        r_nodes = [n for n in cycle if n[0] == 'R']
        c_nodes = [n for n in cycle if n[0] == 'C']
        if len(r_nodes) != 2 or len(c_nodes) != 2:
            continue
        (ry1, rlo1, rhi1) = G.nodes[r_nodes[0]]['data']
        (ry2, rlo2, rhi2) = G.nodes[r_nodes[1]]['data']
        (cx1, clo1, chi1) = G.nodes[c_nodes[0]]['data']
        (cx2, clo2, chi2) = G.nodes[c_nodes[1]]['data']

        y_top, y_bot = sorted((ry1, ry2))
        x_left, x_right = sorted((cx1, cx2))
        width, height = x_right - x_left, y_bot - y_top
        if width < min_size or height < min_size:
            continue
        if width > max_size_ratio * w_img or height > max_size_ratio * h_img:
            continue
        rects.append((int(x_left), int(y_top), int(x_right), int(y_bot)))
    return rects


def detect_panels_graph(gray, binary_img, w_img, h_img, corner_tol=25,
                         min_size=180, max_size_ratio=0.85,
                         min_area_ratio=0.0004, max_area_ratio=0.5):
    rows = _find_dashed_lines_morph(binary_img, 'h')
    cols = _find_dashed_lines_morph(binary_img, 'v')

    G = _build_panel_graph(rows, cols, corner_tol)
    rects = _rectangles_from_cycles(G, min_size, max_size_ratio, w_img, h_img)

    img_area = w_img * h_img
    rects = [r for r in rects
             if min_area_ratio * img_area <= (r[2] - r[0]) * (r[3] - r[1]) <= max_area_ratio * img_area]
    # Reject the fixed right-margin template sidebar and title-block area
    # (both are recurring drawing-template furniture, not real panels --
    # see v3 notes on the sidebar; the title block additionally clutters
    # this specific graph method with many small gridded table cells).
    rects = [r for r in rects if not (r[0] > 0.9 * w_img)]
    rects = [r for r in rects if not (r[0] > 0.8 * w_img and r[1] > 0.5 * h_img)]
    rects = _dedupe_rects(rects)
    return _remove_nested_rects(rects)


def _iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0


def _dedupe_rects(rects, iou_thresh=0.6):
    rects = sorted(rects, key=lambda r: (r[2] - r[0]) * (r[3] - r[1]), reverse=True)
    kept = []
    for r in rects:
        if not any(_iou(r, k) > iou_thresh for k in kept):
            kept.append(r)
    return kept


def _remove_nested_rects(rects, contain_thresh=0.92):
    if not rects:
        return []
    rects_sorted = sorted(rects, key=lambda r: (r[2] - r[0]) * (r[3] - r[1]), reverse=True)
    kept = []
    for r in rects_sorted:
        rx1, ry1, rx2, ry2 = r
        r_area = max(1, (rx2 - rx1) * (ry2 - ry1))
        nested = False
        for k in kept:
            kx1, ky1, kx2, ky2 = k
            ix1, iy1 = max(rx1, kx1), max(ry1, ky1)
            ix2, iy2 = min(rx2, kx2), min(ry2, ky2)
            inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
            if inter / r_area >= contain_thresh:
                nested = True
                break
        if not nested:
            kept.append(r)
    return kept


# --------------------------------------------------------------------------
# BUSBAR DETECTION -- single longest thick horizontal stroke per panel
# --------------------------------------------------------------------------

def _perp_thickness(binary_img, x1, y1, x2, y2, samples=7, max_probe=20):
    h, w = binary_img.shape
    xs = np.linspace(x1, x2, samples).astype(int)
    y_at = np.linspace(y1, y2, samples)
    thicknesses = []
    for x, y in zip(xs, y_at):
        y = int(round(y))
        if not (0 <= x < w and 0 <= y < h):
            continue
        lo = y
        while lo > max(0, y - max_probe) and binary_img[lo - 1, x] > 0:
            lo -= 1
        hi = y
        while hi < min(h - 1, y + max_probe) and binary_img[hi + 1, x] > 0:
            hi += 1
        thicknesses.append(hi - lo + 1)
    return float(np.median(thicknesses)) if thicknesses else 0.0


def detect_thick_horizontal_segments(gray, binary_img, min_thickness=5.0,
                                      min_length=18, angle_tol=3.0):
    lines = _run_lsd(gray)
    out = []
    for (x1, y1, x2, y2) in lines:
        length = np.hypot(x2 - x1, y2 - y1)
        if length < min_length:
            continue
        angle = _seg_angle(x1, y1, x2, y2)
        if not (angle <= angle_tol or angle >= 180 - angle_tol):
            continue
        thickness = _perp_thickness(binary_img, x1, y1, x2, y2)
        if thickness < min_thickness:
            continue
        y = (y1 + y2) / 2.0
        lo, hi = (x1, x2) if x1 <= x2 else (x2, x1)
        out.append((lo, y, hi, length))
    return out


def _cluster_rows(segs, y_tol):
    segs = sorted(segs, key=lambda s: s[1])
    rows = []
    for s in segs:
        placed = False
        for row in rows:
            if abs(s[1] - np.mean([r[1] for r in row])) <= y_tol:
                row.append(s)
                placed = True
                break
        if not placed:
            rows.append([s])
    return rows


def longest_busbar_for_panel(binary_img, all_segs, panel, y_tol=6, bridge_gap=55):
    """
    Bridge small gaps (component legs crossing the bar) within each row,
    then return ONLY the single longest resulting run in the whole panel
    -- that is "the busbar" for this panel.
    """
    x1, y1, x2, y2 = panel
    margin = max(4, int(min(x2 - x1, y2 - y1) * 0.02))
    ix1, iy1, ix2, iy2 = x1 + margin, y1 + margin, x2 - margin, y2 - margin

    inside = [s for s in all_segs if ix1 <= s[0] and s[2] <= ix2 and iy1 <= s[1] <= iy2]
    if not inside:
        return _fallback_longest(binary_img, panel)

    rows = _cluster_rows(inside, y_tol)
    best = None
    for row in rows:
        row = sorted(row, key=lambda s: s[0])
        y = float(np.mean([s[1] for s in row]))
        cur_lo, cur_hi = row[0][0], row[0][2]
        for s in row[1:]:
            if s[0] <= cur_hi + bridge_gap:
                cur_hi = max(cur_hi, s[2])
            else:
                if best is None or (cur_hi - cur_lo) > (best[2] - best[0]):
                    best = (cur_lo, y, cur_hi)
                cur_lo, cur_hi = s[0], s[2]
        if best is None or (cur_hi - cur_lo) > (best[2] - best[0]):
            best = (cur_lo, y, cur_hi)

    if best is None:
        return _fallback_longest(binary_img, panel)
    lo, y, hi = best
    return [(int(lo), int(round(y)), int(hi), int(round(y)), hi - lo)]


def _fallback_longest(binary_img, panel):
    """
    No thick-line candidate survived at all (busbar too fragmented for
    any run to register). Fall back to a per-row ink-density projection:
    the row whose ink spans the widest fraction of the panel, excluding
    rows right at the panel's own border.
    """
    x1, y1, x2, y2 = panel
    margin = max(4, int(min(x2 - x1, y2 - y1) * 0.02))
    ix1, iy1, ix2, iy2 = x1 + margin, y1 + margin, x2 - margin, y2 - margin
    interior = binary_img[iy1:iy2, ix1:ix2]
    if interior.size == 0:
        return []

    for length_ratio in (0.3, 0.15, 0.08):
        length = max(12, int((x2 - x1) * length_ratio))
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (length, 1))
        opened = cv2.morphologyEx(interior, cv2.MORPH_OPEN, kernel)
        num, labels, stats, _ = cv2.connectedComponentsWithStats(opened, connectivity=8)
        best = None
        for i in range(1, num):
            x, y, w, h, area = stats[i]
            if h > 14:
                continue
            if best is None or w > best[2]:
                best = (x, y, w, h)
        if best is not None:
            x, y, w, h = best
            yy = iy1 + y + h // 2
            return [(ix1 + x, yy, ix1 + x + w, yy, w)]

    pw = ix2 - ix1
    ih = interior.shape[0]
    edge_exclude = max(6, int(ih * 0.08))
    best_row, best_span = None, 0
    for y in range(edge_exclude, ih - edge_exclude):
        xs = np.where(interior[y] > 0)[0]
        if xs.size < 2:
            continue
        span = xs.max() - xs.min()
        if span > best_span and span >= 0.3 * pw:
            best_span = span
            best_row = (y, xs.min(), xs.max())
    if best_row is not None:
        y, xlo, xhi = best_row
        yy = iy1 + y
        return [(ix1 + xlo, yy, ix1 + xhi, yy, xhi - xlo)]
    return []


# --------------------------------------------------------------------------
# Pipeline
# --------------------------------------------------------------------------

def annotate(image, panel_busbar_pairs):
    out = image.copy()
    for (x1, y1, x2, y2), bbs in panel_busbar_pairs:
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 0, 255), 2)
        for (bx1, by1, bx2, by2, _l) in bbs:
            cv2.line(out, (bx1, by1), (bx2, by2), (0, 200, 0), 3)
    cv2.putText(out, "Busbar", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 0), 2)
    cv2.line(out, (100, 20), (140, 20), (0, 200, 0), 3)
    cv2.putText(out, "Panel", (160, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    cv2.rectangle(out, (240, 12), (270, 28), (0, 0, 255), 2)
    return out


def run_pipeline(image_path, output_path):
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(image_path)
    gray, binary = preprocess_image(image)
    h, w = gray.shape

    panels = detect_panels_graph(gray, binary, w, h)
    all_segs = detect_thick_horizontal_segments(gray, binary)

    pairs = []
    for (x1, y1, x2, y2) in panels:
        bbs = longest_busbar_for_panel(binary, all_segs, (x1, y1, x2, y2))
        pairs.append(((x1, y1, x2, y2), bbs))

    total = sum(len(b) for _, b in pairs)
    annotated = annotate(image, pairs)
    cv2.imwrite(output_path, annotated)
    print(f"Detected {len(pairs)} panel(s), {total} busbar(s) total.")
    for (px1, py1, px2, py2), bbs in pairs:
        print(f"  panel ({px1},{py1})-({px2},{py2}): {len(bbs)} busbar(s)")
    return annotated, pairs


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("image")
    ap.add_argument("-o", "--output", default=None)
    args = ap.parse_args()
    out = args.output or (os.path.splitext(args.image)[0] + "_annotated_v4.png")
    run_pipeline(args.image, out)
