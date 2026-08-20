export interface TileBox {
  x1: number;
  y1: number;
  x2: number;
  y2: number;
}

/**
 * Compute overlapping tile bounding boxes in image pixel coordinates.
 * Matches backend tile_image logic in app/tiling.py.
 */
export function computeTileBoxes(
  W: number,
  H: number,
  gridSize: number,
  overlap: number
): TileBox[] {
  if (gridSize <= 1 || W <= 0 || H <= 0) return [];
  const denominator = gridSize - (gridSize - 1) * overlap;
  const tileW = Math.ceil(W / denominator);
  const tileH = Math.ceil(H / denominator);
  const maxXStart = Math.max(0, W - tileW);
  const maxYStart = Math.max(0, H - tileH);

  const positionsX =
    gridSize === 1
      ? [0]
      : Array.from({ length: gridSize }, (_, gx) =>
          Math.round((gx * maxXStart) / (gridSize - 1))
        );
  const positionsY =
    gridSize === 1
      ? [0]
      : Array.from({ length: gridSize }, (_, gy) =>
          Math.round((gy * maxYStart) / (gridSize - 1))
        );

  const boxes: TileBox[] = [];
  for (const y of positionsY) {
    for (const x of positionsX) {
      boxes.push({
        x1: x,
        y1: y,
        x2: Math.min(x + tileW, W),
        y2: Math.min(y + tileH, H),
      });
    }
  }
  return boxes;
}

/**
 * Compute overlapping adaptive tile bounding boxes in image pixel coordinates.
 * Matches backend adaptive_tile_image logic in app/component/tiling.py.
 */
export function computeAdaptiveTileBoxes(
  W: number,
  H: number,
  targetSymbolPx: number,
  estimatedSymbolPx: number,
  modelInputSize: number, // resolution
  overlap: number,
  enableScaleNorm: boolean,
  targetReferenceHeight: number = 60.0
): { boxes: TileBox[]; gx: number; gy: number; scale: number } {
  if (W <= 0 || H <= 0) return { boxes: [], gx: 1, gy: 1, scale: 1.0 };

  // 1. Scale Normalization
  let s = 1.0;
  let newW = W;
  let newH = H;
  if (enableScaleNorm && estimatedSymbolPx > 0) {
    s = targetReferenceHeight / estimatedSymbolPx;
    const maxPx = 50_000_000;
    const projectedPx = (W * s) * (H * s);
    if (projectedPx > maxPx) {
      const capScale = Math.sqrt(maxPx / (W * H));
      s = Math.min(s, capScale);
    }
    newW = Math.max(32, Math.round(W * s));
    newH = Math.max(32, Math.round(H * s));
  }

  // 2. Assume no auto-crop for client-side live preview estimate
  const cw = newW;
  const ch = newH;

  // 3. Adaptive Grid Size (gx, gy)
  const effectiveSymbolPx = estimatedSymbolPx > 0 ? estimatedSymbolPx * s : targetReferenceHeight;
  const baseGx = Math.max(1, Math.round((targetSymbolPx * newW) / (modelInputSize * effectiveSymbolPx)));
  const baseGy = Math.max(1, Math.round((targetSymbolPx * newH) / (modelInputSize * effectiveSymbolPx)));
  const gx = Math.max(baseGx, Math.round((targetSymbolPx * cw) / (modelInputSize * effectiveSymbolPx)));
  const gy = Math.max(baseGy, Math.round((targetSymbolPx * ch) / (modelInputSize * effectiveSymbolPx)));

  const tileW = gx > 1 ? cw / (gx - (gx - 1) * overlap) : cw;
  const tileH = gy > 1 ? ch / (gy - (gy - 1) * overlap) : ch;
  const strideX = gx > 1 ? tileW * (1 - overlap) : cw;
  const strideY = gy > 1 ? tileH * (1 - overlap) : ch;

  const positionsX = Array.from({ length: gx }, (_, c) => c * strideX);
  if (gx > 1) {
    positionsX[positionsX.length - 1] = Math.max(0.0, cw - tileW);
  }
  const positionsY = Array.from({ length: gy }, (_, r) => r * strideY);
  if (gy > 1) {
    positionsY[positionsY.length - 1] = Math.max(0.0, ch - tileH);
  }

  const boxes: TileBox[] = [];
  for (const py of positionsY) {
    for (const px of positionsX) {
      const tx1 = Math.round(px);
      const ty1 = Math.round(py);
      const tx2 = Math.min(cw, Math.round(px + tileW));
      const ty2 = Math.min(ch, Math.round(py + tileH));
      if (tx2 <= tx1 || ty2 <= ty1) continue;

      // Map back to original image space
      boxes.push({
        x1: tx1 / s,
        y1: ty1 / s,
        x2: tx2 / s,
        y2: ty2 / s
      });
    }
  }

  return { boxes, gx, gy, scale: s };
}
