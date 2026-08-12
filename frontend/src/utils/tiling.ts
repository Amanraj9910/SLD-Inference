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
