/**
 * Generates a visually distinct HSL color for a given class ID.
 * Matches the per-class palette logic from the notebook.
 */
export function classColor(classId: number, alpha = 1): string {
  const hue = (classId * 47 + 13) % 360;   // spread hues evenly
  return `hsla(${hue}, 80%, 60%, ${alpha})`;
}

/** Same color as classColor but returned as {r,g,b} 0-255 */
export function classColorRGB(classId: number): { r: number; g: number; b: number } {
  const hue = (classId * 47 + 13) % 360;
  // HSL → RGB (s=0.8, l=0.6)
  const s = 0.8, l = 0.6;
  const c = (1 - Math.abs(2 * l - 1)) * s;
  const x = c * (1 - Math.abs(((hue / 60) % 2) - 1));
  const m = l - c / 2;
  let r = 0, g = 0, b = 0;
  if (hue < 60)       { r = c; g = x; b = 0; }
  else if (hue < 120) { r = x; g = c; b = 0; }
  else if (hue < 180) { r = 0; g = c; b = x; }
  else if (hue < 240) { r = 0; g = x; b = c; }
  else if (hue < 300) { r = x; g = 0; b = c; }
  else                { r = c; g = 0; b = x; }
  return {
    r: Math.round((r + m) * 255),
    g: Math.round((g + m) * 255),
    b: Math.round((b + m) * 255),
  };
}
