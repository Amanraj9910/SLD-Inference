import { useCallback, useEffect, useRef, useState } from 'react';
import type { KonvaEventObject } from 'konva/lib/Node';
import { Stage, Layer, Image as KonvaImage, Rect, Text, Group } from 'react-konva';
import { useAppStore } from '../store/appStore';
import { classColor } from '../utils/palette';
import type { Detection } from '../store/appStore';

interface CanvasSize { w: number; h: number; }

/** Tooltip position + content */
interface Tooltip {
  x: number;
  y: number;
  label: string;
  color: string;
}

export function ImageCanvas() {
  const containerRef = useRef<HTMLDivElement>(null);
  const [canvasSize, setCanvasSize] = useState<CanvasSize>({ w: 800, h: 600 });
  const [htmlImage, setHtmlImage] = useState<HTMLImageElement | null>(null);
  const [imageNaturalSize, setImageNaturalSize] = useState({ w: 1, h: 1 });
  const [tooltip, setTooltip] = useState<Tooltip | null>(null);
  const [hoveredKey, setHoveredKey] = useState<string | null>(null);

  const {
    currentImageUrl,
    detectionResults,
    thresholds,
    visibleModels,
    showLabels,
    models,
  } = useAppStore();

  // ── Resize observer: make canvas fill container ──────────────────────────
  useEffect(() => {
    if (!containerRef.current) return;
    const ro = new ResizeObserver(entries => {
      const { width, height } = entries[0].contentRect;
      if (width > 0 && height > 0) setCanvasSize({ w: width, h: height });
    });
    ro.observe(containerRef.current);
    return () => ro.disconnect();
  }, []);

  // ── Load image ───────────────────────────────────────────────────────────
  useEffect(() => {
    if (!currentImageUrl) { setHtmlImage(null); return; }
    const img = new window.Image();
    img.onload = () => {
      setHtmlImage(img);
      setImageNaturalSize({ w: img.naturalWidth, h: img.naturalHeight });
    };
    img.src = currentImageUrl;
  }, [currentImageUrl]);

  // ── Scale factor: image → canvas ─────────────────────────────────────────
  const scaleX = htmlImage ? canvasSize.w / imageNaturalSize.w : 1;
  const scaleY = htmlImage ? canvasSize.h / imageNaturalSize.h : 1;

  // ── Collect all visible detections ───────────────────────────────────────
  const visibleDetections: Array<{
    key: string;
    modelId: string;
    arch: string;
    det: Detection;
    className: string;
    color: string;
  }> = [];

  for (const [modelId, modelDets] of Object.entries(detectionResults)) {
    if (!visibleModels[modelId]) continue;
    const threshold = thresholds[modelId] ?? 0;
    const modelInfo = models.find(m => m.model_id === modelId);
    const arch = modelInfo?.arch ?? 'dfine';

    modelDets.detections
      .filter(d => d.score >= threshold)
      .forEach((det, idx) => {
        const className =
          modelDets.class_names[det.class_id] ?? `class_${det.class_id}`;
        visibleDetections.push({
          key: `${modelId}-${idx}`,
          modelId,
          arch,
          det,
          className,
          color: classColor(det.class_id),
        });
      });
  }

  const handleBoxEnter = useCallback(
    (key: string, det: Detection, className: string, color: string, e: KonvaEventObject<MouseEvent>) => {
      setHoveredKey(key);
      const stage = e.target.getStage();
      if (!stage) return;
      const pos = stage.getPointerPosition();
      if (!pos) return;
      setTooltip({
        x: pos.x + 12,
        y: pos.y - 8,
        label: `${className}  ${(det.score * 100).toFixed(1)}%`,
        color,
      });
    },
    []
  );

  const handleBoxLeave = useCallback(() => {
    setHoveredKey(null);
    setTooltip(null);
  }, []);

  // ── Drop zone when no image ───────────────────────────────────────────────
  if (!currentImageUrl) {
    return (
      <div ref={containerRef} className="flex-1 flex items-center justify-center">
        <p className="text-slate-500 text-sm">Upload an image to begin</p>
      </div>
    );
  }

  return (
    <div ref={containerRef} className="flex-1 relative">
      <Stage width={canvasSize.w} height={canvasSize.h}>
        {/* Background image */}
        <Layer>
          {htmlImage && (
            <KonvaImage
              image={htmlImage}
              width={canvasSize.w}
              height={canvasSize.h}
            />
          )}
        </Layer>

        {/* Detection boxes */}
        <Layer>
          {visibleDetections.map(({ key, arch, det, className, color }) => {
            const [x1, y1, x2, y2] = det.box;
            const rx = x1 * scaleX;
            const ry = y1 * scaleY;
            const rw = (x2 - x1) * scaleX;
            const rh = (y2 - y1) * scaleY;
            const isHovered = hoveredKey === key;
            const isDashed = arch === 'rfdetr';
            const strokeWidth = isHovered ? 3 : 1.5;

            return (
              <Group key={key}>
                <Rect
                  x={rx}
                  y={ry}
                  width={rw}
                  height={rh}
                  stroke={color}
                  strokeWidth={strokeWidth}
                  dash={isDashed ? [6, 3] : undefined}
                  fill={isHovered ? classColor(det.class_id, 0.15) : 'transparent'}
                  onMouseEnter={e => handleBoxEnter(key, det, className, color, e)}
                  onMouseLeave={handleBoxLeave}
                  onMouseMove={e => {
                    const stage = e.target.getStage();
                    if (!stage) return;
                    const pos = stage.getPointerPosition();
                    if (pos) setTooltip(t => t ? { ...t, x: pos.x + 12, y: pos.y - 8 } : t);
                  }}
                />
                {/* Label: always visible if showLabels, hover-only otherwise */}
                {(showLabels || isHovered) && (
                  <Text
                    x={rx + 2}
                    y={ry + 2}
                    text={`${className}  ${(det.score * 100).toFixed(0)}%`}
                    fontSize={11}
                    fontFamily="Inter, sans-serif"
                    fill={color}
                    shadowColor="black"
                    shadowBlur={3}
                    shadowOpacity={0.8}
                    listening={false}
                  />
                )}
              </Group>
            );
          })}

          {/* Floating tooltip */}
          {tooltip && (
            <Group x={tooltip.x} y={tooltip.y} listening={false}>
              <Rect
                width={tooltip.label.length * 7 + 16}
                height={24}
                fill="rgba(15,23,42,0.92)"
                cornerRadius={4}
                stroke={tooltip.color}
                strokeWidth={1}
              />
              <Text
                x={8}
                y={5}
                text={tooltip.label}
                fontSize={12}
                fontFamily="Inter, sans-serif"
                fill="#f1f5f9"
              />
            </Group>
          )}
        </Layer>
      </Stage>

      {/* Detection count badge */}
      {visibleDetections.length > 0 && (
        <div className="absolute top-3 right-3 glass rounded-full px-3 py-1 text-xs text-slate-300 pointer-events-none">
          {visibleDetections.length} detection{visibleDetections.length !== 1 ? 's' : ''}
        </div>
      )}
    </div>
  );
}
