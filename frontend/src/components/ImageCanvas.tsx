import { useCallback, useEffect, useRef, useState } from 'react';
import type { KonvaEventObject } from 'konva/lib/Node';
import type Konva from 'konva';
import { Stage, Layer, Image as KonvaImage, Rect, Text, Group } from 'react-konva';
import { useAppStore } from '../store/appStore';
import { classColor } from '../utils/palette';
import type { Detection } from '../store/appStore';
import { ZoomIn, ZoomOut, RotateCcw, Move } from 'lucide-react';

interface CanvasSize { w: number; h: number; }

interface Tooltip {
  x: number;
  y: number;
  label: string;
  color: string;
}

export function ImageCanvas() {
  const containerRef = useRef<HTMLDivElement>(null);
  const stageRef = useRef<Konva.Stage>(null);
  const [canvasSize, setCanvasSize] = useState<CanvasSize>({ w: 800, h: 600 });
  const [htmlImage, setHtmlImage] = useState<HTMLImageElement | null>(null);
  const [imageNaturalSize, setImageNaturalSize] = useState({ w: 1, h: 1 });
  const [tooltip, setTooltip] = useState<Tooltip | null>(null);
  const [hoveredKey, setHoveredKey] = useState<string | null>(null);

  // ── Zoom & Pan State ─────────────────────────────────────────────────────
  const [zoomScale, setZoomScale] = useState(1);
  const [stagePos, setStagePos] = useState({ x: 0, y: 0 });

  const {
    currentImageUrl,
    detectionResults,
    thresholds,
    visibleModels,
    showLabels,
    models,
  } = useAppStore();

  // Reset zoom & position when a new image is loaded
  const resetZoom = useCallback(() => {
    setZoomScale(1);
    setStagePos({ x: 0, y: 0 });
  }, []);

  // ── Resize observer ──────────────────────────────────────────────────────
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
      resetZoom();
    };
    img.src = currentImageUrl;
  }, [currentImageUrl, resetZoom]);

  // Scale factor for base image to canvas container
  const scaleX = htmlImage ? canvasSize.w / imageNaturalSize.w : 1;
  const scaleY = htmlImage ? canvasSize.h / imageNaturalSize.h : 1;

  // ── Wheel Zoom Centered on Pointer ───────────────────────────────────────
  const handleWheel = (e: KonvaEventObject<WheelEvent>) => {
    e.evt.preventDefault();
    const stage = stageRef.current;
    if (!stage) return;

    const oldScale = zoomScale;
    const pointer = stage.getPointerPosition();
    if (!pointer) return;

    const mousePointTo = {
      x: (pointer.x - stage.x()) / oldScale,
      y: (pointer.y - stage.y()) / oldScale,
    };

    const speed = 1.12;
    const newScale = e.evt.deltaY < 0 ? oldScale * speed : oldScale / speed;
    const clampedScale = Math.max(0.5, Math.min(15, newScale));

    setZoomScale(clampedScale);
    setStagePos({
      x: pointer.x - mousePointTo.x * clampedScale,
      y: pointer.y - mousePointTo.y * clampedScale,
    });
  };

  const handleZoomBtn = (direction: 'in' | 'out') => {
    const factor = direction === 'in' ? 1.25 : 0.8;
    const newScale = Math.max(0.5, Math.min(15, zoomScale * factor));
    setZoomScale(newScale);
  };

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
        x: (pos.x - stage.x()) / zoomScale + 12 / zoomScale,
        y: (pos.y - stage.y()) / zoomScale - 8 / zoomScale,
        label: `${className}  ${(det.score * 100).toFixed(1)}%`,
        color,
      });
    },
    [zoomScale]
  );

  const handleBoxLeave = useCallback(() => {
    setHoveredKey(null);
    setTooltip(null);
  }, []);

  if (!currentImageUrl) {
    return (
      <div ref={containerRef} className="flex-1 flex items-center justify-center">
        <p className="text-slate-500 text-sm">Upload an image to begin</p>
      </div>
    );
  }

  return (
    <div ref={containerRef} className="flex-1 relative overflow-hidden select-none">
      <Stage
        ref={stageRef}
        width={canvasSize.w}
        height={canvasSize.h}
        scaleX={zoomScale}
        scaleY={zoomScale}
        x={stagePos.x}
        y={stagePos.y}
        draggable={true}
        onWheel={handleWheel}
        onDragEnd={e => {
          setStagePos({ x: e.target.x(), y: e.target.y() });
        }}
        style={{ cursor: zoomScale > 1 ? 'grab' : 'default' }}
      >
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
            const strokeWidth = (isHovered ? 3 : 1.5) / zoomScale;

            return (
              <Group key={key}>
                <Rect
                  x={rx}
                  y={ry}
                  width={rw}
                  height={rh}
                  stroke={color}
                  strokeWidth={strokeWidth}
                  dash={isDashed ? [6 / zoomScale, 3 / zoomScale] : undefined}
                  fill={isHovered ? classColor(det.class_id, 0.15) : 'transparent'}
                  onMouseEnter={e => handleBoxEnter(key, det, className, color, e)}
                  onMouseLeave={handleBoxLeave}
                  onMouseMove={e => {
                    const stage = e.target.getStage();
                    if (!stage) return;
                    const pos = stage.getPointerPosition();
                    if (pos) setTooltip(t => t ? { ...t, x: (pos.x - stage.x()) / zoomScale + 12 / zoomScale, y: (pos.y - stage.y()) / zoomScale - 8 / zoomScale } : t);
                  }}
                />
                {(showLabels || isHovered) && (
                  <Text
                    x={rx + 2 / zoomScale}
                    y={ry + 2 / zoomScale}
                    text={`${className}  ${(det.score * 100).toFixed(0)}%`}
                    fontSize={11 / zoomScale}
                    fontFamily="Inter, sans-serif"
                    fill={color}
                    shadowColor="black"
                    shadowBlur={3 / zoomScale}
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
                width={(tooltip.label.length * 7 + 16) / zoomScale}
                height={24 / zoomScale}
                fill="rgba(15,23,42,0.92)"
                cornerRadius={4 / zoomScale}
                stroke={tooltip.color}
                strokeWidth={1 / zoomScale}
              />
              <Text
                x={8 / zoomScale}
                y={5 / zoomScale}
                text={tooltip.label}
                fontSize={12 / zoomScale}
                fontFamily="Inter, sans-serif"
                fill="#f1f5f9"
              />
            </Group>
          )}
        </Layer>
      </Stage>

      {/* Floating Toolbar — Zoom Controls */}
      <div className="absolute bottom-4 right-4 glass rounded-xl px-2 py-1.5 flex items-center gap-1.5 border border-slate-700/60 shadow-xl z-20">
        <button
          onClick={() => handleZoomBtn('out')}
          className="p-1.5 rounded-lg text-slate-400 hover:text-slate-200 hover:bg-slate-800/60 transition-all"
          title="Zoom out"
        >
          <ZoomOut size={16} />
        </button>
        <span className="text-xs text-slate-300 font-mono font-medium px-1.5 tabular-nums min-w-12 text-center">
          {Math.round(zoomScale * 100)}%
        </span>
        <button
          onClick={() => handleZoomBtn('in')}
          className="p-1.5 rounded-lg text-slate-400 hover:text-slate-200 hover:bg-slate-800/60 transition-all"
          title="Zoom in"
        >
          <ZoomIn size={16} />
        </button>
        <div className="h-4 w-px bg-slate-700/60 mx-0.5" />
        <button
          onClick={resetZoom}
          className="p-1.5 rounded-lg text-slate-400 hover:text-cyan-400 hover:bg-slate-800/60 transition-all"
          title="Reset zoom & position"
        >
          <RotateCcw size={15} />
        </button>
        <div className="text-[10px] text-slate-500 flex items-center gap-1 pl-1">
          <Move size={12} /> Drag to Pan
        </div>
      </div>

      {/* Detection count badge */}
      {visibleDetections.length > 0 && (
        <div className="absolute top-3 right-3 glass rounded-full px-3 py-1 text-xs text-slate-300 pointer-events-none">
          {visibleDetections.length} detection{visibleDetections.length !== 1 ? 's' : ''}
        </div>
      )}
    </div>
  );
}
