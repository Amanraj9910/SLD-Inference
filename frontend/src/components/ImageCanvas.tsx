import { useCallback, useEffect, useRef, useState } from 'react';
import type { KonvaEventObject } from 'konva/lib/Node';
import type Konva from 'konva';
import { Stage, Layer, Image as KonvaImage, Rect, Text, Group } from 'react-konva';
import { useAppStore } from '../store/appStore';
import { classColor } from '../utils/palette';
import { computeTileBoxes, computeAdaptiveTileBoxes } from '../utils/tiling';
import type { Detection } from '../store/appStore';

interface CanvasSize { w: number; h: number; }

interface Tooltip {
  x: number;
  y: number;
  label: string;
  color: string;
}

export function ImageCanvas({ modelId }: { modelId?: string }) {
  const containerRef = useRef<HTMLDivElement>(null);
  const stageRef = useRef<Konva.Stage>(null);
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
    showOcr,
    zoomScale,
    setZoomScale,
    stagePos,
    setStagePos,
    resetZoom,
    inferenceMode,
    ocrTilingGrid,
    inferSettings,
    showComponentTileGrid,
    showOcrTileGrid,
  } = useAppStore();

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



  // ── Collect all visible detections ───────────────────────────────────────
  const visibleDetections: Array<{
    key: string;
    modelId: string;
    arch: string;
    det: Detection;
    className: string;
    color: string;
  }> = [];

  const detectionsObj = detectionResults.detections || {};
  for (const [mId, modelDets] of Object.entries(detectionsObj)) {
    if (modelId && mId !== modelId) continue;
    if (!visibleModels[mId]) continue;
    const threshold = thresholds[mId] ?? 0;
    const modelInfo = models.find(m => m.model_id === mId);
    const arch = modelInfo?.arch ?? 'dfine';

    modelDets.detections
      .filter(d => d.score >= threshold)
      .forEach((det, idx) => {
        const className =
          modelDets.class_names[det.class_id] ?? `class_${det.class_id}`;
        visibleDetections.push({
          key: `${mId}-${idx}`,
          modelId: mId,
          arch,
          det,
          className,
          color: classColor(det.class_id),
        });
      });
  }

  const visibleOcr: Array<{
    key: string;
    text: string;
    box: [number, number, number, number];
  }> = [];

  if (showOcr && detectionResults.ocr) {
    detectionResults.ocr.forEach((line, idx) => {
      visibleOcr.push({
        key: `ocr-${idx}`,
        text: line.text,
        box: line.box,
      });
    });
  }

  const visiblePanels: Array<{
    key: string;
    x: number;
    y: number;
    width: number;
    height: number;
    border_type: string;
    continuation?: string | null;
    busbar?: { x: number; y: number; width: number; height: number; score: number } | null;
  }> = [];

  const visibleModelIds = Object.keys(detectionsObj).filter(mId => visibleModels[mId]);
  if (visibleModelIds.length > 0) {
    const primaryModelId = visibleModelIds[0];
    const modelDets = detectionsObj[primaryModelId];
    const panels = modelDets?.panels || [];
    panels.forEach((panel: any, idx: number) => {
      visiblePanels.push({
        key: `panel-${idx}`,
        x: panel.x,
        y: panel.y,
        width: panel.width,
        height: panel.height,
        border_type: panel.border_type,
        continuation: panel.continuation,
        busbar: panel.busbar,
      });
    });
  }

  // Live client-side tile grid calculations (instant feedback before inference)
  const showOcrTilingSection =
    showOcrTileGrid && (inferenceMode === 'ocr' || inferenceMode === 'both') && ocrTilingGrid > 1;
  const liveOcrTiles = showOcrTilingSection
    ? computeTileBoxes(imageNaturalSize.w, imageNaturalSize.h, ocrTilingGrid, 0.40)
    : [];

  const showComponentTilingSection =
    showComponentTileGrid &&
    (inferenceMode === 'components' || inferenceMode === 'both');

  let componentTilesToDraw: Array<{ x1: number; y1: number; x2: number; y2: number }> = [];

  if (showComponentTilingSection) {
    if (inferSettings.tilingMode === 'fixed') {
      if (inferSettings.gridSize > 1) {
        componentTilesToDraw = computeTileBoxes(
          imageNaturalSize.w,
          imageNaturalSize.h,
          inferSettings.gridSize,
          inferSettings.overlap
        );
      }
    } else {
      // Adaptive Tiling
      if (detectionResults.component_tiles && detectionResults.component_tiles.length > 0) {
        componentTilesToDraw = detectionResults.component_tiles.map(box => ({
          x1: box[0],
          y1: box[1],
          x2: box[2],
          y2: box[3],
        }));
      } else {
        const activeModelId = modelId || Array.from(useAppStore.getState().selectedModelIds)[0];
        const activeModel = models.find(m => m.model_id === activeModelId);
        const resolution = activeModel?.resolution || 640;
        const refHeight = activeModel?.target_reference_height || 60.0;
        const res = computeAdaptiveTileBoxes(
          imageNaturalSize.w,
          imageNaturalSize.h,
          inferSettings.targetSymbolPx,
          inferSettings.estimatedSymbolPx,
          resolution,
          inferSettings.overlap,
          inferSettings.enableScaleNorm,
          refHeight
        );
        componentTilesToDraw = res.boxes;
      }
    }
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
        onDragMove={e => {
          setStagePos({ x: e.target.x(), y: e.target.y() });
        }}
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

          {/* Live Component Tile Borders (instant client-side feedback) */}
          {componentTilesToDraw.map(({ x1, y1, x2, y2 }, idx) => {
            const rx = x1 * scaleX;
            const ry = y1 * scaleY;
            const rw = (x2 - x1) * scaleX;
            const rh = (y2 - y1) * scaleY;
            const hue = (idx * 137.5 + 200) % 360;
            const strokeColor = `hsl(${hue}, 85%, 45%)`;
            const fillColor = `hsla(${hue}, 85%, 50%, 0.04)`;

            return (
              <Group key={`comp-tile-${idx}`} listening={false}>
                <Rect
                  x={rx}
                  y={ry}
                  width={rw}
                  height={rh}
                  stroke={strokeColor}
                  strokeWidth={2 / zoomScale}
                  dash={[6 / zoomScale, 3 / zoomScale]}
                  fill={fillColor}
                />
                <Rect
                  x={rx + 4 / zoomScale}
                  y={ry + 4 / zoomScale}
                  width={72 / zoomScale}
                  height={16 / zoomScale}
                  fill={`hsl(${hue}, 70%, 25%)`}
                  cornerRadius={3 / zoomScale}
                />
                <Text
                  x={rx + 7 / zoomScale}
                  y={ry + 6 / zoomScale}
                  text={`Comp Tile ${idx + 1}`}
                  fontSize={9 / zoomScale}
                  fontFamily="Inter, sans-serif"
                  fontStyle="bold"
                  fill="#ffffff"
                />
              </Group>
            );
          })}

          {/* Live OCR Tile Borders (instant client-side feedback) */}
          {liveOcrTiles.map(({ x1, y1, x2, y2 }, idx) => {
            const rx = x1 * scaleX;
            const ry = y1 * scaleY;
            const rw = (x2 - x1) * scaleX;
            const rh = (y2 - y1) * scaleY;
            const hue = (idx * 137.5) % 360;
            const strokeColor = `hsl(${hue}, 85%, 45%)`;
            const fillColor = `hsla(${hue}, 85%, 50%, 0.05)`;
            const labelBg = `hsl(${hue}, 80%, 25%)`;

            return (
              <Group key={`ocr-tile-${idx}`} listening={false}>
                <Rect
                  x={rx}
                  y={ry}
                  width={rw}
                  height={rh}
                  stroke={strokeColor}
                  strokeWidth={2 / zoomScale}
                  dash={[8 / zoomScale, 4 / zoomScale]}
                  fill={fillColor}
                />
                <Rect
                  x={rx + 4 / zoomScale}
                  y={ry + 4 / zoomScale}
                  width={62 / zoomScale}
                  height={16 / zoomScale}
                  fill={labelBg}
                  cornerRadius={3 / zoomScale}
                />
                <Text
                  x={rx + 7 / zoomScale}
                  y={ry + 6 / zoomScale}
                  text={`OCR Tile ${idx + 1}`}
                  fontSize={9 / zoomScale}
                  fontFamily="Inter, sans-serif"
                  fontStyle="bold"
                  fill="#ffffff"
                />
              </Group>
            );
          })}

          {/* OCR text boxes */}
          {visibleOcr.map(({ key, text, box }) => {
            const [x1, y1, x2, y2] = box;
            const rx = x1 * scaleX;
            const ry = y1 * scaleY;
            const rw = (x2 - x1) * scaleX;
            const rh = (y2 - y1) * scaleY;
            const isHovered = hoveredKey === key;
            const strokeWidth = (isHovered ? 2.5 : 1.25) / zoomScale;
            const color = '#10b981';

            return (
              <Group key={key}>
                <Rect
                  x={rx}
                  y={ry}
                  width={rw}
                  height={rh}
                  stroke={color}
                  strokeWidth={strokeWidth}
                  dash={[4 / zoomScale, 2 / zoomScale]}
                  fill={isHovered ? 'rgba(16,185,129,0.12)' : 'transparent'}
                  onMouseEnter={e => {
                    setHoveredKey(key);
                    const stage = e.target.getStage();
                    if (!stage) return;
                    const pos = stage.getPointerPosition();
                    if (!pos) return;
                    setTooltip({
                      x: (pos.x - stage.x()) / zoomScale + 12 / zoomScale,
                      y: (pos.y - stage.y()) / zoomScale - 8 / zoomScale,
                      label: text,
                      color,
                    });
                  }}
                  onMouseLeave={handleBoxLeave}
                  onMouseMove={e => {
                    const stage = e.target.getStage();
                    if (!stage) return;
                    const pos = stage.getPointerPosition();
                    if (pos) {
                      setTooltip(t => t ? {
                        ...t,
                        x: (pos.x - stage.x()) / zoomScale + 12 / zoomScale,
                        y: (pos.y - stage.y()) / zoomScale - 8 / zoomScale
                      } : t);
                    }
                  }}
                />
                {(showLabels || isHovered) && (
                  <Text
                    x={rx + 2 / zoomScale}
                    y={ry + 2 / zoomScale}
                    text={text}
                    fontSize={10 / zoomScale}
                    fontFamily="Inter, sans-serif"
                    fill={color}
                    shadowColor="black"
                    shadowBlur={2 / zoomScale}
                    shadowOpacity={0.8}
                    listening={false}
                  />
                )}
              </Group>
            );
          })}

          {/* Deterministic Panel and Busbar boxes */}
          {visiblePanels.map((panel, idx) => {
            const rx = panel.x * scaleX;
            const ry = panel.y * scaleY;
            const rw = panel.width * scaleX;
            const rh = panel.height * scaleY;
            const isHovered = hoveredKey === panel.key;
            const strokeWidth = (isHovered ? 3.5 : 2) / zoomScale;
            const panelColor = '#2563eb'; // Royal Blue
            const busbarColor = '#10b981'; // Emerald Green

            let label = `P${idx + 1}`;
            if (panel.continuation === 'left') label += '[L]';
            else if (panel.continuation === 'right') label += '[R]';
            else if (panel.continuation === 'both') label += '[L+R]';

            return (
              <Group key={panel.key}>
                {/* Panel rectangle */}
                <Rect
                  x={rx}
                  y={ry}
                  width={rw}
                  height={rh}
                  stroke={panelColor}
                  strokeWidth={strokeWidth}
                  dash={panel.border_type === 'dashed' || panel.border_type === 'dashed_with_continuation' ? [6 / zoomScale, 3 / zoomScale] : undefined}
                  fill={isHovered ? 'rgba(37,99,235,0.08)' : 'transparent'}
                  onMouseEnter={e => {
                    setHoveredKey(panel.key);
                    const stage = e.target.getStage();
                    if (!stage) return;
                    const pos = stage.getPointerPosition();
                    if (!pos) return;
                    setTooltip({
                      x: (pos.x - stage.x()) / zoomScale + 12 / zoomScale,
                      y: (pos.y - stage.y()) / zoomScale - 8 / zoomScale,
                      label: `${label} (${panel.border_type})`,
                      color: panelColor,
                    });
                  }}
                  onMouseLeave={handleBoxLeave}
                  onMouseMove={e => {
                    const stage = e.target.getStage();
                    if (!stage) return;
                    const pos = stage.getPointerPosition();
                    if (pos) {
                      setTooltip(t => t ? {
                        ...t,
                        x: (pos.x - stage.x()) / zoomScale + 12 / zoomScale,
                        y: (pos.y - stage.y()) / zoomScale - 8 / zoomScale
                      } : t);
                    }
                  }}
                />
                {(showLabels || isHovered) && (
                  <Text
                    x={rx + 4 / zoomScale}
                    y={ry + 4 / zoomScale}
                    text={label}
                    fontSize={11 / zoomScale}
                    fontFamily="Inter, sans-serif"
                    fontStyle="bold"
                    fill={panelColor}
                    shadowColor="black"
                    shadowBlur={2 / zoomScale}
                    shadowOpacity={0.8}
                    listening={false}
                  />
                )}

                {/* Busbar rectangle if present */}
                {(() => {
                  const busbar = panel.busbar;
                  if (!busbar) return null;
                  const bx = busbar.x * scaleX;
                  const by = busbar.y * scaleY;
                  const bw = busbar.width * scaleX;
                  const bh = busbar.height * scaleY;
                  const busbarKey = `${panel.key}-busbar`;
                  const isBusbarHovered = hoveredKey === busbarKey;
                  const bStrokeWidth = (isBusbarHovered ? 4 : 2.5) / zoomScale;

                  return (
                    <Group key={busbarKey}>
                      <Rect
                        x={bx}
                        y={by}
                        width={bw}
                        height={bh}
                        stroke={busbarColor}
                        strokeWidth={bStrokeWidth}
                        fill={isBusbarHovered ? 'rgba(16,185,129,0.2)' : 'transparent'}
                        onMouseEnter={e => {
                          setHoveredKey(busbarKey);
                          const stage = e.target.getStage();
                          if (!stage) return;
                          const pos = stage.getPointerPosition();
                          if (!pos) return;
                          setTooltip({
                            x: (pos.x - stage.x()) / zoomScale + 12 / zoomScale,
                            y: (pos.y - stage.y()) / zoomScale - 8 / zoomScale,
                            label: `Busbar (Len: ${busbar.width})`,
                            color: busbarColor,
                          });
                        }}
                        onMouseLeave={handleBoxLeave}
                        onMouseMove={e => {
                          const stage = e.target.getStage();
                          if (!stage) return;
                          const pos = stage.getPointerPosition();
                          if (pos) {
                            setTooltip(t => t ? {
                              ...t,
                              x: (pos.x - stage.x()) / zoomScale + 12 / zoomScale,
                              y: (pos.y - stage.y()) / zoomScale - 8 / zoomScale
                            } : t);
                          }
                        }}
                      />
                    </Group>
                  );
                })()}
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

      {/* Detection count badge */}
      {(visibleDetections.length > 0 || visibleOcr.length > 0) && (
        <div className="absolute top-3 right-3 glass rounded-full px-3 py-1 text-xs text-slate-300 pointer-events-none">
          {visibleDetections.length > 0 && `${visibleDetections.length} component${visibleDetections.length !== 1 ? 's' : ''}`}
          {visibleDetections.length > 0 && visibleOcr.length > 0 && ' | '}
          {visibleOcr.length > 0 && `${visibleOcr.length} text line${visibleOcr.length !== 1 ? 's' : ''}`}
        </div>
      )}
    </div>
  );
}
