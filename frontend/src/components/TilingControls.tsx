import { useAppStore } from '../store/appStore';
import { Eye, EyeOff } from 'lucide-react';
import { computeAdaptiveTileBoxes } from '../utils/tiling';

export function TilingControls() {
  const {
    inferSettings,
    setInferSettings,
    inferenceMode,
    ocrTilingGrid,
    setOcrTilingGrid,
    showComponentTileGrid,
    toggleShowComponentTileGrid,
    showOcrTileGrid,
    toggleShowOcrTileGrid,
    models,
    selectedModelIds,
    imageDimensions,
  } = useAppStore();

  const selectedModelId = Array.from(selectedModelIds)[0];
  const selectedModel = models.find(m => m.model_id === selectedModelId);
  const modelInputSize = selectedModel?.resolution || 640;
  const targetReferenceHeight = selectedModel?.target_reference_height || 60.0;

  let adaptiveGridStr = '';
  if (imageDimensions) {
    const { gx, gy, boxes } = computeAdaptiveTileBoxes(
      imageDimensions.width,
      imageDimensions.height,
      inferSettings.targetSymbolPx,
      inferSettings.estimatedSymbolPx,
      modelInputSize,
      inferSettings.overlap,
      inferSettings.enableScaleNorm,
      targetReferenceHeight
    );
    adaptiveGridStr = `${gx} × ${gy} (${boxes.length} tile${boxes.length !== 1 ? 's' : ''})`;
  } else {
    adaptiveGridStr = 'Upload image to calculate';
  }

  const adaptive = inferSettings.tilingMode === 'adaptive';
  const showComponentTiling = inferenceMode === 'components' || inferenceMode === 'both';
  const showOcrTiling = inferenceMode === 'ocr' || inferenceMode === 'both';

  return (
    <div className="mt-auto border-t border-slate-200/80 pt-3 space-y-2.5">
      {/* Component Detection Tiling */}
      {showComponentTiling && (
        <div className="space-y-2">
          <div className="flex items-center justify-between px-0.5">
            <p className="text-[9px] font-bold uppercase tracking-wider text-slate-400">Component Tiling</p>
            {(adaptive || inferSettings.gridSize > 1) && (
              <button
                type="button"
                onClick={toggleShowComponentTileGrid}
                className={`flex items-center gap-1 text-[10px] font-semibold px-2 py-0.5 rounded border transition-colors ${
                  showComponentTileGrid
                    ? 'bg-indigo-50 text-indigo-700 border-indigo-200'
                    : 'bg-slate-50 text-slate-500 border-slate-200 hover:text-slate-700'
                }`}
                title="Toggle client-side tile grid overlay"
              >
                {showComponentTileGrid ? <Eye size={11} /> : <EyeOff size={11} />}
                Grid Overlay
              </button>
            )}
          </div>

          <button
            type="button"
            aria-pressed={adaptive}
            onClick={() => setInferSettings({ tilingMode: adaptive ? 'fixed' : 'adaptive' })}
            className={`w-full flex items-center justify-between rounded-xl px-3 py-2 border transition-colors ${
              adaptive ? 'bg-indigo-50 border-indigo-200 text-indigo-800' : 'bg-slate-50 border-slate-200 text-slate-700'
            }`}
          >
            <span className="text-xs font-semibold">Adaptive Tiling</span>
            <span className="flex items-center gap-2 text-[10px] font-bold uppercase tracking-wide">
              {adaptive ? 'On' : 'Off · Fixed'}
              <span className={`w-8 h-4 rounded-full p-0.5 ${adaptive ? 'bg-indigo-600' : 'bg-slate-300'}`}>
                <span className={`block w-3 h-3 rounded-full bg-white transition-transform ${adaptive ? 'translate-x-4' : ''}`} />
              </span>
            </span>
          </button>

          {adaptive ? (
            <div className="space-y-1">
              <div className="flex items-center justify-between">
                <span className="text-[11px] font-medium text-slate-500">Calculated grid</span>
                <span className="text-[11px] text-indigo-700 tabular-nums font-semibold">
                  {adaptiveGridStr}
                </span>
              </div>
              <p className="text-[10px] leading-relaxed text-slate-500">
                Tile count is calculated from the uploaded image and symbol scale.
              </p>
            </div>
          ) : (
            <div className="space-y-1">
              <div className="flex items-center justify-between">
                <span className="text-[11px] font-medium text-slate-500">Fixed tile grid</span>
                <span className="text-[11px] text-slate-700 tabular-nums font-semibold">
                  {inferSettings.gridSize} × {inferSettings.gridSize}
                </span>
              </div>
              <input
                type="range"
                min={1}
                max={10}
                step={1}
                value={inferSettings.gridSize}
                onChange={event => setInferSettings({ gridSize: parseInt(event.target.value, 10) })}
                className="w-full"
              />
            </div>
          )}
        </div>
      )}

      {/* OCR Tiling */}
      {showOcrTiling && (
        <div className="space-y-2">
          {showComponentTiling && <div className="border-t border-slate-100 my-1" />}
          <div className="flex items-center justify-between px-0.5">
            <p className="text-[9px] font-bold uppercase tracking-wider text-emerald-600">OCR Tiling</p>
            {ocrTilingGrid > 1 && (
              <button
                type="button"
                onClick={toggleShowOcrTileGrid}
                className={`flex items-center gap-1 text-[10px] font-semibold px-2 py-0.5 rounded border transition-colors ${
                  showOcrTileGrid
                    ? 'bg-emerald-50 text-emerald-700 border-emerald-200'
                    : 'bg-slate-50 text-slate-500 border-slate-200 hover:text-slate-700'
                }`}
                title="Toggle client-side OCR tile grid overlay"
              >
                {showOcrTileGrid ? <Eye size={11} /> : <EyeOff size={11} />}
                Grid Overlay
              </button>
            )}
          </div>

          <div className="space-y-1">
            <div className="flex items-center justify-between">
              <span className="text-[11px] font-medium text-slate-500">OCR tile grid</span>
              <span className="text-[11px] text-emerald-700 tabular-nums font-semibold">
                {ocrTilingGrid === 1 ? 'Off' : `${ocrTilingGrid} × ${ocrTilingGrid}`}
              </span>
            </div>
            <input
              type="range"
              min={1}
              max={6}
              step={1}
              value={ocrTilingGrid}
              onChange={event => setOcrTilingGrid(parseInt(event.target.value, 10))}
              className="w-full accent-emerald-600"
            />
            <p className="text-[10px] leading-relaxed text-slate-500">
              {ocrTilingGrid === 1
                ? 'Full image sent to OCR (no tiling).'
                : `Image split into ${ocrTilingGrid}×${ocrTilingGrid} tiles with 40% overlap.`}
            </p>
          </div>
        </div>
      )}
    </div>
  );
}
