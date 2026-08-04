import { useState } from 'react';
import { useAppStore } from '../store/appStore';
import {
  Eye,
  EyeOff,
  CheckSquare,
  Square,
  Settings,
  Loader2,
  Cpu,
  ChevronDown,
  ChevronRight,
  Plus,
  Trash2,
  Layers,
  Grid,
} from 'lucide-react';

export function ModelPanel() {
  const {
    models,
    modelsLoading,
    selectedModelIds,
    toggleModelSelected,
    thresholds,
    setThreshold,
    visibleModels,
    toggleModelVisible,
    loadModel,
    openConfigModal,
    openUploadModal,
    deleteModel,
    imageDimensions,
    inferSettings,
    setInferSettings,
  } = useAppStore();

  const [loadingIds, setLoadingIds] = useState<Set<string>>(new Set());
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [deletingId, setDeletingId] = useState<string | null>(null);

  const handleLoad = async (modelId: string) => {
    setLoadingIds(prev => new Set(prev).add(modelId));
    try {
      await loadModel(modelId);
    } catch (err) {
      console.error('Failed to load model', err);
    } finally {
      setLoadingIds(prev => {
        const next = new Set(prev);
        next.delete(modelId);
        return next;
      });
    }
  };

  const handleDelete = async (modelId: string, displayName: string) => {
    if (!confirm(`Are you sure you want to delete '${displayName}' from disk?`)) return;
    setDeletingId(modelId);
    try {
      await deleteModel(modelId);
    } catch (err) {
      alert(`Failed to delete model: ${err}`);
    } finally {
      setDeletingId(null);
    }
  };

  const adaptiveModels = models.filter(m => (m.tiling_mode || 'adaptive') === 'adaptive');
  const fixedModels = models.filter(m => m.tiling_mode === 'fixed');

  const renderModelCard = (model: typeof models[0]) => {
    const selected = selectedModelIds.has(model.model_id);
    const visible = visibleModels[model.model_id] ?? true;
    const threshold = thresholds[model.model_id] ?? model.confidence_default;
    const isLoading = loadingIds.has(model.model_id);
    const hasWeights = model.weights_exist ?? true;

    return (
      <div
        key={model.model_id}
        className={`rounded-xl border transition-all duration-200 shadow-sm ${
          !hasWeights
            ? 'border-red-200 bg-red-50/30'
            : selected
            ? 'border-indigo-500/60 bg-indigo-50/40 ring-1 ring-indigo-500/20'
            : 'border-slate-200 bg-white hover:border-slate-300'
        }`}
      >
        {/* Top row: checkbox, name, buttons */}
        <div className="flex items-center gap-2.5 p-3">
          <button
            onClick={() => hasWeights && toggleModelSelected(model.model_id)}
            disabled={!hasWeights}
            className={`transition-colors shrink-0 ${
              !hasWeights
                ? 'text-slate-300 cursor-not-allowed'
                : 'text-slate-400 hover:text-indigo-600'
            }`}
            title={
              !hasWeights
                ? `Upload ${model.weights_file} to backend/weights/${model.arch}/ to enable`
                : selected
                ? 'Deselect model'
                : 'Select model'
            }
          >
            {selected ? (
              <CheckSquare size={18} className="text-indigo-600" />
            ) : (
              <Square size={18} />
            )}
          </button>

          <div className="flex-1 min-w-0">
            <p className="text-xs font-semibold text-slate-800 truncate">
              {model.display_name}
            </p>
            <div className="flex items-center gap-1.5 mt-0.5 flex-wrap">
              <span
                className={`text-[9px] font-bold uppercase px-1.5 py-0.5 rounded border ${
                  model.arch === 'dfine'
                    ? 'bg-cyan-50 text-cyan-700 border-cyan-200'
                    : 'bg-amber-50 text-amber-700 border-amber-200'
                }`}
              >
                {model.arch}
              </span>
              <span className="text-[9px] font-semibold px-1 py-0.5 rounded bg-slate-100 text-slate-600 border border-slate-200">
                {model.tiling_mode === 'fixed' ? `${model.grid_size || 4}×${model.grid_size || 4} Fixed` : 'Adaptive Size'}
              </span>
              {hasWeights ? (
                model.loaded ? (
                  <span className="text-[10px] font-semibold text-emerald-600">● GPU</span>
                ) : (
                  <span className="text-[10px] text-slate-400">Ready</span>
                )
              ) : (
                <span className="text-[9px] font-semibold text-red-600 bg-red-50 px-1 py-0.5 rounded border border-red-200" title={`Missing ${model.weights_file}`}>
                  Missing .pth
                </span>
              )}
            </div>
          </div>

          <button
            onClick={() => toggleModelVisible(model.model_id)}
            className="text-slate-400 hover:text-slate-600 transition-colors shrink-0"
            title={visible ? 'Hide detections' : 'Show detections'}
          >
            {visible ? <Eye size={16} /> : <EyeOff size={16} />}
          </button>

          <button
            onClick={() => openConfigModal(model.model_id)}
            className="text-slate-400 hover:text-slate-600 transition-colors shrink-0"
            title="Edit class names / threshold"
          >
            <Settings size={16} />
          </button>

          <button
            onClick={() => handleDelete(model.model_id, model.display_name)}
            disabled={deletingId === model.model_id}
            className="text-slate-400 hover:text-red-600 transition-colors shrink-0 disabled:opacity-50"
            title="Delete checkpoint directory from server"
          >
            {deletingId === model.model_id ? (
              <Loader2 size={16} className="animate-spin text-red-600" />
            ) : (
              <Trash2 size={16} />
            )}
          </button>
        </div>

        {selected && (
          <div className="px-3 pb-3 space-y-2.5 fade-in border-t border-slate-100 pt-2.5">
            {!model.loaded && (
              <button
                onClick={() => handleLoad(model.model_id)}
                disabled={isLoading || !hasWeights}
                className="w-full flex items-center justify-center gap-2 text-xs font-semibold
                           bg-indigo-50 hover:bg-indigo-100 text-indigo-700
                           border border-indigo-200 rounded-lg py-1.5
                           transition-colors disabled:opacity-50 disabled:cursor-not-allowed shadow-sm"
              >
                {isLoading ? (
                  <Loader2 size={13} className="animate-spin" />
                ) : (
                  <Cpu size={13} />
                )}
                {isLoading
                  ? 'Loading…'
                  : hasWeights
                  ? 'Load to GPU'
                  : `Missing ${model.weights_file}`}
              </button>
            )}

            <div className="space-y-1">
              <div className="flex items-center justify-between">
                <span className="text-[11px] font-medium text-slate-500">Confidence</span>
                <span className="text-[11px] text-slate-700 tabular-nums font-semibold">
                  {(threshold * 100).toFixed(0)}%
                </span>
              </div>
              <input
                type="range"
                min={0}
                max={1}
                step={0.01}
                value={threshold}
                onChange={e =>
                  setThreshold(model.model_id, parseFloat(e.target.value))
                }
                className="w-full"
              />
            </div>
          </div>
        )}
      </div>
    );
  };

  return (
    <aside className="w-72 glass rounded-2xl p-4 flex flex-col gap-4 overflow-y-auto bg-white border border-slate-200/80 shadow-sm">
      {/* Overall Header */}
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-xs font-bold uppercase tracking-wider text-slate-400">
            Model Registry
          </h2>
          {modelsLoading && (
            <div className="flex items-center gap-1.5 mt-1 text-xs text-slate-500 font-medium">
              <Loader2 size={12} className="animate-spin" /> Scanning…
            </div>
          )}
        </div>
      </div>

      {/* Group 1: Adaptive Tiling Models */}
      <div className="space-y-2">
        <div className="flex items-center justify-between border-b border-indigo-100 pb-1.5">
          <div className="flex items-center gap-1.5 text-xs font-bold text-indigo-950">
            <Layers size={15} className="text-indigo-600" />
            <span>Adaptive Tiling Models</span>
            <span className="text-[10px] bg-indigo-100 text-indigo-700 font-extrabold px-1.5 py-0.2 rounded-full">
              {adaptiveModels.length}
            </span>
          </div>
          <button
            onClick={() => openUploadModal('adaptive')}
            className="flex items-center gap-0.5 text-[10px] font-semibold text-indigo-600 hover:text-indigo-700 bg-indigo-50 hover:bg-indigo-100/80 px-2 py-0.5 rounded transition-all border border-indigo-200/60"
            title="Upload model trained on adaptive tiling"
          >
            <Plus size={12} /> Add
          </button>
        </div>

        <div className="flex flex-col gap-2.5">
          {adaptiveModels.map(renderModelCard)}
          {adaptiveModels.length === 0 && !modelsLoading && (
            <p className="text-[11px] text-slate-400 italic px-1 py-1">
              No adaptive tiling models. Click "+ Add" to upload.
            </p>
          )}
        </div>
      </div>

      {/* Group 2: Fixed Grid Models */}
      <div className="space-y-2 pt-2 border-t border-slate-100">
        <div className="flex items-center justify-between border-b border-slate-100 pb-1.5">
          <div className="flex items-center gap-1.5 text-xs font-bold text-slate-800">
            <Grid size={15} className="text-slate-600" />
            <span>Fixed Grid Models</span>
            <span className="text-[10px] bg-slate-100 text-slate-700 font-extrabold px-1.5 py-0.2 rounded-full">
              {fixedModels.length}
            </span>
          </div>
          <button
            onClick={() => openUploadModal('fixed')}
            className="flex items-center gap-0.5 text-[10px] font-semibold text-slate-600 hover:text-slate-800 bg-slate-100 hover:bg-slate-200/80 px-2 py-0.5 rounded transition-all border border-slate-200"
            title="Upload model trained on fixed grid tiling"
          >
            <Plus size={12} /> Add
          </button>
        </div>

        <div className="flex flex-col gap-2.5">
          {fixedModels.map(renderModelCard)}
          {fixedModels.length === 0 && !modelsLoading && (
            <p className="text-[11px] text-slate-400 italic px-1 py-1">
              No fixed grid models. Click "+ Add" to upload.
            </p>
          )}
        </div>
      </div>

      {/* Inference settings */}
      <div className="mt-auto border-t border-slate-200/80 pt-3">
        <button
          onClick={() => setSettingsOpen(!settingsOpen)}
          className="flex items-center gap-1 text-xs font-semibold text-slate-500 hover:text-slate-800 transition-colors"
        >
          {settingsOpen ? <ChevronDown size={14} /> : <ChevronRight size={14} />}
          Tiling Settings
        </button>

        {settingsOpen && (
          <div className="mt-2.5 space-y-2.5 fade-in bg-slate-50 p-2.5 rounded-xl border border-slate-200/60">
            {/* Use tiling toggle */}
            <div className="flex items-center justify-between">
              <span className="text-[11px] font-medium text-slate-600">Use Tiling</span>
              <label className="toggle">
                <input
                  type="checkbox"
                  checked={inferSettings.useTiling}
                  onChange={e =>
                    setInferSettings({ useTiling: e.target.checked })
                  }
                />
                <span className="toggle-slider" />
              </label>
            </div>

            {inferSettings.useTiling && (
              <>
                {/* Mode Selector */}
                <div className="space-y-1">
                  <span className="text-[11px] text-slate-500 font-medium">Tiling Strategy</span>
                  <div className="grid grid-cols-2 gap-1 bg-slate-200/60 p-1 rounded-lg">
                    <button
                      type="button"
                      onClick={() => setInferSettings({ tilingMode: 'fixed' })}
                      className={`text-[10px] font-semibold py-1 rounded transition-all ${
                        inferSettings.tilingMode === 'fixed'
                          ? 'bg-white text-indigo-700 shadow-sm'
                          : 'text-slate-600 hover:text-slate-900'
                      }`}
                    >
                      Fixed Grid
                    </button>
                    <button
                      type="button"
                      onClick={() => setInferSettings({ tilingMode: 'adaptive' })}
                      className={`text-[10px] font-semibold py-1 rounded transition-all ${
                        inferSettings.tilingMode === 'adaptive'
                          ? 'bg-white text-indigo-700 shadow-sm'
                          : 'text-slate-600 hover:text-slate-900'
                      }`}
                    >
                      Adaptive (Size)
                    </button>
                  </div>
                </div>

                {inferSettings.tilingMode === 'fixed' ? (
                  /* Fixed Grid controls */
                  <div className="space-y-1">
                    <div className="flex items-center justify-between">
                      <span className="text-[11px] text-slate-500">Grid Size</span>
                      <span className="text-[11px] text-slate-700 tabular-nums font-semibold">
                        {inferSettings.gridSize}×{inferSettings.gridSize}
                      </span>
                    </div>
                    <input
                      type="range"
                      min={1}
                      max={8}
                      step={1}
                      value={inferSettings.gridSize}
                      onChange={e =>
                        setInferSettings({ gridSize: parseInt(e.target.value) })
                      }
                      className="w-full"
                    />
                  </div>
                ) : (
                  /* Adaptive (Size-Based) controls */
                  <div className="space-y-2 pt-1">
                    {/* Auto-Calculated Grid Card */}
                    {(() => {
                      const imgW = imageDimensions?.width ?? 0;
                      const imgH = imageDimensions?.height ?? 0;
                      const targetPx = inferSettings.targetSymbolPx || 48;
                      const estPx = inferSettings.estimatedSymbolPx || 48;
                      const modelSize = 640;
                      const autoCols = imgW > 0 ? Math.max(1, Math.round((targetPx * imgW) / (modelSize * estPx))) : null;
                      const autoRows = imgH > 0 ? Math.max(1, Math.round((targetPx * imgH) / (modelSize * estPx))) : null;
                      const totalTiles = autoCols && autoRows ? autoCols * autoRows : null;

                      return (
                        <div className="bg-indigo-50/80 border border-indigo-100/90 rounded-lg p-2.5 space-y-1.5 text-[11px]">
                          <div className="flex items-center justify-between text-indigo-900 font-semibold">
                            <span>Image Dimension Grid</span>
                            <span className="bg-indigo-600 text-white text-[9px] px-1.5 py-0.5 rounded font-bold uppercase tracking-wider">Auto</span>
                          </div>
                          {imgW > 0 && imgH > 0 ? (
                            <div className="space-y-1 text-slate-600 text-[10px]">
                              <div className="flex justify-between">
                                <span>Image Size:</span>
                                <span className="font-mono font-medium text-slate-800">{imgW} × {imgH} px</span>
                              </div>
                              <div className="flex justify-between">
                                <span>Auto Grid (Cols × Rows):</span>
                                <span className="font-mono font-bold text-indigo-700">{autoCols} × {autoRows}</span>
                              </div>
                              <div className="flex justify-between">
                                <span>Total Tiles:</span>
                                <span className="font-mono font-semibold text-slate-800">{totalTiles} crops</span>
                              </div>
                            </div>
                          ) : (
                            <p className="text-slate-500 italic text-[10px]">
                              Upload an image to automatically compute grid columns & rows based on size.
                            </p>
                          )}
                        </div>
                      );
                    })()}

                    {/* Target Symbol Size */}
                    <div className="space-y-1">
                      <div className="flex items-center justify-between">
                        <span className="text-[11px] text-slate-500">Target Symbol Size</span>
                        <span className="text-[11px] text-slate-700 tabular-nums font-semibold">
                          {inferSettings.targetSymbolPx}px
                        </span>
                      </div>
                      <input
                        type="range"
                        min={16}
                        max={128}
                        step={4}
                        value={inferSettings.targetSymbolPx}
                        onChange={e =>
                          setInferSettings({ targetSymbolPx: parseFloat(e.target.value) })
                        }
                        className="w-full"
                      />
                    </div>

                    {/* Estimated Symbol Size */}
                    <div className="space-y-1">
                      <div className="flex items-center justify-between">
                        <span className="text-[11px] text-slate-500">Est. Symbol Size</span>
                        <span className="text-[11px] text-slate-700 tabular-nums font-semibold">
                          {inferSettings.estimatedSymbolPx}px
                        </span>
                      </div>
                      <input
                        type="range"
                        min={16}
                        max={128}
                        step={4}
                        value={inferSettings.estimatedSymbolPx}
                        onChange={e =>
                          setInferSettings({ estimatedSymbolPx: parseFloat(e.target.value) })
                        }
                        className="w-full"
                      />
                    </div>

                    {/* Toggles */}
                    <div className="flex items-center justify-between pt-1">
                      <span className="text-[11px] text-slate-600">Auto-Crop Margins</span>
                      <label className="toggle">
                        <input
                          type="checkbox"
                          checked={inferSettings.enableAutoCrop}
                          onChange={e =>
                            setInferSettings({ enableAutoCrop: e.target.checked })
                          }
                        />
                        <span className="toggle-slider" />
                      </label>
                    </div>

                    <div className="flex items-center justify-between">
                      <span className="text-[11px] text-slate-600">Scale Normalization</span>
                      <label className="toggle">
                        <input
                          type="checkbox"
                          checked={inferSettings.enableScaleNorm}
                          onChange={e =>
                            setInferSettings({ enableScaleNorm: e.target.checked })
                          }
                        />
                        <span className="toggle-slider" />
                      </label>
                    </div>
                  </div>
                )}

                {/* Overlap */}
                <div className="space-y-1 pt-1 border-t border-slate-200/60">
                  <div className="flex items-center justify-between">
                    <span className="text-[11px] text-slate-500">Overlap</span>
                    <span className="text-[11px] text-slate-700 tabular-nums font-semibold">
                      {(inferSettings.overlap * 100).toFixed(0)}%
                    </span>
                  </div>
                  <input
                    type="range"
                    min={0}
                    max={0.5}
                    step={0.05}
                    value={inferSettings.overlap}
                    onChange={e =>
                      setInferSettings({ overlap: parseFloat(e.target.value) })
                    }
                    className="w-full"
                  />
                </div>
              </>
            )}
          </div>
        )}
      </div>
    </aside>
  );
}
