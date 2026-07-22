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
    inferSettings,
    setInferSettings,
  } = useAppStore();

  const [loadingIds, setLoadingIds] = useState<Set<string>>(new Set());
  const [settingsOpen, setSettingsOpen] = useState(false);

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

  return (
    <aside className="w-72 glass rounded-2xl p-4 flex flex-col gap-4 overflow-y-auto bg-white border border-slate-200/80 shadow-sm">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-xs font-bold uppercase tracking-wider text-slate-400">
            Models
          </h2>
          {modelsLoading && (
            <div className="flex items-center gap-1.5 mt-1 text-xs text-slate-500 font-medium">
              <Loader2 size={12} className="animate-spin" /> Scanning…
            </div>
          )}
        </div>

        <button
          onClick={openUploadModal}
          className="flex items-center gap-1 text-xs font-semibold text-indigo-600 hover:text-indigo-700 bg-indigo-50 hover:bg-indigo-100/80 px-2.5 py-1 rounded-lg transition-all border border-indigo-200/60 shadow-sm"
          title="Upload a new .pth model checkpoint & manifest"
        >
          <Plus size={14} /> Add Model
        </button>
      </div>

      {/* Model list */}
      <div className="flex flex-col gap-3">
        {models.map(model => {
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
                {/* Select checkbox */}
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

                {/* Name + arch badge */}
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

                {/* Eye toggle */}
                <button
                  onClick={() => toggleModelVisible(model.model_id)}
                  className="text-slate-400 hover:text-slate-600 transition-colors shrink-0"
                  title={visible ? 'Hide detections' : 'Show detections'}
                >
                  {visible ? <Eye size={16} /> : <EyeOff size={16} />}
                </button>

                {/* Config gear */}
                <button
                  onClick={() => openConfigModal(model.model_id)}
                  className="text-slate-400 hover:text-slate-600 transition-colors shrink-0"
                  title="Edit class names / threshold"
                >
                  <Settings size={16} />
                </button>
              </div>

              {/* Expanded controls when selected */}
              {selected && (
                <div className="px-3 pb-3 space-y-2.5 fade-in border-t border-slate-100 pt-2.5">
                  {/* Load button */}
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

                  {/* Threshold slider */}
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
        })}

        {models.length === 0 && !modelsLoading && (
          <p className="text-xs text-slate-400 text-center py-4">
            No models found in weights/
          </p>
        )}
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
                {/* Grid size */}
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

                {/* Overlap */}
                <div className="space-y-1">
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
