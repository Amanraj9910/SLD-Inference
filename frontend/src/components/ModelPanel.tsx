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
    <aside className="w-72 glass rounded-xl p-4 flex flex-col gap-4 overflow-y-auto">
      {/* Header */}
      <div>
        <h2 className="text-sm font-semibold uppercase tracking-wider text-slate-400">
          Models
        </h2>
        {modelsLoading && (
          <div className="flex items-center gap-2 mt-2 text-xs text-slate-500">
            <Loader2 size={14} className="animate-spin" /> Scanning weights…
          </div>
        )}
      </div>

      {/* Model list */}
      <div className="flex flex-col gap-3">
        {models.map(model => {
          const selected = selectedModelIds.has(model.model_id);
          const visible = visibleModels[model.model_id] ?? true;
          const threshold = thresholds[model.model_id] ?? model.confidence_default;
          const isLoading = loadingIds.has(model.model_id);

          return (
            <div
              key={model.model_id}
              className={`rounded-lg border transition-all duration-200 ${
                selected
                  ? 'border-indigo-500/50 bg-indigo-500/5'
                  : 'border-slate-700/50 bg-slate-800/30'
              }`}
            >
              {/* Top row: checkbox, name, buttons */}
              <div className="flex items-center gap-2 p-3">
                {/* Select checkbox */}
                <button
                  onClick={() => toggleModelSelected(model.model_id)}
                  className="text-slate-400 hover:text-indigo-400 transition-colors shrink-0"
                  title={selected ? 'Deselect model' : 'Select model'}
                >
                  {selected ? (
                    <CheckSquare size={18} className="text-indigo-400" />
                  ) : (
                    <Square size={18} />
                  )}
                </button>

                {/* Name + arch badge */}
                <div className="flex-1 min-w-0">
                  <p className="text-sm font-medium text-slate-200 truncate">
                    {model.display_name}
                  </p>
                  <div className="flex items-center gap-2 mt-0.5">
                    <span
                      className={`text-[10px] font-semibold uppercase px-1.5 py-0.5 rounded ${
                        model.arch === 'dfine'
                          ? 'bg-cyan-500/15 text-cyan-400'
                          : 'bg-amber-500/15 text-amber-400'
                      }`}
                    >
                      {model.arch}
                    </span>
                    <span className="text-[10px] text-slate-500">
                      {model.num_classes} classes
                    </span>
                    {model.loaded && (
                      <span className="text-[10px] text-emerald-400">● GPU</span>
                    )}
                  </div>
                </div>

                {/* Eye toggle */}
                <button
                  onClick={() => toggleModelVisible(model.model_id)}
                  className="text-slate-500 hover:text-slate-300 transition-colors shrink-0"
                  title={visible ? 'Hide detections' : 'Show detections'}
                >
                  {visible ? <Eye size={16} /> : <EyeOff size={16} />}
                </button>

                {/* Config gear */}
                <button
                  onClick={() => openConfigModal(model.model_id)}
                  className="text-slate-500 hover:text-slate-300 transition-colors shrink-0"
                  title="Edit class names / threshold"
                >
                  <Settings size={16} />
                </button>
              </div>

              {/* Expanded controls when selected */}
              {selected && (
                <div className="px-3 pb-3 space-y-2 fade-in">
                  {/* Load button */}
                  {!model.loaded && (
                    <button
                      onClick={() => handleLoad(model.model_id)}
                      disabled={isLoading}
                      className="w-full flex items-center justify-center gap-2 text-xs font-medium
                                 bg-indigo-600/20 hover:bg-indigo-600/30 text-indigo-300
                                 border border-indigo-500/30 rounded-md py-1.5
                                 transition-colors disabled:opacity-50"
                    >
                      {isLoading ? (
                        <Loader2 size={14} className="animate-spin" />
                      ) : (
                        <Cpu size={14} />
                      )}
                      {isLoading ? 'Loading…' : 'Load to GPU'}
                    </button>
                  )}

                  {/* Threshold slider */}
                  <div className="space-y-1">
                    <div className="flex items-center justify-between">
                      <span className="text-[11px] text-slate-500">Confidence</span>
                      <span className="text-[11px] text-slate-400 tabular-nums font-medium">
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
          <p className="text-xs text-slate-600 text-center py-4">
            No models found in weights/
          </p>
        )}
      </div>

      {/* Inference settings */}
      <div className="mt-auto border-t border-slate-700/50 pt-3">
        <button
          onClick={() => setSettingsOpen(!settingsOpen)}
          className="flex items-center gap-1 text-xs text-slate-500 hover:text-slate-300 transition-colors"
        >
          {settingsOpen ? <ChevronDown size={14} /> : <ChevronRight size={14} />}
          Tiling Settings
        </button>

        {settingsOpen && (
          <div className="mt-2 space-y-2 fade-in">
            {/* Use tiling toggle */}
            <div className="flex items-center justify-between">
              <span className="text-[11px] text-slate-500">Use Tiling</span>
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
                    <span className="text-[11px] text-slate-500">Grid</span>
                    <span className="text-[11px] text-slate-400 tabular-nums font-medium">
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
                    <span className="text-[11px] text-slate-400 tabular-nums font-medium">
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
