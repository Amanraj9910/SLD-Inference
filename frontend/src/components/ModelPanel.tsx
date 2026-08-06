import { type DragEvent, useState } from 'react';
import { useAppStore } from '../store/appStore';
import { TilingControls } from './TilingControls';
import {
  CheckSquare,
  Cpu,
  Eye,
  EyeOff,
  Folder,
  FolderPlus,
  Loader2,
  Plus,
  Settings,
  Square,
  Trash2,
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
    groups,
    createGroup,
    deleteGroup,
    moveModelToGroup,
  } = useAppStore();

  const [loadingIds, setLoadingIds] = useState<Set<string>>(new Set());
  const [deletingId, setDeletingId] = useState<string | null>(null);

  const handleLoad = async (modelId: string) => {
    setLoadingIds(previous => new Set(previous).add(modelId));
    try {
      await loadModel(modelId);
    } catch (error) {
      console.error('Failed to load model', error);
    } finally {
      setLoadingIds(previous => {
        const next = new Set(previous);
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
    } catch (error) {
      alert(`Failed to delete model: ${error}`);
    } finally {
      setDeletingId(null);
    }
  };

  const handleCreateGroup = () => {
    const name = window.prompt('Name this component group', 'Circuit Breakers');
    if (name) createGroup(name);
  };

  const handleDrop = (event: DragEvent, groupId: string | null) => {
    event.preventDefault();
    const modelId = event.dataTransfer.getData('text/model-id');
    if (modelId) moveModelToGroup(modelId, groupId);
  };

  const renderModelCard = (model: typeof models[number]) => {
    const selected = selectedModelIds.has(model.model_id);
    const visible = visibleModels[model.model_id] ?? true;
    const threshold = thresholds[model.model_id] ?? model.confidence_default;
    const isLoading = loadingIds.has(model.model_id);
    const hasWeights = model.weights_exist ?? true;

    return (
      <div
        key={model.model_id}
        draggable={hasWeights}
        onDragStart={event => event.dataTransfer.setData('text/model-id', model.model_id)}
        className={`rounded-xl border transition-all duration-200 shadow-sm ${
          !hasWeights
            ? 'border-red-200 bg-red-50/30'
            : selected
            ? 'border-indigo-500/60 bg-indigo-50/40 ring-1 ring-indigo-500/20'
            : 'border-slate-200 bg-white hover:border-slate-300'
        }`}
      >
        <div className="flex items-center gap-2.5 p-3">
          <button
            onClick={() => hasWeights && toggleModelSelected(model.model_id)}
            disabled={!hasWeights}
            className={`transition-colors shrink-0 ${!hasWeights ? 'text-slate-300 cursor-not-allowed' : 'text-slate-400 hover:text-indigo-600'}`}
            title={selected ? 'Deselect model' : 'Select model'}
          >
            {selected ? <CheckSquare size={18} className="text-indigo-600" /> : <Square size={18} />}
          </button>

          <div className="flex-1 min-w-0">
            <p className="text-xs font-semibold text-slate-800 truncate">{model.display_name}</p>
            <div className="flex items-center gap-1.5 mt-0.5 flex-wrap">
              <span className={`text-[9px] font-bold uppercase px-1.5 py-0.5 rounded border ${model.arch === 'dfine' ? 'bg-cyan-50 text-cyan-700 border-cyan-200' : 'bg-amber-50 text-amber-700 border-amber-200'}`}>
                {model.arch}
              </span>
              {hasWeights ? (
                model.loaded ? <span className="text-[10px] font-semibold text-emerald-600">GPU</span> : <span className="text-[10px] text-slate-400">Ready</span>
              ) : (
                <span className="text-[9px] font-semibold text-red-600 bg-red-50 px-1 py-0.5 rounded border border-red-200">Missing .pth</span>
              )}
            </div>
          </div>

          <button onClick={() => toggleModelVisible(model.model_id)} className="text-slate-400 hover:text-slate-600 transition-colors shrink-0" title={visible ? 'Hide detections' : 'Show detections'}>
            {visible ? <Eye size={16} /> : <EyeOff size={16} />}
          </button>
          <button onClick={() => openConfigModal(model.model_id)} className="text-slate-400 hover:text-slate-600 transition-colors shrink-0" title="Edit class names and threshold">
            <Settings size={16} />
          </button>
          <button onClick={() => handleDelete(model.model_id, model.display_name)} disabled={deletingId === model.model_id} className="text-slate-400 hover:text-red-600 transition-colors shrink-0 disabled:opacity-50" title="Delete checkpoint">
            {deletingId === model.model_id ? <Loader2 size={16} className="animate-spin text-red-600" /> : <Trash2 size={16} />}
          </button>
        </div>

        {selected && (
          <div className="px-3 pb-3 space-y-2.5 fade-in border-t border-slate-100 pt-2.5">
            {!model.loaded && (
              <button onClick={() => handleLoad(model.model_id)} disabled={isLoading || !hasWeights} className="w-full flex items-center justify-center gap-2 text-xs font-semibold bg-indigo-50 hover:bg-indigo-100 text-indigo-700 border border-indigo-200 rounded-lg py-1.5 transition-colors disabled:opacity-50">
                {isLoading ? <Loader2 size={13} className="animate-spin" /> : <Cpu size={13} />}
                {isLoading ? 'Loading...' : hasWeights ? 'Load model' : `Missing ${model.weights_file}`}
              </button>
            )}
            <div className="space-y-1">
              <div className="flex items-center justify-between">
                <span className="text-[11px] font-medium text-slate-500">Confidence</span>
                <span className="text-[11px] text-slate-700 tabular-nums font-semibold">{(threshold * 100).toFixed(0)}%</span>
              </div>
              <input type="range" min={0} max={1} step={0.01} value={threshold} onChange={event => setThreshold(model.model_id, parseFloat(event.target.value))} className="w-full" />
            </div>
          </div>
        )}
      </div>
    );
  };

  const groupedModelIds = new Set(groups.flatMap(group => group.modelIds));
  const ungroupedModels = models.filter(model => !groupedModelIds.has(model.model_id));

  return (
    <aside className="w-72 glass rounded-2xl p-4 flex flex-col gap-4 overflow-y-auto bg-white border border-slate-200/80 shadow-sm">
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-xs font-bold uppercase tracking-wider text-slate-400">Model Registry</h2>
          {modelsLoading && <div className="flex items-center gap-1.5 mt-1 text-xs text-slate-500 font-medium"><Loader2 size={12} className="animate-spin" /> Scanning...</div>}
        </div>
        <button onClick={() => openUploadModal()} className="flex items-center gap-1 text-[10px] font-semibold text-indigo-600 hover:text-indigo-700 bg-indigo-50 hover:bg-indigo-100 px-2 py-1 rounded-lg border border-indigo-200/70" title="Upload a model checkpoint">
          <Plus size={12} /> Add model
        </button>
      </div>

      <div className="space-y-2">
        <div className="flex items-center justify-between">
          <span className="text-[11px] font-bold uppercase tracking-wide text-slate-500">Component groups</span>
          <button onClick={handleCreateGroup} className="flex items-center gap-1 text-[10px] font-semibold text-slate-600 hover:text-indigo-700" title="Create a component group">
            <FolderPlus size={13} /> New group
          </button>
        </div>

        {groups.map(group => (
          <div key={group.id} className="rounded-xl border border-indigo-100 bg-indigo-50/30 p-2 space-y-2" onDragOver={event => event.preventDefault()} onDrop={event => handleDrop(event, group.id)}>
            <div className="flex items-center justify-between px-1">
              <div className="flex items-center gap-1.5 text-xs font-semibold text-indigo-900"><Folder size={14} className="text-indigo-600" />{group.name}</div>
              <button onClick={() => deleteGroup(group.id)} className="text-[10px] text-slate-400 hover:text-red-600" title="Delete group">Remove</button>
            </div>
            {group.modelIds.map(modelId => models.find(model => model.model_id === modelId)).filter(Boolean).map(model => renderModelCard(model!))}
            {group.modelIds.length === 0 && <p className="text-[10px] text-indigo-500/70 italic px-1 py-2">Drop a model here</p>}
          </div>
        ))}

        <div className="space-y-2" onDragOver={event => event.preventDefault()} onDrop={event => handleDrop(event, null)}>
          {groups.length > 0 && <p className="text-[10px] font-semibold uppercase tracking-wide text-slate-400 px-1 pt-1">Unassigned models</p>}
          {ungroupedModels.map(renderModelCard)}
          {models.length === 0 && !modelsLoading && <p className="text-[11px] text-slate-400 italic px-1 py-2">No model checkpoints found. Click “Add model” to upload one.</p>}
        </div>
      </div>

      <TilingControls />
    </aside>
  );
}
