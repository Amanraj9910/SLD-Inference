import { useState, useEffect } from 'react';
import { useAppStore } from '../store/appStore';
import { X, Save, AlertTriangle } from 'lucide-react';

export function ClassConfigModal() {
  const { configModalModelId, closeConfigModal, models, updateModelConfig } =
    useAppStore();

  const model = models.find(m => m.model_id === configModalModelId);

  const [classText, setClassText] = useState('');
  const [threshold, setThreshold] = useState(0.2);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Sync local state when the modal opens / model changes
  useEffect(() => {
    if (model) {
      setClassText(model.class_names.join('\n'));
      setThreshold(model.confidence_default);
      setError(null);
    }
  }, [model]);

  if (!configModalModelId || !model) return null;

  const handleSave = async () => {
    setSaving(true);
    setError(null);
    try {
      const names = classText
        .split('\n')
        .map(l => l.trim())
        .filter(l => l.length > 0);

      if (names.length !== model.num_classes) {
        setError(
          `You provided ${names.length} class names but the checkpoint expects exactly ${model.num_classes}. ` +
          `Each line becomes one class name.`
        );
        setSaving(false);
        return;
      }

      await updateModelConfig(model.model_id, {
        class_names: names,
        confidence_default: threshold,
      });
      closeConfigModal();
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setSaving(false);
    }
  };

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 backdrop-blur-sm"
      onClick={closeConfigModal}
    >
      <div
        className="glass rounded-2xl w-full max-w-lg mx-4 p-6 space-y-5 fade-in shadow-2xl"
        onClick={e => e.stopPropagation()}
      >
        {/* Header */}
        <div className="flex items-start justify-between">
          <div>
            <h3 className="text-lg font-semibold text-slate-100">
              Configure Model
            </h3>
            <p className="text-xs text-slate-500 mt-0.5">{model.display_name}</p>
          </div>
          <button
            onClick={closeConfigModal}
            className="text-slate-500 hover:text-slate-300 transition-colors"
          >
            <X size={20} />
          </button>
        </div>

        {/* num_classes (read-only) */}
        <div className="flex items-center gap-3 bg-slate-800/50 rounded-lg p-3">
          <AlertTriangle size={16} className="text-amber-400 shrink-0" />
          <div>
            <p className="text-xs text-slate-300">
              <span className="font-semibold text-amber-300">num_classes = {model.num_classes}</span>{' '}
              — baked into the checkpoint's output layer. Changing it requires a
              different checkpoint.
            </p>
          </div>
        </div>

        {/* Default confidence */}
        <div className="space-y-1">
          <label className="text-xs font-medium text-slate-400">
            Default Confidence Threshold
          </label>
          <div className="flex items-center gap-3">
            <input
              type="range"
              min={0}
              max={1}
              step={0.01}
              value={threshold}
              onChange={e => setThreshold(parseFloat(e.target.value))}
              className="flex-1"
            />
            <span className="text-sm text-slate-300 tabular-nums font-medium w-12 text-right">
              {(threshold * 100).toFixed(0)}%
            </span>
          </div>
        </div>

        {/* Class names editor */}
        <div className="space-y-1">
          <label className="text-xs font-medium text-slate-400">
            Class Names{' '}
            <span className="text-slate-600">(one per line, {model.num_classes} total)</span>
          </label>
          <textarea
            value={classText}
            onChange={e => setClassText(e.target.value)}
            rows={Math.min(model.num_classes, 12)}
            className="w-full bg-slate-800/50 border border-slate-700/60 rounded-lg p-3
                       text-sm text-slate-200 font-mono leading-relaxed
                       placeholder:text-slate-600 focus:border-indigo-500/50
                       focus:ring-1 focus:ring-indigo-500/30 outline-none resize-y"
            placeholder={`Line 1: class name for id=0\nLine 2: class name for id=1\n…`}
          />
          <p className="text-[10px] text-slate-600">
            {classText.split('\n').filter(l => l.trim()).length} / {model.num_classes} names
          </p>
        </div>

        {/* Error */}
        {error && (
          <div className="bg-red-500/10 border border-red-500/30 rounded-lg p-3 text-xs text-red-300">
            {error}
          </div>
        )}

        {/* Actions */}
        <div className="flex justify-end gap-3">
          <button
            onClick={closeConfigModal}
            className="px-4 py-2 text-xs font-medium text-slate-400 hover:text-slate-200 transition-colors"
          >
            Cancel
          </button>
          <button
            onClick={handleSave}
            disabled={saving}
            className="flex items-center gap-2 px-4 py-2 text-xs font-semibold
                       bg-indigo-600 hover:bg-indigo-500 text-white rounded-lg
                       transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
          >
            <Save size={14} />
            {saving ? 'Saving…' : 'Save Config'}
          </button>
        </div>
      </div>
    </div>
  );
}
