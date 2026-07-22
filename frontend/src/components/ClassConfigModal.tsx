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
      className="fixed inset-0 z-50 flex items-center justify-center bg-slate-900/40 backdrop-blur-sm"
      onClick={closeConfigModal}
    >
      <div
        className="glass rounded-2xl w-full max-w-lg mx-4 p-6 space-y-5 fade-in shadow-xl bg-white border border-slate-200"
        onClick={e => e.stopPropagation()}
      >
        {/* Header */}
        <div className="flex items-start justify-between border-b border-slate-100 pb-3">
          <div>
            <h3 className="text-base font-bold text-slate-900">
              Configure Model
            </h3>
            <p className="text-xs text-slate-500 mt-0.5">{model.display_name}</p>
          </div>
          <button
            onClick={closeConfigModal}
            className="text-slate-400 hover:text-slate-600 transition-colors"
          >
            <X size={20} />
          </button>
        </div>

        {/* num_classes (read-only) */}
        <div className="flex items-center gap-3 bg-amber-50/60 border border-amber-200/60 rounded-xl p-3">
          <AlertTriangle size={16} className="text-amber-600 shrink-0" />
          <div>
            <p className="text-xs text-amber-900 leading-relaxed">
              <span className="font-bold text-amber-700">num_classes = {model.num_classes}</span>{' '}
              — baked into the output layer shape. Changing it requires a different checkpoint.
            </p>
          </div>
        </div>

        {/* Default confidence */}
        <div className="space-y-1.5">
          <label className="text-xs font-semibold text-slate-700">
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
            <span className="text-xs font-semibold text-slate-900 tabular-nums w-12 text-right">
              {(threshold * 100).toFixed(0)}%
            </span>
          </div>
        </div>

        {/* Class names editor */}
        <div className="space-y-1.5">
          <div className="flex items-center justify-between">
            <label className="text-xs font-semibold text-slate-700">Class Names</label>
            <span className="text-[10px] text-slate-400 font-medium">
              1 per line, {model.num_classes} total
            </span>
          </div>
          <textarea
            value={classText}
            onChange={e => setClassText(e.target.value)}
            rows={Math.min(model.num_classes, 10)}
            className="w-full bg-slate-50 border border-slate-200 rounded-xl p-3
                       text-xs text-slate-800 font-mono leading-relaxed
                       placeholder:text-slate-400 focus:border-indigo-500/50
                       focus:ring-2 focus:ring-indigo-500/10 outline-none resize-y"
            placeholder={`Line 1: class name for id=0\nLine 2: class name for id=1\n…`}
          />
          <p className="text-[10px] text-slate-400 text-right">
            {classText.split('\n').filter(l => l.trim()).length} / {model.num_classes} filled
          </p>
        </div>

        {/* Error */}
        {error && (
          <div className="bg-red-50 border border-red-200 rounded-xl p-3 text-xs text-red-700 font-medium">
            {error}
          </div>
        )}

        {/* Actions */}
        <div className="flex justify-end gap-3 pt-2 border-t border-slate-100">
          <button
            onClick={closeConfigModal}
            className="px-4 py-2 text-xs font-medium text-slate-500 hover:text-slate-800 transition-colors"
          >
            Cancel
          </button>
          <button
            onClick={handleSave}
            disabled={saving}
            className="flex items-center gap-2 px-4 py-2 text-xs font-semibold
                       bg-indigo-600 hover:bg-indigo-700 text-white rounded-xl
                       transition-all shadow-sm disabled:opacity-50"
          >
            <Save size={14} />
            {saving ? 'Saving…' : 'Save Config'}
          </button>
        </div>
      </div>
    </div>
  );
}
