import { useState, useRef } from 'react';
import { useAppStore } from '../store/appStore';
import { X, Upload, FileCode, AlertCircle, Loader2 } from 'lucide-react';

export function UploadModelModal() {
  const { uploadModalOpen, closeUploadModal, uploadModel } = useAppStore();

  const [arch, setArch] = useState<'dfine' | 'rfdetr'>('dfine');
  const [displayName, setDisplayName] = useState('');
  const [file, setFile] = useState<File | null>(null);
  const [classNamesText, setClassNamesText] = useState('');
  const [resolution, setResolution] = useState(640);
  const [confidenceDefault, setConfidenceDefault] = useState(0.20);
  const [gridSize, setGridSize] = useState(4);
  const [overlap, setOverlap] = useState(0.20);

  const [uploading, setUploading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const fileInputRef = useRef<HTMLInputElement>(null);

  if (!uploadModalOpen) return null;

  const handleArchChange = (newArch: 'dfine' | 'rfdetr') => {
    setArch(newArch);
    setResolution(newArch === 'dfine' ? 640 : 560);
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError(null);

    if (!displayName.trim()) {
      setError('Please enter a display name for the model.');
      return;
    }

    if (!file) {
      setError('Please select a PyTorch weight file (.pth or .pt).');
      return;
    }

    const classNames = classNamesText
      .split('\n')
      .map(s => s.trim())
      .filter(Boolean);

    if (classNames.length === 0) {
      setError('Please enter at least 1 class name (one per line).');
      return;
    }

    setUploading(true);
    try {
      await uploadModel({
        file,
        arch,
        displayName: displayName.trim(),
        numClasses: classNames.length,
        classNames,
        resolution,
        confidenceDefault,
        gridSize,
        overlap,
      });
      closeUploadModal();
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setUploading(false);
    }
  };

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-slate-900/40 backdrop-blur-sm"
      onClick={closeUploadModal}
    >
      <div
        className="glass rounded-2xl w-full max-w-lg max-h-[90vh] mx-4 p-6 space-y-4 fade-in shadow-xl bg-white border border-slate-200 overflow-y-auto"
        onClick={e => e.stopPropagation()}
      >
        {/* Header */}
        <div className="flex items-start justify-between border-b border-slate-100 pb-3">
          <div>
            <h3 className="text-base font-bold text-slate-900 flex items-center gap-2">
              <FileCode size={18} className="text-indigo-600" /> Upload Model Checkpoint
            </h3>
            <p className="text-xs text-slate-500 mt-0.5">
              Add a new D-FINE or RF-DETR weight file and manifest
            </p>
          </div>
          <button
            onClick={closeUploadModal}
            className="text-slate-400 hover:text-slate-600 transition-colors"
          >
            <X size={20} />
          </button>
        </div>

        <form onSubmit={handleSubmit} className="space-y-4">
          {/* Architecture selection */}
          <div className="space-y-1">
            <label className="text-xs font-semibold text-slate-700">Architecture</label>
            <div className="grid grid-cols-2 gap-3">
              <button
                type="button"
                onClick={() => handleArchChange('dfine')}
                className={`py-2 px-3 rounded-xl text-xs font-semibold border transition-all ${
                  arch === 'dfine'
                    ? 'border-cyan-500/60 bg-cyan-50 text-cyan-800'
                    : 'border-slate-200 bg-white text-slate-600 hover:bg-slate-50'
                }`}
              >
                D-FINE
              </button>
              <button
                type="button"
                onClick={() => handleArchChange('rfdetr')}
                className={`py-2 px-3 rounded-xl text-xs font-semibold border transition-all ${
                  arch === 'rfdetr'
                    ? 'border-amber-500/60 bg-amber-50 text-amber-800'
                    : 'border-slate-200 bg-white text-slate-600 hover:bg-slate-50'
                }`}
              >
                RF-DETR
              </button>
            </div>
          </div>

          {/* Model Display Name */}
          <div className="space-y-1">
            <label className="text-xs font-semibold text-slate-700">Model Display Name</label>
            <input
              type="text"
              required
              placeholder="e.g. D-FINE SLD Stage 2"
              value={displayName}
              onChange={e => setDisplayName(e.target.value)}
              className="w-full bg-slate-50 border border-slate-200 rounded-xl px-3 py-2 text-xs text-slate-800 focus:border-indigo-500/50 focus:ring-2 focus:ring-indigo-500/10 outline-none"
            />
          </div>

          {/* .pth Weight file picker */}
          <div className="space-y-1">
            <label className="text-xs font-semibold text-slate-700">
              Checkpoint File (.pth / .pt)
            </label>
            <div
              onClick={() => fileInputRef.current?.click()}
              className="border-2 border-dashed border-slate-200 hover:border-indigo-500/50 rounded-xl p-4 text-center cursor-pointer bg-slate-50/50 transition-all"
            >
              <Upload size={20} className="mx-auto text-slate-400 mb-1" />
              <p className="text-xs text-slate-700 font-semibold">
                {file ? file.name : 'Click to select .pth file'}
              </p>
              {file && (
                <p className="text-[10px] text-slate-400 mt-0.5 font-mono">
                  {(file.size / (1024 * 1024)).toFixed(1)} MB
                </p>
              )}
            </div>
            <input
              ref={fileInputRef}
              type="file"
              accept=".pth,.pt"
              className="hidden"
              onChange={e => {
                const f = e.target.files?.[0];
                if (f) setFile(f);
              }}
            />
          </div>

          {/* Class names text area */}
          <div className="space-y-1">
            <div className="flex items-center justify-between">
              <label className="text-xs font-semibold text-slate-700">Class Names</label>
              <span className="text-[10px] text-slate-400 font-medium">
                {classNamesText.split('\n').filter(s => s.trim()).length} classes
              </span>
            </div>
            <textarea
              required
              rows={4}
              placeholder={`Enter 1 class name per line:\nACB\nATS\nAmmeter\n...`}
              value={classNamesText}
              onChange={e => setClassNamesText(e.target.value)}
              className="w-full bg-slate-50 border border-slate-200 rounded-xl p-3 text-xs text-slate-800 font-mono leading-relaxed focus:border-indigo-500/50 focus:ring-2 focus:ring-indigo-500/10 outline-none resize-y"
            />
          </div>

          {/* Parameters grid */}
          <div className="grid grid-cols-2 gap-3 pt-1">
            <div className="space-y-1">
              <label className="text-[11px] font-semibold text-slate-600">Resolution (px)</label>
              <input
                type="number"
                value={resolution}
                onChange={e => setResolution(parseInt(e.target.value) || 640)}
                className="w-full bg-slate-50 border border-slate-200 rounded-xl px-3 py-1.5 text-xs text-slate-800 outline-none"
              />
            </div>
            <div className="space-y-1">
              <label className="text-[11px] font-semibold text-slate-600">Default Confidence</label>
              <input
                type="number"
                step={0.05}
                min={0}
                max={1}
                value={confidenceDefault}
                onChange={e => setConfidenceDefault(parseFloat(e.target.value) || 0.2)}
                className="w-full bg-slate-50 border border-slate-200 rounded-xl px-3 py-1.5 text-xs text-slate-800 outline-none"
              />
            </div>
            <div className="space-y-1">
              <label className="text-[11px] font-semibold text-slate-600">Tiling Grid Size</label>
              <input
                type="number"
                min={1}
                max={8}
                value={gridSize}
                onChange={e => setGridSize(parseInt(e.target.value) || 4)}
                className="w-full bg-slate-50 border border-slate-200 rounded-xl px-3 py-1.5 text-xs text-slate-800 outline-none"
              />
            </div>
            <div className="space-y-1">
              <label className="text-[11px] font-semibold text-slate-600">Tile Overlap</label>
              <input
                type="number"
                step={0.05}
                min={0}
                max={0.5}
                value={overlap}
                onChange={e => setOverlap(parseFloat(e.target.value) || 0.2)}
                className="w-full bg-slate-50 border border-slate-200 rounded-xl px-3 py-1.5 text-xs text-slate-800 outline-none"
              />
            </div>
          </div>

          {/* Error Message */}
          {error && (
            <div className="flex items-center gap-2 bg-red-50 border border-red-200 rounded-xl p-2.5 text-xs text-red-700 font-medium">
              <AlertCircle size={14} className="shrink-0 text-red-500" />
              <span>{error}</span>
            </div>
          )}

          {/* Submit Actions */}
          <div className="flex justify-end gap-3 pt-2 border-t border-slate-100">
            <button
              type="button"
              onClick={closeUploadModal}
              className="px-4 py-2 text-xs font-medium text-slate-500 hover:text-slate-800 transition-colors"
            >
              Cancel
            </button>
            <button
              type="submit"
              disabled={uploading}
              className="flex items-center gap-2 px-4 py-2 text-xs font-semibold bg-indigo-600 hover:bg-indigo-700 text-white rounded-xl transition-all shadow-sm disabled:opacity-50"
            >
              {uploading ? <Loader2 size={14} className="animate-spin" /> : <Upload size={14} />}
              {uploading ? 'Uploading Checkpoint…' : 'Upload Model'}
            </button>
          </div>
        </form>
      </div>
    </div>
  );
}
