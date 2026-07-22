import React, { useCallback, useEffect, useRef, useState } from 'react';
import { useAppStore } from './store/appStore';
import { ModelPanel } from './components/ModelPanel';
import { ImageCanvas } from './components/ImageCanvas';
import { ClassConfigModal } from './components/ClassConfigModal';
import { UploadModelModal } from './components/UploadModelModal';
import { LogsModal } from './components/LogsModal';
import { Legend } from './components/Legend';
import {
  Eye,
  EyeOff,
  Upload,
  Play,
  Loader2,
  AlertCircle,
  Zap,
  Terminal,
} from 'lucide-react';

export default function App() {
  const {
    fetchModels,
    currentImageUrl,
    setImage,
    selectedModelIds,
    isInferring,
    inferError,
    runInfer,
    showLabels,
    toggleShowLabels,
    detectionResults,
  } = useAppStore();

  const fileInputRef = useRef<HTMLInputElement>(null);
  const [logsOpen, setLogsOpen] = useState(false);

  // Fetch models on mount
  useEffect(() => {
    fetchModels();
  }, [fetchModels]);

  // File input handler
  const handleFile = useCallback(
    (file: File) => {
      if (!file.type.startsWith('image/')) return;
      setImage(file);
    },
    [setImage]
  );

  // Drag & drop
  const handleDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      const file = e.dataTransfer.files[0];
      if (file) handleFile(file);
    },
    [handleFile]
  );

  const hasResults = Object.keys(detectionResults).length > 0;
  const canRunInfer = selectedModelIds.size > 0 && currentImageUrl !== null;

  return (
    <div className="h-screen flex flex-col bg-slate-50 text-slate-900">
      {/* ── Top bar ───────────────────────────────────────────────────── */}
      <header className="glass border-b border-slate-200/80 px-6 py-3.5 flex items-center gap-4 shrink-0 bg-white/80">
        {/* Logo / title */}
        <div className="flex items-center gap-2.5">
          <div className="w-8 h-8 rounded-lg bg-indigo-600/10 flex items-center justify-center border border-indigo-600/20">
            <Zap size={18} className="text-indigo-600" />
          </div>
          <div>
            <h1 className="text-sm font-bold text-slate-900 tracking-tight">
              SLD Inference Viewer
            </h1>
            <p className="text-[10px] text-slate-500 font-medium">Multi-Model Inspection</p>
          </div>
        </div>

        <div className="flex-1" />

        {/* Eye toggle */}
        <button
          onClick={toggleShowLabels}
          className={`flex items-center gap-1.5 text-xs font-medium px-3 py-1.5 rounded-lg border transition-all ${
            showLabels
              ? 'border-indigo-600/40 bg-indigo-50 text-indigo-700'
              : 'border-slate-200 bg-white text-slate-600 hover:bg-slate-50 hover:text-slate-900'
          }`}
          title={showLabels ? 'Labels always visible' : 'Labels only on hover'}
        >
          {showLabels ? <Eye size={14} /> : <EyeOff size={14} />}
          <span>{showLabels ? 'Labels On' : 'Labels Off'}</span>
        </button>

        {/* Server logs button */}
        <button
          onClick={() => setLogsOpen(true)}
          className="flex items-center gap-1.5 text-xs font-medium px-3 py-1.5 rounded-lg
                     border border-slate-200 bg-white
                     text-slate-600 hover:text-slate-900 hover:bg-slate-50
                     transition-all shadow-sm"
          title="View live backend server logs"
        >
          <Terminal size={14} className="text-slate-500" />
          <span>Server Logs</span>
        </button>

        {/* Upload button */}
        <button
          onClick={() => fileInputRef.current?.click()}
          className="flex items-center gap-1.5 text-xs font-medium px-3 py-1.5 rounded-lg
                     border border-slate-200 bg-white
                     text-slate-700 hover:text-slate-900 hover:bg-slate-50
                     transition-all shadow-sm"
        >
          <Upload size={14} />
          <span>Upload Image</span>
        </button>
        <input
          ref={fileInputRef}
          type="file"
          accept="image/*"
          className="hidden"
          onChange={e => {
            const f = e.target.files?.[0];
            if (f) handleFile(f);
            e.target.value = '';
          }}
        />

        {/* Run inference */}
        <button
          onClick={runInfer}
          disabled={!canRunInfer || isInferring}
          className="flex items-center gap-1.5 text-xs font-semibold px-4 py-1.5 rounded-lg
                     bg-indigo-600 hover:bg-indigo-700 text-white
                     disabled:opacity-40 disabled:cursor-not-allowed
                     transition-all shadow-sm shadow-indigo-600/20"
        >
          {isInferring ? (
            <Loader2 size={14} className="animate-spin" />
          ) : (
            <Play size={14} />
          )}
          {isInferring ? 'Running…' : 'Run Inference'}
        </button>
      </header>

      {/* ── Main area ─────────────────────────────────────────────────── */}
      <div className="flex-1 flex min-h-0">
        {/* Left sidebar */}
        <div className="p-4 shrink-0">
          <ModelPanel />
        </div>

        {/* Center canvas area */}
        <main className="flex-1 flex flex-col p-4 pl-0 gap-4 min-w-0">
          {/* Canvas / drop zone */}
          <div
            className={`flex-1 glass rounded-2xl overflow-hidden flex bg-white border border-slate-200/80 shadow-sm ${
              !currentImageUrl
                ? 'items-center justify-center border-2 border-dashed border-slate-200 bg-slate-50/50'
                : ''
            }`}
            onDragOver={e => e.preventDefault()}
            onDrop={handleDrop}
          >
            {currentImageUrl ? (
              <ImageCanvas />
            ) : (
              <div className="text-center space-y-3 p-8">
                <div className="w-14 h-14 mx-auto rounded-2xl bg-white border border-slate-200 flex items-center justify-center shadow-sm">
                  <Upload size={22} className="text-slate-400" />
                </div>
                <div>
                  <p className="text-sm font-semibold text-slate-700">
                    Drag & drop an SLD image here
                  </p>
                  <p className="text-xs text-slate-400 mt-1">or click Upload Image above</p>
                </div>
              </div>
            )}
          </div>

          {/* Error banner */}
          {inferError && (
            <div className="flex items-center justify-between bg-red-50 border border-red-200 rounded-xl px-4 py-2.5 text-xs text-red-700 fade-in shadow-sm">
              <div className="flex items-center gap-2">
                <AlertCircle size={15} className="shrink-0 text-red-500" />
                <span>{inferError}</span>
              </div>
              <button
                onClick={() => setLogsOpen(true)}
                className="underline hover:text-red-900 font-semibold text-[11px] shrink-0 ml-2"
              >
                View Backend Logs
              </button>
            </div>
          )}

          {/* Legend */}
          {hasResults && <Legend />}
        </main>
      </div>

      {/* ── Modals ────────────────────────────────────────────────────── */}
      <ClassConfigModal />
      <UploadModelModal />
      <LogsModal isOpen={logsOpen} onClose={() => setLogsOpen(false)} />
    </div>
  );
}
