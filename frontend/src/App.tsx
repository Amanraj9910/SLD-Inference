import React, { useCallback, useEffect, useRef, useState } from 'react';
import { useAppStore } from './store/appStore';
import { ModelPanel } from './components/ModelPanel';
import { ImageCanvas } from './components/ImageCanvas';
import { ClassConfigModal } from './components/ClassConfigModal';
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
    <div className="h-screen flex flex-col">
      {/* ── Top bar ───────────────────────────────────────────────────── */}
      <header className="glass border-b border-slate-700/40 px-5 py-3 flex items-center gap-4 shrink-0">
        {/* Logo / title */}
        <div className="flex items-center gap-2.5">
          <Zap size={20} className="text-indigo-400" />
          <h1 className="text-base font-bold gradient-text tracking-tight">
            SLD Inference Viewer
          </h1>
        </div>

        <div className="flex-1" />

        {/* Eye toggle */}
        <button
          onClick={toggleShowLabels}
          className={`flex items-center gap-1.5 text-xs px-3 py-1.5 rounded-lg border transition-all ${
            showLabels
              ? 'border-indigo-500/50 bg-indigo-500/10 text-indigo-300'
              : 'border-slate-700/50 bg-slate-800/30 text-slate-500 hover:text-slate-300'
          }`}
          title={showLabels ? 'Labels always visible' : 'Labels only on hover'}
        >
          {showLabels ? <Eye size={14} /> : <EyeOff size={14} />}
          <span>{showLabels ? 'Labels On' : 'Labels Off'}</span>
        </button>

        {/* Server logs button */}
        <button
          onClick={() => setLogsOpen(true)}
          className="flex items-center gap-1.5 text-xs px-3 py-1.5 rounded-lg
                     border border-slate-700/50 bg-slate-800/30
                     text-slate-400 hover:text-cyan-300 hover:border-cyan-500/40
                     transition-all"
          title="View live backend server logs"
        >
          <Terminal size={14} className="text-cyan-400" />
          <span>Server Logs</span>
        </button>

        {/* Upload button */}
        <button
          onClick={() => fileInputRef.current?.click()}
          className="flex items-center gap-1.5 text-xs px-3 py-1.5 rounded-lg
                     border border-slate-700/50 bg-slate-800/30
                     text-slate-400 hover:text-slate-200 hover:border-slate-600
                     transition-all"
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
                     bg-indigo-600 hover:bg-indigo-500 text-white
                     disabled:opacity-40 disabled:cursor-not-allowed
                     transition-all shadow-lg shadow-indigo-500/20"
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
        <div className="p-3 shrink-0">
          <ModelPanel />
        </div>

        {/* Center canvas area */}
        <main className="flex-1 flex flex-col p-3 pl-0 gap-3 min-w-0">
          {/* Canvas / drop zone */}
          <div
            className={`flex-1 glass rounded-xl overflow-hidden flex ${
              !currentImageUrl
                ? 'items-center justify-center border-2 border-dashed border-slate-700/60'
                : ''
            }`}
            onDragOver={e => e.preventDefault()}
            onDrop={handleDrop}
          >
            {currentImageUrl ? (
              <ImageCanvas />
            ) : (
              <div className="text-center space-y-3 p-8">
                <div className="w-16 h-16 mx-auto rounded-2xl bg-slate-800/50 border border-slate-700/50 flex items-center justify-center">
                  <Upload size={24} className="text-slate-600" />
                </div>
                <p className="text-sm text-slate-500">
                  Drag & drop an SLD image here
                </p>
                <p className="text-xs text-slate-600">or use the Upload button above</p>
              </div>
            )}
          </div>

          {/* Error banner */}
          {inferError && (
            <div className="flex items-center justify-between bg-red-500/10 border border-red-500/30 rounded-lg px-4 py-2.5 text-xs text-red-300 fade-in">
              <div className="flex items-center gap-2">
                <AlertCircle size={14} className="shrink-0" />
                <span>{inferError}</span>
              </div>
              <button
                onClick={() => setLogsOpen(true)}
                className="underline hover:text-white font-medium text-[11px] shrink-0 ml-2"
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
      <LogsModal isOpen={logsOpen} onClose={() => setLogsOpen(false)} />
    </div>
  );
}
