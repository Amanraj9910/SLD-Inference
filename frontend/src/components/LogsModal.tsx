import { useState, useEffect } from 'react';
import { api } from '../api/client';
import { X, RefreshCw, Terminal } from 'lucide-react';

interface LogsModalProps {
  isOpen: boolean;
  onClose: () => void;
}

export function LogsModal({ isOpen, onClose }: LogsModalProps) {
  const [logs, setLogs] = useState<string>('Loading backend logs…');
  const [loading, setLoading] = useState(false);

  const fetchLogs = async () => {
    setLoading(true);
    try {
      const res = await api.getLogs(150);
      setLogs(res.logs || 'No log output yet.');
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : String(err);
      setLogs(`Failed to fetch backend logs: ${msg}`);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    if (isOpen) {
      fetchLogs();
    }
  }, [isOpen]);

  if (!isOpen) return null;

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/70 backdrop-blur-md"
      onClick={onClose}
    >
      <div
        className="glass rounded-2xl w-full max-w-4xl max-h-[85vh] mx-4 flex flex-col fade-in shadow-2xl overflow-hidden border border-slate-700/60"
        onClick={e => e.stopPropagation()}
      >
        {/* Header */}
        <div className="flex items-center justify-between px-6 py-4 border-b border-slate-700/50 bg-slate-900/50">
          <div className="flex items-center gap-2.5">
            <Terminal size={20} className="text-cyan-400" />
            <div>
              <h3 className="text-base font-semibold text-slate-100">
                Backend Server Logs
              </h3>
              <p className="text-xs text-slate-400">
                Live output from <code className="font-mono text-cyan-300">backend.log</code>
              </p>
            </div>
          </div>

          <div className="flex items-center gap-3">
            <button
              onClick={fetchLogs}
              disabled={loading}
              className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-medium bg-slate-800 hover:bg-slate-700 text-slate-300 transition-colors disabled:opacity-50"
            >
              <RefreshCw size={14} className={loading ? 'animate-spin' : ''} />
              <span>Refresh</span>
            </button>
            <button
              onClick={onClose}
              className="text-slate-400 hover:text-slate-200 transition-colors"
            >
              <X size={20} />
            </button>
          </div>
        </div>

        {/* Log Viewer Content */}
        <div className="flex-1 p-6 overflow-y-auto font-mono text-xs text-slate-300 bg-slate-950/80 leading-relaxed whitespace-pre-wrap selection:bg-cyan-500/30 selection:text-white">
          {logs}
        </div>
      </div>
    </div>
  );
}
