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
      className="fixed inset-0 z-50 flex items-center justify-center bg-slate-900/40 backdrop-blur-sm"
      onClick={onClose}
    >
      <div
        className="glass rounded-2xl w-full max-w-4xl max-h-[85vh] mx-4 flex flex-col fade-in shadow-xl overflow-hidden bg-white border border-slate-200"
        onClick={e => e.stopPropagation()}
      >
        {/* Header */}
        <div className="flex items-center justify-between px-6 py-4 border-b border-slate-100 bg-slate-50/80">
          <div className="flex items-center gap-2.5">
            <Terminal size={18} className="text-slate-700" />
            <div>
              <h3 className="text-sm font-bold text-slate-900">
                Backend Server Logs
              </h3>
              <p className="text-xs text-slate-500">
                Output from <code className="font-mono text-slate-700 font-semibold">backend.log</code>
              </p>
            </div>
          </div>

          <div className="flex items-center gap-3">
            <button
              onClick={fetchLogs}
              disabled={loading}
              className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-semibold bg-white hover:bg-slate-100 text-slate-700 border border-slate-200 transition-all shadow-sm disabled:opacity-50"
            >
              <RefreshCw size={13} className={loading ? 'animate-spin' : ''} />
              <span>Refresh</span>
            </button>
            <button
              onClick={onClose}
              className="text-slate-400 hover:text-slate-600 transition-colors"
            >
              <X size={20} />
            </button>
          </div>
        </div>

        {/* Log Viewer Content (Dark terminal viewport inside white modal) */}
        <div className="flex-1 p-6 overflow-y-auto font-mono text-xs text-slate-200 bg-slate-900 leading-relaxed whitespace-pre-wrap selection:bg-indigo-500/30 selection:text-white">
          {logs}
        </div>
      </div>
    </div>
  );
}
