import { useState, useMemo, useEffect } from 'react';
import { useAppStore } from '../store/appStore';
import { X, Copy, Check, Braces, FileText, Search, Download, Cpu, Layers } from 'lucide-react';

interface JSONOutputModalProps {
  isOpen: boolean;
  onClose: () => void;
}

export function JSONOutputModal({ isOpen, onClose }: JSONOutputModalProps) {
  const { detectionResults, models } = useAppStore();
  const [userTab, setUserTab] = useState<string | null>(null);
  const [searchQuery, setSearchQuery] = useState<string>('');
  const [copied, setCopied] = useState(false);

  // Reset tab selection & search query whenever modal is reopened
  useEffect(() => {
    if (isOpen) {
      setUserTab(null);
      setSearchQuery('');
    }
  }, [isOpen]);

  // 1. Map model_id to display_name
  const modelNameMap = useMemo(() => {
    const map: Record<string, string> = {};
    models.forEach(m => {
      map[m.model_id] = m.display_name;
    });
    return map;
  }, [models]);

  // 2. Unconditional Data Parsing
  const rawOcr = detectionResults?.ocr || [];
  const modelIds = Object.keys(detectionResults?.detections || {});
  const q = searchQuery.trim().toLowerCase();

  const defaultTab = rawOcr.length > 0 ? 'ocr' : (modelIds[0] || 'ocr');
  const validTab = userTab && (userTab === 'ocr' || userTab === 'all' || modelIds.includes(userTab))
    ? userTab
    : defaultTab;

  const filteredOcr = useMemo(() => {
    if (!q) return rawOcr;
    return rawOcr.filter(line => line && typeof line.text === 'string' && line.text.toLowerCase().includes(q));
  }, [rawOcr, q]);

  const ocrFormatted = useMemo(() => {
    return filteredOcr.map(line => {
      const box = Array.isArray(line?.box) ? line.box : [0, 0, 0, 0];
      const x1 = typeof box[0] === 'number' ? box[0] : 0;
      const y1 = typeof box[1] === 'number' ? box[1] : 0;
      const x2 = typeof box[2] === 'number' ? box[2] : 0;
      const y2 = typeof box[3] === 'number' ? box[3] : 0;

      return {
        text: line?.text || '',
        coordinates: {
          x_min: Number(x1.toFixed(2)),
          y_min: Number(y1.toFixed(2)),
          x_max: Number(x2.toFixed(2)),
          y_max: Number(y2.toFixed(2)),
        },
      };
    });
  }, [filteredOcr]);

  const modelFormattedMap = useMemo(() => {
    const map: Record<
      string,
      {
        model_id: string;
        display_name: string;
        total_detections: number;
        detections: Array<{
          class_name: string;
          confidence: number;
          box: { x_min: number; y_min: number; x_max: number; y_max: number };
        }>;
        panels: any[];
      }
    > = {};

    Object.entries(detectionResults?.detections || {}).forEach(([mId, mData]) => {
      const classNames = Array.isArray(mData?.class_names) ? mData.class_names : [];
      const rawDets = Array.isArray(mData?.detections) ? mData.detections : [];

      const allDets = rawDets.map(det => {
        const box = Array.isArray(det?.box) ? det.box : [0, 0, 0, 0];
        const score = typeof det?.score === 'number' ? det.score : 0;
        const cid = typeof det?.class_id === 'number' ? det.class_id : 0;
        const className = classNames[cid] || `Class ${cid}`;

        const x1 = typeof box[0] === 'number' ? box[0] : 0;
        const y1 = typeof box[1] === 'number' ? box[1] : 0;
        const x2 = typeof box[2] === 'number' ? box[2] : 0;
        const y2 = typeof box[3] === 'number' ? box[3] : 0;

        return {
          class_name: className,
          confidence: Number(score.toFixed(4)),
          box: {
            x_min: Number(x1.toFixed(2)),
            y_min: Number(y1.toFixed(2)),
            x_max: Number(x2.toFixed(2)),
            y_max: Number(y2.toFixed(2)),
          },
        };
      });

      const filteredDets = q
        ? allDets.filter(det => det.class_name.toLowerCase().includes(q))
        : allDets;
      const rawPanels = Array.isArray(mData?.panels) ? mData.panels : [];

      map[mId] = {
        model_id: mId,
        display_name: modelNameMap[mId] || mId,
        total_detections: filteredDets.length,
        detections: filteredDets,
        panels: rawPanels,
      };
    });

    return map;
  }, [detectionResults?.detections, modelNameMap, q]);

  // Total matches calculation
  const ocrMatchCount = ocrFormatted.length;
  const modelMatchCounts = Object.values(modelFormattedMap).reduce(
    (acc, m) => acc + m.total_detections,
    0
  );
  const totalMatchCount =
    validTab === 'ocr'
      ? ocrMatchCount
      : validTab === 'all'
      ? ocrMatchCount + modelMatchCounts
      : modelFormattedMap[validTab]?.total_detections || 0;

  // Standalone JSON Payload per tab
  let activeData: any = null;
  if (validTab === 'ocr') {
    activeData = {
      type: 'ocr_text',
      count: ocrFormatted.length,
      lines: ocrFormatted,
    };
  } else if (validTab === 'all') {
    activeData = {
      ocr: ocrFormatted,
      models: modelFormattedMap,
    };
  } else if (modelFormattedMap[validTab]) {
    activeData = modelFormattedMap[validTab];
  } else {
    activeData = detectionResults;
  }

  const jsonString = JSON.stringify(activeData || {}, null, 2);

  const handleCopy = async () => {
    try {
      await navigator.clipboard.writeText(jsonString);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch (err) {
      console.error('Failed to copy JSON: ', err);
    }
  };

  const handleDownload = () => {
    const blob = new Blob([jsonString], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `sld_${validTab}_output.json`;
    a.click();
    URL.revokeObjectURL(url);
  };

  const handleDownloadPanels = () => {
    const modelData = modelFormattedMap[validTab];
    const panels = modelData?.panels || [];
    const blob = new Blob([JSON.stringify(panels, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'panel.json';
    a.click();
    URL.revokeObjectURL(url);
  };

  if (!isOpen) return null;

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-slate-950/70 backdrop-blur-md"
      onClick={onClose}
    >
      <div
        className="glass rounded-2xl w-full max-w-5xl max-h-[88vh] mx-4 flex flex-col fade-in shadow-2xl overflow-hidden bg-slate-950 border border-slate-800"
        onClick={e => e.stopPropagation()}
      >
        {/* Header */}
        <div className="flex items-center justify-between px-6 py-4 border-b border-slate-800 bg-slate-900/90 shrink-0">
          <div className="flex items-center gap-2.5">
            <Braces size={20} className="text-indigo-400" />
            <div>
              <h3 className="text-sm font-bold text-white">
                Inference JSON Output
              </h3>
              <p className="text-xs text-slate-400">
                Separate JSON models for component detection & OCR text coordinates
              </p>
            </div>
          </div>

          <div className="flex items-center gap-2.5">
            {validTab !== 'ocr' && validTab !== 'all' && (
              <button
                onClick={handleDownloadPanels}
                className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-semibold bg-emerald-600 hover:bg-emerald-500 text-white border border-emerald-700 transition-all shadow-sm"
                title="Download panel.json for this model"
              >
                <Download size={13} />
                <span>Download panel.json</span>
              </button>
            )}
            <button
              onClick={handleDownload}
              className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-semibold bg-slate-800 hover:bg-slate-700 text-slate-200 border border-slate-700 transition-all shadow-sm"
              title="Download standalone JSON file for current tab"
            >
              <Download size={13} />
              <span>Download</span>
            </button>
            <button
              onClick={handleCopy}
              className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-semibold bg-indigo-600 hover:bg-indigo-500 text-white transition-all shadow-sm"
              title="Copy current tab JSON to clipboard"
            >
              {copied ? <Check size={13} /> : <Copy size={13} />}
              <span>{copied ? 'Copied!' : 'Copy JSON'}</span>
            </button>
            <button
              onClick={onClose}
              className="text-slate-400 hover:text-white transition-colors ml-2"
            >
              <X size={20} />
            </button>
          </div>
        </div>

        {/* Model & OCR Tabs Bar (Always at top) */}
        <div className="flex border-b border-slate-800 bg-slate-900/80 px-6 gap-2 overflow-x-auto shrink-0 z-10">
          {/* OCR Tab */}
          <button
            onClick={() => setUserTab('ocr')}
            className={`py-3 px-3 text-xs font-semibold border-b-2 transition-all flex items-center gap-1.5 whitespace-nowrap ${
              validTab === 'ocr'
                ? 'border-emerald-500 text-emerald-400 bg-emerald-500/10'
                : 'border-transparent text-slate-400 hover:text-slate-200'
            }`}
          >
            <FileText size={14} className={validTab === 'ocr' ? 'text-emerald-400' : ''} />
            <span>OCR Text Only</span>
            <span className="ml-1 text-[10px] px-1.5 py-0.5 rounded-full bg-emerald-950 text-emerald-300 border border-emerald-800 font-bold tabular-nums">
              {ocrFormatted.length}
            </span>
          </button>

          {/* Separate Tab for Each Model */}
          {modelIds.map(mId => {
            const modelData = modelFormattedMap[mId];
            const count = modelData?.total_detections ?? 0;
            const isActive = validTab === mId;
            const displayName = modelNameMap[mId] || mId;

            return (
              <button
                key={mId}
                onClick={() => setUserTab(mId)}
                className={`py-3 px-3 text-xs font-semibold border-b-2 transition-all flex items-center gap-1.5 whitespace-nowrap ${
                  isActive
                    ? 'border-indigo-500 text-indigo-400 bg-indigo-500/10'
                    : 'border-transparent text-slate-400 hover:text-slate-200'
                }`}
              >
                <Cpu size={14} className={isActive ? 'text-indigo-400' : ''} />
                <span>Components ({displayName})</span>
                <span className="ml-1 text-[10px] px-1.5 py-0.5 rounded-full bg-indigo-950 text-indigo-300 border border-indigo-800 font-bold tabular-nums">
                  {count}
                </span>
              </button>
            );
          })}

          {/* All Combined Tab */}
          <button
            onClick={() => setUserTab('all')}
            className={`py-3 px-3 text-xs font-semibold border-b-2 transition-all flex items-center gap-1.5 whitespace-nowrap ${
              validTab === 'all'
                ? 'border-violet-500 text-violet-400 bg-violet-500/10'
                : 'border-transparent text-slate-400 hover:text-slate-200'
            }`}
          >
            <Layers size={14} className={validTab === 'all' ? 'text-violet-400' : ''} />
            <span>All Combined</span>
          </button>
        </div>

        {/* Search Bar + Match Stats Bar */}
        <div className="flex items-center justify-between px-6 py-2.5 bg-slate-900/60 border-b border-slate-800 gap-4 shrink-0">
          {/* Search Input */}
          <div className="relative flex-1 max-w-md">
            <Search size={14} className="absolute left-3 top-1/2 -translate-y-1/2 text-slate-400" />
            <input
              type="text"
              value={searchQuery}
              onChange={e => setSearchQuery(e.target.value)}
              placeholder="Filter by class name or OCR text (e.g. MCCB, 400A)..."
              className="w-full pl-9 pr-8 py-1.5 text-xs bg-slate-800/80 rounded-lg border border-slate-700 text-slate-100 placeholder-slate-400 focus:outline-none focus:ring-2 focus:ring-indigo-500/40 focus:border-indigo-400 transition-all"
            />
            {searchQuery && (
              <button
                onClick={() => setSearchQuery('')}
                className="absolute right-2.5 top-1/2 -translate-y-1/2 text-slate-400 hover:text-white"
              >
                <X size={13} />
              </button>
            )}
          </div>

          {/* Match stats */}
          <div className="text-xs font-medium text-slate-400 bg-slate-900 px-3 py-1 rounded-lg border border-slate-800 shadow-sm">
            Active view:{' '}
            <span className="font-bold text-emerald-400 tabular-nums">
              {validTab === 'ocr' ? 'OCR Text' : validTab === 'all' ? 'All Combined' : modelNameMap[validTab] || validTab}
            </span>{' '}
            ({totalMatchCount} items)
          </div>
        </div>

        {/* Content Container (Scrollable JSON view) */}
        <div className="flex-1 p-6 overflow-y-auto font-mono text-xs bg-slate-950 leading-relaxed min-h-[350px]">
          <pre className="text-emerald-400 font-mono text-xs overflow-x-auto whitespace-pre-wrap break-all leading-relaxed select-text">
            <code>{jsonString}</code>
          </pre>
        </div>
      </div>
    </div>
  );
}
