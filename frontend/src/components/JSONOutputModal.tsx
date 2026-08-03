import { useState } from 'react';
import { useAppStore } from '../store/appStore';
import { X, Copy, Check, Braces, FileText } from 'lucide-react';

interface JSONOutputModalProps {
  isOpen: boolean;
  onClose: () => void;
}

type TabType = 'ocr' | 'detections' | 'full';

export function JSONOutputModal({ isOpen, onClose }: JSONOutputModalProps) {
  const { detectionResults } = useAppStore();
  const [activeTab, setActiveTab] = useState<TabType>('ocr');
  const [copied, setCopied] = useState(false);

  if (!isOpen) return null;

  // Prepare filtered data structures
  const ocrData = detectionResults.ocr || [];
  
  // Format coordinate & text JSON for OCR
  const ocrFormatted = ocrData.map(line => ({
    text: line.text,
    coordinates: {
      x_min: line.box[0],
      y_min: line.box[1],
      x_max: line.box[2],
      y_max: line.box[3]
    }
  }));

  const detectionsFormatted = Object.entries(detectionResults.detections || {}).reduce(
    (acc, [modelId, modelData]) => {
      acc[modelId] = {
        class_names: modelData.class_names,
        detections: modelData.detections.map(det => ({
          class_name: modelData.class_names[det.class_id] || `Class ${det.class_id}`,
          score: Number(det.score.toFixed(4)),
          box: {
            x_min: Number(det.box[0].toFixed(2)),
            y_min: Number(det.box[1].toFixed(2)),
            x_max: Number(det.box[2].toFixed(2)),
            y_max: Number(det.box[3].toFixed(2))
          }
        }))
      };
      return acc;
    },
    {} as Record<string, any>
  );

  // Get current JSON string based on active tab
  let jsonString = '';
  if (activeTab === 'ocr') {
    jsonString = JSON.stringify(ocrFormatted, null, 2);
  } else if (activeTab === 'detections') {
    jsonString = JSON.stringify(detectionsFormatted, null, 2);
  } else {
    jsonString = JSON.stringify(detectionResults, null, 2);
  }

  const handleCopy = async () => {
    try {
      await navigator.clipboard.writeText(jsonString);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch (err) {
      console.error('Failed to copy text: ', err);
    }
  };

  const hasData = ocrFormatted.length > 0 || Object.keys(detectionsFormatted).length > 0;

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
            <Braces size={18} className="text-indigo-600" />
            <div>
              <h3 className="text-sm font-bold text-slate-900">
                Inference JSON Output
              </h3>
              <p className="text-xs text-slate-500">
                Structured data including coordinates and OCR text detections
              </p>
            </div>
          </div>

          <div className="flex items-center gap-3">
            {hasData && (
              <button
                onClick={handleCopy}
                className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-semibold bg-white hover:bg-slate-100 text-slate-700 border border-slate-200 transition-all shadow-sm"
              >
                {copied ? <Check size={13} className="text-emerald-600" /> : <Copy size={13} />}
                <span>{copied ? 'Copied!' : 'Copy JSON'}</span>
              </button>
            )}
            <button
              onClick={onClose}
              className="text-slate-400 hover:text-slate-600 transition-colors"
            >
              <X size={20} />
            </button>
          </div>
        </div>

        {/* Tabs Bar */}
        {hasData && (
          <div className="flex border-b border-slate-100 bg-slate-50/30 px-6 gap-6">
            <button
              onClick={() => setActiveTab('ocr')}
              className={`py-3 text-xs font-semibold border-b-2 transition-all flex items-center gap-1.5 ${
                activeTab === 'ocr'
                  ? 'border-indigo-600 text-indigo-600'
                  : 'border-transparent text-slate-500 hover:text-slate-800'
              }`}
            >
              <FileText size={14} />
              <span>OCR Coordinates & Text ({ocrFormatted.length})</span>
            </button>
            <button
              onClick={() => setActiveTab('detections')}
              className={`py-3 text-xs font-semibold border-b-2 transition-all flex items-center gap-1.5 ${
                activeTab === 'detections'
                  ? 'border-indigo-600 text-indigo-600'
                  : 'border-transparent text-slate-500 hover:text-slate-800'
              }`}
            >
              <Braces size={14} />
              <span>Model Detections</span>
            </button>
            <button
              onClick={() => setActiveTab('full')}
              className={`py-3 text-xs font-semibold border-b-2 transition-all flex items-center gap-1.5 ${
                activeTab === 'full'
                  ? 'border-indigo-600 text-indigo-600'
                  : 'border-transparent text-slate-500 hover:text-slate-800'
              }`}
            >
              <Braces size={14} />
              <span>Full Raw Response</span>
            </button>
          </div>
        )}

        {/* Content */}
        <div className="flex-1 p-6 overflow-y-auto font-mono text-xs text-slate-200 bg-slate-900 leading-relaxed whitespace-pre selection:bg-indigo-500/30 selection:text-white min-h-[300px]">
          {hasData ? (
            jsonString
          ) : (
            <div className="h-full flex flex-col items-center justify-center text-slate-400 font-sans gap-2 py-12">
              <Braces size={32} className="text-slate-600 animate-pulse" />
              <p className="font-semibold text-sm">No results available yet</p>
              <p className="text-xs text-slate-500">Run model inference first to view the coordinate & text-name JSON output.</p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
