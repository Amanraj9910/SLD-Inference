import { ZoomIn, ZoomOut, RotateCcw, Move } from 'lucide-react';
import { useAppStore } from '../store/appStore';

export function ZoomControls() {
  const { zoomScale, setZoomScale, resetZoom } = useAppStore();

  const handleZoomBtn = (direction: 'in' | 'out') => {
    const factor = direction === 'in' ? 1.25 : 0.8;
    const newScale = Math.max(0.5, Math.min(15, zoomScale * factor));
    setZoomScale(newScale);
  };

  return (
    <div className="absolute bottom-4 right-4 glass rounded-xl px-2 py-1.5 flex items-center gap-1.5 border border-slate-200 bg-white/95 text-slate-800 shadow-xl z-20 transition-all">
      <button
        onClick={() => handleZoomBtn('out')}
        className="p-1.5 rounded-lg text-slate-500 hover:text-slate-900 hover:bg-slate-100 transition-all"
        title="Zoom out"
      >
        <ZoomOut size={16} />
      </button>
      <span className="text-xs text-slate-700 font-mono font-semibold px-1.5 tabular-nums min-w-12 text-center">
        {Math.round(zoomScale * 100)}%
      </span>
      <button
        onClick={() => handleZoomBtn('in')}
        className="p-1.5 rounded-lg text-slate-500 hover:text-slate-900 hover:bg-slate-100 transition-all"
        title="Zoom in"
      >
        <ZoomIn size={16} />
      </button>
      <div className="h-4 w-px bg-slate-200 mx-0.5" />
      <button
        onClick={resetZoom}
        className="p-1.5 rounded-lg text-slate-500 hover:text-indigo-600 hover:bg-slate-100 transition-all"
        title="Reset zoom & position"
      >
        <RotateCcw size={15} />
      </button>
      <div className="text-[10px] text-slate-400 flex items-center gap-1 pl-1">
        <Move size={12} /> Drag to Pan
      </div>
    </div>
  );
}
