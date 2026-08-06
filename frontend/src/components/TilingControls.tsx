import { useAppStore } from '../store/appStore';

export function TilingControls() {
  const { inferSettings, setInferSettings } = useAppStore();
  const adaptive = inferSettings.tilingMode === 'adaptive';

  return (
    <div className="mt-auto border-t border-slate-200/80 pt-3 space-y-2.5">
      <button
        type="button"
        aria-pressed={adaptive}
        onClick={() => setInferSettings({ tilingMode: adaptive ? 'fixed' : 'adaptive' })}
        className={`w-full flex items-center justify-between rounded-xl px-3 py-2.5 border transition-colors ${
          adaptive ? 'bg-indigo-50 border-indigo-200 text-indigo-800' : 'bg-slate-50 border-slate-200 text-slate-700'
        }`}
      >
        <span className="text-xs font-semibold">Adaptive Tiling</span>
        <span className="flex items-center gap-2 text-[10px] font-bold uppercase tracking-wide">
          {adaptive ? 'On' : 'Off · Fixed'}
          <span className={`w-8 h-4 rounded-full p-0.5 ${adaptive ? 'bg-indigo-600' : 'bg-slate-300'}`}>
            <span className={`block w-3 h-3 rounded-full bg-white transition-transform ${adaptive ? 'translate-x-4' : ''}`} />
          </span>
        </span>
      </button>

      {adaptive ? (
        <p className="text-[10px] leading-relaxed text-slate-500">
          Tile count is calculated from the uploaded image and symbol scale.
        </p>
      ) : (
        <div className="space-y-1">
          <div className="flex items-center justify-between">
            <span className="text-[11px] font-medium text-slate-500">Fixed tile grid</span>
            <span className="text-[11px] text-slate-700 tabular-nums font-semibold">
              {inferSettings.gridSize} × {inferSettings.gridSize}
            </span>
          </div>
          <input
            type="range"
            min={1}
            max={10}
            step={1}
            value={inferSettings.gridSize}
            onChange={event => setInferSettings({ gridSize: parseInt(event.target.value, 10) })}
            className="w-full"
          />
        </div>
      )}
    </div>
  );
}
