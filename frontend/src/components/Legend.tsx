import { useMemo } from 'react';
import { useAppStore } from '../store/appStore';
import { classColor } from '../utils/palette';

export function Legend() {
  const { detectionResults, visibleModels, thresholds } = useAppStore();

  // Collect unique (classId, className) pairs across all visible results
  const items = useMemo(() => {
    const seen = new Map<number, string>();
    for (const [modelId, modelDets] of Object.entries(detectionResults)) {
      if (!visibleModels[modelId]) continue;
      const threshold = thresholds[modelId] ?? 0;
      for (const det of modelDets.detections) {
        if (det.score >= threshold && !seen.has(det.class_id)) {
          const name =
            modelDets.class_names[det.class_id] ?? `class_${det.class_id}`;
          seen.set(det.class_id, name);
        }
      }
    }
    return [...seen.entries()]
      .sort((a, b) => a[0] - b[0])
      .map(([classId, name]) => ({ classId, name }));
  }, [detectionResults, visibleModels, thresholds]);

  if (items.length === 0) return null;

  return (
    <div className="glass rounded-2xl px-5 py-3 flex flex-wrap gap-x-5 gap-y-2 items-center bg-white border border-slate-200/80 shadow-sm">
      <span className="text-[10px] uppercase tracking-wider text-slate-400 font-bold mr-1">
        Legend
      </span>
      {items.map(({ classId, name }) => (
        <div key={classId} className="flex items-center gap-2">
          <span
            className="inline-block w-3 h-3 rounded-full shadow-sm"
            style={{ backgroundColor: classColor(classId) }}
          />
          <span className="text-xs font-semibold text-slate-700">{name}</span>
        </div>
      ))}
    </div>
  );
}
