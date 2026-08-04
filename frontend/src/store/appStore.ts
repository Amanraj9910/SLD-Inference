import { create } from 'zustand';
import { api } from '../api/client';

// ─── Domain types (mirroring backend schemas) ──────────────────────────────

export interface ModelInfo {
  model_id: string;
  arch: 'dfine' | 'rfdetr';
  display_name: string;
  weights_file: string;
  num_classes: number;
  resolution: number;
  class_names: string[];
  confidence_default: number;
  grid_size: number;
  overlap: number;
  tiling_mode?: 'fixed' | 'adaptive';
  target_symbol_px?: number;
  estimated_symbol_px?: number;
  enable_auto_crop?: boolean;
  enable_scale_norm?: boolean;
  iou_threshold: number;
  loaded: boolean;
  weights_exist: boolean;
}

export interface Detection {
  box: [number, number, number, number]; // [x1, y1, x2, y2]
  class_id: number;
  score: number;
}

export interface ModelDetections {
  class_names: string[];
  detections: Detection[];
}

export interface OCRLine {
  text: string;
  box: [number, number, number, number]; // [x1, y1, x2, y2]
}

export interface InferResponse {
  detections: Record<string, ModelDetections>;
  ocr: OCRLine[] | null;
}

// ─── Inference settings ────────────────────────────────────────────────────

export interface InferSettings {
  useTiling: boolean;
  tilingMode: 'fixed' | 'adaptive';
  gridSize: number;
  overlap: number;
  targetSymbolPx: number;
  estimatedSymbolPx: number;
  enableAutoCrop: boolean;
  enableScaleNorm: boolean;
}

// ─── Store ─────────────────────────────────────────────────────────────────

interface AppState {
  // Model registry
  models: ModelInfo[];
  modelsLoading: boolean;
  fetchModels: () => Promise<void>;
  loadModel: (modelId: string) => Promise<void>;
  updateModelConfig: (
    modelId: string,
    patch: {
      class_names?: string[];
      confidence_default?: number;
      tiling_mode?: 'fixed' | 'adaptive';
      target_symbol_px?: number;
      estimated_symbol_px?: number;
      enable_auto_crop?: boolean;
      enable_scale_norm?: boolean;
    }
  ) => Promise<void>;
  deleteModel: (modelId: string) => Promise<void>;

  // Selection
  selectedModelIds: Set<string>;
  toggleModelSelected: (modelId: string) => void;

  // Per-model UI state
  thresholds: Record<string, number>;          // client-side threshold per model
  setThreshold: (modelId: string, v: number) => void;
  visibleModels: Record<string, boolean>;       // layer visibility
  toggleModelVisible: (modelId: string) => void;

  // Global UI toggles
  showLabels: boolean;
  toggleShowLabels: () => void;
  showOcr: boolean;
  toggleShowOcr: () => void;

  // Inference settings
  inferSettings: InferSettings;
  setInferSettings: (patch: Partial<InferSettings>) => void;

  // Image + detections
  currentImageUrl: string | null;              // object URL for canvas
  currentImageFile: File | null;
  imageDimensions: { width: number; height: number } | null;
  setImage: (file: File) => void;

  detectionResults: InferResponse;
  isInferring: boolean;
  inferError: string | null;
  runInfer: () => Promise<void>;

  // Config modal
  configModalModelId: string | null;
  openConfigModal: (modelId: string) => void;
  closeConfigModal: () => void;

  // Upload model modal
  uploadModalOpen: boolean;
  uploadModalMode: 'adaptive' | 'fixed';
  openUploadModal: (mode?: 'adaptive' | 'fixed') => void;
  closeUploadModal: () => void;
  uploadModel: (params: Parameters<typeof api.uploadModel>[0]) => Promise<void>;
}

export const useAppStore = create<AppState>((set, get) => ({
  // ── Model registry ──────────────────────────────────────────────────────
  models: [],
  modelsLoading: false,

  fetchModels: async () => {
    set({ modelsLoading: true });
    try {
      const models = await api.getModels();
      set(state => {
        // Initialise defaults for new models
        const thresholds = { ...state.thresholds };
        const visibleModels = { ...state.visibleModels };
        for (const m of models) {
          if (!(m.model_id in thresholds)) {
            thresholds[m.model_id] = m.confidence_default;
          }
          if (!(m.model_id in visibleModels)) {
            visibleModels[m.model_id] = true;
          }
        }
        return { models, thresholds, visibleModels };
      });
    } finally {
      set({ modelsLoading: false });
    }
  },

  loadModel: async (modelId: string) => {
    const updated = await api.loadModel(modelId);
    set(state => ({
      models: state.models.map(m => (m.model_id === modelId ? updated : m)),
    }));
  },

  updateModelConfig: async (modelId, patch) => {
    const updated = await api.updateConfig(modelId, patch);
    set(state => ({
      models: state.models.map(m => (m.model_id === modelId ? updated : m)),
      thresholds:
        patch.confidence_default !== undefined
          ? { ...state.thresholds, [modelId]: patch.confidence_default }
          : state.thresholds,
    }));
  },

  deleteModel: async (modelId: string) => {
    await api.deleteModel(modelId);
    await get().fetchModels();
    set(state => {
      const nextSelected = new Set(state.selectedModelIds);
      nextSelected.delete(modelId);
      return { selectedModelIds: nextSelected };
    });
  },

  // ── Selection ────────────────────────────────────────────────────────────
  selectedModelIds: new Set(),

  toggleModelSelected: (modelId: string) =>
    set(state => {
      const next = new Set(state.selectedModelIds);
      if (next.has(modelId)) next.delete(modelId);
      else next.add(modelId);
      return { selectedModelIds: next };
    }),

  // ── Per-model UI ─────────────────────────────────────────────────────────
  thresholds: {},
  setThreshold: (modelId, v) =>
    set(state => ({ thresholds: { ...state.thresholds, [modelId]: v } })),

  visibleModels: {},
  toggleModelVisible: (modelId: string) =>
    set(state => ({
      visibleModels: {
        ...state.visibleModels,
        [modelId]: !state.visibleModels[modelId],
      },
    })),

  // ── Global UI ────────────────────────────────────────────────────────────
  showLabels: false,
  toggleShowLabels: () => set(s => ({ showLabels: !s.showLabels })),
  showOcr: true,
  toggleShowOcr: () => set(s => ({ showOcr: !s.showOcr })),

  // ── Inference settings ───────────────────────────────────────────────────
  inferSettings: {
    useTiling: true,
    tilingMode: 'adaptive',
    gridSize: 4,
    overlap: 0.2,
    targetSymbolPx: 48,
    estimatedSymbolPx: 48,
    enableAutoCrop: false,
    enableScaleNorm: false,
  },
  setInferSettings: patch =>
    set(s => ({ inferSettings: { ...s.inferSettings, ...patch } })),

  // ── Image ────────────────────────────────────────────────────────────────
  currentImageUrl: null,
  currentImageFile: null,
  imageDimensions: null,
  setImage: (file: File) => {
    const prev = get().currentImageUrl;
    if (prev) URL.revokeObjectURL(prev);
    const url = URL.createObjectURL(file);

    const img = new Image();
    img.onload = () => {
      set({ imageDimensions: { width: img.naturalWidth, height: img.naturalHeight } });
    };
    img.src = url;

    set({
      currentImageUrl: url,
      currentImageFile: file,
      imageDimensions: null,
      detectionResults: { detections: {}, ocr: null },
      inferError: null,
    });
  },

  // ── Inference ────────────────────────────────────────────────────────────
  detectionResults: { detections: {}, ocr: null },
  isInferring: false,
  inferError: null,

  runInfer: async () => {
    const { currentImageFile, selectedModelIds, inferSettings } = get();
    if (!currentImageFile || selectedModelIds.size === 0) return;

    set({ isInferring: true, inferError: null });
    try {
      const results = await api.infer({
        image: currentImageFile,
        modelIds: [...selectedModelIds],
        useTiling: inferSettings.useTiling,
        tilingMode: inferSettings.tilingMode,
        gridSize: inferSettings.gridSize,
        overlap: inferSettings.overlap,
        targetSymbolPx: inferSettings.targetSymbolPx,
        estimatedSymbolPx: inferSettings.estimatedSymbolPx,
        enableAutoCrop: inferSettings.enableAutoCrop,
        enableScaleNorm: inferSettings.enableScaleNorm,
      });
      set({ detectionResults: results });
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : String(err);
      set({ inferError: msg });
    } finally {
      set({ isInferring: false });
    }
  },

  // ── Config modal ─────────────────────────────────────────────────────────
  configModalModelId: null,
  openConfigModal: (modelId: string) => set({ configModalModelId: modelId }),
  closeConfigModal: () => set({ configModalModelId: null }),

  // ── Upload model modal ───────────────────────────────────────────────────
  uploadModalOpen: false,
  uploadModalMode: 'adaptive',
  openUploadModal: (mode = 'adaptive') => set({ uploadModalOpen: true, uploadModalMode: mode }),
  closeUploadModal: () => set({ uploadModalOpen: false }),

  uploadModel: async params => {
    const newModel = await api.uploadModel(params);
    await get().fetchModels();
    set(state => {
      const nextSelected = new Set(state.selectedModelIds);
      nextSelected.add(newModel.model_id);
      return { selectedModelIds: nextSelected, uploadModalOpen: false };
    });
  },
}));
