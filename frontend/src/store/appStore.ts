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

export type InferResponse = Record<string, ModelDetections>;

// ─── Inference settings ────────────────────────────────────────────────────

export interface InferSettings {
  useTiling: boolean;
  gridSize: number;
  overlap: number;
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
    patch: { class_names?: string[]; confidence_default?: number }
  ) => Promise<void>;

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

  // Inference settings
  inferSettings: InferSettings;
  setInferSettings: (patch: Partial<InferSettings>) => void;

  // Image + detections
  currentImageUrl: string | null;              // object URL for canvas
  currentImageFile: File | null;
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
  openUploadModal: () => void;
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

  // ── Inference settings ───────────────────────────────────────────────────
  inferSettings: { useTiling: true, gridSize: 4, overlap: 0.2 },
  setInferSettings: patch =>
    set(s => ({ inferSettings: { ...s.inferSettings, ...patch } })),

  // ── Image ────────────────────────────────────────────────────────────────
  currentImageUrl: null,
  currentImageFile: null,
  setImage: (file: File) => {
    const prev = get().currentImageUrl;
    if (prev) URL.revokeObjectURL(prev);
    set({
      currentImageUrl: URL.createObjectURL(file),
      currentImageFile: file,
      detectionResults: {},
      inferError: null,
    });
  },

  // ── Inference ────────────────────────────────────────────────────────────
  detectionResults: {},
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
        gridSize: inferSettings.gridSize,
        overlap: inferSettings.overlap,
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
  openUploadModal: () => set({ uploadModalOpen: true }),
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
