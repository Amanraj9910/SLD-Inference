import axios from 'axios';
import type { ModelInfo, InferResponse } from '../store/appStore';

// In development Vite proxies /api → localhost:8000 via vite.config.ts
// In production Nginx serves the built files and proxies /api → uvicorn
const BASE = import.meta.env.VITE_API_BASE_URL ?? '/api';

const http = axios.create({
  baseURL: BASE,
  timeout: 300_000, // 5 min — tiled inference can be slow
});

export const api = {
  /** Fetch all discovered model manifests */
  getModels(): Promise<ModelInfo[]> {
    return http.get<ModelInfo[]>('/models').then(r => r.data);
  },

  /** Eagerly load a model onto GPU */
  loadModel(modelId: string): Promise<ModelInfo> {
    return http.post<ModelInfo>(`/models/${modelId}/load`).then(r => r.data);
  },

  /** Update class names and/or confidence threshold */
  updateConfig(
    modelId: string,
    payload: { class_names?: string[]; confidence_default?: number }
  ): Promise<ModelInfo> {
    return http
      .put<ModelInfo>(`/models/${modelId}/config`, payload)
      .then(r => r.data);
  },

  /** Run inference on an uploaded image */
  infer(params: {
    image: File;
    modelIds: string[];
    useTiling: boolean;
    gridSize: number;
    overlap: number;
  }): Promise<InferResponse> {
    const form = new FormData();
    form.append('image', params.image);
    form.append(
      'body',
      JSON.stringify({
        model_ids: params.modelIds,
        use_tiling: params.useTiling,
        grid_size: params.gridSize,
        overlap: params.overlap,
      })
    );
    return http
      .post<InferResponse>('/infer', form, {
        headers: { 'Content-Type': 'multipart/form-data' },
      })
      .then(r => r.data);
  },

  /** Fetch live server logs for debugging */
  getLogs(lines = 100): Promise<{ logs: string }> {
    return http.get<{ logs: string }>(`/logs?lines=${lines}`).then(r => r.data);
  },

  /** Upload a new model checkpoint (.pth) and metadata manifest */
  uploadModel(params: {
    file: File;
    arch: 'dfine' | 'rfdetr';
    displayName: string;
    numClasses: number;
    classNames: string[];
    resolution: number;
    confidenceDefault: number;
    gridSize: number;
    overlap: number;
    modelId?: string;
  }): Promise<ModelInfo> {
    const form = new FormData();
    form.append('file', params.file);
    form.append(
      'manifest',
      JSON.stringify({
        arch: params.arch,
        display_name: params.displayName,
        num_classes: params.numClasses,
        class_names: params.classNames,
        resolution: params.resolution,
        confidence_default: params.confidenceDefault,
        grid_size: params.gridSize,
        overlap: params.overlap,
        model_id: params.modelId,
      })
    );
    return http
      .post<ModelInfo>('/models/upload', form, {
        headers: { 'Content-Type': 'multipart/form-data' },
        timeout: 600_000,
      })
      .then(r => r.data);
  },

  /** Delete a model checkpoint and directory */
  deleteModel(modelId: string): Promise<{ status: string }> {
    return http.delete<{ status: string }>(`/models/${modelId}`).then(r => r.data);
  },
};
