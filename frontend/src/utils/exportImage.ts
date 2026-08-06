import { classColor } from './palette';
import type {
  Detection,
  InferSettings,
  ModelInfo,
  InferResponse,
} from '../store/appStore';

interface ExportImageOptions {
  imageUrl: string;
  originalFileName: string;
  results: InferResponse;
  models: ModelInfo[];
  thresholds: Record<string, number>;
  visibleModels: Record<string, boolean>;
  inferSettings: InferSettings;
  includeOcr: boolean;
}

function safeFilePart(value: string, maxLength = 48): string {
  return value
    .replace(/\.[^.]+$/, '')
    .replace(/[^a-zA-Z0-9]+/g, '-')
    .replace(/^-+|-+$/g, '')
    .slice(0, maxLength) || 'image';
}

function loadImage(imageUrl: string): Promise<HTMLImageElement> {
  return new Promise((resolve, reject) => {
    const image = new Image();
    image.onload = () => resolve(image);
    image.onerror = () => reject(new Error('Unable to load the source image for export.'));
    image.src = imageUrl;
  });
}

function drawDetection(
  context: CanvasRenderingContext2D,
  detection: Detection,
  className: string,
  scale: number,
): void {
  const [x1, y1, x2, y2] = detection.box;
  const color = classColor(detection.class_id);
  const label = `${className} ${(detection.score * 100).toFixed(0)}%`;
  const fontSize = Math.max(12, Math.round(14 * scale));
  context.strokeStyle = color;
  context.lineWidth = Math.max(1, 2 * scale);
  context.strokeRect(x1, y1, x2 - x1, y2 - y1);
  context.font = `600 ${fontSize}px Arial`;
  const textWidth = context.measureText(label).width;
  const textHeight = fontSize + 6 * scale;
  context.fillStyle = color;
  context.fillRect(x1, Math.max(0, y1 - textHeight), textWidth + 8 * scale, textHeight);
  context.fillStyle = '#ffffff';
  context.fillText(label, x1 + 4 * scale, Math.max(fontSize, y1 - 4 * scale));
}

export async function downloadAnnotatedImage(options: ExportImageOptions): Promise<void> {
  const image = await loadImage(options.imageUrl);
  const canvas = document.createElement('canvas');
  canvas.width = image.naturalWidth;
  canvas.height = image.naturalHeight;
  const context = canvas.getContext('2d');
  if (!context) throw new Error('Canvas export is not supported by this browser.');

  context.drawImage(image, 0, 0);
  const modelParts: string[] = [];
  for (const [modelId, modelData] of Object.entries(options.results.detections || {})) {
    if (!options.visibleModels[modelId]) continue;
    const modelInfo = options.models.find(model => model.model_id === modelId);
    modelParts.push(safeFilePart(modelInfo?.display_name || modelId));
    const threshold = options.thresholds[modelId] ?? 0;
    for (const detection of modelData.detections) {
      if (detection.score < threshold) continue;
      drawDetection(
        context,
        detection,
        modelData.class_names[detection.class_id] || `class_${detection.class_id}`,
        Math.max(1, image.naturalWidth / 2000),
      );
    }
  }

  if (options.includeOcr) {
    context.strokeStyle = '#10b981';
    context.lineWidth = Math.max(1, image.naturalWidth / 2000);
    context.font = `${Math.max(12, Math.round(image.naturalWidth / 140))}px Arial`;
    for (const line of options.results.ocr || []) {
      const [x1, y1, x2, y2] = line.box;
      context.strokeRect(x1, y1, x2 - x1, y2 - y1);
      context.fillStyle = '#047857';
      context.fillText(line.text, x1, Math.max(14, y1 - 4));
    }
  }

  const mode = options.inferSettings.tilingMode === 'adaptive'
    ? 'adaptive'
    : `${options.inferSettings.gridSize}x${options.inferSettings.gridSize}`;
  const modelPart = modelParts.length > 0 ? modelParts.join('+').slice(0, 120) : 'model';
  const fileName = `${safeFilePart(options.originalFileName, 80)}_${modelPart}_${mode}.png`;
  const blob = await new Promise<Blob | null>(resolve => canvas.toBlob(resolve, 'image/png'));
  if (!blob) throw new Error('Unable to encode the annotated image.');

  const link = document.createElement('a');
  link.href = URL.createObjectURL(blob);
  link.download = fileName;
  link.click();
  window.setTimeout(() => URL.revokeObjectURL(link.href), 1000);
}
