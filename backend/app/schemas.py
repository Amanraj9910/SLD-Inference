"""
Pydantic request / response schemas for the API.
"""
from __future__ import annotations

from typing import Any
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Model-registry schemas
# ---------------------------------------------------------------------------

class ModelInfo(BaseModel):
    """Metadata for a single checkpoint discovered in weights/."""
    model_id: str
    arch: str                           # "dfine" | "rfdetr"
    display_name: str
    weights_file: str
    num_classes: int
    resolution: int
    class_names: list[str]
    confidence_default: float
    grid_size: int = 4
    overlap: float = 0.20
    tiling_mode: str = "fixed"          # "fixed" | "adaptive"
    target_symbol_px: float = 48.0
    estimated_symbol_px: float = 48.0
    enable_auto_crop: bool = False
    enable_scale_norm: bool = False
    iou_threshold: float = 0.50
    loaded: bool = False                # True once the GPU model is in memory
    weights_exist: bool = True          # True if the .pth file actually exists on disk


class UpdateConfigRequest(BaseModel):
    """Body for PUT /api/models/{model_id}/config."""
    class_names: list[str] | None = None
    confidence_default: float | None = None
    tiling_mode: str | None = None
    target_symbol_px: float | None = None
    estimated_symbol_px: float | None = None
    enable_auto_crop: bool | None = None
    enable_scale_norm: bool | None = None


class UploadModelManifest(BaseModel):
    """Metadata attached to model upload requests."""
    arch: str                          # "dfine" | "rfdetr"
    display_name: str
    num_classes: int
    class_names: list[str]
    resolution: int = 640
    confidence_default: float = 0.20
    grid_size: int = 4
    overlap: float = 0.20
    tiling_mode: str = "fixed"         # "fixed" | "adaptive"
    target_symbol_px: float = 48.0
    estimated_symbol_px: float = 48.0
    enable_auto_crop: bool = False
    enable_scale_norm: bool = False
    iou_threshold: float = 0.50
    model_id: str | None = None


# ---------------------------------------------------------------------------
# Inference schemas
# ---------------------------------------------------------------------------

class InferRequest(BaseModel):
    """JSON body attached to the multipart /api/infer request."""
    model_ids: list[str] = Field(..., min_length=1)
    use_tiling: bool = True
    tiling_mode: str | None = None       # None uses each model manifest
    grid_size: int | None = Field(None, ge=1, le=10)
    overlap: float | None = Field(None, ge=0.0, lt=1.0)
    target_symbol_px: float | None = Field(None, gt=0.0)
    estimated_symbol_px: float | None = Field(None, gt=0.0)
    enable_auto_crop: bool | None = None
    enable_scale_norm: bool | None = None



class Detection(BaseModel):
    """Single bounding-box detection."""
    box: list[float]        # [x1, y1, x2, y2] in image pixels
    class_id: int
    score: float


class ModelDetections(BaseModel):
    """All detections from one model for the submitted image."""
    class_names: list[str]
    detections: list[Detection]


class OCRLine(BaseModel):
    """Single bounding-box text OCR detection."""
    text: str
    box: list[float]        # [x1, y1, x2, y2] in image pixels


class InferResponse(BaseModel):
    """Consolidated inference response containing model detections and OCR text."""
    detections: dict[str, ModelDetections]
    ocr: list[OCRLine] | None = None
