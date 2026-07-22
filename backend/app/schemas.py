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
    iou_threshold: float = 0.50
    loaded: bool = False                # True once the GPU model is in memory
    weights_exist: bool = True          # True if the .pth file actually exists on disk


class UpdateConfigRequest(BaseModel):
    """Body for PUT /api/models/{model_id}/config."""
    class_names: list[str] | None = None
    confidence_default: float | None = None


# ---------------------------------------------------------------------------
# Inference schemas
# ---------------------------------------------------------------------------

class InferRequest(BaseModel):
    """JSON body attached to the multipart /api/infer request."""
    model_ids: list[str] = Field(..., min_length=1)
    use_tiling: bool = True
    grid_size: int = Field(4, ge=1, le=10)
    overlap: float = Field(0.20, ge=0.0, lt=1.0)


class Detection(BaseModel):
    """Single bounding-box detection."""
    box: list[float]        # [x1, y1, x2, y2] in image pixels
    class_id: int
    score: float


class ModelDetections(BaseModel):
    """All detections from one model for the submitted image."""
    class_names: list[str]
    detections: list[Detection]


# Response is a dict keyed by model_id
InferResponse = dict[str, ModelDetections]
