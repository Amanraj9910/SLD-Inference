"""
/api/models — list, load, and configure checkpoints.
"""
from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException

from app.models import registry
from app.schemas import ModelInfo, UpdateConfigRequest

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/models", tags=["models"])


@router.get("", response_model=list[ModelInfo])
def list_models() -> list[ModelInfo]:
    """Return all discovered checkpoints and their current config."""
    return registry.scan()


@router.post("/{model_id}/load", response_model=ModelInfo)
def load_model(model_id: str) -> ModelInfo:
    """
    Eagerly load a checkpoint into GPU memory.
    Normally models are loaded on first inference, but this endpoint
    lets the UI pre-warm a model before the user uploads an image.
    """
    try:
        registry.get_or_load(model_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except FileNotFoundError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        logger.exception("Failed to load model '%s'", model_id)
        raise HTTPException(status_code=500, detail=str(exc))

    info = registry.get_model_info(model_id)
    if info is None:
        raise HTTPException(status_code=404, detail=f"Model '{model_id}' not found after load.")
    return info


@router.put("/{model_id}/config", response_model=ModelInfo)
def update_config(model_id: str, body: UpdateConfigRequest) -> ModelInfo:
    """
    Live-edit class names and/or confidence threshold for a model.
    Rewrites manifest.json on disk — no server restart needed.
    num_classes is read-only (baked into checkpoint output-layer shape).
    """
    try:
        return registry.update_manifest(
            model_id,
            class_names=body.class_names,
            confidence_default=body.confidence_default,
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except Exception as exc:
        logger.exception("Failed to update config for '%s'", model_id)
        raise HTTPException(status_code=500, detail=str(exc))
