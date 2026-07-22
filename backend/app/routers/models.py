"""
/api/models — list, load, and configure checkpoints.
"""
from __future__ import annotations

import json
import logging
import re
import shutil
from pathlib import Path

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from app.config import settings
from app.models import registry
from app.schemas import ModelInfo, UpdateConfigRequest, UploadModelManifest

logger = logging.getLogger(__name__)
router = APIRouter(tags=["models"])


@router.get("/models", response_model=list[ModelInfo])
@router.get("/api/models", response_model=list[ModelInfo])
def list_models() -> list[ModelInfo]:
    """Return all discovered checkpoints and their current config."""
    return registry.scan()


@router.post("/models/{model_id}/load", response_model=ModelInfo)
@router.post("/api/models/{model_id}/load", response_model=ModelInfo)
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
    except (FileNotFoundError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        logger.exception("Failed to load model '%s'", model_id)
        raise HTTPException(status_code=500, detail=str(exc))

    info = registry.get_model_info(model_id)
    if info is None:
        raise HTTPException(status_code=404, detail=f"Model '{model_id}' not found after load.")
    return info


@router.put("/models/{model_id}/config", response_model=ModelInfo)
@router.put("/api/models/{model_id}/config", response_model=ModelInfo)
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


@router.post("/models/upload", response_model=ModelInfo)
@router.post("/api/models/upload", response_model=ModelInfo)
async def upload_model(
    file: UploadFile = File(..., description=".pth checkpoint file"),
    manifest: str = Form(..., description="JSON string matching UploadModelManifest"),
) -> ModelInfo:
    """
    Upload a new model checkpoint (.pth) and its metadata manifest.
    Creates a new subfolder in weights/ and rescans the registry.
    """
    try:
        data = UploadModelManifest(**json.loads(manifest))
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Invalid manifest JSON: {exc}")

    if not file.filename.endswith((".pth", ".pt")):
        raise HTTPException(status_code=400, detail="File must be a PyTorch checkpoint (.pth or .pt)")

    # Derive clean model_id
    raw_id = data.model_id or data.display_name
    clean_id = re.sub(r"[^a-zA-Z0-9_-]", "_", raw_id.lower()).strip("_")
    if not clean_id:
        clean_id = f"{data.arch}_custom_{int(file.size or 0)}"

    weights_base = settings.resolved_weights_dir
    target_dir = weights_base / clean_id
    target_dir.mkdir(parents=True, exist_ok=True)

    # Save .pth file using shutil.copyfileobj (guarantees complete byte-for-byte stream)
    weights_path = target_dir / file.filename
    try:
        with open(weights_path, "wb") as f:
            shutil.copyfileobj(file.file, f)
    except Exception as exc:
        logger.exception("Failed to write weight file %s", weights_path)
        raise HTTPException(status_code=500, detail=f"Failed to save weight file: {exc}")

    # Build manifest dict
    manifest_dict = {
        "arch": data.arch.lower(),
        "model_id": clean_id,
        "display_name": data.display_name,
        "weights_file": file.filename,
        "num_classes": data.num_classes,
        "resolution": data.resolution,
        "confidence_default": data.confidence_default,
        "grid_size": data.grid_size,
        "overlap": data.overlap,
        "iou_threshold": data.iou_threshold,
        "class_names": data.class_names,
    }

    manifest_path = target_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest_dict, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    logger.info("New model '%s' uploaded and manifest written to %s", clean_id, manifest_path)

    # Rescan registry
    registry.scan()

    info = registry.get_model_info(clean_id)
    if info is None:
        raise HTTPException(status_code=500, detail="Model uploaded but scan failed to locate it.")
    return info


@router.delete("/models/{model_id}")
@router.delete("/api/models/{model_id}")
def delete_model(model_id: str) -> dict:
    """Delete a model directory from weights/ and clear it from memory."""
    try:
        registry.delete_model(model_id)
        return {"status": "ok", "message": f"Model '{model_id}' deleted successfully."}
    except Exception as exc:
        logger.exception("Failed to delete model '%s'", model_id)
        raise HTTPException(status_code=500, detail=str(exc))


