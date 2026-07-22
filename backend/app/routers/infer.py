"""
POST /api/infer — run one or more models on an uploaded image.

Request:  multipart/form-data
  - image : UploadFile  (JPEG / PNG)
  - body  : JSON string matching InferRequest schema

Response: dict[model_id → ModelDetections]
"""
from __future__ import annotations

import json
import logging

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from PIL import Image
import io

from app.models import registry
from app.schemas import Detection, InferRequest, InferResponse, ModelDetections
from app.tiling import merge_detections, tile_image

logger = logging.getLogger(__name__)
router = APIRouter(tags=["infer"])


@router.post("/infer", response_model=InferResponse)
@router.post("/api/infer", response_model=InferResponse)
async def run_infer(
    image: UploadFile = File(..., description="SLD image (JPEG or PNG)"),
    body: str = Form(..., description="JSON string matching InferRequest"),
) -> InferResponse:
    """
    Run the requested models on the uploaded image (optionally with tiling)
    and return raw detections (score ≥ MIN_SCORE_FLOOR).

    Threshold filtering is intentionally deferred to the frontend so the
    slider gives instant feedback without additional network round-trips.
    """
    # ── Parse request body ───────────────────────────────────────────────
    try:
        req = InferRequest(**json.loads(body))
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Invalid body JSON: {exc}")

    # ── Load image ───────────────────────────────────────────────────────
    try:
        image_bytes = await image.read()
        pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Cannot open image: {exc}")

    W, H = pil_image.size
    response: InferResponse = {}

    # ── Run each model sequentially (safe on single GPU) ─────────────────
    for model_id in req.model_ids:
        try:
            wrapper = registry.get_or_load(model_id)
        except KeyError:
            raise HTTPException(status_code=404, detail=f"Model '{model_id}' manifest not found.")
        except (FileNotFoundError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        except Exception as exc:
            logger.exception("Failed to load model '%s'", model_id)
            raise HTTPException(status_code=500, detail=f"Model load error: {exc}")

        manifest = wrapper.manifest
        iou_threshold = manifest.get("iou_threshold", 0.50)

        try:
            if req.use_tiling:
                tiles = tile_image(pil_image, grid_size=req.grid_size, overlap=req.overlap)
                tile_results = []
                for tile, x_off, y_off in tiles:
                    det = wrapper.infer(tile)
                    tile_results.append((det, x_off, y_off))
                detections = merge_detections(tile_results, W, H, iou_threshold)
            else:
                detections = wrapper.infer(pil_image)

        except Exception as exc:
            logger.exception("Inference failed for model '%s'", model_id)
            raise HTTPException(status_code=500, detail=f"Inference error ({model_id}): {exc}")

        # Convert sv.Detections → API schema
        det_list: list[Detection] = []
        if len(detections) > 0:
            for box, cid, score in zip(
                detections.xyxy,
                detections.class_id,
                detections.confidence,
            ):
                det_list.append(
                    Detection(
                        box=[float(box[0]), float(box[1]), float(box[2]), float(box[3])],
                        class_id=int(cid),
                        score=float(score),
                    )
                )

        response[model_id] = ModelDetections(
            class_names=manifest.get("class_names", []),
            detections=det_list,
        )

    return response
