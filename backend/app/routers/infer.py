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
import asyncio
import httpx

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from PIL import Image
import io
import torch

from app.config import settings
from app.component.models import registry
from app.schemas import Detection, InferRequest, InferResponse, ModelDetections, OCRLine
from app.component.tiling import adaptive_tile_image, merge_adaptive_detections, merge_detections
from app.ocr import run_azure_ocr_tiled
from app.busbar_panel import detect as detect_busbar_panel
from pathlib import Path
import sys
import cv2
import numpy as np

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
    and return raw detections (score ≥ MIN_SCORE_FLOOR) along with OCR text detections.

    Supports inference_mode:
      - "components": Only GPU model inference, no OCR
      - "ocr": Only Azure Document Intelligence OCR, no GPU inference
      - "both": Both component detection and OCR (default, current behavior)

    Threshold filtering is intentionally deferred to the frontend so the
    slider gives instant feedback without additional network round-trips.
    """
    # ── Parse request body ───────────────────────────────────────────────
    try:
        req = InferRequest(**json.loads(body))
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Invalid body JSON: {exc}")

    mode = req.inference_mode or "both"
    run_components = mode in ("components", "both")
    run_ocr = mode in ("ocr", "both")

    logger.info("Inference mode: %s (components=%s, ocr=%s, ocr_grid=%d)",
                mode, run_components, run_ocr, req.ocr_grid_size)

    # Validate that model_ids are provided when running component detection
    if run_components and len(req.model_ids) == 0:
        raise HTTPException(status_code=422, detail="model_ids required for component detection")

    # ── Load image ───────────────────────────────────────────────────────
    try:
        image_bytes = await image.read()
        pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Cannot open image: {exc}")

    W, H = pil_image.size
    detections_dict: dict[str, ModelDetections] = {}

    # Start OCR task if needed (runs in parallel with GPU inference)
    ocr_task = None
    if run_ocr:
        ocr_task = asyncio.create_task(
            run_azure_ocr_tiled(pil_image, image_bytes, req.ocr_grid_size)
        )

    # ── Run deterministic panel detector (once per upload, if components mode active) ──
    deterministic_panels = []
    if run_components:
        try:
            img_bgr = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
            det_res = detect_busbar_panel(img_bgr)
            deterministic_panels = det_res.get("panels", [])
            logger.info("Deterministic panel detector found %d panels", len(deterministic_panels))
        except Exception as exc:
            logger.exception("Failed to run deterministic panel detector")

    component_tiles: list[list[float]] | None = None
    if run_components:
        for model_id in req.model_ids:
            try:
                # Checkpoint deserialization can take minutes for a large model.
                # Do it off the event loop so health checks and other requests
                # remain responsive while this request is starting.
                wrapper = await asyncio.to_thread(registry.get_or_load, model_id)
            except KeyError:
                if ocr_task:
                    ocr_task.cancel()
                raise HTTPException(status_code=404, detail=f"Model '{model_id}' manifest not found.")
            except (FileNotFoundError, ValueError) as exc:
                if ocr_task:
                    ocr_task.cancel()
                raise HTTPException(status_code=400, detail=str(exc))
            except Exception as exc:
                if ocr_task:
                    ocr_task.cancel()
                logger.exception("Failed to load model '%s'", model_id)
                raise HTTPException(status_code=500, detail=f"Model load error: {exc}")

            manifest = wrapper.manifest
            iou_threshold = manifest.get("iou_threshold", 0.50)
            effective_tiling_mode = req.tiling_mode or manifest.get("tiling_mode", "fixed")
            effective_overlap = (
                req.overlap if req.overlap is not None else manifest.get("overlap", 0.20)
            )

            try:
                if req.use_tiling:
                    if effective_tiling_mode == "adaptive":
                        tiles_info = adaptive_tile_image(
                            pil_image,
                            target_symbol_px=(
                                req.target_symbol_px
                                if req.target_symbol_px is not None
                                else manifest.get("target_symbol_px", 48.0)
                            ),
                            estimated_symbol_px=(
                                req.estimated_symbol_px
                                if req.estimated_symbol_px is not None
                                else manifest.get("estimated_symbol_px", 48.0)
                            ),
                            model_input_size=manifest.get("resolution", 640),
                            overlap=effective_overlap,
                            enable_auto_crop=(
                                req.enable_auto_crop
                                if req.enable_auto_crop is not None
                                else manifest.get("enable_auto_crop", False)
                            ),
                            enable_scale_norm=(
                                req.enable_scale_norm
                                if req.enable_scale_norm is not None
                                else manifest.get("enable_scale_norm", False)
                            ),
                            target_reference_height=manifest.get("target_reference_height", 60.0),
                        )
                        logger.info(
                            "Adaptive inference model='%s' image=%sx%s tiles=%d estimated_symbol_px=%.1f "
                            "target_symbol_px=%.1f crop=%s scale_norm=%s",
                            model_id,
                            W,
                            H,
                            len(tiles_info),
                            float(req.estimated_symbol_px if req.estimated_symbol_px is not None else manifest.get("estimated_symbol_px", 48.0)),
                            float(req.target_symbol_px if req.target_symbol_px is not None else manifest.get("target_symbol_px", 48.0)),
                            req.enable_auto_crop if req.enable_auto_crop is not None else manifest.get("enable_auto_crop", False),
                            req.enable_scale_norm if req.enable_scale_norm is not None else manifest.get("enable_scale_norm", False),
                        )
                        adaptive_results = []
                        for tile_resized, tx1, ty1, tw, th, cx1, cy1, s in tiles_info:
                            det = await asyncio.to_thread(wrapper.infer, tile_resized)
                            adaptive_results.append((det, tx1, ty1, tw, th, cx1, cy1, s))
                        
                        if component_tiles is None:
                            component_tiles = []
                            for _, tx1, ty1, tw, th, cx1, cy1, s in tiles_info:
                                x1 = (tx1 + cx1) / s
                                y1 = (ty1 + cy1) / s
                                x2 = (tx1 + tw + cx1) / s
                                y2 = (ty1 + th + cy1) / s
                                component_tiles.append([float(x1), float(y1), float(x2), float(y2)])

                        detections = merge_adaptive_detections(
                            adaptive_results,
                            W,
                            H,
                            model_input_size=manifest.get("resolution", 640),
                            iou_threshold=iou_threshold,
                        )
                    else:
                        grid_sz = req.grid_size if req.grid_size is not None else manifest.get("grid_size", 4)
                        tiles = tile_image(pil_image, grid_size=grid_sz, overlap=effective_overlap)
                        
                        if component_tiles is None:
                            component_tiles = []
                            W_orig, H_orig = pil_image.size
                            denominator = grid_sz - (grid_sz - 1) * effective_overlap
                            tile_w = int(np.ceil(W_orig / denominator))
                            tile_h = int(np.ceil(H_orig / denominator))
                            for _, x_off, y_off in tiles:
                                x2 = min(x_off + tile_w, W_orig)
                                y2 = min(y_off + tile_h, H_orig)
                                component_tiles.append([float(x_off), float(y_off), float(x2), float(y2)])

                        tile_results = []
                        for tile, x_off, y_off in tiles:
                            det = await asyncio.to_thread(wrapper.infer, tile)
                            tile_results.append((det, x_off, y_off))
                        detections = merge_detections(tile_results, W, H, iou_threshold)
                else:
                    detections = await asyncio.to_thread(wrapper.infer, pil_image)

            except Exception as exc:
                if ocr_task:
                    ocr_task.cancel()
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

            logger.info(
                "Inference result model='%s' tiling=%s raw_detections=%d returned_detections=%d",
                model_id,
                effective_tiling_mode if req.use_tiling else "disabled",
                len(detections),
                len(det_list),
            )
            if len(det_list) == 0:
                logger.warning(
                    "No detections for model='%s'. Check the raw score floor (%.3f), "
                    "adaptive preprocessing, and that the expected checkpoint was loaded.",
                    model_id,
                    settings.min_score_floor,
                )

            detections_dict[model_id] = ModelDetections(
                class_names=manifest.get("class_names", []),
                detections=det_list,
                panels=deterministic_panels,
            )

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Await OCR results if running
    ocr_results: list[OCRLine] | None = None
    ocr_tile_boxes: list[list[float]] | None = None
    if ocr_task is not None:
        lines, tile_boxes = await ocr_task
        ocr_results = lines if lines else None
        ocr_tile_boxes = tile_boxes if tile_boxes else None

    return InferResponse(
        detections=detections_dict,
        ocr=ocr_results,
        ocr_tiles=ocr_tile_boxes,
        component_tiles=component_tiles,
    )

