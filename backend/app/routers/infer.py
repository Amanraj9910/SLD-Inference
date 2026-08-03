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

from app.config import settings
from app.models import registry
from app.schemas import Detection, InferRequest, InferResponse, ModelDetections, OCRLine
from app.tiling import merge_detections, tile_image

logger = logging.getLogger(__name__)
router = APIRouter(tags=["infer"])


async def run_azure_ocr(image_bytes: bytes) -> list[OCRLine]:
    endpoint = settings.azure_document_intelligence_endpoint
    key = settings.azure_document_intelligence_key

    if not endpoint or not key:
        logger.info("Azure Document Intelligence endpoint or key not configured. Skipping OCR.")
        return []

    if not endpoint.endswith("/"):
        endpoint += "/"

    # API version 2024-11-30 is recommended for v4.0 GA Document Intelligence
    url = f"{endpoint}documentintelligence/documentModels/prebuilt-read:analyze?api-version=2024-11-30"

    headers = {
        "Ocp-Apim-Subscription-Key": key,
        "Content-Type": "application/octet-stream"
    }

    try:
        async with httpx.AsyncClient() as client:
            logger.info("Submitting OCR request to Azure Document Intelligence...")
            response = await client.post(url, headers=headers, content=image_bytes, timeout=30.0)
            if response.status_code != 202:
                logger.error("Azure OCR request submission failed: %s - %s", response.status_code, response.text)
                return []

            operation_location = response.headers.get("Operation-Location")
            if not operation_location:
                logger.error("Azure OCR response missing Operation-Location header")
                return []

            # Poll for the result
            for poll_idx in range(30):
                await asyncio.sleep(1.0)
                res = await client.get(operation_location, headers={"Ocp-Apim-Subscription-Key": key})
                if res.status_code != 200:
                    logger.error("Azure OCR polling failed: %s", res.status_code)
                    return []

                result_data = res.json()
                status = result_data.get("status")
                if status == "succeeded":
                    ocr_lines = []
                    pages = result_data.get("analyzeResult", {}).get("pages", [])
                    for page in pages:
                        lines = page.get("lines", [])
                        for line in lines:
                            content = line.get("content", "")
                            polygon = line.get("polygon", [])
                            if len(polygon) == 8:
                                # Convert [x1, y1, x2, y2, x3, y3, x4, y4] to [x_min, y_min, x_max, y_max]
                                x_coords = polygon[0::2]
                                y_coords = polygon[1::2]
                                box = [
                                    float(min(x_coords)),
                                    float(min(y_coords)),
                                    float(max(x_coords)),
                                    float(max(y_coords))
                                ]
                                ocr_lines.append(OCRLine(text=content, box=box))
                    logger.info("Azure OCR finished successfully. Found %d lines.", len(ocr_lines))
                    return ocr_lines
                elif status == "failed":
                    logger.error("Azure OCR analyze operation failed: %s", result_data)
                    return []

    except Exception as exc:
        logger.exception("Error during Azure Document Intelligence OCR execution")
        return []

    return []


@router.post("/infer", response_model=InferResponse)
@router.post("/api/infer", response_model=InferResponse)
async def run_infer(
    image: UploadFile = File(..., description="SLD image (JPEG or PNG)"),
    body: str = Form(..., description="JSON string matching InferRequest"),
) -> InferResponse:
    """
    Run the requested models on the uploaded image (optionally with tiling)
    and return raw detections (score ≥ MIN_SCORE_FLOOR) along with OCR text detections.

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
    detections_dict: dict[str, ModelDetections] = {}

    # ── Run OCR in parallel with GPU model loading/inference if enabled ──
    ocr_task = run_azure_ocr(image_bytes)

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

        detections_dict[model_id] = ModelDetections(
            class_names=manifest.get("class_names", []),
            detections=det_list,
        )

    # Await OCR results
    ocr_results = await ocr_task

    return InferResponse(
        detections=detections_dict,
        ocr=ocr_results if ocr_results else None
    )
