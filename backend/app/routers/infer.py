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
from app.models import registry
from app.schemas import Detection, InferRequest, InferResponse, ModelDetections, OCRLine
from app.tiling import adaptive_tile_image, merge_adaptive_detections, merge_detections, tile_image

logger = logging.getLogger(__name__)
router = APIRouter(tags=["infer"])
async def run_azure_ocr(image_bytes: bytes, max_retries: int = 5) -> list[OCRLine]:
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
            # Retry loop for rate limiting (429) on initial POST
            for attempt in range(max_retries):
                logger.info("Submitting OCR request to Azure Document Intelligence... (attempt %d/%d)", attempt + 1, max_retries)
                response = await client.post(url, headers=headers, content=image_bytes, timeout=30.0)

                if response.status_code == 429:
                    wait_time = min(2 ** attempt * 2, 30)  # 2s, 4s, 8s, 16s, 30s
                    logger.warning("Azure OCR rate limited (429). Retrying in %ds... (attempt %d/%d)",
                                   wait_time, attempt + 1, max_retries)
                    await asyncio.sleep(wait_time)
                    continue

                if response.status_code != 202:
                    logger.error("Azure OCR request submission failed: %s - %s", response.status_code, response.text)
                    return []

                operation_location = response.headers.get("Operation-Location")
                if not operation_location:
                    logger.error("Azure OCR response missing Operation-Location header")
                    return []

                # Poll for the result with resilience to transient errors
                for poll_idx in range(90):
                    await asyncio.sleep(1.0)
                    try:
                        res = await client.get(operation_location, headers={"Ocp-Apim-Subscription-Key": key})
                        if res.status_code != 200:
                            logger.warning("Azure OCR polling status %s (poll %d/90). Retrying...", res.status_code, poll_idx + 1)
                            await asyncio.sleep(2.0)
                            continue

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
                    except Exception as poll_exc:
                        logger.warning("Exception during Azure OCR polling: %s. Retrying...", poll_exc)
                        await asyncio.sleep(2.0)
                        continue

                logger.error("Azure OCR polling timed out after 90 seconds")
                return []

            logger.error("Azure OCR: exhausted %d retries due to rate limiting", max_retries)
            return []

    except Exception as exc:
        logger.exception("Error during Azure Document Intelligence OCR execution")
        return []


def _box_iou(box_a: list[float], box_b: list[float]) -> float:
    """Compute IoU = Intersection / Union for two [x1, y1, x2, y2] boxes."""
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    if inter == 0.0:
        return 0.0
    area_a = max(0.0, box_a[2] - box_a[0]) * max(0.0, box_a[3] - box_a[1])
    area_b = max(0.0, box_b[2] - box_b[0]) * max(0.0, box_b[3] - box_b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _text_similarity(text_a: str, text_b: str) -> float:
    """Compute text similarity using SequenceMatcher (0.0 to 1.0)."""
    from difflib import SequenceMatcher
    if not text_a or not text_b:
        return 0.0
    return SequenceMatcher(None, text_a.lower().strip(), text_b.lower().strip()).ratio()


def _is_substring_match(str1: str, str2: str) -> bool:
    s1, s2 = str1.lower().strip(), str2.lower().strip()
    return (s1 in s2 or s2 in s1) if (len(s1) >= 2 and len(s2) >= 2) else (s1 == s2)


def _box_containment(box_a: list[float], box_b: list[float]) -> float:
    """Compute what fraction of box_a is inside box_b (0.0 to 1.0)."""
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, box_a[2] - box_a[0]) * max(0.0, box_a[3] - box_a[1])
    return inter / area_a if area_a > 0 else 0.0


def _deduplicate_ocr_lines(ocr_lines: list[OCRLine]) -> list[OCRLine]:
    """
    Deduplicate OCR text detections safely using IoU, containment, and text similarity.

    Two OCR detections are duplicates ONLY when:
    1. IoU > 0.35 AND (text_similarity > 0.35 OR substring match)
    2. IoU > 0.20 AND text_similarity > 0.80 (nearly identical text in almost same spot)
    3. Substring match AND one box is >70% contained in the other (partial tile crop cleanup)

    Distinct small text inside or near larger text boxes WILL NOT be deleted
    because IoU (Intersection / Union) for boxes of different sizes is low.
    """
    if not ocr_lines:
        return []

    n = len(ocr_lines)
    suppressed = [False] * n

    for i in range(n):
        if suppressed[i]:
            continue
        for j in range(i + 1, n):
            if suppressed[j]:
                continue

            box_i = ocr_lines[i].box
            box_j = ocr_lines[j].box
            text_i = ocr_lines[i].text
            text_j = ocr_lines[j].text

            iou = _box_iou(box_i, box_j)
            text_sim = _text_similarity(text_i, text_j)
            sub_match = _is_substring_match(text_i, text_j)

            containment_i_in_j = _box_containment(box_i, box_j)
            containment_j_in_i = _box_containment(box_j, box_i)
            high_containment = (containment_i_in_j > 0.70 or containment_j_in_i > 0.70)

            is_dup = False
            if iou > 0.35 and (text_sim > 0.35 or sub_match):
                is_dup = True
            elif iou > 0.20 and text_sim > 0.80:
                is_dup = True
            elif sub_match and high_containment:
                is_dup = True

            if is_dup:
                # Keep the longer or more complete text string
                if len(text_j.strip()) > len(text_i.strip()):
                    logger.debug(
                        "OCR dedup: suppressing [%d] '%s' in favor of [%d] '%s' (iou=%.2f, sim=%.2f)",
                        i, text_i[:30], j, text_j[:30], iou, text_sim
                    )
                    suppressed[i] = True
                    break
                else:
                    logger.debug(
                        "OCR dedup: suppressing [%d] '%s' in favor of [%d] '%s' (iou=%.2f, sim=%.2f)",
                        j, text_j[:30], i, text_i[:30], iou, text_sim
                    )
                    suppressed[j] = True

    kept = [line for line, is_suppressed in zip(ocr_lines, suppressed) if not is_suppressed]
    return kept


def _sort_ocr_reading_order(ocr_lines: list[OCRLine]) -> list[OCRLine]:
    """Sort OCR lines in reading order: top-to-bottom then left-to-right."""
    if not ocr_lines:
        return []

    def sort_key(line: OCRLine) -> tuple[float, float]:
        y_center = (line.box[1] + line.box[3]) / 2.0
        x_center = (line.box[0] + line.box[2]) / 2.0
        return (y_center, x_center)

    return sorted(ocr_lines, key=sort_key)


async def run_azure_ocr_tiled(
    pil_image: Image.Image,
    image_bytes: bytes,
    ocr_grid_size: int,
) -> tuple[list[OCRLine], list[list[float]]]:
    """
    Run Azure OCR with pure N x N tiling strategy.
    Baseline 1x1 is currently disabled for pure N x N isolation testing.
    """
    if ocr_grid_size <= 1:
        results = await run_azure_ocr(image_bytes)
        return _sort_ocr_reading_order(results), []

    # 1. Baseline 1x1 OCR commented out for now to isolate pure N x N tiling logic
    # logger.info("OCR Hybrid: running 1x1 full image baseline...")
    # full_image_ocr = await run_azure_ocr(image_bytes)
    # logger.info("OCR Hybrid: 1x1 full image returned %d lines", len(full_image_ocr))

    # 2. Tile the image with 40% overlap so text lines crossing boundaries are not cut off
    ocr_overlap = 0.40
    tiles = tile_image(pil_image, grid_size=ocr_grid_size, overlap=ocr_overlap)
    tile_boxes = [
        [float(x_off), float(y_off), float(x_off + tile_img.width), float(y_off + tile_img.height)]
        for tile_img, x_off, y_off in tiles
    ]

    logger.info(
        "OCR Pure Tiling: running grid %dx%d (%d tiles, overlap=%.0f%%)...",
        ocr_grid_size, ocr_grid_size, len(tiles), ocr_overlap * 100
    )

    sem = asyncio.Semaphore(1)  # Process 1 tile at a time to prevent rate limits

    async def process_tile(tile_idx: int, tile_img: Image.Image, x_off: int, y_off: int) -> list[OCRLine]:
        async with sem:
            await asyncio.sleep(0.3)  # Gentle delay between requests
            logger.info("OCR tiling: processing tile %d/%d (offset=%d,%d)...", tile_idx + 1, len(tiles), x_off, y_off)
            buf = io.BytesIO()
            tile_img.save(buf, format="PNG")
            tile_bytes = buf.getvalue()

            tile_ocr = await run_azure_ocr(tile_bytes)

            remapped: list[OCRLine] = []
            for line in tile_ocr:
                remapped.append(OCRLine(
                    text=line.text,
                    box=[
                        line.box[0] + x_off,
                        line.box[1] + y_off,
                        line.box[2] + x_off,
                        line.box[3] + y_off,
                    ],
                ))
            logger.info("OCR tiling: tile %d/%d returned %d lines", tile_idx + 1, len(tiles), len(remapped))
            return remapped

    tasks = [process_tile(idx, tile_img, x_off, y_off)
             for idx, (tile_img, x_off, y_off) in enumerate(tiles)]
    tile_results = await asyncio.gather(*tasks, return_exceptions=True)

    all_ocr_lines: list[OCRLine] = []

    for idx, result in enumerate(tile_results):
        if isinstance(result, Exception):
            logger.error("OCR tiling: tile %d failed: %s", idx, result)
            continue
        all_ocr_lines.extend(result)

    logger.info("OCR Hybrid: collected %d raw lines (1x1 baseline + %d tiles), deduplicating...",
                len(all_ocr_lines), len(tiles))

    deduped = _deduplicate_ocr_lines(all_ocr_lines)
    sorted_lines = _sort_ocr_reading_order(deduped)

    logger.info("OCR Hybrid: %d lines after deduplication (removed %d duplicates)",
                len(sorted_lines), len(all_ocr_lines) - len(sorted_lines))

    return sorted_lines, tile_boxes


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

    # ── Run each model sequentially (safe on single GPU) ─────────────────
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
    )

