"""
OCR processing services using Azure Document Intelligence.
"""
from __future__ import annotations

import asyncio
import logging
import io
import httpx
from PIL import Image

from app.config import settings
from app.schemas import OCRLine
from app.component.tiling import tile_image

logger = logging.getLogger(__name__)


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

    # Tile the image with 40% overlap so text lines crossing boundaries are not cut off
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
