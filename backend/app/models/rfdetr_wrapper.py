"""
RF-DETR model wrapper — ported from notebook Cell 6.

Uses the official `rfdetr` pip package (RFDETRLarge).
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import supervision as sv
from PIL import Image

from app.config import settings
from app.models.base import BaseModelWrapper

logger = logging.getLogger(__name__)


class RFDETRWrapper(BaseModelWrapper):
    """
    RF-DETR inference wrapper.

    The `rfdetr` package handles its own weight loading and device placement
    internally — we just construct RFDETRLarge with the checkpoint path.
    """

    def load(self) -> None:
        try:
            from rfdetr import RFDETRLarge  # noqa: F401 — import test
        except ImportError as exc:
            raise RuntimeError(
                "rfdetr package not installed. Run: pip install rfdetr"
            ) from exc

        manifest = self.manifest
        weights_path = Path(self.weights_dir) / manifest["weights_file"]
        if not weights_path.exists():
            raise FileNotFoundError(f"RF-DETR weights not found: {weights_path}")

        from rfdetr import RFDETRLarge

        logger.info("Loading RF-DETR from %s ...", weights_path)
        self._rfdetr = RFDETRLarge(
            pretrain_weights=str(weights_path),
            num_classes=manifest["num_classes"],
        )
        self._floor = settings.min_score_floor
        self._loaded = True
        logger.info("RF-DETR loaded.")

    def infer(self, pil_image: Image.Image) -> sv.Detections:
        if not self._loaded:
            self.load()

        # rfdetr.predict returns sv.Detections directly
        detections: sv.Detections = self._rfdetr.predict(
            pil_image.convert("RGB"),
            threshold=self._floor,
        )
        return detections
