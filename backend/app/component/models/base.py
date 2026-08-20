"""
Abstract base class every model wrapper must implement.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import supervision as sv
from PIL import Image


class BaseModelWrapper(ABC):
    """
    Common interface for D-FINE and RF-DETR wrappers.

    Subclasses receive the parsed manifest dict at construction time and
    are expected to load weights lazily (or eagerly) via load().
    """

    def __init__(self, manifest: dict, weights_dir: str) -> None:
        self.manifest = manifest
        self.weights_dir = weights_dir
        self._loaded = False

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    @abstractmethod
    def load(self) -> None:
        """Load weights onto GPU.  Called once per process lifetime."""
        ...

    @abstractmethod
    def infer(self, pil_image: Image.Image) -> sv.Detections:
        """
        Run inference on a single PIL image.
        Must return raw detections with score >= MIN_SCORE_FLOOR;
        threshold filtering is done client-side.
        """
        ...
