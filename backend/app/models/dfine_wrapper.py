"""
D-FINE model wrapper — ported from notebook Cells 3 + 4.

Architecture: HGNetv2-B5 backbone, HybridEncoder, DFINETransformer.
The YAML config is reconstructed inline from the manifest parameters;
if a sibling config.yml exists next to the weights it will be used instead
(preferred, as it guarantees an exact architecture match).
"""
from __future__ import annotations

import os
import sys
import tempfile
import logging
from pathlib import Path

import numpy as np
import supervision as sv
import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image

from app.config import settings
from app.models.base import BaseModelWrapper

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# YAML template — mirrors the reconstructed config from the notebook exactly.
# num_classes and eval_spatial_size are injected from the manifest at runtime.
# ---------------------------------------------------------------------------
_DFINE_CFG_TEMPLATE = """\
task: detection
model: DFINE
postprocessor: DFINEPostProcessor
num_classes: {num_classes}
eval_spatial_size: [{resolution}, {resolution}]
use_focal_loss: True

DFINE:
  backbone: HGNetv2
  encoder: HybridEncoder
  decoder: DFINETransformer

HGNetv2:
  name: B5
  return_idx: [1, 2, 3]
  freeze_stem_only: True
  freeze_at: 0
  freeze_norm: True
  pretrained: False

HybridEncoder:
  in_channels: [512, 1024, 2048]
  feat_strides: [8, 16, 32]
  hidden_dim: 384
  use_encoder_idx: [2]
  num_encoder_layers: 1
  nhead: 8
  dim_feedforward: 2048
  dropout: 0.0
  enc_act: 'gelu'
  expansion: 1.0
  depth_mult: 1
  act: 'silu'

DFINETransformer:
  feat_channels: [384, 384, 384]
  feat_strides: [8, 16, 32]
  hidden_dim: 256
  num_levels: 3
  num_layers: 6
  eval_idx: -1
  num_queries: 300
  num_denoising: 100
  label_noise_ratio: 0.5
  box_noise_scale: 1.0
  layer_scale: 1
  num_points: [3, 6, 3]
  cross_attn_method: default
  query_select_method: default
  reg_max: 32
  reg_scale: 8

DFINEPostProcessor:
  num_top_queries: 300
"""


class _DFINEInferModel(nn.Module):
    """Thin wrapper that combines backbone + postprocessor for a single forward pass."""

    def __init__(self, cfg) -> None:
        super().__init__()
        self.model = cfg.model.deploy()
        self.postprocessor = cfg.postprocessor.deploy()

    def forward(self, images: torch.Tensor, orig_target_sizes: torch.Tensor):
        outputs = self.model(images)
        outputs = self.postprocessor(outputs, orig_target_sizes)
        return outputs  # (labels, boxes, scores)


class DFINEWrapper(BaseModelWrapper):
    """
    D-FINE inference wrapper.

    Drops the D-FINE repo path into sys.path to import src.core.YAMLConfig,
    then builds the model from a config YAML and loads the checkpoint.
    """

    def load(self) -> None:
        # ── 1. Make D-FINE importable ────────────────────────────────────
        dfine_path = settings.dfine_repo_path
        if dfine_path not in sys.path:
            sys.path.insert(0, dfine_path)
        try:
            from src.core import YAMLConfig  # noqa: F401 – import test
        except ImportError as exc:
            raise RuntimeError(
                f"Cannot import D-FINE from '{dfine_path}'. "
                "Set DFINE_REPO_PATH in .env to the cloned D-FINE repo root."
            ) from exc

        # ── 2. Resolve paths ─────────────────────────────────────────────
        manifest = self.manifest
        weights_path = Path(self.weights_dir) / manifest["weights_file"]
        if not weights_path.exists():
            raise FileNotFoundError(f"D-FINE weights not found: {weights_path}")

        # Optional: use a hand-crafted config.yml next to the .pth
        custom_cfg = weights_path.parent / "config.yml"
        num_classes = manifest["num_classes"]
        resolution = manifest.get("resolution", 640)

        # ── 3. Write / locate the config file ────────────────────────────
        if custom_cfg.exists():
            logger.info("D-FINE: using custom config %s", custom_cfg)
            cfg_path = str(custom_cfg)
        else:
            yaml_text = _DFINE_CFG_TEMPLATE.format(
                num_classes=num_classes,
                resolution=resolution,
            )
            tmp = tempfile.NamedTemporaryFile(
                mode="w", suffix=".yml", delete=False, prefix="dfine_cfg_"
            )
            tmp.write(yaml_text)
            tmp.close()
            cfg_path = tmp.name
            logger.info("D-FINE: wrote reconstructed config to %s", cfg_path)

        # ── 4. Build model ───────────────────────────────────────────────
        from src.core import YAMLConfig

        cfg = YAMLConfig(cfg_path, resume=str(weights_path))

        try:
            checkpoint = torch.load(str(weights_path), map_location="cpu", weights_only=False)
        except Exception as exc:
            err_str = str(exc)
            if "PytorchStreamReader" in err_str or "zip archive" in err_str or "miniz" in err_str:
                raise ValueError(
                    f"Checkpoint file '{weights_path.name}' is corrupted or incomplete on disk (truncated upload). "
                    f"Please re-upload the weight file to server."
                ) from exc
            raise

        if "ema" in checkpoint:
            state = checkpoint["ema"]["module"]
        elif "model" in checkpoint:
            state = checkpoint["model"]
        else:
            state = checkpoint  # raw state_dict

        cfg.model.load_state_dict(state)

        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._model = _DFINEInferModel(cfg).to(self._device).eval()
        logger.info("D-FINE loaded on %s from %s", self._device, weights_path)

        # ── 5. Build transform (matches training pipeline exactly) ───────
        self._transform = T.Compose([
            T.Resize((resolution, resolution)),
            T.ToTensor(),
        ])

        self._floor = settings.min_score_floor
        self._loaded = True

    @torch.no_grad()
    def infer(self, pil_image: Image.Image) -> sv.Detections:
        if not self._loaded:
            self.load()

        W, H = pil_image.size
        img_t = self._transform(pil_image.convert("RGB")).unsqueeze(0).to(self._device)
        orig_size = torch.tensor([[W, H]], device=self._device)

        labels, boxes, scores = self._model(img_t, orig_size)

        labels = labels[0].cpu().numpy().astype(int)
        boxes  = boxes[0].cpu().numpy().astype(np.float32)
        scores = scores[0].cpu().numpy().astype(np.float32)

        keep = scores >= self._floor
        if not keep.any():
            return sv.Detections.empty()

        return sv.Detections(
            xyxy=boxes[keep],
            confidence=scores[keep],
            class_id=labels[keep],
        )
