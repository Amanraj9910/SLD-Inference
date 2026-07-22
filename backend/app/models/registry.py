"""
Model registry — scans weights/ for manifest.json files, holds loaded
model instances in memory so the GPU pays the load cost once per process.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Iterator

from app.config import settings
from app.models.base import BaseModelWrapper
from app.schemas import ModelInfo

logger = logging.getLogger(__name__)

# In-memory store: model_id → loaded wrapper instance
_registry: dict[str, BaseModelWrapper] = {}

# In-memory manifest cache (keeps display metadata even for unloaded models)
_manifests: dict[str, dict] = {}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _manifest_iter(weights_dir: Path) -> Iterator[tuple[str, Path, dict]]:
    """Yield (model_id, weights_subdir, manifest_dict) for every manifest found."""
    for manifest_path in sorted(weights_dir.glob("*/manifest.json")):
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception as exc:
            logger.warning("Skipping %s — parse error: %s", manifest_path, exc)
            continue

        arch = manifest.get("arch", "").lower()
        if arch not in ("dfine", "rfdetr"):
            logger.warning("Skipping %s — unknown arch '%s'", manifest_path, arch)
            continue

        # model_id is derived from the subfolder name, e.g. weights/dfine/ → "dfine"
        folder_name = manifest_path.parent.name
        model_id = manifest.get("model_id", folder_name)
        manifest["_weights_subdir"] = str(manifest_path.parent)
        yield model_id, manifest_path.parent, manifest


def _build_wrapper(model_id: str, manifest: dict) -> BaseModelWrapper:
    arch = manifest["arch"].lower()
    weights_subdir = manifest["_weights_subdir"]

    if arch == "dfine":
        from app.models.dfine_wrapper import DFINEWrapper
        return DFINEWrapper(manifest, weights_subdir)
    elif arch == "rfdetr":
        from app.models.rfdetr_wrapper import RFDETRWrapper
        return RFDETRWrapper(manifest, weights_subdir)
    else:
        raise ValueError(f"Unknown arch '{arch}' for model '{model_id}'")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def scan() -> list[ModelInfo]:
    """
    Scan weights/ and return ModelInfo for every discovered manifest.
    Also populates the internal manifest cache (_manifests).
    """
    weights_dir = settings.resolved_weights_dir
    if not weights_dir.exists():
        logger.warning("weights_dir '%s' does not exist — no models discovered.", weights_dir)
        return []

    results: list[ModelInfo] = []
    for model_id, weights_subdir, manifest in _manifest_iter(weights_dir):
        _manifests[model_id] = manifest
        weights_file = manifest.get("weights_file", "")
        weights_path = weights_subdir / weights_file if weights_file else None
        weights_exist = bool(weights_path and weights_path.is_file() and weights_path.stat().st_size > 1024 * 10)

        results.append(
            ModelInfo(
                model_id=model_id,
                arch=manifest["arch"],
                display_name=manifest.get("display_name", model_id),
                weights_file=weights_file,
                num_classes=manifest["num_classes"],
                resolution=manifest.get("resolution", 640),
                class_names=manifest.get("class_names", []),
                confidence_default=manifest.get("confidence_default", 0.20),
                grid_size=manifest.get("grid_size", 4),
                overlap=manifest.get("overlap", 0.20),
                iou_threshold=manifest.get("iou_threshold", 0.50),
                loaded=model_id in _registry,
                weights_exist=weights_exist,
            )
        )
    return results


def get_model_info(model_id: str) -> ModelInfo | None:
    """Return ModelInfo for a single model, or None if not found."""
    for info in scan():
        if info.model_id == model_id:
            return info
    return None


def get_or_load(model_id: str) -> BaseModelWrapper:
    """
    Return the loaded wrapper for *model_id*, loading it on first call.
    Raises KeyError if the model_id is not in any discovered manifest.
    Raises FileNotFoundError if the .pth weight file is missing on disk.
    """
    if model_id in _registry:
        return _registry[model_id]

    # Ensure manifests are populated
    if not _manifests:
        scan()

    if model_id not in _manifests:
        raise KeyError(f"Model '{model_id}' not found in weights/")

    manifest = _manifests[model_id]
    weights_subdir = Path(manifest["_weights_subdir"])
    weights_file = manifest.get("weights_file", "")
    if not (weights_subdir / weights_file).is_file():
        raise FileNotFoundError(
            f"Checkpoint file '{weights_file}' for model '{model_id}' was not found at {weights_subdir / weights_file}. "
            f"Please upload '{weights_file}' to the server."
        )

    wrapper = _build_wrapper(model_id, manifest)
    wrapper.load()
    _registry[model_id] = wrapper
    logger.info("Model '%s' loaded and cached.", model_id)
    return wrapper


def update_manifest(
    model_id: str,
    class_names: list[str] | None,
    confidence_default: float | None,
) -> ModelInfo:
    """
    Update class_names and/or confidence_default for a model — both in the
    in-memory manifest and on disk.  No model reload is required because these
    fields don't touch the network weights.
    """
    if not _manifests:
        scan()

    if model_id not in _manifests:
        raise KeyError(f"Model '{model_id}' not found.")

    manifest = _manifests[model_id]
    if class_names is not None:
        manifest["class_names"] = class_names
    if confidence_default is not None:
        manifest["confidence_default"] = confidence_default

    # Persist to disk
    weights_subdir = Path(manifest["_weights_subdir"])
    manifest_path = weights_subdir / "manifest.json"
    serialisable = {k: v for k, v in manifest.items() if not k.startswith("_")}
    manifest_path.write_text(
        json.dumps(serialisable, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    logger.info("manifest.json updated for model '%s'.", model_id)

    if model_id in _registry:
        _registry[model_id].manifest = manifest

    weights_file = manifest.get("weights_file", "")
    weights_path = weights_subdir / weights_file if weights_file else None
    weights_exist = bool(weights_path and weights_path.is_file() and weights_path.stat().st_size > 1024 * 10)

    return ModelInfo(
        model_id=model_id,
        arch=manifest["arch"],
        display_name=manifest.get("display_name", model_id),
        weights_file=weights_file,
        num_classes=manifest["num_classes"],
        resolution=manifest.get("resolution", 640),
        class_names=manifest["class_names"],
        confidence_default=manifest["confidence_default"],
        grid_size=manifest.get("grid_size", 4),
        overlap=manifest.get("overlap", 0.20),
        iou_threshold=manifest.get("iou_threshold", 0.50),
        loaded=model_id in _registry,
        weights_exist=weights_exist,
    )


def delete_model(model_id: str) -> None:
    """Delete a model's subfolder from weights/ and uncache it."""
    if model_id in _registry:
        del _registry[model_id]

    if not _manifests:
        scan()

    if model_id in _manifests:
        manifest = _manifests[model_id]
        weights_subdir = Path(manifest["_weights_subdir"])
        if weights_subdir.exists():
            import shutil
            shutil.rmtree(weights_subdir, ignore_errors=True)
        del _manifests[model_id]
        logger.info("Model '%s' directory removed from disk.", model_id)

    scan()

