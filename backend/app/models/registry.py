"""
Model registry — scans weights/ for manifest.json files, holds loaded
model instances in memory so the GPU pays the load cost once per process.
"""
from __future__ import annotations

import json
import logging
import threading
from pathlib import Path
from typing import Iterator

from app.config import settings
from app.models.base import BaseModelWrapper
from app.schemas import ModelInfo

logger = logging.getLogger(__name__)

# In-memory store: model_id → loaded wrapper instance
_registry: dict[str, BaseModelWrapper] = {}

# Checkpoint identity for each loaded wrapper.  This lets a running server
# notice when a .pth file is replaced on disk instead of silently reusing the
# old model object.
_loaded_signatures: dict[str, tuple] = {}
_registry_lock = threading.Lock()

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

        class_names = manifest.get("class_names", [])
        if isinstance(class_names, list):
            cleaned_names = [
                name.strip().strip(",").strip('"') if isinstance(name, str) else name
                for name in class_names
            ]
            if cleaned_names != class_names:
                logger.warning(
                    "Normalizing serialized class names in %s; regenerate the manifest from the processed COCO categories.",
                    manifest_path,
                )
                manifest["class_names"] = cleaned_names
            if len(cleaned_names) != manifest.get("num_classes"):
                logger.warning(
                    "Model '%s' declares %s classes but has %s class names.",
                    manifest.get("model_id", manifest_path.parent.name),
                    manifest.get("num_classes"),
                    len(cleaned_names),
                )

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


def _weights_signature(manifest: dict) -> tuple:
    """Return the parts of a manifest that determine the loaded checkpoint."""
    weights_subdir = Path(manifest["_weights_subdir"])
    weights_file = manifest.get("weights_file", "")
    weights_path = weights_subdir / weights_file
    try:
        stat = weights_path.stat()
        file_signature = (str(weights_path.resolve()), stat.st_size, stat.st_mtime_ns)
    except FileNotFoundError:
        file_signature = (str(weights_path.resolve()), None, None)
    return (
        manifest.get("arch", "").lower(),
        manifest.get("num_classes"),
        manifest.get("resolution", 640),
        file_signature,
    )


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
    discovered_ids: set[str] = set()
    for model_id, weights_subdir, manifest in _manifest_iter(weights_dir):
        discovered_ids.add(model_id)
        signature = _weights_signature(manifest)
        if model_id in _registry and _loaded_signatures.get(model_id) != signature:
            logger.info(
                "Checkpoint for model '%s' changed on disk; invalidating cached model.",
                model_id,
            )
            del _registry[model_id]
            _loaded_signatures.pop(model_id, None)
        _manifests[model_id] = manifest
        if model_id in _registry:
            _registry[model_id].manifest = manifest
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
                tiling_mode=manifest.get("tiling_mode", "fixed"),
                target_symbol_px=float(manifest.get("target_symbol_px", 48.0)),
                estimated_symbol_px=float(manifest.get("estimated_symbol_px", 48.0)),
                enable_auto_crop=bool(manifest.get("enable_auto_crop", False)),
                enable_scale_norm=bool(manifest.get("enable_scale_norm", False)),
                iou_threshold=manifest.get("iou_threshold", 0.50),
                loaded=model_id in _registry,
                weights_exist=weights_exist,
            )
        )

    for model_id in set(_manifests) - discovered_ids:
        _manifests.pop(model_id, None)
        _registry.pop(model_id, None)
        _loaded_signatures.pop(model_id, None)

    return results


def get_model_info(model_id: str) -> ModelInfo | None:
    """Return ModelInfo for a single model, or None if not found."""
    for info in scan():
        if info.model_id == model_id:
            return info
    return None


def get_or_load(model_id: str) -> BaseModelWrapper:
    """Return a loaded model, serializing concurrent first-load requests."""
    with _registry_lock:
        return _get_or_load(model_id)


def _get_or_load(model_id: str) -> BaseModelWrapper:
    """
    Return the loaded wrapper for *model_id*, loading it on first call.
    Raises KeyError if the model_id is not in any discovered manifest.
    Raises FileNotFoundError if the .pth weight file is missing on disk.
    """
    # Rescan on every load request so a manually replaced checkpoint is
    # detected even when this process has already loaded the model once.
    scan()

    if model_id in _registry:
        return _registry[model_id]

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
    _loaded_signatures[model_id] = _weights_signature(manifest)
    logger.info("Model '%s' loaded and cached.", model_id)
    return wrapper


def update_manifest(
    model_id: str,
    class_names: list[str] | None = None,
    confidence_default: float | None = None,
    tiling_mode: str | None = None,
    target_symbol_px: float | None = None,
    estimated_symbol_px: float | None = None,
    enable_auto_crop: bool | None = None,
    enable_scale_norm: bool | None = None,
) -> ModelInfo:
    """
    Update class_names, confidence_default, and adaptive tiling fields for a model
    — both in the in-memory manifest and on disk.
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
    if tiling_mode is not None:
        manifest["tiling_mode"] = tiling_mode
    if target_symbol_px is not None:
        manifest["target_symbol_px"] = target_symbol_px
    if estimated_symbol_px is not None:
        manifest["estimated_symbol_px"] = estimated_symbol_px
    if enable_auto_crop is not None:
        manifest["enable_auto_crop"] = enable_auto_crop
    if enable_scale_norm is not None:
        manifest["enable_scale_norm"] = enable_scale_norm

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
        tiling_mode=manifest.get("tiling_mode", "fixed"),
        target_symbol_px=float(manifest.get("target_symbol_px", 48.0)),
        estimated_symbol_px=float(manifest.get("estimated_symbol_px", 48.0)),
        enable_auto_crop=bool(manifest.get("enable_auto_crop", False)),
        enable_scale_norm=bool(manifest.get("enable_scale_norm", False)),
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
