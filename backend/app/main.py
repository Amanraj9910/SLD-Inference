"""
FastAPI application entry point.
"""
from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

from fastapi import FastAPI, Query, Response
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.routers import infer, models

backend_dir = Path(__file__).resolve().parent.parent
log_path = backend_dir / "backend.log"


def configure_logging() -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s | %(levelname)-8s | %(name)s - %(message)s")

    for handler in list(root_logger.handlers):
        if (
            isinstance(handler, logging.FileHandler)
            and not isinstance(handler, RotatingFileHandler)
            and Path(getattr(handler, "baseFilename", "")).resolve() == log_path.resolve()
        ):
            root_logger.removeHandler(handler)
            handler.close()

    file_handler = next(
        (
            handler
            for handler in root_logger.handlers
            if isinstance(handler, RotatingFileHandler)
            and Path(getattr(handler, "baseFilename", "")).resolve() == log_path.resolve()
        ),
        None,
    )
    if file_handler is None:
        file_handler = RotatingFileHandler(
            log_path,
            maxBytes=10 * 1024 * 1024,
            backupCount=3,
            encoding="utf-8",
        )
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)

    if not any(isinstance(handler, logging.StreamHandler) and not isinstance(handler, RotatingFileHandler) for handler in root_logger.handlers):
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        root_logger.addHandler(stream_handler)


configure_logging()

app = FastAPI(
    title="SLD Inference API",
    description="Multi-model D-FINE / RF-DETR inference server for SLD images.",
    version="1.0.0",
)

# ── CORS ────────────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Routers ──────────────────────────────────────────────────────────────────
app.include_router(models.router)
app.include_router(infer.router)


from pathlib import Path

# ── Health check ─────────────────────────────────────────────────────────────
@app.get("/health", tags=["health"])
@app.get("/api/health", tags=["health"])
def health() -> dict:
    return {"status": "ok"}


# ── Server logs ──────────────────────────────────────────────────────────────
@app.get("/logs", tags=["logs"])
@app.get("/api/logs", tags=["logs"])
def get_logs(response: Response, lines: int = Query(default=100, ge=1, le=1000)) -> dict:
    """Return the tail of backend.log for live debugging in the UI."""
    response.headers["Cache-Control"] = "no-store"
    if not log_path.is_file():
        return {"logs": "backend.log not found on server yet."}

    try:
        content = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
        tail_lines = content[-lines:]
        return {"logs": "\n".join(tail_lines)}
    except Exception as exc:
        return {"logs": f"Error reading log file: {exc}"}
