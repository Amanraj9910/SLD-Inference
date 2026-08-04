"""
FastAPI application entry point.
"""
from __future__ import annotations

import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.routers import infer, models

from pathlib import Path

backend_dir = Path(__file__).resolve().parent.parent
log_path = backend_dir / "backend.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s — %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(log_path, mode="a", encoding="utf-8"),
    ],
)

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
def get_logs(lines: int = 100) -> dict:
    """Return the tail of backend.log for live debugging in the UI."""
    backend_dir = Path(__file__).resolve().parent.parent
    possible_paths = [
        backend_dir / "backend.log",
        Path("/opt/SLD-Inference/backend/backend.log"),
    ]
    
    log_file = None
    for p in possible_paths:
        if p.is_file():
            log_file = p
            break

    if log_file is None:
        return {"logs": "backend.log not found on server yet."}

    try:
        content = log_file.read_text(encoding="utf-8", errors="ignore").splitlines()
        tail_lines = content[-lines:] if len(content) > lines else content
        return {"logs": "\n".join(tail_lines)}
    except Exception as exc:
        return {"logs": f"Error reading log file: {exc}"}

