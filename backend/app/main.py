"""
FastAPI application entry point.
"""
from __future__ import annotations

import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.routers import infer, models

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s — %(message)s",
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


# ── Health check ─────────────────────────────────────────────────────────────
@app.get("/health", tags=["health"])
def health() -> dict:
    return {"status": "ok"}
