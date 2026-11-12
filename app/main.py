from __future__ import annotations

import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI

from app.api.routes import router
from app.config import CROPS_DIR, LOG_LEVEL, MODELS_DIR, ORIGINALS_DIR
from app.logging_config import setup_logging


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Setup logging
    setup_logging(LOG_LEVEL)

    # Create data directories if they don't exist
    for directory in [ORIGINALS_DIR, CROPS_DIR, MODELS_DIR]:
        Path(directory).mkdir(parents=True, exist_ok=True)

    yield


app = FastAPI(
    title="Voucher OCR API",
    version="1.0.0",
    lifespan=lifespan,
)

app.include_router(router)


@app.get("/health")
async def health_check():
    return {"status": "ok", "version": "1.0.0"}
