"""AlphaForge dashboard — application entry point.

Run from the repository root with either

    uvicorn app.server:app --port 8501
    alphaforge dashboard            # same thing via the CLI

The server is read-only: it loads the bundled Parquet cache, computes
factor statistics and re-runs headline backtests on demand. It never
writes to disk, so a fresh clone gives a fully working demo offline.
"""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.api import router

STATIC_DIR = Path(__file__).parent / "static"


def create_app() -> FastAPI:
    app = FastAPI(
        title="AlphaForge Dashboard",
        description=(
            "REST API behind the AlphaForge web dashboard: factor "
            "statistics, live vectorised backtests, out-of-sample ML "
            "diagnostics and risk analytics on real US equity data."
        ),
        version="1.0.0",
    )
    app.include_router(router)
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    @app.get("/", include_in_schema=False)
    def index() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html")

    return app


app = create_app()
