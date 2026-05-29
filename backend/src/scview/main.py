"""scView backend – FastAPI application entry point."""

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware

from scview.config import get_settings
from scview.api.router import api_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Create required data directories on startup."""
    settings = get_settings()
    data_dir = Path(settings.DATA_DIR)
    for subdir in ("uploads", "converted", "cache"):
        (data_dir / subdir).mkdir(parents=True, exist_ok=True)
    yield


app = FastAPI(
    title="scView",
    description="Single-cell RNA-seq visualisation backend",
    version="0.1.0",
    lifespan=lifespan,
)

# ---------------------------------------------------------------------------
# Middleware
# ---------------------------------------------------------------------------
settings = get_settings()

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(GZipMiddleware, minimum_size=1000)

# ---------------------------------------------------------------------------
# Routers
# ---------------------------------------------------------------------------
app.include_router(api_router)


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------
@app.get("/health", tags=["health"])
async def health_check():
    return {"status": "ok"}
