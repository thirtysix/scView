"""scView backend – FastAPI application entry point."""

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Depends, FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from scview.config import get_settings
from scview.core.guards import require_access
from scview.api.router import api_router

logger = logging.getLogger(__name__)

# Routes that legitimately accept large bodies (file uploads) — exempt from the
# JSON body-size cap, which still bounds them via MAX_UPLOAD_SIZE_MB.
_UPLOAD_PATH_HINTS = ("upload", "ingest", "convert")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Create required data directories and sweep stale ingest sessions on startup."""
    settings = get_settings()
    data_dir = Path(settings.DATA_DIR)
    for subdir in ("uploads", "converted", "cache"):
        (data_dir / subdir).mkdir(parents=True, exist_ok=True)
    # Remove abandoned ingest staging sessions older than the TTL (24 h).
    from scview.dependencies import get_ingest_session_manager

    get_ingest_session_manager().sweep_expired()

    # Deployment-posture self-check — make an insecure public exposure loud.
    logger.info("scView deployment mode: %s", settings.DEPLOYMENT_MODE)
    for warn in settings.deployment_warnings():
        logger.warning("DEPLOYMENT (public): %s", warn)
    yield


class BodySizeLimitMiddleware(BaseHTTPMiddleware):
    """Reject oversized JSON bodies on non-upload routes before they're parsed."""

    def __init__(self, app, max_bytes: int):
        super().__init__(app)
        self.max_bytes = max_bytes

    async def dispatch(self, request: Request, call_next):
        if not any(h in request.url.path for h in _UPLOAD_PATH_HINTS):
            cl = request.headers.get("content-length")
            if cl and cl.isdigit() and int(cl) > self.max_bytes:
                return JSONResponse(status_code=413, content={"detail": "Request body too large."})
        return await call_next(request)


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
app.add_middleware(BodySizeLimitMiddleware, max_bytes=settings.MAX_JSON_BODY_KB * 1024)

# ---------------------------------------------------------------------------
# Routers — every /api route passes the (optional) shared-secret access gate.
# ---------------------------------------------------------------------------
app.include_router(api_router, dependencies=[Depends(require_access)])


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------
@app.get("/health", tags=["health"])
async def health_check():
    return {"status": "ok"}
