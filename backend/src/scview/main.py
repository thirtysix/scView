"""scView backend – FastAPI application entry point."""

import logging
import threading
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Depends, FastAPI, Request
from fastapi.exception_handlers import http_exception_handler
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.middleware.base import BaseHTTPMiddleware

from scview.config import get_settings
from scview.core import large_mode
from scview.core.dataset_manager import DatasetBusyError, DatasetMemoryError
from scview.core.large_mode import LargeDatasetError
from scview.core.guards import require_access
from scview.api.router import api_router

logger = logging.getLogger(__name__)

# uvicorn configures only its own loggers, so without a handler here the app's
# INFO lines (e.g. MSigDB download progress) never reach `docker logs`. Scoped
# to the scview namespace to keep third-party INFO chatter out.
_app_log = logging.getLogger("scview")
if not _app_log.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter("%(levelname)s:     %(name)s - %(message)s"))
    _app_log.addHandler(_handler)
    _app_log.setLevel(logging.INFO)

# Routes that legitimately accept large bodies (file uploads) — exempt from the
# JSON body-size cap, which still bounds them via MAX_UPLOAD_SIZE_MB.
_UPLOAD_PATH_HINTS = ("upload", "ingest", "convert", "seq-metrics", "velocity/attach", "atac/fragments", "atac/metrics")  # seq-metrics: raw matrices


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

    # Fetch MSigDB gene sets in the background on first start (~45 MB), so
    # startup isn't held up and an offline machine still comes up.
    if settings.MSIGDB_DIR and settings.MSIGDB_AUTO_DOWNLOAD:
        from scview.core.msigdb_fetch import ensure_msigdb

        threading.Thread(
            target=ensure_msigdb, args=(Path(settings.MSIGDB_DIR),), daemon=True
        ).start()

    # Deployment-posture self-check — make an insecure public exposure loud.
    logger.info("scView deployment mode: %s", settings.DEPLOYMENT_MODE)
    for warn in settings.deployment_warnings():
        logger.warning("DEPLOYMENT (public): %s", warn)

    from scview.core.memory_budget import mark_running, mark_stopped

    if mark_running():
        logger.warning("The previous backend stopped without shutting down (killed, most often for memory).")
    yield
    mark_stopped()


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


@app.exception_handler(DatasetBusyError)
async def dataset_busy_handler(_request: Request, exc: DatasetBusyError):
    return JSONResponse(
        status_code=503,
        content={"detail": "This dataset is being processed. Try again when the run has finished."},
        headers={"Retry-After": "30"},
    )


@app.exception_handler(DatasetMemoryError)
async def dataset_memory_handler(_request: Request, exc: DatasetMemoryError):
    # 507: the request may work once memory is raised; not a proxy's 502/503/504.
    return JSONResponse(status_code=507, content={"detail": str(exc)})


def _large_refusal(exc: LargeDatasetError) -> JSONResponse:
    # 422: work refused on a large dataset (its matrices stay on disk), with the reason;
    # the UI shows it as a note in the panel, not as an error.
    return JSONResponse(status_code=422, content={"detail": str(exc), "large_dataset": True})


@app.exception_handler(LargeDatasetError)
async def large_dataset_handler(_request: Request, exc: LargeDatasetError):
    return _large_refusal(exc)


@app.exception_handler(StarletteHTTPException)
async def http_error_handler(request: Request, exc: StarletteHTTPException):
    # Many endpoints turn any failure into a 500 ("... failed: {e}"). When the failure
    # was a large-dataset refusal (the guard on its on-disk matrix, core/large_mode.py),
    # the refusal is what the client gets: a 422 with the plain reason.
    if exc.status_code >= 500:
        refusal = large_mode.refusal_in(exc)
        if refusal is not None:
            return _large_refusal(refusal)
    return await http_exception_handler(request, exc)


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------
@app.get("/health", tags=["health"])
async def health_check():
    return {"status": "ok"}
