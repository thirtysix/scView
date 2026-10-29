"""GWAS trait links (``core/gwas_catalog.py``): the pinned GWAS Catalog release.

``GET /gwas/status`` says whether it is on disk; ``POST /gwas/download`` fetches and
prepares it (SSE progress); ``GET /gwas/gene/{symbol}`` one gene's traits;
``POST /datasets/{id}/gwas/clusters`` the trait overlap of each group's markers
(kept in the results store, a History step that adds no data), and ``GET`` the
last one. Nothing is sent to a third party except the pinned downloads.
"""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from scview.config import Settings
from scview.core import gwas_catalog as gc
from scview.core.orthologs import OrthologsUnavailable
from scview.core.dataset_manager import DatasetManager
from scview.core.obs_writes import dataset_edit, save_edit, uns_snapshot
from scview.core.species import detect_species
from scview.core.sse import keepalive
from scview.dependencies import get_dataset_manager, get_settings_dep

logger = logging.getLogger(__name__)

router = APIRouter()


class ClustersRequest(BaseModel):
    column: str = Field(min_length=1, max_length=256)
    include_measurements: bool = False
    n_markers: int = Field(default=gc.N_MARKERS, ge=gc.MIN_QUERY, le=500)
    species: str = Field(default="", pattern="^(|human|mouse)$")


def _dest(settings: Settings) -> Path:
    return Path(settings.DATA_DIR) / "gwas"


def _status(settings: Settings) -> dict:
    dest = _dest(settings)
    meta = gc.read_meta(dest)
    return {
        "release": gc.RELEASE, "downloaded": meta is not None, "raw_present": gc.raw_present(dest),
        "auto_download": gc.auto_download(), "n_traits": (meta or {}).get("n_traits"),
        "n_genes": (meta or {}).get("n_genes"), "n_loci": (meta or {}).get("n_loci"),
        "built_at": (meta or {}).get("built_at"),
        "download_mb": round((gc.ASSOC_BYTES + gc.MAPPINGS_BYTES) / 1e6),
        "assoc_url": gc.ASSOC_URL, "mappings_url": gc.MAPPINGS_URL,
        "attribution": gc.ATTRIBUTION, "citation": gc.CITATION, "licence": gc.LICENCE, "caveat": gc.CAVEAT,
    }


@router.get("/gwas/status")
async def gwas_status(settings: Settings = Depends(get_settings_dep)):
    """Whether the GWAS Catalog release is on disk, and what it holds."""
    return _status(settings)


@router.post("/gwas/download")
async def gwas_download(settings: Settings = Depends(get_settings_dep)):
    """Fetch (or take the files placed by hand) and prepare the catalog, as SSE:
    ``progress`` events, then ``complete`` (the status) or ``error``."""
    dest = _dest(settings)
    loop = asyncio.get_running_loop()
    queue: asyncio.Queue = asyncio.Queue()

    def _progress(event: dict) -> None:
        loop.call_soon_threadsafe(queue.put_nowait, ("progress", event))

    def _work() -> None:
        try:
            gc.ensure(dest, allow_download=gc.auto_download(), progress=_progress)
            loop.call_soon_threadsafe(queue.put_nowait, ("complete", _status(settings)))
        except Exception as e:  # GwasUnavailable carries the message to show
            logger.warning("GWAS Catalog download failed: %s", e)
            loop.call_soon_threadsafe(queue.put_nowait, ("error", {"error": str(e)}))

    async def events():
        task = asyncio.ensure_future(asyncio.to_thread(_work))
        try:
            while True:
                kind, data = await queue.get()
                yield f"event: {kind}\ndata: {json.dumps(data)}\n\n"
                if kind in ("complete", "error"):
                    break
        finally:
            if task.done():
                task.result()
            # else: the download goes on in its thread; a later request finds it ready.

    return StreamingResponse(keepalive(events()), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


async def _catalog(settings: Settings) -> gc.Catalog:
    """The built catalog (read from disk off the event loop the first time), or 409."""
    try:
        return await asyncio.to_thread(gc.load, _dest(settings))
    except gc.GwasUnavailable as e:
        raise HTTPException(status_code=409, detail=str(e))


@router.get("/gwas/gene/{symbol}")
async def gwas_gene(symbol: str, species: str = "", settings: Settings = Depends(get_settings_dep)):
    """The traits the catalog maps one gene for, smallest p first (a mouse symbol
    via its human ortholog with ``species=mouse``)."""
    if len(symbol) > 64:
        raise HTTPException(status_code=422, detail="Gene symbol too long.")
    if species not in ("", "human", "mouse"):
        raise HTTPException(status_code=422, detail="species must be human or mouse.")
    cat = await _catalog(settings)

    def _run() -> dict | None:
        names = [symbol]
        if species == "mouse":
            from scview.core import orthologs

            names = orthologs.mouse_to_human([symbol]).get(symbol) or [symbol]
        for name in names:
            got = gc.gene_traits(cat, name)
            if got is not None:
                got["query"] = symbol
                got["via_ortholog"] = species == "mouse"
                return got
        return None

    try:
        got = await asyncio.to_thread(_run)
    except OrthologsUnavailable as e:
        raise HTTPException(status_code=503, detail=str(e))
    if got is None:
        raise HTTPException(status_code=404, detail=f"The GWAS Catalog ({gc.RELEASE}) maps no genome-wide "
                                                    f"significant association to {symbol}.")
    return got


@router.post("/datasets/{dataset_id}/gwas/clusters")
async def gwas_clusters(
    dataset_id: str,
    body: ClustersRequest,
    dm: DatasetManager = Depends(get_dataset_manager),
    settings: Settings = Depends(get_settings_dep),
):
    """The trait overlap of each group's markers; a History step, kept for GET."""
    cat = await _catalog(settings)
    async with dataset_edit(dataset_id) as held:
        adaptor = await dm.get_or_load_dataset(dataset_id)
        if adaptor is None:
            raise HTTPException(status_code=404, detail=f"Dataset '{dataset_id}' not found.")
        adata = adaptor.adata
        if body.column not in adata.obs.columns:
            raise HTTPException(status_code=422, detail=f"Column '{body.column}' not found.")
        output_path = str(dm.derived_h5ad_path(dataset_id, adaptor.h5ad_path))

        def _run() -> dict:
            species = body.species
            if not species:
                call = detect_species(adata)
                if call.species not in ("human", "mouse"):
                    raise ValueError(f"Could not tell whether this dataset is human or mouse ({call.evidence}).")
                species = call.species
            result = gc.run(cat, adata, body.column, include_measurements=body.include_measurements,
                            n_markers=body.n_markers, species=species, dataset_id=dataset_id)
            before = uns_snapshot(adata)
            gc.record(adata, result)
            save_edit(adaptor, adata, output_path, before)
            return result

        try:
            result = await held.run(_run)
        except (ValueError, KeyError) as e:
            raise HTTPException(status_code=422, detail=str(e).strip("'\""))
        except OrthologsUnavailable as e:
            raise HTTPException(status_code=503, detail=str(e))
        except Exception as e:
            dm.evict(dataset_id)  # the cached copy may hold the step without it being saved
            logger.error("GWAS trait links failed for %s: %s", dataset_id, e, exc_info=True)
            raise HTTPException(status_code=500, detail=f"Could not compute GWAS trait links: {e!s}")
        dm.saved_in_place(dataset_id, adaptor, output_path)
        gc.remember(adaptor, result)  # after the step is recorded: stored as current
        return result


@router.get("/datasets/{dataset_id}/gwas/clusters")
async def last_gwas_clusters(dataset_id: str, column: str | None = None,
                             dm: DatasetManager = Depends(get_dataset_manager)):
    """The last run on this dataset (on ``column``, when given), or 404."""
    adaptor = await dm.get_or_load_dataset(dataset_id)
    if adaptor is None:
        raise HTTPException(status_code=404, detail=f"Dataset '{dataset_id}' not found.")
    r = gc.last(adaptor, column)
    if r is None:
        raise HTTPException(status_code=404, detail="No GWAS trait links computed for this dataset yet.")
    return r
