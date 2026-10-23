"""Memory and time estimates before a load or a run (``core/preflight.py``).

Both read file headers only, in a worker thread, and never refuse anything: the
verdict is a warning, and the user decides.
"""

from __future__ import annotations

import asyncio
import logging

from fastapi import APIRouter, Depends, HTTPException, Query

from scview.core import preflight
from scview.core.dataset_manager import DatasetManager
from scview.core.ingestion.session import IngestSessionManager
from scview.core.pipeline import ALL_STEPS
from scview.dependencies import get_dataset_manager, get_ingest_session_manager

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/ingest/session/{sid}/preflight")
async def ingest_preflight(
    sid: str,
    mgr: IngestSessionManager = Depends(get_ingest_session_manager),
    dm: DatasetManager = Depends(get_dataset_manager),
):
    """For each sample staged in the session: its header (cells, genes, storage,
    layers), the memory it takes once loaded and at peak while it is read and
    saved, the rough time, and a verdict against the memory scView has free
    (``fits`` / ``tight`` / ``wont_fit``, with suggestions). Several samples get
    a ``merge`` entry for the commit as a whole."""
    if not mgr.exists(sid):
        raise HTTPException(status_code=404, detail="Ingest session not found.")
    return await asyncio.to_thread(preflight.session_preflight, mgr, sid, dm)


@router.get("/datasets/{dataset_id}/preflight")
async def dataset_preflight(
    dataset_id: str,
    steps: str = Query(..., description="Comma-separated pipeline step names"),
    n_top_genes: int | None = Query(default=None, ge=1),
    n_comps: int | None = Query(default=None, ge=1),
    n_neighbors: int | None = Query(default=None, ge=1),
    normalization_method: str | None = None,
    annotation_method: str | None = None,
    doublet_channel_key: str | None = None,
    dm: DatasetManager = Depends(get_dataset_manager),
):
    """Peak memory and rough time of running ``steps`` on the dataset: per step
    (transient memory over what it holds, seconds as a range) and for the run,
    with a verdict against the memory scView has free. The shape comes from the
    cached dataset, else from its file's header (the dataset is not loaded)."""
    names = [s.strip() for s in steps.split(",") if s.strip()]
    unknown = [s for s in names if s not in ALL_STEPS]
    if not names or unknown:
        raise HTTPException(
            status_code=422,
            detail=(f"Unknown step{'s' if len(unknown) != 1 else ''}: {', '.join(unknown)}. "
                    if unknown else "No steps given. ")
            + "Valid steps: " + ", ".join(ALL_STEPS) + ".",
        )
    if dm.get_dataset_info(dataset_id) is None:
        raise HTTPException(status_code=404, detail=f"Dataset '{dataset_id}' not found.")
    overrides = {k: v for k, v in {
        "n_top_genes": n_top_genes, "n_comps": n_comps, "n_neighbors": n_neighbors,
        "normalization_method": normalization_method, "annotation_method": annotation_method,
        "doublet_channel_key": doublet_channel_key,  # its largest channel sets the peak
    }.items() if v is not None}
    try:
        result = await asyncio.to_thread(preflight.dataset_preflight, dm, dataset_id, names,
                                         overrides)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Dataset '{dataset_id}' has no data file.")
    return result
