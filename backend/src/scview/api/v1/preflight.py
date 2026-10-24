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
    chunked: str | None = None,
    clustering_method: str | None = None,
    leiden_iterations: int | None = None,
    umap_3d: str | None = None,
    regress_out: str | None = Query(default=None, description="Comma-separated obs columns"),
    ambient_rho: float | None = Query(default=None, gt=0, lt=1),
    dm: DatasetManager = Depends(get_dataset_manager),
):
    """Peak memory and rough time of running ``steps`` on the dataset: per step
    (transient memory over what it holds, seconds as a range) and for the run,
    with a verdict against the memory scView has free. The shape comes from the
    cached dataset, else from its file's header (the dataset is not loaded). ``chunked``
    (auto | on | off, as the run's param): ``mode`` says whether the run would go to
    large-dataset mode, whose estimate (with the disk it writes) then replaces the
    in-memory one (kept as ``in_memory_estimate``). A run that clusters with Leiden gets
    ``leiden``: its time with the full optimisation and with fast Leiden
    (``leiden_iterations`` 2), and ``offer_fast`` when the full one is over 10 minutes.
    A run with the UMAP step gets ``umap_3d``: whether it makes the 3D UMAP (``umap_3d``
    auto | on | off; auto leaves it out in large-dataset mode) and the time it adds.
    ``regress_out`` (comma-separated, the scaling step's): the fit's time and memory."""
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
        "chunked": chunked,
        "clustering_method": clustering_method, "leiden_iterations": leiden_iterations,
        "umap_3d": umap_3d,
        "regress_out": None if regress_out is None else [k for k in regress_out.split(",") if k.strip()],
        "ambient_rho": ambient_rho,  # set: no estimate (and no clustering of its own)
    }.items() if v is not None}
    try:
        if chunked is not None:
            from scview.core.pipeline import check_chunked

            check_chunked(chunked)
        if leiden_iterations is not None:
            from scview.core.leiden import check_iterations

            check_iterations(leiden_iterations)
        if umap_3d is not None:
            from scview.core.pipeline import check_umap_3d

            check_umap_3d(umap_3d)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    try:
        result = await asyncio.to_thread(preflight.dataset_preflight, dm, dataset_id, names,
                                         overrides)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Dataset '{dataset_id}' has no data file.")
    return result
