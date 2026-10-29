"""Upstream TF candidates from promoter motifs (``core/tf_motifs.py``).

``GET /tf-motifs/status`` says which motif tables are built (and, with ``dataset_id``,
which one the dataset needs); ``POST /tf-motifs/prepare`` downloads the inputs and
builds a table in a child process, streamed (SSE); the build carries on when the
client goes away, and a second request follows the running one.
``POST /datasets/{id}/tf-motifs/run`` tests each group's markers (409 while the table
is not built) and records the run (provenance step ``tf_motif_enrichment``, which
only adds a result); ``GET /datasets/{id}/tf-motifs`` gives the last run, or 404.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from scview.config import Settings
from scview.core import large_mode
from scview.core import tf_motifs as tm
from scview.core.dataset_manager import DatasetManager
from scview.core.obs_writes import dataset_edit, save_edit, uns_snapshot
from scview.core.species import detect_species
from scview.core.sse import keepalive
from scview.dependencies import get_dataset_manager, get_settings_dep

logger = logging.getLogger(__name__)

router = APIRouter()

MAX_GROUPS = 100
POLL_SECONDS = 0.5
ACTIVITY_HINT = ("Score TF activity once (Activity > TFs (CollecTRI)) and run again to see "
                 "whether each candidate's activity agrees.")


class PrepareRequest(BaseModel):
    species: Literal["human", "mouse"] = "human"
    assembly: Literal["hg38", "mm10", "mm39"] | None = None


class RunRequest(BaseModel):
    column: str = Field(min_length=1, max_length=256)
    n_markers: int = Field(default=tm.DEFAULT_MARKERS, ge=10, le=500)


def _dest(settings: Settings) -> Path:
    return Path(settings.DATA_DIR) / "tf_motifs"  # the data volume


def _sse(event: str, data: dict[str, Any]) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


async def _adaptor(dm: DatasetManager, dataset_id: str):
    adaptor = await dm.get_or_load_dataset(dataset_id)
    if adaptor is None:
        raise HTTPException(status_code=404, detail=f"Dataset '{dataset_id}' not found.")
    return adaptor


def _assembly_status(dest: Path, a: tm.Assembly) -> dict:
    meta = tm.read_meta(dest, a.name)
    build = tm.current_build(a.name)
    return {
        "assembly": a.name, "species": a.species, "label": a.label, "prepared": meta is not None,
        "estimate_seconds": a.estimate_seconds, "building": build is not None,
        "progress": build.last if build is not None else None,
        "files": [{"name": s.name, "url": s.url, "sha256": s.sha256, "bytes": s.size,
                   "downloaded": (tm.raw_dir(dest) / s.filename).is_file()} for s in tm.sources_for(a.name)],
        **({"meta": meta} if meta else {}),
    }


@router.get("/tf-motifs/status")
async def status(
    dataset_id: str = "",
    dm: DatasetManager = Depends(get_dataset_manager),
    settings: Settings = Depends(get_settings_dep),
):
    """Which motif tables are built; with ``dataset_id``, the one that dataset needs."""
    dest = _dest(settings)
    assemblies = await asyncio.to_thread(lambda: [_assembly_status(dest, a) for a in tm.ASSEMBLIES.values()])
    body: dict[str, Any] = {
        "assemblies": assemblies, "auto_download": settings.TF_MOTIFS_AUTO_DOWNLOAD,
        "citation": tm.CITATION, "licence": tm.LICENCE, "caveats": tm.CAVEATS,
        "p_threshold": tm.P_THRESHOLD, "promoter_bp": tm.PROMOTER_LEN,
    }
    if dataset_id:
        adata = (await _adaptor(dm, dataset_id)).adata
        species = detect_species(adata).species
        asm = tm.assembly_for(adata, species)
        body["dataset"] = {"species": species, "assembly": asm,
                           "prepared": any(a["prepared"] for a in assemblies if a["assembly"] == asm)}
    return body


@router.post("/tf-motifs/prepare")
async def prepare(body: PrepareRequest, settings: Settings = Depends(get_settings_dep)):
    """Download the inputs and build the motif table (minutes, in a child process at 2
    threads). Events: ``progress`` (stage, message, done / total), ``result`` (the
    table's summary), ``complete``; or ``error``."""
    asm = body.assembly or tm.DEFAULT_ASSEMBLY[body.species]
    if tm.ASSEMBLIES[asm].species != body.species:
        raise HTTPException(status_code=422, detail=f"{asm} is not a {body.species} assembly.")
    dest = _dest(settings)

    async def events():
        meta = await asyncio.to_thread(tm.read_meta, dest, asm)
        if meta is None:
            build = tm.start_build(dest, asm, auto_download=settings.TF_MOTIFS_AUTO_DOWNLOAD)
            yield _sse("start", {"assembly": asm, "estimate_seconds": tm.ASSEMBLIES[asm].estimate_seconds,
                                 "running_for": round(time.time() - build.started, 1)})
            sent = 0
            while True:
                done = build.done  # read before the events, so none is missed
                for e in build.events[sent:]:
                    yield _sse("progress", e)
                sent = len(build.events)
                if done:
                    break
                await asyncio.sleep(POLL_SECONDS)
            if build.error:
                yield _sse("error", {"error": build.error})
                return
            meta = await asyncio.to_thread(tm.read_meta, dest, asm)
            if meta is None:
                yield _sse("error", {"error": "The motif table was built but cannot be read; try again."})
                return
        yield _sse("result", {"assembly": asm, "meta": meta})
        yield _sse("complete", {})

    return StreamingResponse(keepalive(events()), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


def _activity(adata, column: str, settings: Settings, species: str) -> tuple[dict | None, str | None]:
    """CollecTRI group-vs-rest ranks for the agreement column, when the network is on disk
    (never fetched from here); (None, why) otherwise."""
    from scview.api.v1 import activity as activity_api
    from scview.core import activity_nets

    net_dir = Path(settings.DATA_DIR) / "networks"
    if activity_nets.read_meta(net_dir, "collectri") is None or (
            species == "mouse" and activity_nets.read_mouse_meta(net_dir, "collectri") is None):
        return None, ACTIVITY_HINT
    try:
        model, _ = activity_api._model(adata, "collectri", net_dir, species)
        groups, codes = activity_api._group_codes(adata.obs[column])
        means, sizes = model.group_means(codes, len(groups))
    except Exception as e:  # noqa: BLE001 - the agreement column is optional
        logger.info("TF motifs: no CollecTRI activity for the agreement column: %s", e)
        return None, ACTIVITY_HINT
    return tm.activity_ranks(groups, model.net.sources, means, sizes), None


@router.post("/datasets/{dataset_id}/tf-motifs/run")
async def run(
    dataset_id: str,
    body: RunRequest,
    dm: DatasetManager = Depends(get_dataset_manager),
    settings: Settings = Depends(get_settings_dep),
):
    """Each group's marker genes against the promoter motif table: candidates per group."""
    from scview.core import activity, marker_selection

    dest = _dest(settings)
    async with dataset_edit(dataset_id) as held:
        adaptor = await _adaptor(dm, dataset_id)
        large_mode.check(adaptor, "Upstream TF candidates", large_mode.WHOLE_MATRIX)
        adata = adaptor.adata
        if body.column not in adata.obs.columns:
            raise HTTPException(status_code=404, detail=f"Column '{body.column}' not found.")
        n_groups = adata.obs[body.column].nunique(dropna=True)
        if not 1 <= n_groups <= MAX_GROUPS:
            raise HTTPException(status_code=422,
                                detail=f"'{body.column}' has {n_groups} groups; at most {MAX_GROUPS}.")
        species = detect_species(adata).species
        asm = tm.assembly_for(adata, species)
        if asm is None:
            raise HTTPException(status_code=422, detail=(
                "The species of this dataset's genes is unknown: promoter motifs need human or "
                "mouse gene symbols."))
        table = await asyncio.to_thread(tm.load_table, dest, asm)
        if table is None:
            raise HTTPException(status_code=409, detail=(
                f"The promoter motif table for {tm.ASSEMBLIES[asm].label} is not built yet: "
                "press Prepare first (a one-off download and scan of a few minutes)."))
        output_path = str(dm.derived_h5ad_path(dataset_id, adaptor.h5ad_path))

        def _run() -> dict:
            t0 = time.time()
            markers = marker_selection.group_markers(adata, body.column, n=body.n_markers)
            if not markers:
                raise ValueError(f"No marker genes are stored for '{body.column}': compute marker "
                                 "genes for it first (Markers tab).")
            M, genes, source = activity.lognorm_matrix(adata)
            measured = activity.measured_genes(M)
            universe = [str(g) for g, ok in zip(genes, measured) if ok]
            res = tm.enrich(table, universe, markers)
            tfs = {t for g in res["groups"] for r in g["rows"] for t in r["tfs"]}
            stats = tm.marker_stats(adata, body.column, tfs, M, genes)
            act, hint = _activity(adata, body.column, settings, species)
            tm.annotate(res, stats, act)
            counts = adata.obs[body.column].astype(str).value_counts()
            for g in res["groups"]:
                g["n_cells"] = int(counts.get(g["group"], 0))
            result = {
                "result_id": uuid.uuid4().hex[:16], "column": body.column, "assembly": asm,
                "species": species, "n_markers_requested": body.n_markers, **res,
                "p_threshold": tm.P_THRESHOLD, "pseudocount": tm.PSEUDOCOUNT,
                "min_overlap": tm.MIN_OVERLAP, "promoter_bp": tm.PROMOTER_LEN,
                "activity": {"available": act is not None, "hint": hint, "network": "CollecTRI"},
                "matrix_source": source, "citation": tm.CITATION, "licence": tm.LICENCE,
                "caveats": tm.CAVEATS, "table": {k: table.meta.get(k) for k in ("n_genes", "n_motifs", "hit_rate")},
                "elapsed_seconds": round(time.time() - t0, 2),
                "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            }
            before = uns_snapshot(adata)
            tm.record(adata, result)
            save_edit(adaptor, adata, output_path, before)
            return result

        try:
            result = await held.run(_run)
        except ValueError as e:
            raise HTTPException(status_code=422, detail=str(e))
        except Exception as e:
            dm.evict(dataset_id)  # the cached copy may hold the step without it being saved
            logger.error("TF motif enrichment failed for %s: %s", dataset_id, e, exc_info=True)
            raise HTTPException(status_code=500, detail=f"Could not test the promoter motifs: {e!s}")
        dm.saved_in_place(dataset_id, adaptor, output_path)
        tm.remember(adaptor, result)  # after the step is recorded: stored as current
        return result


@router.get("/datasets/{dataset_id}/tf-motifs")
async def last_run(dataset_id: str, allow_empty: bool = False,
                   dm: DatasetManager = Depends(get_dataset_manager)):
    """The last run on this dataset, or 404; with ``allow_empty`` null instead (the panel asks on open,
    and a 404 is a console error in the browser)."""
    adaptor = await _adaptor(dm, dataset_id)
    r = await asyncio.to_thread(tm.last, adaptor)
    if r is None:
        if allow_empty:
            return None
        raise HTTPException(status_code=404, detail="No upstream TF candidates computed for this dataset yet.")
    return r
