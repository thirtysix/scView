"""RNA velocity endpoints (``core/velocity_input.py``, ``core/velocity.py``).

``GET /datasets/{id}/velocity``: whether spliced and unspliced counts are attached, what
matched, the last run and its estimate (never a 404: the panel asks on open). ``POST
.../velocity/attach``: a velocyto .loom, an .h5ad with the layers, or a STARsolo
Velocyto/filtered folder's files, stored beside the originals (``uploads/{id}/velocity/``,
read-only) and matched to the cells. ``POST .../velocity/run-stream``: the run (SSE: ``start``,
``progress``, ``watchdog``, ``result``, ``complete`` or ``error``). ``GET .../velocity/grid``,
``.../phase`` and ``.../genes`` read the saved run.

A run reads the store and copies obsm and the graph out first; it never reads X, and holds the
edit lock only for the final save, so a large dataset runs too. A client that goes away stops
it after the block it is on, and nothing is saved.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import re
import threading
import uuid
from pathlib import Path
from typing import Any, Literal

from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from scview.core import preflight
from scview.core import velocity as vel
from scview.core import velocity_input as vi
from scview.core.dataset_manager import DatasetManager
from scview.core.obs_writes import dataset_edit, edit_room, editable, save_edit, uns_snapshot
from scview.core.run_watchdog import RunWatchdog
from scview.core.sse import keepalive
from scview.dependencies import get_dataset_manager

logger = logging.getLogger(__name__)

router = APIRouter()

_DONE = object()
_running: set[str] = set()  # datasets with a run streaming, or counts being attached: one at a time each
_SAFE = re.compile(r"[^A-Za-z0-9._-]+")
_WHAT = "RNA velocity"


class RunRequest(BaseModel):
    method: Literal["stochastic", "steady_state"] = "stochastic"
    n_top_genes: int = Field(default=vel.N_TOP_GENES, ge=100, le=20_000)
    n_neighbors: int = Field(default=vel.N_NEIGHBORS, ge=5, le=100)
    group_by: str | None = Field(default=None, max_length=256)


def _sse(event: str, data: dict[str, Any]) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


async def _adaptor(dm: DatasetManager, dataset_id: str):
    adaptor = await dm.get_or_load_dataset(dataset_id)
    if adaptor is None:
        raise HTTPException(status_code=404, detail="Dataset not found.")
    return adaptor


async def _read(fn, *args):
    try:
        return await asyncio.to_thread(fn, *args)
    except vel.VelocityError as e:
        raise HTTPException(status_code=e.status, detail=str(e))


def _estimate(dm: DatasetManager, dataset_id: str, adaptor, *, method: str = "stochastic",
              n_top_genes: int = vel.N_TOP_GENES, n_neighbors: int = vel.N_NEIGHBORS) -> dict[str, Any] | None:
    """The run's memory and time, from the store's header; None with no counts yet."""
    if not vi.has_store(dm.data_dir, dataset_id):
        return None
    n, n_genes, nnz_s, nnz_u = vel.stored_values(dm.data_dir, dataset_id)
    adata = adaptor.adata
    conn, _dist, _k, _why = vel._stored_neighbours(adata, n_neighbors, _all_cells(adata))
    return preflight.estimate_velocity(int(adata.n_obs), n, nnz_s, nnz_u, n_top_genes=n_top_genes, n_neighbors=n_neighbors,
                                       method=method, has_graph=conn is not None, has_pca="X_pca" in adata.obsm,
                                       n_store_genes=n_genes)


def _all_cells(adata):
    import numpy as np

    return np.arange(adata.n_obs)


@router.get("/datasets/{dataset_id}/velocity")
async def get_velocity(dataset_id: str, dm: DatasetManager = Depends(get_dataset_manager)):
    """Status: what is attached, what matched, the last run, the estimate for the defaults."""
    adaptor = await _adaptor(dm, dataset_id)

    def work() -> dict[str, Any]:
        return {**vel.status(adaptor, dm.data_dir, dataset_id), "estimate": _estimate(dm, dataset_id, adaptor),
                "running": dataset_id in _running}

    return await _read(work)


async def _store_uploads(dm: DatasetManager, dataset_id: str, files: list[UploadFile]) -> list[tuple[Path, str, bool]]:
    """Each upload streamed into ``uploads/{id}/velocity/`` (hashed name, read-only):
    (path, original name, newly written)."""
    folder = Path(dm.data_dir) / "uploads" / dataset_id / "velocity"
    folder.mkdir(parents=True, exist_ok=True)
    out: list[tuple[Path, str, bool]] = []
    for file in files:
        name = Path(file.filename or "velocity").name
        partial = folder / f".upload-{os.getpid()}-{id(file)}.partial"
        digest = hashlib.sha256()
        total = 0
        try:
            with open(partial, "wb") as fh:
                while chunk := await file.read(1024 * 1024):
                    total += len(chunk)
                    if total > vi.MAX_UPLOAD_BYTES:
                        raise HTTPException(status_code=413, detail=(
                            f"{name} exceeds the {vi.MAX_UPLOAD_BYTES // 1024**3} GB limit for velocity files."))
                    digest.update(chunk)
                    fh.write(chunk)
        except BaseException:
            partial.unlink(missing_ok=True)
            raise
        dest = folder / f"{digest.hexdigest()[:12]}_{_SAFE.sub('_', name).strip('_') or 'velocity'}"
        if dest.exists():
            partial.unlink(missing_ok=True)
            out.append((dest, name, False))
            continue
        os.replace(partial, dest)
        try:
            dest.chmod(0o444)
        except OSError:
            pass
        out.append((dest, name, True))
    return out


@router.post("/datasets/{dataset_id}/velocity/attach")
async def attach_velocity(dataset_id: str, files: list[UploadFile] = File(...),
                          dm: DatasetManager = Depends(get_dataset_manager)):
    """Attach spliced and unspliced counts. Answers with what matched (cells and genes); a file
    that does not match a clear majority of the cells is refused, and nothing changes. Attaching
    replaces earlier counts and drops an earlier run."""
    await _adaptor(dm, dataset_id)
    if dataset_id in _running:
        raise HTTPException(status_code=409, detail="A velocity run is working on this dataset: wait for it or cancel it.")
    _running.add(dataset_id)  # no run starts on counts that are being replaced
    try:
        return await _attach(dm, dataset_id, files)
    finally:
        _running.discard(dataset_id)


async def _attach(dm: DatasetManager, dataset_id: str, files: list[UploadFile]) -> dict[str, Any]:
    stored = await _store_uploads(dm, dataset_id, files)
    paths = [p for p, _n, _new in stored]

    async with dataset_edit(dataset_id) as held:
        adaptor = await _adaptor(dm, dataset_id)
        output_path = str(dm.derived_h5ad_path(dataset_id, adaptor.h5ad_path))
        backed = await edit_room(dm, dataset_id, adaptor)

        def work() -> dict[str, Any]:
            data = editable(adaptor)
            before = uns_snapshot(data)
            old_keys = [k for k in data.obsm.keys() if str(k).startswith("velocity_")]
            record = vi.attach(data, dataset_id, dm.data_dir, paths, stored_as=[p.name for p in paths],
                               label=stored[0][1] if len(stored) == 1 else None)
            save_edit(adaptor, data, output_path, before, obsm_keys=old_keys, var=True)
            return record

        try:
            record = await held.run(work)
        except vi.VelocityInputError as e:
            for p, _n, new in stored:
                if new:
                    p.unlink(missing_ok=True)
            raise HTTPException(status_code=e.status, detail=str(e))
        except Exception as e:
            dm.evict(dataset_id)
            logger.error("Velocity attach failed for %s: %s", dataset_id, e, exc_info=True)
            raise HTTPException(status_code=500, detail=f"Could not attach the velocity counts: {e!s}")
        if backed:
            dm.evict(dataset_id)
            await dm.load_dataset(dataset_id)
        else:
            dm.saved_in_place(dataset_id, adaptor, output_path)
    return {"attached": True, "source": record}


async def _save(dm: DatasetManager, dataset_id: str, p: vel.Prepared, out: dict, extra: dict, resources: dict,
                ) -> dict[str, Any]:
    async with dataset_edit(dataset_id) as held:
        adaptor = await _adaptor(dm, dataset_id)
        if adaptor.cells_fingerprint() != p.fingerprint:
            raise vel.VelocityError("This dataset's cells changed while velocity ran (a pipeline run or a filter); "
                                    "nothing was saved. Run it again.", 409)
        output_path = str(dm.derived_h5ad_path(dataset_id, adaptor.h5ad_path))
        backed = await edit_room(dm, dataset_id, adaptor)

        def work() -> dict[str, Any]:
            data = editable(adaptor)
            before = uns_snapshot(data)
            old = [k for k in data.obsm.keys() if str(k).startswith("velocity_")]
            run = vel.save(data, p, out, extra, resources=resources)
            keys = sorted({*old, *run["obsm_keys"].values()})
            save_edit(adaptor, data, output_path, before, obsm_keys=keys, var=True)
            return run

        try:
            run = await held.run(work)
        except Exception:
            dm.evict(dataset_id)  # the cached copy may hold arrays that were never saved
            raise
        if backed:
            dm.evict(dataset_id)
            await dm.load_dataset(dataset_id)
            adaptor = await _adaptor(dm, dataset_id)
        else:
            dm.saved_in_place(dataset_id, adaptor, output_path)
        vel.remember(adaptor, run, extra["result_id"])
        return run


@router.post("/datasets/{dataset_id}/velocity/run-stream")
async def run_velocity(dataset_id: str, body: RunRequest, dm: DatasetManager = Depends(get_dataset_manager)):
    """Run velocity and stream it. The estimate comes first; the memory it needs is reserved
    until the run ends (a load of another dataset meanwhile counts it as used). Saved only at
    the end of the run."""
    adaptor = await _adaptor(dm, dataset_id)
    if dataset_id in _running:
        raise HTTPException(status_code=409, detail="A velocity run is already working on this dataset.")
    if not vi.has_store(dm.data_dir, dataset_id):
        raise HTTPException(status_code=422, detail=(
            "This dataset has no spliced and unspliced counts yet: attach them first (a velocyto .loom, an .h5ad "
            "with those layers, or a STARsolo Velocyto folder)."))
    est = await asyncio.to_thread(_estimate, dm, dataset_id, adaptor, method=body.method, n_top_genes=body.n_top_genes,
                                  n_neighbors=body.n_neighbors)
    await dm.reserve(dataset_id, est["peak_bytes"], what=_WHAT)
    params = vel.Params(method=body.method, n_top_genes=body.n_top_genes, n_neighbors=body.n_neighbors,
                        group_by=body.group_by)
    del adaptor

    async def events():
        if dataset_id in _running:  # another request started between the check and here
            yield _sse("error", {"error": "A velocity run is already working on this dataset."})
            return
        _running.add(dataset_id)
        dm.reserve_run(dataset_id, est["peak_bytes"])
        stop = threading.Event()
        watchdog = RunWatchdog()
        sent = 0
        gen = None
        pending: asyncio.Future | None = None

        def finish(fut: asyncio.Future | None = None) -> None:
            if fut is not None and not fut.cancelled():
                fut.exception()  # retrieved: a stopped run is not an error
            if gen is not None:
                gen.close()
            watchdog.stop()
            vel.drop_next_files(dm.data_dir, dataset_id)
            dm.release_run(dataset_id)
            _running.discard(dataset_id)

        try:
            watchdog.start()
            yield _sse("start", {"method": body.method, "estimate": est})
            watchdog.step_started(vel.STEP)
            adaptor = await _adaptor(dm, dataset_id)
            pending = asyncio.ensure_future(asyncio.to_thread(vel.prepare, adaptor, dm.data_dir, dataset_id, params))
            del adaptor
            prepared = await asyncio.shield(pending)
            pending = None
            yield _sse("progress", {"step": "prepared", "message": prepared.neighbours_note,
                                    "n_cells": int(prepared.positions.size)})
            gen = vel.iter_run(prepared, stop.is_set)
            while True:
                pending = asyncio.ensure_future(asyncio.to_thread(next, gen, _DONE))
                item = await asyncio.shield(pending)
                pending = None
                for warning in watchdog.events[sent:]:
                    yield _sse("watchdog", warning)
                sent = len(watchdog.events)
                if item is _DONE:
                    break
                if item["type"] == "progress":
                    yield _sse("progress", {k: v for k, v in item.items() if k != "type"})
                else:
                    watchdog.step_finished(vel.STEP)
                    res = watchdog.summary()
                    resources = {"peak_gb": res["peak_gb"], "limit_gb": res["limit_gb"]}
                    extra = {"result_id": uuid.uuid4().hex[:12], "resources": res}
                    run = await _save(dm, dataset_id, prepared, item["out"], extra, resources)
                    yield _sse("result", {"run": vel.public_run(run)})
            yield _sse("complete", {})
        except vel.VelocityCancelled:
            return
        except vel.VelocityError as e:
            yield _sse("error", {"error": str(e)})
        except Exception as e:  # noqa: BLE001 - surface to the client, keep the server up
            logger.error("Velocity failed for %s: %s", dataset_id, e, exc_info=True)
            detail = e.detail if isinstance(e, HTTPException) else str(e)
            yield _sse("error", {"error": f"RNA velocity failed: {detail}"})
        finally:
            stop.set()  # a thread mid-block stops before the next one
            if pending is not None and not pending.done():
                pending.add_done_callback(finish)
            else:
                finish()

    return StreamingResponse(keepalive(events()), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@router.get("/datasets/{dataset_id}/velocity/grid")
async def velocity_grid(
    dataset_id: str,
    embedding: str = Query(max_length=256),
    density: float = Query(default=1.0, ge=0.2, le=2.0),
    dm: DatasetManager = Depends(get_dataset_manager),
):
    """The arrows on a grid over this embedding (scVelo's default grid at density 1, 50 x 50 before
    empty points go). An embedding the run did not see is projected now from the run's graph."""
    adaptor = await _adaptor(dm, dataset_id)
    return await _read(vel.grid_for, adaptor, dm.data_dir, dataset_id, embedding, density)


@router.get("/datasets/{dataset_id}/velocity/phase")
async def velocity_phase(
    dataset_id: str,
    gene: str = Query(max_length=256),
    color_by: str | None = Query(default=None, max_length=256),
    dm: DatasetManager = Depends(get_dataset_manager),
):
    """A gene's phase portrait: spliced and unspliced first moments with the fitted gamma."""
    adaptor = await _adaptor(dm, dataset_id)
    return await _read(vel.phase, adaptor, dm.data_dir, dataset_id, gene, color_by)


@router.get("/datasets/{dataset_id}/velocity/genes")
async def velocity_genes(dataset_id: str, dm: DatasetManager = Depends(get_dataset_manager)):
    """The fitted genes: gamma, r2 and whether each passed as a velocity gene."""
    adaptor = await _adaptor(dm, dataset_id)
    return await _read(vel.gene_table, adaptor.adata)
