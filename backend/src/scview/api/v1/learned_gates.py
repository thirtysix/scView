"""Learned gates (core/learned_gate.py): train a gate on a selection, keep it in the
library, export / import it, check and apply it to any dataset.

``POST /datasets/{id}/learned-gates/train-stream`` trains on the selection and streams
it (SSE); it writes nothing into the dataset and keeps the gate as a draft, shown with
its metrics before ``POST /learned-gates/{gid}/save`` keeps it in the library
(``DATA_DIR/learned_gates/``) and records the training step on the source dataset, the
selection's cells in the results store. ``GET /learned-gates`` lists the library,
``GET /learned-gates/{gid}`` downloads one gate file, ``POST /learned-gates/import``
takes one, ``DELETE /learned-gates/{gid}`` removes it.
``POST /datasets/{id}/learned-gates/{gid}/check`` reports the species, gene mapping,
coverage and warnings without writing; ``.../apply`` writes obs ``gate_<name>`` (in /
out) and ``gate_<name>_score`` under the dataset's edit lock.

A large dataset (core/large_mode.py) is read from its counts file: training reads the
training cells' rows only, applying the gate's genes for every cell in one pass by row
blocks (before the edit lock is taken); both give the values the dataset in memory gives.

A client that goes away (the Cancel button) stops training after the fit it is on.
"""

from __future__ import annotations

import asyncio
import json
import logging
import threading
from collections import OrderedDict
from typing import Any, Literal

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import Response, StreamingResponse
from pydantic import BaseModel, Field

from scview.config import Settings
from scview.core import large_mode, results_store
from scview.core import learned_gate as lg
from scview.core.dataset_manager import DatasetManager
from scview.core.obs_writes import dataset_edit, edit_room, editable, save_edit, uns_snapshot
from scview.core.run_watchdog import RunWatchdog
from scview.core.sse import keepalive
from scview.dependencies import get_dataset_manager, get_settings_dep

logger = logging.getLogger(__name__)

router = APIRouter()

_DONE = object()
_running: set[str] = set()  # datasets with a training streaming: one at a time each
# Trained gates not yet saved (gate, its selection, the source's fingerprint): the newest few.
_drafts: OrderedDict[str, dict[str, Any]] = OrderedDict()
_MAX_DRAFTS = 16

_WHAT = "Learned gates"


class TrainRequest(BaseModel):
    name: str = Field(min_length=1, max_length=60)
    # The selection as the frontend encodes it: {n_cells, indices} or {n_cells, bitmask}
    cells: dict[str, Any]
    # None picks the column; "" means none
    sample_column: str | None = Field(default=None, max_length=200)
    condition_column: str | None = Field(default=None, max_length=200)
    exclude_cc_stress: bool = True
    negatives: Literal["own_samples", "all"] = "own_samples"


class ApplyRequest(BaseModel):
    sample_column: str | None = Field(default=None, max_length=200)  # for the per-sample counts


def _sse(event: str, data: dict[str, Any]) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


async def _adaptor(dm: DatasetManager, dataset_id: str):
    adaptor = await dm.get_or_load_dataset(dataset_id)
    if adaptor is None:
        raise HTTPException(status_code=404, detail="Dataset not found.")
    return adaptor


def _http(e: lg.LearnedGateError) -> HTTPException:
    return HTTPException(status_code=e.status, detail=str(e))


def _expression(fallback: list[str]):
    from scview.core.label_transfer import LabelTransferError, expression_of

    def get(adata, rows=None):  # every cell's values: prepare takes the rows it needs
        try:
            return expression_of(adata, fallback)
        except LabelTransferError as e:
            raise lg.LearnedGateError(str(e).replace("mapping compares", "a learned gate reads"), 422) from e
    return get


def _large_rows(counts):
    """A large dataset's values of the training cells alone, from its counts file."""
    from scview.core.label_transfer import LARGE_SOURCE, lognorm_rows

    def get(adata, rows):
        return lognorm_rows(counts, rows), adata.var.copy(), LARGE_SOURCE
    return get


class _LargeColumns:
    """A large dataset's log-normalised values at some columns, every cell
    (``label_transfer.lognorm_columns``: one pass over its counts), kept once read so the
    pass runs before the edit lock is taken and the edit reuses it."""

    def __init__(self, adata, counts) -> None:
        self.adata, self.counts, self.shape = adata, counts, counts.shape
        self._read: dict[tuple[int, ...], Any] = {}

    def __call__(self, cols):
        from scview.core.label_transfer import lognorm_columns

        key = tuple(int(c) for c in cols)
        if key not in self._read:
            self._read[key] = lognorm_columns(self.counts, cols)
        return self._read[key]


def _isg_genes(settings: Settings, species: str) -> set[str] | None:
    """The MSigDB Hallmark interferon alpha + gamma genes, or None without MSigDB."""
    try:
        from scview.core.msigdb_loader import HALLMARK, get_msigdb_loader

        loader = get_msigdb_loader(settings.MSIGDB_DIR)
        if loader is None:
            return None
        coll = HALLMARK.get(species, HALLMARK["human"])
        genes: set[str] = set()
        for name in lg.ISG_SETS:
            gs = loader.get_gene_set(name, coll) or loader.get_gene_set(name, "", species)
            genes |= set((gs or {}).get("genes") or [])
        return genes or None
    except Exception as e:  # noqa: BLE001 - the check is a caution, never a failure
        logger.debug("No interferon gene sets for the learned gate check: %s", e)
        return None


def _keep_draft(gate_id: str, draft: dict[str, Any]) -> None:
    _drafts[gate_id] = draft
    while len(_drafts) > _MAX_DRAFTS:
        _drafts.popitem(last=False)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


@router.post("/datasets/{dataset_id}/learned-gates/train-stream")
async def train_stream(
    dataset_id: str,
    body: TrainRequest,
    dm: DatasetManager = Depends(get_dataset_manager),
    settings: Settings = Depends(get_settings_dep),
):
    """Train a learned gate on the selection and stream it. Events: ``start`` (cells,
    negatives, candidate genes), ``progress`` after each penalty value, ``watchdog``
    memory warnings, ``result`` (the draft gate, its metrics, genes for and against, the
    cells the model and the outline disagree on), then ``complete``, or ``error``.
    Writes nothing; ``POST /learned-gates/{gid}/save`` keeps the gate."""
    if dataset_id in _running:
        raise HTTPException(status_code=409, detail="A learned gate is already training on this dataset.")
    adaptor = await _adaptor(dm, dataset_id)
    adata = adaptor.adata
    large = large_mode.is_large(adaptor)
    if large:
        counts = adaptor.counts_matrix(_WHAT)
        await dm.reserve(dataset_id, lg.large_estimate(counts, train=True)["peak_bytes"], what=_WHAT)
        expression = _large_rows(counts)
    else:
        expression = _expression(dm.h5ad_candidates(dataset_id)[1:])

    def prep(src) -> lg.TrainInput:
        data = src.to_memory() if getattr(src, "isbacked", False) and not large else src
        return lg.prepare(data, body.cells, expression=expression, sample_column=body.sample_column,
                          condition_column=body.condition_column, exclude_cc_stress=body.exclude_cc_stress,
                          negatives=body.negatives)

    try:
        inp = await asyncio.to_thread(prep, adata)
    except lg.LearnedGateError as e:
        raise _http(e)
    fingerprint = await asyncio.to_thread(adaptor.cells_fingerprint)
    overlap = await asyncio.to_thread(lg.cluster_overlap, adata.obs, inp.info["selection"], int(adata.n_obs))
    info = dm.get_dataset_info(dataset_id) or {}
    source_name = str(info.get("name") or dataset_id)
    isg = await asyncio.to_thread(_isg_genes, settings, inp.info.get("species") or "human")
    del adata, adaptor  # the run needs only its copy

    async def events():
        if dataset_id in _running:  # another request started between the check and here
            yield _sse("error", {"error": "A learned gate is already training on this dataset."})
            return
        _running.add(dataset_id)
        stop = threading.Event()
        watchdog = RunWatchdog()
        sent = 0
        gen = lg.iter_train(inp, name=body.name, stop=stop.is_set)
        pending: asyncio.Future | None = None

        def finish(fut: asyncio.Future | None = None) -> None:
            if fut is not None and not fut.cancelled():
                fut.exception()  # retrieved: a stopped run is not an error
            gen.close()
            watchdog.stop()
            _running.discard(dataset_id)

        try:
            watchdog.start()
            yield _sse("start", {k: inp.info[k] for k in ("n_selected", "n_pool", "n_pos", "n_neg", "sample_column",
                                                          "condition_column", "samples", "negatives", "n_up",
                                                          "n_down", "source", "excluded_rules")}
                       | {"n_candidates": len(inp.genes), "warnings": inp.warnings, "folds": lg.N_FOLDS,
                          "n_c": len(lg.C_GRID)})
            watchdog.step_started(lg.STEP_TRAIN)
            while True:
                # Each fit runs in a worker thread. Shielded, so a client going away
                # cancels this wait, not the thread's future, whose callback then
                # closes the run.
                pending = asyncio.ensure_future(asyncio.to_thread(next, gen, _DONE))
                item = await asyncio.shield(pending)
                pending = None
                for warning in watchdog.events[sent:]:
                    yield _sse("watchdog", warning)
                sent = len(watchdog.events)
                if item is _DONE:
                    break
                if item["type"] == "trained":
                    watchdog.step_finished(lg.STEP_TRAIN)
                    gate: lg.LearnedGate = item["gate"]
                    gate = gate.model_copy(update={
                        "source": gate.source.model_copy(update={
                            "dataset_id": dataset_id, "name": source_name, "cells_fingerprint": fingerprint}),
                        "isg_weight_share": lg.isg_share(gate, isg)})
                    _keep_draft(gate.id, {"gate": gate, "cells": body.cells, "dataset_id": dataset_id,
                                          "fingerprint": fingerprint, "report": item["report"]})
                    yield _sse("result", {"gate": lg.gate_dict(gate), "summary": lg.summary(gate),
                                          "report": item["report"], "overlap": overlap, "saved": False,
                                          "resources": watchdog.summary()})
                else:
                    yield _sse(item["type"], {k: v for k, v in item.items() if k != "type"})
            yield _sse("complete", {})
        except lg.LearnedGateCancelled:
            return
        except lg.LearnedGateError as e:
            yield _sse("error", {"error": str(e)})
        except Exception as e:  # noqa: BLE001 — surface to the client, keep the server up
            logger.error("Learned gate training failed for %s: %s", dataset_id, e, exc_info=True)
            yield _sse("error", {"error": f"Training the learned gate failed: {e}"})
        finally:
            stop.set()  # a thread mid-fit stops before the next one
            if pending is not None and not pending.done():
                pending.add_done_callback(finish)
            else:
                finish()

    return StreamingResponse(
        keepalive(events()),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


async def _edit(dm: DatasetManager, dataset_id: str, edit, what: str, fingerprint: str | None = None):
    """Run ``edit(adata)`` in a worker thread holding the dataset's edit lock, then save
    the obs/uns change; returns (result, the adaptor after the save)."""
    async with dataset_edit(dataset_id) as held:
        adaptor = await _adaptor(dm, dataset_id)
        if fingerprint is not None and adaptor.cells_fingerprint() != fingerprint:
            raise HTTPException(status_code=409, detail=(
                "This dataset's cells changed since the gate was trained (a pipeline run or a filter); "
                "train it again."))
        output_path = str(dm.derived_h5ad_path(dataset_id, adaptor.h5ad_path))
        backed = await edit_room(dm, dataset_id, adaptor)

        def _run():
            work = editable(adaptor)
            before = uns_snapshot(work)
            result = edit(work)
            save_edit(adaptor, work, output_path, before)
            return result

        try:
            result = await held.run(_run)
        except lg.LearnedGateError as exc:  # raised before any change
            raise _http(exc)
        except Exception as exc:
            dm.evict(dataset_id)  # the cached copy may hold columns that were never saved
            logger.error("%s failed for %s: %s", what, dataset_id, exc, exc_info=True)
            raise HTTPException(status_code=500, detail=f"{what} failed: {exc!s}")
        if backed:
            dm.evict(dataset_id)
            await dm.load_dataset(dataset_id)
        else:
            dm.saved_in_place(dataset_id, adaptor, output_path)
    return result, await _adaptor(dm, dataset_id)


@router.post("/learned-gates/{gate_id}/save")
async def save_gate(gate_id: str, dm: DatasetManager = Depends(get_dataset_manager)):
    """Keep a trained gate in the library and record the training on its source dataset
    (the selection's cells in the results store)."""
    draft = _drafts.get(gate_id)
    if draft is None:
        raise HTTPException(status_code=404, detail=(
            "That trained gate is no longer held (a restart, or many gates trained since): train it again."))
    gate: lg.LearnedGate = draft["gate"]
    dataset_id = draft["dataset_id"]

    def edit(adata) -> None:
        lg.record_training(adata, gate, draft["report"])

    _, adaptor = await _edit(dm, dataset_id, edit, "Recording the learned gate", draft["fingerprint"])
    await asyncio.to_thread(lg.save_gate, dm.data_dir, gate)
    results_store.keep(adaptor, lg.TRAIN_KIND, {
        "gate_id": gate.id, "name": gate.name, "cells": draft["cells"], "cv": gate.cv.model_dump(),
        "sample_column": gate.source.sample_column, "condition_column": gate.source.condition_column,
        "samples": gate.source.samples, "negatives": gate.source.negatives,
    }, key=gate.id, label=f"Learned gate '{gate.name}': training selection")
    _drafts.pop(gate_id, None)
    return {**lg.summary(gate), "saved": True}


# ---------------------------------------------------------------------------
# Library
# ---------------------------------------------------------------------------


@router.get("/learned-gates")
async def list_gates(dm: DatasetManager = Depends(get_dataset_manager)):
    """The library, newest first."""
    return {"gates": await asyncio.to_thread(lg.list_gates, dm.data_dir), "score_note": lg.SCORE_NOTE,
            "max_file_bytes": lg.MAX_FILE_BYTES}


@router.get("/learned-gates/{gate_id}")
async def download_gate(gate_id: str, dm: DatasetManager = Depends(get_dataset_manager)):
    """One gate file (JSON), as a download."""
    try:
        gate = await asyncio.to_thread(lg.load_gate, dm.data_dir, gate_id)
    except lg.LearnedGateError as e:
        raise _http(e)
    from scview.core.label_transfer import slug

    return Response(content=gate.model_dump_json(indent=1), media_type="application/json",
                    headers={"Content-Disposition": f'attachment; filename="learned_gate_{slug(gate.name)}.json"'})


@router.post("/learned-gates/import")
async def import_gate(request: Request, dm: DatasetManager = Depends(get_dataset_manager)):
    """Add a gate file (the JSON body, at most 256 KB) to the library, checked against
    the format: no code is ever read from it."""
    raw = await request.body()
    if len(raw) > lg.MAX_FILE_BYTES:
        raise HTTPException(status_code=413, detail=f"A gate file is at most {lg.MAX_FILE_BYTES // 1024} KB.")
    try:
        gate = await asyncio.to_thread(lg.import_gate, dm.data_dir, raw)
    except lg.LearnedGateError as e:
        raise _http(e)
    return lg.summary(gate)


@router.delete("/learned-gates/{gate_id}")
async def delete_gate(gate_id: str, dm: DatasetManager = Depends(get_dataset_manager)):
    try:
        gone = await asyncio.to_thread(lg.delete_gate, dm.data_dir, gate_id)
    except lg.LearnedGateError as e:
        raise _http(e)
    if not gone:
        raise HTTPException(status_code=404, detail="That learned gate is not in the library.")
    return {"deleted": gate_id}


# ---------------------------------------------------------------------------
# Check and apply
# ---------------------------------------------------------------------------


def _load(dm: DatasetManager, gate_id: str) -> lg.LearnedGate:
    try:
        return lg.load_gate(dm.data_dir, gate_id)
    except lg.LearnedGateError as e:
        raise _http(e)


@router.post("/datasets/{dataset_id}/learned-gates/{gate_id}/check")
async def check_gate(
    dataset_id: str,
    gate_id: str,
    dm: DatasetManager = Depends(get_dataset_manager),
    settings: Settings = Depends(get_settings_dep),
):
    """Whether the gate can be applied here: species and gene mapping, the share of its
    genes and weight found, refusals and warnings, the columns it would write, and the
    sample column the counts would be split by. Writes nothing."""
    gate = await asyncio.to_thread(_load, dm, gate_id)
    adaptor = await _adaptor(dm, dataset_id)
    adata = adaptor.adata
    counts, unreadable = None, None
    if large_mode.is_large(adaptor):
        try:
            counts = adaptor.counts_matrix(_WHAT)
        except large_mode.LargeDatasetError as e:  # no counts file: reported, not raised
            unreadable = str(e)

    def work() -> dict[str, Any]:
        rep = lg.check(gate, adata, dataset_id=dataset_id, isg_genes=_isg_genes(settings, gate.species))
        col = rep.get("columns", {}).get("column")
        record = adata.uns.get(lg.UNS_KEY) or {}
        rep["replaces"] = bool(col and col in record)
        if col and col in adata.obs.columns and col not in record:
            rep["ok"] = False
            rep["refusal"] = f"This dataset already has a column '{col}' that no learned gate wrote; rename the gate."
        if unreadable:
            rep["ok"] = False
            rep["refusal"] = unreadable
        if counts is not None:
            rep["large"] = lg.large_estimate(counts)
        rep["sample_column"] = lg.pick_sample_column(adata.obs)
        rep["gate"] = lg.summary(gate)
        return rep

    return await asyncio.to_thread(work)


@router.post("/datasets/{dataset_id}/learned-gates/{gate_id}/apply")
async def apply_gate(
    dataset_id: str,
    gate_id: str,
    body: ApplyRequest | None = None,
    dm: DatasetManager = Depends(get_dataset_manager),
    settings: Settings = Depends(get_settings_dep),
):
    """Score every cell with the gate and write obs ``gate_<name>`` ("in"/"out") and
    ``gate_<name>_score``; the result (counts per sample, coverage, warnings) is kept in
    the results store for the co-pilot."""
    gate = await asyncio.to_thread(_load, dm, gate_id)
    adaptor = await _adaptor(dm, dataset_id)
    body = body or ApplyRequest()
    if body.sample_column and body.sample_column not in adaptor.adata.obs.columns:
        raise HTTPException(status_code=404, detail=f"No column '{body.sample_column}' in this dataset.")
    expression = _expression(dm.h5ad_candidates(dataset_id)[1:])
    isg = await asyncio.to_thread(_isg_genes, settings, gate.species)
    read: _LargeColumns | None = None
    if large_mode.is_large(adaptor):
        # The pass over the counts runs here, outside the edit lock (it can take minutes);
        # the edit reuses it unless the dataset was reloaded meanwhile.
        counts = adaptor.counts_matrix(_WHAT)
        await dm.reserve(dataset_id, lg.large_estimate(counts)["peak_bytes"], what=_WHAT)
        read = _LargeColumns(adaptor.adata, counts)

        def warm() -> None:
            read(lg.columns_read(gate, read.adata, read.adata.var, dataset_id=dataset_id, isg_genes=isg))

        try:
            await asyncio.to_thread(warm)
        except lg.LearnedGateError as e:
            raise _http(e)

    def edit(adata) -> dict[str, Any]:
        if large_mode.is_large(adata):
            from scview.core.label_transfer import LARGE_SOURCE

            cols = read if read is not None and read.adata is adata else \
                _LargeColumns(adata, large_mode.counts_of(adata))
            X, var, source = cols, adata.var, LARGE_SOURCE
        else:
            X, var, source = expression(adata)
        return lg.apply(adata, gate, X=X, var=var, source=source, dataset_id=dataset_id,
                        sample_column=body.sample_column, isg_genes=isg)

    result, after = await _edit(dm, dataset_id, edit, "Applying the learned gate")
    results_store.keep(after, lg.KIND, result, key=gate.id, label=f"Learned gate '{gate.name}' applied",
                       params={"gate_id": gate.id, "column": result["column"]})
    return result
