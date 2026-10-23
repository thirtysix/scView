"""Cell-cell communication: ligand-receptor scoring with LIANA's consensus rank aggregate.

Between the groups of one obs column, optionally on one condition's cells only
(e.g. Kang's STIM), so conditions can be run one at a time; a differential
comparison between conditions is not done here. Computed on demand and never
written to the dataset: the last few results are kept in memory for the CSV
download and the co-pilot. See ``core/ccc.py`` and ``core/lr_resource.py``.
"""

from __future__ import annotations

import asyncio
import io
import json
import logging
import re
import threading
import time
import uuid
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from scview.config import Settings
from scview.core import ccc, large_mode, lr_resource
from scview.core.activity import lognorm_matrix
from scview.core.confounding import condition_columns
from scview.core.dataset_manager import DatasetManager
from scview.core.species import detect_species
from scview.core.sse import keepalive
from scview.dependencies import get_dataset_manager, get_settings_dep

logger = logging.getLogger(__name__)

router = APIRouter()

MAX_ROWS = 5000  # rows sent to the browser, strongest magnitude first; the CSV has all
MAX_PERMS = 5000
RESOURCE_LABELS = {"consensus": "Consensus (human)", "mouseconsensus": "Consensus (mouse)"}

_running: set[str] = set()  # datasets with a run going: one at a time each


class CommunicationRequest(BaseModel):
    groupby: str = Field(min_length=1, max_length=256)
    subset_column: str | None = Field(default=None, max_length=256)
    subset_value: str | None = Field(default=None, max_length=1024)
    resource: str = Field(default="auto", max_length=32)  # auto | consensus | mouseconsensus
    expr_prop: float = Field(default=ccc.EXPR_PROP, ge=0.0, le=1.0)
    n_perms: int = Field(default=ccc.N_PERMS, ge=0, le=MAX_PERMS)


def _sse(event: str, data: dict[str, Any]) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


def _res_dir(settings: Settings) -> Path:
    return Path(settings.DATA_DIR) / "networks"  # the data volume, next to CollecTRI / PROGENy


async def _adaptor(dm: DatasetManager, dataset_id: str):
    adaptor = await dm.get_or_load_dataset(dataset_id)
    if adaptor is None:
        raise HTTPException(status_code=404, detail=f"Dataset '{dataset_id}' not found.")
    return adaptor


def _groups(obs: pd.DataFrame, col: str, mask: np.ndarray | None = None) -> int:
    if col not in obs.columns:
        raise HTTPException(status_code=404, detail=f"Column '{col}' not found.")
    s = obs[col] if mask is None else obs[col][mask]
    return int(s.nunique(dropna=True))


def _mask(obs: pd.DataFrame, column: str | None, value: str | None) -> np.ndarray | None:
    if not column:
        return None
    if column not in obs.columns:
        raise HTTPException(status_code=404, detail=f"Column '{column}' not found.")
    mask = (obs[column].astype(str) == str(value)).to_numpy() & obs[column].notna().to_numpy()
    if not mask.any():
        raise HTTPException(status_code=400, detail=f"No cells have {column} = {value}.")
    return mask


def _choose_resource(choice: str, adata, table: pd.DataFrame | None = None) -> tuple[str, str, str]:
    """(resource, species, why): the user's choice, else by the species call, else
    (species unknown) the resource with more of its genes in the dataset."""
    if choice not in ("auto", *lr_resource.RESOURCES):
        raise HTTPException(status_code=400, detail=f"Unknown resource '{choice}'.")
    call = detect_species(adata)
    if choice != "auto":
        return choice, call.species, "chosen"
    if call.species in ("human", "mouse"):
        name = "consensus" if call.species == "human" else "mouseconsensus"
        return name, call.species, call.evidence
    if table is None:
        return "consensus", call.species, call.evidence
    genes = adata.raw.var_names if adata.raw is not None else adata.var_names
    cov = {n: ccc.resource_coverage(lr_resource.select(table, n), genes) for n in lr_resource.RESOURCES}
    best = max(cov, key=cov.get)
    return best, call.species, (f"Species unknown; {best} has more of its genes here "
                                f"({cov[best]:.0%} vs {min(cov.values()):.0%}).")


def _num(a, digits: int = 4) -> list:
    """Floats to ``digits`` significant digits for JSON; NaN as null."""
    a = np.asarray(a, dtype=np.float64)
    out: list = []
    for v in a.tolist():
        out.append(None if not np.isfinite(v) else float(f"{v:.{digits}g}"))
    return out


ROW_TEXT = ["source", "target", "ligand_complex", "receptor_complex", "ligand", "receptor"]
ROW_NUM = ["magnitude_rank", "specificity_rank", "lr_means", "cellphone_pvals", "expr_prod",
           "scaled_weight", "lr_logfc", "spec_weight", "lrscore", "ligand_props", "receptor_props"]


def result_body(r: ccc.Result) -> dict[str, Any]:
    tab = r.table.head(MAX_ROWS)
    roles = r.roles()
    return {
        "result_id": r.id,
        "groupby": r.groupby,
        "subset": {"column": r.subset[0], "value": r.subset[1]} if r.subset else None,
        "resource": r.resource,
        "resource_label": RESOURCE_LABELS[r.resource],
        "species": r.species,
        "matrix_source": r.matrix_source,
        "n_cells": r.n_cells,
        "groups": r.groups,
        "group_sizes": r.group_sizes,
        "dropped_groups": r.dropped_groups,
        "expr_prop": r.expr_prop,
        "n_perms": r.n_perms,
        "seed": ccc.SEED,
        "n_resource": r.n_resource,
        "n_rows": int(len(r.table)),
        "n_interactions": int(r.table[["ligand_complex", "receptor_complex"]].drop_duplicates().shape[0]),
        "n_specific": int((r.table["specificity_rank"] <= ccc.SPECIFIC).sum()),
        "specific_threshold": ccc.SPECIFIC,
        "rows_shown": int(len(tab)),
        "columns": {**{c: tab[c].astype(str).tolist() for c in ROW_TEXT},
                    **{c: _num(tab[c]) for c in ROW_NUM}},
        "pair_counts": r.counts.astype(int).tolist(),
        "pair_strength": [_num(row) for row in r.strength],
        "roles": {
            "outgoing": roles["outgoing"].tolist(), "incoming": roles["incoming"].tolist(),
            "outgoing_strength": _num(roles["outgoing_strength"]),
            "incoming_strength": _num(roles["incoming_strength"]),
        },
        "elapsed_seconds": r.elapsed_seconds,
        "citation": lr_resource.CITATION,
    }


@router.get("/communication/resource")
async def resource_status(settings: Settings = Depends(get_settings_dep)):
    """Whether the LIANA resource is downloaded yet, and where from."""
    meta = lr_resource.read_meta(_res_dir(settings))
    return {"downloaded": meta is not None, "url": lr_resource.URL, "version": lr_resource.VERSION,
            "citation": lr_resource.CITATION, "fetched_at": (meta or {}).get("fetched_at"),
            "n_interactions": (meta or {}).get("n_interactions")}


@router.get("/datasets/{dataset_id}/communication/options")
async def communication_options(
    dataset_id: str,
    groupby: str = "",
    subset_column: str = "",
    subset_value: str = "",
    n_perms: int = ccc.N_PERMS,
    dm: DatasetManager = Depends(get_dataset_manager),
):
    """Condition-like columns to subset by (with their values), the species and
    resource that would be used, and, with ``groupby``, a run-time estimate."""
    adata = (await _adaptor(dm, dataset_id)).adata
    obs = adata.obs
    conditions = [{"column": c, "values": sorted(map(str, obs[c].dropna().unique()))}
                  for c in condition_columns(obs)]
    resource, species, why = await asyncio.to_thread(_choose_resource, "auto", adata)
    body: dict[str, Any] = {
        "conditions": conditions, "species": species, "species_evidence": why,
        "resource": resource, "max_groups": ccc.MAX_GROUPS, "max_perms": MAX_PERMS,
        "default_expr_prop": ccc.EXPR_PROP, "default_n_perms": ccc.N_PERMS,
        "min_cells": ccc.MIN_CELLS,
    }
    if groupby:
        mask = _mask(obs, subset_column or None, subset_value)
        n_groups = _groups(obs, groupby, mask)
        n_cells = int(adata.n_obs if mask is None else mask.sum())
        body.update(n_cells=n_cells, n_groups=n_groups,
                    estimate_seconds=ccc.estimate_seconds(n_cells, n_groups, max(0, n_perms)))
    return body


@router.post("/datasets/{dataset_id}/communication/run")
async def run_communication(
    dataset_id: str,
    body: CommunicationRequest,
    dm: DatasetManager = Depends(get_dataset_manager),
    settings: Settings = Depends(get_settings_dep),
):
    """Score every sender x receiver x ligand-receptor pair and stream it.

    Events: ``start`` (cells, groups, resource, time estimate), ``progress``
    (permutations done), ``result`` (the summary and the strongest rows), then
    ``complete``, or ``error``. A client that goes away stops the run at its
    next batch of permutations."""
    if body.resource not in ("auto", *lr_resource.RESOURCES):
        raise HTTPException(status_code=400, detail=f"Unknown resource '{body.resource}'.")
    adaptor = await _adaptor(dm, dataset_id)
    large_mode.check(adaptor, "Ligand-receptor communication",
                     "it reads the chosen cells' values and totals over every gene in memory")
    adata = adaptor.adata
    obs = adata.obs
    mask = _mask(obs, body.subset_column, body.subset_value)
    n_groups = _groups(obs, body.groupby, mask)
    if not 2 <= n_groups <= ccc.MAX_GROUPS:
        raise HTTPException(status_code=400, detail=(
            f"'{body.groupby}' has {n_groups} groups{' in these cells' if mask is not None else ''}; "
            f"communication is scored between 2 and {ccc.MAX_GROUPS}."))
    if dataset_id in _running:
        raise HTTPException(status_code=409, detail="A communication run is already going on this dataset.")
    n_cells = int(adata.n_obs if mask is None else mask.sum())
    subset = (body.subset_column, str(body.subset_value)) if mask is not None else None
    res_dir = _res_dir(settings)

    async def events():
        if dataset_id in _running:
            yield _sse("error", {"error": "A communication run is already going on this dataset."})
            return
        _running.add(dataset_id)
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue = asyncio.Queue()
        stop = threading.Event()

        def put(event: str, data: dict) -> None:
            loop.call_soon_threadsafe(queue.put_nowait, (event, data))

        def work() -> None:
            t0 = time.time()
            try:
                table = lr_resource.load(res_dir)
                resource, species, why = _choose_resource(body.resource, adata, table)
                put("start", {"n_cells": n_cells, "n_groups": n_groups, "resource": resource,
                              "resource_label": RESOURCE_LABELS[resource], "species": species,
                              "species_evidence": why, "n_perms": body.n_perms,
                              "estimate_seconds": ccc.estimate_seconds(n_cells, n_groups, body.n_perms)})
                M, genes, source = lognorm_matrix(adata)
                labels = obs[body.groupby]
                if mask is not None:
                    M, labels = M[mask], labels[mask]
                prep = ccc.prepare(M, genes, labels, lr_resource.select(table, resource))
                del M
                out = ccc.score(prep, expr_prop=body.expr_prop, n_perms=body.n_perms,
                                progress=lambda d, n: put("progress", {"done": d, "total": n}),
                                stop=stop.is_set)
                result = ccc.Result(
                    id=uuid.uuid4().hex[:12], dataset_id=dataset_id, groupby=body.groupby, subset=subset,
                    resource=resource, species=species, matrix_source=source,
                    groups=prep.stats.groups, group_sizes=prep.stats.sizes.tolist(),
                    dropped_groups=prep.dropped_groups, expr_prop=body.expr_prop, n_perms=body.n_perms,
                    n_resource=prep.n_resource, table=out, elapsed_seconds=round(time.time() - t0, 2),
                )
                ccc.keep(result)
                ccc.remember(adaptor, result)
                put("result", result_body(result))
                put("complete", {})
            except InterruptedError:
                put("complete", {})
            except lr_resource.ResourceUnavailable as e:
                put("error", {"error": str(e)})
            except (ccc.CCCError, ValueError) as e:
                put("error", {"error": str(e)})
            except Exception as e:  # noqa: BLE001 - surface to the client, keep the server up
                logger.error("Communication run failed for %s: %s", dataset_id, e, exc_info=True)
                put("error", {"error": f"The run failed: {e}"})
            finally:
                put("_done", {})

        fut = loop.run_in_executor(None, work)
        try:
            while True:
                event, data = await queue.get()
                if event == "_done":
                    break
                yield _sse(event, data)
                if event in ("complete", "error"):
                    break
        finally:
            stop.set()  # a client gone: the worker stops at its next permutation batch

            def release(_f) -> None:
                _running.discard(dataset_id)

            if fut.done():
                _running.discard(dataset_id)
            else:
                fut.add_done_callback(release)

    return StreamingResponse(
        keepalive(events()),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.get("/datasets/{dataset_id}/communication/results/{result_id}.csv")
async def communication_csv(dataset_id: str, result_id: str):
    """Every scored row of a kept result, strongest magnitude first."""
    r = ccc.recall(dataset_id, result_id)
    if r is None:
        raise HTTPException(status_code=404, detail="That result is no longer kept on the server: run it again.")

    def _csv() -> str:
        buf = io.StringIO()
        r.table.to_csv(buf, index=False, float_format="%.6g")
        return buf.getvalue()

    text = await asyncio.to_thread(_csv)
    where = f"_{r.subset[1]}" if r.subset else ""
    name = re.sub(r"[^A-Za-z0-9_.-]", "_", f"scview_communication_{r.groupby}{where}") + ".csv"
    return StreamingResponse(iter([text]), media_type="text/csv",
                             headers={"Content-Disposition": f'attachment; filename="{name}"'})
