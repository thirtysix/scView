"""The MCP tools: thin wrappers over scView's ``/api/v1`` routes (``mcp/api.py``).

Each tool calls the route the UI calls, so validation, edit locks, memory checks,
large-dataset refusals and provenance are the UI's. What a tool adds is the
shape: the columns that matter, the first ``limit`` rows with the total, numbers
to 4 significant digits (``mcp/compact.py``), the route's warnings named, and the
route's one-sentence error as the tool's error. No tool deletes anything.
"""

from __future__ import annotations

import asyncio
import base64
import gzip
import io
import json
from collections.abc import Callable
from typing import Annotated, Any, Literal

import numpy as np
import pandas as pd
from mcp.server.mcpserver import Image
from mcp.server.mcpserver.exceptions import ToolError
from mcp.types import TextContent, ToolAnnotations
from pydantic import Field

from scview.core.cell_filter import CellFilter
from scview.mcp import api, compact, jobs, params, plots
from scview.mcp.compact import DEFAULT_LIMIT, clamp, dumps, num, pick, rows, table

READ = ToolAnnotations(read_only_hint=True, destructive_hint=False, open_world_hint=False)
WRITE = ToolAnnotations(read_only_hint=False, destructive_hint=False, idempotent_hint=False,
                        open_world_hint=False)
WEB_READ = ToolAnnotations(read_only_hint=True, destructive_hint=False, open_world_hint=True)
WEB_WRITE = ToolAnnotations(read_only_hint=False, destructive_hint=False, open_world_hint=True)

TOOLS: list[tuple[Callable, ToolAnnotations]] = []


def tool(annotations: ToolAnnotations):
    def register(fn: Callable) -> Callable:
        TOOLS.append((fn, annotations))
        return fn
    return register


# --- shared argument types -------------------------------------------------------------

DatasetId = Annotated[str, Field(description="The dataset's id (from list_datasets).")]
Limit = Annotated[int, Field(ge=1, le=compact.MAX_LIMIT, description="Rows to return (default 20, max 200).")]
Filter = Annotated[CellFilter, Field(description=(
    "Cells by conditions joined by AND/OR, as the UI's cell filter: e.g. "
    '{"op":"and","items":[{"kind":"category","column":"donor_id","values":["C1"]},'
    '{"kind":"range","column":"DPS","min":11,"max":11}]}. Items: category (column, values), '
    "range (column, min and/or max), gene (gene, op >,>=,<,<=, value) or a group "
    '{"kind":"group","op":"or","conditions":[...]}.'))]


P0_NOTE = ("A p value of 0 is what the test returned: below double precision (under about 1e-300), "
           "not rounded here.")


def _max_chars(limit: int) -> int:
    return compact.MAX_CHARS if limit <= DEFAULT_LIMIT else int(compact.MAX_CHARS * limit / DEFAULT_LIMIT)


def _fail(e: api.RouteError) -> ToolError:
    if e.large_dataset:
        return ToolError(f"Refused on a large dataset: {e.detail}")
    return ToolError(e.detail)


async def _get(path: str, **query: Any) -> Any:
    try:
        return await api.get(path, **query)
    except api.RouteError as e:
        raise _fail(e) from None


async def _post(path: str, body: Any = None, **query: Any) -> Any:
    try:
        return await api.post(path, body, **query)
    except api.RouteError as e:
        raise _fail(e) from None


async def _raw(path: str, **query: Any) -> tuple[dict[str, str], bytes]:
    try:
        return await api.request("GET", path, query)
    except api.RouteError as e:
        raise _fail(e) from None


def _ds(dataset_id: str) -> str:
    return f"/datasets/{dataset_id}"


async def _indices(dataset_id: str, flt: CellFilter) -> np.ndarray:
    """The rows matching a filter, from the cell-filter route's bitmask."""
    r = await _post(f"{_ds(dataset_id)}/cells/filter", {"filter": flt.model_dump(), "mask": True})
    bits = np.unpackbits(np.frombuffer(base64.b64decode(r["mask"]), dtype=np.uint8), bitorder="little")
    return np.flatnonzero(bits[: r["n_cells"]])


# --- data ----------------------------------------------------------------------------------

_LIST_KEYS = ("id", "name", "status", "n_cells", "n_genes", "source", "format", "file_format",
              "created_at", "processed", "error_message")


@tool(READ)
async def list_datasets() -> str:
    """The datasets in this scView library: id, name, status, cells and genes. Start here."""
    data = await _get("/datasets")
    out = [pick(d, *_LIST_KEYS) for d in data]
    return dumps({"datasets": out, "n_total": len(out)})


@tool(READ)
async def get_dataset(dataset_id: DatasetId) -> str:
    """One dataset: cells, genes, processing steps done, embeddings, obs columns (type and
    number of values; the values when 30 or fewer), expression layers, active clustering,
    large-dataset mode, and other modalities present."""
    d = await _get(_ds(dataset_id))
    cols = []
    for c in d.get("obs_columns") or []:
        entry = {"name": c["name"], "dtype": c.get("dtype"), "n_unique": c.get("n_unique")}
        if c.get("values") is not None and (c.get("n_unique") or 99) <= 30:
            entry["values"] = c["values"]
        if c.get("dtype") == "bool" and (c.get("n_unique") or 0) < 2:
            # The route lists both for a bool column (its colour slots): name the one present.
            vals = (await _get(f"{_ds(dataset_id)}/metadata/{c['name']}", format="json"))["values"]
            entry["values"] = sorted({str(v) for v in vals if v is not None})
        cols.append(entry)
    out = pick(d, "id", "name", "status", "n_cells", "n_genes", "error_message", "active_clustering",
               "user_annotation_columns", "app_annotation_columns", "available_embeddings",
               "default_expression_layer")
    out["expression_layers"] = [
        (layer.get("key") or layer.get("name")) if isinstance(layer, dict) else layer
        for layer in d.get("expression_layers") or []]
    large = d.get("large")
    out["large_mode"] = bool(large) if not isinstance(large, dict) else bool(large.get("large", True))
    out["modalities"] = [k for k in ("protein", "atac", "perturbation", "spatial") if d.get(k)]
    if d.get("status") == "ready":
        try:
            recipe = await api.get(f"{_ds(dataset_id)}/recipe")
            out["steps_done"] = [s.get("step") for s in recipe.get("recipe") or []]
        except api.RouteError:
            pass
    out["obs_columns"] = cols
    return dumps(out)


@tool(READ)
async def describe_column(dataset_id: DatasetId,
                          column: Annotated[str, Field(description="An obs (cell metadata) column.")],
                          limit: Limit = DEFAULT_LIMIT) -> str:
    """One obs column: each value's cell count (largest first) for a category, or a
    numeric summary (min, quartiles, max, mean, missing)."""
    d = await _get(f"{_ds(dataset_id)}/metadata/{column}", format="json")
    s = pd.Series(d["values"])
    out: dict[str, Any] = {"dataset_id": dataset_id, "column": column, "n_cells": int(len(s)),
                           "n_missing": int(s.isna().sum())}
    numeric = pd.to_numeric(s, errors="coerce")
    if s.notna().any() and numeric.notna().sum() == s.notna().sum() and s.map(type).ne(bool).all() \
            and s.nunique() > 12:
        q = numeric.quantile([0, 0.25, 0.5, 0.75, 1]).tolist()
        out.update(kind="numeric", min=q[0], q25=q[1], median=q[2], q75=q[3], max=q[4],
                   mean=float(numeric.mean()))
    else:
        counts = s.astype(str).where(s.notna()).value_counts()
        out.update(kind="categories", n_values=int(len(counts)),
                   counts=[[str(k), int(v)] for k, v in counts.head(clamp(limit)).items()])
        if len(counts) > clamp(limit):
            out["n_shown"] = clamp(limit)
    return dumps(out)


@tool(READ)
async def crosstab(dataset_id: DatasetId,
                   row: Annotated[str, Field(description="Obs column for the rows.")],
                   col: Annotated[str, Field(description="Obs column for the columns.")],
                   limit: Limit = DEFAULT_LIMIT) -> str:
    """Cell counts for every pair of values of two obs columns (e.g. donor x condition,
    cell type x condition): which donors are in which condition, and how many cells."""
    d = await _get(f"{_ds(dataset_id)}/metadata/crosstab", row=row, col=col)
    return dumps({"dataset_id": dataset_id, **d}, _max_chars(limit))


@tool(READ)
async def memory_status() -> str:
    """The memory scView may use, uses now and has left (bytes, and GB), and whether the
    previous backend was killed (most often for memory)."""
    d = await _get("/system/memory")
    out = dict(d)
    for k in ("limit_bytes", "used_bytes", "available_bytes"):
        if isinstance(d.get(k), (int, float)):
            out[k.replace("_bytes", "_gb")] = round(d[k] / 2**30, 2)
    return dumps(out)


# --- public datasets (CELLxGENE Discover) -----------------------------------------------------

@tool(WEB_READ)
async def search_public_datasets(
    query: Annotated[str | None, Field(description="Words in the title or collection.")] = None,
    organism: Annotated[str | None, Field(description="e.g. Homo sapiens")] = None,
    tissue: Annotated[str | None, Field(description="e.g. blood")] = None,
    disease: Annotated[str | None, Field(description="e.g. COVID-19")] = None,
    assay: Annotated[str | None, Field(description="e.g. 10x 3' v3")] = None,
    max_mb: Annotated[float | None, Field(gt=0, description="Largest download, MB.")] = None,
    sort: Annotated[Literal["cells", "size", "title", "newest"], Field(
        description="Order: most cells first (default), largest download, title, or newest.")] = "cells",
    limit: Limit = DEFAULT_LIMIT,
) -> str:
    """Search CELLxGENE Discover's public datasets (listing cached a day). Import one with
    import_public_dataset and its dataset_version_id."""
    d = await _get("/public-import/discover/datasets", q=query, organism=organism, tissue=tissue,
                   disease=disease, assay=assay, max_mb=max_mb, sort=sort, limit=clamp(limit))
    found = d.get("datasets") or d.get("results") or []
    keep = ("dataset_version_id", "dataset_id", "title", "collection_name", "cell_count", "size_bytes",
            "organism", "tissue", "disease", "assay", "citation")
    out = []
    for r in found:
        e = pick(r, *keep)
        if "size_bytes" in e:
            e["size_mb"] = round(e.pop("size_bytes") / 2**20, 1)
        for k in ("organism", "tissue", "disease", "assay"):
            if isinstance(e.get(k), list):
                e[k] = e[k][:5]
        out.append(e)
    return dumps({"datasets": out, "n_total": d.get("total", d.get("n_total", len(out))),
                  "fetched_at": d.get("fetched_at")}, _max_chars(limit))


def _line(event: str, data: Any) -> str | None:
    """A job's progress line from one stream event."""
    if not isinstance(data, dict):
        return f"{event}: {str(data)[:200]}"
    if event == "step_start":
        return f"step {data.get('index', 0) + 1}/{data.get('total')}: {data.get('step')} running"
    if event == "step_done":
        how = "skipped" if data.get("skipped") else f"done in {num(data.get('elapsed') or 0)} s"
        return f"step {data.get('index', 0) + 1}: {data.get('step')} {how}"
    if event == "step_error":
        return f"step {data.get('step')} failed: {data.get('error')}"
    if event == "substep":
        return f"{data.get('step')}: {data.get('message') or data.get('label') or data.get('name') or ''}"[:200]
    if event in ("complete", "result"):
        return "finished; reading the result back"
    msg = data.get("message") or data.get("stage") or data.get("step") or data.get("error")
    if event == "progress" and data.get("fraction") is not None:
        msg = f"{msg or ''} {round(100 * float(data['fraction']))}%".strip()
    return f"{event}: {msg}"[:200] if msg else None


@tool(WEB_WRITE)
async def import_public_dataset(
    dataset_version_id: Annotated[str, Field(description="From search_public_datasets.")],
    dataset_id: Annotated[str | None, Field(
        description="Its dataset_id: needed only for a version no longer current.")] = None,
    name: Annotated[str | None, Field(description="Name in the library (default: the title).")] = None,
) -> str:
    """Download a CELLxGENE Discover dataset into the library (a job: returns job_id at once;
    follow it with job_status; the new dataset's id is in the result)."""
    body = {"dataset_version_id": dataset_version_id, "dataset_id": dataset_id, "name": name}

    async def cancel(job: jobs.Job) -> None:
        jid = (job.events.get("job") or {}).get("job_id")
        if not jid:
            raise api.RouteError(404, "not started")
        await api.post(f"/public-import/jobs/{jid}/cancel")

    def finish(job: jobs.Job) -> dict[str, Any]:
        return pick(job.events.get("complete"), "dataset_id")

    job = jobs.new("import", None, "/public-import/discover/import-stream", body,
                   progress_line=_line, finish=finish, cancel_route=cancel)
    return dumps({"job_id": job.id, "status": job.status,
                  "next": "job_status(job_id, wait_seconds=50) until complete"})


# --- QC and processing ---------------------------------------------------------------------------

@tool(READ)
async def assess_dataset(dataset_id: DatasetId) -> str:
    """Which processing steps the data already has (raw counts? normalised? PCA, clusters,
    UMAP?), the species, and the steps scView recommends running next."""
    d = await _get(f"{_ds(dataset_id)}/assessment")
    done, not_done, cautions = {}, {}, []
    for k, v in d.items():
        if isinstance(v, dict) and "done" in v:
            (done if v["done"] else not_done)[k] = v.get("details")
            if v.get("caution"):
                cautions.append(f"{k}: {v['caution']}")
    rest = {k: v for k, v in d.items() if not (isinstance(v, dict) and "done" in v) and v is not None}
    return dumps({"dataset_id": dataset_id, "done": done, "not_done": not_done, "cautions": cautions, **rest})


@tool(READ)
async def qc_filter_preview(
    dataset_id: DatasetId,
    mode: Annotated[Literal["mad", "fixed"] | None, Field(
        description="mad (default): each sample's outliers; fixed: the cut-offs below.")] = None,
    min_genes: Annotated[float | None, Field(description="Fixed mode: fewest genes per cell (default 200).")] = None,
    max_genes: Annotated[float | None, Field(description="Fixed mode: most genes per cell (0 = no cap).")] = None,
    min_counts: Annotated[float | None, Field(description="Fixed mode: fewest counts per cell (0 = none).")] = None,
    max_pct_mt: Annotated[float | None, Field(description="% mito ceiling (default 20).")] = None,
    mad_k: Annotated[float | None, Field(description=(
        "MAD mode: cut counts and genes per cell below each sample's median minus k MADs (log1p; default 5)."))] = None,
    mad_k_mt: Annotated[float | None, Field(description="MAD mode: % mito above median + k MADs (default 3).")] = None,
    drop_doublets: Annotated[bool | None, Field(description="Also drop the cells called doublets (default no).")] = None,
    sample_column: Annotated[str, Field(description="The samples (default: detected).")] = "",
    limit: Limit = DEFAULT_LIMIT,
) -> str:
    """The cells the filtering step would remove with these thresholds (nothing is changed):
    overall, per rule, per sample and per cluster, with warnings for a sample or cluster
    mostly lost. Thresholds left out take the pipeline's defaults."""
    d = await _get(f"{_ds(dataset_id)}/qc/filter-preview", mode=mode, min_genes=min_genes,
                   max_genes=max_genes, min_counts=min_counts, max_pct_mt=max_pct_mt, mad_k=mad_k,
                   mad_k_mt=mad_k_mt, drop_doublets=drop_doublets, sample_column=sample_column)
    n = clamp(limit)
    out = {"dataset_id": dataset_id, **pick(d, "n_cells", "n_removed", "n_kept", "fraction_removed",
                                            "thresholds", "has_mt", "by_criterion", "warnings", "doublets",
                                            "grid_note", "tissue_cells_note")}
    samples = d.get("samples") or {}
    if samples.get("groups"):
        out["samples"] = {"column": samples.get("column"), **table(
            samples["groups"], ["name", "n_cells", "n_removed", "fraction", "median_genes", "median_counts",
                                "median_pct_mt", "flags"], n)}
    mad = d.get("mad") or {}
    if mad.get("samples"):
        out["mad_thresholds"] = {"column": mad.get("column"), **table(
            mad["samples"], ["name", "min_counts", "max_counts", "min_genes", "max_genes", "max_pct_mt"], n)}
    clusters = d.get("clusters") or {}
    if clusters.get("groups"):
        out["clusters"] = {"column": clusters.get("column"), **table(
            clusters["groups"], ["name", "n_cells", "n_removed", "fraction"], n)}
    audit = d.get("audit") or {}
    flagged = [dict(t, **(t.get("test") or {})) for t in audit.get("by_type") or []
               if (t.get("test") or {}).get("p") is not None and t["test"]["p"] < 0.05]
    if audit:
        out["removal_by_type_and_condition"] = {
            "label": (audit.get("label") or {}).get("column"),
            "condition": (audit.get("condition") or {}).get("column"),
            "flagged": table(flagged, ["name", "n_cells", "n_removed", "high", "high_rate", "low",
                                       "low_rate", "max_diff_pp", "p"], n)}
    return dumps(out, _max_chars(limit))


Steps = Annotated[list[str], Field(min_length=1, description=(
    "Pipeline steps, run in the pipeline's order. A full run from counts: qc_metrics, "
    "doublet_detection, filtering, normalization, log_transform, highly_variable_genes, scaling, pca, "
    "batch_correction (with params.batch_key), neighbors, clustering, embeddings, marker_genes. "
    "Others: reset_to_counts, ambient_correction, uncorrected_embedding, enrichment, cell_cycle, "
    "dissociation_stress, cell_type_annotation."))]
Params = Annotated[dict[str, Any] | None, Field(description=(
    "Pipeline parameters (only the ones to change; unknown names are refused)."))]


def _check_params(p: dict[str, Any] | None) -> dict[str, Any]:
    p = dict(p or {})
    msg = params.unknown_message(list(p))
    if msg:
        raise ToolError(msg)
    try:
        params.PipelineParamsInput(**p)
    except Exception as e:  # pydantic ValidationError: a value of the wrong type
        raise ToolError(f"Invalid pipeline parameter value: {e}") from None
    return p


def _check_steps(steps: list[str]) -> None:
    from scview.core.step_names import unknown_steps, unknown_steps_message

    bad = unknown_steps(steps)
    if bad:
        raise ToolError(unknown_steps_message(bad))


@tool(READ)
async def sex_check(
    dataset_id: DatasetId,
    column: Annotated[str, Field(description="The samples / donors (default: detected).")] = "",
    limit: Limit = DEFAULT_LIMIT,
) -> str:
    """Each sample's sex from its expression (XIST vs Y-chromosome genes): female / male /
    mixed / ambiguous / undetermined with the evidence, against any sex metadata column, and the
    warnings: mismatches (possible swaps), mixed samples, and sex confounded with a condition
    (then XIST and Y genes top any comparison between them)."""
    d = await _get(f"{_ds(dataset_id)}/qc/sex", column=column)
    out = {"dataset_id": dataset_id,
           **pick(d, "column", "reason", "species", "metadata_column", "female_genes", "male_genes",
                  "source", "summary", "warnings", "notes"),
           "samples": table(d.get("groups") or [], ["name", "n_cells", "call", "confidence", "metadata",
                                                    "mismatch", "frac_female", "frac_male", "evidence"],
                            clamp(limit)),
           "confounding": [pick(c, "condition", "kind", "levels") for c in d.get("confounding") or []]}
    return dumps(out, _max_chars(limit))


@tool(READ)
async def preflight(dataset_id: DatasetId, steps: Steps, params: Annotated[dict[str, Any] | None, Field(
        description="The run's parameters, as run_pipeline takes them (its schema lists them).")] = None) -> str:
    """Before a run: its peak memory and rough time per step and in all, against the memory
    scView has free (verdict fits / tight / wont_fit, with suggestions), and whether it would
    run in large-dataset mode. Nothing is run."""
    _check_steps(steps)
    p = _check_params(params)
    query: dict[str, Any] = {"steps": ",".join(steps)}
    for k in ("n_top_genes", "n_comps", "n_neighbors", "normalization_method", "annotation_method",
              "doublet_channel_key", "doublet_method", "chunked", "clustering_method",
              "leiden_iterations", "umap_3d", "ambient_rho"):
        if p.get(k) is not None:
            query[k] = p[k]
    if p.get("regress_out"):
        query["regress_out"] = ",".join(p["regress_out"])
    d = await _get(f"{_ds(dataset_id)}/preflight", **query)
    gb = 2**30
    steps_ = [{"step": s.get("step"), "peak_gb": round((s.get("peak_bytes") or 0) / gb, 2),
               "seconds": s.get("seconds")} for s in d.get("steps") or []]
    head = d.get("header") or {}
    out = {"dataset_id": dataset_id, "verdict": d.get("verdict"),
           **pick(d, "mode", "leiden", "umap_3d", "in_memory", "in_memory_estimate", "regress_out"),
           "data": pick(head, "n_cells", "n_genes", "format", "backed", "large", "unit"),
           "steps": table(steps_, ["step", "peak_gb", "seconds"], 40)}
    return dumps(out)


def _pipeline_result(job: jobs.Job) -> dict[str, Any]:
    r = job.events.get("complete") or {}
    out = pick(r, "status", "stop_reason", "steps_run", "steps_skipped", "skipped", "errors",
               "notes", "cleared", "elapsed_seconds", "unknown_steps", "chunked", "mode")
    out["dataset_id"] = job.dataset_id
    out["note"] = "Saved: the open browser UI shows the new state after a reload of the dataset."
    return out


async def _start_run(dataset_id: str, steps: list[str], p: dict[str, Any], kind: str,
                     finish: Callable = _pipeline_result) -> jobs.Job:
    if any(j.dataset_id == dataset_id for j in jobs.running()):
        busy = next(j for j in jobs.running() if j.dataset_id == dataset_id)
        raise ToolError(f"Job {busy.id} ({busy.kind}) is still running on this dataset; "
                        "wait for it with job_status or stop it with cancel_job.")

    async def cancel(job: jobs.Job) -> None:
        await api.post(f"{_ds(dataset_id)}/assessment/cancel")

    return jobs.new(kind, dataset_id, f"{_ds(dataset_id)}/assessment/run-stream",
                    {"steps": steps, "params": p}, progress_line=_line, finish=finish,
                    cancel_route=cancel)


@tool(WRITE)
async def run_pipeline(dataset_id: DatasetId, steps: Steps, params: Params = None) -> str:
    """Run pipeline steps on a dataset and save the result (a job: returns job_id at once;
    follow it with job_status, stop it with cancel_job). Check preflight first on a big dataset.
    The original upload is never changed. Unknown step or parameter names are refused."""
    _check_steps(steps)
    p = _check_params(params)
    job = await _start_run(dataset_id, steps, p, "pipeline")
    return dumps({"job_id": job.id, "status": job.status, "dataset_id": dataset_id, "steps": steps,
                  "params": p, "next": "job_status(job_id, wait_seconds=50) until complete"})


@tool(READ)
async def get_recipe(dataset_id: DatasetId) -> str:
    """The dataset's recorded processing, in order: each step with its parameters and how the
    replay script reproduces it, and the source the data came from."""
    d = await _get(f"{_ds(dataset_id)}/recipe")
    steps = []
    for s in d.get("recipe") or []:
        e = pick(s, "step", "tool", "params", "timestamp")
        e["params"] = {k: v if len(json.dumps(v, default=str)) <= 200
                       else f"({len(json.dumps(v, default=str))} chars: see the UI's History)"
                       for k, v in (e.get("params") or {}).items()}
        rep = s.get("replay")
        if isinstance(rep, dict):
            e["replay"] = rep.get("status")
        steps.append(e)
    return dumps({"dataset_id": dataset_id, "steps": steps, "source": d.get("source")})


@tool(READ)
async def get_replay_script(
    dataset_id: DatasetId,
    offset: Annotated[int, Field(ge=0, description="Character to start from (page through a long script).")] = 0,
    max_chars: Annotated[int, Field(ge=500, le=60000, description="Page length in characters.")] = 7000,
) -> str:
    """The standalone Python (scanpy) script that replays the recorded steps from the input
    file, one page at a time (it is often longer than one page)."""
    _, data = await _raw(f"{_ds(dataset_id)}/recipe/script")
    text = data.decode("utf-8", "replace")
    end = min(len(text), offset + max_chars)
    head = (f"# scView replay script, characters {offset}-{end} of {len(text)}"
            + (f"; next page: offset={end}" if end < len(text) else "") + "\n")
    return head + text[offset:end]


# --- groups and labels ---------------------------------------------------------------------------

@tool(WRITE)
async def get_markers(
    dataset_id: DatasetId,
    column: Annotated[str, Field(description="Obs column of the groups (e.g. the clustering).")],
    group: Annotated[str | None, Field(description="One group only (default: every group).")] = None,
    n_genes: Annotated[int, Field(ge=1, le=100, description="Top genes per group.")] = 10,
) -> str:
    """Marker genes of each group of a column vs the rest (Wilcoxon), top n per group with
    log fold change, adjusted p and the share of cells expressing in and out. Computed and
    saved with the dataset the first time a column is asked for."""
    d = await _get(f"{_ds(dataset_id)}/markers", groupby_column=column, groupby=group or "",
                   format="json", n_genes=n_genes)
    if not d.get("groups"):
        return dumps({"dataset_id": dataset_id, "column": column, "groups": [],
                      "reason": d.get("reason")})
    sample = next(iter(d["markers"].values()))[0]
    cols = [c for c in ("gene", "logfoldchange", "pval_adj", "pct_in", "pct_out", "pct_nz_group",
                        "pct_nz_reference", "pts", "pts_rest", "specificity") if c in sample]
    out = {g: [[num(r.get(c)) for c in cols] for r in recs[:n_genes]] for g, recs in d["markers"].items()}
    extra = {"note": P0_NOTE} if "pval_adj" in cols and any(
        r[cols.index("pval_adj")] == 0 for rs in out.values() for r in rs) else {}
    return dumps({"dataset_id": dataset_id, "column": column, "columns": cols, "markers": out, **extra},
                 _max_chars(DEFAULT_LIMIT * max(1, n_genes // 10)))


@tool(WRITE)
async def annotate_celltypist(
    dataset_id: DatasetId,
    model: Annotated[str, Field(description="CellTypist model, e.g. Immune_All_Low.pkl (default), "
                                            "Immune_All_High.pkl, Healthy_COVID19_PBMC.pkl.")] = "Immune_All_Low.pkl",
    groupby: Annotated[str, Field(description="Clusters for the majority vote (default: the active clustering).")] = "",
    target_column: Annotated[str, Field(description="Obs column to write.")] = "celltypist",
) -> str:
    """Label cells with a CellTypist reference model (the pipeline's cell_type_annotation step;
    a job). The result names the labels' confidence and the model's gene overlap. The model is
    downloaded on first use."""
    p = {"annotation_method": "celltypist", "celltypist_model": model, "annotation_groupby": groupby,
         "annotation_target": target_column}

    async def finish(job: jobs.Job) -> dict[str, Any]:
        out = _pipeline_result(job)
        try:
            s = await api.get(f"{_ds(dataset_id)}/annotation/celltypist", column=target_column)
            out["celltypist"] = s.get("result")
        except api.RouteError as e:
            out["celltypist_summary_error"] = e.detail
        return out

    job = await _start_run(dataset_id, ["cell_type_annotation"], p, "celltypist", finish)
    return dumps({"job_id": job.id, "status": job.status, "dataset_id": dataset_id,
                  "target_column": target_column, "next": "job_status(job_id, wait_seconds=50)"})


@tool(WRITE)
async def set_labels(
    dataset_id: DatasetId,
    column: Annotated[str, Field(description="Annotation column to write (a user column).")],
    label: Annotated[str, Field(min_length=1, description="The value to give the matching cells.")],
    filter: Filter,
    new_column: Annotated[bool, Field(description=(
        "Create the column (every other cell 'Unassigned'). False: write into an existing user column."))] = False,
) -> str:
    """Give every cell matching a filter one label in a user annotation column (as a lasso and
    "Label" in the UI): name a cluster ({"kind":"category","column":"leiden","values":["3"]}),
    or build a column such as a sample id from donor AND day. Saved with the dataset."""
    idx = await _indices(dataset_id, filter)
    if len(idx) == 0:
        raise ToolError("No cell matches the filter; nothing was labelled.")
    r = await _post(f"{_ds(dataset_id)}/labels", {"column": column, "label": label,
                                                    "indices": idx.tolist(), "new_column": new_column})
    out = pick(r, "column", "label", "created")
    out.update(dataset_id=dataset_id, n_labelled=int(len(idx)),
               n_changed=sum(len(g.get("indices") or []) for g in r.get("previous") or []))
    return dumps(out)


@tool(WRITE)
async def subcluster(
    dataset_id: DatasetId,
    column: Annotated[str, Field(description="Obs column of the groups.")],
    group: Annotated[str, Field(description="The group to split.")],
    resolution: Annotated[float, Field(gt=0, le=5, description="Leiden resolution on the group's own graph.")] = 0.5,
) -> str:
    """Split one group into sub-clusters on its own variable genes, PCA and graph, saved as
    the column '<column>_sub' (other cells keep their group)."""
    r = await _post(f"{_ds(dataset_id)}/clustering/subcluster",
                    {"column": column, "group": group, "resolution": resolution})
    return dumps({"dataset_id": dataset_id, **r})


# --- statistics -----------------------------------------------------------------------------------

def _de_warnings(r: dict[str, Any]) -> list[str]:
    w = []
    conf = r.get("confounding")
    if conf:
        w.append("confounding: " + str(conf.get("message") or conf))
    if (r.get("sex") or {}).get("message"):
        w.append("sex: " + r["sex"]["message"])
    comp = r.get("composition") or {}
    if comp.get("warn"):
        w.append("composition: " + str(comp.get("message")))
    if r.get("underreplicated"):
        w.append(f"under-replicated: donors A={r.get('n_donors_a')}, B={r.get('n_donors_b')} "
                 f"(replicate column {r.get('replicate_key')}); a cell-level p value overstates "
                 "the evidence, check with method='pseudobulk' or per_donor_expression")
    ts = r.get("transcriptome_size") or {}
    if ts.get("caution"):
        w.append(f"transcriptome size differs {num(ts.get('ratio'))}x between the sides "
                 f"(median counts {num(ts.get('median_a'))} vs {num(ts.get('median_b'))})")
    if r.get("dropped_samples"):
        w.append(f"samples left out: {', '.join(map(str, r['dropped_samples'][:10]))}")
    return w


@tool(READ)
async def differential_expression(
    dataset_id: DatasetId,
    column: Annotated[str | None, Field(description=(
        "Between two groups: the obs column holding them (e.g. condition); with group_a and "
        "group_b. One group vs the rest: the column of `group`."))] = None,
    group_a: Annotated[str | None, Field(description="Between groups: side A (e.g. STIM).")] = None,
    group_b: Annotated[str | None, Field(description="Between groups: side B, the reference (e.g. CTRL).")] = None,
    within_column: Annotated[str | None, Field(description="Restrict to one subgroup: its column (e.g. cell_type).")] = None,
    within_value: Annotated[str | None, Field(description="Restrict to one subgroup: its value (e.g. CD14 Mono).")] = None,
    group: Annotated[str | None, Field(description="One group vs every other cell (with `column`).")] = None,
    filter: Annotated[CellFilter | None, Field(description="Selected cells vs every other cell.")] = None,
    method: Annotated[Literal["wilcoxon", "t-test", "logreg", "pseudobulk"], Field(description=(
        "wilcoxon / t-test (Welch) / logreg are cell-level. pseudobulk sums each donor's counts "
        "and tests with DESeq2 (or voom), the donor-aware test for conditions; between groups only."))] = "wilcoxon",
    replicate_key: Annotated[str | None, Field(description="Donor/sample column (default: detected).")] = None,
    pb_method: Annotated[Literal["deseq2", "voom", "voombygroup"], Field(
        description="Pseudobulk model: DESeq2 (default), limma-voom, or voomByGroup (a variance per group).")] = "deseq2",
    min_pct: Annotated[float | None, Field(ge=0, le=1, description="Cell-level: test genes detected in at least this share of either side (default 0.01).")] = None,
    padj_max: Annotated[float, Field(gt=0, le=1, description="Count and list genes at adjusted p at most this.")] = 0.05,
    lfc_min: Annotated[float, Field(ge=0, description="And |log2 fold change| at least this.")] = 0.25,
    limit: Limit = DEFAULT_LIMIT,
) -> str:
    """Differential expression: A vs B of one column (optionally within one cell type), one
    group vs the rest, or filtered cells vs the rest. Returns the result_id (for enrichment and
    get_result), counts significant at padj_max and |log2FC| >= lfc_min, the top up and down
    genes, and the route's warnings (confounding with batch, cell-type composition shift,
    too few donors, transcriptome size). Across conditions prefer method='pseudobulk'."""
    if group_a is not None or group_b is not None:
        if not (column and group_a and group_b):
            raise ToolError("Between groups needs column, group_a and group_b.")
        body: dict[str, Any] = {"split_by": column, "group_a": group_a, "group_b": group_b,
                                "within_column": within_column, "within_value": within_value,
                                "replicate_key": replicate_key, "method": method, "pb_method": pb_method}
        if min_pct is not None:
            body["min_pct"] = min_pct
        r = await _post(f"{_ds(dataset_id)}/de/compare", body)
    else:
        if method == "pseudobulk":
            raise ToolError("Pseudobulk compares two groups: give column, group_a and group_b.")
        if filter is None:
            if not (column and group):
                raise ToolError("Give column + group_a + group_b, column + group, or a filter.")
            filter = CellFilter(op="and", items=[{"kind": "category", "column": column, "values": [group]}])
        idx = await _indices(dataset_id, filter)
        body = {"indices": idx.tolist(), "label": group or "selection", "method": method}
        if min_pct is not None:
            body["min_pct"] = min_pct
        r = await _post(f"{_ds(dataset_id)}/de", body)
    genes = r.get("genes") or []
    has_p = any(g.get("pval_adj") is not None for g in genes)

    def sig(g: dict[str, Any]) -> bool:
        lfc = g.get("logfoldchange") or 0.0
        p = g.get("pval_adj")
        return abs(lfc) >= lfc_min and (p is not None and p <= padj_max if has_p else True)

    def key(g: dict[str, Any]) -> tuple:
        p = g.get("pval_adj")
        return (p if p is not None else 1.0, -abs(g.get("logfoldchange") or 0.0))

    up = sorted((g for g in genes if (g.get("logfoldchange") or 0) > 0 and sig(g)), key=key)
    down = sorted((g for g in genes if (g.get("logfoldchange") or 0) < 0 and sig(g)), key=key)
    if not has_p:  # logistic regression: no p values; rank by coefficient
        up = sorted(up, key=lambda g: -(g.get("stat") or 0))
        down = sorted(down, key=lambda g: g.get("stat") or 0)
    n = clamp(limit)
    cols = ["gene", "logfoldchange", "pval_adj", "pval"] if has_p else ["gene", "logfoldchange", "stat"]
    out: dict[str, Any] = {
        "dataset_id": dataset_id, "result_id": r.get("result_id"), "label": r.get("label"),
        "method": r.get("method"), "n_a": r.get("n_selected"), "n_b": r.get("n_rest"),
        **pick(r, "replicate_key", "n_donors_a", "n_donors_b", "n_samples_a", "n_samples_b",
               "design", "pb_method", "counts_source", "min_pct", "n_genes_skipped"),
        "n_genes_tested": r.get("n_genes_tested") or len(genes),
        "rule": f"padj <= {padj_max} and |log2FC| >= {lfc_min}" if has_p else f"|coef| ranked, |log2FC| >= {lfc_min}",
        "n_up": len(up), "n_down": len(down), "columns": cols,
        "up": [[num(g.get(c)) for c in cols] for g in up[:n]],
        "down": [[num(g.get(c)) for c in cols] for g in down[:n]],
        "warnings": _de_warnings(r),
    }
    if (out["method"] != "pseudobulk" and not r.get("underreplicated")
            and (r.get("n_donors_a") or 0) >= 2 and (r.get("n_donors_b") or 0) >= 2):
        # The UI's caption says the same: a cell-level test treats cells as independent.
        out["warnings"].append(
            f"cell-level test across donors ({r.get('n_donors_a')} vs {r.get('n_donors_b')}): cells are "
            "treated as independent, so p values are optimistic when the sides also differ by donor; "
            "confirm with method='pseudobulk' or per_donor_expression")
    if any(g.get("pval") == 0 for g in up[:n] + down[:n]):
        out["note"] = P0_NOTE
    comp = r.get("composition") or {}
    if comp.get("groups") and comp.get("warn"):
        out["composition_shift"] = [pick(g, "group", "share_a", "share_b", "diff_points", "p")
                                    for g in comp["groups"][:5] if g.get("flagged")]
    return dumps(out, _max_chars(limit))


@tool(READ)
async def compare_de_results(
    dataset_id: DatasetId,
    result_a: Annotated[str, Field(description="A differential_expression result_id (e.g. the cell-level one).")],
    result_b: Annotated[str, Field(description="Another result_id (e.g. pseudobulk on the same comparison).")],
    padj_max: Annotated[float, Field(gt=0, le=1, description="Significant at adjusted p at most this.")] = 0.05,
    lfc_min: Annotated[float, Field(ge=0, le=20, description="And |log2FC| at least this.")] = 0.25,
    limit: Limit = DEFAULT_LIMIT,
) -> str:
    """Compare two DE results' significant genes (the DE overlap the UI draws): how many each
    calls, how many both call, their sign agreement, and the top genes only one of them calls
    (largest |log2FC| first). Use case: cell-level Wilcoxon vs donor-level pseudobulk."""
    d = await _post(f"{_ds(dataset_id)}/de/overlap", {
        "result_ids": [result_a, result_b], "direction": "both", "padj_max": padj_max, "lfc_min": lfc_min})
    n = clamp(limit)
    sets = d.get("sets") or []
    pair = (d.get("pairs") or [{}])[0]
    by_sets = {tuple(i["sets"]): i for i in d.get("intersections") or []}

    def genes(key: tuple[int, ...]) -> dict[str, Any]:
        i = by_sets.get(key) or {"n": 0, "n_up": 0, "n_down": 0, "genes": []}
        recs = [{"gene": g["gene"], "lfc_a": (g.get("lfc") or [None, None])[0],
                 "lfc_b": (g.get("lfc") or [None, None])[1]} for g in i.get("genes") or []]
        return {**pick(i, "n", "n_up", "n_down"), **table(recs, ["gene", "lfc_a", "lfc_b"], n)}

    shared, disc = pair.get("shared", 0), pair.get("discordant", 0)
    out = {
        "dataset_id": dataset_id, "rule": f"padj <= {padj_max} and |log2FC| >= {lfc_min}",
        "a": pick(sets[0], "name", "method", "n", "n_up", "n_down", "n_tested") if sets else None,
        "b": pick(sets[1], "name", "method", "n", "n_up", "n_down", "n_tested") if len(sets) > 1 else None,
        "n_common_tested": d.get("n_common_tested"),
        "both": shared, "discordant_sign": disc,
        "sign_agreement": round(shared / (shared + disc), 4) if shared + disc else None,
        "jaccard": pair.get("jaccard"),
        "only_a": genes((0,)), "only_b": genes((1,)),
        "note": "lfc_* is null where that result did not test the gene (under its detection "
                "cut-off or the pseudobulk count filter).",
        "notes": d.get("notes") or [],
    }
    if disc:
        out["discordant_genes"] = [g["gene"] for g in (d.get("discordant") or {}).get("genes", [])[:n]]
    return dumps(out, _max_chars(limit))


@tool(READ)
async def per_donor_expression(
    dataset_id: DatasetId,
    gene: Annotated[str, Field(description="One gene.")],
    groupby: Annotated[str, Field(description="Obs column of the groups (default: the active clustering).")] = "",
    sample: Annotated[str, Field(description="Donor / sample column (default: detected).")] = "",
    condition: Annotated[str, Field(description="Condition column to split each group by.")] = "",
    method: Annotated[Literal["mean", "cpm"], Field(description=(
        "mean of the log-normalised values per donor, or cpm: log2 CPM of the donor's summed counts."))] = "mean",
    groups: Annotated[list[str] | None, Field(description="Only these groups (default: all).")] = None,
    conditions: Annotated[list[str] | None, Field(description=(
        "The two condition values to test (default: the column's two, or its two besides "
        "'Unassigned'); other values are still listed, not tested."))] = None,
    reference: Annotated[str | None, Field(description=(
        "Baseline of the two: mean_diff is the other minus this (default: the first in the column's order)."))] = None,
    limit: Limit = DEFAULT_LIMIT,
) -> str:
    """One gene's pseudobulk value per donor in each group (and condition), with the design
    (paired when donors are in both tested conditions) and the per-group donor-level test
    (paired Wilcoxon signed-rank + paired t, or unpaired) of two conditions; `difference`
    names the sign of mean_diff: the donor-aware check of a DE gene."""
    d = await _get(f"{_ds(dataset_id)}/expression/per-donor", gene=gene, groupby=groupby,
                   sample=sample, condition=condition, method=method, groups=groups or None,
                   conditions=conditions or None, reference=reference or None)
    points = d.pop("points", None) or []
    if not d.get("groups_not_found"):
        d.pop("groups_not_found", None)
    # Per group x condition: the donors' values (compact: donor -> value).
    table: dict[str, dict[str, dict[str, Any]]] = {}
    for p in points:
        g, c = str(p.get("group")), str(p.get("condition", ""))
        donor = str(p.get("sample") or p.get("donor") or p.get("sample_id"))
        table.setdefault(g, {}).setdefault(c or "all", {})[donor] = num(p.get("value"))
    out = {"dataset_id": dataset_id, **d, "values_by_group": dict(list(table.items())[: clamp(limit)])}
    if len(table) > clamp(limit):
        out["n_groups"] = len(table)
    return dumps(out, _max_chars(limit))


@tool(READ)
async def composition_test(
    dataset_id: DatasetId,
    condition_key: Annotated[str, Field(description="Obs column of the conditions.")],
    group_key: Annotated[str, Field(description="Obs column of the cell types / clusters.")],
    sample_key: Annotated[str | None, Field(description="Donor / sample column (default: detected).")] = None,
    conditions: Annotated[list[str] | None, Field(description="Values to compare (default: all).")] = None,
    reference: Annotated[str | None, Field(description="Baseline condition (default: the first).")] = None,
    transform: Annotated[Literal["logit", "asin"], Field(description=(
        "Scale the per-sample shares are tested on: logit (propeller's default) or arcsine square root."))] = "logit",
    paired: Annotated[bool | None, Field(description="Default: paired when every donor is in every condition.")] = None,
    limit: Limit = DEFAULT_LIMIT,
) -> str:
    """Do cell types make up different shares of the cells between conditions, with donors as
    replicates (propeller-style: per-sample proportions, limma moderated t/F, BH FDR)."""
    body = {"condition_key": condition_key, "group_key": group_key, "sample_key": sample_key,
            "conditions": conditions, "reference": reference, "transform": transform, "paired": paired}
    d = await _post(f"{_ds(dataset_id)}/composition/test", body)
    groups = [_flat(g) for g in d.pop("groups", []) or []]
    d.pop("samples", None)  # per-sample shares: in get_result's table
    cols = list(dict.fromkeys(k for g in groups for k in g))
    out = {"dataset_id": dataset_id, **d, "groups": table(groups, cols, clamp(limit))}
    return dumps(out, _max_chars(limit))


def _flat(d: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    """Nested dicts as one level: {"props": {"CTRL": 0.1}} -> {"props_CTRL": 0.1}."""
    out: dict[str, Any] = {}
    for k, v in d.items():
        if isinstance(v, dict):
            out.update(_flat(v, f"{prefix}{k}_"))
        else:
            out[f"{prefix}{k}"] = v
    return out


def _milo_result(job: jobs.Job) -> dict[str, Any]:
    r = dict(job.events.get("result") or {})
    nh = r.pop("nhoods", None) or {}
    for k in ("layouts", "edges", "index_cell", "seed", "timings"):
        r.pop(k, None)
    out = {"dataset_id": job.dataset_id, **r}
    if isinstance(nh, dict) and nh.get("spatial_fdr"):
        cols = ["label", "label_fraction", "size", "log_fc", "pval", "spatial_fdr"]
        recs = [{c: (nh.get(c) or [None] * (i + 1))[i] for c in cols} for i in range(len(nh["spatial_fdr"]))]
        recs.sort(key=lambda x: 1.0 if x["spatial_fdr"] is None else x["spatial_fdr"])
        out["top_neighbourhoods"] = table(recs, cols, 15)
    return out


@tool(READ)
async def milo(
    dataset_id: DatasetId,
    condition_key: Annotated[str, Field(description="Obs column of the conditions.")],
    sample_key: Annotated[str | None, Field(description="Donor / sample column (default: detected).")] = None,
    conditions: Annotated[list[str] | None, Field(description="[reference, tested] (default: the first two).")] = None,
    covariates: Annotated[list[str] | None, Field(description="Per-sample columns in the design (batch, sex).")] = None,
    paired: Annotated[bool | None, Field(description=(
        "Donors in both conditions as pairs (default: paired when every donor is in both)."))] = None,
    label_key: Annotated[str | None, Field(description="Cell types to label neighbourhoods by.")] = None,
    prop: Annotated[float | None, Field(gt=0, le=1, description="Share of cells sampled as index cells.")] = None,
) -> str:
    """Milo neighbourhood differential abundance between two conditions, donors as replicates,
    no clusters needed (a job: returns job_id; job_status gives the result: neighbourhoods
    significant at spatial FDR and the labels they fall in). The dataset is not changed."""
    body = {"condition_key": condition_key, "sample_key": sample_key, "conditions": conditions,
            "covariates": covariates, "paired": paired, "label_key": label_key, "prop": prop}
    plan = await _post(f"{_ds(dataset_id)}/milo/plan", body)
    job = jobs.new("milo", dataset_id, f"{_ds(dataset_id)}/milo/run", body,
                   progress_line=_line, finish=_milo_result)
    return dumps({"job_id": job.id, "status": job.status, "plan": plan,
                  "next": "job_status(job_id, wait_seconds=50)"})


@tool(READ)
async def expression_by_group(
    dataset_id: DatasetId,
    genes: Annotated[list[str], Field(min_length=1, max_length=300, description="Gene symbols, up to 300.")],
    groupby: Annotated[str, Field(description="Obs column (default: the active clustering).")] = "",
    layer: Annotated[str, Field(description=(
        "Expression layer (default: the dataset's own default, which can be raw COUNTS, e.g. 'raw'; "
        "the result names the layer used); get_dataset lists the layers."))] = "",
    limit: Limit = DEFAULT_LIMIT,
) -> str:
    """Each gene's mean expression and share of expressing cells in each group (the dot plot's
    numbers). mean is over all of a group's cells; fraction is the share with a value above 0.
    The default layer may be raw counts (then mean is mean counts): the result names it."""
    d = await _get(f"{_ds(dataset_id)}/expression/by-group", genes=genes, groupby=groupby, layer=layer)
    g = d["groups"][: clamp(limit)]
    out = {"dataset_id": dataset_id, "groupby": d["groupby"], "layer": d["layer"], "genes": d["genes"],
           "groups": g, "mean": d["mean"][: len(g)], "fraction": d["fraction"][: len(g)],
           "note": "mean[i][j] and fraction[i][j] are group i, gene j"}
    if len(d["groups"]) > len(g):
        out["n_groups"] = len(d["groups"])
    missing = [x for x in genes if x not in d["genes"]]
    if missing:
        out["genes_missing"] = missing
    return dumps(out, _max_chars(limit))


async def _de_table(dataset_id: str, result_id: str) -> pd.DataFrame:
    _, data = await _raw(f"{_ds(dataset_id)}/results/{result_id}/table.tsv.gz")
    try:
        data = gzip.decompress(data)
    except OSError:
        pass  # already plain
    return pd.read_csv(io.BytesIO(data), sep="\t")


def _col(df: pd.DataFrame, *names: str) -> str | None:
    return next((n for n in names if n in df.columns), None)


@tool(READ)
async def enrichment(
    dataset_id: DatasetId,
    de_result_id: Annotated[str | None, Field(description="A differential_expression result_id.")] = None,
    genes: Annotated[list[str] | None, Field(description="A gene list (ORA against the dataset's genes).")] = None,
    column: Annotated[str | None, Field(description="Marker group: its obs column (with group).")] = None,
    group: Annotated[str | None, Field(description="Marker group: the group (its top markers).")] = None,
    method: Annotated[Literal["ora", "gsea"], Field(description=(
        "ora: hypergeometric over-representation (of a DE result's significant up and down genes, "
        "each tested on its own against the genes the DE tested). gsea: preranked GSEA of a DE "
        "result or a marker group."))] = "ora",
    collections: Annotated[list[str] | None, Field(description=(
        "MSigDB collections, e.g. h.all, c2.cp.reactome, c2.cp.kegg_medicus, c2.cp.wikipathways, "
        "c5.go.bp, c5.go.mf, c5.go.cc, c8.all, c7.immunesigdb (default: the species' set)."))] = None,
    padj_max: Annotated[float, Field(gt=0, le=1, description="ORA of a DE result: genes significant at this.")] = 0.05,
    lfc_min: Annotated[float, Field(ge=0, description="ORA of a DE result: and |log2FC| at least this.")] = 0.25,
    n_genes: Annotated[int, Field(ge=5, le=1000, description="ORA of a marker group: its top n genes.")] = 100,
    min_size: Annotated[int | None, Field(ge=5, le=500, description="GSEA: smallest set size tested (default 15).")] = None,
    limit: Limit = DEFAULT_LIMIT,
) -> str:
    """Pathway / gene-set enrichment (MSigDB): ORA of a DE result's up and down genes, of a
    gene list or of a group's markers, or preranked GSEA of a DE result or a marker group.
    Returns the top terms with overlap, p and adjusted p (NES for GSEA)."""
    n = clamp(limit)
    sources = sum(bool(x) for x in (de_result_id, genes, group))
    if sources != 1:
        raise ToolError("Give exactly one of: de_result_id, genes, or column + group.")
    if group and not column:
        raise ToolError("A marker group needs its column.")
    ora_cols = ["term", "collection", "overlap_count", "gene_count", "pvalue", "adjusted_pvalue", "genes"]
    gsea_cols = ["term", "collection", "nes", "pvalue", "fdr", "set_size", "leading_edge"]

    def terms(records: list[dict[str, Any]], cols: list[str]) -> dict[str, Any]:
        cut = [{**r, **{k: r[k][:10] for k in ("genes", "leading_edge") if isinstance(r.get(k), list)}}
               for r in records[:n]]
        return table(cut, cols, n)

    if method == "gsea":
        if genes:
            raise ToolError("GSEA needs a ranking: a de_result_id or a marker group, not a gene list.")
        body: dict[str, Any] = {"de_result_id": de_result_id or "", "column": column or "",
                                "group": group or "", "collections": collections, "max_terms": n}
        if min_size is not None:
            body["min_size"] = min_size
        d = await _post(f"{_ds(dataset_id)}/enrichment/gsea", body)
        out = {"dataset_id": dataset_id, "method": "gsea",
               **pick(d, "rank_metric_label", "n_genes_ranked", "n_excluded_ribo_mito", "n_sets_tested",
                      "permutations", "collections", "warning", "panel_caution", "de_method"),
               "note": "leading_edge: first 10 genes; a pvalue of 0 is below 1/permutations"}
        for side in ("up", "down"):
            s = d.get(side) or {}
            out[side] = {**pick(s, "n_terms", "n_significant"), **terms(s.get("results") or [], gsea_cols)}
        return dumps(out, _max_chars(limit))

    if group:
        try:
            d = await api.post(f"{_ds(dataset_id)}/enrichment/compute-local",
                               {"column": column, "group": group, "n_genes": n_genes, "collections": collections})
        except api.RouteError as e:
            hint = " Call get_markers for that column first." if "No marker genes" in e.detail else ""
            raise ToolError(e.detail + hint) from None
        res = d.get("results") or []
        return dumps({"dataset_id": dataset_id, "method": "ora", "source": f"top {n_genes} markers of "
                      f"{column}={group}", **pick(d, "n_genes_used", "warning", "panel_caution"),
                      "n_terms": len(res), "terms": terms(res, ora_cols)}, _max_chars(limit))

    if de_result_id:
        df = await _de_table(dataset_id, de_result_id)
        gcol = _col(df, "gene", "names")
        lcol = _col(df, "logfoldchange", "logfoldchanges", "log2FoldChange")
        pcol = _col(df, "pval_adj", "pvals_adj", "padj")
        if gcol is None or lcol is None:
            raise ToolError("This result holds no gene table with fold changes.")
        lfc = pd.to_numeric(df[lcol], errors="coerce")
        ok = lfc.abs() >= lfc_min
        if pcol is not None and pd.to_numeric(df[pcol], errors="coerce").notna().any():
            ok &= pd.to_numeric(df[pcol], errors="coerce") <= padj_max
        up = df.loc[ok & (lfc > 0), gcol].astype(str).tolist()
        down = df.loc[ok & (lfc < 0), gcol].astype(str).tolist()
        background = df[gcol].astype(str).tolist()
        source = f"DE {de_result_id}: {len(up)} up, {len(down)} down of {len(background)} tested"
    else:
        all_genes = (await _get(f"{_ds(dataset_id)}/genes"))["genes"]
        up, down, background = list(genes or []), [], all_genes
        source = f"gene list of {len(up)} against the dataset's {len(all_genes)} genes"
    if not up and not down:
        raise ToolError("No significant genes to test at these thresholds.")
    d = await _post(f"{_ds(dataset_id)}/enrichment/gene-lists",
                    {"up": up[:5000], "down": down[:5000], "background": background[:60000],
                     "collections": collections, "max_terms": n})
    out = {"dataset_id": dataset_id, "method": "ora", "source": source,
           **pick(d, "background_n", "collections", "warning", "panel_caution", "species")}
    for side in ("up", "down"):
        s = d.get(side) or {}
        if s.get("n_genes_requested"):
            out[side] = {**pick(s, "n_genes_used", "n_terms", "n_significant"),
                         "terms": terms(s.get("results") or [], ora_cols)}
    out["note"] = "genes: the first 10 of overlap_count"
    return dumps(out, _max_chars(limit))


@tool(READ)
async def score_gene_set(
    dataset_id: DatasetId,
    genes: Annotated[list[str], Field(min_length=1, max_length=2000, description="The set's gene symbols.")],
    groupby: Annotated[str, Field(description="Summarise the per-cell scores by this obs column "
                                              "(default: the active clustering).")] = "",
    method: Annotated[Literal["score_genes", "aucell", "ssgsea", "gsva"], Field(description=(
        "score_genes: scanpy's mean minus a matched control set (default); aucell: the set's share of "
        "each cell's top-ranked genes; ssgsea / gsva: rank walks (refused on very large data)."))] = "score_genes",
    limit: Limit = DEFAULT_LIMIT,
) -> str:
    """Score every cell for a gene set (scanpy score_genes by default, or AUCell / ssGSEA /
    GSVA) and summarise the scores per group (mean, median, share above 0)."""
    d = await _post(f"{_ds(dataset_id)}/genesets/score", {"gene_set": genes, "method": method})
    scores = np.asarray(d.pop("scores"), dtype=np.float64)
    if not groupby:
        groupby = (await _get(_ds(dataset_id))).get("active_clustering") or ""
    out: dict[str, Any] = {"dataset_id": dataset_id, **{k: v for k, v in d.items() if k != "score_name"}}
    out["genes_found"] = out.get("genes_found", [])[:50]
    if groupby:
        labels = pd.Series((await _get(f"{_ds(dataset_id)}/metadata/{groupby}", format="json"))["values"])
        frame = pd.DataFrame({"g": labels.astype(str).where(labels.notna()), "s": scores})
        agg = frame.groupby("g", observed=True)["s"].agg(
            n="size", mean="mean", median="median", share_above_0=lambda s: float((s > 0).mean()))
        agg = agg.sort_values("mean", ascending=False).reset_index().rename(columns={"g": "group"})
        out["groupby"] = groupby
        out.update(rows(agg.to_dict("records"), limit=clamp(limit)))
    else:
        out["summary"] = {"mean": float(scores.mean()), "median": float(np.median(scores))}
    return dumps(out, _max_chars(limit))


# --- results, figures, jobs --------------------------------------------------------------------------

@tool(READ)
async def list_results(
    dataset_id: DatasetId,
    kind: Annotated[str | None, Field(description="Only this kind (e.g. de, markers, enrichment, composition).")] = None,
    limit: Limit = DEFAULT_LIMIT,
) -> str:
    """The dataset's stored results (DE, markers, enrichment, composition, ...), newest first:
    id, kind, label, and whether computed on an earlier version of the data (stale)."""
    d = await _get(f"{_ds(dataset_id)}/results", kind=kind)
    res = d.get("results") or []
    return dumps({"dataset_id": dataset_id, **rows(res, ["id", "kind", "label", "created_at", "stale", "status"],
                                                   clamp(limit))}, _max_chars(limit))


@tool(READ)
async def get_result(
    dataset_id: DatasetId,
    result_id: Annotated[str, Field(description="From list_results or a tool's result_id.")],
    limit: Limit = DEFAULT_LIMIT,
    offset: Annotated[int, Field(ge=0, description="Row to start from (page through a long table).")] = 0,
    sort_by: Annotated[str | None, Field(description="Column to sort by (ascending; prefix '-' for descending).")] = None,
) -> str:
    """A stored result's parameters and its table's rows (from offset, limit of them)."""
    meta = await _get(f"{_ds(dataset_id)}/results/{result_id}")
    out: dict[str, Any] = {"dataset_id": dataset_id, **pick(meta, "id", "kind", "label", "created_at",
                                                            "stale", "params")}
    try:
        df = await _de_table(dataset_id, result_id)
    except ToolError as e:
        out["table"] = str(e)
        return dumps(out)
    if sort_by:
        col = sort_by.lstrip("-")
        if col not in df.columns:
            raise ToolError(f"No column '{col}'; the columns are: {', '.join(df.columns)}.")
        df = df.sort_values(col, ascending=not sort_by.startswith("-"))
    n = clamp(limit)
    out["columns"] = list(df.columns)
    out["n_rows"] = len(df)
    out["offset"] = offset
    out["rows"] = [[num(v) for v in r] for r in df.iloc[offset: offset + n].itertuples(index=False)]
    return dumps(out, _max_chars(limit))


async def _adaptor(dataset_id: str):
    dm = api.dataset_manager()
    await _get(_ds(dataset_id))  # a 404 / not-ready error as the route words it
    adaptor = await dm.get_or_load_dataset(dataset_id)
    if adaptor is None:
        raise ToolError("Dataset not found.")
    return adaptor


@tool(READ)
async def plot(
    dataset_id: DatasetId,
    kind: Annotated[Literal["embedding", "violin", "dotplot"], Field(description=(
        "embedding: cells on a UMAP/tSNE/PCA coloured by color_by or gene; violin: one gene by "
        "groupby; dotplot: genes x groupby (size = share expressing, colour = mean)."))] = "embedding",
    embedding: Annotated[str, Field(description="obsm key, e.g. X_umap (default), X_tsne, X_pca.")] = "X_umap",
    color_by: Annotated[str | None, Field(description="Obs column to colour by (embedding).")] = None,
    gene: Annotated[str | None, Field(description="Gene to colour by (embedding) or to plot (violin).")] = None,
    groupby: Annotated[str, Field(description="Groups (violin, dotplot; default: the active clustering).")] = "",
    genes: Annotated[list[str] | None, Field(max_length=60, description="Genes (dotplot).")] = None,
    layer: Annotated[str, Field(description="Expression layer (default: the dataset's).")] = "",
    within_column: Annotated[str | None, Field(description=(
        "Only one subgroup: its obs column (e.g. cell_type). Violin and dotplot use only its cells; "
        "an embedding colours them and draws the other cells grey."))] = None,
    within_value: Annotated[str | None, Field(description="Only one subgroup: its value (e.g. CD14 Monocyte).")] = None,
) -> list:
    """A PNG figure of the dataset (at most 1600 px, under 300 KB). A dataset over 50,000 cells
    is drawn from a random 50,000 (said in the caption). The same views are in the browser UI."""
    if (within_column is None) != (within_value is None):
        raise ToolError("Give both within_column and within_value, or neither.")
    scope = f" in {within_column} = {within_value}" if within_column else ""
    if kind == "dotplot":
        if not genes:
            raise ToolError("A dot plot needs genes.")
        if within_column:
            d = await _dotplot_within(dataset_id, genes, groupby, layer, within_column, within_value or "")
        else:
            d = await _get(f"{_ds(dataset_id)}/expression/by-group", genes=genes, groupby=groupby, layer=layer)
        caption = f"{len(d['groups'])} groups of {d['groupby']}{scope}, layer {d['layer']}"
        png = await asyncio.to_thread(plots.dotplot, d, f"Dot plot by {d['groupby']}{scope}", caption)
        return [Image(data=png, format="png"), TextContent(type="text", text=caption)]

    adaptor = await _adaptor(dataset_id)
    lay = layer or None

    def _within() -> np.ndarray | None:
        if not within_column:
            return None
        try:
            col = adaptor.get_obs_column(within_column)
        except KeyError:
            raise ToolError(f"Column '{within_column}' not found.")
        mask = (col.astype(str).where(col.notna()) == str(within_value)).to_numpy()
        if not mask.any():
            raise ToolError(f"No cell has {within_column} = '{within_value}'.")
        return mask

    def _values_of_gene(g: str) -> np.ndarray:
        if g not in set(adaptor._var_names_for_layer(lay or adaptor.default_expression_layer())):
            raise ToolError(f"Gene '{g}' is not in the expression layer.")
        return adaptor.get_expression([g], layer=lay)[:, 0]

    def _draw() -> tuple[bytes, str]:
        n = adaptor.n_cells()
        rows_ = plots.sample(n)
        note = f"{n:,} cells" if rows_ is None else f"random {len(rows_):,} of {n:,} cells"
        focus = _within()
        if focus is not None:
            note += f"; {int(focus.sum()):,} with {within_column} = {within_value}"
        if kind == "embedding":
            try:
                xy = adaptor.get_embedding(embedding)[:, :2]
            except KeyError:
                names = [e["name"] for e in adaptor.available_embeddings()]
                raise ToolError(f"No embedding '{embedding}'; this dataset has: {', '.join(names) or 'none'}.")
            if color_by and gene:
                raise ToolError("Colour by an obs column or a gene, not both.")
            values: Any = None
            label = ""
            if color_by:
                try:
                    values = adaptor.get_obs_column(color_by)
                except KeyError:
                    raise ToolError(f"Column '{color_by}' not found.")
                label = color_by
            elif gene:
                values, label = _values_of_gene(gene), gene
            if rows_ is not None:
                xy = xy[rows_]
                focus = focus[rows_] if focus is not None else None
                if values is not None:
                    values = values.iloc[rows_] if isinstance(values, pd.Series) else values[rows_]
            title = (f"{embedding} coloured by {label}" if label else embedding) + scope
            return plots.embedding(np.asarray(xy, dtype=float), values, title, label, note, focus), note
        if not gene:
            raise ToolError("A violin plot needs a gene.")
        by = groupby or adaptor.active_clustering_column() or ""
        if not by:
            raise ToolError("No grouping column: give groupby.")
        try:
            groups = adaptor.get_obs_column(by)
        except KeyError:
            raise ToolError(f"Column '{by}' not found.")
        values = _values_of_gene(gene)
        if focus is not None:  # the subgroup's cells only, sampled among themselves
            values, groups = values[focus], groups[focus]
            rows_ = plots.sample(len(values))
            note = f"{len(values):,} cells with {within_column} = {within_value}" + (
                f" (random {len(rows_):,} drawn)" if rows_ is not None else "")
        if rows_ is not None:
            values, groups = values[rows_], groups.iloc[rows_]
        return plots.violin(values, groups, f"{gene} by {by}{scope}",
                            f"{gene} ({lay or adaptor.default_expression_layer()})", note), note

    try:
        png, note = await asyncio.to_thread(_draw)
    except ToolError:
        raise
    except Exception as e:  # a refusal on a large dataset, a bad column type
        raise ToolError(f"Could not draw the plot: {e}") from None
    return [Image(data=png, format="png"), TextContent(type="text", text=f"{kind}: {note}, {len(png):,} bytes")]


async def _dotplot_within(dataset_id: str, genes: list[str], groupby: str, layer: str,
                          column: str, value: str) -> dict[str, Any]:
    """The dot plot's numbers over one subgroup's cells: the by-group route's selection row,
    once per group of ``groupby`` among those cells (in the column's order)."""
    flt = CellFilter(op="and", items=[{"kind": "category", "column": column, "values": [value]}])
    within = await _indices(dataset_id, flt)
    if len(within) == 0:
        raise ToolError(f"No cell has {column} = '{value}'.")
    if not groupby:
        groupby = (await _get(_ds(dataset_id))).get("active_clustering") or ""
    values = (await _get(f"{_ds(dataset_id)}/metadata/{groupby}", format="json"))["values"]
    labels = pd.Series(values, dtype="object").astype(str).where(pd.Series(values).notna())
    order = list(dict.fromkeys(v for v in labels if isinstance(v, str)))
    sub = labels.iloc[within]
    present = set(sub.dropna())
    out: dict[str, Any] = {"groupby": groupby, "groups": [], "mean": [], "fraction": [],
                           "genes": [], "layer": layer}
    for g in sorted(present, key=lambda x: (order.index(x), x)):
        cells = within[(sub == g).to_numpy()]
        r = await _post(f"{_ds(dataset_id)}/expression/by-group",
                        {"genes": genes, "groupby": groupby, "layer": layer, "cells": cells.tolist(),
                         "include_groups": False})
        sel = r["selection"]
        out.update(genes=r["genes"], layer=r["layer"])
        out["groups"].append(g)
        out["mean"].append(sel["mean"])
        out["fraction"].append(sel["fraction"])
    return out


@tool(READ)
async def job_status(
    job_id: Annotated[str, Field(description="From the tool that started the job.")],
    wait_seconds: Annotated[float, Field(ge=0, le=50, description="Wait up to this long for it to finish.")] = 20,
) -> str:
    """A job's state: running (with its latest progress line), or complete with its result,
    error, or cancelled. Waits up to wait_seconds (max 50) for it to finish first."""
    job = jobs.get(job_id)
    if job is None:
        raise ToolError(f"No job '{job_id}' (jobs are kept in memory and lost when scView restarts).")
    await jobs.wait(job, wait_seconds)
    return dumps(job.summary())


@tool(WRITE)
async def cancel_job(job_id: Annotated[str, Field(description="A running job's id.")]) -> str:
    """Stop a running job. A pipeline run stops before its next step and saves nothing (the
    dataset stays as it was); an import keeps nothing."""
    job = jobs.get(job_id)
    if job is None:
        raise ToolError(f"No job '{job_id}'.")
    await jobs.cancel(job)
    await jobs.wait(job, 10)
    return dumps(job.summary())
