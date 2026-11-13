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
import re
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
    '{"kind":"group","op":"or","conditions":[...]}. No other keys (an unknown one is refused) and '
    "no negation: list the values to keep instead."))]


P0_NOTE = ("A p value of 0 is what the test returned: below double precision (under about 1e-300), "
           "not rounded here.")


def _max_chars(limit: int) -> int:
    return compact.MAX_CHARS if limit <= DEFAULT_LIMIT else int(compact.MAX_CHARS * limit / DEFAULT_LIMIT)


def _fail(e: api.RouteError) -> ToolError:
    if e.large_dataset:
        return ToolError(f"Refused on a large dataset: {e.detail}")
    return ToolError(e.detail)


_MISSING_COLUMN = re.compile(r"^Column '([^']+)' not found")
_DATASET_PATH = re.compile(r"^/datasets/([^/]+)/")


async def _route_error(e: api.RouteError, path: str) -> ToolError:
    """The route's error; a missing obs column also names the closest columns (any route)."""
    col, ds = _MISSING_COLUMN.match(e.detail or ""), _DATASET_PATH.match(path)
    if col and ds and "Closest:" not in e.detail:
        try:
            return await _column_error(ds.group(1), col.group(1))
        except Exception:  # noqa: BLE001 — the hint is extra; the route's own error stands
            pass
    return _fail(e)


async def _get(path: str, **query: Any) -> Any:
    try:
        return await api.get(path, **query)
    except api.RouteError as e:
        raise await _route_error(e, path) from None


async def _post(path: str, body: Any = None, **query: Any) -> Any:
    try:
        return await api.post(path, body, **query)
    except api.RouteError as e:
        raise await _route_error(e, path) from None


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
    """One dataset: cells, genes, processing steps done, embeddings, every obs column (name,
    type and number of values, always all of them), the values of the small categorical
    columns (30 or fewer values, as many as fit; describe_column gives the rest), expression
    layers, active clustering, large-dataset mode, and other modalities present."""
    d = await _get(_ds(dataset_id))
    cols, values = [], {}
    for c in d.get("obs_columns") or []:
        cols.append([c["name"], c.get("dtype"), c.get("n_unique")])
        if c.get("values") is not None and (c.get("n_unique") or 99) <= 30:
            values[c["name"]] = c["values"]
        if c.get("dtype") == "bool" and (c.get("n_unique") or 0) < 2:
            # The route lists both for a bool column (its colour slots): name the one present.
            vals = (await _get(f"{_ds(dataset_id)}/metadata/{c['name']}", format="json"))["values"]
            values[c["name"]] = sorted({str(v) for v in vals if v is not None})
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
            out["steps_done"] = _runs([s.get("step") for s in recipe.get("recipe") or []])
        except api.RouteError:
            pass
    # Every column's name and type, never cut (Yang 2021: 22 of 45 shown, the author labels
    # among the hidden); the small columns' values while they fit, fewest values first.
    out["obs_columns"] = {"columns": ["name", "dtype", "n_unique"], "rows": cols, "n_total": len(cols)}
    budget = compact.MAX_CHARS - len(dumps(out, 10**9))
    kept, left = {}, []
    for name, vals in sorted(values.items(), key=lambda kv: len(json.dumps(kv[1], default=str))):
        size = len(json.dumps({name: vals}, default=str))
        if size <= budget:
            kept[name] = vals
            budget -= size
        else:
            left.append(name)
    out["column_values"] = {name: kept[name] for name in values if name in kept}
    if left:
        out["column_values_left_out"] = f"{', '.join(left)}: describe_column lists them"
    return dumps(out, 10**9)


def _runs(steps: list[Any]) -> list[str]:
    """Steps in order, a step repeated back to back written once with its count
    ('label_cells x13')."""
    out: list[str] = []
    for s in steps:
        name = str(s)
        if out and out[-1].split(" x")[0] == name:
            n = out[-1].partition(" x")[2]
            out[-1] = f"{name} x{int(n or 1) + 1}"
        else:
            out.append(name)
    return out


async def _column_error(dataset_id: str, name: str) -> ToolError:
    """A column that is not there: the closest names, and every column when there are few."""
    names = [c["name"] for c in (await _get(_ds(dataset_id))).get("obs_columns") or []]
    return ToolError(column_message(name, names))


def column_message(name: str, names: list[str]) -> str:
    """'Column ... not found.' with the closest of ``names`` (spelling, or containing it),
    and all of them when there are at most 60."""
    import difflib

    names = [str(n) for n in names]
    lower = {n.lower(): n for n in names}
    close = difflib.get_close_matches(name.lower(), list(lower), n=5, cutoff=0.5)
    close += [lower[n] for n in lower if name.lower() in n and lower[n] not in close][: 5 - len(close)]
    close = [lower.get(c, c) for c in close]
    msg = f"Column '{name}' not found."
    if close:
        msg += f" Closest: {', '.join(close)}."
    if len(names) <= 60:
        msg += f" The {len(names)} obs columns: {', '.join(names)}."
    else:
        msg += f" The dataset has {len(names)} obs columns: get_dataset lists them all."
    return msg


async def _obs_values(dataset_id: str, column: str) -> pd.Series:
    """One obs column's values (missing as None), or the closest names when it is not there."""
    try:
        d = await api.get(f"{_ds(dataset_id)}/metadata/{column}", format="json")
    except api.RouteError as e:
        if e.status == 404 and e.detail.startswith("Column"):
            raise await _column_error(dataset_id, column) from None
        raise _fail(e) from None
    return pd.Series(d["values"], dtype=object)


def _is_numeric(s: pd.Series) -> bool:
    present = s.dropna()
    if present.empty or present.map(type).eq(bool).any():
        return False
    return bool(pd.to_numeric(present, errors="coerce").notna().all())


def _by_group(values: pd.Series, groups: list[pd.Series], names: list[str]) -> list[dict[str, Any]]:
    """n, missing, mean, median and quartiles of a numeric column per group (cells whose
    group is missing are left out)."""
    frame = pd.DataFrame({f"g{i}": g.astype(str).where(g.notna()) for i, g in enumerate(groups)})
    frame["v"] = pd.to_numeric(values, errors="coerce")
    keys = [f"g{i}" for i in range(len(groups))]
    out = []
    for key, sub in frame.dropna(subset=keys).groupby(keys, sort=True, observed=True):
        key = key if isinstance(key, tuple) else (key,)
        v = sub["v"].dropna()
        rec: dict[str, Any] = dict(zip(names, key))
        rec.update(n=int(len(sub)), n_missing=int(len(sub) - len(v)))
        if len(v):
            q = v.quantile([0.25, 0.5, 0.75]).tolist()
            rec.update(mean=float(v.mean()), median=q[1], q25=q[0], q75=q[2], iqr=q[2] - q[0],
                       min=float(v.min()), max=float(v.max()))
        out.append(rec)
    return out


@tool(READ)
async def describe_column(
    dataset_id: DatasetId,
    column: Annotated[str, Field(description="An obs (cell metadata) column.")],
    groupby: Annotated[list[str] | None, Field(max_length=2, description=(
        "One or two obs columns (e.g. ['sample'] or ['donor_id', 'condition']): summarise a numeric "
        "column per group (n, mean, median, quartiles, IQR) instead of over every cell."))] = None,
    limit: Limit = DEFAULT_LIMIT,
    offset: Annotated[int, Field(ge=0, description="With groupby: the group row to start from.")] = 0,
) -> str:
    """One obs column: each value's cell count (largest first) for a category, or a
    numeric summary (min, quartiles, max, mean, missing). With groupby, a numeric column
    (depth, genes per cell, % mito, doublet score) per group: per sample, or per donor x
    condition. A missing column's error names the closest columns."""
    s = await _obs_values(dataset_id, column)
    if groupby:
        if column in groupby:
            raise ToolError("groupby must name other columns than the one described.")
        if not _is_numeric(s):
            raise ToolError(f"'{column}' is not numeric: count its values per group with crosstab "
                            f"(row '{column}', col '{groupby[0]}').")
        groups = [await _obs_values(dataset_id, g) for g in groupby]
        recs = _by_group(s, groups, list(groupby))
        n = clamp(limit)
        cols = [*groupby, "n", "n_missing", "mean", "median", "q25", "q75", "iqr", "min", "max"]
        out = {"dataset_id": dataset_id, "column": column, "groupby": list(groupby),
               "n_cells": int(len(s)), "n_missing": int(s.isna().sum()),
               "n_without_group": int(sum(pd.concat(groups, axis=1).isna().any(axis=1))),
               "offset": offset, **table(recs[offset:], cols, n)}
        out["n_total"] = len(recs)
        if offset + n < len(recs):
            out["next_offset"] = offset + n
        return dumps(out, _max_chars(limit))
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
    try:
        d = await api.get(f"{_ds(dataset_id)}/metadata/crosstab", row=row, col=col)
    except api.RouteError as e:
        if e.status == 404 and "not found" in e.detail.lower() and "dataset" not in e.detail.lower():
            names = [c["name"] for c in (await _get(_ds(dataset_id))).get("obs_columns") or []]
            missing = next((c for c in (row, col) if c not in names), None)
            if missing is not None:
                raise ToolError(column_message(missing, names)) from None
        raise _fail(e) from None
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
    query: Annotated[str | None, Field(description=(
        "Words (not a phrase), each anywhere in the title, collection name or the collection's "
        "description (its abstract). With no dataset holding them all, those with the most."))] = None,
    organism: Annotated[str | None, Field(description="e.g. Homo sapiens")] = None,
    tissue: Annotated[str | None, Field(description="e.g. blood")] = None,
    disease: Annotated[str | None, Field(description="e.g. COVID-19")] = None,
    assay: Annotated[str | None, Field(description=(
        "e.g. 10x 3' v3. Tissue, disease and assay match the exact name first; when no dataset "
        "has it, names holding its words (Visium -> Visium Spatial Gene Expression V1), "
        "listed under matched."))] = None,
    max_mb: Annotated[float | None, Field(gt=0, description="Largest download, MB.")] = None,
    sort: Annotated[Literal["cells", "size", "title", "newest"], Field(
        description="Order: most cells first (default), largest download, title, or newest.")] = "cells",
    limit: Limit = DEFAULT_LIMIT,
    offset: Annotated[int, Field(ge=0, description=(
        "The match to start from: page through the matches with the reply's next_offset."))] = 0,
) -> str:
    """Search CELLxGENE Discover's public datasets (listing cached a day). Check one's memory
    need with check_public_dataset; import it with import_public_dataset and its
    dataset_version_id. A page holds up to limit matches, fewer when they do not fit the
    reply: n_shown and next_offset say where the next page starts."""
    d = await _get("/public-import/discover/datasets", q=query, organism=organism, tissue=tissue,
                   disease=disease, assay=assay, max_mb=max_mb, sort=sort, limit=clamp(limit),
                   offset=offset)
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
    total = int(d.get("total", d.get("n_total", len(out))) or 0)
    res: dict[str, Any] = {"n_total": total, "offset": offset, "fetched_at": d.get("fetched_at")}
    if d.get("partial"):
        res["partial"] = {**d["partial"], "note": "No dataset has every word: these have the most."}
    if d.get("matched"):
        res["matched"] = d["matched"]
    # Whole records, as many as fit the reply (a cut list said "100 of 200 shown" with no
    # way to the rest): the count shown and the next page's start are then exact.
    budget = _max_chars(limit) - len(dumps(res, 10**9)) - 120
    shown: list[dict[str, Any]] = []
    for e in out:
        size = len(dumps(e, 10**9)) + 1
        if shown and size > budget:
            break
        shown.append(e)
        budget -= size
    res.update(datasets=shown, n_shown=len(shown))
    if offset + len(shown) < total:
        res["next_offset"] = offset + len(shown)
    return dumps(res, 10**9)


def _memory(m: dict[str, Any] | None) -> dict[str, Any]:
    gb = 2**30
    out = pick(m or {}, "verdict", "message", "source", "n_cells", "n_genes")
    for k in ("in_memory_bytes", "backed_bytes", "limit_bytes"):
        if isinstance((m or {}).get(k), (int, float)):
            out[k.replace("_bytes", "_gb")] = round(m[k] / gb, 2)
    return out


@tool(WEB_READ)
async def check_public_dataset(
    dataset_version_id: Annotated[str, Field(description="From search_public_datasets.")],
    dataset_id: Annotated[str | None, Field(
        description="Its dataset_id: needed only for a version no longer current.")] = None,
) -> str:
    """Before an import: one Discover dataset version's size, the disk the import needs, and
    the memory opening it needs (verdict fits in memory / backed: its matrices stay on disk /
    wont_fit; read from the file's header on Discover, nothing is downloaded)."""
    d = await _get(f"/public-import/discover/datasets/{dataset_version_id}", dataset_id=dataset_id)
    out = pick(d, "dataset_version_id", "dataset_id", "title", "cell_count", "disk")
    out["size_mb"] = round((d.get("size_bytes") or 0) / 2**20, 1)
    out["memory"] = _memory(d.get("memory"))
    return dumps(out)


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
    follow it with job_status; the new dataset's id is in the result). Refused before any
    download when the file could not open even with its matrices on disk."""
    body = {"dataset_version_id": dataset_version_id, "dataset_id": dataset_id, "name": name}
    detail = await _get(f"/public-import/discover/datasets/{dataset_version_id}", dataset_id=dataset_id)
    memory = _memory(detail.get("memory"))
    if memory.get("verdict") == "wont_fit":
        raise ToolError(f"Not downloaded: {memory.get('message')}")

    async def cancel(job: jobs.Job) -> None:
        jid = (job.events.get("job") or {}).get("job_id")
        if not jid:
            raise api.RouteError(404, "not started")
        await api.post(f"/public-import/jobs/{jid}/cancel")

    def finish(job: jobs.Job) -> dict[str, Any]:
        return pick(job.events.get("complete"), "dataset_id")

    job = jobs.new("import", None, "/public-import/discover/import-stream", body,
                   progress_line=_line, finish=finish, cancel_route=cancel)
    return dumps({"job_id": job.id, "status": job.status, "memory": memory,
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
                  "female_genes_missing", "male_genes_missing",
                  "source", "summary", "warnings", "notes"),
           "samples": table(d.get("groups") or [], ["name", "n_cells", "call", "confidence", "metadata",
                                                    "mismatch", "frac_female", "frac_male", "evidence"],
                            clamp(limit)),
           "confounding": [pick(c, "condition", "kind", "levels") for c in d.get("confounding") or []]}
    return dumps(out, _max_chars(limit))


@tool(READ)
async def library_concordance(
    dataset_id: DatasetId,
    sample_column: Annotated[str, Field(description=(
        "The samples / donors whose libraries are compared (default: the donor column the QC "
        "table uses)."))] = "",
    library_column: Annotated[str, Field(description=(
        "The libraries: one value per sequencing library, each within one sample (default: "
        "detected: a library / lane / channel column, a non-person sample column such as "
        "biosample_id, or the barcodes' prefix or -N suffix)."))] = "",
    celltype_column: Annotated[str, Field(description=(
        "The cell types the libraries' mixes are compared over (default: an author cell-type "
        "column, else the clustering)."))] = "",
    limit: Limit = DEFAULT_LIMIT,
) -> str:
    """Samples sequenced as two or more libraries: each library's cells, median UMIs and
    genes, and a flag when two libraries of one sample differ in cell-type mix (or depth)
    more than different samples alike in condition, tissue and sex do (90% quantile of those
    distances, floor 0.15, chi-square p < 1e-3). A flagged library may hold another sample's
    cells or a failed capture. A sample's libraries from different tissues or conditions are
    compared only within one (``split_by``). ``reason`` says why nothing was checked."""
    d = await _get(f"{_ds(dataset_id)}/qc/libraries", sample_column=sample_column,
                   library_column=library_column, celltype_column=celltype_column)
    samples = []
    for s in d.get("samples") or []:
        pair = s.get("pair") or {}
        samples.append({"name": s.get("name"), "status": s.get("status"), "flags": s.get("flags"),
                        "libraries": [pick(lib, "name", "n_cells", "median_counts", "median_genes", "too_small")
                                      for lib in s.get("libraries") or []],
                        **({"pair": pick(pair, "a", "b", "distance", "p", "above_share", "depth_log2_ratio",
                                         "differing")} if pair else {}),
                        **({"message": s["message"]} if s.get("message") else {})})
    n = clamp(limit)
    out = {"dataset_id": dataset_id,
           **pick(d, "status", "reason", "sample_column", "library_column", "library_source",
                  "celltype_column", "split_by", "n_checked", "n_flagged", "flagged", "reference", "note"),
           "samples": samples[:n]}
    if len(samples) > n:
        out["samples_total"] = len(samples)
    return dumps(out, _max_chars(limit))


@tool(READ)
async def preflight(dataset_id: DatasetId, steps: Steps, params: Annotated[dict[str, Any] | None, Field(
        description="The run's parameters, as run_pipeline takes them (its schema lists them).")] = None) -> str:
    """Before a run: its peak memory and rough time per step and in all, against the memory
    scView has free (verdict fits / tight / wont_fit, with suggestions), and whether it would
    run in large-dataset mode (mode.converts: the dataset would be converted for good, with
    what that turns off) or without the matrices (mode.light: steps that need none). Nothing
    is run."""
    _check_steps(steps)
    p = _check_params(params)
    d = await _get(f"{_ds(dataset_id)}/preflight", **_preflight_query(steps, p))
    gb = 2**30
    steps_ = [{"step": s.get("step"), "peak_gb": round((s.get("peak_bytes") or 0) / gb, 2),
               "seconds": s.get("seconds")} for s in d.get("steps") or []]
    head = d.get("header") or {}
    out = {"dataset_id": dataset_id, "verdict": d.get("verdict"),
           **pick(d, "mode", "leiden", "umap_3d", "in_memory", "in_memory_estimate", "regress_out"),
           "data": pick(head, "n_cells", "n_genes", "format", "backed", "large", "unit"),
           "steps": table(steps_, ["step", "peak_gb", "seconds"], 40)}
    return dumps(out)


def _preflight_query(steps: list[str], p: dict[str, Any]) -> dict[str, Any]:
    query: dict[str, Any] = {"steps": ",".join(steps)}
    for k in ("n_top_genes", "n_comps", "n_neighbors", "normalization_method", "annotation_method",
              "doublet_channel_key", "doublet_method", "chunked", "clustering_method",
              "leiden_iterations", "umap_3d", "ambient_rho"):
        if p.get(k) is not None:
            query[k] = p[k]
    if p.get("regress_out"):
        query["regress_out"] = ",".join(p["regress_out"])
    return query


def _pipeline_result(job: jobs.Job) -> dict[str, Any]:
    r = job.events.get("complete") or {}
    out = pick(r, "status", "stop_reason", "steps_run", "steps_skipped", "skipped", "errors",
               "notes", "cleared", "elapsed_seconds", "unknown_steps", "chunked", "mode")
    out["dataset_id"] = job.dataset_id
    out["note"] = "Saved: the open browser UI shows the new state after a reload of the dataset."
    return out


async def _start_run(dataset_id: str, steps: list[str], p: dict[str, Any], kind: str,
                     finish: Callable = _pipeline_result, allow_large_mode: bool = True) -> jobs.Job:
    if any(j.dataset_id == dataset_id for j in jobs.running()):
        busy = next(j for j in jobs.running() if j.dataset_id == dataset_id)
        raise ToolError(f"Job {busy.id} ({busy.kind}) is still running on this dataset; "
                        "wait for it with job_status or stop it with cancel_job.")

    async def cancel(job: jobs.Job) -> None:
        await api.post(f"{_ds(dataset_id)}/assessment/cancel")

    return jobs.new(kind, dataset_id, f"{_ds(dataset_id)}/assessment/run-stream",
                    {"steps": steps, "params": p, "allow_large_mode": allow_large_mode},
                    progress_line=_line, finish=finish, cancel_route=cancel)


@tool(WRITE)
async def run_pipeline(dataset_id: DatasetId, steps: Steps, params: Params = None,
                       allow_large_mode: Annotated[bool, Field(description=(
                           "Let a run that does not fit in memory convert the dataset to "
                           "large-dataset layout for good (cell-level DE and other whole-matrix tools "
                           "are then refused on it; pseudobulk DE stays; .raw leaves its layers). Default false: such a "
                           "run is refused, naming what it would turn off."))] = False) -> str:
    """Run pipeline steps on a dataset and save the result (a job: returns job_id at once;
    follow it with job_status, stop it with cancel_job). Check preflight first on a big dataset.
    Steps that need no expression matrix (neighbors, clustering, embeddings on the stored PCA)
    never convert it. The original upload is never changed. Unknown step or parameter names are
    refused."""
    _check_steps(steps)
    p = _check_params(params)
    if not allow_large_mode:
        mode = (await _get(f"{_ds(dataset_id)}/preflight", **_preflight_query(steps, p))).get("mode") or {}
        if mode.get("converts"):
            raise ToolError(
                f"Not run: {mode.get('reason')} That converts the dataset for good: "
                f"{mode.get('layout_after')}; no longer available on it: "
                f"{'; '.join(mode.get('turns_off') or [])}. Run cell-level DE and the other tools first, free "
                "memory, or pass allow_large_mode=true to convert it.")
    job = await _start_run(dataset_id, steps, p, "pipeline", allow_large_mode=allow_large_mode)
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
    saved with the dataset the first time a column is asked for. The current markers live in
    the dataset itself, one set per column, so they have no result_id and list_results does not
    show them (it lists earlier markers a re-run replaced): call this again to read them."""
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
    if d.get("skipped_groups"):
        extra["skipped_groups"] = d["skipped_groups"]
        extra["skipped_reason"] = d.get("skipped_reason")
    if d.get("matrix"):  # what the test ranked: log-normalised values (core/marker_matrix.py)
        extra["ranked_on"] = d["matrix"]
    return dumps({"dataset_id": dataset_id, "column": column, "columns": cols, "markers": out, **extra},
                 _max_chars(DEFAULT_LIMIT * max(1, n_genes // 10)))


@tool(WRITE)
async def annotate_celltypist(
    dataset_id: DatasetId,
    model: Annotated[str, Field(description="CellTypist model, e.g. Immune_All_Low.pkl (default), "
                                            "Immune_All_High.pkl, Healthy_COVID19_PBMC.pkl.")] = "Immune_All_Low.pkl",
    groupby: Annotated[str, Field(description="Clusters for the majority vote (default: the active clustering).")] = "",
    target_column: Annotated[str, Field(description="Obs column to write.")] = "celltypist",
    overwrite: Annotated[bool, Field(description=(
        "Replace target_column when it is a column that came with the data (an author or curated "
        "annotation); without it such a column is refused."))] = False,
) -> str:
    """Label cells with a CellTypist reference model (the pipeline's cell_type_annotation step;
    a job). The result names the labels' confidence and the model's gene overlap. The model is
    downloaded on first use."""
    p = {"annotation_method": "celltypist", "celltypist_model": model, "annotation_groupby": groupby,
         "annotation_target": target_column, "annotation_overwrite": overwrite}

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


@tool(READ)
async def cell_ontology_ids(
    dataset_id: DatasetId,
    column: Annotated[str, Field(description="Obs column of cell-type labels (an annotator's or the author's).")],
) -> str:
    """Each label of a column with its Cell Ontology (CL) id by strict name matching only: the
    same name or exact synonym, or the same once case, plurals, '+'/'positive', hyphens and a
    trailing 'cells' are set aside. Abbreviations ('CD14 Mono', 'Mk') get no id. For a column the
    LLM annotator wrote, the id the model gave is kept when it agrees with the label (source
    'model'). The ontology (a pinned CL release) is downloaded on first use."""
    d = await _get(f"{_ds(dataset_id)}/annotation/ontology", column=column)
    cols = ["label", "cl_id", "cl_name", "how", "source", "n_cells"]
    return dumps({"dataset_id": dataset_id, "column": column, "version": d.get("version"),
                  "n_labels": d["n_labels"], "n_matched": d["n_matched"], "columns": cols,
                  "labels": [[r.get(c) for c in cols] for r in d["labels"]]})


@tool(WEB_READ)
async def marker_db_check(
    dataset_id: DatasetId,
    column: Annotated[str, Field(description=(
        "Label column to check, e.g. an author cell_type or an annotation scView wrote."))],
    species: Annotated[Literal["human", "mouse"] | None, Field(description=(
        "Database rows to use (default: the dataset's species, told from the gene names)."))] = None,
    retry: Annotated[bool, Field(description=(
        "Try a database whose download failed a moment ago again now."))] = False,
    tissue: Annotated[str, Field(max_length=200, description=(
        "The dataset's tissue, e.g. 'kidney': suggestions keep to its types and those found in every "
        "tissue (immune, blood, vessels, stroma). Default: the data's own tissue column, else its title; "
        "none known: types of any tissue."))] = "",
    limit: Limit = DEFAULT_LIMIT,
) -> str:
    """The UI's marker database check (PanglaoDB, CellMarker, ACT): for each label of a column, its
    database type matched by name, the verdict (supported / weak support, also for one shared gene
    whatever its p / no database support / too few database markers / not in database) with p and
    the overlapping genes, and the 3
    database types the group's top 25 markers fit best, of the dataset's tissue and of every tissue
    when the tissue is known ('tissue' in the reply says which). Changes nothing in the dataset. The
    databases are downloaded on first use (about a minute); after that cluster_evidence also
    compares markers with them."""
    d = await _post(f"{_ds(dataset_id)}/annotation/marker-db/check",
                    {"column": column, "species": species, "retry": retry, "tissue": tissue})
    n = clamp(limit)
    labels = [{**r, "overlap": ", ".join(r.get("overlap") or []),
               "suggestions": "; ".join(f"{s['cell_type']} (p {num(s['p'])})" for s in r.get("suggestions") or [])}
              for r in d.get("labels") or []]
    return dumps({"dataset_id": dataset_id,
                  **pick(d, "column", "species", "species_note", "verdict_counts", "cellmarker",
                         "verdict_sources", "suggestion_sources", "tissue", "note"),
                  "labels": table(labels, ["label", "verdict_text", "db_type", "p", "overlap", "n_db_genes",
                                           "suggestions"], n)}, _max_chars(n))


@tool(READ)
async def cluster_evidence(
    dataset_id: DatasetId,
    column: Annotated[str, Field(description=(
        "Grouping column: a clustering or a cell-type column (default: the active clustering)."))] = "",
) -> str:
    """Why each group of a column was labelled or flagged, in plain words from scView's own rules
    (never a probability): its markers against random genes (best marker-database p vs what 1 in
    10 random lists reach), median genes and % mito against the typical cluster with the cluster
    QC flags that fired, a doublet-like cluster's two parents, database types as shares of the
    evidence (named groupings only), and the label scView gave ('Low-quality cells') or would give
    itself in an AI annotation, or the note the AI got. Markers must be computed first; the
    marker lines need the marker databases (marker_db_check downloads them)."""
    d = await _get(f"{_ds(dataset_id)}/clusters/evidence", column=column or None)
    return dumps({"dataset_id": dataset_id, **pick(d, "column", "n_groups", "markers", "typing", "annotation", "notes"),
                  "clusters": [{"group": c["group"], "n_cells": c["n_cells"], "label": c["label"],
                                "decided": c["decided"], "evidence": [ln["text"] for ln in c["lines"]]}
                               for c in d["clusters"]]})


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
    "Label" in the UI): name a cluster (filter {"op":"and","items":[{"kind":"category",
    "column":"leiden","values":["3"]}]}), or build a column such as a sample id from donor AND
    day. An unknown key in the filter is refused. Saved with the dataset."""
    idx = await _indices(dataset_id, filter)
    if len(idx) == 0:
        raise ToolError("No cell matches the filter; nothing was labelled.")
    r = await _post(f"{_ds(dataset_id)}/labels", {"column": column, "label": label,
                                                    "indices": idx.tolist(), "new_column": new_column})
    out = pick(r, "column", "label", "created", "n_labels", "note")
    out.update(dataset_id=dataset_id, n_labelled=int(len(idx)),
               n_changed=sum(len(g.get("indices") or []) for g in r.get("previous") or []
                             if g.get("value") != r.get("label")))
    return dumps(out)


@tool(WRITE)
async def combine_columns(
    dataset_id: DatasetId,
    columns: Annotated[list[str], Field(min_length=2, max_length=4, description=(
        "Two to four obs columns, in the order their values are joined (e.g. ['donor_id', 'depot']).")
    )],
    name: Annotated[str, Field(min_length=1, description="Name of the new user annotation column.")],
    separator: Annotated[str, Field(min_length=1, max_length=5, description=(
        "Text between the values."))] = " | ",
) -> str:
    """Make a new annotation column whose value is each cell's values of 2-4 columns joined
    ("HFD01 | ING"): a sample id from donor AND depot (or day, library) in one call, instead
    of one set_labels call per pair. A cell missing a value in any column gets none. Saved with
    the dataset; set_labels can edit it afterwards."""
    r = await _post(f"{_ds(dataset_id)}/labels/combine",
                    {"columns": columns, "name": name, "separator": separator})
    return dumps({"dataset_id": dataset_id, **r})


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
        w.append(f"under-replicated: '{r.get('n_donors_column') or r.get('replicate_key')}' values "
                 f"A={r.get('n_donors_a')}, B={r.get('n_donors_b')}; a cell-level p value overstates "
                 "the evidence, check with method='pseudobulk' or per_donor_expression")
    ts = r.get("transcriptome_size") or {}
    if ts.get("caution"):
        w.append(f"transcriptome size differs {num(ts.get('ratio'))}x between the sides "
                 f"(median counts {num(ts.get('median_a'))} vs {num(ts.get('median_b'))})")
    if r.get("dropped_samples"):
        w.append(f"samples left out: {', '.join(map(str, r['dropped_samples'][:10]))}")
    w += [f"technology: {x}" for x in r.get("technology_warnings") or []]
    w += [f"replicates: {x}" for x in r.get("replicate_warnings") or []]
    w += [f"covariate: {x}" for x in r.get("covariate_notes") or []]
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
    within_value: Annotated[str | list[str] | None, Field(description=(
        "Restrict to one subgroup: its value (e.g. CD14 Mono), or a list of values whose cells are "
        "pooled."))] = None,
    group: Annotated[str | None, Field(description="One group vs every other cell (with `column`).")] = None,
    filter: Annotated[CellFilter | None, Field(description="Selected cells vs every other cell.")] = None,
    method: Annotated[Literal["wilcoxon", "t-test", "logreg", "pseudobulk"], Field(description=(
        "wilcoxon / t-test (Welch) / logreg are cell-level. pseudobulk sums each donor's counts "
        "and tests with limma-voom (or DESeq2), the donor-aware test for conditions; between groups only."))] = "wilcoxon",
    replicate_key: Annotated[str | None, Field(description="Donor/sample column (default: detected).")] = None,
    pb_method: Annotated[Literal["deseq2", "voom", "voombygroup", "edger_ql"], Field(
        description="Pseudobulk model: limma-voom (default), voomByGroup (a variance per group), edgeR QL (quasi-likelihood F-test; FDR held like voom), or DESeq2 (its FDR runs high below 6 donors per side).")] = "voom",
    covariates: Annotated[list[str] | None, Field(max_length=4, description=(
        "Pseudobulk: per-sample obs columns in the design before the condition (sex, batch, age); "
        "refused when one varies within a sample or is confounded with the condition."))] = None,
    numeric_covariates: Annotated[list[str] | None, Field(max_length=4, description=(
        "Pseudobulk: covariates read as numbers, one slope each (an age; text such as "
        "'54-year-old stage' is read as 54). Added to covariates."))] = None,
    donor_key: Annotated[str | None, Field(description=(
        "Pseudobulk: the person column the replicate column's samples nest in, for the "
        "repeated-donor warning (default: detected)."))] = None,
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
        if covariates or donor_key or numeric_covariates:
            if method != "pseudobulk":
                raise ToolError("covariates, numeric_covariates and donor_key are for method='pseudobulk'.")
            body.update({"covariates": covariates, "donor_key": donor_key,
                         "numeric_covariates": numeric_covariates})
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
        **pick(r, "replicate_key", "n_donors_a", "n_donors_b", "n_donors_column", "n_samples_a", "n_samples_b",
               "design", "pb_method", "counts_source", "min_pct", "n_genes_skipped", "covariates"),
        "n_genes_tested": r.get("n_genes_tested") or len(genes),
        "rule": f"padj <= {padj_max} and |log2FC| >= {lfc_min}" if has_p else f"|coef| ranked, |log2FC| >= {lfc_min}",
        "n_up": len(up), "n_down": len(down), "columns": cols,
        "up": [[num(g.get(c)) for c in cols] for g in up[:n]],
        "down": [[num(g.get(c)) for c in cols] for g in down[:n]],
        "warnings": _de_warnings(r),
    }
    if not out.get("covariates"):
        out.pop("covariates", None)
    if r.get("fold_change_note"):
        out["fold_change_note"] = r["fold_change_note"]
    sig_note = (r.get("significance") or {}).get("note")
    if sig_note and not up and not down:
        out["none_significant"] = sig_note
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
    common_tested_only: Annotated[bool, Field(description=(
        "Count only the genes both results tested (the cell-level detection cut-off and the "
        "pseudobulk count filter keep different genes)."))] = False,
    limit: Limit = DEFAULT_LIMIT,
) -> str:
    """Compare two DE results' significant genes (the DE overlap the UI draws): how many each
    calls, how many both call, their sign agreement, and the top genes only one of them calls
    (largest |log2FC| first). Use case: cell-level Wilcoxon vs donor-level pseudobulk."""
    d = await _post(f"{_ds(dataset_id)}/de/overlap", {
        "result_ids": [result_a, result_b], "direction": "both", "padj_max": padj_max, "lfc_min": lfc_min,
        "restrict_to_common_tested": common_tested_only})
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
        "notes": [_api_words(x) for x in d.get("notes") or []],
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
    paired: Annotated[bool | None, Field(description=(
        "Default: paired when every donor is in every condition. True: pair the sample values found "
        "in every condition and leave the others out (the result says how many)."))] = None,
    exclude_samples: Annotated[list[str] | None, Field(description=(
        "Values of the sample column to leave out (e.g. one donor's samples, for a leave-one-out)."))] = None,
    within_column: Annotated[str | None, Field(description=(
        "Test shares among a subset only: its column (e.g. a broad cell class)."))] = None,
    within_value: Annotated[str | list[str] | None, Field(description=(
        "The subset's value, or a list of values, of within_column (e.g. 'Immune': T cells among "
        "immune cells)."))] = None,
    donor_key: Annotated[str | None, Field(description=(
        "Person column the samples nest in, for the repeated-donor warning (default: detected)."))] = None,
    covariates: Annotated[list[str] | None, Field(max_length=4, description=(
        "Per-sample obs columns in the design before the condition (sex, batch; the animal column "
        "blocks on the animal when the condition varies within it, e.g. depot)."))] = None,
    limit: Limit = DEFAULT_LIMIT,
) -> str:
    """Do cell types make up different shares of the cells between conditions, with donors as
    replicates (propeller-style: per-sample proportions, limma moderated t/F, BH FDR). Returns
    the stored result_id; `comparison` names the conditions, subset and samples left out."""
    body = {"condition_key": condition_key, "group_key": group_key, "sample_key": sample_key,
            "conditions": conditions, "reference": reference, "transform": transform, "paired": paired,
            "exclude_samples": exclude_samples, "within_column": within_column,
            "within_value": within_value, "donor_key": donor_key, "covariates": covariates}
    d = await _post(f"{_ds(dataset_id)}/composition/test", body)
    groups = [_flat(g) for g in d.pop("groups", []) or []]
    d.pop("samples", None)  # per-sample shares: in get_result's table
    cols = list(dict.fromkeys(k for g in groups for k in g))
    out = {"dataset_id": dataset_id, **d, "groups": table(groups, cols, clamp(limit))}
    return dumps(out, _max_chars(limit))


def _api_words(note: str) -> str:
    """A DE overlap note in this tool's terms: the UI's checkbox is an argument here."""
    return note.replace('tick "only genes tested in all" to count only those',
                        "pass common_tested_only=true to count only those")


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
        cols = ["label", "label_fraction", "size", "n_counted", "log_fc", "pval", "spatial_fdr", "top_sample",
                "top_sample_share"]
        by_cond = list((nh.get("counts") or {}).values())
        nh["n_counted"] = [int(sum(c[i] for c in by_cond)) for i in range(len(nh["spatial_fdr"]))] \
            if by_cond else None
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
    significant at spatial FDR, the labels they fall in, the graph's k and representation,
    and which sample fills each one: samples_in_significant). Cells in neither condition are
    not counted but the neighbourhoods span the whole graph (the plan warns). Rebuild the graph
    on another representation with run_pipeline(["neighbors"], {"neighbors_use_rep": "X_pca"}).
    The dataset is not changed."""
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
    offset: Annotated[int, Field(ge=0, description=(
        "The group to start from: page through many groups with the result's next_offset."))] = 0,
) -> str:
    """Each gene's mean expression and share of expressing cells in each group (the dot plot's
    numbers). mean is over all of a group's cells; fraction is the share with a value above 0.
    The default layer may be raw counts (then mean is mean counts): the result names it.
    Groups come a page at a time (at most limit, fewer when many genes): next_offset gives the
    next page."""
    d = await _get(f"{_ds(dataset_id)}/expression/by-group", genes=genes, groupby=groupby, layer=layer)
    out = {"dataset_id": dataset_id, "groupby": d["groupby"], "layer": d["layer"],
           **pick(d, "layer_label"), "genes": d["genes"], "n_groups": len(d["groups"]), "offset": offset,
           "note": "mean[i][j] and fraction[i][j] are group i, gene j"}
    missing = [x for x in genes if x not in d["genes"]]
    if missing:
        out["genes_missing"] = missing
    # As many groups as fit the page (Liu 2021: 197 genes x 100 groups came back as 25 groups
    # with no way to the rest), each whole.
    budget = _max_chars(limit) - len(dumps(out, 10**9)) - 200
    end = offset
    while end < min(len(d["groups"]), offset + clamp(limit)):
        size = len(dumps([d["groups"][end], d["mean"][end], d["fraction"][end]], 10**9))
        if end > offset and size > budget:
            break
        budget -= size
        end += 1
    out.update(groups=d["groups"][offset:end], mean=d["mean"][offset:end], fraction=d["fraction"][offset:end])
    if end < len(d["groups"]):
        out["next_offset"] = end
    return dumps(out, 10**9)


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
                      "permutations", "collections", "warning", "panel_caution", "de_method", "result_id"),
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


@tool(WRITE)
async def score_gene_set(
    dataset_id: DatasetId,
    genes: Annotated[list[str], Field(min_length=1, max_length=2000, description="The set's gene symbols.")],
    name: Annotated[str, Field(max_length=200, description=(
        "The set's name, e.g. HALLMARK_TNFA_SIGNALING_VIA_NFKB. With a name the score is also saved "
        "as an obs column score_<name> (saved_column), offered in Compare colourings and Colour by; "
        "scoring the same name again replaces it. Empty: not saved."))] = "",
    groupby: Annotated[str, Field(description="Summarise the per-cell scores by this obs column "
                                              "(default: the active clustering).")] = "",
    method: Annotated[Literal["score_genes", "aucell", "ssgsea", "gsva"], Field(description=(
        "score_genes: scanpy's mean minus a matched control set (default); aucell: the set's share of "
        "each cell's top-ranked genes; ssgsea / gsva: rank walks (refused on very large data)."))] = "score_genes",
    sample: Annotated[str, Field(description=(
        "Donor / sample column: the mean score per sample within each group (samples with at "
        "least 10 cells there), the donor-level view of a signature."))] = "",
    condition: Annotated[str, Field(description=(
        "With sample: the condition column; each group's donor-level test between two of its "
        "values (paired signed-rank + paired t when donors are in both, else rank-sum + Welch t), "
        "as per_donor_expression tests a gene."))] = "",
    conditions: Annotated[list[str] | None, Field(min_length=2, max_length=2, description=(
        "With condition: [reference, tested], the two values to compare (needed when the column "
        "has more than two; cells with any other value are left out). mean_diff = tested minus "
        "reference."))] = None,
    limit: Limit = DEFAULT_LIMIT,
    offset: Annotated[int, Field(ge=0, description="With sample: the per-sample row to start from.")] = 0,
) -> str:
    """Score every cell for a gene set (scanpy score_genes by default, or AUCell / ssGSEA /
    GSVA, on log-normalised values) and summarise the scores per group (mean, median, share
    above 0). With sample (and condition): each sample's mean score within each group, and
    per group the donor-level test between two conditions, in one call. With name the score
    is also saved as an obs column (saved_column; remove_saved_score drops it)."""
    if (condition or conditions) and not sample:
        raise ToolError("A donor-level test needs the sample (donor) column: give sample.")
    if conditions and not condition:
        raise ToolError("conditions needs the condition column it is values of.")
    body: dict[str, Any] = {"gene_set": genes, "method": method}
    if name.strip():
        body.update(score_name=name.strip(), save=True)
    d = await _post(f"{_ds(dataset_id)}/genesets/score", body)
    scores = np.asarray(d.pop("scores"), dtype=np.float64)
    if not groupby:
        groupby = (await _get(_ds(dataset_id))).get("active_clustering") or ""
    out: dict[str, Any] = {"dataset_id": dataset_id,
                           **{k: v for k, v in d.items() if k != "score_name"
                              and not (k in ("saved_column", "save_note") and v is None)}}
    out["genes_found"] = out.get("genes_found", [])[:50]
    if sample:
        if not groupby:
            raise ToolError("No grouping column: give groupby (e.g. the cell types).")
        cols = {"group": await _obs_values(dataset_id, groupby), "sample": await _obs_values(dataset_id, sample)}
        if condition:
            cols["condition"] = await _obs_values(dataset_id, condition)
        out.update(groupby=groupby, sample=sample,
                   **_per_sample_scores(scores, cols, conditions, clamp(limit), offset))
        if condition:
            out["condition"] = condition
        return dumps(out, 10**9)  # a page of limit rows: its size is bounded already
    if groupby:
        labels = await _obs_values(dataset_id, groupby)
        frame = pd.DataFrame({"g": labels.astype(str).where(labels.notna()), "s": scores})
        agg = frame.groupby("g", observed=True)["s"].agg(
            n="size", mean="mean", median="median", share_above_0=lambda s: float((s > 0).mean()))
        agg = agg.sort_values("mean", ascending=False).reset_index().rename(columns={"g": "group"})
        out["groupby"] = groupby
        out.update(rows(agg.to_dict("records"), limit=clamp(limit)))
    else:
        out["summary"] = {"mean": float(scores.mean()), "median": float(np.median(scores))}
    return dumps(out, _max_chars(limit))


@tool(WRITE)
async def remove_saved_score(
    dataset_id: DatasetId,
    column: Annotated[str, Field(min_length=1, max_length=200, description=(
        "The saved score's obs column (score_<set> or activity_<source>, the saved_column a scoring "
        "returned). Only a column scView saved as a score can be removed."))],
) -> str:
    """Remove a saved per-cell score: its obs column and its record. The dataset's own columns
    and hand labels are never removed; scoring the set again brings the column back."""
    try:
        _, data = await api.request("DELETE", f"{_ds(dataset_id)}/scores/{column}")
    except api.RouteError as e:
        raise _fail(e) from None
    return dumps({"dataset_id": dataset_id, **json.loads(data)}, compact.MAX_CHARS)


SAMPLE_MIN_CELLS = 10  # per sample x group, as per_donor_expression (core/donor_expression.py)


def _per_sample_scores(scores: np.ndarray, cols: dict[str, pd.Series], conditions: list[str] | None,
                       limit: int, offset: int) -> dict[str, Any]:
    """Each (group, condition, sample)'s mean score over its cells (at least
    SAMPLE_MIN_CELLS of them), and with a condition, per group the donor-level test of
    core/paired_design.py between the two conditions (Liu 2021 built 67 label columns and
    ~200 calls for this)."""
    from scview.core import paired_design

    def text(s: pd.Series) -> pd.Series:
        return s.astype(str).where(s.notna())

    frame = pd.DataFrame({k: text(v) for k, v in cols.items()})
    frame["s"] = scores
    has_cond = "condition" in frame
    out: dict[str, Any] = {"min_cells": SAMPLE_MIN_CELLS}
    if has_cond:
        present = list(dict.fromkeys(frame["condition"].dropna()))
        order = [str(c) for c in cols["condition"].dtype.categories] if isinstance(
            cols["condition"].dtype, pd.CategoricalDtype) else sorted(present)
        levels = [c for c in order if c in present]
        if conditions:
            missing = [c for c in conditions if c not in present]
            if missing:
                raise ToolError(f"No cell has condition {', '.join(map(repr, missing))}; its values are: "
                                f"{', '.join(levels[:30])}.")
            pair = [str(c) for c in conditions]
        elif len(levels) == 2:
            pair = levels
        else:
            raise ToolError(f"The condition column has {len(levels)} values ({', '.join(levels[:30])}): "
                            "pick two with conditions=[reference, tested].")
        n_left = int(frame["condition"].notna().sum() - frame["condition"].isin(pair).sum())
        frame = frame[frame["condition"].isin(pair)]
        out.update(conditions=pair, direction=f"mean_diff = {pair[1]} minus {pair[0]}")
        if n_left:
            out["cells_left_out"] = f"{n_left:,} cells with another condition value"
    keys = ["group", "condition", "sample"] if has_cond else ["group", "sample"]
    agg = frame.dropna(subset=keys).groupby(keys, observed=True, sort=True)["s"].agg(
        n_cells="size", mean_score="mean").reset_index()
    small = agg[agg["n_cells"] < SAMPLE_MIN_CELLS]
    agg = agg[agg["n_cells"] >= SAMPLE_MIN_CELLS]
    if len(small):
        out["n_dropped"] = int(len(small))
        out["dropped"] = [" / ".join(str(r[k]) for k in keys) + f" ({int(r['n_cells'])} cells)"
                          for _, r in small.head(20).iterrows()]
    recs = agg.to_dict("records")
    out["values"] = table(recs[offset:], [*keys, "n_cells", "mean_score"], limit)
    out["values"]["n_total"] = len(recs)
    out["offset"] = offset
    if offset + limit < len(recs):
        out["next_offset"] = offset + limit
    if has_cond:
        groups = list(dict.fromkeys(agg["group"]))
        points = [{"group": r["group"], "condition": r["condition"], "sample": r["sample"],
                   "value": r["mean_score"]} for r in recs]
        both = frame.dropna(subset=["sample", "condition"])
        design = paired_design.detect_design(set(zip(both["sample"], both["condition"])), out["conditions"])
        out["design"] = design
        if design["paired"]:
            tests = paired_design.group_paired_tests(points, out["conditions"], groups)
            tcols = ["group", "n_pairs", "mean_diff", "wilcoxon_p", "ttest_p", "caution"]
        else:
            tests = paired_design.group_unpaired_tests(points, out["conditions"], groups)
            tcols = ["group", "n_a", "n_b", "mean_diff", "mannwhitney_p", "welch_p", "caution"]
        out["test"] = ("paired: Wilcoxon signed-rank + paired t over the donors in both" if design["paired"]
                       else "unpaired: Wilcoxon rank-sum + Welch t on the donors' mean scores")
        out["tests"] = table([{"group": g, **t} for g, t in tests.items()], tcols, 200)
    return out


# --- results, figures, jobs --------------------------------------------------------------------------

@tool(READ)
async def list_results(
    dataset_id: DatasetId,
    kind: Annotated[str | None, Field(description="Only this kind (e.g. de, markers, enrichment, gsea, composition, milo).")] = None,
    limit: Limit = DEFAULT_LIMIT,
) -> str:
    """The dataset's stored results (DE, markers, enrichment, composition, ...), newest first:
    id, kind, label, and whether computed on an earlier version of the data (stale). Markers and
    enrichment listed here are earlier ones a re-run replaced; the current ones live in the
    dataset and come from get_markers / enrichment."""
    d = await _get(f"{_ds(dataset_id)}/results", kind=kind)
    res = d.get("results") or []
    return dumps({"dataset_id": dataset_id, **rows(res, ["id", "kind", "label", "created_at", "stale", "stale_reason", "status"],
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
                                                            "stale", "stale_reason", "params")}
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
    within_value: Annotated[str | list[str] | None, Field(description=(
        "Only one subgroup: its value (e.g. CD14 Monocyte), or a list of values whose cells are "
        "pooled."))] = None,
) -> list:
    """A PNG figure of the dataset (at most 1600 px, under 300 KB). A dataset over 50,000 cells
    is drawn from a random 50,000 (said in the caption). The same views are in the browser UI."""
    if isinstance(within_value, list) and not within_value:
        within_value = None
    if (within_column is None) != (within_value is None):
        raise ToolError("Give both within_column and within_value, or neither.")
    within_values = [] if within_value is None else (
        [within_value] if isinstance(within_value, str) else [str(v) for v in within_value])
    within_value = ", ".join(within_values) if within_values else None  # for the captions
    scope = f" in {within_column} = {within_value}" if within_column else ""
    if kind == "dotplot":
        if not genes:
            raise ToolError("A dot plot needs genes.")
        if within_column:
            d = await _dotplot_within(dataset_id, genes, groupby, layer, within_column, within_values)
        else:
            d = await _get(f"{_ds(dataset_id)}/expression/by-group", genes=genes, groupby=groupby, layer=layer)
        caption = f"{len(d['groups'])} groups of {d['groupby']}{scope}, layer {d.get('layer_label') or d['layer']}"
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
            raise ToolError(column_message(within_column, list(adaptor.adata.obs.columns)))
        mask = col.astype(str).where(col.notna()).isin(within_values).to_numpy()
        if not mask.any():
            raise ToolError(f"No cell has {within_column} = '{within_value}'.")
        return mask

    def _values_of_gene(g: str) -> np.ndarray:
        if g not in set(adaptor._var_names_for_layer(lay or adaptor.default_expression_layer())):
            raise ToolError(f"Gene '{g}' is not in the expression layer.")
        return adaptor.get_expression([g], layer=lay)[:, 0]

    def _layer_words() -> str:
        # What the values are ("Log-normalized (raw)", "Counts (raw)"), not the bare key.
        return adaptor.layer_label(lay or adaptor.default_expression_layer())

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
                    raise ToolError(column_message(color_by, list(adaptor.adata.obs.columns)))
                label = color_by
            elif gene:
                values, label = _values_of_gene(gene), f"{gene}, {_layer_words()}"
            # A published embedding can leave cells out (NaN coordinates: CELLxGENE sets often
            # do): draw the placed cells only, sampled among themselves, and say how many.
            placed = np.flatnonzero(np.isfinite(np.asarray(xy, dtype=float)).all(axis=1))
            if len(placed) == 0:
                raise ToolError(f"No cell has a position on '{embedding}' (every coordinate is missing).")
            if len(placed) < n:
                pick_ = plots.sample(len(placed))
                rows_ = placed if pick_ is None else placed[pick_]
                note = (f"{len(placed):,} of {n:,} cells placed on {embedding} ({n - len(placed):,} have "
                        "no coordinates, not drawn)" if pick_ is None else
                        f"random {len(rows_):,} of the {len(placed):,} cells placed on {embedding} "
                        f"({n - len(placed):,} of {n:,} have no coordinates)")
                if focus is not None:
                    note += (f"; {int(focus[placed].sum()):,} of {int(focus.sum()):,} with "
                             f"{within_column} = {within_value} placed")
                    if not focus[placed].any():
                        raise ToolError(f"No cell with {within_column} = '{within_value}' has a "
                                        f"position on '{embedding}'.")
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
            raise ToolError(column_message(by, list(adaptor.adata.obs.columns)))
        values = _values_of_gene(gene)
        if focus is not None:  # the subgroup's cells only, sampled among themselves
            values, groups = values[focus], groups[focus]
            rows_ = plots.sample(len(values))
            note = f"{len(values):,} cells with {within_column} = {within_value}" + (
                f" (random {len(rows_):,} drawn)" if rows_ is not None else "")
        if rows_ is not None:
            values, groups = values[rows_], groups.iloc[rows_]
        return plots.violin(values, groups, f"{gene} by {by}{scope}", f"{gene}, {_layer_words()}", note), note

    try:
        png, note = await asyncio.to_thread(_draw)
    except ToolError:
        raise
    except Exception as e:  # a refusal on a large dataset, a bad column type
        raise ToolError(f"Could not draw the plot: {e}") from None
    return [Image(data=png, format="png"), TextContent(type="text", text=f"{kind}: {note}, {len(png):,} bytes")]


async def _dotplot_within(dataset_id: str, genes: list[str], groupby: str, layer: str,
                          column: str, values: str | list[str]) -> dict[str, Any]:
    """The dot plot's numbers over one subgroup's cells (``values`` of ``column``, one or
    several): the by-group route's selection row, once per group of ``groupby`` among those
    cells (in the column's order)."""
    values = [values] if isinstance(values, str) else list(values)
    flt = CellFilter(op="and", items=[{"kind": "category", "column": column, "values": values}])
    within = await _indices(dataset_id, flt)
    if len(within) == 0:
        raise ToolError(f"No cell has {column} = '{', '.join(values)}'.")
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
        out.update(genes=r["genes"], layer=r["layer"], layer_label=r.get("layer_label"))
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
