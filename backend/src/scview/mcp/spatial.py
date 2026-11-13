"""The MCP tools for spatial data and for pooling datasets: wrappers over the routes the
UI's spatial tabs and Compare > Joint embedding call (``api/v1/spatial_autocorr.py``,
``spatial_domains.py``, ``spot_deconvolution.py``, ``integration.py``).

A run streams (SSE) as in the UI, so each is a job (``mcp/jobs.py``); the tool waits up
to ``WAIT_SECONDS`` for it and answers with the result when it is done by then (most
runs on a slide take a second or two), else with the job id for ``job_status``. Only
the arguments given are sent, so the route's own defaults apply. Replies keep the
route's fields: a table comes back as its first ``limit`` rows with the total, a field
this module does not know (a newer route's) is passed on in the same compact form, and a
large matrix is named by its shape instead of sent. The tools register on import
(``mcp/server.py`` imports this module).
"""

from __future__ import annotations

import io
from typing import Annotated, Any, Literal

import numpy as np
import pandas as pd
from mcp.server.mcpserver.exceptions import ToolError
from pydantic import Field

from scview.mcp import api, jobs
from scview.mcp.compact import DEFAULT_LIMIT, clamp, dumps, num, pick, table
from scview.mcp.tools import (
    P0_NOTE,
    READ,
    WRITE,
    DatasetId,
    Limit,
    _check_params,
    _ds,
    _get,
    _line,
    _max_chars,
    _post,
    _raw,
    tool,
)

WAIT_SECONDS = 20.0  # a run that ends by then is answered at once
MAX_SECTION_COLUMNS = 12  # sections shown side by side in a per-section table
MAX_MATRIX_ROWS = 40  # a matrix with more rows (labels x labels x distances) is named, not sent
NEXT = "job_status(job_id, wait_seconds=50) until complete"


def _body(**kw: Any) -> dict[str, Any]:
    """The arguments given (not None): the route's defaults stand for the rest."""
    return {k: v for k, v in kw.items() if v is not None}


def _shape(v: Any) -> list[int]:
    out = []
    while isinstance(v, list):
        out.append(len(v))
        v = v[0] if v else None
    return out


def _compact(v: Any, n: int) -> Any:
    """A route's value made small: a list of records as a table (first ``n`` rows, the
    total), a matrix of more than ``MAX_MATRIX_ROWS`` rows as its shape, the rest as is."""
    if isinstance(v, dict):
        return {str(k): _compact(x, n) for k, x in v.items() if x is not None}
    if isinstance(v, list) and v and all(isinstance(x, dict) for x in v):
        cols = list(dict.fromkeys(k for x in v[:n] for k in x))
        return table([{k: _compact(x.get(k), n) for k in cols if k in x} for x in v], cols, n)
    if isinstance(v, list) and len(v) > MAX_MATRIX_ROWS and v and isinstance(v[0], list):
        return {"shape": _shape(v), "note": "not shown here (a large array): the UI draws it"}
    return v


# Bookkeeping of a saved run that says nothing about its result.
NOISE = ("available", "created_at", "version", "run_id", "cells_fingerprint", "labels_fingerprint",
         "inputs_fingerprint", "stage_seconds", "threads", "peak_rss_gb", "visited_pairs")


def _rest(d: dict[str, Any], skip: tuple[str, ...], n: int) -> dict[str, Any]:
    """Every field of ``d`` but ``skip`` and the bookkeeping, compact: what a newer route
    adds comes through. The run's memory as its peak only."""
    out = {k: _compact(v, n) for k, v in d.items() if k not in skip and k not in NOISE and k != "resources"
           and v is not None}
    peak = (d.get("resources") or {}).get("peak_gb") if isinstance(d.get("resources"), dict) else None
    if peak is not None:
        out["peak_gb"] = peak
    return out


def _sections_wide(rows: list[dict[str, Any]], key: str, n: int) -> dict[str, Any] | None:
    """Per row (a gene) each section's value of ``key`` from the row's ``by_library``
    ({section: {I, z, ...}}): a table gene x sections, at most MAX_SECTION_COLUMNS of them
    side by side (the reply says how many there are)."""
    libs = list(dict.fromkeys(lib for r in rows for lib in (r.get("by_library") or {})))
    if not libs:
        return None
    shown = libs[:MAX_SECTION_COLUMNS]
    recs = [{"gene": r.get("gene"), **{lib: ((r.get("by_library") or {}).get(lib) or {}).get(key) for lib in shown}}
            for r in rows]
    out = {"value": key, **table(recs, ["gene", *shown], n)}
    if len(libs) > len(shown):
        out["sections_shown"] = f"{len(shown)} of {len(libs)}"
    return out


async def _started(job: jobs.Job, limit: int) -> str:
    """The job's result when it ends within WAIT_SECONDS, else its id to follow."""
    await jobs.wait(job, WAIT_SECONDS)
    out = job.summary()
    if job.status == "running":
        out["next"] = NEXT
    return dumps(out, _max_chars(limit))


def _result(job: jobs.Job) -> dict[str, Any]:
    r = job.events.get("result")
    if not isinstance(r, dict):
        raise ToolError("The run ended without a result.")
    return r


# --- spatially variable genes (Moran's I) ------------------------------------------------------

Section = Annotated[str | None, Field(max_length=256, description=(
    "One section only: its library_id (get_dataset lists the sections). Default: every section."))]


def _autocorr_reply(r: dict[str, Any], dataset_id: str, n: int) -> dict[str, Any]:
    rows = r.get("rows") or []
    out = {"dataset_id": dataset_id, **_rest(r, ("rows", "created_at"), n),
           "genes_ranked": table(rows, ["gene", "I", "z", "pval", "pval_adj"], n)}
    wide = _sections_wide(rows[:n], "I", n)
    if wide:
        out["by_section"] = wide
    out["genes_ranked"]["note"] = "highest I first; get_spatial_autocorr pages through every gene"
    if any(r.get("pval") == 0 for r in rows[:n]):
        out["note"] = P0_NOTE
    return out


@tool(WRITE)
async def spatial_autocorr(
    dataset_id: DatasetId,
    genes: Annotated[Literal["hvg", "all"] | list[str], Field(description=(
        "hvg (the highly variable genes, default), all, or a list of gene symbols."))] = "hvg",
    library_id: Section = None,
    layer: Annotated[str | None, Field(max_length=256, description=(
        "Expression layer (default: the log-normalised values the UI uses)."))] = None,
    limit: Limit = DEFAULT_LIMIT,
) -> str:
    """Spatially variable genes: Moran's I per gene on the tissue graph (= squidpy's
    spatial_autocorr), on every section or one (library_id). Saved with the dataset as the
    UI's Spatial genes tab runs it, with a result_id. A job: waits up to 20 s and answers
    with the genes of highest I, else returns job_id for job_status."""
    body = _body(genes=genes, library_id=library_id, layer=layer)

    def finish(job: jobs.Job) -> dict[str, Any]:
        return _autocorr_reply(_result(job), dataset_id, clamp(limit))

    job = jobs.new("spatial_autocorr", dataset_id, f"{_ds(dataset_id)}/spatial/autocorr/run-stream", body,
                   progress_line=_line, finish=finish)
    return await _started(job, limit)


@tool(READ)
async def get_spatial_autocorr(
    dataset_id: DatasetId,
    library_id: Annotated[str | None, Field(max_length=256, description=(
        "The last run on this section; 'all': the last run over every section. Default: the "
        "last run."))] = None,
    result_id: Annotated[str | None, Field(max_length=64, description=(
        "A kept run's id (from spatial_autocorr or list_results): every gene it tested, a page "
        "at a time."))] = None,
    limit: Limit = DEFAULT_LIMIT,
    offset: Annotated[int, Field(ge=0, description="With result_id: the gene row to start from.")] = 0,
) -> str:
    """A saved Moran's I run: the last one (on a section, or over all), or every gene of a
    kept run by result_id, highest I first."""
    n = clamp(limit)
    if result_id:
        _, data = await _raw(f"{_ds(dataset_id)}/spatial/autocorr/{result_id}.csv")
        df = pd.read_csv(io.BytesIO(data))
        out = {"dataset_id": dataset_id, "result_id": result_id, "columns": list(df.columns),
               "n_rows": len(df), "offset": offset,
               "rows": [[num(v) for v in r] for r in df.iloc[offset:offset + n].itertuples(index=False)]}
        if offset + n < len(df):
            out["next_offset"] = offset + n
        return dumps(out, _max_chars(limit))
    r = await _get(f"{_ds(dataset_id)}/spatial/autocorr", library_id=library_id)
    return dumps(_autocorr_reply(r, dataset_id, n), _max_chars(limit))


# --- domains, neighbourhood enrichment, niches, co-occurrence ------------------------------------

def _per_section(per_library: Any, names: list[str], n: int, what: str) -> dict[str, Any] | None:
    """{section: [count per group]} as a table section x groups, the first n sections."""
    if not isinstance(per_library, dict) or not per_library:
        return None
    recs = [{"section": lib, **{f"{what} {names[i] if i < len(names) else i}": c for i, c in enumerate(counts)}}
            for lib, counts in per_library.items() if isinstance(counts, list)]
    cols = list(dict.fromkeys(k for r in recs for k in r))
    return table(recs, cols, n)


def _domains_reply(r: dict[str, Any], dataset_id: str, n: int) -> dict[str, Any]:
    k = int(r.get("k") or 0)
    names = [str(i) for i in range(k)]
    out = {"dataset_id": dataset_id,
           **_rest(r, ("per_library", "scan", "stage_seconds", "inputs_fingerprint", "cells_fingerprint",
                       "run_id", "version"), n)}
    sections = _per_section(r.get("per_library"), names, n, "domain")
    if sections:
        out["per_section"] = sections
    scan = r.get("scan")
    if isinstance(scan, dict) and scan.get("ks"):
        out["stability_scan"] = pick(scan, "ks", "mean", "peaks", "suggested", "has_peak", "repeats")
    return out


async def _spatial_run(dataset_id: str, kind: str, path: str, body: dict[str, Any], limit: int,
                       reply) -> str:
    def finish(job: jobs.Job) -> dict[str, Any]:
        return reply(_result(job), dataset_id, clamp(limit))

    job = jobs.new(kind, dataset_id, f"{_ds(dataset_id)}{path}", body, progress_line=_line, finish=finish)
    return await _started(job, limit)


Overwrite = Annotated[bool | None, Field(description=(
    "Replace an earlier run's column of the same name (default: refused, naming a free one)."))]
Seed = Annotated[int | None, Field(ge=0, le=2**31 - 1, description="Random seed (default 0).")]


@tool(WRITE)
async def spatial_domains(
    dataset_id: DatasetId,
    k: Annotated[int | None, Field(ge=2, le=40, description="Number of domains (default 7).")] = None,
    suggest_k: Annotated[bool, Field(description=(
        "Scan k from k_min to k_max (repeats seeds each) and use the most stable k (the UI's "
        "Suggest k); k is then ignored."))] = False,
    k_min: Annotated[int | None, Field(ge=2, le=40, description="Suggest k: smallest k scanned (default 2).")] = None,
    k_max: Annotated[int | None, Field(ge=2, le=40, description="Suggest k: largest k scanned (default 15).")] = None,
    repeats: Annotated[int | None, Field(ge=2, le=20, description="Suggest k: seeds per k (default 5).")] = None,
    n_hops: Annotated[int | None, Field(ge=0, le=6, description=(
        "Rings of neighbours averaged into each spot's features (default 3)."))] = None,
    n_pcs: Annotated[int | None, Field(ge=2, le=500, description=(
        "Principal components used (default 10; Harmony's when the dataset has several sections and them)."))] = None,
    seed: Seed = None,
    column: Annotated[str | None, Field(max_length=64, description=(
        "Obs column to write (default spatial_domain)."))] = None,
    overwrite: Overwrite = None,
    limit: Limit = DEFAULT_LIMIT,
) -> str:
    """Spatial domains (CellCharter-style: each spot's PCs with its neighbours' means, a
    Gaussian mixture), saved as an obs column; per section, each domain's spots. A job:
    waits up to 20 s, else returns job_id for job_status. Colour the tissue by the column
    with plot(embedding='spatial', color_by=column)."""
    body = _body(k=k, k_min=k_min, k_max=k_max, repeats=repeats, n_hops=n_hops, n_pcs=n_pcs, seed=seed,
                 column=column, overwrite=overwrite)
    if suggest_k:
        body["k"] = None  # null: the route scans and picks
    return await _spatial_run(dataset_id, "spatial_domains", "/spatial/domains/run-stream", body, limit,
                              _domains_reply)


def _pairs(labels: list[str], z: list[list[Any]], count: list[list[Any]] | None, n: int) -> dict[str, Any]:
    """Pairs of different labels, most enriched and most avoided, and each label with itself."""
    pairs = [{"a": labels[i], "b": labels[j], "z": z[i][j],
              **({"links": count[i][j]} if count is not None else {})}
             for i in range(len(labels)) for j in range(i + 1, len(labels)) if z[i][j] is not None]
    cols = ["a", "b", "z", "links"]
    return {"most_together": table(sorted([p for p in pairs if p["z"] > 0], key=lambda p: -p["z"]), cols, n),
            "most_apart": table(sorted([p for p in pairs if p["z"] < 0], key=lambda p: p["z"]), cols, n),
            "self": table([{"label": labels[i], "z": z[i][i]} for i in range(len(labels))], ["label", "z"], n)}


def _enrichment_reply(r: dict[str, Any], dataset_id: str, n: int) -> dict[str, Any]:
    labels = [str(x) for x in r.get("labels") or []]
    out = {"dataset_id": dataset_id,
           **_rest(r, ("zscore", "count", "labels", "sizes", "cells_fingerprint", "labels_fingerprint",
                       "version"), n)}
    out["labels"] = table([{"label": lab, "n": s} for lab, s in zip(labels, r.get("sizes") or [], strict=False)],
                          ["label", "n"], n)
    if r.get("zscore"):
        out.update(_pairs(labels, r["zscore"], r.get("count"), n))
        out["note"] = ("z > 0: neighbours more often than the labels shuffled within each section; z < 0: "
                       "less often.")
    return out


@tool(WRITE)
async def nhood_enrichment(
    dataset_id: DatasetId,
    column: Annotated[str, Field(max_length=256, description="Obs column of labels (cell types, domains).")],
    n_perms: Annotated[int | None, Field(ge=10, le=10_000, description="Label shuffles (default 1000).")] = None,
    seed: Seed = None,
    limit: Limit = DEFAULT_LIMIT,
) -> str:
    """Neighbourhood enrichment (= squidpy's nhood_enrichment): which labels sit next to
    which on the tissue graph more or less often than the labels shuffled within each
    section (z). Saved with the dataset. A job: waits up to 20 s, else job_id."""
    return await _spatial_run(dataset_id, "nhood_enrichment", "/spatial/nhood-enrichment/run-stream",
                              _body(column=column, n_perms=n_perms, seed=seed), limit, _enrichment_reply)


def _niches_reply(r: dict[str, Any], dataset_id: str, n: int) -> dict[str, Any]:
    labels = [str(x) for x in r.get("labels") or []]
    niches = [str(x) for x in r.get("niches") or []]
    out = {"dataset_id": dataset_id,
           **_rest(r, ("fractions", "per_library", "labels", "cells_fingerprint", "labels_fingerprint",
                       "version"), n)}
    fr = r.get("fractions") or []
    recs = []
    for i, row in enumerate(fr):
        vals = np.asarray([np.nan if v is None else v for v in row], dtype=float)
        top = np.argsort(-np.nan_to_num(vals, nan=-1))[:3]
        recs.append({"niche": niches[i] if i < len(niches) else str(i),
                     "n": (r.get("sizes") or [None] * (i + 1))[i],
                     "mostly": "; ".join(f"{labels[j]} {vals[j]:.0%}" for j in top if j < len(labels)
                                         and np.isfinite(vals[j]) and vals[j] > 0)})
    if recs:
        out["niches"] = table(recs, ["niche", "n", "mostly"], n)
    sections = _per_section(r.get("per_library"), niches, n, "niche")
    if sections:
        out["per_section"] = sections
    return out


@tool(WRITE)
async def spatial_niches(
    dataset_id: DatasetId,
    label_column: Annotated[str | None, Field(max_length=256, description=(
        "Obs column of cell labels whose mix around each cell makes the niches (cells, Xenium, "
        "Visium HD 8 um bins)."))] = None,
    n_niches: Annotated[int | None, Field(ge=2, le=40, description="Number of niches (default 8).")] = None,
    k_window: Annotated[int | None, Field(ge=2, le=200, description=(
        "Nearest neighbours in each window (default 20)."))] = None,
    seed: Seed = None,
    column: Annotated[str | None, Field(max_length=64, description="Obs column to write (default spatial_niche).")] = None,
    overwrite: Overwrite = None,
    limit: Limit = DEFAULT_LIMIT,
) -> str:
    """Composition niches: groups of places with the same mix of labels around them,
    saved as an obs column; each niche's main labels and its count per section. The route
    says what it takes on spots (Visium spots mix cells). A job: waits up to 20 s, else job_id."""
    body = _body(label_column=label_column, n_niches=n_niches, k_window=k_window, seed=seed, column=column,
                 overwrite=overwrite)
    return await _spatial_run(dataset_id, "spatial_niches", "/spatial/niches/run-stream", body, limit,
                              _niches_reply)


CO_DISTANCES = 5  # distances shown per pair: the first usable one to the last, evenly
MIN_PAIRS = 30  # as core/spatial_cooccurrence.MIN_PAIRS: fewer pairs at a distance say little


def _cooccurrence_reply(r: dict[str, Any], dataset_id: str, n: int, label: str | None = None) -> dict[str, Any]:
    labels = [str(x) for x in r.get("labels") or []]
    out = {"dataset_id": dataset_id,
           **_rest(r, ("occ", "count", "labels", "sizes", "distances", "cells_fingerprint", "labels_fingerprint",
                       "version"), n)}
    out["labels"] = table([{"label": lab, "n": s} for lab, s in zip(labels, r.get("sizes") or [], strict=False)],
                          ["label", "n"], n)
    occ = np.asarray(r.get("occ") or [], dtype=float)
    dist = list(r.get("distances") or [])
    if occ.ndim != 3 or not dist:
        return out
    count = np.asarray(r.get("count") or np.zeros_like(occ), dtype=float)
    usable = [t for t in range(occ.shape[2]) if (count[:, :, t] >= MIN_PAIRS).sum() > len(labels)]
    first = usable[0] if usable else 0
    picks = sorted({int(round(x)) for x in np.linspace(first, occ.shape[2] - 1, CO_DISTANCES)})
    unit = r.get("distance_unit") or ""
    names = [f"at {num(float(dist[t]))} {unit}".strip() for t in picks]
    # squidpy's score is the same both ways round when the counts are: each pair once then.
    symmetric = bool(np.allclose(occ, occ.transpose(1, 0, 2), equal_nan=True))
    k = len(labels)
    pairs = [(i, j) for i in range(k) for j in range(k) if i != j and (i < j or not symmetric)]
    if label is not None:
        pairs = [(i, j) if labels[i] == label else (j, i) for i, j in pairs if label in (labels[i], labels[j])
                 and (symmetric or labels[i] == label)]
    recs = []
    for i, j in pairs:
        recs.append({"label": labels[i], "with": labels[j], **{nm: occ[i, j, t] for nm, t in zip(names, picks)},
                     "pairs_at_first": int(count[i, j, picks[0]])})
    recs.sort(key=lambda x: -(x[names[0]] if np.isfinite(x[names[0]]) else -1))
    if label is not None and not recs:
        raise ToolError(f"No label '{label}' in this run; its labels: {', '.join(labels[:40])}.")
    out["pairs"] = table(recs, ["label", "with", *names, "pairs_at_first"], n)
    out["pairs"]["note"] = (f"score = how much more often 'with' is found within that distance of 'label' than "
                            f"among all pairs at that range (> 1 together, < 1 apart); {len(dist)} distances, "
                            f"{len(picks)} shown; pairs_at_first: pairs counted at the first one shown.")
    return out


@tool(WRITE)
async def cooccurrence(
    dataset_id: DatasetId,
    column: Annotated[str, Field(max_length=256, description="Obs column of labels (cell types, domains).")],
    interval: Annotated[int | None, Field(ge=2, le=200, description="Distances scored (default 50).")] = None,
    max_distance: Annotated[float | None, Field(gt=0, description=(
        "Score distances evenly from 0 to this, in the positions' units (default: squidpy's range)."))] = None,
    label: Annotated[str | None, Field(max_length=256, description=(
        "Reply with this label's pairs only (the run covers every pair either way)."))] = None,
    limit: Limit = DEFAULT_LIMIT,
) -> str:
    """Co-occurrence (= squidpy's co_occurrence): how much more often one label is found
    within each distance of another than among all pairs at that range. Pairs are counted
    within each section. Saved with the dataset. A job: waits up to 20 s, else job_id."""

    def reply(r: dict[str, Any], did: str, n: int) -> dict[str, Any]:
        return _cooccurrence_reply(r, did, n, label)

    return await _spatial_run(dataset_id, "cooccurrence", "/spatial/cooccurrence/run-stream",
                              _body(cluster_key=column, interval=interval, max_distance=max_distance), limit, reply)


# --- spot composition (deconvolution of spots against a reference) -------------------------------

Reference = Annotated[str, Field(max_length=128, description=(
    "The reference dataset's id: single cells or nuclei with cell-type labels, same species."))]
CelltypeColumn = Annotated[str | None, Field(max_length=256, description=(
    "The reference's cell-type column (default: guessed from its columns)."))]
Method = Annotated[Literal["rctd", "nnls"] | None, Field(description=(
    "rctd (default: RCTD-style, a platform effect per section) or nnls (non-negative least squares)."))]
Exclude = Annotated[list[str] | None, Field(max_length=500, description="Reference labels to leave out.")]
Merge = Annotated[dict[str, str] | None, Field(description="Reference labels to merge: {label: new name}.")]


def _reference_reply(r: dict[str, Any], dataset_id: str, n: int) -> dict[str, Any]:
    est = r.get("estimate") or {}
    out = {"dataset_id": dataset_id, **_rest(r, ("estimate", "label_columns"), n)}
    if est:
        out["estimate"] = {"peak_gb": round((est.get("peak_bytes") or 0) / 2**30, 2),
                           **pick(est, "seconds", "seconds_range", "note", "large_reference")}
    cols = r.get("label_columns") or []
    out["label_columns"] = [c.get("column") if isinstance(c, dict) else c for c in cols][:40]
    return out


@tool(READ)
async def spot_composition_reference(
    dataset_id: DatasetId,
    reference_id: Reference,
    celltype_column: CelltypeColumn = None,
    method: Method = None,
    exclude_labels: Exclude = None,
    merge_labels: Merge = None,
    limit: Limit = DEFAULT_LIMIT,
) -> str:
    """Before spot_composition: whether a reference can split this dataset's spots into
    cell types (ok, or the reason), its cells per type (under 25 flagged), catch-all labels
    worth leaving out (suggested_exclude), species against the spots', shared genes, and the
    run's memory and time. Reads the reference's tables only."""
    path = f"{_ds(dataset_id)}/spot-deconvolution/reference/{reference_id}"
    if exclude_labels or merge_labels:
        if not celltype_column:
            raise ToolError("Leaving out or merging labels needs celltype_column.")
        r = await _post(f"{path}/labels", _body(celltype_column=celltype_column, method=method,
                                                exclude_labels=exclude_labels, merge_labels=merge_labels))
    else:
        r = await _get(path, celltype_column=celltype_column, method=method)
    return dumps(_reference_reply(r, dataset_id, clamp(limit)), _max_chars(limit))


def _composition_reply(r: dict[str, Any], dataset_id: str, n: int) -> dict[str, Any]:
    types = [str(t) for t in r.get("types") or []]
    out = {"dataset_id": dataset_id,
           **_rest(r, ("types", "type_stats", "libraries", "clusters", "s_e", "sources", "created_at", "reference",
                       "resources"), n)}
    out["reference"] = pick(r.get("reference") or {}, "id", "name", "celltype_column", "n_reference_cells")
    stats = sorted(r.get("type_stats") or [], key=lambda t: -(t.get("rna_mean") or 0))  # largest share first
    out["types"] = table(stats, ["type", "n_reference_cells", "rna_mean", "rna_max", "n_top"], n)
    libs = r.get("libraries") or []
    shown = [str(t.get("type")) for t in stats if str(t.get("type")) in types][:MAX_SECTION_COLUMNS] or \
        types[:MAX_SECTION_COLUMNS]
    recs = [{"section": lib.get("id"), "n_spots": lib.get("n_spots"), "n_fitted": lib.get("n_fitted"),
             **{t: (lib.get("bulk") or [None] * len(types))[types.index(t)] for t in shown}} for lib in libs]
    out["per_section_shares"] = table(recs, ["section", "n_spots", "n_fitted", *shown], n)
    if len(types) > len(shown):
        out["per_section_shares"]["types_shown"] = f"{len(shown)} of {len(types)}"
    out["note_shares"] = ("per_section_shares: each type's share of the section's RNA over its fitted spots; types "
                          "with the largest mean share first")
    return out


@tool(WRITE)
async def spot_composition(
    dataset_id: DatasetId,
    reference_id: Reference,
    celltype_column: Annotated[str, Field(max_length=256, description=(
        "The reference's cell-type column (spot_composition_reference shows the candidates)."))],
    method: Method = None,
    min_umi: Annotated[int | None, Field(ge=0, le=100_000, description=(
        "Spots with fewer counts are not fitted (default: the unit's own floor)."))] = None,
    exclude_labels: Exclude = None,
    merge_labels: Merge = None,
    limit: Limit = DEFAULT_LIMIT,
) -> str:
    """Spot composition: each Visium spot's cell-type shares from a single-cell reference
    (RCTD-style), saved as obsm spot_deconv and obs spot_deconv_top; per section each
    type's share, and the notes (types over their expected share, spots not fitted). A job:
    waits up to 20 s, else job_id; check the reference first with spot_composition_reference."""
    body = _body(reference_id=reference_id, celltype_column=celltype_column, method=method, min_umi=min_umi,
                 exclude_labels=exclude_labels, merge_labels=merge_labels)
    return await _spatial_run(dataset_id, "spot_composition", "/spot-deconvolution/run-stream", body, limit,
                              _composition_reply)


# --- joint embedding (pooling datasets, e.g. one CELLxGENE file per slide) -----------------------

DatasetIds = Annotated[list[str], Field(min_length=2, max_length=50, description=(
    "The datasets to pool (2 or more; the server caps how many, 8 by default)."))]
Labels = Annotated[list[str] | None, Field(description=(
    "A name per dataset for the joint 'dataset' column, in the same order (default: their names)."))]


def _plan_reply(p: dict[str, Any], n: int) -> dict[str, Any]:
    merge = p.get("merge") or {}
    shared = p.get("shared_columns") or []
    out = {**pick(p, "ok", "est_cells", "blocking", "notes"),
           "preflight": pick(p.get("preflight") or {}, "level", "sentence", "peak_stage", "seconds", "suggestions",
                             "notes"),
           "datasets": table(p.get("datasets") or [], ["id", "name", "label", "n_cells", "n_genes", "counts_source",
                                                       "error"], n),
           "merge": pick(merge, "intersection", "union", "recommended_join", "est_cells", "est_genes",
                         "suspicious_low_overlap", "warnings"),
           "shared_columns": shared[:60], "n_shared_columns": len(shared),
           "dropped_columns": table(p.get("dropped_columns") or [], ["column", "reason", "datasets"], n)}
    rec = merge.get("reconciliation") or {}
    if rec.get("needed"):
        out["merge"]["gene_names"] = pick(rec, "feasible", "message", "overlap_before", "overlap_after")
    out.update(_rest(p, ("datasets", "merge", "shared_columns", "dropped_columns", *out), n))
    return out


@tool(READ)
async def joint_embedding_plan(dataset_ids: DatasetIds, labels: Labels = None, limit: Limit = DEFAULT_LIMIT) -> str:
    """Before joint_embedding: whether the datasets can be pooled (counts found, genes
    shared), the cells and genes of the joint dataset, the obs columns it keeps and those it
    leaves out with why, design notes (a condition only in one dataset), and its memory and
    time (preflight). Changes nothing."""
    p = await _post("/integrate/plan", _body(dataset_ids=dataset_ids, labels=labels))
    return dumps(_plan_reply(p, clamp(limit)), _max_chars(limit))


@tool(WRITE)
async def joint_embedding(
    dataset_ids: DatasetIds,
    labels: Labels = None,
    name: Annotated[str | None, Field(max_length=120, description="The joint dataset's name.")] = None,
    method: Annotated[Literal["harmony", "scanorama", "bbknn", "none"], Field(description=(
        "Batch correction across the datasets (default harmony); none: one plain embedding."))] = "harmony",
    join: Annotated[Literal["inner", "outer"], Field(description=(
        "inner: the genes every dataset has (default); outer: all genes, missing ones as 0."))] = "inner",
    params: Annotated[dict[str, Any] | None, Field(description=(
        "Pipeline parameters for the joint run, as run_pipeline takes them (only the ones to change)."))] = None,
) -> str:
    """Pool datasets into a new joint dataset (the UI's Compare > Joint embedding): counts
    merged with a 'dataset' column (each dataset's sections become library_id values named by its label),
    then QC metrics, normalisation, HVG, PCA, batch correction, clustering and UMAP. The
    sources are not changed. A job: returns job_id at once (minutes); job_status gives the
    new dataset_id; cancel_job stops it and keeps nothing. Check joint_embedding_plan first."""
    p = _check_params(params)
    plan = await _post("/integrate/plan", _body(dataset_ids=dataset_ids, labels=labels))
    if not plan.get("ok"):
        raise ToolError("Not started: " + "; ".join(plan.get("blocking") or ["the plan was refused."]))
    seen: dict[str, Any] = {}

    def progress(event: str, data: Any) -> str | None:
        if isinstance(data, dict) and data.get("dataset_id") and "dataset_id" not in seen:
            seen["dataset_id"] = data["dataset_id"]  # the first stage event names the joint dataset
        return _line(event, data)

    async def cancel(job: jobs.Job) -> None:
        if not seen.get("dataset_id"):
            raise api.RouteError(404, "not started")
        await api.post(f"{_ds(seen['dataset_id'])}/assessment/cancel")

    def finish(job: jobs.Job) -> dict[str, Any]:
        done = job.events.get("complete") or {}
        out = {**pick(done, "dataset_id", "name"),
               **pick(job.events.get("pipeline_complete") or {}, "status", "steps_run", "elapsed_seconds", "notes")}
        out["next"] = "get_dataset(dataset_id); plot(dataset_id, embedding='spatial') draws each section"
        return out

    body = _body(dataset_ids=dataset_ids, labels=labels, name=name, method=method, join=join, params=p or None)
    job = jobs.new("joint_embedding", None, "/integrate/run-stream", body, progress_line=progress, finish=finish,
                   cancel_route=cancel)
    return dumps({"job_id": job.id, "status": job.status, "est_cells": plan.get("est_cells"),
                  "preflight": (plan.get("preflight") or {}).get("sentence"), "next": NEXT})
