"""The MCP tools for the other modalities: CRISPR screens (guide assignment, Mixscape), cell
hashing, and 10x Multiome ATAC (fragment QC, LSI + gene activity, WNN, differential peaks,
peak-gene links, motifs, chromVAR). Wrappers over the routes the UI's Perturbation,
Cell hashing, Accessibility and Protein tabs call (``api/v1/perturbation.py``, ``hashing.py``,
``atac.py``, ``atac_fragments.py``, ``atac_links.py``, ``atac_motifs.py``, ``wnn.py``).

A run that streams (SSE) is a job, as the spatial tools' (``mcp/spatial.py``): the tool waits
up to 20 s and answers with the result, else with the job id for ``job_status``. Only the
arguments given are sent, so the route's own defaults apply. A stored result (differential
peaks, links, motif enrichment, chromVAR) has a result_id that ``get_result`` pages through.
The tools register on import (``mcp/server.py`` imports this module).
"""

from __future__ import annotations

from typing import Annotated, Any, Literal

import numpy as np
from mcp.server.mcpserver.exceptions import ToolError
from pydantic import Field

from scview.mcp import api, jobs
from scview.mcp.compact import DEFAULT_LIMIT, clamp, dumps, num, pick, table
from scview.mcp.spatial import _body, _compact, _rest, _result, _started
from scview.mcp.tools import P0_NOTE, READ, WRITE, DatasetId, Limit, _ds, _get, _line, _max_chars, _post, tool

GET_RESULT = "get_result(dataset_id, result_id) pages through every row"


def _rows(records: list[dict[str, Any]] | None, n: int, skip: tuple[str, ...] = ()) -> dict[str, Any]:
    """A list of records as a compact table, without ``skip`` columns."""
    recs = [{k: v for k, v in r.items() if k not in skip} for r in records or []]
    cols = list(dict.fromkeys(k for r in recs[:n] for k in r))
    return table(recs, cols, n)


# --- CRISPR screens: guide assignment and Mixscape ------------------------------------------------

def _per_target(p: dict[str, Any], n: int) -> dict[str, Any]:
    """Cells per target with its guides (the preview lists them; the saved call has the mapping)."""
    control, mapping = p.get("control"), p.get("mapping") or {}
    recs = []
    for t in p.get("per_target") or []:
        guides = t.get("guides") or sorted(g for g, v in mapping.items() if v == t.get("target"))
        recs.append({"target": t.get("target"), "n_cells": t.get("n_cells"),
                     "control": True if t.get("target") == control else None,
                     "n_guides": len(guides), "guides": ", ".join(map(str, guides[:8]))})
    recs.sort(key=lambda r: -(r["n_cells"] or 0))
    guided = any(r["n_guides"] for r in recs)  # a saved call keeps no guide list: leave the columns out
    return table(recs, ["target", "n_cells", "control", *(("n_guides", "guides") if guided else ())], n)


def _assignment_reply(p: dict[str, Any], dataset_id: str, n: int, saved: bool) -> dict[str, Any]:
    out: dict[str, Any] = {"dataset_id": dataset_id, "saved": saved,
                           **pick(p, "source", "source_label", "column", "params", "control", "n_cells", "counts")}
    out["n_targets"] = p["n_targets"] if p.get("n_targets") is not None else len(p.get("targets") or [])
    out["per_target"] = _per_target(p, n)
    if p.get("targets_without_cells"):
        # the preview lists {target, guides}, the saved call the names
        out["targets_without_cells"] = [t.get("target") if isinstance(t, dict) else str(t)
                                        for t in p["targets_without_cells"]]
    auto, full = p.get("auto_mapping"), p.get("mapping") or {}
    if p.get("guides") or full:
        out["n_guides"] = len(p.get("guides") or full)
    overrides = {g: t for g, t in full.items() if auto is not None and auto.get(g) != t}
    if overrides:
        out["mapping_overrides"] = overrides
    if not saved:
        out["note"] = ("A preview: nothing saved. guide_assignment(save=true) with the same arguments writes "
                       "guide_call, the perturbation column and n_guides.")
    return out


@tool(WRITE)
async def guide_assignment(
    dataset_id: DatasetId,
    save: Annotated[bool, Field(description=(
        "false (default): a preview, nothing changed. true: save each cell's guide and target (guide_call, a "
        "perturbation column, n_guides) as the Perturbation tab's Save does."))] = False,
    source: Annotated[Literal["counts", "cellranger", "column"] | None, Field(description=(
        "counts: the guide counts (obsm guide_counts, CRISPR Guide Capture); cellranger: Cell Ranger's "
        "protospacer calls; column: an obs column holding each cell's guide or target. Default: the suggested one."))] = None,
    column: Annotated[str | None, Field(max_length=256, description="source column: the obs column.")] = None,
    min_umi: Annotated[float | None, Field(ge=0, description=(
        "counts: a guide counts in a cell from this many UMIs (default 5, the Papalexi 2021 rule)."))] = None,
    min_ratio: Annotated[float | None, Field(ge=0, description=(
        "counts: the top guide must have this many times the second's UMIs, else the cell is 'multiple' "
        "(default 2; 1 = the top guide always; 0 = any second guide makes it multiple; between 0 and 1 is "
        "refused)."))] = None,
    mapping: Annotated[dict[str, str] | None, Field(description=(
        "Guide -> target overrides (e.g. {\"eGFPg1\": \"eGFP\"}); the rest are mapped from the guide names."))] = None,
    control: Annotated[str | None, Field(max_length=256, description=(
        "The control target (default: detected, e.g. NT or non-targeting)."))] = None,
    limit: Limit = DEFAULT_LIMIT,
) -> str:
    """CRISPR screen guide assignment: each cell's guide and target from guide counts, Cell
    Ranger calls or an obs column, with the cells per target, the control, cells with several
    guides ('multiple') and none. A preview unless save=true. Then mixscape."""
    body = _body(source=source, column=column, min_umi=min_umi, min_ratio=min_ratio, mapping=mapping,
                 control=control)
    n = clamp(limit)
    if not save:
        p = await _post(f"{_ds(dataset_id)}/perturbation/preview", body)
        return dumps(_assignment_reply(p, dataset_id, n, False), _max_chars(limit))
    s = await _post(f"{_ds(dataset_id)}/perturbation/assign", body)
    call = s.get("call") or {}
    out = _assignment_reply(call, dataset_id, n, True)
    out["columns_written"] = call.get("columns")
    out["next"] = "mixscape(dataset_id) classifies each target's cells as KO or NP (not perturbed)"
    return dumps(out, _max_chars(limit))


_TARGET_COLUMNS = ("target", "n_cells", "n_ko", "n_np", "ko_fraction", "n_de_genes", "rounds", "top_de_genes", "note")


def _mixscape_reply(r: dict[str, Any], dataset_id: str, n: int) -> dict[str, Any]:
    targets = [{**{k: t.get(k) for k in _TARGET_COLUMNS if k != "top_de_genes"},
                "top_de_genes": ", ".join(str(g.get("gene")) for g in (t.get("top_de_genes") or [])[:6])}
               for t in r.get("targets") or []]
    targets.sort(key=lambda t: -(t.get("ko_fraction") or 0))
    out: dict[str, Any] = {
        "dataset_id": dataset_id,
        **pick(r, "labels", "control", "split_by", "n_cells", "n_control", "n_targets", "n_genes", "gene_source",
               "expression_source", "summary", "warnings", "excluded", "seconds", "guide_column",
               "replicate_column"),
        "params": pick(r.get("params") or {}, "n_neighbors", "n_dims", "logfc_threshold", "pval_cutoff",
                       "min_de_genes", "iter_num", "lda", "de_genes"),
        "targets": table(targets, list(_TARGET_COLUMNS), n),
    }
    rank = {t["target"]: i for i, t in enumerate(targets)}  # the targets' order: highest KO share first
    for key in ("per_guide", "per_replicate"):
        if r.get(key):
            recs = sorted(r[key], key=lambda x: rank.get(x.get("target"), len(rank)))
            out[key] = _rows(recs, n)
    if r.get("protein"):
        out["protein"] = _protein_reply(r["protein"], n)
    out["columns_written"] = ["mixscape_class", "mixscape_class_global", "mixscape_p_ko"]
    out["note"] = ("KO: the guide's cells that moved away from the controls; NP: not perturbed (escapers). "
                   "per_guide / per_replicate: each target's KO share per guide and per replicate. "
                   + ("protein: each protein's shift (CLR) vs the controls in all / KO / NP cells per target; "
                      "flags = a protein effect with no RNA KO class. " if r.get("protein") else "")
                   + "list_results(kind='mixscape') keeps each run.")
    return out


_PROTEIN_COLUMNS = ("target", "protein", "ko_fraction", "all_n", "all_diff", "all_padj", "ko_diff", "ko_padj",
                    "np_diff", "np_padj", "n_replicates", "same_sign", "p_replicate", "unit", "significant",
                    "flag")


def _protein_reply(pr: dict[str, Any], n: int) -> dict[str, Any]:
    """The Mixscape run's protein check: flags in words, then the target x protein rows
    (flags first, then significant, by cell-level adjusted p)."""
    recs = []
    for r in pr.get("rows") or []:
        rec = {k: r.get(k) for k in ("target", "protein", "ko_fraction", "n_replicates", "same_sign", "p_replicate",
                                     "unit", "significant", "flag")}
        for g in ("all", "ko", "np"):
            s = r.get(g) or {}
            rec[f"{g}_diff"], rec[f"{g}_padj"] = s.get("diff"), s.get("padj")
            if g == "all":
                rec["all_n"] = s.get("n")
        recs.append(rec)
    recs.sort(key=lambda x: (not x["flag"], not x["significant"],
                             1.0 if x["all_padj"] is None else x["all_padj"]))
    return {**pick(pr, "layer", "proteins", "control", "n_control", "replicate_column", "notes"),
            "flags": [f["text"] for f in pr.get("flags") or []],
            "rows": table(recs, list(_PROTEIN_COLUMNS), n)}


@tool(WRITE)
async def mixscape(
    dataset_id: DatasetId,
    labels: Annotated[str | None, Field(max_length=256, description=(
        "Obs column of each cell's target (default: the saved guide assignment's perturbation column)."))] = None,
    control: Annotated[str | None, Field(max_length=256, description=(
        "The control value of labels (default: the saved assignment's control)."))] = None,
    split_by: Annotated[str | None, Field(max_length=256, description=(
        "Compare each cell with controls of its own replicate / batch / lane (the UI's 'Within')."))] = None,
    de_genes: Annotated[Literal["all", "variable"] | None, Field(description=(
        "DE genes from: all (every gene, Seurat RunMixscape's rule, default) or variable (the variable "
        "genes, as scView 0.77)."))] = None,
    n_dims: Annotated[int | None, Field(ge=2, le=200, description="PCs for the neighbours (default 15).")] = None,
    n_neighbors: Annotated[int | None, Field(ge=2, le=500, description="Control neighbours per cell (default 20).")] = None,
    logfc_threshold: Annotated[float | None, Field(ge=0, description="DE |log fold change| (default 0.25).")] = None,
    pval_cutoff: Annotated[float | None, Field(gt=0, le=1, description="DE adjusted p (default 0.05).")] = None,
    min_de_genes: Annotated[int | None, Field(ge=1, le=10_000, description=(
        "A target with fewer DE genes gets no KO class (default 5)."))] = None,
    iter_num: Annotated[int | None, Field(ge=1, le=100, description="Mixture rounds (default 10).")] = None,
    lda: Annotated[bool | None, Field(description="Also the LDA of the perturbation signatures (default true).")] = None,
    guide_column: Annotated[str | None, Field(max_length=256, description=(
        "Obs column of each cell's guide for the per-guide KO table (default: the assignment's)."))] = None,
    replicate_column: Annotated[str | None, Field(max_length=256, description=(
        "Obs column of each cell's replicate, for the per-replicate KO table and the protein check's "
        "replicate unit (default: split_by)."))] = None,
    limit: Limit = DEFAULT_LIMIT,
) -> str:
    """Mixscape (Papalexi 2021): per target, which guide cells are knocked out (KO) and which
    escaped (NP), saved as mixscape_class / mixscape_class_global / mixscape_p_ko; the per-target
    table (cells, KO share, DE genes), each target's KO share per guide and per replicate. With
    protein (CITE / ECCITE-seq): each protein's shift vs the controls per target, and flags for a
    protein effect with no RNA KO class (judged on replicate_column / split_by when given). Run
    guide_assignment(save=true) first, or name labels and control. A job: waits up to 20 s,
    else job_id."""
    body = _body(labels=labels, control=control, split_by=split_by, de_genes=de_genes, n_dims=n_dims,
                 n_neighbors=n_neighbors, logfc_threshold=logfc_threshold, pval_cutoff=pval_cutoff,
                 min_de_genes=min_de_genes, iter_num=iter_num, lda=lda, guide_column=guide_column,
                 replicate_column=replicate_column)

    def finish(job: jobs.Job) -> dict[str, Any]:
        return _mixscape_reply(_result(job), dataset_id, clamp(limit))

    job = jobs.new("mixscape", dataset_id, f"{_ds(dataset_id)}/perturbation/mixscape/run-stream", body,
                   progress_line=_line, finish=finish)
    return await _started(job, limit)


@tool(READ)
async def get_mixscape(dataset_id: DatasetId, limit: Limit = DEFAULT_LIMIT) -> str:
    """The last Mixscape run on the current cells: per target, per guide and per replicate."""
    r = await _get(f"{_ds(dataset_id)}/perturbation/mixscape")
    return dumps(_mixscape_reply(r, dataset_id, clamp(limit)), _max_chars(limit))


# --- cell hashing ----------------------------------------------------------------------------------

def _hashing_reply(s: dict[str, Any], dataset_id: str, n: int) -> dict[str, Any]:
    out: dict[str, Any] = {"dataset_id": dataset_id, **pick(s, "n_cells", "tags", "suggested", "doublet_source")}
    call = s.get("call")
    if isinstance(call, dict):
        c = pick(call, "method", "label", "tags", "split_by", "counts", "n_uncalled", "warnings",
                 "saved_with_problems", "thresholds")
        per_call = sorted((call.get("per_call") or {}).items(), key=lambda kv: -kv[1])
        c["per_call"] = table([{"call": k, "n": v} for k, v in per_call], ["call", "n"], n)
        if isinstance(call.get("per_tag"), list):
            c["per_tag"] = _rows(call["per_tag"], n)
        elif isinstance(call.get("per_tag"), dict):
            c["per_tag"] = _compact(call["per_tag"], n)
        if call.get("groups"):
            c["groups"] = _compact(call["groups"], n)
        out["call"] = c
        out["columns_written"] = ["hash_call", "hash_class", "hash_max", "hash_second", "hash_margin"]
    if s.get("samples"):
        out["samples"] = s["samples"]
    if s.get("split_suggestion"):
        out["split_suggestion"] = s["split_suggestion"]
    return out


@tool(WRITE)
async def hashing_demux(
    dataset_id: DatasetId,
    tags: Annotated[list[str] | None, Field(max_length=50, description=(
        "The hashtags (2 to 50): tag counts (obsm hto_counts) and/or antibody features, which then leave the "
        "protein modality. Default: the suggested ones; the error lists every tag when none is suggested."))] = None,
    method: Annotated[Literal["htodemux", "hashedDrops"] | None, Field(description=(
        "htodemux (Seurat HTODemux, default) or hashedDrops (DropletUtils)."))] = None,
    quantile: Annotated[float | None, Field(gt=0.5, lt=1, description=(
        "HTODemux: the positive quantile of each tag's background (default 0.99)."))] = None,
    split_by: Annotated[str | None, Field(max_length=256, description=(
        "Call within each value of this obs column (a lane or channel: the UI's 'Within'), for lanes "
        "counted against different tag panels."))] = None,
    allow_warnings: Annotated[bool, Field(description=(
        "Save a call the checks find unsafe (structural zeros, a zero-ambient tag), as the UI's 'Save anyway'. "
        "Default false: refused with the reason."))] = False,
    limit: Limit = DEFAULT_LIMIT,
) -> str:
    """Cell hashing demultiplexing (HTODemux or hashedDrops): each cell's tag, Singlet /
    Doublet / Negative, saved as hash_call, hash_class, ... (the QC's doublet source then
    prefers it). A call the checks find unsafe is refused with the reason unless
    allow_warnings. Name each tag's sample with hashing_samples."""
    if not tags:
        s = await _get(f"{_ds(dataset_id)}/hashing")
        tags = list(s.get("suggested") or [])
        if not tags:
            have = list(s.get("tags") or []) + [p for p in s.get("proteins") or [] if p not in (s.get("tags") or [])]
            if not have:
                raise ToolError("This dataset has no hashtag counts or antibody features to demultiplex.")
            raise ToolError(f"Name the tags: none is suggested. Tags and antibody features: {', '.join(have[:60])}.")
    body = _body(tags=tags, method=method, quantile=quantile, split_by=split_by, allow_warnings=allow_warnings)
    try:
        s = await api.post(f"{_ds(dataset_id)}/hashing/demux", body)
    except api.RouteError as e:
        # the route's refusal word for word; its "Save anyway" is this tool's allow_warnings
        hint = " (Here: allow_warnings=true.)" if e.status == 409 and "Save anyway" in e.detail else ""
        raise ToolError(e.detail + hint) from None
    return dumps(_hashing_reply(s, dataset_id, clamp(limit)), _max_chars(limit))


@tool(WRITE)
async def hashing_samples(
    dataset_id: DatasetId,
    mapping: Annotated[dict[str, str], Field(description=(
        "Each tag's sample name, e.g. {\"HTO1\": \"donor A\"}; doublets and negatives stay empty."))],
    column: Annotated[str | None, Field(max_length=64, description="Obs column to write (default hash_sample).")] = None,
    limit: Limit = DEFAULT_LIMIT,
) -> str:
    """Write a sample column from the saved hashing call: each singlet's tag named by mapping."""
    s = await _post(f"{_ds(dataset_id)}/hashing/samples", _body(mapping=mapping, column=column))
    return dumps(_hashing_reply(s, dataset_id, clamp(limit)), _max_chars(limit))


# --- ATAC (10x Multiome) ---------------------------------------------------------------------------

def _plan(p: dict[str, Any] | None) -> dict[str, Any] | None:
    if not isinstance(p, dict):
        return None
    return {k: v for k, v in p.items() if not isinstance(v, (list, dict)) or k in ("lsi_dims",)}


@tool(READ)
async def atac_status(
    dataset_id: DatasetId,
    groupby: Annotated[str | None, Field(max_length=256, description=(
        "WNN: the mean RNA weight per group of this obs column."))] = None,
) -> str:
    """The ATAC (10x Multiome) state of a dataset: peaks and genome, LSI / gene activity / WNN
    runs (stale when the cells changed since), what a run would need, the fragments file and its
    QC, the motif scan and chromVAR. The multiome path: atac_fragment_qc, atac_lsi, wnn(modality=
    'atac'), atac_gene_activity, atac_differential, atac_links, atac_motif_enrichment,
    atac_chromvar, plot(kind='coverage')."""
    s = await _get(f"{_ds(dataset_id)}/atac", allow_empty=True, groupby=groupby or "")
    if not s.get("available"):
        return dumps({"dataset_id": dataset_id, "available": False, "reason": s.get("reason")})
    out: dict[str, Any] = {"dataset_id": dataset_id, "available": True,
                           **pick(s, "n_cells", "running", "qc_columns", "plan_reason",
                                                          "wnn_reason")}
    info = s.get("info") or {}
    out["info"] = {k: v for k, v in info.items() if not isinstance(v, (list, dict)) or k == "computed"}
    for key in ("lsi", "wnn"):
        sec = s.get(key)
        if isinstance(sec, dict):
            out[key] = {k: v for k, v in sec.items()
                        if k in ("created_at", "n_clusters", "dims_used", "dropped_first", "n_selected", "stale",
                                 "clusters_column", "mean_rna_weight", "seconds", "method")}
            if key == "wnn" and sec.get("by_group"):
                out[key]["by_group"] = _compact(sec["by_group"], 40)
    if s.get("activity"):
        out["activity"] = {k: v for k, v in s["activity"].items() if not isinstance(v, list)}
    out["plan"] = _plan(s.get("plan"))
    out["wnn_plan"] = _plan(s.get("wnn_plan"))
    try:
        frag = await api.get(f"{_ds(dataset_id)}/atac/fragments")
        out["fragments"] = {k: v for k, v in frag.items() if not isinstance(v, list)
                            and k not in ("qc",)} if isinstance(frag, dict) else frag
        qc = frag.get("qc") if isinstance(frag, dict) else None
        if isinstance(qc, dict):
            out["fragments"]["qc"] = {k: v for k, v in qc.items() if not isinstance(v, (list, dict))}
    except api.RouteError as e:
        out["fragments"] = e.detail
    try:
        m = await api.get(f"{_ds(dataset_id)}/atac/motifs")
        if m.get("available"):
            out["motifs"] = {"genome": m.get("genome"),
                             "scan": pick(m.get("scan") or {}, "ready", "running", "n_peaks", "estimate_seconds"),
                             "chromvar": pick(m.get("chromvar") or {}, "run_id", "n_cells", "seconds", "stale")
                             if m.get("chromvar") else None,
                             "chromvar_estimate": m.get("chromvar_estimate")}
        else:
            out["motifs"] = m.get("reason")
    except api.RouteError as e:
        out["motifs"] = e.detail
    return dumps(out, 12000)


MAX_VALUES = 12  # a list of numbers longer than this (sdev per component, a histogram) is cut
_LEAN_SKIP = ("top_peaks", "stage_seconds", "cells_fingerprint", "inputs_fingerprint", "run_id")


def _lean(v: Any) -> Any:
    """A run's summary without its long number lists (they are for the UI's plots): the
    first MAX_VALUES values and how many there are."""
    if isinstance(v, dict):
        return {k: _lean(x) for k, x in v.items() if k not in _LEAN_SKIP}
    if isinstance(v, list):
        if len(v) > MAX_VALUES and all(isinstance(x, (int, float)) or x is None for x in v):
            return {"first": v[:MAX_VALUES], "n": len(v)}
        return [_lean(x) for x in v]
    return v


def _atac_reply(r: dict[str, Any], dataset_id: str, n: int) -> dict[str, Any]:
    return {"dataset_id": dataset_id, **_lean(_rest(r, ("cells_fingerprint", "inputs_fingerprint"), n))}


def _sections(*keep: str):
    """The reply of a run whose result is the dataset's whole ATAC record: only the ``keep``
    sections (the run's own) and what has been computed, not the earlier runs again."""
    def reply(r: dict[str, Any], dataset_id: str, n: int) -> dict[str, Any]:
        out = _atac_reply({k: v for k, v in r.items() if k in keep or k in ("resources", "peak_gb")}, dataset_id, n)
        computed = (r.get("info") or {}).get("computed")
        if computed:
            out["computed"] = computed
        return out
    return reply


async def _atac_job(dataset_id: str, kind: str, path: str, body: dict[str, Any], limit: int, reply=_atac_reply,
                    after=None) -> str:
    async def finish(job: jobs.Job) -> dict[str, Any]:
        out = reply(_result(job), dataset_id, clamp(limit))
        if after is not None:
            out.update(await after(job))
        return out

    job = jobs.new(kind, dataset_id, f"{_ds(dataset_id)}{path}", body, progress_line=_line, finish=finish)
    return await _started(job, limit)


@tool(WRITE)
async def atac_fragment_qc(dataset_id: DatasetId, limit: Limit = DEFAULT_LIMIT) -> str:
    """Fragment QC of the attached atac_fragments.tsv.gz (+ .tbi): per-cell TSS enrichment and
    nucleosome signal (Signac's definitions), fragments per cell, the length ladder; saved as
    obs columns (atac_tss_enrichment, atac_nucleosome_signal, ...) that the filtering step's ATAC
    cuts use. One pass over the file (minutes for a large one). A job: waits up to 20 s, else job_id."""
    return await _atac_job(dataset_id, "atac_fragment_qc", "/atac/fragments/qc-stream", {}, limit)


@tool(WRITE)
async def atac_lsi(
    dataset_id: DatasetId,
    n_lsi: Annotated[int | None, Field(ge=3, le=200, description="LSI components (default 50).")] = None,
    dims_to: Annotated[int | None, Field(ge=2, le=200, description=(
        "The graph's last LSI component (default: all of them).")) ] = None,
    drop_first: Annotated[Literal["auto", "yes", "no"] | None, Field(description=(
        "Leave out the first component: auto (when it follows depth, r > 0.75; default), yes or no."))] = None,
    resolution: Annotated[float | None, Field(gt=0, le=10, description="Leiden on the ATAC graph (default 1).")] = None,
    activity: Annotated[bool | None, Field(description="Also gene activity (default true).")] = None,
    activity_source: Annotated[Literal["auto", "fragments", "peaks"] | None, Field(description=(
        "Gene activity from: fragments (Signac's GeneActivity: every fragment over the gene body + 2 kb), "
        "peaks (counts in called peaks there), or auto (fragments when a fragments file is attached; default)."))] = None,
    limit: Limit = DEFAULT_LIMIT,
) -> str:
    """ATAC analysis (Signac): TF-IDF on the peaks, LSI with its depth check, a graph, ATAC UMAP
    (X_umap_atac) and Leiden clusters, and gene activity per gene. Saved with the dataset. A job:
    waits up to 20 s, else job_id. Then wnn(modality='atac'), atac_gene_activity, atac_differential."""
    body = _body(n_lsi=n_lsi, dims_to=dims_to, drop_first=drop_first, resolution=resolution, activity=activity,
                 activity_source=activity_source)
    return await _atac_job(dataset_id, "atac_lsi", "/atac/run-stream", body, limit, _sections("lsi", "activity"))


@tool(WRITE)
async def wnn(
    dataset_id: DatasetId,
    modality: Annotated[Literal["atac", "protein"], Field(description=(
        "The second modality beside RNA: atac (LSI, after atac_lsi) or protein (CITE-seq)."))],
    lsi_first: Annotated[int | None, Field(ge=1, le=200, description=(
        "atac: the first LSI component used (default 2, as Seurat's multiome vignette)."))] = None,
    lsi_last: Annotated[int | None, Field(ge=2, le=200, description="atac: the last LSI component (default: all).")] = None,
    protein_pcs: Annotated[int | None, Field(ge=2, le=200, description=(
        "protein: protein PCs (default min(18, proteins - 1)).")) ] = None,
    resolution: Annotated[float | None, Field(gt=0, le=10, description="Leiden on the WNN graph (default 1).")] = None,
    limit: Limit = DEFAULT_LIMIT,
) -> str:
    """Weighted nearest neighbours (Seurat FindMultiModalNeighbors): per-cell RNA vs ATAC or
    protein weights, a joint graph, a WNN UMAP and WNN clusters, saved with the dataset (the
    columns are in the reply). A job: waits up to 20 s, else job_id."""
    if modality == "atac":
        if protein_pcs is not None:
            raise ToolError("protein_pcs is for modality='protein'.")
        return await _atac_job(dataset_id, "wnn", "/atac/wnn/run-stream",
                               _body(lsi_first=lsi_first, lsi_last=lsi_last, resolution=resolution), limit,
                               _sections("wnn"))
    if lsi_first is not None or lsi_last is not None:
        raise ToolError("lsi_first and lsi_last are for modality='atac'.")
    return await _atac_job(dataset_id, "wnn", "/wnn/run-stream", _body(protein_pcs=protein_pcs, resolution=resolution),
                           limit)


@tool(READ)
async def atac_gene_activity(
    dataset_id: DatasetId,
    gene: Annotated[str, Field(max_length=256, description="One gene.")],
    groupby: Annotated[str | None, Field(max_length=256, description=(
        "Obs column of groups (default: the active clustering)."))] = None,
    limit: Limit = DEFAULT_LIMIT,
) -> str:
    """One gene's ATAC gene activity (log1p per 10k, from the last atac_lsi run) per group:
    cells, mean, median and share above 0, highest mean first; its correlation with the gene's
    RNA over all cells, and which activity it is (fragments or peaks)."""
    by = groupby
    if not by:
        d = await _get(_ds(dataset_id))
        by = d.get("active_clustering") or ""
    r = await _get(f"{_ds(dataset_id)}/atac/activity", gene=gene, groupby=by or "")
    out: dict[str, Any] = {"dataset_id": dataset_id, **pick(r, "gene", "kind", "caption", "n_cells", "n_nonzero", "r",
                                                          "violin_error")}
    out["r_note"] = "Pearson r of activity with the gene's RNA over every cell"
    if r.get("violin"):
        recs = []
        for g, vals in r["violin"].items():
            v = np.asarray(vals, dtype=float)
            recs.append({"group": g, "n": int(v.size), "mean": float(v.mean()) if v.size else None,
                         "median": float(np.median(v)) if v.size else None,
                         "pct_above_0": float((v > 0).mean() * 100) if v.size else None})
        recs.sort(key=lambda x: -(x["mean"] or 0))
        out["groupby"] = by
        out["by_group"] = table(recs, ["group", "n", "mean", "median", "pct_above_0"], clamp(limit))
    return dumps(out, _max_chars(limit))


def _pages(rows: dict[str, list[dict[str, Any]]] | None, n: int, per_group: int) -> dict[str, Any]:
    out = {}
    for g, recs in list((rows or {}).items())[:n]:
        out[str(g)] = _rows(recs[:per_group], per_group, skip=("peak_index", "group"))
    return out


def _da_reply(r: dict[str, Any], dataset_id: str, n: int) -> dict[str, Any]:
    groups = list(r.get("groups") or [])
    counts = [{"group": g, "n_tested": (r.get("n_tested") or {}).get(g),
               "n_significant": (r.get("n_significant") or {}).get(g)} for g in groups]
    out: dict[str, Any] = {"dataset_id": dataset_id, "result_id": r.get("id"),
                           **pick(r, "column", "vs", "test", "fold_change", "method", "note", "n_fit"),
                           "params": pick(r.get("params") or {}, "min_pct", "max_peaks", "on", "latent", "lrt_peaks",
                                          "standard_only", "n_peaks_left_out", "n_cells_vs"),
                           "counts": table(counts, ["group", "n_tested", "n_significant"], n),
                           "top_peaks": _pages(r.get("rows"), n, min(n, 5 if len(groups) > 2 else 10))}
    out["next"] = (f"{GET_RESULT} (sort_by='pval_adj'); atac_motif_enrichment(da_id=result_id) tests motifs in "
                   "each group's peaks")
    return out


@tool(WRITE)
async def atac_differential(
    dataset_id: DatasetId,
    column: Annotated[str, Field(max_length=256, description="Obs column of the groups (cell types, clusters).")],
    groups: Annotated[list[str] | None, Field(max_length=50, description=(
        "Only these groups, each vs the rest (default: every group). With vs: exactly one group, A."))] = None,
    vs: Annotated[str | None, Field(max_length=512, description=(
        "A vs B: the one group in groups against this group (Signac FindMarkers ident.1 / ident.2), "
        "not against the rest."))] = None,
    test: Annotated[Literal["lr", "wilcoxon"] | None, Field(description=(
        "lr: logistic regression with a depth covariate (Seurat's LR test; the UI's default); wilcoxon on "
        "TF-IDF (the route's default, faster)."))] = None,
    min_pct: Annotated[float | None, Field(ge=0, le=1, description=(
        "Test peaks open in at least this share of either side (default 0.05)."))] = None,
    max_peaks: Annotated[int | None, Field(ge=10, le=500_000, description="Rows kept per group.")] = None,
    lrt_peaks: Annotated[int | None, Field(ge=1, le=500_000, description=(
        "lr: true likelihood-ratio fits per group (the rest from the score test)."))] = None,
    standard_only: Annotated[bool | None, Field(description=(
        "Leave out chrM, unplaced scaffolds and alt contigs (default true, Signac's keepStandardChromosomes)."))] = None,
    fold_change: Annotated[Literal["signac", "tfidf"] | None, Field(description=(
        "log2fc formula. signac (default): Signac's FoldChange for peaks, mean log1p(TF-IDF) + 1e-4, the numbers "
        "FindMarkers prints. tfidf: mean TF-IDF + 1, scView's before 0.79, about 4x smaller."))] = None,
    limit: Limit = DEFAULT_LIMIT,
) -> str:
    """Differential accessibility: each group's peaks vs the rest, or one group vs another
    (groups=[A], vs=B), with the nearest gene, distance to TSS and peak class; kept as a result
    (result_id) that get_result pages through and atac_motif_enrichment tests. A job: waits up
    to 20 s, else job_id."""
    if vs is not None and (not groups or len(groups) != 1):
        raise ToolError("A vs B needs exactly one group in groups (A) and vs (B).")
    body = _body(column=column, groups=groups, vs=vs, test=test, min_pct=min_pct, max_peaks=max_peaks,
                 lrt_peaks=lrt_peaks, standard_only=standard_only, fold_change=fold_change)
    return await _atac_job(dataset_id, "atac_differential", "/atac/differential", body, limit, _da_reply)


def _links_reply(r: dict[str, Any], dataset_id: str, n: int) -> dict[str, Any]:
    out = {"dataset_id": dataset_id, "result_id": r.get("id"),
           **pick(r, "total", "counts", "method", "caption", "seconds", "stale"),
           "params": _compact(r.get("params") or {}, n),
           "top_links": _rows(r.get("rows"), n)}
    if r.get("missing"):
        out["genes_not_found"] = r["missing"][:20]
    out["next"] = f"{GET_RESULT}; plot(kind='coverage', gene=...) draws the locus"
    return out


@tool(WRITE)
async def atac_links(
    dataset_id: DatasetId,
    genes: Annotated[Literal["variable", "all"] | list[str], Field(description=(
        "variable (the variable genes, default), all, or a list of gene symbols."))] = "variable",
    cells_column: Annotated[str | None, Field(max_length=256, description=(
        "Only some cells: an obs column (with cells_value)."))] = None,
    cells_value: Annotated[list[str] | None, Field(max_length=50, description=(
        "Only some cells: the values of cells_column kept."))] = None,
    limit: Limit = DEFAULT_LIMIT,
) -> str:
    """Peak-gene links (Signac LinkPeaks): peaks within 500 kb of a gene's TSS whose accessibility
    correlates with its expression, against 200 GC- and depth-matched peaks from other
    chromosomes; a candidate element, never proof of regulation. Downloads the genome sequence
    on first use. Kept as a result (result_id). A job: waits up to 20 s, else job_id."""
    if (cells_column is None) != (not cells_value):
        raise ToolError("Give both cells_column and cells_value, or neither.")
    body = _body(cells_column=cells_column, cells_value=cells_value)
    if isinstance(genes, list):
        body.update(genes="list", gene_list=genes)
    else:
        body["genes"] = genes
    return await _atac_job(dataset_id, "atac_links", "/atac/links", body, limit, _links_reply)


async def _scan_then(job: jobs.Job, on_event, dataset_id: str, then) -> None:
    """The motif scan (at once when the peaks were scanned before), then ``then``."""
    job.stage = "scan"
    await api.stream(f"{_ds(dataset_id)}/atac/motifs/scan", {}, on_event, job.disconnect)
    if job.error is not None or "cancelled" in job.events or job.cancel_requested:
        return
    job.events.pop("result", None)
    job.stage = "run"
    await then(on_event)


async def _cancel_scan(job: jobs.Job) -> None:
    if job.stage != "scan":
        raise api.RouteError(404, "not scanning")  # leave the stream: the run stops
    await api.post(f"{_ds(job.dataset_id)}/atac/motifs/scan/cancel")


def _motif_reply(r: dict[str, Any], dataset_id: str, n: int) -> dict[str, Any]:
    out = {"dataset_id": dataset_id, "result_id": r.get("id"),
           **pick(r, "da_id", "column", "genome", "method", "caveats"),
           "params": pick(r.get("params") or {}, "da_test", "p_column", "p_max", "min_pct", "min_log2fc",
                          "background", "seed"),
           "summary": _compact(r.get("summary"), n),
           "top_motifs": _pages(r.get("rows"), n, min(n, 5 if len(r.get("groups") or []) > 2 else 10))}
    out["next"] = GET_RESULT
    return out


@tool(WRITE)
async def atac_motif_enrichment(
    dataset_id: DatasetId,
    da_id: Annotated[str | None, Field(max_length=64, description=(
        "An atac_differential result_id (default: the last one on these cells)."))] = None,
    groups: Annotated[list[str] | None, Field(max_length=50, description="Only these groups of it.")] = None,
    p_column: Annotated[Literal["pval", "pval_adj"] | None, Field(description=(
        "Query peaks by pval (default) or pval_adj."))] = None,
    p_max: Annotated[float | None, Field(gt=0, le=1, description="Query peaks at p below this (default 0.005).")] = None,
    min_pct: Annotated[float | None, Field(ge=0, le=1, description=(
        "Query peaks open in more than this share of the group (default 0.05; Signac's vignette uses 0.2, "
        "which on nuclei keeps mostly promoters and GC-rich motifs)."))] = None,
    min_log2fc: Annotated[float | None, Field(description="Query peaks at log2 fold change above this.")] = None,
    background: Annotated[int | None, Field(ge=100, le=500_000, description=(
        "GC-matched background peaks (default 40000)."))] = None,
    seed: Annotated[int | None, Field(ge=0, le=2**31 - 1, description="Random seed of the background.")] = None,
    limit: Limit = DEFAULT_LIMIT,
) -> str:
    """Motifs enriched in each group's differential peaks (Signac FindMotifs) against
    GC-matched background peaks, JASPAR 2024 (879 motifs), with each TF's own RNA in the group.
    Scans the peaks first when they were not scanned yet (downloads the genome and JASPAR on
    first use; a minute or more). Kept as a result. A job: waits up to 20 s, else job_id."""
    body = _body(da_id=da_id, groups=groups, p_column=p_column, p_max=p_max, min_pct=min_pct,
                 min_log2fc=min_log2fc, background=background, seed=seed)

    async def run(job: jobs.Job, on_event) -> None:
        async def enrich(on_event_) -> None:
            r = await api.post(f"{_ds(dataset_id)}/atac/motifs/enrichment", body)
            await on_event_("result", r)
        await _scan_then(job, on_event, dataset_id, enrich)

    def finish(job: jobs.Job) -> dict[str, Any]:
        return _motif_reply(_result(job), dataset_id, clamp(limit))

    job = jobs.new("atac_motif_enrichment", dataset_id, "", body, progress_line=_line, finish=finish, runner=run,
                   cancel_route=_cancel_scan)
    return await _started(job, limit)


def _chromvar_groups(g: dict[str, Any], n: int) -> dict[str, Any]:
    """Each group's motifs most above the rest (mean z, adjusted p), three per group."""
    recs = []
    for group, rows in (g.get("rows") or {}).items():
        top = sorted(rows or [], key=lambda r: -(r.get("mean_diff") or 0))[:3]
        recs.append({"group": group, "top_motifs": "; ".join(f"{r.get('motif')} z {num(r.get('mean_z'))}" for r in top),
                     "padj_of_first": top[0].get("pval_adj") if top else None})
    out = {"column": g.get("column"), **table(recs, ["group", "top_motifs", "padj_of_first"], n)}
    out["note"] = "each group's mean chromVAR z against the rest's; z of a motif is relative to matched background peaks"
    return out


def _chromvar_reply(r: dict[str, Any], dataset_id: str, n: int) -> dict[str, Any]:
    out = {"dataset_id": dataset_id, "result_id": r.get("run_id"),
           **pick(r, "n_cells", "n_peaks_used", "n_peaks_no_counts", "n_peaks_left_out", "n_peaks_nonstandard",
                  "iterations", "genome", "seconds", "stale", "method", "caveats")}
    out["most_variable"] = _rows(r.get("ranking"), n)
    if any(x.get("p") == 0 for x in (r.get("ranking") or [])[:n]):
        out["note"] = P0_NOTE
    return out


@tool(WRITE)
async def atac_chromvar(
    dataset_id: DatasetId,
    seed: Annotated[int | None, Field(ge=0, le=2**31 - 1, description="Random seed of the background peaks.")] = None,
    column: Annotated[str | None, Field(max_length=256, description=(
        "Also each group's mean z of the most variable motifs, by this obs column."))] = None,
    limit: Limit = DEFAULT_LIMIT,
) -> str:
    """chromVAR motif deviations per cell (JASPAR 2024): motifs ranked by variability, and with
    column each group's mean z. Scans the peaks first when they were not scanned yet. Saved with
    the dataset (a motif's z per cell can be drawn on the plot in the UI). A job: waits up to
    20 s, else job_id."""
    body = _body(seed=seed)

    async def run(job: jobs.Job, on_event) -> None:
        async def chromvar(on_event_) -> None:
            await api.stream(f"{_ds(dataset_id)}/atac/motifs/chromvar", body, on_event_, job.disconnect)
        await _scan_then(job, on_event, dataset_id, chromvar)

    async def finish(job: jobs.Job) -> dict[str, Any]:
        n = clamp(limit)
        out = _chromvar_reply(_result(job), dataset_id, n)
        if column:
            g = await _get(f"{_ds(dataset_id)}/atac/motifs/chromvar/groups", column=column)
            out["by_group"] = _chromvar_groups(g, n)
        return out

    job = jobs.new("atac_chromvar", dataset_id, "", body, progress_line=_line, finish=finish, runner=run,
                   cancel_route=_cancel_scan)
    return await _started(job, limit)

