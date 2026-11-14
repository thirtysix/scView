"""The MCP tools for trajectories and RNA velocity: PAGA, diffusion pseudotime, the genes that
change along it, and the velocity run, its direction check and its result. Wrappers over the
routes the UI's Trajectory and Velocity tabs call (``api/v1/trajectory.py``, ``velocity.py``).

A velocity run streams (SSE), so it is a job as the spatial runs are (``mcp/spatial.py``): the
tool waits up to 20 s and answers with the result, else with the job id for ``job_status``.
PAGA and pseudotime answer at once. Only the arguments given are sent, so the route's own
defaults apply. Warnings come back as the routes word them (a wrong root, arrows that point
backward), never reworded. The arrows, the PAGA graph and a pseudotime by group are drawn by
``plot`` (``overlay``, ``value_column``, ``kind='phase'``). The tools register on import
(``mcp/server.py`` imports this module).
"""

from __future__ import annotations

from typing import Annotated, Any, Literal

import pandas as pd
from mcp.server.mcpserver.exceptions import ToolError
from pydantic import Field

from scview.mcp import jobs
from scview.mcp.compact import DEFAULT_LIMIT, clamp, dumps, pick, table
from scview.mcp.spatial import _body, _result, _started
from scview.mcp.tools import (
    P0_NOTE,
    READ,
    WRITE,
    DatasetId,
    Limit,
    _by_group,
    _ds,
    _get,
    _line,
    _max_chars,
    _obs_values,
    _post,
    tool,
)

TREND_POINTS = 20  # bins of a gene's trend when genes are named (the route's floor is 10)
TRENDS_SHOWN = 5  # ranked genes whose binned trend comes with the table
ATTACH = ("velocity counts are not attached: upload_instructions has the curl recipe (POST "
          "/api/v1/datasets/<id>/velocity/attach with a velocyto .loom, an .h5ad with spliced and unspliced "
          "layers, or STARsolo Velocyto files). An .h5ad uploaded with those layers attaches them itself.")


# --- PAGA ----------------------------------------------------------------------------------------

def _paga_reply(view: dict[str, Any], dataset_id: str, n: int) -> dict[str, Any]:
    rec = view.get("paga")
    if not rec:
        return {"dataset_id": dataset_id, "paga": None,
                "note": "No PAGA graph saved: paga(dataset_id, groupby=<cluster or cell type column>) computes one."}
    groups, sizes = rec.get("groups") or [], rec.get("sizes") or []
    tree = {(int(i), int(j)) for i, j in rec.get("tree") or []}
    degree = [0] * len(groups)
    for i, j in tree:
        degree[i] += 1
        degree[j] += 1
    edges = [{"from": groups[i], "to": groups[j], "connectivity": w, "in_tree": (min(i, j), max(i, j)) in tree}
             for i, j, w in rec.get("edges") or []]
    edges.sort(key=lambda e: -(e["connectivity"] or 0))
    out: dict[str, Any] = {
        "dataset_id": dataset_id, **pick(rec, "groupby", "graph", "model", "n_cells"),
        "n_groups": len(groups),
        "groups": table([{"group": g, "n_cells": s, "tree_links": d} for g, s, d in zip(groups, sizes, degree)],
                        ["group", "n_cells", "tree_links"], max(n, len(groups))),
        "tree": [f"{groups[i]} - {groups[j]}" for i, j in sorted(tree)],
        "tree_ends": [g for g, d in zip(groups, degree) if d == 1],
        "edges": table(edges, ["from", "to", "connectivity", "in_tree"], n),
    }
    if view.get("stale"):
        out["stale"] = view["stale"]
    if view.get("root_group"):
        out["pseudotime_root_group"] = view["root_group"]
    out["note"] = ("connectivity: PAGA's (0-1; edges between the groups against random wiring of the same cells), "
                   "highest first; tree: PAGA's maximum spanning tree. A process usually starts at one of tree_ends: "
                   "pseudotime(root_column=groupby, root_value=<that group>). Draw it: "
                   "plot(embedding=..., overlay=['paga']).")
    return out


@tool(WRITE)
async def paga(
    dataset_id: DatasetId,
    groupby: Annotated[str | None, Field(max_length=256, description=(
        "Obs column whose groups become the graph's nodes (clusters or cell types; at most 200 groups). "
        "Computes PAGA on the neighbour graph and saves it. Leave out to read the saved graph."))] = None,
    limit: Limit = DEFAULT_LIMIT,
) -> str:
    """PAGA (Wolf 2019, scanpy's model): how strongly the groups of a column connect on the
    neighbour graph, the maximum spanning tree, and the groups at its ends (candidate starts for
    pseudotime). Saved with the dataset as the UI's Trajectory tab does; stale says when the
    grouping or the graph changed since. Without groupby: the saved graph."""
    if groupby:
        await _post(f"{_ds(dataset_id)}/trajectory/paga", {"groupby": groupby})
    view = await _get(f"{_ds(dataset_id)}/trajectory/paga")
    return dumps(_paga_reply(view, dataset_id, clamp(limit)), _max_chars(limit))


# --- diffusion pseudotime ----------------------------------------------------------------------

def _run_record(p: dict[str, Any]) -> dict[str, Any]:
    return pick(p, "column", "root_named_column", "root_column", "root_value", "root_cell", "graph",
                "n_unreachable", "n_detached", "warnings", "stale")


async def _medians(dataset_id: str, column: str, groupby: str, n: int) -> dict[str, Any]:
    """Each group's pseudotime (median and quartiles), earliest first. Cells with no finite
    value (unreachable or left out) count as missing."""
    values = pd.to_numeric(await _obs_values(dataset_id, column), errors="coerce")
    values = values.where(values.abs() != float("inf"))
    recs = _by_group(values, [await _obs_values(dataset_id, groupby)], [groupby])
    recs.sort(key=lambda r: (r.get("median") is None, r.get("median") or 0.0))
    return table(recs, [groupby, "n", "n_missing", "median", "q25", "q75"], max(n, 1))


@tool(WRITE)
async def pseudotime(
    dataset_id: DatasetId,
    root_column: Annotated[str | None, Field(max_length=256, description=(
        "Obs column holding the start group (clusters or cell types; often PAGA's groupby)."))] = None,
    root_value: Annotated[str | None, Field(max_length=256, description=(
        "The start group's value in root_column: the cell state the process begins in (an end of the PAGA "
        "tree, a progenitor, the earliest stage). The root cell is the group's most central cell."))] = None,
    limit: Limit = DEFAULT_LIMIT,
) -> str:
    """Diffusion pseudotime (scanpy's DPT) from a root group, saved as the obs column
    scview_dpt_pseudotime and as one column per root (root_named_column, kept when another
    root runs). The reply has the route's warnings word for word (a root inside the PAGA tree,
    a pseudotime that runs against another one, weakly linked cells left out), n_detached and
    n_unreachable, and each root_column group's median pseudotime, earliest first. Without
    root_column and root_value: the pseudotime columns the dataset holds and how each was made."""
    n = clamp(limit)
    if root_column is None and root_value is None:
        d = await _get(f"{_ds(dataset_id)}/trajectory")
        out = {"dataset_id": dataset_id, "pseudotime_columns": d.get("pseudotime_columns") or [],
               "runs": {c: _run_record(p) for c, p in (d.get("runs") or {}).items()},
               "velocity_counts_attached": bool(d.get("velocity_layers"))}
        if d.get("computed"):
            out["latest"] = _run_record(d["computed"])
        if not out["pseudotime_columns"]:
            out["note"] = "No pseudotime yet: pseudotime(root_column, root_value) computes one."
        return dumps(out, _max_chars(limit))
    if not root_column or root_value is None:
        raise ToolError("Give both root_column and root_value (or neither, to list the pseudotime columns).")
    r = await _post(f"{_ds(dataset_id)}/trajectory/dpt", {"root_column": root_column, "root_value": root_value})
    out = {"dataset_id": dataset_id, "warnings": r.get("warnings") or [], **_run_record(r),
           **pick(r, "root_rule", "diffmap_n_comps", "n_dcs")}
    column = r.get("root_named_column") or r.get("column")
    out["by_group"] = await _medians(dataset_id, column, root_column, n)
    out["next"] = (f"pseudotime_genes(column='{column}'); plot(kind='violin', value_column='{column}', "
                   f"groupby='{root_column}'); plot(color_by='{column}')")
    try:  # a wrong root at a tree end passes the checks above when it is the first pseudotime run
        attached = bool((await _get(f"{_ds(dataset_id)}/trajectory")).get("velocity_layers"))
    except ToolError:
        attached = False
    if attached:
        out["next"] += (f"; velocity_direction(order_column='{column}') checks this root against the RNA "
                        "velocity (many backward transitions = the root is probably the wrong end)")
    return dumps(out, _max_chars(limit))


# --- genes along pseudotime --------------------------------------------------------------------

_TREND_COLUMNS = ["gene", "f_stat", "pval", "pval_adj", "r2", "amplitude", "peak_pseudotime", "pattern", "n_detected"]


@tool(READ)
async def pseudotime_genes(
    dataset_id: DatasetId,
    column: Annotated[str, Field(max_length=256, description=(
        "A pseudotime column (pseudotime() lists them; e.g. scview_dpt_pseudotime, or a root's own column)."))],
    genes: Annotated[list[str] | None, Field(max_length=10, description=(
        "Up to 10 genes: each one's mean expression along the pseudotime, in bins of equal cell counts, "
        "instead of the ranking."))] = None,
    n_top: Annotated[int | None, Field(ge=1, le=200, description="Ranking: genes returned (default 50).")] = None,
    n_bins: Annotated[int | None, Field(ge=5, le=100, description=(
        "Bins of equal cell counts along the pseudotime (ranking default 40; genes default 20, at least 10)."))] = None,
    groups_column: Annotated[str | None, Field(max_length=256, description=(
        "Ranking on one branch only: the obs column of its groups (e.g. PAGA's groupby)."))] = None,
    groups: Annotated[list[str] | None, Field(max_length=200, description=(
        "Ranking on one branch only: the groups of groups_column on it, e.g. the PAGA tree path from the "
        "root to one end."))] = None,
    limit: Limit = DEFAULT_LIMIT,
) -> str:
    """The genes whose log-normalised expression changes most along a pseudotime (spline F-test,
    Benjamini-Hochberg), each with its pattern (rising, falling, transient), peak and amplitude;
    the first 5 with their binned trend. On one branch with groups_column + groups. With genes:
    those genes' trends instead. Bins hold equal numbers of cells, each at its median pseudotime.
    Changes nothing."""
    n = clamp(limit)
    if genes:
        if groups_column or groups:
            raise ToolError("groups_column and groups restrict the ranking; leave them out with genes.")
        bins = max(10, n_bins or TREND_POINTS)
        d = await _get(f"{_ds(dataset_id)}/trajectory/{column}/genes", genes=",".join(genes), n_bins=bins)
        found = {g: v for g, v in (d.get("genes") or {}).items() if v.get("found")}
        missing = [g for g, v in (d.get("genes") or {}).items() if not v.get("found")]
        first = next(iter(found.values()), {})
        out = {"dataset_id": dataset_id, "column": column, "binning": d.get("binning"), "n_bins": d.get("n_bins"),
               "bins": {"pseudotime": first.get("binned_pseudotime"), "n_cells": first.get("binned_counts")},
               "mean_expression": {g: v.get("binned_expression") for g, v in found.items()}}
        if missing:
            out["not_found"] = missing
        return dumps(out, _max_chars(limit))
    if (groups_column is None) != (not groups):
        raise ToolError("Give both groups_column and groups for one branch, or neither.")
    body = _body(n_top=n_top or max(n, 50), n_bins=n_bins, groups_column=groups_column, groups=groups)
    r = await _post(f"{_ds(dataset_id)}/trajectory/{column}/trend-genes", body)
    ranked = r.get("genes") or []
    out = {"dataset_id": dataset_id, **pick(r, "column", "n_cells", "n_genes_tested", "min_detected", "expression",
                                            "method", "restrict"),
           "genes": table(ranked, _TREND_COLUMNS, n),
           "bins": pick(r.get("bins") or {}, "binning", "pseudotime", "n_cells"),
           "trends": {g["gene"]: g.get("mean") for g in ranked[:min(TRENDS_SHOWN, n)]}}
    out["note"] = ("trends: each bin's mean expression (log-normalised) at bins.pseudotime. "
                   "pseudotime_genes(column, genes=[...]) gives any gene's trend.")
    if any(g.get("pval") == 0 for g in ranked[:n]):
        out["p_note"] = P0_NOTE
    return dumps(out, _max_chars(limit))


# --- RNA velocity ------------------------------------------------------------------------------

_TRANSITION_COLUMNS = ["from", "to", "status", "n_cells", "mean_cos", "low", "high", "share_forward"]
_CLUSTER_COLUMNS = ["group", "n_cells", "coherence", "length", "unspliced_fraction", "n_velocity_genes", "top_genes"]


def _gene_flags(flags: dict[str, Any] | None, n: int) -> dict[str, Any] | None:
    if not flags:
        return None
    if not flags.get("checked"):
        return pick(flags, "checked", "reason", "rule")
    recs = [{"gene": g.get("gene"), "rank": g.get("rank"), "r2": g.get("r2"),
             "steps": "; ".join(f"{s.get('from')} -> {s.get('to')} (spliced {s.get('spliced_from')} -> "
                                f"{s.get('spliced_to')}, u/s {s.get('ratio_from')} -> {s.get('ratio_to')})"
                                for s in g.get("steps") or [])}
            for g in flags.get("genes") or []]
    return {**pick(flags, "checked", "rule", "n_velocity_genes", "n_flagged", "top_n", "n_in_top", "steps",
                   "n_cells", "sampled"),
            "genes": table(recs, ["gene", "rank", "r2", "steps"], n)}


def _direction(d: dict[str, Any] | None, n: int) -> dict[str, Any] | None:
    if not d:
        return None
    if not d.get("checked"):
        return pick(d, "checked", "reason", "embedding")
    out = {**pick(d, "checked", "label", "kind", "spec", "group_column", "embedding", "n_forward", "n_backward",
                  "n_unclear", "derived"),
           "transitions": table(d.get("transitions") or [], _TRANSITION_COLUMNS, n),
           "cells": pick(d.get("cells") or {}, "n_cells", "n_run_cells", "mean_cos", "share_forward")}
    if d.get("skipped"):
        out["skipped"] = d["skipped"]
    flags = _gene_flags(d.get("gene_flags"), n)
    if flags:
        out["gene_flags"] = flags
    out["note"] = d.get("note")
    return out


def _velocity_reply(run: dict[str, Any] | None, dataset_id: str, n: int, stale: str | None = None) -> dict[str, Any]:
    if not run:
        return {"dataset_id": dataset_id, "run": None}
    params = run.get("params") or {}
    out: dict[str, Any] = {
        "dataset_id": dataset_id, "warnings": run.get("warnings") or [],
        **pick(run, "result_id", "method", "n_cells", "cells_without", "n_genes_fit", "n_velocity_genes",
               "median_confidence", "median_length", "unspliced_fraction", "group_column", "embeddings"),
        "params": pick(params, "n_top_genes", "n_neighbors", "n_pcs", "exclude_sets", "exclude_given"),
    }
    if stale:
        out["stale"] = stale
    if run.get("excluded"):
        out["excluded"] = pick(run["excluded"], "sets", "n_given", "n_matched", "n_unmatched", "unmatched",
                               "n_passing_filter")
    direction = _direction(run.get("direction"), n)
    if direction:
        out["direction"] = direction
    clusters = [{**c, "top_genes": ", ".join(map(str, c.get("top_genes") or []))} for c in run.get("clusters") or []]
    out["groups"] = table(clusters, _CLUSTER_COLUMNS, n)
    out["note"] = run.get("note")
    return out


def _direction_spec(order, transitions, root, order_column, column_order, group_by) -> dict[str, Any] | None:
    spec = _body(order=order, transitions=[list(t) for t in transitions] if transitions else None, root=root,
                 column=order_column, column_order=column_order)
    if not spec:
        return None
    if group_by:
        spec["group_by"] = group_by
    return spec


Order = Annotated[list[str] | None, Field(max_length=200, description=(
    "Direction check: groups of group_by, earliest first (each consecutive pair is an expected transition), "
    "e.g. ['Blood progenitors 1', 'Blood progenitors 2', 'Erythroid1', 'Erythroid2', 'Erythroid3']."))]
Transitions = Annotated[list[list[str]] | None, Field(max_length=200, description=(
    "Direction check: expected transitions as [from, to] pairs of group_by's groups (branches allowed)."))]
Root = Annotated[str | None, Field(max_length=256, description=(
    "Direction check: the earliest group; transitions follow the PAGA tree of group_by away from it "
    "(run paga on group_by first)."))]
OrderColumn = Annotated[str | None, Field(max_length=256, description=(
    "Direction check: a column that orders the cells (a pseudotime or time, numeric; or a stage such as E7.5)."))]
ColumnOrder = Annotated[list[str] | None, Field(max_length=500, description=(
    "Direction check: the values of a text order_column, earliest first."))]
Embedding = Annotated[str | None, Field(max_length=256, description=(
    "The 2D plot the direction is checked on (default X_umap)."))]


def _check_transitions(transitions: list[list[str]] | None) -> None:
    if transitions and any(len(t) != 2 for t in transitions):
        raise ToolError("Each transition is a [from, to] pair.")


@tool(WRITE)
async def velocity(
    dataset_id: DatasetId,
    method: Annotated[Literal["stochastic", "steady_state"] | None, Field(description=(
        "stochastic (second moments, scVelo's default; default) or steady_state."))] = None,
    n_top_genes: Annotated[int | None, Field(ge=100, le=20_000, description="Genes fitted (default 2000).")] = None,
    n_neighbors: Annotated[int | None, Field(ge=5, le=100, description="Neighbours for the moments (default 30).")] = None,
    group_by: Annotated[str | None, Field(max_length=256, description=(
        "Obs column for the per-group table and the direction check's groups (default: the active clustering)."))] = None,
    exclude_genes: Annotated[list[str] | None, Field(max_length=5000, description=(
        "Genes to leave out of the fit, by name (before the variable genes are chosen)."))] = None,
    exclude_sets: Annotated[list[Literal["murk_barile2021"]] | None, Field(max_length=5, description=(
        "Built-in lists to leave out: murk_barile2021 = the 89 mouse erythroid MURK genes (Barile 2021), whose "
        "transcription steps up along erythroid maturation and turns the arrows backward."))] = None,
    order: Order = None,
    transitions: Transitions = None,
    root: Root = None,
    order_column: OrderColumn = None,
    column_order: ColumnOrder = None,
    embedding: Embedding = None,
    limit: Limit = DEFAULT_LIMIT,
) -> str:
    """RNA velocity (scVelo 0.3.3's stochastic or steady-state model) on the attached spliced and
    unspliced counts, saved with the dataset. Then a direction check: the arrows against an
    expected order (order, transitions, root, or order_column; default: scView's pseudotime when
    there is one), per transition forward / backward / unclear, and the velocity genes whose
    pattern can turn arrows backward (gene_flags). Warnings word for word. A job: waits up to
    20 s, else returns job_id for job_status. get_velocity reads the run back."""
    _check_transitions(transitions)
    body = _body(method=method, n_top_genes=n_top_genes, n_neighbors=n_neighbors, group_by=group_by,
                 exclude_genes=exclude_genes, exclude_sets=exclude_sets, embedding=embedding,
                 direction=_direction_spec(order, transitions, root, order_column, column_order, None))

    def finish(job: jobs.Job) -> dict[str, Any]:
        return _velocity_reply((_result(job) or {}).get("run"), dataset_id, clamp(limit))

    job = jobs.new("velocity", dataset_id, f"{_ds(dataset_id)}/velocity/run-stream", body,
                   progress_line=_line, finish=finish)
    return await _started(job, limit)


@tool(READ)
async def get_velocity(
    dataset_id: DatasetId,
    genes: Annotated[bool, Field(description=(
        "Also the fitted genes: velocity genes first, by r2 (gamma, r2), limit rows."))] = False,
    limit: Limit = DEFAULT_LIMIT,
) -> str:
    """Whether spliced and unspliced counts are attached (and how to attach them), and the last
    velocity run: its parameters, warnings word for word, the direction check (per transition,
    gene_flags), the per-group table, stale when the cells or graph changed since, and the
    memory and time estimate for a run."""
    n = clamp(limit)
    s = await _get(f"{_ds(dataset_id)}/velocity")
    out: dict[str, Any] = {"dataset_id": dataset_id, "attached": bool(s.get("attached"))}
    if not s.get("attached"):
        out["note"] = ATTACH
        return dumps(out, _max_chars(limit))
    src = s.get("source") or {}
    out["source"] = pick(src, "kind", "file", "files", "stored_as", "n_matched", "cells_now", "cells_without_now")
    if s.get("running"):
        out["running"] = True
    est = s.get("estimate") or {}
    if est:
        out["estimate"] = {"peak_gb": round((est.get("peak_bytes") or 0) / 2**30, 2), **pick(est, "seconds")}
    if s.get("run"):
        out["run"] = _velocity_reply(s["run"], dataset_id, n, s.get("stale"))
        out["run"].pop("dataset_id", None)
    else:
        out["run"] = None
        out["next"] = "velocity(dataset_id) runs it"
    if genes and s.get("run"):
        t = await _get(f"{_ds(dataset_id)}/velocity/genes")
        recs = t.get("genes") if isinstance(t, dict) else t
        out["genes"] = table(recs or [], ["gene", "velocity_gene", "gamma", "r2"], n)
    return dumps(out, _max_chars(limit))


@tool(WRITE)
async def velocity_direction(
    dataset_id: DatasetId,
    group_by: Annotated[str | None, Field(max_length=256, description=(
        "Column of the groups the transitions run between (default: the run's group column)."))] = None,
    order: Order = None,
    transitions: Transitions = None,
    root: Root = None,
    order_column: OrderColumn = None,
    column_order: ColumnOrder = None,
    embedding: Embedding = None,
    limit: Limit = DEFAULT_LIMIT,
) -> str:
    """Check the saved velocity run's arrows against an expected order (or on another 2D plot)
    without running it again: per transition the mean cosine of the arrows at the group boundary
    with its 95% interval, forward / backward / unclear, and gene_flags. The check is kept with
    the run (its warnings follow it). Seconds."""
    _check_transitions(transitions)
    spec = _direction_spec(order, transitions, root, order_column, column_order, group_by) or {}
    if group_by and "group_by" not in spec:
        spec["group_by"] = group_by
    r = await _post(f"{_ds(dataset_id)}/velocity/direction", _body(**spec, embedding=embedding))
    n = clamp(limit)
    out = {"dataset_id": dataset_id, "warnings": r.get("warnings") or [],
           "direction": _direction(r.get("direction"), n)}
    return dumps(out, _max_chars(limit))

