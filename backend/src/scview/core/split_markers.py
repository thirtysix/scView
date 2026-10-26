"""Marker genes with calibrated p-values: count splitting.

A marker p-value tests genes on the same counts that made the clusters ("double
dipping"): clustering puts cells with different expression in different
clusters, so every cluster has "significant" markers, even clusters cut out of
one population (Neufeld et al. 2024, "Inference after latent variable
estimation for single-cell RNA sequencing data", Biostatistics, PMID 36511385;
Song et al., ClusterDE, PMID 41060830). This check keeps the test honest, with
the merge check's machinery (``core/merge_check.py``):

1. The raw counts are split into a training and a test half by negative
   binomial count splitting (epsilon 0.5, each gene at its own overdispersion:
   ``merge_check.estimate_overdispersion`` + ``split_counts``).
2. The recorded clustering recipe (normalisation, variable genes, scaling, PCA,
   Harmony, the neighbour graph and Leiden in its child process) re-runs on the
   training half (``hvg_sensitivity.run_setting_graph``), at the resolution that
   gives as many clusters as the column checked (``merge_check.resolution_for_count``).
3. Each training cluster is tested against the rest on the TEST half's
   log-normalised counts (Wilcoxon, BH; ``core/rank_genes.py``, scanpy's test),
   and each checked group reads the training cluster that matches it best
   (Jaccard, ``hvg_sensitivity.compare``).
4. The ordinary markers are the same test of the checked groups on all the
   counts (normalised the same way). A marker is a gene up in the group, in at
   least ``MIN_PTS`` of its cells, with log2 fold change >= ``MIN_LOGFC`` and
   adjusted p < ``MAX_PADJ`` (the thresholds of ``core/marker_selection.py``,
   ribosomal and mitochondrial genes left out). An ordinary marker is "kept"
   when the split test also has it up at adjusted p < ``MAX_PADJ``; the rest are
   "lost": markers only by double dipping, or too weak to show at half depth.

On a population with no structure, forced into clusters, the split p-values are
uniform and the ordinary ones are not (``tests/test_split_markers.py``, a KS
test on simulated negative binomial counts). Count splitting removes the reuse of
noise, not differences the cells really have: Kang's CD14 monocytes (CTRL)
forced into two clusters split by depth (median 3,006 vs 1,992 counts), which
both halves of a cell share and normalisation does not fully remove, so 41% of
genes still test at p < 0.05 on the held-out half (35% ordinary); the same with
counts simulated from one mean profile at the cells' own depths 28% (61%); real
cells of one depth band 10.5% (19%).

Caveats, also shown in the UI: the test half holds half of the molecules, so a
weak marker can be lost for want of power, not only for being an artefact; a
real difference such as sequencing depth keeps its "markers"; the
tested clusters are the training half's, standing for the checked groups only
as far as they match (the Jaccard per group, the ARI overall); one split with
one seed. Nothing is saved to the dataset; a group's split result can be kept
with the DE results (``core/de_results.py``) for GSEA and the overlap view.
"""

from __future__ import annotations

import hashlib
import logging
import threading
import time
import weakref
from collections import OrderedDict
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from scview.core import hvg_sensitivity as hs
from scview.core import merge_check as mc
from scview.core import results_store
from scview.core.gene_filters import is_ribo_mito
from scview.core.marker_selection import MAX_PADJ, MIN_LOGFC, MIN_PTS

logger = logging.getLogger(__name__)

MAX_GROUPS = mc.MAX_GROUPS
MIN_GROUP_CELLS = 3  # a training cluster smaller than this is not tested
GENES_PER_GROUP = 300  # gene rows returned per group (ordinary score order): the Markers table shows 100
LIST_GENES = 10  # kept / lost names listed per group
KEEPS_ENOUGH = 10  # a group "keeps its markers" with at least this many kept
# A group whose best training cluster overlaps it less than this (Jaccard) has no
# counterpart on the training half and is not tested. Kang, resolution 0.5: a
# 24-cell group's best match was pDC's cluster at 0.008, the others 0.47-0.94;
# over-split pieces of one type (tests) match their training pieces at 0.11-0.28.
MIN_JACCARD = 0.1

# Time (2 cores, server warm, measured 2026-10-25): Kang copy (13,836 cells,
# resolution 0.5, 13 groups) 38-46 s, of which the fit and split 4 s, the
# ordinary test 2.5 s, the training run 13 s, matching the cluster count 20 s,
# the held-out test 0.7 s; paul15 (2,730 cells, dense) 7 s. Beyond one
# variable-gene setting's estimate (hs.estimate_seconds):
EXTRA_SECONDS_PER_1K_CELLS = 2.0


class SplitMarkersError(ValueError):
    """The check cannot run on this dataset or column: a 422 with the reason."""


def check(adata) -> None:
    try:
        hs.check(adata)
    except hs.SensitivityError as exc:
        raise SplitMarkersError(str(exc)) from None
    if hs._counts(adata)[0] is None:
        raise SplitMarkersError(
            "Count splitting splits the raw counts, which this dataset does not keep "
            "(layers['counts'], .raw or X)."
        )


def plan(adata, column: str | None = None) -> mc.Plan:
    """The column to check (default the active clustering), as the merge check
    plans it, matching its number of clusters on the training half."""
    check(adata)
    try:
        return mc.plan(adata, column or None, match_count=True)
    except mc.MergeCheckError as exc:
        raise SplitMarkersError(str(exc)) from None


def estimate_seconds(n_cells: int, rec: hs.Recorded) -> float:
    return round(hs.estimate_seconds(n_cells, 1, rec) + EXTRA_SECONDS_PER_1K_CELLS * n_cells / 1000, 1)


# ---------------------------------------------------------------------------
# The test
# ---------------------------------------------------------------------------


@dataclass
class GroupTests:
    """Each tested group vs the rest, genes in the matrix's order (rows: groups)."""

    groups: list[int]  # codes tested
    scores: np.ndarray
    lfc: np.ndarray
    pvals: np.ndarray
    padj: np.ndarray
    pct_in: np.ndarray
    pct_out: np.ndarray

    def row(self, code: int) -> int | None:
        try:
            return self.groups.index(code)
        except ValueError:
            return None


def groups_vs_rest(log, codes: np.ndarray, var_names: pd.Index) -> GroupTests:
    """Wilcoxon of every group of ``codes`` (cells with -1 left out) with at
    least MIN_GROUP_CELLS cells against all other labelled cells, on ``log``
    (log-normalised), every gene, BH within the group; scanpy's test
    (``core/rank_genes.py``)."""
    import anndata as ad

    from scview.core import rank_genes

    keep = np.flatnonzero(codes >= 0)
    sub = codes[keep]
    sizes = np.bincount(sub)
    tested = [int(c) for c in np.flatnonzero(sizes >= MIN_GROUP_CELLS)]
    g = len(var_names)
    empty = np.full((len(tested), g), np.nan)
    out = GroupTests(tested, *(empty.copy() for _ in range(6)))
    if len(tested) < 1 or np.unique(sub).size < 2:
        out.groups = []
        return out
    X = log[keep] if keep.size < codes.size else log
    tmp = ad.AnnData(X=X, obs=pd.DataFrame(
        {"g": pd.Categorical([str(c) for c in sub], categories=[str(c) for c in range(sizes.size)])},
        index=[str(i) for i in range(keep.size)]))
    tmp.var_names = pd.Index(var_names).astype(str)
    tmp.uns["log1p"] = {"base": None}
    rank_genes.rank_genes_groups(tmp, "g", groups=[str(c) for c in tested], reference="rest",
                                 method="wilcoxon", pts=True, n_genes=g, use_raw=False)
    r = tmp.uns["rank_genes_groups"]
    index = pd.Index(tmp.var_names)
    for i, c in enumerate(tested):
        name = str(c)
        pos = index.get_indexer(np.asarray(r["names"][name]).astype(str))
        out.scores[i, pos] = np.asarray(r["scores"][name], dtype=np.float64)
        out.lfc[i, pos] = np.asarray(r["logfoldchanges"][name], dtype=np.float64)
        out.pvals[i, pos] = np.asarray(r["pvals"][name], dtype=np.float64)
        out.padj[i, pos] = np.asarray(r["pvals_adj"][name], dtype=np.float64)
        out.pct_in[i] = r["pts"][name].reindex(index).to_numpy(dtype=np.float64)
        out.pct_out[i] = r["pts_rest"][name].reindex(index).to_numpy(dtype=np.float64)
    return out


def marker_mask(t: GroupTests, row: int, eligible: np.ndarray) -> np.ndarray:
    """The group's markers: up, in MIN_PTS of its cells, MIN_LOGFC, adjusted p < MAX_PADJ."""
    with np.errstate(invalid="ignore"):
        return (eligible & (t.lfc[row] >= MIN_LOGFC) & (t.pct_in[row] >= MIN_PTS)
                & (t.padj[row] < MAX_PADJ))


def _num(x: float, digits: int | None = None) -> float | None:
    if x is None or not np.isfinite(x):
        return None
    return round(float(x), digits) if digits is not None else float(x)


@dataclass
class _Stage:
    """A run, kept so a group's split result can be stored for GSEA afterwards."""

    plan: mc.Plan
    ordinary: GroupTests
    split: GroupTests
    match: list[int | None]  # per checked group: its training cluster's code
    train_names: list[str]
    var_names: list[str]
    result: dict[str, Any] = field(default_factory=dict)


def summarise(p: mc.Plan, ordinary: GroupTests, split: GroupTests, comparison: dict[str, Any],
              train_names: list[str], var_names: pd.Index) -> tuple[dict[str, Any], list[int | None]]:
    """Per checked group: its training match and its ordinary markers kept / lost."""
    names = np.asarray(var_names).astype(str)
    eligible = ~np.array([is_ribo_mito(n) for n in names], dtype=bool)
    train_code = {n: i for i, n in enumerate(train_names)}
    by_ref = {c["cluster"]: c for c in comparison["clusters"]}
    matched = [train_code.get(by_ref[n]["match"]) if n in by_ref and by_ref[n]["jaccard"] >= MIN_JACCARD
               else None for n in p.names]
    counts = pd.Series([m for m in matched if m is not None]).value_counts()
    groups = []
    for code, name in enumerate(p.names):
        row_o = ordinary.row(code)
        m = matched[code]
        row_s = split.row(m) if m is not None else None
        info = by_ref.get(name, {})
        g: dict[str, Any] = {
            "group": name, "n_cells": int(info.get("n_cells", int((p.codes == code).sum()))),
            "train_cluster": train_names[m] if m is not None else None,
            "jaccard": info.get("jaccard"), "shared": bool(m is not None and counts.get(m, 0) > 1),
            "tested": row_o is not None and row_s is not None, "reason": None,
            "n_ordinary": 0, "n_kept": 0, "n_lost": 0, "n_split": 0, "n_split_only": 0,
            "kept": [], "lost": [], "split_only": [], "genes": [],
        }
        groups.append(g)
        if not g["tested"]:
            g["reason"] = ("too few cells to test" if row_o is None or (m is not None and row_s is None) else
                           f"no training cluster matches it (best Jaccard {info.get('jaccard', 0):.2f})")
            continue
        ord_m = marker_mask(ordinary, row_o, eligible)
        spl_m = marker_mask(split, row_s, eligible)
        with np.errstate(invalid="ignore"):
            up_sig = (split.lfc[row_s] > 0) & (split.padj[row_s] < MAX_PADJ)
        kept, lost = ord_m & up_sig, ord_m & ~up_sig
        o_score = np.nan_to_num(ordinary.scores[row_o], nan=-np.inf)
        s_score = np.nan_to_num(split.scores[row_s], nan=-np.inf)

        def top(mask: np.ndarray, score: np.ndarray, n: int = LIST_GENES) -> list[str]:
            idx = np.flatnonzero(mask)
            return names[idx[np.argsort(-score[idx], kind="stable")][:n]].tolist()

        g.update(n_ordinary=int(ord_m.sum()), n_kept=int(kept.sum()), n_lost=int(lost.sum()),
                 n_split=int(spl_m.sum()), n_split_only=int((spl_m & ~ord_m).sum()),
                 kept=top(kept, s_score), lost=top(lost, o_score), split_only=top(spl_m & ~ord_m, s_score))
        rows = np.argsort(-o_score, kind="stable")[:GENES_PER_GROUP]
        g["genes"] = [{
            "gene": names[j],
            "logfoldchange": _num(ordinary.lfc[row_o, j], 4), "pval": _num(ordinary.pvals[row_o, j]),
            "pval_adj": _num(ordinary.padj[row_o, j]),
            "split_logfoldchange": _num(split.lfc[row_s, j], 4), "split_pval": _num(split.pvals[row_s, j]),
            "split_pval_adj": _num(split.padj[row_s, j]),
            "split_pct_in": _num(split.pct_in[row_s, j], 4), "split_pct_out": _num(split.pct_out[row_s, j], 4),
            "marker": bool(ord_m[j]), "kept": bool(kept[j]),
        } for j in rows]
    tested = [g for g in groups if g["tested"]]
    return {
        "reference": p.reference,
        "reference_exists": p.exists,
        "method": p.rec.method,
        "resolution": p.rec.resolution,
        "n_groups": len(p.names),
        "n_tested_groups": len(tested),
        "n_keep_enough": sum(g["n_kept"] >= KEEPS_ENOUGH for g in tested),
        "n_ordinary": sum(g["n_ordinary"] for g in tested),
        "n_kept": sum(g["n_kept"] for g in tested),
        "train": {"n_clusters": len(train_names), "ari": comparison["ari"], "ami": comparison["ami"]},
        "groups": groups,
        "notes": list(p.notes),
        "epsilon": mc.EPSILON,
        "seed": mc.SEED,
        "thresholds": {"padj": MAX_PADJ, "min_logfc": MIN_LOGFC, "min_pts": MIN_PTS,
                       "keeps_enough": KEEPS_ENOUGH, "genes_per_group": GENES_PER_GROUP,
                       "min_jaccard": MIN_JACCARD},
    }, matched


# ---------------------------------------------------------------------------
# Runs kept (the last few, across datasets)
# ---------------------------------------------------------------------------

_stages: OrderedDict[str, _Stage] = OrderedDict()
_STAGE_CACHE = 2
_stages_lock = threading.Lock()


def _key(adata, p: mc.Plan) -> str:
    return hashlib.sha1(("split_markers:" + mc._stage_key(adata, p)).encode()).hexdigest()


def cached(adata, p: mc.Plan) -> _Stage | None:
    with _stages_lock:
        s = _stages.get(_key(adata, p))
        if s is not None:
            _stages.move_to_end(_key(adata, p))
        return s


def iter_split_markers(adata, p: mc.Plan, stop: Callable[[], bool] | None = None) -> Iterator[dict[str, Any]]:
    """Split, re-cluster the training half, test on the test half, compare with
    the ordinary markers. Yields a header, ``{"step": ...}`` items, then the
    result. ``adata`` is not changed; ``stop`` is checked between steps."""
    rec = p.rec
    stage = cached(adata, p)
    yield {"n_cells": int(adata.n_obs), "reference": p.reference, "reference_exists": p.exists,
           "recorded": rec.to_dict(), "cached": stage is not None,
           "estimate_seconds": 0.0 if stage else estimate_seconds(adata.n_obs, rec)}
    if stage is not None:
        yield stage.result
        return
    t0 = time.perf_counter()
    rng = np.random.default_rng(mc.SEED)
    counts, _ = hs._counts(adata)
    yield {"step": "overdispersion", "label": "Estimating each gene's overdispersion"}
    theta, theta_info = mc.estimate_overdispersion(counts, rng)
    hs._check_stop(stop)
    yield {"step": "ordinary", "label": "Testing the groups on all the counts (the ordinary markers)"}
    ordinary = groups_vs_rest(mc._normalise(counts, rec.target_sum), p.codes, adata.var_names)
    hs._check_stop(stop)
    yield {"step": "split", "label": "Splitting the counts into two halves"}
    train, test = mc.split_counts(counts, theta, rng)
    del counts
    hs._check_stop(stop)
    yield {"step": "train", "label": "Re-clustering the training half"}
    t_train = time.perf_counter()
    inp = hs.inputs_from_counts(adata, rec, train)
    _, labels, graph = hs.run_setting_graph(inp, rec.current, rec, stop)
    del inp, train
    train_resolution = rec.resolution
    n_ref = int(np.unique(p.codes[p.codes >= 0]).size)
    if len(labels.categories) != n_ref:
        yield {"step": "resolution", "label": f"Finding the resolution that gives {n_ref} clusters on the training half"}
        train_resolution, labels = mc.resolution_for_count(graph, rec.method, rec.resolution, labels, n_ref,
                                                           stop, rec.leiden_iterations)
    del graph
    train_seconds = time.perf_counter() - t_train
    hs._check_stop(stop)
    yield {"step": "test", "label": "Testing the training clusters on the held-out half"}
    t_test = time.perf_counter()
    train_codes = np.asarray(labels.codes, dtype=np.int64)
    train_names = [str(c) for c in labels.categories]
    split = groups_vs_rest(mc._normalise(test, rec.target_sum), train_codes, adata.var_names)
    del test
    comparison = hs.compare(p.codes, p.names, labels)
    result, matched = summarise(p, ordinary, split, comparison, train_names, adata.var_names)
    result.update(train_resolution=train_resolution, overdispersion=theta_info,
                  seconds={"train": round(train_seconds, 1), "test": round(time.perf_counter() - t_test, 1),
                           "total": round(time.perf_counter() - t0, 1)})
    stage = _Stage(p, ordinary, split, matched, train_names, list(map(str, adata.var_names)), result)
    with _stages_lock:
        _stages[_key(adata, p)] = stage
        while len(_stages) > _STAGE_CACHE:
            _stages.popitem(last=False)
    yield result


def run(adata, p: mc.Plan) -> dict[str, Any]:
    *_, result = iter_split_markers(adata, p)
    return result


# ---------------------------------------------------------------------------
# A group's split result as a DE result (GSEA by id, the overlap view)
# ---------------------------------------------------------------------------


@dataclass
class _Gene:
    gene: str
    logfoldchange: float
    pval: float
    pval_adj: float
    stat: float


def keep_group(dataset_id: str, stage: _Stage, group: str) -> str:
    """Store ``group``'s split test (its training cluster vs the rest, on the test
    half) with the DE results; returns the result id."""
    from scview.core import de_results

    try:
        code = stage.plan.names.index(group)
    except ValueError:
        raise SplitMarkersError(f"No group '{group}' in '{stage.plan.reference}'.") from None
    m = stage.match[code]
    row = stage.split.row(m) if m is not None else None
    if row is None:
        raise SplitMarkersError(f"'{group}' has no tested training cluster.")
    t = stage.split
    genes = [_Gene(g, float(t.lfc[row, j]), float(t.pvals[row, j]), float(t.padj[row, j]), float(t.scores[row, j]))
             for j, g in enumerate(stage.var_names) if np.isfinite(t.pvals[row, j])]
    return de_results.remember(dataset_id, genes, method="wilcoxon", comparison={
        "mode": "selection", "label": f"{group} (count split, held-out half)", "n_a": int(
            stage.result["groups"][code]["n_cells"]), "column": stage.plan.reference, "group": group,
        "count_split": True})


# ---------------------------------------------------------------------------
# Co-pilot grounding: the last check run on each loaded dataset
# ---------------------------------------------------------------------------

_last: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()
_last_lock = threading.Lock()


def remember(adaptor, result: dict[str, Any]) -> None:
    try:
        with _last_lock:
            _last[adaptor] = result
    except TypeError:  # not weak-referenceable (a test double)
        return
    results_store.keep(adaptor, "count_split", result, key=str(result.get("reference", "")),
                       label=f"Count-split marker check of '{result.get('reference')}'")


def grounding_lines(adaptor, obs: pd.DataFrame) -> tuple[list[str], str | None]:
    """For the co-pilot: how many groups keep their markers on held-out counts."""
    r = results_store.remembered(_last, _last_lock, adaptor, "count_split")
    if r is None or (r.get("reference_exists") and r["reference"] not in obs.columns):
        return [], None
    tested = [g for g in r["groups"] if g["tested"]]
    t = r["thresholds"]
    detail = (f"count-split check of '{r['reference']}': {r['n_keep_enough']}/{len(tested)} groups keep >= "
              f"{t['keeps_enough']} markers on held-out counts; {r['n_kept']:,} of {r['n_ordinary']:,} ordinary "
              f"markers kept (training half ARI {r['train']['ari']:.2f})")
    lines = ["\n## Count-split marker check (last run; markers re-tested on held-out counts)",
             f"- {detail} [result:count_split]."]
    for g in sorted(tested, key=lambda g: (g["n_kept"], g["group"]))[:8]:
        lost = f"; lost e.g. {', '.join(g['lost'][:4])}" if g["lost"] else ""
        lines.append(f"- {g['group']}: keeps {g['n_kept']} of {g['n_ordinary']} markers{lost}")
    lines.append("- Caveat: the held-out half has half the molecules, so weak markers can be lost for lack of "
                 "power; a group keeping few markers may not be a distinct population.")
    return lines, detail
