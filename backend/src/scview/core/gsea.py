"""Preranked GSEA of a DE result or a marker group, with leading-edge genes.

ORA tests only the genes past a cut-off; GSEA (gseapy ``prerank``) walks every
tested gene in rank order and asks whether a set's genes bunch up at the top
(positive NES: higher in the first group) or the bottom (negative NES). The
leading edge is the part of the set that drives the score.

The ranking is the test's own statistic where there is one:

- cell-level Wilcoxon DE: scanpy's z-score of the rank-sum test;
- pseudobulk DESeq2: ``sign(log2FC) * -log10(p)`` (the DE result carries no
  statistic, and p alone has no direction);
- a marker group: scanpy's score for the group against the rest.

Cost is about gene sets x permutations in gseapy's Rust core, which releases
the GIL, so a worker thread keeps the server answering. Measured on Kang 2018
STIM vs CTRL within CD14 Mono (14,053 genes, 2 threads): Hallmark (50 sets) at
1,000 permutations 1.6 s; the eight default collections (6,835 sets of 15-500
genes) 21 s at 100 permutations and 183 s at 1,000. So the permutation count
shrinks as the collections grow (``permutations_for``). With many sets that
costs little: GSEA's FDR pools the null scores of every set, 683,500 of them at
100 permutations; only the nominal p is floored at 1/permutations.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import threading
import time
from collections import OrderedDict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from scview.core.gene_filters import is_ribo_mito

logger = logging.getLogger(__name__)

THREADS = 2
MIN_SIZE = 15  # GSEA's own defaults for the set size within the ranking
MAX_SIZE = 500
MIN_PERMUTATIONS = 100
MAX_PERMUTATIONS = 1000
# Gene sets x permutations per run: ~20 s on 2 threads at the rate measured above.
PERMUTATION_BUDGET = 700_000
# At MIN_PERMUTATIONS, about 30 s; more sets are refused rather than run for minutes.
MAX_SETS = 10_000
SIGNIFICANT_FDR = 0.05
# -log10 of a p-value of 0 (DESeq2 underflow) would be infinite.
_P_FLOOR = 1e-300

RANK_LABELS = {
    "wilcoxon_score": "Wilcoxon statistic (z-score)",
    "signed_log10_p": "sign(log2FC) × −log10(p)",
    "marker_score": "marker score (Wilcoxon z-score vs the rest)",
}


class GseaError(ValueError):
    """A request GSEA can't run (an empty ranking, too many gene sets...)."""


@dataclass
class Ranking:
    genes: list[str]
    values: np.ndarray  # descending
    metric: str
    n_excluded_ribo_mito: int = 0

    def series(self) -> pd.Series:
        return pd.Series(self.values, index=self.genes)


def _finite(v: Any) -> bool:
    try:
        return v is not None and math.isfinite(float(v))
    except (TypeError, ValueError):
        return False


def build_ranking(
    pairs: Iterable[tuple[str, float]], metric: str, *, exclude_ribo_mito: bool
) -> Ranking:
    """Genes sorted by value, highest first (ties by name, so a run is repeatable).
    Non-finite values and repeated genes are dropped."""
    seen: set[str] = set()
    genes: list[str] = []
    values: list[float] = []
    excluded = 0
    for gene, value in pairs:
        gene = str(gene)
        if gene in seen or not _finite(value):
            continue
        seen.add(gene)
        if exclude_ribo_mito and is_ribo_mito(gene):
            excluded += 1
            continue
        genes.append(gene)
        values.append(float(value))
    if len(genes) < 2:
        raise GseaError("Too few ranked genes for GSEA.")
    v = np.asarray(values, dtype=np.float64)
    order = np.lexsort((np.asarray(genes, dtype=object), -v))
    return Ranking([genes[i] for i in order], v[order], metric, excluded)


def ranking_from_de(
    genes: Iterable[Mapping[str, Any]], *, rank_by: str = "auto", exclude_ribo_mito: bool = True
) -> Ranking:
    """A DE result's genes ranked by its statistic, else by signed -log10 p."""
    rows = list(genes)
    if rank_by == "auto":
        rank_by = "score" if rows and all(_finite(r.get("score")) for r in rows) else "signed_p"
    if rank_by == "score":
        pairs = [(r["gene"], r.get("score")) for r in rows]
        metric = "wilcoxon_score"
    else:
        pairs = []
        for r in rows:
            lfc, p = r.get("logfoldchange"), r.get("pval")
            value = None
            if _finite(lfc) and _finite(p):
                value = float(np.sign(lfc)) * -math.log10(max(float(p), _P_FLOOR))
            pairs.append((r["gene"], value))
        metric = "signed_log10_p"
    return build_ranking(pairs, metric, exclude_ribo_mito=exclude_ribo_mito)


def ranking_from_markers(rgg: Mapping[str, Any], group: str, *, exclude_ribo_mito: bool = True) -> Ranking:
    """Every gene of a marker group ranked by scanpy's score against the rest."""
    names = rgg.get("names")
    fields = getattr(getattr(names, "dtype", None), "names", None) or ()
    if group not in fields:
        raise GseaError(f"No markers for group '{group}'.")
    scores = rgg.get("scores")
    if scores is None or group not in (getattr(scores.dtype, "names", None) or ()):
        raise GseaError("These markers carry no scores to rank by.")
    pairs = zip((str(g) for g in names[group]), (float(s) for s in scores[group]))
    return build_ranking(pairs, "marker_score", exclude_ribo_mito=exclude_ribo_mito)


def permutations_for(n_sets: int, requested: int | None = None) -> int:
    """As many permutations as the budget allows for ``n_sets`` (at most the
    requested number), never fewer than MIN_PERMUTATIONS."""
    cap = MAX_PERMUTATIONS if requested is None else min(requested, MAX_PERMUTATIONS)
    by_budget = PERMUTATION_BUDGET // max(n_sets, 1)
    return int(max(MIN_PERMUTATIONS, min(cap, by_budget)))


def _mostly_upper(genes: list[str]) -> bool:
    return bool(genes) and sum(g.isupper() for g in genes) >= 0.9 * len(genes)


def sets_in_range(
    gene_sets: Mapping[str, list[str]], genes: list[str], min_size: int, max_size: int
) -> dict[str, list[str]]:
    """The sets with min_size..max_size genes in the ranking, matched as gseapy
    matches them (upper-cased when the ranking is not upper-case, as for mouse)."""
    upper = not _mostly_upper(genes)
    universe = {g.upper() for g in genes} if upper else set(genes)
    out = {}
    for term, members in gene_sets.items():
        hits = {m.upper() for m in members} if upper else set(members)
        if min_size <= len(hits & universe) <= max_size:
            out[term] = list(members)
    return out


@dataclass
class GseaRun:
    """One prerank run: every tested set, split by the sign of its NES."""

    up: list[dict] = field(default_factory=list)  # NES > 0, most significant first
    down: list[dict] = field(default_factory=list)
    n_sets: int = 0
    permutations: int = 0
    seconds: float = 0.0


def _num(v: Any, default: float = float("nan")) -> float:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return default
    return f if math.isfinite(f) else default


def run_prerank(
    ranking: Ranking,
    gene_sets: Mapping[str, list[str]],
    *,
    min_size: int = MIN_SIZE,
    max_size: int = MAX_SIZE,
    permutations: int | None = None,
    seed: int = 123,
) -> GseaRun:
    """gseapy prerank of ``ranking`` against the in-range sets of ``gene_sets``."""
    import gseapy as gp

    usable = sets_in_range(gene_sets, ranking.genes, min_size, max_size)
    if not usable:
        raise GseaError(
            f"No gene set has {min_size}-{max_size} of its genes in the ranking; "
            "pick other collections."
        )
    if len(usable) > MAX_SETS:
        raise GseaError(
            f"{len(usable):,} gene sets have {min_size}-{max_size} genes in the ranking; GSEA "
            f"runs at most {MAX_SETS:,} at a time. Pick fewer collections."
        )
    n_perm = permutations_for(len(usable), permutations)
    t0 = time.perf_counter()
    res = gp.prerank(
        rnk=ranking.series(),
        gene_sets=usable,
        min_size=min_size,
        max_size=max_size,
        permutation_num=n_perm,
        ascending=None,  # already sorted, ties broken by name
        threads=THREADS,
        seed=seed,
        outdir=None,
        no_plot=True,
        verbose=False,
    )
    seconds = time.perf_counter() - t0
    df = res.res2d
    run = GseaRun(n_sets=0 if df is None else len(df), permutations=n_perm, seconds=seconds)
    for _, row in (df.iterrows() if df is not None else []):
        nes = _num(row.get("NES"))
        if not math.isfinite(nes) or nes == 0:
            continue
        lead = [g for g in str(row.get("Lead_genes") or "").split(";") if g]
        tag = str(row.get("Tag %") or "")
        size = int(tag.split("/")[1]) if "/" in tag and tag.split("/")[1].isdigit() else 0
        rec = {
            "term": str(row.get("Term", "")),
            "es": _num(row.get("ES"), 0.0),
            "nes": nes,
            "pvalue": _num(row.get("NOM p-val"), 1.0),
            "fdr": min(_num(row.get("FDR q-val"), 1.0), 1.0),
            "set_size": size,
            "leading_edge": lead,
        }
        (run.up if nes > 0 else run.down).append(rec)
    for side in (run.up, run.down):
        side.sort(key=lambda r: (r["fdr"], r["pvalue"], -abs(r["nes"])))
    return run


# ---------------------------------------------------------------------------
# Cache: one run per (ranking, gene sets, parameters), so the same DE result or
# marker group asked for again is answered at once.
# ---------------------------------------------------------------------------

_CACHE_SIZE = 8
_cache: OrderedDict[str, GseaRun] = OrderedDict()
_cache_lock = threading.Lock()


def cache_key(ranking: Ranking, gene_set_ids: Iterable[str], params: Mapping[str, Any]) -> str:
    h = hashlib.sha256()
    h.update("\0".join(ranking.genes).encode())
    h.update(np.ascontiguousarray(ranking.values, dtype=np.float64).tobytes())
    h.update(json.dumps([ranking.metric, sorted(gene_set_ids), dict(params)], sort_keys=True,
                        default=str).encode())
    return h.hexdigest()


def cached(key: str) -> GseaRun | None:
    with _cache_lock:
        run = _cache.get(key)
        if run is not None:
            _cache.move_to_end(key)
        return run


def remember(key: str, run: GseaRun) -> None:
    with _cache_lock:
        _cache[key] = run
        _cache.move_to_end(key)
        while len(_cache) > _CACHE_SIZE:
            _cache.popitem(last=False)


def clear_cache() -> None:
    with _cache_lock:
        _cache.clear()
