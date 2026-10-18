"""The last few DE results, kept so a follow-up request can name one by id.

A DE result lives in the browser, but sending it back for GSEA does not fit:
every tested gene with its fold change, p and statistic is ~1.3 MB of JSON for
Kang's 14k genes, past the 512 KB request cap (``MAX_JSON_BODY_KB``). So each DE
response carries an id, and GSEA asks for the result by it. Process memory,
like the dataset cache (the production backend is one worker); a result that
has dropped out (after KEEP newer ones, or a restart) is run again.
"""

from __future__ import annotations

import threading
import uuid
from collections import OrderedDict
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

import numpy as np

KEEP = 8  # ~1 MB each for 14k genes
# The cell-level methods, whose statistic GSEA ranks by, and its metric name.
SCORE_METRICS = {"wilcoxon": "wilcoxon_score", "t-test": "t_score", "logreg": "logreg_coef"}


@dataclass(frozen=True)
class StoredResult:
    dataset_id: str
    genes: list[str]
    logfoldchanges: np.ndarray
    pvals: np.ndarray
    stats: np.ndarray  # the test statistic; NaN where there is none
    # wilcoxon | t-test | logreg (cell-level, core/de_methods.py) | pseudobulk
    method: str = "wilcoxon"
    # The DE's detection prefilter (core/de_prefilter.py): GSEA ranks and
    # activity scores only the genes it tested, so their responses say so.
    min_pct: float | None = None
    n_skipped: int = 0

    @property
    def score_metric(self) -> str:
        """What GSEA calls a ranking by this result's statistic (core/gsea.py
        RANK_LABELS)."""
        return SCORE_METRICS.get(self.method, "wilcoxon_score")

    def rows(self) -> list[dict[str, Any]]:
        """The genes as GSEA takes them (core/gsea.py ranking_from_de). GSEA ranks
        a cell-level result by its statistic (Wilcoxon z, Welch t, logistic
        regression coefficient) and a pseudobulk one by signed -log10 p, so a
        pseudobulk result's Wald statistic is not passed as a score. A logistic
        regression has no p-value: its pval is NaN."""
        use_stat = self.method in SCORE_METRICS
        out = []
        for g, lfc, p, s in zip(self.genes, self.logfoldchanges, self.pvals, self.stats):
            out.append({
                "gene": g,
                "logfoldchange": float(lfc),
                "pval": float(p),
                "score": float(s) if use_stat and np.isfinite(s) else None,
            })
        return out

    def profile(self) -> tuple[list[str], list[float]]:
        """Genes with a finite statistic, and the statistic: what TF and pathway
        activity is scored on (core/activity.py score_profile)."""
        keep = np.isfinite(self.stats)
        return [g for g, k in zip(self.genes, keep) if k], self.stats[keep].tolist()


_results: OrderedDict[str, StoredResult] = OrderedDict()
_lock = threading.Lock()


def remember(
    dataset_id: str,
    genes: Iterable[Any],
    method: str = "wilcoxon",
    min_pct: float | None = None,
    n_skipped: int = 0,
) -> str:
    """Keep a DE result's genes (objects with gene, logfoldchange, pval and an
    optional stat) and the prefilter that chose them; returns the id to ask for
    them by."""
    rows = list(genes)
    stored = StoredResult(
        dataset_id=dataset_id,
        genes=[str(g.gene) for g in rows],
        logfoldchanges=np.array([g.logfoldchange for g in rows], dtype=np.float64),
        pvals=np.array([np.nan if g.pval is None else g.pval for g in rows], dtype=np.float64),
        stats=np.array(
            [np.nan if getattr(g, "stat", None) is None else g.stat for g in rows], dtype=np.float64
        ),
        method=method,
        min_pct=min_pct,
        n_skipped=n_skipped,
    )
    result_id = uuid.uuid4().hex
    with _lock:
        _results[result_id] = stored
        while len(_results) > KEEP:
            _results.popitem(last=False)
    return result_id


def recall(dataset_id: str, result_id: str) -> StoredResult | None:
    """The stored result, or None when it is gone or belongs to another dataset."""
    with _lock:
        stored = _results.get(result_id)
        if stored is None or stored.dataset_id != dataset_id:
            return None
        _results.move_to_end(result_id)
        return stored


def clear() -> None:
    with _lock:
        _results.clear()
