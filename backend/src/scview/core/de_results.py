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


@dataclass(frozen=True)
class StoredResult:
    dataset_id: str
    genes: list[str]
    logfoldchanges: np.ndarray
    pvals: np.ndarray
    scores: np.ndarray  # NaN where the test gives none (pseudobulk)

    def rows(self) -> list[dict[str, Any]]:
        """The genes as GSEA takes them (core/gsea.py ranking_from_de)."""
        out = []
        for g, lfc, p, s in zip(self.genes, self.logfoldchanges, self.pvals, self.scores):
            out.append({
                "gene": g,
                "logfoldchange": float(lfc),
                "pval": float(p),
                "score": float(s) if np.isfinite(s) else None,
            })
        return out


_results: OrderedDict[str, StoredResult] = OrderedDict()
_lock = threading.Lock()


def remember(dataset_id: str, genes: Iterable[Any]) -> str:
    """Keep a DE result's genes (objects with gene, logfoldchange, pval and an
    optional score); returns the id to ask for them by."""
    rows = list(genes)
    stored = StoredResult(
        dataset_id=dataset_id,
        genes=[str(g.gene) for g in rows],
        logfoldchanges=np.array([g.logfoldchange for g in rows], dtype=np.float64),
        pvals=np.array([g.pval for g in rows], dtype=np.float64),
        scores=np.array(
            [np.nan if getattr(g, "score", None) is None else g.score for g in rows], dtype=np.float64
        ),
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
