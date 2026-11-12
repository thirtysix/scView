"""Marker lists that do not beat 9 of 10 random gene lists of their size (no LLM).

The annotation benchmark's T4 controls (2026-11-08) gave the annotator 25 random genes as a
"cluster": the default model named a cell type every time (0 of 54). This check finds such lists. From
v0.71.0 the annotator labelled them "Unidentified" itself; since v0.75.0 (user decision 2026-11-12) it
sends them to the AI with ``NOTE`` beside their markers (``pipeline._annotation_decided``), because
disease and injury states missing from the databases (kidney PT_VCAM1) look random to this check. A
note in the prompt moved other clusters' calls in the benchmark (arms rm, ua): the price of not hiding
those states.

Each list (the annotator's own markers, ``core/marker_selection.py``) is scored by its best
hypergeometric match in the marker databases (``core/marker_db.py``, the annotate view), counting
only genes some database type lists. lncRNAs, pseudogenes and bare Ensembl ids are in no database;
counted, they made real lists look random (Tabula Sapiens liver hepatocytes: 4 of 25 named). The
null is the best match of ``N_NULL`` random lists with the same number of named genes, drawn from the
named genes detected in >= 10% of cells. A list that does not beat ``NULL_QUANTILE`` of them is
flagged; a list with fewer than ``MIN_NAMED`` named genes is not judged. Every list with a named
gene also gets its best database type (``best``), judged or not.

Sweep (research/batch42/random-markers/, 2026-11-10): T4 shuffled lists 15/15 flagged; 0 of ~270
clean groups (T4 backgrounds, author and Leiden clusters of 6 datasets, doublets, low quality).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import scipy.sparse as sp
from scipy.stats import hypergeom

logger = logging.getLogger(__name__)

N_NULL = 500
NULL_QUANTILE = 0.90
MIN_NAMED = 10
POOL_MIN_DETECTED = 0.10
SEED = 0
NOTE = "markers do not beat 9 of 10 random gene lists of the same size against the marker databases"


@dataclass
class RandomCheck:
    flagged: dict[str, dict] = field(default_factory=dict)  # group -> evidence
    checked: int = 0
    too_few_named: list[str] = field(default_factory=list)
    skipped: str | None = None  # why nothing was checked
    best: dict[str, str] = field(default_factory=dict)  # group -> best database type (any named gene)
    # group -> {named, best_type, best_log10p, null_cut (judged lists only)}: every list with a named
    # gene, flagged or not (core/cluster_evidence.py says it in words)
    scores: dict[str, dict] = field(default_factory=dict)

    def record(self) -> dict:
        return {"flagged": sorted(self.flagged), "checked": self.checked,
                "too_few_named": self.too_few_named, "skipped": self.skipped}


def _detected_share(adata, names) -> np.ndarray | None:
    """Per-gene share of cells with a count, from qc_metrics' ``n_cells_by_counts`` (no matrix read)."""
    for var in ((adata.raw.var if adata.raw is not None else None), adata.var):
        if var is not None and "n_cells_by_counts" in var and len(var) == len(names) and adata.n_obs:
            return np.asarray(var["n_cells_by_counts"], dtype=float) / adata.n_obs
    return None


class _Index:
    """The database's type sets over the named measured genes, as a sparse type x gene matrix."""

    def __init__(self, sdb, measured):
        upper = {}
        for g in measured:
            upper.setdefault(str(g).upper(), str(g))
        sets = {t: {g.upper() for g in sdb.types[t]["genes"]} & upper.keys() for t in sdb.names}
        from scview.core.marker_db import MIN_DB_GENES

        self.types = [t for t in sdb.names if len(sets[t]) >= MIN_DB_GENES]
        named = sorted(set().union(*(sets[t] for t in self.types))) if self.types else []
        self.col = {g: i for i, g in enumerate(named)}
        rows, cols = [], []
        for i, t in enumerate(self.types):
            for g in sets[t]:
                rows.append(i)
                cols.append(self.col[g])
        self.M = sp.csr_matrix((np.ones(len(rows), dtype=np.int32), (rows, cols)),
                               shape=(len(self.types), len(named)))
        self.size = np.asarray(self.M.sum(axis=1)).ravel()
        self.U = len(named)
        self.measured = upper

    def query(self, genes) -> list[int]:
        return sorted({self.col[u] for u in (str(g).upper() for g in genes) if u in self.col})

    def best(self, Q: sp.csr_matrix) -> tuple[np.ndarray, np.ndarray]:
        """(-log10 p of the best type, its row) for each query row of ``Q`` (lists x genes, 0/1)."""
        K = (Q @ self.M.T).toarray()
        n_q = np.asarray(Q.sum(axis=1)).ravel()[:, None]
        p = hypergeom.sf(K - 1, self.U, self.size[None, :], n_q)
        s = -np.log10(np.clip(p, 1e-300, 1.0))
        s[K == 0] = 0.0
        i = np.argmax(s, axis=1)
        return s[np.arange(len(i)), i], i


def _rows(idx_lists: list[list[int]], n_cols: int) -> sp.csr_matrix:
    rows = [r for r, cols in enumerate(idx_lists) for _ in cols]
    cols = [c for cols in idx_lists for c in cols]
    return sp.csr_matrix((np.ones(len(cols), dtype=np.int32), (rows, cols)), shape=(len(idx_lists), n_cols))


def check(adata, marker_map: dict[str, list[str]], db=None) -> RandomCheck:
    """Which lists in ``marker_map`` look random. Best effort: never fetches the databases
    (``skipped`` says why nothing was checked)."""
    from scview.core import marker_db as mdb

    if db is None:
        try:
            db = mdb.load(mdb.marker_db_dir())
        except Exception as exc:  # unreadable tables: the annotation goes on without the check
            logger.info("Random-marker check skipped: %s", exc)
            return RandomCheck(skipped="marker databases unreadable")
    if db is None:
        return RandomCheck(skipped="marker databases not downloaded")
    species, _ = mdb._species(adata, None)
    names = mdb.measured_genes(adata)
    idx = _Index(db.species(species, db.view("annotate", species)), names)
    if not idx.types:
        return RandomCheck(skipped="no database types for these genes")

    det = _detected_share(adata, names)
    if det is not None:
        pool_up = {str(g).upper() for g, d in zip(names, det) if d >= POOL_MIN_DETECTED}
        pool = [c for g, c in idx.col.items() if g in pool_up]
    else:
        pool = list(idx.col.values())

    out = RandomCheck()
    queries = {g: idx.query(genes) for g, genes in marker_map.items()}
    judged = {g: q for g, q in queries.items() if len(q) >= MIN_NAMED}
    out.too_few_named = sorted(g for g in queries if g not in judged)
    named = [g for g, q in queries.items() if q]
    if named:
        s_all, b_all = idx.best(_rows([queries[g] for g in named], idx.U))
        out.best = {g: idx.types[b] for g, b in zip(named, b_all)}
        out.scores = {g: {"named": len(queries[g]), "best_type": idx.types[b], "best_log10p": round(float(sc), 2)}
                      for g, sc, b in zip(named, s_all, b_all)}
    if not judged:
        return out
    groups = list(judged)
    s, best = idx.best(_rows([judged[g] for g in groups], idx.U))
    rng = np.random.default_rng(SEED)
    cut: dict[int, float] = {}
    for g, score, b in zip(groups, s, best):
        n = len(judged[g])
        if n > len(pool):
            continue
        if n not in cut:
            draws = [list(rng.choice(pool, n, replace=False)) for _ in range(N_NULL)]
            cut[n] = float(np.quantile(idx.best(_rows(draws, idx.U))[0], NULL_QUANTILE))
        out.checked += 1
        out.scores[g]["null_cut"] = round(cut[n], 2)
        if score <= cut[n]:
            out.flagged[g] = {"named": n, "best_type": idx.types[b], "best_log10p": round(float(score), 2),
                              "null_cut": round(cut[n], 2)}
    return out
