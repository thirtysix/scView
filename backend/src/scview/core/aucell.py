"""AUCell: a per-cell gene-set score from the order of genes inside each cell.

Aibar et al. 2017 (SCENIC, Nat Methods, PMID 28991892). Within one cell the genes are
ranked by expression, highest first. A set's score is the area under its recovery
curve over the top ``max_rank`` genes: each set gene at rank r <= max_rank adds
``max_rank - r``, and the sum is divided by the most a set of that size could reach
(its genes at ranks 1, 2, ...). So the score is 0 when no set gene is among the cell's
top genes and 1 when they fill the top ranks. Only the order of genes within the cell
counts, so unlike a mean expression (``sc.tl.score_genes``) the score does not scale
with the cell's depth; how much it still follows depth on a dataset is measured by
``depth_check``.

This is decoupler 2.2's ``dc.mt.aucell`` (the reference; agreement on Kang and paul15
is in the C097 report), read from its source:

- genes never detected in any cell are left out (decoupler's ``extract(empty=True)``),
  and the default ``max_rank`` is 5% of the genes left, rounded up, at least 2;
- ties are broken in one fixed random order of those genes, numpy's
  ``default_rng(0)`` permutation, as decoupler shuffles the columns before an ordinal
  rank. Ties matter most in a cell that detects fewer genes than ``max_rank``: its
  lowest counted ranks are then undetected genes, in that order;
- the maximum is exact: ``sum(max_rank - i for i in 1..min(k, max_rank - 1))`` for a set
  of k detected genes. pySCENIC and R's AUCell divide by ``max_rank * k`` instead (a
  constant factor per set) and break ties at random on each run;
- a cell with no detected gene gets 0 (decoupler drops it).

Ranks need no log transform or normalisation: any transform that keeps a cell's order
(counts, CPM, log CPM) gives the same ranks, so ``rank_matrix`` takes whichever of
those the dataset holds, never z-scores or Pearson residuals. Ranking every cell is the
cost (about 0.6 s for Kang's 14k cells); a set then costs one pass over each cell's top
genes, so the ranking is cached per loaded dataset and ``max_rank``.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy import stats

from scview.core.object_cache import ObjectCache

TOP_FRACTION = 0.05  # decoupler's n_up and pySCENIC's auc_threshold default
_SCORE_ROWS = 4096  # cells per block when a set is scored
_MEASURE_ROWS = 4096  # dense rows per block when detected genes are counted
_KEEP_RANKINGS = 2  # max_rank values kept per dataset


# ---------------------------------------------------------------------------
# What to rank
# ---------------------------------------------------------------------------


def _non_negative(M) -> bool:
    sample = M.data[:200_000] if sp.issparse(M) else np.asarray(M[:1000])
    return sample.size == 0 or float(np.min(sample)) >= 0


def rank_matrix(adata):
    """Expression to rank: (matrix, gene names, where from).

    ``.raw``, else X, when non-negative (log-normalised or counts: within a cell both
    give the same order), else the counts. Never z-scores or Pearson residuals: a
    per-gene centring reorders genes within a cell."""
    from scview.core.normalization import find_counts
    from scview.core.pipeline import looks_like_counts

    def kind(M) -> str:
        return "counts" if looks_like_counts(M) else "log-normalised"

    if adata.raw is not None and _non_negative(adata.raw.X):
        return adata.raw.X, adata.raw.var_names, f"{kind(adata.raw.X)} .raw"
    if _non_negative(adata.X):
        return adata.X, adata.var_names, f"{kind(adata.X)} X"
    try:
        counts, source = find_counts(adata)
    except ValueError:
        raise ValueError(
            "AUCell needs expression to rank (log-normalised values or raw counts), and "
            "this dataset holds only scaled values."
        ) from None
    return counts, adata.var_names, f"counts in {source}"


def detected_genes(M) -> np.ndarray:
    """Genes with a non-zero value in at least one cell (dense: in row blocks)."""
    if sp.issparse(M):
        M = sp.csr_matrix(M)
        return np.bincount(M.indices[M.data != 0], minlength=M.shape[1]) > 0
    M = np.asarray(M)
    seen = np.zeros(M.shape[1], dtype=bool)
    for i in range(0, M.shape[0], _MEASURE_ROWS):
        seen |= np.any(M[i:i + _MEASURE_ROWS] != 0, axis=0)
    return seen


def default_max_rank(n_ranked: int) -> int:
    """decoupler's ``n_up``: 5% of the ranked genes, rounded up, at least 2."""
    return int(np.clip(np.ceil(TOP_FRACTION * n_ranked), 2, n_ranked))


def check_max_rank(max_rank: int | None, n_ranked: int) -> int:
    if n_ranked < 2:
        raise ValueError("AUCell needs at least 2 detected genes to rank.")
    if max_rank is None:
        return default_max_rank(n_ranked)
    if not 2 <= int(max_rank) <= n_ranked:
        raise ValueError(
            f"max_rank must be between 2 and the {n_ranked:,} detected genes, not {max_rank}."
        )
    return int(max_rank)


def tie_order(detected: np.ndarray) -> np.ndarray:
    """The detected genes (indices) in the order that breaks ties: decoupler's
    ``_break_ties`` column shuffle."""
    idx = np.flatnonzero(detected)
    perm = np.random.default_rng(seed=0).choice(np.arange(idx.size), idx.size, replace=False)
    return idx[perm]


# ---------------------------------------------------------------------------
# Rankings
# ---------------------------------------------------------------------------


@dataclass
class Rankings:
    """Every cell's top genes, in rank order."""

    top: np.ndarray  # cells x (max_rank - 1) gene indices: rank 1, 2, ... (max_rank adds 0)
    max_rank: int
    genes: pd.Index  # every gene of the ranked matrix
    detected: np.ndarray  # bool per gene: the genes ranked
    n_detected: np.ndarray  # per cell: genes with a non-zero value
    source: str

    @property
    def n_ranked(self) -> int:
        return int(self.detected.sum())

    @property
    def n_cells(self) -> int:
        return int(self.top.shape[0])

    def share_below_max_rank(self) -> float:
        """Share of cells detecting fewer genes than ``max_rank``: their lowest counted
        ranks are ties among undetected genes."""
        return float(np.mean(self.n_detected < self.max_rank)) if self.n_cells else 0.0


def _row_entries(M, r: int):
    """(values, gene indices) of row r's non-zero entries."""
    if sp.issparse(M):
        a, b = M.indptr[r], M.indptr[r + 1]
        v, g = M.data[a:b], M.indices[a:b]
        keep = v != 0
        return v[keep], g[keep]
    row = np.asarray(M[r]).ravel()
    g = np.flatnonzero(row)
    return row[g], g


def build(M, genes, source: str = "", max_rank: int | None = None,
          detected: np.ndarray | None = None) -> Rankings:
    """Rank every cell of ``M`` (cells x genes, non-negative; CSR or dense): its top
    ``max_rank - 1`` genes by value, highest first, ties in ``tie_order``."""
    if sp.issparse(M):
        M = sp.csr_matrix(M)
    else:
        M = np.asarray(M)
    genes = pd.Index(np.asarray(genes, dtype=str))
    n_cells, n_genes = M.shape
    if len(genes) != n_genes:
        raise ValueError("One gene name per column is needed to rank.")
    if detected is None:
        detected = detected_genes(M)
    n_up = check_max_rank(max_rank, int(detected.sum()))
    keep = n_up - 1
    order = tie_order(detected)
    pos = np.zeros(n_genes, dtype=np.int64)
    pos[order] = np.arange(order.size)
    fill = order[:keep]  # undetected genes of a cell come from these, in this order
    dtype = np.uint16 if n_genes <= np.iinfo(np.uint16).max + 1 else np.uint32
    top = np.zeros((n_cells, keep), dtype=dtype)
    n_det = np.zeros(n_cells, dtype=np.int64)
    expressed = np.zeros(n_genes, dtype=bool)
    for r in range(n_cells):
        v, g = _row_entries(M, r)
        n_det[r] = g.size
        o = np.lexsort((pos[g], -v))  # value, highest first; ties by the shuffled position
        m = min(o.size, keep)
        top[r, :m] = g[o[:m]]
        if m < keep:
            # The first undetected genes in tie order: among the first ``keep`` of that
            # order at most ``m`` are this cell's detected genes, so they are enough.
            expressed[g] = True
            top[r, m:] = fill[~expressed[fill]][: keep - m]
            expressed[g] = False
    return Rankings(top=top, max_rank=n_up, genes=genes, detected=detected,
                    n_detected=n_det, source=source)


def max_auc(k: int, max_rank: int) -> float:
    """The largest AUC a set of k ranked genes can reach (genes at ranks 1..k)."""
    kk = min(int(k), max_rank - 1)
    return float(kk * max_rank - kk * (kk + 1) / 2)


def score(rk: Rankings, gene_idx) -> np.ndarray:
    """Every cell's AUC for the set with these gene indices (float64, 0 to 1)."""
    idx = np.unique(np.asarray(gene_idx, dtype=np.int64))
    idx = idx[rk.detected[idx]]
    out = np.zeros(rk.n_cells, dtype=np.float64)
    if idx.size == 0:
        return out
    in_set = np.zeros(len(rk.genes), dtype=bool)
    in_set[idx] = True
    keep = rk.max_rank - 1
    w = np.arange(keep, 0, -1, dtype=np.float64)  # rank j + 1 adds max_rank - (j + 1)
    for i in range(0, rk.n_cells, _SCORE_ROWS):
        out[i:i + _SCORE_ROWS] = in_set[rk.top[i:i + _SCORE_ROWS]] @ w
    out /= max_auc(idx.size, rk.max_rank)
    out[rk.n_detected == 0] = 0.0
    return out


@dataclass
class SetMatch:
    """A gene list matched to the ranked genes."""

    idx: list[int]  # detected genes of the set, as indices into ``Rankings.genes``
    found: list[str]  # the dataset's names of those genes
    missing: list[str]  # requested names the dataset does not have
    undetected: list[str]  # the dataset has them, but no cell expresses them


def match(rk: Rankings, gene_set: list[str]) -> SetMatch:
    """Each name exactly, else ignoring case (as ORA matches MSigDB's upper-case
    human symbols to a mouse dataset's Cd3e)."""
    names = [str(g) for g in rk.genes]
    exact = {g: i for i, g in enumerate(names)}
    upper: dict[str, int] = {}
    for i, g in enumerate(names):
        upper.setdefault(g.upper(), i)
    out = SetMatch([], [], [], [])
    seen: set[int] = set()
    for g in gene_set:
        i = exact.get(g)
        if i is None:
            i = upper.get(str(g).upper())
        if i is None:
            out.missing.append(g)
        elif i not in seen:
            seen.add(i)
            if rk.detected[i]:
                out.idx.append(i)
                out.found.append(names[i])
            else:
                out.undetected.append(names[i])
    return out


# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------

# loaded dataset object -> {"base": (M, genes, source, detected), max_rank: Rankings}.
# A pipeline run drops the object, so rankings never outlive the values they ranked.
_CACHE = ObjectCache()
_BUILD = threading.Lock()  # one build at a time: a second request waits for the first


def rankings(adata, max_rank: int | None = None) -> Rankings:
    """This loaded dataset's rankings for ``max_rank`` (None: the default), built on
    first use. Call in a worker thread."""
    with _BUILD:
        entry = _CACHE.get(adata)
        if entry is None:
            M, genes, source = rank_matrix(adata)
            entry = {"base": (M, genes, source, detected_genes(M)), "built": []}
            _CACHE.put(adata, entry)
        M, genes, source, detected = entry["base"]
        n_up = check_max_rank(max_rank, int(detected.sum()))
        rk = entry.get(n_up)
        if rk is None:
            rk = build(M, genes, source, n_up, detected=detected)
            entry[n_up] = rk
            entry["built"].append(n_up)
            while len(entry["built"]) > _KEEP_RANKINGS:
                entry.pop(entry["built"].pop(0), None)
        return rk


# ---------------------------------------------------------------------------
# Depth check
# ---------------------------------------------------------------------------

MIN_GROUP_CELLS = 20  # a group smaller than this gives no within-group correlation
STRONG, WEAK = 0.3, 0.1  # |rho| for "follows depth" / "weakly"


def spearman(a, b) -> float | None:
    """Spearman's rho (average ranks for ties); None when either side is constant."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    ok = np.isfinite(a) & np.isfinite(b)
    a, b = a[ok], b[ok]
    if a.size < 3:
        return None
    ra, rb = stats.rankdata(a), stats.rankdata(b)
    ra -= ra.mean()
    rb -= rb.mean()
    den = np.sqrt((ra * ra).sum() * (rb * rb).sum())
    if den == 0:
        return None
    return float((ra * rb).sum() / den)


def correlations(x: np.ndarray, y: np.ndarray, codes: np.ndarray | None,
                 min_cells: int = MIN_GROUP_CELLS) -> dict:
    """rho over all cells, and the median of the within-group rhos (groups of at
    least ``min_cells`` cells; code -1 = no group), so groups that differ in depth and
    in score do not pass for a depth effect."""
    out = {"overall": spearman(x, y), "within": None, "n_groups": 0}
    if codes is None:
        return out
    rhos = []
    for c in np.unique(codes[codes >= 0]):
        m = codes == c
        if m.sum() >= min_cells:
            rho = spearman(x[m], y[m])
            if rho is not None:
                rhos.append(rho)
    if rhos:
        out["within"] = float(np.median(rhos))
        out["n_groups"] = len(rhos)
    return out


def verdict(rho: float | None) -> str:
    if rho is None:
        return "cannot be checked against depth"
    if abs(rho) >= STRONG:
        return "follows depth"
    if abs(rho) >= WEAK:
        return "follows depth weakly"
    return "does not follow depth"


def depth_check(scores: dict[str, np.ndarray], depth: dict[str, np.ndarray],
                codes: np.ndarray | None) -> dict:
    """Spearman of each method's score with each depth measure (overall and within
    groups), and between the methods."""
    out: dict = {"methods": {}, "between": None}
    for name, s in scores.items():
        out["methods"][name] = {k: correlations(s, d, codes) for k, d in depth.items()}
    names = list(scores)
    if len(names) == 2:
        out["between"] = correlations(scores[names[0]], scores[names[1]], codes)
    return out


def _rho_text(rho: float | None) -> str:
    return "n/a" if rho is None else f"{rho:.2f}"


METRIC_TEXT = {"total_counts": "total counts", "n_genes": "genes detected"}


def headline(c: dict) -> float | None:
    """The within-group rho, else the overall one."""
    return c["within"] if c["within"] is not None else c["overall"]


def strongest_metric(check: dict) -> str:
    """The depth measure either method follows most (largest |rho|), so the sentence
    compares both methods on the same measure without hiding the worse one."""
    def size(metric: str) -> float:
        rhos = [headline(m[metric]) for m in check["methods"].values()]
        return max((abs(r) for r in rhos if r is not None), default=-1.0)

    metrics = list(next(iter(check["methods"].values())))
    return max(metrics, key=size)


def sentence(check: dict, labels: dict[str, str], metric: str, where: str) -> str:
    """'score_genes follows depth on this data (rho 0.41 with total counts, within
    clusters); AUCell does not follow depth (rho 0.05).' ``where``: 'within clusters'
    or 'within <column> groups'."""
    parts = []
    for i, (name, label) in enumerate(labels.items()):
        c = check["methods"][name][metric]
        rho = headline(c)
        here = where if c["within"] is not None else "over all cells"
        tail = (f" on this data (rho {_rho_text(rho)} with {METRIC_TEXT.get(metric, metric)}, {here})"
                if i == 0 else f" (rho {_rho_text(rho)})")
        parts.append(f"{label} {verdict(rho)}{tail}")
    return "; ".join(parts) + "."
