"""TF and pathway activity with the univariate linear model (ULM), as decoupler fits it.

For one profile (a cell's expression, or a DE result's statistics) and one source
(a TF in CollecTRI, a pathway in PROGENy), ULM fits ``y = b0 + b1 * w`` over every
measured gene, where a gene's ``w`` is its signed weight in the source's network
and 0 when it is not a target. The activity is the t-value of ``b1``: positive
when the targets the source activates are high and the ones it represses low
(Badia-i-Mompel et al. 2022, decoupler, PMID 36699385).

The t-value of a one-covariate fit is the Pearson correlation rescaled,
``t = r sqrt(df / ((1 - r)(1 + r)))`` with ``df = n_genes - 2``, and r needs only
the product ``X @ W`` and each row's and column's mean and spread. So every cell
and every source cost one sparse product: no densified matrix, no loop over cells,
and no dependency on decoupler itself. The numbers match decoupler 2.0.3's
``dc.mt.ulm`` (``tests/test_activity.py`` checks the regression definition; the
Kang comparison is in the C073 report).

As decoupler does, genes with no expression in any cell (or a zero statistic, for a
DE profile) are left out, and sources with fewer than ``TMIN`` targets among the
remaining genes are dropped.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy import stats

TMIN = 5  # decoupler's default: fewer targets measured than this, no score
_EPS = 2.2e-16  # decoupler's guard in the t-value at r = +-1
_ROWS = 20_000  # cells per block when every source is scored
KEEP_BLOCK_BYTES = 64 * 1024**2  # working copies of one block of cells while CellScores fills
_BLOCK_COPIES = 8  # float64 cells x sources arrays _t and the product hold at once, about


def tval(r: np.ndarray, df: int) -> np.ndarray:
    return r * np.sqrt(df / ((1.0 - r + _EPS) * (1.0 + r + _EPS)))


@dataclass
class Network:
    """A network restricted to the measured genes: the ``genes x sources`` weights."""

    sources: list[str]
    n_targets: np.ndarray  # measured targets per source
    W: sp.csc_matrix  # rows: every gene of the profile, 0 for a non-target
    n_features: int  # measured genes (the regression's sample size)
    mean_w: np.ndarray = field(init=False)
    sd_w: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        # Over the measured genes; unmeasured rows of W are 0 and not counted.
        n = self.n_features
        s1 = np.asarray(self.W.sum(axis=0)).ravel()
        s2 = np.asarray(self.W.multiply(self.W).sum(axis=0)).ravel()
        self.mean_w = s1 / n
        self.sd_w = np.sqrt(np.maximum(s2 - n * self.mean_w**2, 0.0) / (n - 1))

    @property
    def df(self) -> int:
        return self.n_features - 2


def restrict(net: pd.DataFrame, genes, measured: np.ndarray, tmin: int = TMIN) -> Network:
    """``net`` (source, target, weight) on a profile with gene names ``genes``, of which
    ``measured`` (bool mask) are the genes ULM regresses over."""
    genes = pd.Index(np.asarray(genes, dtype=str))
    if genes.has_duplicates:
        raise ValueError("Gene names must be unique to score activity.")
    n_features = int(measured.sum())
    if n_features < 3:
        raise ValueError("Too few measured genes to score activity.")
    pos = genes.get_indexer(net["target"].astype(str))
    keep = pos >= 0
    keep[keep] = measured[pos[keep]]
    edges = net.loc[keep, ["source", "weight"]].assign(row=pos[keep])
    counts = edges["source"].value_counts()
    sources = sorted(counts.index[counts >= tmin])
    if not sources:
        raise ValueError(
            f"No source has {tmin} or more targets among this dataset's genes: the network "
            "uses human gene symbols."
        )
    col = pd.Index(sources)
    edges = edges[edges["source"].isin(col)]
    W = sp.csc_matrix(
        (edges["weight"].to_numpy(float), (edges["row"].to_numpy(), col.get_indexer(edges["source"]))),
        shape=(len(genes), len(sources)),
    )
    return Network(sources=list(sources), n_targets=counts[sources].to_numpy(int), W=W,
                   n_features=n_features)


def _non_negative(M) -> bool:
    from scview.core.working_copy import on_disk

    if on_disk(M):  # a backed dataset's matrix: its first rows, read (a CSR or an array)
        M = M[: min(M.shape[0], 1000)]
    sample = M.data[:200_000] if sp.issparse(M) else np.asarray(M[:1000])
    return sample.size == 0 or float(np.min(sample)) >= 0


class OnDiskMatrix(ValueError):
    """``lognorm_matrix`` asked to read a backed dataset's matrix whole without ``read``:
    its caller holds no room for it (``working_copy.reading`` with ``lognorm_bytes``)."""


@dataclass
class Lognorm:
    """The matrix ``lognorm_matrix`` scores, before anything is read. For a dataset opened
    backed (``core/working_copy.py``) ``M`` is its X or .raw's X on disk, read as indexed."""

    M: object
    names: pd.Index
    source: str
    counts: bool  # M holds raw counts: their shifted log is taken, a cell (row) at a time
    where: str | None  # the h5ad matrix M is ("X", "raw/X"), for its size; None: in memory
    cols: np.ndarray | None = None  # M's columns of ``names`` (on-disk counts in .raw only)

    @property
    def on_disk(self) -> bool:
        from scview.core.working_copy import on_disk

        return on_disk(self.M)

    def _log(self, block):
        from scview.core.normalization import shifted_log

        if self.cols is not None:
            block = block[:, self.cols]
        return shifted_log(block) if self.counts else block

    def whole(self):
        """All of it in memory: a backed dataset's matrix read whole (hold the room first)."""
        from scview.core.working_copy import read_matrix

        return self._log(read_matrix(self.M))

    def rows(self, keep: np.ndarray):
        """The cells ``keep`` (a mask, or ascending positions) in memory: ``whole()[keep]``,
        with only those rows read (the shifted log is per cell)."""
        from scview.core.working_copy import read_rows

        keep = np.asarray(keep)
        if keep.dtype != bool:
            mask = np.zeros(self.M.shape[0], dtype=bool)
            mask[keep] = True
            keep = mask
        return self._log(read_rows(self.M, keep) if self.on_disk else self.M[keep])

    def blocks(self, rows: int = _ROWS):
        """(first cell, the next ``rows`` cells in memory), every cell in order: a backed
        dataset's matrix is read a block at a time, never whole."""
        for start in range(0, self.M.shape[0], rows):
            yield start, self._log(self.M[start:start + rows])


def lognorm_source(adata) -> Lognorm:
    """Which matrix to score, read only to judge it (a backed dataset's first rows).

    ``.raw`` after scView's pipeline (under either normalisation method), else X
    when it holds log values, else the shifted log of the counts. Never z-scores or
    Pearson residuals: ULM regresses expression levels on the network weights."""
    from scview.core import large_mode
    from scview.core.normalization import find_counts
    from scview.core.pipeline import looks_like_counts
    from scview.core.working_copy import on_disk

    def is_lognorm(M) -> bool:
        return not looks_like_counts(M) and _non_negative(M)

    raw = adata.raw
    if raw is not None and is_lognorm(raw.X):
        return Lognorm(raw.X, raw.var_names, "log-normalised .raw", False, "raw/X")
    if is_lognorm(adata.X):
        return Lognorm(adata.X, adata.var_names, "log-normalised X", False, "X")
    if (raw is not None and on_disk(raw.X) and not raw.var_names.equals(adata.var_names)
            and not large_mode.is_large(adata)
            and not ("counts" in adata.layers and looks_like_counts(adata.layers["counts"]))
            and not len(adata.var_names.difference(raw.var_names)) and looks_like_counts(raw.X)):
        # find_counts's choice, but cut to X's genes as it is read: find_counts cuts it whole
        return Lognorm(raw.X, adata.var_names, "shifted log of the counts in .raw", True, "raw/X",
                       cols=raw.var_names.get_indexer(adata.var_names))
    try:
        counts, source = find_counts(adata, read_only=True)
    except ValueError:
        raise ValueError(
            "No log-normalised expression or raw counts to score activity on: run "
            "normalization and log transform first."
        ) from None
    where = {".raw": "raw/X", "X": "X"}.get(source) if on_disk(counts) else None
    return Lognorm(counts, adata.var_names, f"shifted log of the counts in {source}", True, where)


def lognorm_matrix(adata, read: bool = False):
    """Log-normalised expression to score, in memory: (matrix, gene names, where from), as
    ``lognorm_source`` picks it. A backed dataset's matrix is read whole only with ``read``,
    by a caller holding the room for it (``lognorm_bytes``); else ``OnDiskMatrix``."""
    choice = lognorm_source(adata)
    if choice.on_disk and not read:
        raise OnDiskMatrix(
            "This dataset is read from its file as it is used (it opened backed), and this "
            "would read its whole expression matrix into memory without room held for it."
        )
    return choice.whole(), choice.names, choice.source


def lognorm_bytes(adaptor, n_rows: int | None = None) -> int:
    """Memory ``lognorm_matrix(read=True)`` adds on a backed dataset (``n_rows``: of
    ``Lognorm.rows`` of that many cells), from the file header: the rows read, and when they
    are counts the shifted log's two float64 copies. 0 when the matrix is in memory."""
    from scview.core import preflight

    adata = adaptor.adata
    if not getattr(adata, "isbacked", False):
        return 0
    choice = lognorm_source(adata)
    if not choice.on_disk or choice.where is None:
        return 0
    mm = preflight.read_header(adaptor.h5ad_path, "h5ad").matrix(choice.where)
    if mm is None or not mm.n_cells:
        return 0
    n = int(adata.n_obs if n_rows is None else n_rows)
    need = preflight.rows_bytes(mm, n)
    if choice.counts:  # astype(float64) and the scaled copy: values and indices
        need += 2 * 12 * int(mm.nnz * min(1.0, n / mm.n_cells))
    return int(need)


def measured_genes(M) -> np.ndarray:
    """Genes expressed in at least one cell (decoupler drops the others)."""
    if sp.issparse(M):
        M = sp.csr_matrix(M)
        return np.bincount(M.indices[M.data != 0], minlength=M.shape[1]) > 0
    return np.count_nonzero(np.asarray(M), axis=0) > 0


@dataclass
class CellModel:
    """Everything ULM needs to score cells of one matrix against one network."""

    M: object  # cells x genes, log-normalised; CSR or dense
    net: Network
    mean_x: np.ndarray
    sd_x: np.ndarray

    @classmethod
    def build(cls, M, genes, net: pd.DataFrame, tmin: int = TMIN) -> "CellModel":
        if sp.issparse(M):
            M = sp.csr_matrix(M)
        else:
            M = np.asarray(M)
        measured = measured_genes(M)
        network = restrict(net, genes, measured, tmin)
        n = network.n_features
        if sp.issparse(M):
            s1 = np.asarray(M.sum(axis=1), dtype=np.float64).ravel()
            s2 = np.asarray(M.multiply(M).sum(axis=1), dtype=np.float64).ravel()
        else:
            Md = M.astype(np.float64, copy=False)
            s1, s2 = Md.sum(axis=1), (Md * Md).sum(axis=1)
        mean_x = s1 / n  # unmeasured genes are 0 in every cell: they add nothing
        sd_x = np.sqrt(np.maximum(s2 - n * mean_x**2, 0.0) / (n - 1))
        return cls(M=M, net=network, mean_x=mean_x, sd_x=sd_x)

    def _t(self, xw: np.ndarray, rows: slice, cols) -> np.ndarray:
        nw, n = self.net, self.net.n_features
        mx, sx = self.mean_x[rows, None], self.sd_x[rows, None]
        cov = (xw - n * mx * nw.mean_w[cols]) / (n - 1)
        with np.errstate(divide="ignore", invalid="ignore"):
            r = cov / (sx * nw.sd_w[cols])
        t = tval(r, nw.df)
        t[~np.isfinite(t)] = 0.0  # an empty cell: no evidence either way
        return t

    def source(self, name: str) -> np.ndarray:
        """Every cell's activity for one source (float32)."""
        j = self.net.sources.index(name)
        w = self.net.W[:, j]
        xw = np.asarray(self.M @ w.toarray()).reshape(-1, 1)
        return self._t(xw, slice(None), [j]).ravel().astype(np.float32)

    def blocks(self, rows: int = _ROWS):
        """(row slice, cells x sources activity) for every source, a block of cells at a time."""
        n = self.M.shape[0]
        for i in range(0, n, rows):
            sl = slice(i, min(n, i + rows))
            xw = self.M[sl] @ self.net.W
            xw = xw.toarray() if sp.issparse(xw) else np.asarray(xw)
            yield sl, self._t(xw, sl, slice(None))

    def all(self, rows: int = _ROWS) -> np.ndarray:
        """cells x sources activity (float32); memory grows with both."""
        out = np.empty((self.M.shape[0], len(self.net.sources)), dtype=np.float32)
        for sl, t in self.blocks(rows):
            out[sl] = t
        return out

    def group_means(self, codes: np.ndarray, n_groups: int) -> tuple[np.ndarray, np.ndarray]:
        """(groups x sources mean activity, cells per group); code -1 = no group."""
        return _group_means(self.blocks(), len(self.net.sources), codes, n_groups)


def _group_means(blocks, n_sources: int, codes: np.ndarray, n_groups: int) -> tuple[np.ndarray, np.ndarray]:
    sums = np.zeros((n_groups, n_sources))
    sizes = np.bincount(codes[codes >= 0], minlength=n_groups)
    for sl, t in blocks:
        c = codes[sl]
        ok = c >= 0
        onehot = sp.csr_matrix((np.ones(ok.sum()), (c[ok], np.flatnonzero(ok))),
                               shape=(n_groups, t.shape[0]))
        sums += onehot @ t
    with np.errstate(invalid="ignore"):
        means = sums / sizes[:, None]
    return means, sizes


def kept_bytes(n_cells: int, n_sources: int) -> int:
    """What ``CellScores`` keeps: every cell's activity for every source, float32."""
    return 4 * int(n_cells) * int(n_sources)


@dataclass
class CellScores:
    """Every cell's activity for every source of one network, kept in place of the
    ``CellModel`` it was scored with, and used as one (``net``, ``source``, ``group_means``).
    A backed dataset's model holds its matrix read from the file; these are ``kept_bytes``
    (Kang's 14k cells x CollecTRI's ~600 sources scored: 34 MB). Filled with
    ``KEEP_BLOCK_BYTES`` of working copies beside them at most."""

    net: Network
    scores: np.ndarray  # cells x sources, float32

    @classmethod
    def of(cls, model: CellModel) -> "CellScores":
        rows = max(1, KEEP_BLOCK_BYTES // (_BLOCK_COPIES * 8 * max(1, len(model.net.sources))))
        return cls(net=model.net, scores=model.all(rows=min(rows, _ROWS)))

    def source(self, name: str) -> np.ndarray:
        return self.scores[:, self.net.sources.index(name)]

    def blocks(self, rows: int = _ROWS):
        n = self.scores.shape[0]
        for i in range(0, n, rows):
            sl = slice(i, min(n, i + rows))
            yield sl, self.scores[sl]

    def group_means(self, codes: np.ndarray, n_groups: int) -> tuple[np.ndarray, np.ndarray]:
        return _group_means(self.blocks(), len(self.net.sources), codes, n_groups)


def score_profile(values, genes, net: pd.DataFrame, tmin: int = TMIN) -> pd.DataFrame:
    """ULM on one profile, e.g. a DE result's per-gene statistic: one row per source with
    its activity (t), two-sided p-value, BH-adjusted p and number of measured targets."""
    y = np.asarray(values, dtype=np.float64)
    genes = np.asarray(genes, dtype=str)
    ok = np.isfinite(y)
    y, genes = y[ok], genes[ok]
    measured = y != 0  # decoupler drops an all-zero feature, here a zero statistic
    network = restrict(net, genes, measured, tmin)
    n = network.n_features
    y = np.where(measured, y, 0.0)
    mean_y = y.sum() / n
    sd_y = np.sqrt(max((y * y).sum() - n * mean_y**2, 0.0) / (n - 1))
    xw = np.asarray(network.W.T @ y).ravel()
    with np.errstate(divide="ignore", invalid="ignore"):
        r = (xw - n * mean_y * network.mean_w) / (n - 1) / (sd_y * network.sd_w)
    t = tval(r, network.df)
    t[~np.isfinite(t)] = 0.0
    p = 2 * stats.t.sf(np.abs(t), network.df)
    return pd.DataFrame({
        "source": network.sources,
        "score": t,
        "pvalue": p,
        "padj": stats.false_discovery_control(p, method="bh"),
        "n_targets": network.n_targets,
    })
