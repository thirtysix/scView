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
    sample = M.data[:200_000] if sp.issparse(M) else np.asarray(M[:1000])
    return sample.size == 0 or float(np.min(sample)) >= 0


def lognorm_matrix(adata):
    """Log-normalised expression to score: (matrix, gene names, where from).

    ``.raw`` after scView's pipeline (under either normalisation method), else X
    when it holds log values, else the shifted log of the counts. Never z-scores or
    Pearson residuals: ULM regresses expression levels on the network weights."""
    from scview.core.normalization import find_counts, shifted_log
    from scview.core.pipeline import looks_like_counts

    def is_lognorm(M) -> bool:
        return not looks_like_counts(M) and _non_negative(M)

    if adata.raw is not None and is_lognorm(adata.raw.X):
        return adata.raw.X, adata.raw.var_names, "log-normalised .raw"
    if is_lognorm(adata.X):
        return adata.X, adata.var_names, "log-normalised X"
    try:
        counts, source = find_counts(adata)
    except ValueError:
        raise ValueError(
            "No log-normalised expression or raw counts to score activity on: run "
            "normalization and log transform first."
        ) from None
    return shifted_log(counts), adata.var_names, f"shifted log of the counts in {source}"


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

    def all(self) -> np.ndarray:
        """cells x sources activity (float32); memory grows with both."""
        out = np.empty((self.M.shape[0], len(self.net.sources)), dtype=np.float32)
        for sl, t in self.blocks():
            out[sl] = t
        return out

    def group_means(self, codes: np.ndarray, n_groups: int) -> tuple[np.ndarray, np.ndarray]:
        """(groups x sources mean activity, cells per group); code -1 = no group."""
        sums = np.zeros((n_groups, len(self.net.sources)))
        sizes = np.bincount(codes[codes >= 0], minlength=n_groups)
        for sl, t in self.blocks():
            c = codes[sl]
            ok = c >= 0
            onehot = sp.csr_matrix((np.ones(ok.sum()), (c[ok], np.flatnonzero(ok))),
                                   shape=(n_groups, t.shape[0]))
            sums += onehot @ t
        with np.errstate(invalid="ignore"):
            means = sums / sizes[:, None]
        return means, sizes


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
