"""Mixscape: which cells of a CRISPR screen were really perturbed (N12).

A guide does not knock out its target in every cell that carries it (in-frame
edits, a missed guide). Mixscape (Papalexi et al. 2021, Nat Genet 53:322,
PMID 33649593; Seurat ``CalcPerturbSig`` + ``RunMixscape``, pertpy
``Mixscape``) splits each target's cells into perturbed ("KO") and not
perturbed ("NP"):

1. Perturbation signature: each cell's log-normalised expression subtracted
   from the mean of its ``n_neighbors`` (20) nearest control (non-targeting)
   cells in PCA space (the first ``n_dims`` = 15 PCs), the mean taken on the
   count scale (``log1p(mean(expm1(x)))``) as Seurat does; within each value of
   ``split_by`` (a replicate or batch column) when given. The neighbours are
   exact (pertpy uses approximate ones, NNDescent).
2. Per target, a Wilcoxon test of its cells against the controls on the
   log-normalised expression of the variable genes (scanpy's
   ``rank_genes_groups`` with ``reference``, Benjamini-Hochberg; Seurat's and
   pertpy's default ``de.assay``); genes with |log2 FC| >= 0.25 and adjusted
   p < 0.05. Fewer than ``min_de_genes`` (5): the whole target is "NP".
3. The signature of those genes, scaled per gene over the target's and the
   controls' cells, is projected on the difference of the target's and the
   controls' mean signature. A two-component Gaussian mixture is fitted to the
   projections, the control component's mean and variance fixed at the
   controls' own; a target cell with a posterior above 0.5 for the other
   component is "KO". The KO cells then give the next difference vector, up
   to ``iter_num`` (10) rounds or until no cell changes class; fewer than
   ``min_de_genes`` KO cells make the whole target "NP" (Seurat's rule).

The mixture is fitted by EM to convergence with both weights free, the control
component's mean and variance held at the controls' (Seurat's
``normalmixEM(mean.constr, sd.constr)``; pertpy 1.0.3 fixes the variance only
in the stored parameters, not in its next E-step, so its control component
widens; the posterior includes the mixing weights, as pertpy's does).

Outputs (obs): ``mixscape_class`` ("<target> KO", "<target> NP", "NT"),
``mixscape_class_global`` (KO / NP / NT), ``mixscape_p_ko`` (the posterior,
target cells only). Cells with no single target ("multiple", "none") are left
out (empty). Per target: cells, KO fraction, DE genes, the posterior histogram.
An optional linear discriminant projection (sklearn) of the KO and control
cells on the signature of the DE genes gives a 2-D view.
"""

from __future__ import annotations

import json
import logging
import threading
import time
import weakref
from collections.abc import Callable, Iterator
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import scipy.sparse as sp

from scview.core import results_store

logger = logging.getLogger(__name__)

N_NEIGHBORS = 20
N_DIMS = 15
LOGFC = 0.25
PVAL = 0.05
MIN_DE_GENES = 5
ITER_NUM = 10
N_HVG = 2000  # variable genes when the dataset has none marked
MIN_TARGET_CELLS = 5
MAX_SIGNATURE_VALUES = 300_000_000  # cells x DE genes kept as float32 (1.2 GB)
MAX_LDA_POINTS = 5000
HIST_BINS = 20
EM_TOL = 1e-8
EM_MAX_ITER = 1000
REG_VAR = 1e-6  # added to the free component's variance (sklearn's reg_covar)
UNS_KEY = "scview_mixscape"
CLASS, GLOBAL, P_KO = "mixscape_class", "mixscape_class_global", "mixscape_p_ko"
OBS_COLUMNS = (CLASS, GLOBAL, P_KO)
KO, NP, NT = "KO", "NP", "NT"
TAG = "result:mixscape"

_WHAT = "Mixscape"


class MixscapeError(ValueError):
    """A request Mixscape cannot serve; ``status`` is the HTTP code."""

    def __init__(self, message: str, status: int = 400) -> None:
        super().__init__(message)
        self.status = status


class MixscapeCancelled(Exception):
    """The client went away: nothing is saved."""


@dataclass
class Params:
    labels: str = "perturbation"
    control: str = "NT"
    split_by: str | None = None
    n_neighbors: int = N_NEIGHBORS
    n_dims: int = N_DIMS
    logfc_threshold: float = LOGFC
    pval_cutoff: float = PVAL
    min_de_genes: int = MIN_DE_GENES
    iter_num: int = ITER_NUM
    scale: bool = True
    lda: bool = True


@dataclass
class MixscapeResult:
    params: dict
    labels: str
    control: str
    split_by: str | None
    n_cells: int  # cells used: the targets' and the controls'
    n_control: int
    n_targets: int
    n_genes: int  # variable genes tested
    gene_source: str
    expression_source: str
    targets: list[dict]
    warnings: list[str] = field(default_factory=list)
    lda: dict | None = None
    seconds: float = 0.0
    created: str = ""
    n_cells_total: int = 0

    def summary(self) -> dict:
        n_target_cells = sum(t["n_cells"] for t in self.targets)
        n_ko = sum(t["n_ko"] for t in self.targets)
        return {"n_target_cells": n_target_cells, "n_ko": n_ko,
                "ko_fraction": (n_ko / n_target_cells) if n_target_cells else 0.0,
                "n_targets_with_ko": sum(1 for t in self.targets if t["n_ko"] > 0)}

    def to_dict(self) -> dict:
        d = asdict(self)
        d["summary"] = self.summary()
        return d


@dataclass
class Plan:
    params: Params
    rows: np.ndarray  # cell positions used (targets + controls)
    labels: np.ndarray  # their labels (str)
    targets: list[str]
    skipped: dict[str, int]  # targets with too few cells -> their count
    splits: list[tuple[str, np.ndarray]]  # (name, mask over rows)
    gene_idx: np.ndarray  # columns of the expression matrix tested
    gene_names: np.ndarray
    gene_source: str
    expression_source: str
    n_cells_total: int

    def to_dict(self) -> dict:
        return {"labels": self.params.labels, "control": self.params.control, "split_by": self.params.split_by,
                "n_cells": int(self.rows.size), "n_control": int((self.labels == self.params.control).sum()),
                "n_targets": len(self.targets), "targets": self.targets, "skipped": self.skipped,
                "n_genes": int(self.gene_idx.size), "gene_source": self.gene_source,
                "expression_source": self.expression_source, "n_splits": len(self.splits),
                "params": asdict(self.params)}


# ---------------------------------------------------------------------------
# Plan: which cells, which genes
# ---------------------------------------------------------------------------


def _expression(adata):
    from scview.core.activity import lognorm_matrix

    try:
        return lognorm_matrix(adata)
    except ValueError as e:
        raise MixscapeError(str(e), 422) from None


def _variable_genes(adata, M, var_names) -> tuple[np.ndarray, str]:
    """Columns of ``M`` to test: the dataset's variable genes, else the ``N_HVG``
    with the highest variance of the log values."""
    names = pd.Index(np.asarray(var_names).astype(str))
    hv = adata.var.get("highly_variable") if hasattr(adata, "var") else None
    if hv is not None and bool(np.asarray(hv, dtype=bool).any()):
        chosen = adata.var_names[np.asarray(hv, dtype=bool)].astype(str)
        idx = names.get_indexer(chosen)
        idx = np.sort(idx[idx >= 0])
        if idx.size >= MIN_DE_GENES:
            return idx, f"{idx.size:,} variable genes (var highly_variable)"
    if sp.issparse(M):
        Mc = sp.csr_matrix(M)
        mean = np.asarray(Mc.mean(axis=0)).ravel()
        sq = np.asarray(Mc.multiply(Mc).mean(axis=0)).ravel()
    else:
        A = np.asarray(M, dtype=np.float64)
        mean, sq = A.mean(axis=0), (A * A).mean(axis=0)
    var = sq - mean * mean
    n = min(N_HVG, names.size)
    idx = np.sort(np.argsort(-var, kind="stable")[:n])
    return idx, f"{n:,} genes with the highest variance (no variable genes marked)"


def plan(adata, params: Params) -> Plan:
    """Check a request and pick the cells and genes (nothing heavy is computed)."""
    from scview.core import large_mode

    if large_mode.is_large(adata):
        raise MixscapeError(f"{_WHAT} is not available for large datasets yet: {large_mode.WHOLE_MATRIX}.", 422)
    obs = adata.obs
    p = params
    if p.labels not in obs.columns:
        raise MixscapeError(f"Column '{p.labels}' not found: assign the guides first.")
    if p.split_by and p.split_by not in obs.columns:
        raise MixscapeError(f"Column '{p.split_by}' not found.")
    if not 1 <= p.n_neighbors <= 200:
        raise MixscapeError("Neighbours must be 1 to 200.")
    if not 0 <= p.logfc_threshold <= 10 or not 0 < p.pval_cutoff <= 1:
        raise MixscapeError("The fold-change threshold must be 0-10 and the p cut-off in (0, 1].")
    if not 1 <= p.min_de_genes <= 1000 or not 1 <= p.iter_num <= 100:
        raise MixscapeError("Minimum DE genes must be 1-1000 and rounds 1-100.")
    pca = adata.obsm.get("X_pca") if hasattr(adata, "obsm") else None
    if pca is None:
        raise MixscapeError("No PCA: run the pipeline (PCA) first.", 422)
    from scview.core import crispr

    lab = obs[p.labels].astype(object).where(obs[p.labels].notna(), None)
    lab = lab.map(lambda v: None if v is None else str(v))
    if not (lab == p.control).any():
        raise MixscapeError(f"No cells labelled '{p.control}' (the control) in '{p.labels}'.")
    left_out = {crispr.NONE, crispr.MULTIPLE, "", "nan"}
    counts = lab[~lab.isin(left_out) & lab.notna()].value_counts()
    targets, skipped = [], {}
    for t, n in counts.items():
        if t == p.control:
            continue
        if n < MIN_TARGET_CELLS:
            skipped[str(t)] = int(n)
        else:
            targets.append(str(t))
    if not targets:
        raise MixscapeError(f"No target has {MIN_TARGET_CELLS} or more cells.")
    if len(targets) > crispr.MAX_TARGETS:
        raise MixscapeError(f"{len(targets):,} targets: more than {crispr.MAX_TARGETS:,}.")
    keep = lab.isin(set(targets) | {p.control}).to_numpy()
    rows = np.flatnonzero(keep)
    labels = lab.to_numpy()[rows].astype(str)
    if p.split_by:
        sv = obs[p.split_by].astype(object).to_numpy()[rows]
        sv = np.array(["" if v is None or (isinstance(v, float) and np.isnan(v)) else str(v) for v in sv])
        splits = [(s, sv == s) for s in pd.unique(sv)]
    else:
        splits = [("all", np.ones(rows.size, dtype=bool))]
    for name, m in splits:
        n_ctrl = int(((labels == p.control) & m).sum())
        if n_ctrl < p.n_neighbors:
            where = f" in '{p.split_by}' = {name}" if p.split_by else ""
            raise MixscapeError(f"Only {n_ctrl} control cells{where}: the signature needs at least "
                                f"{p.n_neighbors} (lower the neighbours, or drop the split).")
    M, var_names, where = _expression(adata)
    gene_idx, gene_source = _variable_genes(adata, M, var_names)
    return Plan(params=p, rows=rows, labels=labels, targets=sorted(targets), skipped=skipped, splits=splits,
                gene_idx=gene_idx, gene_names=np.asarray(var_names).astype(str)[gene_idx],
                gene_source=gene_source, expression_source=where, n_cells_total=int(adata.n_obs))


# ---------------------------------------------------------------------------
# The steps
# ---------------------------------------------------------------------------


def _subset(adata, pl: Plan):
    """The tested genes of the used cells: CSR float32 (or dense float32)."""
    M, _, _ = _expression(adata)
    if sp.issparse(M):
        E = sp.csr_matrix(M)[pl.rows][:, pl.gene_idx]
        E = sp.csr_matrix(E, dtype=np.float32)
        E.sort_indices()
        return E
    return np.asarray(np.asarray(M)[pl.rows][:, pl.gene_idx], dtype=np.float32)


def neighbours(pca: np.ndarray, labels: np.ndarray, control: str, splits, k: int) -> list[tuple]:
    """Per split: (rows of the split, rows of its controls, the k nearest controls of
    each split row as positions into its controls). Exact Euclidean neighbours."""
    from sklearn.neighbors import NearestNeighbors

    out = []
    for _, m in splits:
        rows = np.flatnonzero(m)
        ctrl = np.flatnonzero(m & (labels == control))
        nn = NearestNeighbors(n_neighbors=k).fit(pca[ctrl])
        _, ind = nn.kneighbors(pca[rows], n_neighbors=k)
        out.append((rows, ctrl, ind))
    return out


def de_genes(E, labels: np.ndarray, targets: list[str], control: str, gene_names: np.ndarray,
             logfc: float, pval: float, min_de: int) -> dict[str, pd.DataFrame]:
    """Per target: its DE genes against the controls (Wilcoxon, BH), columns gene /
    pos (column of E) / logfc / pval_adj / score, rank order; empty when fewer than
    ``min_de``."""
    import anndata as ad

    from scview.core import rank_genes

    test = ad.AnnData(X=E, obs=pd.DataFrame({"g": pd.Categorical(labels)},
                                            index=[str(i) for i in range(len(labels))]),
                      var=pd.DataFrame(index=pd.Index(gene_names).astype(str)))
    rank_genes.rank_genes_groups(test, "g", method="wilcoxon", groups=targets, reference=control,
                                 use_raw=False, n_genes=int(E.shape[1]))
    rgg = test.uns.pop("rank_genes_groups")
    pos = pd.Index(gene_names.astype(str))
    out = {}
    for t in targets:
        df = pd.DataFrame({"gene": np.asarray(rgg["names"][t]).astype(str),
                           "logfc": np.asarray(rgg["logfoldchanges"][t], dtype=np.float64),
                           "pval_adj": np.asarray(rgg["pvals_adj"][t], dtype=np.float64),
                           "score": np.asarray(rgg["scores"][t], dtype=np.float64)})
        df = df[(df["logfc"].abs() >= logfc) & (df["pval_adj"] < pval)]
        if len(df) < min_de:
            df = df.iloc[0:0]
        df.insert(1, "pos", pos.get_indexer(df["gene"]))
        out[t] = df.reset_index(drop=True)
    return out


def signature(E, nbrs: list[tuple], cols: np.ndarray, n_rows: int, k: int,
              stop: Callable[[], bool] | None = None, chunk: int = 20_000) -> np.ndarray:
    """Rows x ``cols``: log1p(mean over the k control neighbours of expm1(x)) - x."""
    S = np.zeros((n_rows, cols.size), dtype=np.float32)
    if cols.size == 0:
        return S
    Ec = E[:, cols]
    for rows, ctrl, ind in nbrs:
        Xc = Ec[ctrl]
        Xc = Xc.toarray() if sp.issparse(Xc) else np.asarray(Xc)
        Xc = np.expm1(Xc.astype(np.float64))
        for start in range(0, rows.size, chunk):
            if stop is not None and stop():
                raise MixscapeCancelled()
            r = rows[start:start + chunk]
            idx = ind[start:start + chunk]
            W = sp.csr_matrix((np.full(idx.size, 1.0 / k), (np.repeat(np.arange(r.size), k), idx.ravel())),
                              shape=(r.size, ctrl.size))
            x = Ec[r]
            x = x.toarray() if sp.issparse(x) else np.asarray(x)
            S[r] = (np.log1p(W @ Xc) - x.astype(np.float64)).astype(np.float32)
    return S


def _scale(X: np.ndarray) -> np.ndarray:
    """Per column: zero mean, unit standard deviation (ddof 1), as scanpy's ``scale``."""
    X = np.asarray(X, dtype=np.float64)
    mean = X.mean(axis=0)
    n = X.shape[0]
    std = np.sqrt(((X - mean) ** 2).sum(axis=0) / max(1, n - 1))
    std[std == 0] = 1.0
    return (X - mean) / std


def _log_norm(x: np.ndarray, mean: float, var: float) -> np.ndarray:
    return -0.5 * (np.log(2 * np.pi * var) + (x - mean) ** 2 / var)


def fit_mixture(x: np.ndarray, m0: float, v0: float, m1: float, v1: float) -> tuple[np.ndarray, dict]:
    """EM for two 1-D Gaussians, component 0 fixed at (m0, v0); returns the
    posterior of component 1 for every value and the fitted parameters."""
    x = np.asarray(x, dtype=np.float64)
    v0 = max(float(v0), REG_VAR)
    v1 = max(float(v1), REG_VAR) if np.isfinite(v1) else max(v0, REG_VAR)
    m1 = float(m1)
    w = np.array([0.5, 0.5])
    prev = -np.inf
    n_iter = 0
    for n_iter in range(1, EM_MAX_ITER + 1):
        lp = np.stack([np.log(w[0]) + _log_norm(x, m0, v0), np.log(w[1]) + _log_norm(x, m1, v1)], axis=1)
        top = lp.max(axis=1, keepdims=True)
        ll = top[:, 0] + np.log(np.exp(lp - top).sum(axis=1))
        r = np.exp(lp - ll[:, None])
        mean_ll = float(ll.mean())
        nk = r.sum(axis=0) + 10 * np.finfo(float).eps
        w = nk / nk.sum()
        m1 = float((r[:, 1] * x).sum() / nk[1])
        v1 = float((r[:, 1] * (x - m1) ** 2).sum() / nk[1]) + REG_VAR
        if abs(mean_ll - prev) < EM_TOL:
            break
        prev = mean_ll
    lp = np.stack([np.log(w[0]) + _log_norm(x, m0, v0), np.log(w[1]) + _log_norm(x, m1, v1)], axis=1)
    top = lp.max(axis=1, keepdims=True)
    post = np.exp(lp[:, 1] - (top[:, 0] + np.log(np.exp(lp - top).sum(axis=1))))
    return post, {"weights": [float(w[0]), float(w[1])], "means": [float(m0), m1],
                  "sds": [float(np.sqrt(v0)), float(np.sqrt(v1))], "em_iterations": n_iter}


def classify(dat: np.ndarray, is_target: np.ndarray, is_control: np.ndarray, iter_num: int,
             min_ko: int, scale: bool = True) -> dict:
    """One target: ``dat`` = signature of its DE genes for its cells and the
    controls (rows). Returns the KO mask and posterior over its cells, rounds."""
    X = _scale(dat) if scale else np.asarray(dat, dtype=np.float64)
    nt_mean = X[is_control].mean(axis=0)
    tgt = np.flatnonzero(is_target)
    ko = np.ones(tgt.size, dtype=bool)
    post = np.zeros(tgt.size)
    rounds, fit, collapsed = 0, {}, False
    for rounds in range(1, iter_num + 1):
        vec = X[tgt[ko]].mean(axis=0) - nt_mean
        denom = float(vec @ vec)
        if not np.isfinite(denom) or denom == 0:
            ko[:] = False
            post[:] = 0.0
            collapsed = True
            break
        pvec = X @ vec / denom
        nt, gd = pvec[is_control], pvec[tgt[ko]]
        p_all, fit = fit_mixture(pvec, nt.mean(), nt.var(ddof=1), gd.mean(),
                                 gd.var(ddof=1) if gd.size > 1 else np.nan)
        post = p_all[tgt]
        new = post > 0.5
        if new.sum() < min_ko:
            ko[:] = False
            collapsed = True
            break
        changed = not np.array_equal(new, ko)
        ko = new
        if not changed:
            break
    return {"ko": ko, "post": post, "rounds": rounds, "fit": fit, "collapsed": collapsed}


def _lda(S: np.ndarray, labels: np.ndarray, rng: np.random.Generator) -> dict | None:
    """2-D linear discriminant projection of the KO and control cells."""
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

    classes = np.unique(labels)
    if classes.size < 2 or S.shape[1] == 0:
        return None
    n_comp = int(min(2, classes.size - 1, S.shape[1]))
    X = _scale(S)
    try:
        Z = LinearDiscriminantAnalysis(n_components=n_comp).fit(X, labels).transform(X)
    except (ValueError, np.linalg.LinAlgError) as e:
        logger.info("Mixscape LDA failed: %s", e)
        return None
    if Z.shape[1] == 1:
        Z = np.column_stack([Z[:, 0], rng.uniform(-0.5, 0.5, Z.shape[0])])
    idx = np.arange(Z.shape[0])
    if idx.size > MAX_LDA_POINTS:
        idx = np.sort(rng.choice(idx, MAX_LDA_POINTS, replace=False))
    return {"x": np.round(Z[idx, 0], 4).tolist(), "y": np.round(Z[idx, 1], 4).tolist(),
            "label": labels[idx].tolist(), "n_cells": int(Z.shape[0]), "n_shown": int(idx.size),
            "dims": n_comp}


def iter_mixscape(adata, pl: Plan, *, stop: Callable[[], bool] | None = None) -> Iterator[dict]:
    """Run Mixscape step by step: ``{"type": "progress", ...}`` per step, then
    ``{"type": "result", "result": MixscapeResult, "columns": {...}}``."""
    t0 = time.perf_counter()
    p = pl.params

    def check():
        if stop is not None and stop():
            raise MixscapeCancelled()

    n_steps = 3 + len(pl.targets) + (1 if p.lda else 0)
    yield {"type": "progress", "step": "neighbours", "done": 0, "total": n_steps,
           "message": f"Finding each cell's {p.n_neighbors} nearest control cells"}
    pca = np.asarray(adata.obsm["X_pca"])[pl.rows]
    pca = pca[:, :min(p.n_dims, pca.shape[1])].astype(np.float64)
    nbrs = neighbours(pca, pl.labels, p.control, pl.splits, p.n_neighbors)
    check()
    yield {"type": "progress", "step": "de", "done": 1, "total": n_steps,
           "message": f"Testing {len(pl.targets)} targets against the controls (Wilcoxon)"}
    E = _subset(adata, pl)
    de = de_genes(E, pl.labels, pl.targets, p.control, pl.gene_names, p.logfc_threshold, p.pval_cutoff,
                  p.min_de_genes)
    check()
    union = np.unique(np.concatenate([d["pos"].to_numpy() for d in de.values()] + [np.zeros(0, int)]))
    if union.size * pl.rows.size > MAX_SIGNATURE_VALUES:
        raise MixscapeError(f"The signature would hold {union.size:,} genes x {pl.rows.size:,} cells: more "
                            "than this run keeps in memory. Raise the fold-change threshold.", 422)
    yield {"type": "progress", "step": "signature", "done": 2, "total": n_steps,
           "message": f"Perturbation signature of {union.size:,} genes"}
    S = signature(E, nbrs, union, pl.rows.size, p.n_neighbors, stop)
    del E
    col_of = {int(g): i for i, g in enumerate(union)}
    is_ctrl = pl.labels == p.control
    cls = np.where(is_ctrl, NT, "").astype(object)
    glob = np.where(is_ctrl, NT, "").astype(object)
    pko = np.full(pl.rows.size, np.nan, dtype=np.float32)
    rows_out, warnings = [], []
    edges = np.linspace(0, 1, HIST_BINS + 1)
    for i, t in enumerate(pl.targets):
        check()
        yield {"type": "progress", "step": "classify", "done": 3 + i, "total": n_steps,
               "message": f"Classifying {t} ({i + 1}/{len(pl.targets)})"}
        is_t = pl.labels == t
        d = de[t]
        row = {"target": t, "n_cells": int(is_t.sum()), "n_de_genes": int(len(d)),
               "top_de_genes": [{"gene": g, "logfc": round(float(lf), 3), "pval_adj": float(pa)}
                                for g, lf, pa in d.sort_values("pval_adj", kind="stable")[["gene", "logfc", "pval_adj"]]
                                .head(10).itertuples(index=False)]}
        if len(d) == 0:
            post = np.zeros(int(is_t.sum()))
            ko = np.zeros(int(is_t.sum()), dtype=bool)
            row.update({"rounds": 0, "note": f"fewer than {p.min_de_genes} DE genes: all NP"})
        else:
            sub = np.flatnonzero(is_t | is_ctrl)
            dat = S[sub][:, [col_of[int(g)] for g in d["pos"].to_numpy()]]
            out = classify(dat, is_t[sub], is_ctrl[sub], p.iter_num, p.min_de_genes, p.scale)
            ko, post = out["ko"], out["post"]
            row.update({"rounds": out["rounds"], "fit": out["fit"]})
            if out["collapsed"]:
                row["note"] = f"fewer than {p.min_de_genes} cells called KO: all NP"
        tpos = np.flatnonzero(is_t)
        cls[tpos] = np.where(ko, f"{t} {KO}", f"{t} {NP}")
        glob[tpos] = np.where(ko, KO, NP)
        pko[tpos] = post.astype(np.float32)
        n_ko = int(ko.sum())
        row.update({"n_ko": n_ko, "n_np": int(row["n_cells"] - n_ko),
                    "ko_fraction": n_ko / row["n_cells"] if row["n_cells"] else 0.0,
                    "posterior_hist": {"edges": edges.round(3).tolist(),
                                       "ko": np.histogram(post[ko], bins=edges)[0].tolist(),
                                       "np": np.histogram(post[~ko], bins=edges)[0].tolist()}})
        rows_out.append(row)
    lda = None
    if p.lda:
        check()
        yield {"type": "progress", "step": "lda", "done": n_steps - 1, "total": n_steps,
               "message": "Projecting the KO and control cells (LDA)"}
        keep = (glob == KO) | is_ctrl
        if (glob == KO).any():
            lab = np.where(is_ctrl, NT, pl.labels).astype(str)[keep]
            lda = _lda(S[keep], lab, np.random.default_rng(0))
    if pl.skipped:
        warnings.append(f"{len(pl.skipped)} target(s) with fewer than {MIN_TARGET_CELLS} cells left out: "
                        + ", ".join(f"{k} ({v})" for k, v in list(pl.skipped.items())[:10]))
    result = MixscapeResult(
        params=asdict(p), labels=p.labels, control=p.control, split_by=p.split_by, n_cells=int(pl.rows.size),
        n_control=int(is_ctrl.sum()), n_targets=len(pl.targets), n_genes=int(pl.gene_idx.size),
        gene_source=pl.gene_source, expression_source=pl.expression_source, targets=rows_out,
        warnings=warnings, lda=lda, seconds=round(time.perf_counter() - t0, 2),
        created=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"), n_cells_total=pl.n_cells_total)
    yield {"type": "result", "result": result,
           "columns": {"rows": pl.rows, CLASS: cls, GLOBAL: glob, P_KO: pko}}


def run(adata, params: Params) -> tuple[MixscapeResult, dict]:
    """The whole run at once (tests, scripts)."""
    pl = plan(adata, params)
    for item in iter_mixscape(adata, pl):
        if item["type"] == "result":
            return item["result"], item["columns"]
    raise RuntimeError("Mixscape gave no result")  # pragma: no cover


# ---------------------------------------------------------------------------
# Saving into the dataset
# ---------------------------------------------------------------------------


def save(adata, result: MixscapeResult, columns: dict) -> dict:
    """Write the three obs columns (cells outside the run left empty), the record
    in uns and a provenance step."""
    from scview.core import provenance
    from scview.core.obs_writes import forget_column_results

    n = adata.n_obs
    rows = columns["rows"]
    if result.n_cells_total != n or (rows.size and rows.max() >= n):
        raise MixscapeError("The dataset's cells changed during the run; nothing was saved. Run it again.", 409)
    cls = np.full(n, None, dtype=object)
    glob = np.full(n, None, dtype=object)
    pko = np.full(n, np.nan, dtype=np.float32)
    cls[rows], glob[rows], pko[rows] = columns[CLASS], columns[GLOBAL], columns[P_KO]
    order = [NT] + [f"{t['target']} {s}" for t in result.targets for s in (KO, NP)]
    obs = adata.obs
    obs[CLASS] = pd.Categorical(cls, categories=[c for c in order if c in set(cls[rows])])
    obs[GLOBAL] = pd.Categorical(glob, categories=[c for c in (KO, NP, NT) if c in set(glob[rows])])
    obs[P_KO] = pko
    for c in (CLASS, GLOBAL):
        forget_column_results(adata, c)
    record = {k: v for k, v in result.to_dict().items() if k not in ("lda",)}
    for t in record["targets"]:
        t.pop("posterior_hist", None)
        t.pop("fit", None)
    adata.uns[UNS_KEY] = json.dumps(record, default=str)
    try:
        provenance.record_step(adata, step="mixscape", tool="scview mixscape (Papalexi 2021)", params={
            **result.params, "n_cells": result.n_cells, "n_targets": result.n_targets,
            "gene_source": result.gene_source, "summary": result.summary(),
            "ko_per_target": {t["target"]: t["n_ko"] for t in result.targets},
        })
    except Exception as e:  # provenance must never block the result
        logger.warning("Could not record Mixscape in provenance: %s", e)
    return result.to_dict()


def saved_record(adata) -> dict | None:
    raw = adata.uns.get(UNS_KEY) if hasattr(adata, "uns") else None
    try:
        rec = json.loads(raw) if isinstance(raw, str) else None
    except ValueError:
        return None
    return rec if isinstance(rec, dict) and CLASS in adata.obs.columns else None


# ---------------------------------------------------------------------------
# The last result per loaded dataset (co-pilot, the tab after a reload)
# ---------------------------------------------------------------------------

_last: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()
_last_lock = threading.Lock()


def remember(adaptor, result: MixscapeResult) -> None:
    try:
        with _last_lock:
            _last[adaptor] = result
    except TypeError:  # not weak-referenceable (a test double)
        return
    results_store.keep(adaptor, "mixscape", result, key=f"{result.labels}|{result.control}",
                       label=f"Mixscape: {result.n_targets} targets vs {result.control}",
                       params=result.params)


def last_result(adaptor) -> MixscapeResult | None:
    r = results_store.remembered(_last, _last_lock, adaptor, "mixscape")
    if r is None:
        return None
    obs = adaptor.adata.obs if hasattr(adaptor, "adata") else None
    if obs is not None and (r.n_cells_total != len(obs) or r.labels not in obs.columns):
        return None  # cells filtered, or the labels gone, since
    return r


def grounding_lines(adaptor) -> tuple[list[str], str | None]:
    """For the co-pilot: the guide assignment and the last Mixscape run (KO
    fraction per target)."""
    from scview.core import crispr

    adata = adaptor.adata
    lines, detail = crispr.grounding_lines(adata)
    r = last_result(adaptor)
    if r is None:
        return (["\n## CRISPR screen", *lines], detail or None) if lines else ([], None)
    s = r.summary()
    ranked = sorted(r.targets, key=lambda t: -t["ko_fraction"])
    per = "; ".join(f"{t['target']}: {t['n_ko']:,} of {t['n_cells']:,} cells KO ({t['ko_fraction']:.0%}), "
                    f"{t['n_de_genes']} DE genes" for t in ranked[:12])
    none = [t["target"] for t in r.targets if t["n_ko"] == 0]
    mdetail = (f"{r.n_targets} targets vs {r.n_control:,} '{r.control}' cells: {s['n_ko']:,} of "
               f"{s['n_target_cells']:,} target cells KO ({s['ko_fraction']:.0%}), "
               f"{s['n_targets_with_ko']} targets with any KO cells")
    out = ["\n## CRISPR screen (guide assignment and Mixscape)", *lines,
           f"- Mixscape (Papalexi 2021; signature vs {r.params.get('n_neighbors')} nearest control cells, "
           f"DE |log2FC| >= {r.params.get('logfc_threshold')}, adj p < {r.params.get('pval_cutoff')}): {mdetail}; "
           f"KO fraction per target: {per}" + (f"; no KO cells (all NP): {', '.join(none[:15])}" if none else "")
           + f" [{TAG}]"]
    return out, mdetail
