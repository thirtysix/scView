"""Gene scores and CellTypist for a dataset whose matrix stays on disk (large-dataset mode).

``score_genes`` / ``score_genes_cell_cycle`` give the scores ``sc.tl.score_genes`` and
``sc.tl.score_genes_cell_cycle`` give on the log-normalised matrix in memory, bit for bit:

- the genes' mean expression (which bins them for the control genes) comes from the gene
  store (``core/gene_store.py``): each gene's values summed in float32 in cell order, as
  scipy sums a CSR matrix's columns for scanpy's ``_sparse_nanmean`` (a dense matrix,
  which scanpy averages in float32, is followed too: ``dense=True``);
- the control genes are drawn by scanpy's own ``_score_genes_bins`` with its seed;
- each cell's mean over the gene set and over the control genes is scanpy's own
  ``_nan_means`` on that cell's row, one block of rows at a time: a row's mean never
  depends on the other rows.

``celltypist_annotate`` predicts a block of cells at a time with ``celltypist.annotate``
(each cell's prediction is its own: normalise, log, scale its genes, the model's linear
decision, the sigmoid), then votes over the clustering with CellTypist's own
``majority_vote``. CellTypist's own over-clustering (used when there is no clustering)
needs a neighbour graph of the whole input and is refused here.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from typing import Any

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp

from scview.core import chunked

logger = logging.getLogger(__name__)

# Working memory of one CellTypist block: the model's genes, dense, in float64 (the
# scaled input and its copies).
CELLTYPIST_BLOCK_BYTES = 192 * 2**20
CELLTYPIST_MIN_ROWS = 512


# ---------------------------------------------------------------------------
# gene means
# ---------------------------------------------------------------------------

def gene_means(store, *, dense: bool = False) -> np.ndarray:
    """Every gene's mean over all cells, as ``scanpy.tools._score_genes._nan_means(X,
    axis=0)`` gives it for the matrix the store holds: float32 sums in cell order (what
    ``csr.sum(axis=0)`` adds), then float64 over the cell count; ``dense``: as numpy's
    ``nanmean`` of a dense float32 matrix (the same sums, the mean in float32)."""
    ones = np.ones(store.n_cells, dtype=store.dtype)
    sums = np.zeros(store.n_genes, dtype=store.dtype)
    for g0, g1, block in store.blocks():
        # CSR (genes x cells) times ones: one running float32 sum per gene, cells ascending
        sums[g0:g1] = sp.csr_matrix(block.T) @ ones
    if dense:
        return (sums.astype(np.float64) / store.n_cells).astype(store.dtype)
    return sums.astype(np.float64) / store.n_cells


# ---------------------------------------------------------------------------
# per-cell means over gene sets, by row blocks
# ---------------------------------------------------------------------------

def _row_means_block(b, sets: list[np.ndarray], dense: bool) -> list[np.ndarray]:
    from scanpy.tools._score_genes import _nan_means

    if dense:
        d = b.toarray() if sp.issparse(b) else np.asarray(b)
        return [np.nanmean(d[:, idx], axis=1, dtype="float64") for idx in sets]
    b = sp.csr_matrix(b)
    return [_nan_means(b[:, idx], axis=1, dtype="float64") for idx in sets]


def row_means(X, sets: Sequence[np.ndarray], *, dense: bool = False) -> list[np.ndarray]:
    """Each cell's mean over each set of gene positions (one pass over the row blocks of
    ``X``), as scanpy's ``_nan_means(X[:, idx], axis=1, dtype="float64")``."""
    sets = [np.asarray(s, dtype=np.int64) for s in sets]
    parts: list[list[np.ndarray]] = [[] for _ in sets]
    nb = X.numblocks[0]
    for r in chunked._each_block(X, _row_means_block, [sets] * nb, [dense] * nb):
        for k, v in enumerate(r):
            parts[k].append(np.asarray(v).ravel())
    return [np.concatenate(p) if p else np.zeros(0) for p in parts]


# ---------------------------------------------------------------------------
# score_genes
# ---------------------------------------------------------------------------

def _controls(adata: ad.AnnData, means: np.ndarray, gene_list, *, ctrl_size: int,
              random_state: int | None, n_bins: int = 25) -> tuple[pd.Index, pd.Index]:
    """(the genes scored, the control genes) as ``sc.tl.score_genes`` picks them."""
    from scanpy.tools._score_genes import _check_score_genes_args, _score_genes_bins

    if random_state is not None:
        np.random.seed(random_state)
    gene_list, gene_pool, _ = _check_score_genes_args(adata, gene_list, None, use_raw=False,
                                                      layer=None)
    n = adata.n_vars

    def get_subset(genes):
        if len(genes) != n:  # only the whole gene pool is averaged here
            raise AssertionError("gene means are taken over every gene")
        return means[None, :]

    control = pd.Index([], dtype="string")
    for r_genes in _score_genes_bins(gene_list, gene_pool, ctrl_as_ref=True, ctrl_size=ctrl_size,
                                     n_bins=n_bins, get_subset=get_subset):
        control = control.union(r_genes)
    if len(control) == 0:
        raise RuntimeError("No control genes found in any cut. Try setting `ctrl_as_ref=False`.")
    return gene_list, control


def _positions(adata: ad.AnnData, genes: pd.Index) -> np.ndarray:
    """The columns scanpy's ``get_subset`` takes (all of X when every gene is asked for)."""
    if len(genes) == adata.n_vars:
        return np.arange(adata.n_vars)
    return adata.var_names.get_indexer(genes)


def score_sets(adata: ad.AnnData, X, means: np.ndarray, sets: list[dict], *,
               dense: bool = False) -> None:
    """Several ``sc.tl.score_genes`` calls in one pass over X: each of ``sets`` is a dict of
    ``genes``, ``score_name``, ``ctrl_size`` and ``random_state``, drawn in order (each call
    seeds numpy's generator itself), written to ``adata.obs[score_name]`` (float64)."""
    picks = [_controls(adata, means, s["genes"], ctrl_size=int(s["ctrl_size"]),
                       random_state=s.get("random_state", 0)) for s in sets]
    idx = []
    for genes, control in picks:
        idx += [_positions(adata, genes), _positions(adata, control)]
    m = row_means(X, idx, dense=dense)
    for k, s in enumerate(sets):
        score = m[2 * k] - m[2 * k + 1]
        adata.obs[s["score_name"]] = pd.Series(np.array(score).ravel(), index=adata.obs_names,
                                               dtype="float64")
        logger.info("Score %r from the gene store: %d genes, %d control genes", s["score_name"],
                    len(picks[k][0]), len(picks[k][1]))


def score_genes(adata: ad.AnnData, X, means: np.ndarray, gene_list, *, score_name: str,
                ctrl_size: int = 50, random_state: int | None = 0, dense: bool = False) -> None:
    """``sc.tl.score_genes(adata, gene_list, ...)`` on the log-normalised matrix ``X`` (row
    blocks), with ``means`` from ``gene_means``."""
    score_sets(adata, X, means, [{"genes": gene_list, "score_name": score_name,
                                  "ctrl_size": ctrl_size, "random_state": random_state}],
               dense=dense)


def score_genes_cell_cycle(adata: ad.AnnData, X, means: np.ndarray, *, s_genes, g2m_genes,
                           dense: bool = False) -> None:
    """``sc.tl.score_genes_cell_cycle``: S and G2M scores (one pass over X) and the phase."""
    ctrl_size = min(len(s_genes), len(g2m_genes))
    score_sets(adata, X, means, [
        {"genes": s_genes, "score_name": "S_score", "ctrl_size": ctrl_size, "random_state": 0},
        {"genes": g2m_genes, "score_name": "G2M_score", "ctrl_size": ctrl_size, "random_state": 0},
    ], dense=dense)
    scores = adata.obs[["S_score", "G2M_score"]]
    phase = pd.Series("S", index=scores.index)
    phase[scores["G2M_score"] > scores["S_score"]] = "G2M"
    phase[np.all(scores < 0, axis=1)] = "G1"
    adata.obs["phase"] = phase


# ---------------------------------------------------------------------------
# CellTypist
# ---------------------------------------------------------------------------

def _as_block(b):
    return sp.csr_matrix(b) if sp.issparse(b) else np.asarray(b)


def _sub_blocks(n: int, rows: int) -> list[tuple[int, int]]:
    """Row ranges of at most ``rows``; a short tail joins the range before it, so no
    prediction runs on a handful of rows (BLAS takes another path for tiny matrices)."""
    out = [(a, min(n, a + rows)) for a in range(0, n, rows)]
    if len(out) > 1 and out[-1][1] - out[-1][0] < rows // 2:
        a, _ = out[-2]
        out[-2:] = [(a, n)]
    return out


def celltypist_annotate(model: Any, counts, obs: pd.DataFrame, var_names: pd.Index,
                        groupby: str, *, progress: Callable[[int, int], None] | None = None):
    """``celltypist.annotate(input, model, majority_voting=True, over_clustering=groupby)``
    where ``input`` is ``normalize_total(target_sum=1e4)`` + ``log1p`` of ``counts`` (row
    blocks), predicted a block at a time. Returns CellTypist's ``AnnotationResult`` (labels
    with ``majority_voting``, the probability matrix in float32; no decision matrix)."""
    import scanpy as sc
    from celltypist import classifier
    from celltypist.models import Model

    if not groupby:
        raise ValueError(
            "CellTypist in large-dataset mode votes over an existing clustering: its own "
            "over-clustering needs the whole matrix in memory. Run clustering first.")
    model = model if isinstance(model, Model) else Model.load(model)
    n_feat = max(1, int(np.isin(var_names, model.classifier.features).sum()))
    rows = max(CELLTYPIST_MIN_ROWS, CELLTYPIST_BLOCK_BYTES // (n_feat * 8 * 3))
    names = pd.Index(obs.index)
    labels: list[np.ndarray] = []
    probs: list[np.ndarray] = []
    classes = None
    start = 0
    total = counts.shape[0]
    for b in chunked._each_block(counts, _as_block):
        for a, z in _sub_blocks(b.shape[0], rows):
            inp = ad.AnnData(X=b[a:z].copy(),
                             obs=pd.DataFrame(index=names[start + a:start + z]),
                             var=pd.DataFrame(index=var_names))
            sc.pp.normalize_total(inp, target_sum=1e4)
            sc.pp.log1p(inp)
            pred = classifier.Classifier(filename=inp, model=model).celltype(mode="best match")
            labels.append(pred.predicted_labels["predicted_labels"].astype(str).to_numpy())
            probs.append(pred.probability_matrix.to_numpy(dtype=np.float32))
            classes = pred.probability_matrix.columns
        start += b.shape[0]
        if progress is not None:
            progress(start, total)
    lab = pd.DataFrame(np.concatenate(labels), columns=["predicted_labels"], index=names,
                       dtype="category")
    prob = pd.DataFrame(np.concatenate(probs), columns=classes, index=names)
    result = classifier.AnnotationResult(lab, None, prob, None)
    if result.cell_count <= 50:
        logger.warning("CellTypist: %d cells are too few for majority voting", result.cell_count)
        return result
    over = pd.Series(obs[groupby].values, index=names)  # as annotate reads its obs column
    return classifier.Classifier.majority_vote(result, over, min_prop=0)
