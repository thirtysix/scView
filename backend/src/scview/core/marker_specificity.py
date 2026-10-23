"""How specific a marker gene is to its group, beside how strongly it marks it.

A one-vs-rest Wilcoxon test ranks a gene high when its group expresses it more
than all other cells pooled, so a gene shared by two of thirteen groups ranks as
well as one found in a single group. Two columns tell them apart:

- **tau** (Yanai et al. 2005, Bioinformatics 21:650), over the groups' mean
  expression in the matrix the markers were ranked on (log-normalised ``.raw``
  after the pipeline): ``sum(1 - mean_i / max_mean) / (n_groups - 1)``. 0 means
  even across the groups, 1 expressed in one group only. Of the tissue
  specificity indices it was the most robust in a benchmark
  (Kryuchkova-Mostacci & Robinson-Rechavi 2017, Brief Bioinform 18:205).
- the **detection fraction** in the group against the highest other group (not
  the pooled rest, which a large unrelated population dilutes), and their ratio.

Tau is one value per gene, so it is computed once, when the markers are, and
kept inside the markers' own uns entry: whatever replaces or drops the markers
(re-marking, a merge, new labels) takes it along. Detection fractions come from
the markers' ``pts`` (scanpy ``pts=True``), or are computed here when absent.

No bimodality coefficient: on sparse counts any on/off gene is a two-point
distribution, whose coefficient is exactly 1 whatever its detection rate, so it
cannot rank markers.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
import scipy.sparse as sp

logger = logging.getLogger(__name__)

# Inside a markers entry (uns["rank_genes_groups__<column>"]): a DataFrame
# indexed by gene with a "tau" column.
SPECIFICITY_FIELD = "scview_specificity"

# Cells per block when summing a large or backed matrix; a dense block is also
# held under _DENSE_ELEMENTS values (50k cells x 30k genes would otherwise copy GBs).
_CHUNK = 20_000
_DENSE_ELEMENTS = 20_000_000


def group_names(rgg: Any) -> list[str]:
    names = getattr(getattr(rgg.get("names"), "dtype", None), "names", None)
    return [str(g) for g in names] if names else []


def has_specificity(rgg: Any) -> bool:
    return isinstance(rgg, dict) and isinstance(rgg.get(SPECIFICITY_FIELD), pd.DataFrame)


def tau(means: np.ndarray) -> np.ndarray:
    """Tau per gene from (n_groups, n_genes) mean expression; NaN where a gene is
    never expressed, or for fewer than two groups."""
    means = np.clip(np.asarray(means, dtype=np.float64), 0, None)
    n = means.shape[0]
    out = np.full(means.shape[1], np.nan)
    if n < 2:
        return out
    top = means.max(axis=0)
    ok = top > 0
    out[ok] = (1.0 - means[:, ok] / top[ok]).sum(axis=0) / (n - 1)
    return out


def _ranked_matrix(adata: Any, rgg: Any):
    """The matrix and gene names the markers were ranked on."""
    params = rgg.get("params", {}) if isinstance(rgg, dict) else {}
    layer = params.get("layer")
    if layer and layer in adata.layers:
        return adata.layers[layer], adata.var_names
    use_raw = params.get("use_raw")
    if use_raw is None:  # not recorded: scanpy's own default
        use_raw = adata.raw is not None
    if use_raw and adata.raw is not None:
        return adata.raw.X, adata.raw.var_names
    return adata.X, adata.var_names


def _group_sums(X, codes: np.ndarray, n_groups: int) -> tuple[np.ndarray, np.ndarray]:
    """Per-group sums of X and of (X > 0), shape (n_groups, n_genes), in blocks of
    cells so a backed matrix is never read whole. Cells with code -1 are left out."""
    n_cells, n_genes = X.shape
    sums = np.zeros((n_groups, n_genes))
    detected = np.zeros((n_groups, n_genes))
    if hasattr(X, "row_blocks"):  # a large dataset's X on disk: blocks of a bounded size
        blocks = X.row_blocks()
    else:
        rows = _CHUNK if sp.issparse(X) else min(_CHUNK, max(1, _DENSE_ELEMENTS // max(n_genes, 1)))
        blocks = ((start, None) for start in range(0, n_cells, rows))
    for start, block in blocks:
        stop = min(start + (block.shape[0] if block is not None else rows), n_cells)
        c = codes[start:stop]
        keep = np.flatnonzero(c >= 0)
        if keep.size == 0:
            continue
        if block is None:
            block = X[start:stop]
        if hasattr(block, "to_memory"):  # a backed sparse dataset's slice
            block = block.to_memory()
        if keep.size < len(c):
            block = block[keep]
        onehot = sp.csr_matrix(
            (np.ones(keep.size), (c[keep], np.arange(keep.size))), shape=(n_groups, keep.size)
        )
        if sp.issparse(block):
            block = sp.csr_matrix(block)
            sums += (onehot @ block).toarray()
            hit = block.copy()
            hit.data = (hit.data > 0).astype(np.float32)
            detected += (onehot @ hit).toarray()
        else:
            block = np.asarray(block)
            sums += onehot @ block
            detected += onehot @ (block > 0).astype(np.float32)
    return sums, detected


def compute(adata: Any, rgg: Any, column: str) -> tuple[pd.DataFrame, pd.DataFrame] | None:
    """(tau frame, detection fractions genes x groups) for the markers ``rgg`` of
    ``column``, or None when the column or its groups can't be found."""
    groups = group_names(rgg)
    if len(groups) < 2 or column not in adata.obs.columns:
        return None
    X, var_names = _ranked_matrix(adata, rgg)
    # Labels as the markers saw them: bool / numeric columns were ranked as strings.
    labels = adata.obs[column].astype(str).to_numpy()
    codes = pd.Categorical(labels, categories=groups).codes.astype(np.int64)
    counts = np.bincount(codes[codes >= 0], minlength=len(groups)).astype(np.float64)
    if (counts == 0).any():  # the markers' groups are not this column's any more
        return None
    sums, detected = _group_sums(X, codes, len(groups))
    means = sums / counts[:, None]
    index = pd.Index([str(g) for g in var_names])
    frame = pd.DataFrame({"tau": tau(means).astype(np.float32)}, index=index)
    pts = pd.DataFrame((detected / counts[:, None]).T, index=index, columns=groups)
    return frame, pts


def attach(adata: Any, rgg: Any, column: str) -> bool:
    """Store tau (and ``pts`` when the markers lack them) in the markers dict ``rgg``.

    Returns whether ``rgg`` changed. Best effort: a failure is logged and leaves
    the markers as they were, shown without the specificity columns.
    """
    try:
        result = compute(adata, rgg, column)
    except Exception as exc:  # never fail the markers over their extra columns
        logger.warning("Marker specificity for '%s' failed: %s", column, exc)
        return False
    if result is None:
        return False
    frame, pts = result
    rgg[SPECIFICITY_FIELD] = frame
    if not isinstance(rgg.get("pts"), pd.DataFrame):
        rgg["pts"] = pts
    return True


def columns_for_group(rgg: Any, group: str, genes: list[str]) -> dict[str, np.ndarray | list]:
    """The specificity columns for ``genes`` of ``group``, aligned with ``genes``.

    ``pct_max_other`` is the highest detection fraction among the other groups
    (``max_other_group`` names it) and ``pct_ratio`` the group's own over it
    (inf when no other group detects the gene). Columns that can't be computed
    are left out.
    """
    out: dict[str, np.ndarray | list] = {}
    spec = rgg.get(SPECIFICITY_FIELD)
    if isinstance(spec, pd.DataFrame) and "tau" in spec.columns:
        out["tau"] = spec["tau"].reindex(genes).to_numpy(dtype=np.float64)
    pts = rgg.get("pts")
    if isinstance(pts, pd.DataFrame) and group in pts.columns and pts.shape[1] > 1:
        sub = pts.reindex(genes)
        own = sub[group].to_numpy(dtype=np.float64)
        others = sub.drop(columns=[group])
        vals = others.to_numpy(dtype=np.float64)
        has = ~np.isnan(vals).all(axis=1)
        best = np.full(len(genes), np.nan)
        best[has] = np.nanmax(vals[has], axis=1)
        idx = np.zeros(len(genes), dtype=np.int64)
        idx[has] = np.nanargmax(vals[has], axis=1)
        names = [str(c) for c in others.columns]
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.where(best > 0, own / best, np.where(own > 0, np.inf, np.nan))
        out["pct_max_other"] = best
        out["max_other_group"] = [names[i] if h else None for i, h in zip(idx, has)]
        out["pct_ratio"] = ratio
    return out
