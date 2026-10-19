"""Each group's marker genes, as annotation and the co-pilot should see them.

``rank_genes_groups`` stores genes in wilcoxon score order. The score grows with
the number of cells as much as with specificity, so broadly high, abundant
transcripts lead the list: on Kang, score order puts FTL, CD63 and ANXA5 first in
"CD14 Mono" (CD14 at rank 70, LYZ at 18) and RPLP0/1/2 and RPSA in the T-cell top
15. Fed straight to a cell-type lookup that order is a poor description of a
group.

``group_markers`` keeps the genes expressed in at least ``min_pts`` of the group
(0.2) with a log fold change of at least ``min_logfc``, drops ribosomal and
mitochondrial genes (``gene_filters.is_ribo_mito``, the one definition), and sorts
by log fold change. Measured on Kang (research/marker-db-assessment, item 7): a
CellMarker 2.0 lookup from these markers named 0.69 of the cell types correctly
against 0.46 from score order (CD14 Mono and pDC became correct), and 25 markers
per group did clearly better than 10.

The order is a choice (``order``):

- ``logfc``: log fold change, highest first (the default);
- ``score``: the stored wilcoxon order, with the same filters;
- ``pts_diff``: fraction expressing in the group minus in the rest; the report
  paired it with ``min_logfc=1``.

Over-representation (ORA) queries use it too (``ora_query_genes``); the ORA
universe stays the full ranked list. The Markers table, dot plot and heatmap show
the full stored result, which the user sorts and filters; they don't use this.
"""

from __future__ import annotations

import logging
from typing import Literal

import numpy as np
import pandas as pd
import scipy.sparse as sp

from scview.core.gene_filters import is_ribo_mito
from scview.core.ora_background import resolve_markers_key

logger = logging.getLogger(__name__)

Order = Literal["logfc", "score", "pts_diff"]
ORDERS: tuple[str, ...] = ("logfc", "score", "pts_diff")
DEFAULT_N = 25
MIN_PTS = 0.2
MIN_LOGFC = 0.25
_CHUNK = 20000  # cells per block when counting expressing cells


def group_markers(
    adata,
    groupby: str,
    *,
    n: int = DEFAULT_N,
    min_pts: float = MIN_PTS,
    min_logfc: float = MIN_LOGFC,
    key: str | None = None,
    order: Order = "logfc",
    exclude_ribo_mito: bool = True,
) -> dict[str, list[str]]:
    """Per group of ``groupby`` (by label, in the stored group order): up to ``n``
    marker genes, best first.

    Reads the stored rank_genes_groups result for the column (``key`` overrides
    which one) and its ``pts``; without stored pts the fraction expressing is
    counted from the log-normalised matrix (``.raw`` when it holds log values).
    A gene whose fraction or fold change is unknown (NaN) fails that filter; a
    result without fold changes is not filtered or ordered by them. Empty when
    the column has no stored markers.
    """
    if order not in ORDERS:
        raise ValueError(f"Unknown marker order '{order}' (expected one of {', '.join(ORDERS)}).")
    key = key or resolve_markers_key(adata.uns, groupby)
    if key is None or key not in adata.uns:
        return {}
    rgg = adata.uns[key]
    try:
        names = rgg["names"]
        groups = [str(g) for g in names.dtype.names]
    except (KeyError, TypeError, AttributeError):
        return {}

    pts, pts_rest = _stored(rgg, "pts"), _stored(rgg, "pts_rest")
    if pts is None or (order == "pts_diff" and pts_rest is None):
        pts, pts_rest = _fractions(adata, groupby, groups) or (None, None)

    has_lfc = "logfoldchanges" in rgg and rgg["logfoldchanges"] is not None
    out: dict[str, list[str]] = {}
    for group in groups:
        genes = np.asarray(names[group]).astype(str)
        keep = np.ones(genes.size, dtype=bool)
        if exclude_ribo_mito:
            keep &= np.fromiter((not is_ribo_mito(g) for g in genes), dtype=bool, count=genes.size)
        frac = _lookup(pts, group, genes)
        if frac is not None:
            keep &= frac >= min_pts  # NaN compares False
        lfc = (np.asarray(rgg["logfoldchanges"][group], dtype=float)[: genes.size]
               if has_lfc else None)
        if lfc is not None:
            keep &= lfc >= min_logfc
        idx = np.flatnonzero(keep)  # stored (score) order
        if order == "logfc" and lfc is not None:
            idx = idx[np.argsort(-lfc[idx], kind="stable")]
        elif order == "pts_diff" and frac is not None:
            rest = _lookup(pts_rest, group, genes)
            if rest is not None:
                idx = idx[np.argsort(-(frac[idx] - rest[idx]), kind="stable")]
        out[group] = genes[idx[:n]].tolist()
    return out


def ora_query_genes(
    adata, column: str, group: str, n: int, *, exclude_ribo_mito: bool = True,
    key: str | None = None,
) -> list[str]:
    """The over-representation query for one group: ``group_markers``' choice of up
    to ``n`` genes (expressed in >= 20% of the group, up by fold change), not the
    stored score order. Fewer than ``n`` when fewer genes pass; empty when none do.
    With no ``column`` the groupby is read from the stored result's params."""
    groupby = column
    if not groupby:
        k = key or resolve_markers_key(adata.uns, "")
        try:
            groupby = str(adata.uns[k]["params"]["groupby"]) if k else ""
        except (KeyError, TypeError):
            groupby = ""
    return group_markers(
        adata, groupby, n=n, key=key, exclude_ribo_mito=exclude_ribo_mito,
    ).get(str(group), [])


def _stored(rgg, field: str):
    try:
        return rgg[field]
    except (KeyError, TypeError):
        return None


def _lookup(table, group: str, genes: np.ndarray) -> np.ndarray | None:
    """``table``'s values for ``genes`` in ``group``, aligned with ``genes``.

    A DataFrame (scanpy >= 1.8) is indexed by gene; a record array (older files)
    is aligned with the ranked names by position."""
    if table is None:
        return None
    if isinstance(table, pd.DataFrame):
        cols = {str(c): c for c in table.columns}
        if group not in cols:
            return None
        col = pd.Series(table[cols[group]].to_numpy(dtype=float),
                        index=table.index.astype(str))
        return col.reindex(genes).to_numpy()
    try:
        values = np.asarray(table[group], dtype=float)[: genes.size]
    except (KeyError, ValueError, TypeError, IndexError):
        return None
    return values if values.size == genes.size else None


def _fractions(adata, groupby: str, groups: list[str]) -> tuple[pd.DataFrame, pd.DataFrame] | None:
    """Fraction of cells expressing each gene in each group and in the rest of the
    cells, from the log-normalised matrix, as scanpy's pts/pts_rest. Sparse- and
    dense-safe, counted in blocks of cells. None when it can't be counted."""
    if groupby not in adata.obs.columns:
        return None
    from scview.core.activity import lognorm_matrix

    try:
        M, var_names, _ = lognorm_matrix(adata)
    except ValueError as exc:
        logger.warning("marker selection: no fraction expressing for %s: %s", groupby, exc)
        return None
    labels = adata.obs[groupby].astype(str).to_numpy()
    codes = np.full(labels.size, -1)
    for i, g in enumerate(groups):
        codes[labels == g] = i
    # One-hot cells -> groups; cells outside every group count only towards "all".
    onehot = sp.csr_matrix(
        (np.ones(int((codes >= 0).sum()), dtype=np.float32),
         (codes[codes >= 0], np.flatnonzero(codes >= 0))),
        shape=(len(groups), labels.size),
    )
    counts = np.zeros((len(groups), M.shape[1]))
    total = np.zeros(M.shape[1])
    for start in range(0, M.shape[0], _CHUNK):
        block = M[start:start + _CHUNK]
        if sp.issparse(block):
            block = sp.csr_matrix(block)
            nz = sp.csr_matrix((block.data != 0, block.indices, block.indptr),
                               shape=block.shape, dtype=np.float32)
        else:
            nz = (np.asarray(block) != 0).astype(np.float32)
        part = onehot[:, start:start + _CHUNK] @ nz
        counts += part.toarray() if sp.issparse(part) else np.asarray(part)
        total += np.asarray(nz.sum(axis=0)).ravel()
    size = np.bincount(codes[codes >= 0], minlength=len(groups)).astype(float)[:, None]
    rest = labels.size - size
    with np.errstate(divide="ignore", invalid="ignore"):
        pts = np.where(size > 0, counts / size, np.nan)
        pts_rest = np.where(rest > 0, (total[None, :] - counts) / rest, np.nan)
    index = pd.Index([str(v) for v in var_names])
    return (pd.DataFrame(pts.T, index=index, columns=groups),
            pd.DataFrame(pts_rest.T, index=index, columns=groups))
