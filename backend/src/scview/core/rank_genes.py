"""scView's one call for Wilcoxon and t-test markers and DE: ``sc.tl.rank_genes_groups``.

``rank_genes_groups`` takes scanpy's arguments and writes the same ``uns`` entry. When
``sparse_markers`` computes the test (Wilcoxon without tie correction, ``t-test``,
``t-test_overestim_var``; each group against the rest, a subset of groups, or a
reference group) and the matrix is in memory as float32 or float64 (CSR, CSC or
dense), the matrix is read by blocks of genes (``core/matrix_genes.py``) and only its
stored values are sorted; ``mask_var`` (one boolean per gene, or a var column) cuts each
block to its genes (``GeneSubset``), with no copy of them. Anything else (logistic
regression, tie correction, a matrix on disk, integer values, stored duplicates, a group
that is not a category) runs scanpy's own, which also raises scanpy's errors.

Equal to scanpy 1.11: names in order, scores, p-values, adjusted p-values and pts bit
for bit (rank sums are exact). Means and variances, so fold changes and the t-test, are
float64 sums in cell order, the rest as all cells minus the group: bit for bit whenever
those sums are exact (float32 log values, as the pipeline stores them), else within
float precision, as scanpy's own differ between its numba thread counts (and for a CSC
matrix, whose squares scanpy takes in float64). A reference comparison is ranked within
the two groups' cells, as scanpy's ``reference=`` mode ranks them: bit for bit too.

100k neurons x 28k genes (201M values in rows of descending genes), 21 groups, 2 cores,
in memory: Wilcoxon with pts 16-17 s at +0.35 GB over the loaded data (rows shuffled
into no order: 24 s at +0.60 GB), scanpy 162-201 s at +3.2 GB (it copies each group's
rest of the matrix, then ranks dense chunks of 10M values); t-test 9.5 s against 32 s.
Every field bit for bit equal.
"""

from __future__ import annotations

import logging
from typing import Any

from scview.core import matrix_genes, sparse_markers

logger = logging.getLogger(__name__)

SPARSE = "sparse"
SCANPY = "scanpy"


def _matrix(adata: Any, use_raw: bool, layer: str | None) -> tuple[Any, Any]:
    """The matrix scanpy ranks for these arguments, and its gene names."""
    if layer is not None:
        return adata.layers[layer], adata.var_names
    if use_raw:
        return adata.raw.X, adata.raw.var_names
    return adata.X, adata.var_names


def _widen_half(adata: Any, use_raw: bool | None, layer: str | None) -> None:
    """Store a float16 ranked matrix as float32, in place: scanpy's numba kernels raise
    "float16" on it (a published h5ad saved in half precision, e.g. infercnvpy's
    maynard2020_3k), and the blocks reader takes float32/float64 only."""
    import anndata as ad
    import numpy as np

    if use_raw is None:
        use_raw = adata.raw is not None
    if use_raw and adata.raw is None or layer is not None and layer not in adata.layers:
        return  # unsupported() names the problem
    matrix, _ = _matrix(adata, use_raw, layer)
    if np.dtype(getattr(matrix, "dtype", object)) != np.float16:
        return
    wide = matrix.astype(np.float32)
    logger.info("rank_genes_groups: float16 %s stored as float32",
                f"layer {layer!r}" if layer is not None else ".raw" if use_raw else "X")
    if layer is not None:
        adata.layers[layer] = wide
    elif use_raw:
        adata.raw = ad.AnnData(X=wide, obs=adata.obs[[]], var=adata.raw.var.copy())
    else:
        adata.X = wide


def _gene_mask(adata: Any, mask_var: Any, use_raw: bool | None, layer: str | None):
    """``mask_var`` as one boolean per gene of the ranked matrix, or None: a var column
    name (of X's genes) or a boolean array, as scanpy takes it."""
    import numpy as np

    if use_raw is None:
        use_raw = adata.raw is not None
    if isinstance(mask_var, str):
        if use_raw or mask_var not in adata.var.columns:
            return None
        mask_var = adata.var[mask_var].to_numpy()
    mask = np.asarray(mask_var)
    if mask.dtype != bool:
        return None
    if (use_raw and adata.raw is None) or (layer is not None and layer not in adata.layers):
        return None
    _, names = _matrix(adata, use_raw, layer)
    return mask if mask.shape == (len(names),) else None


def unsupported(adata: Any, groupby: str, *, method: str = "wilcoxon", groups: Any = "all",
                reference: str = "rest", use_raw: bool | None = None, layer: str | None = None,
                corr_method: str = "benjamini-hochberg", n_genes: int | None = None,
                tie_correct: bool = False, mask_var: Any = None, **kwds: Any) -> str | None:
    """Why scanpy ranks this call rather than ``sparse_markers``, or None."""
    if method not in sparse_markers.METHODS:
        return f"method {method!r}"
    if tie_correct:
        return "tie correction"
    if mask_var is not None and _gene_mask(adata, mask_var, use_raw, layer) is None:
        return "a gene mask that is not one boolean per gene"
    if kwds:
        return f"arguments {sorted(kwds)}"
    if corr_method not in ("benjamini-hochberg", "bonferroni"):
        return f"correction {corr_method!r}"
    if n_genes is not None and (not isinstance(n_genes, int) or n_genes < 1):
        return f"n_genes={n_genes!r}"
    if groupby not in adata.obs.columns:
        return f"no column {groupby!r}"
    labels = adata.obs[groupby]
    if labels.dtype.name != "category":
        return f"a {labels.dtype} column"
    categories = labels.cat.categories
    if reference != "rest" and reference not in categories:
        return f"reference {reference!r} is not a group"
    if not (isinstance(groups, str) and groups == "all"):
        if isinstance(groups, (str, int)):
            return "groups not given as a sequence"
        names = [str(g) if isinstance(g, int) else g for g in groups]
        if not names or any(g not in categories for g in names):
            return "groups that are not categories"
        if reference != "rest" and set(names) <= {reference}:
            return "no group besides the reference"
    if use_raw is None:
        use_raw = adata.raw is not None
    if use_raw and adata.raw is None:
        return "use_raw without .raw"
    if layer is not None and (use_raw or layer not in adata.layers):
        return f"layer {layer!r}"
    matrix, _ = _matrix(adata, use_raw, layer)
    reason = matrix_genes.unsupported(matrix)
    return reason


def rank_genes_groups(
    adata: Any,
    groupby: str,
    *,
    method: str,
    groups: Any = "all",
    reference: str = "rest",
    pts: bool = False,
    use_raw: bool | None = None,
    layer: str | None = None,
    key_added: str | None = None,
    corr_method: str = "benjamini-hochberg",
    n_genes: int | None = None,
    rankby_abs: bool = False,
    tie_correct: bool = False,
    mask_var: Any = None,
    specificity: bool = False,
    max_entries: int | None = None,
    progress: sparse_markers.Progress | None = None,
    **kwds: Any,
) -> str:
    """``sc.tl.rank_genes_groups(adata, groupby, ...)`` with the same arguments (``method``
    named: scanpy's default is the t-test), by blocks of genes when ``sparse_markers``
    computes it (else scanpy's own); writes
    ``adata.uns[key_added or "rank_genes_groups"]``. Returns which ran: "sparse" | "scanpy".

    ``specificity``: the blocks path also stores the tau frame ``marker_specificity``
    would attach (from the group means it has); scanpy's path leaves that to the caller.
    ``max_entries``: values per block (default ``matrix_genes.block_entries``)."""
    # scanpy's first step (sanitize_anndata): string columns become categorical
    adata.strings_to_categoricals()
    _widen_half(adata, use_raw, layer)
    reason = unsupported(adata, groupby, method=method, groups=groups, reference=reference,
                         use_raw=use_raw, layer=layer, corr_method=corr_method,
                         n_genes=n_genes, tie_correct=tie_correct, mask_var=mask_var, **kwds)
    if reason is None:
        if use_raw is None:
            use_raw = adata.raw is not None
        matrix, var_names = _matrix(adata, use_raw, layer)
        source = matrix_genes.MatrixGenes(matrix, var_names)
        if max_entries is None:
            max_entries = matrix_genes.block_entries(source.nnz)
        mask = None if mask_var is None else _gene_mask(adata, mask_var, use_raw, layer)
        if mask is not None:  # the kept genes, cut from each block: no copy of them
            source = matrix_genes.GeneSubset(source, mask)
        try:
            sparse_markers.rank_genes_groups(
                adata, groupby, source, method=method,
                groups=groups, reference=reference, pts=pts, use_raw=use_raw, layer=layer,
                key_added=key_added, corr_method=corr_method, n_genes=n_genes,
                rankby_abs=rankby_abs, specificity=specificity, check_store=False,
                max_entries=max_entries, progress=progress, mask_var=mask)
            return SPARSE
        except matrix_genes.DuplicateEntries as exc:  # found while reading; nothing written
            reason = str(exc)
    import scanpy as sc

    logger.info("rank_genes_groups on '%s' by scanpy: %s", groupby, reason)
    if mask_var is not None:
        kwds["mask_var"] = mask_var
    sc.tl.rank_genes_groups(adata, groupby, method=method, groups=groups, reference=reference,
                            pts=pts, use_raw=use_raw, layer=layer, key_added=key_added,
                            corr_method=corr_method, n_genes=n_genes, rankby_abs=rankby_abs,
                            tie_correct=tie_correct, **kwds)
    return SCANPY
