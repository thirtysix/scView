"""PCA diagnostics: how much each component explains, where the scree curve
bends, which genes load on each component, and how many components the
neighbour graph was built from.

Everything comes from what scanpy stored: ``uns["pca"]`` (variance,
variance_ratio, params), ``varm["PCs"]`` (genes x PCs; a gene outside the
variable set has an all-zero row when ``use_highly_variable`` was on) and
``uns["neighbors"]["params"]``. The expression matrix is never read, so dense
and sparse datasets cost the same.

Two numbers summarise the curve, each with one documented rule:

- the **elbow** (Kneedle, Satopaa et al. 2011): on the scree curve (variance
  ratio against PC index, both axes scaled to 0..1) the PC farthest below the
  straight line from the first to the last PC;
- **PCs to 90%**: the smallest number of leading PCs whose variance sums to
  90% of the variance the stored PCs retain together (not of the total
  variance: with 50 of thousands of possible PCs the total is never reached).

Neither is a rule to follow blindly: scanpy's default of all 50 PCs for the
neighbour graph is usually fine, and a late PC can still carry a rare cell
type. The loadings say what a component is: a PC whose top genes are
ribosomal, mitochondrial or cell-cycle genes carries a technical or
cell-state signal rather than a cell-type one.
"""

from __future__ import annotations

import logging

import numpy as np
import scipy.sparse as sp

logger = logging.getLogger(__name__)

TOP_K = 15  # genes per sign per PC
SHARE = 0.90  # the "PCs to 90%" rule

METHOD = (
    "variance_ratio: each PC's share of the total variance of the data PCA was run on "
    "(scanpy uns['pca']['variance_ratio']); cumulative: running sum of those shares. "
    "elbow: Kneedle, the PC on the scree curve (variance ratio vs PC index, both axes "
    "scaled to 0..1) farthest below the chord from the first to the last PC. "
    "pcs_90: the smallest number of leading PCs whose variance reaches 90% of the "
    "variance all stored PCs retain together. Loadings: varm['PCs'], the top genes by "
    "signed loading on each PC; genes with a zero loading (outside the variable set) "
    "are left out."
)


def kneedle_elbow(values: np.ndarray) -> int:
    """1-based index of the elbow of a decreasing curve: the point farthest below the
    chord from its first to its last point, both axes scaled to 0..1 (Kneedle). Two
    points or fewer, or a flat curve: the last point."""
    y = np.asarray(values, dtype=np.float64)
    n = len(y)
    if n < 3 or not np.isfinite(y).all() or y.max() == y.min():
        return max(n, 1)
    x = np.linspace(0.0, 1.0, n)
    yn = (y - y.min()) / (y.max() - y.min())
    # Chord from (0, yn[0]) to (1, yn[-1]); signed distance below it.
    chord = yn[0] + (yn[-1] - yn[0]) * x
    below = chord - yn
    return int(np.argmax(below)) + 1


def pcs_to_share(variance_ratio: np.ndarray, share: float = SHARE) -> int:
    """Smallest number of leading PCs whose variance sums to ``share`` of the stored
    PCs' summed variance."""
    v = np.asarray(variance_ratio, dtype=np.float64)
    total = float(v.sum())
    if total <= 0:
        return len(v)
    cum = np.cumsum(v) / total
    return int(np.searchsorted(cum, share - 1e-12)) + 1


def _loadings_matrix(adata):
    if "PCs" not in adata.varm:
        return None
    m = adata.varm["PCs"]
    if sp.issparse(m):
        m = m.toarray()  # genes x PCs: small
    return np.asarray(m, dtype=np.float64)


def top_loadings(pcs: np.ndarray, genes: list[str], k: int = TOP_K) -> list[dict]:
    """Per PC, the ``k`` largest positive and ``k`` most negative loadings (gene,
    loading), zero loadings excluded."""
    out = []
    for j in range(pcs.shape[1]):
        col = pcs[:, j]
        nz = np.flatnonzero(col != 0)
        order = nz[np.argsort(col[nz])]
        neg = [i for i in order[:k] if col[i] < 0]
        pos = [i for i in order[::-1][:k] if col[i] > 0]
        out.append({
            "pc": j + 1,
            "positive": [{"gene": genes[i], "loading": float(col[i])} for i in pos],
            "negative": [{"gene": genes[i], "loading": float(col[i])} for i in neg],
        })
    return out


def _neighbors(adata, n_pcs: int) -> dict | None:
    """How many PCs the neighbour graph took, and from which representation.
    scanpy records ``n_pcs`` only when given; otherwise it used every column of
    ``use_rep`` (default X_pca)."""
    params = adata.uns.get("neighbors", {}).get("params") if "neighbors" in adata.uns else None
    if not isinstance(params, dict):
        return None
    use_rep = params.get("use_rep")
    rep = str(use_rep) if use_rep else "X_pca"
    n = params.get("n_pcs")
    if n is None and rep in adata.obsm:
        n = int(adata.obsm[rep].shape[1])
    elif n is not None:
        n = int(n)
    return {
        "use_rep": rep,
        "n_pcs": n,
        "n_neighbors": int(params["n_neighbors"]) if "n_neighbors" in params else None,
        # X_pca_harmony etc. has the same columns as X_pca, so the count still applies.
        "from_pca": rep == "X_pca" or rep.startswith("X_pca_"),
    }


def pca_diagnostics(adata, k: int = TOP_K) -> dict:
    """The diagnostics for one dataset (see the module docstring). Without a PCA:
    ``{"available": False, "reason": ...}``."""
    if "X_pca" not in adata.obsm:
        return {"available": False, "reason": "No PCA yet: run the pipeline's PCA step first."}
    X_pca = adata.obsm["X_pca"]
    n_cells, n_pcs = int(X_pca.shape[0]), int(X_pca.shape[1])
    uns = adata.uns.get("pca") if "pca" in adata.uns else None
    uns = uns if isinstance(uns, dict) else {}
    params = uns.get("params") if isinstance(uns.get("params"), dict) else {}

    ratio_basis = "total"
    if "variance_ratio" in uns and len(uns["variance_ratio"]) == n_pcs:
        variance_ratio = np.asarray(uns["variance_ratio"], dtype=np.float64)
        variance = (np.asarray(uns["variance"], dtype=np.float64)
                    if "variance" in uns and len(uns["variance"]) == n_pcs
                    else np.var(np.asarray(X_pca, dtype=np.float64), axis=0, ddof=1))
    else:
        # A PCA from elsewhere (a converted Seurat object): only the stored PCs'
        # variances are known, so the shares are of their sum, not of the total.
        variance = np.var(np.asarray(X_pca, dtype=np.float64), axis=0, ddof=1)
        variance_ratio = variance / variance.sum() if variance.sum() > 0 else variance
        ratio_basis = "retained"
    cumulative = np.cumsum(variance_ratio)

    pcs = _loadings_matrix(adata)
    genes = [str(g) for g in adata.var_names]
    loadings: list[dict] = []
    n_genes_used = None
    if pcs is not None and pcs.shape[0] == adata.n_vars and pcs.shape[1] >= n_pcs:
        pcs = pcs[:, :n_pcs]
        n_genes_used = int(np.count_nonzero(np.any(pcs != 0, axis=1)))
        loadings = top_loadings(pcs, genes, k)
    mask_var = params.get("mask_var")
    if n_genes_used is None and isinstance(mask_var, str) and mask_var in adata.var:
        n_genes_used = int(np.asarray(adata.var[mask_var]).astype(bool).sum())

    method = METHOD
    if ratio_basis == "retained":
        method = method.replace(
            "each PC's share of the total variance of the data PCA was run on "
            "(scanpy uns['pca']['variance_ratio'])",
            "no stored variance ratios: each PC's share of the stored PCs' summed variance "
            "(from X_pca), so the cumulative curve ends at 1")
    return {
        "available": True,
        "n_cells": n_cells,
        "n_pcs": n_pcs,
        "n_genes_total": int(adata.n_vars),
        "n_genes_used": n_genes_used,
        "params": {
            "n_comps": n_pcs,
            "use_highly_variable": _bool_or_none(params.get("use_highly_variable")),
            "mask_var": mask_var if isinstance(mask_var, str) else None,
            "zero_center": _bool_or_none(params.get("zero_center")),
        },
        "variance": [float(v) for v in variance],
        "variance_ratio": [float(v) for v in variance_ratio],
        "cumulative": [float(v) for v in cumulative],
        "ratio_basis": ratio_basis,
        "retained": float(variance_ratio.sum()),
        "elbow": kneedle_elbow(variance_ratio),
        "pcs_90": pcs_to_share(variance_ratio),
        "neighbors": _neighbors(adata, n_pcs),
        "loadings": loadings,
        "method": method,
    }


def _bool_or_none(v) -> bool | None:
    if v is None:
        return None
    try:
        return bool(v)
    except Exception:  # pragma: no cover - odd uns values
        return None


def grounding_lines(adata) -> tuple[list[str], str | None]:
    """For the co-pilot: the first PCs' variance, the elbow and PC1's top genes."""
    try:
        d = pca_diagnostics(adata, k=6)
    except Exception as exc:  # pragma: no cover - defensive
        logger.debug("pca grounding failed: %s", exc)
        return [], None
    if not d["available"]:
        return [], None
    vr = d["variance_ratio"]
    first = "; ".join(f"PC{i + 1} {100 * v:.1f}%" for i, v in enumerate(vr[:3]))
    basis = "the total variance" if d["ratio_basis"] == "total" else "the stored PCs' variance"
    detail = (f"{d['n_pcs']} PCs: {first} of {basis}; all {d['n_pcs']} together {100 * d['retained']:.1f}%; "
              f"elbow at PC{d['elbow']} (Kneedle); {d['pcs_90']} PCs reach 90% of the retained variance")
    nb = d["neighbors"]
    if nb and nb["n_pcs"]:
        detail += f"; the neighbour graph used {nb['n_pcs']} PCs of {nb['use_rep']}"
    lines = ["\n## PCA (stored components)", f"- {detail} [result:pca]."]
    if d["loadings"]:
        pc1 = d["loadings"][0]
        pos = ", ".join(g["gene"] for g in pc1["positive"])
        neg = ", ".join(g["gene"] for g in pc1["negative"])
        lines.append(f"- PC1 top loadings: positive {pos or 'none'}; negative {neg or 'none'}")
    lines.append("- Caveat: the elbow and the 90% count are rules of thumb; a late PC can still carry "
                 "a rare cell type, and a PC led by ribosomal, mitochondrial or cell-cycle genes is a "
                 "technical or cell-state axis.")
    return lines, detail
