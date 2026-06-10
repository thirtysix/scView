"""Differential expression for an arbitrary cell selection (lasso/cluster) vs the rest.

Powers the volcano plot: the user lassoes (or clicks) a set of cells, and we run a
one-vs-rest Wilcoxon test over all genes, returning log fold-change and adjusted
p-values per gene. Unlike the marker endpoint (which groups by an obs column), the
group here is an ad-hoc list of cell indices.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from scview.dependencies import get_dataset_manager
from scview.core.dataset_manager import DatasetManager

logger = logging.getLogger(__name__)

router = APIRouter()

# Below this many cells on either side, a Wilcoxon test isn't meaningful.
_MIN_GROUP = 3


class DERequest(BaseModel):
    indices: list[int]  # cell indices forming the "selection" group
    label: str = "selection"  # display label for the selection group


class DEGene(BaseModel):
    gene: str
    logfoldchange: float
    pval: float
    pval_adj: float


class DEResponse(BaseModel):
    n_selected: int
    n_rest: int
    label: str
    genes: list[DEGene]


@router.post("/datasets/{dataset_id}/de", response_model=DEResponse)
async def differential_expression(
    dataset_id: str,
    body: DERequest,
    dm: DatasetManager = Depends(get_dataset_manager),
) -> DEResponse:
    """One-vs-rest Wilcoxon DE for the selected cells, over all genes (for a volcano)."""
    adaptor = await dm.get_or_load_dataset(dataset_id)
    if adaptor is None:
        raise HTTPException(status_code=404, detail="Dataset not found.")

    import numpy as np
    import pandas as pd
    import scanpy as sc

    adata = adaptor.adata
    n = int(adata.n_obs)

    idx = sorted({int(i) for i in body.indices if 0 <= int(i) < n})
    if len(idx) < _MIN_GROUP:
        raise HTTPException(
            status_code=400,
            detail=f"Select at least {_MIN_GROUP} cells (got {len(idx)}).",
        )
    if n - len(idx) < _MIN_GROUP:
        raise HTTPException(
            status_code=400,
            detail="The selection covers (almost) all cells; nothing to compare against.",
        )

    work = adata.to_memory() if getattr(adata, "isbacked", False) else adata.copy()

    mask = np.zeros(n, dtype=bool)
    mask[idx] = True
    work.obs["_de_group"] = pd.Categorical(
        np.where(mask, "selection", "rest"), categories=["rest", "selection"]
    )

    try:
        sc.tl.rank_genes_groups(
            work,
            groupby="_de_group",
            groups=["selection"],
            reference="rest",
            method="wilcoxon",
            n_genes=work.n_vars,
        )
        rgg = work.uns["rank_genes_groups"]
        names = rgg["names"]["selection"]
        lfc = rgg["logfoldchanges"]["selection"]
        pv = rgg["pvals"]["selection"]
        pv_adj = rgg["pvals_adj"]["selection"]
    except Exception as exc:
        logger.error("DE computation failed for %s: %s", dataset_id, exc)
        raise HTTPException(status_code=500, detail=f"DE computation failed: {exc!s}")

    genes: list[DEGene] = []
    for i in range(len(names)):
        lf = float(lfc[i])
        p = float(pv[i])
        pa = float(pv_adj[i])
        # Drop non-finite rows (a gene with zero variance yields NaNs).
        if not (np.isfinite(lf) and np.isfinite(p) and np.isfinite(pa)):
            continue
        genes.append(DEGene(gene=str(names[i]), logfoldchange=lf, pval=p, pval_adj=pa))

    return DEResponse(n_selected=len(idx), n_rest=n - len(idx), label=body.label, genes=genes)
