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


class DECompareRequest(BaseModel):
    """Group A vs group B of one obs column, optionally within one subgroup —
    e.g. dataset=STIM vs dataset=CTRL within cell_type=CD14 Mono."""

    split_by: str
    group_a: str
    group_b: str
    within_column: str | None = None
    within_value: str | None = None


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


def _wilcoxon(work, mask_a, mask_b, dataset_id: str) -> list[DEGene]:
    """Wilcoxon DE of the ``mask_a`` cells against the ``mask_b`` cells of ``work``
    (an in-memory AnnData the caller may modify), over all genes."""
    import numpy as np
    import pandas as pd
    import scanpy as sc

    keep = mask_a | mask_b
    if not keep.all():
        work = work[keep].copy()
        mask_a = mask_a[keep]
    work.obs["_de_group"] = pd.Categorical(
        np.where(mask_a, "selection", "rest"), categories=["rest", "selection"]
    )
    try:
        sc.tl.rank_genes_groups(
            work,
            groupby="_de_group",
            groups=["selection"],
            reference="rest",
            method="wilcoxon",
            # Every tested gene: the test runs on .raw when present, which can hold
            # more genes than X (an HVG subset); capping at X's width kept only the
            # top-scoring, i.e. up-regulated, genes.
            n_genes=work.raw.n_vars if work.raw is not None else work.n_vars,
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
    return genes


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

    mask = np.zeros(n, dtype=bool)
    mask[idx] = True
    work = adata.to_memory() if getattr(adata, "isbacked", False) else adata.copy()
    genes = _wilcoxon(work, mask, ~mask, dataset_id)
    return DEResponse(n_selected=len(idx), n_rest=n - len(idx), label=body.label, genes=genes)


@router.post("/datasets/{dataset_id}/de/compare", response_model=DEResponse)
async def differential_expression_compare(
    dataset_id: str,
    body: DECompareRequest,
    dm: DatasetManager = Depends(get_dataset_manager),
) -> DEResponse:
    """Wilcoxon DE between two groups of an obs column (A vs B), optionally
    restricted to one subgroup — the cross-dataset / cross-condition comparison
    for a joint dataset. ``n_selected``/``n_rest`` are the A/B cell counts."""
    adaptor = await dm.get_or_load_dataset(dataset_id)
    if adaptor is None:
        raise HTTPException(status_code=404, detail="Dataset not found.")

    import numpy as np

    adata = adaptor.adata
    obs = adata.obs
    for col in (body.split_by, body.within_column):
        if col and col not in obs.columns:
            raise HTTPException(status_code=404, detail=f"Column '{col}' not found.")
    if body.group_a == body.group_b:
        raise HTTPException(status_code=400, detail="Pick two different groups to compare.")

    split = obs[body.split_by].astype(str).to_numpy()
    within = np.ones(adata.n_obs, dtype=bool)
    if body.within_column and body.within_value is not None:
        within = obs[body.within_column].astype(str).to_numpy() == body.within_value
    mask_a = within & (split == body.group_a)
    mask_b = within & (split == body.group_b)
    n_a, n_b = int(mask_a.sum()), int(mask_b.sum())
    if min(n_a, n_b) < _MIN_GROUP:
        scope = f" within {body.within_value}" if body.within_column else ""
        raise HTTPException(
            status_code=400,
            detail=(f"Each group needs at least {_MIN_GROUP} cells{scope} "
                    f"({body.group_a}: {n_a}, {body.group_b}: {n_b})."),
        )

    # Subset before copying: only the compared cells are materialised.
    keep = mask_a | mask_b
    sub = adata[keep]
    work = sub.to_memory() if getattr(adata, "isbacked", False) else sub.copy()
    genes = _wilcoxon(work, mask_a[keep], mask_b[keep], dataset_id)
    label = body.group_a + (f" ({body.within_value})" if body.within_column else "")
    return DEResponse(n_selected=n_a, n_rest=n_b, label=label, genes=genes)
