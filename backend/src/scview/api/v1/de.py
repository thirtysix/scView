"""Differential expression for an arbitrary cell selection (lasso/cluster) vs the rest.

Powers the volcano plot: the user lassoes (or clicks) a set of cells, and we run a
one-vs-rest Wilcoxon test over all genes, returning log fold-change and adjusted
p-values per gene. Unlike the marker endpoint (which groups by an obs column), the
group here is an ad-hoc list of cell indices. Between-groups comparisons can
instead run donor-aware pseudobulk DESeq2 (``core/pseudobulk.py``).
"""

from __future__ import annotations

import asyncio
import logging
from typing import Literal

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from scview.dependencies import get_dataset_manager
from scview.core.confounding import for_comparison, recorded_batch_key
from scview.core.replicates import donor_candidates, pick_donor
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
    replicate_key: str | None = None  # donor/sample obs column; overrides detection
    # "pseudobulk": per-donor count sums tested with DESeq2 (needs replicate_key
    # or a detectable donor column, and raw counts).
    method: Literal["wilcoxon", "pseudobulk"] = "wilcoxon"


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
    # Replicate guard (only set by /de/compare); additive, all optional.
    replicate_key: str | None = None
    n_donors_a: int | None = None
    n_donors_b: int | None = None
    top_donor_share_a: float | None = None
    top_donor_share_b: float | None = None
    underreplicated: bool = False
    # Pseudobulk only (method="pseudobulk"); additive, all optional.
    method: str = "wilcoxon"
    n_samples_a: int | None = None
    n_samples_b: int | None = None
    n_genes_tested: int | None = None
    design: str | None = None
    counts_source: str | None = None
    dropped_samples: list[str] = []
    # /de/compare: the two sides are also different batches (core/confounding.py).
    confounding: dict | None = None


def _detect_replicate_key(obs, exclude: set[str], mask_a=None, mask_b=None) -> str | None:
    """The most donor-like obs column (not one used to split/scope); with the two
    groups' masks, one with at least 2 donors in each group (core/replicates.py)."""
    candidates = donor_candidates(obs, exclude)
    groups = [mask_a, mask_b] if mask_a is not None and mask_b is not None else []
    return pick_donor(obs, candidates, groups)


def _donor_stats(donors) -> tuple[int, float]:
    """(number of distinct donors, share of cells held by the largest donor)."""
    counts = donors.value_counts(dropna=True)
    total = int(counts.sum())
    if total == 0:
        return 0, 0.0
    return int((counts > 0).sum()), float(counts.max()) / total


def _replicate_guard(obs, mask_a, mask_b, key: str | None) -> dict:
    if not key:
        return {"replicate_key": None, "underreplicated": False}
    col = obs[key].astype(str).where(obs[key].notna())
    na, sa = _donor_stats(col[mask_a])
    nb, sb = _donor_stats(col[mask_b])
    under = na < 2 or nb < 2 or sa > 0.5 or sb > 0.5
    return {
        "replicate_key": key,
        "n_donors_a": na,
        "n_donors_b": nb,
        "top_donor_share_a": round(sa, 4),
        "top_donor_share_b": round(sb, 4),
        "underreplicated": bool(under),
    }


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

    def _run() -> list[DEGene]:
        work = adata.to_memory() if getattr(adata, "isbacked", False) else adata.copy()
        return _wilcoxon(work, mask, ~mask, dataset_id)

    # Seconds on a large dataset: in a worker thread, so the server keeps answering.
    genes = await asyncio.to_thread(_run)
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
    for col in (body.split_by, body.within_column, body.replicate_key):
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

    # The two sides being different batches (datasets, libraries, runs): the
    # result then mixes the batch effect with any biological one.
    # The batch-correction key counts unless it is the compared column itself:
    # Harmony changes the embedding, not the expression this test runs on.
    batch_key = recorded_batch_key(adata)
    extra = [batch_key] if batch_key and batch_key != body.split_by else []
    hit = for_comparison(obs, body.split_by, mask_a | mask_b, extra)
    confounding = hit.to_dict() if hit else None

    if body.method == "pseudobulk":
        resp = await _pseudobulk_compare(adata, body, mask_a, mask_b, dataset_id)
        resp.confounding = confounding
        return resp

    # Subset before copying: only the compared cells are materialised.
    keep = mask_a | mask_b

    def _run() -> list[DEGene]:
        sub = adata[keep]
        work = sub.to_memory() if getattr(adata, "isbacked", False) else sub.copy()
        return _wilcoxon(work, mask_a[keep], mask_b[keep], dataset_id)

    genes = await asyncio.to_thread(_run)
    label = body.group_a + (f" ({body.within_value})" if body.within_column else "")
    key = body.replicate_key or _detect_replicate_key(
        obs, {c for c in (body.split_by, body.within_column) if c}, mask_a, mask_b
    )
    guard = _replicate_guard(obs, mask_a, mask_b, key)
    return DEResponse(n_selected=n_a, n_rest=n_b, label=label, genes=genes,
                      confounding=confounding, **guard)


async def _pseudobulk_compare(adata, body: DECompareRequest, mask_a, mask_b, dataset_id: str):
    """Donor-aware A vs B: per-donor count sums, DESeq2 Wald test (off the event loop)."""
    from scview.core.pseudobulk import PseudobulkError, pseudobulk_de

    key = body.replicate_key or _detect_replicate_key(
        adata.obs, {c for c in (body.split_by, body.within_column) if c}, mask_a, mask_b
    )
    if not key:
        raise HTTPException(
            status_code=400,
            detail="Pseudobulk DE needs a donor or sample column; none was found, so pick one.",
        )
    if key == body.split_by:
        raise HTTPException(
            status_code=400, detail="The donor column must differ from the column being compared."
        )

    keep = mask_a | mask_b

    def _run():
        sub = adata[keep]
        if getattr(adata, "isbacked", False):
            sub = sub.to_memory()
        return pseudobulk_de(
            sub, mask_a[keep], sub.obs[key],
            label_a=body.group_a, label_b=body.group_b, donor_key=key,
        )

    try:
        res = await asyncio.to_thread(_run)
    except PseudobulkError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        logger.error("Pseudobulk DE failed for %s: %s", dataset_id, exc, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Pseudobulk DE failed: {exc!s}")

    label = body.group_a + (f" ({body.within_value})" if body.within_column else "")
    return DEResponse(
        n_selected=res.n_cells_a,
        n_rest=res.n_cells_b,
        label=label,
        genes=[DEGene(gene=g, logfoldchange=lf, pval=p, pval_adj=pa) for g, lf, p, pa in res.genes],
        replicate_key=key,
        n_donors_a=res.n_samples_a,
        n_donors_b=res.n_samples_b,
        method="pseudobulk",
        n_samples_a=res.n_samples_a,
        n_samples_b=res.n_samples_b,
        n_genes_tested=res.n_genes_tested,
        design=res.design,
        counts_source=res.counts_source,
        dropped_samples=res.dropped_samples,
    )
