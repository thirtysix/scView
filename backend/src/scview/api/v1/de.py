"""Differential expression for an arbitrary cell selection (lasso/cluster) vs the rest.

Powers the volcano plot: the user lassoes (or clicks) a set of cells, and we run a
one-vs-rest cell-level test (Wilcoxon by default; Welch t-test or logistic
regression, ``core/de_methods.py``) over the genes detected in enough cells
(``core/de_prefilter.py``), returning log fold-change and adjusted p-values per
gene. Unlike the marker endpoint (which groups by an obs column), the
group here is an ad-hoc list of cell indices. Between-groups comparisons can
instead run donor-aware pseudobulk DESeq2 (``core/pseudobulk.py``). The
``/methods`` endpoints run one comparison under every cell-level test and say
how far they agree. A between-groups comparison also reports the two sides'
composition (``core/de_composition.py``), before (``GET /de/composition``)
and with the result.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Literal

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field

from scview.dependencies import get_dataset_manager
from scview.core import de_composition, de_methods, de_prefilter, de_results, transcriptome_size
from scview.core.confounding import for_comparison, recorded_batch_key
from scview.core.replicates import donor_candidates, pick_donor
from scview.core.dataset_manager import DatasetManager

logger = logging.getLogger(__name__)

router = APIRouter()

# Below this many cells on either side, a cell-level test isn't meaningful.
_MIN_GROUP = 3


# Test only genes detected in at least this fraction of the cells of either
# group (core/de_prefilter.py); 0 tests every gene.
_MinPct = Field(default=de_prefilter.DEFAULT_MIN_PCT, ge=0.0, le=1.0)
# The cell-level tests (core/de_methods.py): "t-test" is Welch's.
CellMethod = Literal["wilcoxon", "t-test", "logreg"]
_TopN = Field(default=de_methods.TOP_N, ge=5, le=de_methods.MAX_TOP_N)


class DERequest(BaseModel):
    indices: list[int]  # cell indices forming the "selection" group
    label: str = "selection"  # display label for the selection group
    min_pct: float = _MinPct
    method: CellMethod = "wilcoxon"


class DEMethodsRequest(BaseModel):
    """Selection vs rest under every cell-level test (``/de/methods``)."""

    indices: list[int]
    label: str = "selection"
    min_pct: float = _MinPct
    top_n: int = _TopN


class _CompareScope(BaseModel):
    """Group A vs group B of one obs column, optionally within one subgroup —
    e.g. dataset=STIM vs dataset=CTRL within cell_type=CD14 Mono."""

    split_by: str
    group_a: str
    group_b: str
    within_column: str | None = None
    within_value: str | None = None
    replicate_key: str | None = None  # donor/sample obs column; overrides detection


class DECompareRequest(_CompareScope):
    # "pseudobulk": per-donor count sums tested with DESeq2 (needs replicate_key
    # or a detectable donor column, and raw counts); the others are cell-level.
    method: Literal["wilcoxon", "t-test", "logreg", "pseudobulk"] = "wilcoxon"
    # The cell-level test only: pseudobulk filters genes on their summed counts.
    min_pct: float = _MinPct
    # The label column the composition check uses (core/de_composition.py);
    # None: the active clustering. Never changes the test.
    composition_by: str | None = None


class DECompareMethodsRequest(_CompareScope):
    """A vs B under every cell-level test (``/de/compare/methods``)."""

    min_pct: float = _MinPct
    top_n: int = _TopN


class DEGene(BaseModel):
    gene: str
    logfoldchange: float
    # None for logistic regression, which gives no p-value (core/de_methods.py).
    pval: float | None
    pval_adj: float | None
    # The test statistic: Wilcoxon z-score, Welch t, the logistic regression
    # coefficient, or DESeq2's Wald statistic for pseudobulk. TF and pathway
    # activity is scored on it, GSEA ranks a cell-level result by it (p-values
    # saturate at 0 for strong genes).
    stat: float | None = None


class TranscriptomeSize(BaseModel):
    """Median raw total counts per cell of the two sides (core/transcriptome_size.py):
    ``caution`` when one is ``threshold``-fold the other or more, since
    library-size normalisation then shifts fold changes of broadly expressed genes."""

    source: str  # the obs column, or where the counts were summed from
    median_a: float
    median_b: float
    ratio: float | None  # larger over smaller; None when a median is 0
    caution: bool
    threshold: float
    # A seeded sample of each side's per-cell totals, for a violin.
    values_a: list[float]
    values_b: list[float]


class DECompositionGroup(BaseModel):
    group: str
    n_a: int
    n_b: int
    share_a: float  # of side A's labelled compared cells
    share_b: float
    diff_points: float  # (share_a - share_b) x 100
    flagged: bool  # past threshold_points with min_cells on a side
    one_sided: bool  # nearly absent from one side: maybe a state the comparison creates


class DEComposition(BaseModel):
    """Each side's share of every group of ``column`` among the compared cells
    (core/de_composition.py): ``warn`` when a group's shares differ by more than
    ``threshold_points``, since the DE then partly measures composition."""

    column: str | None  # None: not checked (``note`` says why)
    label_a: str
    label_b: str
    auto: bool  # the column was chosen automatically (active clustering)
    n_a: int
    n_b: int
    groups: list[DECompositionGroup]  # largest difference first, at most MAX_GROUPS
    n_groups: int
    warn: bool
    message: str | None
    note: str | None
    threshold_points: float
    min_cells: int
    n_missing: int = 0
    nested: bool = False  # the sides are different groups of ``column``


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
    # The test: wilcoxon | t-test | logreg (cell-level) | pseudobulk.
    method: str = "wilcoxon"
    # Pseudobulk only (method="pseudobulk"); additive, all optional.
    n_samples_a: int | None = None
    n_samples_b: int | None = None
    n_genes_tested: int | None = None
    design: str | None = None
    counts_source: str | None = None
    dropped_samples: list[str] = []
    # /de/compare: the two sides are also different batches (core/confounding.py).
    confounding: dict | None = None
    # /de/compare: the two sides' cluster/state composition (core/de_composition.py).
    composition: DEComposition | None = None
    # Names this result for a follow-up such as GSEA (core/de_results.py), which
    # could not take every gene back within the request size cap.
    result_id: str | None = None
    # Wilcoxon: the detection cut-off applied (None when there is none: scaled
    # expression, or pseudobulk), the genes it left out (n_genes_tested above
    # counts the rest), and every gene's larger detection fraction of the two
    # sides as a histogram (core/de_prefilter.py HIST_BINS).
    min_pct: float | None = None
    n_genes_skipped: int | None = None
    detection: list[int] | None = None
    # The two sides' RNA content; None when the dataset holds no raw counts.
    transcriptome_size: TranscriptomeSize | None = None


class DEMethodSummary(BaseModel):
    method: str
    label: str  # e.g. "Welch t-test"
    stat_label: str  # what its statistic is: z-score, t statistic, coefficient
    has_pvalues: bool
    n_significant: int | None  # adjusted p <= 0.05; None without p-values
    seconds: float


class DEMethodPair(BaseModel):
    """Two tests' agreement: Jaccard of their top-N genes higher in A (``up``)
    and higher in B (``down``), and the Spearman correlation of the statistics."""

    a: str
    b: str
    jaccard_up: float | None
    shared_up: int
    jaccard_down: float | None
    shared_down: int
    spearman: float | None


class DEMethodUnique(BaseModel):
    """A gene in one test's top N on a side and no other test's: each test's
    rank of it on that side (1 = top)."""

    gene: str
    method: str
    side: Literal["up", "down"]
    ranks: dict[str, int | None]


class DEMethodsResponse(BaseModel):
    n_selected: int
    n_rest: int
    label: str
    top_n: int
    methods: list[DEMethodSummary]
    pairs: list[DEMethodPair]
    unique: list[DEMethodUnique]
    # Tests asked about that cannot run here, with the reason (MAST: R only).
    unavailable: dict[str, str]
    # The prefilter every test ran behind (as in DEResponse).
    min_pct: float | None = None
    n_genes_tested: int
    n_genes_skipped: int | None = None


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


@dataclass
class _CellTest:
    genes: list[DEGene]
    prefilter: de_prefilter.Prefilter
    method: str = "wilcoxon"

    def fields(self) -> dict:
        """The response's prefilter fields."""
        pre = self.prefilter
        return {"min_pct": pre.min_pct, "n_genes_tested": pre.n_tested,
                "n_genes_skipped": pre.n_skipped, "detection": pre.histogram or None,
                "method": self.method}


def _prepare(work, mask_a, mask_b, min_pct: float = 0.0):
    """The AnnData a cell-level test runs on: the ``mask_a`` cells (group
    "selection") and the ``mask_b`` cells ("rest") of ``work`` (an in-memory
    AnnData the caller may modify), log-normalised, over the genes detected in
    at least ``min_pct`` of either side's cells (every gene at 0); and that
    prefilter."""
    import anndata as ad
    import numpy as np
    import pandas as pd
    import scanpy as sc

    from scview.core.pipeline import looks_like_counts

    keep = mask_a | mask_b
    if not keep.all():
        work = work[keep].copy()
        mask_a = mask_a[keep]
    # Log-normalised expression, never counts: an unprocessed upload can hold counts
    # in .raw beside a log-normalised X, or counts alone, and the fold change is
    # taken of whatever is tested (Kang: ISG15 log2FC 184 on counts).
    use_raw = work.raw is not None and not looks_like_counts(work.raw.X)
    if not use_raw and looks_like_counts(work.X):
        sc.pp.normalize_total(work, target_sum=1e4)
        sc.pp.log1p(work)
    # Every gene of the tested matrix: .raw can hold more genes than X (an HVG
    # subset), and testing only X's width kept the up-regulated genes alone.
    M = work.raw.X if use_raw else work.X
    names = work.raw.var_names if use_raw else work.var_names
    pre = de_prefilter.prefilter(M, mask_a, min_pct)
    if not pre.keep.any():
        raise HTTPException(
            status_code=400,
            detail=(f"No gene is detected in {min_pct * 100:g}% or more of the cells of "
                    "either group; lower the minimum detection."),
        )
    test = ad.AnnData(
        X=M if pre.keep.all() else M[:, pre.keep],
        obs=pd.DataFrame(
            {"_de_group": pd.Categorical(np.where(mask_a, "selection", "rest"),
                                         categories=["rest", "selection"])},
            index=pd.RangeIndex(len(mask_a)).astype(str),
        ),
        var=pd.DataFrame(index=names[pre.keep]),
        # The log base scanpy undoes to take the fold change.
        uns={"log1p": work.uns["log1p"]} if "log1p" in work.uns else {},
    )
    return test, pre


def _run_method(test, method: str, dataset_id: str) -> de_methods.MethodResult:
    try:
        return de_methods.run(test, method)
    except Exception as exc:
        logger.error("DE (%s) failed for %s: %s", method, dataset_id, exc)
        raise HTTPException(status_code=500, detail=f"DE computation failed: {exc!s}")


def _de_genes(res: de_methods.MethodResult) -> list[DEGene]:
    """The response rows, highest statistic first. A gene without a finite fold
    change (or p-value, for a test that gives one) is dropped: zero variance."""
    import numpy as np

    order = np.argsort(-np.nan_to_num(res.scores, nan=-np.inf), kind="stable")
    genes: list[DEGene] = []
    for i in order:
        lf = float(res.logfoldchanges[i])
        z = float(res.scores[i])
        if res.pvals is None:
            if not (np.isfinite(lf) and np.isfinite(z)):
                continue
            p = pa = None
        else:
            p, pa = float(res.pvals[i]), float(res.pvals_adj[i])
            if not (np.isfinite(lf) and np.isfinite(p) and np.isfinite(pa)):
                continue
        genes.append(DEGene(gene=str(res.genes[i]), logfoldchange=lf, pval=p, pval_adj=pa,
                            stat=z if np.isfinite(z) else None))
    return genes


def _cell_test(work, mask_a, mask_b, dataset_id: str, min_pct: float = 0.0,
               method: str = "wilcoxon") -> _CellTest:
    """``method`` DE of the ``mask_a`` cells against the ``mask_b`` cells of
    ``work`` over the genes the prefilter keeps (``_prepare``)."""
    test, pre = _prepare(work, mask_a, mask_b, min_pct)
    return _CellTest(_de_genes(_run_method(test, method, dataset_id)), pre, method)


def _methods(work, mask_a, mask_b, dataset_id: str, min_pct: float, top_n: int) -> dict:
    """Every cell-level test on the same prefiltered genes, and their agreement."""
    test, pre = _prepare(work, mask_a, mask_b, min_pct)
    results = [_run_method(test, m, dataset_id) for m in de_methods.METHODS]
    agree = de_methods.agreement(results, top_n)
    summaries = []
    for r in results:
        sig = None
        if r.pvals_adj is not None:
            sig = int((r.pvals_adj <= 0.05).sum())
        summaries.append(DEMethodSummary(
            method=r.method, label=de_methods.LABELS[r.method],
            stat_label=de_methods.STAT_LABELS[r.method],
            has_pvalues=de_methods.HAS_PVALUES[r.method], n_significant=sig,
            seconds=round(r.seconds, 2),
        ))
    return {
        "methods": summaries, "unavailable": de_methods.UNAVAILABLE,
        "min_pct": pre.min_pct, "n_genes_tested": pre.n_tested,
        "n_genes_skipped": pre.n_skipped, **agree,
    }


def _remember(dataset_id: str, res: _CellTest) -> str:
    return de_results.remember(dataset_id, res.genes, method=res.method,
                               min_pct=res.prefilter.min_pct, n_skipped=res.prefilter.n_skipped)


def _selection_mask(adata, indices: list[int]):
    """The selected cells as a mask; 400 when either side is too small."""
    import numpy as np

    n = int(adata.n_obs)
    idx = sorted({int(i) for i in indices if 0 <= int(i) < n})
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
    return mask


@router.post("/datasets/{dataset_id}/de", response_model=DEResponse)
async def differential_expression(
    dataset_id: str,
    body: DERequest,
    dm: DatasetManager = Depends(get_dataset_manager),
) -> DEResponse:
    """One-vs-rest cell-level DE (``method``, Wilcoxon by default) for the
    selected cells, over the genes detected in at least ``min_pct`` of either
    side's cells (for a volcano)."""
    adaptor = await dm.get_or_load_dataset(dataset_id)
    if adaptor is None:
        raise HTTPException(status_code=404, detail="Dataset not found.")

    adata = adaptor.adata
    mask = _selection_mask(adata, body.indices)
    n_sel = int(mask.sum())

    def _run() -> tuple[_CellTest, dict | None]:
        work = adata.to_memory() if getattr(adata, "isbacked", False) else adata.copy()
        size = transcriptome_size.check(work, mask, ~mask)  # before X is normalised
        return _cell_test(work, mask, ~mask, dataset_id, body.min_pct, body.method), size

    # Seconds on a large dataset: in a worker thread, so the server keeps answering.
    res, size = await asyncio.to_thread(_run)
    return DEResponse(n_selected=n_sel, n_rest=int(adata.n_obs) - n_sel, label=body.label,
                      genes=res.genes, result_id=_remember(dataset_id, res),
                      transcriptome_size=size, **res.fields())


@router.post("/datasets/{dataset_id}/de/methods", response_model=DEMethodsResponse)
async def differential_expression_methods(
    dataset_id: str,
    body: DEMethodsRequest,
    dm: DatasetManager = Depends(get_dataset_manager),
) -> DEMethodsResponse:
    """The selection vs the rest under every cell-level test (Wilcoxon, Welch
    t-test, logistic regression) on the same genes, and how far they agree."""
    adaptor = await dm.get_or_load_dataset(dataset_id)
    if adaptor is None:
        raise HTTPException(status_code=404, detail="Dataset not found.")
    adata = adaptor.adata
    mask = _selection_mask(adata, body.indices)
    n_sel = int(mask.sum())

    def _run() -> dict:
        work = adata.to_memory() if getattr(adata, "isbacked", False) else adata.copy()
        return _methods(work, mask, ~mask, dataset_id, body.min_pct, body.top_n)

    out = await asyncio.to_thread(_run)
    return DEMethodsResponse(n_selected=n_sel, n_rest=int(adata.n_obs) - n_sel,
                             label=body.label, **out)


def _scope_masks(adata, body: _CompareScope, *extra: str | None):
    """Group A's and group B's cells (within the scope); 404 for an unknown
    column (``extra`` ones too), 400 when the groups are the same."""
    import numpy as np

    obs = adata.obs
    for col in (body.split_by, body.within_column, body.replicate_key, *extra):
        if col and col not in obs.columns:
            raise HTTPException(status_code=404, detail=f"Column '{col}' not found.")
    if body.group_a == body.group_b:
        raise HTTPException(status_code=400, detail="Pick two different groups to compare.")

    split = obs[body.split_by].astype(str).to_numpy()
    within = np.ones(adata.n_obs, dtype=bool)
    if body.within_column and body.within_value is not None:
        within = obs[body.within_column].astype(str).to_numpy() == body.within_value
    return within & (split == body.group_a), within & (split == body.group_b)


def _compare_masks(adata, body: _CompareScope, *extra: str | None):
    """``_scope_masks``, and 400 when a group is too small."""
    mask_a, mask_b = _scope_masks(adata, body, *extra)
    n_a, n_b = int(mask_a.sum()), int(mask_b.sum())
    if min(n_a, n_b) < _MIN_GROUP:
        scope = f" within {body.within_value}" if body.within_column else ""
        raise HTTPException(
            status_code=400,
            detail=(f"Each group needs at least {_MIN_GROUP} cells{scope} "
                    f"({body.group_a}: {n_a}, {body.group_b}: {n_b})."),
        )
    return mask_a, mask_b


def _compare_label(body: _CompareScope) -> str:
    return body.group_a + (f" ({body.within_value})" if body.within_column else "")


def _composition(adata, body: _CompareScope, mask_a, mask_b, column: str | None) -> DEComposition:
    """The two sides' composition by ``column`` (the active clustering when None)."""
    res = de_composition.for_comparison(
        adata.obs, mask_a, mask_b, body.group_a, body.group_b, split_by=body.split_by,
        column=column, default=de_composition.default_column(adata),
        within_column=body.within_column if body.within_value is not None else None,
        within_value=body.within_value,
    )
    return DEComposition(**res)


@router.get("/datasets/{dataset_id}/de/composition", response_model=DEComposition)
async def differential_expression_composition(
    dataset_id: str,
    split_by: str,
    group_a: str,
    group_b: str,
    within_column: str | None = None,
    within_value: str | None = None,
    column: str | None = Query(default=None, description="Label column; default the active clustering"),
    dm: DatasetManager = Depends(get_dataset_manager),
) -> DEComposition:
    """Each side's share of every cluster/state among the cells a ``/de/compare``
    of these groups would test, before it is run. Obs only (milliseconds), so
    on the event loop; a group with no cells gets a note, not a 400."""
    adaptor = await dm.get_or_load_dataset(dataset_id)
    if adaptor is None:
        raise HTTPException(status_code=404, detail="Dataset not found.")
    scope = _CompareScope(split_by=split_by, group_a=group_a, group_b=group_b,
                          within_column=within_column or None, within_value=within_value)
    mask_a, mask_b = _scope_masks(adaptor.adata, scope, column)
    return _composition(adaptor.adata, scope, mask_a, mask_b, column or None)


@router.post("/datasets/{dataset_id}/de/compare", response_model=DEResponse)
async def differential_expression_compare(
    dataset_id: str,
    body: DECompareRequest,
    dm: DatasetManager = Depends(get_dataset_manager),
) -> DEResponse:
    """Cell-level DE (``method``, Wilcoxon by default) or pseudobulk between two
    groups of an obs column (A vs B), optionally restricted to one subgroup —
    the cross-dataset / cross-condition comparison for a joint dataset.
    ``n_selected``/``n_rest`` are the A/B cell counts."""
    adaptor = await dm.get_or_load_dataset(dataset_id)
    if adaptor is None:
        raise HTTPException(status_code=404, detail="Dataset not found.")

    adata = adaptor.adata
    obs = adata.obs
    mask_a, mask_b = _compare_masks(adata, body, body.composition_by)
    n_a, n_b = int(mask_a.sum()), int(mask_b.sum())

    # The two sides being different batches (datasets, libraries, runs): the
    # result then mixes the batch effect with any biological one.
    # The batch-correction key counts unless it is the compared column itself:
    # Harmony changes the embedding, not the expression this test runs on.
    batch_key = recorded_batch_key(adata)
    extra = [batch_key] if batch_key and batch_key != body.split_by else []
    hit = for_comparison(obs, body.split_by, mask_a | mask_b, extra)
    confounding = hit.to_dict() if hit else None
    # The two sides' cluster mix: a shift makes the DE partly a composition test.
    composition = _composition(adata, body, mask_a, mask_b, body.composition_by)

    if body.method == "pseudobulk":
        resp = await _pseudobulk_compare(adata, body, mask_a, mask_b, dataset_id)
        resp.confounding = confounding
        resp.composition = composition
        return resp

    # Subset before copying: only the compared cells are materialised.
    keep = mask_a | mask_b

    def _run() -> tuple[_CellTest, dict | None]:
        sub = adata[keep]
        work = sub.to_memory() if getattr(adata, "isbacked", False) else sub.copy()
        size = transcriptome_size.check(work, mask_a[keep], mask_b[keep])  # before X is normalised
        res = _cell_test(work, mask_a[keep], mask_b[keep], dataset_id, body.min_pct, body.method)
        return res, size

    res, size = await asyncio.to_thread(_run)
    key = body.replicate_key or _detect_replicate_key(
        obs, {c for c in (body.split_by, body.within_column) if c}, mask_a, mask_b
    )
    guard = _replicate_guard(obs, mask_a, mask_b, key)
    return DEResponse(n_selected=n_a, n_rest=n_b, label=_compare_label(body), genes=res.genes,
                      result_id=_remember(dataset_id, res), transcriptome_size=size,
                      confounding=confounding, composition=composition, **guard, **res.fields())


@router.post("/datasets/{dataset_id}/de/compare/methods", response_model=DEMethodsResponse)
async def differential_expression_compare_methods(
    dataset_id: str,
    body: DECompareMethodsRequest,
    dm: DatasetManager = Depends(get_dataset_manager),
) -> DEMethodsResponse:
    """A vs B (as ``/de/compare``) under every cell-level test on the same
    genes, and how far they agree."""
    adaptor = await dm.get_or_load_dataset(dataset_id)
    if adaptor is None:
        raise HTTPException(status_code=404, detail="Dataset not found.")
    adata = adaptor.adata
    mask_a, mask_b = _compare_masks(adata, body)
    keep = mask_a | mask_b

    def _run() -> dict:
        sub = adata[keep]
        work = sub.to_memory() if getattr(adata, "isbacked", False) else sub.copy()
        return _methods(work, mask_a[keep], mask_b[keep], dataset_id, body.min_pct, body.top_n)

    out = await asyncio.to_thread(_run)
    return DEMethodsResponse(n_selected=int(mask_a.sum()), n_rest=int(mask_b.sum()),
                             label=_compare_label(body), **out)


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
        res = pseudobulk_de(
            sub, mask_a[keep], sub.obs[key],
            label_a=body.group_a, label_b=body.group_b, donor_key=key,
        )
        # DESeq2's size factors normalise library size too.
        return res, transcriptome_size.check(sub, mask_a[keep], mask_b[keep])

    try:
        res, size = await asyncio.to_thread(_run)
    except PseudobulkError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        logger.error("Pseudobulk DE failed for %s: %s", dataset_id, exc, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Pseudobulk DE failed: {exc!s}")

    label = body.group_a + (f" ({body.within_value})" if body.within_column else "")
    genes = [DEGene(gene=g, logfoldchange=lf, pval=p, pval_adj=pa, stat=res.stats.get(g))
             for g, lf, p, pa in res.genes]
    return DEResponse(
        n_selected=res.n_cells_a,
        n_rest=res.n_cells_b,
        label=label,
        genes=genes,
        result_id=de_results.remember(dataset_id, genes, method="pseudobulk"),
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
        transcriptome_size=size,
    )
