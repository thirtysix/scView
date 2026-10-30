"""Gene set endpoints – Phase 5.

Provides gene set collection listing, search (MSigDB and custom collections),
and per-cell scoring: ``scanpy.tl.score_genes`` (the set's mean expression minus a
binned control set), AUCell (ranks within each cell, ``core/aucell.py``), ssGSEA
(a rank-weighted walk within each cell, ``core/ssgsea.py``) or GSVA (each gene placed
in its own distribution over the cells first, ``core/gsva.py``), and a check of which
of them follows sequencing depth on the dataset.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Literal

import numpy as np
import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field

from scview.config import Settings
from scview.core import aucell, gsva, msigdb_species, ssgsea, targeted_panel, working_copy
from scview.core import large_mode
from scview.core.custom_genesets import get_custom_store, is_custom_id
from scview.core.dataset_manager import DatasetManager
from scview.core.geneset_sources import custom_collections_for_dataset
from scview.core.msigdb_loader import get_msigdb_loader
from scview.dependencies import get_dataset_manager, get_settings_dep

logger = logging.getLogger(__name__)

router = APIRouter()

# ---------------------------------------------------------------------------
# MSigDB collection catalogue — now uses subcategory-level loader
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------


ScoreMethod = Literal["score_genes", "aucell", "ssgsea", "gsva"]
METHOD_LABELS = {"score_genes": "score_genes", "aucell": "AUCell", "ssgsea": "ssGSEA", "gsva": "GSVA"}
_WALKS = {"ssgsea": ssgsea.score_set, "gsva": gsva.score_set}  # method -> scorer


class GeneSetScoreRequest(BaseModel):
    """Body for the gene-set scoring endpoint. ``max_rank``: AUCell's top genes per
    cell (default 5% of the detected genes); ignored by score_genes."""

    gene_set: list[str]
    score_name: str = "gene_set_score"
    method: ScoreMethod = "score_genes"
    max_rank: int | None = Field(default=None, ge=2)


class GeneSetScoreResponse(BaseModel):
    """Per-cell scores returned by the scoring endpoint."""

    score_name: str
    n_cells: int
    scores: list[float]
    min_score: float
    max_score: float
    genes_found: list[str]
    genes_missing: list[str]
    method: ScoreMethod = "score_genes"
    panel_caution: str | None = None  # a targeted panel biases enrichment (core/targeted_panel.py)
    # AUCell only: the top genes per cell counted, the matrix ranked, the set's genes
    # the dataset has but no cell expresses (left out, as decoupler does), and the share
    # of cells detecting fewer genes than max_rank (their lowest counted ranks are ties
    # among undetected genes).
    max_rank: int | None = None
    source: str | None = None
    genes_undetected: list[str] = []
    share_below_max_rank: float | None = None
    # ssGSEA / GSVA: how the score was computed, in one line (genes ranked, the CDF).
    # GSVA's genes_undetected: the set's genes it leaves out (the same in every cell).
    note: str | None = None


class ScoreCompareRequest(BaseModel):
    """Body for the depth check: the set, AUCell's ``max_rank``, the grouping the
    within-group correlations use (default: the active clustering), and the methods
    (default all; score_genes and AUCell always)."""

    gene_set: list[str]
    max_rank: int | None = Field(default=None, ge=2)
    groupby: str | None = None
    methods: list[ScoreMethod] | None = None


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.get("/datasets/{dataset_id}/genesets/collections")
async def list_geneset_collections(
    dataset_id: str,
    dm: DatasetManager = Depends(get_dataset_manager),
    settings: Settings = Depends(get_settings_dep),
):
    """List available gene-set collections (subcategory-level).

    MSigDB collections first (on a mouse dataset the mouse ones, then the human
    ones), then the custom ones (category "Custom"), which also say how many of
    their genes this dataset measures. ``species``: the dataset's.
    """
    adaptor = await dm.get_or_load_dataset(dataset_id)
    if adaptor is None:
        raise HTTPException(status_code=404, detail="Dataset not found.")

    species = await asyncio.to_thread(msigdb_species.dataset_species, adaptor.adata)
    loader = get_msigdb_loader(settings.MSIGDB_DIR)
    if loader is not None and species == "mouse" and settings.MSIGDB_AUTO_DOWNLOAD:
        from scview.core.msigdb_fetch import start_mouse_download

        start_mouse_download(loader.dir)  # first use: fetched in the background
    # Reads every collection on disk the first time: off the event loop.
    collections = (await asyncio.to_thread(loader.available_collections, species)
                   if loader is not None else [])
    collections += custom_collections_for_dataset(
        adaptor, get_custom_store(settings.DATA_DIR)
    )
    return {
        "dataset_id": dataset_id,
        "collections": collections,
        "species": species,
    }


@router.get("/datasets/{dataset_id}/genesets/search")
async def search_genesets(
    dataset_id: str,
    q: str = Query(default="", description="Search query for gene set name"),
    collection: str = Query(default="", description="Filter by collection id"),
    limit: int = Query(default=50, ge=1, le=200, description="Max results"),
    dm: DatasetManager = Depends(get_dataset_manager),
    settings: Settings = Depends(get_settings_dep),
):
    """Search gene-set collections (MSigDB, then custom) by name.

    An empty query lists the collection's sets. A set whose name matches the
    query exactly comes first, so looking a result term up by name finds that
    term rather than the first longer name containing it.
    """
    adaptor = await dm.get_or_load_dataset(dataset_id)
    if adaptor is None:
        raise HTTPException(status_code=404, detail="Dataset not found.")

    loader = get_msigdb_loader(settings.MSIGDB_DIR)
    store = get_custom_store(settings.DATA_DIR)
    custom = is_custom_id(collection)
    query = q.strip()
    # Across all collections, a mouse dataset's own (mouse) collections come first.
    species = None if collection else msigdb_species.dataset_species(adaptor.adata)

    results: list[dict] = []
    if query:
        exact = None
        if not custom and loader is not None:
            exact = loader.get_gene_set(query, collection=collection, species=species)
        if exact is None and (custom or not collection):
            exact = store.get_gene_set(query, collection=collection)
        if exact is not None:
            results.append(exact)
    if not custom and loader is not None:
        results += loader.search(query, collection=collection, limit=limit, species=species)
    if custom or not collection:
        results += store.search(query, collection=collection, limit=limit)

    seen: set[tuple[str, str]] = set()
    unique: list[dict] = []
    for r in results:
        key = (r["collection"], r["name"])
        if key not in seen:
            seen.add(key)
            unique.append(r)
    unique = unique[:limit]

    body = {
        "results": unique,
        "total": len(unique),
        "query": q,
        "collection": collection,
    }
    if loader is None and not custom:
        body["message"] = "MSigDB directory not configured. Set MSIGDB_DIR environment variable."
    return body


def _match(available, gene_set: list[str]) -> tuple[list[str], list[str]]:
    """(found, missing): a name matches exactly, else ignoring case, the way ORA matches
    MSigDB's upper-case human symbols (and custom sets, stored upper-case) to a mouse
    dataset's Cd3e."""
    available = set(available)
    by_upper: dict[str, str] = {}
    for g in available:
        by_upper.setdefault(g.upper(), g)
    found: list[str] = []
    missing: list[str] = []
    for g in gene_set:
        hit = g if g in available else by_upper.get(g.upper())
        if hit is None:
            missing.append(g)
        elif hit not in found:
            found.append(hit)
    return found, missing


def _none_found(gene_set: list[str]) -> HTTPException:
    return HTTPException(
        status_code=400,
        detail=f"None of the {len(gene_set)} provided genes were found in the dataset.",
    )


def _score_genes(adata, genes_found: list[str], gene_set: list[str], score_key: str):
    """``sc.tl.score_genes`` on .raw when it has the genes, else X: (scores, genes
    used, genes missing, where from). Call in a worker thread."""
    import scanpy as sc

    # A light AnnData sharing the arrays, so the score column is not written into the
    # dataset (a backed one is read in first by the endpoint, core/working_copy.py).
    work_adata = adata.to_memory() if hasattr(adata, "to_memory") else adata.copy()

    # Filter genes_found to those actually present in the expression matrix
    # that score_genes will use (raw if use_raw, else X).
    has_raw = work_adata.raw is not None
    if has_raw:
        raw_var = set(work_adata.raw.var_names.tolist())
        genes_in_raw = [g for g in genes_found if g in raw_var]
    else:
        genes_in_raw = []
    main_var = set(work_adata.var_names.tolist())
    genes_in_main = [g for g in genes_found if g in main_var]

    # Try with raw first (log-normalised), fall back to main X
    for use_raw, gene_list in [(True, genes_in_raw), (False, genes_in_main)]:
        if not gene_list or (use_raw and not has_raw):
            continue
        try:
            sc.tl.score_genes(
                work_adata,
                gene_list=gene_list,
                score_name=score_key,
                use_raw=use_raw,
            )
        except Exception as inner_err:
            logger.warning(
                "score_genes failed with use_raw=%s: %s — trying fallback",
                use_raw, inner_err,
            )
            continue
        # What was actually used
        used = {g.upper() for g in gene_list}
        missing = [g for g in gene_set if g.upper() not in used]
        scores = np.nan_to_num(work_adata.obs[score_key].values.astype(float), nan=0.0)
        return scores, gene_list, missing, ".raw" if use_raw else "X"

    raise HTTPException(
        status_code=400,
        detail=(
            f"Could not score gene set. {len(genes_found)} genes were found "
            "but scoring failed. This may happen if too few genes overlap "
            "with the expression matrix."
        ),
    )


# Why AUCell, ssGSEA, GSVA and the method comparison are not run on a large dataset.
_RANKS_EVERY_GENE = "it ranks every gene in every cell, which needs the whole matrix"


def _score_genes_large(adaptor, genes_found: list[str], gene_set: list[str]):
    """``_score_genes`` for a large dataset: scanpy's score_genes from the gene store
    (``core/large_views.py``, the same scores), on X (log-normalised)."""
    from scview.core import large_views

    store = adaptor.gene_store()
    if store is None:
        large_mode.refuse("Gene set scoring", (adaptor.large_info() or {}).get("gene_store_note")
                          or "the gene-major copy is not ready")
    scores, used, _ = large_views.score_genes(store, adaptor.adata.var_names, genes_found,
                                              means=adaptor.gene_means())
    upper = {g.upper() for g in used}
    missing = [g for g in gene_set if g.upper() not in upper]
    return np.nan_to_num(np.asarray(scores, dtype=float), nan=0.0), list(used), missing, "X"


def _aucell(adata, gene_set: list[str], max_rank: int | None):
    """(scores, the set matched, the rankings). Call in a worker thread."""
    rk = aucell.rankings(adata, max_rank)
    m = aucell.match(rk, gene_set)
    if not m.idx:
        if m.undetected:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"None of the set's genes is expressed in any cell ({len(m.undetected)} "
                    "found but never detected), so AUCell cannot rank them."
                ),
            )
        raise _none_found(gene_set)
    return aucell.score(rk, m.idx), m, rk


async def _in_thread(fn, dataset_id: str):
    try:
        # Seconds on a large dataset: in a worker thread, so the server keeps answering.
        return await asyncio.to_thread(fn)
    except HTTPException:
        raise
    except ValueError as e:  # nothing to rank, max_rank out of range, too large
        raise HTTPException(status_code=422, detail=str(e))
    except ImportError:
        raise HTTPException(
            status_code=500,
            detail="scanpy is required for gene set scoring but is not installed.",
        )
    except Exception as e:
        logger.error("Gene set scoring failed for dataset %s: %s", dataset_id, e)
        raise HTTPException(status_code=500, detail=f"Scoring failed: {e!s}")


@router.post("/datasets/{dataset_id}/genesets/score")
async def score_genesets(
    dataset_id: str,
    body: GeneSetScoreRequest,
    dm: DatasetManager = Depends(get_dataset_manager),
):
    """Per-cell gene-set scores for every cell of the dataset.

    ``method``: ``score_genes`` (the default: ``scanpy.tl.score_genes``, on .raw when
    present), ``aucell`` (``core/aucell.py``; ``max_rank`` top genes per cell),
    ``ssgsea`` (``core/ssgsea.py``) or ``gsva`` (``core/gsva.py``); the last two refuse
    (422) a dataset above ``ssgsea.MAX_VALUES`` non-zero values.
    """
    adaptor = await dm.get_or_load_dataset(dataset_id)
    if adaptor is None:
        raise HTTPException(status_code=404, detail="Dataset not found.")

    if not body.gene_set or len(body.gene_set) == 0:
        raise HTTPException(
            status_code=400,
            detail="gene_set must contain at least one gene name.",
        )

    adata = adaptor.adata

    if body.method == "aucell":
        large_mode.check(adaptor, "AUCell scoring", _RANKS_EVERY_GENE)

        def _run_aucell() -> GeneSetScoreResponse:
            scores, m, rk = _aucell(adata, body.gene_set, body.max_rank)
            return GeneSetScoreResponse(
                score_name=body.score_name,
                n_cells=int(len(scores)),
                scores=scores.tolist(),
                min_score=float(np.min(scores)),
                max_score=float(np.max(scores)),
                genes_found=m.found,
                genes_missing=m.missing,
                method="aucell",
                panel_caution=targeted_panel.enrichment_caution(adata),
                max_rank=rk.max_rank,
                source=rk.source,
                genes_undetected=m.undetected,
                share_below_max_rank=round(rk.share_below_max_rank(), 4),
            )

        return await _in_thread(_run_aucell, dataset_id)

    if body.method in _WALKS:
        label = METHOD_LABELS[body.method]
        large_mode.check(adaptor, f"{label} scoring", _RANKS_EVERY_GENE)

        def _run_walk() -> GeneSetScoreResponse:
            r = _WALKS[body.method](adata, body.gene_set)
            return GeneSetScoreResponse(
                score_name=body.score_name,
                n_cells=int(len(r.scores)),
                scores=r.scores.tolist(),
                min_score=float(np.min(r.scores)),
                max_score=float(np.max(r.scores)),
                genes_found=r.match.found,
                genes_missing=r.match.missing,
                method=body.method,
                panel_caution=targeted_panel.enrichment_caution(adata),
                source=r.source,
                genes_undetected=r.match.undetected,
                note=r.note,
            )

        return await _in_thread(_run_walk, dataset_id)

    # Determine which genes are present in the dataset.
    genes_found, _ = _match(adaptor.gene_names(), body.gene_set)
    if len(genes_found) == 0:
        raise _none_found(body.gene_set)

    large = adaptor.is_large
    if not large:  # a large dataset is scored from its gene store, never read in
        await working_copy.make_room(dm, dataset_id, adaptor)

    def _score() -> GeneSetScoreResponse:
        if large:
            scores, used, missing, source = _score_genes_large(adaptor, genes_found, body.gene_set)
        else:
            scores, used, missing, source = _score_genes(
                working_copy.in_memory(adaptor), genes_found, body.gene_set, body.score_name)
        return GeneSetScoreResponse(
            score_name=body.score_name,
            n_cells=int(len(scores)),
            scores=scores.tolist(),
            min_score=float(np.min(scores)),
            max_score=float(np.max(scores)),
            genes_found=used,
            genes_missing=missing,
            source=source,
            panel_caution=targeted_panel.enrichment_caution(adata),
        )

    return await _in_thread(_score, dataset_id)


def _group_codes(adata, column: str | None) -> np.ndarray | None:
    """Each cell's group index (-1: no label), or None without a grouping."""
    if not column or column not in adata.obs.columns:
        return None
    codes, _ = pd.factorize(adata.obs[column], use_na_sentinel=True)
    return np.asarray(codes, dtype=np.int64)


@router.post("/datasets/{dataset_id}/genesets/score/compare")
async def compare_score_methods(
    dataset_id: str,
    body: ScoreCompareRequest,
    dm: DatasetManager = Depends(get_dataset_manager),
):
    """Which scoring method follows sequencing depth on this dataset (C097).

    Scores the set with each method (score_genes, AUCell, ssGSEA, GSVA; ``methods``
    narrows the last two), then Spearman's rho of each score with the cell's total
    counts and with its detected genes: over all cells, and the median of the rhos
    within each group of the clustering (groups of 20+ cells), so cell types that
    differ in depth and in score do not pass for depth sensitivity. Also the rho
    between score_genes and AUCell (``between``), of each method with score_genes
    (``agreement``), and one sentence saying what the numbers show. ssGSEA or GSVA on
    a dataset above their size limit is left out, with the reason in ``skipped``.
    """
    from scview.core.transcriptome_size import per_cell_totals

    adaptor = await dm.get_or_load_dataset(dataset_id)
    if adaptor is None:
        raise HTTPException(status_code=404, detail="Dataset not found.")
    large_mode.check(adaptor, "Comparing scoring methods (AUCell)", _RANKS_EVERY_GENE)
    if not body.gene_set:
        raise HTTPException(status_code=400, detail="gene_set must contain at least one gene name.")
    adata = adaptor.adata
    groupby = body.groupby or adaptor.active_clustering_column()
    if body.groupby and body.groupby not in adata.obs.columns:
        raise HTTPException(status_code=404, detail=f"Column '{body.groupby}' not found.")
    active = groupby == adaptor.active_clustering_column()
    genes_found, _ = _match(adaptor.gene_names(), body.gene_set)
    if not genes_found:
        raise _none_found(body.gene_set)

    await working_copy.make_room(dm, dataset_id, adaptor)

    def _check() -> dict:
        adata = working_copy.in_memory(adaptor)
        au, m, rk = _aucell(adata, body.gene_set, body.max_rank)
        sg, _, _, _ = _score_genes(adata, genes_found, body.gene_set, "_scview_depth_check")
        scores = {"score_genes": sg, "aucell": au}
        skipped: dict[str, str] = {}
        for name, run in _WALKS.items():
            if body.methods is not None and name not in body.methods:
                continue
            try:
                scores[name] = run(adata, body.gene_set).scores
            except ValueError as e:  # too large, or no gene it can rank
                skipped[name] = str(e)
        depth: dict[str, np.ndarray] = {}
        totals = per_cell_totals(adata)
        if totals is not None:
            depth["total_counts"] = totals[0]
        depth["n_genes"] = rk.n_detected.astype(np.float64)
        codes = _group_codes(adata, groupby)
        check = aucell.depth_check(scores, depth, codes)
        check["between"] = aucell.correlations(sg, au, codes)
        metric = aucell.strongest_metric(check)
        where = "within clusters" if active else f"within {groupby} groups"
        n_groups = check["methods"]["aucell"][metric]["n_groups"]
        return {
            "n_cells": int(rk.n_cells),
            "groupby": groupby if n_groups else None,
            "n_groups": n_groups,
            "min_group_cells": aucell.MIN_GROUP_CELLS,
            "depth": {
                "total_counts": totals[1] if totals is not None else None,
                "n_genes": f"genes detected per cell in the {rk.source}",
            },
            "metric": metric,
            "methods": check["methods"],
            "between": check["between"],
            "agreement": {k: aucell.correlations(sg, v, codes) for k, v in scores.items() if k != "score_genes"},
            "skipped": skipped,
            "max_rank": rk.max_rank,
            "share_below_max_rank": round(rk.share_below_max_rank(), 4),
            "genes_found": m.found,
            "genes_missing": m.missing,
            "sentence": aucell.sentence(check, {k: METHOD_LABELS[k] for k in scores}, metric, where),
        }

    return await _in_thread(_check, dataset_id)
