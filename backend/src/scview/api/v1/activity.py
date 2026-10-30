"""TF and pathway activity: CollecTRI TFs and PROGENy pathways scored with ULM.

Per cell (to colour the embedding by one source, and a group x source heatmap of
the mean), and on a DE result's per-gene statistics (which TFs and pathways the
comparison moves). Computed on demand, never written to the dataset: one sparse
product scores every cell (0.5 s for Kang's 14k cells x 625 TFs on 2 cores), and
the per-dataset model is kept in memory for the loaded dataset object only.
See ``core/activity.py`` and ``core/activity_nets.py``.
"""

from __future__ import annotations

import asyncio
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from scview.config import Settings
from scview.core import activity, activity_nets, de_results, large_mode, orthologs, targeted_panel
from scview.core.dataset_manager import DatasetManager
from scview.core.object_cache import ObjectCache
from scview.core.species import detect_species
from scview.dependencies import get_dataset_manager, get_settings_dep

logger = logging.getLogger(__name__)

router = APIRouter()

MAX_GROUPS = 100  # as for markers
MAX_DE_GENES = 100_000

MOUSE_NOTE = (
    "Mouse dataset: the human CollecTRI and PROGENy networks were mapped to mouse through "
    "HGNC HCOP orthologs, as decoupler does; a gene with no confident ortholog is left out."
)

# (loaded dataset object, network) -> activity.CellModel. A pipeline run drops the
# object, so a model never outlives the expression values it was built from.
_MODELS = ObjectCache()


class DEActivityRequest(BaseModel):
    """A DE result's per-gene statistic (Wilcoxon z-score, Welch t, logistic
    regression coefficient or DESeq2 Wald statistic):
    by the id the DE response carried (core/de_results.py), which keeps a whole
    transcriptome out of the request, or as explicit genes and stats."""

    result_id: str | None = Field(default=None, max_length=64)
    genes: list[str] = Field(default_factory=list, max_length=MAX_DE_GENES)
    stats: list[float] = Field(default_factory=list, max_length=MAX_DE_GENES)


def _spec(network: str) -> activity_nets.NetworkSpec:
    try:
        return activity_nets.spec(network)
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e.args[0]))


def _net_dir(settings: Settings) -> Path:
    return Path(settings.DATA_DIR) / "networks"  # the data volume


def _network(network: str, dest: Path, species: str = "human") -> pd.DataFrame:
    """The network table in the dataset's species, fetched (and for mouse mapped
    through orthologs) on first use (call in a worker thread)."""
    return activity_nets.load_for(network, dest, species)


async def _dataset(dm: DatasetManager, dataset_id: str):
    adaptor = await dm.get_or_load_dataset(dataset_id)
    if adaptor is None:
        raise HTTPException(status_code=404, detail=f"Dataset '{dataset_id}' not found.")
    return adaptor.adata


def _species(adata) -> str:
    """human | mouse | unknown; mouse networks are mapped through orthologs."""
    return detect_species(adata).species


def _model(adata, network: str, dest: Path, species: str = "human") -> tuple[activity.CellModel, str]:
    """The cached ULM model for this loaded dataset, built on first use (worker thread)."""
    large_mode.check(adata, "TF and pathway activity", large_mode.WHOLE_MATRIX)
    cached = _MODELS.get(adata, network)
    if cached is not None:
        return cached
    M, genes, source = activity.lognorm_matrix(adata)
    model = activity.CellModel.build(M, genes, _network(network, dest, species))
    _MODELS.put(adata, (model, source), key=network)
    return model, source


async def _in_thread(fn, *args):
    try:
        return await asyncio.to_thread(fn, *args)
    except HTTPException:
        raise
    except activity_nets.NetworkUnavailable as e:
        raise HTTPException(status_code=503, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))


def _round(a, digits: int = 3) -> list:
    return np.round(np.asarray(a, dtype=np.float64), digits).tolist()


def _group_codes(col: pd.Series) -> tuple[list[str], np.ndarray]:
    """The groups present, in the column's own order when categorical, and each
    cell's group index (-1 for a missing label)."""
    labels = col.astype(str).where(col.notna())
    present = set(labels.dropna())
    if isinstance(col.dtype, pd.CategoricalDtype):
        order = [str(c) for c in col.cat.categories if str(c) in present]
    else:
        order = sorted(present)
    return order, pd.Index(order).get_indexer(labels)


@router.get("/activity/networks")
async def list_networks(settings: Settings = Depends(get_settings_dep)):
    """The networks scView can score, and whether each is downloaded yet."""
    out = []
    for s in activity_nets.NETWORKS.values():
        meta = activity_nets.read_meta(_net_dir(settings), s.name)
        out.append({
            "name": s.name, "label": s.label, "kind": s.kind, "citation": s.citation,
            "url": s.url, "downloaded": meta is not None,
            "n_sources": (meta or {}).get("n_sources"), "n_edges": (meta or {}).get("n_edges"),
            "fetched_at": (meta or {}).get("fetched_at"),
            "mouse": activity_nets.read_mouse_meta(_net_dir(settings), s.name),
        })
    hcop = orthologs.read_meta(_net_dir(settings))
    return {"networks": out, "orthologs": {
        "label": orthologs.LABEL, "url": orthologs.URL, "citation": orthologs.CITATION,
        "licence": orthologs.LICENCE, "downloaded": hcop is not None,
        "auto_download": settings.ORTHOLOGS_AUTO_DOWNLOAD, **({"meta": hcop} if hcop else {})}}


@router.get("/datasets/{dataset_id}/activity/{network}")
async def activity_summary(
    dataset_id: str,
    network: str,
    groupby: str = "",
    dm: DatasetManager = Depends(get_dataset_manager),
    settings: Settings = Depends(get_settings_dep),
):
    """The sources scored on this dataset and, with ``groupby``, each group's mean
    activity per source (for the heatmap)."""
    s = _spec(network)
    adata = await _dataset(dm, dataset_id)
    species = _species(adata)
    if groupby:
        if groupby not in adata.obs.columns:
            raise HTTPException(status_code=404, detail=f"Column '{groupby}' not found.")
        n_groups = adata.obs[groupby].nunique(dropna=True)
        if not 1 <= n_groups <= MAX_GROUPS:
            raise HTTPException(
                status_code=400,
                detail=f"'{groupby}' has {n_groups} groups; activity is averaged over at most {MAX_GROUPS}.",
            )

    def _run() -> dict:
        t0 = time.time()
        model, source = _model(adata, network, _net_dir(settings), species)
        net = model.net
        body = {
            "network": s.name, "label": s.label, "kind": s.kind, "citation": s.citation,
            "species": species, "matrix_source": source, "n_cells": int(adata.n_obs),
            "n_genes_measured": net.n_features, "sources": net.sources,
            "n_targets": net.n_targets.tolist(), "groupby": groupby or None,
            "panel_caution": targeted_panel.enrichment_caution(adata),
        }
        if groupby:
            groups, codes = _group_codes(adata.obs[groupby])
            means, sizes = model.group_means(codes, len(groups))
            body.update(groups=groups, group_sizes=sizes.tolist(),
                        mean=[_round(row) for row in means])
        if species == "mouse":
            body["orthologs"] = {"note": MOUSE_NOTE, **activity_nets.note_mouse_run(
                adata, network, _net_dir(settings), len(net.sources), net.n_features)}
        body["elapsed_seconds"] = round(time.time() - t0, 2)
        return body

    return await _in_thread(_run)


@router.get("/datasets/{dataset_id}/activity/{network}/cells")
async def activity_cells(
    dataset_id: str,
    network: str,
    source: str,
    dm: DatasetManager = Depends(get_dataset_manager),
    settings: Settings = Depends(get_settings_dep),
):
    """Every cell's activity for one source, in cell order (to colour the embedding)."""
    s = _spec(network)
    adata = await _dataset(dm, dataset_id)
    species = _species(adata)

    def _run() -> dict:
        model, _ = _model(adata, network, _net_dir(settings), species)
        if source not in model.net.sources:
            raise HTTPException(
                status_code=404,
                detail=f"'{source}' is not scored on this dataset ({s.label} sources need "
                f"at least {activity.TMIN} targets among its genes).",
            )
        j = model.net.sources.index(source)
        return {"network": s.name, "source": source, "n_targets": int(model.net.n_targets[j]),
                "scores": _round(model.source(source))}

    return await _in_thread(_run)


@router.post("/datasets/{dataset_id}/activity/{network}/de")
async def activity_de(
    dataset_id: str,
    network: str,
    body: DEActivityRequest,
    dm: DatasetManager = Depends(get_dataset_manager),
    settings: Settings = Depends(get_settings_dep),
):
    """ULM on a DE result's per-gene statistic: each source's activity (t), p-value
    and BH-adjusted p, strongest first."""
    s = _spec(network)
    prefilter: dict = {}
    if body.result_id:
        stored = de_results.recall(dataset_id, body.result_id)
        if stored is None:
            raise HTTPException(
                status_code=404,
                detail="That DE result is no longer kept on the server: run the DE again.",
            )
        genes, stats = stored.profile()
        prefilter["de_method"] = stored.method
        # Only the genes the DE's detection prefilter let through have a statistic.
        if stored.min_pct is not None:
            prefilter.update(de_min_pct=stored.min_pct, de_n_genes_skipped=stored.n_skipped)
    else:
        genes, stats = body.genes, body.stats
    if len(genes) != len(stats):
        raise HTTPException(status_code=400, detail="genes and stats must have the same length.")
    if not genes:
        raise HTTPException(status_code=400, detail="No genes given.")
    adata = await _dataset(dm, dataset_id)
    species = _species(adata)

    def _run() -> dict:
        res = activity.score_profile(stats, genes, _network(network, _net_dir(settings), species))
        res = res.reindex(res["score"].abs().sort_values(ascending=False).index)
        return {
            "network": s.name, "label": s.label, "kind": s.kind, "citation": s.citation,
            "n_genes": len(genes), "species": species,
            **({"orthologs_note": MOUSE_NOTE} if species == "mouse" else {}),
            **prefilter,
            "results": [
                {"source": r.source, "score": round(float(r.score), 3), "pvalue": float(r.pvalue),
                 "padj": float(r.padj), "n_targets": int(r.n_targets)}
                for r in res.itertuples()
            ],
        }

    return await _in_thread(_run)
