"""Cytokine response (CytoSig signatures, ridge regression + permutation z): which
cytokines' signalling each group's expression looks like. See ``core/cytokine_activity.py``.

Per group vs the rest (or all cells), or condition A vs B within each group. Group
sums come from the log-normalised matrix in memory, or from a large dataset's gene
store (only the ~4,900 signature genes are read). Results are kept for the loaded
dataset (the same request answers from the store) and the last one is a co-pilot fact.
"""

from __future__ import annotations

import asyncio
import time
from pathlib import Path
from typing import Literal

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from scview.config import Settings
from scview.core import activity, cytokine_activity as ca, large_mode, orthologs
from scview.core.dataset_manager import DatasetManager
from scview.core.species import detect_species
from scview.dependencies import get_dataset_manager, get_settings_dep

router = APIRouter()

MAX_GROUPS = 100


class CytokineRequest(BaseModel):
    mode: Literal["groups", "conditions"] = "groups"
    # groups: required; conditions: optional (empty = all cells as one group).
    groupby: str = Field(default="", max_length=256)
    reference: Literal["rest", "all"] = "rest"
    condition: str = Field(default="", max_length=256)
    a: str = Field(default="", max_length=256)
    b: str = Field(default="", max_length=256)
    n_permutations: int = Field(default=ca.N_PERMUTATIONS, ge=100, le=ca.N_PERMUTATIONS)


def _dest(settings: Settings) -> Path:
    return Path(settings.DATA_DIR) / "networks" / "cytosig"  # the data volume


@router.get("/cytokines/signature")
async def signature_status(settings: Settings = Depends(get_settings_dep)):
    """Whether the CytoSig signature is downloaded, and where it comes from."""
    meta = ca.read_meta(_dest(settings))
    return {"downloaded": meta is not None, "url": ca.URL, "licence": ca.LICENCE,
            "citation": ca.CITATION, "sha256": ca.SHA256,
            "auto_download": settings.CYTOSIG_AUTO_DOWNLOAD, **({"meta": meta} if meta else {})}


def _check(req: CytokineRequest, obs) -> None:
    for col in (req.groupby, req.condition):
        if col and col not in obs.columns:
            raise HTTPException(status_code=404, detail=f"Column '{col}' not found.")
    if req.mode == "groups" and not req.groupby:
        raise HTTPException(status_code=400, detail="Choose a group column (groupby).")
    if req.mode == "conditions":
        if not req.condition or not req.a or not req.b or req.a == req.b:
            raise HTTPException(status_code=400,
                                detail="Choose a condition column and two different values (a, b).")
        present = set(obs[req.condition].dropna().astype(str))
        for v in (req.a, req.b):
            if v not in present:
                raise HTTPException(status_code=400, detail=f"'{v}' is not a value of '{req.condition}'.")
    if req.groupby:
        n = obs[req.groupby].nunique(dropna=True)
        if not 1 <= n <= MAX_GROUPS:
            raise HTTPException(status_code=400,
                                detail=f"'{req.groupby}' has {n} groups; at most {MAX_GROUPS}.")


@router.post("/datasets/{dataset_id}/cytokines/activity")
async def cytokine_activity(
    dataset_id: str,
    body: CytokineRequest,
    dm: DatasetManager = Depends(get_dataset_manager),
    settings: Settings = Depends(get_settings_dep),
):
    """z-score, permutation p and ridge coefficient per group x cytokine (CytoSig's 43)."""
    adaptor = await dm.get_or_load_dataset(dataset_id)
    if adaptor is None:
        raise HTTPException(status_code=404, detail=f"Dataset '{dataset_id}' not found.")
    adata = adaptor.adata
    _check(body, adata.obs)
    key = tuple(body.model_dump().values())
    stored = ca.recall(adaptor, key)
    if stored is not None:
        ca.remember(adaptor, key, stored)  # the co-pilot's fact follows what is shown
        return {**stored, "reused": True}

    def _run() -> dict:
        t0 = time.time()
        sig = ca.load(_dest(settings), auto_download=settings.CYTOSIG_AUTO_DOWNLOAD)
        species = detect_species(adata).species
        table, why = None, ""
        if species == "mouse":
            try:
                table = orthologs.load(Path(settings.DATA_DIR) / "networks",
                                       auto_download=settings.ORTHOLOGS_AUTO_DOWNLOAD)
            except orthologs.OrthologsUnavailable as e:
                why = str(e)
        if body.mode == "groups":
            design = ca.design_groups(adata.obs, body.groupby)
        else:
            design = ca.design_conditions(adata.obs, body.groupby or None, body.condition, body.a, body.b)
        if large_mode.is_large(adaptor):
            store = adaptor.gene_store()
            if store is None:
                large_mode.refuse("Cytokine response", (adaptor.large_info() or {}).get("gene_store_note")
                                  or "the gene-major copy is not ready")
            match = ca.match_genes(sig, adata.var_names, species, table, why)
            sums, sizes = ca.group_sums_store(store, match.positions, design.codes, design.k)
            source = "log-normalised X (gene store)"
        else:
            M, genes, source = activity.lognorm_matrix(adata)
            match = ca.match_genes(sig, genes, species, table, why)
            sums, sizes = ca.group_sums(M, match.positions, design.codes, design.k)
        res = ca.score(match, design, sums, sizes, reference=body.reference,
                       n_perm=body.n_permutations)
        return {
            **res, "mode": body.mode, "groupby": body.groupby or None, "reference": body.reference,
            "condition": body.condition or None, "a": body.a or None, "b": body.b or None,
            "species": species, "mapped_by": match.mapped_by, "matrix_source": source,
            "citation": ca.CITATION, "source_url": ca.URL, "licence": ca.LICENCE,
            "elapsed_seconds": round(time.time() - t0, 2),
        }

    try:
        result = await asyncio.to_thread(_run)
    except ca.SignatureUnavailable as e:
        raise HTTPException(status_code=503, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    ca.remember(adaptor, key, result)
    return {**result, "reused": False}
