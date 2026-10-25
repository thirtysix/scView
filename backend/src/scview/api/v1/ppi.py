"""Protein interaction subnetwork of a gene list (STRING physical links).

``POST /ppi/network`` takes any list; ``GET /datasets/{id}/ppi`` resolves the list
server-side from a stored DE result, a group's markers or an NMF program. The
network is fetched on first use (core/ppi.py); the analysis runs in a worker
thread. Nothing is written to the dataset; the last few dataset queries are kept
against the loaded dataset object for the co-pilot.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from scview.config import Settings
from scview.core import ppi
from scview.core.dataset_manager import DatasetManager
from scview.core.species import detect_species
from scview.dependencies import get_dataset_manager, get_settings_dep

router = APIRouter()

MAX_REQUEST_GENES = 5000  # more are refused; the first MAX_GENES are analysed


class PPIRequest(BaseModel):
    species: str = Field(pattern="^(human|mouse)$")
    genes: list[str] = Field(max_length=MAX_REQUEST_GENES)
    score: int = ppi.MIN_SCORE
    logfc: dict[str, float] | None = None


def _dest(settings: Settings) -> Path:
    return Path(settings.DATA_DIR) / "networks"


def _check_score(score: int) -> None:
    if score not in ppi.SCORES:
        raise HTTPException(status_code=422,
                            detail=f"score must be one of {', '.join(map(str, ppi.SCORES))}.")


async def _in_thread(fn, *args):
    try:
        return await asyncio.to_thread(fn, *args)
    except HTTPException:
        raise
    except ppi.PPIUnavailable as e:
        raise HTTPException(status_code=503, detail=str(e))
    except LookupError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))


@router.get("/ppi/status")
async def ppi_status(settings: Settings = Depends(get_settings_dep)):
    """Which species' STRING networks are on disk, and the attribution."""
    out = []
    for s in ppi.SPECIES.values():
        meta = ppi.read_meta(_dest(settings), s.species)
        out.append({"species": s.species, "taxid": s.taxid, "downloaded": meta is not None,
                    "n_proteins": (meta or {}).get("n_proteins"), "n_links": (meta or {}).get("n_links"),
                    "fetched_at": (meta or {}).get("fetched_at"), "links_url": s.links_url})
    return {"version": ppi.STRING_VERSION, "licence": ppi.LICENCE, "attribution": ppi.ATTRIBUTION,
            "citation": ppi.CITATION, "scores": list(ppi.SCORES), "max_genes": ppi.MAX_GENES,
            "auto_download": settings.PPI_AUTO_DOWNLOAD, "species": out}


@router.post("/ppi/network")
async def ppi_network(req: PPIRequest, settings: Settings = Depends(get_settings_dep)):
    """The STRING subnetwork of any gene list."""
    _check_score(req.score)

    def _run() -> dict:
        net = ppi.load_network(req.species, _dest(settings), auto_download=settings.PPI_AUTO_DOWNLOAD)
        return ppi.analyse(net, req.genes, score=req.score, logfc=req.logfc)

    return await _in_thread(_run)


@router.get("/datasets/{dataset_id}/ppi")
async def dataset_ppi(
    dataset_id: str,
    source: str,
    n: int = 100,
    direction: str = "up",
    score: int = ppi.MIN_SCORE,
    species: str = "",
    dm: DatasetManager = Depends(get_dataset_manager),
    settings: Settings = Depends(get_settings_dep),
):
    """The subnetwork of a list taken from this dataset: ``de:<result id>``,
    ``markers:<column>/<group>`` or ``program:<k>``; ``n`` genes (at most 500)."""
    _check_score(score)
    if not 1 <= n <= ppi.MAX_GENES:
        raise HTTPException(status_code=422, detail=f"n must be between 1 and {ppi.MAX_GENES}.")
    if species and species not in ppi.SPECIES:
        raise HTTPException(status_code=422, detail="species must be human or mouse.")
    adaptor = await dm.get_or_load_dataset(dataset_id)
    if adaptor is None:
        raise HTTPException(status_code=404, detail=f"Dataset '{dataset_id}' not found.")
    adata = adaptor.adata

    def _run() -> dict:
        sp_name = species
        call = None
        if not sp_name:
            call = detect_species(adata)
            if call.species not in ppi.SPECIES:
                raise ValueError(f"Could not tell whether this dataset is human or mouse ({call.evidence}). "
                                 "Choose the species.")
            sp_name = call.species
        label, genes, lfc = ppi.resolve_source(adata, dataset_id, source, n, direction)
        if not genes:
            raise LookupError(f"No genes in {label} pass the cut-offs.")
        net = ppi.load_network(sp_name, _dest(settings), auto_download=settings.PPI_AUTO_DOWNLOAD)
        result = ppi.analyse(net, genes, score=score, logfc=lfc)
        result.update(label=label, source=source, direction=direction,
                      species_evidence=call.evidence if call else "chosen by the user")
        ppi.remember(adata, label, result)
        return result

    return await _in_thread(_run)
