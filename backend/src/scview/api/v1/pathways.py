"""Reactome pathway diagrams (core/reactome.py) and their colouring by a DE result.

``GET /pathways/reactome/{st_id}/diagram`` is the renderer's model of the
pathway's diagram (or of the nearest ancestor's that draws it), fetched from
reactome.org on first use; ``GET /datasets/{id}/pathways/reactome/{st_id}/values``
is each node's log fold change from a stored DE result (``de_result``) or one
group's stored markers (``column`` + ``group``). Nothing about the dataset is
ever sent to Reactome. Nothing is written to the dataset.
"""

from __future__ import annotations

import asyncio

from fastapi import APIRouter, Depends, HTTPException

from scview.config import Settings
from scview.core import de_results, reactome
from scview.core.dataset_manager import DatasetManager
from scview.dependencies import get_dataset_manager, get_settings_dep

router = APIRouter()


async def _in_thread(fn, *args):
    try:
        return await asyncio.to_thread(fn, *args)
    except HTTPException:
        raise
    except reactome.ReactomeUnavailable as e:
        raise HTTPException(status_code=503, detail=str(e))
    except LookupError as e:  # PathwayNotFound too
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))


def _diagram(st_id: str, settings: Settings) -> dict:
    return reactome.get_diagram(st_id, settings.DATA_DIR, auto_download=settings.REACTOME_AUTO_DOWNLOAD)


@router.get("/pathways/reactome/{st_id}/diagram")
async def reactome_diagram(st_id: str, settings: Settings = Depends(get_settings_dep)):
    """The diagram model: compartments, nodes (class, box, label, genes), reactions,
    connectors, the outline of a pathway drawn inside an ancestor, release and
    attribution. ``too_broad`` with ``suggestions`` for an overview of sub-pathways."""
    return await _in_thread(_diagram, st_id, settings)


@router.get("/datasets/{dataset_id}/pathways/reactome/{st_id}/values")
async def reactome_values(
    dataset_id: str,
    st_id: str,
    de_result: str = "",
    column: str = "",
    group: str = "",
    dm: DatasetManager = Depends(get_dataset_manager),
    settings: Settings = Depends(get_settings_dep),
):
    """Per gene and per node: log fold change (complex / set: mean of its measured
    genes) and genes changed at adjusted p < 0.05, from ``de_result`` (an id from a DE
    response) or the markers of ``group`` in ``column``."""
    if bool(de_result) == bool(column and group):
        raise HTTPException(status_code=422,
                            detail="Send either de_result, or column and group (a group's markers).")
    try:
        reactome.check_st_id(st_id)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    adaptor = None
    if column:
        adaptor = await dm.get_or_load_dataset(dataset_id)
        if adaptor is None:
            raise HTTPException(status_code=404, detail=f"Dataset '{dataset_id}' not found.")

    def _run() -> dict:
        if de_result:
            stored = de_results.recall(dataset_id, de_result)
            if stored is None:
                raise LookupError(f"The server no longer holds this DE result (it keeps the last "
                                  f"{de_results.KEEP}); run the DE again.")
            table = reactome.table_from_result(stored)
            source = f"de:{de_result}"
        else:
            table = reactome.table_from_markers(adaptor.adata, column, group)
            source = f"markers:{column}/{group}"
        out = reactome.diagram_values(_diagram(st_id, settings), table)
        out["source"] = source
        return out

    return await _in_thread(_run)
