"""Spatially variable genes (``core/spatial_autocorr.py``): Moran's I per gene.

``POST /spatial/autocorr`` runs it on the dataset's spots (all sections, or one) and
saves it with the dataset (uns ``moranI``, a provenance step the replay script
repeats); ``GET /spatial/autocorr`` gives the last run (results store) or 404;
``GET /spatial/autocorr/{result_id}.csv`` every gene of a kept run.
"""

from __future__ import annotations

import io
import logging
from typing import Any, Literal

import numpy as np
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from scview.core import large_mode, results_store
from scview.core import spatial_autocorr as sa
from scview.core.dataset_manager import DatasetManager
from scview.core.obs_writes import dataset_edit, save_edit, uns_snapshot
from scview.dependencies import get_dataset_manager

logger = logging.getLogger(__name__)

router = APIRouter()

MAX_ROWS = 2000  # rows sent to the browser, highest I first; the CSV has all


class AutocorrRequest(BaseModel):
    genes: Literal["hvg", "all"] | list[str] = "hvg"
    library_id: str | None = Field(default=None, max_length=256)
    layer: str | None = Field(default=None, max_length=256)


def _num(a, digits: int = 6) -> list:
    out: list = []
    for v in np.asarray(a, dtype=np.float64).tolist():
        out.append(None if not np.isfinite(v) else float(f"{v:.{digits}g}"))
    return out


def result_body(r: sa.Result) -> dict[str, Any]:
    t = r.table.head(MAX_ROWS)
    cols = {c: _num(t[c]) for c in ("I", "expected_I", "var_I", "z", "pval", "pval_adj")}
    rows = [{"gene": str(g), **{c: cols[c][i] for c in cols}} for i, g in enumerate(t["gene"])]
    return {
        "result_id": r.id, "method": "moran", "n_obs": r.n_obs, "n_genes": r.n_genes,
        "neighbors": r.neighbors, "libraries": r.libraries, "genes": r.genes_choice,
        "matrix_source": r.matrix_source, "unit": r.unit, "n_constant": r.n_constant,
        "notes": r.notes, "rows_shown": len(rows), **sa.summary_counts(r),
        "elapsed_seconds": r.elapsed_seconds, "created_at": r.created_at, "rows": rows,
    }


async def _adaptor(dm: DatasetManager, dataset_id: str):
    adaptor = await dm.get_or_load_dataset(dataset_id)
    if adaptor is None:
        raise HTTPException(status_code=404, detail=f"Dataset '{dataset_id}' not found.")
    return adaptor


NO_SPATIAL = ("This dataset has no spatial coordinates: spatially variable genes need spot or cell "
              "positions (obsm['spatial'], e.g. a Space Ranger folder).")


@router.post("/datasets/{dataset_id}/spatial/autocorr")
async def run_autocorr(
    dataset_id: str,
    body: AutocorrRequest,
    dm: DatasetManager = Depends(get_dataset_manager),
):
    """Moran's I for the chosen genes; saved with the dataset and kept for GET."""
    async with dataset_edit(dataset_id) as held:
        adaptor = await _adaptor(dm, dataset_id)
        large_mode.check(adaptor, "Spatially variable genes", large_mode.WHOLE_MATRIX)
        adata = adaptor.adata
        if not sa.has_spatial(adata):
            raise HTTPException(status_code=422, detail=NO_SPATIAL)
        output_path = str(dm.derived_h5ad_path(dataset_id, adaptor.h5ad_path))

        def _run() -> sa.Result:
            r = sa.run(adata, genes=body.genes, library_id=body.library_id, layer=body.layer,
                       dataset_id=dataset_id)
            before = uns_snapshot(adata)
            sa.record(adata, r, body.genes, body.library_id, body.layer)
            save_edit(adaptor, adata, output_path, before)
            return r

        try:
            r = await held.run(_run)
        except sa.SpatialAutocorrError as e:
            raise HTTPException(status_code=422, detail=str(e))
        except KeyError as e:
            raise HTTPException(status_code=404, detail=str(e).strip("'\""))
        except Exception as e:
            dm.evict(dataset_id)  # the cached copy may hold the step without it being saved
            logger.error("Moran's I failed for %s: %s", dataset_id, e, exc_info=True)
            raise HTTPException(status_code=500, detail=f"Could not compute spatially variable genes: {e!s}")
        dm.saved_in_place(dataset_id, adaptor, output_path)
        sa.remember(adaptor, r)  # after the step is recorded: stored as current
        return result_body(r)


@router.get("/datasets/{dataset_id}/spatial/autocorr")
async def last_autocorr(dataset_id: str, allow_empty: bool = False,
                        dm: DatasetManager = Depends(get_dataset_manager)):
    """The last run on this dataset, or 404; with ``allow_empty`` null instead (the panel asks on open,
    and a 404 is a console error in the browser)."""
    adaptor = await _adaptor(dm, dataset_id)
    r = sa.last(adaptor)
    if r is None:
        if allow_empty:
            return None
        raise HTTPException(status_code=404, detail="No spatially variable genes computed for this dataset yet.")
    return result_body(r)


@router.get("/datasets/{dataset_id}/spatial/autocorr/{result_id}.csv")
async def autocorr_csv(dataset_id: str, result_id: str, dm: DatasetManager = Depends(get_dataset_manager)):
    """Every tested gene of a kept run, highest I first."""
    adaptor = await _adaptor(dm, dataset_id)
    r = sa.last(adaptor)
    if r is None or r.id != result_id:  # an older run, or the store has it only
        r = results_store.recall(dataset_id, result_id, sa.KIND) if results_store.safe_id(result_id) else None
    if r is None:
        raise HTTPException(status_code=404, detail="That result is no longer kept on the server: run it again.")
    buf = io.StringIO()
    r.table.to_csv(buf, index=False)
    return StreamingResponse(iter([buf.getvalue()]), media_type="text/csv", headers={
        "Content-Disposition": f'attachment; filename="moran_{result_id}.csv"'})
