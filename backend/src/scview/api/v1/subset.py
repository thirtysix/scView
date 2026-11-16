"""A new dataset made of some of a dataset's cells: ``POST /datasets/{id}/subset``
(``core/subset.py``). The parent is never changed; the new dataset is then processed like
any other (Data Assessment, or ``POST /datasets/{new id}/assessment/run-stream``)."""

from __future__ import annotations

import asyncio
import logging
import uuid
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from scview.config import Settings
from scview.core import large_mode, subset, working_copy
from scview.core.dataset_manager import DatasetManager, clean_dataset_name
from scview.core.obs_writes import dataset_edit
from scview.dependencies import get_dataset_manager, get_settings_dep

logger = logging.getLogger(__name__)

router = APIRouter()

WHAT = "A subset"
WHY = ("it reads the chosen cells' counts into memory, and a large dataset's matrices stay on disk. "
       "Make the subset from a smaller copy of the data (import only the cells you need), or filter "
       "the cells out of the large dataset with a pipeline run")


class SubsetResponse(BaseModel):
    dataset_id: str | None  # None for a dry run
    name: str
    n_cells: int
    n_parent_cells: int
    n_genes: int | None = None
    described: str
    counts_source: str | None = None
    carried: list[str]
    not_carried: list[str]
    renamed: dict[str, str]
    dry_run: bool = False
    next: str


@router.post("/datasets/{dataset_id}/subset", response_model=SubsetResponse)
async def make_subset(
    dataset_id: str,
    body: subset.SubsetRequest,
    dm: DatasetManager = Depends(get_dataset_manager),
    settings: Settings = Depends(get_settings_dep),
) -> SubsetResponse:
    """A new dataset of the cells ``filter``, ``groups`` or ``mask`` choose: their counts,
    obs (scView's own columns as ``parent_*``), the parent's UMAP as ``X_umap_parent``,
    per-cell modalities and the input files read by barcode. ``dry_run``: what it would
    hold, nothing made. Refused: fewer than ``subset.MIN_CELLS`` cells, every cell, a
    large dataset (422)."""
    adaptor = await dm.get_or_load_dataset(dataset_id)
    if adaptor is None:
        raise HTTPException(status_code=404, detail="Dataset not found.")
    if large_mode.is_large(adaptor):
        raise large_mode.LargeDatasetError(WHAT, WHY)
    info = dm.get_dataset_info(dataset_id) or {}
    parent_name = str(info.get("name") or dataset_id)
    name = None
    if body.name is not None and body.name.strip():
        try:
            name = clean_dataset_name(body.name)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e)) from None
    data_dir = Path(settings.DATA_DIR)

    # The edit lock: no run or edit changes the parent while its cells are chosen and read.
    async with dataset_edit(dataset_id) as edit:
        try:
            sel = await edit.run(subset.select, adaptor, body)
        except subset.SubsetError as e:
            raise HTTPException(status_code=e.status, detail=str(e)) from None
        name = name or subset.default_name(parent_name, sel)
        adata = adaptor.adata
        n_all = int(adata.n_obs)
        what = await asyncio.to_thread(subset.contents, adata, data_dir, dataset_id)
        if body.dry_run:
            return SubsetResponse(
                dataset_id=None, name=name, n_cells=sel.n, n_parent_cells=n_all, described=sel.described,
                carried=what.carried, not_carried=what.not_carried, renamed=what.renamed, dry_run=True,
                next="Send the same request without dry_run to make the dataset.")

        new_id = uuid.uuid4().hex
        parent = {"id": dataset_id, "name": parent_name, "source": info.get("source"),
                  "filename": Path(adaptor.h5ad_path).name if getattr(adaptor, "h5ad_path", None) else None}
        fallback = dm.h5ad_candidates(dataset_id)[1:]
        need = await asyncio.to_thread(subset.read_need, getattr(adaptor, "h5ad_path", None), sel.n, n_all)

        def make() -> tuple[int, str]:
            sub, counts_source = subset.build(adata, sel.keep, fallback, what)
            subset.record(sub, parent=parent, sel=sel, n_parent=n_all, counts_source=counts_source, info=what)
            subset.persist(data_dir, new_id, name, sub, parent=parent, sel=sel, n_parent=n_all, info=what)
            return int(sub.n_vars), counts_source

        async with working_copy.reading(dm, dataset_id, need, f"{WHAT} of {sel.n:,} cells") as held:
            try:
                n_genes, counts_source = await held.run_with(edit, make)
            except subset.SubsetError as e:
                raise HTTPException(status_code=e.status, detail=str(e)) from None
            except Exception as e:
                logger.error("Subset of %s failed: %s", dataset_id, e, exc_info=True)
                raise HTTPException(status_code=500, detail=f"The subset could not be made: {e}") from None

    return SubsetResponse(
        dataset_id=new_id, name=name, n_cells=sel.n, n_parent_cells=n_all, n_genes=n_genes,
        described=sel.described, counts_source=counts_source, carried=what.carried,
        not_carried=what.not_carried, renamed=what.renamed,
        next=("Process the new dataset: Data Assessment > run the pipeline (MCP: run_pipeline with its id; "
              "the default steps start from its counts)."))
