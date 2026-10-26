"""Metadata endpoints – obs column info, values, and summaries."""

from typing import Literal

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import Response
from pydantic import BaseModel

from scview.dependencies import get_dataset_manager
from scview.core import category_order, cell_ontology, large_layout
from scview.core.dataset_manager import DatasetManager
from scview.core.arrow_serializer import series_to_arrow_ipc
from scview.core.obs_writes import dataset_edit, edit_room, editable, save_edit, uns_snapshot
from scview.core.uns_keys import enrichment_key, markers_key

router = APIRouter()

ARROW_CONTENT_TYPE = "application/vnd.apache.arrow.stream"


@router.get("/datasets/{dataset_id}/metadata")
async def list_metadata_columns(
    dataset_id: str,
    dm: DatasetManager = Depends(get_dataset_manager),
):
    adaptor = await dm.get_or_load_dataset(dataset_id)
    if adaptor is None:
        raise HTTPException(status_code=404, detail="Dataset not found.")
    return adaptor.obs_columns_info()


@router.get("/datasets/{dataset_id}/metadata/summary")
async def metadata_summary(
    dataset_id: str,
    groupby: str = Query(default=""),
    dm: DatasetManager = Depends(get_dataset_manager),
):
    adaptor = await dm.get_or_load_dataset(dataset_id)
    if adaptor is None:
        raise HTTPException(status_code=404, detail="Dataset not found.")

    if not groupby:
        # Default: try common categorical columns
        for candidate in ("leiden", "louvain", "seurat_clusters", "cluster", "celltype"):
            try:
                adaptor.get_obs_column(candidate)
                groupby = candidate
                break
            except KeyError:
                continue

    if not groupby:
        return {"groupby": None, "counts": {}}

    try:
        counts = adaptor.get_obs_summary(groupby)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Column '{groupby}' not found.")

    return {"groupby": groupby, "counts": counts}


@router.get("/datasets/{dataset_id}/metadata/cell/{index}")
async def get_cell_metadata(
    dataset_id: str,
    index: int,
    dm: DatasetManager = Depends(get_dataset_manager),
):
    """Return all obs column values for a single cell at the given index."""
    adaptor = await dm.get_or_load_dataset(dataset_id)
    if adaptor is None:
        raise HTTPException(status_code=404, detail="Dataset not found.")
    try:
        return adaptor.get_cell_metadata(index)
    except IndexError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/datasets/{dataset_id}/metadata/crosstab")
async def get_crosstab(
    dataset_id: str,
    row: str = Query(...),
    col: str = Query(...),
    facet: str = Query(default="", description="Optional third column: counts per its value too"),
    dm: DatasetManager = Depends(get_dataset_manager),
):
    """Return a cross-tabulation of two obs columns (optionally per value of a third)."""
    adaptor = await dm.get_or_load_dataset(dataset_id)
    if adaptor is None:
        raise HTTPException(status_code=404, detail="Dataset not found.")
    try:
        return adaptor.get_obs_crosstab(row, col, facet or None)
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.get("/datasets/{dataset_id}/metadata/{column}")
async def get_metadata_column(
    dataset_id: str,
    column: str,
    format: str = Query(default="arrow", description="Response format: arrow or json"),
    dm: DatasetManager = Depends(get_dataset_manager),
):
    """Return values for a single obs column as Arrow IPC or JSON."""
    adaptor = await dm.get_or_load_dataset(dataset_id)
    if adaptor is None:
        raise HTTPException(status_code=404, detail="Dataset not found.")

    try:
        series = adaptor.get_obs_column(column)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Column '{column}' not found.")

    if format == "json":
        # Missing values as null: NaN is not valid JSON and made this a 500.
        values = series.astype(object).where(series.notna(), None).tolist()
        return {"column": column, "values": values}

    ipc_bytes = series_to_arrow_ipc(series, name=column)
    return Response(content=ipc_bytes, media_type=ARROW_CONTENT_TYPE)


def _marker_keys(adata, column: str) -> list[str]:
    keys = [markers_key(column)] if markers_key(column) in adata.uns else []
    rgg = adata.uns.get("rank_genes_groups")
    if isinstance(rgg, dict) and rgg.get("params", {}).get("groupby") == column:
        keys.append("rank_genes_groups")
    return keys


def _follow_rename(adata, column: str, old: str, new: str, merge: bool) -> None:
    """Carry a category rename into the column's markers and cached enrichment,
    which are keyed by group name and otherwise keep the old label."""
    import pandas as pd

    from scview.core import rank_genes
    from scview.core.ora_background import meta_key
    from scview.core.pipeline import drop_enrichment_cache

    if merge:
        # Two groups became one: markers and enrichment no longer apply.
        drop_enrichment_cache(adata, column)
        keys = _marker_keys(adata, column)
        if keys and large_layout.is_large(adata):
            # A large dataset's matrix stays on disk: the markers go, and are computed
            # again from the gene store on the next visit to the column.
            for key in keys:
                adata.uns.pop(key, None)
        elif keys:
            # by blocks of genes, bit for bit scanpy's (core/rank_genes.py)
            rank_genes.rank_genes_groups(adata, column, method="wilcoxon", pts=True,
                                         key_added=markers_key(column), specificity=True)
            if "rank_genes_groups" in keys:
                adata.uns["rank_genes_groups"] = adata.uns[markers_key(column)]
        return
    for key in _marker_keys(adata, column):
        rgg = adata.uns[key]
        for field in ("names", "scores", "logfoldchanges", "pvals", "pvals_adj"):
            arr = rgg.get(field)
            names = getattr(getattr(arr, "dtype", None), "names", None)
            if names and old in names:
                arr.dtype.names = tuple(new if n == old else n for n in names)
        for field in ("pts", "pts_rest"):
            df = rgg.get(field)
            if isinstance(df, pd.DataFrame) and old in df.columns:
                rgg[field] = df.rename(columns={old: new})
    for k_old, k_new in ((enrichment_key(column, old), enrichment_key(column, new)),
                         (meta_key(column, old), meta_key(column, new))):
        if k_old in adata.uns:
            adata.uns[k_new] = adata.uns.pop(k_old)


class CategoryOrderBody(BaseModel):
    # Every category once, in the new order; or a quick sort instead.
    order: list[str] | None = None
    sort: Literal["natural", "size"] | None = None


@router.post("/datasets/{dataset_id}/metadata/{column}/order")
async def set_category_order(
    dataset_id: str,
    column: str,
    body: CategoryOrderBody,
    dm: DatasetManager = Depends(get_dataset_manager),
):
    """Set the order of a label column's categories (core/category_order.py), which
    the legend, violin, dot plot and composition views list groups by. Persists to
    the derived layer, never the original upload."""
    async with dataset_edit(dataset_id) as held:
        adaptor = await dm.get_or_load_dataset(dataset_id)
        if adaptor is None:
            raise HTTPException(status_code=404, detail="Dataset not found.")
        output_path = str(dm.derived_h5ad_path(dataset_id, adaptor.h5ad_path))
        backed = await edit_room(dm, dataset_id, adaptor)

        def _run() -> list[str]:
            adata = editable(adaptor)
            before = uns_snapshot(adata)
            order = category_order.reorder(adata, column, body.order, body.sort)
            # Only obs (and a colours list in uns) changed: no matrix rewrite.
            save_edit(adaptor, adata, output_path, before)
            return order

        try:
            order = await held.run(_run)
        except category_order.OrderError as exc:  # raised before any change
            raise HTTPException(status_code=exc.status, detail=str(exc))
        except Exception as exc:
            dm.evict(dataset_id)  # may hold an order that was never saved
            raise HTTPException(status_code=500, detail=f"Could not save the order: {exc!s}")
        if backed:
            dm.evict(dataset_id)
            await dm.load_dataset(dataset_id)
        else:
            dm.saved_in_place(dataset_id, adaptor, output_path)
    return {"column": column, "order": order}


class RenameCategoryBody(BaseModel):
    old: str
    new: str


@router.post("/datasets/{dataset_id}/metadata/{column}/rename")
async def rename_obs_category(
    dataset_id: str,
    column: str,
    body: RenameCategoryBody,
    dm: DatasetManager = Depends(get_dataset_manager),
):
    """Rename a category in a categorical obs column (e.g. correct a cell-type label).

    Persists to the derived layer, never the original upload. If the new label already
    exists, the two categories are merged.
    """
    new = body.new.strip()
    if not new:
        raise HTTPException(status_code=400, detail="New label must not be empty.")

    # One edit of the dataset's file at a time (markers, labels, a run).
    async with dataset_edit(dataset_id) as held:
        adaptor = await dm.get_or_load_dataset(dataset_id)
        if adaptor is None:
            raise HTTPException(status_code=404, detail="Dataset not found.")
        output_path = str(dm.derived_h5ad_path(dataset_id, adaptor.h5ad_path))
        backed = await edit_room(dm, dataset_id, adaptor)

        def _run() -> None:
            adata = editable(adaptor)
            if column not in adata.obs.columns:
                raise HTTPException(status_code=404, detail=f"Column '{column}' not found.")
            col = adata.obs[column]
            if col.dtype.name != "category":
                col = col.astype("category")
            cats = list(col.cat.categories)
            if body.old not in cats:
                raise HTTPException(
                    status_code=404, detail=f"'{body.old}' is not a category of '{column}'."
                )
            merge = new in cats and new != body.old
            if merge:
                # Through the categories, not str: astype(str) made missing values "nan".
                col = col.copy()
                col[col == body.old] = new
                adata.obs[column] = col.cat.remove_unused_categories()
            else:
                adata.obs[column] = col.cat.rename_categories({body.old: new})
            _follow_rename(adata, column, body.old, new, merge)
            category_order.follow_rename(adata, column, body.old, new, merge)
            cell_ontology.follow_rename(adata, column, body.old, new, merge)
            # A full write (markers are edited in place), beside the file then swapped in.
            save_edit(adaptor, adata, output_path)

        try:
            # Markers may be recomputed on a merge: seconds, so in a worker thread.
            await held.run(_run)
        except HTTPException:
            raise  # raised before anything changed
        except Exception as exc:
            dm.evict(dataset_id)  # may hold a rename that was never saved
            raise HTTPException(status_code=500, detail=f"Could not save the rename: {exc!s}")
        if backed:
            dm.evict(dataset_id)
            await dm.load_dataset(dataset_id)
        else:
            dm.saved_in_place(dataset_id, adaptor, output_path)
    return {"column": column, "old": body.old, "new": new}
