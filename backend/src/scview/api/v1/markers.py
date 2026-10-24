"""Marker gene endpoints."""

import logging

import math

import anndata as ad
import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import Response

from scview.dependencies import get_dataset_manager
from scview.core.dataset_manager import DatasetManager
from scview.core import marker_specificity
from scview.core.arrow_serializer import dataframe_to_arrow_ipc
from scview.core.obs_writes import dataset_edit
from scview.core import working_copy
from scview.core.large_mode import LargeDatasetError
from scview.core.ora_background import resolve_markers_key
from scview.core.pipeline import drop_enrichment_cache
from scview.core.uns_keys import markers_column, markers_key

logger = logging.getLogger(__name__)

router = APIRouter()

ARROW_CONTENT_TYPE = "application/vnd.apache.arrow.stream"

# The pipeline's limit too: every gene is kept per group, so a column with
# thousands of levels (barcodes, free text) would cost a lot to compute and store.
MAX_MARKER_GROUPS = 100

# No markers is an expected state (an unprocessed upload), not a client error: 200
# with an empty table and this reason, so opening a dataset logs no failed request.
NO_MARKERS = "No marker genes yet: run the pipeline or pick a column."
# The base columns of a marker table (``AnnDataAdaptor.get_markers``), for an empty one.
MARKER_COLUMNS = ("group", "gene", "logfoldchange", "pval", "pval_adj")


def _no_markers(format: str, reason: str):
    """An empty marker table with ``reason``: JSON as the frontend's MarkersResponse
    (no groups) plus ``reason``; Arrow as an empty table, the reason in a header."""
    if format == "json":
        return {"groups": [], "markers": {}, "reason": reason}
    empty = pd.DataFrame({c: pd.Series(dtype=object if c in ("group", "gene") else float)
                          for c in MARKER_COLUMNS})
    return Response(content=dataframe_to_arrow_ipc(empty), media_type=ARROW_CONTENT_TYPE,
                    headers={"X-Scview-Reason": reason})


def _compute_markers(adata: ad.AnnData, column: str) -> dict:
    """Wilcoxon markers for every group of ``column``, as the pipeline computes them
    (by blocks of genes, ``core/rank_genes.py``: bit for bit scanpy's, with the
    specificity frame from the group means unless scanpy ran).

    Runs on a light AnnData sharing the dataset's matrix rather than on a copy of
    the whole dataset, and leaves the dataset's own obs column as it is.
    """
    from scview.core import rank_genes

    if getattr(adata, "isbacked", False):
        adata = adata.to_memory()
    # scanpy ranks .raw when there is one; give it that matrix directly.
    use_raw = adata.raw is not None
    src = adata.raw if use_raw else adata
    labels = adata.obs[column]
    # rank_genes_groups needs a categorical groupby; coerce bool / numeric /
    # object columns (e.g. predicted_doublet) the way the pipeline does.
    if not isinstance(labels.dtype, pd.CategoricalDtype):
        labels = labels.astype(str).astype("category")
    work = ad.AnnData(
        X=src.X,
        obs=pd.DataFrame({column: labels.values}, index=adata.obs_names),
        var=pd.DataFrame(index=src.var_names),
    )
    if "log1p" in adata.uns:  # the base the fold changes undo, as for the pipeline's
        work.uns["log1p"] = adata.uns["log1p"]
    rank_genes.rank_genes_groups(work, column, method="wilcoxon", pts=True, use_raw=False,
                                 specificity=True)
    rgg = work.uns["rank_genes_groups"]
    rgg["params"]["use_raw"] = use_raw  # what the ranking actually read
    return rgg


def _compute_markers_large(adaptor, column: str) -> dict:
    """``_compute_markers`` for a large dataset: the same Wilcoxon test from the gene
    store (``core/sparse_markers.py``, equal to scanpy's), its specificity columns from
    the group means computed on the way; the matrix stays on disk."""
    from scview.core import large_mode, sparse_markers

    store = adaptor.gene_store()
    if store is None:
        large_mode.refuse("Marker genes for a new column",
                          (adaptor.large_info() or {}).get("gene_store_note")
                          or "the gene-major copy is not ready")
    adata = adaptor.adata
    labels = adata.obs[column]
    if not isinstance(labels.dtype, pd.CategoricalDtype):
        labels = labels.astype(str).astype("category")
    work = ad.AnnData(obs=pd.DataFrame({column: labels.values}, index=adata.obs_names),
                      var=pd.DataFrame(index=adata.var_names))
    if "log1p" in adata.uns:
        work.uns["log1p"] = adata.uns["log1p"]
    sparse_markers.rank_genes_groups(work, column, store, method="wilcoxon", pts=True,
                                     use_raw=False, specificity=True, check_store=False)
    rgg = work.uns["rank_genes_groups"]
    rgg["params"]["use_raw"] = False
    return rgg


def _compute_and_save(dm: DatasetManager, dataset_id: str, adaptor, column: str) -> None:
    """Compute ``column``'s markers into the dataset and save them with it (in memory:
    a backed dataset is read in first, after ``working_copy.make_room``; a large one
    stays on disk, its markers from the gene store)."""
    large = adaptor.is_large
    adata = adaptor.adata if large else working_copy.in_memory(adaptor)
    if column not in adata.obs.columns:
        raise HTTPException(status_code=404, detail=f"Column '{column}' not found in obs.")
    n_groups = adata.obs[column].nunique()
    if n_groups < 2:
        raise HTTPException(
            status_code=400,
            detail=f"Column '{column}' has only one group; need at least two to compute markers.",
        )
    if n_groups > MAX_MARKER_GROUPS:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Column '{column}' has {n_groups} groups; markers are computed "
                f"for at most {MAX_MARKER_GROUPS}."
            ),
        )
    try:
        key = markers_key(column)
        if large:  # specificity comes with them
            adata.uns[key] = _compute_markers_large(adaptor, column)
        else:
            adata.uns[key] = _compute_markers(adata, column)
        drop_enrichment_cache(adata, column)
        if not marker_specificity.has_specificity(adata.uns[key]):  # scanpy ran: best effort
            marker_specificity.attach(adata, adata.uns[key], column)
    except LargeDatasetError:
        raise
    except Exception as e:
        logger.error("On-demand marker computation failed: %s", e)
        raise HTTPException(status_code=500, detail=f"Marker computation failed: {e!s}")
    try:
        dm.persist_uns(dataset_id, adaptor, [key])
    except Exception as e:  # still served from memory; recomputed after a reload
        logger.warning("Could not save markers for '%s': %s", column, e)


def _specificity_target(adata, column: str) -> tuple[str, str] | None:
    """(uns key, obs column) of stored markers that lack their specificity columns:
    markers saved before they existed, or a file's own."""
    key = resolve_markers_key(adata.uns, column)
    if key is None:
        return None
    rgg = adata.uns[key]
    if not isinstance(rgg, dict) or marker_specificity.has_specificity(rgg):
        return None
    col = column or markers_column(key) or str(rgg.get("params", {}).get("groupby") or "")
    if not col or col not in adata.obs.columns:
        return None
    return key, col


def _attach_and_save(dm: DatasetManager, dataset_id: str, adaptor, key: str, column: str) -> None:
    adata = adaptor.adata
    rgg = adata.uns[key]
    if not marker_specificity.attach(adata, rgg, column):
        return
    # The default key is often the very same entry as the column's own.
    keys = [k for k in dict.fromkeys((key, markers_key(column), "rank_genes_groups"))
            if adata.uns.get(k) is rgg]
    try:
        dm.persist_uns(dataset_id, adaptor, keys)
    except Exception as e:  # still served from memory; recomputed after a reload
        logger.warning("Could not save marker specificity for '%s': %s", column, e)


async def ensure_specificity(dm: DatasetManager, dataset_id: str, adaptor, column: str = ""):
    """Give a column's stored markers their specificity columns, once: computed and
    saved with the markers, as on-demand markers are. Returns the current adaptor."""
    if _specificity_target(adaptor.adata, column) is None:
        return adaptor
    async with dataset_edit(dataset_id) as held:
        adaptor = await dm.get_or_load_dataset(dataset_id)
        if adaptor is None:
            raise HTTPException(status_code=404, detail="Dataset not found.")
        target = _specificity_target(adaptor.adata, column)
        if target is not None:
            # Group means over the whole matrix: a second or so, in a worker thread.
            await held.run(_attach_and_save, dm, dataset_id, adaptor, *target)
    return adaptor


@router.get("/datasets/{dataset_id}/markers")
async def get_markers(
    dataset_id: str,
    groupby: str = Query(default="", description="Filter to a specific group"),
    groupby_column: str = Query(default="", description="Compute markers for this obs column"),
    format: str = Query(default="arrow", description="Response format: arrow or json"),
    n_genes: int = Query(default=100, ge=1, le=5000, description="Max genes per group"),
    dm: DatasetManager = Depends(get_dataset_manager),
):
    """Return marker gene table from rank_genes_groups.

    If groupby_column is provided, returns that column's markers, computing and
    saving them the first time. ``groupby`` limits the table to one group.
    No markers stored and no column named (an unprocessed dataset), or an empty
    table, is 200 with no groups and a ``reason`` (Arrow: the ``X-Scview-Reason``
    header); 404 stays for a dataset or a ``groupby_column`` that does not exist.
    """
    adaptor = await dm.get_or_load_dataset(dataset_id)
    if adaptor is None:
        raise HTTPException(status_code=404, detail="Dataset not found.")

    # Markers for a specific obs column: stored ones, else computed once and stored
    # under the column's key (as the pipeline does), so the next visit is instant.
    if groupby_column:
        if not adaptor.has_markers(column=groupby_column):
            # Saving rewrites the dataset's file, so one edit at a time; a request
            # that waited here usually finds the markers already computed.
            async with dataset_edit(dataset_id) as held:
                adaptor = await dm.get_or_load_dataset(dataset_id)
                if adaptor is None:
                    raise HTTPException(status_code=404, detail="Dataset not found.")
                if not adaptor.has_markers(column=groupby_column):
                    if not adaptor.is_large:  # a large dataset is not read in
                        await working_copy.make_room(dm, dataset_id, adaptor)
                    # Seconds to tens of seconds: in a worker thread, so the server
                    # keeps answering meanwhile.
                    await held.run(_compute_and_save, dm, dataset_id, adaptor, groupby_column)
        adaptor = await ensure_specificity(dm, dataset_id, adaptor, groupby_column)
        df = adaptor.get_markers(
            groupby=groupby if groupby else None,
            column=groupby_column,
            n_genes=n_genes,
        )
    else:
        if not adaptor.has_markers():
            return _no_markers(format, NO_MARKERS)
        adaptor = await ensure_specificity(dm, dataset_id, adaptor)
        df = adaptor.get_markers(groupby=groupby if groupby else None, n_genes=n_genes)

    if df is None or df.empty:
        return _no_markers(format, f"No marker genes for group '{groupby}'." if groupby else NO_MARKERS)

    if format == "json":
        records = df.to_dict(orient="records")
        # Replace NaN/Inf with None so JSON serialization succeeds
        for rec in records:
            for key, val in rec.items():
                if isinstance(val, float) and (math.isnan(val) or math.isinf(val)):
                    rec[key] = None
        # Group by cluster/group to match frontend MarkersResponse shape
        groups = list(dict.fromkeys(r["group"] for r in records))
        markers_by_group = {g: [r for r in records if r["group"] == g] for g in groups}
        return {"groups": groups, "markers": markers_by_group}

    ipc_bytes = dataframe_to_arrow_ipc(df)
    return Response(content=ipc_bytes, media_type=ARROW_CONTENT_TYPE)
