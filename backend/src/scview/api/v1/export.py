"""Export endpoints – Phase 5.

Provides data export in CSV and Excel formats for markers, metadata, and
expression subsets, and a download of the dataset's own .h5ad file.
"""

from __future__ import annotations

import asyncio
import io
import logging
import re
from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel

from scview.api.v1.markers import ensure_specificity
from scview.core import large_layout, large_mode
from scview.core.dataset_manager import DatasetManager
from scview.core.obs_writes import dataset_edit
from scview.dependencies import get_dataset_manager

logger = logging.getLogger(__name__)

router = APIRouter()


# ---------------------------------------------------------------------------
# Request model
# ---------------------------------------------------------------------------


class ExportRequest(BaseModel):
    """Body for the export endpoint."""

    format: str = "csv"  # "csv" | "xlsx"
    data_type: str = "markers"  # "markers" | "metadata" | "expression"
    params: dict[str, Any] = {}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Excel's hard row limit (1,048,576) minus the header row.
_XLSX_MAX_ROWS = 1_048_575
# Cells (rows x columns) one Excel export may hold. openpyxl writes ~100k cells a
# second (50 genes x Kang's 13,836 cells: 7 s, +0.27 GB), so 2M is ~20 s; a
# 500-gene expression table of Kang (7M) was over a minute. CSV has no cap.
XLSX_MAX_CELLS = 2_000_000


def file_stem(text: str | None, fallback: str = "dataset") -> str:
    """Filename-safe stem for a dataset or column name ('PBMC 3k (v2)' -> 'PBMC_3k_v2').

    Mirrors ``exportFilename`` in the frontend so plots, tables and the .h5ad
    from one dataset share a prefix."""
    stem = re.sub(r"_{2,}", "_", re.sub(r"[^A-Za-z0-9._-]+", "_", text or ""))
    stem = stem.strip("._-")[:80].rstrip("._-")
    return stem or fallback


def _sheet_name(name: str, taken: set[str]) -> str:
    """An Excel sheet name: at most 31 characters, none of []:*?/\\, unique."""
    base = re.sub(r"[\[\]:*?/\\]", "_", name)[:31] or "markers"
    out, n = base, 2
    while out.lower() in taken:
        suffix = f"_{n}"
        out, n = base[: 31 - len(suffix)] + suffix, n + 1
    taken.add(out.lower())
    return out


_H5AD_SOURCES = {
    "derived": "processed",
    "ingested": "imported",
    "converted": "converted from R (.rds)",
    "uploads": "as uploaded",
}


def _active_h5ad(dm: DatasetManager, dataset_id: str) -> Path:
    """The .h5ad scView reads for a dataset (processed output when there is one)."""
    if dm.get_dataset_info(dataset_id) is None:
        raise HTTPException(status_code=404, detail="Dataset not found.")
    candidates = dm.h5ad_candidates(dataset_id)
    if not candidates:
        raise HTTPException(status_code=404, detail="This dataset has no .h5ad file yet.")
    return Path(candidates[0])


def _h5ad_download_name(dm: DatasetManager, dataset_id: str) -> str:
    info = dm.get_dataset_info(dataset_id) or {}
    return f"scview_{file_stem(info.get('name'), dataset_id)}.h5ad"


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.get("/datasets/{dataset_id}/export/h5ad/info")
async def h5ad_download_info(
    dataset_id: str,
    raw: bool = False,
    format: str = "h5ad",
    embedding: list[str] | None = Query(None),
    column: list[str] | None = Query(None),
    genes: str = "all",
    dm: DatasetManager = Depends(get_dataset_manager),
):
    """Size and name of the .h5ad a download would send, so the UI can show the
    size before the user starts a multi-GB transfer. A large dataset's download is
    one file put together from its h5ad and counts (``large``; the size is close, not
    exact). ``format=cellxgene|vitessce``: the export for that viewer with these
    options instead (``export_formats.format_info``; ``raw`` = include raw counts)."""
    if format != "h5ad":
        from scview.api.v1.export_formats import FormatOptions, format_info

        return await format_info(dm, dataset_id, format, FormatOptions(
            embeddings=embedding, columns=column, include_raw=raw, genes=genes))
    path = _active_h5ad(dm, dataset_id)
    layer = path.parent.parent.name
    out = {
        "filename": _h5ad_download_name(dm, dataset_id),
        "size_bytes": path.stat().st_size,
        "source": layer,
        "source_label": _H5AD_SOURCES.get(layer, layer),
    }
    if await asyncio.to_thread(large_mode.file_is_large, path):
        adaptor = await dm.get_or_load_dataset(dataset_id)
        counts = _large_counts_path(adaptor)
        out["size_bytes"] = await asyncio.to_thread(large_mode.export_bytes, path, counts, raw)
        out["large"] = True
        out["with_counts"] = counts is not None
    return out


def _large_counts_path(adaptor) -> Path | None:
    """The counts sidecar to put in an export, or None when it cannot be used."""
    try:
        adaptor.counts_matrix("Exporting the counts")
    except large_mode.LargeDatasetError:
        return None
    return large_layout.counts_path(adaptor.h5ad_path, adaptor.adata.uns)


def _export_fresh(out: Path, sources: list[Path]) -> bool:
    try:
        built = out.stat().st_mtime_ns
    except OSError:
        return False
    return all(src.stat().st_mtime_ns <= built for src in sources if src.exists())


async def _large_export(dm: DatasetManager, dataset_id: str, raw: bool) -> Path:
    """A large dataset's self-contained h5ad (X, layers['counts'], optionally .raw = X),
    put together from its files chunk by chunk (``large_mode.export_h5ad``, nothing is
    read into memory) in the dataset's cache folder, and kept until its files change."""
    async with dataset_edit(dataset_id) as held:  # no in-place edit while it is copied
        adaptor = await dm.get_or_load_dataset(dataset_id)
        if adaptor is None:
            raise HTTPException(status_code=404, detail="Dataset not found.")
        counts = _large_counts_path(adaptor)
        out = dm.data_dir / "cache" / dataset_id / ("export_raw.h5ad" if raw else "export.h5ad")
        sources = [Path(adaptor.h5ad_path)] + ([counts] if counts is not None else [])
        if not _export_fresh(out, sources):
            out.parent.mkdir(parents=True, exist_ok=True)
            await held.run(lambda: large_mode.export_h5ad(
                adaptor.adata, adaptor.h5ad_path, counts, out, raw=raw))
        return out


@router.get("/datasets/{dataset_id}/export/h5ad")
async def download_h5ad(
    dataset_id: str,
    raw: bool = False,
    with_results: bool = False,
    dm: DatasetManager = Depends(get_dataset_manager),
):
    """Stream the dataset's active .h5ad (processing results included) from disk.
    ``with_results``: a zip of the h5ad beside a ``results/`` folder of the stored
    results as .tsv.gz and their ``manifest.json`` (core/results_export.py).

    Every write path saves the file before it returns (pipeline steps evict the
    cache afterwards), so the file on disk is what the app shows. It is sent in
    chunks; nothing is loaded into memory. A large dataset keeps its counts in a
    sidecar, so its download is one h5ad put together from both first
    (``_large_export``; ``raw`` adds .raw = X)."""
    path = _active_h5ad(dm, dataset_id)
    if await asyncio.to_thread(large_mode.file_is_large, path):
        path = await _large_export(dm, dataset_id, raw)
    if with_results:
        from scview.core import results_export

        name = _h5ad_download_name(dm, dataset_id)
        out = dm.data_dir / "cache" / dataset_id / "export_with_results.zip"
        out.parent.mkdir(parents=True, exist_ok=True)
        await asyncio.to_thread(results_export.results_zip, dataset_id, out, dm.get_dataset(dataset_id),
                                path, name)
        return FileResponse(out, media_type="application/zip", filename=name[:-len(".h5ad")] + "_with_results.zip",
                            headers={"Content-Encoding": "identity"})
    return FileResponse(
        path,
        media_type="application/x-hdf5",
        filename=_h5ad_download_name(dm, dataset_id),
        # A set Content-Encoding makes the app's GZipMiddleware pass the body
        # through: re-gzipping a multi-GB, already-compressed file at level 9
        # would tie up the single worker and drop Content-Length (no progress bar).
        headers={"Content-Encoding": "identity"},
    )


@router.post("/datasets/{dataset_id}/export")
async def export_dataset(
    dataset_id: str,
    body: ExportRequest,
    dm: DatasetManager = Depends(get_dataset_manager),
):
    """Export dataset data as a downloadable CSV or Excel file.

    Supported ``data_type`` values:

    * **markers** – marker gene table for the obs column in ``params.column``
      (every column with stored markers when omitted; one sheet each in
      Excel), optionally filtered to one group via ``params.group``; at most
      ``params.n_genes`` (default 100) genes per group
    * **metadata** – the full obs DataFrame (cell metadata)
    * **expression** – expression matrix for selected genes (pass
      ``params.genes`` as a list of gene names)
    """
    adaptor = await dm.get_or_load_dataset(dataset_id)
    if adaptor is None:
        raise HTTPException(status_code=404, detail="Dataset not found.")

    fmt = body.format.lower()
    if fmt not in ("csv", "xlsx"):
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported format '{fmt}'.  Use 'csv' or 'xlsx'.",
        )

    data_type = body.data_type.lower()
    params = body.params
    info = dm.get_dataset_info(dataset_id) or {}
    prefix = f"scview_{file_stem(info.get('name'), dataset_id)}"

    # ---- Build the DataFrame to export ----
    df: pd.DataFrame
    # Markers for several columns go to Excel as one sheet per column.
    sheets: dict[str, pd.DataFrame] | None = None
    keep_index = True

    if data_type == "markers":
        if not adaptor.has_markers():
            raise HTTPException(
                status_code=404,
                detail="No marker genes found in this dataset.",
            )
        group = params.get("group", None)
        column = str(params.get("column") or "")
        try:
            n_genes = min(max(int(params.get("n_genes", 100)), 1), 5000)
        except (TypeError, ValueError):
            raise HTTPException(status_code=400, detail="params.n_genes must be a number.")
        if column and not adaptor.has_markers(column=column):
            raise HTTPException(
                status_code=404,
                detail=f"No stored marker genes for '{column}'.",
            )
        # "" is the file's own unkeyed rank_genes_groups (no groupby recorded).
        columns = [column] if column else (adaptor.marker_columns() or [""])
        frames: dict[str, pd.DataFrame] = {}
        for col in columns:
            # Markers saved before the specificity columns get them now, as in the table.
            adaptor = await ensure_specificity(dm, dataset_id, adaptor, col)
            part = adaptor.get_markers(
                groupby=group if group else None, column=col or None, n_genes=n_genes,
            )
            if part is not None and not part.empty:
                frames[col] = part
        if not frames:
            raise HTTPException(
                status_code=404,
                detail="No marker genes found for the specified group.",
            )
        df = pd.concat(
            [part.assign(grouping=col or "markers") for col, part in frames.items()],
            ignore_index=True,
        )
        df = df[["grouping", *[c for c in df.columns if c != "grouping"]]]
        if len(frames) > 1:
            taken: set[str] = set()
            sheets = {_sheet_name(col or "markers", taken): part for col, part in frames.items()}
        keep_index = False  # a row number, not a cell or gene id
        filename_base = f"{prefix}_markers"
        if column:
            filename_base += f"_{file_stem(column, 'column')}"
        if group:
            filename_base += f"_{file_stem(str(group), 'group')}"

    elif data_type == "metadata":
        df = adaptor.adata.obs.copy()
        # Convert categorical columns to strings for cleaner export
        for col in df.columns:
            if hasattr(df[col], "cat"):
                df[col] = df[col].astype(str)
        filename_base = f"{prefix}_metadata"

    elif data_type == "expression":
        genes = params.get("genes", [])
        if not genes or not isinstance(genes, list):
            raise HTTPException(
                status_code=400,
                detail="params.genes must be a non-empty list of gene names.",
            )

        if len(genes) > 500:
            raise HTTPException(
                status_code=400,
                detail="At most 500 genes can be exported at once.",
            )

        # Build expression DataFrame
        # Names from the same layer the values come from (.raw can hold more genes than X).
        layer = adaptor.default_expression_layer()
        present = set(adaptor._var_names_for_layer(layer))
        found_genes = [g for g in genes if g in present]
        expr = adaptor.get_expression(found_genes, layer=layer)

        if expr.shape[1] == 0:
            raise HTTPException(
                status_code=404,
                detail="None of the requested genes were found.",
            )

        df = pd.DataFrame(
            expr,
            index=adaptor.adata.obs_names,
            columns=found_genes,
        )
        filename_base = f"{prefix}_expression"

    else:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Unknown data_type '{data_type}'.  "
                "Supported: 'markers', 'metadata', 'expression'."
            ),
        )

    # ---- Serialise to bytes ----
    # Excel writing is pure Python (openpyxl): in a worker thread, so the server
    # keeps answering meanwhile.
    def _serialise() -> tuple[io.BytesIO, str, str]:
        buf = io.BytesIO()

        if fmt == "csv":
            df.to_csv(buf, index=keep_index)
            buf.seek(0)
            media_type = "text/csv; charset=utf-8"
            filename = f"{filename_base}.csv"
        else:
            if max(len(part) for part in (sheets or {"": df}).values()) > _XLSX_MAX_ROWS:
                raise HTTPException(
                    status_code=400,
                    detail="Too many rows for an Excel sheet; export as CSV instead.",
                )
            n_cells = sum(len(p) * (p.shape[1] + int(keep_index)) for p in (sheets or {"": df}).values())
            if n_cells > XLSX_MAX_CELLS:
                raise HTTPException(
                    status_code=413,
                    detail=(f"This table has {n_cells:,} cells; an Excel export holds at most "
                            f"{XLSX_MAX_CELLS:,}. Export it as CSV instead."),
                )
            if sheets:
                with pd.ExcelWriter(buf, engine="openpyxl") as writer:
                    for name, part in sheets.items():
                        part.to_excel(writer, sheet_name=name, index=False)
            else:
                df.to_excel(buf, index=keep_index, engine="openpyxl")
            buf.seek(0)
            media_type = (
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
            filename = f"{filename_base}.xlsx"
        return buf, media_type, filename

    buf, media_type, filename = await asyncio.to_thread(_serialise)

    return StreamingResponse(
        buf,
        media_type=media_type,
        headers={
            "Content-Disposition": f'attachment; filename="{filename}"',
        },
    )
