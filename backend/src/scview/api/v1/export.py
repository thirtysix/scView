"""Export endpoints – Phase 5.

Provides data export in CSV and Excel formats for markers, metadata, and
expression subsets.
"""

from __future__ import annotations

import io
import logging
from typing import Any

import numpy as np
import pandas as pd
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from scview.core.dataset_manager import DatasetManager
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
# Endpoints
# ---------------------------------------------------------------------------


@router.post("/datasets/{dataset_id}/export")
async def export_dataset(
    dataset_id: str,
    body: ExportRequest,
    dm: DatasetManager = Depends(get_dataset_manager),
):
    """Export dataset data as a downloadable CSV or Excel file.

    Supported ``data_type`` values:

    * **markers** – marker gene table (optionally filtered by group via
      ``params.group``)
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

    # ---- Build the DataFrame to export ----
    df: pd.DataFrame

    if data_type == "markers":
        if not adaptor.has_markers():
            raise HTTPException(
                status_code=404,
                detail="No marker genes found in this dataset.",
            )
        group = params.get("group", None)
        df = adaptor.get_markers(groupby=group if group else None)
        if df is None or df.empty:
            raise HTTPException(
                status_code=404,
                detail="No marker genes found for the specified group.",
            )
        filename_base = f"markers_{group or 'all'}_{dataset_id}"

    elif data_type == "metadata":
        df = adaptor.adata.obs.copy()
        # Convert categorical columns to strings for cleaner export
        for col in df.columns:
            if hasattr(df[col], "cat"):
                df[col] = df[col].astype(str)
        filename_base = f"metadata_{dataset_id}"

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
        expr = adaptor.get_expression(genes)
        var_names = adaptor.adata.var_names.tolist()
        found_genes = [g for g in genes if g in var_names]

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
        filename_base = f"expression_{dataset_id}"

    else:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Unknown data_type '{data_type}'.  "
                "Supported: 'markers', 'metadata', 'expression'."
            ),
        )

    # ---- Serialise to bytes ----
    buf = io.BytesIO()

    if fmt == "csv":
        df.to_csv(buf, index=True)
        buf.seek(0)
        media_type = "text/csv; charset=utf-8"
        filename = f"{filename_base}.csv"
    else:
        df.to_excel(buf, index=True, engine="openpyxl")
        buf.seek(0)
        media_type = (
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
        filename = f"{filename_base}.xlsx"

    return StreamingResponse(
        buf,
        media_type=media_type,
        headers={
            "Content-Disposition": f'attachment; filename="{filename}"',
        },
    )
