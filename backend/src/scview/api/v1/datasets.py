"""Dataset management endpoints – upload, list, get info, delete."""

import json
import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File

from scview.config import Settings
from scview.dependencies import get_settings_dep, get_dataset_manager
from scview.core.dataset_manager import DatasetManager
from scview.core.conversion import trigger_conversion
from scview.models.enums import DatasetStatus
from scview.models.schemas import DatasetUploadResponse

logger = logging.getLogger(__name__)

router = APIRouter()

ALLOWED_EXTENSIONS = {".h5ad", ".rds", ".rdata"}


def _suffix(filename: str) -> str:
    return Path(filename).suffix.lower()


def _make_readonly(path: Path) -> None:
    """Best-effort: mark an ingested source file read-only.

    Defense-in-depth so an unexpected process/bug surfaces a loud
    PermissionError instead of silently overwriting an irreplaceable source.
    All legitimate processing writes to the separate `derived/` layer, so the
    original never needs to be rewritten. Failures (e.g. cross-container
    ownership) are logged and ignored — the `derived/` redirect is the
    primary guarantee.
    """
    try:
        path.chmod(0o444)
    except OSError as e:
        logger.warning("Could not mark %s read-only: %s", path, e)


@router.post("/datasets/upload", response_model=DatasetUploadResponse, status_code=201)
async def upload_dataset(
    file: UploadFile = File(...),
    settings: Settings = Depends(get_settings_dep),
    dm: DatasetManager = Depends(get_dataset_manager),
):
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided.")

    ext = _suffix(file.filename)
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{ext}'. Allowed: {', '.join(sorted(ALLOWED_EXTENSIONS))}",
        )

    dataset_id = str(uuid.uuid4())
    upload_dir = Path(settings.DATA_DIR) / "uploads" / dataset_id
    upload_dir.mkdir(parents=True, exist_ok=True)

    dest_path = upload_dir / file.filename
    total_bytes = 0
    max_bytes = settings.max_upload_bytes

    with open(dest_path, "wb") as fh:
        while chunk := await file.read(1024 * 1024):
            total_bytes += len(chunk)
            if total_bytes > max_bytes:
                fh.close()
                dest_path.unlink(missing_ok=True)
                raise HTTPException(
                    status_code=413,
                    detail=f"File exceeds maximum upload size of {settings.MAX_UPLOAD_SIZE_MB} MB.",
                )
            fh.write(chunk)

    # Protect the user's original upload — never written to again.
    _make_readonly(dest_path)

    needs_conversion = ext in (".rds", ".rdata")
    status = DatasetStatus.converting if needs_conversion else DatasetStatus.ready

    meta = {
        "id": dataset_id,
        "name": Path(file.filename).stem,
        "filename": file.filename,
        "extension": ext,
        "size_bytes": total_bytes,
        "status": status.value,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "n_cells": None,
        "n_genes": None,
        "available_embeddings": [],
        "obs_columns": [],
    }
    meta_path = upload_dir / "metadata.json"
    with open(meta_path, "w") as fh:
        json.dump(meta, fh, indent=2)

    if needs_conversion:
        converted_dir = Path(settings.DATA_DIR) / "converted" / dataset_id
        converted_dir.mkdir(parents=True, exist_ok=True)
        output_path = converted_dir / f"{Path(file.filename).stem}.h5ad"
        try:
            await trigger_conversion(
                rds_path=str(dest_path),
                output_path=str(output_path),
                converter_url=settings.CONVERTER_URL,
            )
            meta["status"] = DatasetStatus.ready.value
            with open(meta_path, "w") as fh:
                json.dump(meta, fh, indent=2)
            # Conversion output is also a regenerate-able source — keep it
            # immutable; processing writes to the derived/ layer instead.
            _make_readonly(output_path)
        except Exception as e:
            meta["status"] = DatasetStatus.error.value
            meta["error_message"] = str(e)
            with open(meta_path, "w") as fh:
                json.dump(meta, fh, indent=2)

    # Auto-load the dataset to populate metadata (cell/gene counts, embeddings)
    if meta["status"] == DatasetStatus.ready.value:
        await dm.load_dataset(dataset_id)

    return DatasetUploadResponse(id=dataset_id, name=meta["name"], status=meta["status"])


@router.get("/datasets")
async def list_datasets(
    dm: DatasetManager = Depends(get_dataset_manager),
):
    return dm.list_datasets()


@router.post("/datasets/prune")
async def prune_datasets(
    dm: DatasetManager = Depends(get_dataset_manager),
):
    """Remove dataset entries that can't be opened (failed conversions / missing data)."""
    removed = dm.prune_orphans()
    return {"removed": removed, "count": len(removed)}


@router.get("/datasets/{dataset_id}")
async def get_dataset_info(
    dataset_id: str,
    dm: DatasetManager = Depends(get_dataset_manager),
):
    info = dm.get_dataset_info(dataset_id)
    if info is None:
        raise HTTPException(status_code=404, detail="Dataset not found.")

    # Always ensure adaptor is loaded for ready datasets
    # (needed after restart when metadata.json has n_cells but adaptor is not in memory)
    adaptor = dm.get_dataset(dataset_id)
    if adaptor is None and info.get("status") == "ready":
        await dm.load_dataset(dataset_id)
        info = dm.get_dataset_info(dataset_id)
        adaptor = dm.get_dataset(dataset_id)
    if adaptor is not None:
        embs = adaptor.available_embeddings()
        info["obs_columns"] = adaptor.obs_columns_info()
        info["active_clustering"] = adaptor.active_clustering_column()
        info["available_embeddings"] = [e["name"] for e in embs]
        info["embedding_dimensions"] = {e["name"]: e["dimensions"] for e in embs}
        info["n_cells"] = adaptor.n_cells()
        info["n_genes"] = adaptor.n_genes()
        info["expression_layers"] = adaptor.available_expression_layers()
        info["default_expression_layer"] = adaptor.default_expression_layer()

    return info


@router.delete("/datasets/{dataset_id}", status_code=204)
async def delete_dataset(
    dataset_id: str,
    dm: DatasetManager = Depends(get_dataset_manager),
):
    removed = dm.remove_dataset(dataset_id)
    if not removed:
        raise HTTPException(status_code=404, detail="Dataset not found.")
    return None
