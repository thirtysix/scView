"""Custom gene-set collections: upload, list, rename, delete.

A collection is uploaded once (.gmt, or a gene_set,gene .csv/.tsv) and can then
be used by id wherever an MSigDB collection can; see core/custom_genesets.py.
"""

from __future__ import annotations

import logging
from pathlib import Path

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from fastapi.responses import Response
from pydantic import BaseModel

from scview.config import Settings
from scview.core.custom_genesets import (
    MAX_FILE_BYTES,
    CustomCollectionError,
    GeneSetFileError,
    get_custom_store,
    is_custom_id,
    parse_gene_set_file,
)
from scview.core.dataset_manager import DatasetManager
from scview.core.geneset_sources import custom_collections_for_dataset
from scview.dependencies import get_dataset_manager, get_settings_dep

logger = logging.getLogger(__name__)

router = APIRouter()


class RenameCollectionRequest(BaseModel):
    name: str


def _not_found(cid: str) -> HTTPException:
    return HTTPException(status_code=404, detail=f"Custom collection '{cid}' not found.")


@router.get("/genesets/custom")
async def list_custom_collections(settings: Settings = Depends(get_settings_dep)):
    """Every custom collection (metadata only)."""
    return {"collections": get_custom_store(settings.DATA_DIR).list()}


@router.get("/datasets/{dataset_id}/genesets/custom")
async def list_custom_collections_for_dataset(
    dataset_id: str,
    dm: DatasetManager = Depends(get_dataset_manager),
    settings: Settings = Depends(get_settings_dep),
):
    """Custom collections with how many of their genes this dataset measures."""
    adaptor = await dm.get_or_load_dataset(dataset_id)
    if adaptor is None:
        raise HTTPException(status_code=404, detail="Dataset not found.")
    store = get_custom_store(settings.DATA_DIR)
    return {
        "dataset_id": dataset_id,
        "collections": custom_collections_for_dataset(adaptor, store),
    }


@router.post("/genesets/custom/upload", status_code=201)
async def upload_custom_collection(
    file: UploadFile = File(...),
    name: str = Form(default=""),
    settings: Settings = Depends(get_settings_dep),
):
    """Save a .gmt / .csv / .tsv file as a named custom collection.

    Nothing is saved unless the whole file is valid; a 400 lists what was wrong.
    """
    filename = file.filename or ""
    chunks: list[bytes] = []
    total = 0
    while chunk := await file.read(1024 * 1024):
        total += len(chunk)
        if total > MAX_FILE_BYTES:
            raise HTTPException(
                status_code=413,
                detail=f"File exceeds the {MAX_FILE_BYTES // (1024 * 1024)} MB limit for gene-set files.",
            )
        chunks.append(chunk)

    try:
        parsed = parse_gene_set_file(b"".join(chunks), filename)
    except GeneSetFileError as e:
        raise HTTPException(status_code=400, detail={"message": e.message, "problems": e.problems})

    display_name = name.strip() or Path(filename).stem.strip() or "Custom collection"
    store = get_custom_store(settings.DATA_DIR)
    try:
        meta = store.create(display_name, parsed, source_filename=filename)
    except CustomCollectionError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {**meta, "warnings": parsed.warnings}


@router.patch("/genesets/custom/{collection_id}")
async def rename_custom_collection(
    collection_id: str,
    body: RenameCollectionRequest,
    settings: Settings = Depends(get_settings_dep),
):
    """Rename a custom collection (its id, and so cached results, are unchanged)."""
    if not is_custom_id(collection_id):
        raise _not_found(collection_id)
    store = get_custom_store(settings.DATA_DIR)
    try:
        meta = store.rename(collection_id, body.name)
    except CustomCollectionError as e:
        raise HTTPException(status_code=400, detail=str(e))
    if meta is None:
        raise _not_found(collection_id)
    return meta


@router.delete("/genesets/custom/{collection_id}", status_code=204)
async def delete_custom_collection(
    collection_id: str,
    settings: Settings = Depends(get_settings_dep),
):
    """Delete a custom collection.

    Cached enrichment results that used it are left in place and from then on
    show the collection as deleted; asking to compute with it again is a 400.
    """
    if not is_custom_id(collection_id):
        raise _not_found(collection_id)
    if not get_custom_store(settings.DATA_DIR).delete(collection_id):
        raise _not_found(collection_id)
    return Response(status_code=204)
