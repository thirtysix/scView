"""TCR / BCR contigs (``core/vdj.py``): attach a contig table, and the repertoire views.

``POST /vdj`` stores the uploaded table beside the dataset's originals
(``uploads/{id}/vdj/``, read-only), joins it to the cells and saves the dataset;
``POST /vdj/clonotypes`` recomputes the clonotypes from the stored chains (after
cells were removed, or with V genes in the definition); ``DELETE /vdj`` removes
every V(D)J column. The GET endpoints read the saved columns only: no matrix.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import os
import re
from pathlib import Path
from typing import Any, Callable

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from pydantic import BaseModel

from scview.core import vdj
from scview.core.dataset_manager import DatasetManager
from scview.core.obs_writes import dataset_edit, uns_snapshot, write_derived
from scview.dependencies import get_dataset_manager

logger = logging.getLogger(__name__)

router = APIRouter()

_SAFE = re.compile(r"[^A-Za-z0-9._-]+")


async def _adaptor(dm: DatasetManager, dataset_id: str):
    adaptor = await dm.get_or_load_dataset(dataset_id)
    if adaptor is None:
        raise HTTPException(status_code=404, detail="Dataset not found.")
    return adaptor


async def _read(dm: DatasetManager, dataset_id: str, fn: Callable[[Any], dict]) -> dict:
    adaptor = await _adaptor(dm, dataset_id)
    try:
        return await asyncio.to_thread(fn, adaptor.adata)
    except vdj.VdjError as e:
        raise HTTPException(status_code=e.status, detail=str(e))


async def _edit(dm: DatasetManager, dataset_id: str, fn: Callable[[Any], dict]) -> dict:
    """Run ``fn`` on the dataset under its edit lock, save obs, uns and the chain
    table, and answer with the new summary."""
    async with dataset_edit(dataset_id) as held:
        adaptor = await _adaptor(dm, dataset_id)
        output_path = str(dm.derived_h5ad_path(dataset_id, adaptor.h5ad_path))
        backed = getattr(adaptor.adata, "isbacked", False)

        def work() -> dict:
            data = adaptor.adata.to_memory() if backed else adaptor.adata
            before = uns_snapshot(data)
            result = fn(data)
            write_derived(data, output_path, adaptor.h5ad_path, before, obsm_keys=(vdj.CHAINS_KEY,))
            return {**result, "summary": vdj.summary(data)}

        try:
            out = await held.run(work)
        except vdj.VdjError as e:
            raise HTTPException(status_code=e.status, detail=str(e))  # raised before any change
        except Exception as e:
            dm.evict(dataset_id)  # the cached copy may hold columns that were never saved
            logger.error("V(D)J edit failed for %s: %s", dataset_id, e, exc_info=True)
            raise HTTPException(status_code=500, detail=f"Could not save the V(D)J data: {e!s}")
        if backed:
            dm.evict(dataset_id)
            await dm.load_dataset(dataset_id)
        else:
            dm.saved_in_place(dataset_id, adaptor, output_path)
        return out


@router.get("/datasets/{dataset_id}/vdj")
async def get_vdj(dataset_id: str, dm: DatasetManager = Depends(get_dataset_manager)):
    """What is attached, what matched both ways (cells without contigs, contig
    barcodes without cells), chain status, clonotypes; how to attach when nothing is."""
    return await _read(dm, dataset_id, vdj.summary)


async def _store_upload(dm: DatasetManager, dataset_id: str, file: UploadFile) -> tuple[Path, str, str, bool]:
    """Stream the upload into ``uploads/{id}/vdj/`` (hashed name, read-only):
    (path, original name, sha256, newly written)."""
    folder = Path(dm.data_dir) / "uploads" / dataset_id / "vdj"
    folder.mkdir(parents=True, exist_ok=True)
    name = Path(file.filename or "contigs.csv").name
    partial = folder / f".upload-{os.getpid()}-{id(file)}.partial"
    digest = hashlib.sha256()
    total = 0
    try:
        with open(partial, "wb") as fh:
            while chunk := await file.read(1024 * 1024):
                total += len(chunk)
                if total > vdj.MAX_UPLOAD_BYTES:
                    raise HTTPException(status_code=413, detail=(
                        f"File exceeds the {vdj.MAX_UPLOAD_BYTES // (1024 * 1024)} MB limit for contig tables."))
                digest.update(chunk)
                fh.write(chunk)
    except BaseException:
        partial.unlink(missing_ok=True)
        raise
    sha = digest.hexdigest()
    dest = folder / f"{sha[:12]}_{_SAFE.sub('_', name).strip('_') or 'contigs'}"
    if dest.exists():
        partial.unlink(missing_ok=True)
        return dest, name, sha, False
    os.replace(partial, dest)
    try:
        dest.chmod(0o444)
    except OSError:
        pass
    return dest, name, sha, True


@router.post("/datasets/{dataset_id}/vdj")
async def attach_vdj(
    dataset_id: str,
    file: UploadFile = File(...),
    sample_column: str = Form(default=""),
    sample: str = Form(default=""),
    replace: bool = Form(default=False),
    same_v_gene: str = Form(default=""),
    dm: DatasetManager = Depends(get_dataset_manager),
):
    """Attach a Cell Ranger contig annotations CSV or an AIRR rearrangement TSV.
    ``sample_column`` + ``sample``: the sample the contigs belong to, when the
    dataset holds several (``__barcode_suffix__`` names the cells' barcode
    suffix). ``replace``: drop that sample's earlier contigs first.
    ``same_v_gene`` (true/false; empty keeps the dataset's setting)."""
    await _adaptor(dm, dataset_id)
    path, name, sha, new = await _store_upload(dm, dataset_id, file)
    svg = None if same_v_gene == "" else same_v_gene.lower() in ("1", "true", "yes")

    def fn(adata) -> dict:
        return {"attached": vdj.attach(adata, path, file_name=name, sha256=sha,
                                       sample_column=sample_column or None, sample=sample or None,
                                       replace=replace, same_v_gene=svg, stored_as=path.name)}

    try:
        return await _edit(dm, dataset_id, fn)
    except HTTPException:
        if new:
            path.unlink(missing_ok=True)
        raise


class DefinitionRequest(BaseModel):
    same_v_gene: bool = False


@router.post("/datasets/{dataset_id}/vdj/clonotypes")
async def redefine_clonotypes(
    dataset_id: str,
    body: DefinitionRequest,
    dm: DatasetManager = Depends(get_dataset_manager),
):
    """Recompute clonotypes, clone sizes and expansion from the stored chains:
    after cells were removed, or with (``same_v_gene``) the V genes required too."""
    return await _edit(dm, dataset_id, lambda adata: vdj.set_definition(adata, body.same_v_gene))


@router.delete("/datasets/{dataset_id}/vdj")
async def remove_vdj(dataset_id: str, dm: DatasetManager = Depends(get_dataset_manager)):
    """Remove every V(D)J column, the chain table and the attach record (the
    stored contig files stay beside the originals)."""
    return await _edit(dm, dataset_id, vdj.remove)


def _group(adata, group_by: str, kind: str) -> str:
    col = group_by or vdj.default_group(adata, kind)
    if not col:
        raise vdj.VdjError("Name a column to group the cells by (group_by).")
    return col


@router.get("/datasets/{dataset_id}/vdj/clonotypes")
async def get_clonotypes(dataset_id: str, group_by: str = "", group: str | None = None, n: int = 25,
                         dm: DatasetManager = Depends(get_dataset_manager)):
    """The largest clonotypes (within ``group`` of ``group_by`` when named), their
    chains, cells per group and cell indices. Default ``group_by``: the cell types,
    else the clustering."""
    return await _read(dm, dataset_id, lambda a: vdj.clonotypes(
        a, group_by=group_by or vdj.default_group(a, "celltype"), group=group, n=n))


@router.get("/datasets/{dataset_id}/vdj/diversity")
async def get_diversity(dataset_id: str, group_by: str = "", rarefy: bool = False, seed: int = 0,
                        dm: DatasetManager = Depends(get_dataset_manager)):
    """Richness, Shannon, Gini-Simpson, clonality and Gini per group; ``rarefy``
    subsamples every group to the smallest (seeded). Default ``group_by``: samples."""
    return await _read(dm, dataset_id, lambda a: vdj.diversity(
        a, _group(a, group_by, "sample"), rarefy=rarefy, seed=seed))


@router.get("/datasets/{dataset_id}/vdj/overlap")
async def get_overlap(dataset_id: str, group_by: str = "", metric: str = "morisita_horn",
                      dm: DatasetManager = Depends(get_dataset_manager)):
    """Clonotype overlap between groups (Morisita-Horn or Jaccard)."""
    return await _read(dm, dataset_id, lambda a: vdj.overlap(a, _group(a, group_by, "sample"), metric))


@router.get("/datasets/{dataset_id}/vdj/usage")
async def get_usage(dataset_id: str, group_by: str = "", chain: str = "", gene: str = "v",
                    count: str = "clonotypes", dm: DatasetManager = Depends(get_dataset_manager)):
    """V or J gene usage of one locus (``chain``, default the commonest of TRB / IGH)
    per group, over clonotypes (each once) or cells."""
    return await _read(dm, dataset_id, lambda a: vdj.usage(
        a, _group(a, group_by, "sample"), chain or None, gene, count))
