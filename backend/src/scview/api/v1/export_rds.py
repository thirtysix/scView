"""Download a dataset as an R object: a Seurat object or a SingleCellExperiment (.rds).

``GET  /datasets/{id}/export/rds/info?format=seurat|sce`` what the object will hold, what
is left out, and whether the converter has the memory for it.
``POST /datasets/{id}/export/rds?format=seurat|sce`` starts the export (202, a job): the
backend writes a staging h5ad under the dataset's edit lock, the R converter builds and
saves the object (``core/rds_export.py``, ``converter/export_rds.R``).
``GET  /datasets/{id}/export/rds/{job}/status`` the job's stage, then its report.
``GET  /datasets/{id}/export/rds/{job}`` the finished .rds, sent once and then removed
(unfetched files are removed after an hour).
"""

from __future__ import annotations

import asyncio
import logging

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse
from starlette.background import BackgroundTask

from scview.config import Settings
from scview.core import cellxgene_export as cx
from scview.core import large_mode, rds_export
from scview.core.dataset_manager import DatasetManager
from scview.core.obs_writes import dataset_edit
from scview.dependencies import get_dataset_manager, get_settings_dep

logger = logging.getLogger(__name__)

router = APIRouter()


def _fmt(fmt: str) -> str:
    if fmt not in rds_export.FORMATS:
        raise HTTPException(status_code=400, detail=f"Unknown R format '{fmt}': use seurat or sce.")
    return fmt


def _name(dm: DatasetManager, dataset_id: str) -> str:
    return (dm.get_dataset_info(dataset_id) or {}).get("name") or dataset_id


def _download_name(dm: DatasetManager, dataset_id: str, fmt: str) -> str:
    from scview.api.v1.export import file_stem

    return f"scview_{file_stem(_name(dm, dataset_id), dataset_id)}{rds_export.FORMATS[fmt]['suffix']}"


async def _adaptor(dm: DatasetManager, dataset_id: str):
    if dm.get_dataset_info(dataset_id) is None:
        raise HTTPException(status_code=404, detail="Dataset not found.")
    adaptor = await dm.get_or_load_dataset(dataset_id)
    if adaptor is None:
        raise HTTPException(status_code=404, detail="Dataset not found.")
    return adaptor


async def _plan(dm: DatasetManager, dataset_id: str, fmt: str):
    adaptor = await _adaptor(dm, dataset_id)
    try:
        return await asyncio.to_thread(rds_export.plan, adaptor, fmt, _name(dm, dataset_id), dataset_id)
    except (rds_export.ExportError, cx.ExportError) as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except large_mode.LargeDatasetError as exc:
        raise HTTPException(status_code=422, detail=str(exc))


def _too_big(mem: dict) -> str:
    gb = 1024 ** 3
    return (f"Building this R object needs about {mem['needs_bytes'] / gb:.1f} GB in the R converter, "
            f"more than its {mem['limit_bytes'] / gb:.1f} GB limit. Raise SCVIEW_MEMORY_GB, or subset "
            "the dataset first, or download the .h5ad and read it in R with zellkonverter or "
            "SeuratDisk.")


@router.get("/datasets/{dataset_id}/export/rds/info")
async def rds_export_info(
    dataset_id: str,
    format: str = "seurat",
    dm: DatasetManager = Depends(get_dataset_manager),
    settings: Settings = Depends(get_settings_dep),
):
    """What the object will hold and leave out, its memory estimate, the file name, and
    whether the converter is running (``converter``: ok | busy | down)."""
    fmt = _fmt(format)
    p = await _plan(dm, dataset_id, fmt)
    out = await asyncio.to_thread(rds_export.summary, p, fmt)
    del p
    out["filename"] = _download_name(dm, dataset_id, fmt)
    out["converter"] = await rds_export.converter_state(settings.CONVERTER_URL)
    if not out["fits"]:
        out["too_big"] = _too_big(out)
    running = rds_export.running_for(dataset_id, fmt)
    out["running_job"] = running.id if running else None
    return out


@router.post("/datasets/{dataset_id}/export/rds", status_code=202)
async def start_rds_export(
    dataset_id: str,
    format: str = "seurat",
    dm: DatasetManager = Depends(get_dataset_manager),
    settings: Settings = Depends(get_settings_dep),
):
    """Start writing the .rds; answers at once with the job (poll ``/status``)."""
    fmt = _fmt(format)
    rds_export.sweep(dm.data_dir)
    p = await _plan(dm, dataset_id, fmt)
    mem = rds_export.memory_check(p)
    del p
    if not mem["fits"]:
        raise HTTPException(status_code=507, detail=_too_big(mem))
    if rds_export.running_for(dataset_id, fmt) is not None:
        raise HTTPException(status_code=409, detail="This export is already being written.")
    state = await rds_export.converter_state(settings.CONVERTER_URL)
    if state == "down":
        raise HTTPException(status_code=503, detail=rds_export.CONVERTER_DOWN)

    name = _name(dm, dataset_id)
    job = rds_export.new_job(dm.data_dir, dataset_id, fmt, _download_name(dm, dataset_id, fmt))
    if state == "busy":
        job.message = "Waiting for the R converter (it is converting another file)"

    async def write() -> dict:
        note = cx.provenance(name, dataset_id, fmt)
        async with dataset_edit(dataset_id) as held:  # one consistent state of the dataset
            adaptor = await dm.get_or_load_dataset(dataset_id)
            if adaptor is None:
                raise rds_export.ExportError("Dataset not found.")

            def work(a) -> dict:
                plan = rds_export.plan(a, fmt, name, dataset_id)
                rds_export.write_staging(plan, job.folder / rds_export.STAGING, note,
                                         progress=lambda m: setattr(job, "message", m))
                return plan.meta

            task = held.run(work, adaptor)
            del adaptor  # a reload meanwhile must not keep this copy alive
            return await task

    rds_export.start(job, rds_export.run(job, write, settings.CONVERTER_URL))
    return job.to_dict()


def _job(dataset_id: str, job_id: str) -> rds_export.Job:
    job = rds_export.get_job(job_id)
    if job is None or job.dataset_id != dataset_id:
        raise HTTPException(status_code=404, detail=(
            "This export is gone: a finished file is kept for an hour, and only until it is "
            "downloaded. Start the export again."))
    return job


@router.get("/datasets/{dataset_id}/export/rds/{job_id}/status")
async def rds_export_status(dataset_id: str, job_id: str):
    """``status``: writing | converting | done (with ``report``, ``size_bytes``) | error."""
    return _job(dataset_id, job_id).to_dict()


@router.get("/datasets/{dataset_id}/export/rds/{job_id}")
async def download_rds(dataset_id: str, job_id: str):
    """The finished .rds, streamed from disk; removed once sent."""
    job = _job(dataset_id, job_id)
    if job.status == "error":
        raise HTTPException(status_code=409, detail=job.error or "The export failed.")
    if job.status != "done":
        raise HTTPException(status_code=409, detail="The .rds is still being written.")
    if not job.output.is_file():
        raise HTTPException(status_code=404, detail="The .rds file is gone; start the export again.")
    return FileResponse(
        job.output, media_type="application/octet-stream", filename=job.filename,
        headers={"Content-Encoding": "identity"},  # gzip-compressed already: no re-gzip
        background=BackgroundTask(rds_export.remove, job),
    )
