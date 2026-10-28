"""Conversion orchestrator – talks to the R-based converter sidecar service.

An uploaded .rds / .RData (a Seurat object, a SingleCellExperiment, any
SummarizedExperiment) is converted in the background: the upload returns at
once with status "converting", because a large object takes longer than a
proxy waits for one response (nginx: 300 s). Meanwhile the converter writes
its stages to a progress file (which class it found, what it is writing), which
GET /datasets/{id} returns; at the end the dataset turns "ready", carrying the
converter's report of what it kept and skipped, or "error" with its sentence.
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import Callable, Coroutine
from pathlib import Path
from typing import Any

import httpx

logger = logging.getLogger(__name__)

PROGRESS_FILE = "conversion_progress.json"

# The converter answers once the whole object is written, which for a large one
# takes many minutes: only connecting and sending are bounded tightly.
READ_TIMEOUT_S = 6 * 3600.0
TIMEOUT = httpx.Timeout(connect=10.0, read=READ_TIMEOUT_S, write=60.0, pool=10.0)

INTERRUPTED = ("The conversion stopped when scView restarted, so it never finished. "
               "Add the file again to convert it.")

_RUNNING: dict[str, asyncio.Task] = {}


class ConversionError(RuntimeError):
    """The converter refused the file or failed; the message is its own sentence."""


async def trigger_conversion(
    rds_path: str,
    output_path: str,
    converter_url: str,
    progress_path: str | None = None,
    transport: httpx.AsyncBaseTransport | None = None,
) -> dict:
    """POST to the converter service to convert an .rds / .RData to h5ad.

    Parameters
    ----------
    rds_path : str
        Absolute path to the uploaded .rds / .rdata file.
    output_path : str
        Absolute path where the resulting .h5ad should be written.
    converter_url : str
        Base URL of the converter sidecar (e.g. http://converter:8001).
    progress_path : str, optional
        Where the converter writes its stages while it runs.
    transport : httpx transport, optional
        For tests.

    Returns
    -------
    dict – the converter's body: ``status``, ``message`` and ``report``.

    Raises
    ------
    ConversionError – with the converter's own reason ("This file holds a
    list object, which scView cannot convert. ..."), or why it was not reached.
    """
    body = {"input_path": rds_path, "output_path": output_path}
    if progress_path:
        body["progress_path"] = progress_path
    try:
        async with httpx.AsyncClient(timeout=TIMEOUT, transport=transport) as client:
            resp = await client.post(f"{converter_url}/convert", json=body)
    except httpx.TimeoutException as e:
        raise ConversionError(
            f"The R converter did not finish within {READ_TIMEOUT_S / 3600:.0f} hours."
        ) from e
    except httpx.TransportError as e:
        raise ConversionError(
            "The R converter service could not be reached "
            f"({type(e).__name__}): is the converter container running?"
        ) from e

    try:
        data = resp.json()
    except ValueError:
        data = None
    if not isinstance(data, dict):
        data = {}
    if resp.status_code >= 400 or data.get("status") != "ok":
        reason = data.get("error")
        raise ConversionError(
            str(reason) if reason else f"The R converter answered {resp.status_code} without saying why."
        )
    return data


def complete_spatial(output_path: str | Path) -> None:
    """A converted Seurat Visium object whose images kept no array grid (VisiumV2, Seurat
    >= 5.1): the grid recovered from the spot centres and saved in the .h5ad
    (``core/seurat_spatial.py``), so it opens as Visium spots. Never raises: the file
    stays as converted when this fails."""
    from scview.core import seurat_spatial

    try:
        if seurat_spatial.complete_converted(output_path):
            logger.info("Visium grid recovered for %s", Path(output_path).name)
    except Exception:  # noqa: BLE001
        logger.exception("Could not complete the spatial data of %s", output_path)


def read_progress(path: Path) -> dict | None:
    """The converter's latest stage, or None before it wrote one."""
    try:
        data = json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def is_running(dataset_id: str) -> bool:
    task = _RUNNING.get(dataset_id)
    return task is not None and not task.done()


def task_for(dataset_id: str) -> asyncio.Task | None:
    return _RUNNING.get(dataset_id)


def start(dataset_id: str, job: Coroutine[Any, Any, None]) -> asyncio.Task:
    """Run a conversion in the background, kept referenced until it ends."""
    task = asyncio.get_running_loop().create_task(job)
    _RUNNING[dataset_id] = task

    def _forget(t: asyncio.Task, did: str = dataset_id) -> None:
        if _RUNNING.get(did) is t:
            _RUNNING.pop(did, None)

    task.add_done_callback(_forget)
    return task


async def convert_upload(
    dataset_id: str,
    rds_path: Path,
    output_path: Path,
    converter_url: str,
    dm: Any,
    protect: Callable[[Path], None],
) -> None:
    """Convert an upload, then mark it ready (with the report) or failed.

    ``dm`` is the DatasetManager (metadata + load); ``protect`` marks the
    converted file read-only (it is a source: processing writes to derived/)."""
    progress_path = output_path.parent / PROGRESS_FILE
    try:
        result = await trigger_conversion(
            str(rds_path), str(output_path), converter_url, progress_path=str(progress_path)
        )
    except Exception as e:  # noqa: BLE001 - any failure becomes the dataset's error
        logger.warning("Conversion of %s failed: %s", dataset_id, e)
        dm.update_dataset_metadata(dataset_id, {"status": "error", "error_message": str(e)})
        return

    await asyncio.to_thread(complete_spatial, output_path)
    protect(output_path)
    report = result.get("report") if isinstance(result.get("report"), dict) else {}
    dm.update_dataset_metadata(dataset_id, {"status": "ready", "conversion": report})
    try:
        # Fills in cells, genes, embeddings; GET /datasets/{id} retries a failed load.
        await dm.load_dataset(dataset_id)
    except Exception:  # noqa: BLE001
        logger.exception("Loading the converted dataset %s failed", dataset_id)
