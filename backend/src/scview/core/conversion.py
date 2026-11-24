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
import math
import time
from collections.abc import Callable, Coroutine
from datetime import datetime
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


# After the connection drops mid-conversion: how long to wait for a converter that
# restarts (its container was killed, most likely for memory) before calling it down.
RESTART_WAIT_S = 60.0
RESTART_POLL_S = 2.0

GiB = 1024 ** 3

SMALLER_OBJECT_TIP = ("Or save a smaller object in R first, with only the counts (for Seurat v5: "
                      "DietSeurat(obj, layers = \"counts\")): scView does not read the data and "
                      "scale.data layers, but the converter has to load them.")


class ConversionError(RuntimeError):
    """The converter refused the file or failed; the message is its own sentence."""


def converter_memory_advice(limit_bytes: int | None, target_gb: int | None, *, hosted: bool,
                            name_limit: bool = True) -> str:
    """What to do when the converter has too little memory: the ONE place for the local
    and hosted (DEPLOYMENT_MODE=public) wording. Locally the converter gets scView's own
    limit (docker-compose.yml: SCVIEW_MEMORY_GB). ``name_limit`` False: the sentence
    before already named the limit, so the hosted wording does not say it again."""
    if hosted:
        from scview.core.memory_budget import more_memory_advice

        return more_memory_advice(limit_bytes=limit_bytes if name_limit else None, target_gb=target_gb,
                                  service="the R converter", hosted_site=True)
    at_least = f" to at least {target_gb} GB" if target_gb else ""
    if limit_bytes is None:
        return ("Free memory on this computer by closing other programs, or give Docker or Podman "
                f"more memory{at_least} (on Windows and macOS in its settings).")
    return (f"Raise SCVIEW_MEMORY_GB in the .env file in the scView folder{at_least} (the converter "
            "has the same limit as scView), then restart scView.")


# A size to raise the converter's limit to: the estimate under 70% of it (the backend
# preflight's FITS_BELOW), for its error and the server's own use.
TARGET_SHARE = 0.7


def target_gb(need_bytes: int, used_bytes: int = 0) -> int:
    """The converter limit, in whole GB, that a conversion needing ``need_bytes`` fits."""
    return math.ceil((need_bytes / TARGET_SHARE + used_bytes) / GiB)


def _gb(b: float) -> str:
    g = b / GiB
    return f"{g:.0f}" if g >= 10 else f"{g:.1f}"


def out_of_memory_message(doing: str, limit_bytes: int | None, need_bytes: int | None, *,
                          hosted: bool, restarted: bool = False, tip: bool = True) -> str:
    """The sentence for work the converter ran out of memory on (``doing``: "converting
    x.rds"), with what to do. The measured fact leads (the limit it ran out of); the
    estimate follows only as what it adds: above the limit, the size this object needs;
    at or under it, that the estimate was too low for this object (on the hub one said
    "needs about 2.0 GB" of a converter that had 3.0 GB and ran out). Each clause once."""
    of = f"its {_gb(limit_bytes)} GB" if limit_bytes else "memory"
    lead = (f"The converter stopped and restarted while {doing}: it most likely ran out of {of}."
            if restarted else f"The converter ran out of {of} {doing}.")
    parts = [lead]
    target = target_gb(need_bytes) if need_bytes else None
    if need_bytes and limit_bytes and need_bytes <= limit_bytes:
        parts.append(f"{'If so, the' if restarted else 'The'} estimate (about {_gb(need_bytes)} GB) "
                     f"was too low for this object: it needs more than {_gb(limit_bytes)} GB.")
        if target and target * GiB <= limit_bytes:
            target = None  # no size the estimate can vouch for: only "more"
    elif need_bytes:
        parts.append(f"This object needs about {_gb(need_bytes)} GB.")
    parts.append(converter_memory_advice(limit_bytes, target, hosted=hosted, name_limit=False))
    if tip:
        parts.append(SMALLER_OBJECT_TIP)
    return " ".join(parts)


def hosted() -> bool:
    try:
        from scview.config import get_settings

        return get_settings().is_public
    except Exception:  # noqa: BLE001 - wording only
        return False


async def _need_bytes(rds_path: str) -> int | None:
    """The converter preflight's estimate for the file, for the out-of-memory sentence."""
    from scview.core import conversion_preflight

    try:
        ser = await asyncio.to_thread(conversion_preflight.serialized_size_cached, rds_path)
    except OSError:
        return None
    return conversion_preflight.estimate_peak(ser.bytes)


async def _measured(rds_path: str, peak_bytes: int) -> dict:
    """A conversion's measured peak beside the estimate it had, for the dataset's
    conversion report (``memory``) and the log: real objects to refit the estimate on."""
    record: dict[str, Any] = {"peak_bytes": peak_bytes}
    from scview.core import conversion_preflight

    try:
        ser = await asyncio.to_thread(conversion_preflight.serialized_size_cached, rds_path)
    except OSError:
        ser = None
    if ser is not None:
        record.update(serialized_bytes=ser.bytes, serialized_exact=ser.exact,
                      estimate_bytes=conversion_preflight.estimate_peak(ser.bytes))
    logger.info("Converted %s: measured peak %.2f GiB%s", Path(rds_path).name, peak_bytes / GiB,
                "" if ser is None else (f", estimate {record['estimate_bytes'] / GiB:.2f} GiB "
                                        f"({ser.bytes / GiB:.2f} GiB decompressed)"))
    return record


async def _after_drop(converter_url: str, sent_at: float,
                      transport: httpx.AsyncBaseTransport | None) -> dict | None:
    """The connection dropped mid-conversion: the converter's /health once it answers
    again (with ``restarted``: it started after the request was sent), or None when it
    stays down for ``RESTART_WAIT_S``."""
    timeout = httpx.Timeout(connect=3.0, read=5.0, write=5.0, pool=3.0)
    deadline = time.monotonic() + RESTART_WAIT_S
    while True:
        try:
            async with httpx.AsyncClient(timeout=timeout, transport=transport) as client:
                r = await client.get(f"{converter_url}/health")
            data = r.json() if r.status_code == 200 else None
        except (httpx.HTTPError, ValueError):
            data = None
        if isinstance(data, dict):
            started = data.get("started_at")
            try:
                t = datetime.fromisoformat(str(started)).timestamp()
            except ValueError:
                t = None
            data["restarted"] = t is not None and t >= sent_at - 1.0
            return data
        if time.monotonic() >= deadline:
            return None
        await asyncio.sleep(RESTART_POLL_S)


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
    dict – the converter's body: ``status``, ``message`` and ``report``; the report
    carries the measured peak beside the estimate (``memory``) when the converter
    measured one.

    Raises
    ------
    ConversionError – with the converter's own reason ("This file holds a
    list object, which scView cannot convert. ..."), or why it was not reached.
    """
    body = {"input_path": rds_path, "output_path": output_path}
    if progress_path:
        body["progress_path"] = progress_path
    name = Path(rds_path).name
    sent_at = time.time()
    try:
        async with httpx.AsyncClient(timeout=TIMEOUT, transport=transport) as client:
            resp = await client.post(f"{converter_url}/convert", json=body)
    except httpx.TimeoutException as e:
        raise ConversionError(
            f"The R converter did not finish within {READ_TIMEOUT_S / 3600:.0f} hours."
        ) from e
    except httpx.TransportError as e:
        down = ConversionError(
            "The R converter service could not be reached "
            f"({type(e).__name__}): is the converter container running?"
        )
        if isinstance(e, httpx.ConnectError):
            raise down from e
        # It took the request and the connection dropped: killed (its container, for
        # memory, most likely) or down. Its /health says which once it is back.
        health = await _after_drop(converter_url, sent_at, transport)
        if health is None:
            raise down from e
        if health.get("restarted"):
            raise ConversionError(out_of_memory_message(
                f"converting {name}", health.get("memory_limit_bytes"), await _need_bytes(rds_path),
                hosted=hosted(), restarted=True)) from e
        raise ConversionError(
            f"The connection to the R converter dropped while it converted {name} "
            f"({type(e).__name__}). Add the file again to convert it."
        ) from e

    try:
        data = resp.json()
    except ValueError:
        data = None
    if not isinstance(data, dict):
        data = {}
    if data.get("error_kind") == "out_of_memory":
        limit = data.get("memory_limit_bytes")
        raise ConversionError(out_of_memory_message(
            f"converting {name}", int(limit) if isinstance(limit, (int, float)) else None, await _need_bytes(rds_path),
            hosted=hosted()))
    if resp.status_code >= 400 or data.get("status") != "ok":
        reason = data.get("error")
        raise ConversionError(
            str(reason) if reason else f"The R converter answered {resp.status_code} without saying why."
        )
    peak = data.get("peak_bytes")
    if isinstance(peak, (int, float)) and not isinstance(peak, bool) and peak > 0:
        report = data.get("report") if isinstance(data.get("report"), dict) else {}
        report["memory"] = await _measured(rds_path, int(peak))
        data["report"] = report
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
        # The converter preflight, shown while it converts: a warning, never a refusal.
        from scview.core import conversion_preflight

        verdict = await asyncio.to_thread(conversion_preflight.verdict_for, rds_path, converter_url,
                                          hosted=hosted())
        dm.update_dataset_metadata(dataset_id, {"conversion_preflight": verdict})
    except Exception:  # noqa: BLE001 - advisory only
        logger.exception("Converter preflight of %s failed", dataset_id)
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
