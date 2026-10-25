"""Exports for other viewers: a CELLxGENE-compatible h5ad and a Vitessce bundle.

``GET /export/h5ad/info?format=cellxgene|vitessce`` (``export.py``) answers with
``format_info``: the choices (embeddings, label columns), what the file holds and leaves
out, its size, and for CELLxGENE the Discover readiness report.
``POST /export/{format}/build-stream`` writes the file in a worker thread while holding
the dataset's edit lock (so the file is one consistent state of the dataset) and streams
``progress`` events, then ``complete`` with the name to fetch, or ``error``. A client that
goes away stops the build at its next step and the partial file is removed.
``GET /export/files/{name}`` sends the finished file from disk. Built files are kept in
the dataset's cache folder until the dataset's files change (one per format).
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import re
import threading
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel

from scview.core import cellxgene_export as cx
from scview.core import large_mode
from scview.core import vitessce_export as vx
from scview.core.dataset_manager import DatasetManager
from scview.core.obs_writes import dataset_edit
from scview.core.sse import keepalive
from scview.dependencies import get_dataset_manager

logger = logging.getLogger(__name__)

router = APIRouter()

FORMATS = {
    "cellxgene": {"suffix": "_cellxgene.h5ad", "media": "application/x-hdf5"},
    "vitessce": {"suffix": "_vitessce.zip", "media": "application/zip"},
}
_FILE_NAME = re.compile(r"^(cellxgene|vitessce)-[0-9a-f]{16}\.(h5ad|zip)$")
_running: set[tuple[str, str]] = set()


class FormatOptions(BaseModel):
    embeddings: list[str] | None = None  # obsm keys; None: every 2D/3D layout
    columns: list[str] | None = None  # obs columns; None: all (CELLxGENE) / labels (Vitessce)
    include_raw: bool = False  # CELLxGENE: raw counts as .raw
    genes: str = "all"  # Vitessce: all | variable


def _sse(event: str, data: dict[str, Any]) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


def _check_format(fmt: str) -> None:
    if fmt not in FORMATS:
        raise HTTPException(status_code=404, detail=f"Unknown export format '{fmt}'.")


def _cx_options(o: FormatOptions) -> cx.Options:
    return cx.Options(embeddings=o.embeddings, columns=o.columns, include_raw=o.include_raw)


def _download_name(dm: DatasetManager, dataset_id: str, fmt: str) -> str:
    from scview.api.v1.export import file_stem

    info = dm.get_dataset_info(dataset_id) or {}
    return f"scview_{file_stem(info.get('name'), dataset_id)}{FORMATS[fmt]['suffix']}"


def _plan(adaptor, fmt: str, o: FormatOptions):
    if fmt == "cellxgene":
        return cx.plan(adaptor, _cx_options(o))
    return vx.plan(adaptor, _cx_options(o), genes=o.genes)


def _summary(adaptor, fmt: str, o: FormatOptions, name: str) -> dict[str, Any]:
    """The choices, the plan's contents and its size (and readiness for CELLxGENE)."""
    from scview.core.species import detect_species

    adata = adaptor.adata
    p = _plan(adaptor, fmt, o)
    out: dict[str, Any] = {
        "format": fmt,
        "embeddings": cx.embedding_choices(adata, vx.MAX_DIMS if fmt == "vitessce" else 3),
        "label_columns": vx.label_columns(adata),
        "has_variable_genes": "highly_variable" in adata.var.columns,
        "n_cells": int(adata.n_obs),
    }
    if fmt == "cellxgene":
        out.update({
            "size_bytes": cx.estimate_bytes(p),
            "has_counts": p.counts is not None,
            "x_from": p.x.where,
            "raw_from": p.raw.where if p.raw is not None else None,
            "n_genes": int(p.var.shape[0]),
            "obs_columns": [str(c) for c in p.obs.columns],
            "exported_embeddings": list(p.obsm),
            "dropped": p.dropped,
            "notes": p.notes,
            "renamed": p.renamed,
            "readiness": cx.readiness(p, species=detect_species(adata).species, title=name,
                                     uns=dict(adata.uns)),
        })
    else:
        out.update({
            "size_bytes": vx.estimate_bytes(p),
            "x_from": p["x"].where,
            "n_genes": int(p["var"].shape[0]),
            "obs_columns": [str(c) for c in p["obs"].columns],
            "exported_embeddings": list(p["obsm"]),
            "dropped": p["dropped"],
            "notes": p["notes"],
        })
    return out


async def _adaptor(dm: DatasetManager, dataset_id: str):
    if dm.get_dataset_info(dataset_id) is None:
        raise HTTPException(status_code=404, detail="Dataset not found.")
    adaptor = await dm.get_or_load_dataset(dataset_id)
    if adaptor is None:
        raise HTTPException(status_code=404, detail="Dataset not found.")
    return adaptor


async def format_info(dm: DatasetManager, dataset_id: str, fmt: str, o: FormatOptions) -> dict[str, Any]:
    """For ``GET /export/h5ad/info?format=...``: what a download with these options holds."""
    _check_format(fmt)
    adaptor = await _adaptor(dm, dataset_id)
    name = (dm.get_dataset_info(dataset_id) or {}).get("name") or dataset_id
    try:
        out = await asyncio.to_thread(_summary, adaptor, fmt, o, name)
    except cx.ExportError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    out["filename"] = _download_name(dm, dataset_id, fmt)
    out["large"] = bool(getattr(adaptor, "is_large", False))
    return out


def _sources(adaptor) -> list[Path]:
    from scview.core import large_layout

    paths = [Path(adaptor.h5ad_path)]
    if getattr(adaptor, "is_large", False):
        paths.append(large_layout.counts_path(adaptor.h5ad_path, adaptor.adata.uns))
    return paths


def _file_key(adaptor, fmt: str, o: FormatOptions) -> str:
    from scview.core.provenance import _scview_version

    stamp = [(str(p), p.stat().st_mtime_ns, p.stat().st_size) for p in _sources(adaptor) if p.exists()]
    blob = json.dumps({"fmt": fmt, "o": o.model_dump(), "src": stamp, "v": _scview_version()},
                      sort_keys=True)
    return hashlib.sha1(blob.encode()).hexdigest()[:16]


def _exports_dir(dm: DatasetManager, dataset_id: str) -> Path:
    return dm.data_dir / "cache" / dataset_id / "exports"


@router.post("/datasets/{dataset_id}/export/{fmt}/build-stream")
async def build_export(
    dataset_id: str,
    fmt: str,
    body: FormatOptions,
    dm: DatasetManager = Depends(get_dataset_manager),
):
    """Write the export in a worker thread and stream its progress; ``complete`` carries
    ``file`` (for ``GET /export/files/{file}``), ``filename`` and ``size_bytes``."""
    _check_format(fmt)
    adaptor = await _adaptor(dm, dataset_id)
    try:  # bad options and large-dataset refusals answer before the stream starts
        await asyncio.to_thread(_plan, adaptor, fmt, body)
    except cx.ExportError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    del adaptor  # a reload meanwhile must not keep this copy alive
    if (dataset_id, fmt) in _running:
        raise HTTPException(status_code=409, detail="This export is already being written.")
    info = dm.get_dataset_info(dataset_id) or {}
    name = info.get("name") or dataset_id
    download = _download_name(dm, dataset_id, fmt)

    async def events():
        if (dataset_id, fmt) in _running:
            yield _sse("error", {"error": "This export is already being written."})
            return
        _running.add((dataset_id, fmt))
        stop = threading.Event()
        messages: asyncio.Queue[str] = asyncio.Queue()
        loop = asyncio.get_running_loop()

        def progress(msg: str) -> None:
            loop.call_soon_threadsafe(messages.put_nowait, msg)

        try:
            async with dataset_edit(dataset_id) as held:
                ad_ = await dm.get_or_load_dataset(dataset_id)
                if ad_ is None:
                    yield _sse("error", {"error": "Dataset not found."})
                    return
                folder = _exports_dir(dm, dataset_id)
                ext = "h5ad" if fmt == "cellxgene" else "zip"
                out = folder / f"{fmt}-{_file_key(ad_, fmt, body)}.{ext}"
                if not out.exists():
                    folder.mkdir(parents=True, exist_ok=True)
                    from scview.api.v1.export import file_stem

                    note = cx.provenance(name, dataset_id, fmt)

                    def work(adaptor) -> None:
                        p = _plan(adaptor, fmt, body)
                        if fmt == "cellxgene":
                            cx.write(p, out, note, stop=stop.is_set, progress=progress)
                        else:
                            vx.write(p, out, name, file_stem(name, dataset_id), note,
                                     stop=stop.is_set, progress=progress)
                        for old in folder.glob(f"{fmt}-*.{ext}"):  # one kept per format
                            if old != out:
                                old.unlink(missing_ok=True)

                    task = asyncio.ensure_future(held.run(work, ad_))
                    # Retrieved when the client went away mid-write (Cancelled).
                    task.add_done_callback(lambda t: t.cancelled() or t.exception())
                    del ad_
                    while not task.done():
                        getter = asyncio.ensure_future(messages.get())
                        done, _ = await asyncio.wait({task, getter}, return_when=asyncio.FIRST_COMPLETED)
                        if getter in done:
                            yield _sse("progress", {"message": getter.result()})
                        else:
                            getter.cancel()
                    task.result()
                else:
                    del ad_
            yield _sse("complete", {"file": out.name, "filename": download,
                                    "size_bytes": out.stat().st_size})
        except cx.Cancelled:
            pass
        except large_mode.LargeDatasetError as exc:
            yield _sse("error", {"error": str(exc)})
        except Exception as e:  # noqa: BLE001 - surface to the client, keep the server up
            logger.error("%s export failed for %s: %s", fmt, dataset_id, e, exc_info=True)
            yield _sse("error", {"error": f"The export failed: {e}"})
        finally:
            stop.set()  # a thread mid-write stops at its next step
            _running.discard((dataset_id, fmt))

    return StreamingResponse(
        keepalive(events()),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.get("/datasets/{dataset_id}/export/files/{name}")
async def download_export(
    dataset_id: str,
    name: str,
    dm: DatasetManager = Depends(get_dataset_manager),
):
    """A finished export, streamed from disk."""
    if dm.get_dataset_info(dataset_id) is None or not _FILE_NAME.match(name):
        raise HTTPException(status_code=404, detail="Export not found.")
    path = _exports_dir(dm, dataset_id) / name
    if not path.is_file():
        raise HTTPException(status_code=404, detail="Export not found; write it again.")
    fmt = name.split("-", 1)[0]
    return FileResponse(path, media_type=FORMATS[fmt]["media"],
                        filename=_download_name(dm, dataset_id, fmt),
                        headers={"Content-Encoding": "identity"})
