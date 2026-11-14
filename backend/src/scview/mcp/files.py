"""The MCP tools for getting files into scView: how to upload from the client's disk
(``upload_instructions``), and ``import_server_path`` for a file or folder on the SERVER.

The MCP server runs inside the backend (a container), so it cannot see the client's
disk: a local file goes up over the REST routes the UI uses (``api/v1/ingest.py``).
``import_server_path`` reads only under the folders set in ``MCP_IMPORT_DIRS`` (off when
empty, and always off in public mode: a hub never reads its own disk for a caller). A path
must be absolute; it is resolved with its links and must stay inside one of those folders,
as must every file a folder holds (a link pointing out is refused). The files are COPIED
into an ingest session (never moved or linked: the originals are never touched), then the
session is committed as an upload is.
"""

from __future__ import annotations

import asyncio
import hashlib
import os
import shutil
from pathlib import Path
from typing import Annotated, Any

from mcp.server.mcpserver.exceptions import ToolError
from pydantic import Field

from scview.mcp import api, jobs
from scview.mcp.compact import dumps, pick
from scview.mcp.spatial import NEXT, _body
from scview.mcp.tools import READ, WRITE, _line, tool

MAX_FILES = 2000  # a folder with more is not one experiment
_CHUNK = 4 * 1024 * 1024

UPLOAD_RECIPE = """\
The MCP server cannot read files on your machine: upload them over scView's REST API (the routes the
browser UI uses). BASE is the backend's address (http://127.0.0.1:8080 by default; the UI's port 3000
serves /api too). With ACCESS_TOKEN set, add -H "Authorization: Bearer <token>" to every call.

One .h5ad (or .loom, .h5, .rds):
  curl -F "file=@/path/data.h5ad" BASE/api/v1/datasets/upload          -> {"dataset_id": ...}

Several files or a folder (10x MTX trio, Cell Ranger outs/, Multiome h5 + atac_fragments.tsv.gz + .tbi +
per_barcode_metrics.csv, V(D)J contigs, Space Ranger spatial/, several samples to merge):
  SID=$(curl -s -X POST BASE/api/v1/ingest/session | python3 -c 'import sys,json;print(json.load(sys.stdin)["session_id"])')
  curl -F "files=@outs/filtered_feature_bc_matrix.h5" -F "files=@outs/atac_fragments.tsv.gz" \\
       -F "files=@outs/atac_fragments.tsv.gz.tbi" -F "files=@outs/per_barcode_metrics.csv" \\
       BASE/api/v1/ingest/session/$SID/files
  # A folder keeps its layout: one -F "paths=<path inside the folder>" per file, in the files' order,
  # e.g. -F "files=@s1/outs/filtered_feature_bc_matrix/matrix.mtx.gz" \\
  #      -F "paths=s1/outs/filtered_feature_bc_matrix/matrix.mtx.gz" (same-named files stay apart, and the
  # folder names say which sample a file belongs to). Side files (fragments, metrics, contigs, guide calls,
  # spatial/) are recognised by name and attached to their sample. BAM, Loupe and HTML files are skipped.
  curl BASE/api/v1/ingest/session/$SID                                   # what was detected, issues
  curl -X POST -H "Content-Type: application/json" -d '{"name": "my data"}' BASE/api/v1/ingest/session/$SID/options
  curl -X POST BASE/api/v1/ingest/session/$SID/commit                    -> {"dataset_id": ...}

A side file for a dataset already in the library:
  curl --data-binary @atac_fragments.tsv.gz "BASE/api/v1/datasets/<id>/atac/fragments?name=atac_fragments.tsv.gz"
  curl --data-binary @atac_fragments.tsv.gz.tbi "BASE/api/v1/datasets/<id>/atac/fragments?name=atac_fragments.tsv.gz.tbi"
  curl -F "file=@per_barcode_metrics.csv" BASE/api/v1/datasets/<id>/atac/metrics

RNA velocity counts (spliced + unspliced) for a dataset already in the library: a velocyto .loom (one per
sample, sent together), an .h5ad with spliced and unspliced layers, or STARsolo's Velocyto files
(features.tsv, barcodes.tsv, spliced.mtx, unspliced.mtx, ambiguous.mtx). Matched to the cells by barcode:
  curl -F "files=@sample1.loom" -F "files=@sample2.loom" BASE/api/v1/datasets/<id>/velocity/attach
An .h5ad uploaded with spliced and unspliced layers has them attached at upload. Then get_velocity.

Or drag the files onto the UI's Data tab. Then list_datasets.
"""


@tool(READ)
async def upload_instructions() -> str:
    """How to add files from your own machine: the MCP server runs inside scView's container and
    cannot read your disk, so local files go up over the REST API (curl recipes for one file, a
    multi-file or folder upload with its side files, a side file or RNA velocity counts for an existing
    dataset),
    or through the UI's Data tab. import_server_path reads files already on the server."""
    s = api.settings()
    text = UPLOAD_RECIPE
    if s.ACCESS_TOKEN:
        text += "\nThis server has ACCESS_TOKEN set: send it with each call.\n"
    if s.mcp_import_dirs:
        text += ("\nFiles already on the SERVER under " + ", ".join(s.mcp_import_dirs)
                 + " can be imported with import_server_path instead.\n")
    return text


def _allowed_roots() -> list[Path]:
    s = api.settings()
    if s.is_public:
        raise ToolError("import_server_path is off in public mode: upload the files instead (upload_instructions).")
    dirs = s.mcp_import_dirs
    if not dirs:
        raise ToolError("import_server_path is off: the server sets no MCP_IMPORT_DIRS. Upload the files instead "
                        "(upload_instructions), or set MCP_IMPORT_DIRS to the server folders it may read.")
    roots = []
    for d in dirs:
        try:
            roots.append(Path(d).resolve(strict=True))
        except (OSError, RuntimeError):
            continue
    if not roots:
        raise ToolError("None of the MCP_IMPORT_DIRS folders exists on the server.")
    return roots


def _inside(p: Path, roots: list[Path]) -> Path | None:
    return next((r for r in roots if p == r or p.is_relative_to(r)), None)


def resolve_import(path: str, roots: list[Path]) -> tuple[Path, list[tuple[str, Path]], list[dict[str, Any]]]:
    """The real path and the files to stage ((relative path, real file)); refused (ToolError)
    when the path is not absolute, does not exist, or it or any file it holds leaves the
    allowed folders. Files scView never reads are listed as skipped."""
    from scview.core.ingestion.paths import UploadPathError, junk_reason, safe_relative_path

    if not path or "\x00" in path:
        raise ToolError("Give an absolute path on the server.")
    raw = Path(path)
    if not raw.is_absolute():
        raise ToolError(f"'{path}' is not absolute: give the full path on the server, under "
                        f"{', '.join(map(str, roots))}.")
    try:
        real = raw.resolve(strict=True)
    except (OSError, RuntimeError):
        raise ToolError(f"'{path}' does not exist on the server.") from None
    root = _inside(real, roots)
    if root is None:
        raise ToolError(f"'{path}' is outside the folders import_server_path may read "
                        f"({', '.join(map(str, roots))}).")
    if real == root:
        raise ToolError("Name a file or a folder inside the import folder, not the folder itself.")
    files: list[tuple[str, Path]] = []
    skipped: list[dict[str, Any]] = []
    if real.is_file():
        files.append((real.name, real))
    elif real.is_dir():
        for dirpath, dirnames, filenames in os.walk(real, followlinks=False):
            for d in list(dirnames):
                if Path(dirpath, d).is_symlink():
                    raise ToolError(f"'{Path(dirpath, d)}' is a link to a folder: give its target instead.")
            for f in sorted(filenames):
                p = Path(dirpath, f)
                target = p.resolve()
                if _inside(target, roots) is None or not target.is_file():
                    raise ToolError(f"'{p}' links outside the folders import_server_path may read.")
                rel = f"{real.name}/{p.relative_to(real).as_posix()}"
                try:
                    rel = safe_relative_path(rel)
                except UploadPathError as e:
                    raise ToolError(str(e)) from None
                why = junk_reason(rel)
                if why:
                    skipped.append({"path": rel, "reason": why})
                    continue
                files.append((rel, target))
                if len(files) > MAX_FILES:
                    raise ToolError(f"'{path}' holds more than {MAX_FILES} files: name the experiment's folder.")
    else:
        raise ToolError(f"'{path}' is neither a file nor a folder.")
    if not files:
        raise ToolError(f"'{path}' holds no file scView reads.")
    return real, files, skipped


def _copy(src: Path, dest: Path) -> str:
    """Copy ``src`` to ``dest`` (never a link: the original stays untouched); its sha256."""
    digest = hashlib.sha256()
    try:
        with open(src, "rb") as fin, open(dest, "wb") as fout:
            while chunk := fin.read(_CHUNK):
                digest.update(chunk)
                fout.write(chunk)
        shutil.copystat(src, dest, follow_symlinks=True)
    except BaseException:
        dest.unlink(missing_ok=True)
        raise
    return digest.hexdigest()


@tool(WRITE)
async def import_server_path(
    path: Annotated[str, Field(max_length=4096, description=(
        "An absolute path ON THE SERVER (not your machine) to a file or a folder (e.g. a Cell Ranger outs/), "
        "under one of the folders the server allows (MCP_IMPORT_DIRS)."))],
    name: Annotated[str | None, Field(max_length=200, description="The dataset's name (default: from the files).")] = None,
) -> str:
    """Import a file or folder already on the SERVER (a mounted share, a batch node's disk) as a
    new dataset, as an upload of the same files would: copied in (the original is never
    changed), detected, bundled with its side files, saved. Only under the folders the server
    sets in MCP_IMPORT_DIRS; off when it sets none and always in public mode. For files on your
    own machine see upload_instructions. A job: returns job_id at once; job_status gives the
    new dataset_id."""
    roots = _allowed_roots()
    real, files, skipped = await asyncio.to_thread(resolve_import, path, roots)
    from scview.dependencies import get_ingest_session_manager

    mgr = api.dependency(get_ingest_session_manager)
    total = sum(f.stat().st_size for _, f in files)

    async def run(job: jobs.Job, on_event) -> None:
        sid = mgr.create_session()
        job.stage = sid
        partials: list[Path] = []
        committed = False
        try:
            items = []
            for i, (rel, src) in enumerate(files):
                if job.disconnect.is_set():  # cancel_job: nothing kept
                    return
                await on_event("progress", {"message": f"copying {i + 1}/{len(files)}: {rel}"})
                partial = mgr.partial_path(sid)
                partials.append(partial)
                items.append((rel, partial, await asyncio.to_thread(_copy, src, partial)))
            await asyncio.to_thread(mgr.stage, sid, items)
            if name:
                await api.post(f"/ingest/session/{sid}/options", _body(name=name))
            state = await api.get(f"/ingest/session/{sid}")
            await on_event("progress", {"message": "saving the dataset"})
            done = await api.post(f"/ingest/session/{sid}/commit")
            committed = True
            await on_event("complete", {"dataset_id": done.get("dataset_id"), "state": state})
        finally:
            for p in partials:
                p.unlink(missing_ok=True)
            if not committed:
                mgr.discard(sid)

    def finish(job: jobs.Job) -> dict[str, Any]:
        done = job.events.get("complete") or {}
        state = done.get("state") or {}
        validation = state.get("validation") or {}
        out = {**pick(done, "dataset_id"), "path": str(real), "n_files": len(files), "bytes": total,
               "warnings": [i.get("message") for i in validation.get("issues") or []
                            if isinstance(i, dict) and i.get("message")][:10]}
        if skipped:
            out["skipped"] = skipped[:20]
        out["next"] = "get_dataset(dataset_id), then assess_dataset"
        return out

    job = jobs.new("import_server_path", None, "", {"path": str(real)}, progress_line=_line, finish=finish,
                   runner=run)
    return dumps({"job_id": job.id, "status": job.status, "path": str(real), "n_files": len(files), "bytes": total,
                  **({"skipped": skipped[:20]} if skipped else {}), "next": NEXT})
