"""Store a dataset already in the library by rows (CSR), when the user asks.

New datasets are stored by rows since v0.85 (``core/csr_store.py``). One uploaded or converted
before that keeps its column-stored (CSC) matrices: it is never rewritten on open, in the
background or in bulk, and never deleted. This is the rewrite the app OFFERS where the layout
blocks something: large-dataset mode reads the matrix in blocks of cells, which a CSC matrix
cannot give (``chunked.read_lazy``), so a run that does not fit in memory is refused
(``refusal``, raised by ``processing._make_run_room``; ``GET .../preflight`` says it before
the run as ``mode.by_columns``). Marker genes and cell-level DE read a backed dataset whole
either way and their memory estimates do not depend on the layout, so a 507 there stays a 507
after a rewrite: nothing is offered there.

What it writes (``targets``), each as a new file swapped in with ``os.replace``, so a kill
mid-write leaves the dataset as it was (``sweep`` removes what a kill left):

- the stored original (upload, conversion or import), when column-stored, as a row-stored copy
  in ``ingested/{id}/``, as a v0.85 upload has it. "Reset from original" and the mito re-filter
  read that copy from then on (``mito_refilter.source_file``), so a reset does not bring the
  column-stored matrix back. The upload itself is never changed; an ingest's own file in
  ``ingested/`` is replaced by its row-stored equal.
- the derived file (labels, results), when the dataset has one and it is column-stored.

Values, their dtype, the cell and gene order and everything else in the file stay as they are.
The step is recorded in the dataset's active file (``STEP``), so History shows it; it changes no
value, so no stored result goes stale (``results_store.ADDITIVE_STEPS``) and the replay script
has nothing to run for it. Before anything is written: the disk the new files and the
conversion's temporary files need against what is free (``NoDiskSpace``, a 507 with both
numbers), and the conversion's memory reserved in the budget (``csr_store.working_bytes``).
"""

from __future__ import annotations

import logging
import os
import shutil
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import h5py

from scview.core import csr_store, provenance

logger = logging.getLogger(__name__)

STEP = "store_by_rows"
TOOL = "scview.store_by_rows"
# The words of the run refusal the UI and the MCP tools look for (frontend api/pipelineRun.ts).
REFUSAL_PHRASE = "store the dataset by rows first"
# The matrices large-dataset mode reads in blocks of cells (``large_run._Run.open``).
LARGE_MODE_READS = ("X", "layers/counts", "raw/X")
PART = ".byrows.part"
GB = 1024**3

# The new file against the one it is made from: measured 1.15 (gzip: 0.468 -> 0.540 GB; the
# row pointers and each cell's gene numbers compress less well than a gene's cell numbers) and
# 1.00 (uncompressed) on 73,260 x 10,237.
OUT_SHARE = 1.25
DISK_MARGIN = 1.1
DISK_FLOOR = 256 * 1024**2  # left free whatever the estimate
# Seconds per stored value on 2 cores, by how the source's values are stored (see
# ``tests/test_store_by_rows.py`` for the measured runs these come from).
SECONDS_PER_VALUE = {True: 1.7e-7, False: 0.45e-7}
SECONDS_FLOOR = 2.0


class NoDiskSpace(RuntimeError):
    """Not enough free disk for the copy; the message names the GB needed and free."""

    def __init__(self, need: int, free: int, what: str = "Storing this dataset by rows") -> None:
        self.need, self.free = int(need), int(free)
        super().__init__(
            f"{what} writes a second copy of it: that needs about {_gb(need)} GB of disk and "
            f"{_gb(free)} GB is free. Free some disk space, then try again.")


def _gb(n: int | float) -> str:
    v = n / GB
    return f"{v:.0f}" if v >= 10 else f"{v:.1f}" if v >= 0.1 else f"{v:.2f}"


@dataclass
class Target:
    """One file to write: ``src`` read, ``dst`` swapped in; ``record``: the step is recorded
    in it (the dataset's active file)."""

    src: Path
    dst: Path
    record: bool
    matrices: list[dict] = field(default_factory=list)  # csr_store.csc_sizes

    @property
    def out_bytes(self) -> int:
        return int(self.src.stat().st_size * OUT_SHARE)

    @property
    def temp_bytes(self) -> int:
        return max((m["temp_bytes"] for m in self.matrices), default=0)

    @property
    def seconds(self) -> float:
        return sum(m["nnz"] * SECONDS_PER_VALUE[bool(m["compressed"])] for m in self.matrices)


def blocks_large_mode(path: str | Path | None) -> list[str]:
    """The column-stored matrices of the file that large-dataset mode would read by rows
    ([] when none: nothing is blocked)."""
    if path is None:
        return []
    return [w for w in csr_store.csc_in_file(path) if w in LARGE_MODE_READS]


def targets(dm, dataset_id: str) -> list[Target]:
    """The files a rewrite writes, in order ([] when the dataset's active file has no
    column-stored matrix: nothing to do)."""
    from scview.core import mito_refilter

    active = dm._resolve_h5ad(dataset_id)
    if active is None or not csr_store.csc_in_file(active):
        return []
    out: list[Target] = []
    source = mito_refilter.source_file(dm, dataset_id)
    if source is not None:
        sizes = csr_store.csc_sizes(source)
        if sizes:
            dst = Path(dm.data_dir) / "ingested" / dataset_id / Path(source).name
            out.append(Target(Path(source), dst, record=source == active, matrices=sizes))
    if source != active:  # the derived file
        out.append(Target(Path(active), Path(active), record=True, matrices=csr_store.csc_sizes(active)))
    return out


def disk_need(todo: list[Target]) -> int:
    """Disk the rewrite takes at its peak: every new file, and the largest matrix's temporary
    files (one matrix is converted at a time), with a margin."""
    if not todo:
        return 0
    return int((sum(t.out_bytes for t in todo) + max(t.temp_bytes for t in todo)) * DISK_MARGIN)


def free_disk(folder: str | Path) -> int:
    """Free bytes of the disk holding ``folder`` (its nearest existing parent)."""
    p = Path(folder)
    while not p.exists() and p != p.parent:
        p = p.parent
    return int(shutil.disk_usage(p).free)


def check_disk(folder: str | Path, need: int, what: str = "Storing this dataset by rows") -> None:
    """``NoDiskSpace`` unless ``need`` bytes fit the disk of ``folder`` with ``DISK_FLOOR`` left."""
    free = free_disk(folder)
    if need + DISK_FLOOR > free:
        raise NoDiskSpace(need, free, what)


def file_disk_need(src: str | Path) -> int:
    """``disk_need`` of one file's row-stored copy (the upload-time rewrite)."""
    return disk_need([Target(Path(src), Path(src), False, csr_store.csc_sizes(src))])


def plan(dm, dataset_id: str) -> dict:
    """What the UI and an agent need before offering or starting the rewrite, from the files'
    headers only: which matrices are column-stored, whether that blocks large-dataset mode,
    and the rewrite's seconds, disk (needed, free) and memory."""
    active = dm._resolve_h5ad(dataset_id)
    todo = targets(dm, dataset_id)
    stored = csr_store.csc_in_file(active) if active is not None else []
    need = disk_need(todo)
    free = free_disk(dm.data_dir)
    seconds = max(SECONDS_FLOOR, sum(t.seconds for t in todo)) if todo else 0.0
    out = {
        "dataset_id": dataset_id,
        "column_stored": stored,
        "blocks_large_mode": blocks_large_mode(active),
        "files": len(todo),
        "seconds": round(seconds),
        "disk_bytes": need,
        "disk_free_bytes": free,
        "enough_disk": not todo or need + DISK_FLOOR <= free,
        "memory_bytes": max((csr_store.working_bytes(t.src) for t in todo), default=0),
    }
    if todo:
        out["text"] = (f"about {_seconds_text(seconds)}, {_gb(need)} GB of disk "
                       f"({_gb(free)} GB free)")
    return out


def _seconds_text(seconds: float) -> str:
    if seconds < 90:
        return f"{max(5, int(round(seconds / 5.0)) * 5)} s"
    return f"{int(round(seconds / 60.0))} min"


def refusal(dm, dataset_id: str, reason: str, blocked: list[str]) -> str:
    """The sentence a run gets when it would go to large-dataset mode on a column-stored
    dataset: why, and the one action that unblocks it (with its cost)."""
    p = plan(dm, dataset_id)
    cost = (f" (about {_seconds_text(p['seconds'])} and {_gb(p['disk_bytes'])} GB of disk; no value changes)"
            if p["files"] else "")
    return (f"{reason} That mode reads the matrix in blocks of cells, and this dataset's is stored "
            f"by columns ({', '.join(blocked)}). Nothing was run: {REFUSAL_PHRASE}{cost}, then run "
            "it again. Data Assessment offers it beside Run; an agent calls the tool "
            f"store_by_rows (POST /api/v1/datasets/{dataset_id}/storage/by-rows).")


# ---------------------------------------------------------------------------
# The rewrite
# ---------------------------------------------------------------------------


def _shape(f: h5py.File) -> tuple[int, int]:
    x = f["X"]
    shape = x.shape if isinstance(x, h5py.Dataset) else x.attrs["shape"]
    return int(shape[0]), int(shape[1])


def _record(path: Path, converted: list[str], seconds: float) -> None:
    """Append the step to the provenance stored in the h5ad ``path`` (nothing else is read)."""
    from anndata.io import read_elem, write_elem

    key = provenance.UNS_KEY
    with h5py.File(path, "a") as f:
        if "uns" not in f:
            write_elem(f, "uns", {})
        uns = f["uns"]
        n_obs, n_vars = _shape(f)
        shim = SimpleNamespace(uns={key: read_elem(uns[key])} if key in uns else {},
                               n_obs=n_obs, n_vars=n_vars)
        provenance.record_step(
            shim, step=STEP, tool=TOOL, params={"matrices": list(converted)},
            note="Matrices stored by rows (CSR) instead of by columns: same values and order.",
            resources={"seconds": round(seconds, 1)})
        if key in uns:
            del uns[key]
        write_elem(uns, key, shim.uns[key])


def leftovers(folder: Path) -> list[Path]:
    """What a killed rewrite left in ``folder``: its unfinished file, the conversion's
    temporary folders."""
    if not folder.is_dir():
        return []
    return [p for p in folder.iterdir()
            if (p.name.startswith(".") and p.name.endswith(PART))
            or (p.is_dir() and p.name.startswith(".csr-"))]


def _remove(paths: list[Path]) -> int:
    for p in paths:
        if p.is_dir():
            shutil.rmtree(p, ignore_errors=True)
        else:
            p.unlink(missing_ok=True)
    return len(paths)


def sweep(data_dir: str | Path) -> int:
    """At startup (nothing is being written yet): remove what a killed rewrite or a killed
    upload-time conversion left under ``derived/`` and ``ingested/``. How many were removed."""
    n = 0
    for sub in ("derived", "ingested"):
        base = Path(data_dir) / sub
        if base.is_dir():
            for d in base.iterdir():
                n += _remove(leftovers(d))
    if n:
        logger.info("Removed %d unfinished store-by-rows file(s) left by a stopped backend", n)
    return n


def run(dm, dataset_id: str, progress=None) -> dict:
    """Rewrite the dataset's column-stored files (``targets``). Call it inside
    ``obs_writes.dataset_edit`` (``edit.run``), after ``check_disk``: nothing else may save the
    dataset meanwhile. The caller drops the cached copy afterwards (``dm.evict``).
    ``progress(done, total)``: stored values handled, of all the files'. Returns what was
    converted, the seconds and the active file's size before and after; ``converted`` is empty
    (and nothing written) for a dataset with no column-stored matrix."""
    t0 = time.monotonic()
    active = dm._resolve_h5ad(dataset_id)
    if active is None:
        raise csr_store.CsrStoreError(f"Dataset {dataset_id} has no h5ad file.")
    before = os.path.getsize(active)
    todo = targets(dm, dataset_id)
    if not todo:
        return {"converted": [], "seconds": 0.0, "files": [], "bytes_before": before,
                "bytes_after": before}
    total = sum(csr_store.work_values(t.src) for t in todo)
    state = {"done": 0, "said": 0.0}

    def step(n: int) -> None:
        state["done"] += int(n)
        now = time.monotonic()
        if progress is not None and (now - state["said"] >= 0.5 or state["done"] >= total):
            state["said"] = now
            progress(min(state["done"], total), total)

    converted: list[str] = []
    files: list[str] = []
    for t in todo:
        t.dst.parent.mkdir(parents=True, exist_ok=True)
        _remove(leftovers(t.dst.parent))  # of a rewrite that was killed
        partial = t.dst.with_name(f".{t.dst.name}.{uuid.uuid4().hex[:8]}{PART}")
        try:
            done = csr_store.rewrite_file(t.src, partial, workdir=t.dst.parent, progress=step)
            if t.record:
                _record(partial, done, time.monotonic() - t0)
                converted = done
            os.replace(partial, t.dst)
        finally:
            partial.unlink(missing_ok=True)
        if t.dst.parent.parent.name != "derived":
            t.dst.chmod(0o444)  # a source like the upload: never written to again
        files.append(str(t.dst))
    now = dm._resolve_h5ad(dataset_id)
    out = {"converted": converted, "seconds": round(time.monotonic() - t0, 1), "files": files,
           "active": now, "bytes_before": before, "bytes_after": os.path.getsize(now)}
    dm.update_dataset_metadata(dataset_id, {"stored_by_rows": {
        "matrices": converted, "at": datetime.now(timezone.utc).isoformat(), "asked": True}})
    logger.info("Dataset %s stored by rows (CSR): %s, %d file(s) in %.1f s", dataset_id,
                ", ".join(converted), len(files), out["seconds"])
    return out
