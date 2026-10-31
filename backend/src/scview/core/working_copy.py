"""The dataset in memory, for work that needs every value at hand.

The adaptor opens an h5ad larger than ``anndata_adaptor.BACKED_MIN_BYTES`` (2 GiB
by default; ``SCVIEW_BACKED_MIN_MB``) in anndata's backed mode: X and .raw's X stay
on disk and are read as they are used. That suits viewing, but processing (scanpy's
QC metrics, filtering, normalisation, variable genes, PCA, ...) needs the matrices in
memory. Handed the backed object, a pipeline run failed at its first step (``AxisError:
axis 1 is out of bounds for array of dimension 0``), every step after it failed in turn
except doublet detection, which ran for 20+ minutes on 500k cells, and the run then
saved.

This module is the one place that decides how such work gets its data: ``make_room``
before it (on the event loop: the memory budget), ``in_memory`` in the worker thread
that does it. Today a backed dataset is read into memory whole, once, within the
memory budget (``DatasetManager.reserve``: other open datasets closed first, least
recently used; a 507 when it cannot fit even then), and the adaptor serves that copy
from then on: its backed object and file handle are let go, so the work never holds
two copies. A chunked path (QC..PCA in blocks) would replace the read here, not in the
callers.

Work that needs only some of a backed dataset's values (a test on two groups, the counts of
a joint embedding) reads those: ``on_disk`` / ``read_rows`` / ``read_matrix``, without
closing the file others read (``AnnData.to_memory`` closes it, and a request reading it at
that moment fails). It takes ``reading``: room for what it reads, a 507 naming the dataset
when that cannot fit, and a reservation until the work has ended, so requests running at
the same time count each other.
"""

from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager

import numpy as np
import scipy.sparse as sp

from scview.core import preflight

logger = logging.getLogger(__name__)

DENSE_ROWS = 4096  # rows of an on-disk dense matrix read at a time


def _backed(adaptor) -> bool:
    return bool(getattr(adaptor.adata, "isbacked", False))


def copy_bytes(adaptor) -> int:
    """Memory reading the adaptor's dataset into memory adds: its on-disk matrices
    (X, .raw's X) from the file header; 0 when it is in memory already."""
    if not _backed(adaptor):
        return 0
    h = preflight.read_header(adaptor.h5ad_path, "h5ad")
    if hasattr(h, "backed"):
        h.backed = True  # open backed, whatever its size (the manager may have chosen it)
    return preflight.backed_bytes(h)


async def make_room(dm, dataset_id: str, adaptor, then: int = 0) -> bool:
    """Before work on ``adaptor``'s dataset that needs it in memory: room for reading a
    backed dataset in, and ``then`` bytes the work needs beyond the in-memory copy (a
    run's estimated peak, ``preflight.run_need``). Refused with ``DatasetMemoryError``
    (a 507) only when the copy itself cannot fit with every other dataset closed; what
    the work needs beyond it is asked for, never refused (a run's pre-step check stops
    a step that cannot fit). Returns whether the dataset is backed: ``in_memory``, in
    the worker thread, then reads it in."""
    try:
        copy = await asyncio.to_thread(copy_bytes, adaptor)
    except Exception as e:  # never block the work on the estimate itself
        logger.warning("No in-memory estimate for %s: %s", dataset_id, e)
        copy = -1 if _backed(adaptor) else 0
    backed = copy != 0
    if copy > 0:
        await dm.reserve(dataset_id, copy, extra=max(0, then),
                         what="Reading this dataset into memory")
    elif then > 0:
        await dm.free_memory_for(then, keep=dataset_id)
    return backed


def in_memory(adaptor):
    """``adaptor``'s AnnData in memory: a backed dataset read whole, after
    ``make_room``, which the adaptor serves from now on. Call it in the worker thread
    that does the work (reading a large file takes seconds to minutes)."""
    return adaptor.to_memory()


def on_disk(M) -> bool:
    """Whether ``M`` is a backed dataset's matrix on disk, read only when indexed."""
    return M is not None and not sp.issparse(M) and not isinstance(M, np.ndarray)


def read_matrix(M):
    """``M`` in memory: an on-disk matrix read whole, one in memory as it is."""
    if not on_disk(M):
        return M
    return M.to_memory() if hasattr(M, "to_memory") else np.asarray(M)


def read_rows(M, keep: np.ndarray):
    """The rows of on-disk ``M`` where the mask ``keep`` is True, in memory: what
    ``read_matrix(M)[keep]`` gives, with only those rows held."""
    if keep.all():
        return read_matrix(M)
    if hasattr(M, "to_memory"):
        if getattr(M, "format", "csr") != "csr":  # rows of a CSC cost a pass over all of it
            return M.to_memory()[keep]
        return M[keep]  # a mask: anndata reads runs of rows as slices
    out = np.empty((int(keep.sum()), *M.shape[1:]), dtype=M.dtype)
    at = 0
    for start in range(0, M.shape[0], DENSE_ROWS):
        part = keep[start:start + DENSE_ROWS]
        if part.any():
            block = M[start:start + part.size][part]
            out[at:at + len(block)] = block
            at += len(block)
    return out


def rows_bytes(adaptor, wheres: tuple[str, ...] | list[str], n_rows: int) -> int:
    """Memory ``n_rows`` rows of the on-disk matrices ``wheres`` ("X", "raw/X") take, from
    the file header (``preflight``); 0 when the dataset is in memory."""
    if not _backed(adaptor):
        return 0
    h = preflight.read_header(adaptor.h5ad_path, "h5ad")
    return sum(preflight.rows_bytes(h.matrix(w), n_rows) for w in wheres)


def counts_bytes(adaptor) -> int:
    """Memory that copying a backed dataset's counts and log-normalising them adds
    (``preflight.label_transfer_need``'s share of one dataset); 0 when it is in memory."""
    if not _backed(adaptor):
        return 0
    return int(preflight.LT_COUNTS_COPIES * preflight._lt_counts_bytes(
        preflight.read_header(adaptor.h5ad_path, "h5ad")))


def file_bytes(path: str) -> int:
    """The peak of reading the h5ad ``path`` whole, whatever its size (``estimate_load``:
    its header would say an open of a big file leaves its matrices on disk)."""
    h = preflight.read_header(path, "h5ad")
    if hasattr(h, "backed"):
        h.backed = False
    return int(preflight.estimate_load(h, ingest=False).peak_bytes)


class HeldRead:
    """Work inside ``reading``: each ``run`` is a worker thread the reservation waits for,
    even when the request is cancelled (as ``obs_writes.DatasetEdit``)."""

    def __init__(self) -> None:
        self.tasks: list[asyncio.Future] = []

    async def run(self, fn, *args):
        task = asyncio.ensure_future(asyncio.to_thread(fn, *args))
        self.tasks.append(task)
        return await asyncio.shield(task)


@asynccontextmanager
async def reading(dm, dataset_id: str, need: int, what: str, also: tuple[str, ...] | list[str] = ()):
    """Hold room for a read of ``need`` bytes (0: nothing to read, nothing is checked) until
    the block and the threads it started with ``held.run`` have ended, however they end.
    ``DatasetManager.reserve`` first: other open datasets closed, least recently used (never
    ``dataset_id`` or ``also``, the datasets read), and ``DatasetMemoryError`` (a 507) with
    ``what`` as the sentence's subject when it cannot fit."""
    held = HeldRead()
    if need <= 0:
        yield held
        return
    ids = (dataset_id, *also)
    guard = dm.hold_read(ids, 0)  # not dropped to make the room
    try:
        await dm.reserve(dataset_id, need, what=what)
        release = dm.hold_read(ids, need, what)
    finally:
        guard()
    try:
        yield held
    finally:
        running = [t for t in held.tasks if not t.done()]
        if not running:
            release()
        else:
            left = len(running)

            def _done(_t) -> None:
                nonlocal left
                left -= 1
                if left == 0:
                    release()

            for t in running:
                t.add_done_callback(_done)
