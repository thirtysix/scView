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
"""

from __future__ import annotations

import asyncio
import logging

from scview.core import preflight

logger = logging.getLogger(__name__)


def _backed(adaptor) -> bool:
    return bool(getattr(adaptor.adata, "isbacked", False))


def copy_bytes(adaptor) -> int:
    """Memory reading the adaptor's dataset into memory adds: its on-disk matrices
    (X, .raw's X) from the file header; 0 when it is in memory already."""
    if not _backed(adaptor):
        return 0
    return preflight.backed_bytes(preflight.read_header(adaptor.h5ad_path, "h5ad"))


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
