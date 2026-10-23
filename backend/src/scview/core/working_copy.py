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
two copies.

A dataset processed in large-dataset mode (``core/large_layout.py``) keeps its counts
in a sidecar, not in a layer: read into memory for an in-memory run, the counts come in as
``layers['counts']`` and the copy stops being marked large (saved by that run, it is an
ordinary h5ad and the sidecars go). A run in large-dataset mode is never read in at all
(``core/large_run.py``; ``core/processing.py`` chooses the mode).
"""

from __future__ import annotations

import asyncio
import logging

from scview.core import large_layout, preflight

logger = logging.getLogger(__name__)


def _backed(adaptor) -> bool:
    return bool(getattr(adaptor.adata, "isbacked", False))


def large_info(h5ad_path) -> dict | None:
    """``uns["scview_large"]`` of an h5ad on disk (read alone), or None: not a large dataset."""
    import h5py
    from anndata.io import read_elem

    try:
        with h5py.File(h5ad_path, "r") as f:
            key = f"uns/{large_layout.KEY}"
            return read_elem(f[key]) if key in f else None
    except OSError:
        return None


def large_counts_matrix(h5ad_path, uns: dict | None = None) -> preflight.Matrix | None:
    """The counts sidecar of a large dataset as a header matrix (``layers/counts``), or None
    (not large, or unreadable). ``uns``: the dataset's, when at hand (else read)."""
    import h5py

    if uns is None:
        info = large_info(h5ad_path)
        uns = {large_layout.KEY: info} if info else None
    if not uns or not large_layout.is_large(uns):
        return None
    path = large_layout.counts_path(h5ad_path, uns)
    try:
        with h5py.File(path, "r") as f:
            g = f[large_layout.COUNTS_KEY]
            n, m = (int(v) for v in g.attrs["shape"])
            return preflight.Matrix("layers/counts", n, m, True, int(g["data"].shape[0]),
                                    str(g["data"].dtype), str(g["indices"].dtype))
    except (OSError, KeyError) as e:
        logger.warning("Counts sidecar of %s unreadable: %s", h5ad_path, e)
        return None


def _large_counts(adaptor) -> preflight.Matrix | None:
    adata = adaptor.adata
    return large_counts_matrix(adaptor.h5ad_path, adata.uns) if large_layout.is_large(adata) else None


def copy_bytes(adaptor) -> int:
    """Memory reading the adaptor's dataset into memory adds: its on-disk matrices
    (X, .raw's X) from the file header, and a large dataset's counts sidecar; 0 when it
    is in memory already."""
    counts = _large_counts(adaptor)
    extra = counts.nbytes if counts is not None else 0
    if not _backed(adaptor):
        return extra
    return preflight.backed_bytes(preflight.read_header(adaptor.h5ad_path, "h5ad")) + extra


def run_header(adaptor) -> preflight.Header:
    """The header of the dataset as an in-memory run holds it (``preflight.in_memory_header``,
    with a large dataset's counts as the layer they become)."""
    h = preflight.in_memory_header(adaptor.adata)
    counts = _large_counts(adaptor)
    if counts is not None:
        h.matrices = [mm for mm in h.matrices if mm.where != "layers/counts"] + [counts]
    return h


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
    ``make_room``, which the adaptor serves from now on (a large dataset's counts sidecar
    as ``layers['counts']``). Call it in the worker thread that does the work (reading a
    large file takes seconds to minutes)."""
    adata = adaptor.to_memory()
    if large_layout.is_large(adata):
        import h5py
        from anndata.io import read_elem

        path = large_layout.counts_path(adaptor.h5ad_path, adata.uns)
        why = large_layout.counts_mismatch(path, adata.obs_names, adata.var_names)
        if why is not None:
            raise large_layout.LargeLayoutError(
                f"This large dataset's counts can't be read into memory ({why}).")
        with h5py.File(path, "r") as f:
            adata.layers["counts"] = read_elem(f[large_layout.COUNTS_KEY])
        del adata.uns[large_layout.KEY]  # an ordinary dataset in memory from now on
        logger.info("Read the counts of large dataset %s into memory", adaptor.h5ad_path)
    return adata
