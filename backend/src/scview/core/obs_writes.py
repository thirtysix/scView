"""Saving a dataset after an obs edit made outside the pipeline.

Hand labels (``core/user_labels.py``) and a kept clustering resolution
(``core/resolution_sweep.py``) each add or change one obs column (plus a few uns
entries), so ``write_derived`` can save them without rewriting the matrix.
Writes to one dataset are serialised: two requests writing the same
``.partial`` file at once would corrupt it.
"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
import os
import shutil
import weakref

from scview.core.uns_keys import markers_key

# One set per event loop: an asyncio.Lock belongs to the loop it was first
# contended on (one loop in the server; a fresh one per test).
_locks: weakref.WeakKeyDictionary[asyncio.AbstractEventLoop, dict[str, asyncio.Lock]] = (
    weakref.WeakKeyDictionary()
)


def forget_column_results(adata, column: str) -> None:
    """Drop the markers and cached enrichment computed for ``column``.

    Both are keyed by group name, so after cells change group a surviving label
    would otherwise be served results for its old members. Markers are computed
    again on the next visit to the column.
    """
    from scview.core.pipeline import drop_enrichment_cache

    adata.uns.pop(markers_key(column), None)
    rgg = adata.uns.get("rank_genes_groups")
    if isinstance(rgg, dict) and rgg.get("params", {}).get("groupby") == column:
        del adata.uns["rank_genes_groups"]
    drop_enrichment_cache(adata, column)


def dataset_lock(dataset_id: str) -> asyncio.Lock:
    """The lock held while a request edits and saves this dataset."""
    locks = _locks.setdefault(asyncio.get_running_loop(), {})
    lock = locks.get(dataset_id)
    if lock is None:
        lock = locks[dataset_id] = asyncio.Lock()
    return lock


class DatasetEdit:
    """Work run inside ``dataset_edit``: each ``run`` is a worker thread the lock
    waits for, even when the request is cancelled."""

    def __init__(self) -> None:
        self.tasks: list[asyncio.Future] = []

    async def run(self, fn, *args):
        task = asyncio.ensure_future(asyncio.to_thread(fn, *args))
        self.tasks.append(task)
        # Shielded: a cancelled request stops waiting, the thread carries on.
        return await asyncio.shield(task)


@asynccontextmanager
async def dataset_edit(dataset_id: str):
    """Hold the dataset's edit lock for a request that edits it in worker threads
    (``edit.run(fn, ...)``). A client that goes away cancels the request, but the
    thread keeps writing; the lock is then handed back only when the thread ends,
    so no other edit can start on a half-written file."""
    lock = dataset_lock(dataset_id)
    await lock.acquire()
    edit = DatasetEdit()
    try:
        yield edit
    finally:
        running = [t for t in edit.tasks if not t.done()]
        if not running:
            lock.release()
        else:
            left = len(running)

            def _done(_t) -> None:
                nonlocal left
                left -= 1
                if left == 0:
                    lock.release()

            for t in running:
                t.add_done_callback(_done)


def is_locked(dataset_id: str) -> bool:
    """True while a request holds this dataset's edit lock (creates no lock)."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return False
    lock = _locks.get(loop, {}).get(dataset_id)
    return lock is not None and lock.locked()


async def edit_room(dm, dataset_id: str, adaptor) -> bool:
    """``working_copy.make_room`` before an obs / uns edit; whether the dataset is read
    into memory for it (the caller reloads it afterwards). A large dataset is not: its
    tables are in memory already and the edit is written into its h5ad in place."""
    if getattr(adaptor, "is_large", False):
        return False
    from scview.core import working_copy

    return await working_copy.make_room(dm, dataset_id, adaptor)


def editable(adaptor):
    """The AnnData an obs / uns edit changes (in a worker thread, after ``edit_room``):
    the dataset in memory (``working_copy.in_memory``), or a large dataset's backed one,
    whose X stays on disk."""
    if getattr(adaptor, "is_large", False):
        return adaptor.adata
    from scview.core import working_copy

    return working_copy.in_memory(adaptor)


def save_edit(adaptor, adata, output_path: str, uns_before: dict[str, int] | None = None,
              obsm_keys: tuple[str, ...] | list[str] | None = (),
              obsp_keys: tuple[str, ...] | list[str] | None = (), var: bool = False) -> None:
    """Save an obs / uns (/ obsm / obsp) edit of ``editable(adaptor)``: ``write_derived``,
    or for a large dataset its obs, the uns entries the edit replaced or removed,
    ``obsm_keys`` and ``obsp_keys`` written into its derived h5ad in place, X neither
    read nor copied. ``var``: the edit changed var too."""
    if getattr(adaptor, "is_large", False):
        from scview.core import large_mode
        from scview.core.dataset_manager import large_derived

        large_derived(adaptor, output_path)
        changed, removed = large_mode.changed_uns(adata, uns_before)
        adaptor.write_in_place(obs=True, uns_keys=changed, uns_removed=removed,
                               obsm_keys=list(obsm_keys or ()), obsp_keys=list(obsp_keys or ()), var=var)
        return
    write_derived(adata, output_path, adaptor.h5ad_path, uns_before, obsm_keys, obsp_keys, var=var)


def uns_snapshot(adata) -> dict[str, int]:
    """Identity of each top-level ``uns`` entry, taken before an edit, so
    ``write_derived`` can tell which entries the edit replaced or removed."""
    return {k: id(v) for k, v in adata.uns.items()}


def write_derived(
    adata,
    output_path: str,
    source_path: str | None = None,
    uns_before: dict[str, int] | None = None,
    obsm_keys: tuple[str, ...] | list[str] | None = (),
    obsp_keys: tuple[str, ...] | list[str] | None = (),
    var: bool = False,
) -> None:
    """Write ``adata`` to its derived file: beside it first, then swapped in, so a
    failed write leaves the old file and a backed dataset reading it intact.

    When the dataset was loaded from ``output_path`` itself and the edit only
    touched obs and uns (``uns_before`` from ``uns_snapshot``), the file is
    copied and just obs plus the replaced or removed uns entries are rewritten:
    a full write re-compresses the whole matrix (8 s on a 14k-cell dense file,
    against 0.1 s). Edits must replace an uns entry, not mutate it in place.
    ``obsm_keys``: obsm entries the edit set or removed, rewritten the same way;
    ``obsp_keys`` the same for obsp (a graph, core/wnn.py); ``var``: var rewritten too. A backed
    ``adata`` is saved the same way from ``source_path``, wherever that is: its matrices were never read in.
    """
    from scview.core import uns_history

    uns_history.before_save(output_path, adata)
    partial = output_path + ".partial"
    # anndata refuses to write an index named like a column with other values (an
    # uploaded h5ad's obs "cell_id"); the pipeline's save and the ingest drop the name too.
    for frame in (adata.obs, adata.var):
        if frame.index.name is not None and frame.index.name in frame.columns:
            frame.index.name = None
    if (
        uns_before is not None
        and source_path
        and (getattr(adata, "isbacked", False)
             or (os.path.exists(output_path) and os.path.samefile(source_path, output_path)))
    ):
        import h5py
        from anndata.io import write_elem

        shutil.copyfile(source_path, partial)
        try:
            with h5py.File(partial, "a") as f:
                del f["obs"]
                write_elem(f, "obs", adata.obs)
                if var:
                    del f["var"]
                    write_elem(f, "var", adata.var)
                if "uns" not in f:
                    write_elem(f, "uns", {})
                group = f["uns"]
                for key in [k for k in uns_before if k not in adata.uns]:
                    if key in group:
                        del group[key]
                for key, value in adata.uns.items():
                    if uns_before.get(key) != id(value):
                        if key in group:
                            del group[key]
                        write_elem(group, key, value)
                if obsm_keys and "obsm" not in f:
                    write_elem(f, "obsm", {})
                for key in obsm_keys or ():
                    if key in f["obsm"]:
                        del f["obsm"][key]
                    if key in adata.obsm:
                        write_elem(f["obsm"], key, adata.obsm[key])
                if obsp_keys and "obsp" not in f:
                    write_elem(f, "obsp", {})
                for key in obsp_keys or ():
                    if key in f["obsp"]:
                        del f["obsp"][key]
                    if key in adata.obsp:
                        write_elem(f["obsp"], key, adata.obsp[key])
        except Exception:
            os.remove(partial)
            raise
    else:
        adata.write_h5ad(partial, compression="gzip")
    os.replace(partial, output_path)
