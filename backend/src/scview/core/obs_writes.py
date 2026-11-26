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


# The edit holding each dataset's lock now (``current_edit``), per event loop as the locks.
_edits: weakref.WeakKeyDictionary[asyncio.AbstractEventLoop, dict[str, "DatasetEdit"]] = (
    weakref.WeakKeyDictionary()
)


class DatasetEdit:
    """Work run inside ``dataset_edit``: each ``run`` is a worker thread the lock
    waits for, even when the request is cancelled. ``on_end`` callbacks run when the
    lock is handed back (the memory held for a backed dataset's read-in is let go there)."""

    def __init__(self) -> None:
        self.tasks: list[asyncio.Future] = []
        self._on_end: list = []

    def on_end(self, fn) -> None:
        self._on_end.append(fn)

    def _end(self) -> None:
        fns, self._on_end = self._on_end, []
        for fn in fns:
            try:
                fn()
            except Exception:  # noqa: BLE001 - the lock is handed back whatever a release does
                pass

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
    edits = _edits.setdefault(asyncio.get_running_loop(), {})
    edits[dataset_id] = edit

    def finish() -> None:
        if edits.get(dataset_id) is edit:
            del edits[dataset_id]
        edit._end()
        lock.release()

    try:
        yield edit
    finally:
        running = [t for t in edit.tasks if not t.done()]
        if not running:
            finish()
        else:
            left = len(running)

            def _done(_t) -> None:
                nonlocal left
                left -= 1
                if left == 0:
                    finish()

            for t in running:
                t.add_done_callback(_done)


def current_edit(dataset_id: str) -> DatasetEdit | None:
    """The ``dataset_edit`` holding this dataset's lock now, if any."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return None
    return _edits.get(loop, {}).get(dataset_id)


def is_locked(dataset_id: str) -> bool:
    """True while a request holds this dataset's edit lock (creates no lock)."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return False
    lock = _locks.get(loop, {}).get(dataset_id)
    return lock is not None and lock.locked()


async def edit_room(dm, dataset_id: str, adaptor, labels_only: bool = False) -> bool:
    """``working_copy.make_room`` before an obs / uns edit; whether the dataset is backed
    (the caller reloads it afterwards). Call it inside ``dataset_edit``: the room for the
    read-in is held until that edit ends. A large dataset is not read in: its tables are
    in memory already and the edit is written into its h5ad in place.

    ``labels_only``: the edit reads and changes obs and uns only (never X, raw, a layer or
    obsp), so a backed dataset is NOT read in either: no room is asked for and none held,
    and requests beside the edit have nothing to wait for. ``editable`` takes the same
    flag; ``save_edit`` then copies the file and rewrites obs and the changed uns entries
    (it needs ``uns_before``)."""
    if getattr(adaptor, "is_large", False):
        return False
    if skips_read_in(adaptor, labels_only):
        return True
    from scview.core import working_copy

    return await working_copy.make_room(dm, dataset_id, adaptor)


def skips_read_in(adaptor, labels_only: bool) -> bool:
    """Whether a ``labels_only`` edit leaves this backed dataset on disk: its file must be
    a current-format h5ad, since obs and uns are written into a copy of it (an older
    file is read in and written whole, as before)."""
    return bool(labels_only and getattr(adaptor.adata, "isbacked", False)
                and _current_h5ad(adaptor.h5ad_path))


def editable(adaptor, labels_only: bool = False):
    """The AnnData an obs / uns edit changes (in a worker thread, after ``edit_room``):
    the dataset in memory (``working_copy.in_memory``), or a large dataset's backed one,
    whose X stays on disk. ``labels_only`` (as given to ``edit_room``): a backed dataset's
    own object, its obs and uns in memory and its matrices left on disk."""
    if getattr(adaptor, "is_large", False) or skips_read_in(adaptor, labels_only):
        return adaptor.adata
    from scview.core import working_copy

    return working_copy.in_memory(adaptor)


def save_edit(adaptor, adata, output_path: str, uns_before: dict[str, int] | None = None,
              obsm_keys: tuple[str, ...] | list[str] | None = (),
              obsp_keys: tuple[str, ...] | list[str] | None = (), var: bool = False,
              source_unchanged: bool = False) -> None:
    """Save an obs / uns (/ obsm / obsp) edit of ``editable(adaptor)``: ``write_derived``,
    or for a large dataset its obs, the uns entries the edit replaced or removed,
    ``obsm_keys`` and ``obsp_keys`` written into its derived h5ad in place, X neither
    read nor copied. ``var``: the edit changed var too. ``source_unchanged``: see
    ``write_derived``."""
    if getattr(adaptor, "is_large", False):
        from scview.core import large_mode
        from scview.core.dataset_manager import large_derived

        large_derived(adaptor, output_path)
        changed, removed = large_mode.changed_uns(adata, uns_before)
        adaptor.write_in_place(obs=True, uns_keys=changed, uns_removed=removed,
                               obsm_keys=list(obsm_keys or ()), obsp_keys=list(obsp_keys or ()), var=var)
        return
    write_derived(adata, output_path, adaptor.h5ad_path, uns_before, obsm_keys, obsp_keys, var=var,
                  source_unchanged=source_unchanged)


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
    source_unchanged: bool = False,
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
    ``source_unchanged``: the caller's edit touched only obs, uns and the keys named, so the
    in-memory matrices are still ``source_path``'s (the first hand label on a dataset with
    no derived file yet): copied too, when that file is a current-format h5ad, rather than
    re-compressed (81 s on a 96k-cell, 1.2 GB file).
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
             or (os.path.exists(output_path) and os.path.samefile(source_path, output_path))
             or (source_unchanged and _current_h5ad(source_path)))
    ):
        import h5py
        from anndata.io import write_elem

        from scview.core import h5_copy

        # Everything the edit did not touch is copied into a new file and the rest written
        # beside it: deleting from a file copy left each old obs in it (``compact_copy``).
        gone = [k for k in uns_before if k not in adata.uns]
        try:
            with h5py.File(source_path, "r") as f:
                in_file = {g: set(f[g].keys()) if g in f else set() for g in ("uns", "obsm", "obsp")}
            fresh_uns = [k for k, v in adata.uns.items()
                         # From another file (source_unchanged), also what the dataset gained in
                         # memory since it was read (as a full write would save it).
                         if uns_before.get(k) != id(v) or (source_unchanged and k not in in_file["uns"])]
            if source_unchanged:
                obsm_keys = [*(obsm_keys or ()), *(k for k in adata.obsm if k not in in_file["obsm"])]
            obsm_keys, obsp_keys = list(dict.fromkeys(obsm_keys or ())), list(dict.fromkeys(obsp_keys or ()))
            h5_copy.compact_copy(source_path, partial, [
                "obs", *(["var"] if var else []), *(f"uns/{k}" for k in (*gone, *fresh_uns)),
                *(f"obsm/{k}" for k in obsm_keys), *(f"obsp/{k}" for k in obsp_keys)])
            with h5py.File(partial, "a") as f:
                write_elem(f, "obs", adata.obs)
                if var:
                    write_elem(f, "var", adata.var)
                for name, keys, held in (("uns", fresh_uns, adata.uns), ("obsm", obsm_keys, adata.obsm),
                                         ("obsp", obsp_keys, adata.obsp)):
                    if name not in f and (keys or name == "uns"):
                        write_elem(f, name, {})
                    for key in keys:
                        if key in held:
                            write_elem(f[name], key, held[key])
        except Exception:
            if os.path.exists(partial):
                os.remove(partial)
            raise
    else:
        adata.write_h5ad(partial, compression="gzip")
    os.replace(partial, output_path)


def _current_h5ad(path: str) -> bool:
    """An h5ad written by anndata 0.8 or later (each element names its encoding), so obs
    and uns written into a copy of it read back with the rest."""
    import h5py

    try:
        with h5py.File(path, "r") as f:
            enc = f.attrs.get("encoding-type")
            return (enc.decode() if isinstance(enc, bytes) else enc) == "anndata"
    except (OSError, ValueError):
        return False
