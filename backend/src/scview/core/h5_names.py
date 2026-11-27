"""Names with "/" in an h5ad: escaped on write, restored on read.

HDF5 reads "/" in a key as a path. anndata writes every dict key and every DataFrame
column under its own name, so a label "T/NK" used as a key becomes a group "T" holding
"NK". What that did before this module (anndata 0.12.19):

- "T" and "T/NK" together (marker ``pts`` columns, a colour dict, two obs columns): the
  write fails, "Incompatible object (Dataset) already exists". A finished run lost its save.
- "T/NK" alone as a dict key (uns, layers, obsm): written, read back as ``{"T": {"NK": ..}}``.
- "T/NK" alone as a DataFrame column: written nested, read back right.
- "", ".", a leading "/": the file no longer opens.

anndata will refuse "/" in h5 keys outright (its FutureWarning), so nesting is no way out.

Here every such name is stored percent-escaped ("T%2fNK") and read back as it was typed.
Nothing else is marked in the file: an outside reader (plain anndata, scanpy, R) opens it
and sees the escaped name. Only names that need it change: "/" -> "%2f", a "%" that would
read as an escape -> "%25", "." -> "%2e", "" -> "%00". A file written before this module
holds none of these, so its names read as they always did.

``install()`` wraps anndata's own dict and DataFrame writers and readers, so
``write_h5ad``, ``read_h5ad`` (backed too), ``write_elem``, ``read_elem`` and ``write_zarr``
all follow it; ``scview/__init__`` installs it when anndata is first imported. Code that
addresses an h5 group by a name itself (``f["obs"][column]``, ``write_elem(f["uns"], key,
value)``) uses ``stored`` / ``child`` / ``has`` / ``drop`` / ``names`` / ``write_elem``
below: ``child`` also finds the nested entry an older file holds.

``REPLAY_CODE`` is copied into replay scripts (keep it self-contained).
"""

from __future__ import annotations

import contextlib
import re
from typing import Any

# Lower case on purpose: core/uns_keys.py has long put "%2F" (upper case) INSIDE the names
# of marker and enrichment keys, in memory and in every file since. Those names are not
# escapes of this module and pass through unchanged, both ways.
_NEEDS = re.compile(r"/|%(?=25|2f|2e|00)")
_ESCAPED = re.compile(r"%2f|%25(?=25|2f|2e|00)")
_PLAIN = {"": "%00", ".": "%2e"}
_WHOLE = {v: k for k, v in _PLAIN.items()}


def needs_escape(name: Any) -> bool:
    """A name HDF5 cannot hold as one key, or one that would read back as an escape."""
    return isinstance(name, str) and (name in _PLAIN or _NEEDS.search(name) is not None)


def escape(name: Any) -> Any:
    """The h5 key ``name`` is stored under. One-to-one; a name that needs nothing is itself."""
    if not isinstance(name, str):
        return name
    if name in _PLAIN:
        return _PLAIN[name]
    if _NEEDS.search(name) is None:
        return name
    return _NEEDS.sub(lambda m: "%2f" if m.group() == "/" else "%25", name)


def unescape(key: Any) -> Any:
    """The name an h5 key stands for (``escape`` undone)."""
    if not isinstance(key, str) or "%" not in key:
        return key
    if key in _WHOLE:
        return _WHOLE[key]
    return _ESCAPED.sub(lambda m: "/" if m.group() == "%2f" else "%", key)


def escape_frame(df):
    """``df`` with escaped column names and index name (a shallow plain DataFrame), or
    ``df`` itself when no name needs it."""
    import pandas as pd

    if not (any(needs_escape(c) for c in df.columns) or needs_escape(df.index.name)):
        return df
    out = pd.DataFrame(df, copy=False)  # not a view of an AnnData: renaming must not copy it
    out.columns = [escape(c) for c in df.columns]
    if needs_escape(df.index.name):
        out.index = out.index.rename(escape(df.index.name))
    return out


def unescape_frame(df):
    """Column names and index name as typed, in place."""
    if any(isinstance(c, str) and "%" in c for c in df.columns):
        df.columns = [unescape(c) for c in df.columns]
    if isinstance(df.index.name, str) and "%" in df.index.name:
        df.index.name = unescape(df.index.name)
    return df


def escape_keys(d: dict) -> dict:
    """``d`` keyed by escaped names (a shallow copy), or ``d`` itself when no key needs it."""
    if not any(needs_escape(k) for k in d):
        return d
    return {escape(k): v for k, v in d.items()}


# --- addressing an h5 group by a name -------------------------------------------------


def stored(name: str) -> str:
    """The key of ``name`` inside its group ("uns/" + stored(key) for a path)."""
    return escape(name)


def path(*names: str) -> str:
    """An h5 path of names, each escaped: ``path("uns", key)``."""
    return "/".join(escape(n) for n in names)


def _old(group, name: str):
    """The nested entry a file written before the escape holds for ``name``, or None: h5py
    made the groups on the way to it itself, so none of them says what it encodes (a dict
    "T" holding a key "NK" does, and is not the name "T/NK")."""
    if "/" not in name.strip("/") or not name or name != name.strip("/") or "//" in name:
        return None
    try:
        at = group
        for part in name.split("/")[:-1]:
            at = at.get(part, getclass=False)
            if at is None or not hasattr(at, "keys") or "encoding-type" in at.attrs:
                return None
        return group.get(name)
    except (KeyError, ValueError, TypeError):
        return None


def has(group, name: str) -> bool:
    return escape(name) in group or _old(group, name) is not None


def child(group, name: str):
    """``group[name]``; KeyError (naming it) when absent."""
    key = escape(name)
    if key in group:
        return group[key]
    old = _old(group, name)
    if old is None:
        raise KeyError(name)
    return old


def drop(group, name: str) -> bool:
    """``del group[name]`` when present, an older file's nested entry too."""
    found = False
    key = escape(name)
    if key in group:
        del group[key]
        found = True
    if _old(group, name) is not None:
        del group[name]
        found = True
    return found


def names(group) -> list[str]:
    """The names of the group's entries, as typed."""
    return [unescape(k) for k in group.keys()]


def write_elem(group, name: str, value, **kwargs) -> None:
    """``anndata.io.write_elem(group, name, value)`` with ``name`` as ONE key of ``group``
    (never a path), replacing what is there."""
    from anndata.io import write_elem as _write

    _install()
    if _old(group, name) is not None:
        del group[name]
    _write(group, escape(name), value, **kwargs)


# --- anndata's writers and readers ----------------------------------------------------

_installed = False
_plain = False


@contextlib.contextmanager
def plain():
    """anndata's own writers and readers, as before this module: for tests that need a
    file as an earlier version wrote it (names with "/" nested). Not for the server: the
    switch is process-wide."""
    global _plain
    was, _plain = _plain, True
    try:
        yield
    finally:
        _plain = was


def _wrap_write(func, fix):
    def write(f, k, elem, *args, **kwargs):
        return func(f, k, elem if _plain else fix(elem), *args, **kwargs)

    write.__wrapped__ = func
    write._scview_names = True
    return write


def _wrap_read(func, fix):
    def read(elem, *args, **kwargs):
        out = func(elem, *args, **kwargs)
        return out if _plain else fix(out)

    read.__wrapped__ = func
    read._scview_names = True
    return read


def _unescape_keys(d):
    if isinstance(d, dict) and any(isinstance(k, str) and "%" in k for k in d):
        return {unescape(k): v for k, v in d.items()}
    return d


def _install() -> None:
    """Wrap anndata's registered dict and DataFrame writers / readers (h5 and zarr). Their
    registry is private: ``tests/test_h5_names.py`` fails loudly if a new anndata moves it."""
    global _installed
    if _installed:
        return
    import pandas as pd
    from anndata._io.specs.registry import _REGISTRY

    wrapped = 0
    for key, func in list(_REGISTRY.write.items()):
        src = key[1]
        if getattr(func, "_scview_names", False) or not isinstance(src, type):
            continue
        if issubclass(src, pd.DataFrame):
            _REGISTRY.write[key] = _wrap_write(func, escape_frame)
            wrapped += 1
        elif issubclass(src, dict):
            _REGISTRY.write[key] = _wrap_write(func, escape_keys)
            wrapped += 1
    for table in (_REGISTRY.read, _REGISTRY.read_partial):
        for key, func in list(table.items()):
            kind = getattr(key[1], "encoding_type", None)
            if getattr(func, "_scview_names", False):
                continue
            if kind == "dataframe":
                table[key] = _wrap_read(func, unescape_frame)
                wrapped += 1
            elif kind == "dict":
                table[key] = _wrap_read(func, _unescape_keys)
                wrapped += 1
    if not wrapped:
        raise RuntimeError("anndata's element registry has no dict / dataframe entries to wrap (h5_names)")
    _installed = True


def install() -> None:
    _install()


def installed() -> bool:
    return _installed


def install_on_import() -> None:
    """Install now when anndata is loaded, else the moment it is first imported (children
    that never read a file, like the Leiden one, do not pay for the import)."""
    import sys

    if "anndata" in sys.modules and getattr(sys.modules["anndata"], "AnnData", None) is not None:
        _install()
        return
    if any(isinstance(f, _OnAnndata) for f in sys.meta_path):
        return
    sys.meta_path.insert(0, _OnAnndata())


class _OnAnndata:
    """A finder that finds nothing: it sees ``import anndata`` start, steps aside, and has
    the real loader install the wrappers once the package has run."""

    def find_spec(self, fullname, path=None, target=None):
        if fullname != "anndata":
            return None
        import importlib.util
        import sys

        sys.meta_path.remove(self)
        spec = importlib.util.find_spec(fullname)
        if spec is None or spec.loader is None:
            return spec
        loader = spec.loader
        run = loader.exec_module

        def exec_module(module):
            run(module)
            _install()

        try:
            loader.exec_module = exec_module
        except AttributeError:  # a loader that takes no attributes: install at first use
            pass
        return spec


# --- replay scripts -------------------------------------------------------------------

REPLAY_NEEDS_PATTERN = _NEEDS.pattern
REPLAY_PLAIN_NAMES = dict(_PLAIN)
REPLAY_CONSTANTS = ("REPLAY_NEEDS_PATTERN", "REPLAY_PLAIN_NAMES")


def h5_key(name):
    """The key a name is stored under in the h5ad ("T/NK" -> "T%2fNK"): HDF5 reads "/" as a path."""
    import re

    if not isinstance(name, str):
        return name
    if name in REPLAY_PLAIN_NAMES:
        return REPLAY_PLAIN_NAMES[name]
    return re.sub(REPLAY_NEEDS_PATTERN, lambda m: "%2f" if m.group() == "/" else "%25", name)


def h5_safe_names(adata):
    """Rename, in place, every obs / var column, uns key (at any depth), DataFrame column and
    layer / obsm / varm / obsp name that holds "/" to its ``h5_key``; the names changed."""
    import pandas as pd

    changed = []

    def key(name):
        new = h5_key(name)
        if new != name:
            changed.append(name)
        return new

    def frame(df):
        out = pd.DataFrame(df, copy=False)
        out.columns = [key(c) for c in df.columns]
        return out

    def walk(value):
        if isinstance(value, pd.DataFrame):
            return frame(value)
        if isinstance(value, dict):
            return {key(k): walk(v) for k, v in value.items()}
        return value

    for df in (adata.obs, adata.var):
        df.columns = [key(c) for c in df.columns]
    for slot in (adata.obsm, adata.varm, adata.layers, adata.obsp, adata.varp):
        for k in list(slot.keys()):
            slot[key(k)] = walk(slot.pop(k))
    for k in list(adata.uns.keys()):
        adata.uns[key(k)] = walk(adata.uns.pop(k))
    return changed


REPLAY_CODE = (h5_key, h5_safe_names)
