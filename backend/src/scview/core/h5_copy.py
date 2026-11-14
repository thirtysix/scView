"""Copy an HDF5 group or dataset into another file with its object references still valid.

anndata 0.7.x wrote each categorical column with an attribute ``categories`` that is an HDF5
OBJECT REFERENCE to ``obs/__categories/<column>``. ``h5py``'s ``copy`` (H5Ocopy) copies the
reference as it is, so in the new file it points into the old one and anndata fails on read with
"Invalid HDF5 object reference". scVelo's downloadable datasets (gastrulation_erythroid and
others) are in this layout. ``copy`` here copies, then points every reference attribute in the
copy at the copied object (or, for one outside the copied tree, at the object of the same path
in the new file).
"""

from __future__ import annotations

import h5py
import numpy as np


def copy(src: h5py.Group | h5py.Dataset, parent: h5py.Group, name: str) -> None:
    """``src`` into ``parent[name]``, references re-pointed."""
    src.file.copy(src, parent, name=name)
    repoint(src, parent[name])


def _target(src_root: str, dst_root: str, path: str) -> str:
    if path == src_root or path.startswith(src_root.rstrip("/") + "/"):
        return dst_root + path[len(src_root):]
    return path


def _fixed(value, src_file: h5py.File, dst_file: h5py.File, src_root: str, dst_root: str):
    """The attribute value with its object references re-made in ``dst_file``; None when it
    holds none (or one cannot be resolved there)."""
    if isinstance(value, h5py.Reference):
        if not value:
            return None
        path = src_file[value].name
        dst = dst_file.get(_target(src_root, dst_root, path))
        return None if dst is None else dst.ref
    if isinstance(value, np.ndarray) and h5py.check_dtype(ref=value.dtype) is h5py.Reference:
        out = np.empty(value.shape, dtype=h5py.ref_dtype)
        for i, v in np.ndenumerate(value):
            got = _fixed(v, src_file, dst_file, src_root, dst_root) if v else None
            out[i] = got if got is not None else h5py.Reference()
        return out
    return None


def repoint(src: h5py.Group | h5py.Dataset, dst: h5py.Group | h5py.Dataset) -> int:
    """Re-make the reference attributes of ``dst`` (a copy of ``src``) and everything under it.
    Returns how many were re-made."""
    src_file, dst_file = src.file, dst.file
    n = 0

    def fix(s, d) -> None:
        nonlocal n
        for key in list(s.attrs.keys()):
            try:
                value = s.attrs[key]
            except (OSError, TypeError):  # pragma: no cover - an attribute h5py cannot read
                continue
            got = _fixed(value, src_file, dst_file, src.name, dst.name)
            if got is not None:
                d.attrs[key] = got
                n += 1

    fix(src, dst)
    if isinstance(src, h5py.Group):
        def visit(rel: str, obj) -> None:
            if isinstance(obj, (h5py.Group, h5py.Dataset)) and rel in dst:
                fix(obj, dst[rel])

        src.visititems(visit)
    return n
