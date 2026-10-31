"""Segmented cell and nucleus outlines of a spatial dataset, kept beside it on disk:

    DATA_DIR/outlines/<dataset id>/<kind>.h5    (kind: cell, nucleus)

An outline is only drawn, never analysed, so it stays out of the h5ad: it would be read on
every open and rewritten on every save. Like ``modality_store``, a store keeps every cell it
arrived with and is addressed by obs name, so filtering needs no rewrite. The folder goes with
the dataset.

Layout: ``ids`` (UTF-8, the dataset's obs names), ``offsets`` (int64, n + 1), ``xy`` (float32,
vertices x 2), attrs ``source``. Coordinates are in the units of ``obsm["spatial"]`` (Xenium
microns, Visium HD full-resolution pixels). A ring is open: its first vertex is not repeated.

``encode`` is the wire format of ``GET /datasets/{id}/spatial/outlines``, little-endian:
``int32 n_cells, int32 n_vertices, int32 offsets[n_cells + 1], float32 xy[2 * n_vertices]``, one
outline per obs name in order; a cell with no outline has an empty one.
"""

from __future__ import annotations

import os
import shutil
from collections.abc import Sequence
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

SUBDIR = "outlines"
KINDS = ("cell", "nucleus")
_STR = h5py.string_dtype("utf-8")


class OutlineError(RuntimeError):
    """An outline store is missing, or what was given to it does not fit."""


def store_dir(data_dir: str | Path, dataset_id: str) -> Path:
    return Path(data_dir) / SUBDIR / dataset_id


def store_path(data_dir: str | Path, dataset_id: str, kind: str) -> Path:
    if kind not in KINDS:
        raise OutlineError(f"Unknown outline kind '{kind}': use one of {', '.join(KINDS)}.")
    return store_dir(data_dir, dataset_id) / f"{kind}.h5"


def remove(data_dir: str | Path, dataset_id: str) -> None:
    shutil.rmtree(store_dir(data_dir, dataset_id), ignore_errors=True)


def kinds_of(data_dir: str | Path, dataset_id: str) -> list[str]:
    """The outlines this dataset has, in ``KINDS`` order."""
    return [k for k in KINDS if store_path(data_dir, dataset_id, k).is_file()]


def write(path: str | Path, ids: Sequence, offsets: np.ndarray, xy: np.ndarray, *, source: str = "") -> None:
    """Write one store through a part file: a stopped write leaves nothing that looks finished."""
    offsets = np.asarray(offsets, dtype=np.int64)
    xy = np.asarray(xy, dtype=np.float32)
    n = len(ids)
    if offsets.shape != (n + 1,) or (n + 1 and offsets[0] != 0) or np.any(np.diff(offsets) < 0):
        raise OutlineError(f"Outline offsets do not fit {n} cells.")
    if xy.ndim != 2 or xy.shape[1] != 2 or xy.shape[0] != offsets[-1]:
        raise OutlineError(f"Outline vertices do not fit their offsets ({xy.shape} for {offsets[-1]}).")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_name(path.name + ".part")
    try:
        with h5py.File(part, "w") as f:
            f.create_dataset("ids", data=np.asarray([str(i) for i in ids], dtype=object), dtype=_STR)
            f.create_dataset("offsets", data=offsets)
            f.create_dataset("xy", data=xy, chunks=True if len(xy) else None)
            f.attrs["source"] = source
        os.replace(part, path)
    finally:
        part.unlink(missing_ok=True)


def read_for(path: str | Path, obs_names: Sequence) -> tuple[np.ndarray, np.ndarray]:
    """(offsets, xy) with one outline per obs name, in that order. A name the store does not
    hold gets an empty outline."""
    with h5py.File(path, "r") as f:
        ids = pd.Index(f["ids"].asstr()[:])
        offsets = f["offsets"][:]
        xy = f["xy"][:]
    rows = ids.get_indexer(pd.Index([str(o) for o in obs_names]))
    lengths = np.where(rows >= 0, offsets[rows + 1] - offsets[rows], 0)
    out_offsets = np.zeros(len(rows) + 1, dtype=np.int64)
    out_offsets[1:] = np.cumsum(lengths)
    # The vertices of each kept row, in obs order, without a Python loop per cell.
    kept = rows >= 0
    starts = np.repeat(offsets[rows[kept]], lengths[kept])
    within = np.arange(out_offsets[-1], dtype=np.int64) - np.repeat(out_offsets[:-1][kept], lengths[kept])
    return out_offsets, xy[starts + within]


def encode(offsets: np.ndarray, xy: np.ndarray) -> bytes:
    n_cells, n_vertices = len(offsets) - 1, int(offsets[-1])
    if n_vertices > np.iinfo(np.int32).max:
        raise OutlineError(f"{n_vertices} outline vertices are more than one response can carry.")
    head = np.array([n_cells, n_vertices], dtype="<i4")
    return head.tobytes() + offsets.astype("<i4").tobytes() + xy.astype("<f4").tobytes()
