"""SeuratDisk's .h5Seurat files, read natively with h5py (no R).

An .h5Seurat is a Seurat object written by SeuratDisk's ``SaveH5Seurat``: root
attributes ``version``, ``active.assay``, ``project``; groups ``assays/<assay>/``
(``counts`` and ``data`` as compressed-column groups ``data`` / ``indices`` /
``indptr`` with a ``dims`` attribute = (features, cells), ``features``),
``meta.data/`` (one dataset per column; a factor is a group of ``levels`` and
1-based ``values``; a logical is integers with ``s3class`` "logical"; NA strings are
"NA"), ``reductions/<name>/cell.embeddings``, ``graphs/``, ``images/<slice>/`` and
``cell.names``. HDF5 stores R's arrays with their dimensions reversed, so h5py sees a
(cells x dims) R matrix as (dims, cells).

The result is what the R converter makes of the same object (``converter/convert.R``,
so the rest of scView cannot tell the two apart): X the active assay's ``counts``
(its ``data`` when counts are empty), var the gene names only, obs the meta.data,
obsm ``X_<reduction>``, a CITE-seq assay (ADT, Protein...) as obsm
``protein_counts`` (cells x proteins; ``core/protein.py`` adds the CLR). Graphs,
scale.data, commands and tools are left out. Visium images become obsm/uns
``spatial`` and obs ``library_id`` (``core/seurat_spatial.py``).
"""

from __future__ import annotations

import logging
from pathlib import Path

import anndata as ad
import h5py
import numpy as np
import pandas as pd
import scipy.sparse as sp

from scview.core import seurat_spatial

logger = logging.getLogger(__name__)

# convert.R's names for a CITE-seq antibody assay, in its order of preference.
ADT_ASSAYS = ("ADT", "Protein", "protein", "adt", "CITE", "Antibody")
MAX_PROTEINS = 1000  # convert.R: a larger "panel" is not antibodies
_R_INT_NA = np.iinfo(np.int32).min
# Seurat image classes read here: Visium (V1 table, V2 centroids); other FOV-based ones
# (Xenium, CosMx, Vizgen via Seurat) and SlideSeq without an image.
_VISIUM = ("VisiumV1", "VisiumV2")


class H5SeuratError(ValueError):
    """The file is not an .h5Seurat scView can read; the message says why."""


def is_h5seurat(f: h5py.File) -> bool:
    """SeuratDisk's layout: root attributes version + active.assay, groups assays/ and
    meta.data, the cell.names dataset."""
    keys = set(f.keys())
    return ({"version", "active.assay"} <= set(f.attrs.keys())
            and {"assays", "meta.data", "cell.names"} <= keys
            and isinstance(f["assays"], h5py.Group))


def summary(path: str | Path) -> dict:
    """The active assay, cells, genes and reductions, from the file's metadata."""
    with h5py.File(path, "r") as f:
        assay = _main_assay(f)
        group = f["assays"][assay]
        n_genes, n_cells = dims(matrix_node(group))
        return {"version": _attr(f, "version"), "assay": assay, "n_cells": n_cells,
                "n_genes": n_genes, "assays": sorted(f["assays"].keys()),
                "reductions": sorted(f.get("reductions", {}).keys()),
                "images": sorted(f.get("images", {}).keys())}


def read(path: str | Path) -> ad.AnnData:
    """The .h5Seurat at ``path`` as an AnnData (cells x genes), as convert.R makes it."""
    path = Path(path)
    with h5py.File(path, "r") as f:
        if not is_h5seurat(f):
            raise H5SeuratError(f"{path.name} is an HDF5 file without SeuratDisk's .h5Seurat layout.")
        assay = _main_assay(f)
        group = f["assays"][assay]
        cells = _strings(f["cell.names"])
        node = matrix_node(group)
        X = _matrix(node)
        n_genes, n_cells = dims(node)
        if n_cells != len(cells):
            raise H5SeuratError(f"The {assay} assay holds {n_cells:,} cells but the file names "
                                f"{len(cells):,}.")
        features = _strings(group["features"]) if "features" in group else []
        if len(features) != n_genes:
            raise H5SeuratError(f"The {assay} assay holds {n_genes:,} features but names "
                                f"{len(features):,}.")
        obs = _meta_data(f["meta.data"], cells)
        adata = ad.AnnData(X=X, obs=obs, var=pd.DataFrame(index=pd.Index(features)))
        for name, emb in _reductions(f, n_cells):
            adata.obsm[f"X_{name}"] = emb
        protein = _protein(f, assay, n_cells, adata.obs_names)
        if protein is not None:
            adata.obsm["protein_counts"] = protein
        slices = _images(f)
    if slices:
        result = seurat_spatial.attach(adata, slices)
        if result and "skipped" not in result:
            logger.info("h5Seurat %s: spatial slices %s", path.name, ", ".join(result["libraries"]))
    logger.info("Read %s: %s assay, %d cells x %d genes, reductions %s", path.name, assay,
                adata.n_obs, adata.n_vars, ", ".join(adata.obsm.keys()) or "none")
    return adata


# ---------------------------------------------------------------------------
# Assays
# ---------------------------------------------------------------------------


def _main_assay(f: h5py.File) -> str:
    """The active assay; when that is an antibody assay, RNA (or Spatial) instead, as the
    genes go in X and the antibodies beside them."""
    assays = list(f["assays"].keys())
    if not assays:
        raise H5SeuratError("The .h5Seurat file holds no assay.")
    active = _attr(f, "active.assay")
    if active in assays and active not in ADT_ASSAYS:
        return active
    for name in ("RNA", "Spatial"):
        if name in assays:
            return name
    return next((a for a in assays if a not in ADT_ASSAYS), assays[0])


def matrix_node(group: h5py.Group):
    """``counts``, or ``data`` when the counts are empty (a Seurat object kept without
    them), as convert.R falls back."""
    for slot in ("counts", "data"):
        node = group.get(slot)
        if node is not None and all(dims(node)):
            return node
    raise H5SeuratError(f"The {Path(group.name).name} assay holds neither counts nor data.")


def dims(node) -> tuple[int, int]:
    """(features, cells) of an assay matrix: a sparse group's ``dims``, a dense dataset's
    shape reversed (h5py sees R's (features, cells) as (cells, features))."""
    if isinstance(node, h5py.Group):
        dims = node.attrs.get("dims")
        if dims is None or len(dims) != 2:
            return 0, 0
        return int(dims[0]), int(dims[1])
    if node.ndim != 2:
        return 0, 0
    return int(node.shape[1]), int(node.shape[0])


def _matrix(node) -> sp.csr_matrix | np.ndarray:
    """An assay matrix as cells x features: R's compressed-column (features x cells) is
    compressed rows of its transpose, so the arrays are used as they are."""
    n_features, n_cells = dims(node)
    if isinstance(node, h5py.Group):
        return sp.csr_matrix((node["data"][:], node["indices"][:], node["indptr"][:]),
                             shape=(n_cells, n_features))
    return np.asarray(node[:])


def _protein(f: h5py.File, main: str, n_cells: int, obs_names: pd.Index) -> pd.DataFrame | None:
    """A CITE-seq assay's counts as a cells x proteins table (convert.R: the first of
    ADT_ASSAYS present, at most MAX_PROTEINS, the same cells)."""
    name = next((a for a in ADT_ASSAYS if a in f["assays"] and a != main), None)
    if name is None:
        return None
    group = f["assays"][name]
    try:
        node = matrix_node(group)
    except H5SeuratError:
        return None
    n_prot, n = dims(node)
    features = _strings(group["features"]) if "features" in group else []
    if n != n_cells or not 0 < n_prot <= MAX_PROTEINS or len(features) != n_prot:
        logger.info("Assay %s not kept as proteins (%d features x %d cells)", name, n_prot, n)
        return None
    m = _matrix(node)
    dense = m.toarray() if sp.issparse(m) else np.asarray(m)
    return pd.DataFrame(dense, index=obs_names, columns=pd.Index(features))


# ---------------------------------------------------------------------------
# meta.data and reductions
# ---------------------------------------------------------------------------


def _meta_data(group: h5py.Group, cells: list[str]) -> pd.DataFrame:
    """The cell table: columns in the ``colnames`` order, factors as categoricals."""
    index = _strings(group["_index"]) if "_index" in group else cells
    if len(index) != len(cells):
        index = cells
    order = [str(c) for c in _attr_list(group, "colnames")] or [k for k in group.keys() if k != "_index"]
    cols = {}
    for name in order:
        node = group.get(name)
        if node is None:
            continue
        values = _column(node, len(index))
        if values is None:
            logger.info("meta.data column %s not read (%s)", name, type(node).__name__)
            continue
        cols[name] = values
    return pd.DataFrame(cols, index=pd.Index(index))


def _column(node, n: int):
    if isinstance(node, h5py.Group):
        if "levels" not in node or "values" not in node:
            return None
        levels = _strings(node["levels"])
        codes = np.asarray(node["values"][:], dtype=np.int64) - 1
        if len(codes) != n:
            return None
        codes[(codes < 0) | (codes >= len(levels))] = -1  # NA
        return pd.Categorical.from_codes(codes, categories=pd.Index(levels).unique()) \
            if len(set(levels)) == len(levels) else pd.Categorical(
                [levels[c] if c >= 0 else None for c in codes])
    if node.ndim != 1 or node.shape[0] != n:
        return None
    raw = node[:]
    if _attr(node, "s3class") == "logical":
        vals = np.asarray(raw, dtype=np.int64)
        na = (vals != 0) & (vals != 1)
        if not na.any():
            return vals == 1
        return pd.array([None if m else bool(v) for v, m in zip(vals, na)], dtype="boolean")
    if raw.dtype.kind in ("O", "S", "U"):
        out = np.array([_s(v) for v in raw], dtype=object)
        out[out == "NA"] = None  # SeuratDisk writes R's NA strings as "NA"
        return out
    if raw.dtype.kind == "i" and raw.dtype.itemsize == 4 and (raw == _R_INT_NA).any():
        return np.where(raw == _R_INT_NA, np.nan, raw.astype(np.float64))
    if raw.dtype.kind in "iufb":
        return raw
    return None


def _reductions(f: h5py.File, n_cells: int):
    """(name, cells x dims float64) of each reduction with cell embeddings."""
    for name, group in (f.get("reductions") or {}).items():
        node = group.get("cell.embeddings") if isinstance(group, h5py.Group) else None
        if node is None or node.ndim != 2:
            continue
        emb = np.asarray(node[:], dtype=np.float64)
        if emb.shape[1] == n_cells:
            emb = emb.T  # R's (cells, dims) reversed
        if emb.shape[0] != n_cells:
            logger.info("Reduction %s not kept: %s for %d cells", name, emb.shape, n_cells)
            continue
        yield name, emb


# ---------------------------------------------------------------------------
# Images (spatial)
# ---------------------------------------------------------------------------


def _images(f: h5py.File) -> list[seurat_spatial.SeuratSlice]:
    out = []
    for name, group in (f.get("images") or {}).items():
        if not isinstance(group, h5py.Group):
            continue
        cls = _attr(group, "s4class").split(":")[-1]
        try:
            s = _slice(name, cls, group)
        except (KeyError, ValueError) as e:
            logger.info("Image %s (%s) not read: %s", name, cls, e)
            continue
        if s is None:
            logger.info("Image %s of class %s not read", name, cls or "unknown")
            continue
        out.append(s)
    return out


def _slice(name: str, cls: str, group: h5py.Group) -> seurat_spatial.SeuratSlice | None:
    sf = {}
    if isinstance(group.get("scale.factors"), h5py.Group):
        for k, node in group["scale.factors"].items():
            if isinstance(node, h5py.Dataset) and node.size:
                sf[k] = float(np.asarray(node[:]).ravel()[0])
    image = None
    if isinstance(group.get("image"), h5py.Dataset) and group["image"].ndim in (2, 3):
        image = np.asarray(group["image"][:]).T  # R's (rows, cols, channels) reversed
    grid = None
    coords = group.get("coordinates")
    if cls == "VisiumV1" or (isinstance(coords, h5py.Group) and "imagerow" in coords):
        cells = _strings(coords["_index"])
        imagerow = np.asarray(coords["imagerow"][:], dtype=np.float64)
        imagecol = np.asarray(coords["imagecol"][:], dtype=np.float64)
        xy = np.column_stack([imagecol, imagerow])
        if "row" in coords and "col" in coords:
            grid = (np.asarray(coords["row"][:], dtype=np.int64),
                    np.asarray(coords["col"][:], dtype=np.int64))
        return seurat_spatial.SeuratSlice(name, cls or "VisiumV1", cells, xy, image, sf, grid)
    centroids = group.get("boundaries/centroids")
    if isinstance(centroids, h5py.Group) and "coords" in centroids:
        c = np.asarray(centroids["coords"][:], dtype=np.float64)
        c = c.T if c.shape[0] == 2 and c.shape[1] != 2 else c  # R's (cells, 2) reversed
        cells = _strings(centroids["cells"])
        # VisiumV2: x = imagerow, y = imagecol (Seurat's own swap); another FOV: (x, y).
        xy = c[:, ::-1] if cls == "VisiumV2" else c
        return seurat_spatial.SeuratSlice(name, cls, cells, xy, image if cls in _VISIUM else None,
                                          sf if cls in _VISIUM else {}, None)
    if isinstance(coords, h5py.Group) and "x" in coords and "y" in coords:  # SlideSeq
        cells = _strings(coords["_index"])
        xy = np.column_stack([np.asarray(coords["x"][:], dtype=np.float64),
                              np.asarray(coords["y"][:], dtype=np.float64)])
        return seurat_spatial.SeuratSlice(name, cls, cells, xy, None, {}, None)
    return None


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def _s(v) -> str:
    return v.decode("utf-8", "replace") if isinstance(v, bytes) else str(v)


def _strings(node) -> list[str]:
    return [_s(v) for v in np.asarray(node[:]).ravel()]


def _attr_list(node, key: str) -> list:
    v = node.attrs.get(key)
    if v is None:
        return []
    return [_s(x) for x in np.asarray(v).ravel()]


def _attr(node, key: str) -> str:
    vals = _attr_list(node, key)
    return vals[0] if vals else ""
