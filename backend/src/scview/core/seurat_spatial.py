"""Seurat's Visium images in scView's spatial layout (``core/spatial.py``, contract.md).

Seurat keeps one image per slice in ``obj@images[[slice]]``, written to an .h5Seurat
(``core/ingestion/h5seurat.py``) or carried over by the R converter (``convert.R``):

* ``VisiumV1`` (Seurat 3.2 - 5.0): ``coordinates``, a table of ``tissue``, ``row``,
  ``col`` (the Visium array grid) and ``imagerow`` / ``imagecol`` in FULL-resolution
  pixels (Space Ranger's pxl_row_in_fullres / pxl_col_in_fullres).
* ``VisiumV2`` (Seurat >= 5.1): ``boundaries$centroids``, centres (x = imagerow,
  y = imagecol, full-res pixels) without the array grid.

Both hold one ``image`` (the low-res PNG Seurat read, numbers 0-1) and
``scale.factors`` (``spot``, ``fiducial``, ``hires``, ``lowres``); Seurat places the
spots on the image with ``lowres``, so the image is stored as "lowres". obsm
["spatial"] is (imagecol, imagerow): x across, y down, as Space Ranger's own import.

A VisiumV2 slice lost the array grid: ``visium_grid`` recovers it from the spot
centres (a hexagonal lattice, checked to fit every spot), so the slice still counts
as Visium spots; when it does not fit, the slice is "cells" (contract.md).
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import pandas as pd

from scview.core import spatial

logger = logging.getLogger(__name__)

SOURCE = "seurat"
GRID_INFERRED = "inferred from the spot centres (Seurat VisiumV2 keeps no array grid)"
# A recovered grid is kept only when every spot sits this close to its lattice point
# (in lattice steps: 0.5 would be halfway to the next one).
GRID_TOLERANCE = 0.25


class SeuratSlice:
    """One Seurat image: its spots (cell names), their full-res (x, y), the array grid
    when known, the image (HxWx3) and the scale factors."""

    def __init__(self, name: str, image_class: str, cells: list[str], xy: np.ndarray,
                 image: np.ndarray | None, scale_factors: Mapping,
                 grid: tuple[np.ndarray, np.ndarray] | None = None) -> None:
        self.name = name
        self.image_class = image_class
        self.cells = [str(c) for c in cells]
        self.xy = np.asarray(xy, dtype=np.float64).reshape(-1, 2)
        self.image = image
        self.scale_factors = dict(scale_factors)
        self.grid = grid
        self.grid_inferred = False


def as_uint8_image(img) -> np.ndarray | None:
    """Seurat's image (numbers 0-1, or 0-255) as uint8 HxWx3; None when it is no image."""
    if img is None:
        return None
    a = np.asarray(img)
    if a.ndim == 2:
        a = np.stack([a] * 3, axis=-1)
    if a.ndim != 3 or a.shape[2] < 3 or a.shape[0] < 1 or a.shape[1] < 1:
        return None
    a = a[:, :, :3]
    if a.dtype == np.uint8:
        return np.ascontiguousarray(a)
    a = np.nan_to_num(a.astype(np.float64), nan=0.0)
    if a.max(initial=0.0) <= 1.0:
        a = a * 255.0
    return np.ascontiguousarray(np.clip(np.round(a), 0, 255).astype(np.uint8))


def visium_grid(xy: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    """Visium array (row, col) of spot centres on a hexagonal lattice: along a row the
    columns step by 2, the next row is shifted by one column, and row + col is even
    counted from the lowest row and column of tissue. Checked on Space Ranger's own
    positions: equal to its grid less a constant on V1_Human_Lymph_Node (4,035 spots),
    and its mirror image (the same neighbours) on CytAssist_FFPE_Mouse_Brain, whose
    column 0 is on the image's right. None when the spots do not sit on one lattice
    (fewer than 3, a spot off its point by more than ``GRID_TOLERANCE`` steps, two on
    one point, or too few neighbours to measure the steps: a sparse subset)."""
    from scipy.spatial import cKDTree

    xy = np.asarray(xy, dtype=np.float64)
    if xy.ndim != 2 or len(xy) < 3 or not np.isfinite(xy).all():
        return None
    dist, idx = cKDTree(xy).query(xy, k=min(7, len(xy)))
    d = float(np.median(dist[:, 1]))
    if not np.isfinite(d) or d <= 0:
        return None
    # Neighbour steps: on the lattice, one of six, repeating every 60 degrees.
    near = (dist[:, 1:] > 0) & (dist[:, 1:] < 1.2 * d)
    vec = xy[idx[:, 1:]][near] - np.repeat(xy, near.sum(axis=1), axis=0)
    ang = np.arctan2(vec[:, 1], vec[:, 0])
    theta = np.angle(np.exp(6j * ang).mean()) / 6
    # Of the three lattice directions, the one nearest the image's x axis runs along the
    # rows (columns + 2); +60 degrees from it (down the image) is the next row, column + 1.
    row_dir = min((theta + k * np.pi / 3 for k in range(3)), key=lambda a: abs(np.sin(a)))
    if np.cos(row_dir) < 0:
        row_dir += np.pi
    which = np.round(((ang - row_dir) % (2 * np.pi)) / (np.pi / 3)).astype(int) % 6
    steps = _STEPS[which]  # (column, row) of each neighbour step
    # The lattice -> pixel map from every step at once (shear and all), then fitted again
    # with an origin on every spot's lattice point.
    A = np.linalg.lstsq(steps, vec, rcond=None)[0].T  # pixels = A @ (column, row)
    if not np.isfinite(A).all() or abs(np.linalg.det(A)) < 1e-9:
        return None
    lat = np.linalg.solve(A, (xy - xy[0]).T).T
    for _ in range(2):
        k = np.round(lat)
        design = np.column_stack([k, np.ones(len(k))])
        fit = np.linalg.lstsq(design, xy, rcond=None)[0]  # rows: column, row, origin
        A, origin = fit[:2].T, fit[2]
        if abs(np.linalg.det(A)) < 1e-9:
            return None
        lat = np.linalg.solve(A, (xy - origin).T).T
    k = np.round(lat)
    if np.abs(lat - k).max() > GRID_TOLERANCE:
        return None
    cols, rows = k[:, 0] - k[:, 0].min(), k[:, 1] - k[:, 1].min()
    if len(np.unique((rows + cols) % 2)) != 1:
        return None
    if (rows[0] + cols[0]) % 2:
        cols = cols + 1
    pairs = rows.astype(np.int64) * 1_000_003 + cols.astype(np.int64)
    if len(np.unique(pairs)) != len(pairs):
        return None
    return rows.astype(np.int64), cols.astype(np.int64)


# The six neighbour steps of Visium's lattice as (column, row), counter-clockwise from
# along the row in image coordinates (y down): +60 degrees is the row below.
_STEPS = np.array([[2, 0], [1, 1], [-1, 1], [-2, 0], [-1, -1], [1, -1]], dtype=np.float64)


def attach(adata, slices: list[SeuratSlice]) -> dict | None:
    """Put Seurat's slices on the cells of ``adata``: obsm["spatial"], obs library_id
    (the slice names), array_row / array_col (when every slice has its grid, recovered
    or kept), uns["spatial"][slice] and uns["scview_spatial"]. Every cell must be a
    spot of one slice, else nothing is attached (the reason is logged and returned
    under "skipped"). Returns the summary for the ingest history (None without slices)."""
    slices = [s for s in slices if len(s.cells)]
    if not slices:
        return None
    names = pd.Index(adata.obs_names.astype(str))
    n = adata.n_obs
    coords = np.full((n, 2), np.nan)
    libs = np.full(n, None, dtype=object)
    rows_arr = np.full(n, -1, dtype=np.int64)
    cols_arr = np.full(n, -1, dtype=np.int64)
    for s in slices:
        if s.grid is None:
            s.grid = visium_grid(s.xy)
            s.grid_inferred = s.grid is not None
        pos = names.get_indexer(pd.Index(s.cells))
        keep = pos >= 0
        coords[pos[keep]] = s.xy[keep]
        libs[pos[keep]] = s.name
        if s.grid is not None:
            rows_arr[pos[keep]] = s.grid[0][keep]
            cols_arr[pos[keep]] = s.grid[1][keep]
    placed = np.isfinite(coords).all(axis=1)
    if not placed.all():
        why = (f"{int((~placed).sum()):,} of {n:,} cells are in none of the object's images "
               f"({', '.join(s.name for s in slices)}), so the tissue positions were left out")
        logger.warning("Seurat spatial data not attached: %s", why)
        return {"skipped": why}
    adata.obsm[spatial.KEY] = coords
    adata.obs[spatial.LIBRARY_COLUMN] = pd.Categorical(libs, categories=[s.name for s in slices])
    if all(s.grid is not None for s in slices):
        adata.obs["array_row"] = rows_arr
        adata.obs["array_col"] = cols_arr
    uns = {}
    for s in slices:
        sf = s.scale_factors
        scalefactors = {k: float(sf[src]) for k, src in (
            ("tissue_lowres_scalef", "lowres"), ("tissue_hires_scalef", "hires"),
            ("spot_diameter_fullres", "spot"), ("fiducial_diameter_fullres", "fiducial"))
            if _finite(sf.get(src))}
        img = as_uint8_image(s.image)
        meta = {"source": SOURCE, "spaceranger_version": None, "seurat_image_class": s.image_class}
        if s.grid_inferred:
            meta["array_grid"] = GRID_INFERRED
        uns[s.name] = {"images": {"lowres": img} if img is not None and
                       "tissue_lowres_scalef" in scalefactors else {},
                       "scalefactors": scalefactors, "metadata": meta}
    adata.uns[spatial.KEY] = uns
    adata.uns[spatial.INFO_KEY] = spatial.technology(adata)
    counts = adata.obs[spatial.LIBRARY_COLUMN].value_counts()
    return {"source": SOURCE, **adata.uns[spatial.INFO_KEY],
            "libraries": {s.name: int(counts.get(s.name, 0)) for s in slices},
            "images": {s.name: sorted(uns[s.name]["images"]) for s in slices},
            "image_classes": {s.name: s.image_class for s in slices},
            **({"array_grid": GRID_INFERRED} if any(s.grid_inferred
                                                   for s in slices) else {})}


def _finite(v) -> bool:
    try:
        return bool(np.isfinite(float(np.asarray(v).ravel()[0])))
    except (TypeError, ValueError, IndexError):
        return False


def complete_converted(path: str | Path) -> bool:
    """A converted Seurat Visium .h5ad (``convert.R``) whose slices came without the
    array grid (VisiumV2): the grid recovered (``visium_grid``) and written back, so it
    opens as Visium spots. A file that is not one is never read whole. Returns whether
    it was changed."""
    import anndata as ad
    import h5py

    path = Path(path)
    try:
        with h5py.File(path, "r") as f:
            if "spatial" not in f.get("obsm", {}) or "array_row" in f.get("obs", {}):
                return False
            libs = f.get("uns/spatial")
            if libs is None or not any(
                    _h5_str(libs.get(f"{lib}/metadata/source")) == SOURCE for lib in libs):
                return False
    except OSError:
        return False
    adata = ad.read_h5ad(path)
    lib_col = adata.obs.get(spatial.LIBRARY_COLUMN)
    if lib_col is None:
        return False
    xy = np.asarray(adata.obsm[spatial.KEY], dtype=np.float64)[:, :2]
    rows = np.full(adata.n_obs, -1, dtype=np.int64)
    cols = np.full(adata.n_obs, -1, dtype=np.int64)
    lib_of = lib_col.astype(str).to_numpy()
    for lib in pd.unique(lib_of):
        where = lib_of == lib
        grid = visium_grid(xy[where])
        if grid is None:
            logger.info("No Visium grid recovered for slice %s of %s", lib, path.name)
            return False
        rows[where], cols[where] = grid
    adata.obs["array_row"] = rows
    adata.obs["array_col"] = cols
    for lib in adata.uns.get(spatial.KEY, {}):
        meta = adata.uns[spatial.KEY][lib].setdefault("metadata", {})
        meta["array_grid"] = GRID_INFERRED
    adata.uns[spatial.INFO_KEY] = spatial.technology(adata)
    tmp = path.with_name(f".{path.name}.grid.part")
    adata.write_h5ad(tmp)
    tmp.replace(path)
    return True


def _h5_str(node) -> str:
    if node is None:
        return ""
    try:
        v = node[()]
    except (TypeError, ValueError, OSError):
        return ""
    if isinstance(v, np.ndarray):
        v = v.ravel()[0] if v.size else ""
    return v.decode() if isinstance(v, bytes) else str(v)
