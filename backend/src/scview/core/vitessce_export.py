"""A Vitessce bundle: the dataset as AnnData-Zarr plus a view config, in one .zip.

The zip holds ``<name>.zarr/`` (zarr format v2 with consolidated metadata, the layout
Vitessce's ``anndata.zarr`` loader reads), ``config.json`` (a Vitessce view config,
schema ``CONFIG_VERSION``: a scatterplot per embedding, cell sets from the chosen label
columns, their sizes, a gene list and a heatmap) and ``README.txt`` on opening it.

Vitessce reads one gene at a time from X by fetching the zarr chunks that hold it, so X
is written dense (float32) in chunks of ``rows x GENE_CHUNK`` genes, row band by row
band from the dataset's sparse X (in memory, or from its file for a large or backed
dataset): never the whole dense matrix in memory. The heatmap starts on at most
``HEATMAP_GENES`` genes (``var['heatmap_genes']``: the most variable of the variable
genes, or of all genes when none are marked), since a heatmap of every gene is more
than a browser draws.

The config is written by hand (no ``vitessce`` runtime dependency); its shape matches
what the ``vitessce`` Python package 3.8.4 builds (``VitessceConfig(schema_version=
"1.0.17")`` with an ``AnnDataWrapper``), checked against ``@vitessce/schemas``.
"""

from __future__ import annotations

import json
import os
import shutil
import zipfile
from pathlib import Path
from typing import Any, Callable

import h5py
import numpy as np
import pandas as pd
import scipy.sparse as sp

from scview.core import cellxgene_export as cx

CONFIG_VERSION = "1.0.17"
GENE_CHUNK = 16
BAND_BYTES = 64 << 20  # one dense row band in memory while writing
HEATMAP_GENES = 500
MAX_VALUES = 2_000_000_000  # dense values (8 GB as float32): more is not for a browser
MAX_SET_CATEGORIES = 500
DEFAULT_PORT = 8000
# Bytes per stored non-zero and per zero after blosc-zstd of dense float32 chunks,
# measured on Kang (13,836 x 14,053, 7.5% non-zero): see estimate_bytes.
_BYTES_PER_NONZERO = 6.0
_BYTES_PER_ZERO = 0.11


def label_columns(adata) -> list[dict[str, Any]]:
    """obs columns that can be cell sets: categoricals / text / true-false with 2 to
    ``MAX_SET_CATEGORIES`` values; ``default`` for those with at most 100."""
    out: list[dict[str, Any]] = []
    for col in adata.obs.columns:
        s = adata.obs[col]
        if pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s):
            continue
        n = int(s.nunique())
        if 2 <= n <= MAX_SET_CATEGORIES:
            out.append({"name": str(col), "n_values": n, "default": n <= 100})
    return out


def _rows_per_band(n_genes: int) -> int:
    rows = BAND_BYTES // max(1, 4 * n_genes)
    return int(max(256, min(8192, rows // 256 * 256 or 256)))


def _gene_mask(var: pd.DataFrame, genes: str) -> np.ndarray | None:
    if genes == "all":
        return None
    if genes != "variable":
        raise cx.ExportError("genes must be 'all' or 'variable'.")
    if "highly_variable" not in var.columns:
        raise cx.ExportError("This dataset has no variable genes marked; export all genes.")
    mask = np.asarray(var["highly_variable"], dtype=bool)
    if not mask.any():
        raise cx.ExportError("No genes are marked variable; export all genes.")
    return mask


def _nnz(m: cx.Matrix) -> int:
    if m.value is not None:
        v = m.value
        return int(v.nnz) if sp.issparse(v) else int(np.count_nonzero(v[: min(2000, v.shape[0])])
                                                      * v.shape[0] / max(1, min(2000, v.shape[0])))
    with h5py.File(m.path, "r") as f:
        node = f[m.key]
        if isinstance(node, h5py.Group):
            return int(node["data"].shape[0])
        return int(np.prod(node.shape))


def plan(adaptor, opts: cx.Options, genes: str = "all") -> dict[str, Any]:
    """What the bundle holds: obs, var, embeddings, the expression matrix and gene mask."""
    adata = adaptor.adata
    picked, obsm_dropped = cx._pick_embeddings(adata, opts.embeddings)
    if not picked:
        raise cx.ExportError("Choose at least one embedding (a 2D layout) for the scatterplots.")
    if opts.columns is None:
        cols = [c["name"] for c in label_columns(adata) if c["default"]][:6]
    else:
        cols = opts.columns
    obs, notes, _left, n_cells = cx.clean_obs(adata.obs, cols)
    for name in obs.columns:
        if not isinstance(obs[name].dtype, pd.CategoricalDtype):
            raise cx.ExportError(f"'{name}' holds numbers, not labels; it cannot be a cell set.")
        if len(obs[name].cat.categories) > MAX_SET_CATEGORIES:
            raise cx.ExportError(f"'{name}' has more than {MAX_SET_CATEGORIES} values.")
    x, _counts, mnotes = cx.matrices(adaptor)
    notes += mnotes
    var_src = adata.raw.var if x.where == ".raw" and adata.raw is not None else adata.var
    mask = _gene_mask(var_src, genes)
    var, vnotes, n_genes = cx.symbol_var(var_src)
    notes += vnotes
    keep = [c for c in ("gene_ids", "highly_variable") if c in var.columns]
    var = var[keep].copy()
    if "highly_variable" in var.columns:
        var["highly_variable"] = np.asarray(var["highly_variable"], dtype=bool)
    if mask is not None:
        var = var.iloc[np.flatnonzero(mask)]
    n_out = int(var.shape[0])
    if x.shape[0] * n_out > MAX_VALUES:
        raise cx.ExportError(
            f"{x.shape[0]:,} cells x {n_out:,} genes is more than a browser can load "
            f"({MAX_VALUES:,} values at most); export the variable genes only.")
    obsm = {c["export_name"]: np.ascontiguousarray(np.asarray(adata.obsm[c["key"]]), dtype=np.float32)
            for c in picked}
    if n_cells:
        notes.append(f"{n_cells:,} duplicate cell names made unique.")
    if n_genes:
        notes.append(f"{n_genes:,} duplicate gene names made unique.")
    return {"obs": obs, "var": var, "obsm": obsm, "picked": picked, "x": x, "mask": mask,
            "genes": genes, "notes": notes, "dropped": {"obsm": obsm_dropped}}


def estimate_bytes(p: dict[str, Any]) -> int:
    """About how big the zip is: non-zeros and zeros of the dense X after compression
    (rates measured on Kang), plus the embeddings and labels."""
    x: cx.Matrix = p["x"]
    n_genes = int(p["var"].shape[0])
    nnz = _nnz(x)
    if p["mask"] is not None:
        nnz = int(nnz * n_genes / max(1, x.shape[1]))
    values = x.shape[0] * n_genes
    total = nnz * _BYTES_PER_NONZERO + max(0, values - nnz) * _BYTES_PER_ZERO
    total += sum(a.nbytes for a in p["obsm"].values())
    total += int(p["obs"].memory_usage(deep=True).sum() // 4) + n_genes * 24
    return int(total) + 64 * 1024


def _band(x: cx.Matrix, start: int, stop: int) -> np.ndarray:
    """Rows ``start:stop`` of X as a dense float32 array."""
    if x.value is not None:
        block = x.value[start:stop]
    else:
        with h5py.File(x.path, "r") as f:
            node = f[x.key]
            if isinstance(node, h5py.Group):
                ptr = node["indptr"][start:stop + 1]
                lo, hi = int(ptr[0]), int(ptr[-1])
                block = sp.csr_matrix(
                    (node["data"][lo:hi], node["indices"][lo:hi], ptr - lo),
                    shape=(stop - start, x.shape[1]))
            else:
                block = node[start:stop]
    if sp.issparse(block):
        return block.toarray().astype(np.float32, copy=False)
    return np.asarray(block, dtype=np.float32)


def config(name: str, url: str, p: dict[str, Any]) -> dict[str, Any]:
    """The Vitessce view config (schema ``CONFIG_VERSION``)."""
    embeddings = [{"path": f"obsm/{c['export_name']}", "dims": [0, 1], "embeddingType": c["label"]}
                  for c in p["picked"]]
    sets = [{"name": str(col), "path": f"obs/{col}"} for col in p["obs"].columns]
    options: dict[str, Any] = {
        "obsEmbedding": embeddings,
        "obsFeatureMatrix": {"path": "X", "initialFeatureFilterPath": "var/heatmap_genes"},
    }
    if sets:
        options["obsSets"] = sets
    letters = [chr(ord("A") + i) for i in range(len(embeddings))]
    layout: list[dict[str, Any]] = []
    w = 12 // max(1, min(len(embeddings), 3))
    for i, letter in enumerate(letters):
        layout.append({"component": "scatterplot",
                       "coordinationScopes": {"dataset": "A", "embeddingType": letter},
                       "x": (i % 3) * w, "y": (i // 3) * 4, "w": w, "h": 4})
    y = ((len(letters) + 2) // 3) * 4
    for component, x0, width in (("obsSets", 0, 4), ("obsSetSizes", 4, 4), ("featureList", 8, 4)):
        if component != "featureList" and not sets:
            continue
        layout.append({"component": component, "coordinationScopes": {"dataset": "A"},
                       "x": x0, "y": y, "w": width, "h": 4})
    layout.append({"component": "heatmap", "coordinationScopes": {"dataset": "A"},
                   "x": 0, "y": y + 4, "w": 12, "h": 5})
    return {
        "version": CONFIG_VERSION,
        "name": name,
        "description": "Exported from scView",
        "datasets": [{"uid": "A", "name": name, "files": [{
            "fileType": "anndata.zarr",
            "url": url,
            "options": options,
            "coordinationValues": {"obsType": "cell", "featureType": "gene",
                                   "featureValueType": "expression"},
        }]}],
        "coordinationSpace": {"dataset": {"A": "A"},
                              "embeddingType": {letter: e["embeddingType"]
                                                for letter, e in zip(letters, embeddings)}},
        "layout": layout,
        "initStrategy": "auto",
    }


def readme(name: str, zarr_name: str, port: int = DEFAULT_PORT) -> str:
    return f"""{name}: Vitessce bundle exported from scView
{'=' * (len(name) + 37)}

Contents
  {zarr_name}/   the dataset as AnnData-Zarr (zarr format v2, consolidated metadata)
  config.json    a Vitessce view config: a scatterplot per embedding, cell sets,
                 cell set sizes, a gene list and a heatmap
  README.txt     this file

config.json points at http://127.0.0.1:{port}/{zarr_name}. To open the bundle:

1. In a browser (vitessce.io)
   Unzip, then serve this folder on your own computer with CORS allowed, e.g.

     npx http-server --cors -a 127.0.0.1 -p {port}

   and open
     http://vitessce.io/#?url=http://127.0.0.1:{port}/config.json
   (The data stays on your computer; the page only reads it from 127.0.0.1.
   Chrome may ask to allow the page to reach local addresses.)

2. In Python / Jupyter (the vitessce package, pip install "vitessce[all]")

     import json
     from vitessce import VitessceConfig
     with open("config.json") as f:
         vc = VitessceConfig.from_dict(json.load(f))
     vc.widget()     # with the folder served as in 1.

   or build your own view from the data:  AnnDataWrapper(adata_path="{zarr_name}", ...)

The expression matrix is the log-normalised X as scView showed it. The heatmap
starts on var['heatmap_genes'] (at most {HEATMAP_GENES} genes); search the gene
list for any other gene. Provenance (scView version, dataset, date) is in
{zarr_name}/uns/scview_export.
"""


def write(p: dict[str, Any], out_path: Path, name: str, stem: str, note: dict[str, str],
          stop: Callable[[], bool] = lambda: False,
          progress: Callable[[str], None] = lambda _m: None) -> Path:
    """Write the bundle's zip at ``out_path`` (a working folder beside it, removed)."""
    import anndata as ad
    import numcodecs
    import zarr

    out_path = Path(out_path)
    work = out_path.with_name(f".{out_path.name}.dir")
    tmp = out_path.with_name(f".{out_path.name}.part")
    zarr_name = f"{stem}.zarr"
    store = work / zarr_name
    shutil.rmtree(work, ignore_errors=True)
    work.mkdir(parents=True)
    x: cx.Matrix = p["x"]
    var: pd.DataFrame = p["var"].copy()
    mask = p["mask"]
    cols = np.flatnonzero(mask) if mask is not None else None
    n_obs, n_genes = x.shape[0], int(var.shape[0])
    try:
        progress("Writing cell and gene tables")
        meta = ad.AnnData(obs=p["obs"], var=var, obsm=p["obsm"], uns={"scview_export": note})
        with ad.settings.override(zarr_write_format=2):
            meta.write_zarr(store)
        if stop():
            raise cx.Cancelled()
        g = zarr.open_group(store, mode="a", zarr_format=2, use_consolidated=False)
        if "X" in g:
            del g["X"]
        rows = _rows_per_band(n_genes)
        arr = g.create_array("X", shape=(n_obs, n_genes), chunks=(min(rows, n_obs), GENE_CHUNK),
                             dtype="float32", fill_value=0.0,
                             compressors=numcodecs.Blosc(cname="zstd", clevel=5,
                                                         shuffle=numcodecs.Blosc.SHUFFLE))
        arr.attrs.update({"encoding-type": "array", "encoding-version": "0.2.0"})
        total = np.zeros(n_genes)
        total_sq = np.zeros(n_genes)
        for start in range(0, n_obs, rows):
            if stop():
                raise cx.Cancelled()
            stop_row = min(n_obs, start + rows)
            progress(f"Writing expression: cells {start + 1:,}-{stop_row:,} of {n_obs:,}")
            band = _band(x, start, stop_row)
            if cols is not None:
                band = band[:, cols]
            arr[start:stop_row, :] = band
            total += band.sum(axis=0, dtype=np.float64)
            total_sq += np.square(band, dtype=np.float64).sum(axis=0)
        mean = total / max(1, n_obs)
        variance = total_sq / max(1, n_obs) - mean ** 2
        pool = np.asarray(var["highly_variable"], dtype=bool) if "highly_variable" in var.columns else None
        if pool is None or not pool.any():
            pool = np.ones(n_genes, dtype=bool)
        order = np.argsort(-np.where(pool, variance, -np.inf), kind="stable")
        heat = np.zeros(n_genes, dtype=bool)
        heat[order[: min(HEATMAP_GENES, int(pool.sum()))]] = True
        from anndata.io import write_elem

        var["heatmap_genes"] = heat
        del g["var"]
        write_elem(g, "var", var)
        zarr.consolidate_metadata(store, zarr_format=2)
        (work / "config.json").write_text(json.dumps(
            config(name, f"http://127.0.0.1:{DEFAULT_PORT}/{zarr_name}", p), indent=2))
        (work / "README.txt").write_text(readme(name, zarr_name))
        if stop():
            raise cx.Cancelled()
        progress("Packing the zip")
        with zipfile.ZipFile(tmp, "w", compression=zipfile.ZIP_STORED, allowZip64=True) as zf:
            for path in sorted(work.rglob("*")):
                if path.is_file():
                    zf.write(path, path.relative_to(work).as_posix())
        os.replace(tmp, out_path)
    finally:
        shutil.rmtree(work, ignore_errors=True)
        tmp.unlink(missing_ok=True)
    return out_path


def config_paths(cfg: dict[str, Any]) -> list[str]:
    """Every store path a config names (for the check that each exists)."""
    out: list[str] = []
    for ds in cfg["datasets"]:
        for f in ds["files"]:
            opts = f.get("options", {})
            out += [e["path"] for e in opts.get("obsEmbedding", [])]
            out += [s["path"] for s in opts.get("obsSets", [])]
            fm = opts.get("obsFeatureMatrix")
            if fm:
                out.append(fm["path"])
                if fm.get("initialFeatureFilterPath"):
                    out.append(fm["initialFeatureFilterPath"])
    return out
