"""What an AnnData object holds (C038): the matrices, the cell and gene tables, the
embeddings and graphs, and uns as a tree, with the obvious problems flagged.

Cheap at any size. Tables (obs, var), obsm/obsp/varm shapes and uns are in memory
whichever way the dataset is open. Matrices are never read whole: in memory their
values are sampled (``SAMPLE_VALUES`` stored values, or ``SAMPLE_ROWS`` rows of a
dense one); on disk (a backed or large dataset) shape, format, dtype and the number of
stored values come from the h5ad's headers, and the values from a few contiguous
slices of the stored data. A large dataset's counts sidecar is summarised the same way.
"""

from __future__ import annotations

import contextlib
import logging
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pandas as pd
import scipy.sparse as sp

logger = logging.getLogger(__name__)

SAMPLE_VALUES = 200_000  # stored values sampled per matrix
SAMPLE_ROWS = 500  # rows sampled from a dense matrix
SLICES = 4  # contiguous slices read from an on-disk matrix's stored values
MAX_COLUMNS = 300  # obs / var columns listed
EXAMPLES = 3
SHORT_STR = 80  # uns strings shown up to this length
UNS_DEPTH = 5
UNS_CHILDREN = 60
SEED = 0


# ---------------------------------------------------------------------------
# Matrices


def _value_stats(vals: np.ndarray, *, has_zeros: bool) -> dict[str, Any]:
    """Range and integer check of sampled values (``has_zeros``: implicit zeros too)."""
    vals = np.asarray(vals).ravel()
    finite = vals[np.isfinite(vals)] if vals.dtype.kind == "f" else vals
    n_nan = int(vals.size - finite.size)
    if finite.size == 0:
        return {"n_sampled": int(vals.size), "min": 0.0 if has_zeros else None,
                "max": 0.0 if has_zeros else None, "integer": has_zeros, "negative": False,
                "n_nan": n_nan}
    lo, hi = float(finite.min()), float(finite.max())
    if has_zeros:
        lo, hi = min(lo, 0.0), max(hi, 0.0)
    integer = bool(finite.dtype.kind in "iub" or np.all(np.mod(finite, 1) == 0))
    return {"n_sampled": int(vals.size), "min": lo, "max": hi, "integer": integer,
            "negative": lo < 0, "n_nan": n_nan}


def _counts_like(stats: dict[str, Any]) -> bool:
    return bool(stats["integer"] and not stats["negative"] and (stats["max"] or 0) > 0)


def _finish(out: dict[str, Any]) -> dict[str, Any]:
    n, m = out["shape"]
    nnz = out.get("nnz")
    out["density"] = (round(nnz / (n * m), 6) if nnz is not None and n * m > 0 else None)
    out["looks_like_counts"] = _counts_like(out["sample"])
    return out


def matrix_memory(M) -> dict[str, Any]:
    """Summary of an in-memory matrix (scipy sparse or dense)."""
    rng = np.random.default_rng(SEED)
    if sp.issparse(M):
        data = M.data
        # With replacement: drawing without it permutes all of a large matrix's values.
        idx = (np.sort(rng.integers(0, data.size, SAMPLE_VALUES)) if data.size > SAMPLE_VALUES
               else slice(None))
        nnz = int(M.nnz)
        stats = _value_stats(data[idx], has_zeros=nnz < M.shape[0] * M.shape[1])
        fmt = M.format
    else:
        arr = np.asarray(M)
        rows = (np.sort(rng.choice(arr.shape[0], SAMPLE_ROWS, replace=False))
                if arr.shape[0] > SAMPLE_ROWS else slice(None))
        sample = arr[rows]
        stats = _value_stats(sample, has_zeros=False)
        nnz = None
        fmt = "dense"
    return _finish({"shape": [int(M.shape[0]), int(M.shape[1])], "dtype": str(M.dtype),
                    "format": fmt, "on_disk": False, "nnz": nnz, "sample": stats})


def _slices(n: int, total: int) -> list[slice]:
    """Up to ``SLICES`` contiguous slices covering ``total`` of ``n`` positions, spread
    evenly (the first at 0)."""
    if n <= total:
        return [slice(0, n)]
    each = max(total // SLICES, 1)
    starts = np.linspace(0, n - each, SLICES).astype(np.int64)
    return [slice(int(s), int(s) + each) for s in starts]


def _encoding(node) -> str:
    enc = node.attrs.get("encoding-type", node.attrs.get("h5sparse_format", ""))
    enc = enc.decode() if isinstance(enc, bytes) else str(enc)
    return {"csr": "csr_matrix", "csc": "csc_matrix"}.get(enc, enc)


def matrix_h5(node) -> dict[str, Any]:
    """Summary of a matrix in an HDF5 file (an h5ad's X / raw/X / layer, or a sidecar),
    from its headers and a few slices of its stored values."""
    if isinstance(node, h5py.Group):
        enc = _encoding(node)
        shape = [int(x) for x in node.attrs["shape"]]
        data = node["data"]
        nnz = int(data.shape[0])
        vals = np.concatenate([data[s] for s in _slices(nnz, SAMPLE_VALUES)]) if nnz else np.zeros(0)
        return _finish({"shape": shape, "dtype": str(data.dtype),
                        "format": enc.replace("_matrix", "") or "sparse", "on_disk": True,
                        "nnz": nnz, "sample": _value_stats(vals, has_zeros=nnz < shape[0] * shape[1])})
    shape = [int(x) for x in node.shape[:2]]
    rows = SAMPLE_ROWS // SLICES if shape[0] > SAMPLE_ROWS else shape[0]
    parts = [node[s] for s in _slices(shape[0], max(rows * SLICES, 1))] if shape[0] else [np.zeros((0, 0))]
    return _finish({"shape": shape, "dtype": str(node.dtype), "format": "dense", "on_disk": True,
                    "nnz": None, "sample": _value_stats(np.concatenate(parts), has_zeros=False)})


# ---------------------------------------------------------------------------
# Tables and uns


def column_summary(name: str, s: pd.Series) -> dict[str, Any]:
    n = int(s.size)
    n_missing = int(s.isna().sum())
    out: dict[str, Any] = {"name": str(name), "dtype": str(s.dtype), "n_missing": n_missing,
                           "all_missing": n > 0 and n_missing == n}
    if isinstance(s.dtype, pd.CategoricalDtype):
        cats = s.cat.categories
        out.update(kind="categorical", n_unique=int(s.nunique(dropna=True)),
                   n_categories=int(len(cats)), examples=[str(c) for c in cats[:EXAMPLES]])
    elif pd.api.types.is_bool_dtype(s.dtype):
        out.update(kind="bool", n_unique=int(s.nunique(dropna=True)),
                   examples=[str(v) for v in s.dropna().unique()[:EXAMPLES]])
    elif pd.api.types.is_numeric_dtype(s.dtype):
        vals = s.to_numpy(dtype=float, na_value=np.nan)
        fin = vals[np.isfinite(vals)]
        out.update(kind="numeric", n_unique=int(s.nunique(dropna=True)),
                   min=float(fin.min()) if fin.size else None,
                   max=float(fin.max()) if fin.size else None,
                   examples=[_short(v) for v in s.dropna().iloc[:EXAMPLES]])
    else:
        vals = s.dropna()
        out.update(kind="text", n_unique=int(vals.nunique()),
                   examples=[_short(v) for v in pd.unique(vals.iloc[:1000])[:EXAMPLES]])
    return out


def table_summary(df: pd.DataFrame) -> dict[str, Any]:
    cols = list(df.columns)
    return {"n_rows": int(df.shape[0]), "n_columns": len(cols),
            "columns": [column_summary(c, df[c]) for c in cols[:MAX_COLUMNS]],
            "truncated": len(cols) > MAX_COLUMNS}


def _short(v: Any) -> Any:
    if isinstance(v, (np.generic,)):
        v = v.item()
    if isinstance(v, float) and not np.isfinite(v):
        return None
    if isinstance(v, (bool, int, float)) or v is None:
        return v
    text = str(v)
    return text if len(text) <= SHORT_STR else text[:SHORT_STR] + "…"


def _shape_of(v: Any) -> dict[str, Any]:
    shape = getattr(v, "shape", None)
    return {"shape": [int(x) for x in shape] if shape is not None else None,
            "dtype": str(getattr(v, "dtype", "")) or None}


def uns_node(key: str, v: Any, depth: int = 0) -> dict[str, Any]:
    """One uns entry: its type and size; values only for short scalars and strings."""
    node: dict[str, Any] = {"key": str(key)}
    if isinstance(v, Mapping):
        keys = sorted(v.keys(), key=str)  # as a file stores them, whichever way it was read
        node.update(type="dict", size=len(keys))
        if depth < UNS_DEPTH:
            node["children"] = [uns_node(k, v[k], depth + 1) for k in keys[:UNS_CHILDREN]]
            node["more"] = max(len(keys) - UNS_CHILDREN, 0)
    elif isinstance(v, pd.DataFrame):
        node.update(type="DataFrame", shape=[int(x) for x in v.shape])
    elif sp.issparse(v):
        node.update(type="sparse", **_shape_of(v))
    elif isinstance(v, np.ndarray):
        if v.ndim == 0:
            node.update(type=str(v.dtype), value=_short(v.item()))
        else:
            node.update(type="array", **_shape_of(v))
            if v.dtype.names:
                node["fields"] = list(v.dtype.names)[:UNS_CHILDREN]
    elif isinstance(v, (str, bytes)):
        text = v.decode(errors="replace") if isinstance(v, bytes) else v
        node.update(type="str", size=len(text))
        if len(text) <= SHORT_STR:
            node["value"] = text
    elif isinstance(v, (bool, int, float, np.generic)) or v is None:
        node.update(type=type(v.item() if isinstance(v, np.generic) else v).__name__, value=_short(v))
    elif isinstance(v, (list, tuple)):
        node.update(type="list", size=len(v))
        if len(v) <= 5 and all(isinstance(x, (str, bool, int, float)) for x in v):
            shown = [_short(x) for x in v]
            if sum(len(str(x)) for x in shown) <= SHORT_STR:
                node["value"] = shown
    else:
        node.update(type=type(v).__name__, **_shape_of(v))
    return node


# ---------------------------------------------------------------------------
# The whole object


def _lock(adaptor):
    """The lock that orders reads of a large dataset's file against in-place writes."""
    return getattr(getattr(adaptor, "_large", None), "lock", None) or contextlib.nullcontext()


def _matrices_on_disk(path: str, layers: list[str], has_raw: bool) -> dict[str, Any]:
    out: dict[str, Any] = {"X": None, "raw": None, "layers": {}}
    with h5py.File(path, "r") as f:
        if "X" in f:
            out["X"] = matrix_h5(f["X"])
        if has_raw and "raw" in f and "X" in f["raw"]:
            out["raw"] = matrix_h5(f["raw"]["X"])
        for name in layers:
            if "layers" in f and name in f["layers"]:
                out["layers"][name] = matrix_h5(f["layers"][name])
    return out


def summarise(adaptor) -> dict[str, Any]:
    """The structure of the dataset ``adaptor`` serves, with its issues (``flags``)."""
    adata = adaptor.adata
    backed = bool(getattr(adata, "isbacked", False))
    large = bool(getattr(adaptor, "is_large", False))
    layer_names = [str(k) for k in adata.layers.keys()]
    raw = adata.raw
    if backed:
        with _lock(adaptor):
            mats = _matrices_on_disk(adaptor.h5ad_path, layer_names, raw is not None)
    else:
        mats = {"X": matrix_memory(adata.X) if adata.X is not None else None,
                "raw": matrix_memory(raw.X) if raw is not None else None,
                "layers": {k: matrix_memory(adata.layers[k]) for k in layer_names}}
    if mats["raw"] is not None:
        mats["raw"].update(n_vars=int(raw.n_vars),
                           var_columns=[str(c) for c in raw.var.columns][:MAX_COLUMNS])
    sidecars: dict[str, Any] = {}
    if large:
        from scview.core import large_layout

        counts_path, _ = large_layout.sidecar_paths(adaptor.h5ad_path)
        try:
            with _lock(adaptor), h5py.File(counts_path, "r") as f:
                if large_layout.COUNTS_KEY in f:
                    sidecars["counts"] = matrix_h5(f[large_layout.COUNTS_KEY])
        except OSError as exc:
            logger.info("structure: no counts sidecar for %s: %s", adaptor.h5ad_path, exc)

    def keyed(m) -> list[dict[str, Any]]:
        return [{"key": str(k), **_shape_of(m[k]),
                 "kind": "sparse" if sp.issparse(m[k]) else
                 ("DataFrame" if isinstance(m[k], pd.DataFrame) else "array")}
                for k in m.keys()]

    out: dict[str, Any] = {
        "n_obs": int(adata.n_obs), "n_vars": int(adata.n_vars),
        "mode": "large" if large else ("backed" if backed else "memory"),
        "file_bytes": _size(adaptor.h5ad_path),
        "X": mats["X"], "raw": mats["raw"], "layers": mats["layers"], "sidecars": sidecars,
        "obs": table_summary(adata.obs), "var": table_summary(adata.var),
        "obs_names": _names(adata.obs_names), "var_names": _names(adata.var_names),
        "obsm": keyed(adata.obsm), "obsp": keyed(adata.obsp), "varm": keyed(adata.varm),
        "varp": keyed(adata.varp),
        "uns": [uns_node(k, adata.uns[k]) for k in sorted(adata.uns.keys(), key=str)],
        "sampling": {"values": SAMPLE_VALUES, "rows": SAMPLE_ROWS, "seed": SEED},
    }
    out["flags"] = flags(out)
    return out


def _size(path: str) -> int | None:
    try:
        return Path(path).stat().st_size
    except OSError:
        return None


def _names(index: pd.Index) -> dict[str, Any]:
    dup = index[index.duplicated()]
    return {"n": int(index.size), "n_duplicated": int(dup.size),
            "duplicated_examples": [str(x) for x in pd.unique(dup)[:5]],
            "examples": [str(x) for x in index[:EXAMPLES]]}


def flags(s: dict[str, Any]) -> list[dict[str, str]]:
    """Problems worth a look: no counts anywhere, duplicate names, empty columns."""
    out: list[dict[str, str]] = []
    X = s["X"]
    counts_where = [name for name, m in (("X", X), (".raw", s["raw"]),
                                         *((f"layers['{k}']", v) for k, v in s["layers"].items()),
                                         *((f"the {k} sidecar", v) for k, v in s["sidecars"].items()))
                    if m is not None and m["looks_like_counts"]]
    if X is not None and not X["looks_like_counts"] and not counts_where:
        what = ("negative values (scaled)" if X["sample"]["negative"]
                else "non-integer values (normalised)")
        out.append({"level": "warn", "where": "X",
                    "text": f"X holds {what} and there is no .raw or layer with counts: QC by "
                            "counts, doublet detection, pseudobulk DE and Pearson residuals need "
                            "raw counts."})
    if X is not None and X["sample"]["negative"] and s["raw"] is None:
        out.append({"level": "info", "where": "X",
                    "text": "X holds negative values (scaled) and there is no .raw: expression "
                            "plots will show scaled values."})
    for m_name, m in (("X", X), (".raw", s["raw"])):
        if m is not None and m["sample"]["n_nan"]:
            out.append({"level": "warn", "where": m_name,
                        "text": f"{m_name} has NaN values in the sample ({m['sample']['n_nan']:,})."})
    for axis, label in (("var_names", "gene"), ("obs_names", "cell")):
        d = s[axis]
        if d["n_duplicated"]:
            out.append({"level": "warn", "where": axis,
                        "text": f"{d['n_duplicated']:,} duplicate {label} names (e.g. "
                                f"{', '.join(d['duplicated_examples'][:3])}): a {label} looked up by "
                                "name gets the first."})
    for table in ("obs", "var"):
        empty = [c["name"] for c in s[table]["columns"] if c["all_missing"]]
        if empty:
            out.append({"level": "info", "where": table,
                        "text": f"{len(empty)} {table} column(s) are all missing: "
                                + ", ".join(empty[:5]) + ("…" if len(empty) > 5 else "") + "."})
    return out
