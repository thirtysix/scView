"""A CELLxGENE-compatible copy of a dataset, and how far it is from a Discover submission.

``build`` writes an h5ad that ``cellxgene launch`` (the desktop explorer, CELLxGENE
Annotate) opens as it is:

- embeddings as obsm ``X_<name>`` with 2 (or 3) float32 columns (a wider one, PCA, keeps
  its first two), the names the Discover schema allows (``^X_[a-zA-Z][a-zA-Z0-9_.-]*$``);
- unique cell and gene names (made unique, and the response says how many changed);
- obs text and true/false columns as categoricals, the colours scView keeps for them as
  ``uns['<column>_colors']``;
- var index = gene symbols; Ensembl ids, when the index held them, kept in
  ``var['gene_ids']``;
- X the log-normalised matrix (``.raw`` when X holds an older scView's scaled values);
  ``.raw`` the raw counts, or nothing, by choice;
- scView's own results (uns, layers, graphs, loadings, non-embedding obsm) left out, and
  listed (``Plan.dropped``).

``readiness`` reports which fields a CELLxGENE Discover submission needs are present,
missing or not in the expected form, read from the published schema (``SCHEMA_VERSION``).
It reports only: no ontology term is ever filled in or guessed.

Matrices are never read whole into memory for this: a dataset in memory has its X and
counts written as they are (``anndata.io.write_elem``); one on disk (large, or opened
backed) has them copied from its files chunk by chunk (``large_mode._copy_matrix``).
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import h5py
import numpy as np
import pandas as pd
import scipy.sparse as sp

from scview.core import large_layout, large_mode

# CELLxGENE Discover dataset schema the readiness report follows:
# https://github.com/chanzuckerberg/single-cell-curation/blob/main/schema/5.3.0/schema.md
# (read 2026-10-25). Against 5.3.0: organism_ontology_term_id lives in uns (6.0.0;
# obs organism columns are reserved), several terms are joined with " || " (disease,
# self-reported ethnicity, 6.0.0), tissue_type "cell culture" became "primary cell
# culture" and "cell line" was added (Cellosaurus CVCL_ tissue; "na" donor, sex, stage
# and ethnicity; 7.0.0), and optional perturbation / experimental-condition fields
# arrived (7.1.0).
SCHEMA_VERSION = "7.1.0"
SCHEMA_URL = ("https://github.com/chanzuckerberg/single-cell-curation/blob/main/schema/"
              f"{SCHEMA_VERSION}/schema.md")

EMBEDDING_NAME = re.compile(r"^X_[a-zA-Z][a-zA-Z0-9_.-]*$")
ENSEMBL = re.compile(r"^ENS[A-Z]*G\d{6,}(\.\d+)?$")
# var columns that may hold gene symbols when the index holds Ensembl ids.
SYMBOL_COLUMNS = ("feature_name", "gene_symbols", "gene_symbol", "symbol", "symbols",
                  "gene_name", "gene_names", "name")
# var columns that may hold Ensembl ids next to symbol names (as species.py reads them).
ID_COLUMNS = ("gene_ids", "gene_id", "ensembl_id", "ensembl", "gene_ensembl", "feature_id", "id")
# obsm entries that are per-cell values, not layouts.
NOT_EMBEDDINGS = {"X_nmf_usage"}


class ExportError(ValueError):
    """A bad option (unknown column or embedding): a 400 with this message."""


class Cancelled(Exception):
    """The client went away; the partial file is removed."""


@dataclass
class Options:
    embeddings: list[str] | None = None  # obsm keys; None: every 2D/3D layout
    columns: list[str] | None = None  # obs columns; None: all
    include_raw: bool = False  # raw counts as .raw


@dataclass
class Matrix:
    """One matrix of the export: in memory (``value``), or in a file (``path`` +
    ``key``) to copy from. ``where`` names it for the user."""

    where: str
    shape: tuple[int, int]
    value: Any = None
    path: Path | None = None
    key: str | None = None
    is_counts: bool = False


@dataclass
class Plan:
    obs: pd.DataFrame
    var: pd.DataFrame
    obsm: dict[str, np.ndarray]
    uns: dict[str, Any]
    x: Matrix
    raw: Matrix | None
    counts: Matrix | None  # what .raw would hold, whether or not it was chosen
    dropped: dict[str, list[str]] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)
    renamed: dict[str, int] = field(default_factory=dict)  # names made unique


# ---------------------------------------------------------------------------
# Embeddings


def _label(key: str) -> str:
    suffix = key[2:] if key.startswith("X_") else key
    known = {"umap": "UMAP", "tsne": "t-SNE", "pca": "PCA", "umap_3d": "UMAP 3D",
             "draw_graph_fa": "ForceAtlas2", "phate": "PHATE", "diffmap": "Diffusion map"}
    return known.get(suffix.lower(), suffix)


def embedding_choices(adata, max_dims: int = 3) -> list[dict[str, Any]]:
    """The obsm layouts an export can carry: name, columns, whether chosen by default
    (2 to ``max_dims`` columns), and the name it gets in the export."""
    out: list[dict[str, Any]] = []
    taken: set[str] = set()
    for key in adata.obsm.keys():
        if key in NOT_EMBEDDINGS or not str(key).startswith("X_"):
            continue
        val = adata.obsm[key]
        if isinstance(val, pd.DataFrame) or sp.issparse(val):
            continue
        shape = getattr(val, "shape", ())
        dtype = getattr(val, "dtype", None)
        if len(shape) != 2 or shape[1] < 2 or dtype is None or dtype.kind not in "fiu":
            continue
        name = key if EMBEDDING_NAME.match(key) else "X_" + (
            re.sub(r"[^a-zA-Z0-9_.-]+", "_", key[2:]).lstrip("_.-0123456789") or "layout")
        base, n = name, 2
        while name in taken:
            name, n = f"{base}_{n}", n + 1
        taken.add(name)
        out.append({"key": key, "label": _label(key), "n_dims": int(shape[1]),
                    "default": int(shape[1]) <= max_dims, "export_name": name})
    return out


def _pick_embeddings(adata, chosen: list[str] | None, max_dims: int = 3) -> tuple[list[dict], list[str]]:
    choices = embedding_choices(adata, max_dims)
    by_key = {c["key"]: c for c in choices}
    if chosen is None:
        picked = [c for c in choices if c["default"]]
    else:
        missing = [k for k in chosen if k not in by_key]
        if missing:
            raise ExportError(f"Not an embedding of this dataset: {', '.join(missing)}.")
        picked = [by_key[k] for k in dict.fromkeys(chosen)]
    keys = {c["key"] for c in picked}
    dropped = [str(k) for k in adata.obsm.keys() if k not in keys]
    return picked, dropped


def embedding_array(adata, choice: dict) -> np.ndarray:
    arr = np.asarray(adata.obsm[choice["key"]])
    if arr.shape[1] > 3:
        arr = arr[:, :2]
    return np.ascontiguousarray(arr, dtype=np.float32)


# ---------------------------------------------------------------------------
# obs and var


def _unique(index: pd.Index) -> tuple[pd.Index, int]:
    """``index`` as unique strings (anndata's ``make_unique`` rule: '-1', '-2' added);
    how many names changed."""
    from anndata.utils import make_index_unique

    idx = pd.Index(index.astype(str))
    if idx.is_unique:
        return idx, 0
    new = make_index_unique(idx)
    return new, int((new != idx).sum())


def clean_obs(obs: pd.DataFrame, columns: list[str] | None) -> tuple[pd.DataFrame, list[str], list[str], int]:
    """(obs to export, notes, columns left out, cell names made unique). Text and
    true/false columns become categoricals with text categories; a '/' in a column name
    (a path separator in h5ad and zarr) becomes '_'."""
    if columns is None:
        keep = list(obs.columns)
    else:
        missing = [c for c in columns if c not in obs.columns]
        if missing:
            raise ExportError(f"Not a cell metadata column: {', '.join(missing)}.")
        keep = list(dict.fromkeys(columns))
    left_out = [str(c) for c in obs.columns if c not in keep]
    notes: list[str] = []
    out = pd.DataFrame(index=obs.index)
    for col in keep:
        s = obs[col]
        name = str(col).replace("/", "_")
        if name != str(col):
            notes.append(f"Column '{col}' renamed '{name}' ('/' is not allowed in a column name).")
        if isinstance(s.dtype, pd.CategoricalDtype):
            cats = s.cat.categories
            if not all(isinstance(c, str) for c in cats):
                s = s.cat.rename_categories([str(c) for c in cats])
        elif pd.api.types.is_bool_dtype(s):
            s = pd.Categorical(s.map({True: "True", False: "False"}), categories=["False", "True"])
        elif pd.api.types.is_numeric_dtype(s):
            pass
        else:
            s = pd.Categorical(s.where(s.isna(), s.astype(str)))
        out[name] = s
    out.index, n_renamed = _unique(obs.index)
    out.index.name = None
    return out, notes, left_out, n_renamed


def symbol_var(var: pd.DataFrame) -> tuple[pd.DataFrame, list[str], int]:
    """(var with gene symbols as its index, notes, gene names made unique). When the
    index holds Ensembl ids and a column holds symbols, the symbols become the index and
    the ids go to ``gene_ids``."""
    notes: list[str] = []
    out = var.copy()
    idx = pd.Index(out.index.astype(str))
    sample = idx[:: max(1, len(idx) // 2000)]
    if len(sample) and np.mean([bool(ENSEMBL.match(v)) for v in sample]) > 0.5:
        for col in SYMBOL_COLUMNS:
            if col not in out.columns:
                continue
            vals = out[col].astype(str)
            if vals.isin(["", "nan", "None"]).mean() > 0.5:
                continue
            if np.mean([bool(ENSEMBL.match(v)) for v in vals.iloc[:: max(1, len(vals) // 2000)]]) > 0.5:
                continue
            ids = idx
            sym = vals.where(~vals.isin(["", "nan", "None"]), ids.to_series(index=out.index))
            if col != "gene_ids":
                out = out.drop(columns=[col])
            if "gene_ids" not in out.columns:
                out.insert(0, "gene_ids", np.asarray(ids))
            out.index = pd.Index(np.asarray(sym))
            notes.append(f"Gene names: symbols from var['{col}']; Ensembl ids kept in var['gene_ids'].")
            break
        else:
            notes.append("Gene names are Ensembl ids (no symbol column found); kept as they are.")
    out.index, n_renamed = _unique(out.index)
    out.index.name = None
    for col in list(out.columns):
        s = out[col]
        if not (isinstance(s.dtype, pd.CategoricalDtype) or pd.api.types.is_numeric_dtype(s)
                or pd.api.types.is_bool_dtype(s)):
            out[col] = s.astype(str).where(s.notna(), None)
        if "/" in str(col):
            out = out.rename(columns={col: str(col).replace("/", "_")})
    return out, notes, n_renamed


def _colors(uns: dict, obs: pd.DataFrame, original: dict[str, str]) -> dict[str, list[str]]:
    """``<column>_colors`` for each exported categorical with one colour per category."""
    out: dict[str, list[str]] = {}
    for name, src in original.items():
        if not isinstance(obs[name].dtype, pd.CategoricalDtype):
            continue
        cols = uns.get(f"{src}_colors")
        if cols is None:
            continue
        cols = [str(c) for c in np.asarray(cols).ravel()]
        if len(cols) == len(obs[name].cat.categories):
            out[f"{name}_colors"] = cols
    return out


# ---------------------------------------------------------------------------
# Matrices


def _h5_counts(path: Path, key: str) -> bool:
    """Whether a stored matrix's first values are non-negative integers (raw counts)."""
    try:
        with h5py.File(path, "r") as f:
            node = f[key]
            data = node["data"] if isinstance(node, h5py.Group) else node
            if data.ndim == 2:
                vals = np.asarray(data[: min(200, data.shape[0])]).ravel()
            else:
                vals = np.asarray(data[: min(100_000, data.shape[0])])
    except (OSError, KeyError):
        return False
    vals = vals[vals != 0].astype(np.float64)
    return bool(vals.size) and bool(np.all(vals >= 0) and np.all(np.mod(vals, 1) == 0))


def _h5_shape(path: Path, key: str) -> tuple[int, int] | None:
    try:
        with h5py.File(path, "r") as f:
            if key not in f:
                return None
            node = f[key]
            if isinstance(node, h5py.Group):
                shape = node.attrs.get("shape")
                return (int(shape[0]), int(shape[1])) if shape is not None else None
            return (int(node.shape[0]), int(node.shape[1])) if node.ndim == 2 else None
    except OSError:
        return None


def storage_bytes(path: Path | None, key: str | None) -> int | None:
    """Bytes a stored element takes in its file (compressed chunks), None when absent."""
    if path is None or key is None:
        return None
    try:
        with h5py.File(path, "r") as f:
            if key not in f:
                return None
            node = f[key]
            if isinstance(node, h5py.Dataset):
                return int(node.id.get_storage_size())
            total = 0
            stack = [node]
            while stack:
                g = stack.pop()
                for v in g.values():
                    if isinstance(v, h5py.Dataset):
                        total += int(v.id.get_storage_size())
                    elif isinstance(v, h5py.Group):
                        stack.append(v)
            return total
    except OSError:
        return None


def matrices(adaptor) -> tuple[Matrix, Matrix | None, list[str]]:
    """(the expression matrix to export as X, the raw counts or None, notes).

    In memory: X (its ``.raw`` when X holds an older scView's scaled values), counts from
    ``layers['counts']``, a counts ``.raw``, or X itself. On disk: X from the h5ad, counts
    from a large dataset's sidecar or the file's ``layers/counts``."""
    from scview.core.pipeline import looks_like_counts
    from scview.core.scaling import holds_scaled

    adata = adaptor.adata
    h5ad = Path(adaptor.h5ad_path)
    n, g = int(adata.n_obs), int(adata.n_vars)
    notes: list[str] = []
    if getattr(adaptor, "is_large", False) or adata.isbacked:
        shape = _h5_shape(h5ad, "X")
        if shape != (n, g):
            raise large_mode.LargeDatasetError(
                "This export", "the expression matrix in its file does not match the cells "
                "and genes shown; process or reload the dataset first")
        x = Matrix("X", (n, g), path=h5ad, key="X", is_counts=_h5_counts(h5ad, "X"))
        counts: Matrix | None = None
        if getattr(adaptor, "is_large", False):
            try:
                adaptor.counts_matrix("Exporting the counts")
                cpath = large_layout.counts_path(h5ad, adata.uns)
                counts = Matrix("the counts file", (n, g), path=cpath,
                                key=large_layout.COUNTS_KEY, is_counts=True)
            except large_mode.LargeDatasetError as exc:
                notes.append(f"No raw counts: {exc.why}.")
        elif _h5_shape(h5ad, "layers/counts") == (n, g):
            counts = Matrix("layers['counts']", (n, g), path=h5ad, key="layers/counts",
                            is_counts=True)
        if counts is None and x.is_counts:
            counts = x
        return x, counts, notes

    X = adata.X
    x = Matrix("X", (n, g), value=X, path=h5ad, key="X")
    raw = adata.raw
    if X is not None and holds_scaled(X) and raw is not None and not looks_like_counts(raw.X):
        if list(raw.var_names) == list(adata.var_names):
            x = Matrix(".raw", (n, g), value=raw.X, path=h5ad, key="raw/X")
            notes.append("X holds scaled values, so the log-normalised .raw is exported as X.")
        else:
            notes.append("X holds scaled values (z-scores); exported as they are.")
    x.is_counts = looks_like_counts(x.value)
    counts = None
    if "counts" in adata.layers:
        counts = Matrix("layers['counts']", (n, g), value=adata.layers["counts"],
                        path=h5ad, key="layers/counts", is_counts=True)
    elif raw is not None and list(raw.var_names) == list(adata.var_names) and looks_like_counts(raw.X):
        counts = Matrix(".raw", (n, g), value=raw.X, path=h5ad, key="raw/X", is_counts=True)
    elif x.is_counts:
        counts = x
    return x, counts, notes


# ---------------------------------------------------------------------------
# The plan


def plan(adaptor, opts: Options) -> Plan:
    """What the export holds and leaves out, without writing anything."""
    adata = adaptor.adata
    picked, obsm_dropped = _pick_embeddings(adata, opts.embeddings)
    obs, notes, cols_dropped, n_cells = clean_obs(adata.obs, opts.columns)
    original = {str(c).replace("/", "_"): str(c) for c in (opts.columns or list(adata.obs.columns))}
    x, counts, mnotes = matrices(adaptor)
    notes += mnotes
    var_src = adata.var
    if x.where == ".raw" and adata.raw is not None:
        var_src = adata.raw.var
    var, vnotes, n_genes = symbol_var(var_src)
    notes += vnotes
    obsm = {c["export_name"]: embedding_array(adata, c) for c in picked}
    for c in picked:
        if c["n_dims"] > 3:
            notes.append(f"{c['label']} has {c['n_dims']} columns; its first two are exported.")
        if not np.isfinite(obsm[c["export_name"]]).all():
            notes.append(f"{c['label']} has missing values (cells without a position).")
    colors = _colors(adata.uns, obs, original)
    uns: dict[str, Any] = dict(colors)
    for key in ("X_umap", "X_tsne"):
        if key in obsm:
            uns["default_embedding"] = key
            break
    raw = counts if (opts.include_raw and counts is not None and counts is not x) else None
    if opts.include_raw and counts is None:
        notes.append("No raw counts were found, so the file has no .raw.")
    elif opts.include_raw and counts is x:
        notes.append("X holds the raw counts already, so the file has no separate .raw.")
    dropped = {
        "uns": sorted(str(k) for k in adata.uns.keys() if k not in colors),
        "obsm": obsm_dropped,
        "layers": [str(k) for k in adata.layers.keys()],
        "obsp": [str(k) for k in adata.obsp.keys()],
        "varm": [str(k) for k in adata.varm.keys()],
        "obs": cols_dropped,
    }
    if adata.raw is not None and (raw is None or raw.where != ".raw") and x.where != ".raw":
        dropped["raw"] = [".raw (raw counts)" if looks_counts_raw(adata) else ".raw (log-normalised values)"]
    p = Plan(obs=obs, var=var, obsm=obsm, uns=uns, x=x, raw=raw, counts=counts,
             dropped={k: v for k, v in dropped.items() if v}, notes=notes,
             renamed={"cells": n_cells, "genes": n_genes})
    if n_cells:
        notes.append(f"{n_cells:,} duplicate cell names made unique ('-1', '-2' added).")
    if n_genes:
        notes.append(f"{n_genes:,} duplicate gene names made unique ('-1', '-2' added).")
    return p


def looks_counts_raw(adata) -> bool:
    from scview.core.pipeline import looks_like_counts

    return adata.raw is not None and looks_like_counts(adata.raw.X)


def estimate_bytes(p: Plan) -> int:
    """About how big the file is: the stored size of its matrices in the dataset's own
    files (gzip like the export), the embeddings and a share for obs/var."""
    total = 0
    for m in (p.x, p.raw):
        if m is None:
            continue
        stored = storage_bytes(m.path, m.key)
        if stored is None:
            val = m.value
            nnz = int(val.nnz) if sp.issparse(val) else int(np.prod(m.shape))
            stored = nnz * 6
        total += stored
    total += sum(int(a.nbytes) for a in p.obsm.values())
    total += int(p.obs.memory_usage(deep=True).sum() // 4) + int(p.var.memory_usage(deep=True).sum() // 4)
    return int(total) + 64 * 1024


# ---------------------------------------------------------------------------
# Writing


def provenance(dataset_name: str, dataset_id: str, kind: str) -> dict[str, str]:
    from scview.core.provenance import _scview_version

    return {"scview_version": _scview_version(), "dataset_name": dataset_name,
            "dataset_id": dataset_id, "format": kind,
            "exported_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat()}


def _write_matrix(f: h5py.File | h5py.Group, name: str, m: Matrix) -> None:
    if m.value is not None:
        from anndata.io import write_elem

        val = m.value
        if not (sp.issparse(val) or isinstance(val, np.ndarray)):
            val = val[:] if hasattr(val, "__getitem__") else np.asarray(val)
        write_elem(f, name, val, dataset_kwargs={"compression": "gzip"})
        return
    with h5py.File(m.path, "r") as src:
        large_mode._copy_matrix(src[m.key], f, name)


def write(p: Plan, out_path: Path, note: dict[str, str],
          stop: Callable[[], bool] = lambda: False,
          progress: Callable[[str], None] = lambda _m: None) -> Path:
    """Write the plan's h5ad at ``out_path`` (beside it first, then moved into place)."""
    import anndata as ad
    from anndata.io import write_elem

    out_path = Path(out_path)
    tmp = out_path.with_name(f".{out_path.name}.part")
    uns = dict(p.uns)
    uns["scview_export"] = note
    meta = ad.AnnData(obs=p.obs, var=p.var, obsm=p.obsm, uns=uns)
    try:
        progress("Writing cell and gene tables")
        meta.write_h5ad(tmp, compression="gzip")
        if stop():
            raise Cancelled()
        with h5py.File(tmp, "r+") as out:
            if "X" in out:
                del out["X"]
            progress(f"Writing the expression matrix (from {p.x.where})")
            _write_matrix(out, "X", p.x)
            if stop():
                raise Cancelled()
            if p.raw is not None:
                progress(f"Writing raw counts (from {p.raw.where})")
                if "raw" in out:
                    del out["raw"]
                grp = out.create_group("raw")
                grp.attrs["encoding-type"] = "raw"
                grp.attrs["encoding-version"] = "0.1.0"
                _write_matrix(grp, "X", p.raw)
                write_elem(grp, "var", pd.DataFrame(index=p.var.index))
                write_elem(grp, "varm", {})
        os.replace(tmp, out_path)
    finally:
        tmp.unlink(missing_ok=True)
    return out_path


# ---------------------------------------------------------------------------
# Readiness for CELLxGENE Discover

# (field, allowed term prefixes or literal values, what it is) for the obs columns a
# curator MUST annotate (schema 7.1.0, "obs (Cell Metadata)"); "na" is the value for
# cell lines (tissue_type "cell line"). organism_ontology_term_id is checked in uns.
_ONTOLOGY_FIELDS: list[tuple[str, tuple[str, ...], tuple[str, ...], str]] = [
    ("assay_ontology_term_id", ("EFO",), (), "assay (EFO)"),
    ("tissue_ontology_term_id", ("UBERON", "CL", "WBbt", "ZFA", "FBbt", "CVCL"), (),
     "tissue (UBERON; Cellosaurus CVCL_ for a cell line)"),
    ("disease_ontology_term_id", ("MONDO", "PATO"), (), "disease (MONDO, or PATO:0000461 for normal)"),
    ("cell_type_ontology_term_id", ("CL", "WBbt", "ZFA", "FBbt"), ("unknown", "na"), "cell type (CL)"),
    ("development_stage_ontology_term_id", ("HsapDv", "MmusDv", "UBERON", "WBls", "ZFS", "FBdv"),
     ("unknown", "na"), "development stage (HsapDv / MmusDv)"),
    ("sex_ontology_term_id", ("PATO",), ("unknown", "na"), "sex (PATO)"),
    ("self_reported_ethnicity_ontology_term_id", ("HANCESTRO", "AfPO"), ("unknown", "na"),
     "self-reported ethnicity (HANCESTRO; 'na' for non-human)"),
]
_LITERAL_FIELDS = {
    "suspension_type": ("cell", "nucleus", "na"),
    "tissue_type": ("tissue", "organoid", "primary cell culture", "cell line"),
}
# Columns Discover adds itself from the term ids: curators MUST NOT annotate them.
_ADDED_BY_DISCOVER = ("assay", "cell_type", "development_stage", "disease", "organism",
                      "self_reported_ethnicity", "sex", "tissue", "observation_joinid",
                      "experimental_condition", "perturbation_types")
# Fields joined with " || " when a cell has several terms (6.0.0).
_MULTI_TERM = ("disease_ontology_term_id", "self_reported_ethnicity_ontology_term_id")
_VAR_ADDED_BY_DISCOVER = ("feature_name", "feature_reference", "feature_biotype",
                          "feature_length", "feature_type")
_DEPRECATED_OBS = ("ethnicity", "ethnicity_ontology_term_id", "organism", "organism_ontology_term_id")
# OBO CURIE (EFO:0009922), or Cellosaurus's underscore form (CVCL_1P02).
_TERM = re.compile(r"^(?:([A-Za-z]+):\d+|(CVCL)_[A-Z0-9]+)$")
_DONOR_HINTS = re.compile(r"donor|patient|individual|subject|^ind$|replicate|sample", re.I)


def _item(key: str, label: str, status: str, detail: str) -> dict[str, str]:
    return {"key": key, "label": label, "status": status, "detail": detail}


def _bad_terms(values: pd.Series, prefixes: tuple[str, ...], literals: tuple[str, ...],
               multi: bool = False) -> list[str]:
    bad: list[str] = []
    for v in pd.unique(values.astype(str)):
        parts = v.split(" || ") if multi else [v]
        ok = v in literals or (len(set(parts)) == len(parts) and parts == sorted(parts) and all(
            (m := _TERM.match(p)) is not None and (m.group(1) or m.group(2)) in prefixes
            for p in parts))
        if not ok:
            bad.append(v)
    return bad


def readiness(p: Plan, species: str | None = None, title: str | None = None,
              uns: dict | None = None) -> dict[str, Any]:
    """Which Discover fields the export has, lacks or has in another form. ``status``:
    ok | missing | invalid | warn. ``uns``: the dataset's own uns, where the organism is
    read. Nothing is filled in."""
    obs, var = p.obs, p.var
    items: list[dict[str, str]] = []
    org = (uns or {}).get("organism_ontology_term_id")
    org_label = "organism (uns, NCBITaxon)"
    m = _TERM.match(org) if isinstance(org, str) else None
    if org is None:
        where = (" (it is in the cell metadata; schema 6.0.0 moved it to uns)"
                 if "organism_ontology_term_id" in obs.columns else "")
        items.append(_item("organism_ontology_term_id", org_label, "missing",
                           "uns['organism_ontology_term_id'] is not set" + where + "."))
    elif m is None or m.group(1) != "NCBITaxon":
        items.append(_item("organism_ontology_term_id", org_label, "invalid",
                           f"Not an NCBITaxon term: {org}."))
    else:
        items.append(_item("organism_ontology_term_id", org_label, "ok", str(org)))
    for key, prefixes, literals, label in _ONTOLOGY_FIELDS:
        if key not in obs.columns:
            items.append(_item(key, label, "missing", "Not in the cell metadata."))
            continue
        s = obs[key]
        if s.isna().any():
            items.append(_item(key, label, "invalid", f"{int(s.isna().sum()):,} cells have no value."))
            continue
        bad = _bad_terms(s, prefixes, literals, multi=key in _MULTI_TERM)
        if key == "disease_ontology_term_id":
            bad += [v for v in pd.unique(s.astype(str)) if v.startswith("PATO") and v != "PATO:0000461"]
        if bad:
            shown = ", ".join(bad[:3]) + (" …" if len(bad) > 3 else "")
            items.append(_item(key, label, "invalid",
                               "Not a term id of the expected kind (several terms: sorted, "
                               f"joined with ' || '): {shown}."))
        else:
            items.append(_item(key, label, "ok", f"{s.nunique():,} values."))
    for key, allowed in _LITERAL_FIELDS.items():
        if key not in obs.columns:
            items.append(_item(key, key.replace("_", " "), "missing",
                               f"Not in the cell metadata (one of: {', '.join(allowed)})."))
            continue
        bad = [v for v in pd.unique(obs[key].astype(str)) if v not in allowed]
        items.append(_item(key, key.replace("_", " "), "invalid" if bad else "ok",
                           f"Values must be one of: {', '.join(allowed)}." if bad else "Valid values."))
    if "donor_id" in obs.columns:
        items.append(_item("donor_id", "donor id", "ok", f"{obs['donor_id'].nunique():,} donors."))
    else:
        hints = [c for c in obs.columns if _DONOR_HINTS.search(str(c))]
        hint = f" Column(s) {', '.join(repr(c) for c in hints[:3])} may hold it." if hints else ""
        items.append(_item("donor_id", "donor id", "missing", "Not in the cell metadata." + hint))
    if "is_primary_data" in obs.columns:
        ok = pd.api.types.is_bool_dtype(obs["is_primary_data"]) or set(
            pd.unique(obs["is_primary_data"].astype(str))) <= {"True", "False"}
        items.append(_item("is_primary_data", "is primary data", "ok" if ok else "invalid",
                           "True/False per cell." if ok else "Must be True or False."))
    else:
        items.append(_item("is_primary_data", "is primary data", "missing",
                           "Not in the cell metadata (True unless the cells were published before)."))
    clash = [c for c in _ADDED_BY_DISCOVER if c in obs.columns]
    if clash:
        items.append(_item("reserved_obs", "reserved column names", "invalid",
                           f"Discover adds {', '.join(clash)} itself from the term ids: rename "
                           "these columns before submitting."))
    old = [c for c in obs.columns if c in _DEPRECATED_OBS or str(c).startswith("__")]
    if old:
        items.append(_item("deprecated_obs", "deprecated column names", "invalid",
                           f"Not allowed: {', '.join(old)}."))
    # Genes
    ens_index = np.mean([bool(ENSEMBL.match(v)) for v in var.index[:: max(1, len(var) // 2000)]]) > 0.5
    id_col = next((c for c in ID_COLUMNS if c in var.columns), None)
    if ens_index:
        items.append(_item("var_index", "gene ids", "ok", "Gene names are Ensembl ids."))
    elif id_col:
        items.append(_item("var_index", "gene ids", "warn",
                           f"Discover needs Ensembl ids as the gene index; they are in var['{id_col}'] "
                           "(this file keeps symbols for the desktop explorer)."))
    else:
        items.append(_item("var_index", "gene ids", "missing",
                           "Discover needs Ensembl gene ids; this dataset has gene symbols only."))
    if "feature_is_filtered" in var.columns:
        items.append(_item("feature_is_filtered", "feature_is_filtered", "ok", "In var."))
    else:
        items.append(_item("feature_is_filtered", "feature_is_filtered", "missing",
                           "Not in var (True for genes zeroed in X but kept in raw.X)."))
    vclash = [c for c in _VAR_ADDED_BY_DISCOVER if c in var.columns]
    if vclash:
        items.append(_item("reserved_var", "reserved gene columns", "invalid",
                           f"Discover adds {', '.join(vclash)} itself: remove them before submitting."))
    # Matrices
    if p.x.is_counts:
        items.append(_item("raw_counts", "raw counts", "ok", "X holds raw counts."))
    elif p.raw is not None:
        items.append(_item("raw_counts", "raw counts", "ok", f"raw.X holds raw counts (from {p.raw.where})."))
    elif p.counts is not None:
        items.append(_item("raw_counts", "raw counts", "missing",
                           "The dataset has raw counts; tick 'Include raw counts' to put them in raw.X."))
    else:
        items.append(_item("raw_counts", "raw counts", "missing",
                           "No raw counts found; Discover requires them (raw.X, or X when unnormalised)."))
    if p.x.value is not None and not sp.issparse(p.x.value) and isinstance(p.x.value, np.ndarray):
        zeros = float(np.mean(p.x.value[: min(2000, p.x.shape[0])] == 0))
        if zeros >= 0.5:
            items.append(_item("sparse", "sparse matrix", "invalid",
                               f"X is dense with {zeros:.0%} zeros; Discover requires CSR sparse at 50% or more."))
    items.append(_item("embedding", "embedding", "ok" if p.obsm else "missing",
                       f"{len(p.obsm)} in obsm." if p.obsm else "At least one X_ embedding is required."))
    items.append(_item("title", "dataset title", "missing",
                       "uns['title'] is not set" + (f" (the scView name is '{title}')." if title else ".")))
    hints: list[str] = []
    if species in ("human", "mouse"):
        hints.append(f"scView's species call from gene names: {species}. Use it to choose the "
                     "organism term yourself; no term is filled in for you.")
    n_ok = sum(i["status"] == "ok" for i in items)
    return {"schema_version": SCHEMA_VERSION, "schema_url": SCHEMA_URL, "items": items,
            "n_ok": n_ok, "n_items": len(items),
            "ready": all(i["status"] == "ok" for i in items), "hints": hints}
