"""Feature types in a plain .h5ad upload (``POST /datasets/upload``).

An .h5ad written from a 10x feature-barcoding run keeps every feature type in X
(``var["feature_types"]``): CITE-seq antibodies ("Antibody Capture") and CellPlex
tags ("Multiplexing Capture") sit beside the genes and enter the variable genes,
the normalisation and the marker lists. The import wizard takes them out at load
(``core/protein.py`` ``split_features``); this does the same for the plain upload,
with the same rules: genes stay in X, antibodies go to ``obsm["protein_counts"]`` +
``obsm["protein_clr"]``, CellPlex tags to ``obsm["hto_counts"]``, CRISPR guides to
``obsm["guide_counts"]``, other types (custom) are counted and dropped.

Where and how: at upload, from the stored original (which stays byte-identical
and read-only) into ``ingested/{id}/``, the layer the dataset is read from (before
the upload; processing still writes to ``derived/``). The new file is written one
block of rows at a time straight from the original's HDF5 groups, so the split
never holds the dataset in memory: its peak is one block (``BLOCK_NNZ`` stored
values of a sparse matrix, ``BLOCK_VALUES`` of a dense one) plus the small
antibody / tag tables, whatever the file's size. Doing it at first open instead
would mean re-splitting on every open until something saved the dataset, in the
memory the open already needs, and could not tell an upload made before this
change (left as it was) from a new one.
"""

from __future__ import annotations

import logging
from pathlib import Path

import anndata as ad
import h5py
import numpy as np
import pandas as pd
from anndata.io import read_elem, write_elem

from scview.core import crispr, hashing, protein, provenance

logger = logging.getLogger(__name__)

# Measured on a 37,104-cell CITE-seq file (45.7M stored values): 2M per block peaks at
# +0.14 GiB in 7.3 s, 16M at +0.60 GiB in 7.4 s (the time is the gzip writing).
BLOCK_NNZ = 2_000_000  # stored values of a sparse matrix per block
BLOCK_VALUES = 4_000_000  # values of a dense matrix per block
_GZIP = {"compression": "gzip", "compression_opts": 4}  # anndata's own default


class SplitUnsupported(RuntimeError):
    """The file stores a matrix in a layout this streaming split does not read."""


def feature_types(path: Path) -> np.ndarray | None:
    """``var["feature_types"]`` read from the file (nothing else is read), or None."""
    try:
        with h5py.File(path, "r") as f:
            var = f.get("var")
            if not isinstance(var, h5py.Group) or "feature_types" not in var:
                return None
            return np.asarray(pd.Series(read_elem(var["feature_types"])).astype(str))
    except (OSError, KeyError, ValueError, TypeError) as e:
        logger.warning("Could not read feature types of %s: %s", path, e)
        return None


def needs_split(path: Path) -> bool:
    """Whether the file has genes and another feature type in X (what the import
    wizard's ``split_features`` would take out)."""
    types = feature_types(path)
    if types is None:
        return False
    genes = types == protein.GENE_TYPE
    return bool(genes.any()) and not bool(genes.all())


def split_h5ad(src: Path, dst: Path, *, original_filename: str, atac_store: Path | None = None) -> dict:
    """Write ``src`` to ``dst`` with only the gene features in X, layers and ``.raw``
    (when its own ``feature_types`` say which), the antibodies and CellPlex tags in
    obsm, and the split recorded in the provenance (an "ingest" step). Returns what
    was split. Raises SplitUnsupported for a matrix layout it cannot stream.
    ``atac_store``: where Peaks features are written (``core/atac_import.py``), per-cell
    totals going to obs; without it they are only left out."""
    types = feature_types(src)
    if types is None:
        raise SplitUnsupported("no var['feature_types']")
    genes = types == protein.GENE_TYPE
    is_ab = types == protein.ANTIBODY_TYPE
    is_cmo = types == hashing.CMO_TYPE
    is_guide = types == crispr.GUIDE_TYPE
    take_ab = is_ab if 0 < is_ab.sum() <= protein.MAX_PROTEINS else np.zeros_like(is_ab)
    take_cmo = is_cmo if 0 < is_cmo.sum() <= hashing.MAX_TAGS else np.zeros_like(is_cmo)
    take_guide = is_guide if 0 < is_guide.sum() <= crispr.MAX_GUIDES else np.zeros_like(is_guide)
    is_peak = types == protein.PEAKS_TYPE  # to the ATAC store (atac_store), never counted as dropped
    other = pd.Series(types[~genes & ~is_ab & ~is_cmo & ~take_guide & ~is_peak]).value_counts()

    with h5py.File(src, "r") as fi, h5py.File(dst, "w") as fo:
        for k, v in fi.attrs.items():
            fo.attrs[k] = v
        var = read_elem(fi["var"])
        obs_names = _obs_names(fi["obs"])
        extracted: list[np.ndarray] = []
        for key in fi.keys():
            if key == "X":
                extracted = _subset_matrix(fi["X"], fo, "X", genes, [take_ab, take_cmo, take_guide])
            elif key == "layers":
                g = _group_like(fi["layers"], fo, "layers")
                for name in fi["layers"].keys():
                    _subset_matrix(fi["layers"][name], g, name, genes, [])
            elif key == "var":
                write_elem(fo, "var", var.iloc[np.flatnonzero(genes)])
            elif key in ("varm", "varp"):
                g = _group_like(fi[key], fo, key)
                for name in fi[key].keys():
                    m = read_elem(fi[key][name])
                    m = m.iloc[genes] if isinstance(m, pd.DataFrame) else m[genes]
                    if key == "varp":
                        m = m[:, genes]
                    write_elem(g, name, m)
            elif key == "raw":
                _copy_raw(fi["raw"], fo)
            else:  # obs, obsm, obsp, uns: unchanged
                fi.copy(fi[key], fo, name=key)

        holder = ad.AnnData(obs=pd.DataFrame(index=obs_names),
                            var=pd.DataFrame(index=pd.Index(var.index[genes].astype(str))))
        uns = fi.get("uns")
        for key in (protein.UNS_KEY, hashing.UNS_KEY, crispr.UNS_KEY, provenance.UNS_KEY):
            if isinstance(uns, h5py.Group) and key in uns:
                holder.uns[key] = read_elem(uns[key])
        split = protein.ProteinSplit()
        split.other_features = {str(k): int(v) for k, v in other.items()}
        atac_params = _write_atac(src, atac_store, holder, is_peak, original_filename, fo)
        names = [str(v) for v in var.index]
        if not extracted:  # no X to take them from
            take_ab, take_cmo = np.zeros_like(take_ab), np.zeros_like(take_cmo)
            take_guide = np.zeros_like(take_guide)
        if take_ab.any():
            protein.attach(holder, extracted[0], [n for n, t in zip(names, take_ab) if t], split)
        elif split.other_features or is_ab.any():
            holder.uns[protein.UNS_KEY] = split.to_dict()
        if take_cmo.any():
            hashing.attach_counts(holder, extracted[1], [n for n, t in zip(names, take_cmo) if t],
                                  hashing.CMO_TYPE)
        if take_guide.any():
            crispr.attach_counts(holder, extracted[2], [n for n, t in zip(names, take_guide) if t])
        summary = {
            "n_genes": int(genes.sum()),
            "n_antibodies": int(take_ab.sum()),
            "n_multiplexing_tags": int(take_cmo.sum()),
            "n_guides": int(take_guide.sum()),
            "other_features": split.other_features,
            "location": "obsm protein_counts / protein_clr, hto_counts, guide_counts",
        }
        if atac_params:
            summary["n_peaks"] = int(is_peak.sum())
        if is_ab.sum() > protein.MAX_PROTEINS:
            summary["note"] = f"{int(is_ab.sum())} antibody features: more than a panel, dropped from X"
        provenance.init_source(holder, origin="uploaded", original_filename=original_filename, fmt="h5ad")
        provenance.record_step(holder, step="ingest", tool="scview.ingestion", params={
            "formats": ["h5ad"], "feature_split": summary,
            **({"protein": {"n_proteins": split.n_proteins, "controls": split.controls,
                            "normalisation": "CLR per cell (Seurat margin = 2)"}}
               if split.n_proteins else {}),
            **atac_params,
        })
        obsm = fo["obsm"] if "obsm" in fo else _new_dict(fo, "obsm")
        for key, value in holder.obsm.items():
            if key in obsm:
                del obsm[key]
            write_elem(obsm, key, value)
        uns_out = fo["uns"] if "uns" in fo else _new_dict(fo, "uns")
        for key, value in holder.uns.items():
            if key in uns_out:
                del uns_out[key]
            write_elem(uns_out, key, value)
    return summary


def _write_atac(src: Path, store: Path | None, holder: ad.AnnData, is_peak: np.ndarray, filename: str,
                fo: h5py.File) -> dict:
    """The Peaks features into their store; the per-cell totals go straight into the new
    file's obs (copied as stored). Returns the history entry's ATAC part ({} without peaks)."""
    if store is None or not is_peak.any():
        return {}
    from scview.core import atac_import
    from scview.core.ingestion.bundling import BundleFile, FileKind, FileRole, IngestUnit, UnitFormat

    unit = IngestUnit(label=Path(filename).stem, format=UnitFormat.anndata, complete=True, files=[BundleFile(
        path=str(src), name=filename, kind=FileKind.anndata_h5ad, role=FileRole.data, rel_path=filename)])
    plan = atac_import.plan_unit(unit)
    if plan is None or plan.rows is None:
        if plan is not None and plan.note:
            holder.uns[atac_import.LEFT_OUT_KEY] = plan.note
        return {}
    info = atac_import.attach_obs(holder, atac_import.write_store([plan], store))
    if isinstance(fo["obs"], h5py.Group):
        order = [str(c) for c in fo["obs"].attrs.get("column-order", [])]
        for column in (atac_import.OBS_TOTAL, atac_import.OBS_N_PEAKS):
            if column in fo["obs"]:
                del fo["obs"][column]
            write_elem(fo["obs"], column, holder.obs[column].to_numpy())
            order = [c for c in order if c != column] + [column]
        fo["obs"].attrs["column-order"] = np.asarray(order, dtype=object)
    holder.obs = holder.obs.drop(columns=[atac_import.OBS_TOTAL, atac_import.OBS_N_PEAKS])
    return atac_import.provenance_params(info)


# ---------------------------------------------------------------------------
# Streaming one matrix
# ---------------------------------------------------------------------------


def _obs_names(obs) -> pd.Index:
    """The cell names alone (a million-cell obs table can hold many columns)."""
    idx = obs.attrs.get("_index") if isinstance(obs, h5py.Group) else None
    if isinstance(idx, bytes):
        idx = idx.decode()
    if idx and idx in obs:
        return pd.Index(np.asarray(read_elem(obs[idx])).astype(str))
    return pd.Index(read_elem(obs).index.astype(str))


def _new_dict(parent: h5py.Group, name: str) -> h5py.Group:
    g = parent.create_group(name)
    g.attrs["encoding-type"] = "dict"
    g.attrs["encoding-version"] = "0.1.0"
    return g


def _group_like(src: h5py.Group, parent: h5py.Group, name: str) -> h5py.Group:
    g = parent.create_group(name)
    for k, v in src.attrs.items():
        g.attrs[k] = v
    return g


def _copy_raw(raw: h5py.Group, fo: h5py.File) -> None:
    """``.raw`` keeps its own features: subset only when its var says which are genes."""
    rvar = read_elem(raw["var"]) if "var" in raw else None
    types = (rvar["feature_types"].astype(str).to_numpy()
             if isinstance(rvar, pd.DataFrame) and "feature_types" in rvar.columns else None)
    keep = None if types is None else types == protein.GENE_TYPE
    if keep is None or keep.all() or not keep.any():
        raw.file.copy(raw, fo, name="raw")
        return
    g = _group_like(raw, fo, "raw")
    for key in raw.keys():
        if key == "X":
            _subset_matrix(raw["X"], g, "X", keep, [])
        elif key == "var":
            write_elem(g, "var", rvar.iloc[np.flatnonzero(keep)])
        elif key == "varm":
            vg = _group_like(raw["varm"], g, "varm")
            for name in raw["varm"].keys():
                m = read_elem(raw["varm"][name])
                write_elem(vg, name, m.iloc[keep] if isinstance(m, pd.DataFrame) else m[keep])
        else:
            raw.file.copy(raw[key], g, name=key)


def _subset_matrix(src, parent: h5py.Group, name: str, keep: np.ndarray,
                   extract: list[np.ndarray]) -> list[np.ndarray]:
    """Write ``src`` (cells x features) to ``parent[name]`` with the ``keep`` columns
    only, a block at a time; return the ``extract`` columns as small dense arrays."""
    enc = src.attrs.get("encoding-type", "array" if isinstance(src, h5py.Dataset) else None)
    if isinstance(enc, bytes):
        enc = enc.decode()
    if enc == "csr_matrix":
        return _subset_csr(src, parent, name, keep, extract)
    if enc == "csc_matrix":
        return _subset_csc(src, parent, name, keep, extract)
    if enc == "array" and isinstance(src, h5py.Dataset) and src.ndim == 2:
        return _subset_dense(src, parent, name, keep, extract)
    raise SplitUnsupported(f"{name} is stored as {enc!r}")


def _remap(mask: np.ndarray) -> np.ndarray:
    out = np.full(mask.shape[0], -1, dtype=np.int64)
    out[mask] = np.arange(int(mask.sum()))
    return out


def _index_dtype(src) -> np.dtype:
    """One dtype for the new indices and indptr: anndata assigns them to the scipy
    matrix as stored, and scipy's kernels (``eliminate_zeros``, used by scanpy's
    rank_genes_groups) refuse an int64 indptr beside int32 indices. The source's
    indices dtype, unless its value count needs int64."""
    dtype = np.dtype(src["indices"].dtype)
    if src["data"].shape[0] > np.iinfo(np.int32).max or dtype.itemsize > 4:
        return np.dtype(np.int64)
    return np.dtype(np.int32)


def _sparse_out(parent: h5py.Group, name: str, enc: str, shape: tuple[int, int], src) -> h5py.Group:
    g = parent.create_group(name)
    g.attrs["encoding-type"] = enc
    g.attrs["encoding-version"] = "0.1.0"
    g.attrs["shape"] = np.asarray(shape, dtype=np.int64)
    dtypes = {"data": src["data"].dtype, "indices": _index_dtype(src)}
    for key, dtype in dtypes.items():
        g.create_dataset(key, shape=(0,), maxshape=(None,), dtype=dtype,
                         chunks=(min(1 << 16, max(1, int(src[key].shape[0]))),), **_GZIP)
    return g


def _append(ds: h5py.Dataset, values: np.ndarray) -> None:
    if values.size:
        n = ds.shape[0]
        ds.resize((n + values.size,))
        ds[n:] = values


def _blocks(indptr: np.ndarray, budget: int):
    """Contiguous [start, end) ranges of the major axis holding about ``budget`` values each."""
    n = len(indptr) - 1
    start = 0
    while start < n:
        end = int(np.searchsorted(indptr, indptr[start] + budget, side="right")) - 1
        end = min(n, max(end, start + 1))
        yield start, end
        start = end


def _subset_csr(src, parent, name, keep, extract):
    n_obs, n_var = (int(x) for x in src.attrs["shape"])
    indptr = src["indptr"][:].astype(np.int64)
    remap = _remap(keep)
    out = _sparse_out(parent, name, "csr_matrix", (n_obs, int(keep.sum())), src)
    new_indptr = np.zeros(n_obs + 1, dtype=np.int64)
    side = [(m, _remap(m), np.zeros((n_obs, int(m.sum())), dtype=np.float32)) for m in extract]
    for start, end in _blocks(indptr, BLOCK_NNZ):
        lo, hi = indptr[start], indptr[end]
        data, idx = src["data"][lo:hi], src["indices"][lo:hi].astype(np.int64)
        rows = np.repeat(np.arange(end - start), np.diff(indptr[start:end + 1]))
        k = keep[idx]
        _append(out["data"], data[k])
        _append(out["indices"], remap[idx[k]].astype(out["indices"].dtype))
        new_indptr[start + 1:end + 1] = new_indptr[start] + np.cumsum(
            np.bincount(rows[k], minlength=end - start))
        for m, rm, table in side:
            sel = m[idx]
            np.add.at(table, (start + rows[sel], rm[idx[sel]]), data[sel])
    out.create_dataset("indptr", data=new_indptr.astype(out["indices"].dtype))
    return [t for _, _, t in side]


def _subset_csc(src, parent, name, keep, extract):
    n_obs, n_var = (int(x) for x in src.attrs["shape"])
    indptr = src["indptr"][:].astype(np.int64)
    out = _sparse_out(parent, name, "csc_matrix", (n_obs, int(keep.sum())), src)
    lengths = []
    side = [(m, _remap(m), np.zeros((n_obs, int(m.sum())), dtype=np.float32)) for m in extract]
    for start, end in _blocks(indptr, BLOCK_NNZ):
        lo, hi = indptr[start], indptr[end]
        data, idx = src["data"][lo:hi], src["indices"][lo:hi]
        cols = np.repeat(np.arange(start, end), np.diff(indptr[start:end + 1]))
        k = keep[cols]
        _append(out["data"], data[k])
        _append(out["indices"], idx[k].astype(out["indices"].dtype))
        lengths.append(np.diff(indptr[start:end + 1])[keep[start:end]])
        for m, rm, table in side:
            sel = m[cols]
            np.add.at(table, (idx[sel].astype(np.int64), rm[cols[sel]]), data[sel])
    out.create_dataset("indptr", data=np.concatenate([[0], np.cumsum(np.concatenate(lengths or [[]]))])
                       .astype(out["indices"].dtype))
    return [t for _, _, t in side]


def _subset_dense(src, parent, name, keep, extract):
    n_obs, n_var = src.shape
    cols = np.flatnonzero(keep)
    out = parent.create_dataset(name, shape=(n_obs, len(cols)), dtype=src.dtype, chunks=True, **_GZIP)
    out.attrs["encoding-type"] = "array"
    out.attrs["encoding-version"] = "0.2.0"
    side = [(np.flatnonzero(m), np.zeros((n_obs, int(m.sum())), dtype=np.float32)) for m in extract]
    step = max(1, BLOCK_VALUES // max(1, n_var))
    for start in range(0, n_obs, step):
        block = src[start:start + step]
        out[start:start + step] = block[:, cols]
        for idx, table in side:
            table[start:start + step] = block[:, idx]
    return [t for _, t in side]
