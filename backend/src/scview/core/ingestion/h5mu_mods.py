"""The modalities of an .h5mu (MuData on disk) beside RNA, kept at import.

The RNA modality becomes the dataset (``loading._load_h5mu``) and ATAC peaks go to their own
store (core/atac_import.py). The others are mapped to what scView already reads, as a 10x .h5's
feature types are:

- adt / prot / protein / cite -> the protein panel (obsm ``protein_counts`` + ``protein_clr``,
  core/protein.py, as Antibody Capture features are);
- hto / hashtags -> obsm ``hto_counts`` (core/hashing.py);
- gdo / crispr / guides / sgrna -> obsm ``guide_counts`` (core/crispr.py).

Each comes from the modality's ``counts`` layer when it has one, else its X, and must be counts
(no negative or fractional values). Rows follow the RNA cells by name; a cell a modality lacks
gets 0. A modality that is none of these, or that cannot be kept, is named in an import issue
(``plan`` feeds the validation step) instead of disappearing without a word, as pertpy's
papalexi_2021 (rna, adt, hto, gdo) did before v0.78.

obs: MuData keeps a table per modality and one of its own, whose columns are named
``<mod>:<column>`` once pulled (``rna:replicate``). The RNA modality's obs is the dataset's; a
column of the shared table or of another modality that the RNA obs lacks is added by its plain
name (``replicate``), or as ``<mod>_<column>`` when that name is taken.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import anndata as ad
import h5py
import numpy as np
import pandas as pd
import scipy.sparse as sp

logger = logging.getLogger(__name__)

PROTEIN_MODS = ("adt", "prot", "protein", "cite", "antibody")
HASHTAG_MODS = ("hto", "hashtags", "hashtag", "cmo")
GUIDE_MODS = ("gdo", "crispr", "guides", "guide", "sgrna")
ROLE_WORDS = {"protein": "the protein panel (Protein tab)", "hashtags": "cell hashing tags (Hashing)",
              "guides": "CRISPR guide counts (Perturbation)"}
MAX_OBS_COLUMNS = 200  # columns copied from the shared and other tables at most


@dataclass
class ModPlan:
    """One non-RNA modality: what it becomes (``role``: protein | hashtags | guides | atac | None)
    and, when it is left out, why."""

    name: str
    n_cells: int
    n_features: int
    role: str | None
    reason: str | None = None


def _role(name: str) -> str | None:
    from scview.core import atac_import

    low = name.lower()
    if name in atac_import.ATAC_MODS or low in ("atac", "peaks"):
        return "atac"
    if low in PROTEIN_MODS:
        return "protein"
    if low in HASHTAG_MODS:
        return "hashtags"
    if low in GUIDE_MODS:
        return "guides"
    return None


def _limit(role: str) -> int:
    from scview.core import crispr, hashing, protein

    return {"protein": protein.MAX_PROTEINS, "hashtags": hashing.MAX_TAGS, "guides": crispr.MAX_GUIDES}[role]


def _shape(g: h5py.Group) -> tuple[int, int]:
    x = g["X"]
    if isinstance(x, h5py.Dataset):
        return int(x.shape[0]), int(x.shape[1])
    shape = x.attrs.get("shape", x.attrs.get("h5sparse_shape"))
    return int(shape[0]), int(shape[1])


def rna_mod(f: h5py.File) -> str | None:
    from scview.core import atac_import

    return next((m for m in atac_import.RNA_MODS if m in f.get("mod", {})), None)


def plan(path: str | Path) -> list[ModPlan]:
    """Every modality but RNA, with what it becomes (reads shapes only)."""
    out: list[ModPlan] = []
    with h5py.File(path, "r") as f:
        rna = rna_mod(f)
        for name in f.get("mod", {}):
            if name == rna:
                continue
            try:
                n, k = _shape(f["mod"][name])
            except (KeyError, TypeError, IndexError):
                out.append(ModPlan(name, 0, 0, None, "its matrix could not be read"))
                continue
            role = _role(name)
            reason = None
            if role is None:
                reason = "scView has no place for this modality"
            elif role != "atac" and k > _limit(role):
                reason = f"{k:,} features is more than {_limit(role):,}: not a {role} panel"
            out.append(ModPlan(name, n, k, None if reason else role, reason))
    return out


def sentences(path: str | Path) -> tuple[str | None, str | None]:
    """(what the other modalities become, what is left out and why) for the import issues."""
    mods = plan(path)
    kept = [m for m in mods if m.role and m.role != "atac"]
    left = [m for m in mods if m.role is None]
    kept_text = None
    if kept:
        kept_text = ("Besides RNA, this .h5mu's " + "; ".join(
            f"{m.name} ({m.n_features:,} features) becomes {ROLE_WORDS[m.role]}" for m in kept) + ".")
    left_text = None
    if left:
        left_text = ("Not read from this .h5mu: " + "; ".join(f"{m.name} ({m.n_features:,} features: {m.reason})"
                                                            for m in left) + ".")
    return kept_text, left_text


def _counts(g: h5py.Group) -> np.ndarray | None:
    """A modality's counts as a dense float array (its counts layer, else X), or None when the
    values are not counts."""
    from anndata.io import read_elem

    node = g["layers"]["counts"] if "layers" in g and "counts" in g["layers"] else g["X"]
    m = read_elem(node)
    dense = m.toarray() if sp.issparse(m) else np.asarray(m)
    dense = dense.astype(np.float64, copy=False)
    if dense.size and (np.nanmin(dense) < 0 or not np.allclose(dense, np.round(dense))):
        return None
    return dense


def _aligned(values: np.ndarray, names: pd.Index, cells: pd.Index) -> np.ndarray:
    """``values`` (rows named ``names``) in the order of ``cells``; 0 for a cell without a row."""
    idx = names.get_indexer(cells)
    out = np.zeros((len(cells), values.shape[1]), dtype=np.float64)
    hit = idx >= 0
    out[hit] = values[idx[hit]]
    return out


def adopt(f: h5py.File, adata: ad.AnnData, rna: str) -> list[str]:
    """Put the other modalities of the open .h5mu ``f`` on ``adata`` (its RNA cells) and the obs
    columns the RNA table lacks; returns a sentence per modality left out at this stage."""
    from anndata.io import read_elem

    from scview.core import crispr, hashing, protein
    from scview.core.ingestion import feature_split as fs

    left: list[str] = []
    cells = adata.obs_names.astype(str)
    for p in plan(f.filename):
        if p.role in (None, "atac"):
            continue
        g = f["mod"][p.name]
        try:
            values = _counts(g)
            names = fs._obs_names(g["obs"])
            features = [str(v) for v in read_elem(g["var"]).index]
        except Exception as e:  # one unreadable modality never stops the import
            logger.warning("h5mu modality %s not read: %s", p.name, e)
            left.append(f"{p.name}: could not be read ({e})")
            continue
        if values is None:
            left.append(f"{p.name}: its values are not counts")
            continue
        shared = int((names.get_indexer(cells) >= 0).sum())
        if shared == 0:
            left.append(f"{p.name}: none of its cells are among the RNA cells")
            continue
        counts = _aligned(values, names, cells)
        if p.role == "protein":
            protein.attach(adata, counts, features, protein.ProteinSplit())
        elif p.role == "hashtags":
            hashing.attach_counts(adata, counts, features, f"h5mu {p.name}")
        else:
            crispr.attach_counts(adata, counts, features, f"h5mu {p.name}")
    _adopt_obs(f, adata, rna)
    return left


def _adopt_obs(f: h5py.File, adata: ad.AnnData, rna: str) -> None:
    """Columns of the shared obs table and of the other modalities' obs that the RNA obs lacks."""
    from anndata.io import read_elem

    cells = adata.obs_names.astype(str)
    tables: list[tuple[str | None, pd.DataFrame]] = []
    if "obs" in f:
        tables.append((None, read_elem(f["obs"])))
    for name in f.get("mod", {}):
        if name != rna and "obs" in f["mod"][name]:
            tables.append((name, read_elem(f["mod"][name]["obs"])))
    added = 0
    for mod, table in tables:
        if table.shape[1] == 0:
            continue
        table.index = table.index.astype(str)
        idx = table.index.get_indexer(cells)
        if not (idx >= 0).any():
            continue
        for col in table.columns:
            if added >= MAX_OBS_COLUMNS:
                return
            source, _, plain = str(col).rpartition(":") if mod is None else (mod, "", str(col))
            if source == rna or (mod is None and not source):
                source = None  # the RNA modality's own column, or one of the shared table
            if plain in adata.obs.columns:
                if source is None:
                    continue  # the same column the RNA obs already has
                name = f"{source}_{plain}"
                if name in adata.obs.columns or _same(adata.obs[plain], table[col], idx):
                    continue
            else:
                name = plain
            s = table[col]
            hit = idx >= 0
            values = pd.Series(pd.NA, index=adata.obs_names, dtype=object)
            values[hit] = s.to_numpy(dtype=object)[idx[hit]]
            adata.obs[name] = _typed(values, s)
            added += 1


def _same(have: pd.Series, other: pd.Series, idx: np.ndarray) -> bool:
    """Whether ``other`` (rows at ``idx``) holds the values ``have`` does."""
    hit = idx >= 0
    try:
        a = have.astype(str).to_numpy()[hit]
        b = other.astype(str).to_numpy()[idx[hit]]
    except (TypeError, ValueError):
        return False
    return bool(np.array_equal(a, b))


def _typed(values: pd.Series, like: pd.Series) -> pd.Series:
    """``values`` (object) back to the source column's kind: numbers stay numbers, the rest
    categorical (what h5ad writes cleanly)."""
    if pd.api.types.is_numeric_dtype(like.dtype) and not pd.api.types.is_bool_dtype(like.dtype):
        return pd.to_numeric(values, errors="coerce")
    if pd.api.types.is_bool_dtype(like.dtype):
        return values.astype("boolean")
    return values.astype("string").astype("category")
