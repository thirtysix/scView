"""CITE-seq: antibody-derived tag (ADT) counts as a protein modality.

A 10x feature-barcoding run puts the antibody counts ("Antibody Capture") in
the same matrix as the genes. Left there, they are read as genes: they skew the
library sizes, enter the variable-gene selection and the PCA, and are named
like genes ("CD3_TotalSeqB") in every gene list. This module takes them out of
the gene matrix into ``obsm["protein_counts"]`` (a DataFrame, columns = protein
names) and adds ``obsm["protein_clr"]``, the counts normalised per cell by the
centred log-ratio, as Seurat's ``NormalizeData(method = "CLR", margin = 2)``
does (the normalisation of its weighted-nearest-neighbour vignette):

    clr(x) = log1p(x / exp(sum(log1p(x)) / n))      for each cell's n proteins

ADT counts carry a background from free antibody in the droplet that the
per-cell ratio cancels in part; DSB, which subtracts that background measured
in empty droplets, needs the unfiltered matrix and is not done here.

Other non-gene features are taken out of the gene matrix too: CellPlex tags
into ``obsm["hto_counts"]`` (core/hashing.py), CRISPR guides into
``obsm["guide_counts"]`` (core/crispr.py), custom features only counted.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp

COUNTS_KEY = "protein_counts"
CLR_KEY = "protein_clr"
UNS_KEY = "scview_protein"
GENE_TYPE = "Gene Expression"
ANTIBODY_TYPE = "Antibody Capture"
# obsm keys other tools use for the same counts (totalVI's protein_expression).
FOREIGN_KEYS = ("protein_expression", "protein", "adt", "ADT")
MAX_PROTEINS = 1000  # more is not a panel: a misread table

# BioLegend TotalSeq and BD AbSeq suffixes on 10x feature names.
_SUFFIX = re.compile(r"[_.-](TotalSeq[-_]?[ABCD]|TotalA|TotalB|TotalC|AbSeq|ADT)$", re.I)
# Isotype controls: "IgG1_control", "Mouse IgG2a isotype Ctrl", "Isotype_MOPC-21", "rat_IgG2b".
_CONTROL = re.compile(r"isotype|(^|[_\s.-])ctrl($|[_\s.-])|ig[gmae]\d?[abc]?[_\s.-]?(kappa|lambda)?[_\s.-]?control"
                      r"|^(mouse|rat|hamster|armenian)[_\s.-]?ig[gmae]", re.I)


@dataclass
class ProteinSplit:
    """What split_features did, for the provenance note and the ingest summary."""

    n_proteins: int = 0
    names: list[str] = field(default_factory=list)  # display names, in matrix order
    original: dict[str, str] = field(default_factory=dict)  # display -> feature name, when renamed
    controls: list[str] = field(default_factory=list)
    other_features: dict[str, int] = field(default_factory=dict)  # feature type -> count dropped

    def to_dict(self) -> dict:
        return {"n_proteins": self.n_proteins, "names": self.names, "original": self.original,
                "controls": self.controls, "other_features": self.other_features,
                "normalisation": "CLR per cell (Seurat margin = 2)"}


def display_name(feature: str) -> str:
    """'CD3_TotalSeqB' -> 'CD3'; a name without a known suffix is kept."""
    return _SUFFIX.sub("", feature).strip() or feature


def is_control(name: str) -> bool:
    return bool(_CONTROL.search(name))


def clr(counts) -> np.ndarray:
    """Centred log-ratio per cell (rows = cells): Seurat's CLR with margin = 2."""
    x = counts.toarray() if sp.issparse(counts) else np.asarray(counts)
    x = np.asarray(x, dtype=np.float64)
    if x.shape[1] == 0:
        return x.astype(np.float32)
    denom = np.exp(np.log1p(x).sum(axis=1, keepdims=True) / x.shape[1])
    return np.log1p(x / denom).astype(np.float32)


def _unique(names: list[str]) -> list[str]:
    seen: dict[str, int] = {}
    out = []
    for n in names:
        k = seen.get(n, 0)
        out.append(n if k == 0 else f"{n}-{k}")
        seen[n] = k + 1
    return out


def attach(adata: ad.AnnData, counts, features: list[str], split: ProteinSplit) -> None:
    """Store ADT ``counts`` (cells x ``features``) as the protein modality."""
    shown = _unique([display_name(f) for f in features])
    split.n_proteins = len(shown)
    split.names = shown
    split.original = {s: f for s, f in zip(shown, features) if s != f}
    split.controls = [s for s, f in zip(shown, features) if is_control(f)]
    dense = counts.toarray() if sp.issparse(counts) else np.asarray(counts)
    adata.obsm[COUNTS_KEY] = pd.DataFrame(dense.astype(np.float32), index=adata.obs_names, columns=shown)
    adata.obsm[CLR_KEY] = pd.DataFrame(clr(dense), index=adata.obs_names, columns=shown)
    adata.uns[UNS_KEY] = split.to_dict()


def split_features(adata: ad.AnnData) -> tuple[ad.AnnData, ProteinSplit | None]:
    """Take antibody (and other non-gene) features out of the gene matrix.

    With a ``var["feature_types"]`` column that has "Gene Expression" and any
    other type, returns the genes only, with the antibodies in obsm (see the
    module docstring). Without antibodies, returns ``adata`` unchanged and
    None; an obsm protein matrix from another tool (``protein_expression``) is
    adopted under scView's keys instead.
    """
    types = adata.var.get("feature_types")
    if types is None or not (types.astype(str) == GENE_TYPE).any():
        return adata, adopt_obsm(adata)
    types = types.astype(str)
    if (types == GENE_TYPE).all():
        return adata, adopt_obsm(adata)

    from scview.core import crispr, hashing

    split = ProteinSplit()
    is_ab = (types == ANTIBODY_TYPE).to_numpy()
    # CellPlex multiplexing tags are kept as tag counts (core/hashing.py), not dropped;
    # CRISPR guides as guide counts (core/crispr.py).
    is_cmo = (types == hashing.CMO_TYPE).to_numpy()
    is_guide = (types == crispr.GUIDE_TYPE).to_numpy()
    take_guide = 0 < is_guide.sum() <= crispr.MAX_GUIDES
    other = types[(types != GENE_TYPE) & (types != ANTIBODY_TYPE) & ~is_cmo & ~(is_guide & take_guide)]
    split.other_features = {str(k): int(v) for k, v in other.value_counts().items()}
    genes = adata[:, (types == GENE_TYPE).to_numpy()].copy()
    if 0 < is_ab.sum() <= MAX_PROTEINS:
        attach(genes, adata.X[:, is_ab], [str(v) for v in adata.var_names[is_ab]], split)
    elif split.other_features or is_ab.any():
        genes.uns[UNS_KEY] = split.to_dict()
    if 0 < is_cmo.sum() <= hashing.MAX_TAGS:
        hashing.attach_counts(genes, adata.X[:, is_cmo], [str(v) for v in adata.var_names[is_cmo]],
                              hashing.CMO_TYPE)
    if take_guide:
        crispr.attach_counts(genes, adata.X[:, is_guide], [str(v) for v in adata.var_names[is_guide]])
    return genes, (split if split.n_proteins or split.other_features else None)


def adopt_obsm(adata: ad.AnnData) -> ProteinSplit | None:
    """A protein matrix another tool left in obsm (totalVI's
    ``protein_expression``): copied under scView's keys, CLR added. A dataset
    that already has scView's keys gets the CLR (re)made only if missing."""
    if COUNTS_KEY in adata.obsm:
        if CLR_KEY not in adata.obsm:
            m = adata.obsm[COUNTS_KEY]
            adata.obsm[CLR_KEY] = pd.DataFrame(clr(np.asarray(m)), index=adata.obs_names,
                                               columns=list(m.columns))
        return None
    for key in FOREIGN_KEYS:
        m = adata.obsm.get(key)
        if isinstance(m, pd.DataFrame) and 0 < m.shape[1] <= MAX_PROTEINS:
            split = ProteinSplit()
            values = m.to_numpy(dtype=np.float64)
            if np.nanmin(values) < 0 or not np.allclose(values, np.round(values)):
                return None  # not counts (already normalised): leave it alone
            attach(adata, values, [str(c) for c in m.columns], split)
            return split
    return None


def has_protein(adata: ad.AnnData) -> bool:
    m = adata.obsm.get(CLR_KEY) if hasattr(adata, "obsm") else None
    return isinstance(m, pd.DataFrame) and m.shape[1] > 0


def protein_names(adata: ad.AnnData) -> list[str]:
    return [str(c) for c in adata.obsm[CLR_KEY].columns] if has_protein(adata) else []


def values(adata: ad.AnnData, name: str, normalised: bool = True) -> np.ndarray:
    """One protein's per-cell values (CLR, or raw counts): float32, cells order."""
    key = CLR_KEY if normalised else COUNTS_KEY
    m = adata.obsm.get(key)
    if not isinstance(m, pd.DataFrame) or name not in m.columns:
        raise KeyError(name)
    return m[name].to_numpy(dtype=np.float32)


def common_panel(adatas: list[ad.AnnData]) -> list[str]:
    """For a merge: keep the proteins every dataset measured (counts only; the
    CLR is made again on the merged panel) and drop the modality from all of
    them when one has none. Returns the shared panel ([] when dropped)."""
    if not adatas or not all(COUNTS_KEY in a.obsm for a in adatas):
        for a in adatas:
            for key in (COUNTS_KEY, CLR_KEY):
                a.obsm.pop(key, None)
        return []
    shared = [str(c) for c in adatas[0].obsm[COUNTS_KEY].columns
              if all(c in a.obsm[COUNTS_KEY].columns for a in adatas[1:])]
    for a in adatas:
        a.obsm[COUNTS_KEY] = a.obsm[COUNTS_KEY][shared]
        a.obsm.pop(CLR_KEY, None)
    return shared


def renormalise(adata: ad.AnnData) -> None:
    """The CLR again from the counts (after a merge changed the panel)."""
    m = adata.obsm.get(COUNTS_KEY)
    if isinstance(m, pd.DataFrame) and m.shape[1] > 0:
        adata.obsm[CLR_KEY] = pd.DataFrame(clr(m.to_numpy()), index=adata.obs_names, columns=list(m.columns))
    else:
        adata.obsm.pop(COUNTS_KEY, None)
