"""make_export_fixtures.py <out_dir>: staging h5ad files as the backend writes them for
an .rds export (core/rds_export.py), and export_expected.json computed here from the same
arrays, never from the converter.

export_full.h5ad    X log-normalised (CSR), layers/counts (CSC, so the reader converts),
                    obs of every column kind, var, obsm X_pca/X_umap/X_umap_3d, varm PCs,
                    antibody counts + CLR, gene names with '_'
export_counts.h5ad  counts only (dense), no X: a dataset that is not normalised yet
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import anndata as ad
import h5py
import numpy as np
import pandas as pd
import scipy.sparse as sp

try:
    from anndata.io import write_elem
except ImportError:
    from anndata.experimental import write_elem

out = Path(sys.argv[1])
out.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(7)
n, g = 50, 30

counts = rng.poisson(0.8, (n, g)).astype(np.float32)
counts[:, 0] = 0  # a gene with no counts
logn = np.log1p(counts / np.maximum(counts.sum(1, keepdims=True), 1) * 1e4)
cells = [f"AAAC{i:03d}-1" for i in range(n)]
genes = [f"Gene_{i}" if i < 4 else f"G{i}" for i in range(g)]
genes[5] = "Gene-1"  # becomes a duplicate of "Gene_1" once '_' -> '-'

obs = pd.DataFrame(index=cells)
obs["leiden"] = pd.Categorical([str(i % 3) for i in range(n)], categories=["2", "0", "1"])
stage = pd.Categorical(["late" if i % 2 else "early" for i in range(n)], categories=["early", "late"],
                       ordered=True)
obs["stage"] = stage
donor = pd.Categorical([f"d{i % 4}" for i in range(n)])
donor[3] = np.nan
obs["donor"] = donor
obs["is_stim"] = (np.arange(n) % 2) == 0
obs["n_genes"] = (counts > 0).sum(1).astype(np.int64)
score = rng.normal(size=n)
score[4] = np.nan
obs["score"] = score
var = pd.DataFrame(index=genes)
var["highly_variable"] = np.arange(g) < 12
var["means"] = logn.mean(0)

pca = rng.normal(size=(n, 5))
pcs = np.zeros((g, 5))
pcs[:12] = rng.normal(size=(12, 5))
protein = rng.poisson(6, (n, 3)).astype(np.float64)
clr = np.log1p(protein / np.exp(np.log1p(protein).sum(1, keepdims=True) / 3))
meta = {
    "format": "seurat", "dataset_name": "Fixture set", "dataset_id": "fx", "scview_version": "0.0.test",
    "exported_at": "2026-10-26T00:00:00+00:00", "x_from": "X", "counts_from": "layers['counts']",
    "cluster_column": "leiden", "variable_genes": genes[:12],
    "colors": {"leiden_colors": ["#ff0000", "#00ff00", "#0000ff"]}, "recipe": ["normalize", "pca"],
    "provenance": json.dumps({"schema_version": 1}), "protein_names": ["CD3_TotalSeqB", "CD4", "CD19"],
    "left_out": ["obsp: connectivities - neighbour graphs"], "notes": [],
    "pca_variance": [5.0, 4.0, 3.0, 2.0, 1.0], "pca_variance_ratio": [0.3, 0.2, 0.1, 0.05, 0.01],
}

full = out / "export_full.h5ad"
ad.AnnData(obs=obs, var=var, obsm={"X_pca": pca, "X_umap": rng.normal(size=(n, 2)),
                                   "X_umap_3d": rng.normal(size=(n, 3)).astype(np.float32)},
           varm={"PCs": pcs},
           uns={"scview_rexport": json.dumps(meta), "protein_counts": protein, "protein_clr": clr}
           ).write_h5ad(full)
with h5py.File(full, "r+") as f:
    if "X" in f:
        del f["X"]
    write_elem(f, "X", sp.csr_matrix(logn))
    layers = f.require_group("layers")
    write_elem(layers, "counts", sp.csc_matrix(counts))

meta2 = dict(meta, x_from=None, counts_from="X", cluster_column=None, variable_genes=[],
             protein_names=[], left_out=[], pca_variance=[], pca_variance_ratio=[])
small = out / "export_counts.h5ad"
ad.AnnData(obs=obs[["leiden", "is_stim"]], var=pd.DataFrame(index=[f"g{i}" for i in range(g)]),
           uns={"scview_rexport": json.dumps(meta2)}).write_h5ad(small)
with h5py.File(small, "r+") as f:
    if "X" in f:
        del f["X"]
    write_elem(f.require_group("layers"), "counts", counts)  # dense

expected = {
    "n_cells": n, "n_genes": g,
    "counts_sum": float(counts.sum(dtype=np.float64)), "counts_nnz": int((counts != 0).sum()),
    "logn_sum": float(logn.sum(dtype=np.float64)),
    "count_00_11": float(counts[11, 0]), "count_cell3_gene7": float(counts[3, 7]),
    "leiden_levels": ["2", "0", "1"], "leiden_first": [str(i % 3) for i in range(5)],
    "stage_levels": ["early", "late"], "donor_na": 3,
    "n_genes_first": [int(v) for v in obs["n_genes"].iloc[:5]], "score_na": 4,
    "is_stim_first": [bool(v) for v in obs["is_stim"].iloc[:5]],
    "pca_00": float(pca[0, 0]), "pca_cell7_dim3": float(pca[7, 3]),
    # Gene_0..Gene_3 -> Gene-0..Gene-3, and the existing Gene-1 made unique (Gene-1.1)
    "n_renamed": 5, "gene_5_seurat": "Gene-1.1",
    "protein_sum": float(protein.sum()), "clr_sum": float(clr.sum()),
}
(out / "export_expected.json").write_text(json.dumps(expected))
print(f"[export fixtures] wrote {full.name}, {small.name}")
