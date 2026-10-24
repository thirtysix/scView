"""A large dataset (``core/large_layout.py``) made from a small AnnData, for tests.

``make_large`` writes what a chunked run leaves in ``derived/<id>/``: the h5ad with X =
log-normalised CSR and no .raw or layers (``chunked.write_h5ad``), the counts sidecar
(``chunked.write_counts`` + ``large_layout.stamp_counts``), the gene store of X
(``gene_store.build``) and ``uns['scview_large']`` (``large_layout.record``). It uses the
chunked writers themselves, so the tests never depend on the pipeline that makes the
layout for real. ``small_twin`` is the same dataset as the in-memory pipeline would keep
it (X log-normalised, counts in ``layers['counts']``), to compare views against.
"""

from __future__ import annotations

import json
from pathlib import Path

import anndata as ad
import h5py
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp

from scview.core import chunked, gene_store, large_layout


def counts_adata(n_cells: int = 600, n_genes: int = 300, seed: int = 0) -> ad.AnnData:
    """Synthetic counts in three groups (each group's own up-regulated genes), MT- genes,
    QC columns as the pipeline's qc step writes them, a UMAP, leiden labels and donors."""
    rng = np.random.default_rng(seed)
    groups = rng.integers(0, 3, n_cells)
    base = rng.gamma(0.4, 1.0, n_genes)
    rate = np.tile(base, (n_cells, 1))
    for g in range(3):
        rate[groups == g, g * 20:(g + 1) * 20] *= 6.0
    counts = rng.poisson(rate * rng.uniform(0.5, 2.0, (n_cells, 1))).astype(np.float32)
    counts[0, :] = 0  # a cell with no counts
    counts[0, 5] = 1
    X = sp.csr_matrix(counts)
    names = [f"MT-{i}" for i in range(5)] + [f"G{i}" for i in range(5, n_genes)]
    obs = pd.DataFrame(
        {"leiden": pd.Categorical([str(g) for g in groups]),
         "donor": pd.Categorical([f"d{i % 4}" for i in range(n_cells)]),
         "condition": pd.Categorical(["ctrl" if i % 4 < 2 else "stim" for i in range(n_cells)])},
        index=[f"cell{i}" for i in range(n_cells)],
    )
    adata = ad.AnnData(X=X, obs=obs, var=pd.DataFrame(index=names))
    adata.var["mt"] = adata.var_names.str.startswith("MT-")
    sc.pp.calculate_qc_metrics(adata, qc_vars=["mt"], percent_top=None, log1p=False, inplace=True)
    adata.obsm["X_umap"] = rng.normal(size=(n_cells, 2)).astype(np.float32)
    adata.obsm["X_pca"] = rng.normal(size=(n_cells, 10)).astype(np.float32)
    adata.uns["scview_active_clustering"] = "leiden"
    return adata


def make_large(folder: str | Path, counts: ad.AnnData | None = None, *, stem: str = "data",
               rows: int = 128, gene_store_too: bool = True) -> Path:
    """Write the large layout of ``counts`` (an AnnData with counts in X) into ``folder``;
    returns the h5ad's path."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    counts = counts if counts is not None else counts_adata()
    src = folder / f".{stem}.counts-src.h5ad"
    ad.AnnData(X=sp.csr_matrix(counts.X), obs=counts.obs[[]], var=counts.var[[]]).write_h5ad(src)
    try:
        Xc = chunked.read_lazy(src, "X", chunk_rows=rows)
        h5ad = folder / f"{stem}.h5ad"
        uns = dict(counts.uns)
        large_layout.record(uns, h5ad)
        chunked.write_h5ad(h5ad, chunked.log_normalize(Xc), obs=counts.obs, var=counts.var,
                           obsm=dict(counts.obsm), obsp=dict(counts.obsp), uns=uns)
        counts_path, genes_path = large_layout.sidecar_paths(h5ad)
        chunked.write_counts(counts_path, Xc)
        large_layout.stamp_counts(counts_path, counts.obs_names, counts.var_names)
    finally:
        src.unlink(missing_ok=True)
    if gene_store_too:
        with h5py.File(h5ad, "r") as f:
            gene_store.build(f["X"], genes_path, obs_names=counts.obs_names,
                             var_names=counts.var_names, source="X")
    return h5ad


def small_twin(counts: ad.AnnData) -> ad.AnnData:
    """The same dataset as the in-memory pipeline keeps it: X log-normalised (float32),
    the counts in ``layers['counts']``."""
    adata = counts.copy()
    adata.layers["counts"] = adata.X.copy()
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    adata.X = sp.csr_matrix(adata.X, dtype=np.float32)
    return adata


def register(data_dir: str | Path, dataset_id: str, counts: ad.AnnData | None = None,
             *, where: str = "derived", **kwargs) -> Path:
    """A large dataset under ``data_dir`` as the library lists it: the files in
    ``<where>/<id>/`` and a metadata.json in ``uploads/<id>/``."""
    data_dir = Path(data_dir)
    h5ad = make_large(data_dir / where / dataset_id, counts, **kwargs)
    up = data_dir / "uploads" / dataset_id
    up.mkdir(parents=True, exist_ok=True)
    (up / "metadata.json").write_text(json.dumps(
        {"id": dataset_id, "name": f"large {dataset_id}", "status": "ready"}))
    return h5ad
