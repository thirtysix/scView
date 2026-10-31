"""Synthetic spliced and unspliced counts and the files they arrive in, for the velocity tests."""

from __future__ import annotations

import gzip
from pathlib import Path

import anndata as ad
import h5py
import numpy as np
import pandas as pd
import scipy.io
import scipy.sparse as sp

from scview.core import velocity_input as vi

BASES = "ACGT"


def barcode(i: int) -> str:
    return "".join(BASES[(i >> (2 * k)) & 3] for k in range(10)) + "AAAAAA"


def splicing_counts(n_cells: int = 300, n_genes: int = 260, seed: int = 0) -> ad.AnnData:
    """Cells along a time line t in [0, 1]. Each dynamic gene switches on at its own time;
    unspliced leads spliced, so cells ahead of the switch have a positive velocity. The UMAP's
    first axis is t, the second noise, and arrows should mostly point along +x."""
    rng = np.random.default_rng(seed)
    t = np.sort(rng.uniform(0, 1, n_cells))
    n_dynamic = n_genes - 40
    onset = rng.uniform(0, 0.8, n_dynamic)
    beta = 1.0
    gamma = rng.uniform(0.2, 0.8, n_dynamic)
    alpha = rng.uniform(4, 16, n_dynamic)
    tau = np.maximum(t[:, None] - onset[None, :], 0) * 8
    u = alpha / beta * (1 - np.exp(-beta * tau))
    s = alpha / gamma * (1 - np.exp(-gamma * tau)) + alpha * (np.exp(-beta * tau) - np.exp(-gamma * tau)) / (gamma - beta)
    depth = rng.uniform(0.7, 1.3, (n_cells, 1))
    spliced = rng.poisson(np.maximum(s, 0) * depth + 0.05)
    unspliced = rng.poisson(np.maximum(u, 0) * depth + 0.05)
    # Housekeeping genes keep the totals up; rare genes fail the shared-counts filter.
    spliced = np.c_[spliced, rng.poisson(5.0, (n_cells, 30)), np.zeros((n_cells, 10), dtype=int)]
    unspliced = np.c_[unspliced, rng.poisson(1.0, (n_cells, 30)), np.zeros((n_cells, 10), dtype=int)]
    names = [f"G{i}" for i in range(n_genes)]
    obs = pd.DataFrame({"clusters": pd.Categorical(np.where(t < 0.5, "early", "late"))},
                       index=[barcode(i) + "-1" for i in range(n_cells)])
    a = ad.AnnData(sp.csr_matrix(spliced, dtype=np.float32), obs=obs,
                   var=pd.DataFrame({"gene_ids": [f"ENSG{i:05d}.1" for i in range(n_genes)]}, index=names))
    a.layers["spliced"] = sp.csr_matrix(spliced, dtype=np.float32)
    a.layers["unspliced"] = sp.csr_matrix(unspliced, dtype=np.float32)
    a.layers["ambiguous"] = sp.csr_matrix(rng.poisson(0.2, spliced.shape), dtype=np.float32)
    a.obsm["X_umap"] = np.c_[t * 10, rng.normal(0, 0.3, n_cells)].astype(np.float32)
    return a


class Adaptor:
    def __init__(self, adata):
        self.adata = adata
        self.is_large = False

    def cells_fingerprint(self):
        return "fp"

    def active_clustering_column(self):
        return "clusters"


def legacy_h5ad(path: Path, a: ad.AnnData) -> None:
    """A pre-0.7 anndata layout, as scvelo distributes pancreas: compound obs and var, h5sparse."""
    def write_matrix(group, name, m):
        g = group.create_group(name)
        g.attrs["h5sparse_format"] = "csr"
        g.attrs["h5sparse_shape"] = np.asarray(m.shape)
        g["data"], g["indices"], g["indptr"] = m.data, m.indices, m.indptr

    with h5py.File(path, "w") as f:
        obs = np.zeros(a.n_obs, dtype=[("index", "S32"), ("clusters", "i1")])
        obs["index"] = [n.encode() for n in a.obs_names]
        obs["clusters"] = a.obs["clusters"].cat.codes.to_numpy()
        f["obs"] = obs
        var = np.zeros(a.n_vars, dtype=[("index", "S16")])
        var["index"] = [n.encode() for n in a.var_names]
        f["var"] = var
        write_matrix(f, "X", sp.csr_matrix(a.X))
        layers = f.create_group("layers")
        for k in ("spliced", "unspliced"):
            write_matrix(layers, k, sp.csr_matrix(a.layers[k]))
        uns = f.create_group("uns")
        uns["clusters_categories"] = np.asarray(a.obs["clusters"].cat.categories, dtype="S8")
        f.create_group("obsm")["X_umap"] = a.obsm["X_umap"]


def write_loom(path: Path, a: ad.AnnData, sample="possorted_genome_bam_ABC12") -> None:
    """velocyto's layout: genes x cells, CellID sample:BARCODEx, Gene, Accession, main matrix = spliced."""
    with h5py.File(path, "w") as f:
        f["matrix"] = a.layers["spliced"].T.toarray()
        layers = f.create_group("layers")
        for k in vi.LAYERS:
            layers[k] = a.layers[k].T.toarray()
        cols, rows = f.create_group("col_attrs"), f.create_group("row_attrs")
        cols["CellID"] = np.asarray([f"{sample}:{n.split('-')[0]}x".encode() for n in a.obs_names])
        rows["Gene"] = np.asarray([g.encode() for g in a.var_names])
        rows["Accession"] = np.asarray([g.encode() for g in a.var["gene_ids"]])


def write_starsolo(folder: Path, a: ad.AnnData, zipped=False) -> list[Path]:
    folder.mkdir(parents=True, exist_ok=True)
    out = []
    ext = ".gz" if zipped else ""
    op = gzip.open if zipped else open

    def text(name: str, lines: list[str]) -> None:
        with op(folder / f"{name}{ext}", "wt") as fh:
            fh.write("\n".join(lines) + "\n")
        out.append(folder / f"{name}{ext}")

    text("barcodes.tsv", [n.split("-")[0] for n in a.obs_names])
    text("features.tsv", [f"{i}\t{g}\tGene Expression" for i, g in zip(a.var["gene_ids"], a.var_names)])
    for k in vi.LAYERS:
        scipy.io.mmwrite(folder / f"{k}.mtx", a.layers[k].T.tocoo(), field="integer")
        if zipped:
            with open(folder / f"{k}.mtx", "rb") as fi, gzip.open(folder / f"{k}.mtx.gz", "wb") as fo:
                fo.write(fi.read())
            (folder / f"{k}.mtx").unlink()
        out.append(folder / f"{k}.mtx{ext}")
    return out
