"""Tiny 10x Multiome files for the ATAC tests: a v3 .h5, MEX, .h5ad and .h5mu of one matrix.

Two groups of cells with different genes and different open peaks. Peaks are placed around
a small gene table (``GTF``): each group's genes have their own accessible peaks, so gene
activity, LSI and differential accessibility have an answer to find."""

from __future__ import annotations

import gzip
from pathlib import Path

import anndata as ad
import h5py
import numpy as np
import pandas as pd
import scipy.sparse as sp
from anndata.io import write_elem

# Gene bodies (GTF: 1-based inclusive). GENEA and GENEB mark group A, GENEC and GENED group B.
GENES = [("GENEA", "ENSG00000000001", "chr1", 10_001, 20_000, "+"),
         ("GENEB", "ENSG00000000002", "chr1", 50_001, 60_000, "-"),
         ("GENEC", "ENSG00000000003", "chr2", 10_001, 20_000, "+"),
         ("GENED", "ENSG00000000004", "chr2", 50_001, 60_000, "-")]


def gtf_text() -> str:
    lines = ["##description: tiny"]
    for name, gid, chrom, start, end, strand in GENES:
        lines.append(f'{chrom}\ttest\tgene\t{start}\t{end}\t.\t{strand}\t.\t'
                     f'gene_id "{gid}.1"; gene_type "protein_coding"; gene_name "{name}";')
    return "\n".join(lines) + "\n"


def peak_table() -> pd.DataFrame:
    """Peaks: promoter (2 kb upstream side), gene body, distal, for each gene; then
    background peaks every 100 kb on chr3 (open in both groups at random)."""
    rows = []
    for name, _gid, chrom, start, end, strand in GENES:
        s0 = start - 1
        up = s0 - 1_000 if strand == "+" else end + 500  # within 2 kb upstream
        rows.append((chrom, up, up + 500, name, "promoter"))
        rows.append((chrom, s0 + 3_000, s0 + 3_500, name, "body"))
        rows.append((chrom, (end + 15_000) if strand == "+" else s0 - 13_000,
                     (end + 15_500) if strand == "+" else s0 - 12_500, name, "distal"))
    for i in range(60):
        rows.append(("chr3", 100_000 * (i + 1), 100_000 * (i + 1) + 500, "", "background"))
    df = pd.DataFrame(rows, columns=["chrom", "start", "end", "gene", "kind"])
    df["name"] = [f"{c}:{s}-{e}" for c, s, e in zip(df["chrom"], df["start"], df["end"])]
    return df


def matrices(n_per: int = 40, seed: int = 0) -> tuple[sp.csr_matrix, sp.csr_matrix, list[str], pd.DataFrame]:
    """(gene counts, peak counts, barcodes, peak table): cells x features, group A then B."""
    rng = np.random.default_rng(seed)
    n = 2 * n_per
    genes = rng.poisson(0.2, (n, len(GENES))).astype(np.float32)
    genes[:n_per, :2] += rng.poisson(8, (n_per, 2))
    genes[n_per:, 2:] += rng.poisson(8, (n_per, 2))
    table = peak_table()
    peaks = rng.poisson(0.5, (n, len(table))).astype(np.float32)
    marked = table["gene"].isin(["GENEA", "GENEB"]).to_numpy()
    other = table["gene"].isin(["GENEC", "GENED"]).to_numpy()
    peaks[:, marked | other] = rng.poisson(0.1, (n, int((marked | other).sum())))
    peaks[np.ix_(range(n_per), np.flatnonzero(marked))] += rng.poisson(6, (n_per, int(marked.sum())))
    peaks[np.ix_(range(n_per, n), np.flatnonzero(other))] += rng.poisson(6, (n_per, int(other.sum())))
    # Depth differs by cell: the first LSI component should follow it.
    depth = rng.uniform(0.5, 2.0, n)[:, None]
    peaks = np.round(peaks * depth)
    barcodes = ["".join("ACGT"[(i >> (2 * k)) & 3] for k in range(16)) + "-1" for i in range(n)]
    return sp.csr_matrix(genes), sp.csr_matrix(peaks), barcodes, table


def write_tenx_h5(path: str | Path, n_per: int = 40, seed: int = 0, *, genome: str = "GRCh38",
                  peaks_only: bool = False, extra_peak: bool = False) -> Path:
    """A Cell Ranger ARC style .h5: genes first, then peaks (``peaks_only``: no genes)."""
    g, p, barcodes, table = matrices(n_per, seed)
    names = [x[0] for x in GENES]
    ids = [x[1] for x in GENES]
    intervals = [f"{x[2]}:{x[3] - 1}-{x[3]}" for x in GENES]
    peaks = table["name"].tolist()
    if extra_peak:
        peaks = peaks[:-1] + ["chr4:1-500"]
    if peaks_only:
        full, names, ids, intervals, types = p, [], [], [], []
    else:
        full, types = sp.hstack([g, p]).tocsr(), ["Gene Expression"] * len(names)
    all_names = names + peaks
    all_ids = ids + peaks
    all_intervals = intervals + peaks
    types = types + ["Peaks"] * len(peaks)
    csc = sp.csr_matrix(full)  # cells x features: the file's genes x cells CSC arrays
    csc.sort_indices()
    with h5py.File(path, "w") as f:
        m = f.create_group("matrix")
        m["barcodes"] = np.array(barcodes, dtype="S")
        m["data"] = csc.data.astype(np.int32)
        m["indices"] = csc.indices.astype(np.int64)
        m["indptr"] = csc.indptr.astype(np.int64)
        m["shape"] = np.array([full.shape[1], full.shape[0]], dtype=np.int32)
        fe = m.create_group("features")
        fe["_all_tag_keys"] = np.array([b"genome", b"interval"])
        fe["feature_type"] = np.array(types, dtype="S")
        fe["genome"] = np.array([genome] * len(types), dtype="S")
        fe["id"] = np.array(all_ids, dtype="S")
        fe["interval"] = np.array(all_intervals, dtype="S")
        fe["name"] = np.array(all_names, dtype="S")
    return Path(path)


def write_mex(folder: str | Path, n_per: int = 40, seed: int = 0) -> Path:
    """filtered_feature_bc_matrix/ with Cell Ranger ARC's six-column features.tsv.gz."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    g, p, barcodes, table = matrices(n_per, seed)
    full = sp.hstack([g, p]).tocoo()
    order = np.lexsort((full.col, full.row))
    with gzip.open(folder / "matrix.mtx.gz", "wt") as fh:
        fh.write("%%MatrixMarket matrix coordinate integer general\n")
        fh.write(f"{full.shape[1]} {full.shape[0]} {full.nnz}\n")
        for i in order:
            fh.write(f"{full.col[i] + 1} {full.row[i] + 1} {int(full.data[i])}\n")
    with gzip.open(folder / "barcodes.tsv.gz", "wt") as fh:
        fh.write("\n".join(barcodes) + "\n")
    with gzip.open(folder / "features.tsv.gz", "wt") as fh:
        for name, gid, chrom, start, end, _strand in GENES:
            fh.write(f"{gid}\t{name}\tGene Expression\t{chrom}\t{start - 1}\t{start}\n")
        for _, r in table.iterrows():
            fh.write(f"{r['name']}\t{r['name']}\tPeaks\t{r['chrom']}\t{r['start']}\t{r['end']}\n")
    return folder


def as_anndata(n_per: int = 40, seed: int = 0) -> ad.AnnData:
    """The whole matrix as an AnnData with ``feature_types``."""
    g, p, barcodes, table = matrices(n_per, seed)
    var = pd.DataFrame({"gene_ids": [x[1] for x in GENES] + table["name"].tolist(),
                        "feature_types": ["Gene Expression"] * len(GENES) + ["Peaks"] * len(table),
                        "genome": ["GRCh38"] * (len(GENES) + len(table))},
                       index=[x[0] for x in GENES] + table["name"].tolist())
    return ad.AnnData(X=sp.hstack([g, p]).tocsr().astype(np.float32), obs=pd.DataFrame(index=barcodes), var=var)


def write_h5mu(path: str | Path, n_per: int = 40, seed: int = 0, *, atac_x: str = "counts") -> Path:
    """MuData on disk, by hand: mod/rna + mod/atac (no mudata dependency). ``atac_x``:
    "counts" keeps counts in X; "tfidf" puts other values in X and counts in layers/counts;
    "tfidf_only" has no counts at all."""
    g, p, barcodes, table = matrices(n_per, seed)
    obs = pd.DataFrame(index=barcodes)
    rna = ad.AnnData(X=g, obs=obs.copy(), var=pd.DataFrame(
        {"gene_ids": [x[1] for x in GENES], "feature_types": "Gene Expression"}, index=[x[0] for x in GENES]))
    atac = ad.AnnData(X=p, obs=obs.copy(), var=pd.DataFrame(
        {"feature_types": "Peaks", "genome": "GRCh38", "interval": table["name"].to_numpy()},
        index=table["name"].to_numpy()))
    if atac_x != "counts":
        if atac_x == "tfidf":
            atac.layers["counts"] = p.copy()
        atac.X = p.multiply(0.37).tocsr()
    # As mudata writes it: the version in a 512-byte user block before the HDF5 signature.
    with h5py.File(path, "w", userblock_size=512) as f:
        f.attrs["encoding-type"] = "MuData"
        f.attrs["encoding-version"] = "0.1.0"
        write_elem(f, "obs", obs)
        write_elem(f, "var", pd.DataFrame(index=pd.Index(rna.var_names.tolist() + atac.var_names.tolist())))
        mods = f.create_group("mod")
        write_elem(mods, "rna", rna)
        write_elem(mods, "atac", atac)
    with open(path, "r+b") as fh:
        fh.write(b"MuData (format-version=0.1.0;creator=muon;creator-version=0.1.7)")
    return Path(path)
