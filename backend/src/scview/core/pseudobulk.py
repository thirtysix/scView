"""Pseudobulk differential expression across donors (DESeq2 via pydeseq2).

A cell-level test treats every cell as an independent replicate, so when two
conditions also differ by donor its p-values are far too small. Here each
donor's raw counts are summed per condition into one pseudobulk sample, and
DESeq2's negative-binomial Wald test compares those samples: the replicate is
the donor, not the cell.

When every donor contributes a sample to both conditions (the usual
stimulated-vs-control layout), the design is paired, ``~donor + condition``, so
donor-to-donor baseline differences are absorbed; otherwise it is ``~condition``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import scipy.sparse as sp

logger = logging.getLogger(__name__)

# A donor x condition sample needs this many cells to be a usable sum.
MIN_CELLS_PER_SAMPLE = 10
# Pseudobulk samples (donors) required in each condition.
MIN_REPLICATES = 2
# A gene is tested when it has at least this many counts in as many samples as
# the smaller condition has (the edgeR filterByExpr idea, simplified).
MIN_GENE_COUNT = 10
# Compute bounds: DESeq2 fits one model per gene, so cost is samples x genes.
MAX_SAMPLES = 300
MAX_GENES = 20_000
# Worker processes for pydeseq2's per-gene fits. One keeps it in-process, which
# also keeps a shared laptop or a one-worker server from being saturated.
N_CPUS = 1


class PseudobulkError(ValueError):
    """A request pseudobulk DE cannot serve (no counts, too few donors): a 400."""


@dataclass
class PseudobulkResult:
    genes: list[tuple[str, float, float, float]]  # (gene, log2FC, p, adjusted p)
    n_samples_a: int
    n_samples_b: int
    n_cells_a: int
    n_cells_b: int
    n_genes_tested: int
    design: str
    paired: bool
    counts_source: str
    dropped_samples: list[str] = field(default_factory=list)
    stats: dict[str, float] = field(default_factory=dict)  # gene -> Wald statistic


def raw_counts(adata):
    """(matrix, gene names, where it came from) for the dataset's raw counts:
    ``layers['counts']``, else ``.raw``, else ``X``, whichever first holds counts."""
    from scview.core.large_mode import counts_of
    from scview.core.pipeline import looks_like_counts  # lazy: pipeline is heavy

    large = counts_of(adata)  # a large dataset's counts sidecar (row reads only)
    if large is not None:
        return large, adata.var_names, "the counts file"
    candidates = []
    if "counts" in adata.layers:
        candidates.append((adata.layers["counts"], adata.var_names, "layers['counts']"))
    if adata.raw is not None:
        candidates.append((adata.raw.X, adata.raw.var_names, ".raw"))
    candidates.append((adata.X, adata.var_names, "X"))
    for M, names, source in candidates:
        if looks_like_counts(M):
            return M, names, source
    raise PseudobulkError(
        "Pseudobulk DE sums raw counts, and this dataset has none (not in "
        "layers['counts'], .raw or X; X holds normalised values). Use the cell-level "
        "test, or re-load the dataset from a file that keeps its counts."
    )


def _sum_by_sample(M, codes: np.ndarray, n_samples: int) -> sp.csr_matrix:
    """Sum the rows of ``M`` (cells x genes) into ``n_samples`` rows by ``codes``."""
    S = sp.csr_matrix(
        (np.ones(len(codes)), (codes, np.arange(len(codes)))), shape=(n_samples, len(codes))
    )
    M = M if sp.issparse(M) else np.asarray(M)
    return sp.csr_matrix(S @ M)


def _filter_genes(P: sp.csr_matrix, min_samples: int) -> np.ndarray:
    """Column indices of the genes worth testing, capped at ``MAX_GENES`` by total count."""
    expressed = np.asarray((P >= MIN_GENE_COUNT).sum(axis=0)).ravel() >= min_samples
    idx = np.flatnonzero(expressed)
    if len(idx) > MAX_GENES:
        totals = np.asarray(P[:, idx].sum(axis=0)).ravel()
        idx = np.sort(idx[np.argsort(-totals, kind="stable")[:MAX_GENES]])
    return idx


def pseudobulk_de(
    adata,
    in_a: np.ndarray,
    donors: pd.Series,
    *,
    label_a: str = "A",
    label_b: str = "B",
    donor_key: str = "donor",
) -> PseudobulkResult:
    """DESeq2 Wald test of condition A vs B on per-donor pseudobulk sums.

    ``adata`` holds only the compared cells; ``in_a`` marks the A cells (the
    rest are B) and ``donors`` gives each cell's donor (missing = left out).
    """
    M, gene_names, source = raw_counts(adata)
    donor = donors.astype(str).where(donors.notna()).to_numpy()
    has_donor = pd.notna(donor)
    cond = np.where(in_a, "a", "b")

    # One sample per (donor, condition) present in the data.
    key = pd.Series(cond[has_donor]) + "\x1f" + pd.Series(donor[has_donor].astype(str))
    codes, samples = pd.factorize(key, sort=True)
    n_cells = np.bincount(codes, minlength=len(samples))
    sample_cond = np.array([s.split("\x1f", 1)[0] for s in samples])
    sample_donor = np.array([s.split("\x1f", 1)[1] for s in samples])

    rows = np.flatnonzero(has_donor)
    P = _sum_by_sample(M[rows] if not has_donor.all() else M, codes, len(samples))

    ok = n_cells >= MIN_CELLS_PER_SAMPLE
    names = {"a": label_a, "b": label_b}
    dropped = [f"{sample_donor[i]} ({names[sample_cond[i]]}, {n_cells[i]} cells)"
               for i in np.flatnonzero(~ok)]
    n_a = int((ok & (sample_cond == "a")).sum())
    n_b = int((ok & (sample_cond == "b")).sum())
    if min(n_a, n_b) < MIN_REPLICATES:
        raise PseudobulkError(
            f"Pseudobulk DE needs at least {MIN_REPLICATES} donors in each group, each with "
            f"at least {MIN_CELLS_PER_SAMPLE} cells, in '{donor_key}' ({label_a}: {n_a}, "
            f"{label_b}: {n_b}). Pick another donor column, widen the cell scope, or use "
            "the cell-level test."
        )
    if n_a + n_b > MAX_SAMPLES:
        raise PseudobulkError(
            f"Too many pseudobulk samples ({n_a + n_b}; the limit is {MAX_SAMPLES}). "
            f"'{donor_key}' may be a per-cell column rather than a donor or sample."
        )

    P, sample_cond, sample_donor = P[ok], sample_cond[ok], sample_donor[ok]
    n_cells_kept = n_cells[ok]
    genes_idx = _filter_genes(P, min(n_a, n_b))
    if len(genes_idx) == 0:
        raise PseudobulkError(
            f"No gene has at least {MIN_GENE_COUNT} counts in {min(n_a, n_b)} or more "
            "pseudobulk samples; there is too little signal to test."
        )

    # Neutral labels for the model: formulaic and the contrast never see the
    # user's group or donor names (which may hold spaces or symbols).
    donor_codes = pd.factorize(sample_donor, sort=True)[0]
    index = [f"s{i}" for i in range(len(sample_cond))]
    counts_df = pd.DataFrame(
        P[:, genes_idx].toarray().round().astype(np.int64),
        index=index,
        columns=[str(g) for g in np.asarray(gene_names)[genes_idx]],
    )
    meta = pd.DataFrame(
        {"condition": sample_cond, "donor": [f"d{c}" for c in donor_codes]}, index=index
    )
    by_donor = meta.groupby("donor")["condition"].nunique()
    paired = bool((by_donor == 2).all())
    design = "~donor + condition" if paired else "~condition"

    genes, wald = _run_deseq2(counts_df, meta, design)
    return PseudobulkResult(
        genes=genes,
        n_samples_a=n_a,
        n_samples_b=n_b,
        n_cells_a=int(n_cells_kept[sample_cond == "a"].sum()),
        n_cells_b=int(n_cells_kept[sample_cond == "b"].sum()),
        n_genes_tested=len(genes_idx),
        design=design.replace("donor", donor_key),
        paired=paired,
        counts_source=source,
        dropped_samples=dropped,
        stats=wald,
    )


def _run_deseq2(
    counts_df: pd.DataFrame, meta: pd.DataFrame, design: str
) -> tuple[list[tuple[str, float, float, float]], dict[str, float]]:
    """(gene, log2FC, p, adjusted p) per tested gene, strongest up first, and each
    gene's Wald statistic."""
    from pydeseq2.dds import DeseqDataSet
    from pydeseq2.default_inference import DefaultInference
    from pydeseq2.ds import DeseqStats

    inference = DefaultInference(n_cpus=N_CPUS)
    dds = DeseqDataSet(
        counts=counts_df, metadata=meta, design=design, inference=inference, quiet=True
    )
    dds.deseq2()
    stats = DeseqStats(dds, contrast=["condition", "a", "b"], inference=inference, quiet=True)
    stats.summary()
    res = stats.results_df

    out: list[tuple[str, float, float, float]] = []
    for gene, lfc, p, padj in zip(
        res.index, res["log2FoldChange"], res["pvalue"], res["padj"]
    ):
        # NaN LFC/p: an all-zero or Cook's-outlier gene, not tested. NaN padj
        # alone: DESeq2's independent filtering left a low-mean gene out of the
        # FDR correction, which reads as "not significant".
        if not (np.isfinite(lfc) and np.isfinite(p)):
            continue
        out.append((str(gene), float(lfc), float(p), float(padj) if np.isfinite(padj) else 1.0))
    # Rank like the Wilcoxon results (scanpy's score order): strongest up first,
    # strongest down last. Callers take "the top genes" from the front of each
    # side, and DESeq2 returns genes in input (genome) order.
    stat = res["stat"].to_dict() if "stat" in res else {}

    def _rank(row: tuple[str, float, float, float]) -> float:
        s = stat.get(row[0])
        if s is not None and np.isfinite(s):
            return float(s)
        return float(np.sign(row[1]) * -np.log10(max(row[2], 1e-300)))

    out.sort(key=_rank, reverse=True)
    wald = {str(g): float(s) for g, s in stat.items() if s is not None and np.isfinite(s)}
    return out, wald
