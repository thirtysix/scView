"""Gene programs: non-negative matrix factorisation of the variable genes (N09).

Clusters put each cell in one box; an interferon response, the cell cycle or a
stress signature runs across the boxes. NMF writes each cell's expression of the
variable genes as a weighted sum of ``k`` non-negative gene programs: a program
is a set of gene weights (its spectrum), and each cell gets a usage of each
program. Consensus NMF (Kotliar et al. 2019, eLife 8:e43803, PMID 30839264)
repeats the factorisation from several random starts and keeps the programs
that come back every time.

The steps, following cNMF where it applies:

1. **Input.** The dataset's flagged variable genes (``var["highly_variable"]``;
   ``n_hvg`` keeps the top ones by the stored ranking), log-normalised: ``.raw``
   after scView's pipeline, else X when it holds log values, else the shifted
   log of the counts (log1p of counts per 10,000). Never the scaled X: z-scores
   are negative, and NMF needs non-negative input. Each gene is divided by its
   standard deviation, not centred (cNMF scales its counts the same way), so a
   highly expressed gene does not outweigh the rest; sparse stays sparse.
2. **Replicates.** sklearn ``NMF`` (Frobenius loss, coordinate descent, random
   initialisation, tol 1e-4, up to 1000 iterations) from ``seeds`` seeds 0, 1,
   ... Random starts, as cNMF uses, because the deterministic NNDSVD start
   gives the same answer from every seed, which would make any stability
   measure meaningless. Coordinate descent reached a lower error than
   multiplicative updates on Kang (relative error 0.7972 vs 0.7986 at k = 10),
   in 2.2 s per run on 2 cores.
3. **Consensus.** The ``seeds x k`` spectra, each scaled to unit length, are
   clustered into ``k`` groups by k-means (10 starts, seed 1); a program's
   consensus spectrum is the median of its group, scaled to sum 1. cNMF also
   drops outlying spectra by local density first; with a handful of replicates
   that filter has too little to work with, so it is left out.
4. **Usage.** The usages are refitted with the consensus spectra held fixed
   (non-negative least squares, as cNMF does). ``usage`` (obsm
   ``X_nmf_usage``) is each cell's usages scaled to sum 1: the share of the
   cell's modelled expression each program accounts for. The unscaled usages
   are kept as ``X_nmf_usage_raw``.
5. **Genes.** A program's gene score (cNMF's ``gene_spectra_score`` idea) is
   the least-squares coefficient of each gene's z-scored expression on the
   cells' usage shares: the z-score the gene is predicted to have in a cell
   made entirely of that program. The top genes are the highest scores.

Stability, per program: the mean Pearson correlation between the spectra
grouped into it (one per replicate when it is found every time), with the
replicates it was found in and its members' mean silhouette. A program is
**stable** when every replicate has a member in it and that correlation is at
least 0.9. For the whole factorisation: the silhouette of the k-means grouping
(cNMF's stability) and the relative reconstruction error, ``||X - WH||^2 /
||X||^2`` on the scaled input, both reported per k in a sweep.

What it cannot say: NMF finds the largest sources of variation in the variable
genes, so a program can be a cell type rather than a process, and two
processes that always co-occur share one program; k is a choice, not a fact.
Usage is relative within a cell; a cell low in every program's genes still gets
shares that sum to 1 (a cell with none of the genes gets 0 throughout).
"""

from __future__ import annotations

import json
import logging
import time
import warnings
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd
import scipy.sparse as sp

logger = logging.getLogger(__name__)

OBSM_KEY = "X_nmf_usage"  # usage shares, rows sum to 1
OBSM_RAW_KEY = "X_nmf_usage_raw"  # usages before scaling
UNS_KEY = "scview_programs"
ENRICH_KEY = "scview_programs_enrichment"  # ORA results by program, for the co-pilot
TAG = "result:programs"
VERSION = 1

DEFAULT_K = 10
MIN_K, MAX_K = 2, 30
DEFAULT_SEEDS = 5
MAX_SEEDS = 20
DEFAULT_SWEEP = (5, 10, 15, 20)
MAX_SWEEP = 8  # values of k in one sweep
MIN_GENES = 50
N_TOP = 30
STABLE_CORR = 0.9

SOLVER = "cd"
INIT = "random"
BETA_LOSS = "frobenius"
TOL = 1e-4
MAX_ITER = 1000
KMEANS_SEED = 1
KMEANS_STARTS = 10
THREADS = 2
TARGET_SUM = 1e4
_CHUNK = 10_000  # rows read at a time from a dense or on-disk matrix


class ProgramsError(Exception):
    """A request the dataset cannot satisfy, in a sentence for the user."""


# ---------------------------------------------------------------------------
# Input
# ---------------------------------------------------------------------------


@dataclass
class ProgramInput:
    """The matrix NMF runs on and where it came from."""

    X: sp.csr_matrix  # cells x genes, float32, each gene divided by its sd
    genes: list[str]
    source: str  # where the log values came from, in words
    n_flagged: int  # genes the dataset flags as variable
    n_dropped: int  # flagged genes with no variance in these cells
    n_empty_cells: int  # cells with none of the genes
    notes: list[str] = field(default_factory=list)


def _looks_like_counts(M) -> bool:
    from scview.core.pipeline import looks_like_counts

    return looks_like_counts(M)


def _sample_min(M) -> float | None:
    """The smallest value among the first stored values (sparse) or rows (dense)."""
    if sp.issparse(M):
        sample = np.asarray(M.data[:200_000])
    else:
        sample = np.asarray(M[: min(1000, M.shape[0])])
    return float(np.min(sample)) if sample.size else None


def _is_lognorm(M) -> bool:
    """Non-negative values that are not integer counts: log-normalised expression."""
    if M is None:
        return False
    low = _sample_min(M)
    return low is not None and low >= 0 and not _looks_like_counts(M)


def select_genes(adata, n_hvg: int | None = None) -> tuple[list[str], int, list[str]]:
    """The variable genes to factorise (var order), how many are flagged, and notes.

    ``n_hvg`` below the flagged count keeps the top ones by the stored ranking
    (``highly_variable_rank``, else ``dispersions_norm``, else ``variances_norm``)."""
    var = adata.var
    if "highly_variable" not in var.columns:
        raise ProgramsError(
            "No variable genes are flagged in this dataset: run the pipeline's highly variable "
            "genes step first (gene programs are found among the variable genes).")
    flagged = np.asarray(pd.Series(var["highly_variable"]).fillna(False).astype(bool))
    n_flagged = int(flagged.sum())
    if n_flagged < MIN_GENES:
        raise ProgramsError(
            f"Only {n_flagged} variable genes are flagged; gene programs need at least {MIN_GENES}.")
    notes: list[str] = []
    keep = np.flatnonzero(flagged)
    if n_hvg is not None and n_hvg < n_flagged:
        if n_hvg < MIN_GENES:
            raise ProgramsError(f"Use at least {MIN_GENES} variable genes.")
        ranked = None
        for col, ascending in (("highly_variable_rank", True), ("dispersions_norm", False),
                               ("variances_norm", False)):
            if col in var.columns:
                vals = pd.to_numeric(pd.Series(var[col]).iloc[keep], errors="coerce").to_numpy(np.float64)
                if np.isfinite(vals).any():
                    order = np.argsort(np.where(np.isfinite(vals), vals if ascending else -vals, np.inf),
                                       kind="stable")
                    ranked = keep[order[:n_hvg]]
                    notes.append(f"The top {n_hvg:,} of {n_flagged:,} variable genes by {col}.")
                    break
        if ranked is None:
            notes.append(f"No stored ranking of the variable genes: all {n_flagged:,} are used.")
        else:
            keep = np.sort(ranked)
    elif n_hvg is not None and n_hvg > n_flagged:
        notes.append(f"The dataset flags {n_flagged:,} variable genes; all of them are used "
                     f"({n_hvg:,} were asked for).")
    names = [str(g) for g in adata.var_names[keep]]
    return names, n_flagged, notes


def _columns(M, cols: np.ndarray) -> sp.csr_matrix:
    """``M[:, cols]`` as CSR float32, read in row chunks unless ``M`` is an
    in-memory sparse matrix: a dense matrix is never copied whole."""
    if sp.issparse(M):
        return sp.csr_matrix(sp.csr_matrix(M)[:, cols], dtype=np.float32)
    blocks = []
    for start in range(0, M.shape[0], _CHUNK):
        block = M[start:start + _CHUNK]
        if sp.issparse(block):
            blocks.append(sp.csr_matrix(sp.csr_matrix(block)[:, cols], dtype=np.float32))
        else:
            blocks.append(sp.csr_matrix(np.asarray(block)[:, cols], dtype=np.float32))
    return sp.csr_matrix(sp.vstack(blocks, format="csr"), dtype=np.float32)


def _row_totals(M) -> np.ndarray:
    """Per-cell sums over every gene, read in row chunks for a dense matrix."""
    if sp.issparse(M):
        return np.asarray(M.sum(axis=1), dtype=np.float64).ravel()
    parts = []
    for start in range(0, M.shape[0], _CHUNK):
        parts.append(np.asarray(M[start:start + _CHUNK], dtype=np.float64).sum(axis=1))
    return np.concatenate(parts) if parts else np.zeros(0)


def log_hvg(adata, genes: list[str]) -> tuple[sp.csr_matrix, str]:
    """Log-normalised expression of ``genes`` (cells x genes, CSR float32) and
    where it came from (see the module docstring). Refuses scaled values."""
    raw = adata.raw
    if raw is not None and _is_lognorm(raw.X):
        idx = pd.Index(raw.var_names).get_indexer(genes)
        if (idx >= 0).all():
            return _columns(raw.X, idx), "log-normalised .raw"
    if _is_lognorm(adata.X):
        return _columns(adata.X, pd.Index(adata.var_names).get_indexer(genes)), "log-normalised X"
    from scview.core.normalization import find_counts

    try:
        counts, where = find_counts(adata)
    except ValueError:
        low = _sample_min(adata.X)
        if low is not None and low < 0:
            raise ProgramsError(
                "X holds scaled (centred) values, which are negative, and NMF needs non-negative "
                "input; the dataset has no log-normalised .raw and no counts to take them from. "
                "Run the pipeline from normalisation (it keeps the log values in .raw).") from None
        raise ProgramsError(
            "No log-normalised expression or counts to find programs in: run normalisation and "
            "the log transform first.") from None
    totals = _row_totals(counts)
    sub = _columns(counts, pd.Index(adata.var_names).get_indexer(genes))
    scale = np.divide(TARGET_SUM, totals, out=np.zeros_like(totals), where=totals > 0)
    out = sp.csr_matrix(sp.diags(scale.astype(np.float32)) @ sub, dtype=np.float32)
    out.data = np.log1p(out.data)
    return out, f"shifted log of the counts in {where} (log1p of counts per {TARGET_SUM:,.0f})"


def gene_sd(X: sp.csr_matrix) -> np.ndarray:
    """Each column's standard deviation (n - 1), from sparse sums."""
    n = X.shape[0]
    mean = np.asarray(X.mean(axis=0), dtype=np.float64).ravel()
    sq = np.asarray(X.multiply(X).mean(axis=0), dtype=np.float64).ravel()
    var = np.maximum(sq - mean * mean, 0.0) * (n / max(n - 1, 1))
    return np.sqrt(var)


def prepare_input(adata, n_hvg: int | None = None) -> ProgramInput:
    """The scaled variable-gene matrix NMF runs on (see the module docstring)."""
    if adata.n_obs < 20:
        raise ProgramsError("Gene programs need at least 20 cells.")
    genes, n_flagged, notes = select_genes(adata, n_hvg)
    L, source = log_hvg(adata, genes)
    if L.nnz and float(L.data.min()) < 0:
        raise ProgramsError(
            f"The {source} has negative values among the variable genes; NMF needs non-negative "
            "input. Run the pipeline from normalisation.")
    sd = gene_sd(L)
    keep = sd > 0
    n_dropped = int((~keep).sum())
    if n_dropped:
        L = L[:, np.flatnonzero(keep)]
        genes = [g for g, k in zip(genes, keep) if k]
        sd = sd[keep]
        notes.append(f"{n_dropped} variable gene{'s' if n_dropped != 1 else ''} with no variance in "
                     "these cells left out.")
    if len(genes) < MIN_GENES:
        raise ProgramsError(f"Only {len(genes)} variable genes vary in these cells; gene programs "
                            f"need at least {MIN_GENES}.")
    X = sp.csr_matrix(L @ sp.diags((1.0 / sd).astype(np.float32)), dtype=np.float32)
    X.eliminate_zeros()
    n_empty = int((np.diff(X.indptr) == 0).sum())
    if n_empty:
        notes.append(f"{n_empty:,} cell{'s' if n_empty != 1 else ''} with none of the variable genes "
                     "get a usage of 0 for every program.")
    return ProgramInput(X=X, genes=genes, source=source, n_flagged=n_flagged, n_dropped=n_dropped,
                        n_empty_cells=n_empty, notes=notes)


# ---------------------------------------------------------------------------
# NMF, consensus, usage, gene scores
# ---------------------------------------------------------------------------


def _limits():
    from threadpoolctl import threadpool_limits

    return threadpool_limits(limits=THREADS)


def _nmf(k: int, seed: int):
    from sklearn.decomposition import NMF

    return NMF(n_components=k, init=INIT, solver=SOLVER, beta_loss=BETA_LOSS, tol=TOL,
               max_iter=MAX_ITER, random_state=seed)


def fit_replicate(X, k: int, seed: int) -> tuple[np.ndarray, int]:
    """One NMF run from a random start: (spectra k x genes, iterations used)."""
    from sklearn.exceptions import ConvergenceWarning

    model = _nmf(k, seed)
    with _limits(), warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        model.fit_transform(X)
    return np.asarray(model.components_, dtype=np.float64), int(model.n_iter_)


@dataclass
class Consensus:
    spectra: np.ndarray  # k x genes, rows sum to 1
    labels: np.ndarray  # cluster of each replicate spectrum (replicate-major order)
    runs: np.ndarray  # replicate each spectrum came from
    silhouette: float | None  # of the whole grouping
    member_silhouette: np.ndarray | None  # per spectrum
    programs: list[dict[str, Any]]  # per consensus program: correlation, runs found, ...


def _unit_rows(S: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(S, axis=1, keepdims=True)
    return np.divide(S, norms, out=np.zeros_like(S), where=norms > 0)


def _mean_correlation(M: np.ndarray) -> float | None:
    """Mean pairwise Pearson correlation of the rows of ``M`` (None for one row)."""
    if M.shape[0] < 2:
        return None
    C = np.corrcoef(M)
    iu = np.triu_indices(M.shape[0], 1)
    vals = C[iu]
    vals = vals[np.isfinite(vals)]
    return float(vals.mean()) if vals.size else None


def consensus(spectra: list[np.ndarray], k: int) -> Consensus:
    """Group the replicates' spectra into ``k`` programs (see the module docstring)."""
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_samples

    S = np.vstack(spectra)
    runs = np.repeat(np.arange(len(spectra)), [s.shape[0] for s in spectra])
    U = _unit_rows(S)
    if len(spectra) == 1:
        labels = np.arange(k)
    else:
        with _limits():
            labels = KMeans(n_clusters=k, n_init=KMEANS_STARTS, random_state=KMEANS_SEED).fit_predict(U)
    sil = member_sil = None
    if len(spectra) > 1 and 1 < len(np.unique(labels)) < U.shape[0]:
        member_sil = silhouette_samples(U, labels, metric="euclidean")
        sil = float(member_sil.mean())
    cons = np.zeros((k, S.shape[1]))
    programs = []
    for c in range(k):
        members = np.flatnonzero(labels == c)
        if members.size:
            cons[c] = np.median(U[members], axis=0)
        programs.append({
            "n_members": int(members.size),
            "runs_found": int(np.unique(runs[members]).size),
            "correlation": _mean_correlation(S[members]) if members.size else None,
            "silhouette": (float(member_sil[members].mean())
                           if member_sil is not None and members.size else None),
        })
    totals = cons.sum(axis=1, keepdims=True)
    cons = np.divide(cons, totals, out=np.zeros_like(cons), where=totals > 0)
    return Consensus(spectra=cons, labels=labels, runs=runs, silhouette=sil,
                     member_silhouette=member_sil, programs=programs)


def refit_usage(X, spectra: np.ndarray) -> np.ndarray:
    """Usages (cells x k, non-negative) that best rebuild ``X`` from the fixed
    ``spectra``: NMF with the spectra held, as cNMF refits."""
    from sklearn.decomposition import non_negative_factorization
    from sklearn.exceptions import ConvergenceWarning

    H = np.asarray(spectra, dtype=X.dtype)
    with _limits(), warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        W, _, _ = non_negative_factorization(
            X, H=H, n_components=H.shape[0], init=None, update_H=False, solver=SOLVER,
            beta_loss=BETA_LOSS, tol=TOL, max_iter=MAX_ITER)
    return np.asarray(W, dtype=np.float64)


def relative_error(X, W: np.ndarray, H: np.ndarray, norm2: float | None = None) -> float:
    """``||X - WH||^2 / ||X||^2`` without forming WH (sparse-safe)."""
    if norm2 is None:
        norm2 = float(X.multiply(X).sum()) if sp.issparse(X) else float((np.asarray(X) ** 2).sum())
    if norm2 <= 0:
        return 0.0
    XtW = np.asarray(X.T @ W)  # genes x k
    cross = float((XtW.T * H).sum())
    gram = float(((W.T @ W) * (H @ H.T)).sum())
    return max(norm2 - 2.0 * cross + gram, 0.0) / norm2


def share_usage(W: np.ndarray) -> np.ndarray:
    """Each cell's usages scaled to sum 1 (a cell with no usage stays 0)."""
    totals = W.sum(axis=1, keepdims=True)
    return np.divide(W, totals, out=np.zeros_like(W), where=totals > 0)


def gene_scores(X, usage: np.ndarray) -> np.ndarray:
    """k x genes: the least-squares coefficients of each gene's z-scored
    expression on the usage shares (cells with no usage left out), computed
    from sparse products: the z-scored matrix is never formed."""
    rows = np.flatnonzero(usage.sum(axis=1) > 0)
    U = usage[rows]
    Xr = X[rows] if sp.issparse(X) else np.asarray(X)[rows]
    n = Xr.shape[0]
    mean = np.asarray(Xr.mean(axis=0), dtype=np.float64).ravel()
    sd = gene_sd(sp.csr_matrix(Xr)) if sp.issparse(Xr) else np.asarray(Xr).std(axis=0, ddof=1)
    sd = np.where(sd > 0, sd, 1.0)
    P = np.linalg.pinv(U)  # k x n
    PX = np.asarray(Xr.T @ P.T).T  # k x genes
    P1 = P @ np.ones(n)
    return (PX - np.outer(P1, mean)) / sd[None, :]


def top_genes(scores: np.ndarray, genes: list[str], n: int = N_TOP) -> list[list[dict[str, Any]]]:
    """Per program, the ``n`` genes with the highest score (positive only)."""
    out = []
    for row in scores:
        order = np.argsort(-row, kind="stable")[:n]
        out.append([{"gene": genes[i], "score": round(float(row[i]), 4)} for i in order if row[i] > 0])
    return out


# ---------------------------------------------------------------------------
# One k, and the run
# ---------------------------------------------------------------------------


class ProgramsCancelled(Exception):
    """The run was stopped between replicates (the client went away)."""


@dataclass
class KFit:
    """Everything one value of k produced, programs ordered by mean usage share."""

    k: int
    seeds: list[int]
    spectra: np.ndarray  # k x genes (consensus, rows sum 1)
    usage_raw: np.ndarray  # cells x k
    usage: np.ndarray  # cells x k, rows sum 1 (0 for a cell with no usage)
    scores: np.ndarray  # k x genes
    programs: list[dict[str, Any]]  # stability and usage stats per program
    silhouette: float | None
    error: float
    seconds: float
    iterations: list[int]

    def stats(self) -> dict[str, Any]:
        """The numbers a k sweep compares."""
        corr = [p["correlation"] for p in self.programs if p["correlation"] is not None]
        return {
            "k": self.k,
            "silhouette": _r(self.silhouette),
            "mean_correlation": _r(float(np.mean(corr))) if corr else None,
            "n_stable": int(sum(p["stable"] for p in self.programs)),
            "error": _r(self.error),
            "seconds": round(self.seconds, 1),
        }


def _r(v: float | None, d: int = 4) -> float | None:
    return None if v is None or not np.isfinite(v) else round(float(v), d)


def _finish(inp: ProgramInput, k: int, seeds: list[int], spectra: list[np.ndarray],
            iterations: list[int], t0: float) -> KFit:
    X = inp.X
    cons = consensus(spectra, k)
    W = refit_usage(X, cons.spectra)
    usage = share_usage(W)
    order = np.argsort(-usage.mean(axis=0), kind="stable")  # the most used first
    H, W, usage = cons.spectra[order], W[:, order], usage[:, order]
    scores = gene_scores(X, usage)
    error = relative_error(X, W, H)
    has = usage.sum(axis=1) > 0
    top = np.argmax(usage, axis=1)
    n_has = max(int(has.sum()), 1)
    programs = []
    for j, c in enumerate(order):
        p = dict(cons.programs[c])
        corr = p["correlation"]
        p["stable"] = bool(len(seeds) > 1 and p["runs_found"] == len(seeds) and corr is not None
                           and corr >= STABLE_CORR)
        p["correlation"] = _r(corr)
        p["silhouette"] = _r(p["silhouette"])
        p["mean_usage"] = _r(float(usage[:, j].mean()))
        p["pct_top"] = _r(float(((top == j) & has).sum()) / n_has)
        programs.append(p)
    return KFit(k=k, seeds=list(seeds), spectra=H, usage_raw=W, usage=usage, scores=scores,
                programs=programs, silhouette=cons.silhouette, error=error,
                seconds=time.perf_counter() - t0, iterations=iterations)


def fit_k_steps(inp: ProgramInput, k: int, seeds: list[int],
                stop: Callable[[], bool] | None = None) -> Iterator[dict[str, Any]]:
    """One ``k``: a ``progress`` event after each replicate; the generator returns
    the ``KFit`` (``fit = yield from fit_k_steps(...)``). ``stop`` is checked
    between replicates and raises ``ProgramsCancelled``."""
    t0 = time.perf_counter()
    if k >= min(inp.X.shape):
        raise ProgramsError(f"k = {k} is too many programs for {inp.X.shape[0]:,} cells x "
                            f"{inp.X.shape[1]:,} genes.")
    spectra, iterations = [], []
    for i, seed in enumerate(seeds):
        if stop is not None and stop():
            raise ProgramsCancelled()
        H, n_iter = fit_replicate(inp.X, k, seed)
        spectra.append(H)
        iterations.append(n_iter)
        yield {"type": "progress", "k": k, "run": i + 1, "runs": len(seeds)}
    if stop is not None and stop():
        raise ProgramsCancelled()
    return _finish(inp, k, seeds, spectra, iterations, t0)


def fit_k(inp: ProgramInput, k: int, seeds: list[int]) -> KFit:
    """``fit_k_steps`` run to its end."""
    gen = fit_k_steps(inp, k, seeds)
    while True:
        try:
            next(gen)
        except StopIteration as done:
            return done.value


def clean_request(k: int | None, seeds: int | None,
                  sweep: list[int] | None) -> tuple[int, list[int], list[int]]:
    """Validated (k, the seeds 0..R-1, the ks to fit); the chosen k is fitted last."""
    k = DEFAULT_K if k is None else int(k)
    if not MIN_K <= k <= MAX_K:
        raise ProgramsError(f"k must be between {MIN_K} and {MAX_K}.")
    r = DEFAULT_SEEDS if seeds is None else int(seeds)
    if not 1 <= r <= MAX_SEEDS:
        raise ProgramsError(f"Use between 1 and {MAX_SEEDS} seeds.")
    ks = sorted({int(v) for v in (sweep or [])})
    if len(ks) > MAX_SWEEP:
        raise ProgramsError(f"A sweep takes at most {MAX_SWEEP} values of k.")
    if any(not MIN_K <= v <= MAX_K for v in ks):
        raise ProgramsError(f"Sweep values of k must be between {MIN_K} and {MAX_K}.")
    return k, list(range(r)), [v for v in ks if v != k] + [k]


def estimate_seconds(X, ks: list[int], n_seeds: int) -> float:
    """A rough time for a run, calibrated on Kang (1.06M stored values; k = 10: 2.2 s
    per replicate on 2 cores); it grows with the stored values and faster than k."""
    per = 2.2 * (X.nnz / 1.06e6) * sum((kk / 10) ** 1.5 for kk in ks)
    return round(per * n_seeds + 0.5 * len(ks), 1)


def iter_programs(inp: ProgramInput, k: int, seeds: list[int], ks: list[int],
                  stop: Callable[[], bool] | None = None) -> Iterator[dict[str, Any]]:
    """The run as events: ``progress`` after each replicate (with ``done``/``total``),
    ``k_result`` after each k (its sweep numbers), last ``fit`` holding the chosen
    k's ``KFit`` and the sweep."""
    total = len(ks) * len(seeds)
    done = 0
    t0 = time.perf_counter()
    fits: dict[int, KFit] = {}
    for kk in ks:
        gen = fit_k_steps(inp, kk, seeds, stop)
        while True:
            try:
                event = next(gen)
            except StopIteration as end:
                fits[kk] = end.value
                break
            done += 1
            yield {**event, "done": done, "total": total, "elapsed_s": round(time.perf_counter() - t0, 1)}
        yield {"type": "k_result", **fits[kk].stats()}
    sweep = [fits[v].stats() for v in sorted(fits)] if len(fits) > 1 else []
    yield {"type": "fit", "fit": fits[k], "sweep": sweep}


# ---------------------------------------------------------------------------
# Saved results
# ---------------------------------------------------------------------------


def method_text(k: int, n_seeds: int, n_genes: int, source: str) -> str:
    return (
        f"Consensus NMF (Kotliar et al. 2019): {n_seeds} NMF run{'s' if n_seeds != 1 else ''} "
        f"(sklearn, Frobenius loss, coordinate descent, random starts, seeds 0-{n_seeds - 1}) with "
        f"k = {k} on {n_genes:,} variable genes ({source}), each gene divided by its standard "
        "deviation; the runs' spectra grouped by k-means and each group's median kept; usages "
        "refitted with those spectra fixed. Usage share: a cell's usages scaled to sum 1. Gene "
        "score: the least-squares coefficient of the gene's z-scored expression on the usage "
        f"shares. Stable: found in every run, member spectra correlating >= {STABLE_CORR:g}."
    )


def replay_params(fit: KFit, inp: ProgramInput, n_hvg: int | None) -> dict[str, Any]:
    """What the provenance records (and the replay script reads)."""
    return {
        "k": fit.k, "seeds": list(fit.seeds), "n_hvg": n_hvg, "n_genes": len(inp.genes),
        "source": inp.source, "solver": SOLVER, "init": INIT, "beta_loss": BETA_LOSS, "tol": TOL,
        "max_iter": MAX_ITER, "kmeans_seed": KMEANS_SEED, "kmeans_starts": KMEANS_STARTS,
        "target_sum": TARGET_SUM, "genes": list(inp.genes),
    }


def save(adata, fit: KFit, inp: ProgramInput, *, n_hvg: int | None = None,
         sweep: list[dict] | None = None, resources: dict | None = None,
         fingerprint: str | None = None) -> dict[str, Any]:
    """Store the programs in ``adata`` (obsm usage, uns ``scview_programs``, a
    provenance step) and return the summary. Replaces the uns entries (never
    mutates them), so ``write_derived`` sees them as changed."""
    from scview.core import provenance

    created = datetime.now(timezone.utc).isoformat()
    tops = top_genes(fit.scores, inp.genes)
    programs = [{"index": j + 1, "name": f"P{j + 1}", **fit.programs[j], "top_genes": tops[j]}
                for j in range(fit.k)]
    info = {
        "version": VERSION, "run_id": f"{created}|{fit.k}|{adata.n_obs}", "created_at": created,
        "k": fit.k, "seeds": list(fit.seeds), "n_cells": int(adata.n_obs), "n_genes": len(inp.genes),
        "n_flagged": inp.n_flagged, "n_dropped": inp.n_dropped, "n_empty_cells": inp.n_empty_cells,
        "n_hvg": n_hvg, "source": inp.source, "notes": list(inp.notes),
        "params": {"solver": SOLVER, "init": INIT, "beta_loss": BETA_LOSS, "tol": TOL,
                   "max_iter": MAX_ITER, "kmeans_seed": KMEANS_SEED, "kmeans_starts": KMEANS_STARTS,
                   "scaling": "each gene divided by its standard deviation, not centred"},
        "silhouette": _r(fit.silhouette), "error": _r(fit.error), "seconds": round(fit.seconds, 1),
        "iterations": fit.iterations, "sweep": list(sweep or []), "programs": programs,
        "cells_fingerprint": fingerprint,
        "method": method_text(fit.k, len(fit.seeds), len(inp.genes), inp.source),
    }
    adata.obsm[OBSM_KEY] = fit.usage.astype(np.float32)
    adata.obsm[OBSM_RAW_KEY] = fit.usage_raw.astype(np.float32)
    adata.uns[UNS_KEY] = {
        "info": json.dumps(info),
        "genes": np.asarray(inp.genes, dtype=object),
        "spectra": fit.spectra.astype(np.float32),
        "scores": fit.scores.astype(np.float32),
    }
    adata.uns.pop(ENRICH_KEY, None)  # results of the last run's programs
    provenance.record_step(adata, step="gene_programs", tool="scview.gene_programs",
                           params=replay_params(fit, inp, n_hvg), resources=resources)
    return summary(adata)


def stored(adata) -> dict[str, Any] | None:
    """The saved run's info, or None when there is none (or it no longer fits the
    cells: obsm missing or of another width)."""
    entry = adata.uns.get(UNS_KEY) if UNS_KEY in adata.uns else None
    if not isinstance(entry, dict) or "info" not in entry:
        return None
    try:
        info = json.loads(str(entry["info"]))
    except (TypeError, ValueError):
        logger.warning("Ignoring an unreadable %s", UNS_KEY)
        return None
    U = adata.obsm[OBSM_KEY] if OBSM_KEY in adata.obsm else None
    if U is None or U.shape[1] != info.get("k") or U.shape[0] != adata.n_obs:
        return None
    return info


def usage_matrix(adata, raw: bool = False) -> np.ndarray:
    key = OBSM_RAW_KEY if raw and OBSM_RAW_KEY in adata.obsm else OBSM_KEY
    return np.asarray(adata.obsm[key], dtype=np.float32)


def program_usage(adata, index: int, raw: bool = False) -> np.ndarray:
    """One program's per-cell usage (``index`` 1-based, as P1..Pk)."""
    info = stored(adata)
    if info is None:
        raise ProgramsError("No gene programs for this dataset yet: run them in the Programs tab.")
    if not 1 <= index <= info["k"]:
        raise ProgramsError(f"There are {info['k']} programs (P1 to P{info['k']}); no P{index}.")
    return usage_matrix(adata, raw)[:, index - 1]


def _group_labels(adata, column: str) -> pd.Series:
    if column not in adata.obs.columns:
        raise ProgramsError(f"Column '{column}' not found.")
    s = adata.obs[column]
    return s.astype(str).where(s.notna())


def _groups(adata, column: str) -> tuple[np.ndarray, list[str]]:
    """Each cell's group label (None outside every group) and the groups in the
    column's order (categories, else sorted)."""
    labels = _group_labels(adata, column)
    s = adata.obs[column]
    present = set(labels.dropna().unique())
    order = ([str(c) for c in s.cat.categories] if hasattr(s, "cat")
             else sorted(present, key=str))
    return labels.to_numpy(dtype=object), [g for g in order if g in present]


def by_group(adata, column: str, max_groups: int = 100) -> dict[str, Any]:
    """Mean usage share of each program in each group of ``column`` (cells without a
    label left out), groups in the column's order."""
    labels, groups = _groups(adata, column)
    if len(groups) > max_groups:
        raise ProgramsError(f"'{column}' has {len(groups)} groups; the table takes at most {max_groups}.")
    U = usage_matrix(adata).astype(np.float64)
    means, sizes = [], []
    for g in groups:
        m = labels == g
        sizes.append(int(m.sum()))
        means.append(U[m].mean(axis=0))
    M = np.array(means).reshape(len(groups), U.shape[1])
    return {
        "column": column,
        "groups": groups,
        "n_cells": sizes,
        "mean": [[_r(float(M[gi, j])) for gi in range(len(groups))] for j in range(U.shape[1])],  # [program][group]
    }


def violin(adata, values: np.ndarray, column: str, max_groups: int = 100) -> dict[str, list[float]]:
    """``values`` split by the groups of ``column``, for the violin under the plot
    (empty when the column has more than ``max_groups`` groups)."""
    labels, groups = _groups(adata, column)
    if len(groups) > max_groups:
        return {}
    return {g: [round(float(v), 4) for v in values[labels == g]] for g in groups}


def summary(adata, groupby: str | None = None) -> dict[str, Any]:
    """The saved programs for the UI: ``{"available": False}`` without a run."""
    info = stored(adata)
    if info is None:
        return {"available": False}
    out = {"available": True, **info}
    enr = enrichment_cache(adata, info)
    for p in out["programs"]:
        hit = enr.get(str(p["index"]))
        p["top_term"] = hit["top"][0] if hit and hit.get("top") else None
    if groupby:
        try:
            out["by_group"] = by_group(adata, groupby)
        except ProgramsError as exc:
            out["by_group"] = None
            out["by_group_error"] = str(exc)
    return out


def runnable(adata) -> dict[str, Any]:
    """Whether a run can start, with the defaults; reads only the gene flags and a
    sample of the matrix (no copy)."""
    try:
        genes, n_flagged, _ = select_genes(adata)
        raw = adata.raw
        if raw is not None and _is_lognorm(raw.X):
            source = "log-normalised .raw"
        elif _is_lognorm(adata.X):
            source = "log-normalised X"
        else:
            from scview.core.normalization import find_counts

            try:
                _, where = find_counts(adata)
                source = f"shifted log of the counts in {where}"
            except ValueError:
                low = _sample_min(adata.X)
                raise ProgramsError(
                    "X holds scaled (centred) values and the dataset has no log-normalised .raw or "
                    "counts: NMF needs non-negative input." if low is not None and low < 0 else
                    "No log-normalised expression or counts to find programs in.") from None
        return {"ok": True, "reason": None, "n_flagged": n_flagged, "source": source}
    except ProgramsError as exc:
        return {"ok": False, "reason": str(exc), "n_flagged": None, "source": None}


# ---------------------------------------------------------------------------
# Enrichment of a program's top genes (cached for the co-pilot)
# ---------------------------------------------------------------------------


def enrichment_cache(adata, info: dict | None = None) -> dict[str, Any]:
    """{program index (str): {"collections", "top": [...]}} for the saved run."""
    info = info if info is not None else stored(adata)
    raw = adata.uns.get(ENRICH_KEY) if ENRICH_KEY in adata.uns else None
    if info is None or raw is None:
        return {}
    try:
        data = json.loads(str(raw))
    except (TypeError, ValueError):
        return {}
    if not isinstance(data, dict) or data.get("run_id") != info.get("run_id"):
        return {}
    results = data.get("results")
    return results if isinstance(results, dict) else {}


def remember_enrichment(adata, index: int, collections: list[str], records: list[dict]) -> None:
    """Keep a program's five best terms with the run (replacing the uns entry)."""
    info = stored(adata)
    if info is None:
        return
    results = dict(enrichment_cache(adata, info))
    results[str(index)] = {
        "collections": list(collections),
        "top": [{"term": r["term"], "adjusted_pvalue": float(r["adjusted_pvalue"]),
                 "overlap_count": int(r.get("overlap_count", 0)), "gene_count": int(r.get("gene_count", 0)),
                 "collection": r.get("collection", "")}
                for r in records[:5] if float(r.get("adjusted_pvalue", 1.0)) < 0.05],
    }
    adata.uns[ENRICH_KEY] = json.dumps({"run_id": info["run_id"], "results": results})


def query_genes(info: dict, index: int, n: int = N_TOP) -> list[str]:
    return [g["gene"] for g in info["programs"][index - 1]["top_genes"][:n]]


def background_genes(adata) -> list[str]:
    entry = adata.uns.get(UNS_KEY)
    return [str(g) for g in np.asarray(entry["genes"]).tolist()]


# ---------------------------------------------------------------------------
# Co-pilot
# ---------------------------------------------------------------------------


def grounding_lines(adata) -> tuple[list[str], str]:
    """The saved programs as a fact: k, the stable ones with their top five genes
    and best enriched term when one was computed. Nothing without a run."""
    info = stored(adata)
    if info is None:
        return [], ""
    enr = enrichment_cache(adata, info)
    progs = info["programs"]
    stable = [p for p in progs if p.get("stable")]
    n_seeds = len(info.get("seeds") or [])

    def one(p: dict) -> str:
        genes = ", ".join(g["gene"] for g in p["top_genes"][:5])
        corr = f"r = {p['correlation']:.2f}" if p.get("correlation") is not None else "r n/a"
        text = (f"{p['name']} ({genes}; {corr}; top program in {p['pct_top']:.0%} of cells, mean share "
                f"{p['mean_usage']:.0%})")
        hit = enr.get(str(p["index"]))
        if hit and hit.get("top"):
            t = hit["top"][0]
            text += f", most enriched: {t['term']} (adj. p {t['adjusted_pvalue']:.1e})"
        return text

    shown = stable if stable else progs[:5]
    which = (f"{len(stable)} of {info['k']} are stable" if n_seeds > 1 else
             "stability unknown (one run)")
    detail = (f"k = {info['k']} NMF gene programs on {info['n_genes']:,} variable genes, {n_seeds} "
              f"run(s); {which}")
    line = (f"- Gene programs (consensus NMF): {detail} (found in every run, spectra correlating >= "
            f"{STABLE_CORR:g}). {'Stable' if stable else 'First five'}: "
            + "; ".join(one(p) for p in shown)
            + ". Usage share: the part of a cell's modelled variable-gene expression a program accounts "
              "for; programs can be cell types as well as processes, and k was chosen by the user "
              f"[{TAG}]")
    return ["\n## Gene programs (NMF)", line], detail
