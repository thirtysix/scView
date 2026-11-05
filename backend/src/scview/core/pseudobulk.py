"""Pseudobulk differential expression across donors (limma-voom / voomByGroup via
core/voom.py, edgeR v4 quasi-likelihood via core/edger_ql.py, or DESeq2 via
core/deseq2_vec.py), optionally with RUV factors.

A cell-level test treats every cell as an independent replicate, so when two
conditions also differ by donor its p-values are far too small. Here each
donor's raw counts are summed per condition into one pseudobulk sample, and those
samples are compared (limma-voom's moderated t by default): the replicate is the
donor, not the cell.

When every donor contributes a sample to both conditions (the usual
stimulated-vs-control layout), the design is paired, ``~donor + condition``, so
donor-to-donor baseline differences are absorbed; otherwise it is ``~condition``.

``method``: ``voom`` (default: TMM, one mean-variance trend, moderated t),
``voombygroup`` (a trend per condition, for when one condition's donors vary more)
``edger_ql`` (edgeR v4's quasi-likelihood F-test, glmQLFit legacy=FALSE: matched
R edgeR 4.6 to 3 decimals in p) or ``deseq2`` (negative-binomial Wald test,
vectorised over genes: within 0.01 of R DESeq2's p, and its dispersion trend is
fitted as R fits it, where pydeseq2's departs). voom is the default because it held
its false discovery rate on real donor splits where DESeq2 did not: true FDP at a
nominal 5% was 13% / 7% / 6% at 2 / 3 / 4 donors per side for voom and 34% / 20% /
15% for DESeq2 (R and pydeseq2 alike), over 18 datasets; edgeR QL held it as voom did
(docs/validation/pseudobulk-de.md). Whatever the method, each condition's own trend and residual variance are
compared (``heteroscedasticity``). ``ruv_k`` > 0 adds that many RUVg factors from
empirical control genes (core/ruv.py) to the design.

``covariates`` adds per-sample obs columns (sex, batch, age, chemistry) to the
design, ``~sex + batch + condition`` (as Milo takes them): a column must hold one
value per pseudobulk sample, and one that cannot be separated from the condition
(every level on one side) is refused. Text, boolean and categorical columns, and
numbers with at most ``MAX_NUMERIC_LEVELS`` distinct values, are factors; other
numbers enter as one centred and scaled slope. A covariate the paired design
already holds (constant within each donor, as sex is) is left out with a note.
"""

from __future__ import annotations

import logging
import threading
import weakref
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
# Compute bounds: the count models fit every gene, so cost is samples x genes.
MAX_SAMPLES = 300
MAX_GENES = 20_000
METHODS = ("deseq2", "voom", "voombygroup", "edger_ql")
METHOD_NAMES = {"deseq2": "DESeq2", "voom": "limma-voom", "voombygroup": "voomByGroup",
                "edger_ql": "edgeR QL"}
MAX_RUV_K = 3
# DESeq2 called more false genes than its FDR on real donor splits below this many
# donors on the smaller side (true FDP at a nominal 5%: 34% at 2, 20% at 3, 15% at
# 4, 11% at 6; voom 13%, 7%, 6%, 6%): a chosen DESeq2 run warns under it.
DESEQ2_FDR_DONORS = 6
# One condition's median residual variance this many times the other's flags
# heteroscedastic groups (voomByGroup then fits each its own trend).
HETERO_RATIO = 2.0
# Points per group trend sent for the plot.
TREND_POINTS = 60
# A RUV factor correlated with the condition at least this much may take signal.
RUV_CONDITION_R = 0.5
TAG = "result:pseudobulk"  # the co-pilot's tag for the last run
MAX_COVARIATES = 4
# A numeric covariate with at most this many distinct sample values is a factor
# (batch 1/2/3 is not a slope); more, and it is one slope (age).
MAX_NUMERIC_LEVELS = 5


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
    # gene -> Wald statistic (DESeq2), moderated t (voom) or signed sqrt(F) (edgeR QL)
    stats: dict[str, float] = field(default_factory=dict)
    method: str = "voom"
    ruv_k: int = 0
    heteroscedasticity: dict | None = None
    ruv: dict | None = None
    covariates: list[str] = field(default_factory=list)  # in the design, as asked
    covariate_notes: list[str] = field(default_factory=list)  # left out, and why
    warnings: list[str] = field(default_factory=list)  # design cautions (repeated donors)


def _sample_values(values: pd.Series, codes: np.ndarray, n: int, name: str,
                   names: list[str]) -> np.ndarray:
    """One value per pseudobulk sample from the cells' values; refuses a column that is
    missing for some cells or varies within a sample."""
    v = values.to_numpy()
    missing = pd.isna(v)
    if missing.any():
        raise PseudobulkError(
            f"Covariate '{name}' is missing for {int(missing.sum()):,} of the compared cells: "
            "a covariate needs a value for every sample.")
    frame = pd.DataFrame({"s": codes, "v": v})
    k = frame.groupby("s")["v"].nunique()
    if (k > 1).any():
        bad = names[int(k.index[np.argmax(k.to_numpy() > 1)])]
        raise PseudobulkError(
            f"Covariate '{name}' varies within a sample ({bad} has {int(k.max())} values): a "
            "covariate must be one value per pseudobulk sample (sex, batch, age), not per cell.")
    first = frame.drop_duplicates("s").set_index("s")["v"]
    return first.reindex(range(n)).to_numpy()


def _rank(M: np.ndarray) -> int:
    return int(np.linalg.matrix_rank(M)) if M.size else 0


def _covariate_design(cov: dict[str, np.ndarray], nuisance: np.ndarray, is_a: np.ndarray,
                      sample_cond: np.ndarray, names: dict[str, str]
                      ) -> tuple[np.ndarray, dict[str, pd.Series], list[str], list[str]]:
    """(the covariates' design columns, each used covariate's per-sample values for
    DESeq2's formula, the covariates used, notes on those left out). Refuses one that
    is confounded with the condition or only partly separable from the rest."""
    cols: list[np.ndarray] = []
    meta: dict[str, pd.Series] = {}
    used: list[str] = []
    notes: list[str] = []
    base = nuisance
    for j, (name, raw) in enumerate(cov.items()):
        s = pd.Series(raw)
        numeric = (pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s)
                   and s.nunique() > MAX_NUMERIC_LEVELS)
        if s.nunique() < 2:
            notes.append(f"'{name}' has one value in these samples: left out.")
            continue
        if numeric:
            x = s.to_numpy(dtype=np.float64)
            z = (x - x.mean()) / x.std()
            block = z[:, None]
            meta[f"c{j}"] = pd.Series(z)
        else:
            levels = sorted(s.astype(str).unique())
            codes = pd.Categorical(s.astype(str), categories=levels).codes
            block = (codes[:, None] == np.arange(1, len(levels))[None, :]).astype(np.float64)
            meta[f"c{j}"] = pd.Series([f"v{c}" for c in codes])
        r0 = _rank(base)
        gain = _rank(np.column_stack([base, block])) - r0
        if gain == 0:
            notes.append(f"'{name}' is constant within each donor, which the paired design already "
                         "holds: left out." if nuisance.shape[1] > 1 else
                         f"'{name}' adds nothing the design does not hold: left out.")
            meta.pop(f"c{j}")
            continue
        if gain < block.shape[1]:
            raise PseudobulkError(
                f"Covariate '{name}' is partly collinear with the donors or the other covariates "
                "(some of its values occur with one donor or one level only): leave it out or "
                "merge its rare values.")
        with_cond = np.column_stack([base, block, is_a])
        if _rank(with_cond) < _rank(np.column_stack([base, block])) + 1:
            side = {names[k]: sorted(set(s.astype(str)[sample_cond == k]))[:4] for k in ("a", "b")}
            shown = "; ".join(f"{k}: {', '.join(v)}" for k, v in side.items())
            raise PseudobulkError(
                f"Covariate '{name}' is confounded with the condition ({shown}): the two cannot "
                "be separated, so it cannot be in the design.")
        base = np.column_stack([base, block])
        cols.append(block)
        used.append(name)
    C = np.column_stack(cols) if cols else np.zeros((len(is_a), 0))
    return C, meta, used, notes


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


# Cells per read of a counts matrix on disk (a large dataset's counts file: at most
# large_mode.MAX_READ_BYTES per read; 5,000 cells of 4,500 genes each are 180 MB).
SUM_ROW_BLOCK = 5_000


def _in_memory(M) -> bool:
    return sp.issparse(M) or isinstance(M, np.ndarray)


def sum_rows(M, rows: np.ndarray, codes: np.ndarray, n_samples: int) -> sp.csr_matrix:
    """``_sum_by_sample(M[rows], codes, n_samples)``. On disk (a large dataset's counts
    file, a backed matrix) the cells are read in spans of ``SUM_ROW_BLOCK`` rows, as they
    are stored, so only one span and the sums are ever held; the sums are the same
    (float64, added span by span in row order). ``rows`` ascending."""
    rows = np.asarray(rows, dtype=np.int64)
    if _in_memory(M):
        return _sum_by_sample(M if rows.size == M.shape[0] else M[rows], codes, n_samples)
    from scview.core.large_mode import LargeDatasetError

    total = sp.csr_matrix((n_samples, M.shape[1]), dtype=np.float64)
    step = SUM_ROW_BLOCK
    i = 0
    while i < rows.size:
        lo = int(rows[i])
        hi = min(lo + step, M.shape[0])
        j = int(np.searchsorted(rows, hi))
        try:
            block = M[lo:hi]
        except LargeDatasetError:
            if step == 1:
                raise
            step = max(1, step // 2)  # very deep cells: smaller spans
            continue
        block = sp.csr_matrix(block) if sp.issparse(block) else np.asarray(block)
        part = block[rows[i:j] - lo]
        total = total + _sum_by_sample(part, codes[i:j], n_samples)
        i = j
    return sp.csr_matrix(total)


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
    method: str = "voom",
    ruv_k: int = 0,
    covariates: pd.DataFrame | None = None,
) -> PseudobulkResult:
    """Condition A vs B on per-donor pseudobulk sums: limma-voom / voomByGroup's
    moderated t, edgeR QL's F-test, or DESeq2's Wald test (``method``), with ``ruv_k`` RUVg
    factors in the design.

    ``adata`` holds only the compared cells; ``in_a`` marks the A cells (the
    rest are B) and ``donors`` gives each cell's donor (missing = left out: a large
    dataset is passed whole, its other cells without a donor, and its counts file is
    read in blocks of cells).
    ``covariates``: per-sample obs columns of those cells (one column each) to
    put in the design before the condition.
    """
    if method not in METHODS:
        raise PseudobulkError(f"Unknown pseudobulk method '{method}' ({', '.join(METHODS)}).")
    if not 0 <= ruv_k <= MAX_RUV_K:
        raise PseudobulkError(f"RUV k must be 0 to {MAX_RUV_K}.")
    if covariates is not None and covariates.shape[1] > MAX_COVARIATES:
        raise PseudobulkError(f"At most {MAX_COVARIATES} covariates.")
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
    P = sum_rows(M, rows, codes, len(samples))

    ok = n_cells >= MIN_CELLS_PER_SAMPLE
    names = {"a": label_a, "b": label_b}
    sample_names = [f"{sample_donor[i]} ({names[sample_cond[i]]})" for i in range(len(samples))]
    cov_values = {}
    for name in (covariates.columns if covariates is not None else []):
        col = covariates[name].iloc[rows].reset_index(drop=True)
        cov_values[str(name)] = _sample_values(col, codes, len(samples), str(name), sample_names)
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
    cov_values = {k: v[ok] for k, v in cov_values.items()}
    genes_idx = _filter_genes(P, min(n_a, n_b))
    if len(genes_idx) == 0:
        raise PseudobulkError(
            f"No gene has at least {MIN_GENE_COUNT} counts in {min(n_a, n_b)} or more "
            "pseudobulk samples; there is too little signal to test."
        )

    # Neutral labels: the model never sees the user's group or donor names.
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

    from scview.core import voom as vm

    # The design as a matrix (for voom and RUV): intercept, donors when paired,
    # RUV factors, then A (=1) vs B, whose coefficient is the log2 fold change.
    C = counts_df.to_numpy(dtype=np.float64)
    is_a = (sample_cond == "a").astype(np.float64)
    nuisance = np.ones((len(C), 1))
    if paired:
        dummies = donor_codes[:, None] == np.arange(1, donor_codes.max() + 1)[None, :]
        nuisance = np.column_stack([nuisance, dummies.astype(np.float64)])
    cov_cols, cov_meta, cov_used, cov_notes = _covariate_design(
        cov_values, nuisance, is_a, sample_cond, names)
    nuisance = np.column_stack([nuisance, cov_cols])
    lib = C.sum(axis=1)
    try:
        eff_lib = lib * vm.norm_factors(C, "TMM", lib)
    except vm.VoomError as exc:
        if method != "deseq2" or ruv_k:
            raise PseudobulkError(str(exc)) from exc
        eff_lib = lib
    try:  # a side check: never the reason a DESeq2 run fails
        hetero = _heteroscedasticity(C, np.column_stack([nuisance, is_a]), sample_cond, eff_lib,
                                     names, method)
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("pseudobulk: variance check failed: %s", exc)
        hetero = None

    W = np.zeros((len(C), 0))
    ruv = None
    if ruv_k:
        W, ruv = _ruv(C, nuisance, is_a, eff_lib, sample_cond, sample_donor, n_cells_kept,
                      ruv_k, method, names, paired)
    X = np.column_stack([nuisance, W, is_a])
    if len(vm.full_rank_columns(X)) < X.shape[1] or X.shape[0] - X.shape[1] < 1:
        n_w = W.shape[1]
        raise PseudobulkError(
            f"{len(C)} pseudobulk samples leave no residual degrees of freedom for this design"
            + (f" with {n_w} RUV factor{'s' if n_w != 1 else ''}; use a smaller k." if n_w else ".")
        )
    w_names = [f"W{j + 1}" for j in range(W.shape[1])]
    # Neutral names in the formula (c0, c1 ...); the reported design uses the columns'.
    terms = w_names + (["donor"] if paired else []) + list(cov_meta) + ["condition"]
    shown = dict(zip(cov_meta, cov_used))

    if method == "deseq2":
        genes, stat = _run_deseq2(C, X, list(counts_df.columns))
    elif method == "edger_ql":
        genes, stat = _run_edger_ql(C, X, list(counts_df.columns))
    else:
        genes, stat = _run_voom(C, X, sample_cond if method == "voombygroup" else None,
                                eff_lib, list(counts_df.columns))
    return PseudobulkResult(
        genes=genes,
        n_samples_a=n_a,
        n_samples_b=n_b,
        n_cells_a=int(n_cells_kept[sample_cond == "a"].sum()),
        n_cells_b=int(n_cells_kept[sample_cond == "b"].sum()),
        n_genes_tested=len(genes_idx),
        design="~" + " + ".join(donor_key if t == "donor" else shown.get(t, t) for t in terms),
        paired=paired,
        counts_source=source,
        dropped_samples=dropped,
        stats=stat,
        method=method,
        ruv_k=int(W.shape[1]),
        heteroscedasticity=hetero,
        ruv=ruv,
        covariates=cov_used,
        covariate_notes=cov_notes,
    )


def _ranked(rows: list[tuple[str, float, float, float]], stat: dict
            ) -> list[tuple[str, float, float, float]]:
    """Rank like the Wilcoxon results (scanpy's score order): strongest up first,
    strongest down last, by the statistic, else by signed -log10 p. Callers take
    "the top genes" from the front of each side."""
    def _rank(row: tuple[str, float, float, float]) -> float:
        s = stat.get(row[0])
        if s is not None and np.isfinite(s):
            return float(s)
        return float(np.sign(row[1]) * -np.log10(max(row[2], 1e-300)))

    return sorted(rows, key=_rank, reverse=True)


def _run_deseq2(C: np.ndarray, X: np.ndarray, gene_names: list[str]
                ) -> tuple[list[tuple[str, float, float, float]], dict[str, float]]:
    """(gene, log2FC, p, adjusted p) per tested gene, strongest up first, and each
    gene's Wald statistic: DESeq2's Wald test of the design's last column
    (core/deseq2_vec.py, pydeseq2's steps vectorised over genes)."""
    from scview.core import deseq2_vec

    try:
        res = deseq2_vec.deseq2_wald(np.rint(C).astype(np.int64), X, coef=-1)
    except ValueError as exc:
        raise PseudobulkError(str(exc)) from exc
    out: list[tuple[str, float, float, float]] = []
    wald: dict[str, float] = {}
    for i, gene in enumerate(gene_names):
        lfc, p, padj, st = res.log2fc[i], res.pvalue[i], res.padj[i], res.stat[i]
        # NaN LFC/p: an all-zero or Cook's-outlier gene, not tested. NaN padj
        # alone: DESeq2's independent filtering left a low-mean gene out of the
        # FDR correction, which reads as "not significant".
        if not (np.isfinite(lfc) and np.isfinite(p)):
            continue
        out.append((gene, float(lfc), float(p), float(padj) if np.isfinite(padj) else 1.0))
        if np.isfinite(st):
            wald[gene] = float(st)
    return _ranked(out, wald), wald


def _run_edger_ql(C: np.ndarray, X: np.ndarray, gene_names: list[str]
                  ) -> tuple[list[tuple[str, float, float, float]], dict[str, float]]:
    """(gene, log2FC, p, adjusted p) per gene, strongest up first, and each gene's
    statistic as sign(log2FC) * sqrt(F) (F has one numerator df, so this is the
    t-like signed form the tables and GSEA rank by): edgeR v4's QL F-test of the
    design's last column (core/edger_ql.py)."""
    from scview.core import edger_ql

    try:
        res = edger_ql.glm_ql(C, X, coef=-1)
    except ValueError as exc:
        raise PseudobulkError(str(exc)) from exc
    ok = np.isfinite(res.logFC) & np.isfinite(res.pvalue)
    padj = np.where(np.isfinite(res.padj), res.padj, 1.0)
    rows = [(gene_names[i], float(res.logFC[i]), float(res.pvalue[i]), float(padj[i]))
            for i in np.flatnonzero(ok)]
    signed = np.sign(res.logFC) * np.sqrt(np.maximum(res.F, 0.0))
    stat = {gene_names[i]: float(signed[i]) for i in np.flatnonzero(ok & np.isfinite(signed))}
    return _ranked(rows, stat), stat


def _voom_test(C: np.ndarray, X: np.ndarray, group: np.ndarray | None, lib: np.ndarray):
    """The moderated t-test of the design's last column under voom (``group``
    None) or voomByGroup (a trend per value of ``group``)."""
    from scview.core import voom as vm

    E, w = vm.voom(C, X, group=group, lib_size=lib)
    return vm.ebayes(vm.lm_fit_weighted(E, X, w), X.shape[1] - 1)


def _run_voom(C: np.ndarray, X: np.ndarray, group: np.ndarray | None, lib: np.ndarray,
              gene_names: list[str]) -> tuple[list[tuple[str, float, float, float]], dict[str, float]]:
    """(gene, log2FC, p, adjusted p) per gene, strongest up first, and each gene's
    moderated t."""
    from scipy import stats

    from scview.core.voom import VoomError

    try:
        eb = _voom_test(C, X, group, lib)
    except VoomError as exc:
        raise PseudobulkError(str(exc)) from exc
    ok = np.isfinite(eb.logfc) & np.isfinite(eb.p)
    padj = np.ones(len(eb.p))
    if ok.any():
        padj[ok] = stats.false_discovery_control(eb.p[ok], method="bh")
    rows = [(gene_names[i], float(eb.logfc[i]), float(eb.p[i]), float(padj[i]))
            for i in np.flatnonzero(ok)]
    stat = {gene_names[i]: float(eb.t[i]) for i in np.flatnonzero(ok & np.isfinite(eb.t))}
    return _ranked(rows, stat), stat


def _heteroscedasticity(C: np.ndarray, X: np.ndarray, cond: np.ndarray, lib: np.ndarray,
                        names: dict[str, str], method: str) -> dict | None:
    """Each condition's voom trend (sqrt SD against mean log2 count, fitted on its
    own samples as voomByGroup does) and median residual variance of log2 CPM over
    the better-expressed half of the genes, where donor variation rather than
    counting noise sets it; flagged when one is HETERO_RATIO-fold the other."""
    from scview.core import voom as vm

    try:
        fit = vm.voom_fit(C, X, group=cond, lib_size=lib)
    except vm.VoomError:
        return None
    mean_e = fit.E.mean(axis=0)
    upper = mean_e >= np.median(mean_e)
    groups = []
    for key in ("a", "b"):
        trend, s2 = fit.trends.get(key), fit.s2.get(key)
        if trend is None or s2 is None:
            return None
        groups.append({"group": names[key], "n_samples": int((cond == key).sum()),
                       "median_variance": float(np.nanmedian(s2[upper])),
                       "trend": trend.points(TREND_POINTS)})
    hi, lo = sorted(groups, key=lambda g: -g["median_variance"])
    ratio = hi["median_variance"] / lo["median_variance"] if lo["median_variance"] > 0 else None
    flag = ratio is not None and ratio >= HETERO_RATIO
    if flag:
        tail = ("voomByGroup (in use) fits each group its own trend." if method == "voombygroup"
                else "Try voomByGroup: it fits each group its own trend, so "
                f"{lo['group']} does not take on {hi['group']}'s variance.")
        message = (f"{hi['group']} samples vary {ratio:.1f}x as much as {lo['group']} samples "
                   f"(median residual variance of log-CPM). {tail}")
    else:
        shown = f"{ratio:.1f}x" if ratio is not None else "no estimate"
        message = f"The two groups vary about as much ({shown}); one trend (voom) fits both."
    return {"groups": groups, "ratio": ratio, "higher": hi["group"], "flag": flag,
            "threshold": HETERO_RATIO, "message": message}


def _corr(a: np.ndarray, b: np.ndarray) -> float | None:
    if np.std(a) == 0 or np.std(b) == 0:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def _ruv(C, nuisance, is_a, lib, cond, donor, n_cells, k, method, names, paired):
    """RUVg factors (samples x k) from the genes a first pass found least
    associated with the condition (voomByGroup's for that method, else voom's:
    DESeq2 takes voom's too, which is far quicker), and how the factors line up
    with the samples' condition, depth, cell count and donor."""
    from scview.core import ruv as ruvm
    from scview.core.voom import VoomError

    first = np.column_stack([nuisance, is_a])
    try:
        eb = _voom_test(C, first, cond if method == "voombygroup" else None, lib)
    except VoomError as exc:
        raise PseudobulkError(str(exc)) from exc
    controls = ruvm.empirical_controls(eb.p)
    if len(controls) < 2:
        raise PseudobulkError("Too few tested genes to pick RUV control genes from.")
    W = ruvm.ruvg_factors(ruvm.log_normalised(C, lib), controls, k,
                          nuisance=nuisance if paired else None)
    log_depth = np.log10(np.maximum(C.sum(axis=1), 1))
    factors = []
    for j in range(W.shape[1]):
        w = W[:, j]
        r2 = None
        if paired:  # the share of the factor's variance between donors
            means = pd.Series(w).groupby(np.asarray(donor)).transform("mean").to_numpy()
            tot = float(((w - w.mean()) ** 2).sum())
            r2 = float(((means - w.mean()) ** 2).sum() / tot) if tot > 0 else None
        factors.append({"name": f"W{j + 1}", "r_condition": _corr(w, is_a),
                        "r_log_depth": _corr(w, log_depth),
                        "r_n_cells": _corr(w, n_cells.astype(np.float64)), "r2_donor": r2})
    follows = [f for f in factors
               if f["r_condition"] is not None and abs(f["r_condition"]) >= RUV_CONDITION_R]
    warning = None
    if follows:
        f = follows[0]
        warning = (f"{f['name']} follows the condition (r = {f['r_condition']:.2f}): RUV may "
                   "remove real signal along with the unwanted variation.")
    samples = [{"donor": str(donor[i]), "group": names[cond[i]], "n_cells": int(n_cells[i]),
                "log10_depth": float(log_depth[i]), "W": [float(v) for v in W[i]]}
               for i in range(len(C))]
    summary = {"k": int(W.shape[1]), "n_controls": int(len(controls)),
               "control_source": (f"the {len(controls):,} genes least associated with the "
                                  "condition in a first limma-voom pass"),
               "factors": factors, "samples": samples, "warning": warning}
    return W, summary


# --- The co-pilot's fact: the last pseudobulk run on a dataset ---

_last: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()
_last_lock = threading.Lock()


def method_caution(method: str, n_a: int, n_b: int) -> str | None:
    """The DESeq2 FDR caution for a run with fewer than DESEQ2_FDR_DONORS donors on
    its smaller side (None for voom, or for enough donors)."""
    n = min(n_a, n_b)
    if method != "deseq2" or n >= DESEQ2_FDR_DONORS:
        return None
    return (f"DESeq2 with {n} donors on the smaller side: on real donor splits it called "
            "more false genes than its FDR says (about 1 in 3 at 2 donors per side, 1 in 6 at 4, "
            "for a nominal 1 in 20). limma-voom held its FDR much better; prefer it here, or "
            "read DESeq2's padj as optimistic.")


def summary_params(res: PseudobulkResult, label: str) -> dict:
    """What a run's DE store entry keeps beside its comparison (its params), and
    what the co-pilot's line reads."""
    h = res.heteroscedasticity or {}
    return {
        "pb_method": res.method, "ruv_k": res.ruv_k, "pb_label": label,
        "n_sig": int(sum(1 for g in res.genes if g[3] < 0.05)), "n_tested": res.n_genes_tested,
        "n_samples_a": res.n_samples_a, "n_samples_b": res.n_samples_b,
        "hetero_ratio": h.get("ratio"), "hetero_higher": h.get("higher"),
        "hetero_flag": bool(h.get("flag")),
        "ruv_warning": (res.ruv or {}).get("warning"),
        "covariates": list(getattr(res, "covariates", None) or []),
    }


def remember(adaptor, params: dict) -> None:
    try:
        with _last_lock:
            _last[adaptor] = params
    except TypeError:  # not weak-referenceable (a test double)
        pass


def _last_params(adaptor) -> dict | None:
    try:
        with _last_lock:
            got = _last.get(adaptor)
    except TypeError:
        return None
    if got is not None:
        return got
    from scview.core import results_store

    dataset_id = results_store.dataset_id_of(adaptor)
    if dataset_id is None:
        return None
    for e in results_store.entries(dataset_id, "de", adaptor=adaptor):
        p = e.get("params") or {}
        if p.get("method") == "pseudobulk" and not e.get("stale") and p.get("pb_label"):
            return p
    return None


def grounding_lines(adaptor) -> tuple[list[str], str | None]:
    """For the co-pilot: the last pseudobulk DE run's method, hits and the
    per-group variance check."""
    p = _last_params(adaptor)
    if p is None:
        return [], None
    name = METHOD_NAMES.get(p.get("pb_method") or "deseq2", "DESeq2")
    ruv = f" + RUV k={p['ruv_k']}" if p.get("ruv_k") else ""
    ruv += "".join(f" + {c}" for c in p.get("covariates") or [])
    detail = (f"{p['pb_label']} ({name}{ruv}; {p.get('n_samples_a')} vs {p.get('n_samples_b')} "
              f"donor samples): {p.get('n_sig')} of {p.get('n_tested')} genes at adj. p < 0.05")
    lines = ["\n## Pseudobulk DE (last run; donors as replicates)", f"- {detail} [{TAG}]."]
    ratio = p.get("hetero_ratio")
    if ratio is not None:
        if p.get("hetero_flag"):
            used = "used" if p.get("pb_method") == "voombygroup" else "suggested"
            lines.append(f"- {p.get('hetero_higher')} samples vary {ratio:.1f}x as much as the other "
                         f"group (median residual variance); voomByGroup {used}.")
        else:
            lines.append(f"- The two groups vary about as much ({ratio:.1f}x); one voom trend fits.")
    if p.get("ruv_warning"):
        lines.append(f"- RUV caution: {p['ruv_warning']}")
    return lines, detail
