"""Pseudobulk differential expression across donors (DESeq2 via pydeseq2, or
limma-voom / voomByGroup via core/voom.py), optionally with RUV factors.

A cell-level test treats every cell as an independent replicate, so when two
conditions also differ by donor its p-values are far too small. Here each
donor's raw counts are summed per condition into one pseudobulk sample, and
DESeq2's negative-binomial Wald test compares those samples: the replicate is
the donor, not the cell.

When every donor contributes a sample to both conditions (the usual
stimulated-vs-control layout), the design is paired, ``~donor + condition``, so
donor-to-donor baseline differences are absorbed; otherwise it is ``~condition``.

``method``: ``deseq2`` (default), ``voom`` (TMM, one mean-variance trend, moderated
t) or ``voombygroup`` (a trend per condition, for when one condition's donors vary
more). Whatever the method, each condition's own trend and residual variance are
compared (``heteroscedasticity``). ``ruv_k`` > 0 adds that many RUVg factors from
empirical control genes (core/ruv.py) to the design.
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
# Compute bounds: DESeq2 fits one model per gene, so cost is samples x genes.
MAX_SAMPLES = 300
MAX_GENES = 20_000
# Worker processes for pydeseq2's per-gene fits. One keeps it in-process, which
# also keeps a shared laptop or a one-worker server from being saturated.
N_CPUS = 1
METHODS = ("deseq2", "voom", "voombygroup")
METHOD_NAMES = {"deseq2": "DESeq2", "voom": "limma-voom", "voombygroup": "voomByGroup"}
MAX_RUV_K = 3
# One condition's median residual variance this many times the other's flags
# heteroscedastic groups (voomByGroup then fits each its own trend).
HETERO_RATIO = 2.0
# Points per group trend sent for the plot.
TREND_POINTS = 60
# A RUV factor correlated with the condition at least this much may take signal.
RUV_CONDITION_R = 0.5
TAG = "result:pseudobulk"  # the co-pilot's tag for the last run


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
    # gene -> Wald statistic (DESeq2) or moderated t (voom)
    stats: dict[str, float] = field(default_factory=dict)
    method: str = "deseq2"
    ruv_k: int = 0
    heteroscedasticity: dict | None = None
    ruv: dict | None = None


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
    method: str = "deseq2",
    ruv_k: int = 0,
) -> PseudobulkResult:
    """Condition A vs B on per-donor pseudobulk sums: DESeq2's Wald test, or
    limma-voom / voomByGroup's moderated t (``method``), with ``ruv_k`` RUVg
    factors in the design.

    ``adata`` holds only the compared cells; ``in_a`` marks the A cells (the
    rest are B) and ``donors`` gives each cell's donor (missing = left out).
    """
    if method not in METHODS:
        raise PseudobulkError(f"Unknown pseudobulk method '{method}' ({', '.join(METHODS)}).")
    if not 0 <= ruv_k <= MAX_RUV_K:
        raise PseudobulkError(f"RUV k must be 0 to {MAX_RUV_K}.")
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

    from scview.core import voom as vm

    # The design as a matrix (for voom and RUV): intercept, donors when paired,
    # RUV factors, then A (=1) vs B, whose coefficient is the log2 fold change.
    C = counts_df.to_numpy(dtype=np.float64)
    is_a = (sample_cond == "a").astype(np.float64)
    nuisance = np.ones((len(C), 1))
    if paired:
        dummies = donor_codes[:, None] == np.arange(1, donor_codes.max() + 1)[None, :]
        nuisance = np.column_stack([nuisance, dummies.astype(np.float64)])
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
    design = "~" + " + ".join(w_names + (["donor"] if paired else []) + ["condition"])

    if method == "deseq2":
        for j, wn in enumerate(w_names):
            meta[wn] = W[:, j]
        genes, stat = _run_deseq2(counts_df, meta, design)
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
        design=design.replace("donor", donor_key),
        paired=paired,
        counts_source=source,
        dropped_samples=dropped,
        stats=stat,
        method=method,
        ruv_k=int(W.shape[1]),
        heteroscedasticity=hetero,
        ruv=ruv,
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
    # DESeq2 returns genes in input (genome) order.
    stat = res["stat"].to_dict() if "stat" in res else {}
    wald = {str(g): float(s) for g, s in stat.items() if s is not None and np.isfinite(s)}
    return _ranked(out, stat), wald


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
