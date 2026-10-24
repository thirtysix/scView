"""Marker and DE tests by blocks of genes, equal to ``sc.tl.rank_genes_groups``.

Markers are computed per block of genes from a gene-major source: the store on disk
of a large dataset (``core/gene_store.py``), or the matrix in memory
(``core/matrix_genes.py``, which every other dataset's tests take through
``core/rank_genes.py``), and stored in exactly the structure scanpy 1.11 writes to
``uns`` (params, recarrays ordered by score, pts / pts_rest frames), so every markers
view keeps working. Each group against the rest (a subset of ``groups`` too), or
against a ``reference`` group: then, as scanpy does, ranked within the two groups'
cells only, one pass over the genes per group.

Wilcoxon rank-sum (scanpy's default ``tie_correct=False``, z-score ``scores``): in a
column only the stored values are sorted; every zero shares one average rank
(after any negative values), so a gene's rank sum per group is the sum of its
non-zero ranks in that group plus the zero rank times the group's zeros. Ranks are
half-integers, so the sums are exact in float64 in any order, and the z-scores,
p-values and BH-adjusted p-values come out bit for bit equal to scanpy's. The
order of genes is then made by scanpy's own ``_select_top_n`` on those identical
scores, so tied scores (every gene a group never expresses, for one) keep scanpy's
order too.

Welch t-test (``t-test``, and ``t-test_overestim_var``) from per-group sums and
sums of squares, through the same ``scipy.stats.ttest_ind_from_stats`` call.

Means feed ``logfoldchanges`` (and the t-test). A group's sums run over its cells
in cell order, as scanpy's sparse kernel does on one thread (it splits rows over
numba threads); the rest of the cells are the total minus the group. These columns
agree with scanpy to float precision, and bit for bit whenever the float64 sums of
the values are exact, as they were for float32 log values of Kang and paul15. On
float64 values the t-test can order near-tied genes differently, as scanpy does
between its own thread counts.

Memory: one block of genes (``max_entries`` stored values) plus a few arrays of
n_groups x n_genes, never the matrix.
"""

from __future__ import annotations

import logging
import warnings
from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd

from scview.core.gene_store import GeneStore, GeneStoreError

logger = logging.getLogger(__name__)

METHODS = ("wilcoxon", "t-test", "t-test_overestim_var")
# Stored values per block of genes: the working set is about 60 bytes per value.
MAX_ENTRIES = 4_000_000

Progress = Callable[[str, int, int], None]


class StaleGeneStore(GeneStoreError):
    """The gene store does not hold the matrix the markers would be ranked on."""


def _select_top_n(scores: np.ndarray, n_top: int) -> np.ndarray:
    # scanpy 1.11 tools/_rank_genes_groups._select_top_n, verbatim: the order of
    # tied scores comes from these two calls.
    n_from = scores.shape[0]
    reference_indices = np.arange(n_from, dtype=int)
    partition = np.argpartition(scores, -n_top)[-n_top:]
    partial_indices = np.argsort(scores[partition])[::-1]
    global_indices = reference_indices[partition][partial_indices]

    return global_indices


# ---------------------------------------------------------------------------
# per block


def _rank_sums_f32(gl, cnt, starts_of_gene, vals, code, nb, n_codes, n_cells):
    """Rank sums of the stored values per gene x code (nb, n_codes) for one block of
    float32 values, by sorting one uint64 key (gene | value | code) in place."""
    code_bits = max(1, int(n_codes - 1).bit_length())
    shift_v = np.uint64(code_bits)
    shift_g = np.uint64(32 + code_bits)
    b = vals.view(np.uint32)
    # IEEE float32 bits made unsigned-sortable: flip all bits of negatives, set the
    # sign bit of positives (zeros were dropped when the store was built).
    sortable = np.where(b >= np.uint32(0x80000000), ~b, b | np.uint32(0x80000000))
    key = gl.astype(np.uint64) << shift_g
    key |= sortable.astype(np.uint64) << shift_v
    del sortable
    key |= code.astype(np.uint64)
    key.sort()
    n = key.size
    gv = key >> shift_v  # gene and value: equal for tied values of a gene
    new = np.empty(n, dtype=bool)
    new[0] = True
    np.not_equal(gv[1:], gv[:-1], out=new[1:])
    starts = np.flatnonzero(new)
    del new
    ends = np.empty_like(starts)
    ends[:-1] = starts[1:]
    ends[-1] = n
    run_gene = (gv[starts] >> np.uint64(32)).astype(np.int64)
    positive = (gv[starts] & np.uint64(0xFFFFFFFF)) > np.uint64(0x80000000)
    del gv
    first = starts_of_gene[run_gene]
    # 1-based ranks a+1 .. b among the gene's stored values, averaged over the tie;
    # positive values rank after all the zeros.
    run_rank = 0.5 * ((starts - first) + 1 + (ends - first))
    run_rank += np.where(positive, n_cells - cnt[run_gene], 0)
    rank = np.repeat(run_rank, ends - starts)
    del run_rank, starts, ends, run_gene, positive, first
    slot = (key >> shift_g).astype(np.int64) * n_codes
    slot += (key & np.uint64((1 << code_bits) - 1)).astype(np.int64)
    del key
    sums = np.bincount(slot, weights=rank, minlength=nb * n_codes).reshape(nb, n_codes)
    return sums


def _rank_sums_any(gl, cnt, starts_of_gene, vals, code, nb, n_codes, n_cells):
    """``_rank_sums_f32`` for any value dtype (an indirect sort, slower)."""
    order = np.lexsort((vals, gl))
    g = gl[order]
    v = vals[order]
    c = code[order]
    del order
    n = g.size
    new = np.empty(n, dtype=bool)
    new[0] = True
    new[1:] = (g[1:] != g[:-1]) | (v[1:] != v[:-1])
    starts = np.flatnonzero(new)
    ends = np.empty_like(starts)
    ends[:-1] = starts[1:]
    ends[-1] = n
    run_gene = g[starts]
    first = starts_of_gene[run_gene]
    run_rank = 0.5 * ((starts - first) + 1 + (ends - first))
    run_rank += np.where(v[starts] > 0, n_cells - cnt[run_gene], 0)
    rank = np.repeat(run_rank, ends - starts)
    return np.bincount(g * n_codes + c, weights=rank, minlength=nb * n_codes).reshape(nb, n_codes)


class _Accumulator:
    """Per gene x group counts, sums, sums of squares and rank sums, block by block.

    ``n_cells`` counts the cells taking part; with ``exclude``, cells whose code is
    negative take no part at all (a reference comparison ranks only its two groups).
    ``square_dtype``: the dtype scanpy squares the values in for this matrix (its own,
    or float64 for a CSC matrix, whose column kernel casts first)."""

    def __init__(self, n_genes: int, n_codes: int, n_cells: int, *, ranks: bool,
                 exclude: bool = False, square_dtype: Any = None) -> None:
        self.n_codes = n_codes
        self.n_cells = n_cells
        self.exclude = exclude
        self.square_dtype = square_dtype
        self.nnz = np.zeros((n_genes, n_codes), dtype=np.int64)
        self.sums = np.zeros((n_genes, n_codes))
        self.sq = np.zeros((n_genes, n_codes))
        self.total = np.zeros(n_genes)
        self.total_sq = np.zeros(n_genes)
        self.nnz_total = np.zeros(n_genes, dtype=np.int64)
        self.ranks = np.zeros((n_genes, n_codes)) if ranks else None
        self.zero_rank = np.zeros(n_genes) if ranks else None
        self.all_integer = True

    def add(self, g0: int, g1: int, block: Any, codes: np.ndarray) -> None:
        nb = g1 - g0
        cnt = np.diff(block.indptr).astype(np.int64)
        vals = block.data
        if self.all_integer and block.nnz:  # over the whole matrix, as scanpy checks it
            self.all_integer = bool(np.all(vals == np.round(vals)) and np.all(vals >= 0))
        starts_of_gene = np.asarray(block.indptr[:-1], dtype=np.int64)
        if block.nnz:
            gl = np.repeat(np.arange(nb, dtype=np.int64), cnt)
            code = codes[block.indices]
            if self.exclude:
                keep = code >= 0
                if not keep.all():
                    gl, vals, code = gl[keep], vals[keep], code[keep]
                    cnt = np.bincount(gl, minlength=nb).astype(np.int64)
                    starts_of_gene = np.cumsum(cnt) - cnt
        self.nnz_total[g0:g1] = cnt
        if vals.size == 0:
            if self.ranks is not None:
                self.zero_rank[g0:g1] = 0.5 * (self.n_cells + 1)
            return
        slot = gl * self.n_codes + code
        size = nb * self.n_codes
        self.nnz[g0:g1] = np.bincount(slot, minlength=size).reshape(nb, self.n_codes)
        # bincount adds in input order: a group's cells in cell order, in float64
        self.sums[g0:g1] = np.bincount(slot, weights=vals, minlength=size).reshape(nb, self.n_codes)
        sq_vals = vals if self.square_dtype is None else vals.astype(self.square_dtype)
        square = sq_vals * sq_vals  # in the dtype scanpy's kernel squares it in
        del sq_vals
        self.sq[g0:g1] = np.bincount(slot, weights=square, minlength=size).reshape(nb, self.n_codes)
        self.total[g0:g1] = np.bincount(gl, weights=vals, minlength=nb)
        self.total_sq[g0:g1] = np.bincount(gl, weights=square, minlength=nb)
        del slot, square
        if self.ranks is None:
            return
        fn = (_rank_sums_f32 if vals.dtype == np.float32 and self.n_codes <= 1 << 12
              else _rank_sums_any)
        self.ranks[g0:g1] = fn(gl, cnt, starts_of_gene, vals, code, nb, self.n_codes, self.n_cells)
        negative = np.bincount(gl[vals < 0], minlength=nb)
        # the zeros take ranks negative+1 .. negative+zeros
        self.zero_rank[g0:g1] = 0.5 * (2 * negative + (self.n_cells - cnt) + 1)


def _mean_var(sums: np.ndarray, sq: np.ndarray, n: int) -> tuple[np.ndarray, np.ndarray]:
    """scanpy's ``_get_mean_var`` from sums: mean, unbiased variance."""
    mean = sums / n
    var = sq / n - mean**2
    if n != 1:
        var *= n / (n - 1)
    return mean, var


# ---------------------------------------------------------------------------
# public


def compute(store: GeneStore, labels: Any, **kwargs: Any) -> dict[str, Any]:
    """The ``uns`` entry ``sc.tl.rank_genes_groups`` would write for these labels
    (a categorical, one per store cell), ranked on the store's values.

    Keywords as ``_compute``: ``groupby``, ``method``, ``groups``, ``reference``,
    ``pts``, ``use_raw`` and ``layer`` (only recorded in ``params``), ``corr_method``,
    ``var_names`` (default the store's), ``log1p_base`` (``uns["log1p"]["base"]``),
    ``n_genes``, ``rankby_abs``, ``max_entries``, ``progress``."""
    return _compute(store, labels, **kwargs)[0]


def _select_groups(categories: pd.Index, groups: Any, reference: str) -> list[int]:
    """Positions in ``categories`` of the groups scanpy compares, in its order: every
    category, or ``groups`` as given with the reference appended when not among them."""
    if isinstance(groups, str) and groups == "all":
        return list(range(len(categories)))
    if isinstance(groups, (str, int)):
        raise ValueError("Specify a sequence of groups")
    names = list(groups)
    if names and isinstance(names[0], int):
        names = [str(n) for n in names]
    if reference != "rest" and reference not in set(names):
        names.append(reference)
    if not names:
        raise ValueError("No groups to compare.")
    order = []
    for name in names:
        hit = np.flatnonzero(categories.values == name)
        if hit.size == 0:
            raise ValueError(f"'{name}' is not one of the groups {categories.tolist()}.")
        order.append(int(hit[0]))
    return order


def _accumulate(store: Any, acc: _Accumulator, codes: np.ndarray, max_entries: int,
                progress: Progress | None, *, warn: bool = True) -> None:
    acc.square_dtype = getattr(store, "square_dtype", None)
    for g0, g1, block in store.blocks(max_entries):
        acc.add(g0, g1, block, codes)
        if progress is not None:
            progress("Marker genes", g1, store.n_genes)
    if warn and acc.all_integer and store.nnz:
        logger.warning("It seems you use rank_genes_groups on the raw count data. "
                       "Please logarithmize your data before calling rank_genes_groups.")


def _compute(
    store: GeneStore,
    labels: Any,
    *,
    groupby: str,
    method: str = "wilcoxon",
    groups: Any = "all",
    reference: str = "rest",
    pts: bool = False,
    use_raw: bool = False,
    layer: str | None = None,
    corr_method: str = "benjamini-hochberg",
    var_names: Any = None,
    log1p_base: float | None = None,
    n_genes: int | None = None,
    rankby_abs: bool = False,
    max_entries: int = MAX_ENTRIES,
    progress: Progress | None = None,
) -> tuple[dict[str, Any], np.ndarray]:
    """(the markers entry, the groups' means: len(groups) x n_genes, scanpy's order)."""
    if method not in METHODS:
        raise ValueError(f"Method must be one of {METHODS} for markers from a gene store.")
    if corr_method not in ("benjamini-hochberg", "bonferroni"):
        raise ValueError("Correction method must be one of {'benjamini-hochberg', 'bonferroni'}.")

    cat = labels.cat if isinstance(labels, pd.Series) else pd.Series(labels).cat
    categories = cat.categories
    codes = np.asarray(cat.codes, dtype=np.int64)
    if codes.shape[0] != store.n_cells:
        raise ValueError(f"{codes.shape[0]} labels for a gene store of {store.n_cells} cells")
    if reference != "rest" and reference not in categories:
        raise ValueError(f"reference = {reference} needs to be one of groupby = "
                         f"{categories.tolist()}.")
    order = _select_groups(categories, groups, reference)
    sizes = np.bincount(codes[codes >= 0], minlength=len(categories))
    invalid = [str(categories[c]) for c in order if sizes[c] < 2]
    if invalid:
        raise ValueError(f"Could not calculate statistics for groups {', '.join(invalid)} "
                         "since they only contain one sample.")
    var_names = store.var_names if var_names is None else pd.Index(var_names)
    if len(var_names) != store.n_genes:
        raise ValueError(f"{len(var_names)} gene names for a gene store of {store.n_genes} genes")

    kw = dict(method=method, pts=pts, max_entries=max_entries, progress=progress)
    if reference == "rest":
        results, means, means_rest, pts_in, pts_rest = _versus_rest(store, codes, order, sizes, **kw)
    else:
        iref = order.index(int(categories.get_loc(reference)))
        if len(order) < 2:
            raise ValueError("No group to compare with the reference.")
        results, means, means_rest, pts_in = _versus_reference(store, codes, order, sizes, iref,
                                                               **kw)
        pts_rest = None

    if log1p_base is not None:
        def expm1_func(x):
            return np.expm1(x * np.log(log1p_base))
    else:
        expm1_func = np.expm1

    groups_order = [categories[c] for c in order]
    n_genes_user = store.n_genes if n_genes is None or n_genes > store.n_genes else n_genes
    stats_df = _statistics(results, groups_order, var_names, means, means_rest,
                           n_genes_user=n_genes_user, rankby_abs=rankby_abs,
                           corr_method=corr_method, expm1_func=expm1_func)

    out: dict[str, Any] = {"params": dict(groupby=groupby, reference=reference, method=method,
                                          use_raw=use_raw, layer=layer, corr_method=corr_method)}
    if pts:  # filled while the results were generated
        groups_names = [str(name) for name in groups_order]
        out["pts"] = pd.DataFrame(pts_in.T, index=var_names, columns=groups_names)
        if pts_rest is not None:
            out["pts_rest"] = pd.DataFrame(pts_rest.T, index=var_names, columns=groups_names)
    stats_df.columns = stats_df.columns.swaplevel()
    dtypes = {"names": "O", "scores": "float32", "logfoldchanges": "float32",
              "pvals": "float64", "pvals_adj": "float64"}
    for col in stats_df.columns.levels[0]:
        out[col] = stats_df[col].to_records(index=False, column_dtypes=dtypes[col])
    return out, means


def _versus_rest(store, codes, order, sizes, *, method, pts, max_entries, progress):
    """Each group in ``order`` against every other cell (one pass over the genes):
    (results, means, means of the rest, pts, pts_rest), indexed by position in order."""
    from scipy import stats

    n_cells = int(store.n_cells)
    n_cats = len(sizes)
    codes = codes.copy()
    # unlabelled cells take the extra code n_cats: in every group's rest, in no group
    codes[codes < 0] = n_cats
    acc = _Accumulator(store.n_genes, n_cats + 1, n_cells, ranks=method == "wilcoxon")
    _accumulate(store, acc, codes, max_entries, progress)

    n_list = [int(sizes[c]) for c in order]
    shape = (len(order), store.n_genes)
    means, vars_ = np.zeros(shape), np.zeros(shape)
    means_rest, vars_rest = np.zeros(shape), np.zeros(shape)
    pts_in = np.zeros(shape) if pts else None
    pts_rest = np.zeros(shape) if pts else None
    for g, (c, n_g) in enumerate(zip(order, n_list)):
        means[g], vars_[g] = _mean_var(acc.sums[:, c], acc.sq[:, c], n_g)
        n_rest = n_cells - n_g
        means_rest[g], vars_rest[g] = _mean_var(acc.total - acc.sums[:, c],
                                                acc.total_sq - acc.sq[:, c], n_rest)
        if pts:
            pts_in[g] = acc.nnz[:, c] / n_g
            pts_rest[g] = (acc.nnz_total - acc.nnz[:, c]) / n_rest

    def results():
        if method == "wilcoxon":
            for g, (c, n_active) in enumerate(zip(order, n_list)):
                # scanpy's arithmetic, in its order (Python ints, then float)
                rank_sum = acc.ranks[:, c] + acc.zero_rank * (n_active - acc.nnz[:, c])
                std_dev = np.sqrt(1 * n_active * (n_cells - n_active) * (n_cells + 1) / 12.0)
                scores = (rank_sum - (n_active * (n_cells + 1) / 2.0)) / std_dev
                scores[np.isnan(scores)] = 0
                pvals = 2 * stats.distributions.norm.sf(np.abs(scores))
                yield g, scores, pvals
        else:
            for g, ns_group in enumerate(n_list):
                ns_other = n_cells - ns_group
                ns_rest = ns_other if method == "t-test" else ns_group
                scores, pvals = _welch(means[g], vars_[g], ns_group, means_rest[g],
                                       vars_rest[g], ns_rest)
                yield g, scores, pvals

    return results(), means, means_rest, pts_in, pts_rest


def _versus_reference(store, codes, order, sizes, iref, *, method, pts, max_entries, progress):
    """Each group in ``order`` but the reference (position ``iref``) against the
    reference alone, as scanpy's ``reference=`` mode: ranked within the two groups'
    cells, one pass over the genes per group. (results, means, means of the reference
    per group, pts), filled as the results are generated."""
    from scipy import stats

    ref = order[iref]
    m_active = int(sizes[ref])
    shape = (len(order), store.n_genes)
    means, vars_, means_rest = np.zeros(shape), np.zeros(shape), np.zeros(shape)
    pts_in = np.zeros(shape) if pts else None

    def results():
        first = True
        for g, c in enumerate(order):
            if g == iref:
                continue
            n_active = int(sizes[c])
            local = np.full(codes.shape[0], -1, dtype=np.int64)  # cells of neither: left out
            local[codes == c] = 0
            local[codes == ref] = 1
            acc = _Accumulator(store.n_genes, 2, n_active + m_active,
                               ranks=method == "wilcoxon", exclude=True)
            _accumulate(store, acc, local, max_entries, progress, warn=first)
            first = False
            means[g], vars_[g] = _mean_var(acc.sums[:, 0], acc.sq[:, 0], n_active)
            means[iref], vars_[iref] = _mean_var(acc.sums[:, 1], acc.sq[:, 1], m_active)
            means_rest[g] = means[iref]
            if pts:
                pts_in[g] = acc.nnz[:, 0] / n_active
                pts_in[iref] = acc.nnz[:, 1] / m_active
            if method == "wilcoxon":
                rank_sum = acc.ranks[:, 0] + acc.zero_rank * (n_active - acc.nnz[:, 0])
                # scanpy's reference-mode arithmetic, in its order
                std_dev = np.sqrt(1 * n_active * m_active * (n_active + m_active + 1) / 12.0)
                scores = (rank_sum - (n_active * ((n_active + m_active + 1) / 2.0))) / std_dev
                scores[np.isnan(scores)] = 0
                pvals = 2 * stats.distributions.norm.sf(np.abs(scores))
            else:
                ns_rest = m_active if method == "t-test" else n_active
                scores, pvals = _welch(means[g], vars_[g], n_active, means[iref], vars_[iref],
                                       ns_rest)
            yield g, scores, pvals

    return results(), means, means_rest, pts_in


def _welch(mean1, var1, n1, mean2, var2, n2):
    """scanpy's t-test call and its NaN handling."""
    from scipy import stats

    with np.errstate(invalid="ignore"):
        scores, pvals = stats.ttest_ind_from_stats(
            mean1=mean1, std1=np.sqrt(var1), nobs1=n1,
            mean2=mean2, std2=np.sqrt(var2), nobs2=n2,
            equal_var=False,
        )
    scores[np.isnan(scores)] = 0
    pvals[np.isnan(pvals)] = 1
    return scores, pvals


def _statistics(results, groups_order, var_names, means, means_rest, *, n_genes_user,
                rankby_abs, corr_method, expm1_func) -> pd.DataFrame:
    """scanpy's ``_RankGenes.compute_statistics`` table, built the same way (column
    by column, which pandas calls fragmented: its warning is silenced)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", pd.errors.PerformanceWarning)
        return _statistics_table(results, groups_order, var_names, means, means_rest,
                                 n_genes_user=n_genes_user, rankby_abs=rankby_abs,
                                 corr_method=corr_method, expm1_func=expm1_func)


def _statistics_table(results, groups_order, var_names, means, means_rest, *, n_genes_user,
                      rankby_abs, corr_method, expm1_func) -> pd.DataFrame:
    stats_df = None
    n_genes = len(var_names)
    for group_index, scores, pvals in results:
        group_name = str(groups_order[group_index])
        scores_sort = np.abs(scores) if rankby_abs else scores
        global_indices = _select_top_n(scores_sort, n_genes_user)
        if stats_df is None:
            idx = pd.MultiIndex.from_tuples([(group_name, "names")])
            stats_df = pd.DataFrame(columns=idx)
        stats_df[group_name, "names"] = var_names[global_indices]
        stats_df[group_name, "scores"] = scores[global_indices]
        stats_df[group_name, "pvals"] = pvals[global_indices]
        if corr_method == "benjamini-hochberg":
            from statsmodels.stats.multitest import multipletests

            pvals[np.isnan(pvals)] = 1
            _, pvals_adj, _, _ = multipletests(pvals, alpha=0.05, method="fdr_bh")
        else:
            pvals_adj = np.minimum(pvals * n_genes, 1.0)
        stats_df[group_name, "pvals_adj"] = pvals_adj[global_indices]
        foldchanges = (expm1_func(means[group_index]) + 1e-9) / (
            expm1_func(means_rest[group_index]) + 1e-9
        )
        stats_df[group_name, "logfoldchanges"] = np.log2(foldchanges[global_indices])
    return stats_df


def rank_genes_groups(
    adata: Any,
    groupby: str,
    store: Any,
    *,
    method: str = "wilcoxon",
    groups: Any = "all",
    reference: str = "rest",
    pts: bool = False,
    use_raw: bool | None = None,
    layer: str | None = None,
    key_added: str | None = None,
    corr_method: str = "benjamini-hochberg",
    n_genes: int | None = None,
    rankby_abs: bool = False,
    tie_correct: bool = False,
    specificity: bool = False,
    check_store: bool = True,
    max_entries: int = MAX_ENTRIES,
    progress: Progress | None = None,
    mask_var: Any = None,
) -> None:
    """``sc.tl.rank_genes_groups(adata, groupby, method=..., pts=...)`` with the values
    read from ``store`` (a ``GeneStore``, or a ``matrix_genes.MatrixGenes`` over the
    matrix in memory) instead of the matrix; writes ``adata.uns[key_added]``.

    ``use_raw`` defaults as in scanpy (``.raw`` when present), and a gene store must hold
    that matrix (``store.mismatch``, unless ``check_store=False``), else
    ``StaleGeneStore``. ``groups`` and ``reference`` as scanpy's. ``specificity`` also
    stores the tau frame
    ``marker_specificity.attach`` would add, from the group means computed here
    (saving it a pass over the matrix). ``mask_var``: the genes ``store`` holds, as a
    boolean mask over the matrix's (``matrix_genes.GeneSubset``; scanpy's ``mask_var``)."""
    if use_raw is None:
        use_raw = adata.raw is not None
    elif use_raw is True and adata.raw is None:
        raise ValueError("Received `use_raw=True`, but `adata.raw` is empty.")
    if layer is not None and use_raw:
        raise ValueError("Cannot specify `layer` and have `use_raw=True`.")
    if tie_correct:
        raise ValueError("tie_correct=True is not supported for markers from a gene store.")
    adata.strings_to_categoricals()  # what scanpy's sanitize_anndata does first
    source = f"layers/{layer}" if layer is not None else ("raw" if use_raw else "X")
    if check_store and hasattr(store, "mismatch"):
        reason = store.mismatch(adata, source=source)
        if reason is not None:
            raise StaleGeneStore(f"The gene store does not hold the {source} matrix: {reason}")
    var_names = adata.raw.var_names if use_raw else adata.var_names
    if mask_var is not None:
        var_names = var_names[np.asarray(mask_var, dtype=bool)]
    base = adata.uns.get("log1p", {}).get("base")
    result, means = _compute(
        store, adata.obs[groupby], groupby=groupby, method=method, groups=groups,
        reference=reference, pts=pts, use_raw=use_raw,
        layer=layer, corr_method=corr_method, var_names=var_names, log1p_base=base,
        n_genes=n_genes, rankby_abs=rankby_abs, max_entries=max_entries, progress=progress,
    )
    if specificity:
        from scview.core import marker_specificity

        try:
            index = pd.Index([str(g) for g in var_names])
            result[marker_specificity.SPECIFICITY_FIELD] = pd.DataFrame(
                {"tau": marker_specificity.tau(means).astype(np.float32)}, index=index)
        except Exception as exc:  # never fail the markers over their extra columns
            logger.warning("Marker specificity for '%s' failed: %s", groupby, exc)
    adata.uns[key_added or "rank_genes_groups"] = result
