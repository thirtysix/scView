"""Cell-cell communication compared between two conditions (Kang: CTRL vs STIM).

The ligand-receptor scoring of ``core/ccc.py`` runs once per condition on the
SAME groups, resource and genes, so the two tables differ only in the cells:

- groups: a group with fewer than ``min_cells`` cells in either condition is
  dropped from both (and listed), so every sender and receiver is in both runs;
- genes: the interactions are found once on both conditions' cells, so a
  ligand silent in A (an interferon-induced chemokine in CTRL) still has its
  pairs considered there, where they fail the ``expr_prop`` filter and count as
  gained in B, instead of being left out of A's run altogether;
- the permutations use the same seed in both runs.

Per (sender, receiver, ligand, receptor): the ligand-receptor mean (magnitude,
``lr_means``) and the two aggregate ranks in A and in B, the difference B - A
and a call. An interaction "counts" in a condition when its specificity rank is
at most ``ccc.SPECIFIC`` there (as the single run's pair summary counts it):

- gained: counts in B only, and its LR mean at least ``FOLD`` x higher in B;
- lost: counts in A only, and its LR mean at least ``FOLD`` x higher in A;
- changed: counts in both, LR means ``FOLD`` x apart either way;
- unchanged: the rest (a specificity rank crossing 0.05 with little change in
  expression is noise, not a gain).

Rows that count in neither condition are left out. Per sender-receiver pair the
specific interactions' count and summed LR mean in A and B and their difference
are CellChat's differential number and strength of interactions (Jin et al.
2021, PMID 33597522).

Significance is replicate-aware or absent. With a donor / sample column holding
at least 2 donors in each condition (``donor_expression.detect_sample_key``),
each donor's pseudobulk (mean log-normalised expression of the group's cells in
that donor and condition, at least ``MIN_SAMPLE_CELLS`` cells) gives one
ligand mean x receptor mean product per (sender, receiver) and interaction;
A vs B is tested on those, a Wilcoxon signed-rank test when the same donors
were sampled in both conditions (``paired_design``), else Welch's t-test, and
Benjamini-Hochberg across the tested interactions. Without replicates there is
no test: cells are not independent replicates, and a p-value from them would
measure the cell count, not the condition.
"""

from __future__ import annotations

import threading
import warnings
import weakref
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy import stats

from scview.core import ccc, paired_design
from scview.core.donor_expression import detect_sample_key

FOLD = 1.5  # LR means this far apart: a gain, loss or change
MIN_CELLS = 10  # per group and condition; fewer and the group leaves both runs
MIN_SAMPLE_CELLS = 5  # per donor x condition x group pseudobulk
MIN_DONORS = 2  # per condition, for any test
SHARE_FOLD = 2.0  # a group's share of the cells this far apart: composition note
KEEP = 4

CALLS = ("gained", "lost", "changed", "unchanged")


class CompareError(ccc.CCCError):
    """The two conditions cannot be compared (a 4xx)."""


@dataclass
class CompareResult:
    id: str
    dataset_id: str
    groupby: str
    condition: str
    a: str
    b: str
    resource: str
    species: str
    matrix_source: str
    groups: list[str]
    sizes_a: list[int]
    sizes_b: list[int]
    dropped_groups: dict[str, dict[str, int]]  # group -> {a: cells, b: cells}
    expr_prop: float
    n_perms: int
    min_cells: int
    n_interactions: int  # interactions scorable in both conditions
    table: pd.DataFrame  # one row per compared interaction, biggest change first
    counts_a: np.ndarray
    counts_b: np.ndarray
    strength_a: np.ndarray
    strength_b: np.ndarray
    replicates: dict  # sample_key, paired, test, n_donors, reason, caution
    composition: list[dict]  # per group: share in A and B, fold, flagged
    elapsed_seconds: float = 0.0
    donor_values: dict = field(default_factory=dict)  # "a" / "b": rows x donors arrays, "donors_a/b"

    @property
    def n_cells(self) -> int:
        return int(sum(self.sizes_a) + sum(self.sizes_b))


def _labels(s: pd.Series) -> pd.Series:
    return s.astype(str).where(s.notna())


def _colsum_nonzero(M) -> np.ndarray:
    if sp.issparse(M):
        return np.asarray(M.sum(axis=0, dtype=np.float64)).ravel() != 0
    return np.asarray(M, dtype=np.float64).sum(axis=0) != 0


def shared_groups(labels: pd.Series, cond: pd.Series, a: str, b: str, min_cells: int
                  ) -> tuple[list[str], dict[str, dict[str, int]]]:
    """Groups with ``min_cells`` or more cells in both conditions (in the column's
    order), and the dropped ones with their cells per condition."""
    lab, c = _labels(labels), _labels(cond)
    tab = pd.crosstab(lab, c)
    na = tab[a] if a in tab.columns else pd.Series(0, index=tab.index)
    nb = tab[b] if b in tab.columns else pd.Series(0, index=tab.index)
    ok = (na >= min_cells) & (nb >= min_cells)
    present = set(tab.index[ok])
    if isinstance(labels.dtype, pd.CategoricalDtype):
        order = [str(g) for g in labels.cat.categories if str(g) in present]
    else:
        order = sorted(present)
    dropped = {str(g): {"a": int(na[g]), "b": int(nb[g])} for g in tab.index[~ok]}
    return order, dropped


def split_prepared(M, genes, labels: pd.Series, in_a: np.ndarray, resource: pd.DataFrame
                   ) -> tuple[ccc.Prepared, ccc.Prepared]:
    """One preparation on both conditions' cells (so one set of groups, genes and
    interactions), split into a run per condition with that condition's own
    group statistics. ``M`` and ``labels`` hold both conditions' cells, every
    label one of the shared groups; ``in_a`` marks condition A's cells. A gene
    silent in one condition stays in: its pairs score 0 there (a gain or loss),
    where a run on that condition alone would never have listed them."""
    prep = ccc.prepare(M, genes, labels, resource, min_cells=1)
    out = []
    for mask in (in_a, ~in_a):
        Mc = M[mask]
        expressed = _colsum_nonzero(Mc)
        n_expr = max(1, int(expressed.sum()))
        total = float(Mc.sum(dtype=np.float64)) if sp.issparse(Mc) else float(np.asarray(Mc).sum(dtype=np.float64))
        X = prep.X[mask]
        codes = prep.codes[mask]
        st = ccc.group_stats(X, codes, prep.stats.groups, mat_mean=total / (Mc.shape[0] * n_expr),
                             n_genes=n_expr)
        st.genes = prep.stats.genes
        out.append(ccc.Prepared(X=X, codes=codes, stats=st, interactions=prep.interactions,
                                n_resource=prep.n_resource, n_resource_genes_found=prep.n_resource_genes_found))
    return out[0], out[1]


def _side_min(means: np.ndarray, idx_lists: list[list[int]]) -> np.ndarray:
    """Interactions x groups: the least expressed subunit's mean (a complex's value)."""
    out = np.empty((len(idx_lists), means.shape[0]))
    for i, idx in enumerate(idx_lists):
        out[i] = means[:, idx].min(axis=1)
    return out


def _lr_means(prep: ccc.Prepared, ints: np.ndarray, s: np.ndarray, t: np.ndarray) -> np.ndarray:
    L = _side_min(prep.stats.means, prep.interactions["lig_idx"].tolist())
    R = _side_min(prep.stats.means, prep.interactions["rec_idx"].tolist())
    lm, rm = L[ints, s], R[ints, t]
    out = (lm + rm) / 2.0
    out[(lm == 0) | (rm == 0)] = 0.0
    return out


def call_rows(spec_a: np.ndarray, spec_b: np.ndarray, lr_a: np.ndarray, lr_b: np.ndarray,
              fold: float = FOLD) -> np.ndarray:
    """gained / lost / changed / unchanged per row (see the module docstring)."""
    up = lr_b >= fold * lr_a
    down = lr_a >= fold * lr_b
    up &= lr_b > 0
    down &= lr_a > 0
    out = np.full(len(spec_a), "unchanged", dtype=object)
    out[~spec_a & spec_b & up] = "gained"
    out[spec_a & ~spec_b & down] = "lost"
    out[spec_a & spec_b & (up | down)] = "changed"
    return out


def compare_tables(prep_a: ccc.Prepared, prep_b: ccc.Prepared, tab_a: pd.DataFrame,
                   tab_b: pd.DataFrame, fold: float = FOLD) -> pd.DataFrame:
    """One row per (source, target, interaction) counted in either condition."""
    keys = ["source", "target", "ligand_complex", "receptor_complex"]
    cols = ["magnitude_rank", "specificity_rank", "lr_means"]
    ka = tab_a[keys + cols]
    kb = tab_b[keys + cols]
    m = ka.merge(kb, on=keys, how="outer", suffixes=("_a", "_b"))
    spec_a = (m["specificity_rank_a"] <= ccc.SPECIFIC).to_numpy()
    spec_b = (m["specificity_rank_b"] <= ccc.SPECIFIC).to_numpy()
    m = m[spec_a | spec_b].reset_index(drop=True)
    spec_a = (m["specificity_rank_a"] <= ccc.SPECIFIC).to_numpy()
    spec_b = (m["specificity_rank_b"] <= ccc.SPECIFIC).to_numpy()
    # A row not scored in one condition (under expr_prop there) still has an LR
    # mean there: the same groups and genes, so it is read off that run's means.
    gidx = pd.Index(prep_a.stats.groups)
    ints = pd.MultiIndex.from_frame(prep_a.interactions[["ligand_complex", "receptor_complex"]])
    ii = ints.get_indexer(pd.MultiIndex.from_frame(m[["ligand_complex", "receptor_complex"]]))
    s, t = gidx.get_indexer(m["source"]), gidx.get_indexer(m["target"])
    lr_a = _lr_means(prep_a, ii, s, t)
    lr_b = _lr_means(prep_b, ii, s, t)
    out = pd.DataFrame({
        "source": m["source"], "target": m["target"],
        "ligand_complex": m["ligand_complex"], "receptor_complex": m["receptor_complex"],
        "lr_means_a": lr_a, "lr_means_b": lr_b, "delta": lr_b - lr_a,
        "log2_fold": np.log2((lr_b + 0.01) / (lr_a + 0.01)),
        "magnitude_rank_a": m["magnitude_rank_a"], "magnitude_rank_b": m["magnitude_rank_b"],
        "specificity_rank_a": m["specificity_rank_a"], "specificity_rank_b": m["specificity_rank_b"],
        "call": call_rows(spec_a, spec_b, lr_a, lr_b, fold),
        "_int": ii, "_s": s, "_t": t,
    })
    return out


def composition(groups: list[str], sizes_a, sizes_b, share_fold: float = SHARE_FOLD) -> list[dict]:
    """Each group's share of the compared cells in A and B, and whether it moved
    ``share_fold`` x or more (its pairs then get a note)."""
    sa = np.asarray(sizes_a, dtype=np.float64)
    sb = np.asarray(sizes_b, dtype=np.float64)
    fa, fb = sa / sa.sum(), sb / sb.sum()
    fold = fb / fa
    return [{"group": g, "share_a": float(x), "share_b": float(y), "fold": float(f),
             "flagged": bool(f >= share_fold or f <= 1 / share_fold)}
            for g, x, y, f in zip(groups, fa, fb, fold)]


def _bh(p: np.ndarray) -> np.ndarray:
    out = np.full(p.shape, np.nan)
    ok = np.isfinite(p)
    if ok.any():
        out[ok] = stats.false_discovery_control(p[ok], method="bh")
    return out


def donor_products(M, genes, entity: list[str], prep: ccc.Prepared, labels: pd.Series,
                   donors: pd.Series, cells: np.ndarray, groups: list[str], rows: pd.DataFrame,
                   min_cells: int = MIN_SAMPLE_CELLS) -> tuple[np.ndarray, list[str], np.ndarray]:
    """Rows x donors: ligand mean (sender) x receptor mean (receiver) on each
    donor's pseudobulk among ``cells`` (NaN where either sample has too few
    cells), the donors, and rows x donors: the smaller of the two samples' cells."""
    names = pd.Index(np.asarray(genes, dtype=str))
    first = np.flatnonzero(~names.duplicated())
    cols = first[pd.Index(names[first]).get_indexer(entity)]
    lab = _labels(labels)[cells]
    don = _labels(donors)[cells]
    ok = (lab.isin(groups) & don.notna()).to_numpy()
    idx = np.flatnonzero(cells)[ok]
    donor_names = sorted(set(don[ok]))
    g = pd.Index(groups).get_indexer(lab[ok])
    d = pd.Index(donor_names).get_indexer(don[ok])
    n_g, n_d = len(groups), len(donor_names)
    code = d * n_g + g
    sub = M[idx][:, cols]
    X = sp.csr_matrix(sub, dtype=np.float64) if sp.issparse(sub) else np.asarray(sub, dtype=np.float64)
    onehot = sp.csr_matrix((np.ones(len(code)), (code, np.arange(len(code)))), shape=(n_d * n_g, len(code)))
    sums = onehot @ X
    sums = np.asarray(sums.todense()) if sp.issparse(sums) else np.asarray(sums)
    size = np.bincount(code, minlength=n_d * n_g).astype(np.float64)
    with np.errstate(invalid="ignore", divide="ignore"):
        means = sums / size[:, None]
    means[size < min_cells] = np.nan
    means = means.reshape(n_d, n_g, -1)  # donors x groups x entity genes
    lig_idx = prep.interactions["lig_idx"].tolist()
    rec_idx = prep.interactions["rec_idx"].tolist()
    ii, s, t = rows["_int"].to_numpy(), rows["_s"].to_numpy(), rows["_t"].to_numpy()
    out = np.full((len(rows), n_d), np.nan)
    for k in range(n_d):
        L = _side_min(means[k], lig_idx)  # interactions x groups (NaN for a small sample)
        R = _side_min(means[k], rec_idx)
        out[:, k] = L[ii, s] * R[ii, t]
    size = size.reshape(n_d, n_g)
    n_cells = np.minimum(size[:, s], size[:, t]).T.astype(np.int64)  # rows x donors: the smaller sample
    return out, donor_names, n_cells


def replicate_tests(va: np.ndarray, vb: np.ndarray, donors_a: list[str], donors_b: list[str],
                    paired: bool) -> tuple[np.ndarray, np.ndarray, str]:
    """Per row: (p, donors used, test name). Paired: Wilcoxon signed-rank on the
    donors with a value in both; else Welch's t on each condition's donors."""
    n = va.shape[0]
    p = np.full(n, np.nan)
    used = np.zeros(n, dtype=int)
    if paired:
        both = sorted(set(donors_a) & set(donors_b))
        ia = [donors_a.index(x) for x in both]
        ib = [donors_b.index(x) for x in both]
        A, B = va[:, ia], vb[:, ib]
        for r in range(n):
            ok = np.isfinite(A[r]) & np.isfinite(B[r])
            used[r] = int(ok.sum())
            if used[r] >= paired_design.MIN_PAIRS_TEST:
                w = paired_design.paired_test(A[r, ok], B[r, ok])["wilcoxon_p"]
                p[r] = np.nan if w is None else w
        return p, used, "Wilcoxon signed-rank (paired donors)"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for r in range(n):
            a, b = va[r][np.isfinite(va[r])], vb[r][np.isfinite(vb[r])]
            used[r] = len(a) + len(b)
            if len(a) >= MIN_DONORS and len(b) >= MIN_DONORS and (a.std() > 0 or b.std() > 0
                                                                   or a.mean() != b.mean()):
                v = stats.ttest_ind(b, a, equal_var=False).pvalue
                p[r] = v if np.isfinite(v) else np.nan
    return p, used, "Welch's t-test (donors)"


def compare(M, genes, obs: pd.DataFrame, groupby: str, condition: str, a: str, b: str,
            resource: pd.DataFrame, expr_prop: float = ccc.EXPR_PROP, n_perms: int = ccc.N_PERMS,
            min_cells: int = MIN_CELLS, sample_key: str | None = "auto",
            progress: Callable[[str, int, int], None] | None = None,
            stop: Callable[[], bool] | None = None) -> dict:
    """Both runs, the compared table, the pair summaries, the replicate tests and
    the composition notes, as the parts of a ``CompareResult``."""
    if groupby == condition:
        raise CompareError("The groups and the condition must be different columns.")
    if a == b:
        raise CompareError("Choose two different conditions.")
    labels, cond = obs[groupby], _labels(obs[condition])
    groups, dropped = shared_groups(labels, cond, a, b, min_cells)
    if len(groups) < 2:
        raise CompareError(f"Fewer than 2 '{groupby}' groups have {min_cells} or more cells in both "
                           f"{a} and {b}.")
    if len(groups) > ccc.MAX_GROUPS:
        raise CompareError(f"{len(groups)} groups: communication is scored between at most {ccc.MAX_GROUPS}.")
    in_groups = _labels(labels).isin(groups).to_numpy()
    mask_a = in_groups & (cond == a).to_numpy()
    mask_b = in_groups & (cond == b).to_numpy()
    if sp.issparse(M):
        M = M.tocsr()
    both = mask_a | mask_b
    cat = pd.Series(pd.Categorical(_labels(labels)[both], categories=groups))
    pa, pb = split_prepared(M[both], genes, cat, mask_a[both], resource)
    preps, tabs = {"a": pa, "b": pb}, {}
    for key, name in (("a", a), ("b", b)):
        prep = preps[key]
        cb = None if progress is None else (lambda d, n, name=name: progress(name, d, n))
        try:
            tabs[key] = ccc.score(prep, expr_prop=expr_prop, n_perms=n_perms, progress=cb, stop=stop)
        except ccc.CCCError:
            tabs[key] = pd.DataFrame(columns=["source", "target", "ligand_complex", "receptor_complex",
                                              "ligand", "receptor", "magnitude_rank",
                                              "specificity_rank", "lr_means"])
        if stop is not None and stop():
            raise InterruptedError("stopped")
    table = compare_tables(pa, pb, tabs["a"], tabs["b"])
    counts_a, strength_a = ccc.pair_summary(tabs["a"], groups)
    counts_b, strength_b = ccc.pair_summary(tabs["b"], groups)

    # Replicates
    rep: dict = {"sample_key": None, "paired": False, "test": None, "n_donors": {a: 0, b: 0},
                 "n_pairs": 0, "reason": None, "caution": None}
    donor_values: dict = {}
    table["p_value"] = np.nan
    table["p_adj"] = np.nan
    table["n_donors"] = 0
    both = mask_a | mask_b
    key = detect_sample_key(obs[both], groupby, condition) if sample_key == "auto" else sample_key
    if key is not None and key not in obs.columns:
        raise CompareError(f"Column '{key}' not found.")
    if key is None:
        rep["reason"] = ("No donor or sample column: with no replicates there is no test. The differences "
                         "are shown without p-values (cells are not independent replicates).")
    else:
        don = _labels(obs[key])
        n_a = int(don[mask_a].nunique())
        n_b = int(don[mask_b].nunique())
        rep.update(sample_key=key, n_donors={a: n_a, b: n_b})
        if n_a < MIN_DONORS or n_b < MIN_DONORS:
            rep["reason"] = (f"'{key}' has {n_a} donor{'s' if n_a != 1 else ''} in {a} and {n_b} in {b}: "
                             f"a test needs at least {MIN_DONORS} in each, so the differences are shown "
                             "without p-values.")
        else:
            samples = list(zip(don[both].dropna(), cond[both][don[both].notna()]))
            design = paired_design.detect_design(samples, [a, b])
            paired = bool(design["paired"] and design["n_pairs"] >= paired_design.MIN_PAIRS_TEST)
            va, da, ca = donor_products(M, genes, pa.stats.genes, pa, labels, obs[key], mask_a, groups, table)
            vb, db, cb = donor_products(M, genes, pa.stats.genes, pa, labels, obs[key], mask_b, groups, table)
            p, used, test = replicate_tests(va, vb, da, db, paired)
            table["p_value"] = p
            table["p_adj"] = _bh(p)
            table["n_donors"] = used
            rep.update(paired=paired, n_pairs=int(design["n_pairs"]), test=test)
            if paired and design["n_pairs"] < paired_design.MIN_PAIRS:
                n = design["n_pairs"]
                rep["caution"] = (f"{n} paired donors: an exact signed-rank test cannot go below "
                                  f"p = {2.0 / 2 ** n:.3g}; read the p-values as descriptive.")
            elif paired:
                n = design["n_pairs"]
                rep["caution"] = (f"With {n} paired donors the smallest possible p is {2.0 / 2 ** n:.3g}, "
                                  "so after correction across many interactions few may pass.")
            donor_values = {"a": va, "b": vb, "donors_a": da, "donors_b": db, "cells_a": ca, "cells_b": cb}

    comp = composition(groups, pa.stats.sizes, pb.stats.sizes)
    flagged = {c["group"] for c in comp if c["flagged"]}
    table["composition_note"] = table["source"].isin(flagged) | table["target"].isin(flagged)
    order = np.lexsort((-np.abs(table["delta"].to_numpy()), (table["call"] == "unchanged").to_numpy()))
    table = table.iloc[order].reset_index(drop=True)
    for k in ("a", "b", "cells_a", "cells_b"):
        if k in donor_values:
            donor_values[k] = donor_values[k][order]
    return {
        "groups": groups, "sizes_a": pa.stats.sizes.tolist(), "sizes_b": pb.stats.sizes.tolist(),
        "dropped_groups": dropped, "n_interactions": len(pa.interactions), "table": table,
        "counts_a": counts_a, "counts_b": counts_b, "strength_a": strength_a, "strength_b": strength_b,
        "replicates": rep, "composition": comp, "donor_values": donor_values,
    }


def donor_points(r: CompareResult, row: int) -> dict:
    """One compared row's per-donor products, for a slope / dot plot."""
    if not r.donor_values:
        return {"available": False, "reason": r.replicates.get("reason"), "points": []}
    if not 0 <= row < len(r.table):
        raise IndexError(row)
    pts = []
    vals: dict[str, dict[str, float]] = {r.a: {}, r.b: {}}
    for key, cond in (("a", r.a), ("b", r.b)):
        dv = r.donor_values
        for d, v, n in zip(dv[f"donors_{key}"], dv[key][row], dv[f"cells_{key}"][row]):
            if np.isfinite(v):
                pts.append({"donor": d, "condition": cond, "value": float(v), "n_cells": int(n)})
                vals[cond][d] = float(v)
    rec = r.table.iloc[row]
    paired = None
    if r.replicates["paired"]:
        both = sorted(set(vals[r.a]) & set(vals[r.b]))
        paired = paired_design.paired_test([vals[r.a][d] for d in both], [vals[r.b][d] for d in both])
    return {"available": True, "paired": r.replicates["paired"], "sample_key": r.replicates["sample_key"],
            "test": r.replicates["test"], "points": pts, "paired_test": paired,
            "p_value": None if not np.isfinite(rec["p_value"]) else float(rec["p_value"]),
            "p_adj": None if not np.isfinite(rec["p_adj"]) else float(rec["p_adj"]),
            "label": f"{rec['source']} -> {rec['target']}: {rec['ligand_complex']} -> {rec['receptor_complex']}"}


def estimate_seconds(n_a: int, n_b: int, n_groups: int, n_perms: int = ccc.N_PERMS) -> float:
    """Two single runs, plus the donor tests (about a second per few thousand rows)."""
    return round(ccc.estimate_seconds(n_a, n_groups, n_perms) + ccc.estimate_seconds(n_b, n_groups, n_perms)
                 + 1.0, 1)


# ---------------------------------------------------------------------------
# Kept results (CSV, donor plots) and the co-pilot's facts
# ---------------------------------------------------------------------------

_results: dict[str, CompareResult] = {}
_order: list[str] = []
_lock = threading.Lock()
_last: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()


def keep(result: CompareResult) -> None:
    with _lock:
        _results[result.id] = result
        _order.append(result.id)
        while len(_order) > KEEP:
            _results.pop(_order.pop(0), None)


def recall(dataset_id: str, result_id: str) -> CompareResult | None:
    with _lock:
        r = _results.get(result_id)
    return r if r is not None and r.dataset_id == dataset_id else None


def remember(adaptor, result: CompareResult) -> None:
    try:
        with _lock:
            _last[adaptor] = result
    except TypeError:  # not weak-referenceable (a test double)
        pass


def export_table(r: CompareResult) -> pd.DataFrame:
    """The compared rows as the CSV has them (no internal columns)."""
    return r.table.drop(columns=[c for c in r.table.columns if c.startswith("_")])


def grounding_lines(adaptor, obs: pd.DataFrame) -> tuple[list[str], str | None]:
    """For the co-pilot: the last comparison's biggest gains and losses and the
    sender-receiver pairs whose interaction count moved most."""
    try:
        with _lock:
            r = _last.get(adaptor)
    except TypeError:
        return [], None
    if r is None or r.groupby not in obs.columns or r.condition not in obs.columns:
        return [], None
    if not set(r.groups) <= set(obs[r.groupby].dropna().astype(str)):
        return [], None
    tab = r.table
    n = {c: int((tab["call"] == c).sum()) for c in CALLS}
    rep = r.replicates
    test = (f"tested on {rep['sample_key']} donors ({rep['test']}, BH)" if rep.get("test")
            else "no replicate test (no donors with both conditions)")
    detail = (f"Ligand-receptor communication {r.b} vs {r.a} ({r.condition}) between {len(r.groups)} "
              f"'{r.groupby}' groups: {n['gained']} gained, {n['lost']} lost, {n['changed']} changed, "
              f"{n['unchanged']} unchanged interactions; {test}")
    lines = ["\n## Cell-cell communication compared between conditions (last comparison)",
             f"- {detail} [result:communication_compare]."]
    moved = tab[tab["call"] != "unchanged"].drop_duplicates(["ligand_complex", "receptor_complex"]).head(8)
    for row in moved.itertuples():
        p = (f", adj. p {row.p_adj:.2g}" if np.isfinite(row.p_adj) else "")
        lines.append(f"- {row.call}: {row.source} -> {row.target}: {row.ligand_complex} -> "
                     f"{row.receptor_complex} (LR mean {row.lr_means_a:.2f} in {r.a}, {row.lr_means_b:.2f} "
                     f"in {r.b}{p})")
    diff = r.counts_b - r.counts_a
    if diff.size and np.abs(diff).max() > 0:
        i, j = np.unravel_index(np.argmax(np.abs(diff)), diff.shape)
        lines.append(f"- Largest change in interaction count: {r.groups[i]} -> {r.groups[j]} "
                     f"({int(r.counts_a[i, j])} in {r.a}, {int(r.counts_b[i, j])} in {r.b}).")
    flagged = [c["group"] for c in r.composition if c["flagged"]]
    if flagged:
        lines.append(f"- Composition caveat: {', '.join(flagged)} changed share of the cells "
                     f"{SHARE_FOLD:g}-fold or more between {r.a} and {r.b}.")
    lines.append("- Expression-based predictions of signalling, not measured interactions.")
    return lines, detail
