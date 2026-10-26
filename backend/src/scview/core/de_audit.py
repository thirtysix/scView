"""DE audit: can a cell-level DE result between two conditions be trusted?

A cell-level test (Wilcoxon, Welch) treats every cell as a replicate. When the
two sides are also different donors, donor-to-donor differences count as
condition effects and the p-values are far too small: Squair et al. 2021 (Nat
Commun 12:5692, PMID 34584091) found hundreds of "DE" genes in comparisons with
no true difference, and Zimmerman et al. 2021 (Nat Commun 12:738, PMID
33531494) showed the same pseudoreplication inflating false positives. The
audit runs one comparison three ways:

1. **Donor shuffles.** The condition labels are shuffled across donors, never
   across cells, and the same cell-level test is run again (``DEFAULT_PERMUTATIONS``
   times, at most ``MAX_SECONDS``). When every donor is on both sides
   (paired), a shuffle swaps the two samples of half the donors, the
   exchangeable unit of a paired design; donors sampled on one side only
   (unpaired) are reassigned with the side sizes kept, and a mixed design
   reassigns its donor-x-condition samples. The shuffles are *balanced*: each
   pseudo-side holds the two real sides in the same proportion (half the
   donors swapped; A's samples split as evenly as the sizes allow), so a real
   effect shared by the donors cancels instead of being counted as donor
   noise. Unbalanced shuffles keep part of it: on Kang 2018 CD14 monocytes
   (STIM vs CTRL, 8 paired donors, 1,836 genes observed) 20 any-size swaps
   gave 29-1,366 genes (median 590), 20 balanced ones 10-663 (median 310). With an
   odd number of donors 1/n of a real effect remains, and unequal cell
   numbers leave a little more. Shuffles that only swap the two sides count
   once (the two-sided test finds the same genes), the observed labelling is
   never one, and when there are no more distinct shuffles than asked for,
   every one is run. A shuffle keeps the donor structure and drops the
   condition, so its count of genes at adjusted p < ``ALPHA`` is what donor
   differences produce at this cell-level test; a "condition" that is really
   one donor gives the shuffles as many genes as the observed labelling.
   Needs ``MIN_DONORS`` donors on each side.
2. **Pseudobulk.** The app's pseudobulk DE (``core/pseudobulk.py``) on the
   same cells, donors as replicates: the overlap of the two lists of
   significant genes (Jaccard), the Spearman correlation of their log fold
   changes, and the genes significant at cell level but not in pseudobulk
   (the likely false positives), each with the share of its difference carried
   by its top donor.
3. **Verdict.** One plain sentence from the two, naming the rule it used.

The top-donor share of a gene: the difference of the two sides' mean
log-normalised values, taken again without each donor's cells (both its
samples when paired); the share is the largest part of the difference lost
by leaving one donor out (``1 - without / with``, clipped to 0-1). Near 0 when
the donors carry it evenly, 1 when one donor carries all of it.
"""

from __future__ import annotations

import itertools
import logging
import math
import threading
import time
import weakref
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
import scipy.sparse as sp

from scview.core import results_store

logger = logging.getLogger(__name__)

ALPHA = 0.05  # adjusted p cut-off for "significant", cell level and pseudobulk
DEFAULT_PERMUTATIONS = 20
MAX_PERMUTATIONS = 100
MIN_DONORS = 3  # per side, for the shuffles
MAX_SECONDS = 300.0  # the shuffles stop here, with those done so far
QQ_POINTS = 200
TOP_GENES = 50  # rows of the cell-level-only table
# The verdict calls the shuffles "as many" when their median reaches this share
# of the observed count.
NULL_SHARE = 0.5
P_FLOOR = 1e-300  # -log10 of a p-value of 0 (underflow) is capped at 300
KIND = "de_audit"  # in the results store
TAG = "result:deaudit"  # the co-pilot's tag
SCHEMES = {
    "paired": "each donor's two samples swapped or not",
    "donors": "donors reassigned to the two sides",
    "samples": "donor-x-condition samples reassigned to the two sides",
}


class AuditError(ValueError):
    """A request the audit cannot serve: a 400."""


# ---------------------------------------------------------------------------
# The shuffles
# ---------------------------------------------------------------------------


@dataclass
class Design:
    """The samples (one donor on one side) of a comparison."""

    sample_of_cell: np.ndarray  # per cell: its sample, -1 without a donor
    sample_is_a: np.ndarray  # per sample: on side A
    sample_donor: np.ndarray  # per sample: its donor
    n_donors_a: int
    n_donors_b: int
    scheme: str  # paired | donors | samples

    @property
    def enough(self) -> bool:
        return min(self.n_donors_a, self.n_donors_b) >= MIN_DONORS

    def balance(self) -> tuple[int, int, list[int]]:
        """Paired: (donors, donors, how many donors a shuffle swaps). Otherwise:
        (samples on A, samples on B, how many of A's samples a shuffle keeps on A)."""
        if self.scheme == "paired":
            n = len(np.unique(self.sample_donor))
            return n, n, sorted({n // 2, (n + 1) // 2})
        n_a, n_b = int(self.sample_is_a.sum()), int((~self.sample_is_a).sum())
        if n_a == 0 or n_b == 0:
            return n_a, n_b, []
        even = n_a * n_a / (n_a + n_b)  # A's share of pseudo-A = its share of pseudo-B
        keep = {math.floor(even), math.ceil(even)}
        return n_a, n_b, sorted(j for j in keep if max(0, n_a - n_b) <= j < n_a)

    def n_possible(self) -> int:
        """Distinct balanced shuffles (a swap of the two sides counts as the same
        shuffle); the observed labelling is never one."""
        n_a, n_b, allowed = self.balance()
        if self.scheme == "paired":
            return sum(math.comb(n_a, k) for k in allowed) // 2
        total = sum(math.comb(n_a, j) * math.comb(n_b, n_a - j) for j in allowed)
        return total // 2 if n_a == n_b else total


def design(donors: np.ndarray, in_a: np.ndarray) -> Design:
    """The samples of the compared cells: ``donors`` per cell (None/NaN when
    unknown), ``in_a`` marks side A (the rest are side B)."""
    donor = pd.Series(donors, dtype=object)
    has = donor.notna().to_numpy()
    key = np.where(in_a, "a", "b").astype(object) + "\x1f" + donor.astype(str).to_numpy()
    codes, samples = pd.factorize(pd.Series(key[has]), sort=True)
    sample_of_cell = np.full(len(in_a), -1, dtype=np.int64)
    sample_of_cell[has] = codes
    side = np.array([s.split("\x1f", 1)[0] == "a" for s in samples], dtype=bool)
    sdonor = np.array([s.split("\x1f", 1)[1] for s in samples], dtype=object)
    da, db = set(sdonor[side]), set(sdonor[~side])
    if da == db:
        scheme = "paired"
    elif not da & db:
        scheme = "donors"
    else:
        scheme = "samples"
    return Design(sample_of_cell, side, sdonor, len(da), len(db), scheme)


def _canonical(labels: np.ndarray) -> tuple:
    """One key for a labelling and its swap (A <-> B finds the same genes)."""
    t = tuple(bool(x) for x in labels)
    return min(t, tuple(not x for x in t))


def _labellings(d: Design) -> Iterator[np.ndarray]:
    """Every balanced shuffle (pseudo-A per sample), swaps included."""
    n_a, n_b, allowed = d.balance()
    if d.scheme == "paired":
        donors, didx = np.unique(d.sample_donor, return_inverse=True)
        for k in allowed:
            for chosen in itertools.combinations(range(len(donors)), k):
                flip = np.zeros(len(donors), dtype=bool)
                flip[list(chosen)] = True
                yield d.sample_is_a ^ flip[didx]
        return
    idx_a, idx_b = np.flatnonzero(d.sample_is_a), np.flatnonzero(~d.sample_is_a)
    for j in allowed:
        for ka in itertools.combinations(idx_a, j):
            for kb in itertools.combinations(idx_b, n_a - j):
                lab = np.zeros(len(d.sample_is_a), dtype=bool)
                lab[list(ka) + list(kb)] = True
                yield lab


def _draw(d: Design, rng: np.random.Generator) -> np.ndarray:
    """One balanced shuffle at random, every one equally likely."""
    n_a, n_b, allowed = d.balance()
    if d.scheme == "paired":
        donors, didx = np.unique(d.sample_donor, return_inverse=True)
        flip = np.zeros(len(donors), dtype=bool)
        flip[rng.choice(len(donors), int(rng.choice(allowed)), replace=False)] = True
        return d.sample_is_a ^ flip[didx]
    w = np.array([math.comb(n_a, j) * math.comb(n_b, n_a - j) for j in allowed], dtype=float)
    j = int(rng.choice(allowed, p=w / w.sum()))
    lab = np.zeros(len(d.sample_is_a), dtype=bool)
    lab[rng.choice(np.flatnonzero(d.sample_is_a), j, replace=False)] = True
    lab[rng.choice(np.flatnonzero(~d.sample_is_a), n_a - j, replace=False)] = True
    return lab


def shuffles(d: Design, k: int, seed: int = 0) -> tuple[list[np.ndarray], bool]:
    """Up to ``k`` distinct balanced shuffles (pseudo-A per sample), never the
    observed labelling, and whether that is every one there is."""
    seen = {_canonical(d.sample_is_a)}
    out: list[np.ndarray] = []
    if d.n_possible() <= k:
        for lab in _labellings(d):
            c = _canonical(lab)
            if c not in seen:
                seen.add(c)
                out.append(lab)
        return out, True
    rng = np.random.default_rng(seed)
    for _ in range(k * 50):
        if len(out) >= k:
            break
        lab = _draw(d, rng)
        c = _canonical(lab)
        if c not in seen:
            seen.add(c)
            out.append(lab)
    return out, False


# ---------------------------------------------------------------------------
# Numbers from test results
# ---------------------------------------------------------------------------


def n_significant(padj: np.ndarray | None, alpha: float = ALPHA) -> int:
    if padj is None:
        return 0
    p = np.asarray(padj, dtype=np.float64)
    return int((np.isfinite(p) & (p < alpha)).sum())


def _neglog(p: np.ndarray) -> np.ndarray:
    return -np.log10(np.clip(p, P_FLOOR, 1.0))


def qq(observed: np.ndarray, permuted: list[np.ndarray], n_points: int = QQ_POINTS) -> dict:
    """Quantiles of -log10 p: the observed p-values and the shuffles' pooled
    p-values against the uniform expectation, at up to ``n_points`` ranks spread
    evenly on a log scale (the tail, where they part, gets most points)."""
    obs = np.sort(np.asarray(observed, dtype=np.float64)[np.isfinite(observed)])
    g = obs.size
    if g == 0:
        return {"expected": [], "observed": [], "permuted": []}
    ranks = np.unique(np.round(np.geomspace(1, g, num=min(n_points, g))).astype(np.int64))
    frac = (ranks - 0.5) / g
    out = {"expected": _neglog(frac), "observed": _neglog(obs[ranks - 1]), "permuted": None}
    pooled = np.concatenate([np.asarray(p, dtype=np.float64) for p in permuted]) if permuted else None
    if pooled is not None:
        pooled = np.sort(pooled[np.isfinite(pooled)])
    if pooled is not None and pooled.size:
        at = np.clip(np.floor(frac * pooled.size).astype(np.int64), 0, pooled.size - 1)
        out["permuted"] = _neglog(pooled[at])
    return {k: (None if v is None else [round(float(x), 3) for x in v]) for k, v in out.items()}


def donor_shares(X, genes_idx: np.ndarray, sample_of_cell: np.ndarray,
                 d: Design) -> list[tuple[float | None, str | None]]:
    """(top-donor share, top donor) of each column in ``genes_idx`` of ``X``
    (cells x genes, log-normalised; module docstring)."""
    if len(genes_idx) == 0:
        return []
    keep = sample_of_cell >= 0
    cols = X[:, genes_idx]
    cols = cols.toarray() if sp.issparse(cols) else np.asarray(cols)
    cols = np.asarray(cols, dtype=np.float64)[keep]
    s = sample_of_cell[keep]
    n_s = np.bincount(s, minlength=len(d.sample_is_a)).astype(np.float64)
    sums = np.zeros((len(d.sample_is_a), cols.shape[1]))
    np.add.at(sums, s, cols)
    donors, didx = np.unique(d.sample_donor, return_inverse=True)
    side = d.sample_is_a
    # Per donor and side: summed values and cells (zero where the donor has no sample).
    sum_a, sum_b = np.zeros((len(donors), cols.shape[1])), np.zeros((len(donors), cols.shape[1]))
    np.add.at(sum_a, didx[side], sums[side])
    np.add.at(sum_b, didx[~side], sums[~side])
    n_da = np.bincount(didx[side], weights=n_s[side], minlength=len(donors))
    n_db = np.bincount(didx[~side], weights=n_s[~side], minlength=len(donors))
    tot_a, tot_b = sum_a.sum(axis=0), sum_b.sum(axis=0)
    n_a, n_b = float(n_da.sum()), float(n_db.sum())
    effect = tot_a / n_a - tot_b / n_b
    out: list[tuple[float | None, str | None]] = []
    for j in range(cols.shape[1]):
        if effect[j] == 0:
            out.append((None, None))
            continue
        with np.errstate(divide="ignore", invalid="ignore"):
            without = ((tot_a[j] - sum_a[:, j]) / (n_a - n_da)) - ((tot_b[j] - sum_b[:, j]) / (n_b - n_db))
        lost = np.where(np.isfinite(without), 1 - without / effect[j], 1.0)  # a whole side gone: all
        lost = np.clip(lost, 0.0, 1.0)
        top = int(np.argmax(lost))
        out.append((round(float(lost[top]), 4), str(donors[top])))
    return out


def _spearman(x: np.ndarray, y: np.ndarray) -> float | None:
    from scipy import stats

    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 3 or np.ptp(x[ok]) == 0 or np.ptp(y[ok]) == 0:
        return None
    rho = stats.spearmanr(x[ok], y[ok]).statistic
    return round(float(rho), 4) if np.isfinite(rho) else None


def concordance(cell: pd.DataFrame, pb: pd.DataFrame, alpha: float = ALPHA) -> dict:
    """Agreement of the cell-level and pseudobulk results (DataFrames indexed by
    gene with ``lfc`` and ``padj``)."""
    sig_c = set(cell.index[cell["padj"].lt(alpha)])
    sig_p = set(pb.index[pb["padj"].lt(alpha)])
    union = sig_c | sig_p
    common = cell.index.intersection(pb.index)
    return {
        "n_sig_cell": len(sig_c), "n_sig_pseudobulk": len(sig_p), "n_shared": len(sig_c & sig_p),
        "jaccard": round(len(sig_c & sig_p) / len(union), 4) if union else None,
        "n_genes_common": int(len(common)),
        "spearman_lfc": _spearman(cell.loc[common, "lfc"].to_numpy(dtype=np.float64),
                                  pb.loc[common, "lfc"].to_numpy(dtype=np.float64)),
    }


def _range(counts: list[int]) -> str:
    if not counts:
        return ""
    lo, hi = min(counts), max(counts)
    return f"{lo:,}" if lo == hi else f"{lo:,}-{hi:,}"


def verdict(observed: int, null_counts: list[int], pb: dict | None, pb_name: str,
            perm_note: str | None) -> str:
    """One sentence a user can act on, naming the rule it applied."""
    parts = [f"{observed:,} gene{'s' if observed != 1 else ''} at cell level (adj. p < {ALPHA:g})"]
    null_large = None
    if null_counts:
        med = float(np.median(null_counts))
        null_large = observed == 0 or med >= NULL_SHARE * observed
        n = len(null_counts)
        mid = f", median {med:,.0f}" if n > 2 else ""
        parts.append(f"{_range(null_counts)} in {n} donor shuffle{'s' if n != 1 else ''}{mid}")
    elif perm_note:
        parts.append("no donor shuffles (too few donors)")
    if pb is not None:
        parts.append(f"{pb['n_shared']:,} also in pseudobulk ({pb_name}, {pb['n_sig_pseudobulk']:,} there)")
    if observed == 0:
        return "; ".join(parts) + ": nothing to audit."
    if null_large:
        advice = (f"the shuffles' median reaches half the observed count or more, so most cell-level "
                  f"genes are donor differences: "
                  f"{'trust the pseudobulk list' if pb else 'do not trust the cell-level list'}")
    elif pb is not None and pb["n_sig_pseudobulk"] > 0:
        advice = ("trust the pseudobulk list" + (
            "; the shuffles' median stays under half the observed count, so the comparison's "
            "effect is larger than donor differences" if null_large is False else ""))
    elif pb is not None:
        advice = ("pseudobulk finds none with these donors (it may lack power): "
                  "treat the cell-level genes as candidates, not findings")
    elif null_large is False:
        advice = ("the shuffles' median stays under half the observed count, so the comparison's "
                  "effect is larger than donor differences; without pseudobulk, read the strongest "
                  "genes first")
    else:
        advice = "with neither check available, treat the cell-level list as unconfirmed"
    return "; ".join(parts) + ": " + advice + "."


# ---------------------------------------------------------------------------
# The run
# ---------------------------------------------------------------------------


@dataclass
class AuditResult:
    label: str
    comparison: dict[str, Any]
    donor_key: str
    method: str  # the cell-level test
    scheme: str
    n_donors_a: int
    n_donors_b: int
    n_cells_a: int
    n_cells_b: int
    n_genes_tested: int
    observed: int
    null_counts: list[int]
    n_possible: int
    exhaustive: bool
    empirical_p: float | None  # (1 + shuffles >= observed) / (1 + shuffles)
    stopped: str | None  # why the shuffles stopped early
    permutation_note: str | None  # why there are no shuffles
    qq: dict[str, Any]
    pseudobulk: dict[str, Any] | None
    pseudobulk_note: str | None
    cell_only: list[dict[str, Any]]
    n_cell_only: int
    verdict: str
    seconds: float
    alpha: float = ALPHA
    null_share: float = NULL_SHARE
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        from dataclasses import asdict

        return {**asdict(self), "scheme_text": SCHEMES.get(self.scheme, self.scheme)}


def _relabel(test, labels: np.ndarray) -> None:
    test.obs["_de_group"] = pd.Categorical(np.where(labels, "selection", "rest"),
                                           categories=["rest", "selection"])


def iter_audit(
    test,
    in_a: np.ndarray,
    donors: np.ndarray,
    *,
    method: str = "wilcoxon",
    n_permutations: int = DEFAULT_PERMUTATIONS,
    max_seconds: float = MAX_SECONDS,
    seed: int = 0,
    pseudobulk: Callable[[], Any] | None = None,
    pb_name: str = "DESeq2",
    label: str = "",
    comparison: dict[str, Any] | None = None,
    donor_key: str = "",
    stop: Callable[[], bool] = lambda: False,
) -> Iterator[dict[str, Any]]:
    """Run the audit of ``test`` (an AnnData ``_prepare`` built in api/v1/de.py:
    the compared cells, obs ``_de_group``, the prefilter's genes): yields
    progress dicts, then ``{"result": AuditResult}``. ``donors`` and ``in_a``
    are per cell of ``test``; ``pseudobulk`` runs the app's pseudobulk DE and
    returns its ``PseudobulkResult`` (or raises ``PseudobulkError``). ``stop``
    is checked between steps."""
    from scview.core import de_methods

    t0 = time.perf_counter()
    in_a = np.asarray(in_a, dtype=bool)
    d = design(donors, in_a)
    n_a, n_b = int(in_a.sum()), int((~in_a).sum())

    yield {"step": "observed", "message": "Running the cell-level test as it was"}
    _relabel(test, in_a)
    obs_res = de_methods.run(test, method)
    observed = n_significant(obs_res.pvals_adj)
    genes = np.asarray(obs_res.genes)

    perm_note = None
    if not d.enough:
        perm_note = (f"Donor shuffles need at least {MIN_DONORS} donors on each side "
                     f"('{donor_key}': {d.n_donors_a} and {d.n_donors_b}): with fewer, there are "
                     "too few distinct shuffles to tell donor differences from the comparison.")
    labellings, exhaustive = ([], False) if perm_note else shuffles(d, n_permutations, seed)
    null_counts: list[int] = []
    permuted_p: list[np.ndarray] = []
    stopped = None
    t_perm = time.perf_counter()
    for i, lab in enumerate(labellings):
        if stop():
            return
        if time.perf_counter() - t_perm > max_seconds:
            stopped = f"stopped after {i} shuffles at the {max_seconds:g} s limit"
            break
        cell_lab = np.zeros(len(in_a), dtype=bool)
        known = d.sample_of_cell >= 0
        cell_lab[known] = lab[d.sample_of_cell[known]]
        _relabel(test, cell_lab)
        r = de_methods.run(test, method)
        null_counts.append(n_significant(r.pvals_adj))
        permuted_p.append(np.asarray(r.pvals, dtype=np.float64))
        yield {"step": "shuffle", "i": i + 1, "n": len(labellings), "count": null_counts[-1],
               "message": f"Donor shuffle {i + 1} of {len(labellings)}: {null_counts[-1]:,} genes"}
    _relabel(test, in_a)
    if stop():
        return
    emp = None
    if null_counts:
        emp = round((1 + sum(c >= observed for c in null_counts)) / (1 + len(null_counts)), 4)

    pb_summary = None
    pb_note = None
    pb_frame = None
    if pseudobulk is not None:
        yield {"step": "pseudobulk", "message": f"Running pseudobulk ({pb_name}) on the same cells"}
        from scview.core.pseudobulk import PseudobulkError

        try:
            pr = pseudobulk()
        except PseudobulkError as exc:
            pb_note = str(exc)
        else:
            pb_frame = pd.DataFrame([(g, lf, pa) for g, lf, _p, pa in pr.genes],
                                    columns=["gene", "lfc", "padj"]).drop_duplicates("gene").set_index("gene")
            pb_summary = {"method": pr.method, "name": pb_name, "n_tested": pr.n_genes_tested,
                          "n_samples_a": pr.n_samples_a, "n_samples_b": pr.n_samples_b,
                          "design": pr.design}
        if stop():
            return

    cell = pd.DataFrame({"lfc": obs_res.logfoldchanges, "padj": obs_res.pvals_adj,
                         "pval": obs_res.pvals}, index=pd.Index(genes, name="gene"))
    cell = cell[~cell.index.duplicated()]
    if pb_frame is not None:
        pb_summary.update(concordance(cell, pb_frame))
    sig = cell[cell["padj"].lt(ALPHA)]
    if pb_frame is not None:
        pb_sig = pb_frame.index[pb_frame["padj"].lt(ALPHA)]
        only = sig[~sig.index.isin(pb_sig)]
    else:
        only = sig.iloc[0:0]
    only = only.assign(_a=only["lfc"].abs()).sort_values(["pval", "_a"], ascending=[True, False])
    top = only.head(TOP_GENES)
    col = pd.Index(test.var_names).get_indexer(top.index)
    shares = donor_shares(test.X, col, d.sample_of_cell, d)
    rows = []
    for (g, r), (share, donor) in zip(top.iterrows(), shares):
        in_pb = pb_frame is not None and g in pb_frame.index
        rows.append({
            "gene": str(g), "cell_lfc": round(float(r["lfc"]), 4), "cell_padj": float(r["padj"]),
            "pb_lfc": round(float(pb_frame.at[g, "lfc"]), 4) if in_pb and np.isfinite(pb_frame.at[g, "lfc"]) else None,
            "pb_padj": float(pb_frame.at[g, "padj"]) if in_pb and np.isfinite(pb_frame.at[g, "padj"]) else None,
            "pb_tested": bool(in_pb), "top_donor_share": share, "top_donor": donor,
        })

    notes = []
    if exhaustive and labellings:
        notes.append(f"Only {len(labellings)} distinct donor shuffles exist here; all were run "
                     f"(the smallest possible empirical p is {1 / (1 + len(labellings)):.2g}).")
    n_missing = int((d.sample_of_cell < 0).sum())
    if n_missing:
        notes.append(f"{n_missing:,} cells without a donor in '{donor_key}' were left out.")

    result = AuditResult(
        label=label, comparison=dict(comparison or {}), donor_key=donor_key, method=method,
        scheme=d.scheme, n_donors_a=d.n_donors_a, n_donors_b=d.n_donors_b,
        n_cells_a=n_a, n_cells_b=n_b, n_genes_tested=int(len(genes)), observed=observed,
        null_counts=null_counts, n_possible=int(d.n_possible()), exhaustive=bool(exhaustive),
        empirical_p=emp, stopped=stopped, permutation_note=perm_note,
        qq=qq(obs_res.pvals if obs_res.pvals is not None else np.array([]), permuted_p),
        pseudobulk=pb_summary, pseudobulk_note=pb_note, cell_only=rows, n_cell_only=int(len(only)),
        verdict=verdict(observed, null_counts, pb_summary, pb_name, perm_note),
        seconds=round(time.perf_counter() - t0, 2), notes=notes,
    )
    yield {"result": result}


def run(test, in_a, donors, **kw) -> AuditResult:
    """``iter_audit`` to the end."""
    out = None
    for item in iter_audit(test, in_a, donors, **kw):
        if "result" in item:
            out = item["result"]
    if out is None:
        raise AuditError("The audit was stopped.")
    return out


# ---------------------------------------------------------------------------
# Results store and co-pilot: the last audit on each loaded dataset
# ---------------------------------------------------------------------------

_last: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()
_last_lock = threading.Lock()


def remember(adaptor, result: AuditResult) -> None:
    try:
        with _last_lock:
            _last[adaptor] = result
    except TypeError:  # not weak-referenceable (a test double)
        return
    results_store.keep(adaptor, KIND, result, key=result.label, label=f"DE audit: {result.label}",
                       params={"donor_key": result.donor_key, "method": result.method,
                               "n_permutations": len(result.null_counts)})


def last_result(adaptor) -> AuditResult | None:
    r = results_store.remembered(_last, _last_lock, adaptor, KIND)
    if r is None:
        return None
    obs = adaptor.adata.obs if hasattr(adaptor, "adata") else None
    if obs is not None:
        cols = [r.donor_key, (r.comparison or {}).get("column"), (r.comparison or {}).get("within_column")]
        if any(c and c not in obs.columns for c in cols):
            return None
    return r


def grounding_lines(adaptor) -> tuple[list[str], str | None]:
    """For the co-pilot: the last audit's verdict and its numbers."""
    r = last_result(adaptor)
    if r is None:
        return [], None
    detail = (f"{r.label}, donors from '{r.donor_key}' ({r.n_donors_a} vs {r.n_donors_b}): "
              f"{r.verdict}")
    lines = ["\n## DE audit (last run; cell-level DE vs donor shuffles and pseudobulk)",
             f"- {detail} [{TAG}]."]
    if r.null_counts and r.empirical_p is not None:
        lines.append(f"- Empirical p of the observed count against {len(r.null_counts)} donor shuffles "
                     f"({SCHEMES.get(r.scheme, r.scheme)}): {r.empirical_p:g}.")
    pb = r.pseudobulk
    if pb:
        rho = "n/a" if pb.get("spearman_lfc") is None else f"{pb['spearman_lfc']:.2f}"
        jac = "n/a" if pb.get("jaccard") is None else f"{pb['jaccard']:.2f}"
        lines.append(f"- Cell level vs pseudobulk: Jaccard {jac} of the significant genes, "
                     f"Spearman {rho} of the log fold changes over {pb.get('n_genes_common', 0):,} genes.")
    if r.cell_only:
        top = ", ".join(f"{g['gene']} (top donor {g['top_donor_share']:.0%})" if g.get("top_donor_share")
                        is not None else g["gene"] for g in r.cell_only[:8])
        lines.append(f"- Significant at cell level only ({r.n_cell_only:,}), strongest first: {top}.")
    return lines, detail
