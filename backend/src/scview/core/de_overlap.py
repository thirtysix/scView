"""Overlap of the significant genes of several DE results, for an UpSet plot.

"Which genes respond to IFN-beta in every cell type, and which only in
monocytes?" Each kept DE result (core/de_results.py) gives a gene set: genes
with adjusted p <= ``padj_max`` and |log2FC| >= ``lfc_min``, higher in A
("up"), higher in B ("down") or either ("both"). The sets' EXCLUSIVE
intersections put each gene in exactly one combination of sets (UpSet
semantics), so their sizes add up to the union.

Rules:
- p-values are the adjusted ones the result kept (BH within the result for the
  cell-level tests, DESeq2's padj for pseudobulk). A result that kept raw p
  only is adjusted here, BH over its own tested genes, and a note says so.
- Logistic regression gives no p-value: its set is the top ``LOGREG_TOP_N``
  genes per direction by |coefficient|, among the genes past the |log2FC|
  cut-off whose coefficient has the fold change's sign (the DE table's listing
  rule, cut at the "Compare methods" top N), and a note says so.
- Results can test different genes (the detection prefilter, pseudobulk's count
  filter), so each set reports how many genes its result tested, and
  ``restrict_to_common_tested`` counts only genes every result tested. Without
  it, a gene missing from a set may not have been tested there.
- "both" keeps the sign: a gene up in one result and down in another is
  discordant, counted apart (``discordant``), never as a shared hit.
- Extra sets (the user's gene lists) have no direction; their genes are matched
  to the results' gene names ignoring case.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
from scipy import stats

from scview.core import de_methods
from scview.core.de_results import METHOD_NAMES, StoredResult

Direction = Literal["up", "down", "both"]

LOGREG_TOP_N = de_methods.TOP_N  # genes per direction for a test without p-values
MAX_LISTED = 1000  # genes listed per intersection; ``n`` counts them all
MAX_INTERSECTIONS = 64  # the largest; the rest are counted in ``hidden``


@dataclass
class _Set:
    name: str
    kind: Literal["de", "extra"]
    signs: dict[str, int]  # gene -> +1 (higher in A) / -1; 0 for an extra set
    method: str | None = None
    rule: str | None = None  # padj | padj_bh_here | top_n
    n_tested: int | None = None
    n_not_tested: int | None = None  # extra set: genes no result tested
    result_id: str | None = None


def _bh(p: np.ndarray) -> np.ndarray:
    out = np.full(len(p), np.nan)
    ok = np.isfinite(p)
    if ok.any():
        out[ok] = stats.false_discovery_control(p[ok], method="bh")
    return out


def _has_pvalues(r: StoredResult) -> bool:
    if r.method == "logreg":
        return False
    if r.pvals_adj is not None and np.isfinite(r.pvals_adj).any():
        return True
    return bool(np.isfinite(r.pvals).any())


def significant(
    r: StoredResult, *, direction: Direction, padj_max: float, lfc_min: float
) -> tuple[dict[str, int], str]:
    """The result's set (gene -> +1 higher in A / -1 higher in B) and the rule
    that chose it: ``padj`` (the adjusted p it kept), ``padj_bh_here`` (raw p,
    adjusted here) or ``top_n`` (no p-values: the largest coefficients)."""
    lfc = np.asarray(r.logfoldchanges, dtype=np.float64)
    past = np.isfinite(lfc) & (np.abs(lfc) >= lfc_min) & (lfc != 0)
    if _has_pvalues(r):
        if r.pvals_adj is not None and np.isfinite(r.pvals_adj).any():
            padj, rule = np.asarray(r.pvals_adj, dtype=np.float64), "padj"
        else:
            padj, rule = _bh(np.asarray(r.pvals, dtype=np.float64)), "padj_bh_here"
        sig = past & np.isfinite(padj) & (padj <= padj_max)
        up, down = sig & (lfc > 0), sig & (lfc < 0)
    else:
        rule = "top_n"
        stat = np.asarray(r.stats, dtype=np.float64)
        ok = past & np.isfinite(stat) & (np.sign(stat) == np.sign(lfc))
        up, down = np.zeros(len(lfc), bool), np.zeros(len(lfc), bool)
        for side, mask in ((1, ok & (stat > 0)), (-1, ok & (stat < 0))):
            idx = np.flatnonzero(mask)
            top = idx[np.argsort(-np.abs(stat[idx]), kind="stable")[:LOGREG_TOP_N]]
            (up if side == 1 else down)[top] = True
    out: dict[str, int] = {}
    if direction in ("up", "both"):
        out.update({r.genes[i]: 1 for i in np.flatnonzero(up)})
    if direction in ("down", "both"):
        out.update({r.genes[i]: -1 for i in np.flatnonzero(down)})
    return out, rule


def _unique_names(names: list[str]) -> list[str]:
    seen: dict[str, int] = {}
    out = []
    for n in names:
        seen[n] = seen.get(n, 0) + 1
        out.append(n if seen[n] == 1 else f"{n} ({seen[n]})")
    return out


def overlap(
    results: list[tuple[str, StoredResult]],
    *,
    direction: Direction = "both",
    padj_max: float = 0.05,
    lfc_min: float = 0.25,
    restrict_to_common_tested: bool = False,
    extra_sets: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Set sizes, exclusive intersections (largest first, genes listed up to
    MAX_LISTED each), discordant genes ("both"), pairwise Jaccard and the genes
    each result tested, for ``results`` ((result_id, stored) pairs) and any
    ``extra_sets`` ({name, genes})."""
    extra_sets = extra_sets or []
    tested = [set(r.genes) for _, r in results]
    common = set.intersection(*tested) if tested else set()
    union_tested = set.union(*tested) if tested else set()
    canonical = {g.upper(): g for g in sorted(union_tested)}
    universe = common if restrict_to_common_tested else None

    sets: list[_Set] = []
    notes: list[str] = []
    names = _unique_names([r.label or f"DE result ({METHOD_NAMES.get(r.method, r.method)})"
                           for _, r in results])
    for (rid, r), name in zip(results, names):
        signs, rule = significant(r, direction=direction, padj_max=padj_max, lfc_min=lfc_min)
        if universe is not None:
            signs = {g: s for g, s in signs.items() if g in universe}
        sets.append(_Set(name=name, kind="de", signs=signs, method=r.method, rule=rule,
                         n_tested=len(r.genes), result_id=rid))
        if rule == "top_n":
            notes.append(
                f"{name}: logistic regression gives no p-values, so its set is the top "
                f"{LOGREG_TOP_N} genes per direction by |coefficient|, among genes past "
                f"|log2FC| {lfc_min:g} whose coefficient has the fold change's sign."
            )
        elif rule == "padj_bh_here":
            notes.append(
                f"{name}: only raw p-values were kept, so they were adjusted here "
                f"(Benjamini-Hochberg over its {len(r.genes):,} tested genes)."
            )

    for name, raw in zip(_unique_names([str(e.get("name") or "Gene list") for e in extra_sets]),
                         (e.get("genes") or [] for e in extra_sets)):
        genes: dict[str, int] = {}
        for g in raw:
            g = str(g).strip()
            if g:
                genes.setdefault(canonical.get(g.upper(), g), 0)
        n_not = sum(1 for g in genes if g not in union_tested)
        if universe is not None:
            genes = {g: 0 for g in genes if g in universe}
        sets.append(_Set(name=name, kind="extra", signs=genes, n_not_tested=n_not))

    de_idx = [i for i, s in enumerate(sets) if s.kind == "de"]
    # Each gene's combination: the sets it is in.
    member: dict[str, list[int]] = defaultdict(list)
    for i, s in enumerate(sets):
        for g in s.signs:
            member[g].append(i)

    lfc_of = [dict(zip(r.genes, r.logfoldchanges.tolist())) for _, r in results]

    def row(g: str) -> dict[str, Any]:
        vals = []
        for d in lfc_of:
            v = d.get(g)
            vals.append(round(float(v), 4) if v is not None and np.isfinite(v) else None)
        return {"gene": g, "lfc": vals}

    def order(genes: list[str], combo: tuple[int, ...]) -> list[str]:
        """Largest mean |log2FC| over the combination's DE results first."""
        des = [i for i in combo if sets[i].kind == "de"]

        def key(g: str) -> tuple[float, str]:
            vals = [abs(lfc_of[i].get(g, np.nan)) for i in des]
            vals = [v for v in vals if np.isfinite(v)]
            return (-(sum(vals) / len(vals)) if vals else 0.0, g)

        return sorted(genes, key=key)

    groups: dict[tuple[int, ...], list[str]] = defaultdict(list)
    disc: dict[tuple[int, ...], list[str]] = defaultdict(list)
    for g, idx in member.items():
        combo = tuple(idx)
        signs = {sets[i].signs[g] for i in idx if sets[i].kind == "de"}
        (disc if len(signs) > 1 else groups)[combo].append(g)

    def sign_counts(genes: list[str], combo: tuple[int, ...]) -> tuple[int, int]:
        des = [i for i in combo if sets[i].kind == "de"]
        if not des:
            return 0, 0
        up = sum(1 for g in genes if sets[des[0]].signs[g] > 0)
        return up, len(genes) - up

    inters = []
    for combo, genes in groups.items():
        up, down = sign_counts(genes, combo)
        listed = order(genes, combo)[:MAX_LISTED]
        inters.append({"sets": list(combo), "degree": len(combo), "n": len(genes),
                       "n_up": up, "n_down": down, "genes": [row(g) for g in listed]})
    inters.sort(key=lambda x: (-x["n"], x["degree"], x["sets"]))
    hidden = inters[MAX_INTERSECTIONS:]

    disc_genes = [g for gs in disc.values() for g in gs]
    disc_genes = order(disc_genes, tuple(de_idx))

    pairs = []
    for a in range(len(sets)):
        for b in range(a + 1, len(sets)):
            A, B = sets[a].signs, sets[b].signs
            both = A.keys() & B.keys()
            signed = sets[a].kind == "de" and sets[b].kind == "de"
            n_disc = sum(1 for g in both if A[g] != B[g]) if signed else 0
            shared = len(both) - n_disc
            union = len(A.keys() | B.keys())
            pairs.append({"a": a, "b": b, "shared": shared, "discordant": n_disc,
                          "union": union, "jaccard": round(shared / union, 4) if union else None})

    if restrict_to_common_tested:
        notes.append(f"Only the {len(common):,} genes every result tested are counted.")
    elif any(len(t) != len(common) for t in tested):
        notes.append(
            f"The results tested different genes ({min(len(t) for t in tested):,} to "
            f"{max(len(t) for t in tested):,}; {len(common):,} in all): tick \"only genes tested in "
            "all\" to count only those."
        )

    for s in sets:
        if s.kind == "extra" and s.n_not_tested:
            notes.append(f"{s.name}: {s.n_not_tested:,} of its genes were tested in no result.")

    out_sets = []
    for i, s in enumerate(sets):
        n_up = sum(1 for v in s.signs.values() if v > 0)
        n_down = sum(1 for v in s.signs.values() if v < 0)
        out_sets.append({
            "index": i, "name": s.name, "kind": s.kind, "result_id": s.result_id,
            "method": s.method, "rule": s.rule, "n": len(s.signs), "n_up": n_up, "n_down": n_down,
            "n_tested": s.n_tested, "n_not_tested": s.n_not_tested,
        })

    return {
        "direction": direction,
        "padj_max": padj_max,
        "lfc_min": lfc_min,
        "restrict_to_common_tested": restrict_to_common_tested,
        "logreg_top_n": LOGREG_TOP_N,
        "max_listed": MAX_LISTED,
        "sets": out_sets,
        "result_ids": [rid for rid, _ in results],
        "n_common_tested": len(common),
        "n_union": len(member),
        "intersections": inters[:MAX_INTERSECTIONS],
        "n_intersections": len(inters),
        "hidden_intersections": len(hidden),
        "hidden_genes": sum(x["n"] for x in hidden),
        "discordant": {
            "n": len(disc_genes),
            "combinations": sorted(
                ({"sets": list(c), "n": len(gs)} for c, gs in disc.items()),
                key=lambda x: (-x["n"], x["sets"]),
            ),
            # Each with the sets it is in (its combination).
            "genes": [{**row(g), "sets": member[g]} for g in disc_genes[:MAX_LISTED]],
        },
        "pairs": pairs,
        "notes": notes,
    }
