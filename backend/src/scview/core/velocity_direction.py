"""Does the velocity run point the way an order the user trusts says? (``core/velocity.py``)

Cross-boundary direction (CBDir, as UniTVelo's ``cross_boundary_correctness`` and CellRank's
benchmarks use it): for each expected transition A -> B, every cell of A with cells of B among
its K nearest cells ON THE PLOT gives the cosine between its arrow and its mean displacement to
those B cells; the transition's score is the mean over those boundary cells. UniTVelo averages
the cosine to each B neighbour instead; the mean displacement is less noisy and is what QA
round 5 measured (``research/v079-qa/trajectory/veldir.py``). Neighbours are taken on the plot
the arrows are drawn on, so the check is about the arrows the user reads.

Per cell: the cosine between its arrow and the local gradient of the order (sum over its K
neighbours of (t_j - t_i)(x_j - x_i)), for cells whose neighbours differ in order.

An order comes from (``resolve``): a numeric column (a pseudotime, a time), an ordered column (a
stage, given in order or parsed from numbers like E7.5), groups in an order the user gives,
transitions the user gives, or a root group with the PAGA tree on that column. With none given,
scView's own diffusion pseudotime is used when the dataset has one; else the direction is
unchecked.

What it cannot tell: when arrows and order disagree, which one is wrong. Neighbouring cells are
not independent, so the interval (mean +- 1.96 SE over boundary cells) is narrower than it
should be; a transition is called backward only when the whole interval is below zero.
"""

from __future__ import annotations

import re
from collections import deque
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

K = 30  # neighbours on the plot (QA round 5; UniTVelo uses the graph's)
MIN_BOUNDARY = 10  # boundary cells for a transition's verdict
ORDER_AUC = 0.75  # a numeric order: B is later than A for at least this share of cell pairs
AUC_SAMPLE = 4000  # cells per group in that comparison
MAX_GROUPS = 100
MAX_TRANSITIONS = 60
NN_BLOCK = 50_000
_NUMBER = re.compile(r"[-+]?\d*\.?\d+")

CAPTION = (
    "Direction check: each transition A -> B scores the cells of A next to cells of B on this plot (their {k} nearest) "
    "by the cosine between the cell's arrow and the way to those B cells: +1 straight at B, -1 straight away. "
    "Backward means the whole interval is below 0. It compares the arrows with the order given; when they disagree it "
    "cannot say which is wrong, and the interval treats neighbouring cells as independent, so it is too narrow."
)


class DirectionError(ValueError):
    """An order that cannot be used (unknown group, no PAGA, a cycle): the message says what to give."""


@dataclass
class Ordering:
    kind: str  # pseudotime | column | order | transitions | root
    label: str  # in words, for the panel and the warning
    group_column: str
    labels: np.ndarray  # per dataset cell: its group (str) or None
    t: np.ndarray  # per dataset cell: its place in the order (NaN: none)
    transitions: list[tuple[str, str]] | None  # None: derived from the numeric order (``derive``)
    spec: dict[str, Any] = field(default_factory=dict)


def _labels(adata, column: str) -> tuple[np.ndarray, list[str]]:
    if column not in adata.obs.columns:
        raise DirectionError(f"This dataset has no column '{column}'.")
    col = adata.obs[column]
    if pd.api.types.is_float_dtype(col):
        raise DirectionError(f"'{column}' holds continuous values, not groups: pick a column of groups.")
    labels = col.astype(object).where(col.notna(), None).map(lambda v: None if v is None else str(v)).to_numpy()
    present = {v for v in labels if v is not None}
    if isinstance(col.dtype, pd.CategoricalDtype):
        groups = [str(c) for c in col.cat.categories if str(c) in present]
    else:
        groups = sorted(present)
    if len(groups) > MAX_GROUPS:
        raise DirectionError(f"'{column}' has {len(groups)} groups; the direction check takes at most {MAX_GROUPS}.")
    return labels, groups


def _check_groups(names, groups: list[str], column: str) -> None:
    unknown = [g for g in names if g not in groups]
    if unknown:
        raise DirectionError(f"'{column}' has no group {', '.join(repr(u) for u in unknown[:5])}.")


def _depths(transitions: list[tuple[str, str]]) -> dict[str, float]:
    """Each group's longest path from a group nothing leads to; a cycle is refused."""
    nodes = sorted({g for e in transitions for g in e})
    into = {g: 0 for g in nodes}
    out: dict[str, list[str]] = {g: [] for g in nodes}
    for a, b in transitions:
        out[a].append(b)
        into[b] += 1
    depth = {g: 0.0 for g in nodes}
    queue = deque(g for g in nodes if into[g] == 0)
    seen = 0
    while queue:
        g = queue.popleft()
        seen += 1
        for h in out[g]:
            depth[h] = max(depth[h], depth[g] + 1)
            into[h] -= 1
            if into[h] == 0:
                queue.append(h)
    if seen < len(nodes):
        raise DirectionError("The transitions go round in a circle: give them in one direction.")
    return depth


def _numeric_order(col: pd.Series, column: str, column_order: list[str] | None) -> tuple[np.ndarray, str]:
    """Each cell's place in a column's order, and how it was read."""
    if pd.api.types.is_bool_dtype(col):
        raise DirectionError(f"'{column}' is true/false, not an order.")
    if pd.api.types.is_numeric_dtype(col):
        t = col.to_numpy(dtype=np.float64)
        return np.where(np.isfinite(t), t, np.nan), "its values"
    text = col.astype(object).where(col.notna(), None).map(lambda v: None if v is None else str(v))
    if column_order:
        place = {str(v): i for i, v in enumerate(column_order)}
        missing = [v for v in column_order if v not in set(text.dropna())]
        if len(missing) == len(column_order):
            raise DirectionError(f"None of the values given is in '{column}'.")
        return np.asarray([place.get(v, np.nan) if v is not None else np.nan for v in text], dtype=np.float64), \
            "the order given"
    if isinstance(col.dtype, pd.CategoricalDtype) and col.cat.ordered:
        codes = col.cat.codes.to_numpy().astype(np.float64)
        return np.where(codes >= 0, codes, np.nan), "its categories' order"
    values = sorted(set(text.dropna()))
    parsed = {}
    for v in values:
        found = _NUMBER.findall(v)
        if len(found) != 1:
            raise DirectionError(f"'{column}' is not numeric and its values ({', '.join(values[:4])}) do not each carry "
                                 "one number: give its values in order.")
        parsed[v] = float(found[0])
    return np.asarray([parsed.get(v, np.nan) if v is not None else np.nan for v in text], dtype=np.float64), \
        "the number in each value"


def resolve(adata, spec: dict[str, Any] | None, default_group: str | None) -> tuple[Ordering | None, str | None]:
    """(the order to check against, None) or (None, why the direction is unchecked). Raises
    DirectionError for an order that names what is not there."""
    from scview.core import trajectory

    spec = {k: v for k, v in (spec or {}).items() if v not in (None, "", [])}
    column = spec.get("group_by") or default_group
    if not column:
        return None, "No groups to check transitions between: pick a column of groups (cell types, clusters)."
    labels, groups = _labels(adata, column)
    if len(groups) < 2:
        raise DirectionError(f"'{column}' has {len(groups)} group: a transition needs two.")
    base = {"group_by": column}
    if spec.get("transitions"):
        pairs = [(str(a), str(b)) for a, b in spec["transitions"]][:MAX_TRANSITIONS]
        _check_groups([g for e in pairs for g in e], groups, column)
        depth = _depths(pairs)
        t = np.asarray([depth.get(v, np.nan) if v is not None else np.nan for v in labels], dtype=np.float64)
        label = "the transitions given (" + ", ".join(f"{a} -> {b}" for a, b in pairs[:4]) + (
            ", ..." if len(pairs) > 4 else "") + ")"
        return Ordering("transitions", label, column, labels, t, pairs, {**base, "transitions": [list(e) for e in pairs]}), None
    if spec.get("order"):
        order = [str(g) for g in spec["order"]]
        _check_groups(order, groups, column)
        if len(order) < 2:
            raise DirectionError("An order needs at least two groups.")
        place = {g: float(i) for i, g in enumerate(order)}
        t = np.asarray([place.get(v, np.nan) if v is not None else np.nan for v in labels], dtype=np.float64)
        pairs = list(zip(order[:-1], order[1:]))
        return Ordering("order", f"the order of {column} given", column, labels, t, pairs,
                        {**base, "order": order}), None
    if spec.get("root"):
        root = str(spec["root"])
        _check_groups([root], groups, column)
        paga = trajectory.read_paga(adata)
        if not paga or paga.get("groupby") != column:
            raise DirectionError(f"A root group needs the PAGA graph of '{column}': run PAGA on it (Trajectory), or give "
                                 "the groups in order.")
        names = paga["groups"]
        adj: dict[str, list[str]] = {g: [] for g in names}
        for i, j in paga.get("tree") or []:
            adj[names[i]].append(names[j])
            adj[names[j]].append(names[i])
        depth = {root: 0.0}
        pairs = []
        queue = deque([root])
        while queue:
            g = queue.popleft()
            for h in adj.get(g, []):
                if h not in depth:
                    depth[h] = depth[g] + 1
                    pairs.append((g, h))
                    queue.append(h)
        if not pairs:
            raise DirectionError(f"'{root}' has no edge in the PAGA tree of '{column}'.")
        t = np.asarray([depth.get(v, np.nan) if v is not None else np.nan for v in labels], dtype=np.float64)
        return Ordering("root", f"the PAGA tree of {column} from {root}", column, labels, t, pairs,
                        {**base, "root": root}), None
    order_column = spec.get("column")
    kind = "column"
    if not order_column:
        params = trajectory.read_params(adata)
        if trajectory.DPT_COLUMN in adata.obs.columns and params:
            order_column, kind = trajectory.DPT_COLUMN, "pseudotime"
        else:
            return None, ("No order to check against: this dataset has no pseudotime and none was given. Give the "
                          "groups in order, a root group, or a pseudotime or time column.")
    if order_column not in adata.obs.columns:
        raise DirectionError(f"This dataset has no column '{order_column}'.")
    t, how = _numeric_order(adata.obs[order_column], order_column, spec.get("column_order"))
    if np.isfinite(t).sum() < 2 or np.nanmin(t) == np.nanmax(t):
        raise DirectionError(f"'{order_column}' does not order the cells (one value).")
    if kind == "pseudotime":
        p = trajectory.read_params(adata) or {}
        label = f"scView's diffusion pseudotime from {p.get('root_column')} = {p.get('root_value')}"
    else:
        label = f"{order_column} ({how})"
    out = {**base, "column": order_column}
    if spec.get("column_order"):
        out["column_order"] = [str(v) for v in spec["column_order"]]
    pairs = None
    if order_column == column:  # a stage column that is also the groups: its own order, step by step
        ranked = sorted({v for v in labels if v is not None},
                        key=lambda g: float(np.nanmedian(t[labels == g])) if np.isfinite(t[labels == g]).any() else np.inf)
        ranked = [g for g in ranked if np.isfinite(t[labels == g]).any()]
        pairs = list(zip(ranked[:-1], ranked[1:]))
    return Ordering(kind, label, column, labels, t, pairs, out), None


# ---------------------------------------------------------------------------
# The check, on the cells the run has (run order)
# ---------------------------------------------------------------------------


def _cos(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    na, nb = np.linalg.norm(a, axis=-1), np.linalg.norm(b, axis=-1)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = (a * b).sum(axis=-1) / (na * nb)
    out[(na == 0) | (nb == 0)] = np.nan
    return out


def _neighbours(X: np.ndarray, k: int) -> np.ndarray:
    """Each cell's k nearest other cells on the plot (rows of X), a block of cells at a time."""
    from sklearn.neighbors import NearestNeighbors

    k = min(k, X.shape[0] - 1)
    nn = NearestNeighbors(n_neighbors=k + 1, n_jobs=1).fit(X)
    out = np.empty((X.shape[0], k), dtype=np.int64)
    for a in range(0, X.shape[0], NN_BLOCK):
        _, idx = nn.kneighbors(X[a:a + NN_BLOCK])
        rows = np.arange(a, min(a + NN_BLOCK, X.shape[0]))
        # drop the cell itself (first, except under ties: then the column that holds it, else the last)
        is_self = idx == rows[:, None]
        drop = np.where(is_self.any(axis=1), is_self.argmax(axis=1), k)
        keep = np.ones_like(idx, dtype=bool)
        keep[np.arange(idx.shape[0]), drop] = False
        out[a:a + NN_BLOCK] = idx[keep].reshape(idx.shape[0], k)
    return out


def _auc(later: np.ndarray, earlier: np.ndarray, rng: np.random.Generator) -> float:
    """P(a cell of ``later`` is later than one of ``earlier``), ties half (Mann-Whitney)."""
    from scipy.stats import rankdata

    a = later[np.isfinite(later)]
    b = earlier[np.isfinite(earlier)]
    if not a.size or not b.size:
        return np.nan
    if a.size > AUC_SAMPLE:
        a = rng.choice(a, AUC_SAMPLE, replace=False)
    if b.size > AUC_SAMPLE:
        b = rng.choice(b, AUC_SAMPLE, replace=False)
    r = rankdata(np.concatenate([a, b]))
    return float((r[:a.size].sum() - a.size * (a.size + 1) / 2) / (a.size * b.size))


def _boundary(codes: np.ndarray, idx: np.ndarray, n_groups: int) -> np.ndarray:
    """groups x groups: cells of A with at least one B among their neighbours."""
    out = np.zeros((n_groups, n_groups), dtype=np.int64)
    for s in range(0, codes.size, NN_BLOCK):
        own = codes[s:s + NN_BLOCK]
        nb = codes[idx[s:s + NN_BLOCK]]
        cell = np.repeat(np.arange(own.size), nb.shape[1])
        flat = nb.ravel()
        ok = (flat >= 0) & (own[cell] >= 0) & (flat != own[cell])
        pairs = np.unique(cell[ok] * n_groups + flat[ok])  # (cell, neighbour group) once each
        np.add.at(out, (own[pairs // n_groups], pairs % n_groups), 1)
    return out


def _cross_boundary(X: np.ndarray, V: np.ndarray, codes: np.ndarray, idx: np.ndarray, a: int, b: int) -> np.ndarray:
    """Each boundary cell of group a: cosine of its arrow with its mean displacement to its
    neighbours in group b (cells with none left out)."""
    out = []
    cells = np.flatnonzero(codes == a)
    for s in range(0, cells.size, NN_BLOCK):
        rows = cells[s:s + NN_BLOCK]
        nb = idx[rows]
        hit = codes[nb] == b
        n = hit.sum(axis=1)
        keep = n > 0
        if not keep.any():
            continue
        rows, nb, hit, n = rows[keep], nb[keep], hit[keep], n[keep]
        disp = ((X[nb] - X[rows][:, None, :]) * hit[:, :, None]).sum(axis=1) / n[:, None]
        out.append(_cos(V[rows], disp))
    s = np.concatenate(out) if out else np.zeros(0)
    return s[np.isfinite(s)]


def derive(o: Ordering, codes: np.ndarray, groups: list[str], t: np.ndarray, idx: np.ndarray,
           ) -> tuple[list[tuple[str, str]], list[dict[str, Any]]]:
    """Transitions from a numeric order: groups that meet on the plot (MIN_BOUNDARY cells each way
    counted together), pointed from the earlier to the later when ORDER_AUC of cell pairs agree.
    Also the pairs left out because the order does not separate them."""
    counts = _boundary(codes, idx, len(groups))
    rng = np.random.default_rng(0)
    pairs, skipped = [], []
    for g in range(len(groups)):
        for h in range(g + 1, len(groups)):
            if counts[g, h] + counts[h, g] < MIN_BOUNDARY:
                continue
            auc = _auc(t[codes == h], t[codes == g], rng)
            if not np.isfinite(auc):
                continue
            if auc >= ORDER_AUC:
                pairs.append((groups[g], groups[h], auc))
            elif auc <= 1 - ORDER_AUC:
                pairs.append((groups[h], groups[g], 1 - auc))
            else:
                skipped.append({"groups": [groups[g], groups[h]], "auc": round(float(auc), 3)})
    pairs.sort(key=lambda p: -p[2])
    pairs = pairs[:MAX_TRANSITIONS]
    median = {g: float(np.nanmedian(t[codes == i])) if np.isfinite(t[codes == i]).any() else np.inf
              for i, g in enumerate(groups)}
    pairs.sort(key=lambda p: (median[p[0]], median[p[1]]))  # earliest first, as read along the order
    return [(a, b) for a, b, _ in pairs], skipped


def evaluate(o: Ordering, X: np.ndarray, V: np.ndarray, positions: np.ndarray, embedding: str) -> dict[str, Any]:
    """The check on the run's cells: ``X`` / ``V`` their plot coordinates and arrows (run order),
    ``positions`` their rows in the dataset (to read ``o.labels`` / ``o.t``)."""
    labels = o.labels[positions]
    t = o.t[positions]
    ok = np.isfinite(X).all(axis=1) & np.isfinite(V).all(axis=1)
    rows = np.flatnonzero(ok)
    Xo, Vo, lo, to = X[rows], V[rows], labels[rows], t[rows]
    groups = sorted({v for v in lo if v is not None})
    index = {g: i for i, g in enumerate(groups)}
    codes = np.asarray([index.get(v, -1) if v is not None else -1 for v in lo], dtype=np.int64)
    idx = _neighbours(Xo, K)
    skipped: list[dict[str, Any]] = []
    if o.transitions is None:
        pairs, skipped = derive(o, codes, groups, to, idx)
    else:
        pairs = o.transitions
    rows_out = []
    for a, b in pairs:
        ga, gb = index.get(a, -2), index.get(b, -2)
        s = _cross_boundary(Xo, Vo, codes, idx, ga, gb)
        row: dict[str, Any] = {"from": a, "to": b, "n_cells": int(s.size)}
        if s.size:
            mean = float(s.mean())
            half = float(1.96 * s.std(ddof=1) / np.sqrt(s.size)) if s.size > 1 else float("inf")
            row.update({"mean_cos": round(mean, 3), "low": round(mean - half, 3), "high": round(mean + half, 3),
                        "share_forward": round(float((s > 0).mean()), 3)})
        if s.size == 0:
            row["status"] = "no_boundary"
        elif s.size < MIN_BOUNDARY:
            row["status"] = "few_cells"
        elif row["high"] < 0:
            row["status"] = "backward"
        elif row["low"] > 0:
            row["status"] = "forward"
        else:
            row["status"] = "unclear"
        rows_out.append(row)
    # per cell: the arrow against the local gradient of the order
    fin = np.isfinite(to)
    tt = np.where(fin, to, 0.0)
    c = np.full(Xo.shape[0], np.nan)
    for s in range(0, Xo.shape[0], NN_BLOCK):
        nb = idx[s:s + NN_BLOCK]
        here = slice(s, s + nb.shape[0])
        w = (tt[nb] - tt[here, None]) * (fin[nb] & fin[here, None])
        grad = (w[:, :, None] * (Xo[nb] - Xo[here, None, :])).sum(axis=1)
        c[here] = _cos(Vo[here], grad)
    has = np.isfinite(c)
    by_group = []
    for g in groups:
        m = has & (codes == index[g])
        if m.sum():
            by_group.append({"group": g, "n_cells": int(m.sum()), "mean_cos": round(float(c[m].mean()), 3),
                             "share_forward": round(float((c[m] > 0).mean()), 3)})
    cells = {"n_cells": int(has.sum()), "n_run_cells": int(X.shape[0]),
             "mean_cos": round(float(c[has].mean()), 3) if has.any() else None,
             "share_forward": round(float((c[has] > 0).mean()), 3) if has.any() else None, "by_group": by_group}
    statuses = [r["status"] for r in rows_out]
    return {
        "checked": True, "kind": o.kind, "label": o.label, "spec": o.spec, "group_column": o.group_column,
        "embedding": embedding, "k": K, "min_cells": MIN_BOUNDARY, "transitions": rows_out,
        "n_backward": statuses.count("backward"), "n_forward": statuses.count("forward"),
        "n_unclear": statuses.count("unclear"), "skipped": skipped[:20], "cells": cells,
        "derived": o.transitions is None, "order_auc": ORDER_AUC if o.transitions is None else None,
        "note": CAPTION.format(k=K),
    }


def unchecked(reason: str, embedding: str | None = None) -> dict[str, Any]:
    return {"checked": False, "reason": reason, "embedding": embedding, "note": CAPTION.format(k=K)}


def warnings(direction: dict[str, Any] | None) -> list[str]:
    """The warning a backward transition earns, with its numbers."""
    if not direction or not direction.get("checked"):
        return []
    back = [r for r in direction.get("transitions") or [] if r.get("status") == "backward"]
    if not back:
        return []
    parts = "; ".join(f"{r['from']} -> {r['to']} (mean cosine {r['mean_cos']:+.2f}, {r['n_cells']:,} boundary cells, "
                      f"{r['share_forward']:.0%} forward)" for r in back[:4])
    more = f" and {len(back) - 4} more" if len(back) > 4 else ""
    return [f"The arrows point backward against {direction['label']} at {parts}{more}, on "
            f"{direction['embedding']}. Velocity confidence cannot show this. Known causes: genes whose transcription "
            "rate steps up along the path (MURK genes, Barile 2021), or a graph split by batch. Check the flagged genes, "
            "leave genes out and run again, or do not read the arrows there."]
