"""Minimal marker panel (C084): the fewest genes that still tell the groups apart, for a
flow panel, smFISH or a targeted assay.

Candidates are each group's markers as ``marker_selection.group_markers`` chooses them
(the ONE marker choice: expressed in >= 20% of the group, adjusted p < 0.05, by fold
change, no ribosomal or mitochondrial genes), the first ``per_group`` of each after the
user's exclusions and optional restriction, plus the genes the user requires.

Cells: at most ``CELLS_PER_GROUP`` per group, drawn with a fixed seed, so a large
group does not drown a small one and the run takes the same time on 10k or 1M cells.
Expression: the log-normalised values (``activity.lognorm_matrix``; a large dataset's
X from its gene store, only the candidate genes of the sampled cells), each gene
z-scored over the sampled cells.

Selection: greedy forward selection. The required genes go in first; then at each step
the gene that gives the highest cross-validated balanced accuracy of a multinomial
logistic regression (scikit-learn, ``class_weight="balanced"``, stratified 3-fold CV,
fixed seed) joins the panel. Trying every candidate at every step costs
candidates x steps x folds fits (minutes on Kang), so each step first ranks the
candidates by the score test of the current model (the gradient of its balanced
log-likelihood with respect to the new gene's coefficients, norm over the groups)
and cross-validates only the best ``SCREEN`` of them.

Reported: the path (balanced accuracy and per-group recall after each gene), the same
curve for a baseline of the top markers taken in turn from each group (no selection),
the cross-validated confusion matrix of the final panel, and per gene its standardised
coefficients in a model fitted on all sampled cells (the importance shown: no SHAP),
with the groups it raises.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
import scipy.sparse as sp

from scview.core.object_cache import ObjectCache

logger = logging.getLogger(__name__)

TAG = "result:marker_panel"
SIZE_DEFAULT = 20
SIZE_MAX = 50
PER_GROUP_DEFAULT = 10
PER_GROUP_MAX = 50
MAX_CANDIDATES = 200  # under large_mode.MAX_COLUMNS: one gene-store read
CELLS_PER_GROUP = 500
FOLDS = 3
SEED = 0
SCREEN = 15  # candidates cross-validated per step, after the score-test ranking
MIN_GROUP_CELLS = 10  # fewer and the group is left out (3-fold CV needs a few per fold)
MAX_GROUPS = 60
C = 1.0  # sklearn's default L2 strength
MAX_ITER = 300
WITHIN = 0.01  # "smallest panel within 1 point of the best"
IMPORTANCE_NOTE = ("Importance is the gene's standardised coefficient for each group in a "
                   "logistic regression on all sampled cells (no SHAP): positive raises the "
                   "group's odds.")


class PanelError(ValueError):
    """A panel that cannot be made as asked (a 400 with the reason)."""


class Cancelled(Exception):
    """The client went away; the run stopped before its next step."""


@dataclass
class Plan:
    groupby: str
    groups: list[str]
    group_sizes: dict[str, int]
    cells: np.ndarray  # sorted row positions of the sampled cells
    y: np.ndarray  # group code of each sampled cell
    candidates: list[str]  # in baseline order (each group's best in turn), required first
    marker_of: dict[str, list[str]]  # candidate -> groups it is a marker of
    baseline: list[str]  # top markers, each group's in turn (no required genes)
    required: list[str]
    excluded: list[str]
    missing: list[str]  # required genes not in the dataset
    dropped_groups: list[str]
    size: int
    per_group: int
    restricted: int | None  # size of the restriction list, if any
    seed: int = SEED
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        n = np.bincount(self.y, minlength=len(self.groups))
        return {
            "groupby": self.groupby, "groups": self.groups,
            "n_cells": int(self.cells.size),
            "n_per_group": {g: int(n[i]) for i, g in enumerate(self.groups)},
            "group_sizes": self.group_sizes, "cells_per_group": CELLS_PER_GROUP,
            "n_candidates": len(self.candidates), "per_group": self.per_group, "size": self.size,
            "required": self.required, "excluded": self.excluded, "missing": self.missing,
            "dropped_groups": self.dropped_groups, "restricted": self.restricted,
            "folds": FOLDS, "seed": self.seed, "screen": SCREEN,
            "cv": f"stratified {FOLDS}-fold CV, seed {self.seed}",
            "estimate_seconds": estimate_seconds(len(self.groups), int(self.cells.size), self.size),
            "notes": self.notes,
        }


def estimate_seconds(n_groups: int, n_cells: int, size: int) -> float:
    """Rough run time (each step: SCREEN x FOLDS fits plus one; the baseline: size x
    FOLDS); fits of a few thousand cells and up to 50 genes take ~20-60 ms."""
    per_fit = 0.004 + n_cells * n_groups * 6e-7
    return round((size * (SCREEN * FOLDS + 1) + size * FOLDS + FOLDS + 1) * per_fit, 1)


def _clean(genes: list[str] | None) -> list[str]:
    return list(dict.fromkeys(str(g).strip() for g in (genes or []) if str(g).strip()))


def expression_names(adaptor) -> pd.Index:
    """The genes the panel can read: a large dataset's X (its gene store), else the
    log-normalised matrix ``activity.lognorm_matrix`` picks."""
    adata = adaptor.adata
    if adaptor.is_large:
        return pd.Index(adata.var_names.astype(str))
    from scview.core.activity import lognorm_matrix

    _, names, _ = lognorm_matrix(adata)
    return pd.Index(pd.Index(names).astype(str))


def make_plan(adaptor, groupby: str, *, size: int = SIZE_DEFAULT, per_group: int = PER_GROUP_DEFAULT,
              required: list[str] | None = None, excluded: list[str] | None = None,
              restrict: list[str] | None = None, seed: int = SEED) -> Plan:
    """The groups, sampled cells and candidate genes of a panel run (no fitting)."""
    from scview.core.marker_selection import group_markers

    adata = adaptor.adata
    if groupby not in adata.obs.columns:
        raise KeyError(f"Column '{groupby}' not found in obs.")
    if not 1 <= size <= SIZE_MAX:
        raise PanelError(f"Panel size must be between 1 and {SIZE_MAX}.")
    if not 1 <= per_group <= PER_GROUP_MAX:
        raise PanelError(f"Candidates per group must be between 1 and {PER_GROUP_MAX}.")
    labels = adata.obs[groupby]
    present = labels.notna().to_numpy()
    lab = labels.astype(str).to_numpy()
    if isinstance(labels.dtype, pd.CategoricalDtype):
        order = [str(c) for c in labels.cat.categories]
    else:
        order = sorted(set(lab[present]))
    counts = pd.Series(lab[present]).value_counts()
    sizes = {g: int(counts.get(g, 0)) for g in order if counts.get(g, 0) > 0}
    if len(sizes) > MAX_GROUPS:
        raise PanelError(f"Column '{groupby}' has {len(sizes)} groups; a panel is made for at most "
                         f"{MAX_GROUPS}.")
    groups = [g for g, n in sizes.items() if n >= MIN_GROUP_CELLS]
    dropped = [g for g, n in sizes.items() if n < MIN_GROUP_CELLS]
    if len(groups) < 2:
        raise PanelError(f"Column '{groupby}' needs at least two groups of {MIN_GROUP_CELLS} or more "
                         "cells for a panel.")

    markers = group_markers(adata, groupby, n=5000)
    if not markers:
        raise PanelError(f"No marker genes for '{groupby}' yet: open its markers in the Markers "
                         "table first (they are computed once and saved).")
    names = expression_names(adaptor)
    known = set(names)
    required_in = _clean(required)
    missing = [g for g in required_in if g not in known]
    required_in = [g for g in required_in if g in known][:size]
    excluded_in = _clean(excluded)
    drop = set(excluded_in) | set(required_in)
    allow = set(_clean(restrict)) if restrict else None
    lists: dict[str, list[str]] = {}
    for g in groups:
        keep = [x for x in markers.get(g, []) if x in known and x not in drop
                and (allow is None or x in allow)]
        lists[g] = keep[:per_group]
    baseline: list[str] = []
    marker_of: dict[str, list[str]] = {}
    for rank in range(per_group):
        for g in groups:
            if rank < len(lists[g]):
                gene = lists[g][rank]
                marker_of.setdefault(gene, []).append(g)
                if gene not in baseline:
                    baseline.append(gene)
    baseline = baseline[:MAX_CANDIDATES - len(required_in)]
    if not baseline and not required_in:
        raise PanelError("No candidate genes left after the exclusions and restriction.")

    notes: list[str] = []
    if dropped:
        notes.append(f"{len(dropped)} group(s) with fewer than {MIN_GROUP_CELLS} cells left out: "
                     + ", ".join(dropped[:5]) + ("…" if len(dropped) > 5 else "") + ".")
    if missing:
        notes.append("Required genes not in the dataset: " + ", ".join(missing[:10]) + ".")
    no_markers = [g for g in groups if not lists[g]]
    if no_markers:
        notes.append(f"{len(no_markers)} group(s) gave no candidate markers: "
                     + ", ".join(no_markers[:5]) + ".")
    if len(baseline) + len(required_in) < size:
        notes.append(f"Only {len(baseline) + len(required_in)} candidate genes: the panel stops there.")

    rng = np.random.default_rng(seed)
    cells_parts, y_parts = [], []
    for code, g in enumerate(groups):
        idx = np.flatnonzero(present & (lab == g))
        if idx.size > CELLS_PER_GROUP:
            idx = np.sort(rng.choice(idx, CELLS_PER_GROUP, replace=False))
        cells_parts.append(idx)
        y_parts.append(np.full(idx.size, code))
    cells = np.concatenate(cells_parts)
    y = np.concatenate(y_parts)
    order_idx = np.argsort(cells, kind="stable")
    return Plan(
        groupby=groupby, groups=groups, group_sizes=sizes, cells=cells[order_idx], y=y[order_idx],
        candidates=required_in + baseline, marker_of=marker_of, baseline=baseline,
        required=required_in, excluded=excluded_in, missing=missing, dropped_groups=dropped,
        size=min(size, len(required_in) + len(baseline)), per_group=per_group,
        restricted=len(allow) if allow is not None else None, seed=seed, notes=notes,
    )


def read_matrix(adaptor, genes: list[str], cells: np.ndarray) -> np.ndarray:
    """Log-normalised expression of ``genes`` in ``cells`` (sorted positions), dense
    float64 (cells x genes). A large dataset reads only these genes of these cells from
    its gene store; nothing else of the matrix is read."""
    if not genes:
        return np.zeros((cells.size, 0))
    if adaptor.is_large:
        block = adaptor.gene_block(genes, cells)
        return np.asarray(block.toarray(), dtype=np.float64)
    from scview.core.activity import lognorm_matrix

    M, names, _ = lognorm_matrix(adaptor.adata)
    pos = pd.Index(pd.Index(names).astype(str)).get_indexer(genes)
    if (pos < 0).any():
        raise KeyError("Genes not in the dataset: " + ", ".join(np.asarray(genes)[pos < 0][:5]))
    rows = M[cells]
    if hasattr(rows, "to_memory"):
        rows = rows.to_memory()
    order, inverse = np.unique(pos, return_inverse=True)
    sub = rows[:, order]
    dense = sub.toarray() if sp.issparse(sub) else np.asarray(sub)
    return np.nan_to_num(np.asarray(dense, dtype=np.float64)[:, inverse])


def standardise(X: np.ndarray) -> np.ndarray:
    mean = X.mean(axis=0)
    sd = X.std(axis=0)
    return np.where(sd > 0, (X - mean) / np.where(sd > 0, sd, 1.0), 0.0)


# ---------------------------------------------------------------------------
# Fitting


def _model():
    from sklearn.linear_model import LogisticRegression

    return LogisticRegression(class_weight="balanced", C=C, max_iter=MAX_ITER)


def folds(y: np.ndarray, seed: int = SEED) -> list[tuple[np.ndarray, np.ndarray]]:
    from sklearn.model_selection import StratifiedKFold

    skf = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=seed)
    return list(skf.split(np.zeros(y.size), y))


def cv_predict(Z: np.ndarray, y: np.ndarray, splits, n_classes: int) -> np.ndarray:
    """Each cell's group predicted by the model fitted on the other folds."""
    import warnings

    from sklearn.exceptions import ConvergenceWarning

    pred = np.empty(y.size, dtype=np.int64)
    for train, test in splits:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            m = _model().fit(Z[train], y[train])
        pred[test] = m.predict(Z[test])
    return pred


def recalls(y: np.ndarray, pred: np.ndarray, n_classes: int) -> np.ndarray:
    n = np.bincount(y, minlength=n_classes).astype(float)
    hit = np.bincount(y[pred == y], minlength=n_classes).astype(float)
    return np.divide(hit, n, out=np.zeros(n_classes), where=n > 0)


def _fit_all(Z: np.ndarray, y: np.ndarray):
    import warnings

    from sklearn.exceptions import ConvergenceWarning

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        return _model().fit(Z, y)


def score_test(Zc: np.ndarray, y: np.ndarray, P: np.ndarray, n_classes: int) -> np.ndarray:
    """For each candidate column of ``Zc``: the norm over groups of the gradient of the
    class-balanced multinomial log-likelihood with respect to its coefficients at 0,
    given the current model's probabilities ``P`` (cells x groups)."""
    n = np.bincount(y, minlength=n_classes).astype(float)
    w = (y.size / (n_classes * np.maximum(n, 1)))[y]
    R = -P.copy()
    R[np.arange(y.size), y] += 1.0
    G = Zc.T @ (R * w[:, None])  # candidates x groups
    return np.sqrt((G ** 2).sum(axis=1))


def _probabilities(Z: np.ndarray, y: np.ndarray, n_classes: int) -> np.ndarray:
    if Z.shape[1] == 0:  # the intercept-only model of balanced classes
        return np.full((y.size, n_classes), 1.0 / n_classes)
    return _fit_all(Z, y).predict_proba(Z)


def _entry(gene: str, k: int, y: np.ndarray, pred: np.ndarray, groups: list[str], *,
           required: bool, tried: int) -> dict[str, Any]:
    r = recalls(y, pred, len(groups))
    return {"gene": gene, "n_genes": k, "balanced_accuracy": round(float(r.mean()), 4),
            "recall": {g: round(float(v), 4) for g, v in zip(groups, r)},
            "required": required, "tried": tried}


def iter_run(adaptor, plan: Plan, stop: Callable[[], bool] | None = None,
             matrix: np.ndarray | None = None) -> Iterator[tuple[str, dict[str, Any]]]:
    """Run ``plan``: yields ``("start", plan)``, ``("step", ...)`` per gene added,
    ``("baseline", ...)`` and ``("result", ...)``. ``stop`` is checked before each step."""
    t0 = time.monotonic()
    yield "start", plan.to_dict()
    groups, y, k = plan.groups, plan.y, len(plan.groups)
    X = matrix if matrix is not None else read_matrix(adaptor, plan.candidates, plan.cells)
    Z = standardise(X)
    col = {g: i for i, g in enumerate(plan.candidates)}
    splits = folds(y, plan.seed)

    def check() -> None:
        if stop is not None and stop():
            raise Cancelled()

    selected: list[str] = []
    path: list[dict[str, Any]] = []
    for gene in plan.required:
        check()
        selected.append(gene)
        pred = cv_predict(Z[:, [col[g] for g in selected]], y, splits, k)
        path.append(_entry(gene, len(selected), y, pred, groups, required=True, tried=1))
        yield "step", path[-1]
    remaining = [g for g in plan.baseline if g not in selected]
    while len(selected) < plan.size and remaining:
        check()
        cur = Z[:, [col[g] for g in selected]]
        P = _probabilities(cur, y, k)
        scores = score_test(Z[:, [col[g] for g in remaining]], y, P, k)
        top = [remaining[i] for i in np.argsort(-scores, kind="stable")[:SCREEN]]
        best: tuple[float, str, np.ndarray] | None = None
        for gene in top:
            pred = cv_predict(np.column_stack([cur, Z[:, col[gene]]]), y, splits, k)
            acc = float(recalls(y, pred, k).mean())
            if best is None or acc > best[0] + 1e-12:
                best = (acc, gene, pred)
        assert best is not None
        selected.append(best[1])
        remaining.remove(best[1])
        path.append(_entry(best[1], len(selected), y, best[2], groups, required=False, tried=len(top)))
        yield "step", path[-1]

    check()
    base: list[dict[str, Any]] = []
    for n in range(1, min(plan.size, len(plan.baseline)) + 1):
        pred = cv_predict(Z[:, [col[g] for g in plan.baseline[:n]]], y, splits, k)
        base.append({"gene": plan.baseline[n - 1], "n_genes": n,
                     "balanced_accuracy": round(float(recalls(y, pred, k).mean()), 4)})
    yield "baseline", {"path": base}

    check()
    Zs = Z[:, [col[g] for g in selected]]
    pred = cv_predict(Zs, y, splits, k)
    conf = np.zeros((k, k), dtype=np.int64)
    np.add.at(conf, (y, pred), 1)
    model = _fit_all(Zs, y)
    coef = np.asarray(model.coef_)
    if coef.shape[0] == 1:  # binary: sklearn keeps one row, for the second class
        coef = np.vstack([-coef[0], coef[0]])
    genes_out = []
    for j, gene in enumerate(selected):
        c = coef[:, j]
        pos = c[c > 0]
        helps = ([groups[i] for i in np.argsort(-c) if c[i] > 0 and c[i] >= 0.5 * pos.max()]
                 if pos.size else [])
        genes_out.append({
            "gene": gene, "required": gene in plan.required, "helps": helps[:4],
            "marker_of": plan.marker_of.get(gene, []),
            "coef": {g: round(float(v), 3) for g, v in zip(groups, c)},
        })
    accs = [p["balanced_accuracy"] for p in path]
    best_acc = max(accs) if accs else 0.0
    best_n = next((i + 1 for i, a in enumerate(accs) if a >= best_acc - WITHIN), len(accs))
    result = {
        **plan.to_dict(),
        "panel": selected, "path": path, "baseline": base, "genes": genes_out,
        "confusion": {"groups": groups, "counts": conf.tolist()},
        "balanced_accuracy": round(float(recalls(y, pred, k).mean()), 4),
        "baseline_accuracy": base[-1]["balanced_accuracy"] if base else None,
        "best_n": best_n, "best_accuracy": round(best_acc, 4), "within": WITHIN,
        "importance_note": IMPORTANCE_NOTE, "seconds": round(time.monotonic() - t0, 1),
    }
    _cache.put(adaptor.adata, result)
    yield "result", result


def run(adaptor, plan: Plan, matrix: np.ndarray | None = None) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for event, data in iter_run(adaptor, plan, matrix=matrix):
        if event == "result":
            out = data
    return out


# ---------------------------------------------------------------------------
# The latest result, held on the loaded dataset

_cache = ObjectCache()


def latest(adata) -> dict[str, Any] | None:
    res = _cache.get(adata)
    if res is None or res["groupby"] not in adata.obs.columns:
        return None
    return res


def grounding_lines(adata) -> tuple[list[str], str]:
    """One fact line for the co-pilot once a panel has been made for this dataset."""
    res = latest(adata)
    if res is None or not res["panel"]:
        return [], ""
    k = len(res["panel"])
    worst = sorted(res["path"][-1]["recall"].items(), key=lambda kv: kv[1])[:3]
    base = res["baseline_accuracy"]
    vs = (f"; the top {len(res['baseline'])} markers taken in turn from each group reach "
          f"{base:.2f}" if base is not None else "")
    line = (f"- Minimal marker panel for '{res['groupby']}' ({len(res['groups'])} groups, "
            f"{res['n_cells']:,} sampled cells, up to {res['cells_per_group']} per group, "
            f"{res['cv']}, multinomial logistic regression): {k} genes "
            f"{', '.join(res['panel'])} reach balanced accuracy {res['balanced_accuracy']:.2f}{vs}. "
            f"{res['best_n']} genes come within {res['within']:.2f} of the best. "
            f"Hardest groups (recall): " + ", ".join(f"{g} {r:.2f}" for g, r in worst)
            + f" [{TAG}]")
    return [line], f"{k}-gene panel for '{res['groupby']}', balanced accuracy {res['balanced_accuracy']:.2f}"
