"""How well each cell's label fits: split conformal prediction sets (C058).

For one label column (cell types from any annotation method, a clustering, hand
labels), a classifier learns the labels from the PCA the neighbour graph is built
on, and split conformal prediction turns its probabilities into a prediction set
per cell at an error rate ``alpha``: when cells are exchangeable, a cell's own
label is in its set with probability at least 1 - alpha.

**What this measures:** how well each cell's label agrees with the expression
neighbourhood of the cells carrying that label, not whether the label is
biologically right. A label that is wrong for a whole cluster fits perfectly.

Each checked cell gets a flag:

- *confident*: the set is the cell's own label alone;
- *ambiguous*: the set holds the own label and others, which the expression does
  not tell apart at this alpha (the others are the competing labels);
- *unsure*: the set is empty, no label fits well enough at this alpha;
- *doesn't fit*: the set holds other labels but not the cell's own.

Unsure and doesn't-fit cells are the rejected ones: up to about alpha of all cells
are rejected even when every label is right, because the threshold is set so that
1 - alpha of the labels are kept. A label with far more than alpha of its cells
rejected fits its neighbourhood worse than the rest. Probabilities of at least
``CERTAIN`` count as equally certain (their scores tie at 0, which keeps the
guarantee, conservatively): otherwise near-perfect labels would still lose alpha
of their cells over differences in the fourth decimal. When the classifier is
right more often than 1 - alpha, sets shrink rather than grow: in-between cells
come out unsure rather than ambiguous (Kang's published labels at alpha 0.1),
and a smaller alpha shows which labels they are torn between.

Scores for every cell come from a model that never saw it (cross-fitting): the
cells are split in two halves, stratified by label, and a model trained on one
half scores the other. In each half, a random half of the cells sets the
threshold (calibration) and the other half checks it; coverage on those held-out
cells is what the guarantee is about, and is reported.

The score is LAC, 1 - p(label) (Sadinle, Lei & Wasserman 2019): the smallest
sets on average, and it can give an empty set, the "fits no label" answer this
check wants. APS (Romano et al. 2020) is efficient only with randomised scores,
which would flip confident cells to empty sets at random, and without them its
sets grow far too large. Coverage is marginal over cells: calibrating each label
on its own (Mondrian) would give every label the same share of rejects, hiding
the labels that fit badly, and needs about 1/alpha calibration cells per label.

The classifier is a multinomial logistic regression on the standardised PCs. A
15-nearest-neighbour label vote (``classifier="knn"``) is the more literal
"neighbourhood", but its scores take 16 values, so many cells tie at the
threshold, and a small label can't win a vote among cells of a big one: on Kang
it rejected 44 % of "T activated" cells (logistic regression: 22 %) and caught
88 % of planted wrong labels as doesn't fit (95 %).

A column annotated by CellTypist can be scored by the reference model instead
(``classifier="celltypist"``): the pipeline keeps CellTypist's per-cell
probabilities for the column's labels (:func:`store_reference_probabilities`),
rescaled here to sum to 1 over those labels (CellTypist's are one-vs-rest, and
spread over its ~100 finer classes; unscaled, a 90 % coverage needed p >= 0.005
and most sets held most labels). No model is fitted to the dataset, so the score
is independent of the clusters the labels were voted over: on Kang's Leiden 0.5
clusters it rejected 35 % of the "CD16+ NK cells" cluster, which also holds the
CD8 T cells (logistic regression: 17 %). Its sets are larger, since the model's
own call per cell matched its cluster's vote for 78 % of cells.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import time
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd

from scview.core import provenance
from scview.core.obs_writes import forget_column_results

logger = logging.getLogger(__name__)

DEFAULT_ALPHA = 0.1
MIN_ALPHA, MAX_ALPHA = 0.01, 0.5
MAX_LABELS = 100  # pickers and markers handle at most 100 groups per column
MIN_LABEL_CELLS = 20  # fewer and a label is not checked (5 cells per quarter)
MIN_CELLS = 200
MAX_PCS = 50
MAX_TRAIN = 25_000  # training cells per model; a stratified sample beyond this
K_NEIGHBORS = 15  # for classifier="knn" (the pipeline's neighbour count)
CERTAIN = 0.95  # a label this probable or more is as certain as any (see the module doc)
SEED = 0
CLASSIFIERS = ("logreg", "knn", "celltypist")
DEFAULT_CLASSIFIER = "logreg"
_CLASSIFIER_NAMES = {
    "logreg": "multinomial logistic regression",
    "knn": f"{K_NEIGHBORS}-nearest-neighbour label vote",
    "celltypist": "CellTypist reference model's probabilities",
}

CONFIDENT = "confident"
AMBIGUOUS = "ambiguous"
UNSURE = "unsure"
NOT_FIT = "doesn't fit"
NOT_CHECKED = "not checked"
FLAGS = [CONFIDENT, AMBIGUOUS, UNSURE, NOT_FIT]
REJECTED = (UNSURE, NOT_FIT)  # the own label is not in the set
UNKNOWN = "Unknown"

# Every output column starts with this; scView owns them (the pipeline's reset
# drops scview_* columns, and hand labels can't be given such a name).
PREFIX = "scview_conf_"
# uns: JSON {label column: the last check's settings, coverage and summary}
REGISTRY_KEY = "scview_annotation_confidence"
NOTE = (
    "This measures how well each cell's label agrees with the expression "
    "neighbourhood of the cells carrying that label, not whether the label is right."
)
REFERENCE_NOTE = (
    "This measures how well each cell's label agrees with the CellTypist reference "
    "model's call for that cell, not whether the label is right."
)
# uns: JSON {label column: {"obsm": key, "labels": [...], "model": ..., "fingerprint": ...}}
# for CellTypist's per-cell probabilities of the column's labels (obsm[key]).
REFERENCE_KEY = "scview_reference_probabilities"
_SAFE = re.compile(r"[^A-Za-z0-9_.\-]+")
N_CONFUSIONS = 3  # per label


class ConfidenceError(ValueError):
    """A request that cannot be served; ``status`` is the HTTP code."""

    def __init__(self, message: str, status: int = 400) -> None:
        super().__init__(message)
        self.status = status


# ---------------------------------------------------------------------------
# Registry and column names
# ---------------------------------------------------------------------------


def registry(adata) -> dict[str, dict[str, Any]]:
    """The stored checks, by label column (empty when none or unreadable)."""
    raw = adata.uns.get(REGISTRY_KEY)
    try:
        data = json.loads(raw) if isinstance(raw, str) else {}
    except ValueError:
        data = {}
    return data if isinstance(data, dict) else {}


def _save_registry(adata, data: dict[str, dict[str, Any]]) -> None:
    # A new string each time: write_derived rewrites uns entries it sees replaced.
    adata.uns[REGISTRY_KEY] = json.dumps(data)


def is_output(name: object) -> bool:
    return str(name).startswith(PREFIX)


def column_names(base: str) -> dict[str, str]:
    return {
        "flag": f"{base}_flag",
        "fit": f"{base}_fit",
        "set_size": f"{base}_set_size",
        "alt": f"{base}_alt",
        "unknown": f"{base}_unknown",
    }


def _base_for(adata, column: str, reg: dict[str, dict[str, Any]]) -> str:
    """The output-name stem for ``column``: the one its last check used, else a
    new one whose names no other check uses and no obs column has, so a column
    scView did not write is never overwritten."""
    entry = reg.get(column)
    if isinstance(entry, dict) and isinstance(entry.get("base"), str):
        return entry["base"]
    stem = _SAFE.sub("_", column).strip("_")[:40] or "labels"
    taken = {e.get("base") for c, e in reg.items() if c != column and isinstance(e, dict)}
    base, n = f"{PREFIX}{stem}", 2
    while base in taken or any(c in adata.obs.columns for c in column_names(base).values()):
        base, n = f"{PREFIX}{stem}_{n}", n + 1
    return base


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------


def representation(adata) -> tuple[str, int]:
    """The obsm key the classifier learns from (the batch-corrected PCA when there
    is one, as for the neighbour graph) and how many of its components."""
    for key in ("X_pca_harmony", "X_scanorama", "X_pca"):
        if key in adata.obsm:
            return key, int(min(MAX_PCS, np.asarray(adata.obsm[key]).shape[1]))
    raise ConfidenceError(
        "The confidence check needs a PCA, and this dataset has none. Run the "
        "processing pipeline (through PCA) first."
    )


def _as_labels(col: pd.Series) -> pd.Series:
    return col.astype(str).where(col.notna())


def _label_values(adata, column: str) -> pd.Series:
    if column not in adata.obs.columns:
        raise ConfidenceError(f"Column '{column}' not found.", status=404)
    if is_output(column):
        raise ConfidenceError("That column is an output of the confidence check; pick a label column.")
    return _as_labels(adata.obs[column])


def fingerprint(values: pd.Series) -> str:
    """Identity of a column's labels, so a stored check can tell it went stale."""
    hashed = pd.util.hash_pandas_object(values.fillna("\x00"), index=False).to_numpy()
    return hashlib.sha1(hashed.tobytes()).hexdigest()[:16]


def _labels(adata, column: str) -> tuple[pd.Series, list[str], np.ndarray, list[dict[str, Any]]]:
    """(values, checked labels, codes into them with -1 for unchecked cells, the
    labels left out for having too few cells)."""
    values = _label_values(adata, column)
    counts = values.value_counts()
    if len(counts) > MAX_LABELS:
        raise ConfidenceError(
            f"'{column}' has {len(counts)} different values; the confidence check takes a "
            f"label column with 2 to {MAX_LABELS} labels."
        )
    col = adata.obs[column]
    if isinstance(col.dtype, pd.CategoricalDtype):
        order = [str(c) for c in col.cat.categories if str(c) in counts.index]
    else:
        order = [str(c) for c in counts.index]
    labels = [lab for lab in order if counts[lab] >= MIN_LABEL_CELLS]
    skipped = [{"label": lab, "n_cells": int(counts[lab])} for lab in order if lab not in labels]
    if len(labels) < 2:
        raise ConfidenceError(
            f"The confidence check needs at least two labels with {MIN_LABEL_CELLS} or more "
            f"cells each; '{column}' has {len(labels)}."
        )
    lookup = {lab: i for i, lab in enumerate(labels)}
    codes = values.map(lookup).fillna(-1).to_numpy(dtype=np.int64)
    if int((codes >= 0).sum()) < MIN_CELLS:
        raise ConfidenceError(f"The confidence check needs at least {MIN_CELLS} labelled cells.")
    return values, labels, codes, skipped


def _quarters(codes: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Each checked cell's quarter, stratified by label: 0/1 calibrate halves 0/1,
    2/3 check them; -1 for cells not checked."""
    q = np.full(codes.size, -1, dtype=np.int8)
    for k in range(int(codes.max()) + 1):
        idx = rng.permutation(np.flatnonzero(codes == k))
        q[idx] = np.arange(idx.size) % 4
    return q


# ---------------------------------------------------------------------------
# A reference model's probabilities (CellTypist)
# ---------------------------------------------------------------------------


def _reference_registry(adata) -> dict[str, dict[str, Any]]:
    raw = adata.uns.get(REFERENCE_KEY)
    try:
        data = json.loads(raw) if isinstance(raw, str) else {}
    except ValueError:
        data = {}
    return data if isinstance(data, dict) else {}


def store_reference_probabilities(adata, column: str, probabilities: pd.DataFrame, model: str) -> None:
    """Keep a reference model's per-cell probabilities of ``column``'s labels
    (CellTypist's ``probability_matrix``, cells x model classes) for the check."""
    col = adata.obs[column]
    cats = col.cat.categories if isinstance(col.dtype, pd.CategoricalDtype) else col.dropna().unique()
    labels = [str(c) for c in cats if str(c) in probabilities.columns]
    reg = _reference_registry(adata)
    old = reg.get(column, {}).get("obsm") if isinstance(reg.get(column), dict) else None
    if old:
        adata.obsm.pop(old, None)
    if not labels:
        reg.pop(column, None)
    else:
        key = "scview_refprob_" + hashlib.sha1(column.encode()).hexdigest()[:10]
        frame = probabilities.reindex(index=adata.obs_names, columns=labels)
        adata.obsm[key] = frame.fillna(0.0).to_numpy(dtype=np.float32)
        reg[column] = {"obsm": key, "labels": labels, "model": model,
                       "fingerprint": fingerprint(_as_labels(col))}
    adata.uns[REFERENCE_KEY] = json.dumps(reg)


def drop_reference_probabilities(adata, column: str) -> None:
    """Forget the reference probabilities of ``column`` (it was annotated again
    another way)."""
    reg = _reference_registry(adata)
    entry = reg.pop(column, None)
    if entry is None:
        return
    if isinstance(entry, dict) and entry.get("obsm"):
        adata.obsm.pop(entry["obsm"], None)
    adata.uns[REFERENCE_KEY] = json.dumps(reg)


def reference_probabilities(adata, column: str) -> tuple[np.ndarray, list[str], str]:
    """(probabilities, their labels, model) stored for ``column``; a
    ConfidenceError saying why when there are none to use."""
    entry = _reference_registry(adata).get(column)
    if not isinstance(entry, dict):
        raise ConfidenceError(
            f"No CellTypist probabilities are kept for '{column}': only a column annotated "
            "with CellTypist has them (annotate it again if that was before they were kept)."
        )
    P = adata.obsm.get(entry.get("obsm", "")) if entry.get("obsm") else None
    labels = entry.get("labels") or []
    if P is None or np.asarray(P).shape != (adata.n_obs, len(labels)):
        raise ConfidenceError(f"The CellTypist probabilities kept for '{column}' are incomplete; annotate again.")
    if column not in adata.obs.columns or fingerprint(_as_labels(adata.obs[column])) != entry.get("fingerprint"):
        raise ConfidenceError(
            f"'{column}' changed since CellTypist annotated it, so its probabilities no longer "
            "match; annotate again or score with a classifier on the PCA."
        )
    return np.asarray(P), [str(lab) for lab in labels], str(entry.get("model") or "CellTypist")


def reference_columns(adata) -> dict[str, str]:
    """Label columns whose CellTypist probabilities can be used: {column: model}."""
    out = {}
    for column in _reference_registry(adata):
        try:
            out[column] = reference_probabilities(adata, column)[2]
        except ConfidenceError:
            pass
    return out


def _subsample(y: np.ndarray, cap: int, rng: np.random.Generator) -> np.ndarray:
    """Indices of at most ~``cap`` cells, every label kept in proportion."""
    if y.size <= cap:
        return np.arange(y.size)
    keep = []
    for k in np.unique(y):
        idx = np.flatnonzero(y == k)
        take = max(min(idx.size, MIN_LABEL_CELLS), int(round(idx.size * cap / y.size)))
        keep.append(rng.choice(idx, size=take, replace=False))
    return np.sort(np.concatenate(keep))


# ---------------------------------------------------------------------------
# Classifier and conformal sets
# ---------------------------------------------------------------------------


def _fit_predict(
    classifier: str, X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, n_labels: int
) -> np.ndarray:
    """P(label | cell) for the ``X_test`` cells, from a model of ``X_train``."""
    P = np.zeros((X_test.shape[0], n_labels), dtype=np.float64)
    if classifier == "knn":
        # The share of each label among the cell's nearest training cells.
        from sklearn.neighbors import NearestNeighbors

        k = min(K_NEIGHBORS, X_train.shape[0])
        idx = NearestNeighbors(n_neighbors=k).fit(X_train).kneighbors(X_test, return_distance=False)
        neighbour_labels = y_train[idx]
        for j in range(n_labels):
            P[:, j] = (neighbour_labels == j).mean(axis=1)
        return P
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler().fit(X_train)
    model = LogisticRegression(max_iter=1000).fit(scaler.transform(X_train), y_train)
    P[:, model.classes_] = model.predict_proba(scaler.transform(X_test))
    return P


def _reference_scores(adata, column: str, labels: list[str]) -> tuple[np.ndarray, str]:
    """The stored reference probabilities of ``labels``, rescaled to sum to 1
    per cell, and what they came from."""
    ref, ref_labels, model = reference_probabilities(adata, column)
    missing = [lab for lab in labels if lab not in ref_labels]
    if missing:
        raise ConfidenceError(
            f"The CellTypist probabilities kept for '{column}' lack {', '.join(missing[:3])}; "
            "annotate again."
        )
    P = ref[:, [ref_labels.index(lab) for lab in labels]].astype(np.float64)
    total = P.sum(axis=1, keepdims=True)
    P = np.divide(P, total, out=np.full_like(P, 1.0 / len(labels)), where=total > 0)
    return P, f"CellTypist {model.removesuffix('.pkl')}"


def _threshold(cal_scores: np.ndarray, alpha: float) -> float:
    """The split conformal threshold: the ceil((n + 1)(1 - alpha))-th smallest
    calibration score."""
    n = cal_scores.size
    k = int(np.ceil((n + 1) * (1 - alpha)))
    if k > n:
        raise ConfidenceError(
            f"Too few cells to calibrate at alpha = {alpha:g} ({n} per half); raise alpha."
        )
    return float(np.sort(cal_scores)[k - 1])


def _p_values(cal_scores: np.ndarray, scores: np.ndarray) -> np.ndarray:
    """Conformal p-value of each score: (1 + calibration scores >= it) / (n + 1).
    A label is in a set exactly when its p-value is above alpha."""
    s = np.sort(cal_scores)
    n_ge = s.size - np.searchsorted(s, scores, side="left")
    return (1 + n_ge) / (s.size + 1)


# ---------------------------------------------------------------------------
# The check
# ---------------------------------------------------------------------------


def clean_alpha(alpha: float | None) -> float:
    a = DEFAULT_ALPHA if alpha is None else float(alpha)
    if not MIN_ALPHA <= a <= MAX_ALPHA:
        raise ConfidenceError(f"Alpha must be between {MIN_ALPHA} and {MAX_ALPHA}.")
    return a


def assess(
    adata,
    column: str,
    alpha: float | None = DEFAULT_ALPHA,
    classifier: str = DEFAULT_CLASSIFIER,
) -> dict[str, Any]:
    """Check how well each cell's label in ``column`` fits (see the module doc).

    Writes the flag, fit (the own label's conformal p-value), set size and
    competing-label columns, and keeps the settings, coverage and per-label
    summary in ``uns[REGISTRY_KEY]``. Everything is validated and computed before
    ``adata`` changes, so a ConfidenceError leaves it untouched. Returns
    :func:`summary` for the column.
    """
    t0 = time.perf_counter()
    alpha = clean_alpha(alpha)
    if classifier not in CLASSIFIERS:
        raise ConfidenceError(
            f"Unknown classifier '{classifier}' (expected one of {', '.join(CLASSIFIERS)})."
        )
    values, labels, codes, skipped = _labels(adata, column)
    L = len(labels)
    rng = np.random.default_rng(SEED)
    quarter = _quarters(codes, rng)
    half = np.where(quarter >= 0, quarter % 2, -1)

    if classifier == "celltypist":
        P, rep = _reference_scores(adata, column, labels)
        n_pcs = 0
        tool = f"split conformal prediction (LAC) over {rep} probabilities"
    else:
        rep, n_pcs = representation(adata)
        X = np.asarray(adata.obsm[rep])[:, :n_pcs].astype(np.float64)
        if not np.isfinite(X).all():
            raise ConfidenceError(f"'{rep}' holds missing or infinite values; re-run the pipeline.")
        P = np.zeros((adata.n_obs, L), dtype=np.float64)
        for h in (0, 1):  # a model of each half scores the other
            train = np.flatnonzero(half == 1 - h)
            train = train[_subsample(codes[train], MAX_TRAIN, rng)]
            test = np.flatnonzero(half == h)
            P[test] = _fit_predict(classifier, X[train], codes[train], X[test], L)
        tool = f"split conformal prediction (LAC) over a {_CLASSIFIER_NAMES[classifier]} on {rep}"

    rows = np.flatnonzero(quarter >= 0)
    own = codes[rows]
    at_own = np.arange(rows.size), own
    scores = 1.0 - P[rows]  # LAC score of every label for every checked cell
    scores[scores <= 1.0 - CERTAIN] = 0.0
    own_score = scores[at_own]
    in_set = np.zeros((rows.size, L), dtype=bool)
    fit = np.zeros(rows.size, dtype=np.float64)
    thresholds = []
    for h in (0, 1):
        mine = half[rows] == h
        cal = quarter[rows] == h  # quarter 0 calibrates half 0, quarter 1 half 1
        q = _threshold(own_score[cal], alpha)
        thresholds.append(q)
        in_set[mine] = scores[mine] <= q
        fit[mine] = _p_values(own_score[cal], own_score[mine])

    own_in = in_set[at_own]
    size = in_set.sum(axis=1)
    flag = np.select(
        [own_in & (size == 1), own_in, size == 0], [CONFIDENT, AMBIGUOUS, UNSURE], NOT_FIT
    )
    others = P[rows].copy()
    others[at_own] = -np.inf
    alt = np.where(flag != CONFIDENT, others.argmax(axis=1), -1)  # the nearest other label
    held_out = quarter[rows] >= 2

    # Everything computed: now the dataset changes.
    reg = registry(adata)
    previous = reg.get(column) if isinstance(reg.get(column), dict) else {}
    base = _base_for(adata, column, reg)
    names = column_names(base)
    n = adata.n_obs
    flag_col = np.full(n, NOT_CHECKED, dtype=object)
    flag_col[rows] = flag
    fit_col = np.full(n, np.nan, dtype=np.float32)
    fit_col[rows] = fit
    size_col = np.full(n, np.nan, dtype=np.float32)
    size_col[rows] = size
    alt_col = np.full(n, None, dtype=object)
    alt_col[rows[alt >= 0]] = [labels[i] for i in alt[alt >= 0]]
    cats = FLAGS + ([NOT_CHECKED] if rows.size < n else [])
    adata.obs[names["flag"]] = pd.Categorical(flag_col, categories=cats)
    adata.obs[names["fit"]] = fit_col
    adata.obs[names["set_size"]] = size_col
    adata.obs[names["alt"]] = pd.Categorical(alt_col, categories=labels).remove_unused_categories()
    for key in ("flag", "alt"):
        forget_column_results(adata, names[key])  # markers of the old groups

    cov = float(own_in[held_out].mean())
    n_cal, n_held = int((~held_out).sum()), int(held_out.sum())
    # Sampling spread of held-out coverage: the thresholds' (n_cal) plus the check's (n_held).
    sd = float(np.sqrt(alpha * (1 - alpha) * (1 / n_cal + 1 / n_held)))
    entry: dict[str, Any] = {
        "base": base,
        "columns": {k: v for k, v in names.items() if k != "unknown"},
        "alpha": alpha,
        "score": "lac",
        "classifier": classifier,
        "representation": rep,
        "n_pcs": n_pcs,
        "n_cells": int(n),
        "n_checked": int(rows.size),
        "labels_not_checked": skipped,
        "thresholds": [round(t, 6) for t in thresholds],
        "coverage": {
            "target": round(1 - alpha, 4),
            "held_out": round(cov, 4),
            "n_held_out": n_held,
            "calibration": round(float(own_in[~held_out].mean()), 4),
            "n_calibration": n_cal,
            "sd": round(sd, 4),
            "within_sampling_error": bool(cov >= 1 - alpha - 2 * sd),
        },
        "accuracy": round(float((P[rows].argmax(axis=1) == own).mean()), 4),
        "mean_set_size": round(float(size.mean()), 4),
        "counts": {f: int((flag == f).sum()) for f in FLAGS},
        "labels": _per_label(labels, own, flag, alt, held_out),
        "fingerprint": fingerprint(values),
        "unknown_column": None,
        "elapsed_seconds": 0.0,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    entry["confusions"] = _confusions(entry["labels"])
    entry["elapsed_seconds"] = round(time.perf_counter() - t0, 2)
    reg[column] = entry
    _save_registry(adata, reg)
    unknown = previous.get("unknown_column")
    if unknown and unknown in adata.obs.columns:
        make_unknown_column(adata, column)  # keep the copy in step with the new flags
    try:
        provenance.record_step(
            adata,
            step="annotation_confidence",
            tool=tool,
            params={"column": column, "alpha": alpha, "classifier": classifier,
                    "representation": rep, "n_pcs": n_pcs, "seed": SEED,
                    "coverage_held_out": entry["coverage"]["held_out"],
                    "outputs": list(entry["columns"].values())},
        )
    except Exception as e:  # provenance must never block the result
        logger.warning("Could not record annotation-confidence provenance: %s", e)
    logger.info(
        "Annotation confidence for '%s' (%s, alpha %.2f): held-out coverage %.3f; %s; %.1f s",
        column, classifier, alpha, cov, entry["counts"], entry["elapsed_seconds"],
    )
    return summary(adata, column)


def _per_label(
    labels: list[str], own: np.ndarray, flag: np.ndarray, alt: np.ndarray, held_out: np.ndarray
) -> list[dict[str, Any]]:
    """Per label: the share of its cells with each flag, held-out coverage (not
    guaranteed per label) and the labels its flagged cells are nearest to."""
    out = []
    for k, label in enumerate(labels):
        mine = own == k
        n = int(mine.sum())
        others, counts = np.unique(alt[mine & (flag != CONFIDENT)], return_counts=True)
        order = np.argsort(-counts, kind="stable")[:N_CONFUSIONS]
        held = mine & held_out
        row: dict[str, Any] = {"label": label, "n_cells": n}
        for f in FLAGS:
            row[_FLAG_KEYS[f]] = round(float((flag[mine] == f).mean()), 4)
        row["coverage_held_out"] = (
            round(float(np.isin(flag[held], REJECTED, invert=True).mean()), 4) if held.any() else None
        )
        row["confused_with"] = [
            {"label": labels[int(others[i])], "n_cells": int(counts[i]),
             "share": round(float(counts[i]) / n, 4)}
            for i in order
        ]
        out.append(row)
    return out


_FLAG_KEYS = {CONFIDENT: "confident", AMBIGUOUS: "ambiguous", UNSURE: "unsure", NOT_FIT: "not_fit"}


def _confusions(per_label: list[dict[str, Any]], top: int = 10) -> list[dict[str, Any]]:
    """The commonest (label, nearest other label) pairs among flagged cells."""
    pairs = [
        {"label": row["label"], "other": c["label"], "n_cells": c["n_cells"], "share": c["share"]}
        for row in per_label for c in row["confused_with"]
    ]
    return sorted(pairs, key=lambda p: -p["n_cells"])[:top]


def summary(adata, column: str) -> dict[str, Any] | None:
    """The last check of ``column``, or None when there is none (or its columns
    are gone, e.g. after a reprocess from counts). ``stale`` is true once the
    column's labels changed since the check ran."""
    entry = registry(adata).get(column)
    if not isinstance(entry, dict) or column not in adata.obs.columns:
        return None
    flag_name = (entry.get("columns") or {}).get("flag")
    if not flag_name or flag_name not in adata.obs.columns:
        return None
    out = {k: v for k, v in entry.items() if k not in ("base", "fingerprint")}
    unknown = entry.get("unknown_column")
    out["unknown_column"] = unknown if unknown and unknown in adata.obs.columns else None
    out["column"] = column
    out["stale"] = fingerprint(_label_values(adata, column)) != entry.get("fingerprint")
    out["note"] = REFERENCE_NOTE if entry.get("classifier") == "celltypist" else NOTE
    return out


def make_unknown_column(adata, column: str) -> dict[str, Any]:
    """A copy of ``column`` with every rejected cell (unsure or doesn't fit) set to
    UNKNOWN, written as the check's own ``..._unknown`` column; the label column
    itself is never changed."""
    reg = registry(adata)
    entry = reg.get(column)
    values = _label_values(adata, column)
    flag_name = (entry.get("columns") or {}).get("flag") if isinstance(entry, dict) else None
    if not flag_name or flag_name not in adata.obs.columns:
        raise ConfidenceError(f"Run the confidence check on '{column}' first.", status=409)
    if fingerprint(values) != entry.get("fingerprint"):
        raise ConfidenceError(
            f"'{column}' changed since its confidence check ran; run the check again.", status=409
        )
    name = column_names(entry["base"])["unknown"]
    rejected = adata.obs[flag_name].astype(str).isin(REJECTED).to_numpy()
    col = adata.obs[column]
    if isinstance(col.dtype, pd.CategoricalDtype):
        cats = [str(c) for c in col.cat.categories]
    else:
        cats = sorted(values.dropna().unique())
    out = values.to_numpy(dtype=object).copy()
    out[rejected] = UNKNOWN
    adata.obs[name] = pd.Categorical(
        out, categories=[c for c in cats if c != UNKNOWN] + [UNKNOWN]
    ).remove_unused_categories()
    forget_column_results(adata, name)
    reg[column] = dict(entry, unknown_column=name)
    _save_registry(adata, reg)
    try:
        provenance.record_step(
            adata, step="annotation_confidence_unknown", tool="scview annotation confidence",
            params={"column": column, "output": name, "alpha": entry.get("alpha"),
                    "n_unknown": int(rejected.sum())},
        )
    except Exception as e:  # provenance must never block the result
        logger.warning("Could not record annotation-confidence provenance: %s", e)
    return {"column": column, "unknown_column": name, "n_unknown": int(rejected.sum()),
            "n_cells": int(adata.n_obs)}
