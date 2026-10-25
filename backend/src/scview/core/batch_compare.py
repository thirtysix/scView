"""Compare batch-correction methods on the current PCA, and recommend one.

None / Harmony / Scanorama / BBKNN (core/batch_methods.py) each run on a copy of
``obsm["X_pca"]`` (no reprocessing, nothing saved) and are scored the same way as
the joint embedding's mixing score (core/integration.py ``batch_mixing``): on the
same scored cells and the same number of neighbours ``k`` per cell,

* mixing: the share of a cell's neighbours from another batch, against the share
  were the batches perfectly mixed (``ideal``), and iLISI (effective batches around
  a cell);
* conservation, when a cell-type-like label column exists: the share of a cell's
  neighbours with its label (within its own dataset for a joint dataset's
  ``source_cluster``).

Neighbours are the k nearest in the corrected PCs (none: X_pca), and for BBKNN,
which makes a graph and no embedding, each cell's k strongest graph connections.

``mixing_ratio`` = (iLISI - 1) / (ideal iLISI - 1), 0..1, is what "mixes" is judged
on. The recommendation is a rule, explained in its text: among the methods with a
mixing_ratio of at least ``MIX_OK``, the best conservation; within ``TIE`` of the
best, the simplest (none) or the fastest; without a label column, the simplest
or fastest that mixes. A method whose conservation is ``OVERCORRECTION_DROP``
below none's gets an overcorrection warning. Applying a method is a normal
pipeline run with that ``batch_method`` (nothing from the comparison is kept).
"""

from __future__ import annotations

import logging
import time
from typing import Any, Callable, Iterator

import numpy as np
import pandas as pd

from scview.core import batch_methods as bm
from scview.core.integration import (
    MIXING_MAX_SCORED,
    OVERCORRECTION_DROP,
    _conservation_label,
    _ilisi,
    _label_agreement,
    _neighbours,
)

logger = logging.getLogger(__name__)

NONE = "none"
METHODS: tuple[str, ...] = (NONE, bm.HARMONY, bm.SCANORAMA, bm.BBKNN)
MIX_OK = 0.5  # mixing at least this share of the ideal counts as "mixes acceptably"
TIE = 0.02  # conservation within this of the best is a tie
MAX_CELLS = 200_000  # the in-memory cap; larger datasets are large-dataset mode anyway

# Background, from Luecken et al. 2022, "Benchmarking atlas-level data integration in
# single-cell genomics", Nat Methods 19:41-50 (PMC8748196). The benchmark: 16 methods
# (68 method + preprocessing combinations) on 13 integration tasks (85 batches,
# >1.2 million cells), scored by 14 metrics of batch removal (e.g. kBET, iLISI) and
# biological conservation (e.g. ARI, NMI, cLISI, trajectory), overall score 40 % batch
# removal / 60 % conservation. Findings used here:
# - abstract: "scANVI, Scanorama, scVI and scGen perform well, particularly on complex
#   integration tasks"; Results (Fig. 3a): "Scanorama, scVI and FastMNN (gene) balance
#   these two objectives" and "SAUCIE, LIGER, BBKNN and Seurat v3 tend to favor the
#   removal of batch effects over conservation of biological variation";
# - Discussion: Harmony suits simpler tasks with distinct biological signal but "typically
#   ranks outside the top three when used for complex real data scenarios" (Fig. 3b);
# - Results (scalability, Fig. 5): "ComBat, BBKNN and SAUCIE performed best in terms of
#   runtime"; Discussion: BBKNN is suggested for quickly testing what integration does.
# - Discussion: BBKNN's optimisation is similar to the graph iLISI metric, so mixing
#   metrics flatter it.
BACKGROUND = (
    "In a benchmark of 16 integration methods on 13 tasks (Luecken et al. 2022), Scanorama "
    "was among the best on complex tasks and balanced removing batch effects with keeping "
    "cell types apart. Harmony did well on simpler tasks with distinct cell types but usually "
    "ranked outside the top three on complex ones. BBKNN was among the fastest and leaned "
    "towards removing batch effects over keeping biological differences."
)
BACKGROUND_TIP = (
    "Luecken et al., Nat Methods 19:41 (2022): 68 method and preprocessing combinations on 85 "
    "batches from 23 publications, scored by 14 metrics, 40% batch removal and 60% biological "
    "conservation (cell-type labels, trajectories). scVI and scANVI also ranked high there; "
    "scView does not offer them. The table here scores only this dataset, with two simple "
    "measures, so it can disagree with the benchmark."
)
BBKNN_NOTE = (
    "BBKNN takes the same number of neighbours from every batch, so it mixes the batches "
    "partly by construction and its mixing score flatters it (the benchmark notes the same "
    "for its iLISI)."
)


class CompareError(ValueError):
    """The comparison cannot run on this dataset (the message says why)."""


def default_label(adata, batch_key: str) -> tuple[str | None, str]:
    """The label column conservation is scored with, and its kind (as batch_mixing)."""
    return _conservation_label(adata, batch_key)


def check(adata, batch_key: str, label_column: str | None = None) -> tuple[str | None, str]:
    """(label column, kind) for a comparison of ``batch_key``; CompareError if none can run."""
    if "X_pca" not in adata.obsm:
        raise CompareError("Run PCA first: the methods are compared on the current PCA.")
    if not batch_key or batch_key not in adata.obs.columns:
        raise CompareError(f"Batch column '{batch_key}' not found.")
    batches = adata.obs[batch_key]
    if batches.isna().any():
        raise CompareError(f"Batch column '{batch_key}' has cells with no value.")
    if batches.astype(str).nunique() < 2:
        raise CompareError(f"Batch column '{batch_key}' has only one batch.")
    if adata.n_obs > MAX_CELLS:
        raise CompareError(f"At most {MAX_CELLS:,} cells can be compared.")
    if label_column:
        if label_column not in adata.obs.columns:
            raise CompareError(f"Label column '{label_column}' not found.")
        if label_column == batch_key:
            raise CompareError("The label column must differ from the batch column.")
        kind = "within_dataset" if label_column == "source_cluster" else "shared"
        return label_column, kind
    return default_label(adata, batch_key)


def iter_compare(
    adata,
    batch_key: str,
    *,
    label_column: str | None = None,
    methods: list[str] | None = None,
    n_neighbors: int = 15,
    max_scored: int = MIXING_MAX_SCORED,
    seed: int = 0,
    stop: Callable[[], bool] = lambda: False,
) -> Iterator[tuple[str, dict[str, Any]]]:
    """Run and score each method; yields ``start``, then ``method_start`` / ``method``
    per method, then ``result`` (rows + recommendation). ``stop`` is checked between
    methods (a stopped comparison yields ``stopped``)."""
    label_column, kind = check(adata, batch_key, label_column)
    methods = [m for m in METHODS if m in (methods or METHODS)]
    obs = pd.DataFrame({batch_key: adata.obs[batch_key].astype(str).to_numpy()},
                       index=adata.obs_names.astype(str))
    X = np.asarray(adata.obsm["X_pca"], dtype=np.float32)
    n = X.shape[0]
    names, codes = np.unique(obs[batch_key].to_numpy(), return_inverse=True)
    labels = adata.obs[label_column].astype(str).to_numpy() if label_column else None
    k = int(min(n_neighbors, n - 1))
    rng = np.random.default_rng(seed)
    query = np.sort(rng.choice(n, size=min(n, max_scored), replace=False))
    sizes = np.bincount(codes, minlength=len(names))
    ideal_cell = (n - sizes) / (n - 1)
    ideal = float(ideal_cell[codes[query]].mean())
    shares = sizes / n
    ilisi_ideal = float(1.0 / (shares ** 2).sum())
    yield ("start", {"methods": methods, "n_cells": n, "n_scored": int(query.size), "k": k,
                     "batch_key": batch_key, "n_batches": int(len(names)),
                     "label_column": label_column, "label_kind": kind or None,
                     "ideal": round(ideal, 4), "ilisi_ideal": round(ilisi_ideal, 4)})
    rows: list[dict[str, Any]] = []
    for i, method in enumerate(methods):
        if stop():
            yield ("stopped", {"done": [r["method"] for r in rows]})
            return
        yield ("method_start", {"method": method, "index": i, "total": len(methods),
                                "label": bm.LABELS[method]})
        t0 = time.time()
        row: dict[str, Any] = {"method": method, "label": bm.LABELS[method]}
        try:
            idx = _method_neighbours(method, X, obs, batch_key, query, k, n_neighbors)
            row.update(_score(idx, codes, query, len(names), ilisi_ideal, labels, kind))
        except (bm.BatchMethodError, ValueError) as e:
            row["error"] = str(e)
        row["seconds"] = round(time.time() - t0, 2)
        rows.append(row)
        yield ("method", row)
    _compare_to_none(rows)
    yield ("result", {"rows": rows, **recommend(rows, label_column, kind),
                      "label_column": label_column, "label_kind": kind or None, "k": k,
                      "ideal": round(ideal, 4), "ilisi_ideal": round(ilisi_ideal, 4),
                      "background": BACKGROUND, "background_tip": BACKGROUND_TIP})


def _method_neighbours(method: str, X: np.ndarray, obs: pd.DataFrame, batch_key: str,
                       query: np.ndarray, k: int, n_neighbors: int) -> np.ndarray:
    batches = obs[batch_key].to_numpy()
    if method == bm.BBKNN:
        within = bm.bbknn_within(n_neighbors, np.unique(batches).size)
        _, conn, _ = bm.bbknn_graph(X, batches, neighbors_within_batch=within)
        return bm.graph_neighbours(conn, query, k)
    if method == bm.HARMONY:
        Z = bm.harmony_correct(X, obs, batch_key)
    elif method == bm.SCANORAMA:
        Z = bm.scanorama_correct(X, batches)
    else:
        Z = X
    return _neighbours(np.asarray(Z, dtype=np.float32), query, k)


def _score(idx, codes, query, n_sets, ilisi_ideal, labels, kind) -> dict[str, Any]:
    frac = (codes[idx] != codes[query][:, None]).mean(axis=1)
    ilisi = float(_ilisi(idx, codes, n_sets).mean())
    # Judged on iLISI, not the cross-batch share: a correction that moves one batch onto
    # the other's cells gives every cell neighbours from the other batch only (a high
    # share, no mixing). Scanorama on Kang: share 95% of perfect, iLISI 19%.
    ratio = (ilisi - 1.0) / (ilisi_ideal - 1.0) if ilisi_ideal > 1.0 else None
    out: dict[str, Any] = {
        "mixing": round(float(frac.mean()), 4),
        "ilisi": round(ilisi, 4),
        "mixing_ratio": round(float(np.clip(ratio, 0.0, 1.0)), 4) if ratio is not None else None,
        "conservation": None,
    }
    if labels is not None:
        agree = _label_agreement(idx, labels, codes, query, kind == "within_dataset")
        out["conservation"] = round(float(np.nanmean(agree)), 4) if np.isfinite(agree).any() \
            else None
    return out


def _compare_to_none(rows: list[dict[str, Any]]) -> None:
    """Each row's conservation change from none's, and its overcorrection flag."""
    base = next((r.get("conservation") for r in rows if r["method"] == NONE), None)
    for r in rows:
        c = r.get("conservation")
        r["conservation_change"] = round(c - base, 4) if c is not None and base is not None \
            else None
        r["overcorrection"] = bool(r["conservation_change"] is not None
                                   and r["conservation_change"] <= -OVERCORRECTION_DROP)


def _simplest(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """None first, then the fastest."""
    return min(rows, key=lambda r: (r["method"] != NONE, r.get("seconds") or 0.0))


def _pct(x: float | None) -> str:
    return "n/a" if x is None else f"{x:.0%}"


def recommend(rows: list[dict[str, Any]], label_column: str | None, kind: str) -> dict[str, Any]:
    """The rule-based recommendation: {recommended, reason, warnings, notes}."""
    ok = [r for r in rows if not r.get("error")]
    warnings = [f"{r['label']}: cells' neighbours share their label {_pct(r['conservation'])} "
                f"of the time, down from {_pct(r['conservation'] - r['conservation_change'])} "
                "without correction: it may be merging distinct cell types."
                for r in ok if r.get("overcorrection")]
    notes = [BBKNN_NOTE] if any(r["method"] == bm.BBKNN for r in ok) else []
    if not label_column:
        notes.append("No cell-type-like label column, so conservation (keeping cell types "
                     "apart) was not scored: the choice rests on mixing and speed alone. Pick a "
                     "label column to score it.")
    if not ok:
        return {"recommended": None, "reason": "No method ran on this dataset.",
                "warnings": warnings, "notes": notes}
    mixers = [r for r in ok if (r.get("mixing_ratio") or 0) >= MIX_OK]
    if not mixers:
        best = max(ok, key=lambda r: r.get("mixing_ratio") or 0)
        reason = (f"No method mixed the batches to half of perfect mixing; {best['label']} mixed "
                  f"them most ({_pct(best['mixing_ratio'])} of perfect). The batches may differ "
                  "in cell types, not only by a batch effect: check before correcting.")
        return {"recommended": best["method"], "reason": reason, "warnings": warnings,
                "notes": notes}
    what = ("its own dataset's clusters" if kind == "within_dataset"
            else f"'{label_column}'") if label_column else None
    if what is None or all(r.get("conservation") is None for r in mixers):
        pick = _simplest(mixers)
        if pick["method"] == NONE:
            reason = (f"The batches already mix to {_pct(pick['mixing_ratio'])} of perfect "
                      "without correction, so none is needed.")
        else:
            reason = (f"{pick['label']} is the fastest method that mixes the batches acceptably "
                      f"({_pct(pick['mixing_ratio'])} of perfect mixing, at least "
                      f"{MIX_OK:.0%} counts).")
        return {"recommended": pick["method"], "reason": reason, "warnings": warnings,
                "notes": notes}
    scored = [r for r in mixers if r.get("conservation") is not None]
    top = max(r["conservation"] for r in scored)
    tied = [r for r in scored if r["conservation"] >= top - TIE]
    pick = _simplest(tied)
    others = ", ".join(r["label"] for r in tied if r is not pick)
    reason = (f"{pick['label']}: among the methods that mix the batches acceptably (at least "
              f"{MIX_OK:.0%} of perfect mixing), it keeps {what} together best (neighbours share "
              f"the label {_pct(pick['conservation'])} of the time)")
    if others:
        reason += (f"; {others} came within {TIE:.0%} of that, and "
                   + ("no correction is simplest" if pick["method"] == NONE
                      else f"{pick['label']} was the fastest of them"))
    reason += "."
    if pick["method"] == NONE:
        reason += " The batches already mix without correction."
    return {"recommended": pick["method"], "reason": reason, "warnings": warnings,
            "notes": notes}


def estimate_seconds(n_cells: int, methods: list[str] | None = None) -> float:
    """Rough total seconds for the comparison (measured on Kang, 13,836 cells, 2 CPUs;
    see SECONDS_PER_CELL)."""
    methods = methods or list(METHODS)
    return float(sum(SECONDS_FIXED.get(m, 1.0) + SECONDS_PER_CELL.get(m, 1e-4) * n_cells
                     for m in methods))


# Per method: seconds = fixed + per cell x cells, scoring included (Kang, 2 CPUs).
SECONDS_FIXED = {NONE: 0.5, bm.HARMONY: 1.0, bm.SCANORAMA: 1.0, bm.BBKNN: 2.0}
SECONDS_PER_CELL = {NONE: 1e-4, bm.HARMONY: 5e-4, bm.SCANORAMA: 5e-4, bm.BBKNN: 5e-4}
