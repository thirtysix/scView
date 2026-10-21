"""Two QC states per cell type: is the "high-mito" part a tail or a second state?

A fixed mito cutoff can remove most of one cell type when that type holds a second
state: cells with a low ribosomal-protein (RP) share, high % mito and more ambient
RNA, forming their own UMAP region (the LNCaP/CAF review, 2026-10-21). Such a state
is better offered as a label than cut by a threshold.

Per cell: x = RP share of the NON-mitochondrial counts (RPL*/RPS* genes, as
``per_cell_qc(ribo=True)`` counts them), y = % mito. Each group of a label column
with at least ``MIN_CELLS`` cells gets a Gaussian mixture with 1 and with 2
components (full covariance, ``N_INIT`` starts, fixed seed) on the empirical logit
of both shares (``log((k + 0.5) / (n - k + 0.5))`` from the counts, so a cell
with no mito reads does not pile up at a floor). Two states are called only when

- the BIC improves by more than ``MIN_DELTA_BIC`` with the second component,
- the two components are separated: Ashman's D (distance between the means in
  units of their average spread) of at least ``MIN_SEPARATION``, and
- the smaller component holds at least ``MIN_WEIGHT`` of the group's cells.

The BIC alone over-calls: with thousands of cells any skew or heavy tail is
"improved" by a second Gaussian (on Kang PBMC, 6 of 12 cell types pass the BIC
rule with components overlapping at D = 1.2-1.4). The separation rule keeps a
broad or skewed distribution as one state.

Without MT- genes (Kang) % mito cannot be measured: the fit is then 1-D on the RP
share alone and says so, never on an all-zero mito column.

Per cell assignments are not returned (cells can be 100k+): the response has the
components per group, the flagged state's share overall and per condition (or
sample), and a seeded scatter sample of at most ``N_SCATTER`` cells per group.
``label`` writes the assignment as a categorical obs column.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import time
import weakref
from typing import Any

import numpy as np
import pandas as pd

from scview.core.confounding import condition_columns
from scview.core.obs_writes import forget_column_results
from scview.core.qc_groups import MAX_GROUPS, group_codes, members, sample_grouping

logger = logging.getLogger(__name__)

MIN_CELLS = 100  # smaller groups are not fitted
MIN_DELTA_BIC = 10.0  # BIC(1) - BIC(2) must exceed this ("very strong" on Kass-Raftery)
MIN_SEPARATION = 2.0  # Ashman's D: two Gaussians are cleanly apart from 2 up
MIN_WEIGHT = 0.05  # a smaller "state" is outliers
N_INIT = 3
RANDOM_STATE = 0
MAX_FIT_CELLS = 20_000  # a bigger group is fitted on a seeded sample, then every cell assigned
N_SCATTER = 2_000  # per group
MIN_LEVEL_CELLS = 20  # a condition level's share needs this many of the group's cells
FACT_MAX_CELLS = 50_000  # the co-pilot fits on demand up to this size, else uses the cache

MAIN = "main"
LOW_RP_HIGH_MT = "low-RP high-mito"
LOW_RP = "low-RP"  # the 1-D fit: no mito to say "high-mito"
OTHER = "second state"  # two states, but not the low-RP / high-mito pattern
DEFAULT_NAME = "qc_state"
REGISTRY_KEY = "scview_qc_state_columns"  # columns this module wrote (JSON list)
_NAME = re.compile(r"[A-Za-z][A-Za-z0-9_.\-]{0,59}")

TAG = "result:qcstates"


class QcStatesError(ValueError):
    """A request this cannot serve; ``status`` is the HTTP code."""

    def __init__(self, message: str, status: int = 400) -> None:
        super().__init__(message)
        self.status = status


# ---------------------------------------------------------------------------
# Per-cell features
# ---------------------------------------------------------------------------


def features(cells: dict) -> dict[str, Any]:
    """RP share of non-MT counts (%), % mito (None when unmeasurable) and the
    logit matrix the mixture is fitted on, from ``per_cell_qc(ribo=True)``.

    ``ok`` marks the cells that can be fitted (some counts, finite shares)."""
    pct_ribo = cells.get("pct_ribo")
    if pct_ribo is None:
        raise QcStatesError(
            "No ribosomal-protein share for this dataset: no RPL*/RPS* genes were found in a "
            "counts matrix (or the dataset is read from disk, where it is not computed).")
    tc = np.asarray(cells["total_counts"], dtype=np.float64)
    ribo = np.asarray(pct_ribo, dtype=np.float64) / 100.0 * tc
    has_mt = bool(cells.get("has_mt", True)) and cells.get("pct_mt") is not None
    mt = np.asarray(cells["pct_mt"], dtype=np.float64) / 100.0 * tc if has_mt else np.zeros_like(tc)
    non_mt = tc - mt
    with np.errstate(divide="ignore", invalid="ignore"):
        rp_pct = np.where(non_mt > 0, ribo / non_mt * 100.0, np.nan)
        cols = [np.log((ribo + 0.5) / (non_mt - ribo + 0.5))]
        if has_mt:
            cols.append(np.log((mt + 0.5) / (tc - mt + 0.5)))
    X = np.column_stack(cols)
    ok = (tc > 0) & (non_mt > 0) & np.isfinite(X).all(axis=1) & np.isfinite(rp_pct)
    return {
        "rp_pct": rp_pct,
        "mt_pct": np.asarray(cells["pct_mt"], dtype=np.float64) if has_mt else None,
        "X": X,
        "ok": ok,
        "has_mt": has_mt,
    }


# ---------------------------------------------------------------------------
# The fit
# ---------------------------------------------------------------------------


def _ashman_d(means: np.ndarray, covs: np.ndarray) -> float:
    """Distance between two components' means in units of their average spread
    (Ashman's D in 1-D; the Mahalanobis form in 2-D)."""
    d = means[0] - means[1]
    S = (covs[0] + covs[1]) / 2.0
    try:
        return float(np.sqrt(max(d @ np.linalg.solve(S, d), 0.0)))
    except np.linalg.LinAlgError:
        return 0.0


def _fit(X: np.ndarray, seed: int) -> dict[str, Any]:
    from sklearn.mixture import GaussianMixture

    rng = np.random.default_rng(seed)
    fit_on = X if X.shape[0] <= MAX_FIT_CELLS else X[np.sort(rng.choice(X.shape[0], MAX_FIT_CELLS, replace=False))]
    kw = dict(covariance_type="full", n_init=N_INIT, random_state=RANDOM_STATE, reg_covar=1e-4)
    g1 = GaussianMixture(1, **kw).fit(fit_on)
    g2 = GaussianMixture(2, **kw).fit(fit_on)
    return {
        "g2": g2,
        "delta_bic": float(g1.bic(fit_on) - g2.bic(fit_on)),
        "separation": _ashman_d(g2.means_, g2.covariances_),
        "n_fit": int(fit_on.shape[0]),
    }


def _flagged_component(means: np.ndarray, weights: np.ndarray, has_mt: bool) -> tuple[int, str]:
    """(index, label) of the component to flag: the low-RP one, called low-RP
    high-mito when it also has more mito; when the components split the other way
    (lower RP with LESS mito), the smaller one as a "second state"."""
    low = int(np.argmin(means[:, 0]))
    if not has_mt:
        return low, LOW_RP
    if means[low, 1] > means[1 - low, 1]:
        return low, LOW_RP_HIGH_MT
    return int(np.argmin(weights)), OTHER


def _reason(g: dict, has_mt: bool) -> str:
    if g["n_states"] == 2:
        what = {LOW_RP_HIGH_MT: "the low-RP / high-mito state", LOW_RP: "the low-RP state",
                OTHER: "the smaller state (lower RP without more mito)"}[g["flagged_label"]]
        return (f"Two states: a second component improves the BIC by {g['delta_bic']:,.0f} (over "
                f"{MIN_DELTA_BIC:.0f}) and the two are apart (D = {g['separation']:.1f}, at least "
                f"{MIN_SEPARATION:.0f}); {what} holds {g['flagged_share']:.0%} of its cells.")
    dbic, sep = g["delta_bic"], g["separation"]
    if dbic <= MIN_DELTA_BIC:
        return (f"One state: a second component does not improve the BIC by more than "
                f"{MIN_DELTA_BIC:.0f} ({dbic:+,.0f}).")
    if sep < MIN_SEPARATION:
        return (f"One broad state: the BIC favours two components ({dbic:+,.0f}) but they overlap "
                f"(D = {sep:.1f}, under {MIN_SEPARATION:.0f}), the shape of a skewed or tailed "
                "distribution rather than a second state.")
    return (f"One state: the second component holds only {g['minor_weight']:.0%} of the cells "
            f"(under {MIN_WEIGHT:.0%}), outliers rather than a state.")


def _round(v: float | None, digits: int = 2) -> float | None:
    return None if v is None or not np.isfinite(v) else round(float(v), digits)


def _condition(adata, column: str) -> tuple[str | None, str | None]:
    """The column to split each group's share by: a condition, else the samples."""
    for c in condition_columns(adata.obs):
        if c != column:
            return c, "condition"
    col, _why = sample_grouping(adata)
    if col and col != column:
        return col, "sample"
    return None, None


def default_column(adata) -> tuple[str | None, str | None]:
    """(column, why): an author cell-type column, else the active clustering."""
    from scview.core import cluster_qc
    from scview.core.sample_pseudobulk import default_celltype_column

    col = default_celltype_column(adata.obs)
    if col:
        return col, "celltype"
    col = cluster_qc.default_grouping(adata)
    if col:
        return col, "clustering"
    return None, None


def compute(adata, column: str, cells: dict, reason: str = "requested") -> tuple[dict, np.ndarray]:
    """(response, per-cell state codes) for ``column``. Codes: 0 = main, 1 = the
    flagged state of a two-state group, -1 = not fitted (small group, no label,
    no counts); the response says which label code 1 has in each group.

    Raises KeyError for a missing column, QcStatesError for one that cannot be used."""
    t0 = time.perf_counter()
    if column not in adata.obs.columns:
        raise KeyError(f"Column '{column}' not found in obs.")
    names, codes = group_codes(adata.obs[column])
    if not names:
        raise QcStatesError(f"'{column}' has no labelled cells.")
    if len(names) > MAX_GROUPS:
        raise QcStatesError(f"'{column}' has {len(names)} groups; the QC state check takes at most "
                            f"{MAX_GROUPS}. Pick a cell-type or cluster column.")
    f = features(cells)
    has_mt = f["has_mt"]
    cond_col, cond_kind = _condition(adata, column)
    if cond_col:
        levels, level_codes = group_codes(adata.obs[cond_col])
    else:
        levels, level_codes = [], np.full(adata.n_obs, -1)

    states = np.full(adata.n_obs, -1, dtype=np.int8)
    rng = np.random.default_rng(RANDOM_STATE)
    groups_out: list[dict] = []
    sx: list[np.ndarray] = []
    sy: list[np.ndarray] = []
    sg: list[np.ndarray] = []
    ss: list[np.ndarray] = []
    for gi, (name, idx) in enumerate(zip(names, members(codes, len(names)))):
        fit_idx = idx[f["ok"][idx]]
        g: dict[str, Any] = {"name": name, "n_cells": int(idx.size), "n_fitted": int(fit_idx.size),
                             "fitted": False, "n_states": 1, "delta_bic": None, "separation": None,
                             "flagged": None, "flagged_label": None, "flagged_share": None,
                             "minor_weight": None, "components": [], "per_condition": {}}
        if fit_idx.size < MIN_CELLS:
            g["reason"] = (f"Not fitted: {fit_idx.size:,} cells with counts, under {MIN_CELLS}."
                           if fit_idx.size < idx.size else f"Not fitted: {idx.size:,} cells, under {MIN_CELLS}.")
        else:
            fit = _fit(f["X"][fit_idx], RANDOM_STATE + gi)
            g2 = fit["g2"]
            assign = g2.predict(f["X"][fit_idx])
            weights = g2.weights_
            g.update(fitted=True, delta_bic=round(fit["delta_bic"], 1), separation=round(fit["separation"], 2),
                     minor_weight=round(float(weights.min()), 3), n_fit=fit["n_fit"])
            two = (fit["delta_bic"] > MIN_DELTA_BIC and fit["separation"] >= MIN_SEPARATION
                   and float(weights.min()) >= MIN_WEIGHT and len(np.unique(assign)) == 2)
            if two:
                flag, label = _flagged_component(g2.means_, weights, has_mt)
                is_flag = assign == flag
                states[fit_idx] = is_flag.astype(np.int8)
                g.update(n_states=2, flagged=1, flagged_label=label,
                         flagged_share=round(float(is_flag.mean()), 4))
                parts = [(MAIN, ~is_flag, float(weights[1 - flag])), (label, is_flag, float(weights[flag]))]
            else:
                states[fit_idx] = 0
                parts = [(MAIN, np.ones(fit_idx.size, dtype=bool), 1.0)]
            for label, mask, weight in parts:
                cell = fit_idx[mask]
                g["components"].append({
                    "label": label,
                    "n_cells": int(cell.size),
                    "weight": round(weight, 4),
                    "mean_rp_share": _round(float(np.mean(f["rp_pct"][cell])) if cell.size else None),
                    "mean_pct_mt": (_round(float(np.mean(f["mt_pct"][cell]))) if has_mt and cell.size else None),
                })
            if two and cond_col:
                lv = level_codes[fit_idx]
                for li, level in enumerate(levels):
                    in_level = lv == li
                    n = int(in_level.sum())
                    if n:
                        g["per_condition"][level] = {"n": n, "share": round(float(is_flag[in_level].mean()), 4)}
            elif cond_col:
                lv = level_codes[fit_idx]
                for li, level in enumerate(levels):
                    n = int((lv == li).sum())
                    if n:
                        g["per_condition"][level] = {"n": n, "share": None}
            g["reason"] = _reason(g, has_mt)
        groups_out.append(g)
        # The scatter: a seeded sample of the group's fittable cells.
        pool = fit_idx if fit_idx.size else idx[:0]
        pick = pool if pool.size <= N_SCATTER else np.sort(rng.choice(pool, N_SCATTER, replace=False))
        sx.append(f["rp_pct"][pick])
        if has_mt:
            sy.append(f["mt_pct"][pick])
        sg.append(np.full(pick.size, gi))
        ss.append(np.maximum(states[pick], 0))

    fitted = states >= 0
    n_flag = int((states == 1).sum())
    two_state = [g["name"] for g in groups_out if g["n_states"] == 2]
    cat = lambda parts: np.concatenate(parts) if parts else np.array([])  # noqa: E731
    note = ("No mitochondrial (MT-) genes in this dataset, so % mito can't be measured: each group is "
            "fitted on its ribosomal-protein share alone (1-D), and a low-RP state is not called "
            "high-mito." if not has_mt else "")
    result = {
        "column": column,
        "reason": reason,
        "n_cells": int(adata.n_obs),
        "n_unlabelled": int((codes < 0).sum()),
        "has_mt": has_mt,
        "fit": "rp_mito" if has_mt else "rp_only",
        "note": note,
        "rules": {"min_cells": MIN_CELLS, "min_delta_bic": MIN_DELTA_BIC,
                  "min_separation": MIN_SEPARATION, "min_weight": MIN_WEIGHT,
                  "n_init": N_INIT, "random_state": RANDOM_STATE, "max_fit_cells": MAX_FIT_CELLS},
        "condition": ({"column": cond_col, "kind": cond_kind, "levels": levels} if cond_col else None),
        "groups": groups_out,
        "two_state_groups": two_state,
        "overall": {"n_fitted": int(fitted.sum()), "n_flagged": n_flag,
                    "share": round(n_flag / int(fitted.sum()), 4) if fitted.any() else None},
        "scatter": {
            "x": np.round(cat(sx), 2).tolist(),
            "y": np.round(cat(sy), 2).tolist() if has_mt else None,
            "group": cat(sg).astype(int).tolist(),
            "state": cat(ss).astype(int).tolist(),
            "x_label": "RP share of non-mito counts (%)",
            "y_label": "% mito" if has_mt else None,
            "n_shown": int(sum(p.size for p in sg)),
            "max_per_group": N_SCATTER,
        },
        "seconds": round(time.perf_counter() - t0, 3),
    }
    return result, states


# ---------------------------------------------------------------------------
# Cached on the loaded dataset
# ---------------------------------------------------------------------------

_cache: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()


def _fingerprint(obs: pd.DataFrame, columns: list[str]) -> str:
    """Identity of the columns' values: a relabel changes it, a reload does not."""
    h = hashlib.blake2b(digest_size=16)
    for c in columns:
        names, codes = group_codes(obs[c])
        h.update(c.encode())
        h.update(json.dumps(names).encode())
        h.update(np.ascontiguousarray(codes).tobytes())
    return h.hexdigest()


def _cells(adaptor) -> dict:
    fn = getattr(adaptor, "qc_per_cell", None)
    if callable(fn):
        return fn(ribo=True)
    from scview.core.anndata_adaptor import per_cell_qc

    return per_cell_qc(adaptor.adata, ribo=True)


def for_adaptor(adaptor, column: str | None = None) -> tuple[dict, np.ndarray | None]:
    """``compute`` on a loaded dataset, cached per column (and the condition it
    splits by) until either column's values change. With no ``column``, the
    cell-type column, else the clustering (``reason`` says which)."""
    adata = adaptor.adata
    reason = "requested"
    if not column:
        column, reason = default_column(adata)
        if column is None:
            return {"column": None, "reason": None, "n_cells": int(adata.n_obs), "groups": [],
                    "note": "No cell-type or clustering column found: pick a column.",
                    "two_state_groups": []}, None
    if column not in adata.obs.columns:
        raise KeyError(f"Column '{column}' not found in obs.")
    cond, _ = _condition(adata, column)
    key = (column, _fingerprint(adata.obs, [column] + ([cond] if cond else [])))
    try:
        store = _cache.setdefault(adaptor, {})
    except TypeError:  # an object that takes no weak reference: no cache
        store = {}
    hit = store.get(key)
    if hit is not None and hit[2] == id(adata):
        return {**hit[0], "reason": reason}, hit[1]
    result, states = compute(adata, column, _cells(adaptor), reason)
    store.clear()  # one column at a time is enough; the states array is n_cells bytes
    store[key] = (result, states, id(adata))  # not adata itself: it would outlive a reload
    return result, states


def cached(adaptor, column: str | None = None) -> dict | None:
    """The cached result for ``column`` (any, when None), without computing."""
    try:
        store = _cache.get(adaptor) or {}
    except TypeError:
        return None
    for (col, _fp), (result, _s, adata_id) in store.items():
        if adata_id == id(adaptor.adata) and (column is None or col == column):
            return result
    return None


# ---------------------------------------------------------------------------
# The label column
# ---------------------------------------------------------------------------


def written_columns(adata) -> list[str]:
    raw = adata.uns.get(REGISTRY_KEY)
    try:
        names = json.loads(raw) if isinstance(raw, str) else []
    except ValueError:
        names = []
    return [n for n in names if isinstance(n, str)] if isinstance(names, list) else []


def check_name(adata, column: str, name: str | None) -> str:
    """The label column's name, or QcStatesError (409 for a column this module did not write)."""
    name = (name or DEFAULT_NAME).strip()
    if not _NAME.fullmatch(name) or name.startswith("scview_"):
        raise QcStatesError("Name the column with a letter first, then letters, digits, '_', '.' or '-' "
                            "(up to 60 characters, not starting with 'scview_').")
    if name == column:
        raise QcStatesError(f"'{name}' is the grouping column itself; pick another name.")
    if name in adata.obs.columns and name not in written_columns(adata):
        raise QcStatesError(f"A column '{name}' already exists and was not written by the QC state "
                            "check; pick another name.", status=409)
    return name


def label(adata, result: dict, states: np.ndarray, name: str) -> dict:
    """Write each cell's state as categorical obs column ``name``: the flagged
    state's label in a two-state group, ``main`` for every other cell (one-state
    and unfitted groups, cells without a group). Records a provenance step."""
    from scview.core import provenance

    column = result["column"]
    names, codes = group_codes(adata.obs[column])
    flag_label = np.array([MAIN] * (len(names) + 1), dtype=object)  # last slot: code -1
    by_name = {g["name"]: g for g in result["groups"]}
    for i, n in enumerate(names):
        g = by_name.get(n)
        if g and g["n_states"] == 2:
            flag_label[i] = g["flagged_label"]
    values = np.where(states == 1, flag_label[codes], MAIN)
    used = [MAIN] + [c for c in (LOW_RP_HIGH_MT, LOW_RP, OTHER) if (values == c).any()]
    adata.obs[name] = pd.Categorical(values, categories=used)
    forget_column_results(adata, name)  # markers of an earlier state column
    reg = written_columns(adata)
    if name not in reg:
        reg.append(name)
    adata.uns[REGISTRY_KEY] = json.dumps(reg)  # a new string: write_derived rewrites it
    counts = {c: int((values == c).sum()) for c in used}
    try:
        provenance.record_step(
            adata, step="qc_state_label", tool="scview QC states",
            params={"column": column, "output": name, "fit": result["fit"], "counts": counts,
                    "two_state_groups": result["two_state_groups"], **result["rules"]},
        )
    except Exception as e:  # provenance must never block the result
        logger.warning("Could not record QC state provenance: %s", e)
    return {"name": name, "column": column, "counts": counts,
            "two_state_groups": result["two_state_groups"], "fit": result["fit"]}


# ---------------------------------------------------------------------------
# Co-pilot fact
# ---------------------------------------------------------------------------


def _condition_text(g: dict, cond: dict | None) -> str:
    if not cond:
        return ""
    lv = [(k, v["share"]) for k, v in g["per_condition"].items()
          if v.get("share") is not None and v["n"] >= MIN_LEVEL_CELLS]
    if len(lv) < 2:
        return ""
    lo = min(lv, key=lambda kv: kv[1])
    hi = max(lv, key=lambda kv: kv[1])
    if len(lv) == 2:
        return f", {lo[1]:.0%} in {lo[0]} vs {hi[1]:.0%} in {hi[0]} ({cond['column']})"
    return f", from {lo[1]:.0%} ({lo[0]}) to {hi[1]:.0%} ({hi[0]}) across {cond['column']}"


def grounding_lines(result: dict) -> tuple[list[str], str]:
    """Prompt lines + a one-line detail for the co-pilot's QC facts."""
    if not result.get("column") or not result.get("groups"):
        return [], ""
    col = result["column"]
    fit = ("RP share of non-mito counts vs % mito" if result["has_mt"]
           else "RP share alone: no MT- genes, so % mito is unmeasurable")
    lines = [f"\n## QC states per {col} ({fit}; Gaussian mixture, 1 vs 2 components; two states need "
             f"BIC +{MIN_DELTA_BIC:.0f} and separation D >= {MIN_SEPARATION:.0f}) [{TAG}]"]
    fitted = [g for g in result["groups"] if g["fitted"]]
    two = [g for g in fitted if g["n_states"] == 2]
    word = {LOW_RP_HIGH_MT: "low-RP/high-mito", LOW_RP: "low-RP", OTHER: "in a second state"}
    for g in two:
        lines.append(f"- **{g['name']}**: two QC states (BIC +{g['delta_bic']:,.0f}, D {g['separation']:.1f}): "
                     f"{g['flagged_share']:.0%} of {g['n_fitted']:,} cells {word[g['flagged_label']]}"
                     f"{_condition_text(g, result.get('condition'))} [{TAG}]")
    broad = [g["name"] for g in fitted if g["n_states"] == 1 and (g["delta_bic"] or 0) > MIN_DELTA_BIC
             and (g["separation"] or 0) < MIN_SEPARATION]
    if not two:
        lines.append(f"- No {col} group shows a second QC state: one state in each of the "
                     f"{len(fitted)} fitted groups [{TAG}]")
    if broad:
        shown = ", ".join(broad[:6]) + (f" and {len(broad) - 6} more" if len(broad) > 6 else "")
        lines.append(f"- One broad state (the BIC favours 2 components but they overlap, D < "
                     f"{MIN_SEPARATION:.0f}; not a second state): {shown} [{TAG}]")
    skipped = len(result["groups"]) - len(fitted)
    if skipped:
        lines.append(f"- {skipped} {'group' if skipped == 1 else 'groups'} with under {MIN_CELLS} cells "
                     "not fitted.")
    detail = (f"{col}: two QC states in {len(two)} of {len(fitted)} fitted groups"
              + (f" ({', '.join(g['name'] for g in two[:4])})" if two else ""))
    return lines, detail


def grounding_for(adaptor) -> tuple[list[str], str]:
    """The co-pilot's QC-state lines: the cached fit, else a fresh one on the
    default column for a dataset of up to FACT_MAX_CELLS cells."""
    result = cached(adaptor)
    if result is None:
        if int(adaptor.adata.n_obs) > FACT_MAX_CELLS:
            return [], ""
        result, _ = for_adaptor(adaptor)
    return grounding_lines(result)
