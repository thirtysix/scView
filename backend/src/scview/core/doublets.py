"""Doublet calls: the rate Scrublet expects, and a threshold on its score.

The pipeline's doublet step (``sc.pp.scrublet``) scores every cell and calls a
doublet above a threshold it picks from the score distribution. Two parts of
that are made visible and adjustable here:

- The **expected doublet rate** Scrublet is given (it scales the scores) is
  suggested from the number of cells per sample, by 10x Genomics' rule of
  thumb: about 0.8 % of cells are multiplets for every 1,000 cells captured in
  one channel, so 5,000 cells give ~4 %. Only a rule of thumb: chemistry,
  loading and any filtering before upload move it, and a sample column that is
  not the loading channel (Kang 2018 pooled its donors in each lane) makes it a
  guess. Scrublet runs on all cells at once, so it is given the cell-weighted
  mean of the per-sample rates: the share of all cells expected to be doublets.
- A **threshold** on ``doublet_score``, set on the score histogram. Applying it
  rewrites ``predicted_doublet`` (what the filtering step drops with
  ``drop_doublets``) and keeps Scrublet's own call in
  ``predicted_doublet_scrublet``, so the original can be put back.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from scview.core import cluster_qc, provenance, qc_groups

logger = logging.getLogger(__name__)

RATE_PER_1000_CELLS = 0.008
MIN_RATE = 0.004  # 10x's table starts at ~500 cells captured: 0.4 %
MAX_RATE = 0.25
RULE_OF_THUMB = (
    "Rule of thumb from 10x Genomics: about 0.8 % doublets per 1,000 cells captured "
    "in one channel. Chemistry, loading and filtering before upload all move it."
)

SCORE = "doublet_score"
CALL = "predicted_doublet"
SCRUBLET_CALL = "predicted_doublet_scrublet"  # Scrublet's own call, kept on apply
THRESHOLD_KEY = "scview_doublet_threshold"
N_BINS = 100


class DoubletError(ValueError):
    """A request that can't be met (no scores, a bad threshold); nothing changed."""


def rate_for_cells(n_cells: int) -> float:
    """The expected doublet rate of one channel holding ``n_cells`` captured cells."""
    return float(np.clip(RATE_PER_1000_CELLS * n_cells / 1000.0, MIN_RATE, MAX_RATE))


def _sample_column(adata, column: str | None) -> tuple[str | None, str | None]:
    if column:
        if column not in adata.obs.columns:
            raise DoubletError(f"Column '{column}' not found.")
        return column, "requested"
    return qc_groups.sample_grouping(adata)


def _samples(adata, column: str | None) -> tuple[list[str], np.ndarray]:
    if not column:
        return [], np.full(adata.n_obs, -1)
    order, codes = qc_groups.group_codes(adata.obs[column])
    if len(order) > qc_groups.MAX_GROUPS:
        raise DoubletError(
            f"'{column}' has {len(order)} groups; at most {qc_groups.MAX_GROUPS} samples are shown."
        )
    return order, codes


def suggest_rate(adata, column: str | None = None) -> dict:
    """The expected doublet rate for the whole dataset, from its cells per sample
    (the donor / sample column, else a batch column; else one channel)."""
    col, reason = _sample_column(adata, column)
    order, codes = _samples(adata, col)
    sizes = np.bincount(codes[codes >= 0], minlength=len(order)) if order else np.array([])
    samples = [{"name": g, "n_cells": int(n), "rate": round(rate_for_cells(int(n)), 4)}
               for g, n in zip(order, sizes)]
    labelled = int(sizes.sum()) if order else 0
    if labelled:
        rate = float(sum(s["n_cells"] * rate_for_cells(s["n_cells"]) for s in samples) / labelled)
    else:
        col, reason, samples = None, None, []
        rate = rate_for_cells(int(adata.n_obs))
    return {
        "rate": round(rate, 4),
        "column": col,
        "reason": reason,
        "n_cells": int(adata.n_obs),
        "samples": samples,
        "basis": RULE_OF_THUMB if col else (
            RULE_OF_THUMB + " No sample column was found, so all cells are taken as one channel."
        ),
    }


def scrublet_threshold(adata) -> float | None:
    """The threshold Scrublet picked (None when it ran per batch, or found none)."""
    info = adata.uns.get("scrublet")
    if not isinstance(info, dict) or info.get("threshold") is None:
        return None
    t = float(np.asarray(info["threshold"]).ravel()[0])
    return t if np.isfinite(t) else None


def applied_threshold(adata) -> dict | None:
    """The threshold applied over Scrublet's call, if any."""
    entry = adata.uns.get(THRESHOLD_KEY)
    return dict(entry) if isinstance(entry, dict) and "threshold" in entry else None


def _scores(adata) -> np.ndarray:
    if SCORE not in adata.obs.columns:
        raise DoubletError("No doublet scores: run doublet detection first.")
    return pd.to_numeric(adata.obs[SCORE], errors="coerce").to_numpy(dtype=np.float64)


def _calls_at(scores: np.ndarray, threshold: float) -> np.ndarray:
    # Strictly above, as Scrublet calls them: its own threshold gives its own calls.
    with np.errstate(invalid="ignore"):
        return np.isfinite(scores) & (scores > threshold)


def _check_threshold(threshold: float) -> float:
    t = float(threshold)
    if not np.isfinite(t) or t < 0:
        raise DoubletError("The threshold must be a number of 0 or more.")
    return t


def _recorded_rate(adata) -> float | None:
    """The expected rate the latest doublet step ran with, from provenance."""
    for entry in reversed(provenance.read_provenance(adata).get("history", [])):
        if entry.get("step") == "doublet_detection":
            rate = (entry.get("params") or {}).get("expected_doublet_rate")
            return float(rate) if isinstance(rate, (int, float)) else None
    return None


def summary(adata, threshold: float | None = None, column: str | None = None,
            n_bins: int = N_BINS) -> dict:
    """The doublet-score histogram, and how many cells a threshold calls doublets,
    overall and per sample, beside the rate expected from their size.

    ``threshold`` defaults to the applied one, else Scrublet's. Without scores the
    answer still carries the suggested expected rate (for the doublet step)."""
    suggestion = suggest_rate(adata, column)
    n = int(adata.n_obs)
    if SCORE not in adata.obs.columns:
        return {"available": False, "n_cells": n, "suggestion": suggestion}
    scores = _scores(adata)
    finite = scores[np.isfinite(scores)]
    st = scrublet_threshold(adata)
    applied = applied_threshold(adata)
    if threshold is None:
        threshold = applied["threshold"] if applied else st
    if threshold is not None:
        threshold = _check_threshold(threshold)

    top = max(1.0, float(finite.max())) if finite.size else 1.0
    counts, edges = np.histogram(finite, bins=n_bins, range=(0.0, top))
    now = cluster_qc.doublet_calls(adata.obs)
    out = {
        "available": True,
        "n_cells": n,
        "n_scored": int(finite.size),
        "hist": {"bin_edges": np.round(edges, 4).tolist(), "counts": counts.tolist()},
        "median": round(float(np.median(finite)), 4) if finite.size else None,
        "scrublet_threshold": st,  # exact: the calls at it are Scrublet's own
        "applied": applied,
        "scrublet_call_kept": SCRUBLET_CALL in adata.obs.columns,
        "n_called_now": None if now is None else int(now.sum()),
        "expected_rate_used": _recorded_rate(adata),
        "threshold": threshold,
        "n_called": None,
        "fraction_called": None,
        "suggestion": suggestion,
        "samples": [],
    }
    if threshold is None:
        return out
    called = _calls_at(scores, threshold)
    out["n_called"] = int(called.sum())
    out["fraction_called"] = round(float(called.mean()), 4) if n else 0.0
    order, codes = _samples(adata, suggestion["column"])
    if order:
        sizes = np.bincount(codes[codes >= 0], minlength=len(order))
        hits = np.bincount(codes[(codes >= 0) & called], minlength=len(order))
        out["samples"] = [
            {"name": g, "n_cells": int(s), "n_called": int(h),
             "fraction": round(float(h / s), 4) if s else 0.0,
             "expected_rate": round(rate_for_cells(int(s)), 4)}
            for g, s, h in zip(order, sizes, hits)
        ]
    return out


def clear_threshold(adata) -> None:
    """Forget an applied threshold (a new Scrublet run made new scores and calls)."""
    adata.uns.pop(THRESHOLD_KEY, None)
    if SCRUBLET_CALL in adata.obs.columns:
        del adata.obs[SCRUBLET_CALL]


def apply_threshold(adata, threshold: float | None, column: str | None = None) -> dict:
    """Call doublets at ``threshold`` into ``predicted_doublet``, keeping Scrublet's
    call in ``predicted_doublet_scrublet``; ``None`` puts Scrublet's call back.
    Raises DoubletError before changing anything. Returns the new ``summary``."""
    from scview.core.obs_writes import forget_column_results

    scores = _scores(adata)
    if threshold is not None:
        threshold = _check_threshold(threshold)
    _samples(adata, _sample_column(adata, column)[0])  # a bad column fails here, unchanged
    obs = adata.obs
    previous = applied_threshold(adata)
    if threshold is None:
        if previous is None:
            return summary(adata, None, column)
        if SCRUBLET_CALL in obs.columns:
            obs[CALL] = obs[SCRUBLET_CALL].to_numpy(dtype=bool)
            del obs[SCRUBLET_CALL]
        elif previous.get("created_call"):
            del obs[CALL]  # there was no call before the first threshold
        adata.uns.pop(THRESHOLD_KEY, None)
        params: dict = {"threshold": "scrublet"}
    else:
        t = threshold
        created = previous.get("created_call", False) if previous else CALL not in obs.columns
        if SCRUBLET_CALL not in obs.columns and CALL in obs.columns and previous is None:
            obs[SCRUBLET_CALL] = cluster_qc.doublet_calls(obs)
        called = _calls_at(scores, t)
        obs[CALL] = called
        # A new dict, not an edit of the old one: write_derived saves replaced entries.
        adata.uns[THRESHOLD_KEY] = {"threshold": t, "n_called": int(called.sum()),
                                    "created_call": bool(created)}
        params = {"threshold": t, "n_called": int(called.sum()),
                  "scrublet_threshold": scrublet_threshold(adata)}
    forget_column_results(adata, CALL)  # markers / enrichment of the old calls
    try:
        provenance.record_step(adata, step="doublet_threshold",
                               tool="scview doublet-score threshold", params=params)
    except Exception as e:  # provenance must never block the result
        logger.warning("Could not record the doublet threshold in provenance: %s", e)
    return summary(adata, None, column)
