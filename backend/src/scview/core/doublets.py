"""Doublet calls: Scrublet per capture channel, the rate it expects, and a threshold.

Doublets form inside one capture channel (a 10x GEM well or lane): two cells in one
droplet. The doublet step runs Scrublet on each channel's cells alone, one channel after
another, as ``sc.pp.scrublet(batch_key=...)`` does, each with its own expected rate and
its own automatic threshold (``run_scrublet``; ``scrublet_lean.scrublet_by_channel``).
Run on all cells of a 133-well mouse brain subset at once it was given one rate for
100,000 cells (clipped to 25 %) and flagged 8.7 % of them; per well, 0.7 %.

The channels (``channels``; ``channel_grouping`` gives the column and why), in this order:

1. the doublet step's ``doublet_channel_key``: an obs column, or ``ALL_CELLS`` for all
   cells as one channel;
2. an obs column whose name says it is the capture unit (``CHANNEL_NAMES``, in any case,
   with spaces, dots or hyphens read as "_"): the first of them in that order;
3. the cell barcodes (``barcode_channels``): a 10x barcode (16 or 14 of ACGT) with a
   ``-N`` / ``.N`` suffix (Cell Ranger aggr's GEM well, 1 to 133 in the 1.3M brain) and/or
   a ``PREFIX_`` (samples merged, Seurat style), when they split the cells into 2 or more
   groups. Kang 2018's ``CTRL_AAACATACATTTCC.1`` gives its two lanes, CTRL and STIM. The
   doublet step writes these labels to obs ``CHANNEL_COLUMN``, for the user to see and
   reuse, and step 2 finds that column on the next run;
4. the samples (``qc_groups.sample_grouping``: the donor column, else a batch column): a
   guess, since donors pooled in one lane are not channels (Kang's Donor column holds 8
   donors demultiplexed from each of its 2 lanes), said in ``GUESS_NOTE``;
5. all cells as one channel.

2 to 4 are used only when their median channel holds at least ``MIN_CHANNEL_CELLS`` cells
(a column named "well" in plate data has one cell per well); a candidate passed over is
listed in ``skipped``. A channel under that size is not scored: its cells keep a NaN
score and no call, and the views say how many cells that is.

- The **expected doublet rate** of a channel is 10x Genomics' rule of thumb for its
  cells: about 0.8 % multiplets for every 1,000 cells captured in one channel, so 5,000
  cells give ~4 %. Chemistry, loading and filtering before upload all move it. A rate set
  in the doublet step applies to every channel.
- A **threshold** on ``doublet_score``, set on the score histogram, is one threshold over
  all scores. Applying it rewrites ``predicted_doublet`` (what the filtering step drops with
  ``drop_doublets``, unless cell hashing is the doublet source) and keeps Scrublet's own
  per-channel call in ``predicted_doublet_scrublet``, so it can be put back.
- A channel whose **automatic threshold calls almost nothing** (under
  ``FEW_CALLED_SHARE`` of a channel of ``FEW_CALLED_MIN_CELLS`` or more; Kang's STIM lane,
  9 of 7,263) is named in ``summary``'s ``few_called``, with its scores' histogram, for
  the doublet card to say its threshold may have missed them.

``uns["scrublet"]`` keeps each channel's record in ``scview_channels`` (cells, cells
scored, rate, Scrublet's threshold, cells called, neighbours, status) with the column in
``scview_channel_column``. With one channel of every cell it is Scrublet's own ``uns``
(threshold, simulated scores) beside that, as ``sc.pp.scrublet`` writes it.

A second detector, an scDblFinder-style classifier (core/doublet_classifier.py), runs in
the same channels with the same rates when the doublet step's ``doublet_method`` is
``scdblfinder-style`` or ``both`` (``run_doublets``; ``run_classifier``). It writes
``doublet_score_classifier`` and ``predicted_doublet_classifier`` and its per-channel
record to ``uns[CLASSIFIER_KEY]``; the step drops the results of a detector it did not run.
Its threshold is its own (fit to the artificial doublets and the expected rate); the
threshold set on the histogram is Scrublet's. ``summary`` gives its calls per channel
(``classifier``) and how they agree with Scrublet's (``methods``).

With cell hashing run as well, hashing is the doublet views' primary call and
Scrublet the second opinion (``core/doublet_source.py``); ``summary`` then says how
the two agree at the threshold shown, overall and per hashed sample.
"""

from __future__ import annotations

import logging
import re
import threading
import weakref
from dataclasses import dataclass, replace

import numpy as np
import pandas as pd

from scview.core import doublet_classifier, doublet_source, provenance, qc_groups, scrublet_lean

logger = logging.getLogger(__name__)

RATE_PER_1000_CELLS = 0.008
MIN_RATE = 0.004  # 10x's table starts at ~500 cells captured: 0.4 %
MAX_RATE = 0.25
# Up to this many cells in a channel Scrublet finds neighbours exactly (sklearn) instead
# of with pynndescent: the same scores (Kang: Pearson 1.0, same top 1-5% of cells) in 18 s
# instead of 56 s, and 77 s instead of 125 s on 37k cells, with no numba compile on a fresh
# server. Exact search grows faster with the cell count, so a large channel keeps the
# approximate one.
EXACT_KNN_MAX_CELLS = 50_000
RULE_OF_THUMB = (
    "Rule of thumb from 10x Genomics: about 0.8 % doublets per 1,000 cells captured "
    "in one channel. Chemistry, loading and filtering before upload all move it."
)

# A channel whose automatic threshold called almost nothing (``summary``'s ``few_called``):
# Scrublet's own threshold calls under this share of its cells, in a channel of at least
# FEW_CALLED_MIN_CELLS. On Kang the STIM lane's threshold landed at 0.54 (CTRL's 0.22) and
# called 9 of 7,263 cells (0.12 %; CTRL 179 of 6,573, 2.7 %), where the rule of thumb
# expects ~5.8 %: 1 in 500 flags it, the ~0.1 % first proposed would not. At 2,000 cells the
# rule of thumb expects 1.6 % doublets (32), so calling under 0.2 % (4) there is unlikely by
# chance; in a smaller channel (the 100k brain's GEM wells hold 512-1,034 cells, 0.4-0.8 %
# expected) a few calls or none is ordinary, and 0.2 % is a cell or two.
FEW_CALLED_SHARE = 0.002
FEW_CALLED_MIN_CELLS = 2_000

SCORE = "doublet_score"
CALL = "predicted_doublet"
SCRUBLET_CALL = "predicted_doublet_scrublet"  # Scrublet's own call, kept on apply
THRESHOLD_KEY = "scview_doublet_threshold"
N_BINS = 100

# --- the detectors ------------------------------------------------------------------

SCRUBLET_METHOD, CLASSIFIER_METHOD, BOTH_METHOD = "scrublet", "scdblfinder-style", "both"
METHODS = (SCRUBLET_METHOD, CLASSIFIER_METHOD, BOTH_METHOD)
METHOD_ALIASES = {"classifier": CLASSIFIER_METHOD, "scdblfinder": CLASSIFIER_METHOD,
                  "scdblfinder_style": CLASSIFIER_METHOD}
CLASSIFIER_SCORE = doublet_source.CLASSIFIER_SCORE
CLASSIFIER_CALL = doublet_source.CLASSIFIER_CALL
CLASSIFIER_KEY = "scview_doublet_classifier"  # uns: its per-channel record

# --- capture channels ---------------------------------------------------------------

CHANNEL_COLUMN = "capture_channel"  # the channels read from the barcodes, in obs
ALL_CELLS = "__all__"  # doublet_channel_key: all cells as one channel
ALL_CELLS_LABEL = "all cells"
MIN_CHANNEL_CELLS = scrublet_lean.MIN_CHANNEL_CELLS
# obs column names that say "capture unit", most specific first ("well" last: in plate
# data it is one cell, and the median-size rule passes over it).
CHANNEL_NAMES = (
    CHANNEL_COLUMN, "gem_well", "gem_group", "gem", "channel", "channel_id", "capture",
    "capture_id", "lane", "lane_id", "library", "library_id", "lib", "lib_id", "well",
)
# A 10x cell barcode: an optional prefix and one of _ : # -, the barcode (16 of ACGT for
# v2 / v3 chemistry, 14 for v1), an optional -N or .N (or more: anndata's "-1-0").
BARCODE_PATTERN = (r"^(?:(?P<prefix>.+?)[_:#-])?(?P<found>)(?:[ACGT]{16}|[ACGT]{14})"
                   r"(?P<suffix>(?:[-.]\d+)+)?$")
BARCODE_SAMPLE = 2000  # barcodes checked before all of them are parsed

REASONS = {
    "requested": "the column chosen in the doublet step",
    "requested_all": "all cells as one channel, as chosen in the doublet step",
    "channel_column": "the column that names the capture channel",
    "barcode_prefix": "the prefix of the cell barcodes",
    "barcode_suffix": "the -N suffix of the cell barcodes (the GEM well)",
    "barcode_prefix_suffix": "the prefix and the -N suffix of the cell barcodes",
    "donor": "the donor / sample column (a guess: no capture channel was found)",
    "batch": "the batch column (a guess: no capture channel was found)",
    "none": "no capture channel was found, so all cells are taken as one channel",
}
GUESS_NOTE = (
    "Doublets form within one capture channel, and donors or samples pooled in one lane "
    "are not channels (Kang 2018 demultiplexed 8 donors from each of its 2 lanes), so the "
    "per-channel rates are a guess. If the data has a lane or GEM well column, pick it as "
    "the doublet step's channels."
)


class DoubletError(ValueError):
    """A request that can't be met (no scores, a bad threshold); nothing changed."""


def rate_for_cells(n_cells: int) -> float:
    """The expected doublet rate of one channel holding ``n_cells`` captured cells."""
    return float(np.clip(RATE_PER_1000_CELLS * n_cells / 1000.0, MIN_RATE, MAX_RATE))


@dataclass(frozen=True)
class Channels:
    """The capture channels Scrublet runs in: each cell's index into ``names``."""

    column: str | None  # the obs column (CHANNEL_COLUMN for barcodes); None: all cells as one
    reason: str  # a key of REASONS
    names: list[str]
    codes: np.ndarray  # per cell, an index into names; -1: no channel (not scored)
    derived: bool = False  # read from the barcodes: the step writes them to CHANNEL_COLUMN
    note: str | None = None  # GUESS_NOTE for the donor / batch fallback
    skipped: tuple[str, ...] = ()  # candidates passed over, and why

    @property
    def sizes(self) -> np.ndarray:
        return np.bincount(self.codes[self.codes >= 0], minlength=len(self.names))

    def labels(self) -> pd.Categorical:
        return pd.Categorical.from_codes(self.codes, categories=self.names)


def barcode_channels(names) -> tuple[list[str], np.ndarray, str] | None:
    """The capture channels 10x cell barcodes carry: (labels, codes, kind), or None.

    Every barcode must match ``BARCODE_PATTERN`` (a sample of ``BARCODE_SAMPLE`` is
    checked first, so other names cost little), and the prefixes and suffixes must split
    the cells into 2 or more groups. kind: "barcode_prefix", "barcode_suffix" or
    "barcode_prefix_suffix", whichever part varies. A label is the prefix ("CTRL"), the
    suffix without its first separator ("12", "1-0"), or both ("S1-2"); labels that are
    numbers sort as numbers."""
    pattern = re.compile(BARCODE_PATTERN)
    index = pd.Index(names).astype(str)
    n = len(index)
    if not n or not all(pattern.match(x) for x in index[::max(1, n // BARCODE_SAMPLE)][:BARCODE_SAMPLE]):
        return None
    parts = index.str.extract(pattern)
    if parts["found"].isna().any():
        return None
    p_codes, p_names = pd.factorize(parts["prefix"].fillna(""))
    s_codes, s_names = pd.factorize(parts["suffix"].fillna(""))
    del parts
    kp, ks = len(p_names), len(s_names)
    if kp == 1 and ks == 1:
        return None
    if ks == 1:
        codes, labels, kind = p_codes, [str(p) for p in p_names], "barcode_prefix"
    elif kp == 1:
        codes, labels, kind = s_codes, [str(s)[1:] for s in s_names], "barcode_suffix"
    else:
        used, codes = np.unique(p_codes.astype(np.int64) * ks + s_codes, return_inverse=True)
        labels = [f"{p_names[u // ks]}{s_names[u % ks]}" for u in used]
        kind = "barcode_prefix_suffix"
    order = sorted(range(len(labels)),
                   key=lambda i: (0, int(labels[i]), "") if labels[i].isdigit() else (1, 0, labels[i]))
    remap = np.empty(len(labels), dtype=np.int64)
    remap[order] = np.arange(len(labels))
    return [labels[i] for i in order], remap[np.asarray(codes, dtype=np.int64)], kind


def _norm(name) -> str:
    return re.sub(r"[\s.\-]+", "_", str(name).strip().lower())


def _named_columns(obs: pd.DataFrame) -> list[str]:
    """obs columns named like a capture channel, in CHANNEL_NAMES order."""
    by: dict[str, str] = {}
    for c in obs.columns:
        by.setdefault(_norm(c), str(c))
    return [by[k] for k in CHANNEL_NAMES if k in by]


def _of_column(obs: pd.DataFrame, col: str, reason: str, note: str | None = None) -> Channels:
    names, codes = qc_groups.group_codes(obs[col])
    return Channels(str(col), reason, names, np.asarray(codes, dtype=np.int64), note=note)


def _all_cells(n: int, reason: str, skipped: list[str] | tuple[str, ...] = ()) -> Channels:
    return Channels(None, reason, [ALL_CELLS_LABEL], np.zeros(n, dtype=np.int64),
                    skipped=tuple(skipped))


# Parsing 1.3M barcodes takes ~4 s, and the doublet card, the preflight and the step all
# ask: the last answers, by the obs_names object they were read from (a new AnnData, or
# renamed cells, is a new object; the weak reference lets it go).
_BARCODE_CACHE: list[tuple[weakref.ref, tuple | None]] = []
_BARCODE_CACHE_SIZE = 2
_BARCODE_LOCK = threading.Lock()  # requests read it from worker threads


def _cached_barcode_channels(index: pd.Index) -> tuple[list[str], np.ndarray, str] | None:
    with _BARCODE_LOCK:
        for ref, found in _BARCODE_CACHE:
            if ref() is index:
                return found
    found = barcode_channels(index)
    with _BARCODE_LOCK:
        keep = [e for e in _BARCODE_CACHE if e[0]() is not None][-(_BARCODE_CACHE_SIZE - 1):]
        _BARCODE_CACHE[:] = [*keep, (weakref.ref(index), found)]
    return found


def _from_barcodes(adata) -> Channels | None:
    found = _cached_barcode_channels(adata.obs_names)
    if found is None:
        return None
    labels, codes, kind = found
    return Channels(CHANNEL_COLUMN, kind, list(labels), codes.copy(), derived=True)


def channels(adata, key: str | None = None) -> Channels:
    """The capture channels Scrublet runs in (see the top for the order). ``key`` is the
    doublet step's ``doublet_channel_key``: an obs column, ``ALL_CELLS``, or
    ``CHANNEL_COLUMN`` before the barcodes were written there. Raises DoubletError for
    a column that is not there."""
    obs = adata.obs
    n = int(adata.n_obs)
    if key:
        if key == ALL_CELLS:
            return _all_cells(n, "requested_all")
        if key in obs.columns:
            return _of_column(obs, key, "requested")
        if key == CHANNEL_COLUMN and (ch := _from_barcodes(adata)) is not None:
            return ch
        raise DoubletError(f"Column '{key}' not found.")
    skipped: list[str] = []

    def fits(ch: Channels) -> bool:
        sizes = ch.sizes
        median = float(np.median(sizes)) if sizes.size else 0.0
        if sizes.size and median >= MIN_CHANNEL_CELLS:
            return True
        skipped.append(f"'{ch.column}': {len(ch.names):,} groups with a median of {median:,.0f} cells, "
                       f"under the {MIN_CHANNEL_CELLS} Scrublet needs")
        return False

    def candidates():  # each looked at only when the ones before it were passed over
        for c in _named_columns(obs):
            yield _of_column(obs, c, "channel_column")
        yield _from_barcodes(adata)
        col, why = qc_groups.sample_grouping(adata, max_groups=None)
        if col:
            yield _of_column(obs, col, why, note=GUESS_NOTE)

    for ch in candidates():
        if ch is not None and fits(ch):
            return replace(ch, skipped=tuple(skipped))
    return _all_cells(n, "none", skipped)


def channel_grouping(adata, column: str | None = None) -> tuple[str | None, str]:
    """(column, why) of the capture channels: the obs column (``CHANNEL_COLUMN`` for
    channels read from the barcodes; None: all cells as one channel) and a key of REASONS."""
    ch = channels(adata, column)
    return ch.column, ch.reason


def _rates(sizes: np.ndarray, rate: float | None) -> list[float]:
    """Each channel's expected rate: ``rate`` when set, else its own cells' rule of thumb."""
    if rate is not None and rate > 0:
        return [float(rate)] * len(sizes)
    return [rate_for_cells(int(k)) for k in sizes]


def _suggestion(adata, ch: Channels) -> dict:
    n = int(adata.n_obs)
    sizes = ch.sizes
    scored = sizes >= MIN_CHANNEL_CELLS
    labelled = int(sizes.sum())
    rate = (float(sum(int(k) * rate_for_cells(int(k)) for k in sizes) / labelled) if labelled
            else rate_for_cells(n))
    per = ch.column is not None
    rows = [{"name": g, "n_cells": int(k), "rate": round(rate_for_cells(int(k)), 4), "scored": bool(ok)}
            for g, k, ok in zip(ch.names, sizes, scored)] if per else []
    basis = RULE_OF_THUMB + (" Each channel's rate follows its own cells." if per else
                             " All cells are taken as one channel.")
    return {
        "rate": round(rate, 4),  # the share of all cells expected to be doublets
        "per_channel": per,
        "column": ch.column,
        "reason": ch.reason,
        "reason_text": REASONS[ch.reason],
        "derived": ch.derived,
        "note": ch.note,
        "skipped": list(ch.skipped),
        "n_cells": n,
        "n_channels": len(ch.names),
        "largest_channel": int(sizes.max()) if sizes.size else 0,
        "n_unscored": int(n - sizes[scored].sum()),  # small channels and cells with none
        "min_channel_cells": MIN_CHANNEL_CELLS,
        "channels": rows,
        "basis": basis,
    }


def suggest_rate(adata, column: str | None = None) -> dict:
    """The capture channels the doublet step would use (``channels``; ``column`` as its
    ``doublet_channel_key``) and each one's expected rate, with ``rate`` the share of all
    cells expected to be doublets (the cell-weighted mean of the channels' rates)."""
    return _suggestion(adata, channels(adata, column))


def run_scrublet(adata, counts, rate: float | None = None, key: str | None = None) -> dict:
    """Scrublet in each capture channel of ``adata`` (``channels(adata, key)``) on the rows of
    ``counts`` (cells x genes, in memory or backed), each channel with its own rate (``rate``
    for all when set) and threshold, neighbours exact up to ``EXACT_KNN_MAX_CELLS`` cells.

    Writes ``doublet_score`` and ``predicted_doublet`` (bool; NaN score and False for cells
    not scored), the barcode channels to ``CHANNEL_COLUMN`` when they came from the
    barcodes, and ``uns["scrublet"]``; drops an applied threshold. Returns what the
    provenance records: the channels, why, and each one's rate, threshold and calls."""
    ch = channels(adata, key)
    n = int(adata.n_obs)
    sizes = ch.sizes
    rates = _rates(sizes, rate)
    approx = [int(k) > EXACT_KNN_MAX_CELLS for k in sizes]
    score, call, runs = scrublet_lean.scrublet_by_channel(counts, ch.codes, rates, approx,
                                                          MIN_CHANNEL_CELLS)
    obs = adata.obs
    obs[SCORE] = score
    obs[CALL] = call
    if ch.derived:
        obs[CHANNEL_COLUMN] = ch.labels()
    single = len(runs) == 1 and runs[0]["n_cells"] == n
    neighbors = ["approximate" if a else "exact" for a in approx]
    table = {
        "name": list(ch.names),
        "n_cells": [r["n_cells"] for r in runs],
        "n_scored": [r["n_scored"] for r in runs],
        "expected_doublet_rate": rates,
        "threshold": [np.nan if r["threshold"] is None else r["threshold"] for r in runs],
        "n_called": [r["n_called"] for r in runs],
        "neighbors": neighbors,
        "status": [r["status"] for r in runs],
    }
    info = dict(runs[0]["uns"]) if single and runs[0]["uns"] is not None else {
        "parameters": {"sim_doublet_ratio": 2.0, "random_state": 0, "min_channel_cells": MIN_CHANNEL_CELLS}}
    info["scview_channels"] = table
    info["scview_channel_reason"] = ch.reason
    if ch.column is not None:
        info["scview_channel_column"] = ch.column
    adata.uns["scrublet"] = info
    clear_threshold(adata)  # set on the previous run's scores

    user = rate is not None and rate > 0
    ran = {neighbors[j] for j, r in enumerate(runs) if r["status"] == "scored"}
    found: dict = {
        "expected_doublet_rate": float(rate) if user else (rates[0] if single else None),
        "doublet_channel_key": key or "",
        "channel_column": ch.column,
        "channel_reason": ch.reason,
        "channels": [{"name": g, "n_cells": r["n_cells"], "expected_doublet_rate": rt,
                      "neighbors": nb, "status": r["status"], "threshold": r["threshold"],
                      "n_called": r["n_called"]}
                     for g, r, rt, nb in zip(ch.names, runs, rates, neighbors)],
        "min_channel_cells": MIN_CHANNEL_CELLS,
        "n_unscored": int(np.isnan(score).sum()),
        "neighbors": ran.pop() if len(ran) == 1 else ("mixed" if ran else neighbors[0]),
    }
    if not user:
        found["expected_rate_from"] = (f"{ch.column} channels (per channel)" if ch.column is not None
                                       else "all cells as one channel")
    if ch.note:
        found["channel_note"] = ch.note
    if ch.skipped:
        found["channels_skipped"] = list(ch.skipped)
    failed = [r.get("error") for r in runs if r["status"] == "failed"]
    if failed:
        logger.warning("Scrublet failed in %d channel(s), left unscored: %s", len(failed), failed[0])
    logger.info("Doublet detection (scrublet, %d channel(s) by %s): %d / %d cells flagged, %d not scored",
                len(runs), ch.reason, int(call.sum()), n, found["n_unscored"])
    return found


def run_classifier(adata, counts, rate: float | None = None, key: str | None = None) -> dict:
    """The scDblFinder-style classifier (core/doublet_classifier.py) in each capture
    channel of ``adata``, as ``run_scrublet`` runs Scrublet: the same channels, rates (the
    rule of thumb unless ``rate`` is set) and floor, on the rows of ``counts``.

    Writes ``doublet_score_classifier`` and ``predicted_doublet_classifier`` (NaN score and
    False for cells not scored), the barcode channels to ``CHANNEL_COLUMN`` when they came
    from the barcodes, and the per-channel record to ``uns[CLASSIFIER_KEY]``. Returns what
    the provenance records under ``classifier``: each channel's rate, threshold and calls."""
    ch = channels(adata, key)
    sizes = ch.sizes
    rates = _rates(sizes, rate)
    score, call, runs = doublet_classifier.classify_by_channel(
        counts, ch.codes, rates, MIN_CHANNEL_CELLS, rows=scrublet_lean.channel_rows)
    obs = adata.obs
    obs[CLASSIFIER_SCORE] = score
    obs[CLASSIFIER_CALL] = call
    if ch.derived:
        obs[CHANNEL_COLUMN] = ch.labels()
    table = {
        "name": list(ch.names),
        "n_cells": [r["n_cells"] for r in runs],
        "n_scored": [r["n_scored"] for r in runs],
        "expected_doublet_rate": rates,
        "threshold": [np.nan if r["threshold"] is None else r["threshold"] for r in runs],
        "n_called": [r["n_called"] for r in runs],
        "status": [r["status"] for r in runs],
    }
    info: dict = {"scview_channels": table, "scview_channel_reason": ch.reason,
                  "parameters": {"min_channel_cells": MIN_CHANNEL_CELLS, "seed": doublet_classifier.SEED}}
    if ch.column is not None:
        info["scview_channel_column"] = ch.column
    adata.uns[CLASSIFIER_KEY] = info
    failed = [r.get("error") for r in runs if r["status"] == "failed"]
    if failed:
        logger.warning("The doublet classifier failed in %d channel(s), left unscored: %s",
                       len(failed), failed[0])
    logger.info("Doublet detection (classifier, %d channel(s) by %s): %d / %d cells flagged",
                len(runs), ch.reason, int(call.sum()), int(adata.n_obs))
    return {
        "channel_column": ch.column,
        "channel_reason": ch.reason,
        "channels": [{"name": g, "n_cells": r["n_cells"], "expected_doublet_rate": rt,
                      "status": r["status"], "threshold": r["threshold"], "n_called": r["n_called"]}
                     for g, r, rt in zip(ch.names, runs, rates)],
        "n_called": int(call.sum()),
        "n_unscored": int(np.isnan(score).sum()),
    }


def _drop_scrublet(adata) -> None:
    """Forget Scrublet's results (a doublet step that ran the classifier alone)."""
    for col in (SCORE, CALL, SCRUBLET_CALL):
        if col in adata.obs.columns:
            del adata.obs[col]
    adata.uns.pop("scrublet", None)
    adata.uns.pop(THRESHOLD_KEY, None)


def _drop_classifier(adata) -> None:
    """Forget the classifier's results (a doublet step that ran Scrublet alone)."""
    for col in (CLASSIFIER_SCORE, CLASSIFIER_CALL):
        if col in adata.obs.columns:
            del adata.obs[col]
    adata.uns.pop(CLASSIFIER_KEY, None)


def method_of(value: str | None) -> str:
    """The doublet step's ``doublet_method`` as one of ``METHODS`` (an alias of the
    classifier read as it; empty: Scrublet). Raises DoubletError for anything else."""
    m = (value or SCRUBLET_METHOD).strip().lower()
    m = METHOD_ALIASES.get(m, m)
    if m not in METHODS:
        raise DoubletError(f"Unknown doublet method '{value}': use {', '.join(METHODS)}.")
    return m


def run_doublets(adata, counts, method: str | None = None, rate: float | None = None,
                 key: str | None = None) -> dict:
    """The doublet step: ``method`` (``METHODS``) in each capture channel. Each run
    writes its own columns and drops the other detector's, so the obs holds what this
    step found. Returns what the provenance records: ``doublet_method`` and, for
    Scrublet, ``run_scrublet``'s record at the top level (as before the classifier),
    the classifier's under ``classifier``."""
    m = method_of(method)
    found: dict = {"doublet_method": m}
    if m in (SCRUBLET_METHOD, BOTH_METHOD):
        found.update(run_scrublet(adata, counts, rate, key))
    else:
        _drop_scrublet(adata)
    if m in (CLASSIFIER_METHOD, BOTH_METHOD):
        found["classifier"] = run_classifier(adata, counts, rate, key)
        if m == CLASSIFIER_METHOD:
            ch = found["classifier"]
            found.update(expected_doublet_rate=float(rate) if rate is not None and rate > 0 else None,
                         doublet_channel_key=key or "", channel_column=ch["channel_column"],
                         channel_reason=ch["channel_reason"], min_channel_cells=MIN_CHANNEL_CELLS)
    else:
        _drop_classifier(adata)
    return found


def classifier_record(adata) -> dict | None:
    """The classifier's last channels (``uns[CLASSIFIER_KEY]``), as ``channel_record``."""
    return _record(adata.uns.get(CLASSIFIER_KEY))


def channel_record(adata) -> dict | None:
    """The last doublet step's channels from ``uns["scrublet"]``: the column (None: all
    cells as one), why, and one row per channel; None for scores from before channels
    were recorded, or from another tool."""
    return _record(adata.uns.get("scrublet"))


def _record(info) -> dict | None:
    table = info.get("scview_channels") if isinstance(info, dict) else None
    if not isinstance(table, dict) or "name" not in table:
        return None

    def values(key: str) -> list:
        return np.asarray(table.get(key, []), dtype=object).ravel().tolist()

    names = [str(v) for v in values("name")]
    cols = {k: values(k) for k in ("n_cells", "n_scored", "expected_doublet_rate", "threshold",
                                   "n_called", "neighbors", "status")}
    rows = []
    for i, name in enumerate(names):
        t = cols["threshold"][i] if i < len(cols["threshold"]) else None
        t = float(t) if t is not None and np.isfinite(float(t)) else None
        row = {"name": name, "n_cells": int(cols["n_cells"][i]), "n_scored": int(cols["n_scored"][i]),
               "expected_doublet_rate": float(cols["expected_doublet_rate"][i]), "threshold": t,
               "n_called": int(cols["n_called"][i]), "status": str(cols["status"][i])}
        if i < len(cols["neighbors"]):  # Scrublet's; the classifier has none to choose
            row["neighbors"] = str(cols["neighbors"][i])
        rows.append(row)
    column = info.get("scview_channel_column")
    return {"column": None if column is None else str(column),
            "reason": str(info.get("scview_channel_reason") or ""), "rows": rows}


def channels_text(rows: list[dict], column: str | None = None) -> str:
    """One line on the channels a doublet step ran in (its recorded rows, from
    ``channel_record`` or the provenance), for a caption, the QC report or a prompt (133
    dicts would flood it): "133 channels by gem_well, 512-1,034 cells each, expected
    0.4-0.8 %, 705 called"."""
    rows = [r for r in rows if isinstance(r, dict)]
    if not rows:
        return ""

    def num(r: dict, key: str) -> float:
        v = r.get(key)
        return float(v) if isinstance(v, (int, float)) else 0.0

    sizes = [int(num(r, "n_cells")) for r in rows]
    rates = [num(r, "expected_doublet_rate") for r in rows]
    called = sum(int(num(r, "n_called")) for r in rows)
    small = [r for r in rows if r.get("status") == "too_small"]
    failed = [r for r in rows if r.get("status") == "failed"]
    if len(rows) == 1:
        head = f"one channel of {sizes[0]:,} cells, expected {100 * rates[0]:.1f} %, {called:,} called"
    else:
        head = (f"{len(rows):,} channels{f' by {column}' if column else ''}, {min(sizes):,}-{max(sizes):,} "
                f"cells each, expected {100 * min(rates):.1f}-{100 * max(rates):.1f} %, {called:,} called")
    if small:
        k = sum(int(num(r, "n_cells")) for r in small)
        head += (f"; {len(small):,} channel{'' if len(small) == 1 else 's'} under {MIN_CHANNEL_CELLS} cells "
                 f"not scored ({k:,} cells)")
    if failed:
        head += f"; Scrublet failed in {len(failed):,}"
    return head


def scrublet_threshold(adata) -> float | None:
    """The threshold Scrublet picked (None when it ran per channel, or found none)."""
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
    # A cell not scored (NaN) is not called.
    with np.errstate(invalid="ignore"):
        return np.isfinite(scores) & (scores > threshold)


def _check_threshold(threshold: float) -> float:
    t = float(threshold)
    if not np.isfinite(t) or t < 0:
        raise DoubletError("The threshold must be a number of 0 or more.")
    return t


def _recorded_rate(adata) -> float | None:
    """The one expected rate the latest doublet step ran with, from provenance (None
    when each channel had its own: they are in the channel rows)."""
    for entry in reversed(provenance.read_provenance(adata).get("history", [])):
        if entry.get("step") == "doublet_detection":
            rate = (entry.get("params") or {}).get("expected_doublet_rate")
            return float(rate) if isinstance(rate, (int, float)) else None
    return None


def _grouping(adata, column: str | None, record: dict | None) -> tuple[Channels | None, bool]:
    """The rows of the channel table: ``column`` when named, else the channels the last run
    used (its column, when still in obs), else the channels a run would use now. (None:
    one channel of every cell, no table) and whether they are the recorded run's."""
    if column:
        ch = channels(adata, column)
    elif record is not None:
        col = record["column"]
        if col is None:
            return None, True
        if col in adata.obs.columns:
            return _of_column(adata.obs, col, record["reason"] or "channel_column"), True
        ch = channels(adata, col) if col == CHANNEL_COLUMN else channels(adata)
    else:
        ch = channels(adata)
    if ch.column is None:
        return None, record is not None and record["column"] is None
    return ch, record is not None and record["column"] == ch.column


def summary(adata, threshold: float | None = None, column: str | None = None,
            n_bins: int = N_BINS) -> dict:
    """The doublet-score histogram, how many cells a threshold calls doublets, and per
    capture channel the cells, cells scored, calls and the rate the run used (or would).

    ``threshold`` defaults to the applied one, else Scrublet's when it ran once over every
    cell; with a run per channel there is none (each channel had its own) until one is
    given, and ``n_called_now`` counts the calls. ``column`` names the channels for the
    table (default: the last run's). ``few_called`` names the channels where Scrublet's
    own threshold called almost nothing (``FEW_CALLED_SHARE`` of a channel of at least
    ``FEW_CALLED_MIN_CELLS``): its threshold may have missed the doublets there. Without
    scores the answer still carries the suggestion (the channels and rates the doublet
    step would use)."""
    suggestion = suggest_rate(adata, column)
    n = int(adata.n_obs)
    if SCORE not in adata.obs.columns:
        out: dict = {"available": False, "n_cells": n, "suggestion": suggestion}
        from scview.core import spatial

        if spatial.is_spot_data(adata):  # Visium spots: no doublet step to suggest
            out["not_applicable"] = spatial.DOUBLET_SKIP
        if CLASSIFIER_CALL in adata.obs.columns:  # the classifier ran alone
            ch, _ = _grouping(adata, column, classifier_record(adata))
            out.update(classifier=_classifier_summary(adata, ch, n_bins), methods=None,
                       source=doublet_source.describe(adata.obs))
        return out
    scores = _scores(adata)
    scored = np.isfinite(scores)
    finite = scores[scored]
    st = scrublet_threshold(adata)
    applied = applied_threshold(adata)
    if threshold is None:
        threshold = applied["threshold"] if applied else st
    if threshold is not None:
        threshold = _check_threshold(threshold)
    record = channel_record(adata)

    top = _hist_top(finite, [threshold, st] + [r["threshold"] for r in (record or {}).get("rows", [])])
    counts, edges = np.histogram(finite, bins=n_bins, range=(0.0, top))
    now = doublet_source.scrublet_calls(adata.obs)
    called = _calls_at(scores, threshold) if threshold is not None else None
    ch, recorded = _grouping(adata, column, record)
    rows = _channel_rows(scores, called if called is not None else now, ch,
                         record if recorded else None, edges)
    small = sum(r["n_cells"] for r in (record or {}).get("rows", []) if r["status"] == "too_small")
    out = {
        "available": True,
        "n_cells": n,
        "n_scored": int(finite.size),
        "n_unscored": int(n - finite.size),  # small channels, failed ones, under 3 genes
        "n_unscored_small_channels": int(small),
        "min_channel_cells": MIN_CHANNEL_CELLS,
        "hist": {"bin_edges": np.round(edges, 6).tolist(), "counts": counts.tolist()},
        "median": round(float(np.median(finite)), 4) if finite.size else None,
        "scrublet_threshold": st,  # exact: the calls at it are Scrublet's own
        "per_channel": bool(record is not None and len(record["rows"]) > 1),
        "channel_column": record["column"] if record else None,
        "channel_reason": record["reason"] if record else None,
        "channel_reason_text": REASONS.get(record["reason"]) if record else None,
        "applied": applied,
        "scrublet_call_kept": SCRUBLET_CALL in adata.obs.columns,
        "n_called_now": None if now is None else int(now.sum()),
        "expected_rate_used": _recorded_rate(adata),
        "threshold": threshold,
        "n_called": None if called is None else int(called.sum()),
        "fraction_called": None if called is None else (round(float(called.mean()), 4) if n else 0.0),
        "suggestion": suggestion,
        "channels": rows,
        "channels_column": None if ch is None else ch.column,
        "channels_recorded": recorded,
        # Channels whose automatic threshold called almost nothing: worth a look at their
        # scores, and maybe a threshold set by hand (FEW_CALLED_*)
        "few_called": [r["name"] for r in rows if r["few_called"]],
        "few_called_share": FEW_CALLED_SHARE,
        "few_called_min_cells": FEW_CALLED_MIN_CELLS,
        "source": doublet_source.describe(adata.obs),  # what the doublet views use
        "agreement": _agreement(adata, called if called is not None else now, threshold),
        # The second detector, and how its calls agree with Scrublet's (at the threshold shown)
        "classifier": _classifier_summary(adata, ch, n_bins),
        "methods": _methods(adata, called if called is not None else now, ch),
    }
    return out


def _classifier_summary(adata, ch: Channels | None, n_bins: int = N_BINS) -> dict | None:
    """The classifier's calls (None when it did not run): cells scored, called, its score
    histogram over 0-1, and per channel of ``ch`` the cells, cells scored and called, with
    the threshold and status of its record (when the channels are the ones it ran in)."""
    obs = adata.obs
    calls = doublet_source.classifier_calls(obs)
    if calls is None:
        return None
    scores = (pd.to_numeric(obs[CLASSIFIER_SCORE], errors="coerce").to_numpy(dtype=np.float64)
              if CLASSIFIER_SCORE in obs.columns else np.full(calls.size, np.nan))
    finite = np.isfinite(scores)
    counts, edges = np.histogram(scores[finite], bins=n_bins, range=(0.0, 1.0))
    n = int(calls.size)
    record = classifier_record(adata)
    by = {r["name"]: r for r in record["rows"]} if record else {}
    rows = []
    if ch is not None:
        k = len(ch.names)
        lab = ch.codes >= 0
        sizes = np.bincount(ch.codes[lab], minlength=k)
        scored = np.bincount(ch.codes[lab & finite], minlength=k)
        hits = np.bincount(ch.codes[lab & calls], minlength=k)
        mine = record is not None and record["column"] == ch.column
        for i, name in enumerate(ch.names):
            r = by.get(name, {}) if mine else {}
            rows.append({"name": name, "n_cells": int(sizes[i]), "n_scored": int(scored[i]),
                         "n_called": int(hits[i]),
                         "fraction": round(int(hits[i]) / int(sizes[i]), 4) if sizes[i] else 0.0,
                         "threshold": r.get("threshold"),
                         "status": r.get("status") or ("scored" if scored[i] else "not_scored")})
    elif record and len(record["rows"]) == 1:
        r = record["rows"][0]
        rows.append({"name": r["name"], "n_cells": n, "n_scored": int(finite.sum()),
                     "n_called": int(calls.sum()), "fraction": round(float(calls.mean()), 4) if n else 0.0,
                     "threshold": r["threshold"], "status": r["status"]})
    return {
        "label": doublet_source.CLASSIFIER_LABEL,
        "n_scored": int(finite.sum()),
        "n_unscored": int(n - finite.sum()),
        "n_called": int(calls.sum()),
        "fraction": round(float(calls.mean()), 4) if n else 0.0,
        "median": round(float(np.median(scores[finite])), 4) if finite.any() else None,
        "hist": {"bin_edges": np.round(edges, 6).tolist(), "counts": counts.tolist()},
        "channel_column": record["column"] if record else None,
        "channels": rows,
    }


def _methods(adata, scrublet: np.ndarray | None, ch: Channels | None) -> dict | None:
    """Scrublet's calls (at the threshold shown) x the classifier's: called by both, by
    one only, by neither, over all cells and per channel of ``ch``. None unless both ran."""
    classifier = doublet_source.classifier_calls(adata.obs)
    if scrublet is None or classifier is None:
        return None
    s, c = np.asarray(scrublet, dtype=bool), np.asarray(classifier, dtype=bool)
    out: dict = {"n_cells": int(s.size), "scrublet": int(s.sum()), "classifier": int(c.sum()),
                 "both": int((s & c).sum()), "scrublet_only": int((s & ~c).sum()),
                 "classifier_only": int((~s & c).sum()), "neither": int((~s & ~c).sum()),
                 "channels": []}
    if ch is not None:
        k = len(ch.names)
        lab = ch.codes >= 0

        def count(mask: np.ndarray) -> np.ndarray:
            return np.bincount(ch.codes[lab & mask], minlength=k)

        sizes, both, so, co = count(np.ones_like(s)), count(s & c), count(s & ~c), count(~s & c)
        out["channels"] = [{"name": g, "n_cells": int(sizes[i]), "both": int(both[i]),
                            "scrublet_only": int(so[i]), "classifier_only": int(co[i])}
                           for i, g in enumerate(ch.names)]
    return out


def _hist_top(finite: np.ndarray, thresholds: list) -> float:
    """The histogram's right edge: just past the highest score and every threshold known
    (1 when that is over 1, or nothing is known). Low expected rates (0.4-0.7 % per GEM
    well) put every score under 0.1, which a 0-1 axis squeezed into a few bins."""
    known = [float(t) for t in thresholds if t is not None and np.isfinite(float(t))]
    top = max([float(finite.max()) if finite.size else 0.0, *known])
    if top <= 0:
        return 1.0
    return top * 1.05 if top * 1.05 <= 1.0 else max(1.0, top)


def few_called(n_scored: int, n_called: int) -> bool:
    """Scrublet's own threshold called almost nothing in a channel of this size
    (``FEW_CALLED_SHARE``, ``FEW_CALLED_MIN_CELLS``)."""
    return n_scored >= FEW_CALLED_MIN_CELLS and n_called < FEW_CALLED_SHARE * n_scored


def _channel_rows(scores: np.ndarray, called: np.ndarray | None, ch: Channels | None,
                  record: dict | None, edges: np.ndarray | None = None) -> list[dict]:
    """Per channel: cells now, cells scored, cells called (at the threshold shown, else
    the calls now), the rate the run used (its record) or would use, Scrublet's own
    threshold in that channel, its status (scored | too_small | failed), and the cells
    Scrublet's own threshold calls there now (``scrublet_called``). ``few_called``: that
    threshold called almost nothing (``few_called``); such a row carries its scores'
    histogram over ``edges`` (``hist``), to look for the doublets it may have missed."""
    if ch is None:
        return []
    k = len(ch.names)
    codes = ch.codes
    lab = codes >= 0
    finite = np.isfinite(scores)
    sizes = np.bincount(codes[lab], minlength=k)
    n_scored = np.bincount(codes[lab & finite], minlength=k)
    hits = None if called is None else np.bincount(codes[lab & called], minlength=k)
    by = {r["name"]: r for r in record["rows"]} if record else {}
    rows = []
    for i, name in enumerate(ch.names):
        r = by.get(name, {})
        size = int(sizes[i])
        status = r.get("status") or ("scored" if n_scored[i] else
                                     "too_small" if size < MIN_CHANNEL_CELLS else "not_scored")
        h = None if hits is None else int(hits[i])
        own = r.get("threshold")
        row = {
            "name": name, "n_cells": size, "n_scored": int(n_scored[i]), "n_called": h,
            "fraction": round(h / size, 4) if h is not None and size else 0.0,
            "expected_rate": round(float(r.get("expected_doublet_rate", rate_for_cells(size))), 4),
            "scrublet_threshold": own,
            "status": status,
            "scrublet_called": None,
            "few_called": False,
        }
        if status == "scored" and own is not None:
            mine = lab & (codes == i) & finite
            n_own = int(_calls_at(np.where(mine, scores, np.nan), float(own)).sum())
            row["scrublet_called"] = n_own
            row["few_called"] = few_called(int(n_scored[i]), n_own)
            if row["few_called"] and edges is not None:
                counts, _ = np.histogram(scores[mine], bins=edges)
                row["hist"] = counts.tolist()
        rows.append(row)
    return rows


def _agreement(adata, scrublet: np.ndarray | None, threshold: float | None) -> dict | None:
    """Cell hashing x Scrublet's calls at ``threshold`` (None: its saved call), overall
    and per sample (``qc_groups.sample_grouping``: what hashing tells apart, usually all in
    one channel), when the dataset has both (``doublet_source.agreement``)."""
    hashing = doublet_source.hashing_calls(adata.obs)
    if hashing is None or scrublet is None:
        return None
    column, _ = qc_groups.sample_grouping(adata)
    names, codes = (qc_groups.group_codes(adata.obs[column]) if column
                    else ([], np.full(adata.n_obs, -1)))
    out = doublet_source.agreement(hashing, scrublet, doublet_source.hash_negatives(adata.obs),
                                   names, np.asarray(codes))
    return {**out, "threshold": threshold, "column": column if names else None}


def clear_threshold(adata) -> None:
    """Forget an applied threshold (a new Scrublet run made new scores and calls)."""
    adata.uns.pop(THRESHOLD_KEY, None)
    if SCRUBLET_CALL in adata.obs.columns:
        del adata.obs[SCRUBLET_CALL]


def apply_threshold(adata, threshold: float | None, column: str | None = None) -> dict:
    """Call doublets at ``threshold`` (one threshold over every channel's scores) into
    ``predicted_doublet``, keeping Scrublet's per-channel call in
    ``predicted_doublet_scrublet``; ``None`` puts Scrublet's call back. A cell not
    scored stays uncalled. Raises DoubletError before changing anything. Returns the
    new ``summary``."""
    from scview.core.obs_writes import forget_column_results

    scores = _scores(adata)
    if threshold is not None:
        threshold = _check_threshold(threshold)
    if column:
        channels(adata, column)  # a bad column fails here, unchanged
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
            obs[SCRUBLET_CALL] = doublet_source.scrublet_calls(obs)
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


# The replay script copies these (core/replay_script.py): numpy, pandas and re only.
REPLAY_CONSTANTS = ("BARCODE_PATTERN", "BARCODE_SAMPLE")
REPLAY_CODE = (barcode_channels,)
