"""Which doublet calls the app uses, and where they came from: one place.

Several tools leave per-cell doublet calls in obs: cell hashing (``hash_class``,
``core/hashing.py``: Singlet / Doublet / Negative), Scrublet
(``predicted_doublet``: the pipeline's doublet step, or a threshold on its score,
``core/doublets.py``) and text class columns from other tools (scDblFinder,
DoubletFinder, demuxlet).

When cell hashing has been run it is the primary call: a droplet with two
samples' tags is a doublet by measurement, the ground truth for doublets between
samples. Hashing cannot see two cells of one sample in a droplet; Scrublet
(simulated doublets from the cells' own profiles) is the only call that can, so it
stays visible as the second opinion, and the filtering step can drop either
call or both.

Sources (``resolve``):

- ``auto`` (the default): hashing when ``hash_class`` is present, else Scrublet's
  ``predicted_doublet``, else the first text class column;
- ``hashing``: ``hash_class`` == Doublet only;
- ``scrublet``: ``predicted_doublet`` only;
- ``either``: a hashing Doublet OR a Scrublet doublet (whichever of the two exist).

A hashing ``Negative`` (no tag above background) is not a doublet: usually an empty
droplet or a cell whose tagging failed. ``hash_negatives`` gives them, for the
filtering step's separate opt-in.

A cell Scrublet did not score (its capture channel was too small for it, or it had
under 3 genes: a NaN ``doublet_score``) is not called a doublet; ``scrublet_unscored``
gives them, and ``Calls.unscored`` / ``describe`` carry how many, so a view can say so
rather than count them as singlets silently.

``legacy`` is the order used before this module (Scrublet first, then the text
columns with ``hash_class`` among them): the replay of a filtering recorded before
the choice existed uses it, so an old recipe keeps its meaning.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

AUTO, HASHING, SCRUBLET, EITHER, LEGACY = "auto", "hashing", "scrublet", "either", "legacy"
SOURCES = (AUTO, HASHING, SCRUBLET, EITHER)
HASH_CLASS = "hash_class"  # core/hashing.py CLASS
SCRUBLET_CALL = "predicted_doublet"
SCRUBLET_SCORE = "doublet_score"
# Doublet calls stored as text by other tools (scDblFinder, demuxlet, DoubletFinder).
TEXT_COLUMNS = ("scDblFinder.class", "scDblFinder_class", "multiplets", "doublet_class")
TEXT_PREFIX = "DF.classification"
LABELS = {HASHING: "cell hashing", SCRUBLET: "Scrublet", EITHER: "cell hashing or Scrublet"}
NEGATIVE_NOTE = ("Hashing negatives have no tag above background: usually empty droplets or cells "
                 "whose tagging failed, not doublets.")


@dataclass(frozen=True)
class Calls:
    """Per-cell doublet calls and where they came from."""

    calls: np.ndarray  # bool per cell
    source: str  # hashing | scrublet | either | a text column's name
    label: str  # for captions: "cell hashing", "Scrublet", ...
    columns: tuple[str, ...]  # the obs columns read
    # Cells Scrublet did not score, when its calls are used (else None): not called.
    unscored: np.ndarray | None = None

    @property
    def n(self) -> int:
        return int(self.calls.sum())


def _text_in(col: pd.Series, values: tuple[str, ...]) -> np.ndarray:
    """Cells whose value, as lower-case text, is one of ``values``. A categorical
    column is read through its categories (no string per cell: the filter preview
    asks on every threshold change); a missing value matches nothing."""
    if isinstance(col.dtype, pd.CategoricalDtype):
        hit = np.asarray(pd.Index(col.cat.categories).astype(str).str.lower().isin(values), dtype=bool)
        return np.append(hit, False)[col.cat.codes.to_numpy()]  # code -1 reads the last slot
    return col.astype(str).str.lower().isin(values).to_numpy()


def _text_doublets(col: pd.Series) -> np.ndarray:
    return _text_in(col, ("doublet",))


def scrublet_calls(obs: pd.DataFrame) -> np.ndarray | None:
    """Scrublet's ``predicted_doublet`` as booleans (text after a round trip
    included), or None without it."""
    if SCRUBLET_CALL not in obs.columns:
        return None
    col = obs[SCRUBLET_CALL]
    if pd.api.types.is_bool_dtype(col):
        return col.to_numpy(dtype=bool)
    # Categorical/text "True"/"False" after a round trip: astype(bool) would make
    # every non-empty string True.
    return _text_in(col, ("true", "1", "1.0", "doublet"))


def scrublet_unscored(obs: pd.DataFrame) -> np.ndarray | None:
    """Cells with Scrublet's call but no score (a NaN ``doublet_score``: their channel
    was under the size Scrublet needs, or they had under 3 genes), or None without
    Scrublet's call and score."""
    if SCRUBLET_CALL not in obs.columns or SCRUBLET_SCORE not in obs.columns:
        return None
    return ~np.isfinite(pd.to_numeric(obs[SCRUBLET_SCORE], errors="coerce").to_numpy(dtype=np.float64))


def hashing_calls(obs: pd.DataFrame) -> np.ndarray | None:
    """Cells cell hashing calls doublets (``hash_class`` == Doublet), or None
    when hashing was not run. Uncalled cells (NaN) are not doublets."""
    return _text_doublets(obs[HASH_CLASS]) if HASH_CLASS in obs.columns else None


def hash_negatives(obs: pd.DataFrame) -> np.ndarray | None:
    """Cells hashing calls Negative (no tag above background), or None."""
    if HASH_CLASS not in obs.columns:
        return None
    return _text_in(obs[HASH_CLASS], ("negative",))


def text_call(obs: pd.DataFrame) -> tuple[str, np.ndarray] | None:
    """(column, calls) of the first text class column another tool left, or None."""
    names = [c for c in TEXT_COLUMNS if c in obs.columns]
    names += [str(c) for c in obs.columns if str(c).startswith(TEXT_PREFIX)]
    return (names[0], _text_doublets(obs[names[0]])) if names else None


def _text_label(column: str) -> str:
    return "DoubletFinder" if column.startswith(TEXT_PREFIX) else (
        "scDblFinder" if column.startswith("scDblFinder") else f"'{column}'")


def resolve(obs: pd.DataFrame, source: str = AUTO) -> Calls | None:
    """The doublet calls ``source`` stands for, or None when the dataset has none
    of that kind. Raises ValueError for an unknown source."""
    if source not in SOURCES and source != LEGACY:
        raise ValueError(f"Unknown doublet source '{source}': use {', '.join(SOURCES)}.")
    hashing = hashing_calls(obs)
    scrublet = scrublet_calls(obs)
    if source in (AUTO, HASHING) and hashing is not None:
        return Calls(hashing, HASHING, LABELS[HASHING], (HASH_CLASS,))
    if source == HASHING:
        return None
    if source == EITHER and hashing is not None and scrublet is not None:
        return Calls(hashing | scrublet, EITHER, LABELS[EITHER], (HASH_CLASS, SCRUBLET_CALL),
                     scrublet_unscored(obs))
    if source == EITHER and hashing is not None:
        return Calls(hashing, HASHING, LABELS[HASHING], (HASH_CLASS,))
    if scrublet is not None:
        return Calls(scrublet, SCRUBLET, LABELS[SCRUBLET], (SCRUBLET_CALL,), scrublet_unscored(obs))
    if source in (SCRUBLET, EITHER):
        return None
    if source == LEGACY and hashing is not None:  # hash_class sat among the text columns
        names = [c for c in TEXT_COLUMNS if c in obs.columns]
        if not names:
            return Calls(hashing, HASHING, LABELS[HASHING], (HASH_CLASS,))
    found = text_call(obs)
    if found is None:
        return None
    return Calls(found[1], found[0], _text_label(found[0]), (found[0],))


def options(obs: pd.DataFrame) -> list[str]:
    """The sources that mean something for this dataset (``auto`` always)."""
    has_h, has_s = HASH_CLASS in obs.columns, SCRUBLET_CALL in obs.columns
    return [AUTO] + [s for s, ok in ((HASHING, has_h), (SCRUBLET, has_s), (EITHER, has_h and has_s)) if ok]


def describe(obs: pd.DataFrame, source: str = AUTO) -> dict | None:
    """What the doublet views use (``source``, default auto) and, when hashing and
    Scrublet both exist, how they agree: counts and fractions of all cells. None
    without any doublet calls."""
    used = resolve(obs, source)
    if used is None:
        return None
    n = int(len(obs))
    frac = (lambda k: round(k / n, 4)) if n else (lambda k: 0.0)
    out: dict = {"source": used.source, "label": used.label, "columns": list(used.columns),
                 "n_cells": n, "n_doublets": used.n, "fraction": frac(used.n),
                 "hashing": None, "scrublet": None, "both": None, "options": options(obs)}
    hashing, scrublet = hashing_calls(obs), scrublet_calls(obs)
    if hashing is not None:
        neg = hash_negatives(obs)
        out["hashing"] = {"n_doublets": int(hashing.sum()), "fraction": frac(int(hashing.sum())),
                          "n_negatives": int(neg.sum())}
    if scrublet is not None:
        unscored = scrublet_unscored(obs)
        out["scrublet"] = {"n_doublets": int(scrublet.sum()), "fraction": frac(int(scrublet.sum())),
                           "n_unscored": 0 if unscored is None else int(unscored.sum())}
    if hashing is not None and scrublet is not None:
        both = int((hashing & scrublet).sum())
        out["both"] = {"n": both, "fraction": frac(both)}
    return out


def _pct(f: float) -> str:
    return f"{100 * f:.1f}%"


def text(info: dict | None) -> str:
    """One sentence for a caption or the co-pilot: "12.0% doublets by cell hashing;
    Scrublet flags 7.0%, 4.0% by both" (``describe``'s output). "" without calls."""
    if not info:
        return ""
    n = info["n_doublets"]
    head = f"{n:,} cell{'' if n == 1 else 's'} ({_pct(info['fraction'])}) {'is a doublet' if n == 1 else 'are doublets'} by {info['label']}"
    if info.get("both") is not None:
        s, b = info["scrublet"], info["both"]
        if info["source"] == HASHING:
            head += f"; Scrublet flags {s['n_doublets']:,} ({_pct(s['fraction'])}), {b['n']:,} ({_pct(b['fraction'])}) by both"
        elif info["source"] == EITHER:
            h = info["hashing"]
            head += (f" (hashing {h['n_doublets']:,}, Scrublet {s['n_doublets']:,}, "
                     f"{b['n']:,} by both)")
    k = (info.get("scrublet") or {}).get("n_unscored", 0)
    if k and info["source"] in (SCRUBLET, EITHER):
        head += (f"; {k:,} cell{'' if k == 1 else 's'} not scored by Scrublet "
                 "(a capture channel too small for it, or under 3 genes), so not called")
    return head


def agreement(hashing: np.ndarray, scrublet: np.ndarray, negatives: np.ndarray,
              names: list[str], codes: np.ndarray) -> dict:
    """Hashing x Scrublet doublet calls: the 2 x 2 counts over all cells, the
    hashing negatives Scrublet flags, and per sample (``names`` / ``codes`` from
    ``qc_groups.group_codes``; -1 = no sample) the cells, each call and both."""
    h, s = np.asarray(hashing, dtype=bool), np.asarray(scrublet, dtype=bool)
    neg = np.asarray(negatives, dtype=bool)
    out: dict = {
        "n_cells": int(h.size),
        "hashing": int(h.sum()), "scrublet": int(s.sum()),
        "both": int((h & s).sum()), "hashing_only": int((h & ~s).sum()),
        "scrublet_only": int((~h & s).sum()), "neither": int((~h & ~s).sum()),
        "negatives": int(neg.sum()), "negatives_scrublet": int((neg & s).sum()),
        "samples": [], "unassigned": None,
    }
    if names:
        k = len(names)
        lab = codes >= 0

        def count(mask: np.ndarray) -> np.ndarray:
            return np.bincount(codes[lab & mask], minlength=k)

        sizes, hh, ss, bb = count(np.ones_like(h)), count(h), count(s), count(h & s)
        out["samples"] = [{"name": g, "n_cells": int(sizes[i]), "hashing": int(hh[i]),
                           "scrublet": int(ss[i]), "both": int(bb[i])} for i, g in enumerate(names)]
        none = ~lab
        if none.any():
            out["unassigned"] = {"n_cells": int(none.sum()), "hashing": int((none & h).sum()),
                                 "scrublet": int((none & s).sum()), "both": int((none & h & s).sum())}
    return out
