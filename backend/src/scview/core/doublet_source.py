"""Which doublet calls the app uses, and where they came from: one place.

Several tools leave per-cell doublet calls in obs: cell hashing (``hash_class``,
``core/hashing.py``: Singlet / Doublet / Negative), Scrublet
(``predicted_doublet``: the pipeline's doublet step, or a threshold on its score,
``core/doublets.py``), the doublet step's second detector, an scDblFinder-style
classifier (``predicted_doublet_classifier``, ``core/doublet_classifier.py``), and text
class columns from other tools (scDblFinder, DoubletFinder, demuxlet).

When cell hashing has been run it is the primary call: a droplet with two
samples' tags is a doublet by measurement, the ground truth for doublets between
samples. Hashing cannot see two cells of one sample in a droplet; Scrublet
(simulated doublets from the cells' own profiles) is the only call that can, so it
stays visible as the second opinion, and the filtering step can drop either
call or both.

Sources (``resolve``):

- ``auto`` (the default): hashing when ``hash_class`` is present; else, when both
  detectors ran, their consensus; else the one detector that ran (the classifier's
  ``predicted_doublet_classifier`` or Scrublet's ``predicted_doublet``); else the first
  text class column;
- ``hashing``: ``hash_class`` == Doublet only;
- ``scrublet``: ``predicted_doublet`` only;
- ``classifier``: ``predicted_doublet_classifier`` only;
- ``consensus``: cells BOTH detectors call doublets (the one that ran, when only one did);
- ``either``: a cell any of the calls present calls a doublet (hashing, Scrublet, the
  classifier).

Why consensus for auto once both detectors ran: the two find doublets in different ways
(Scrublet: a cell's neighbourhood among simulated doublets; the classifier: a model
trained on them, with more features), and each makes false calls of its own. A cell both
call is a doublet with more confidence, and removing a real cell (a rare or transitional
state that sits between two types) costs more than keeping a doublet, which the
doublet-like cluster flag and the doublet views still show. ``either`` is there for the
strict cut.

A hashing ``Negative`` (no tag above background) is not a doublet: usually an empty
droplet or a cell whose tagging failed. ``hash_negatives`` gives them, for the
filtering step's separate opt-in.

A cell Scrublet did not score (its capture channel was too small for it, or it had
under 3 genes: a NaN ``doublet_score``) is not called a doublet; ``scrublet_unscored``
gives them (``classifier_unscored`` the classifier's), and ``Calls.unscored`` /
``describe`` carry how many, so a view can say so rather than count them as singlets
silently.

``legacy`` is the order used before this module (Scrublet first, then the text
columns with ``hash_class`` among them): the replay of a filtering recorded before
the choice existed uses it, so an old recipe keeps its meaning.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

AUTO, HASHING, SCRUBLET, EITHER, LEGACY = "auto", "hashing", "scrublet", "either", "legacy"
CLASSIFIER, CONSENSUS = "classifier", "consensus"
SOURCES = (AUTO, HASHING, SCRUBLET, CLASSIFIER, CONSENSUS, EITHER)
HASH_CLASS = "hash_class"  # core/hashing.py CLASS
SCRUBLET_CALL = "predicted_doublet"
SCRUBLET_SCORE = "doublet_score"
CLASSIFIER_CALL = "predicted_doublet_classifier"  # core/doublets.py run_classifier
CLASSIFIER_SCORE = "doublet_score_classifier"
# Doublet calls stored as text by other tools (scDblFinder, demuxlet, DoubletFinder).
TEXT_COLUMNS = ("scDblFinder.class", "scDblFinder_class", "multiplets", "doublet_class")
TEXT_PREFIX = "DF.classification"
CLASSIFIER_LABEL = "the doublet classifier"
LABELS = {HASHING: "cell hashing", SCRUBLET: "Scrublet", CLASSIFIER: CLASSIFIER_LABEL,
          CONSENSUS: "both Scrublet and the doublet classifier", EITHER: "cell hashing or Scrublet"}
# The calls ``either`` joins, in this order (their labels for "a or b or c").
_EITHER_PARTS = ((HASHING, "cell hashing"), (SCRUBLET, "Scrublet"), (CLASSIFIER, CLASSIFIER_LABEL))
NEGATIVE_NOTE = ("Hashing negatives have no tag above background: usually empty droplets or cells "
                 "whose tagging failed, not doublets.")


@dataclass(frozen=True)
class Calls:
    """Per-cell doublet calls and where they came from."""

    calls: np.ndarray  # bool per cell
    source: str  # hashing | scrublet | classifier | consensus | either | a text column's name
    label: str  # for captions: "cell hashing", "Scrublet", ...
    columns: tuple[str, ...]  # the obs columns read
    # Cells a detector whose calls are used did not score (else None): not called.
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


def classifier_calls(obs: pd.DataFrame) -> np.ndarray | None:
    """The classifier's ``predicted_doublet_classifier`` as booleans, or None without it."""
    if CLASSIFIER_CALL not in obs.columns:
        return None
    col = obs[CLASSIFIER_CALL]
    if pd.api.types.is_bool_dtype(col):
        return col.to_numpy(dtype=bool)
    return _text_in(col, ("true", "1", "1.0", "doublet"))


def classifier_unscored(obs: pd.DataFrame) -> np.ndarray | None:
    """Cells with the classifier's call but no score (NaN), or None without both."""
    if CLASSIFIER_CALL not in obs.columns or CLASSIFIER_SCORE not in obs.columns:
        return None
    return ~np.isfinite(pd.to_numeric(obs[CLASSIFIER_SCORE], errors="coerce").to_numpy(dtype=np.float64))


def _any(*masks: np.ndarray | None) -> np.ndarray | None:
    """OR of the masks that exist (None when none does)."""
    found = [m for m in masks if m is not None]
    if not found:
        return None
    out = found[0].copy()
    for m in found[1:]:
        out |= m
    return out


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


def _detector(obs: pd.DataFrame, source: str) -> Calls | None:
    """One detector's calls (Scrublet or the classifier), or None without them."""
    if source == SCRUBLET:
        calls = scrublet_calls(obs)
        return None if calls is None else Calls(calls, SCRUBLET, LABELS[SCRUBLET], (SCRUBLET_CALL,),
                                                scrublet_unscored(obs))
    calls = classifier_calls(obs)
    return None if calls is None else Calls(calls, CLASSIFIER, LABELS[CLASSIFIER], (CLASSIFIER_CALL,),
                                            classifier_unscored(obs))


def _consensus(obs: pd.DataFrame) -> Calls | None:
    """Cells both detectors call (the one that ran alone when only one did)."""
    scr, cls = _detector(obs, SCRUBLET), _detector(obs, CLASSIFIER)
    if scr is None or cls is None:
        return scr if cls is None else cls
    return Calls(scr.calls & cls.calls, CONSENSUS, LABELS[CONSENSUS], (SCRUBLET_CALL, CLASSIFIER_CALL),
                 _any(scr.unscored, cls.unscored))


def _either(obs: pd.DataFrame) -> Calls | None:
    """Cells any of the calls present calls (hashing, Scrublet, the classifier); one
    source alone when it is the only one."""
    hashing = hashing_calls(obs)
    parts = [] if hashing is None else [Calls(hashing, HASHING, LABELS[HASHING], (HASH_CLASS,))]
    parts += [c for c in (_detector(obs, SCRUBLET), _detector(obs, CLASSIFIER)) if c is not None]
    if len(parts) <= 1:
        return parts[0] if parts else None
    labels = [dict(_EITHER_PARTS)[c.source] for c in parts]
    label = " or ".join(labels) if len(labels) == 2 else ", ".join(labels[:-1]) + " or " + labels[-1]
    return Calls(_any(*[c.calls for c in parts]), EITHER, label,
                 tuple(c.columns[0] for c in parts), _any(*[c.unscored for c in parts]))


def resolve(obs: pd.DataFrame, source: str = AUTO) -> Calls | None:
    """The doublet calls ``source`` stands for, or None when the dataset has none
    of that kind. Raises ValueError for an unknown source."""
    if source not in SOURCES and source != LEGACY:
        raise ValueError(f"Unknown doublet source '{source}': use {', '.join(SOURCES)}.")
    hashing = hashing_calls(obs)
    if source in (AUTO, HASHING) and hashing is not None:
        return Calls(hashing, HASHING, LABELS[HASHING], (HASH_CLASS,))
    if source == HASHING:
        return None
    if source == EITHER:
        return _either(obs)
    if source in (SCRUBLET, CLASSIFIER):
        return _detector(obs, source)
    if source in (AUTO, CONSENSUS):
        found = _consensus(obs)
        if found is not None or source == CONSENSUS:
            return found
    if source == LEGACY:
        scrublet = _detector(obs, SCRUBLET)
        if scrublet is not None:
            return scrublet
        if hashing is not None and not [c for c in TEXT_COLUMNS if c in obs.columns]:
            # hash_class sat among the text columns, before DoubletFinder's
            return Calls(hashing, HASHING, LABELS[HASHING], (HASH_CLASS,))
    found = text_call(obs)
    if found is None:
        return None
    return Calls(found[1], found[0], _text_label(found[0]), (found[0],))


def options(obs: pd.DataFrame) -> list[str]:
    """The sources that mean something for this dataset (``auto`` always)."""
    has_h, has_s = HASH_CLASS in obs.columns, SCRUBLET_CALL in obs.columns
    has_c = CLASSIFIER_CALL in obs.columns
    return [AUTO] + [s for s, ok in ((HASHING, has_h), (SCRUBLET, has_s), (CLASSIFIER, has_c),
                                     (CONSENSUS, has_s and has_c),
                                     (EITHER, int(has_h) + int(has_s) + int(has_c) >= 2)) if ok]


def describe(obs: pd.DataFrame, source: str = AUTO) -> dict | None:
    """What the doublet views use (``source``, default auto) and how the calls present
    agree: counts and fractions of all cells (``both``: hashing and Scrublet;
    ``detectors_both``: Scrublet and the classifier). None without any doublet calls."""
    used = resolve(obs, source)
    if used is None:
        return None
    n = int(len(obs))
    frac = (lambda k: round(k / n, 4)) if n else (lambda k: 0.0)
    out: dict = {"source": used.source, "label": used.label, "columns": list(used.columns),
                 "n_cells": n, "n_doublets": used.n, "fraction": frac(used.n),
                 "hashing": None, "scrublet": None, "classifier": None, "both": None,
                 "detectors_both": None, "options": options(obs)}
    hashing, scrublet, classifier = hashing_calls(obs), scrublet_calls(obs), classifier_calls(obs)
    if hashing is not None:
        neg = hash_negatives(obs)
        out["hashing"] = {"n_doublets": int(hashing.sum()), "fraction": frac(int(hashing.sum())),
                          "n_negatives": int(neg.sum())}
    for key, calls, unscored in (("scrublet", scrublet, scrublet_unscored(obs)),
                                 ("classifier", classifier, classifier_unscored(obs))):
        if calls is not None:
            out[key] = {"n_doublets": int(calls.sum()), "fraction": frac(int(calls.sum())),
                        "n_unscored": 0 if unscored is None else int(unscored.sum())}
    if hashing is not None and scrublet is not None:
        both = int((hashing & scrublet).sum())
        out["both"] = {"n": both, "fraction": frac(both)}
    if scrublet is not None and classifier is not None:
        both = int((scrublet & classifier).sum())
        out["detectors_both"] = {"n": both, "fraction": frac(both)}
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
    d, s, c = info.get("detectors_both"), info.get("scrublet"), info.get("classifier")
    if d is not None and s is not None and c is not None:
        if info["source"] == CONSENSUS:
            head += f" (Scrublet {s['n_doublets']:,}, the classifier {c['n_doublets']:,})"
        elif info["source"] == SCRUBLET:
            head += f"; the doublet classifier flags {c['n_doublets']:,}, {d['n']:,} by both"
        elif info["source"] == CLASSIFIER:
            head += f"; Scrublet flags {s['n_doublets']:,}, {d['n']:,} by both"
    for key, name in (("scrublet", "Scrublet"), ("classifier", "the doublet classifier")):
        k = (info.get(key) or {}).get("n_unscored", 0)
        if k and info["source"] in ((SCRUBLET, EITHER, CONSENSUS) if key == "scrublet"
                                    else (CLASSIFIER, EITHER, CONSENSUS)):
            head += (f"; {k:,} cell{'' if k == 1 else 's'} not scored by {name} "
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
