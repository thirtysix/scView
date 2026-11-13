"""Curated vs author cell-type labels: a cautious check that the curation kept their meaning.

CELLxGENE files carry a curated ``cell_type`` (Cell Ontology names) beside the authors' own
labels (``cell_type_original``, ``author_cell_type``). The curation can go wrong: Kuppe 2022's
"Myeloid" became "immature innate lymphoid cell", and scView's DE composition warning then
named that label. This cross-tabs the two columns and, per author label, takes the curated
label most of its cells carry. A mapping is flagged when it looks implausible:

- the two names sit in opposite blood lineages (myeloid words on one side, lymphoid words on
  the other), or
- the names share nothing (no word, no prefix such as "Fib" / "fibroblast", no initials such
  as "pDC" / "plasmacytoid dendritic cell") and the curated label is a progenitor or rare
  kind ("immature", "progenitor", "stem", "precursor", "innate lymphoid") while the author
  group is large (``LARGE_SHARE`` of the cells, or ``LARGE_CELLS``).

It only flags; nothing is renamed. Names alone cannot prove a mapping wrong (an author's
"Cluster 3" shares nothing with any type), so the note says to look, with the cross-tab.
"""

from __future__ import annotations

import re
from typing import Any

import pandas as pd

CURATED = ("cell_type",)
_AUTHOR_RE = re.compile(
    r"^(author[_ .-]?cell[_ .-]?types?|cell[_ .-]?types?[_ .-]?(original|orig|author|authors|submitter)|"
    r"original[_ .-]?cell[_ .-]?types?|celltype[_ .-]?original|author[_ .-]?annotation)$", re.I)
MIN_CELLS = 20  # author labels with fewer cells are listed, never flagged
LARGE_SHARE = 0.05
LARGE_CELLS = 500
MAX_LABELS = 100  # author labels listed (largest first)
TOP = 3  # curated labels shown per author label

_MYELOID = {"myeloid", "monocyte", "monocytes", "mono", "macrophage", "macrophages", "macro", "mac", "mφ",
            "neutrophil", "neutrophils", "granulocyte", "granulocytes", "dendritic", "dc", "mdc", "cdc",
            "mast", "basophil", "eosinophil", "kupffer", "microglia", "microglial", "osteoclast"}
_LYMPHOID = {"lymphoid", "lymphocyte", "lymphocytes", "t", "b", "nk", "ilc", "plasma", "nkt", "treg",
             "lymphoblast"}
_RARE_RE = re.compile(r"immature|progenitor|precursor|stem|innate lymphoid|blast|primordial|"
                      r"embryonic|fetal|unknown|unclassified", re.I)
_STOP = {"cell", "cells", "type", "of", "the", "and", "positive", "negative", "human", "mouse", "like"}


def columns(adata) -> tuple[str, str] | None:
    """(curated, author) column names, or None when the dataset lacks either."""
    obs = adata.obs.columns
    cur = next((c for c in CURATED if c in obs), None)
    auth = next((str(c) for c in obs if _AUTHOR_RE.match(str(c)) and str(c) != cur), None)
    return (cur, auth) if cur and auth else None


def _words(label: str) -> list[str]:
    return [w for w in re.split(r"[^a-z0-9φ]+", str(label).lower()) if w and w not in _STOP]


def _stem(w: str) -> str:
    return w[:-1] if len(w) > 3 and w.endswith("s") and not w.endswith("ss") else w


def _lineage(words: list[str]) -> str | None:
    m = any(w in _MYELOID for w in words)
    lym = any(w in _LYMPHOID for w in words) or any(w.startswith(("cd4", "cd8")) for w in words)
    return "myeloid" if m and not lym else "lymphoid" if lym and not m else None


def related(author: str, curated: str) -> bool:
    """Whether two names share anything: a word, a prefix of 3+ letters ("Fib" / "fibroblast"),
    or initials ("pDC" / "plasmacytoid dendritic cell", "vSMCs" / "smooth muscle cell")."""
    a = [_stem(w) for w in _words(author)]
    c = [_stem(w) for w in _words(curated)]
    if not a or not c:
        return True  # nothing to compare: not flagged on names
    for x in a:
        for y in c:
            if x == y or (min(len(x), len(y)) >= 3 and (x.startswith(y) or y.startswith(x))):
                return True
    initials = "".join(w[0] for w in re.split(r"[^a-z0-9]+", str(curated).lower()) if w)
    for x in a:
        if len(x) >= 2 and x.isalpha() and (x in initials or (len(initials) >= 3 and initials in x)):
            return True
    return False


def _why(author: str, curated: str, n: int, n_total: int) -> str | None:
    la, lc = _lineage(_words(author)), _lineage(_words(curated))
    if la and lc and la != lc:
        return f"'{author}' reads as {la} but its cells are curated as '{curated}' ({lc})"
    large = n >= LARGE_CELLS or n >= LARGE_SHARE * n_total
    if large and _RARE_RE.search(curated) and not related(author, curated):
        return (f"{n:,} cells the authors call '{author}' are curated as '{curated}', a progenitor or "
                "rare kind, and the names share nothing")
    return None


def check(adata) -> dict[str, Any] | None:
    """The cross-tab and its flags; None when the dataset has no curated + author pair."""
    pair = columns(adata)
    if pair is None:
        return None
    cur, auth = pair
    a = adata.obs[auth].astype(str).where(adata.obs[auth].notna())
    c = adata.obs[cur].astype(str).where(adata.obs[cur].notna())
    keep = (a.notna() & c.notna()).to_numpy()
    if keep.sum() == 0:
        return None
    table = pd.crosstab(a[keep], c[keep])
    sizes = table.sum(axis=1).sort_values(ascending=False)
    n_total = int(sizes.sum())
    rows, flags = [], []
    for name in sizes.index[:MAX_LABELS]:
        n = int(sizes[name])
        top = table.loc[name].sort_values(ascending=False)
        top = top[top > 0]
        main = str(top.index[0])
        row = {"author": str(name), "n_cells": n, "curated": main, "share": round(float(top.iloc[0]) / n, 3),
               "others": [{"curated": str(k), "n_cells": int(v)} for k, v in top.iloc[1:TOP].items()]}
        why = _why(str(name), main, n, n_total) if n >= MIN_CELLS else None
        if why:
            row["flag"] = why
            flags.append(why)
        rows.append(row)
    note = None
    if flags:
        note = (f"The curated '{cur}' may not keep the meaning of the authors' '{auth}' for "
                f"{len(flags)} label{'s' if len(flags) != 1 else ''}. Results that name '{cur}' labels (DE "
                f"composition, annotation checks) then name the curated label: check the cross-tab, and "
                f"group by '{auth}' where the curated name looks wrong. Nothing was renamed.")
    return {"curated_column": cur, "author_column": auth, "n_author_labels": int(sizes.size),
            "n_cells": n_total, "rows": rows, "n_flagged": len(flags), "flags": flags[:5], "note": note}

