"""Library concordance: do two libraries of one sample hold the same cells?

A sample sequenced as two (or more) libraries should give the same cell-type mix and
a similar depth in each. When one library holds a very different mix, it may carry
another sample's cells (a swap or a mixed suspension) or a failed capture. Emont
2022's chow mouse NCD07 (inguinal fat) was sequenced as Mm_ING_10-1 and -2: mAd3 +
mAd4 are 48% of one and 1.6% of the other, with the HFD signature in a chow mouse,
and -1 has the lowest depth of all 26 libraries. Nothing said so; a crosstab did.

The columns:

- **samples**: the column named, else the donor / sample column the QC table uses
  (``qc_groups.sample_grouping``; only a donor-like column, never a batch).
- **libraries**: the column named, else the first obs column, in this order, whose
  values nest in the samples (each library one sample, ``NEST_SHARE`` of its cells)
  with some sample holding two or more: one named like a capture channel
  (``doublets._named_columns``: library_id, lane, gem_well ...), a batch-like one, a
  sample-like one that does not name a person (biosample_id), then the cell
  barcodes' prefix or ``-N`` suffix (``doublets.barcode_channels``).
- **what a sample is**: a person's libraries can be different tissues or conditions
  (a mouse's inguinal and epididymal fat). A label column that is one value per
  library and differs between one sample's libraries (not batch- or channel-like)
  splits the sample, so only libraries alike in it are compared (``split_by``).
- **cell types**: an author cell-type column, else the clustering
  (``de_composition.default_column``).

The cut is calibrated, not fixed: each library pair's composition distance (total
variation distance: half the summed absolute difference of the shares, 0 to 1) is
judged against the distances between DIFFERENT samples alike in their sample-level
labels (``reference``: condition, tissue, sex ...; a label is dropped, condition-like
last, while fewer than ``MIN_REF_PAIRS`` pairs remain). A pair is flagged above the
90th percentile of those distances, never under ``FLOOR``, and only when the
difference is not chance (chi-square p < ``MAX_P``). Too few pairs even over all
samples: a fixed ``RULE_CUT``, said in the note. Depth (median UMIs per cell) is
judged the same way on |log2 ratio|, floor 2-fold. Libraries under ``MIN_LIBRARY_CELLS``
are listed as too small, not judged.

Obs only plus the per-cell counts QC already has (``AnnDataAdaptor.qc_per_cell``).
Used by ``GET /qc/libraries`` (the QC per sample table, MCP ``library_concordance``),
the QC report and the co-pilot's QC facts.
"""

from __future__ import annotations

import itertools
import re

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency

from scview.core import doublets, qc_groups
from scview.core.confounding import BATCH_RE, CONDITION_RE, TECH_RE
from scview.core.replicates import PERSON_PATTERNS, donor_candidates

TAG = "result:libraries"

MIN_LIBRARY_CELLS = 200  # a library smaller than this is listed, not judged
NEST_SHARE = 0.95  # a library nests in a sample when one sample holds this share of its cells
MIN_REF_PAIRS = 8  # sample pairs needed to calibrate the cut
QUANTILE = 0.9  # a library pair above this share of the between-sample distances is flagged
FLOOR = 0.15  # ... and never under this distance (15 points of cell-type share moved)
RULE_CUT = 0.30  # the cut when there are too few samples to calibrate
MAX_P = 1e-3  # chi-square of the two libraries' cell-type counts
DEPTH_FLOOR = 1.0  # |log2| of the median UMI ratio: never flag depth under 2-fold
MIN_SHARE_POINTS = 5.0  # a cell type named as differing moves at least this many points
MIN_DESIGN_SAMPLES = 3  # with this many samples checked ...
DESIGN_SHARE = 0.5  # ... more than this share flagged reads as the design, not a fault
MAX_NAMED = 3  # cell types named per flagged pair
MAX_SPLIT_VALUES = 50
MAX_STRATUM_VALUES = 8

_ALIAS_RE = re.compile(r"ontology|term_id", re.I)
_CHANNEL = {doublets._norm(n) for n in doublets.CHANNEL_NAMES}


class LibraryConcordanceError(ValueError):
    """A named column that is not there: a 404."""


def _labels(series: pd.Series) -> pd.Series:
    return series.astype(str).where(series.notna())


def _is_label(series: pd.Series) -> bool:
    return (isinstance(series.dtype, pd.CategoricalDtype) or series.dtype == object
            or pd.api.types.is_string_dtype(series.dtype) or pd.api.types.is_bool_dtype(series.dtype))


def _technical(name: str) -> bool:
    return bool(BATCH_RE.search(name) or TECH_RE.search(name)) or doublets._norm(name) in _CHANNEL


def _nesting(lib: np.ndarray, samp: np.ndarray, n_lib: int) -> np.ndarray | None:
    """Each library's sample (index into the samples), or None when some library of 20+
    cells has no sample holding ``NEST_SHARE`` of its cells (libraries pooling samples)."""
    keep = (lib >= 0) & (samp >= 0)
    if not keep.any():
        return None
    frame = pd.DataFrame({"l": lib[keep], "s": samp[keep]})
    counts = frame.groupby(["l", "s"]).size()
    out = np.full(n_lib, -1)
    for li, sub in counts.groupby(level=0):
        top = sub.idxmax()[1]
        total = int(sub.sum())
        if int(sub.max()) < NEST_SHARE * total and total >= 20:
            return None
        out[int(li)] = int(top)
    return out


def _library_candidates(adata, sample: str) -> list[tuple[str, str]]:
    """(column, source) in the order they are tried; ``doublets.CHANNEL_COLUMN`` with source
    "barcodes" means the barcodes' channels when no such obs column exists."""
    obs = adata.obs
    out: list[tuple[str, str]] = []
    seen = {sample}

    def add(col: str, source: str) -> None:
        if col not in seen and col in obs.columns:
            seen.add(col)
            out.append((col, source))

    for c in doublets._named_columns(obs):
        add(c, "channel_column")
    for c in obs.columns:
        if BATCH_RE.search(str(c)):
            add(str(c), "batch")
    for c in donor_candidates(obs, [sample]):
        if not any(p.search(c) for p in PERSON_PATTERNS):
            add(c, "sample_like")
    if doublets.CHANNEL_COLUMN not in obs.columns:
        out.append((doublets.CHANNEL_COLUMN, "barcodes"))
    return out


def _codes_of(adata, column: str, source: str) -> tuple[list[str], np.ndarray] | None:
    if source == "barcodes":
        found = doublets._cached_barcode_channels(adata.obs_names)
        return None if found is None else (list(found[0]), np.asarray(found[1]))
    names, codes = qc_groups.group_codes(adata.obs[column])
    return names, np.asarray(codes)


def _per_library_columns(obs: pd.DataFrame, lib: np.ndarray, n_lib: int, skip: set[str],
                         max_values: int) -> dict[str, np.ndarray]:
    """Label columns with one value per library (2 to ``max_values`` values, fewer than
    the libraries), each library's value as an index into the column's values
    (-1: none); aliases of an earlier column (the same partition) left out, a
    readable name kept over an ontology id."""
    out: dict[str, np.ndarray] = {}
    parts: dict[tuple, str] = {}
    has = lib >= 0
    cols = sorted((str(c) for c in obs.columns), key=lambda c: bool(_ALIAS_RE.search(c)))
    for name in cols:
        if name in skip or _technical(name):
            continue
        s = obs[name]
        if not _is_label(s):
            continue
        k = int(s.nunique(dropna=True))
        if not 2 <= k <= max_values or k >= n_lib:
            continue
        _, codes = qc_groups.group_codes(s)
        codes = np.asarray(codes)
        frame = pd.DataFrame({"l": lib[has], "v": codes[has]})
        frame = frame[frame["v"] >= 0].drop_duplicates()
        if frame.empty or frame["l"].duplicated().any():
            continue  # varies within a library: a per-cell column
        per = np.full(n_lib, -1)
        per[frame["l"].to_numpy()] = frame["v"].to_numpy()
        key = tuple(pd.factorize(per)[0])
        if key in parts:
            continue
        parts[key] = name
        out[name] = per
    return out


def _value_names(obs: pd.DataFrame, col: str) -> list[str]:
    return qc_groups.group_codes(obs[col])[0]


def _tvd(a: np.ndarray, b: np.ndarray) -> float:
    return float(0.5 * np.abs(a - b).sum())


def _shares(table: np.ndarray) -> np.ndarray:
    tot = table.sum(axis=1, keepdims=True).astype(float)
    return np.divide(table, tot, out=np.zeros(table.shape, float), where=tot > 0)


def _chi2_p(a: np.ndarray, b: np.ndarray) -> float:
    t = np.vstack([a, b])
    t = t[:, t.sum(axis=0) > 0]
    if t.shape[1] < 2:
        return 1.0
    try:
        return float(chi2_contingency(t)[1])
    except ValueError:  # pragma: no cover - degenerate table
        return 1.0


def _median(values: np.ndarray, idx: np.ndarray) -> float | None:
    v = values[idx]
    v = v[np.isfinite(v)]
    return float(np.median(v)) if v.size else None


def _log2_ratio(a: float | None, b: float | None) -> float | None:
    if not a or not b or a <= 0 or b <= 0:
        return None
    return float(abs(np.log2(a / b)))


def _none(reason: str, **cols) -> dict:
    return {"status": "none", "reason": reason, "sample_column": cols.get("sample"),
            "library_column": cols.get("library"), "library_source": cols.get("source"),
            "celltype_column": cols.get("celltype"), "split_by": [], "samples": [],
            "n_checked": 0, "n_flagged": 0, "flagged": [], "reference": None, "note": None}


def _pct(v: float) -> str:
    return f"{100 * v:.1f}%" if v < 0.1 else f"{100 * v:.0f}%"


def _reference_pairs(units: list[dict], strata: dict[str, np.ndarray],
                     cond_cols: set[str]) -> tuple[list[tuple[int, int]], list[str]]:
    """Pairs of units from different samples alike in every kept stratum column, dropping
    columns (condition-like last, the one that frees most pairs first) until
    ``MIN_REF_PAIRS`` pairs remain or none is left."""
    idx = [i for i, u in enumerate(units) if u["n_cells"] >= MIN_LIBRARY_CELLS]
    pairs_all = [(i, j) for i, j in itertools.combinations(idx, 2)
                 if units[i]["sample"] != units[j]["sample"]]
    if not pairs_all:
        return [], []
    a = np.array([p[0] for p in pairs_all])
    b = np.array([p[1] for p in pairs_all])
    same = {c: (v[a] == v[b]) & (v[a] >= 0) for c, v in strata.items()}

    def alike(cols: list[str]) -> np.ndarray:
        out = np.ones(len(pairs_all), bool)
        for c in cols:
            out &= same[c]
        return out

    kept = list(strata)
    while int(alike(kept).sum()) < MIN_REF_PAIRS and kept:
        pool = [c for c in kept if c not in cond_cols] or kept
        best = max(pool, key=lambda c: int(alike([k for k in kept if k != c]).sum()))
        kept.remove(best)
    return [p for p, ok in zip(pairs_all, alike(kept)) if ok], kept


def library_concordance(adata, cells: dict | None = None, *, sample_column: str | None = None,
                        library_column: str | None = None, celltype_column: str | None = None) -> dict:
    """Per sample with two or more libraries: each library's cells, median UMIs and genes,
    the most different library pair's cell-type distance against the calibrated cut, the
    cell types that differ most, and the depth gap. ``cells`` is ``qc_per_cell()``'s
    output (None: depth is not judged). Raises LibraryConcordanceError for a named
    column that does not exist."""
    obs = adata.obs
    for col in (sample_column, library_column, celltype_column):
        if col and col not in obs.columns:
            raise LibraryConcordanceError(f"Column '{col}' not found.")

    sample = sample_column
    if not sample:
        sample, why = qc_groups.sample_grouping(adata, max_groups=None)
        if why != "donor":
            sample = None
    if not sample:
        return _none("No donor or sample column found, so there are no samples to compare "
                     "libraries within.")
    s_names, s_codes = qc_groups.group_codes(obs[sample])
    s_codes = np.asarray(s_codes)

    # The library column: the one named, else the first candidate nesting in the samples.
    tried = [(library_column, "requested")] if library_column else _library_candidates(adata, sample)
    lib = None
    for col, source in tried:
        if col == sample:
            continue
        coded = _codes_of(adata, col, source)
        if coded is None or len(coded[0]) <= len(s_names):
            continue
        l_names, l_codes = coded
        owner = _nesting(l_codes, s_codes, len(l_names))
        if owner is None or np.bincount(owner[owner >= 0], minlength=len(s_names)).max() < 2:
            continue
        lib = (col, source, l_names, l_codes, owner)
        break
    if lib is None:
        if library_column:
            return _none(f"'{library_column}' does not nest in '{sample}' with two or more of its "
                         f"values in one sample (each value must lie within one '{sample}').",
                         sample=sample, library=library_column, source="requested")
        return _none(f"No column of libraries nested in '{sample}' with two or more libraries in one "
                     "sample: each sample was sequenced once, or libraries pool several samples.",
                     sample=sample)
    l_col, l_source, l_names, l_codes, owner = lib
    n_lib = len(l_names)

    celltype = celltype_column
    if not celltype:
        from scview.core.de_composition import default_column
        celltype = default_column(adata)
    if celltype in (sample, l_col):
        celltype = None

    # What splits a sample: one value per library, differing between one sample's libraries.
    per_lib = _per_library_columns(obs, l_codes, n_lib, {sample, l_col, celltype or ""},
                                   MAX_SPLIT_VALUES)
    split_by = []
    for name, per in per_lib.items():
        f = pd.DataFrame({"s": owner, "v": per})
        f = f[(f["s"] >= 0) & (f["v"] >= 0)]
        if (f.groupby("s")["v"].nunique() > 1).any():
            split_by.append(name)

    # Units: a sample's libraries alike in every split column.
    unit_key = [tuple([int(owner[i])] + [int(per_lib[c][i]) for c in split_by]) for i in range(n_lib)]
    keys = sorted({k for k in unit_key if k[0] >= 0})
    where = {k: u for u, k in enumerate(keys)}
    unit_of_lib = np.array([where.get(k, -1) for k in unit_key])
    if np.bincount(unit_of_lib[unit_of_lib >= 0], minlength=len(keys)).max() < 2:
        return _none(f"Each sample's libraries differ in {', '.join(repr(c) for c in split_by)}, so no two "
                     "libraries of one sample are alike enough to compare.",
                     sample=sample, library=l_col, source=l_source)
    split_names = {c: _value_names(obs, c) for c in split_by}
    has_lib = l_codes >= 0
    unit_cell = np.full(len(l_codes), -1)
    unit_cell[has_lib] = unit_of_lib[l_codes[has_lib]]

    t_codes, t_names = None, []
    if celltype:
        t_names, t_codes = qc_groups.group_codes(obs[celltype])
        t_codes = np.asarray(t_codes)
    total = np.asarray(cells["total_counts"], float) if cells and cells.get("total_counts") is not None else None
    genes = np.asarray(cells["n_genes"], float) if cells and cells.get("n_genes") is not None else None

    lib_members = qc_groups.members(np.where(has_lib, l_codes, -1), n_lib)
    unit_members = qc_groups.members(unit_cell, len(keys))

    def table(m: np.ndarray) -> np.ndarray:
        c = t_codes[m]
        return np.bincount(c[c >= 0], minlength=len(t_names))

    units = []
    for u, key in enumerate(keys):
        split = {c: split_names[c][key[1 + j]] for j, c in enumerate(split_by) if key[1 + j] >= 0}
        name = s_names[key[0]] + "".join(f" · {v}" for v in split.values())
        m = unit_members[u]
        units.append({"key": key, "sample": s_names[key[0]], "split": split, "name": name,
                      "n_cells": int(m.size),
                      "comp": _shares(table(m)[None, :])[0] if t_codes is not None else None,
                      "depth": _median(total, m) if total is not None else None})

    # Reference: distances between different samples alike in their sample-level labels.
    strata_cols = _per_library_columns(obs, unit_cell, len(keys), {sample, l_col, celltype or ""},
                                       MAX_STRATUM_VALUES)
    cond_cols = {c for c in strata_cols if CONDITION_RE.search(c)} | set(split_by)
    pairs, matched = _reference_pairs(units, strata_cols, cond_cols)
    comp_ref = [_tvd(units[i]["comp"], units[j]["comp"]) for i, j in pairs] if t_codes is not None else []
    depth_ref = [r for r in (_log2_ratio(units[i]["depth"], units[j]["depth"]) for i, j in pairs)
                 if r is not None]
    calibrated = len(pairs) >= MIN_REF_PAIRS
    if calibrated:
        cut = max(float(np.quantile(comp_ref, QUANTILE)), FLOOR) if comp_ref else None
        depth_cut = (max(float(np.quantile(depth_ref, QUANTILE)), DEPTH_FLOOR)
                     if len(depth_ref) >= MIN_REF_PAIRS else DEPTH_FLOOR)
        basis = "matched" if matched else "all"
        alike = f"alike in {', '.join(matched)}" if matched else "of any kind"
        note = (f"Each pair is judged against {len(pairs)} distances between different samples {alike}: "
                f"flagged above their {QUANTILE:.0%} quantile (at least {FLOOR:.2f}).")
    else:
        cut, depth_cut, basis = (RULE_CUT if t_codes is not None else None), DEPTH_FLOOR, "rule"
        note = (f"Too few samples to calibrate ({len(pairs)} pairs of different samples, "
                f"{MIN_REF_PAIRS} needed): a fixed cut of {RULE_CUT:.2f} was used.")
    if t_codes is None:
        note += " No cell-type or cluster column, so only depth was compared."

    rows = []
    for u, unit in enumerate(units):
        libs = [i for i in range(n_lib) if unit_of_lib[i] == u]
        if len(libs) < 2:
            continue
        lib_rows = []
        for i in libs:
            m = lib_members[i]
            lib_rows.append({"name": l_names[i], "n_cells": int(m.size),
                             "median_counts": _median(total, m) if total is not None else None,
                             "median_genes": _median(genes, m) if genes is not None else None,
                             "too_small": int(m.size) < MIN_LIBRARY_CELLS})
        judged = [(i, r) for i, r in zip(libs, lib_rows) if not r["too_small"]]
        row = {"name": unit["name"], "sample": unit["sample"], "split": unit["split"],
               "libraries": lib_rows, "flagged": False, "flags": [], "pair": None, "message": None}
        if len(judged) < 2:
            row["status"] = "too_small"
            row["message"] = (f"{unit['name']}: {len(libs)} libraries, but fewer than two with "
                              f"{MIN_LIBRARY_CELLS} cells, so they were not compared.")
            rows.append(row)
            continue
        row["status"] = "checked"
        best = None
        for (i, ri), (j, rj) in itertools.combinations(judged, 2):
            d = p = None
            differing = []
            if t_codes is not None:
                ta, tb = table(lib_members[i]), table(lib_members[j])
                sa, sb = _shares(ta[None, :])[0], _shares(tb[None, :])[0]
                d, p = _tvd(sa, sb), _chi2_p(ta, tb)
                order = np.argsort(-np.abs(sa - sb))
                differing = [{"group": t_names[k], "share_a": round(float(sa[k]), 4),
                              "share_b": round(float(sb[k]), 4)}
                             for k in order[:MAX_NAMED] if abs(sa[k] - sb[k]) * 100 >= MIN_SHARE_POINTS]
            ratio = _log2_ratio(ri["median_counts"], rj["median_counts"])
            comp_hit = d is not None and cut is not None and d > cut and p < MAX_P
            depth_hit = ratio is not None and ratio > depth_cut
            pair = {"a": ri["name"], "b": rj["name"],
                    "distance": None if d is None else round(d, 4), "p": p,
                    "differing": differing,
                    "depth_log2_ratio": None if ratio is None else round(ratio, 3),
                    "flags": [f for f, hit in (("composition", comp_hit), ("depth", depth_hit)) if hit]}
            if comp_ref and d is not None:
                pair["above_share"] = round(float(np.mean(np.asarray(comp_ref) < d)), 3)
            rank = (len(pair["flags"]), d if d is not None else 0.0, ratio or 0.0)
            if best is None or rank > best[0]:
                best = (rank, pair, ri, rj)
        _, pair, ri, rj = best
        row["pair"] = pair
        row["flags"] = pair["flags"]
        row["flagged"] = bool(pair["flags"])
        if row["flagged"]:
            row["message"] = _message(unit["name"], pair, ri, rj, cut, depth_cut, len(comp_ref), basis, matched)
        rows.append(row)

    flagged = [r for r in rows if r["flagged"]]
    checked = [r for r in rows if r["status"] == "checked"]
    if len(checked) >= MIN_DESIGN_SAMPLES and len(flagged) > DESIGN_SHARE * len(checked):
        # Most samples' libraries differ: the design (sorted fractions, conditions per
        # library), not a fault in one sample. Said once, nothing flagged.
        note += (f" {len(flagged)} of {len(checked)} samples' libraries differ, so the libraries "
                 "look different by design (sorted fractions or conditions): name the column that "
                 "tells them apart for a check within it.")
        for r in flagged:
            r.update(flagged=False, flags=[], message=None, status="by_design")
        flagged = []
    return {
        "status": "checked", "reason": None,
        "sample_column": sample, "library_column": l_col, "library_source": l_source,
        "celltype_column": celltype, "split_by": split_by,
        "n_libraries": n_lib, "n_samples": len(s_names),
        "samples": rows, "n_checked": sum(r["status"] == "checked" for r in rows),
        "n_flagged": len(flagged), "flagged": [r["name"] for r in flagged],
        "reference": {"basis": basis, "matched_on": matched, "n_pairs": len(pairs),
                      "cut": None if cut is None else round(cut, 4),
                      "depth_cut_log2": round(depth_cut, 3),
                      "quantile": QUANTILE, "floor": FLOOR, "rule_cut": RULE_CUT, "max_p": MAX_P,
                      "min_library_cells": MIN_LIBRARY_CELLS},
        "note": note,
    }


def _message(unit: str, pair: dict, ri: dict, rj: dict, cut: float | None, depth_cut: float,
             n_ref: int, basis: str, matched: list[str]) -> str:
    a, b = pair["a"], pair["b"]
    parts = [f"{unit}: libraries {a} ({ri['n_cells']:,} cells) and {b} ({rj['n_cells']:,}) disagree"]
    if "composition" in pair["flags"]:
        if basis == "rule":
            against = f"over the fixed cut of {cut:.2f}"
        else:
            alike = f" alike in {', '.join(matched)}" if matched else ""
            share = pair.get("above_share")
            against = (f"larger than {share:.0%} of {n_ref} distances between different samples{alike}, "
                       f"cut {cut:.2f}" if share is not None else f"cut {cut:.2f}")
        parts.append(f" in cell types: distance {pair['distance']:.2f} ({against})")
        if pair["differing"]:
            shown = ", ".join(f"{g['group']} {_pct(g['share_a'])} vs {_pct(g['share_b'])}" for g in pair["differing"])
            parts.append(f"; most different: {shown}")
    if ri["median_counts"] and rj["median_counts"]:
        depth = (f"median UMIs {ri['median_counts']:,.0f} vs {rj['median_counts']:,.0f}"
                 + (f", genes {ri['median_genes']:,.0f} vs {rj['median_genes']:,.0f}"
                    if ri["median_genes"] and rj["median_genes"] else ""))
        if "depth" in pair["flags"]:
            parts.append(f"{'; ' if len(parts) > 1 else ' in depth: '}{depth} "
                         f"(over {2 ** depth_cut:.1f}-fold, more than between comparable samples)")
        else:
            parts.append(f"; depth: {depth}")
    parts.append(". One library may hold another sample's cells, or its capture failed: check it "
                 "before pooling the two.")
    return "".join(parts)


def report_lines(result: dict) -> tuple[str | None, list[str]]:
    """(fact, concerns) for the QC report: one line on what was checked, and each flagged
    sample's message."""
    if result.get("status") != "checked" or not result.get("samples"):
        return None, []
    n, k = result["n_checked"], result["n_flagged"]
    lib, samp = result["library_column"], result["sample_column"]
    split = f", compared within one {', '.join(result['split_by'])}" if result["split_by"] else ""
    line = (f"Libraries per sample ('{lib}' within '{samp}'{split}): {n} "
            f"{'sample has' if n == 1 else 'samples have'} two or more libraries; "
            + (f"{k} with libraries that disagree ({', '.join(result['flagged'][:4])})."
               if k else "their libraries agree in cell types and depth."))
    return line, [r["message"] for r in result["samples"] if r["flagged"]]


def grounding_lines(result: dict) -> tuple[list[str], str]:
    """The co-pilot's lines and one-line detail; nothing when no sample has two libraries."""
    line, concerns = report_lines(result)
    if line is None:
        return [], ""
    lines = [f"\n## Library concordance (two or more libraries of one sample) [{TAG}]", f"- {line} [{TAG}]"]
    lines += [f"- {c} [{TAG}]" for c in concerns]
    if result.get("note"):
        lines.append(f"- {result['note']}")
    detail = (f"{result['n_flagged']} of {result['n_checked']} multi-library samples disagree"
              + (f" ({', '.join(result['flagged'][:3])})" if result["flagged"] else ""))
    return lines, detail
