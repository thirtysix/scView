"""Pooled CRISPR screens: each cell's guide and target (N12).

A pooled screen (Perturb-seq, ECCITE-seq, CROP-seq) gives each cell one or more
guide RNAs; the guide says which gene was targeted. Three sources, in order:

- Cell Ranger's own calls (``crispr_analysis/protospacer_calls_per_cell.csv``),
  read at import into obs ``cellranger_guide_call`` ("g1|g2" per cell, "" for
  none) when the file comes with the matrix (``attach_calls``);
- the guide counts: "CRISPR Guide Capture" features of a 10x feature-barcoding
  run, kept at import in ``obsm["guide_counts"]`` (a DataFrame, cells x guides,
  as ``hto_counts`` keeps CellPlex tags). A guide is detected in a cell with at
  least ``min_umi`` UMIs (pertpy ``assign_by_threshold``); with two or more,
  the top one is kept when it has at least ``min_ratio`` times the second
  (else the cell has several guides);
- an obs column the user names (a public .h5ad ships ``perturbation``,
  ``gene``, ``guide_ID``, ...).

A guide's target is its name without the guide suffix (``CD86g1`` -> ``CD86``,
``IFNGR1_2`` -> ``IFNGR1``; kept whole when only the whole name is a gene),
editable per guide. Non-targeting guides (``NT``, ``non-targeting``,
``control``, ``safe``, ``scramble``, ``intergenic``) all map to the one
control label the user picks.

Writes obs ``guide_call`` (the guide, "g1|g2" for several, "none"),
``perturbation`` (the target, the control label, "multiple" when the guides hit
different genes, "none") and ``n_guides``; a column of the same name that
scView did not write is never replaced (the new one gets a ``_scview`` suffix).
The record (source, parameters, mapping, counts) is in ``uns["scview_crispr"]``.
"""

from __future__ import annotations

import json
import logging
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import scipy.sparse as sp

logger = logging.getLogger(__name__)

GUIDE_TYPE = "CRISPR Guide Capture"
GUIDE_KEY = "guide_counts"
UNS_KEY = "scview_crispr"  # JSON string: guide sources, the saved assignment, Cell Ranger file
CR_CALL = "cellranger_guide_call"
CR_N = "cellranger_n_guides"
# obsm keys other tools leave guide counts under (DataFrames with guide names).
FOREIGN_KEYS = ("gdo", "GDO", "guides", "crispr", "sgRNA", "sgrna_counts")
MAX_GUIDES = 2000  # kept as a dense table: more is a genome-wide library (use the calls)
TAG = "result:mixscape"

SOURCE_CELLRANGER, SOURCE_COUNTS, SOURCE_COLUMN = "cellranger", "counts", "column"
SOURCES = (SOURCE_CELLRANGER, SOURCE_COUNTS, SOURCE_COLUMN)
SOURCE_LABELS = {SOURCE_CELLRANGER: "Cell Ranger guide calls", SOURCE_COUNTS: "guide counts",
                 SOURCE_COLUMN: "an obs column"}
MIN_UMI = 5
MIN_RATIO = 2.0
COLUMNS = ("guide_call", "perturbation", "n_guides")
NONE, MULTIPLE = "none", "multiple"
MAX_TARGETS = 2000
MAX_LEVELS = 5000  # an obs column with more values is not a guide / target column

# Guide suffixes: "_2", "-1", ".3", "_g1", "_sgRNA3" (any case), or "g1" / "sg2" right after
# the gene (lowercase only, so a gene ending in G1, CCNG1, keeps its name).
_SUFFIX_SEP = re.compile(r"[_.\-](?:sg|g|guide|grna|sgrna)?\d+[a-z]?$", re.IGNORECASE)
_SUFFIX_G = re.compile(r"(?:sg|g)\d+[a-z]?$")
_CONTROL = re.compile(r"^(nt|ntc|non[-_ ]?target(?:ing|ed)?(?:[-_ ]?control)?|nontarget(?:ing)?|control|ctrl|"
                      r"safe(?:[-_ ]?target(?:ing)?)?|scr(?:amble[d]?)?|neg(?:ative)?[-_ ]?control|"
                      r"no[-_ ]?target|intergenic)$", re.IGNORECASE)
_NONE_VALUE = re.compile(r"^(|none|nan|na|n/a|negative|unassigned|no[-_ ]?guide|blank)$", re.IGNORECASE)
_MULTI_VALUE = re.compile(r"doublet|multiplet|multiple|^multi$", re.IGNORECASE)
_MULTI_SEP = re.compile(r"[|;,+]")
# obs column names that hold guides or targets in public screens (scPerturb, Replogle, Norman, ...).
_COLUMN_HINT = re.compile(r"^(perturbation|perturbation_name|perturbations|gene|genes|target|targets|"
                          r"target_gene|gene_target|gene_targets|guide|guides|guide_id|guide_ids|guide_identity|"
                          r"sgrna|sgrnas|sgrna_id|grna|gene_symbol|ko|knockout|crispr|feature_call)$",
                          re.IGNORECASE)


class CrisprError(ValueError):
    """A request this cannot serve; nothing was changed. ``status`` is the HTTP code."""

    def __init__(self, message: str, status: int = 400) -> None:
        super().__init__(message)
        self.status = status


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# ---------------------------------------------------------------------------
# State (uns JSON) and the guide counts
# ---------------------------------------------------------------------------


def read_state(adata) -> dict:
    raw = adata.uns.get(UNS_KEY) if hasattr(adata, "uns") else None
    try:
        state = json.loads(raw) if isinstance(raw, str) else {}
    except ValueError:
        logger.warning("Ignoring an unreadable %s", UNS_KEY)
        state = {}
    return state if isinstance(state, dict) else {}


def _write_state(adata, state: dict) -> None:
    adata.uns[UNS_KEY] = json.dumps(state, default=str)  # a new string: write_derived saves it


def _uniq(names: list[str]) -> list[str]:
    seen: dict[str, int] = {}
    out = []
    for n in names:
        k = seen.get(n, 0)
        out.append(n if k == 0 else f"{n}-{k}")
        seen[n] = k + 1
    return out


def attach_counts(adata, counts, features: list[str], source: str = GUIDE_TYPE) -> list[str]:
    """Keep guide counts (cells x ``features``) in ``obsm[GUIDE_KEY]``; returns the names."""
    dense = counts.toarray() if sp.issparse(counts) else np.asarray(counts)
    names = _uniq([str(f) for f in features])
    adata.obsm[GUIDE_KEY] = pd.DataFrame(np.asarray(dense, dtype=np.float32), index=adata.obs_names,
                                         columns=names)
    state = read_state(adata)
    state["guides"] = {"source": source, "n_guides": len(names)}
    _write_state(adata, state)
    return names


def guide_counts_frame(adata) -> pd.DataFrame | None:
    """The guide counts (cells x guides): scView's, else a counts table another tool
    left under a known obsm name."""
    if not hasattr(adata, "obsm"):
        return None
    m = adata.obsm.get(GUIDE_KEY)
    if isinstance(m, pd.DataFrame) and m.shape[1] > 0:
        return m
    for key in FOREIGN_KEYS:
        m = adata.obsm.get(key)
        if isinstance(m, pd.DataFrame) and 0 < m.shape[1] <= MAX_GUIDES:
            v = m.to_numpy(dtype=np.float64)
            if np.isfinite(v).all() and v.min() >= 0 and np.allclose(v, np.round(v)):
                return m
    return None


def common_guides(adatas: list) -> None:
    """For a merge: every sample's guide counts on the union of the guides (0 where a
    sample has no such guide, or no guides at all), so none is lost in the join."""
    frames = [a.obsm.get(GUIDE_KEY) for a in adatas]
    if not any(isinstance(f, pd.DataFrame) for f in frames):
        return
    union = list(dict.fromkeys(str(c) for f in frames if isinstance(f, pd.DataFrame) for c in f.columns))
    if len(union) > MAX_GUIDES:
        for a in adatas:
            a.obsm.pop(GUIDE_KEY, None)
        return
    for a, f in zip(adatas, frames):
        base = f if isinstance(f, pd.DataFrame) else pd.DataFrame(index=a.obs_names)
        a.obsm[GUIDE_KEY] = base.reindex(columns=union, fill_value=0).astype(np.float32)


# ---------------------------------------------------------------------------
# Cell Ranger protospacer calls (a side file of the import)
# ---------------------------------------------------------------------------

CALLS_HEADER = ("cell_barcode", "num_features", "feature_call", "num_umis")


def looks_like_calls(first_line: str) -> bool:
    cols = [c.strip().strip('"').lower() for c in first_line.strip().split(",")]
    return tuple(cols[:4]) == CALLS_HEADER


def read_calls(path: str | Path) -> pd.DataFrame:
    """``protospacer_calls_per_cell.csv``: barcode -> "g1|g2" call and guide count."""
    df = pd.read_csv(path, dtype=str)
    df.columns = [c.strip().lower() for c in df.columns]
    missing = [c for c in CALLS_HEADER[:3] if c not in df.columns]
    if missing:
        raise CrisprError(f"{Path(path).name} has no {', '.join(missing)} column.")
    df = df.dropna(subset=["cell_barcode"]).drop_duplicates("cell_barcode")
    return df.set_index("cell_barcode")


def attach_calls(adata, path: str | Path, *, file_name: str, sample_column: str | None = None,
                 sample: str | None = None) -> int:
    """Write Cell Ranger's call per cell into obs ``cellranger_guide_call`` /
    ``cellranger_n_guides``; in a merge, only ``sample``'s cells (barcodes carry
    the ``:<sample>`` suffix there). Returns how many cells matched."""
    calls = read_calls(path)
    names = pd.Index(adata.obs_names.astype(str))
    rows = np.ones(len(names), dtype=bool)
    keys = names
    if sample_column and sample is not None and sample_column in adata.obs.columns:
        rows = (adata.obs[sample_column].astype(str) == str(sample)).to_numpy()
        keys = pd.Index([n.rsplit(":", 1)[0] if n.endswith(f":{sample}") else n for n in names])
    call = adata.obs[CR_CALL].astype(object).to_numpy() if CR_CALL in adata.obs.columns \
        else np.full(len(names), "", dtype=object)
    n_guides = adata.obs[CR_N].to_numpy(dtype=np.int32) if CR_N in adata.obs.columns \
        else np.zeros(len(names), dtype=np.int32)
    pos = calls.index.get_indexer(keys)
    hit = (pos >= 0) & rows
    feature = calls["feature_call"].fillna("").astype(str).to_numpy()
    num = pd.to_numeric(calls.get("num_features"), errors="coerce").fillna(0).astype(np.int32).to_numpy() \
        if "num_features" in calls.columns else np.array([len(f.split("|")) for f in feature], np.int32)
    call[hit] = feature[pos[hit]]
    n_guides[hit] = num[pos[hit]]
    adata.obs[CR_CALL] = pd.Series(call, index=adata.obs_names).fillna("").astype(str)
    adata.obs[CR_N] = n_guides
    state = read_state(adata)
    state.setdefault("cellranger", []).append({"file": file_name, "sample": sample, "n_cells": int(hit.sum()),
                                               "n_rows": int(len(calls))})
    _write_state(adata, state)
    return int(hit.sum())


# ---------------------------------------------------------------------------
# Guide -> target
# ---------------------------------------------------------------------------


def strip_guide(name: str, genes: set[str] | None = None) -> str:
    """``CD86g1`` -> ``CD86``, ``IFNGR1_2`` -> ``IFNGR1``, ``non-targeting_3`` ->
    ``non-targeting``. With ``genes`` (the dataset's gene names): a name that is
    a gene while its stripped form is not stays whole."""
    name = str(name).strip()
    stripped = _SUFFIX_SEP.sub("", name)
    if stripped == name:
        stripped = _SUFFIX_G.sub("", name)
    stripped = stripped or name
    if genes and name in genes and stripped not in genes:
        return name
    return stripped


def is_control(label: str) -> bool:
    return bool(_CONTROL.match(str(label).strip()))


def default_mapping(guides: list[str], genes: set[str] | None = None) -> dict[str, str]:
    return {g: strip_guide(g, genes) for g in guides}


def suggest_control(targets) -> str | None:
    """The control label among the targets: the most frequent control-like one."""
    hits = [t for t in targets if is_control(t)]
    return hits[0] if hits else None


# ---------------------------------------------------------------------------
# Assignment
# ---------------------------------------------------------------------------


def calls_from_counts(counts: np.ndarray, guides: list[str], min_umi: float = MIN_UMI,
                      min_ratio: float = MIN_RATIO) -> list[list[str]]:
    """Each cell's guides: those with >= ``min_umi`` UMIs; of two or more, the top
    one alone when it has >= ``min_ratio`` times the second (0: never)."""
    X = np.asarray(counts, dtype=np.float64)
    out: list[list[str]] = []
    names = np.asarray(guides, dtype=object)
    for row in X:
        hit = np.flatnonzero(row >= min_umi)
        if hit.size > 1:
            order = hit[np.argsort(-row[hit], kind="stable")]
            if min_ratio and row[order[0]] >= min_ratio * row[order[1]]:
                hit = order[:1]
            else:
                hit = order
        out.append([str(g) for g in names[hit]])
    return out


def _split_value(v: Any) -> list[str]:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return []
    s = str(v).strip()
    if _NONE_VALUE.match(s):
        return []
    parts = [p.strip() for p in _MULTI_SEP.split(s) if p.strip()]
    return parts


def calls_from_strings(values) -> list[list[str]]:
    """'g1|g2' / 'g1' / '' per cell (Cell Ranger, or a column) -> guide lists. A
    doublet / multiplet marker counts as several guides of unknown target."""
    out = []
    for v in values:
        parts = _split_value(v)
        if len(parts) == 1 and _MULTI_VALUE.search(parts[0]):
            parts = [MULTIPLE, MULTIPLE + "_"]
        out.append(parts)
    return out


def assign_calls(calls: list[list[str]], mapping: dict[str, str], control: str | None
                 ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(guide_call, perturbation, n_guides)`` per cell from its guide list."""
    n = len(calls)
    guide_call = np.empty(n, dtype=object)
    pert = np.empty(n, dtype=object)
    n_guides = np.zeros(n, dtype=np.int32)
    for i, gs in enumerate(calls):
        n_guides[i] = len(gs)
        if not gs:
            guide_call[i], pert[i] = NONE, NONE
            continue
        guide_call[i] = "|".join(gs)
        targets = set()
        for g in gs:
            t = mapping.get(g, g)
            if g.startswith(MULTIPLE):
                t = None
            elif control is not None and (t == control or is_control(t)):
                t = control
            targets.add(t)
        pert[i] = MULTIPLE if len(targets) != 1 or None in targets else targets.pop()
    return guide_call, pert, n_guides


def _column_values(adata, column: str) -> list[str]:
    col = adata.obs[column]
    return [str(v) for v in pd.unique(col.dropna().astype(str))]


def candidate_columns(adata, named_only: bool = False) -> list[dict[str, Any]]:
    """obs columns that look like guide / target labels: a known name, 2..MAX_LEVELS
    values, at least one control-like (strongest first). ``named_only``: only the
    columns with a known name are read (cheap, for the dataset info)."""
    out = []
    obs = adata.obs
    ours = set((read_state(adata).get("call") or {}).get("columns", {}).values())
    for c in obs.columns:
        if c in ours or c in (CR_CALL, CR_N) or str(c).startswith("mixscape_"):
            continue
        hint = bool(_COLUMN_HINT.match(str(c)))
        if named_only and not hint:
            continue
        s = obs[c]
        if pd.api.types.is_numeric_dtype(s) and not isinstance(s.dtype, pd.CategoricalDtype):
            continue
        try:
            n = s.nunique(dropna=True)
        except TypeError:
            continue
        if not 2 <= n <= MAX_LEVELS:
            continue
        values = _column_values(adata, c)
        stripped = {strip_guide(v) for v in values}
        control = suggest_control(sorted(stripped))
        if hint or control is not None:
            out.append({"column": str(c), "n_values": int(n), "control": control, "named_like": hint})
    out.sort(key=lambda d: (not (d["named_like"] and d["control"]), not d["control"], not d["named_like"]))
    return out


def _source_calls(adata, source: str, column: str | None, min_umi: float, min_ratio: float
                  ) -> tuple[list[list[str]], list[str], dict[str, Any]]:
    """The guide list per cell and the guide names, for ``source``."""
    if source == SOURCE_COUNTS:
        frame = guide_counts_frame(adata)
        if frame is None:
            raise CrisprError("This dataset has no guide counts (CRISPR Guide Capture features).")
        if min_umi <= 0:
            raise CrisprError("The UMI threshold must be above 0.")
        if min_ratio < 0:
            raise CrisprError("The fold rule must be 0 (off) or more.")
        guides = [str(c) for c in frame.columns]
        calls = calls_from_counts(frame.to_numpy(), guides, min_umi, min_ratio)
        return calls, guides, {"min_umi": min_umi, "min_ratio": min_ratio}
    if source == SOURCE_CELLRANGER:
        if CR_CALL not in adata.obs.columns:
            raise CrisprError("No Cell Ranger guide calls were imported with this dataset "
                              "(crispr_analysis/protospacer_calls_per_cell.csv).")
        calls = calls_from_strings(adata.obs[CR_CALL].to_numpy())
    elif source == SOURCE_COLUMN:
        if not column or column not in adata.obs.columns:
            raise CrisprError(f"Column '{column}' not found.")
        s = adata.obs[column]
        if s.nunique(dropna=True) > MAX_LEVELS:
            raise CrisprError(f"'{column}' has more than {MAX_LEVELS:,} values: not a guide or target column.")
        calls = calls_from_strings(s.astype(object).to_numpy())
    else:
        raise CrisprError(f"Unknown source '{source}' (use one of {', '.join(SOURCES)}).")
    guides = sorted({g for gs in calls for g in gs if not g.startswith(MULTIPLE)})
    return calls, guides, {"column": column} if source == SOURCE_COLUMN else {}


def default_source(adata) -> tuple[str | None, str | None]:
    """``(source, column)``: Cell Ranger calls, else guide counts, else the best column."""
    if CR_CALL in adata.obs.columns:
        return SOURCE_CELLRANGER, None
    if guide_counts_frame(adata) is not None:
        return SOURCE_COUNTS, None
    cols = candidate_columns(adata)
    if cols:
        return SOURCE_COLUMN, cols[0]["column"]
    return None, None


def _out_names(adata, state: dict) -> dict[str, str]:
    """Output column per field: the plain name unless a column of that name was not
    written by scView (then ``<name>_scview``)."""
    ours = set((state.get("call") or {}).get("columns", {}).values())
    out = {}
    for c in COLUMNS:
        out[c] = c if (c not in adata.obs.columns or c in ours) else f"{c}_scview"
    return out


def preview(adata, source: str | None = None, column: str | None = None, *, min_umi: float = MIN_UMI,
            min_ratio: float = MIN_RATIO, mapping: dict[str, str] | None = None,
            control: str | None = None) -> dict[str, Any]:
    """What ``assign`` would write (nothing is changed): mapping, control, counts."""
    if source is None:
        source, column = default_source(adata)
        if source is None:
            raise CrisprError("No guides found: import the guide counts (CRISPR Guide Capture), Cell Ranger's "
                              "protospacer calls, or name an obs column with each cell's guide or target.", 422)
    calls, guides, params = _source_calls(adata, source, column, min_umi, min_ratio)
    genes = set(map(str, adata.raw.var_names)) if adata.raw is not None else set(map(str, adata.var_names))
    auto = default_mapping(guides, genes)
    full = {**auto, **{str(k): str(v).strip() for k, v in (mapping or {}).items() if str(v).strip()}}
    targets = sorted(set(full.values()))
    if len(targets) > MAX_TARGETS:
        raise CrisprError(f"{len(targets):,} targets: more than {MAX_TARGETS:,}.")
    control = control or suggest_control(targets)
    if control is not None and control not in targets:
        raise CrisprError(f"The control '{control}' is not among the targets.")
    guide_call, pert, n_guides = assign_calls(calls, full, control)
    s = pd.Series(pert)
    vc = s.value_counts()
    per_target = [{"target": str(k), "n_cells": int(v), "control": str(k) == control,
                   "guides": sorted(g for g, t in full.items() if t == k or (control and str(k) == control
                                                                            and is_control(t)))}
                  for k, v in vc.items() if k not in (NONE, MULTIPLE)]
    return {
        "source": source, "source_label": SOURCE_LABELS[source], "column": column, "params": params,
        "n_cells": int(len(pert)), "guides": guides, "mapping": full, "auto_mapping": auto,
        "control": control, "targets": [t for t in targets if t != control and not is_control(t)],
        "counts": {"assigned": int(len(pert) - vc.get(NONE, 0) - vc.get(MULTIPLE, 0)),
                   "control": int(vc.get(control, 0)) if control else 0,
                   "multiple": int(vc.get(MULTIPLE, 0)), "none": int(vc.get(NONE, 0))},
        "per_target": per_target,
        "_columns": (guide_call, pert, n_guides),
    }


def assign(adata, source: str | None = None, column: str | None = None, *, min_umi: float = MIN_UMI,
           min_ratio: float = MIN_RATIO, mapping: dict[str, str] | None = None,
           control: str | None = None) -> dict[str, Any]:
    """Write ``guide_call`` / ``perturbation`` / ``n_guides`` and the record. Checked
    and computed before anything changes. Returns ``summary``."""
    from scview.core import provenance
    from scview.core.obs_writes import forget_column_results

    p = preview(adata, source, column, min_umi=min_umi, min_ratio=min_ratio, mapping=mapping, control=control)
    if p["control"] is None:
        raise CrisprError("Pick the non-targeting (control) label: none of the targets looks like one.")
    guide_call, pert, n_guides = p.pop("_columns")
    state = read_state(adata)
    names = _out_names(adata, state)
    obs = adata.obs
    obs[names["guide_call"]] = pd.Categorical(guide_call)
    order = [p["control"]] + sorted(t for t in set(pert) if t not in (p["control"], NONE, MULTIPLE)) + \
        [t for t in (MULTIPLE, NONE) if t in set(pert)]
    obs[names["perturbation"]] = pd.Categorical(pert, categories=order)
    obs[names["n_guides"]] = n_guides
    for c in names.values():
        forget_column_results(adata, c)
    record = {k: p[k] for k in ("source", "column", "params", "control", "counts")}
    record.update({"mapping": {g: t for g, t in p["mapping"].items() if p["auto_mapping"].get(g) != t},
                   "n_guides": len(p["guides"]), "n_targets": len(p["targets"]), "columns": names,
                   "n_cells": int(adata.n_obs), "when": _now()})
    state["call"] = record
    _write_state(adata, state)
    try:
        provenance.record_step(adata, step="crispr_assign", tool=f"scview crispr ({SOURCE_LABELS[p['source']]})",
                               params={**{k: record[k] for k in ("source", "column", "params", "control",
                                                                 "mapping", "columns", "counts")}})
    except Exception as e:  # provenance must never block the result
        logger.warning("Could not record the guide assignment in provenance: %s", e)
    return summary(adata)


# ---------------------------------------------------------------------------
# Summary, dataset flag, co-pilot
# ---------------------------------------------------------------------------


def saved_call(adata) -> dict[str, Any] | None:
    """The saved assignment with counts from the current cells, or None."""
    call = read_state(adata).get("call")
    if not isinstance(call, dict):
        return None
    col = (call.get("columns") or {}).get("perturbation")
    if col not in adata.obs.columns:
        return None
    s = adata.obs[col].astype(object)
    vc = s.value_counts()
    control = call.get("control")
    per = [{"target": str(k), "n_cells": int(v)} for k, v in vc.items() if k not in (NONE, MULTIPLE)]
    return {**{k: call.get(k) for k in ("source", "column", "params", "control", "mapping", "columns", "when")},
            "source_label": SOURCE_LABELS.get(call.get("source"), call.get("source")),
            "n_cells_at_call": call.get("n_cells"),
            "counts": {"assigned": int(len(s) - vc.get(NONE, 0) - vc.get(MULTIPLE, 0)),
                       "control": int(vc.get(control, 0)), "multiple": int(vc.get(MULTIPLE, 0)),
                       "none": int(vc.get(NONE, 0))},
            "n_targets": sum(1 for d in per if d["target"] != control),
            "per_target": per}


def summary(adata) -> dict[str, Any]:
    """What the Perturbation tab shows before Mixscape: the sources there are, the
    suggested one, and the saved assignment."""
    frame = guide_counts_frame(adata)
    state = read_state(adata)
    source, column = default_source(adata)
    return {
        "n_cells": int(adata.n_obs),
        "guides": [str(c) for c in frame.columns] if frame is not None else [],
        "has_counts": frame is not None,
        "has_cellranger": CR_CALL in adata.obs.columns,
        "cellranger_files": state.get("cellranger") or [],
        "columns": candidate_columns(adata),
        "obs_columns": [str(c) for c in adata.obs.columns
                        if not pd.api.types.is_float_dtype(adata.obs[c])],
        "suggested": {"source": source, "column": column},
        "defaults": {"min_umi": MIN_UMI, "min_ratio": MIN_RATIO},
        "sources": [{"id": s, "label": SOURCE_LABELS[s]} for s in SOURCES],
        "call": saved_call(adata),
    }


def available(adata) -> dict[str, Any] | None:
    """For the dataset info: whether the Perturbation tab has anything to work on
    (guide counts, Cell Ranger calls, a saved call, or a guide-like column)."""
    try:
        frame = guide_counts_frame(adata)
        call = read_state(adata).get("call")
        cr = CR_CALL in adata.obs.columns
        cols = [] if (frame is not None or cr or call) else candidate_columns(adata, named_only=True)
        cols = [c for c in cols if c["control"]]
        if frame is None and not cr and not call and not cols:
            return None
        return {"n_guides": 0 if frame is None else int(frame.shape[1]), "cellranger": cr,
                "assigned": isinstance(call, dict), "columns": [c["column"] for c in cols[:5]]}
    except Exception as e:  # never break the dataset info
        logger.debug("crispr.available failed: %s", e)
        return None


def grounding_lines(adata) -> tuple[list[str], str]:
    """The co-pilot's fact on the guide assignment (the Mixscape line is added by
    ``core/mixscape.py``). ``(lines, detail)``."""
    call = saved_call(adata)
    if call is None:
        return [], ""
    c = call["counts"]
    n = c["assigned"] + c["multiple"] + c["none"]
    top = sorted((d for d in call["per_target"] if d["target"] != call["control"]), key=lambda d: -d["n_cells"])
    per = ", ".join(f"{d['target']} {d['n_cells']:,}" for d in top[:12])
    detail = (f"{call['n_targets']} targets from {call['source_label']}; {c['assigned']:,} of {n:,} cells "
              f"assigned ({c['control']:,} {call['control']}), {c['multiple']:,} multiple, {c['none']:,} none")
    line = (f"- CRISPR guides: {detail}; cells per target: {per} (column '{call['columns']['perturbation']}') "
            f"[{TAG}]")
    return [line], detail
