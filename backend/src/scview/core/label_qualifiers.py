"""Marker qualifiers for cluster labels that do not tell clusters apart.

An annotator often gives two clusters the same name ("CD14+ monocytes" for the
control and the stimulated cells), or a name that says little about the cluster
("T cells / NK cells", "Unknown", "Low-quality cells"). Such a cluster gets a
short qualifier from the genes that set it apart, "CD14+ monocytes (ISG15+ IFIT3+)",
in a column beside the labels (``<target>_qualified``). The labels themselves stay
clean, so their Cell Ontology match is unchanged.

Deterministic (no LLM), from the per-cluster means and detection fractions of the
matrix the cluster markers were ranked on (log-normalised):

- a label shared by several clusters: each is compared with the others of that
  label, pooled. Up to ``MAX_PLUS`` genes clearly higher (``gene+``) and, more
  sparingly, ``MAX_MINUS`` gene the others clearly express and it does not (``gene-``),
  only in a slot the higher genes left free (two genes at most);
- a mixture, doublet or no-type label of one cluster: compared with all other
  clusters, pooled; higher genes only (a gene missing against everything else
  says nothing).

A gene must clear every cutoff to stand out; when none does, the cluster keeps its
plain label rather than a weak qualifier. The record (``uns["<target>_qualifiers"]``,
JSON) keeps each cluster's genes, what it was compared with and the cutoffs.
"""

from __future__ import annotations

import json
import logging
import re
from collections import defaultdict
from typing import Any

import numpy as np
import pandas as pd

from scview.core.gene_filters import is_ribo_mito

logger = logging.getLogger(__name__)

COLUMN_SUFFIX = "_qualified"
RECORD_SUFFIX = "_qualifiers"

MIN_CELLS = 10  # a cluster, and the cells it is compared with
MAX_PLUS = 2
MAX_MINUS = 1
# Higher: detected in at least PLUS_MIN_DETECTION of the cluster, PLUS_MIN_GAIN more of
# it than of the reference, and a mean log expression at least MIN_LOG_DIFF above the
# reference's (natural log of normalised counts + 1: ~2.7-fold).
PLUS_MIN_DETECTION = 0.5
PLUS_MIN_GAIN = 0.3
MIN_LOG_DIFF = 1.0
# Lower (same-label comparisons only): the reference detects it in at least
# MINUS_MIN_REFERENCE of its cells, the cluster in at most MINUS_MAX_DETECTION.
MINUS_MIN_REFERENCE = 0.5
MINUS_MAX_DETECTION = 0.1

CUTOFFS = {
    "min_cells": MIN_CELLS, "plus_min_detection": PLUS_MIN_DETECTION, "plus_min_gain": PLUS_MIN_GAIN,
    "min_log_diff": MIN_LOG_DIFF, "minus_min_reference": MINUS_MIN_REFERENCE,
    "minus_max_detection": MINUS_MAX_DETECTION, "max_plus": MAX_PLUS, "max_minus": MAX_MINUS,
}

# Names that tell a reader nothing: Ensembl ids, unnamed lncRNAs and clones, antisense,
# mouse Gm / Rik models, versioned ids.
_UNINFORMATIVE = re.compile(
    r"^(?:ENS[A-Z]*G\d|LINC\d|LOC\d|A[CLP]\d{6}|RP\d+-|CT[ABCD]-|Gm\d+$)|-AS\d+$|Rik$|\.\d+$", re.I)
# One-cluster labels that call for a qualifier: a mixture, doublets, or no cell type.
_MIXTURE = re.compile(r"/|\bdoublets?\b|\bmultiplets?\b", re.I)
_NO_TYPE = {"unknown", "unassigned", "low-quality cells", "low quality cells", "none", "other"}


def column_name(target: str) -> str:
    return f"{target}{COLUMN_SUFFIX}"


def record_key(target: str) -> str:
    return f"{target}{RECORD_SUFFIX}"


def informative(gene: str) -> bool:
    return not (is_ribo_mito(gene) or _UNINFORMATIVE.search(gene))


def needs_context(label: str) -> bool:
    """A one-cluster label that says little about the cluster on its own."""
    return bool(_MIXTURE.search(label)) or label.strip().lower() in _NO_TYPE


def cluster_labels(adata, groupby: str, target: str) -> dict[str, str] | None:
    """{cluster: label} when every cluster of ``groupby`` carries one label of
    ``target`` (cluster-level annotation), else None (per-cell labels)."""
    if groupby not in adata.obs.columns or target not in adata.obs.columns:
        return None
    pairs = pd.DataFrame({"g": adata.obs[groupby].astype(str).to_numpy(),
                          "l": adata.obs[target].astype(str).to_numpy()}).drop_duplicates()
    if pairs["g"].duplicated().any():
        return None
    order = ([str(c) for c in adata.obs[groupby].cat.categories] if hasattr(adata.obs[groupby], "cat")
             else sorted(pairs["g"]))
    lab = dict(zip(pairs["g"], pairs["l"]))
    return {g: lab[g] for g in order if g in lab}


def plan(labels: dict[str, str]) -> dict[str, tuple[str, list[str]]]:
    """{cluster: (versus, reference clusters)} for the clusters that get a qualifier."""
    by_label: dict[str, list[str]] = defaultdict(list)
    for g, lab in labels.items():
        by_label[lab].append(g)
    out: dict[str, tuple[str, list[str]]] = {}
    for g, lab in labels.items():
        siblings = [s for s in by_label[lab] if s != g]
        if siblings:
            out[g] = ("same_label", siblings)
        elif needs_context(lab):
            out[g] = ("all", [o for o in labels if o != g])
    return out


def _pick(score: np.ndarray, ok: np.ndarray, names: np.ndarray, n: int) -> list[str]:
    idx = np.flatnonzero(ok)
    if idx.size == 0:
        return []
    return names[idx[np.argsort(-score[idx], kind="stable")[:n]]].tolist()


def qualifiers(sums: np.ndarray, detected: np.ndarray, sizes: np.ndarray, names: np.ndarray,
               groups: list[str], todo: dict[str, tuple[str, list[str]]]) -> dict[str, dict[str, Any]]:
    """Each planned cluster's higher / lower genes from per-group sums of the log matrix
    and of detection (rows = ``groups``, as ``marker_specificity._group_sums`` gives)."""
    pos = {g: i for i, g in enumerate(groups)}
    keep = np.fromiter((informative(str(n)) for n in names), dtype=bool, count=names.size)
    out: dict[str, dict[str, Any]] = {}
    for g, (versus, ref) in todo.items():
        i, r = pos[g], [pos[x] for x in ref if x in pos]
        n_c, n_r = float(sizes[i]), float(sizes[r].sum()) if r else 0.0
        entry: dict[str, Any] = {"versus": versus, "reference": ref, "n_cells": int(n_c),
                                 "plus": [], "minus": []}
        if n_c < MIN_CELLS or n_r < MIN_CELLS:
            entry["skipped"] = "too few cells to compare"
            out[g] = entry
            continue
        mean_c, det_c = sums[i] / n_c, detected[i] / n_c
        mean_r, det_r = sums[r].sum(axis=0) / n_r, detected[r].sum(axis=0) / n_r
        diff, gain = mean_c - mean_r, det_c - det_r
        up = keep & (det_c >= PLUS_MIN_DETECTION) & (gain >= PLUS_MIN_GAIN) & (diff >= MIN_LOG_DIFF)
        entry["plus"] = _pick(gain + 1e-3 * diff, up, names, MAX_PLUS)
        if versus == "same_label" and len(entry["plus"]) < MAX_PLUS:
            down = keep & (det_r >= MINUS_MIN_REFERENCE) & (det_c <= MINUS_MAX_DETECTION) \
                & (-diff >= MIN_LOG_DIFF)
            entry["minus"] = _pick(-gain - 1e-3 * diff, down, names, min(MAX_MINUS, MAX_PLUS - len(entry["plus"])))
        out[g] = entry
    return out


def text(entry: dict[str, Any]) -> str:
    """'ISG15+ IFIT3+ CD4-', or '' when no gene stands out."""
    return " ".join([f"{x}+" for x in entry.get("plus", [])] + [f"{x}-" for x in entry.get("minus", [])])


def qualified_label(label: str, entry: dict[str, Any] | None) -> str:
    q = text(entry) if entry else ""
    return f"{label} ({q})" if q else label


def drop(adata, target: str) -> None:
    adata.obs.drop(columns=[column_name(target)], inplace=True, errors="ignore")
    adata.uns.pop(record_key(target), None)


def apply(adata, groupby: str, target: str) -> dict[str, Any]:
    """Write ``<target>_qualified`` and its record for the clusters of ``groupby``.

    Returns what the step records ({} when nothing was qualified). Any earlier
    qualifier column of ``target`` is dropped first: its comparisons named other
    labels. Best effort: a failure is logged and leaves no column."""
    drop(adata, target)
    try:
        labels = cluster_labels(adata, groupby, target)
        if not labels:
            return {}
        todo = plan(labels)
        if not todo:
            return {}
        from scview.core.marker_specificity import _group_sums, _ranked_matrix
        from scview.core.ora_background import resolve_markers_key

        key = resolve_markers_key(adata.uns, groupby)
        rgg = adata.uns.get(key) if key else None
        if rgg is None:
            return {}
        X, var_names = _ranked_matrix(adata, rgg)
        groups = list(labels)
        codes = pd.Categorical(adata.obs[groupby].astype(str).to_numpy(), categories=groups).codes.astype(np.int64)
        sums, detected = _group_sums(X, codes, len(groups))
        sizes = np.bincount(codes[codes >= 0], minlength=len(groups)).astype(np.float64)
        found = qualifiers(sums, detected, sizes, np.asarray(var_names).astype(str), groups, todo)
    except Exception as exc:  # never fail the annotation for a qualifier
        logger.warning("Label qualifiers for '%s' failed: %s", target, exc)
        return {}
    shown = {g: qualified_label(labels[g], found.get(g)) for g in labels}
    n_q = sum(bool(text(e)) for e in found.values())
    if n_q == 0:
        return {"qualifiers": 0, "qualifiers_considered": len(todo)}
    adata.obs[column_name(target)] = pd.Categorical(
        adata.obs[groupby].astype(str).map(shown).to_numpy(),
        categories=list(dict.fromkeys(shown[g] for g in labels)))
    adata.uns[record_key(target)] = json.dumps({
        "groupby": groupby,
        "clusters": {g: {"label": labels[g], **e} for g, e in found.items()},
        "cutoffs": CUTOFFS,
    })
    logger.info("Label qualifiers: %d of %d clusters considered -> obs['%s']", n_q, len(todo), column_name(target))
    return {"qualifiers": n_q, "qualifiers_considered": len(todo), "qualifier_column": column_name(target)}


def read(adata, target: str) -> dict[str, Any] | None:
    raw = adata.uns.get(record_key(target))
    if not isinstance(raw, str):
        return None
    try:
        return json.loads(raw)
    except ValueError:
        return None


def follow_rename(adata, column: str, large: bool = False) -> None:
    """After a label of ``column`` was renamed or merged: work the qualifiers out again
    (a merge changes which clusters share a label); a large dataset's are dropped."""
    rec = read(adata, column)
    if rec is None:
        return
    if large:
        drop(adata, column)
        return
    apply(adata, str(rec.get("groupby") or ""), column)
