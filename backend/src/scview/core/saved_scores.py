"""Per-cell scores kept as obs columns: a scored gene set, one TF / pathway's activity, one
gene program's usage or one motif's chromVAR z.

Scoring in the app (``api/v1/genesets.py``, ``api/v1/activity.py``, ``api/v1/programs.py``,
``api/v1/atac_motifs.py``) draws a score on the plot; with ``save`` it is also written into
the dataset as one float column, so Compare colourings, Colour by, the histogram, sessions
and the h5ad export see it like any other column. Only the score the user put on the plot
is saved, never a whole matrix of them.

Column names: ``score_<set>`` for a gene set, ``activity_<source>`` for an activity,
``program_k<k>_<n>`` for program n of a k-program run, ``chromvar_<motif>_<JASPAR id>`` for
a motif; cleaned to letters, digits, ``_ . -`` and at most ``NAME_MAX`` characters, made
unique. The record, uns ``scview_scores`` (a JSON string, as the provenance), says per
column what it holds: kind, name, method, the genes used and missing, the matrix scored,
the network, the run it came from, when. Scoring the same set again (any method), the
same source, the same program of the same run or the same motif replaces its column and
its entry. Only a column in the record is ever replaced or removed: a column the user or
the file made is never touched.
"""

from __future__ import annotations

import json
import logging
import re
from datetime import datetime, timezone
from typing import Any

import numpy as np

from scview.core import provenance

logger = logging.getLogger(__name__)

RECORD_KEY = "scview_scores"
KINDS = ("geneset", "activity", "program", "chromvar")
PREFIXES = {"geneset": "score_", "activity": "activity_", "program": "program_", "chromvar": "chromvar_"}
# The record fields that, with the kind, say which score a column holds: the same ones
# saved again replace it.
IDENTITY = {"geneset": ("name",), "activity": ("name", "network"), "program": ("run_id", "program"),
            "chromvar": ("motif_id",)}
# What a run-made score's listing carries beyond the common fields (obs_flag).
_RUN_FLAGS = {"program": ("k", "program", "run_id", "top_genes"), "chromvar": ("motif_id", "run_id", "jaspar")}
NAME_MAX = 64  # whole column name, as a hand-label column (core/user_labels.COLUMN_MAX)
MAX_GENES_RECORDED = 2000  # a set's genes in the record (the scoring endpoint's own cap is larger)

# Scores the pipeline already keeps per cell: offered with the saved ones, never removed.
BUILTIN_SCORE_COLUMNS = ("S_score", "G2M_score", "dissociation_stress_score")

TOOL = "scview.saved_scores"
_UNSAFE = re.compile(r"[^A-Za-z0-9_.\-]+")


class SavedScoreError(ValueError):
    """A request about a saved score that cannot be served; ``status`` is the HTTP code."""

    def __init__(self, message: str, status: int = 400) -> None:
        super().__init__(message)
        self.status = status


def record(adata) -> dict[str, dict[str, Any]]:
    """The saved scores: column -> entry, only for columns still in obs."""
    raw = adata.uns.get(RECORD_KEY)
    if raw is None:
        return {}
    try:
        data = json.loads(raw) if isinstance(raw, (str, bytes)) else dict(raw)
    except (TypeError, ValueError):
        logger.warning("Unreadable %s record; ignored", RECORD_KEY)
        return {}
    if not isinstance(data, dict):
        return {}
    cols = set(map(str, adata.obs.columns))
    return {str(k): v for k, v in data.items() if isinstance(v, dict) and str(k) in cols}


def current_runs(adata, entries: dict[str, dict[str, Any]] | None = None) -> dict[str, str | None]:
    """The run id of the dataset's gene programs and of its chromVAR run now, for the kinds
    the record holds (``obs_flag``'s ``earlier_run``); nothing is read for other kinds."""
    kinds = {e.get("kind") for e in (record(adata) if entries is None else entries).values()}
    out: dict[str, str | None] = {}
    if "program" in kinds:
        from scview.core import gene_programs

        info = gene_programs.stored(adata)
        out["program"] = info.get("run_id") if info else None
    if "chromvar" in kinds:
        # The run itself is on disk (core/atac_chromvar.py); its provenance step names it.
        steps = [h for h in provenance.read_provenance(adata).get("history", []) if h.get("step") == "atac_chromvar"]
        out["chromvar"] = (steps[-1].get("params") or {}).get("run_id") if steps else None
    return out


def obs_flag(entry: dict[str, Any], runs: dict[str, str | None] | None = None) -> dict[str, Any]:
    """What the obs column listing carries per saved score (the UI groups on it). A program
    or motif score also names its run, and ``earlier_run`` says whether the dataset's run
    now is another one (``runs``: ``current_runs``)."""
    flag = {k: entry.get(k) for k in ("kind", "name", "method", "network")}
    kind = entry.get("kind")
    if kind in _RUN_FLAGS:
        flag.update({k: entry.get(k) for k in _RUN_FLAGS[kind]})
        if runs is not None and kind in runs:
            flag["earlier_run"] = entry.get("run_id") != runs[kind]
    return flag


def _same(entry: dict[str, Any], kind: str, ident: dict[str, Any]) -> bool:
    return entry.get("kind") == kind and all(entry.get(f) == ident.get(f) for f in IDENTITY[kind])


def clean_name(kind: str, name: str) -> str:
    """``score_<name>``, ``activity_<name>``... with only safe characters, capped."""
    prefix = PREFIXES[kind]
    body = _UNSAFE.sub("_", str(name)).strip("_.-") or "unnamed"
    return (prefix + body)[:NAME_MAX].rstrip("_.-")


def column_for(adata, kind: str, name: str, network: str | None = None, *,
               ident: dict[str, Any] | None = None, body: str | None = None) -> str:
    """The column this score goes into: its saved column when the same score (the
    ``IDENTITY`` fields, from ``ident``, else ``name`` and ``network``) was saved before,
    else a new name no obs column has, made from ``body`` (else ``name``)."""
    ident = ident if ident is not None else {"name": name, "network": network}
    for col, entry in record(adata).items():
        if _same(entry, kind, ident):
            return col
    base = clean_name(kind, body if body is not None else name)
    taken = {str(c).lower() for c in adata.obs.columns}
    col, i = base, 2
    while col.lower() in taken:
        suffix = f"_{i}"
        col = base[: NAME_MAX - len(suffix)] + suffix
        i += 1
    return col


def _write_record(adata, entries: dict[str, dict[str, Any]]) -> None:
    # A new value, never an in-place change: obs_writes.write_derived saves the uns
    # entries whose identity changed.
    if entries:
        adata.uns[RECORD_KEY] = json.dumps(entries, default=str)
    else:
        adata.uns.pop(RECORD_KEY, None)


def _recorded(record) -> None:
    """Run ``record`` (a ``provenance.record_step`` call); provenance never blocks the edit."""
    try:
        record()
    except Exception as e:
        logger.warning("Could not record a saved score's provenance: %s", e)


def save(adata, *, kind: str, name: str, values, method: str | None = None,
         genes: list[str] | None = None, genes_missing: list[str] | None = None,
         source: str | None = None, network: str | None = None,
         resource: str | None = None, extra: dict[str, Any] | None = None,
         column_body: str | None = None) -> str:
    """Write one per-cell score into ``adata.obs`` and the record; the column name.

    ``genes``: the genes scored (a gene set); ``source``: the matrix scored; ``network``
    and ``resource``: an activity's network and where it came from; ``extra``: other
    settings worth keeping (AUCell's ``max_rank``; for a program its ``run_id`` and
    ``program`` number, for a motif its ``motif_id``: the ``IDENTITY`` fields);
    ``column_body``: the column name after the prefix, when it is not the name. Replaces
    the column of the same score saved before. A NaN (a cell chromVAR gave no value)
    stays NaN. Records the step ``save_score``."""
    if kind not in KINDS:
        raise SavedScoreError(f"Unknown score kind '{kind}'.")
    if not str(name).strip():
        raise SavedScoreError("A saved score needs a name.")
    v = np.asarray(values, dtype=np.float32).ravel()
    if v.shape[0] != adata.n_obs:
        raise SavedScoreError(
            f"{v.shape[0]:,} scores for {adata.n_obs:,} cells: the dataset changed while it was "
            "scored. Score it again.", status=409)
    entries = record(adata)
    ident = {"name": str(name), "network": network, **(extra or {})}
    if any(ident.get(f) is None for f in IDENTITY[kind]):
        raise SavedScoreError(f"A saved {kind} score needs {' and '.join(IDENTITY[kind])}.")
    column = column_for(adata, kind, str(name), network, ident=ident, body=column_body)
    replaced = column in entries
    entry: dict[str, Any] = {
        "kind": kind,
        "name": str(name),
        "method": method,
        "source": source,
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    if genes is not None:
        entry["genes"] = [str(g) for g in genes][:MAX_GENES_RECORDED]
        entry["n_genes"] = len(genes)
        entry["genes_missing"] = [str(g) for g in (genes_missing or [])][:MAX_GENES_RECORDED]
    if network is not None:
        entry["network"] = network
    if resource is not None:
        entry["resource"] = resource
    if extra:
        entry.update({k: v for k, v in extra.items() if v is not None})
    adata.obs[column] = v
    entries[column] = entry
    _write_record(adata, entries)
    params = {"column": column, "replaced": replaced, **{k: v for k, v in entry.items() if k != "created"}}
    _recorded(lambda: provenance.record_step(adata, step="save_score", tool=TOOL, params=params))
    return column


def remove(adata, column: str) -> dict[str, Any]:
    """Drop a saved score's column and its record entry; the entry removed. Only a column
    in the record: any other is a 404, so a file's own columns are never touched."""
    entries = record(adata)
    entry = entries.pop(column, None)
    if entry is None:
        raise SavedScoreError(f"'{column}' is not a saved score.", status=404)
    del adata.obs[column]
    _write_record(adata, entries)
    params = {"column": column, "kind": entry.get("kind"), "name": entry.get("name")}
    _recorded(lambda: provenance.record_step(adata, step="remove_score", tool=TOOL, params=params))
    return entry
