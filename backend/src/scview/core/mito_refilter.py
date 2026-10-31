"""Re-filter with per-type % mito bounds after the dataset was filtered: the two-pass flow.

The usual case is a fresh 10x matrix: no cell types exist before filtering, so the
filtering step judges % mito by one global rule, and only the clustering that
comes after says what the types are. By then the cells the rule removed are gone
from the processed dataset. Re-filtering here starts again from the original
upload (``uploads/`` / ``ingested/``, never written to) with every cell typed:

- a cell the first pass kept takes its label in the chosen column (the pass-1
  clustering, or any label column) by barcode;
- a cell it removed takes the column's value when the original upload has that
  column (an author annotation), else the group whose centroid its expression is
  closest to (``assign_by_expression``: Pearson correlation of log-normalised
  expression over the dataset's variable genes, MT- genes left out, so a cell is
  placed by what it is rather than by how damaged it is);

then the filtering step runs with ``qc_mito_mode`` "per_type" on that column
(with ``floor``, never stricter than the global rule: ``qc_mito_type_floor``), and
the steps the dataset ran before (from its provenance, with their recorded
params) run again on the cells kept. The type column is kept as
``TYPE_COLUMN`` (with an author column, that column itself).

Why this design (measured on ovary_raw, 37,104 cells, its 7 author cell types
hidden from the clustering and used only as the answer; scratchpad
``mitofilter/scripts/twopass_sim.py``): after a default pass 1 (MAD + 20 %
ceiling, 4,034 cells removed), Leiden on the kept cells gave 13 clusters whose
majority author type matched 98.5 % of their cells; the removed cells were given
a cluster whose majority type was their own 97.1 % of the time by nearest
centroid (97.0 % by kNN in the PCA space, which needs the scaling rebuilt).
Filtering by per-cluster bounds from the original then differed from filtering
by the author types themselves in 1,876 cells, and brought back 69 of the 74
cells pass 1 removed that the author types keep. The alternative, filtering the
kept cells again by cluster and reprocessing them, cannot bring any back: from
the same pass 1 it differed in 2,542 cells and removed 6,910 (granulosa cells:
93 vs 24 by author types). With a fixed 10 % pass 1 (the lab's rule, 6,221
removed, granulosa 145 of 471) this flow removed 27 granulosa cells (author
types: 24) and brought back 989 of the 1,537 cells the author types keep; the
alternative removed 146 and 8,312 in all. Its loose variant (pass 1 removes only
cells at 50 % mito or more) does as well (granulosa 31) but needs a pass 1 run
that way before anyone saw the clusters.

What it cannot bring back: cells removed by the counts, genes or doublet rules
(those run again unchanged), cells whose type has too few cells, no spread or a
damaged median (the global rule again), and anything made in scView after pass 1
that is not a pipeline step (labels, gates, subclusters, annotations): the
columns of the original upload are what the new dataset starts with.
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Generator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import anndata as ad
import h5py
import numpy as np
import pandas as pd
import scipy.sparse as sp

from scview.core import provenance, qc_filter, qc_groups
from scview.core.anndata_adaptor import per_cell_qc

logger = logging.getLogger(__name__)

TYPE_COLUMN = "scview_mito_type"  # the types the per-type filter judged by (kept by reset_to_counts)
SKIP_STEPS = ("reset_to_counts", "cell_type_annotation")  # reset is done here; annotation not re-run
ASSIGN_GENES = 2000  # without a variable-gene column: this many most variable genes
MIN_ASSIGN_GENES = 50  # fewer shared genes: no assignment by expression
LOW_R = 0.2  # a removed cell whose best correlation is below this is counted as uncertain
TARGET_SUM = 1e4


class RefilterError(ValueError):
    """A re-filter that cannot run (no original upload, no type column, no % mito)."""

    def __init__(self, message: str, status: int = 400) -> None:
        super().__init__(message)
        self.status = status


def author_column(orig: ad.AnnData, column: str) -> bool:
    """Whether the original upload carries ``column`` itself (an author annotation),
    so every cell, removed or not, has its value. A ``scview_`` column there came
    from an earlier processing, not this dataset's clustering."""
    return column in orig.obs.columns and not str(column).startswith("scview_")


def source_file(dm, dataset_id: str) -> str | None:
    """The stored original (import, conversion or upload; never ``derived/``)."""
    for c in dm.h5ad_candidates(dataset_id):
        p = Path(c)
        if p.parent.parent.name != "derived" and p.is_file():
            return c
    return None


# --- types for the cells pass 1 removed ---------------------------------------------


def _looks_like_counts(M) -> bool:
    from scview.core.pipeline import looks_like_counts  # pipeline imports qc_filter, which we use

    return looks_like_counts(M)


def _lognorm(adata) -> tuple[Any, pd.Index]:
    """The kept cells' log-normalised expression and its genes: ``.raw`` after
    scaling, else X when it holds log values, else worked out from the counts."""
    if adata.raw is not None and not _looks_like_counts(adata.raw.X):
        return adata.raw.X, adata.raw.var_names
    X = adata.X
    if not _looks_like_counts(X) and float(X.min()) >= 0:
        return X, adata.var_names
    counts = adata.layers["counts"] if "counts" in adata.layers else X
    return _log_normalise(sp.csr_matrix(counts)), adata.var_names


def _log_normalise(C: sp.csr_matrix, target: float = TARGET_SUM) -> sp.csr_matrix:
    total = np.asarray(C.sum(axis=1), dtype=np.float64).ravel()
    scale = np.divide(target, total, out=np.zeros_like(total), where=total > 0)
    L = sp.csr_matrix(sp.diags(scale) @ C.astype(np.float64))
    L.data = np.log1p(L.data)
    return L


def _assign_genes(adata, L, names: pd.Index, available: pd.Index) -> list[str]:
    """The genes to place cells by: the variable genes (else the most variable
    ones), without MT- genes, that the original upload has too."""
    var = adata.var
    keep_mt = ~names.str.startswith(("MT-", "mt-"))
    if "highly_variable" in var.columns and var["highly_variable"].astype(bool).any():
        hv = set(var.index[var["highly_variable"].astype(bool).to_numpy()])
        pick = [g for g, ok in zip(names, keep_mt) if ok and g in hv]
    else:
        M = sp.csr_matrix(L)
        mean = np.asarray(M.mean(axis=0)).ravel()
        sq = np.asarray(M.multiply(M).mean(axis=0)).ravel()
        order = np.argsort(-(sq - mean**2), kind="stable")
        pick = [names[i] for i in order if keep_mt[i]][:ASSIGN_GENES]
    have = set(available)
    return [g for g in pick if g in have]


def assign_by_expression(adata, column: str, counts, count_genes: pd.Index,
                         target_sum: float = TARGET_SUM) -> tuple[np.ndarray, np.ndarray, dict]:
    """For cells not in ``adata`` (their raw ``counts``, genes ``count_genes``): the
    group of ``column`` whose centroid in ``adata`` (mean log-normalised expression
    of its cells) is most correlated with the cell's own (Pearson, over
    ``_assign_genes``). Returns (labels, "" where none; best r; info)."""
    L, names = _lognorm(adata)
    genes = _assign_genes(adata, L, names, count_genes)
    m = int(counts.shape[0])
    info: dict = {"method": "nearest_centroid", "metric": "pearson", "genes": len(genes),
                  "target_sum": float(target_sum)}
    if m == 0:
        return np.array([], dtype=object), np.array([]), info
    if len(genes) < MIN_ASSIGN_GENES:
        info["note"] = f"Only {len(genes)} genes shared with the original upload: no assignment."
        return np.full(m, "", dtype=object), np.full(m, np.nan), info
    groups, codes = qc_groups.group_codes(adata.obs[column])
    k = len(groups)
    gi = names.get_indexer(genes)
    Lg = L[:, gi]
    lab = codes >= 0
    ind = sp.csr_matrix((np.ones(int(lab.sum())), (codes[lab], np.flatnonzero(lab))), shape=(k, adata.n_obs))
    sizes = np.asarray(ind.sum(axis=1)).ravel()
    sums = ind @ Lg
    cent = (sums.toarray() if sp.issparse(sums) else np.asarray(sums)) / np.maximum(sizes, 1)[:, None]
    C = sp.csr_matrix(counts)
    R = _log_normalise(C, target_sum)[:, count_genes.get_indexer(genes)].toarray()

    def unit(M: np.ndarray) -> np.ndarray:
        M = M - M.mean(axis=1, keepdims=True)
        norm = np.linalg.norm(M, axis=1, keepdims=True)
        return np.divide(M, norm, out=np.zeros_like(M), where=norm > 0)

    r = unit(R) @ unit(cent).T  # cells x groups
    r[:, sizes == 0] = -np.inf
    best = r.argmax(axis=1)
    score = r[np.arange(m), best]
    ok = np.isfinite(score) & (np.abs(R).sum(axis=1) > 0)
    labels = np.where(ok, np.asarray(groups, dtype=object)[best], "")
    good = score[ok]
    info.update(n=m, n_assigned=int(ok.sum()),
                median_r=round(float(np.median(good)), 3) if good.size else None,
                n_low=int((good < LOW_R).sum()))
    return labels.astype(object), np.where(ok, score, np.nan), info


# --- reading only some cells of the original ----------------------------------------


def _counts_key(f: h5py.File) -> tuple[str, str] | None:
    """(matrix key, var key) of the original's raw counts: layers/counts, raw/X, X."""
    for key, var in (("layers/counts", "var"), ("raw/X", "raw/var"), ("X", "var")):
        if key in f and var in f:
            try:
                if _looks_like_counts(_rows(f[key], np.arange(min(200, _n_rows(f[key]))))):
                    return key, var
            except Exception:  # an unreadable matrix: try the next
                continue
    return None


def _n_rows(node) -> int:
    if isinstance(node, h5py.Dataset):
        return int(node.shape[0])
    return int(node.attrs["shape"][0])


def _rows(node, rows: np.ndarray) -> sp.csr_matrix:
    """Rows of a stored matrix (CSR group or dense dataset), in ``rows`` order."""
    from anndata.io import sparse_dataset

    rows = np.asarray(rows, dtype=np.int64)
    order = np.argsort(rows)
    srt = rows[order]
    if isinstance(node, h5py.Dataset):
        block = sp.csr_matrix(node[srt, :]) if srt.size else sp.csr_matrix((0, node.shape[1]))
    else:
        block = sp.csr_matrix(sparse_dataset(node)[srt])
    back = np.empty_like(order)
    back[order] = np.arange(order.size)
    return block[back]


def read_cells(path: str, names: list[str]) -> tuple[sp.csr_matrix, pd.Index, pd.DataFrame]:
    """The raw counts of the named cells in the stored original, without reading
    the rest of the matrix: (counts, gene names, those cells' obs). Raises
    RefilterError when the original holds no counts or lacks a cell."""
    from anndata.io import read_elem

    with h5py.File(path, "r") as f:
        found = _counts_key(f)
        if found is None:
            raise RefilterError("The original upload holds no raw counts to place the removed cells by.")
        key, var = found
        obs = read_elem(f["obs"])
        obs.index = obs.index.astype(str)
        pos = pd.Index(obs.index).get_indexer(names)
        if (pos < 0).any():
            raise RefilterError("Some cells the filtering step removed are not in the original upload "
                                "(it was changed outside scView?).", status=409)
        genes = pd.Index(read_elem(f[var]).index.astype(str))
        return _rows(f[key], pos), genes, obs.iloc[pos]


# Removed cells' labels per (dataset, file, mtime, column): placed once, then served.
_assigned_cache: dict[tuple, tuple[list[str], np.ndarray]] = {}


def assigned_removed(dm, dataset_id: str, adaptor, column: str) -> tuple[list[str], np.ndarray] | None:
    """For the Mito by cell type card: each cell the filtering step removed (its
    record, in its order) with a ``column`` label (the original's own value when it
    has the column, else by expression) and its recorded % mito. None when there is
    no usable record or no original upload."""
    adata = adaptor.adata
    names = qc_filter.recorded_names(adata)
    gone = qc_filter.recorded_removed_mito(adata, None)
    src = source_file(dm, dataset_id)
    if names is None or gone is None or src is None or column not in adata.obs.columns:
        return None
    try:
        key = (dataset_id, adaptor.h5ad_path, Path(adaptor.h5ad_path).stat().st_mtime_ns, column)
    except OSError:
        return None
    if key in _assigned_cache:
        return _assigned_cache[key]
    try:
        counts, genes, obs = read_cells(src, names)
        if column in obs.columns and not str(column).startswith("scview_"):
            col = obs[column]
            labels = col.astype(str).where(col.notna(), "").tolist()
        else:
            labels = list(assign_by_expression(adata, column, counts, genes)[0])
    except Exception as e:  # the card shows the kept cells only
        logger.warning("Could not place the removed cells of %s: %s", dataset_id, e)
        return None
    for k in [k for k in _assigned_cache if k[0] == dataset_id]:
        del _assigned_cache[k]
    _assigned_cache[key] = (labels, gone[1])
    return _assigned_cache[key]


# --- the run ------------------------------------------------------------------------


def plan_params(adata) -> tuple[list[str], Any, list[str]]:
    """The pipeline steps the dataset ran (its provenance, in pipeline order; not
    reset or annotation) with their recorded params, and the steps left out."""
    from scview.core.pipeline import ALL_STEPS, PipelineParams

    history = [h for h in provenance.read_provenance(adata).get("history", []) if isinstance(h, dict)]
    ran = {h.get("step") for h in history}
    params = PipelineParams()
    fields = set(params.__dataclass_fields__)
    for h in history:  # later runs of a step override earlier ones
        if h.get("step") not in ALL_STEPS:
            continue
        for k, v in (h.get("params") or {}).items():
            if k in fields and not k.startswith("qc_mito_"):
                setattr(params, k, v)
    steps = [s for s in ALL_STEPS if s in ran and s not in SKIP_STEPS]
    if "filtering" not in steps:
        steps.insert(len([s for s in steps if ALL_STEPS.index(s) < ALL_STEPS.index("filtering")]),
                     "filtering")
    skipped = [s for s in ("cell_type_annotation",) if s in ran]
    return steps, params, skipped


@dataclass
class Prepared:
    """The original upload with every cell typed, ready for the filtering step."""

    adata: ad.AnnData
    type_column: str
    column: str
    steps: list[str]
    params: Any
    skipped: list[str]
    reset: bool
    n_original: int
    kept_before: np.ndarray  # per cell of ``adata``: kept by the dataset's last filtering
    typed: dict[str, int]
    assignment: dict | None
    source: str
    notes: list[str] = field(default_factory=list)


def _check(adaptor, column: str | None) -> str:
    """The type column (named, else the card's choice); RefilterError when it can't run."""
    from scview.core import mito_by_type

    adata = adaptor.adata
    cells = adaptor.qc_per_cell() if hasattr(adaptor, "qc_per_cell") else per_cell_qc(adata)
    if not qc_filter._mito_measurable(cells):
        raise RefilterError(mito_by_type.NOT_MEASURABLE)
    if column:
        if column not in adata.obs.columns:
            raise RefilterError(f"Column '{column}' not found.", status=404)
        return column
    col, _ = mito_by_type._column(adata, qc_filter.recorded_filtering(adata))
    if not col:
        raise RefilterError("No cell types or clustering yet: run clustering, or pick a label column.")
    return col


def check_runnable(dm, dataset_id: str, adaptor, column: str | None, mito_mode: str = "per_type") -> str:
    """The checks ``prepare`` makes before reading anything big; the type column."""
    if mito_mode not in qc_filter.MITO_MODES:
        raise RefilterError(f"Unknown % mito mode '{mito_mode}': use 'global' or 'per_type'.")
    col = _check(adaptor, column)
    if source_file(dm, dataset_id) is None:
        raise RefilterError("No original upload is stored for this dataset, so the cells filtering "
                            "removed can't be brought back.", status=409)
    return col


def read_need(dm, dataset_id: str) -> int:
    """Peak memory of ``prepare``'s read of the original upload, whole (``working_copy``);
    0 without one (``check_runnable`` says why) or without an estimate."""
    from scview.core import working_copy

    src = source_file(dm, dataset_id)
    if src is None:
        return 0
    try:
        return working_copy.file_bytes(src)
    except Exception as e:  # noqa: BLE001 - never block the work on the estimate itself
        logger.warning("No read estimate for %s: %s", src, e)
        return 0


def prepare(dm, dataset_id: str, adaptor, column: str | None, mito_mode: str = "per_type",
            floor: bool = False) -> Prepared:
    """Read the original upload and type every cell (module docstring). ``floor``:
    per type never stricter than the global rule. Raises RefilterError when it
    can't run."""
    from scview.core.pipeline import _run_reset_to_counts

    column = check_runnable(dm, dataset_id, adaptor, column, mito_mode)
    src = source_file(dm, dataset_id)
    cur = adaptor.adata
    steps, params, skipped = plan_params(cur)
    reset = "reset_to_counts" in {h.get("step") for h in provenance.read_provenance(cur).get("history", [])}
    orig = ad.read_h5ad(src)
    orig.obs_names = orig.obs_names.astype(str)
    if not cur.obs_names.isin(orig.obs_names).all():
        raise RefilterError("Some of this dataset's cells are not in its original upload, so it can't be "
                            "re-filtered from it.", status=409)
    if reset:
        orig = _run_reset_to_counts(orig, params)
    kept = np.asarray(orig.obs_names.isin(cur.obs_names))
    typed = {"kept": int(kept.sum()), "column": 0, "expression": 0, "none": 0}
    assignment = None
    if author_column(orig, column):
        type_col = column  # the original has every cell's value
        typed["column"] = int((~kept).sum())
        typed["none"] = int(orig.obs[column].isna().sum())
    else:  # made after filtering (a clustering): kept cells by barcode, the rest by expression
        type_col = TYPE_COLUMN
        groups, _ = qc_groups.group_codes(cur.obs[column])
        lab = pd.Series("", index=orig.obs_names, dtype=object)
        src_lab = cur.obs[column]
        lab.loc[cur.obs_names] = src_lab.astype(str).where(src_lab.notna(), "").to_numpy()
        gone = np.flatnonzero(~kept)
        if gone.size:
            labels, _, assignment = assign_by_expression(
                cur, column, _original_counts(orig)[gone], orig.var_names,
                float(getattr(params, "target_sum", TARGET_SUM)))
            lab.iloc[gone] = labels
            typed["expression"] = int(assignment.get("n_assigned", 0))
        values = lab.to_numpy()
        typed["none"] = int((values == "").sum())
        orig.obs[type_col] = pd.Categorical(np.where(values == "", None, values), categories=groups)
    # The new dataset's history: the dataset's own, then this step, then the run.
    prov = provenance.read_provenance(cur)
    prov["current"] = {k: v for k, v in prov.get("current", {}).items() if k == "head"}
    orig.uns[provenance.UNS_KEY] = json.dumps(prov)
    orig.uns.pop(qc_filter.RECORD_KEY, None)
    params.qc_mito_mode = mito_mode
    params.qc_mito_type_column = type_col if mito_mode == "per_type" else ""
    params.qc_mito_type_floor = bool(floor) and mito_mode == "per_type"
    provenance.record_step(orig, step="mito_refilter", tool="scview.mito_refilter", params={
        "column": column, "type_column": type_col, "mito_mode": mito_mode,
        "mito_type_floor": params.qc_mito_type_floor,
        "source_file": Path(src).name, "reset_to_counts": reset,
        "n_original": int(orig.n_obs), "n_kept_before": int(kept.sum()), "typed": typed,
        "assignment": assignment, "steps": steps, "skipped": skipped,
    })
    notes = []
    if skipped:
        notes.append("Cell-type annotation is not re-run: run it again afterwards.")
    return Prepared(orig, type_col, column, steps, params, skipped, reset, int(orig.n_obs), kept, typed,
                    assignment, Path(src).name, notes)


def _original_counts(orig: ad.AnnData):
    if "counts" in orig.layers and _looks_like_counts(orig.layers["counts"]):
        return sp.csr_matrix(orig.layers["counts"])
    if _looks_like_counts(orig.X):
        return sp.csr_matrix(orig.X)
    if orig.raw is not None and _looks_like_counts(orig.raw.X) and not len(orig.var_names.difference(orig.raw.var_names)):
        return sp.csr_matrix(orig.raw[:, orig.var_names].X)
    raise RefilterError("The original upload holds no raw counts to place the removed cells by.")


def _judged(prep: Prepared) -> tuple[ad.AnnData, dict, np.ndarray, qc_filter.TypeBounds | None]:
    """What the filtering step will remove from the prepared cells (a throwaway
    copy: the QC metrics step runs on it in place, as the run's does)."""
    from scview.core.pipeline import _run_qc_metrics

    A = prep.adata
    if "qc_metrics" in prep.steps:
        _run_qc_metrics(A, prep.params)
    t = qc_filter.Thresholds.from_params(prep.params)
    samples = qc_filter.samples_of(A, prep.params.qc_sample_column or None)
    removed, _, types = qc_filter.judge_types(A, per_cell_qc(A), t, samples)
    return A, removed, ~qc_filter.keep_mask(removed, A.n_obs), types


def _doublets_as_the_dataset_has_them(prep: Prepared, cur: ad.AnnData) -> str:
    """For the preview: the doublets the dataset's filtering drops (its recorded
    source, ``doublet_source``) for its cells, the removal record's for the cells it
    removed, written into the prepared cells as the calls the preview judges by.
    Returns a note when those are cell hashing calls the original upload lacks (the
    run cannot drop them; nothing here carries hashing over), else ""."""
    from scview.core import doublet_source

    if not prep.params.drop_doublets:
        return ""
    used = doublet_source.resolve(cur.obs, prep.params.qc_doublet_source)
    if used is None:
        return ""
    calls = pd.Series(False, index=prep.adata.obs_names)
    calls.loc[cur.obs_names] = used.calls
    names, bits = qc_filter.recorded_names(cur), qc_filter.recorded_bits(cur, "doublets")
    if names is not None and bits is not None and len(names) == bits.size:
        calls.loc[[n for n in names if n in calls.index]] = bits[[n in calls.index for n in names]]
    prep.adata.obs[doublet_source.SCRUBLET_CALL] = calls.to_numpy()
    prep.params.qc_doublet_source = doublet_source.SCRUBLET  # the calls just written, whatever their source
    if doublet_source.HASH_CLASS in used.columns and doublet_source.HASH_CLASS not in prep.adata.obs.columns:
        return ("The dataset's doublets are cell hashing calls, which the original upload does not have: "
                "the run drops Scrublet's calls instead (or none, with the source set to hashing). "
                "Demultiplex the new dataset and filter again to drop hashing doublets.")
    return ""


def preview(dm, dataset_id: str, adaptor, column: str | None, mito_mode: str = "per_type",
            floor: bool = False) -> dict:
    """What re-filtering would do, per type: cells, median, the rule and bound,
    removed now (by the dataset's last filtering), removed by the re-filter, brought
    back and newly removed; plus how the removed cells were typed and what it can
    and cannot bring back. ``floor``: per type never stricter than the global rule
    (each row also carries the floor's columns, ``qc_filter.floor_rows``). Raises
    RefilterError when it can't run."""
    prep = prepare(dm, dataset_id, adaptor, column, mito_mode, floor)
    cur = adaptor.adata
    hashing_note = _doublets_as_the_dataset_has_them(prep, cur)
    A, removed, drop, types = _judged(prep)
    kept = prep.kept_before
    rows = []
    names, codes = qc_groups.group_codes(A.obs[prep.type_column])
    table = {r["name"]: r for r in (types.to_dict()["types"] if types is not None else [])}
    for i, name in enumerate(names):
        m = codes == i
        rows.append({**table.get(name, {"name": name}), "n_all": int(m.sum()),
                     "removed_before": int((m & ~kept).sum()), "removed_after": int((m & drop).sum()),
                     "brought_back": int((m & ~kept & ~drop).sum()),
                     "newly_removed": int((m & kept & drop).sum())})
    mito = removed.get("max_pct_mt", np.zeros(A.n_obs, dtype=bool))
    other = drop & ~mito
    back = int((~kept & ~drop).sum())
    stays = int((~kept & drop).sum())
    stays_other = int((~kept & other).sum())
    stays_global = int((~kept & mito & ~(types.own if types is not None else np.zeros(A.n_obs, bool))).sum())
    out = {
        "column": prep.column, "type_column": prep.type_column, "mito_mode": mito_mode,
        "mito_type_floor": bool(prep.params.qc_mito_type_floor),
        "source_file": prep.source, "reset_to_counts": prep.reset,
        "n_original": prep.n_original, "n_current": int(cur.n_obs), "n_after": int((~drop).sum()),
        "removed_before": int((~kept).sum()), "removed_after": int(drop.sum()),
        "brought_back": back, "newly_removed": int((kept & drop).sum()),
        "removed_other": int(other.sum()),
        "typed": prep.typed, "assignment": prep.assignment,
        "types": rows, "rule": types.to_dict() if types is not None else None,
        "steps": prep.steps, "skipped_steps": prep.skipped,
        "can": (f"Brings back {back:,} of the {int((~kept).sum()):,} cells the last filtering removed: "
                + ("those within their type's own % mito bound or the global rule's, whichever is "
                   "higher." if prep.params.qc_mito_type_floor else
                   "those within their type's own % mito bound.")),
        "cannot": (f"{stays:,} stay removed: {stays_other:,} by the counts, genes or doublet rules "
                   f"(applied again unchanged) and {stays - stays_other:,} for % mito"
                   + (f", {stays_global:,} of them in a type that keeps the global rule (too few cells, "
                      "no spread, or damaged)" if stays_global else "")
                   + ". Labels, gates and columns made in scView since are not carried over: the new "
                   "dataset starts from the original upload's columns."),
        "notes": prep.notes + ([] if not prep.params.drop_doublets else
                               ["Doublet calls as the dataset has them; the run calls them again on the "
                                "same cells."]) + ([hashing_note] if hashing_note else []),
        "warnings": types.to_dict()["warnings"] if types is not None else [],
    }
    return out


def run_events(dm, dataset_id: str, adaptor, column: str | None, output_path: str,
               mito_mode: str = "per_type", control=None,
               floor: bool = False) -> Generator[tuple[str, dict], None, None]:
    """The re-filter as pipeline events (``run_pipeline_streamed``'s, after two of
    its own: ``refilter`` with ``stage`` "load" and "types"); saves to ``output_path``
    when the run reaches its end. ``control`` (a ``RunControl``) gives the steps the
    watchdog, Cancel and the pre-step memory check of any pipeline run."""
    from scview.core.pipeline import run_pipeline_streamed

    t0 = time.time()
    yield ("refilter", {"stage": "load", "message": "Reading the original upload"})
    prep = prepare(dm, dataset_id, adaptor, column, mito_mode, floor)
    yield ("refilter", {"stage": "types", "message": "Typed every cell", "typed": prep.typed,
                        "assignment": prep.assignment, "steps": prep.steps,
                        "elapsed": round(time.time() - t0, 2)})
    A, prep.adata = prep.adata, None  # the run owns it now
    yield from run_pipeline_streamed(A, prep.steps, prep.params, output_path, control=control)
