"""Joint multi-dataset integration — one shared embedding across datasets.

Combines 2+ library datasets into a new *joint* dataset. Each contributes its raw
counts (recovered from its own source file when processing replaced them) and
its categorical annotations; genes are aligned with the ingestion merge engine
(intersection by default, Ensembl/symbol reconciliation); a ``dataset`` column
records each cell's origin. The joint dataset is then processed from counts with
that column as the batch key — batch-aware HVGs, PCA, an *uncorrected* UMAP kept
for comparison, Harmony, then neighbors / clustering / UMAP / markers on the
corrected space — so every existing panel works on it unchanged. A batch-mixing
score before and after correction is stored with it (``batch_mixing``).

Sources are never modified: the merged counts land in ``ingested/{id}/``
(read-only) and the processed result in ``derived/{id}/``, like any dataset.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Generator

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp
from pydantic import BaseModel, Field

from scview.core import preflight, protein, provenance, spatial, working_copy
from scview.core.ingestion.merge import MergeJoin, MergePlan, merge_units, plan_merge
from scview.core.ingestion.session import _make_readonly, _safe_name
from scview.core.pipeline import PipelineParams, looks_like_counts, run_pipeline_streamed
from scview.core.uns_keys import BATCH_MIXING_KEY

if TYPE_CHECKING:
    from scview.core.dataset_manager import DatasetManager
    from scview.core.run_watchdog import RunControl

logger = logging.getLogger(__name__)

# obs column naming each cell's source dataset (also the batch key).
LABEL_COLUMN = "dataset"

# Categorical annotations with more levels than this (barcodes, free text) are
# not carried into the joint dataset.
_MAX_CARRY_CATEGORIES = 200
# core/vdj.py's clonotype column: carried whatever its levels, ids prefixed per dataset.
CLONOTYPE_COL = "clonotype"
# Below this, a dataset can't contribute meaningfully to per-batch HVG selection.
MIN_CELLS_PER_DATASET = 20

# Pipeline parameters a caller may override for the joint run.
TUNABLE_PARAMS = ("clustering_resolution", "n_top_genes", "n_comps", "n_neighbors", "umap_min_dist")

_SOURCE_LAYER_LABEL = {
    "derived": "processed file",
    "ingested": "ingested source",
    "converted": "converted source",
    "uploads": "original upload",
}


class IntegrationError(RuntimeError):
    """Raised when datasets can't be combined into a joint dataset."""


# Integration methods: a batch correction on the PCA (core/batch_methods.py: Harmony,
# Scanorama, BBKNN), or plain concatenation (no correction).
METHOD_HARMONY = "harmony"
METHOD_SCANORAMA = "scanorama"
METHOD_BBKNN = "bbknn"
METHOD_NONE = "none"
METHODS = (METHOD_HARMONY, METHOD_SCANORAMA, METHOD_BBKNN, METHOD_NONE)


# ---------------------------------------------------------------------------
# Raw-count recovery
# ---------------------------------------------------------------------------


def _find_counts(adata: ad.AnnData) -> tuple[str, Any, pd.DataFrame] | None:
    """First raw-count matrix in the usual places: (where, matrix, var)."""
    candidates = []
    if "counts" in adata.layers:
        candidates.append(("layers['counts']", adata.layers["counts"], adata.var))
    if adata.raw is not None:
        candidates.append(("raw.X", adata.raw.X, adata.raw.var))
    candidates.append(("X", adata.X, adata.var))
    for where, M, var in candidates:
        if looks_like_counts(M):
            return where, M, var
    return None


def _to_csr(M) -> sp.csr_matrix:
    if working_copy.on_disk(M):  # a backed dataset's: read once, no copy of the copy
        M = working_copy.read_matrix(M)
        return sp.csr_matrix(M, dtype=np.float32) if sp.issparse(M) else sp.csr_matrix(M.astype(np.float32, copy=False))
    if sp.issparse(M):
        return sp.csr_matrix(M, dtype=np.float32, copy=True)
    return sp.csr_matrix(np.asarray(M, dtype=np.float32))


def _open_lazily(path: str) -> ad.AnnData:
    """``path`` with X and .raw's X left on disk: reading counts for a joint embedding or
    a mapping needs one matrix, not all of them. ``.file.close()`` it when done."""
    return ad.read_h5ad(path, backed="r")


@dataclass
class RecoveredCounts:
    X: sp.csr_matrix
    var: pd.DataFrame
    source: str  # human-readable origin, e.g. "raw.X of the original upload"


def recover_counts(adata: ad.AnnData, fallback_paths: list[str]) -> RecoveredCounts:
    """Raw counts for ``adata``'s cells, from the dataset itself or its sources.

    Processing can replace the counts (scaling stores log-normalised data in
    ``.raw``), but the dataset's original upload / ingest still holds them; those
    are read and aligned to the current cells by barcode, so annotations made
    after filtering still line up.
    """
    hit = _find_counts(adata)
    if hit is not None:
        where, M, var = hit
        return RecoveredCounts(_to_csr(M), var.copy(), where)

    want = adata.obs_names
    for path in fallback_paths:
        try:
            src = _open_lazily(path)
        except Exception as e:  # unreadable source: try the next one
            logger.warning("Could not read %s for counts: %s", path, e)
            continue
        try:
            hit = _find_counts(src)
            if hit is None or not src.obs_names.is_unique:
                continue
            pos = src.obs_names.get_indexer(want)
            if (pos < 0).any():
                logger.info("%s lacks %d of the dataset's cells; not using it for counts",
                            path, int((pos < 0).sum()))
                continue
            where, M, var = hit
            layer = _SOURCE_LAYER_LABEL.get(Path(path).parent.parent.name, "source file")
            return RecoveredCounts(_to_csr(M)[pos], var.copy(), f"{where} of the {layer}")
        finally:
            src.file.close()

    raise IntegrationError(
        "no raw counts found (checked layers['counts'], raw.X and X, and the dataset's "
        "source files). Integration re-processes from counts, so this dataset can't join."
    )


# ---------------------------------------------------------------------------
# Units: one dataset's counts + carried annotations
# ---------------------------------------------------------------------------


def _prefixed_clonotypes(s: pd.Series, label: str) -> pd.Series:
    """Clonotype ids are dataset-local too ("1" in A is not "1" in B): prefixed with the
    dataset label; cells without a clonotype keep their value."""
    from scview.core.vdj import NO_CLONOTYPE

    has = s.notna() & (s.astype(str) != NO_CLONOTYPE)
    return s.where(~has, label + ":" + s.astype(str))


# Per-cell QC metrics the joint run's QC step computes again from the joint counts (scanpy
# calculate_qc_metrics names, and the older n_genes / n_counts): not carried.
_RECOMPUTED_RE = re.compile(
    r"^(log1p_)?(n_genes_by_counts|total_counts(_.+)?|pct_counts_.+|n_genes|n_counts)$")
# Why a column is not carried (``DroppedColumn.reason``).
WHY_RECOMPUTED = "recomputed by the joint run"
WHY_ID = "one value per cell (an id)"
WHY_CLUSTERS = "this dataset's own clusters (kept as source_cluster)"
WHY_LEVELS = "too many values to compare"
WHY_SOME = "a number in only some of the datasets"
WHY_EMPTY = "empty"


def _is_number(s: pd.Series) -> bool:
    return pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s)


def _plain_number(s: pd.Series) -> pd.Series:
    """A numeric column as numpy: ints stay ints when no value is missing, else floats."""
    if s.isna().any():
        return s.astype(np.float64)
    if pd.api.types.is_integer_dtype(s):
        return s.astype(np.int64)
    return s.astype(s.dtype.numpy_dtype if hasattr(s.dtype, "numpy_dtype") else s.dtype)


def obs_choice(adata: ad.AnnData, label: str) -> tuple[dict[str, pd.Series], dict[str, str]]:
    """The obs columns a joint dataset keeps from one dataset, and the ones it leaves out
    with the reason (``carried_obs``); ``scview_`` columns are left out unnamed."""
    obs = adata.obs
    active = adata.uns.get("scview_active_clustering")
    keep: dict[str, pd.Series] = {}
    dropped: dict[str, str] = {}
    for col in obs.columns:
        col = str(col)
        if col.startswith("scview_"):
            continue
        if col == active or col in ("leiden", "louvain"):
            dropped[col] = WHY_CLUSTERS
            continue
        s = obs[col]
        if _is_number(s):
            if _RECOMPUTED_RE.match(col):
                dropped[col] = WHY_RECOMPUTED
            elif s.notna().sum() == 0:
                dropped[col] = WHY_EMPTY
            elif (pd.api.types.is_integer_dtype(s) and len(s) > 1
                  and s.nunique(dropna=True) == len(s)):
                dropped[col] = WHY_ID
            else:
                keep[col] = _plain_number(s)
            continue
        if not (isinstance(s.dtype, pd.CategoricalDtype) or pd.api.types.is_string_dtype(s)
                or pd.api.types.is_bool_dtype(s)):
            continue
        n_levels = s.nunique(dropna=True)
        # Too many levels, or nearly one per cell (barcodes / ids): not an annotation.
        # Clonotypes are ids with many levels by nature, yet worth carrying.
        if n_levels == 0:
            dropped[col] = WHY_EMPTY
            continue
        if col != CLONOTYPE_COL and (n_levels > _MAX_CARRY_CATEGORIES
                                     or (n_levels > 1 and n_levels > 0.5 * len(s))):
            dropped[col] = WHY_ID if n_levels > 0.5 * len(s) else WHY_LEVELS
            continue
        name = "source_dataset" if col == LABEL_COLUMN else col
        keep[name] = s.astype(object).where(s.notna(), None)
        if col == CLONOTYPE_COL:
            keep[name] = _prefixed_clonotypes(keep[name], label)
    if active and active in obs.columns:
        keep["source_cluster"] = (label + ":" + obs[active].astype(str)).astype(object)
    return keep, dropped


def carried_obs(adata: ad.AnnData, label: str) -> pd.DataFrame:
    """Annotations worth keeping in the joint dataset: categorical, text and true/false
    columns, and numbers (author scores, % mito, in_tissue; ints stay ints when none is
    missing). Numbers kept only where every dataset has them (``merge_joint``).

    Cluster ids are dataset-local (cluster "3" in A is unrelated to "3" in B), so
    the active clustering is carried as ``source_cluster`` prefixed with the
    dataset label, and other cluster columns are dropped — the joint run
    clusters afresh. Clonotype ids (``clonotype``, core/vdj.py) are prefixed the
    same way. Left out: per-cell QC metrics the joint run computes again, ids (one
    value per cell), and text with too many values (``obs_choice`` says which and why).
    """
    keep, _ = obs_choice(adata, label)
    return pd.DataFrame(keep, index=adata.obs.index)


@dataclass
class Unit:
    dataset_id: str
    name: str
    label: str
    adata: ad.AnnData | None  # counts in X, carried annotations in obs; None once merged
    counts_source: str
    n_cells: int = 0  # kept once ``adata`` is dropped (``release_units``)
    # obs columns of the dataset not carried, with the reason (``obs_choice``)
    dropped: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.adata is not None:
            self.n_cells = int(self.adata.n_obs)


def release_units(units: dict[str, Unit]) -> None:
    """Drop each unit's counts once the joint dataset holds them: a joint run that kept
    them held every dataset's counts twice for its whole length (8 Visium slides: about
    the size of the joint counts again). Every caller holds these same Unit objects, so
    this frees them for all of them."""
    for u in units.values():
        u.adata = None
    from scview.core import memory_budget

    memory_budget.release_memory()


def build_unit(
    adata: ad.AnnData, *, dataset_id: str, name: str, label: str, fallback_paths: list[str]
) -> Unit:
    counts = recover_counts(adata, fallback_paths)
    var = counts.var
    var.index = var.index.astype(str)
    kept, dropped = obs_choice(adata, label)
    obs = pd.DataFrame(kept, index=adata.obs.index)
    obs.index = obs.index.astype(str)
    unit = ad.AnnData(X=counts.X, obs=obs, var=var)
    if protein.has_protein(adata):  # CITE-seq counts ride along (same cells, same order)
        unit.obsm[protein.COUNTS_KEY] = adata.obsm[protein.COUNTS_KEY].set_axis(obs.index)
    if spatial.has_spatial(adata):  # tissue coordinates and images ride along (merge_units keeps them)
        spatial.adopt(adata)
        unit.obsm[spatial.KEY] = np.asarray(adata.obsm[spatial.KEY], dtype=np.float64)
        unit.obs[spatial.LIBRARY_COLUMN] = adata.obs[spatial.LIBRARY_COLUMN].to_numpy()
        unit.obs[spatial.LIBRARY_COLUMN] = unit.obs[spatial.LIBRARY_COLUMN].astype(
            pd.CategoricalDtype(adata.obs[spatial.LIBRARY_COLUMN].cat.categories))
        for col in ("array_row", "array_col"):
            if col in adata.obs.columns:
                unit.obs[col] = adata.obs[col].to_numpy()
        unit.uns[spatial.KEY] = _shown_images_only(adata.uns[spatial.KEY])
    return Unit(dataset_id, name, label, unit, counts.source, dropped=dropped)


def _shown_images_only(libraries: Any) -> dict:
    """``uns["spatial"]`` with each library's images cut to those scView shows
    (``spatial.RESOLUTIONS``). CELLxGENE Visium files also carry the full-resolution
    image (7,874 x 7,874 x 3 per Kuppe slide, 186 MB): 1.5 GB of 8 slides held twice
    during a joint run, never drawn."""
    out = {}
    for lib, entry in dict(libraries).items():
        if isinstance(entry, dict) and isinstance(entry.get("images"), dict):
            entry = dict(entry, images={k: v for k, v in entry["images"].items()
                                        if k in spatial.RESOLUTIONS})
        out[lib] = entry
    return out


def default_name(labels: list[str]) -> str:
    """The joint dataset's name when none is given: the labels, or the first three and how
    many more (28 labels made a name nobody could read)."""
    if len(labels) <= 4:
        return "joint: " + " + ".join(labels)
    return f"joint: {' + '.join(labels[:3])} + {len(labels) - 3} more"


def unique_labels(names: list[str]) -> list[str]:
    """Distinct, non-empty labels for the ``dataset`` column."""
    out: list[str] = []
    for n in names:
        base = (n or "").strip() or "dataset"
        label, k = base, 2
        while label in out:
            label, k = f"{base}_{k}", k + 1
        out.append(label)
    return out


# ---------------------------------------------------------------------------
# Plan (read-only, for the UI)
# ---------------------------------------------------------------------------


class DatasetCheck(BaseModel):
    id: str
    name: str
    label: str
    n_cells: int = 0
    n_genes: int = 0
    counts_source: str | None = None
    columns: list[str] = Field(default_factory=list)
    error: str | None = None


class DroppedColumn(BaseModel):
    """An obs column some dataset has that the joint dataset will not."""

    column: str
    reason: str  # WHY_* above
    datasets: list[str] = Field(default_factory=list)  # labels of the datasets that have it


def numbers_in_some(units: dict[str, Unit]) -> dict[str, list[str]]:
    """Numeric obs columns kept in some units but not in every one: column -> the labels
    of the units that have it. ``merge_joint`` leaves them out (no number for the other
    datasets' cells)."""
    have: dict[str, list[str]] = {}
    for lab, u in units.items():
        for c in u.adata.obs.columns:
            if _is_number(u.adata.obs[c]):
                have.setdefault(str(c), []).append(lab)
    return {c: labs for c, labs in have.items() if len(labs) < len(units)}


def dropped_columns(units: dict[str, Unit]) -> list[DroppedColumn]:
    """Every obs column the joint dataset leaves out, once, with the reason and the
    datasets that have it (the plan's list)."""
    out: dict[str, DroppedColumn] = {}
    for lab, u in units.items():
        for col, why in u.dropped.items():
            out.setdefault(col, DroppedColumn(column=col, reason=why)).datasets.append(lab)
    for col, labs in numbers_in_some(units).items():
        out.setdefault(col, DroppedColumn(column=col, reason=WHY_SOME)).datasets.extend(labs)
    order = [WHY_SOME, WHY_RECOMPUTED, WHY_ID, WHY_LEVELS, WHY_CLUSTERS, WHY_EMPTY]
    return sorted(out.values(), key=lambda d: (order.index(d.reason) if d.reason in order else 99,
                                               d.column.lower()))


class IntegrationPlan(BaseModel):
    datasets: list[DatasetCheck] = Field(default_factory=list)
    merge: MergePlan | None = None
    shared_columns: list[str] = Field(default_factory=list)
    # obs columns the joint dataset leaves out, with why (``dropped_columns``)
    dropped_columns: list[DroppedColumn] = Field(default_factory=list)
    est_cells: int = 0
    blocking: list[str] = Field(default_factory=list)
    # Design notes (not blocking): e.g. each dataset holds one condition.
    notes: list[str] = Field(default_factory=list)
    # The joint run's memory and time (``joint_preflight``'s verdict, as the other preflight notes).
    preflight: dict[str, Any] | None = None
    ok: bool = False


def load_units(
    dm: "DatasetManager", dataset_ids: list[str], labels: list[str] | None = None
) -> tuple[list[DatasetCheck], dict[str, Unit]]:
    """Build a Unit per dataset; failures are reported per dataset, not raised.

    Reads files directly rather than through the manager's LRU cache, so
    checking several datasets doesn't evict the one the user has open.
    """
    names = []
    for did in dataset_ids:
        info = dm.get_dataset_info(did) or {}
        names.append(str(info.get("name") or did))
    given = labels if labels and len(labels) == len(dataset_ids) else names
    final = unique_labels(list(given))

    checks: list[DatasetCheck] = []
    units: dict[str, Unit] = {}
    for did, name, label in zip(dataset_ids, names, final):
        check = DatasetCheck(id=did, name=name, label=label)
        checks.append(check)
        paths = dm.h5ad_candidates(did)
        if not paths:
            check.error = "dataset not found or has no data file"
            continue
        from scview.core import large_mode

        if large_mode.file_is_large(paths[0]):  # never read whole: its matrices stay on disk
            check.error = str(large_mode.LargeDatasetError(
                "A joint embedding", "it reads each dataset's counts into memory"))
            continue
        opened = None
        try:
            cached = dm.get_dataset(did)
            adata = cached.adata if cached is not None else (opened := _open_lazily(paths[0]))
            unit = build_unit(adata, dataset_id=did, name=name, label=label,
                              fallback_paths=paths[1:])
        except IntegrationError as e:
            check.error = str(e)
            continue
        except Exception as e:  # noqa: BLE001 — report, don't crash the plan
            logger.warning("Could not prepare %s for integration: %s", did, e, exc_info=True)
            check.error = f"could not read dataset: {e}"
            continue
        finally:
            if opened is not None:
                opened.file.close()
        check.n_cells, check.n_genes = int(unit.adata.n_obs), int(unit.adata.n_vars)
        check.counts_source = unit.counts_source
        check.columns = [str(c) for c in unit.adata.obs.columns]
        units[label] = unit
    return checks, units


def read_need(dm: "DatasetManager", dataset_ids: list[str]) -> int:
    """Peak memory of ``load_units``, from the file headers: each dataset's counts as the
    float32 CSR its unit keeps, and the largest closed dataset's open (its tables, layers
    and embeddings, one dataset at a time) with its counts as read, before they are cut to
    the unit's copy. Datasets it cannot read count nothing: the plan reports those. Two
    Kang copies: 279 MiB over what was open, estimated 251."""
    from scview.core import large_mode

    counts, opening = 0, 0
    for did in dataset_ids:
        paths = dm.h5ad_candidates(did)
        if not paths or large_mode.file_is_large(paths[0]):
            continue
        try:
            h = preflight.read_header(paths[0], "h5ad")
            one = preflight._lt_counts_bytes(h)
            counts += one
            if dm.get_dataset(did) is None:
                h.backed = True  # ``_open_lazily``
                opening = max(opening, preflight.estimate_load(h, ingest=False).peak_bytes + one)
        except Exception as e:  # noqa: BLE001 - never block a plan on the estimate itself
            logger.warning("No read estimate for %s: %s", did, e)
    return int(counts + opening)


def build_plan(
    checks: list[DatasetCheck],
    units: dict[str, Unit],
    *,
    max_datasets: int,
    max_cells: int,
) -> IntegrationPlan:
    plan = IntegrationPlan(datasets=checks)
    blocking = plan.blocking
    if len(checks) < 2:
        blocking.append("Pick at least two datasets to integrate.")
    if len(checks) > max_datasets:
        blocking.append(f"At most {max_datasets} datasets can be integrated at once.")
    if len({c.id for c in checks}) != len(checks):
        blocking.append("The same dataset was picked twice.")
    for c in checks:
        if c.error:
            blocking.append(f"{c.name}: {c.error}")
        elif c.n_cells < MIN_CELLS_PER_DATASET:
            blocking.append(
                f"{c.name}: only {c.n_cells} cells (need at least {MIN_CELLS_PER_DATASET})."
            )

    plan.est_cells = sum(c.n_cells for c in checks)
    if plan.est_cells > max_cells:
        blocking.append(
            f"The joint dataset would have {plan.est_cells:,} cells; the limit is {max_cells:,}."
        )

    if len(units) >= 2:
        plan.merge = plan_merge({lab: u.adata for lab, u in units.items()})
        col_sets = [set(u.adata.obs.columns) - {"source_cluster"} for u in units.values()]
        plan.shared_columns = sorted(set.intersection(*col_sets))
        plan.dropped_columns = dropped_columns(units)
        m = plan.merge
        genes = m.reconciliation.overlap_after if (
            m.reconciliation and m.reconciliation.feasible) else m.intersection
        if genes == 0:
            blocking.append("These datasets share no genes, so they can't be embedded together.")

    if len(units) >= 2:
        plan.notes = confounding_notes(units)
    plan.ok = not blocking
    return plan


def _quoted(names: list[str]) -> str:
    q = [f"'{n}'" for n in names]
    return q[0] if len(q) == 1 else ", ".join(q[:-1]) + f" and {q[-1]}"


def confounding_notes(units: dict[str, "Unit"]) -> list[str]:
    """When each dataset holds one condition, joining them corrects the condition
    away with the datasets: say so before the user reads the mixing as success.
    Columns that split the datasets the same way (CELLxGENE's ``disease`` and
    ``disease_ontology_term_id``) are one finding, so one note naming them all."""
    import pandas as pd

    from scview.core.confounding import check, condition_columns

    frames = [u.adata.obs.assign(**{"__dataset": lab}) for lab, u in units.items()]
    shared = set.intersection(*(set(f.columns) for f in frames))
    obs = pd.concat([f[sorted(shared)] for f in frames], ignore_index=True)
    findings: dict[tuple, list[str]] = {}
    kinds: dict[tuple, str] = {}
    for cond in condition_columns(obs):
        hit = check(obs, "__dataset", cond)
        if hit is None:
            continue
        # Which datasets share a group: the same partition is the same finding.
        main = obs.groupby("__dataset", observed=True)[cond].agg(
            lambda s: s.astype(str).value_counts().index[0] if s.notna().any() else None)
        key = (hit.kind, tuple(pd.factorize(main.sort_index().to_numpy())[0]))
        findings.setdefault(key, []).append(cond)
        kinds[key] = hit.kind
    notes = []
    for key, cols in findings.items():
        shape = "one" if kinds[key] == "confounded" else "a single"
        what = _quoted(cols)
        group = f"{what} group" if len(cols) == 1 else f"group of {what} (they split the datasets alike)"
        notes.append(
            f"Each dataset holds {shape} {group}, so batch correction removes that difference "
            "from the joint embedding along with the dataset difference: a high mixing score is "
            "expected, and cells of different groups can share clusters. Expression values are "
            "not corrected, so DE between those groups within a cell type still works (use "
            "pseudobulk across donors)."
        )
    return notes[:2]


# ---------------------------------------------------------------------------
# Merge + persist
# ---------------------------------------------------------------------------


def merge_joint(
    units: dict[str, Unit], *, join: MergeJoin = MergeJoin.inner, apply_reconciliation: bool = True,
    copy: bool = True,
) -> ad.AnnData:
    """Concatenate units into one counts AnnData labelled by ``dataset``. ``copy=False``:
    the units' AnnData are changed in place (a caller that drops them after)."""
    adatas = {lab: u.adata for lab, u in units.items()}
    # Pad every unit to the union of annotation columns: concatenating with an
    # inner gene join would otherwise also drop any column one dataset lacks.
    # A number only some datasets have is left out (the plan lists it): the other cells
    # would hold no value, and padding turned the column into text.
    partial = numbers_in_some(units)
    for a in adatas.values():
        a.obs = a.obs.drop(columns=[c for c in partial if c in a.obs.columns])
    numbers = {str(c) for c in next(iter(adatas.values())).obs.columns
               if all(c in a.obs.columns and _is_number(a.obs[c]) for a in adatas.values())}
    cols = list(dict.fromkeys(c for a in adatas.values() for c in a.obs.columns))
    for a in adatas.values():
        for c in cols:
            if c not in a.obs.columns:
                a.obs[c] = None
        a.obs = a.obs[cols]
    recon = plan_merge(adatas).reconciliation if apply_reconciliation else None
    protein.common_panel(list(adatas.values()))  # proteins all of them measured, or none
    joint = merge_units(adatas, join=join, sample_label=LABEL_COLUMN, reconciliation=recon, copy=copy)
    for c in cols:
        if c in numbers:  # ints stay ints when no dataset misses a value (``_plain_number``)
            joint.obs[c] = _plain_number(joint.obs[c])
        else:
            joint.obs[c] = joint.obs[c].astype("category")
    protein.renormalise(joint)
    return joint


def record_integration_provenance(
    joint: ad.AnnData, units: dict[str, Unit], *, join: MergeJoin, method: str,
    apply_reconciliation: bool,
) -> None:
    """Source block + an ``integrate`` history entry. Never raises."""
    merged_from = [
        {"dataset_id": u.dataset_id, "name": u.name, "label": u.label,
         "n_cells": u.n_cells, "counts_source": u.counts_source}
        for u in units.values()
    ]
    try:
        provenance.init_source(
            joint,
            origin="integrated",
            original_filename=" + ".join(u.name for u in units.values()),
            fmt="joint",
            merged_from=merged_from,
            merge={"join": join.value, "label_column": LABEL_COLUMN,
                   "apply_reconciliation": apply_reconciliation, "method": method},
        )
        provenance.record_step(
            joint, step="integrate", tool="scview.integration",
            params={"datasets": [u.dataset_id for u in units.values()], "join": join.value,
                    "method": method, "label_column": LABEL_COLUMN},
        )
    except Exception as e:  # provenance must never block integration
        logger.warning("Could not record integration provenance: %s", e)


def persist_joint(
    data_dir: Path, dataset_id: str, name: str, joint: ad.AnnData, units: dict[str, Unit]
) -> Path:
    """Write the merged counts (read-only) + the metadata sidecar the picker reads."""
    ing = data_dir / "ingested" / dataset_id
    ing.mkdir(parents=True, exist_ok=True)
    path = ing / f"{_safe_name(name)}.h5ad"
    joint.write_h5ad(path, compression="gzip")
    _make_readonly(path)

    up = data_dir / "uploads" / dataset_id
    up.mkdir(parents=True, exist_ok=True)
    meta = {
        "id": dataset_id,
        "name": name,
        "filename": path.name,
        "extension": ".h5ad",
        "size_bytes": path.stat().st_size,
        "status": "ready",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "n_cells": int(joint.n_obs),
        "n_genes": int(joint.n_vars),
        "available_embeddings": [],
        "obs_columns": [],
        "source": "integrated",
        "integrated_from": [
            {"id": u.dataset_id, "name": u.name, "label": u.label} for u in units.values()
        ],
    }
    with open(up / "metadata.json", "w") as fh:
        json.dump(meta, fh, indent=2)
    return path


# ---------------------------------------------------------------------------
# Batch mixing
# ---------------------------------------------------------------------------

# Neighbours per cell for the mixing score.
MIXING_K = 30
# Cells the score is averaged over; a random subset above this keeps it to
# seconds at the 150k-cell cap (standard error ~0.005 at 10k cells).
MIXING_MAX_SCORED = 10_000

# Embedding scored as "corrected": the corrected PCs held, if any (a BBKNN graph is
# scored from its own neighbour lists instead).
_CORRECTED_KEYS = ("X_pca_harmony", "X_scanorama")


def _neighbours(X: np.ndarray, query: np.ndarray, k: int) -> np.ndarray:
    """Indices of each query cell's k nearest neighbours, itself excluded."""
    from sklearn.neighbors import NearestNeighbors

    idx = NearestNeighbors(n_neighbors=k + 1).fit(X).kneighbors(X[query], return_distance=False)
    # Drop the cell itself; among duplicate points it need not come first.
    not_self = idx != query[:, None]
    order = np.argsort(~not_self, axis=1, kind="stable")
    return np.take_along_axis(idx, order, axis=1)[:, :k]


def _cross_dataset_fraction(
    X: np.ndarray, codes: np.ndarray, query: np.ndarray, k: int
) -> np.ndarray:
    """Per query cell, the fraction of its k nearest neighbours from another dataset."""
    idx = _neighbours(X, query, k)
    return (codes[idx] != codes[query][:, None]).mean(axis=1)


def _ilisi(idx: np.ndarray, codes: np.ndarray, n_sets: int) -> np.ndarray:
    """Per query cell, the inverse Simpson index of its neighbours' datasets: the
    effective number of datasets around it (1 = one dataset; uniform kNN weights,
    a simplification of LISI's perplexity-weighted neighbourhood)."""
    counts = np.stack([(codes[idx] == d).sum(axis=1) for d in range(n_sets)], axis=1)
    p = counts / idx.shape[1]
    return 1.0 / (p ** 2).sum(axis=1)


# Cell-type-like columns that, shared by every dataset, show whether correction
# kept cell types apart (bio-conservation); most specific first.
_CONSERVATION_RE = re.compile(r"cell[ ._-]?type|celltype|annotation|^labels?$|^cluster$", re.I)
OVERCORRECTION_DROP = 0.10  # neighbour label agreement lost to correction that warrants a warning


def _conservation_label(adata: ad.AnnData, label_column: str) -> tuple[str | None, str]:
    """(column, kind): a shared cell-type column ("shared"), else each dataset's
    own clusters, compared within the dataset ("within_dataset")."""
    obs = adata.obs
    for col in obs.columns:
        name = str(col)
        if (name != label_column and not name.startswith("scview_") and _CONSERVATION_RE.search(name)
                and obs[col].notna().mean() >= 0.9 and 2 <= obs[col].nunique() <= 100):
            return name, "shared"
    if "source_cluster" in obs.columns:
        return "source_cluster", "within_dataset"
    return None, ""


def _label_agreement(idx: np.ndarray, labels: np.ndarray, codes: np.ndarray,
                     query: np.ndarray, within_dataset: bool) -> np.ndarray:
    """Per query cell, the share of its neighbours with its label (NaN when it
    has no neighbour to compare: e.g. none from its own dataset)."""
    same = labels[idx] == labels[query][:, None]
    if not within_dataset:
        return same.mean(axis=1)
    mine = codes[idx] == codes[query][:, None]
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(mine.sum(axis=1) > 0, (same & mine).sum(axis=1) / mine.sum(axis=1), np.nan)


def _corrected_source(adata: ad.AnnData) -> str | None:
    """What "corrected" is scored on: an obsm key of corrected PCs, "graph" for a BBKNN
    neighbour graph, or None (nothing corrected)."""
    from scview.core.batch_methods import is_bbknn_graph

    key = next((k for k in _CORRECTED_KEYS if k in adata.obsm), None)
    if key is not None:
        return key
    nb = adata.uns.get("neighbors") or {}
    if is_bbknn_graph(adata) and nb.get("connectivities_key", "connectivities") in adata.obsp:
        return "graph"
    return None


def batch_mixing(
    adata: ad.AnnData,
    *,
    label_column: str = LABEL_COLUMN,
    k: int = MIXING_K,
    max_scored: int = MIXING_MAX_SCORED,
    seed: int = 0,
    neighbours: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    """How well the source datasets mix, in the uncorrected and corrected PCA.

    ``neighbours`` gives a space's precomputed neighbour lists instead of a kNN search
    in an embedding: {space name: callable(query, k) -> (len(query), k') indices}. A
    BBKNN graph (no corrected embedding) is scored that way by default, from each
    cell's strongest graph connections (``batch_methods.graph_neighbours``, k' at most
    the graph's neighbours per cell); ``k_by_space`` says the k each space used.

    For each cell, the fraction of its ``k`` nearest neighbours that come from a
    different source dataset, averaged over cells. Were the datasets perfectly
    mixed, a cell of dataset d would see others at rate (n - n_d) / (n - 1), so
    ``ideal`` is that rate averaged over the scored cells (~1 - sum of squared
    proportions). Kang CTRL vs STIM, scored by hand: 0.107 uncorrected, 0.478
    after Harmony, ideal 0.499. None when there is nothing to score.
    """
    if label_column not in adata.obs.columns:
        return None
    neighbours = dict(neighbours or {})
    spaces = [("uncorrected", "X_pca")] if "X_pca" in adata.obsm else []
    corrected = _corrected_source(adata)
    if corrected == "graph" and "corrected" not in neighbours:
        from scview.core.batch_methods import graph_neighbours

        nb = adata.uns["neighbors"]
        conn = adata.obsp[nb.get("connectivities_key", "connectivities")]
        neighbours["corrected"] = lambda q, kk: graph_neighbours(conn, q, kk)
    if corrected is not None:
        spaces.append(("corrected", corrected))
    elif "corrected" in neighbours:
        spaces.append(("corrected", "neighbours"))
    n = int(adata.n_obs)
    names, codes = np.unique(adata.obs[label_column].astype(str).to_numpy(), return_inverse=True)
    if not spaces or len(names) < 2 or n < 3:
        return None
    k = min(k, n - 1)
    rng = np.random.default_rng(seed)
    query = np.sort(rng.choice(n, size=min(n, max_scored), replace=False))
    sizes = np.bincount(codes, minlength=len(names))
    ideal = (n - sizes) / (n - 1)
    q_codes = codes[query]

    out: dict[str, Any] = {
        "k": k,
        "label_column": label_column,
        "n_cells": n,
        "n_scored": int(query.size),
        "ideal": round(float(ideal[q_codes].mean()), 4),
        "spaces": dict(spaces),
        "k_by_space": {},
        "uncorrected": None,
        "corrected": None,
        "datasets": [
            {"label": str(lab), "n_cells": int(sizes[i]), "ideal": round(float(ideal[i]), 4),
             "uncorrected": None, "corrected": None}
            for i, lab in enumerate(names)
        ],
    }
    shares = sizes / n
    out["ilisi"] = {"ideal": round(float(1.0 / (shares ** 2).sum()), 4),
                    "uncorrected": None, "corrected": None}
    cons_col, cons_kind = _conservation_label(adata, label_column)
    cons_labels = (adata.obs[cons_col].astype(str).to_numpy() if cons_col else None)
    agreement: dict[str, np.ndarray] = {}
    for name, key in spaces:
        if name in neighbours:
            idx = np.asarray(neighbours[name](query, k))
        else:
            idx = _neighbours(np.asarray(adata.obsm[key], dtype=np.float32), query, k)
        out["k_by_space"][name] = int(idx.shape[1])
        frac = (codes[idx] != q_codes[:, None]).mean(axis=1)
        out[name] = round(float(frac.mean()), 4)
        for i, row in enumerate(out["datasets"]):
            mine = frac[q_codes == i]
            row[name] = round(float(mine.mean()), 4) if mine.size else None
        out["ilisi"][name] = round(float(_ilisi(idx, codes, len(names)).mean()), 4)
        if cons_labels is not None:
            agreement[name] = _label_agreement(idx, cons_labels, codes, query,
                                               cons_kind == "within_dataset")
    out["conservation"] = _conservation(cons_col, cons_kind, cons_labels, query, agreement)
    out["overcorrection"] = _overcorrection_note(out["conservation"])
    return out


def _conservation(col, kind, labels, query, agreement) -> dict[str, Any] | None:
    """Mean label agreement per space, and the labels that lost the most."""
    if col is None or not agreement:
        return None
    res: dict[str, Any] = {"label_column": col, "kind": kind, "uncorrected": None,
                           "corrected": None, "labels": []}
    for name, vals in agreement.items():
        res[name] = round(float(np.nanmean(vals)), 4) if np.isfinite(vals).any() else None
    if "uncorrected" in agreement and "corrected" in agreement and kind == "shared":
        q_labels = labels[query]
        rows = []
        for lab in np.unique(q_labels):
            m = q_labels == lab
            if m.sum() < 20:
                continue
            before = float(np.nanmean(agreement["uncorrected"][m]))
            after = float(np.nanmean(agreement["corrected"][m]))
            rows.append({"label": str(lab), "n_scored": int(m.sum()),
                         "uncorrected": round(before, 4), "corrected": round(after, 4)})
        rows.sort(key=lambda r: r["corrected"] - r["uncorrected"])
        res["labels"] = rows[:5]
    return res


def _overcorrection_note(cons: dict[str, Any] | None) -> str | None:
    """A warning when correction cost cell-type separation, naming who lost most."""
    if not cons or cons.get("uncorrected") is None or cons.get("corrected") is None:
        return None
    before, after = cons["uncorrected"], cons["corrected"]
    if before - after < OVERCORRECTION_DROP:
        return None
    worst = [r["label"] for r in cons.get("labels", [])
             if r["uncorrected"] - r["corrected"] >= OVERCORRECTION_DROP][:3]
    what = (f"its '{cons['label_column']}' label" if cons["kind"] == "shared"
            else "its own dataset's cluster")
    return (f"After correction a cell's nearest neighbours share {what} {after:.0%} of the time, "
            f"down from {before:.0%}: the correction may be merging distinct cell types"
            + (f" (most for {', '.join(worst)})." if worst else "."))


def record_batch_mixing(adata: ad.AnnData) -> dict[str, Any] | None:
    """Score batch mixing and store it in ``uns``. Never raises."""
    try:
        mixing = batch_mixing(adata)
    except Exception as e:  # a missing score must never fail an integration
        logger.warning("Could not score batch mixing: %s", e)
        return None
    if mixing is not None:
        adata.uns[BATCH_MIXING_KEY] = json.dumps(mixing)
    return mixing


def read_batch_mixing(adata: ad.AnnData) -> dict[str, Any] | None:
    """The stored score, or None (not a joint dataset, or scored before it existed)."""
    raw = adata.uns.get(BATCH_MIXING_KEY)
    if raw is None:
        return None
    try:
        data = json.loads(raw) if isinstance(raw, str) else dict(raw)
    except (TypeError, ValueError):
        return None
    return data if isinstance(data, dict) else None


# ---------------------------------------------------------------------------
# Joint processing
# ---------------------------------------------------------------------------


def joint_steps(method: str) -> list[str]:
    steps = ["qc_metrics", "normalization", "log_transform", "highly_variable_genes",
             "scaling", "pca"]
    if method != METHOD_NONE:
        steps += ["uncorrected_embedding", "batch_correction"]
    return steps + ["neighbors", "clustering", "embeddings", "marker_genes"]


def joint_params(method: str, overrides: dict[str, Any] | None = None) -> PipelineParams:
    params = PipelineParams(
        hvg_batch_key=LABEL_COLUMN,
        batch_key="" if method == METHOD_NONE else LABEL_COLUMN,
        batch_method=METHOD_HARMONY if method == METHOD_NONE else method,
    )
    for key, value in (overrides or {}).items():
        if key in TUNABLE_PARAMS and value is not None:
            setattr(params, key, type(getattr(params, key))(value))
    return params


# ---------------------------------------------------------------------------
# Memory: the joint run's estimate (the plan's note, the run's reservation)
# ---------------------------------------------------------------------------

# Merging over the units held: concat cuts each dataset to the shared genes and stacks the
# cuts (the units are not copied first: ``merge_joint(copy=False)``). 8 Kuppe Visium slides
# (39,936 spots x 35,477 genes, joint counts 0.58 GB by the header): +0.65 GB measured in a
# 6 GB container; 1.5 copies leaves room for its error.
MERGE_COPIES = 1.5
MERGE_LABEL = "merging the datasets"
# What a stopped joint run says it kept, and how to make it fit (``run_watchdog.memory_stop``).
JOINT_NOTHING_KEPT = ("Nothing was kept: the joint dataset was removed, and the source datasets "
                      "are unchanged.")
MERGE_NOTHING_KEPT = "Nothing was kept: the source datasets are unchanged."
JOINT_FIX = ("Raise SCVIEW_MEMORY_GB in .env (then restart scView), or integrate fewer or "
             "smaller datasets.")


def _nested_bytes(d: Any) -> int:
    if isinstance(d, dict):
        return sum(_nested_bytes(v) for v in d.values())
    return int(getattr(d, "nbytes", 0) or 0)


def _unit_bytes(a: ad.AnnData) -> int:
    """Memory one unit holds: its counts, obs, var (the gene names: 28 units hold 28 copies),
    coordinates and tissue images. 28 units of 3,000 cells x 8,000 genes held 448 MiB in a
    fresh process against 460 estimated (the plan's read: 448 measured, 448 estimated)."""
    X = a.X
    n = (X.data.nbytes + X.indices.nbytes + X.indptr.nbytes) if sp.issparse(X) else int(np.asarray(X).nbytes)
    n += int(a.obs.memory_usage(deep=True).sum()) + int(a.var.memory_usage(deep=True).sum())
    n += sum(int(getattr(v, "nbytes", 0) or 0) for v in a.obsm.values())
    return int(n + _nested_bytes(a.uns))


def joint_header(units: dict[str, Unit], n_genes: int) -> preflight.Header:
    """The joint counts as the run will hold them: every unit's cells over ``n_genes``
    genes, each unit's stored values scaled by the share of its genes kept (an inner join
    keeps the shared genes; Visium slides of one study keep 97%)."""
    n = sum(u.adata.n_obs for u in units.values())
    nnz = 0
    for u in units.values():
        share = min(1.0, n_genes / max(1, u.adata.n_vars))
        stored = u.adata.X.nnz if sp.issparse(u.adata.X) else u.adata.X.size
        nnz += int(stored * share)
    h = preflight.Header("h5ad", n, n_genes, [preflight.Matrix("X", n, n_genes, True, nnz, "float32")])
    h.obs_columns = len({c for u in units.values() for c in u.adata.obs.columns}) + 1
    h.obsm_bytes = sum(int(getattr(v, "nbytes", 0) or 0)
                       for u in units.values() for v in u.adata.obsm.values())
    h.uns_bytes = sum(_nested_bytes(u.adata.uns) for u in units.values())
    return h


@dataclass
class JointEstimate:
    """Memory of a joint run, as growth over what the run holds when it starts (the
    units' counts, ``units_bytes``): merging, then the pipeline on the joint counts once
    the units are dropped (``release_units``)."""

    units_bytes: int
    merge_bytes: int  # growth over the units while merging
    run: preflight.RunEstimate  # the pipeline on the joint counts (they are its start)
    header: preflight.Header

    def step_needs(self) -> dict[str, int]:
        """Per stage ("merge", then each step and the save), its growth over the units held."""
        out = {"merge": self.merge_bytes}
        out.update({e.step: max(0, e.peak_bytes - self.units_bytes) for e in self.run.steps})
        return out

    @property
    def need(self) -> int:
        """What the run grows by at its peak, from the units held."""
        return max(self.step_needs().values(), default=0)

    @property
    def peak_bytes(self) -> int:
        return self.units_bytes + self.need

    @property
    def peak_step(self) -> str:
        needs = self.step_needs()
        return max(needs, key=lambda k: needs[k]) if needs else ""


def estimate_joint(units: dict[str, Unit], n_genes: int, *, method: str = METHOD_HARMONY,
                   overrides: dict[str, Any] | None = None) -> JointEstimate:
    """The joint run's memory and time from the units (read for the plan) and the joint
    gene count (``joint_genes``)."""
    h = joint_header(units, n_genes)
    run = preflight.estimate_run(h, steps=joint_steps(method), params=joint_params(method, overrides),
                                 warm=preflight.process_warm())
    held = sum(_unit_bytes(u.adata) for u in units.values())
    return JointEstimate(held, int(MERGE_COPIES * preflight.resident_bytes(h)), run, h)


def joint_genes(plan: IntegrationPlan, join: MergeJoin = MergeJoin.inner) -> int:
    """Genes of the joint dataset: the shared ones (after reconciliation), or every one."""
    m = plan.merge
    if m is None:
        return 0
    if join == MergeJoin.outer:
        return int(m.union)
    return int(m.reconciliation.overlap_after if m.reconciliation and m.reconciliation.feasible
               else m.intersection)


def _stage_label(step: str) -> str:
    return "merging" if step == "merge" else preflight.STEP_LABELS.get(step, step)


def joint_preflight(est: JointEstimate, avail: preflight.Available, cached: preflight.Cached) -> dict:
    """The plan's memory and time note (``preflight.make_verdict``), and ``blocking``: a
    sentence when the run cannot fit in the limit even with every other dataset closed.
    The plan holds the units while this is measured: they count as free."""
    if avail.available_bytes is not None and avail.used_bytes is not None:
        units = min(est.units_bytes, avail.used_bytes)
        avail = preflight.Available(avail.limit_bytes, avail.used_bytes - units,
                                    avail.available_bytes + units, avail.limited, avail.source)
    stage = _stage_label(est.peak_step)
    need = est.peak_bytes
    about = preflight._about(need)
    breakdown = [("The datasets' counts, read", est.units_bytes),
                 ("While merging them", est.units_bytes + est.merge_bytes),
                 ("The joint counts", est.run.start_bytes),
                 (f"At the peak ({stage})", need)]
    h = est.header
    notes = [f"Estimated from the joint shape: {h.n_cells:,} cells x {h.n_genes:,} genes, "
             f"{h.x.nnz:,} stored values."]
    v = preflight.make_verdict(need=need, peak=need, stage=stage, seconds=est.run.seconds,
                               avail=avail, cached=cached,
                               lead=f"{about[0].upper()}{about[1:]} at its peak ({stage})",
                               breakdown=breakdown, notes=notes, alternatives=[])
    blocking = None
    if avail.limited and avail.available_bytes is not None and avail.limit_bytes is not None:
        room = avail.available_bytes + cached.bytes
        if need * (1 - preflight.ERROR_BAND) > room:
            gib = 1024**3
            blocking = (f"The joint run needs about {need / gib:.1f} GB at its peak ({stage}), and scView "
                        f"would have about {room / gib:.1f} GB of its {avail.limit_bytes / gib:.1f} GB "
                        f"limit even with every other dataset closed. {JOINT_FIX}")
    return {"verdict": v, "blocking": blocking}


def joint_before_step(merge_need: int):
    """The joint run's pre-step check (``RunControl.before_step``): the merge's estimated
    growth before merging, ``default_before_step``'s check before each pipeline step, in
    the joint run's words (no step to leave out; the joint dataset is removed)."""
    from scview.core import run_watchdog

    def check(step: str, adata: Any, params: Any) -> str | None:
        if step == "merge":
            return run_watchdog.memory_stop(step, merge_need, label=MERGE_LABEL,
                                            saved=MERGE_NOTHING_KEPT, fix=JOINT_FIX)
        reason = run_watchdog.log_only_stop(step, adata)
        if reason:
            return reason
        return run_watchdog.memory_stop(step, preflight.step_peak_for(adata, step, params),
                                        saved=JOINT_NOTHING_KEPT, fix=JOINT_FIX)

    return check


def run_integration_streamed(
    units: dict[str, Unit],
    *,
    dataset_id: str,
    name: str,
    data_dir: Path,
    output_path_for,
    join: MergeJoin = MergeJoin.inner,
    method: str = METHOD_HARMONY,
    apply_reconciliation: bool = True,
    overrides: dict[str, Any] | None = None,
    control: RunControl | None = None,
    on_merged: Callable[[], None] | None = None,
) -> Generator[tuple[str, dict[str, Any]], None, None]:
    """Merge, persist and process; yields (event, data) for SSE progress.

    ``output_path_for(source_path)`` returns the derived output path, so the
    caller's DatasetManager decides where processed results live. ``control`` (the
    joint run's, ``api/v1/integration.py``) is checked before merging too, as the stage
    "merge" (``joint_before_step``): a stop there ends with ``pipeline_complete`` of that
    status before anything is written. The units' counts are dropped once merged
    (``release_units``), then ``on_merged()`` is called: the pipeline holds the joint
    counts only.
    """
    yield ("stage", {"stage": "merge", "message": f"Merging {len(units)} datasets",
                     "dataset_id": dataset_id})
    halt = control.halt_before("merge", None, None) if control is not None else None
    if halt is not None:
        yield ("pipeline_complete", {"status": halt[0], "stop_reason": halt[1], "steps_run": []})
        return
    joint = merge_joint(units, join=join, apply_reconciliation=apply_reconciliation, copy=False)
    release_units(units)
    if on_merged is not None:
        on_merged()
    record_integration_provenance(joint, units, join=join, method=method,
                                  apply_reconciliation=apply_reconciliation)

    yield ("stage", {"stage": "persist", "message": "Saving the joint dataset"})
    src = persist_joint(data_dir, dataset_id, name, joint, units)
    yield ("created", {"dataset_id": dataset_id, "name": name,
                       "n_cells": int(joint.n_obs), "n_genes": int(joint.n_vars)})

    steps = joint_steps(method)
    params = joint_params(method, overrides)
    # Score once both PCA spaces exist, so the pipeline's final write saves it.
    score_after = "pca" if method == METHOD_NONE else "batch_correction"
    for event, data in run_pipeline_streamed(
        joint, steps, params, output_path=str(output_path_for(str(src))), control=control
    ):
        # The pipeline's own "complete" isn't the end of the integration.
        yield ("pipeline_complete" if event == "complete" else event, data)
        if event in ("step_done", "step_error") and data.get("step") == score_after:
            yield ("substep", {"step": score_after, "message": "Scoring batch mixing"})
            record_batch_mixing(joint)
