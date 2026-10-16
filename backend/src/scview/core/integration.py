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
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Generator

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp
from pydantic import BaseModel, Field

from scview.core import provenance
from scview.core.ingestion.merge import MergeJoin, MergePlan, merge_units, plan_merge
from scview.core.ingestion.session import _make_readonly, _safe_name
from scview.core.pipeline import PipelineParams, looks_like_counts, run_pipeline_streamed
from scview.core.uns_keys import BATCH_MIXING_KEY

if TYPE_CHECKING:
    from scview.core.dataset_manager import DatasetManager

logger = logging.getLogger(__name__)

# obs column naming each cell's source dataset (also the batch key).
LABEL_COLUMN = "dataset"

# Categorical annotations with more levels than this (barcodes, free text) are
# not carried into the joint dataset.
_MAX_CARRY_CATEGORIES = 200
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


# Integration methods: Harmony on the PCA, or plain concatenation (no correction).
METHOD_HARMONY = "harmony"
METHOD_NONE = "none"
METHODS = (METHOD_HARMONY, METHOD_NONE)


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
    if sp.issparse(M):
        return sp.csr_matrix(M, dtype=np.float32, copy=True)
    return sp.csr_matrix(np.asarray(M, dtype=np.float32))


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
            src = ad.read_h5ad(path)
        except Exception as e:  # unreadable source: try the next one
            logger.warning("Could not read %s for counts: %s", path, e)
            continue
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

    raise IntegrationError(
        "no raw counts found (checked layers['counts'], raw.X and X, and the dataset's "
        "source files). Integration re-processes from counts, so this dataset can't join."
    )


# ---------------------------------------------------------------------------
# Units: one dataset's counts + carried annotations
# ---------------------------------------------------------------------------


def carried_obs(adata: ad.AnnData, label: str) -> pd.DataFrame:
    """Categorical annotations worth keeping in the joint dataset.

    Cluster ids are dataset-local (cluster "3" in A is unrelated to "3" in B), so
    the active clustering is carried as ``source_cluster`` prefixed with the
    dataset label, and other cluster columns are dropped — the joint run
    clusters afresh.
    """
    obs = adata.obs
    active = adata.uns.get("scview_active_clustering")
    keep: dict[str, pd.Series] = {}
    for col in obs.columns:
        col = str(col)
        if col == active or col in ("leiden", "louvain") or col.startswith("scview_"):
            continue
        s = obs[col]
        if not (isinstance(s.dtype, pd.CategoricalDtype) or pd.api.types.is_string_dtype(s)
                or pd.api.types.is_bool_dtype(s)):
            continue
        n_levels = s.nunique(dropna=True)
        # Too many levels, or nearly one per cell (barcodes / ids): not an annotation.
        if not 1 <= n_levels <= _MAX_CARRY_CATEGORIES or (n_levels > 1 and n_levels > 0.5 * len(s)):
            continue
        name = "source_dataset" if col == LABEL_COLUMN else col
        keep[name] = s.astype(object).where(s.notna(), None)
    if active and active in obs.columns:
        keep["source_cluster"] = (label + ":" + obs[active].astype(str)).astype(object)
    return pd.DataFrame(keep, index=obs.index)


@dataclass
class Unit:
    dataset_id: str
    name: str
    label: str
    adata: ad.AnnData  # counts in X, carried annotations in obs
    counts_source: str


def build_unit(
    adata: ad.AnnData, *, dataset_id: str, name: str, label: str, fallback_paths: list[str]
) -> Unit:
    if getattr(adata, "isbacked", False):
        adata = adata.to_memory()
    counts = recover_counts(adata, fallback_paths)
    var = counts.var
    var.index = var.index.astype(str)
    obs = carried_obs(adata, label)
    obs.index = obs.index.astype(str)
    unit = ad.AnnData(X=counts.X, obs=obs, var=var)
    return Unit(dataset_id, name, label, unit, counts.source)


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


class IntegrationPlan(BaseModel):
    datasets: list[DatasetCheck] = Field(default_factory=list)
    merge: MergePlan | None = None
    shared_columns: list[str] = Field(default_factory=list)
    est_cells: int = 0
    blocking: list[str] = Field(default_factory=list)
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
        try:
            cached = dm.get_dataset(did)
            adata = cached.adata if cached is not None else ad.read_h5ad(paths[0])
            unit = build_unit(adata, dataset_id=did, name=name, label=label,
                              fallback_paths=paths[1:])
        except IntegrationError as e:
            check.error = str(e)
            continue
        except Exception as e:  # noqa: BLE001 — report, don't crash the plan
            logger.warning("Could not prepare %s for integration: %s", did, e, exc_info=True)
            check.error = f"could not read dataset: {e}"
            continue
        check.n_cells, check.n_genes = int(unit.adata.n_obs), int(unit.adata.n_vars)
        check.counts_source = unit.counts_source
        check.columns = [str(c) for c in unit.adata.obs.columns]
        units[label] = unit
    return checks, units


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
        m = plan.merge
        genes = m.reconciliation.overlap_after if (
            m.reconciliation and m.reconciliation.feasible) else m.intersection
        if genes == 0:
            blocking.append("These datasets share no genes, so they can't be embedded together.")

    plan.ok = not blocking
    return plan


# ---------------------------------------------------------------------------
# Merge + persist
# ---------------------------------------------------------------------------


def merge_joint(
    units: dict[str, Unit], *, join: MergeJoin = MergeJoin.inner, apply_reconciliation: bool = True
) -> ad.AnnData:
    """Concatenate units into one counts AnnData labelled by ``dataset``."""
    adatas = {lab: u.adata for lab, u in units.items()}
    # Pad every unit to the union of annotation columns: concatenating with an
    # inner gene join would otherwise also drop any column one dataset lacks.
    cols = list(dict.fromkeys(c for a in adatas.values() for c in a.obs.columns))
    for a in adatas.values():
        for c in cols:
            if c not in a.obs.columns:
                a.obs[c] = None
        a.obs = a.obs[cols]
    recon = plan_merge(adatas).reconciliation if apply_reconciliation else None
    joint = merge_units(adatas, join=join, sample_label=LABEL_COLUMN, reconciliation=recon)
    for c in cols:
        joint.obs[c] = joint.obs[c].astype("category")
    return joint


def record_integration_provenance(
    joint: ad.AnnData, units: dict[str, Unit], *, join: MergeJoin, method: str,
    apply_reconciliation: bool,
) -> None:
    """Source block + an ``integrate`` history entry. Never raises."""
    merged_from = [
        {"dataset_id": u.dataset_id, "name": u.name, "label": u.label,
         "n_cells": int(u.adata.n_obs), "counts_source": u.counts_source}
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

# Embedding scored as "uncorrected" / "corrected".
_MIXING_SPACES = (("uncorrected", "X_pca"), ("corrected", "X_pca_harmony"))


def _cross_dataset_fraction(
    X: np.ndarray, codes: np.ndarray, query: np.ndarray, k: int
) -> np.ndarray:
    """Per query cell, the fraction of its k nearest neighbours from another dataset."""
    from sklearn.neighbors import NearestNeighbors

    idx = NearestNeighbors(n_neighbors=k + 1).fit(X).kneighbors(X[query], return_distance=False)
    # Drop the cell itself; among duplicate points it need not come first.
    not_self = idx != query[:, None]
    order = np.argsort(~not_self, axis=1, kind="stable")
    idx = np.take_along_axis(idx, order, axis=1)[:, :k]
    return (codes[idx] != codes[query][:, None]).mean(axis=1)


def batch_mixing(
    adata: ad.AnnData,
    *,
    label_column: str = LABEL_COLUMN,
    k: int = MIXING_K,
    max_scored: int = MIXING_MAX_SCORED,
    seed: int = 0,
) -> dict[str, Any] | None:
    """How well the source datasets mix, in the uncorrected and corrected PCA.

    For each cell, the fraction of its ``k`` nearest neighbours that come from a
    different source dataset, averaged over cells. Were the datasets perfectly
    mixed, a cell of dataset d would see others at rate (n - n_d) / (n - 1), so
    ``ideal`` is that rate averaged over the scored cells (~1 - sum of squared
    proportions). Kang CTRL vs STIM, scored by hand: 0.107 uncorrected, 0.478
    after Harmony, ideal 0.499. None when there is nothing to score.
    """
    if label_column not in adata.obs.columns:
        return None
    spaces = [(name, key) for name, key in _MIXING_SPACES if key in adata.obsm]
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
        "uncorrected": None,
        "corrected": None,
        "datasets": [
            {"label": str(lab), "n_cells": int(sizes[i]), "ideal": round(float(ideal[i]), 4),
             "uncorrected": None, "corrected": None}
            for i, lab in enumerate(names)
        ],
    }
    for name, key in spaces:
        frac = _cross_dataset_fraction(
            np.asarray(adata.obsm[key], dtype=np.float32), codes, query, k
        )
        out[name] = round(float(frac.mean()), 4)
        for i, row in enumerate(out["datasets"]):
            mine = frac[q_codes == i]
            row[name] = round(float(mine.mean()), 4) if mine.size else None
    return out


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
    if method == METHOD_HARMONY:
        steps += ["uncorrected_embedding", "batch_correction"]
    return steps + ["neighbors", "clustering", "embeddings", "marker_genes"]


def joint_params(method: str, overrides: dict[str, Any] | None = None) -> PipelineParams:
    params = PipelineParams(
        hvg_batch_key=LABEL_COLUMN,
        batch_key=LABEL_COLUMN if method == METHOD_HARMONY else "",
        batch_method="harmony",
    )
    for key, value in (overrides or {}).items():
        if key in TUNABLE_PARAMS and value is not None:
            setattr(params, key, type(getattr(params, key))(value))
    return params


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
) -> Generator[tuple[str, dict[str, Any]], None, None]:
    """Merge, persist and process; yields (event, data) for SSE progress.

    ``output_path_for(source_path)`` returns the derived output path, so the
    caller's DatasetManager decides where processed results live.
    """
    yield ("stage", {"stage": "merge", "message": f"Merging {len(units)} datasets"})
    joint = merge_joint(units, join=join, apply_reconciliation=apply_reconciliation)
    record_integration_provenance(joint, units, join=join, method=method,
                                  apply_reconciliation=apply_reconciliation)

    yield ("stage", {"stage": "persist", "message": "Saving the joint dataset"})
    src = persist_joint(data_dir, dataset_id, name, joint, units)
    yield ("created", {"dataset_id": dataset_id, "name": name,
                       "n_cells": int(joint.n_obs), "n_genes": int(joint.n_vars)})

    steps = joint_steps(method)
    params = joint_params(method, overrides)
    # Score once both PCA spaces exist, so the pipeline's final write saves it.
    score_after = "batch_correction" if method == METHOD_HARMONY else "pca"
    for event, data in run_pipeline_streamed(
        joint, steps, params, output_path=str(output_path_for(str(src)))
    ):
        # The pipeline's own "complete" isn't the end of the integration.
        yield ("pipeline_complete" if event == "complete" else event, data)
        if event in ("step_done", "step_error") and data.get("step") == score_after:
            yield ("substep", {"step": score_after, "message": "Scoring batch mixing"})
            record_batch_mixing(joint)
