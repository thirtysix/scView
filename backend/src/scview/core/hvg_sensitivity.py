"""Would the clusters survive a different choice of variable genes?

scView picks one set of highly variable genes per pipeline run, and that choice
moves the clusters: pipeComp's benchmark of whole pipelines (Germain et al.
2020, PMID 32873325) and EMD-HVG (Peng et al. 2026, PMID 42323546) both found
the feature-selection method and gene count change the partition. This check
re-selects the variable genes under a few other settings and, for each, re-runs
scaling (or Pearson residuals), PCA, Harmony when the dataset's graph was
corrected, the neighbour graph and the clustering on copies, with the dataset's
own recorded method, resolution, PCs, neighbours and Leiden passes (fast Leiden's
2 when the clustering was made that way). Each partition is compared
with the current clustering:

- ARI and AMI over all cells;
- for each current cluster, its best-match Jaccard index in the new clustering
  (stable at 0.75 or more, Hennig 2007's clusterboot rule), and if not stable,
  whether it splits, merges into a bigger cluster, or dissolves (both);
- how many of its variable genes the setting shares with the current ones.

The current setting itself is re-run first, as a control: it shows how closely a
re-run reproduces the stored clusters (Kang: ARI 0.998), the floor any other
setting's differences should be read against. The dataset is never changed; a
setting's clusters are saved only when asked (``keep``).

Each setting runs the same scanpy calls as the pipeline's steps
(``_run_hvg`` / ``_pearson_hvg``, ``_run_scaling``, ``_run_pca``,
``_run_neighbors``, ``_run_clustering``), on the counts when the dataset keeps
them, so the control reproduces the dataset's own variable genes.

A setting can also change the neighbour graph's k (10 and 30 by default,
around scanpy's 15) or the clustering algorithm (Leiden or Louvain), the other
two choices that move a partition at a fixed resolution; settings with the same
variable genes share their PCs and re-run only the graph and the clustering.
"""

from __future__ import annotations

import dataclasses
import hashlib
import importlib.util
import logging
import re
import time
from collections import OrderedDict
from collections.abc import Callable, Iterator
from dataclasses import asdict, dataclass, field
from typing import Any

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp

from scview.core import leiden, normalization, provenance
from scview.core import resolution_sweep as rs
from scview.core.anndata_adaptor import CLUSTERING_CANDIDATES
from scview.core.obs_writes import forget_column_results

logger = logging.getLogger(__name__)

FLAVORS: tuple[str, ...] = ("seurat", "cell_ranger", "seurat_v3", "pearson_residuals")
FLAVOR_LABELS = {
    "seurat": "Seurat",
    "cell_ranger": "Cell Ranger",
    "seurat_v3": "Seurat v3 (counts)",
    "pearson_residuals": "Pearson residuals",
}
METHOD_LABELS = {"leiden": "Leiden", "louvain": "Louvain"}
MAX_SETTINGS = 10
MIN_GENES, MAX_GENES = 200, 10_000
MIN_NEIGHBORS, MAX_NEIGHBORS = 5, 100
NEIGHBOR_CHOICES = (10, 30)  # tried by default besides the recorded k (scanpy's default 15)
MIN_CELLS = 50
# Each setting re-runs PCA, the neighbour graph and the clustering on every cell:
# about 1 s per 1,000 cells on 2 cores (Kang, 13,836 cells: 10-15 s a setting).
MAX_CELLS = 100_000
MAX_REFERENCE_GROUPS = 500
STABLE = 0.75  # Jaccard: Hennig (2007), a cluster found again at 0.75 or more is stable
PART = 0.1  # a new cluster holding this share of a cluster's cells is one of its parts

# Time estimate (Kang copy, 13,836 cells, 2 cores, measured 2026-10-17): 9-20 s a
# setting once warm, 65 s for the five defaults; Harmony adds 0.3 s.
SECONDS_PER_1K_CELLS = 1.0
HARMONY_SECONDS_PER_1K_CELLS = 0.05
GRAPH_SECONDS_PER_1K_CELLS = 0.9  # a setting sharing another's PCs: the graph and Leiden (Kang: 13-19 s)
FIRST_RUN_SECONDS = 30.0  # numba compiling the neighbour search after a server start

_DEFAULT_GENES = 2000
_CACHE_SIZE = 24  # settings' labels kept for "keep", across datasets


class SensitivityError(ValueError):
    """A check that cannot run on this dataset, or a bad request: a 400."""


class Cancelled(Exception):
    """The client went away; the run stops before its next sub-step."""


@dataclass(frozen=True)
class Setting:
    flavor: str
    n_top_genes: int
    # The neighbour graph's k and the clustering algorithm; None: as recorded.
    n_neighbors: int | None = None
    method: str | None = None

    @property
    def key(self) -> str:
        k = f"{self.flavor}_{self.n_top_genes}"
        if self.n_neighbors is not None:
            k += f"_k{self.n_neighbors}"
        return k + (f"_{self.method}" if self.method else "")

    @property
    def label(self) -> str:
        out = f"{FLAVOR_LABELS.get(self.flavor, self.flavor)}, {self.n_top_genes:,} genes"
        if self.n_neighbors is not None:
            out += f", {self.n_neighbors} neighbours"
        return out + (f", {METHOD_LABELS.get(self.method, self.method)}" if self.method else "")

    @property
    def embedding(self) -> tuple[str, int]:
        """What the PCs depend on: settings differing only in graph or algorithm share them."""
        return self.flavor, self.n_top_genes

    def to_dict(self) -> dict[str, Any]:
        return {"flavor": self.flavor, "n_top_genes": self.n_top_genes, "n_neighbors": self.n_neighbors,
                "method": self.method, "key": self.key, "label": self.label}

    def effective(self, rec: Recorded) -> Recorded:
        """``rec`` with this setting's graph and algorithm."""
        return dataclasses.replace(rec, n_neighbors=self.n_neighbors or rec.n_neighbors,
                                   method=self.method or rec.method)


@dataclass
class Recorded:
    """How the dataset's current clustering was made, from provenance (what had to
    be assumed is listed in ``notes``)."""

    reference: str  # the obs column compared with
    method: str
    resolution: float
    flavor: str
    n_top_genes: int
    hvg_batch_key: str | None
    target_sum: float
    n_comps: int
    n_neighbors: int
    batch_key: str | None  # Harmony's, when the graph was built on X_pca_harmony
    notes: list[str] = field(default_factory=list)
    # Why the control may not reproduce the reference, for the UI to say:
    reference_from_file: bool = False  # the column came with the file, not from scView's clustering
    pcs_made_by: str | None = None  # scView version that made the graph's PCs (None: not recorded)
    pcs_made_on: str | None = None  # and the date
    pcs_older: bool = False  # by an older scView: the re-runs build them with this version's pipeline
    # Leiden's passes the reference was made with (-1: until none improves; 2: fast Leiden),
    # which every re-run uses, so the control reproduces the reference
    leiden_iterations: int = -1
    # obs columns the scaling step regressed out of the variable genes (core/scaling.py)
    regress_out: list[str] = field(default_factory=list)

    @property
    def current(self) -> Setting:
        return Setting(self.flavor, self.n_top_genes)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# What the dataset recorded
# ---------------------------------------------------------------------------


def _last_params(history: list[dict], step: str) -> dict[str, Any] | None:
    for entry in reversed(history):
        if entry.get("step") == step:
            return entry.get("params") or {}
    return None


def _last_entry(history: list[dict], step: str) -> dict[str, Any] | None:
    return next((e for e in reversed(history) if e.get("step") == step), None)


def _older_than_this(made_by: str | None) -> bool:
    from packaging.version import InvalidVersion, Version

    try:
        return made_by is not None and Version(made_by) < Version(provenance._scview_version())
    except InvalidVersion:
        return False


def active_clustering(adata) -> str | None:
    """The dataset's active clustering column, else the one provenance recorded,
    else the column the rest of the app shows as the clustering (a file processed
    elsewhere: Kang's ``cluster``)."""
    active = adata.uns.get("scview_active_clustering")
    if isinstance(active, str) and active in adata.obs.columns:
        return active
    col = (provenance.read_provenance(adata)["current"].get("clustering") or {}).get("column")
    if isinstance(col, str) and col in adata.obs.columns:
        return col
    return next((c for c in CLUSTERING_CANDIDATES if c in adata.obs.columns), None)


def _method_resolution(adata, reference: str, history: list[dict], notes: list[str]) -> tuple[str, float]:
    """The clustering method and resolution: from the reference column's own name
    (``scview_leiden_r0.5``), else what provenance or scanpy recorded for it."""
    m = re.fullmatch(r"scview_(leiden|louvain)_r([0-9.]+)", reference)
    if m:
        return m.group(1), float(m.group(2))
    current = provenance.read_provenance(adata)["current"].get("clustering") or {}
    if current.get("column") == reference and current.get("resolution") is not None:
        return str(current.get("method") or "leiden"), float(current["resolution"])
    params = (adata.uns.get(reference) or {}).get("params") if isinstance(adata.uns.get(reference), dict) else None
    method = rs.dataset_method(adata)
    if isinstance(params, dict) and params.get("resolution") is not None:
        return method, float(params["resolution"])
    recorded = _last_params(history, "clustering") or {}
    if recorded.get("clustering_resolution") is not None:
        notes.append(f"'{reference}' was not made by scView's clustering: re-clustered with the "
                     "dataset's recorded method and resolution.")
        return str(recorded.get("clustering_method") or method), float(recorded["clustering_resolution"])
    notes.append("No clustering resolution is recorded: using 1.0, scanpy's default.")
    return method, 1.0


def _reference_codes(adata, reference: str) -> tuple[np.ndarray, list[str]]:
    """The reference column as codes (-1 for cells without a label) and names, in
    the column's own order when it has one."""
    values = adata.obs[reference]
    if isinstance(values.dtype, pd.CategoricalDtype):
        return (np.asarray(values.cat.codes, dtype=np.int64),
                [str(c) for c in values.cat.categories])
    cat = pd.Categorical(values.astype(str).where(values.notna()))
    return np.asarray(cat.codes, dtype=np.int64), [str(c) for c in cat.categories]


def recorded(adata, reference: str | None = None) -> Recorded:
    """The settings the dataset's clustering was made with; SensitivityError if
    there is no clustering to compare with."""
    reference = reference or active_clustering(adata)
    if not reference:
        raise SensitivityError(
            "This dataset has no clustering to compare with. Run the processing pipeline "
            "(through clustering) in Data Assessment first."
        )
    if reference not in adata.obs.columns:
        raise SensitivityError(f"Column '{reference}' not found.")
    n_groups = adata.obs[reference].nunique()
    if not 2 <= n_groups <= MAX_REFERENCE_GROUPS:
        raise SensitivityError(
            f"'{reference}' has {n_groups} group{'' if n_groups == 1 else 's'}; the comparison needs between 2 and "
            f"{MAX_REFERENCE_GROUPS}."
        )
    history = provenance.read_provenance(adata).get("history", [])
    notes: list[str] = []
    method, resolution = _method_resolution(adata, reference, history, notes)

    hvg = _last_params(history, "highly_variable_genes") or {}
    norm = _last_params(history, "normalization") or {}
    if (hvg.get("normalization_method") or norm.get("normalization_method")) == "pearson_residuals":
        flavor = "pearson_residuals"
    else:
        flavor = hvg.get("hvg_flavor") or (adata.uns.get("hvg") or {}).get("flavor")
        if flavor not in FLAVORS:
            notes.append("The variable genes' flavour is not recorded: assuming Seurat.")
            flavor = "seurat"
    n_top = hvg.get("n_top_genes")
    if not n_top and "highly_variable" in adata.var.columns:
        n_top = int(adata.var["highly_variable"].sum())
    if not n_top:
        notes.append(f"The number of variable genes is not recorded: assuming {_DEFAULT_GENES:,}.")
        n_top = _DEFAULT_GENES

    nb = (adata.uns.get("neighbors") or {}).get("params") or {}
    if "X_pca" in adata.obsm:
        n_comps = int(np.asarray(adata.obsm["X_pca"]).shape[1])
    else:
        n_comps = int((_last_params(history, "pca") or {}).get("n_comps") or 50)
    if isinstance(nb.get("n_pcs"), (int, np.integer)) and 0 < nb["n_pcs"] < n_comps:
        n_comps = int(nb["n_pcs"])
    n_neighbors = int(nb.get("n_neighbors") or (_last_params(history, "neighbors") or {}).get("n_neighbors")
                      or 15)

    batch_key = None
    if nb.get("use_rep") == "X_pca_harmony":
        from scview.core.confounding import recorded_batch_key

        batch_key = recorded_batch_key(adata)
        if batch_key not in adata.obs.columns:
            notes.append("The graph was batch-corrected with a key that is not recorded: "
                         "the re-runs skip batch correction.")
            batch_key = None
    elif nb.get("use_rep") == "X_scanorama" or "bbknn" in nb:
        notes.append("The graph was batch-corrected with "
                     + ("BBKNN" if "bbknn" in nb else "Scanorama")
                     + "; the re-runs only repeat Harmony, so they skip batch correction.")
    hvg_batch_key = hvg.get("hvg_batch_key") or None
    if hvg_batch_key and hvg_batch_key not in adata.obs.columns:
        hvg_batch_key = None
    made = _last_entry(history, "batch_correction" if batch_key else "pca") or {}
    made_by = made.get("scview_version")
    current_col = (provenance.read_provenance(adata)["current"].get("clustering") or {}).get("column")
    return Recorded(
        reference=reference, method=method if method in rs.METHODS else "leiden",
        resolution=float(resolution), flavor=flavor, n_top_genes=int(n_top),
        hvg_batch_key=hvg_batch_key, target_sum=float(norm.get("target_sum") or 1e4),
        n_comps=n_comps, n_neighbors=n_neighbors, batch_key=batch_key, notes=notes,
        reference_from_file=not reference.startswith("scview_") and not reference.endswith("_sub")
        and reference != current_col,
        pcs_made_by=made_by, pcs_made_on=(made.get("timestamp") or "")[:10] or None,
        pcs_older=_older_than_this(made_by),
        leiden_iterations=leiden.recorded_iterations(adata, reference),
        regress_out=_recorded_regress_out(adata, history, notes),
    )


def _recorded_regress_out(adata, history: list[dict], notes: list[str]) -> list[str]:
    """The covariates the last scaling step regressed out, when they are still there."""
    from scview.core import scaling

    try:
        keys = scaling.check_regress_out((_last_params(history, "scaling") or {}).get("regress_out"))
    except ValueError:
        return []
    gone = [k for k in keys if k not in adata.obs.columns]
    if gone:
        notes.append(f"Regress-out of {', '.join(gone)} is recorded but the column is gone: "
                     "the re-runs regress out nothing.")
        return []
    return keys


# ---------------------------------------------------------------------------
# Inputs and settings
# ---------------------------------------------------------------------------


def seurat_v3_available() -> bool:
    """scanpy's seurat_v3 flavour fits a loess with scikit-misc (in the image since v0.17)."""
    return importlib.util.find_spec("skmisc") is not None


def _counts(adata):
    try:
        return normalization.find_counts(adata)
    except ValueError:
        return None, None


def flavors_available(adata) -> dict[str, str | None]:
    """Each flavour, and why it can't run on this dataset (None when it can)."""
    counts, _ = _counts(adata)
    no_counts = "Needs raw counts, which this dataset does not keep."
    return {
        "seurat": None,
        "cell_ranger": None,
        "seurat_v3": (no_counts if counts is None else None) if seurat_v3_available()
        else "Needs scikit-misc, which is not installed.",
        "pearson_residuals": no_counts if counts is None else None,
    }


def check(adata) -> None:
    """Raise SensitivityError when the check cannot run here, saying why."""
    from scview.core import large_layout, large_mode

    if large_layout.is_large(adata):  # its matrices stay on disk
        raise SensitivityError(str(large_mode.LargeDatasetError(
            "This check", "it re-runs normalisation, variable genes and PCA on the whole "
            "matrix in memory")))
    if adata.n_obs < MIN_CELLS:
        raise SensitivityError(f"The check needs at least {MIN_CELLS} cells; this dataset has {adata.n_obs}.")
    if adata.n_obs > MAX_CELLS:
        raise SensitivityError(
            f"The check re-runs PCA, the graph and the clustering on every cell, so it runs on at "
            f"most {MAX_CELLS:,} cells; this dataset has {adata.n_obs:,}."
        )
    if adata.n_vars < MIN_GENES:
        raise SensitivityError(f"The check needs at least {MIN_GENES} genes; this dataset has {adata.n_vars}.")
    if getattr(adata, "isbacked", False):
        raise SensitivityError("This dataset is read from disk as it is used (over 2 GB), "
                               "which the check can't re-process.")


def _round_genes(n: float) -> int:
    return int(max(MIN_GENES, round(n / 100) * 100))


def default_settings(adata, rec: Recorded) -> list[Setting]:
    """The current setting (the control), half and 1.5x as many genes, each other
    flavour that can run here at the current count, then the current genes with
    10 and 30 neighbours and with the other clustering algorithm."""
    available = flavors_available(adata)
    n = min(rec.n_top_genes, adata.n_vars)
    out: list[Setting] = []
    if available.get(rec.flavor) is None:
        out.append(Setting(rec.flavor, n))
    base = rec.flavor if available.get(rec.flavor) is None else "seurat"
    for m in (_round_genes(n / 2), _round_genes(n * 1.5)):
        out.append(Setting(base, min(m, adata.n_vars)))
    for f in ("seurat", "cell_ranger", "pearson_residuals", "seurat_v3"):
        if f != rec.flavor and available[f] is None:
            out.append(Setting(f, n))
    for k in NEIGHBOR_CHOICES:
        if k != rec.n_neighbors:
            out.append(Setting(base, n, n_neighbors=k))
    out.append(Setting(base, n, method=next(m for m in rs.METHODS if m != rec.method)))
    return list(dict.fromkeys(out))[:MAX_SETTINGS]


def clean_settings(adata, rec: Recorded, values: list[dict[str, Any]] | None) -> list[Setting]:
    """The requested settings, validated and deduplicated; the defaults when none."""
    if not values:
        return default_settings(adata, rec)
    available = flavors_available(adata)
    out: list[Setting] = []
    for v in values:
        flavor = str(v.get("flavor") or "")
        if flavor not in FLAVORS:
            raise SensitivityError(f"Unknown flavour '{flavor}'; choose from {', '.join(FLAVORS)}.")
        if available[flavor] is not None:
            raise SensitivityError(f"{FLAVOR_LABELS[flavor]}: {available[flavor]}")
        try:
            n = int(v.get("n_top_genes"))
        except (TypeError, ValueError):
            raise SensitivityError("n_top_genes must be a whole number.") from None
        if not MIN_GENES <= n <= MAX_GENES:
            raise SensitivityError(f"Variable genes must be between {MIN_GENES:,} and {MAX_GENES:,} (got {n}).")
        if n > adata.n_vars:
            raise SensitivityError(f"This dataset has {adata.n_vars:,} genes; {n:,} variable genes is too many.")
        k = v.get("n_neighbors")
        if k is not None:
            try:
                k = int(k)
            except (TypeError, ValueError):
                raise SensitivityError("n_neighbors must be a whole number.") from None
            if not MIN_NEIGHBORS <= k <= MAX_NEIGHBORS:
                raise SensitivityError(f"Neighbours must be between {MIN_NEIGHBORS} and {MAX_NEIGHBORS} (got {k}).")
            if k >= adata.n_obs:
                raise SensitivityError(f"This dataset has {adata.n_obs:,} cells; {k} neighbours is too many.")
        method = v.get("method") or None
        if method is not None and method not in rs.METHODS:
            raise SensitivityError(f"Unknown clustering method '{method}'; choose from {', '.join(rs.METHODS)}.")
        # The recorded value is the current setting's own, not a change.
        out.append(Setting(flavor, n, None if k == rec.n_neighbors else k,
                           None if method == rec.method else method))
    out = list(dict.fromkeys(out))
    if len(out) > MAX_SETTINGS:
        raise SensitivityError(f"Try at most {MAX_SETTINGS} settings at a time.")
    return out


def estimate_seconds(n_cells: int, n_settings: int, rec: Recorded, n_embeddings: int | None = None) -> float:
    """Seconds for ``n_settings``, of which ``n_embeddings`` (default all) need
    their own PCs; the others re-run only the graph and the clustering."""
    per = SECONDS_PER_1K_CELLS + (HARMONY_SECONDS_PER_1K_CELLS if rec.batch_key else 0.0)
    n_emb = n_settings if n_embeddings is None else n_embeddings
    graph_only = (n_settings - n_emb) * GRAPH_SECONDS_PER_1K_CELLS
    return round((n_emb * per + graph_only) * n_cells / 1000 + 2, 1)


def column_name(rec: Recorded, setting: Setting) -> str:
    """The column "keep" saves a setting's clusters as."""
    out = f"{rs.column_name(rec.method, rec.resolution)}_hvg_{setting.flavor}_{setting.n_top_genes}"
    if setting.n_neighbors is not None:
        out += f"_k{setting.n_neighbors}"
    return out + (f"_{setting.method}" if setting.method else "")


@dataclass
class _Inputs:
    var_names: pd.Index
    counts: Any  # cells x genes counts, or None
    log: Any  # log-normalised, as the pipeline's variable-gene step saw it
    obs: pd.DataFrame  # the batch keys' columns only
    current_hvgs: np.ndarray | None


def _inputs(adata, rec: Recorded) -> _Inputs:
    """What every setting reads, built once: the log-normalised matrix is made from
    the counts as the pipeline makes it (normalize_total + log1p), so the control
    picks the dataset's own variable genes; without counts, the stored log values."""
    from scview.core.pipeline import looks_like_counts

    counts, _ = _counts(adata)
    if counts is not None:
        return inputs_from_counts(adata, rec, counts)
    log = None
    raw = adata.raw
    sources = []
    if raw is not None and not len(adata.var_names.difference(raw.var_names)):
        sources.append(raw[:, adata.var_names].X)
    sources.append(adata.X)
    for X in sources:
        if not looks_like_counts(X) and float(X.min()) >= 0:  # scaled data has negatives
            log = X
            break
    if log is None:
        raise SensitivityError(
            "The check needs counts or log-normalised expression; this dataset only has "
            "scaled values. Re-run the pipeline from the counts."
        )
    return _with_obs(adata, rec, None, log)


def inputs_from_counts(adata, rec: Recorded, counts) -> _Inputs:
    """The inputs for ``counts`` (cells x the dataset's genes): ``adata``'s own, or
    a thinned copy of them (core/merge_check.py)."""
    import scanpy as sc

    # In the counts' own dtype, as the pipeline normalises them: float32 instead
    # of Kang's float64 moved the PCA by 1e-5 and lost a 24-cell cluster.
    tmp = ad.AnnData(X=counts.copy())
    sc.pp.normalize_total(tmp, target_sum=rec.target_sum)
    sc.pp.log1p(tmp)
    return _with_obs(adata, rec, counts, tmp.X)


def _with_obs(adata, rec: Recorded, counts, log) -> _Inputs:
    keys = [k for k in (rec.hvg_batch_key, rec.batch_key, *rec.regress_out) if k]
    obs = adata.obs[list(dict.fromkeys(keys))].copy() if keys else pd.DataFrame(index=adata.obs_names)
    hv = adata.var.get("highly_variable")
    current = np.asarray(hv, dtype=bool) if hv is not None else None
    return _Inputs(var_names=adata.var_names.copy(), counts=counts, log=log, obs=obs,
                   current_hvgs=current)


# ---------------------------------------------------------------------------
# One setting
# ---------------------------------------------------------------------------


def _check_stop(stop: Callable[[], bool] | None) -> None:
    if stop is not None and stop():
        raise Cancelled()


def _select(inp: _Inputs, s: Setting, rec: Recorded) -> np.ndarray:
    """The setting's variable genes as a mask over the dataset's genes."""
    import scanpy as sc

    batch_key = rec.hvg_batch_key
    obs = inp.obs[[batch_key]] if batch_key else None
    n = min(s.n_top_genes, len(inp.var_names))
    if s.flavor == "pearson_residuals":  # as pipeline._pearson_hvg
        tmp = ad.AnnData(X=inp.counts, obs=obs)
        tmp.var_names = inp.var_names
        df = sc.experimental.pp.highly_variable_genes(
            tmp, flavor="pearson_residuals", n_top_genes=n, batch_key=batch_key,
            theta=normalization.THETA, inplace=False,
        )
    else:  # as pipeline._run_hvg; seurat_v3 ranks the counts
        tmp = ad.AnnData(X=inp.counts if s.flavor == "seurat_v3" else inp.log, obs=obs)
        tmp.var_names = inp.var_names
        df = sc.pp.highly_variable_genes(tmp, n_top_genes=n, flavor=s.flavor, batch_key=batch_key,
                                         inplace=False)
    return np.asarray(df["highly_variable"], dtype=bool)


def _pcs(inp: _Inputs, mask: np.ndarray, s: Setting, rec: Recorded) -> np.ndarray:
    """PCA of the variable genes: z-scored log values (clipped at 10), or Pearson
    residuals, as the pipeline's scaling and PCA steps make it."""
    import scanpy as sc

    idx = np.flatnonzero(mask)
    if s.flavor == "pearson_residuals":
        X = normalization.pearson_residuals(inp.counts, columns=idx)
    else:
        # Dense float32 first, as core/scaling.py builds the matrix the pipeline's PCA reads.
        part = inp.log[:, idx].astype(np.float32)
        sub = ad.AnnData(X=part.toarray() if sp.issparse(part) else part)
        if rec.regress_out:  # the recorded covariates' fit taken off, as the scaling step does
            from scview.core import scaling

            C = scaling.covariates(inp.obs, rec.regress_out)
            scaling.remove_fit(sub.X, C, scaling.regression_coef(inp.log, idx, C))
        sc.pp.scale(sub, max_value=10)
        X = sub.X
    # Column-major, as the pipeline's PCA reads the variable genes (a masked view
    # of X): row-major moved Kang's PCA by one float32 step, enough to change a
    # Harmony-corrected Leiden partition (ARI 0.85).
    sub = ad.AnnData(X=np.asfortranarray(X))
    n_comps = max(2, min(rec.n_comps, sub.n_obs - 1, sub.n_vars - 1))
    sc.tl.pca(sub, n_comps=n_comps, svd_solver="arpack")
    return np.asarray(sub.obsm["X_pca"])


def _harmony(X: np.ndarray, inp: _Inputs, rec: Recorded) -> np.ndarray:
    from scview.core.pipeline import PipelineParams, _run_batch_correction

    tmp = ad.AnnData(obs=inp.obs[[rec.batch_key]].copy())
    tmp.obsm["X_pca"] = X
    _run_batch_correction(tmp, PipelineParams(batch_key=rec.batch_key))
    return np.asarray(tmp.obsm["X_pca_harmony"])


def _cluster(X: np.ndarray, rec: Recorded) -> tuple[pd.Categorical, sp.csr_matrix]:
    """The neighbour graph and the clustering, as the pipeline builds them:
    (labels, the graph's connectivities)."""
    import scanpy as sc

    tmp = ad.AnnData(obs=pd.DataFrame(index=[str(i) for i in range(X.shape[0])]))
    tmp.obsm["X_rep"] = X
    sc.pp.neighbors(tmp, n_neighbors=min(rec.n_neighbors, tmp.n_obs - 1), use_rep="X_rep")
    return (rs.cluster(tmp, rec.method, rec.resolution, rec.leiden_iterations),
            tmp.obsp["connectivities"])


def _embed(inp: _Inputs, s: Setting, rec: Recorded, stop: Callable[[], bool] | None = None,
           ) -> tuple[np.ndarray, np.ndarray]:
    """(variable-gene mask, the PCs the graph is built on) for one setting."""
    mask = _select(inp, s, rec)
    _check_stop(stop)
    X = _pcs(inp, mask, s, rec)
    _check_stop(stop)
    if rec.batch_key:
        X = _harmony(X, inp, rec)
        _check_stop(stop)
    return mask, X


def run_setting_graph(inp: _Inputs, s: Setting, rec: Recorded, stop: Callable[[], bool] | None = None,
                      embedded: tuple[np.ndarray, np.ndarray] | None = None,
                      ) -> tuple[np.ndarray, pd.Categorical, sp.csr_matrix]:
    """(variable-gene mask, cluster labels, neighbour graph) for one setting;
    checks ``stop`` between sub-steps. ``embedded``: its ``_embed``, when a setting
    with the same genes made it already."""
    mask, X = embedded if embedded is not None else _embed(inp, s, rec, stop)
    labels, graph = _cluster(X, s.effective(rec))
    return mask, labels, graph


def run_setting(inp: _Inputs, s: Setting, rec: Recorded,
                stop: Callable[[], bool] | None = None) -> tuple[np.ndarray, pd.Categorical]:
    """(variable-gene mask, cluster labels) for one setting; checks ``stop``
    between sub-steps."""
    mask, labels, _ = run_setting_graph(inp, s, rec, stop)
    return mask, labels


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------


def _status(jaccard: float, contained: float, purity: float) -> str:
    """stable; else merges (it stays together inside a bigger cluster), splits
    (its cells go to clusters of their own), or dissolves (neither)."""
    if jaccard >= STABLE:
        return "stable"
    if max(contained, purity) < STABLE:
        return "dissolves"
    return "merges" if contained >= purity else "splits"


def compare(ref_codes: np.ndarray, ref_names: list[str], labels) -> dict[str, Any]:
    """A new clustering against the reference: ARI, AMI and, per reference
    cluster, its best-match Jaccard and what became of it. Cells without a
    reference label are left out."""
    from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score

    new = np.asarray(pd.Categorical(labels).codes, dtype=np.int64)
    keep = (ref_codes >= 0) & (new >= 0)
    r, c = ref_codes[keep], new[keep]
    k_r, k_c = len(ref_names), int(new.max()) + 1 if new.size else 0
    table = np.bincount(r * k_c + c, minlength=k_r * k_c).reshape(k_r, k_c).astype(np.float64)
    size_r, size_c = table.sum(axis=1), table.sum(axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        jac = table / (size_r[:, None] + size_c[None, :] - table)
    jac = np.nan_to_num(jac)
    new_names = [str(x) for x in pd.Categorical(labels).categories]
    clusters = []
    for i, name in enumerate(ref_names):
        if size_r[i] == 0:
            continue
        best = int(np.argmax(jac[i]))
        # The new cluster holding most of its cells; of two holding as many, the closer one.
        host = int(np.lexsort((jac[i], table[i]))[-1])
        contained = table[i, host] / size_r[i]
        purity = table[i, host] / size_c[host]
        j = float(jac[i, best])
        clusters.append({
            "cluster": name,
            "n_cells": int(size_r[i]),
            "jaccard": round(j, 4),
            "match": new_names[best],
            "contained": round(float(contained), 4),
            "purity": round(float(purity), 4),
            "parts": int((table[i] >= PART * size_r[i]).sum()),
            "status": _status(j, float(contained), float(purity)),
        })
    sizes = np.sort(np.bincount(new[new >= 0]))[::-1]
    return {
        "ari": round(float(adjusted_rand_score(r, c)), 4),
        "ami": round(float(adjusted_mutual_info_score(r, c)), 4),
        "n_clusters": int(sizes.size),
        "sizes": sizes.astype(int).tolist(),
        "clusters": clusters,
    }


# ---------------------------------------------------------------------------
# Labels kept for "keep"
# ---------------------------------------------------------------------------

_cache: OrderedDict[tuple[str, str], pd.Categorical] = OrderedDict()


def _hash(values) -> bytes:
    return pd.util.hash_pandas_object(pd.Index(values).astype(str), index=False).to_numpy().tobytes()


def fingerprint(adata, rec: Recorded) -> str:
    """Identity of what a setting's clusters depend on: the cells, the genes, the
    matrix they are computed from (its size and sum), the batch columns and the
    recorded parameters. Saving a column changes none of them."""
    import scipy.sparse as sp

    M, _ = _counts(adata)
    if M is None:
        M = adata.raw.X if adata.raw is not None else adata.X
    nnz = int(M.nnz) if sp.issparse(M) else int(np.count_nonzero(M))
    h = hashlib.sha1(_hash(adata.obs_names) + _hash(adata.var_names))
    for key in (rec.hvg_batch_key, rec.batch_key):
        if key:
            h.update(_hash(adata.obs[key].to_numpy()))
    params = (rec.method, rec.resolution, rec.target_sum, rec.n_comps, rec.n_neighbors,
              rec.batch_key, rec.hvg_batch_key, M.shape, nnz, float(M.sum()), rec.leiden_iterations)
    h.update(repr(params).encode())
    return h.hexdigest()


def _remember(fp: str, s: Setting, labels: pd.Categorical) -> None:
    _cache[(fp, s.key)] = labels
    _cache.move_to_end((fp, s.key))
    while len(_cache) > _CACHE_SIZE:
        _cache.popitem(last=False)


# ---------------------------------------------------------------------------
# The run
# ---------------------------------------------------------------------------


def iter_sensitivity(
    adata, settings: list[Setting], rec: Recorded, stop: Callable[[], bool] | None = None,
) -> Iterator[dict[str, Any]]:
    """Re-run each setting and compare it with the reference clustering.

    Yields a header (the recorded settings, the time estimate), then one result
    per setting. ``adata`` is not changed. ``stop`` is checked between
    sub-steps; once it returns True the run raises :class:`Cancelled`.
    """
    check(adata)
    t0 = time.perf_counter()
    ref_codes, ref_names = _reference_codes(adata, rec.reference)
    inp = _inputs(adata, rec)
    fp = fingerprint(adata, rec)
    yield {
        "n_cells": int(adata.n_obs),
        "reference": rec.reference,
        "recorded": rec.to_dict(),
        "current": rec.current.to_dict(),
        "settings": [s.to_dict() for s in settings],
        "n_current_hvgs": None if inp.current_hvgs is None else int(inp.current_hvgs.sum()),
        "estimate_seconds": estimate_seconds(adata.n_obs, len(settings), rec,
                                             len({s.embedding for s in settings})),
        "prepare_seconds": round(time.perf_counter() - t0, 2),
    }
    embedded: dict[tuple[str, int], tuple[np.ndarray, np.ndarray]] = {}
    for s in settings:
        _check_stop(stop)
        t = time.perf_counter()
        # Settings with the same genes share their PCs (a few MB): only the graph or
        # the algorithm changes.
        if s.embedding not in embedded:
            embedded[s.embedding] = _embed(inp, s, rec, stop)
        mask, labels, _ = run_setting_graph(inp, s, rec, stop, embedded=embedded[s.embedding])
        _remember(fp, s, labels)
        shared = None if inp.current_hvgs is None else int((mask & inp.current_hvgs).sum())
        column = column_name(rec, s)
        yield {
            **s.to_dict(),
            "is_current": s == rec.current,
            "n_hvgs": int(mask.sum()),
            "shared_hvgs": shared,
            **compare(ref_codes, ref_names, labels),
            "seconds": round(time.perf_counter() - t, 2),
            "column": column,
            "exists": column in adata.obs.columns,
        }


def keep(adata, setting: Setting, rec: Recorded) -> dict[str, Any]:
    """Save one setting's clusters as ``column_name(rec, setting)``, unless that
    column exists: the labels the run showed when they are still cached, else
    computed again. The active clustering does not change."""
    column = column_name(rec, setting)
    if column in adata.obs.columns:
        return {"column": column, "created": False, "n_clusters": int(adata.obs[column].nunique())}
    check(adata)
    labels = _cache.get((fingerprint(adata, rec), setting.key))
    if labels is None:
        _, labels = run_setting(_inputs(adata, rec), setting, rec)
    adata.obs[column] = pd.Categorical(np.asarray(labels), categories=labels.categories)
    forget_column_results(adata, column)  # left by an earlier column of that name
    try:
        eff = setting.effective(rec)
        provenance.record_step(
            adata, step="keep_hvg_setting",
            tool="scanpy HVG + PCA + neighbors + leiden/louvain (variable-gene check)",
            params={"hvg_flavor": setting.flavor, "n_top_genes": setting.n_top_genes,
                    "clustering_method": eff.method, "clustering_resolution": rec.resolution,
                    "n_comps": rec.n_comps, "n_neighbors": eff.n_neighbors,
                    "batch_key": rec.batch_key, "column": column,
                    "from": "variable-gene sensitivity check",
                    **({"leiden_iterations": rec.leiden_iterations} if eff.method == "leiden" else {})},
        )
    except Exception as e:  # provenance must never block the result
        logger.warning("Could not record keep_hvg_setting provenance: %s", e)
    return {"column": column, "created": True, "n_clusters": int(len(labels.categories))}
