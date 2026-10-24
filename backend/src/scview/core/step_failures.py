"""A failed pipeline step skips only the steps that depend on it.

A run used to carry on after a failed step: every later step ran (a PCA that failed left
the neighbour graph, clusters and UMAP to be built from the previous run's PCs) and the
run was saved as "complete". Now each step has the steps it needs (``DEPENDS_ON``, this
module's own map: ``rerun._DEPENDS_ON`` plans re-runs and is coarser), a step whose
needs failed is skipped, and every step that does not need a failed one still runs.
The run is saved with status "finished_with_errors" and the failed and skipped steps.

    step                   needs (in the run or not: through the whole map)
    reset_to_counts        -
    qc_metrics             reset_to_counts
    doublet_detection      reset_to_counts (not qc_metrics: Scrublet reads the counts)
    filtering              reset_to_counts; doublet_detection only with drop_doublets (it
                           judges cells by the QC values, worked out from the counts when
                           qc_metrics did not write them)
    ambient_correction     filtering (the cells it keeps)
    normalization          filtering, ambient_correction (the counts it normalises: a failed
                           correction must not leave the run going on uncorrected counts)
    log_transform          normalization
    highly_variable_genes  log_transform; filtering + ambient_correction alone for Seurat v3 /
                           Pearson residuals, which rank the counts
    scaling                log_transform (not the variable genes: PCA works out the
                           statistics of genes variable after scaling)
    pca                    highly_variable_genes, scaling
    uncorrected_embedding  pca
    batch_correction       pca
    neighbors              pca (not batch_correction: a failed correction leaves X_pca,
                           which the graph is then built on; PCA drops the old Harmony one)
    clustering, embeddings neighbors
    marker_genes           clustering (it groups by the new clustering)
    enrichment             marker_genes
    cell_cycle,
    dissociation_stress    log_transform (log-normalised values)
    cell_type_annotation   clustering, unless annotation_groupby names another column;
                           then log_transform

A step that failed, or was skipped, may leave results of an earlier run that it would have
replaced. When a step it needs ran in this run (so its inputs changed), those results are
stale and are removed (``clear_outputs``): a failed PCA after new variable genes does not
leave the old PCs, graph, clusters, UMAP and markers looking current. When nothing it
needs ran, they still describe the data and are kept (a failed re-clustering keeps the
clustering there was). Label columns (cell types, kept clusterings) are never removed;
the active clustering stops being active. What was removed is recorded with the run's
outcome in provenance (step ``run_errors``), and the replay script removes the same.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from scview.core import scaling
from scview.core.uns_keys import BATCH_MIXING_KEY, enrichment_prefix, markers_key, meta_prefix

FINISHED_WITH_ERRORS = "finished_with_errors"

DEPENDS_ON: dict[str, tuple[str, ...]] = {
    "reset_to_counts": (),
    "qc_metrics": ("reset_to_counts",),
    "doublet_detection": ("reset_to_counts",),
    "filtering": ("reset_to_counts",),
    "ambient_correction": ("filtering",),
    "normalization": ("filtering", "ambient_correction"),
    "log_transform": ("normalization",),
    "highly_variable_genes": ("log_transform",),
    "scaling": ("log_transform",),
    "pca": ("highly_variable_genes", "scaling"),
    "uncorrected_embedding": ("pca",),
    "batch_correction": ("pca",),
    "neighbors": ("pca",),
    "clustering": ("neighbors",),
    "embeddings": ("neighbors",),
    "marker_genes": ("clustering",),
    "enrichment": ("marker_genes",),
    "cell_cycle": ("log_transform",),
    "dissociation_stress": ("log_transform",),
    "cell_type_annotation": ("clustering",),
}


def _p(params, name: str, default=None):
    v = getattr(params, name, default) if params is not None else default
    return default if v is None else v


def depends_on(step: str, params=None) -> tuple[str, ...]:
    """The steps ``step`` needs directly, for these params."""
    if step == "filtering" and _p(params, "drop_doublets", False):
        return ("reset_to_counts", "doublet_detection")
    if step == "highly_variable_genes" and (
            _p(params, "normalization_method", "log") == "pearson_residuals"
            or _p(params, "hvg_flavor", "seurat") == "seurat_v3"):
        return ("filtering", "ambient_correction")  # ranks the counts
    if step == "cell_type_annotation" and _p(params, "annotation_groupby", ""):
        return ("log_transform",)
    return DEPENDS_ON.get(step, ())


def needs(step: str, params=None) -> set[str]:
    """Every step ``step`` needs, through the whole map (also steps not in the run: a
    run of normalization and variable genes without the log transform still needs the
    normalisation)."""
    out: set[str] = set()
    todo = list(depends_on(step, params))
    while todo:
        s = todo.pop()
        if s not in out:
            out.add(s)
            todo.extend(depends_on(s, params))
    return out


@dataclass
class Outcome:
    """What happened to a run's steps so far; decides whether the next one runs."""

    order: list[str]
    params: Any = None
    succeeded: list[str] = field(default_factory=list)
    failed: dict[str, str] = field(default_factory=dict)  # step -> its error
    skipped: dict[str, str] = field(default_factory=dict)  # step -> the failed step it needs
    cleared: dict[str, list[str]] = field(default_factory=dict)  # where -> keys removed
    # Steps whose earlier results were stale and removed: no longer done
    stale_steps: list[str] = field(default_factory=list)
    active_before: str | None = None  # the active clustering when the run started

    def blocked_by(self, step: str) -> str | None:
        """The failed step ``step`` needs (the nearest one in the run's order), or None."""
        hit = [s for s in self.order if s in self.failed and s in needs(step, self.params)]
        return hit[-1] if hit else None

    def stale(self, step: str) -> bool:
        """A step that failed or was skipped left results of an earlier run that are stale:
        a step it needs ran in this run, so its inputs changed."""
        return bool(needs(step, self.params) & set(self.succeeded))

    @property
    def any(self) -> bool:
        return bool(self.failed)

    def record(self) -> dict[str, Any]:
        """The run's outcome as provenance records it (``run_errors``)."""
        out: dict[str, Any] = {"status": FINISHED_WITH_ERRORS, "failed": dict(self.failed)}
        cleared = {k: list(v) for k, v in self.cleared.items() if v}
        for key, value in (("skipped", dict(self.skipped)), ("stale", list(self.stale_steps)),
                           ("cleared", cleared)):
            if value:  # left out when empty: nothing to say in the History
                out[key] = value
        return out


def skip_reason(step: str, because: str) -> str:
    return f"Skipped: it needs {because.replace('_', ' ')}, which failed."


# ---------------------------------------------------------------------------
# What each step writes, removed when it is stale
# ---------------------------------------------------------------------------

HVG_COLUMNS = ("highly_variable", "means", "dispersions", "dispersions_norm", "variances",
               "variances_norm", "residual_variances", "highly_variable_rank",
               "highly_variable_nbatches", "highly_variable_intersection")
CELL_CYCLE_COLUMNS = ("S_score", "G2M_score", "phase")


def _drop(adata, out: dict[str, list[str]], where: str, keys) -> None:
    store = {"obsm": adata.obsm, "varm": adata.varm, "obsp": adata.obsp, "uns": adata.uns}.get(where)
    for k in keys:
        if where in ("var", "obs"):
            frame = adata.var if where == "var" else adata.obs
            if k in frame.columns:
                del frame[k]
                out.setdefault(where, []).append(k)
        elif store is not None and k in store:
            del store[k]
            out.setdefault(where, []).append(k)


def _column_results(adata, columns) -> list[str]:
    """The uns keys of the markers and enrichment stored for ``columns``."""
    keys = []
    for col in columns:
        if not col:
            continue
        keys.append(markers_key(col))
        for prefix in (enrichment_prefix(col), meta_prefix(col)):
            keys += [k for k in adata.uns if isinstance(k, str) and k.startswith(prefix)]
    return keys


def clear_outputs(adata, step: str, outcome: Outcome) -> dict[str, list[str]]:
    """Remove what ``step`` writes, as left by an earlier run; returns where -> keys
    removed (also added to ``outcome.cleared``)."""
    from scview.core.pipeline import looks_like_counts

    params = outcome.params
    out: dict[str, list[str]] = {}
    if step == "highly_variable_genes":
        _drop(adata, out, "var", HVG_COLUMNS)
        _drop(adata, out, "uns", ["hvg"])
    elif step == "scaling":
        _drop(adata, out, "uns", [scaling.PLAN_KEY])
        _drop(adata, out, "var", [scaling.MEAN, scaling.STD])
        # .raw is the log values scaling set; old ones are stale. Counts there are kept.
        if adata.raw is not None and not looks_like_counts(adata.raw.X):
            adata.raw = None
            out.setdefault("raw", []).append("raw")
    elif step == "pca":
        _drop(adata, out, "obsm", ["X_pca", "X_pca_harmony"])
        _drop(adata, out, "varm", ["PCs"])
        _drop(adata, out, "uns", ["pca", BATCH_MIXING_KEY])
    elif step == "uncorrected_embedding":
        _drop(adata, out, "obsm", ["X_umap_uncorrected"])
    elif step == "batch_correction":
        _drop(adata, out, "obsm", ["X_pca_harmony"])
        _drop(adata, out, "uns", [BATCH_MIXING_KEY])
    elif step == "neighbors":
        nb = adata.uns.get("neighbors")
        keys = ["connectivities", "distances"]
        if isinstance(nb, dict):
            keys += [nb.get("connectivities_key"), nb.get("distances_key")]
        _drop(adata, out, "obsp", [k for k in dict.fromkeys(keys) if k])
        _drop(adata, out, "uns", ["neighbors"])
    elif step == "clustering":
        # The column stays (a label column); it is no longer the active clustering.
        _drop(adata, out, "uns", ["scview_active_clustering"])
    elif step == "embeddings":
        _drop(adata, out, "obsm", ["X_umap", "X_umap_3d"])
        _drop(adata, out, "uns", ["umap"])
    elif step == "marker_genes":
        cols = [outcome.active_before, adata.uns.get("scview_active_clustering"),
                *(_p(params, "marker_columns", []) or [])]
        _drop(adata, out, "uns", ["rank_genes_groups", *_column_results(adata, cols)])
    elif step == "enrichment":
        cols = [outcome.active_before, adata.uns.get("scview_active_clustering"),
                *(_p(params, "enrichment_columns", []) or [])]
        keys = [k for k in _column_results(adata, cols) if k not in
                {markers_key(c) for c in cols if c}]
        _drop(adata, out, "uns", keys)
    elif step == "cell_cycle":
        _drop(adata, out, "obs", CELL_CYCLE_COLUMNS)
    elif step == "dissociation_stress":
        from scview.core.stress import SCORE_COLUMN

        _drop(adata, out, "obs", [SCORE_COLUMN])
    for where, keys in out.items():
        bucket = outcome.cleared.setdefault(where, [])
        bucket.extend(k for k in keys if k not in bucket)
    return out


def settle(adata, step: str, outcome: Outcome) -> None:
    """After ``step`` failed or was skipped: remove its stale results (``stale``); the step
    is then no longer done (``stale_steps``: the assessment reads it from provenance)."""
    if outcome.stale(step):
        clear_outputs(adata, step, outcome)
        outcome.stale_steps.append(step)
