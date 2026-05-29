"""Dependency-aware re-run planning — "edit & re-run from step k".

Changing a step's parameters only invalidates the steps *downstream* of it.
Re-running clustering/embeddings/markers/enrichment is cheap and safe because
they operate on artifacts already in the file (the PCA + neighbour graph),
needing no recompute of the expensive upstream steps and no anchor restore.
Editing an upstream step (normalisation, scaling, PCA, …) changes the
expression matrix or graph, so it needs a full reprocess from the counts anchor
(handled by the existing Reprocess flow).

This module is pure planning; the executor (an endpoint) reuses ``run_pipeline``
on the minimal step set. See ``docs/PROVENANCE.md`` (undo/branching).
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from scview.core.pipeline import ALL_STEPS

# step -> the steps it directly depends on (upstream).
_DEPENDS_ON: dict[str, tuple[str, ...]] = {
    "reset_to_counts": (),
    "qc_metrics": (),
    "filtering": ("qc_metrics",),
    "normalization": ("filtering",),
    "log_transform": ("normalization",),
    "highly_variable_genes": ("log_transform",),
    "scaling": ("highly_variable_genes",),
    "pca": ("scaling",),
    "batch_correction": ("pca",),
    "neighbors": ("pca", "batch_correction"),
    "clustering": ("neighbors",),
    "embeddings": ("neighbors",),
    "marker_genes": ("clustering",),
    "enrichment": ("marker_genes",),
    "cell_cycle": ("log_transform",),
}

# Steps re-runnable in place against existing artifacts (no X/graph recompute,
# no anchor restore). Everything else, when edited, needs a full reprocess.
INPLACE_RERUNNABLE = frozenset(
    {"clustering", "embeddings", "marker_genes", "enrichment", "cell_cycle"}
)


def descendants(step: str) -> set[str]:
    """All steps that transitively depend on ``step``."""
    out: set[str] = set()
    changed = True
    while changed:
        changed = False
        for s, deps in _DEPENDS_ON.items():
            if s in out or s == step:
                continue
            if step in deps or (out & set(deps)):
                out.add(s)
                changed = True
    return out


class RerunPlan(BaseModel):
    edited_step: str
    rerun_steps: list[str] = Field(default_factory=list)  # minimal ordered set to re-run
    kept_steps: list[str] = Field(default_factory=list)  # previously-run, left untouched
    requires_reprocess: bool = False  # True if the edit is upstream
    message: str = ""


def plan_rerun(history_steps: list[str], edited_step: str) -> RerunPlan:
    """Plan the minimal re-run after editing ``edited_step``'s parameters,
    given the steps already run (from provenance history)."""
    if edited_step not in _DEPENDS_ON:
        return RerunPlan(edited_step=edited_step, message=f"Unknown step '{edited_step}'.")

    ran = [s for s in ALL_STEPS if s in set(history_steps)]
    ran_set = set(ran)
    affected = ({edited_step} | descendants(edited_step)) & (ran_set | {edited_step})
    rerun = [s for s in ALL_STEPS if s in affected]
    kept = [s for s in ran if s not in affected]

    if edited_step not in INPLACE_RERUNNABLE:
        return RerunPlan(
            edited_step=edited_step,
            rerun_steps=rerun,
            kept_steps=kept,
            requires_reprocess=True,
            message=(
                f"Changing '{edited_step.replace('_', ' ')}' affects the expression matrix "
                "or neighbour graph, so it needs a full reprocess from counts (use Reprocess "
                "from Scratch)."
            ),
        )

    pretty = ", ".join(s.replace("_", " ") for s in rerun)
    kept_pretty = ", ".join(s.replace("_", " ") for s in kept) or "nothing"
    return RerunPlan(
        edited_step=edited_step,
        rerun_steps=rerun,
        kept_steps=kept,
        requires_reprocess=False,
        message=f"Will re-run: {pretty}. Kept (not recomputed): {kept_pretty}.",
    )
