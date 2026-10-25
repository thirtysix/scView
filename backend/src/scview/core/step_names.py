"""Pipeline step names: which are real, and what a wrong one probably meant.

A run asked for a step by a name the pipeline does not have ("umap"; the step is
"embeddings") used to return 200 with nothing run and no error. The run
endpoints now refuse such a request with 422, naming the valid steps and, where
it is clear, the one that was meant: from the aliases below (the names other
tools use for a step), else from close spellings (difflib).
"""

from __future__ import annotations

import difflib

from scview.core.pipeline import ALL_STEPS

# Names other tools (scanpy, Seurat, tutorials) use -> the step that does it.
STEP_ALIASES: dict[str, str] = {
    "umap": "embeddings",
    "tsne": "embeddings",
    "t_sne": "embeddings",
    "fa": "embeddings",
    "draw_graph": "embeddings",
    "leiden": "clustering",
    "louvain": "clustering",
    "cluster": "clustering",
    "qc": "qc_metrics",
    "normalize": "normalization",
    "normalise": "normalization",
    "normalisation": "normalization",
    "normalize_total": "normalization",
    "log1p": "log_transform",
    "hvg": "highly_variable_genes",
    "hvgs": "highly_variable_genes",
    "scale": "scaling",
    "harmony": "batch_correction",
    "scanorama": "batch_correction",
    "bbknn": "batch_correction",
    "integration": "batch_correction",
    "neighbours": "neighbors",
    "knn": "neighbors",
    "markers": "marker_genes",
    "rank_genes_groups": "marker_genes",
    "doublets": "doublet_detection",
    "scrublet": "doublet_detection",
    "filter": "filtering",
    "celltypist": "cell_type_annotation",
    "annotation": "cell_type_annotation",
}


def _key(name: str) -> str:
    return str(name).strip().lower().replace("-", "_").replace(" ", "_")


def unknown_steps(steps: list[str]) -> list[str]:
    """The requested names that are not pipeline steps, in request order, once each."""
    known = set(ALL_STEPS)
    return list(dict.fromkeys(s for s in steps if s not in known))


def step_hint(name: str) -> str | None:
    """The step ``name`` most likely meant, or None when nothing is close."""
    key = _key(name)
    if key in ALL_STEPS:
        return key  # right name, wrong case or separators
    if key in STEP_ALIASES:
        return STEP_ALIASES[key]
    close = difflib.get_close_matches(key, ALL_STEPS + list(STEP_ALIASES), n=1, cutoff=0.75)
    if not close:
        return None
    return STEP_ALIASES.get(close[0], close[0])


def unknown_steps_message(unknown: list[str]) -> str:
    """One sentence naming each unknown step (with a hint where there is one) and
    then all the valid steps."""
    parts = []
    for name in unknown:
        hint = step_hint(name)
        parts.append(f"'{name}' (did you mean '{hint}'?)" if hint else f"'{name}'")
    noun = "step" if len(unknown) == 1 else "steps"
    return (f"Unknown pipeline {noun} {', '.join(parts)}. "
            f"Valid steps: {', '.join(ALL_STEPS)}.")
