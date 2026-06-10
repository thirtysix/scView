"""Cell-type annotation step: groupby resolution + method dispatch.

The CellTypist path needs a model download (network), so it's exercised manually;
these hermetic tests cover the routing logic that guards it.
"""

from __future__ import annotations

import anndata as ad
import numpy as np
import pandas as pd
import pytest

from scview.core.pipeline import (
    PipelineParams,
    _resolve_annotation_groupby,
    _run_cell_type_annotation,
)


def _clustered(n: int = 20) -> ad.AnnData:
    a = ad.AnnData(X=np.random.RandomState(0).poisson(1, (n, 5)).astype("float32"))
    a.obs["cluster"] = pd.Categorical(["a", "b"] * (n // 2))
    return a


def test_groupby_resolves_to_clustering_column():
    a = _clustered()
    assert _resolve_annotation_groupby(a, PipelineParams()) == "cluster"
    # explicit override is honored
    a.obs["my_clusters"] = a.obs["cluster"]
    assert _resolve_annotation_groupby(
        a, PipelineParams(annotation_groupby="my_clusters")
    ) == "my_clusters"


def test_groupby_empty_without_clustering():
    """No clustering -> "" so CellTypist over-clusters internally (not an error)."""
    a = ad.AnnData(X=np.zeros((6, 3), dtype="float32"))
    assert _resolve_annotation_groupby(a, PipelineParams()) == ""


def test_unknown_and_unimplemented_methods_raise():
    a = _clustered()
    with pytest.raises(ValueError):
        _run_cell_type_annotation(a, PipelineParams(annotation_method="bogus"))
    with pytest.raises(ValueError):  # marker_score not implemented yet
        _run_cell_type_annotation(a, PipelineParams(annotation_method="marker_score"))


def test_llm_requires_clustering():
    """LLM-from-markers needs clusters to label; with none it errors before any LLM call."""
    a = ad.AnnData(X=np.zeros((6, 3), dtype="float32"))
    with pytest.raises(ValueError):
        _run_cell_type_annotation(a, PipelineParams(annotation_method="llm"))


def test_parse_cluster_labels():
    from scview.core.pipeline import _parse_cluster_labels

    raw = "B: B cells\nCD14 Mono: Monocytes\n- NK: NK cells"
    out = _parse_cluster_labels(raw, ["B", "CD14 Mono", "NK"])
    assert out == {"B": "B cells", "CD14 Mono": "Monocytes", "NK": "NK cells"}


def test_annotation_step_is_registered():
    from scview.core.pipeline import ALL_STEPS, _STEP_PROVENANCE, _STEP_RUNNERS

    assert "cell_type_annotation" in ALL_STEPS
    assert "cell_type_annotation" in _STEP_RUNNERS
    assert "cell_type_annotation" in _STEP_PROVENANCE


def test_celltype_cluster_map_for_copilot():
    """The co-pilot's cluster -> cell-type map: prefers the recorded uns mapping,
    else majority cell type per cluster; handles the _celltypeAnno naming."""
    from scview.core.assistant import _celltype_cluster_map

    a = ad.AnnData(X=np.zeros((6, 2), dtype="float32"))
    a.obs["cluster"] = pd.Categorical(["0", "0", "1", "1", "2", "2"])
    a.obs["cell_type"] = pd.Categorical(["B", "B", "NK", "NK", "Mono", "Mono"])

    a.uns["cell_type_llm_mapping"] = {"0": "B cell", "1": "NK cell", "2": "Monocyte"}
    assert _celltype_cluster_map(a, "cell_type") == {"0": "B cell", "1": "NK cell", "2": "Monocyte"}

    del a.uns["cell_type_llm_mapping"]
    a.uns["scview_active_clustering"] = "cluster"
    assert _celltype_cluster_map(a, "cell_type") == {"0": "B", "1": "NK", "2": "Mono"}

    a.obs["cluster_celltypeAnno"] = a.obs["cell_type"].values
    assert _celltype_cluster_map(a, "cluster_celltypeAnno") == {"0": "B", "1": "NK", "2": "Mono"}
