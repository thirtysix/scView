"""Cell-type annotation never overwrites a column scView did not write.

``annotation_target`` defaulted to 'cell_type', so an annotation run without it overwrote
a CELLxGENE file's curated cell_type (research/v075-qa/wilson2022). The default is now
scView's own column (<grouping>_celltypeAnno) and an existing column that came with the
data is refused (422 at the routes) unless ``annotation_overwrite`` is set.
"""

from __future__ import annotations

import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest

from scview.core import pipeline
from scview.core.pipeline import PipelineParams, run_pipeline

API = "/api/v1"
GENES = ["CD3D", "CD3E", "CD3G", "TRAC", "MS4A1", "CD79A", "CD79B", "CD19"]


def _adata(curated: bool = True) -> ad.AnnData:
    rng = np.random.default_rng(0)
    X = (rng.random((20, 8)) * 0.1).astype("float32")
    X[:10, :4] += 3.0
    X[10:, 4:] += 3.0
    a = ad.AnnData(X)
    a.var_names = GENES
    a.obs_names = [f"c{i}" for i in range(20)]
    a.obs["cluster"] = pd.Categorical(["0"] * 10 + ["1"] * 10)
    if curated:
        a.obs["cell_type"] = pd.Categorical(["T cell (curated)"] * 10 + ["B cell (curated)"] * 10)
    return a


def _params(**kw) -> PipelineParams:
    return PipelineParams(annotation_method="marker_score", annotation_marker_sets="builtin", **kw)


def _annotation_records(a) -> list[dict]:
    from scview.core import provenance

    return [h for h in provenance.read_provenance(a)["history"] if h["step"] == "cell_type_annotation"]


def test_default_writes_scviews_own_column_and_records_it():
    a = _adata()
    a, res = run_pipeline(a, ["cell_type_annotation"], _params())
    assert not res.errors, res.errors
    assert list(a.obs["cell_type"].unique()) == ["T cell (curated)", "B cell (curated)"]
    assert set(a.obs["cluster_celltypeAnno"]) == {"T cell", "B cell"}
    [rec] = _annotation_records(a)
    assert rec["params"]["annotation_target"] == "cluster_celltypeAnno"
    # its own column from that run may be written again
    a, res = run_pipeline(a, ["cell_type_annotation"], _params(annotation_target="cluster_celltypeAnno"))
    assert not res.errors, res.errors


def test_a_column_that_came_with_the_data_is_refused_unless_overwrite():
    a = _adata()
    a, res = run_pipeline(a, ["cell_type_annotation"], _params(annotation_target="cell_type"))
    assert "did not write" in res.errors["cell_type_annotation"]
    assert a.obs["cell_type"].iloc[0] == "T cell (curated)"
    a, res = run_pipeline(a, ["cell_type_annotation"],
                          _params(annotation_target="cell_type", annotation_overwrite=True))
    assert not res.errors, res.errors
    assert a.obs["cell_type"].iloc[0] == "T cell"
    [rec] = _annotation_records(a)
    assert rec["params"]["annotation_overwrite"] is True


def test_check_annotation_target_rules():
    a = _adata()
    with pytest.raises(pipeline.AnnotationTargetError, match="annotation_overwrite"):
        pipeline.check_annotation_target(a, "cell_type")
    pipeline.check_annotation_target(a, "cell_type", overwrite=True)
    pipeline.check_annotation_target(a, "new_column")  # not there yet: fine
    a.obs["scview_labels"] = "x"
    pipeline.check_annotation_target(a, "scview_labels")
    # a column an annotation of an older scView wrote (its record has no target: cell_type)
    from scview.core import provenance

    provenance.record_step(a, step="cell_type_annotation", tool="scview.cell_type_annotation",
                           params={"annotation_method": "llm"})
    pipeline.check_annotation_target(a, "cell_type")
    assert pipeline.default_annotation_target("leiden") == "leiden_celltypeAnno"
    assert pipeline.default_annotation_target("") == "celltypeAnno"


def _register(dm, did: str) -> None:
    d = Path(dm.data_dir) / "uploads" / did
    d.mkdir(parents=True, exist_ok=True)
    (d / "metadata.json").write_text(json.dumps({"id": did, "name": did, "status": "ready"}))
    _adata().write_h5ad(d / "data.h5ad")


@pytest.mark.parametrize("path", ["assessment/run", "assessment/run-stream"])
async def test_the_run_routes_refuse_it_with_422(client, dataset_manager, path):
    _register(dataset_manager, "ann")
    body = {"steps": ["cell_type_annotation"],
            "params": {"annotation_method": "marker_score", "annotation_target": "cell_type"}}
    r = await client.post(f"{API}/datasets/ann/{path}", json=body)
    assert r.status_code == 422
    assert "'cell_type' is a column scView did not write" in r.json()["detail"]
