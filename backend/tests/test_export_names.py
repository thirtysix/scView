"""Names holding "/" where they leave scView: shown and exported as typed.

``core/h5_names.py`` stores "CD4/CD8" as ``CD4%2fCD8`` in every h5ad. That key is the
file's business: a Seurat export hands R the column "CD4/CD8" (the staging file stores it
escaped, converter/export_rds.R reads it back; converter/tests covers the R side), a
header read from a file lists a layer and a layout by their names, and a large-dataset
run whose save fails says so in the same words as a run in memory.
"""

from __future__ import annotations

import json
from pathlib import Path

import anndata as ad
import h5py
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

from scview.core import chunked, csr_store, large_run, preflight, rds_export
from scview.core.pipeline import PipelineParams
from tests.test_rds_export import _A, _processed


def _slashed() -> ad.AnnData:
    a = _processed()
    a.obs["CD4/CD8"] = pd.Categorical(np.where(np.arange(a.n_obs) % 2, "T/NK", "T"), categories=["T", "T/NK"])
    a.obs["CD4"] = np.arange(a.n_obs)
    a.uns["CD4/CD8_colors"] = ["#111111", "#222222"]
    a.uns["scview_active_clustering"] = "CD4/CD8"
    return a


def test_seurat_export_keeps_a_slash_in_a_column_name(tmp_path: Path):
    p = rds_export.plan(_A(_slashed()), "seurat", "Set", "s1")
    assert {"CD4/CD8", "CD4"} <= set(p.obs.columns) and "CD4_CD8" not in p.obs.columns
    assert list(p.obs["CD4/CD8"].cat.categories) == ["T", "T/NK"]  # values are never escaped
    assert not [n for n in p.notes if "renamed" in n]
    assert p.meta["cluster_column"] == "CD4/CD8"
    assert p.meta["colors"]["CD4/CD8_colors"] == ["#111111", "#222222"]
    out = rds_export.write_staging(p, tmp_path / "input.h5ad", {})
    with h5py.File(out, "r") as f:  # one key in the file, beside "CD4": what the R side decodes
        assert {"CD4%2fCD8", "CD4"} <= set(f["obs"].keys())
        assert isinstance(f["obs"]["CD4"], h5py.Dataset)
        meta = json.loads(ad.io.read_elem(f["uns/scview_rexport"]))
    assert meta["cluster_column"] == "CD4/CD8"  # JSON text: as typed


def test_left_out_results_are_listed_by_their_column_as_typed():
    from scview.core import cellxgene_export as cx
    from scview.core.uns_keys import markers_key

    a = _slashed()
    a.uns[markers_key("CD4/CD8")] = {"params": {"groupby": "CD4/CD8"}}
    assert "rank_genes_groups__CD4%2FCD8" in a.uns  # the key in memory and in the file
    p = rds_export.plan(_A(a), "seurat", "Set", "s1")
    assert "rank_genes_groups__CD4/CD8" in p.dropped["uns"]
    assert not [k for k in p.dropped["uns"] + p.meta["left_out"] if "%2F" in k]
    c = cx.plan(_A(a), cx.Options())
    assert "rank_genes_groups__CD4/CD8" in c.dropped["uns"]


def test_sce_export_names_the_column_with_an_underscore_and_says_so():
    p = rds_export.plan(_A(_slashed()), "sce", "Set", "s1")
    assert "CD4_CD8" in p.obs.columns and "CD4/CD8" not in p.obs.columns
    assert any("'CD4/CD8' renamed 'CD4_CD8'" in n for n in p.notes)
    assert p.meta["colors"]["CD4_CD8_colors"] == ["#111111", "#222222"]


def _file_with_slash_names(path: Path) -> Path:
    a = _processed()
    a.layers["norm/alt"] = sp.csc_matrix(a.X)
    a.obsm["X_umap/alt"] = np.asarray(a.obsm["X_umap"]).copy()
    a.write_h5ad(path)
    with h5py.File(path, "r") as f:
        assert "norm%2falt" in f["layers"] and "X_umap%2falt" in f["obsm"]
    return path


def test_a_header_read_from_a_file_lists_names_as_typed(tmp_path: Path):
    path = _file_with_slash_names(tmp_path / "d.h5ad")
    h = preflight.read_header(path)
    shown = h.to_dict()
    assert "layers/norm/alt" in [m["where"] for m in shown["matrices"]]
    assert "X_umap/alt" in shown["obsm"]
    assert not [k for k in json.dumps(shown).split('"') if "%2f" in k]
    assert h.matrix("layers/norm%2falt") is not None  # the stored path still addresses the file


def test_column_stored_matrices_are_listed_as_typed(tmp_path: Path):
    path = _file_with_slash_names(tmp_path / "d.h5ad")
    assert "layers/norm/alt" in csr_store.csc_in_file(path)
    assert "layers/norm/alt" in [m["where"] for m in csr_store.csc_sizes(path)]


def test_a_large_run_whose_save_fails_says_it_as_a_run_in_memory_does(tmp_path, monkeypatch):
    monkeypatch.setattr(chunked, "rows_per_block", lambda *a, **k: 100)
    a = _processed()
    a.X = a.layers["counts"].copy()
    del a.layers["counts"]
    a.raw = None
    src = tmp_path / "in" / "d.h5ad"
    src.parent.mkdir()
    a.write_h5ad(src)
    out = tmp_path / "out" / "d.h5ad"

    def fail(self):
        raise TypeError("Incompatible object (Dataset) already exists")

    monkeypatch.setattr(large_run.LargeRun, "save", fail)
    with pytest.raises(Exception) as err:  # noqa: B017, PT011 - the text is the test
        large_run.run(src, ["qc_metrics", "normalization", "log_transform"], PipelineParams(chunked="on"), str(out))
    text = str(err.value)
    assert "could not be saved" in text and "as it was before the run" in text
    assert "Incompatible object" in text
    assert not out.exists()
