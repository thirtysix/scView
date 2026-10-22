"""write_derived: an obs/uns edit to a derived file rewrites only obs and uns."""

from __future__ import annotations

import anndata as ad
import numpy as np
import pandas as pd
import pytest

from scview.core.obs_writes import uns_snapshot, write_derived


def _saved(tmp_path) -> str:
    a = ad.AnnData(np.arange(12, dtype=np.float32).reshape(4, 3))
    a.obs["group"] = pd.Categorical(["a", "a", "b", "b"])
    a.uns["kept"] = {"names": np.array(["x", "y"])}
    a.uns["prov"] = "old"
    a.uns["gone"] = {"v": np.array([1, 2])}
    path = str(tmp_path / "derived.h5ad")
    a.write_h5ad(path, compression="gzip")
    return path


def test_edit_of_the_loaded_derived_file_skips_the_full_write(tmp_path, monkeypatch):
    path = _saved(tmp_path)
    a = ad.read_h5ad(path)
    before = uns_snapshot(a)
    a.obs["label"] = pd.Categorical(["Unassigned", "T", "T", "Unassigned"])
    a.uns["prov"] = "new"
    del a.uns["gone"]
    a.uns["added"] = "x"

    def _no_full_write(*_a, **_k):
        raise AssertionError("full write used")

    monkeypatch.setattr(ad.AnnData, "write_h5ad", _no_full_write)
    write_derived(a, path, path, before)

    b = ad.read_h5ad(path)
    assert list(b.obs["label"]) == ["Unassigned", "T", "T", "Unassigned"]
    assert b.uns["prov"] == "new" and b.uns["added"] == "x"
    assert "gone" not in b.uns
    assert list(b.uns["kept"]["names"]) == ["x", "y"]
    np.testing.assert_array_equal(b.X, a.X)
    assert not (tmp_path / "derived.h5ad.partial").exists()


def test_first_save_from_another_file_writes_the_whole_dataset(tmp_path):
    src = _saved(tmp_path)
    a = ad.read_h5ad(src)
    before = uns_snapshot(a)
    a.obs["label"] = "Unassigned"
    out = str(tmp_path / "out.h5ad")
    write_derived(a, out, src, before)
    b = ad.read_h5ad(out)
    assert "label" in b.obs and b.shape == (4, 3)


def test_failed_partial_write_leaves_the_file(tmp_path, monkeypatch):
    path = _saved(tmp_path)
    a = ad.read_h5ad(path)
    before = uns_snapshot(a)
    a.uns["prov"] = "new"

    import anndata.io

    def _boom(*_a, **_k):
        raise RuntimeError("disk full")

    monkeypatch.setattr(anndata.io, "write_elem", _boom)
    with pytest.raises(RuntimeError):
        write_derived(a, path, path, before)
    assert ad.read_h5ad(path).uns["prov"] == "old"
    assert not (tmp_path / "derived.h5ad.partial").exists()


def test_an_added_obsm_entry_is_written_with_the_partial_write(tmp_path, monkeypatch):
    path = _saved(tmp_path)
    a = ad.read_h5ad(path)
    before = uns_snapshot(a)
    a.obsm["X_umap_s1_k10"] = np.arange(8, dtype=np.float32).reshape(4, 2)
    a.uns["prov"] = "new"

    def _no_full_write(*_a, **_k):
        raise AssertionError("full write used")

    monkeypatch.setattr(ad.AnnData, "write_h5ad", _no_full_write)
    write_derived(a, path, path, before, obsm_keys=["X_umap_s1_k10"])
    b = ad.read_h5ad(path)
    np.testing.assert_array_equal(b.obsm["X_umap_s1_k10"], a.obsm["X_umap_s1_k10"])
    assert b.uns["prov"] == "new"
    # Replaced in place on a second save.
    a.obsm["X_umap_s1_k10"] = a.obsm["X_umap_s1_k10"] * 2
    write_derived(a, path, path, uns_snapshot(a), obsm_keys=["X_umap_s1_k10"])
    np.testing.assert_array_equal(ad.read_h5ad(path).obsm["X_umap_s1_k10"][3], [12, 14])
