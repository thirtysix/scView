"""persist_uns edits a copy and swaps it in: a failure mid-write leaves the dataset readable."""

from __future__ import annotations

import anndata as ad
import numpy as np
import pytest

from tests.test_hvg_sensitivity import _register


def _small() -> ad.AnnData:
    a = ad.AnnData(np.arange(20, dtype=np.float32).reshape(5, 4))
    a.uns["scview_keep"] = "old"
    return a


async def _derived(dataset_manager, did):
    adaptor = await dataset_manager.get_or_load_dataset(did)
    adaptor.adata.uns["scview_x"] = "first"
    assert dataset_manager.persist_uns(did, adaptor, ["scview_x"])  # written to derived/
    return adaptor


@pytest.mark.asyncio
async def test_in_place_save_leaves_no_partial_and_reads_back(dataset_manager):
    did = _register(dataset_manager, _small(), "pu1")
    adaptor = await _derived(dataset_manager, did)
    adaptor.adata.uns["scview_x"] = "second"
    assert dataset_manager.persist_uns(did, adaptor, ["scview_x"])  # in place on the derived file
    target = dataset_manager.derived_h5ad_path(did, adaptor.h5ad_path)
    back = ad.read_h5ad(target)
    assert back.uns["scview_x"] == "second" and back.uns["scview_keep"] == "old"
    assert not list(target.parent.glob("*.part"))


@pytest.mark.asyncio
async def test_failure_mid_write_keeps_the_old_file(dataset_manager, monkeypatch):
    did = _register(dataset_manager, _small(), "pu2")
    adaptor = await _derived(dataset_manager, did)
    import anndata.io

    def boom(*a, **k):
        raise OSError("killed mid-write")

    monkeypatch.setattr(anndata.io, "write_elem", boom)
    adaptor.adata.uns["scview_x"] = "second"
    with pytest.raises(OSError):
        dataset_manager.persist_uns(did, adaptor, ["scview_x"])
    target = dataset_manager.derived_h5ad_path(did, adaptor.h5ad_path)
    assert ad.read_h5ad(target).uns["scview_x"] == "first"
    assert not list(target.parent.glob("*.part"))
