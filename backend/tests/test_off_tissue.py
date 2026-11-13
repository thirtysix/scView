"""Spots outside the tissue in a spatial .h5ad (QA round 3: a CELLxGENE Visium slide holds
the whole capture grid, 713 of 4,992 spots off the tissue, and they went into Moran's I and
the joint embedding).

A new upload or public import leaves them out of the working copy and says how many; the
original stays as sent. A dataset imported before keeps its spots, and the Data Assessment
and the spatial analyses say how many off-tissue spots it holds."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

from scview.core import off_tissue, provenance
from scview.core import spatial_autocorr as sa

API = "/api/v1"


def _grid(n_side: int = 10, off_rows: int = 3, seed: int = 0) -> ad.AnnData:
    """A square Visium-like grid; its first ``off_rows`` rows of spots are off the tissue,
    empty and NaN in an author column, as in a CELLxGENE slide."""
    rng = np.random.default_rng(seed)
    n = n_side * n_side
    row, col = np.divmod(np.arange(n), n_side)
    off = row < off_rows
    counts = rng.poisson(3.0, size=(n, 12)).astype(np.float32)
    counts[off] = 0
    obs = pd.DataFrame({"in_tissue": (~off).astype(np.int64), "array_row": row, "array_col": col,
                        "Cardiomyocyte": np.where(off, np.nan, rng.uniform(0, 1, n))},
                       index=[f"AAAC{i:04d}-1" for i in range(n)])
    a = ad.AnnData(X=sp.csr_matrix(counts), obs=obs, var=pd.DataFrame(index=[f"G{j}" for j in range(12)]))
    a.raw = a.copy()
    a.obsm["spatial"] = np.column_stack([col * 100.0, row * 100.0])
    a.uns["spatial"] = {"slide1": {"images": {"hires": np.zeros((8, 8, 3), np.uint8)},
                                   "scalefactors": {"tissue_hires_scalef": 0.1, "spot_diameter_fullres": 55.0}},
                        "is_single": True}
    return a


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def test_the_in_tissue_column_is_read_in_any_form():
    assert off_tissue.off_mask([1, 0, 1]).tolist() == [False, True, False]
    assert off_tissue.off_mask(np.array([True, False])).tolist() == [False, True]
    assert off_tissue.off_mask(pd.Categorical(["1", "0"])).tolist() == [False, True]
    assert off_tissue.off_mask([1.0, 0.0]).tolist() == [False, True]
    assert off_tissue.off_mask([1, 2]) is None  # not a 0/1 column
    assert off_tissue.off_mask([1, np.nan]) is None


def test_the_working_copy_leaves_the_off_tissue_spots_out(tmp_path):
    a = _grid()
    src, dst = tmp_path / "slide.h5ad", tmp_path / "out.h5ad"
    a.write_h5ad(src)
    before = _sha(src)
    assert off_tissue.file_count(src) == (30, 100)
    summary = off_tissue.drop_in_file(src, dst, original_filename="slide.h5ad")
    assert summary == {"removed": 30, "kept": 70, "column": "in_tissue"}
    assert _sha(src) == before  # the original is only read
    b = ad.read_h5ad(dst)
    assert b.n_obs == 70 and b.raw.n_obs == 70 and b.obsm["spatial"].shape == (70, 2)
    assert (b.obs["in_tissue"] == 1).all() and b.obs_names[0] == "AAAC0030-1"
    assert np.array_equal(b.X.toarray(), a[a.obs["in_tissue"] == 1].X.toarray())
    assert list(b.uns["spatial"]["slide1"]["images"]) == ["hires"]
    step = provenance.read_provenance(b)["history"][-1]
    assert step["step"] == "ingest" and step["tool"] == "scview.off_tissue"
    assert step["params"]["off_tissue"] == summary
    # Nothing to leave out: nothing written.
    b.write_h5ad(tmp_path / "clean.h5ad")
    assert off_tissue.drop_in_file(tmp_path / "clean.h5ad", tmp_path / "x.h5ad", original_filename="c") is None
    assert not (tmp_path / "x.h5ad").exists()
    # Without coordinates in_tissue is just a column: left alone.
    del a.obsm["spatial"]
    a.write_h5ad(tmp_path / "flat.h5ad")
    assert off_tissue.file_count(tmp_path / "flat.h5ad") is None


def test_a_file_too_large_to_rewrite_keeps_its_spots_and_says_so(tmp_path, monkeypatch):
    _grid().write_h5ad(tmp_path / "slide.h5ad")
    monkeypatch.setattr(off_tissue, "_fits", lambda path: False)
    summary = off_tissue.drop_in_file(tmp_path / "slide.h5ad", tmp_path / "out.h5ad", original_filename="s")
    assert summary["removed"] == 0 and summary["off_tissue"] == 30
    assert "too large" in summary["kept_reason"] and not (tmp_path / "out.h5ad").exists()


@pytest.mark.asyncio
async def test_an_upload_leaves_them_out_and_the_upload_stays_as_sent(client, tmp_data_dir, dataset_manager, tmp_path):
    _grid().write_h5ad(tmp_path / "slide.h5ad")
    r = await client.post(f"{API}/datasets/upload",
                          files={"file": ("slide.h5ad", (tmp_path / "slide.h5ad").read_bytes())})
    assert r.status_code == 201, r.text
    did = r.json()["id"]
    meta = json.loads((tmp_data_dir / "uploads" / did / "metadata.json").read_text())
    assert meta["off_tissue"] == {"removed": 30, "kept": 70, "column": "in_tissue"}
    assert ad.read_h5ad(tmp_data_dir / "uploads" / did / "slide.h5ad").n_obs == 100
    info = (await client.get(f"{API}/datasets/{did}")).json()
    assert info["n_cells"] == 70


@pytest.mark.asyncio
async def test_a_public_import_leaves_them_out(client, dataset_manager, monkeypatch, tmp_path):
    from scview.api.v1 import public_import as api
    from scview.core import public_import as pi
    from tests.test_public_import import LIVER_B, LISTING_REC, FakeDiscover, _events, _local_dir, discover_h5ad

    local = tmp_path / "fetched"
    local.mkdir()
    a = discover_h5ad(tmp_path / "plain.h5ad", n=40)
    off = np.arange(40) < 6
    a.obs["in_tissue"] = (~off).astype(np.int64)
    a.obsm["spatial"] = np.column_stack([np.arange(40) % 8 * 100.0, np.arange(40) // 8 * 100.0])
    a.write_h5ad(local / f"{LIVER_B}.h5ad")
    rec = json.loads(json.dumps(LISTING_REC()))
    rec["assets"][0]["filesize"] = (local / f"{LIVER_B}.h5ad").stat().st_size
    (local / f"{LIVER_B}.json").write_text(json.dumps(rec))
    monkeypatch.setattr(pi, "new_client", FakeDiscover().client)
    monkeypatch.setattr(pi, "memory_verdict", lambda p: {"verdict": "fits", "message": "ok"})
    monkeypatch.setattr(pi, "_HEADERS", {})
    _local_dir(monkeypatch, local)
    api._catalogs.clear()
    try:
        r = await client.post(f"{API}/public-import/discover/import-stream", json={"dataset_version_id": LIVER_B})
        events = _events(r.text)
        assert events[-1][0] == "complete", events
        assert ("stage", "Leaving out the 6 spots marked off-tissue (in_tissue 0)") in [
            (e, d.get("message")) for e, d in events]
        did = events[-1][1]["dataset_id"]
        assert dataset_manager.get_dataset(did).adata.n_obs == 34
        meta = json.loads((Path(dataset_manager.data_dir) / "uploads" / did / "metadata.json").read_text())
        assert meta["off_tissue"]["removed"] == 6
        history = provenance.read_provenance(dataset_manager.get_dataset(did).adata)["history"]
        assert [h.get("tool") for h in history if h["step"] == "ingest"] == ["scview.public_import", "scview.off_tissue"]
        assert ad.read_h5ad(Path(dataset_manager.data_dir) / "uploads" / did / f"{LIVER_B}.h5ad").n_obs == 40
    finally:
        api._catalogs.clear()


def test_a_dataset_imported_before_says_how_many_it_holds():
    from scview.core.assessor import assess_preprocessing

    a = _grid()
    state = assess_preprocessing(a)
    assert state.filtering.done is False
    assert state.filtering.details.startswith("30 spots marked off-tissue (in_tissue 0) are included")
    r = sa.run(a, genes="all")
    assert any(n.startswith("30 spots marked off-tissue") for n in r.notes)
    tissue = a[a.obs["in_tissue"] == 1].copy()
    assert off_tissue.note(tissue) is None and off_tissue.count(tissue) == 0
    assert not any("off-tissue" in n for n in sa.run(tissue, genes="all").notes)
