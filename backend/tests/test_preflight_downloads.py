"""Preflight counts the first-use downloads a run waits for: the first annotation run fetched
the marker databases (114 s measured, 20 s estimated)."""

from __future__ import annotations

import json
from pathlib import Path

import anndata as ad
import numpy as np
import pytest
import scipy.sparse as sp

from scview.config import get_settings
from scview.core import marker_db, preflight as pf
from scview.core.pipeline import PipelineParams

API = "/api/v1"
MB = 1 << 20


@pytest.fixture
def no_marker_db(tmp_path, monkeypatch):
    monkeypatch.setattr(marker_db, "marker_db_dir", lambda: tmp_path / "marker_db")
    monkeypatch.setattr(marker_db, "auto_download_enabled", lambda: True)


@pytest.fixture
def llm_key(monkeypatch):
    monkeypatch.setattr(get_settings(), "LLM_API_KEY", "test-key")


def test_the_ai_annotation_waits_for_the_marker_databases(no_marker_db, llm_key):
    out = pf.first_downloads(["clustering", "cell_type_annotation"], PipelineParams())
    assert [d.what for d in out] == ["the marker databases"]
    d = out[0]
    assert d.step == "cell_type_annotation" and d.bytes == pf.MARKER_DB_DOWNLOAD_BYTES
    assert d.seconds == pytest.approx(pf.MARKER_DB_DOWNLOAD_BYTES / pf.DOWNLOAD_BYTES_PER_SECOND
                                      + pf.MARKER_DB_FIXED_SECONDS)
    assert pf.downloads_text(out) == (
        f", including about {pf._dur(d.seconds)} to download the marker databases once "
        "(76 MB)")
    assert pf._dur(d.seconds) == "2 min"


def test_nothing_when_present_off_or_not_fetched(no_marker_db, llm_key, monkeypatch):
    steps = ["cell_type_annotation"]
    assert pf.first_downloads(["clustering"], PipelineParams()) == []
    assert pf.first_downloads(steps, PipelineParams(annotation_method="marker_score",
                                                    annotation_marker_sets="builtin")) == []
    assert len(pf.first_downloads(steps, PipelineParams(annotation_method="marker_score"))) == 1
    monkeypatch.setattr(get_settings(), "LLM_API_KEY", "")
    monkeypatch.setattr(get_settings(), "DEEPINFRA_API_KEY", "")
    assert pf.first_downloads(steps, PipelineParams()) == []  # no key, no fallback: nothing runs
    monkeypatch.setattr(marker_db, "auto_download_enabled", lambda: False)
    assert pf.first_downloads(steps, PipelineParams(annotation_method="marker_score")) == []
    monkeypatch.setattr(marker_db, "auto_download_enabled", lambda: True)
    monkeypatch.setattr(marker_db, "_installed", lambda _d, name: True)
    assert pf.first_downloads(steps, PipelineParams(annotation_method="marker_score")) == []


def test_only_the_parts_still_to_fetch(no_marker_db, monkeypatch):
    """After a first fetch where CellMarker timed out: only it, and not within the cooldown."""
    import time

    monkeypatch.setattr(marker_db, "_installed", lambda _d, name: name != "cellmarker3")
    monkeypatch.setattr(marker_db, "read_manifest", lambda _d: {"failures": {}})
    params = PipelineParams(annotation_method="marker_score")
    (d,) = pf.first_downloads(["cell_type_annotation"], params)
    assert d.bytes == pf.MARKER_DB_PARTS["cellmarker3"][0]
    monkeypatch.setattr(marker_db, "read_manifest",
                        lambda _d: {"failures": {"cellmarker3": {"at_epoch": time.time() - 60}}})
    assert pf.first_downloads(["cell_type_annotation"], params) == []  # not retried yet


def test_a_celltypist_model_not_yet_fetched(tmp_path, monkeypatch):
    monkeypatch.setenv("CELLTYPIST_FOLDER", str(tmp_path))
    params = PipelineParams(annotation_method="celltypist", celltypist_model="Immune_All_Low")
    out = pf.first_downloads(["cell_type_annotation"], params)
    assert [d.what for d in out] == ["the CellTypist model"]
    (tmp_path / "data" / "models").mkdir(parents=True)
    (tmp_path / "data" / "models" / "Immune_All_Low.pkl").write_bytes(b"x")
    assert pf.first_downloads(["cell_type_annotation"], params) == []


def test_mouse_enrichment_waits_for_the_mouse_collections(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from scview.core import species

    monkeypatch.setenv("MSIGDB_DIR", str(tmp_path))
    a = ad.AnnData(sp.csr_matrix(np.ones((5, 4), dtype=np.float32)))
    said = {"species": "mouse"}
    monkeypatch.setattr(species, "detect_species", lambda _a: SimpleNamespace(species=said["species"]))
    out = pf.first_downloads(["enrichment"], PipelineParams(), a)  # the defaults: mouse ones for mouse
    assert [d.what for d in out] == ["the mouse MSigDB collections"]
    said["species"] = "human"
    assert pf.first_downloads(["enrichment"], PipelineParams(), a) == []
    asked = PipelineParams(enrichment_collections=["mh.all"])  # named: whatever the species
    assert len(pf.first_downloads(["enrichment"], asked)) == 1


def test_the_time_goes_into_the_step_and_the_sentence():
    old = (20.0, 80.0)
    out = {"steps": [{"step": "clustering", "seconds": [1.0, 4.0]},
                     {"step": "cell_type_annotation", "seconds": [10.0, 40.0]}],
           "verdict": {"seconds": list(old),
                       "sentence": f"About 1.0 GB more than now at its peak (PCA), {pf.time_text(old)}. "
                                   "scView has 4.0 GB free of its 6.0 GB limit."}}
    dl = [pf.FirstDownload("cell_type_annotation", "the marker databases", 76 * 10**6, 135.0)]
    got = pf._add_downloads(out, dl)
    assert got["steps"][0]["seconds"] == [1.0, 4.0]
    assert got["steps"][1]["seconds"] == [145.0, 175.0]
    assert got["verdict"]["seconds"] == [155.0, 215.0]
    assert got["verdict"]["sentence"].startswith(
        "About 1.0 GB more than now at its peak (PCA), roughly 3 to 4 min, including about 2 min to "
        "download the marker databases once (76 MB). scView has")
    assert got["downloads"] == [{"step": "cell_type_annotation", "what": "the marker databases",
                                 "bytes": 76 * 10**6, "seconds": 135.0}]


async def test_the_route_says_it(client, dataset_manager, tmp_data_dir, no_marker_db, llm_key):
    d = Path(tmp_data_dir) / "uploads" / "x"
    d.mkdir(parents=True)
    (d / "metadata.json").write_text(json.dumps({"id": "x", "name": "x", "status": "ready"}))
    rng = np.random.default_rng(0)
    a = ad.AnnData(sp.csr_matrix(rng.poisson(1.0, (200, 50)).astype(np.float32)))
    a.var_names = [f"G{i}" for i in range(50)]
    a.write_h5ad(d / "data.h5ad")
    await dataset_manager.get_or_load_dataset("x")
    r = await client.get(f"{API}/datasets/x/preflight",
                         params={"steps": "normalization,cell_type_annotation"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert "to download the marker databases once" in body["verdict"]["sentence"]
    step = next(s for s in body["steps"] if s["step"] == "cell_type_annotation")
    assert step["seconds"][0] >= pf.MARKER_DB_FIXED_SECONDS
    assert body["downloads"][0]["what"] == "the marker databases"
