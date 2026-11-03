"""Runs whose steps need no expression matrix (core/large_run.py ``run_light``), and the
opt-in a run needs to put a dataset in large-dataset layout (batch 39 item A).

Otero-Garcia 2022's 96k neurons: a neighbour graph on the authors' PCA was costed with X
and .raw read in (7.2 GB), went to large-dataset mode and converted the dataset for good
(DE refused afterwards, .raw gone from its layers). Without the matrices it takes about
0.5 GB, the result is the in-memory run's bit for bit, and the dataset keeps its layout.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import anndata as ad
import h5py
import numpy as np
import pandas as pd
import pytest
import scanpy as sc
import scipy.sparse as sp

from scview.core import large_layout, large_mode, large_run, memory_budget, preflight, provenance, working_copy
from scview.core.anndata_adaptor import AnnDataAdaptor
from scview.core.pipeline import PipelineParams, run_pipeline
from scview.core.run_watchdog import RunControl

API = "/api/v1"
GB = 1024**3
LIGHT = ["neighbors", "clustering", "embeddings"]
SRC = Path(__file__).resolve().parents[1] / "src" / "scview"


def _processed(n: int = 600, g: int = 120, seed: int = 0) -> ad.AnnData:
    """Counts in .raw, log-normalised X, a PCA: as a published dataset (CELLxGENE's)."""
    rng = np.random.default_rng(seed)
    counts = rng.poisson(rng.gamma(1.0, 1.0, g), (n, g)).astype(np.float32)
    counts[: n // 2, :15] *= 5
    a = ad.AnnData(sp.csr_matrix(counts))
    a.var_names = [f"G{i}" for i in range(g)]
    a.obs_names = [f"c{i}" for i in range(n)]
    a.obs["grp"] = pd.Categorical(np.where(np.arange(n) < n // 2, "a", "b"))
    a.obs["donor"] = pd.Categorical([f"d{i % 4}" for i in range(n)])
    a.raw = a.copy()
    sc.pp.normalize_total(a, target_sum=1e4)
    sc.pp.log1p(a)
    sc.pp.pca(a, n_comps=10)
    a.layers["extra"] = a.X.copy()
    return a


def _write(a: ad.AnnData, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    a.write_h5ad(path, compression="gzip")
    return path


def _params(**kw) -> PipelineParams:
    p = PipelineParams(n_neighbors=10, clustering_method="louvain", umap_3d="off")
    for k, v in kw.items():
        setattr(p, k, v)
    return p


def _stored(path: Path, key: str) -> dict:
    """Every dataset under ``key`` of an h5ad: its values and attributes."""
    out = {}
    with h5py.File(path, "r") as f:
        def visit(name, node):
            if isinstance(node, h5py.Dataset):
                out[name] = (node[()].tobytes() if node.dtype.kind != "O" else list(node[()]),
                             dict(node.attrs))
        node = f[key]
        if isinstance(node, h5py.Dataset):
            visit(key, node)
        else:
            node.visititems(visit)
    return out


# -- which runs ---------------------------------------------------------------------------


def test_only_matrix_free_steps_on_an_ordinary_file(tmp_path, monkeypatch):
    src = _write(_processed(), tmp_path / "d.h5ad")
    assert large_run.light_run_ok(src, ["neighbors"])
    assert large_run.light_run_ok(src, LIGHT + ["uncorrected_embedding", "batch_correction", "enrichment"])
    assert not large_run.light_run_ok(src, ["pca", "neighbors"])
    assert not large_run.light_run_ok(src, ["neighbors", "marker_genes"])
    assert not large_run.light_run_ok(src, [])
    assert not large_run.light_run_ok(None, ["neighbors"])
    monkeypatch.setattr(large_run, "is_legacy_layout", lambda _p: True)  # anndata < 0.7's
    assert not large_run.light_run_ok(src, ["neighbors"])


class _DM:
    def cached_ids(self):
        return []


def test_the_mode_is_light_whatever_the_memory_unless_large_mode_is_chosen(tmp_path, monkeypatch):
    src = _write(_processed(), tmp_path / "d.h5ad")
    adaptor = AnnDataAdaptor(str(src))
    monkeypatch.setattr(memory_budget, "read_budget", lambda *a, **k: memory_budget.MemoryBudget(
        GB, GB, True, "container limit"))  # nothing free
    monkeypatch.setattr(large_run, "in_memory_need", lambda *a: (10**15, 10**15))
    for chunked in ("auto", "off"):
        m = large_run.choose_mode(_DM(), "d", adaptor, ["neighbors"], _params(chunked=chunked))
        assert m.light and not m.chunked and "need no expression matrix" in m.reason
        assert m.to_dict()["light"] is True
    assert large_run.choose_mode(_DM(), "d", adaptor, ["neighbors"], _params(chunked="on")).chunked
    m = large_run.choose_mode(_DM(), "d", adaptor, ["pca", "neighbors"], _params())
    assert m.chunked and not m.light


# -- the run ------------------------------------------------------------------------------


def test_a_light_run_is_the_in_memory_run_and_keeps_the_matrices(tmp_path):
    a = _processed()
    src = _write(a, tmp_path / "up" / "d.h5ad")
    out = tmp_path / "derived" / "d.h5ad"
    out.parent.mkdir()
    res = large_run.run_light(src, LIGHT, _params(), str(out), None)
    assert res.status == "complete" and res.steps_run == LIGHT
    full, ref = run_pipeline(ad.read_h5ad(src), LIGHT, _params())
    assert ref.steps_run == LIGHT
    saved = ad.read_h5ad(out)
    for k in ("connectivities", "distances"):
        assert (saved.obsp[k] != full.obsp[k]).nnz == 0
    np.testing.assert_array_equal(saved.obsm["X_umap"], full.obsm["X_umap"])
    col = "scview_louvain_r0.5"
    assert (saved.obs[col].astype(str) == full.obs[col].astype(str)).all()
    assert saved.uns["scview_active_clustering"] == col
    # The matrices are the source's as stored: X, .raw (with its var) and the layers.
    for key in ("X", "raw", "layers"):
        assert _stored(out, key) == _stored(src, key), key
    assert saved.raw is not None and saved.raw.n_vars == a.n_vars
    assert not large_layout.is_large(saved)
    steps = [h["step"] for h in provenance.read_provenance(saved)["history"]]
    assert steps[-3:] == ["neighbors", "clustering", "embeddings"]


def test_a_light_run_can_replace_its_own_file(tmp_path):
    src = _write(_processed(), tmp_path / "d.h5ad")
    before = _stored(src, "X")
    res = large_run.run_light(src, ["neighbors"], _params(), str(src), None)
    assert res.status == "complete"
    assert _stored(src, "X") == before and "connectivities" in ad.read_h5ad(src).obsp


def test_a_mismatched_matrix_is_never_saved(tmp_path):
    src = _write(_processed(), tmp_path / "d.h5ad")
    light = large_run.read_light(src)
    write = large_run.light_writer(src)
    with pytest.raises(large_run.LargeRunError, match="does not match"):
        write(light[:10].copy(), str(tmp_path / "out.h5ad"))


def test_the_light_estimate_is_the_matrix_free_steps_and_the_libraries_once(tmp_path):
    src = _write(_processed(), tmp_path / "d.h5ad")
    h = preflight.read_header(src, "h5ad")
    warm = preflight.estimate_light_run(h, LIGHT, _params(), warm=True)
    cold = preflight.estimate_light_run(h, LIGHT, _params(), warm=False)
    assert [e.step for e in warm.steps] == LIGHT + ["save"]
    assert warm.peak_bytes == preflight.estimate_chunked_run(h, LIGHT, _params(), large=True).peak_bytes
    assert cold.peak_bytes - warm.peak_bytes == preflight.LIGHT_WARMUP
    # held from the first step that loads the libraries, counted once
    assert all(c.peak_bytes - w.peak_bytes == preflight.LIGHT_WARMUP for c, w in zip(cold.steps, warm.steps))
    # nothing like the in-memory need: no matrix is read
    full = preflight.estimate_run(h, steps=LIGHT, params=_params(), warm=True)
    assert warm.peak_bytes < full.peak_bytes


# -- through the endpoints ----------------------------------------------------------------


def _register(data_dir: Path, did: str = "otero") -> Path:
    d = data_dir / "uploads" / did
    d.mkdir(parents=True, exist_ok=True)
    (d / "metadata.json").write_text(json.dumps({"id": did, "name": did, "status": "ready"}))
    return _write(_processed(), d / "data.h5ad")


def _result(r) -> dict:
    events = [line for line in r.text.splitlines() if line.startswith("data: ")]
    return json.loads(events[-1][len("data: "):])


@pytest.mark.parametrize("path", ["assessment/run", "assessment/run-stream"])
async def test_a_backed_dataset_s_graph_reads_no_matrix_and_keeps_its_layers(
        client, dataset_manager, monkeypatch, path):
    from scview.core import anndata_adaptor

    monkeypatch.setattr(anndata_adaptor, "BACKED_MIN_BYTES", 0)  # opens backed, as Otero's did
    src = _register(Path(dataset_manager.data_dir))
    assert (await dataset_manager.get_or_load_dataset("otero")).adata.isbacked
    monkeypatch.setattr(large_run, "in_memory_need", lambda *a: (10**15, 10**15))  # would not fit

    def boom(*_a, **_k):
        raise AssertionError("read into memory")

    monkeypatch.setattr(working_copy, "in_memory", boom)
    monkeypatch.setattr(large_run, "run", boom)  # never large-dataset mode
    r = await client.post(f"{API}/datasets/otero/{path}", json={"steps": ["neighbors"],
                                                                 "params": {"n_neighbors": 10}})
    assert r.status_code == 200, r.text
    result = _result(r) if path.endswith("stream") else r.json()
    assert result["status"] == "complete" and result["mode"]["light"] is True
    assert result["mode"]["chunked"] is False
    adaptor = await dataset_manager.get_or_load_dataset("otero")
    assert not large_layout.is_large(adaptor.adata) and adaptor.adata.raw is not None
    assert "connectivities" in adaptor.adata.obsp
    out = Path(result["output_path"])
    assert out != src and _stored(out, "raw") == _stored(src, "raw")
    # DE still runs on it (the Otero agent's next call was refused).
    r = await client.post(f"{API}/datasets/otero/de/compare",
                          json={"split_by": "grp", "group_a": "a", "group_b": "b"})
    assert r.status_code == 200, r.text


async def test_a_run_that_would_convert_is_refused_without_the_opt_in(client, dataset_manager, monkeypatch):
    _register(Path(dataset_manager.data_dir))
    monkeypatch.setattr(large_run, "in_memory_need", lambda *a: (10**15, 10**15))

    def boom(*_a, **_k):
        raise AssertionError("ran")

    monkeypatch.setattr(large_run, "run", boom)
    body = {"steps": ["pca", "neighbors"], "allow_large_mode": False}
    r = await client.post(f"{API}/datasets/otero/assessment/run", json=body)
    assert r.status_code == 409, r.text
    detail = r.json()["detail"]
    assert "large-dataset mode" in detail and "differential expression" in detail
    assert ".raw is moved out" in detail and "Nothing was run" in detail
    r = await client.post(f"{API}/datasets/otero/assessment/run-stream", json=body)
    assert "event: error" in r.text and "Nothing was run" in r.text
    assert not (Path(dataset_manager.data_dir) / "derived" / "otero" / "data.h5ad").exists()
    assert not dataset_manager.is_busy("otero")


async def test_preflight_says_light_or_what_a_conversion_turns_off(client, dataset_manager, monkeypatch):
    _register(Path(dataset_manager.data_dir))
    r = await client.get(f"{API}/datasets/otero/preflight", params={"steps": "neighbors,clustering"})
    body = r.json()
    assert body["mode"]["light"] is True and body["mode"]["chunked"] is False
    assert body["verdict"]["sentence"].startswith("Without its matrices")
    assert [s["step"] for s in body["steps"]] == ["neighbors", "clustering", "save"]
    assert body["in_memory_estimate"]["steps"]
    r = await client.get(f"{API}/datasets/otero/preflight", params={"steps": "pca,neighbors", "chunked": "on"})
    body = r.json()
    assert body["mode"]["chunked"] is True and body["mode"]["converts"] is True
    assert body["mode"]["turns_off"] == list(large_mode.OFF_IN_LARGE_MODE)
    assert preflight.CONVERTS_SENTENCE.strip() in body["verdict"]["sentence"]
    assert large_mode.conversion_note() in body["verdict"]["notes"]
    r = await client.get(f"{API}/datasets/otero/preflight", params={"steps": "pca,neighbors"})
    assert r.json()["mode"] == {"chunked": False, "requested": "auto", "reason": "The run fits in memory.",
                                "light": False}


def test_every_large_dataset_refusal_is_named_in_what_a_conversion_turns_off():
    """Each ``large_mode.check(..., "<what>", ...)`` and refusal constant across the API is in
    ``OFF_IN_LARGE_MODE``, so the sentence before a conversion stays complete."""
    named = " ".join(large_mode.OFF_IN_LARGE_MODE).lower()
    found = set()
    for f in list((SRC / "api" / "v1").glob("*.py")) + list((SRC / "core").glob("*.py")):
        text = f.read_text()
        found.update(re.findall(r'large_mode\.check\(\s*\w+,\s*"([^"{]+)"', text))
        # constants a refusal names (``check(adaptor, _WHAT, ...)``, ``LargeDatasetError(_WHAT``)
        for const in set(re.findall(r'(?:check\(\s*\w+,|LargeDatasetError\()\s*(_\w*WHAT)\b', text)):
            found.update(re.findall(rf'^{const} = "([^"]+)"', text, flags=re.M))
    assert {"Differential expression", "The DE audit", "Copy-number inference", "Reference mapping",
            "Gene programs (consensus NMF)"} <= found
    missing = sorted(w for w in found if re.sub(r"^the ", "", w.lower()) not in named)
    assert missing == []
    for _step, (what, _why) in large_run.REFUSED_STEPS.items():
        assert what.lower() in named
    assert large_run.REFUSED_BATCH_METHODS.keys() <= {"scanorama", "bbknn"}


def test_the_control_allows_large_mode_unless_told_not_to():
    assert RunControl().allow_large is True
    assert RunControl(allow_large=False).allow_large is False
