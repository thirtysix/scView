"""The log-normalised matrix of a dataset opened backed but not large (``activity.lognorm_source``).

A dataset processed in memory and reopened after a restart opens backed when its file is over
``SCVIEW_BACKED_MIN_MB``: every feature on ``activity.lognorm_matrix`` failed on it (``float()
argument must be ... not 'csr_matrix'`` judging the first rows), and past that used the on-disk
matrix as if in memory. Each feature here gives the in-memory numbers on the backed file; the
ones that read the matrix whole hold room for it while they work (``working_copy.reading``).
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import scanpy as sc
import scipy.sparse as sp

from scview.api.v1 import communication as communication_api
from scview.core import activity, anndata_adaptor, marker_selection, mixscape
from scview.core import spatial_autocorr as sa
from scview.core import tf_motifs as tm
from scview.core.anndata_adaptor import AnnDataAdaptor
from tests.test_activity import _cells as _act_cells, _lognorm, networks, small_networks  # noqa: F401
from tests.test_backed_reads import GB, _budget
from tests.test_ccc import _dataset as _ccc_dataset, resource_on_disk  # noqa: F401
from tests.test_ccc_compare import _dataset as _cmp_dataset, toy_resource  # noqa: F401
from tests.test_cytokine_activity import _cells as _cyto_cells, signature, small_signature  # noqa: F401
from tests.test_marker_panel import _data as _panel_data
from tests.test_mixscape import planted
from tests.test_pseudotime_genes import _trend_adata
from tests.test_spatial_autocorr import synthetic
from tests.test_tf_motifs import _dataset as _tf_dataset, _dataset_table, _species_human

API = "/api/v1"


def _register(dm, a: ad.AnnData, did: str) -> str:
    d = Path(dm.data_dir) / "uploads" / did
    d.mkdir(parents=True)
    (d / "metadata.json").write_text(json.dumps({"id": did, "name": f"{did} name", "status": "ready"}))
    a.write_h5ad(d / "data.h5ad")
    return did


def _events(text: str) -> list[tuple[str, dict]]:
    out = []
    for block in text.strip().split("\n\n"):
        if block.startswith(":"):
            continue
        lines = dict(line.split(": ", 1) for line in block.splitlines())
        out.append((lines["event"], json.loads(lines["data"])))
    return out


async def _in_memory_then_backed(dm, did: str, monkeypatch, call):
    """``call()`` on the dataset in memory, then on its file reopened backed (as after a restart
    with the file over the backed threshold): both responses."""
    mem = await call()
    assert mem.status_code == 200, mem.text
    assert not (await dm.get_or_load_dataset(did)).adata.isbacked
    dm.evict(did)
    monkeypatch.setattr(anndata_adaptor, "BACKED_MIN_BYTES", 0)
    backed = await call()
    assert backed.status_code == 200, backed.text
    assert (await dm.get_or_load_dataset(did)).adata.isbacked
    return mem.json() if "json" in mem.headers.get("content-type", "") else mem.text, \
        backed.json() if "json" in backed.headers.get("content-type", "") else backed.text


def _no_timing(d):
    """A response without its timings and ids (they differ run to run)."""
    if isinstance(d, dict):
        return {k: _no_timing(v) for k, v in d.items()
                if k not in ("elapsed_seconds", "seconds", "result_id", "id", "created_at", "reused")}
    if isinstance(d, (list, tuple)):
        return [_no_timing(v) for v in d]
    if isinstance(d, np.ndarray):
        return d.tolist()
    return d


# -- the matrix itself -------------------------------------------------------------------


def _counts(n: int = 240, g: int = 50, seed: int = 0) -> ad.AnnData:
    rng = np.random.default_rng(seed)
    X = rng.poisson(rng.gamma(1.0, 1.0, g), (n, g)).astype(np.float32)
    a = ad.AnnData(sp.csr_matrix(X))
    a.var_names = [f"G{i}" for i in range(g)]
    a.obs_names = [f"c{i}" for i in range(n)]
    return a


def _scaled_x(a: ad.AnnData) -> ad.AnnData:
    """X replaced by z-scores: neither counts nor log values."""
    b = a.copy()
    sc.pp.normalize_total(b, target_sum=1e4)
    sc.pp.log1p(b)
    sc.pp.scale(b)
    return b


def _layout(kind: str) -> ad.AnnData:
    a = _counts()
    if kind == "raw_log":  # scView's pipeline: log values in .raw, z-scores in X
        b = _lognorm(a)
        out = _scaled_x(a)
        out.raw = b
        return out
    if kind == "x_log":
        return _lognorm(a)
    if kind == "x_log_dense":
        b = _lognorm(a)
        b.X = np.asarray(b.X.todense())
        return b
    if kind == "x_counts":
        return a
    if kind == "x_counts_dense":
        a.X = np.asarray(a.X.todense())
        return a
    if kind == "raw_counts":  # counts in .raw for X's genes, z-scores in X
        out = _scaled_x(a)
        out.raw = a
        return out
    if kind == "raw_counts_more_genes":  # .raw holds more genes than X (an HVG subset)
        out = _scaled_x(a)[:, 5:40].copy()
        out.raw = a
        return out
    if kind == "layer_counts":
        out = _scaled_x(a)
        out.layers["counts"] = a.X.copy()
        return out
    raise KeyError(kind)


LAYOUTS = ["raw_log", "x_log", "x_log_dense", "x_counts", "x_counts_dense", "raw_counts",
           "raw_counts_more_genes", "layer_counts"]


def _dense(M) -> np.ndarray:
    return M.toarray() if sp.issparse(M) else np.asarray(M)


@pytest.mark.parametrize("kind", LAYOUTS)
def test_the_backed_matrix_is_the_in_memory_one_whole_by_rows_and_by_blocks(tmp_path, monkeypatch, kind):
    path = tmp_path / "d.h5ad"
    _layout(kind).write_h5ad(path)
    mem = ad.read_h5ad(path)
    monkeypatch.setattr(anndata_adaptor, "BACKED_MIN_BYTES", 0)
    adaptor = AnnDataAdaptor(str(path))
    backed = adaptor.adata
    assert backed.isbacked

    M, names, source = activity.lognorm_matrix(mem)
    choice = activity.lognorm_source(backed)  # judged on the first rows: no TypeError
    assert list(choice.names) == list(names) and choice.source == source
    whole = _dense(activity.lognorm_matrix(backed, read=True)[0])
    np.testing.assert_array_equal(whole, _dense(M))
    keep = np.flatnonzero(np.arange(mem.n_obs) % 3 == 1)
    np.testing.assert_array_equal(_dense(choice.rows(keep)), _dense(M)[keep])
    np.testing.assert_array_equal(_dense(activity.lognorm_source(mem).rows(keep)), _dense(M)[keep])
    blocks = np.vstack([_dense(b) for _, b in choice.blocks(70)])
    np.testing.assert_array_equal(blocks, _dense(M))

    if choice.on_disk:  # never read whole without room held for it
        with pytest.raises(activity.OnDiskMatrix):
            activity.lognorm_matrix(backed)
        assert activity.lognorm_bytes(adaptor) > activity.lognorm_bytes(adaptor, keep.size) > 0
    else:  # counts in a layer: in memory already
        assert activity.lognorm_bytes(adaptor) == 0


def test_a_dataset_in_memory_needs_no_room(tmp_path):
    path = tmp_path / "d.h5ad"
    _layout("raw_log").write_h5ad(path)
    adaptor = AnnDataAdaptor(str(path))
    assert not adaptor.adata.isbacked and activity.lognorm_bytes(adaptor) == 0


def test_fractions_expressing_of_a_backed_dataset_are_counted_by_blocks(tmp_path, monkeypatch):
    """The shared marker choice (LLM annotator, co-pilot facts, ORA) without stored pts: the
    backed matrix is read a block of cells at a time, never whole."""
    a = _lognorm(_counts())
    a.obs["grp"] = pd.Categorical(np.where(np.arange(a.n_obs) % 3 == 0, "a", "b"))
    path = tmp_path / "d.h5ad"
    a.write_h5ad(path)
    expected = marker_selection._fractions(ad.read_h5ad(path), "grp", ["a", "b"])
    monkeypatch.setattr(anndata_adaptor, "BACKED_MIN_BYTES", 0)
    backed = AnnDataAdaptor(str(path)).adata
    monkeypatch.setattr(marker_selection, "_CHUNK", 50)
    monkeypatch.setattr(activity.Lognorm, "whole", lambda self: pytest.fail("read whole"))
    got = marker_selection._fractions(backed, "grp", ["a", "b"])
    pd.testing.assert_frame_equal(got[0], expected[0])
    pd.testing.assert_frame_equal(got[1], expected[1])


def test_mixscape_on_a_backed_dataset_reads_blocks_and_classifies_the_same(tmp_path, monkeypatch):
    a, _ = planted()
    path = tmp_path / "d.h5ad"
    a.write_h5ad(path)
    params = mixscape.Params(split_by="rep")
    res_mem, cols_mem = mixscape.run(ad.read_h5ad(path), params)
    monkeypatch.setattr(anndata_adaptor, "BACKED_MIN_BYTES", 0)
    backed = AnnDataAdaptor(str(path)).adata
    monkeypatch.setattr(activity.Lognorm, "whole", lambda self: pytest.fail("read whole"))
    res, cols = mixscape.run(backed, params)
    assert set(cols) == set(cols_mem)
    for k, v in cols_mem.items():
        np.testing.assert_array_equal(np.asarray(cols[k]), np.asarray(v), err_msg=k)
    assert _no_timing(res.to_dict()) == _no_timing(res_mem.to_dict())


# -- each feature through its endpoint ---------------------------------------------------


@pytest.mark.parametrize("layout", ["x_log", "x_counts"])
async def test_cytokine_response(client, dataset_manager, signature, monkeypatch, layout):  # noqa: F811
    a = _cyto_cells()
    if layout == "x_counts":  # the shifted log of the counts, taken as read
        a.X = sp.csr_matrix(np.round(np.expm1(a.X.toarray())).astype(np.float32))
    did = _register(dataset_manager, a, "cyto")
    body = {"mode": "conditions", "groupby": "cell_type", "condition": "stim", "a": "STIM", "b": "CTRL",
            "n_permutations": 200}

    async def call():
        return await client.post(f"{API}/datasets/{did}/cytokines/activity", json=body)

    from scview.core import cytokine_activity as ca

    monkeypatch.setattr(ca, "recall", lambda *_a: None)  # computed again, not served from the store
    mem, backed = await _in_memory_then_backed(dataset_manager, did, monkeypatch, call)
    assert not backed["reused"] and _no_timing(backed) == _no_timing(mem)


@pytest.mark.parametrize("layout", ["x_log", "x_counts"])
async def test_tf_and_pathway_activity(client, dataset_manager, networks, monkeypatch, layout):  # noqa: F811
    a = _act_cells() if layout == "x_counts" else _lognorm(_act_cells())
    did = _register(dataset_manager, a, "act")

    async def summary():
        return await client.get(f"{API}/datasets/{did}/activity/collectri", params={"groupby": "stim"})

    mem, backed = await _in_memory_then_backed(dataset_manager, did, monkeypatch, summary)
    assert _no_timing(backed) == _no_timing(mem)
    cells = await client.get(f"{API}/datasets/{did}/activity/progeny/cells", params={"source": "JAK-STAT"})
    assert cells.status_code == 200, cells.text
    monkeypatch.setattr(anndata_adaptor, "BACKED_MIN_BYTES", 10**15)
    dataset_manager.evict(did)
    in_memory = await client.get(f"{API}/datasets/{did}/activity/progeny/cells", params={"source": "JAK-STAT"})
    assert cells.json() == in_memory.json()


async def test_a_backed_model_is_not_kept(client, dataset_manager, networks, monkeypatch):  # noqa: F811
    """It holds the matrix read from the file: kept, it would sit outside any room held."""
    did = _register(dataset_manager, _lognorm(_act_cells()), "act")
    monkeypatch.setattr(anndata_adaptor, "BACKED_MIN_BYTES", 0)
    built = []
    real = activity.CellModel.build
    monkeypatch.setattr(activity.CellModel, "build", lambda *a, **k: built.append(1) or real(*a, **k))
    for _ in range(2):
        r = await client.get(f"{API}/datasets/{did}/activity/collectri")
        assert r.status_code == 200, r.text
    assert len(built) == 2


async def test_upstream_tf_candidates(client, dataset_manager, test_settings, monkeypatch):
    _species_human(monkeypatch)
    tm.save_table(_dataset_table(), tm.table_path(Path(test_settings.DATA_DIR) / "tf_motifs", "hg38"))
    did = _register(dataset_manager, _tf_dataset(), "tf")

    async def call():
        return await client.post(f"{API}/datasets/{did}/tf-motifs/run", json={"column": "cluster"})

    mem, backed = await _in_memory_then_backed(dataset_manager, did, monkeypatch, call)
    assert _no_timing(backed) == _no_timing(mem) and backed["groups"]


async def test_communication_run_whole_and_on_one_condition(client, dataset_manager, resource_on_disk,  # noqa: F811
                                                            monkeypatch):
    did = _ccc_dataset(dataset_manager)
    for body in ({"groupby": "group", "n_perms": 30, "resource": "consensus"},
                 {"groupby": "group", "subset_column": "stim", "subset_value": "CTRL", "n_perms": 30,
                  "resource": "consensus"}):
        monkeypatch.setattr(anndata_adaptor, "BACKED_MIN_BYTES", 10**15)
        dataset_manager.evict(did)

        async def call(body=body):
            return await client.post(f"{API}/datasets/{did}/communication/run", json=body)

        mem, backed = await _in_memory_then_backed(dataset_manager, did, monkeypatch, call)
        assert dict(_events(backed))["result"]["groups"]
        assert _no_timing(dict(_events(backed))["result"]) == _no_timing(dict(_events(mem))["result"])


async def test_communication_compare(client, dataset_manager, toy_resource, monkeypatch):  # noqa: F811
    did = _cmp_dataset(dataset_manager)

    async def call():
        return await client.post(f"{API}/datasets/{did}/communication/compare",
                                 json={"groupby": "cell_type", "condition": "stim", "a": "CTRL", "b": "STIM",
                                       "n_perms": 30, "resource": "consensus"})

    mem, backed = await _in_memory_then_backed(dataset_manager, did, monkeypatch, call)
    assert [e for e, _ in _events(backed)][-1] == "complete", backed
    assert _no_timing(dict(_events(backed))["result"]) == _no_timing(dict(_events(mem))["result"])


async def test_genes_along_pseudotime(client, dataset_manager, monkeypatch):
    a = _trend_adata()  # dense X
    did = _register(dataset_manager, a, "ptg")

    async def call():
        return await client.post(f"{API}/datasets/{did}/trajectory/dpt_pseudotime/trend-genes",
                                 json={"n_top": 5, "groups_column": "stage", "groups": ["early", "far"]})

    mem, backed = await _in_memory_then_backed(dataset_manager, did, monkeypatch, call)
    assert backed == mem


async def test_spatially_variable_genes(client, dataset_manager, monkeypatch):
    did = _register(dataset_manager, synthetic.visium(), "sp")

    async def call():
        return await client.post(f"{API}/datasets/{did}/spatial/autocorr", json={"genes": "hvg", "library_id": "B"})

    mem, backed = await _in_memory_then_backed(dataset_manager, did, monkeypatch, call)
    assert _no_timing(backed) == _no_timing(mem) and backed["libraries"] == ["B"]


async def test_marker_panel(client, dataset_manager, monkeypatch):
    did = _register(dataset_manager, _panel_data(), "pan")

    async def call():
        return await client.post(f"{API}/datasets/{did}/markers/panel", json={"groupby": "group", "size": 3})

    mem, backed = await _in_memory_then_backed(dataset_manager, did, monkeypatch, call)
    assert _no_timing(_events(backed)) == _no_timing(_events(mem))


async def test_mixscape_stream(client, dataset_manager, monkeypatch):
    a, _ = planted()
    did = _register(dataset_manager, a, "mx")

    async def call():
        return await client.post(f"{API}/datasets/{did}/perturbation/mixscape/run-stream",
                                 json={"labels": "perturbation", "control": "NT", "split_by": "rep"})

    mem, backed = await _in_memory_then_backed(dataset_manager, did, monkeypatch, call)
    assert [e for e, _ in _events(backed)][-2:] == ["result", "complete"], backed
    assert _no_timing(dict(_events(backed))["result"]) == _no_timing(dict(_events(mem))["result"])


# -- room held while the matrix is read ------------------------------------------------


def _spy_reads(monkeypatch, dm) -> list[int]:
    """What other requests see held each time a backed matrix is read (whole or by rows)."""
    seen: list[int] = []
    whole, rows = activity.Lognorm.whole, activity.Lognorm.rows

    def spy_whole(self):
        seen.append(dm.outstanding(GB))
        return whole(self)

    def spy_rows(self, keep):
        seen.append(dm.outstanding(GB))
        return rows(self, keep)

    monkeypatch.setattr(activity.Lognorm, "whole", spy_whole)
    monkeypatch.setattr(activity.Lognorm, "rows", spy_rows)
    return seen


def _needs(monkeypatch, gb: float) -> None:
    monkeypatch.setattr(activity, "lognorm_bytes", lambda *_a: int(gb * GB))
    monkeypatch.setattr(communication_api, "lognorm_bytes", lambda *_a: int(gb * GB))


async def _setup(where: str, client, dm, settings, monkeypatch):
    """(dataset id, the request) for each feature that reads the matrix under a reservation."""
    if where == "cytokines":
        did = _register(dm, _cyto_cells(), "x")
        return did, lambda: client.post(f"{API}/datasets/{did}/cytokines/activity",
                                        json={"groupby": "stim", "n_permutations": 100})
    if where == "activity":
        did = _register(dm, _lognorm(_act_cells()), "x")
        return did, lambda: client.get(f"{API}/datasets/{did}/activity/collectri")
    if where == "tf-motifs":
        _species_human(monkeypatch)
        tm.save_table(_dataset_table(), tm.table_path(Path(settings.DATA_DIR) / "tf_motifs", "hg38"))
        did = _register(dm, _tf_dataset(), "x")
        return did, lambda: client.post(f"{API}/datasets/{did}/tf-motifs/run", json={"column": "cluster"})
    if where == "trend-genes":
        did = _register(dm, _trend_adata(), "x")
        return did, lambda: client.post(f"{API}/datasets/{did}/trajectory/dpt_pseudotime/trend-genes", json={})
    if where == "autocorr":
        did = _register(dm, synthetic.visium(), "x")
        return did, lambda: client.post(f"{API}/datasets/{did}/spatial/autocorr", json={})
    if where == "communication":
        did = _ccc_dataset(dm, "x")
        return did, lambda: client.post(f"{API}/datasets/{did}/communication/run",
                                        json={"groupby": "group", "n_perms": 0, "resource": "consensus"})
    raise KeyError(where)


READERS = ["cytokines", "activity", "tf-motifs", "trend-genes", "autocorr", "communication"]


@pytest.fixture()
def readers_ready(tmp_data_dir, signature, networks):  # noqa: F811
    """The cytokine signature and both activity networks on disk; the LR resource beside them."""
    from tests.test_ccc import _fixture

    _, _, _, res, _ = _fixture()
    table = pd.concat([res.assign(resource="consensus"), res.assign(resource="mouseconsensus")])
    table[["resource", "ligand", "receptor"]].to_csv(networks / "liana_lr_resource.tsv", sep="\t", index=False)
    (networks / "liana_lr_resource.json").write_text(json.dumps({"fetched_at": "2026-10-21"}))


@pytest.mark.parametrize("where", READERS)
async def test_the_room_is_held_while_the_matrix_is_read_and_let_go(client, dataset_manager, test_settings,
                                                                    readers_ready, monkeypatch, where):
    did, call = await _setup(where, client, dataset_manager, test_settings, monkeypatch)
    monkeypatch.setattr(anndata_adaptor, "BACKED_MIN_BYTES", 0)
    await dataset_manager.get_or_load_dataset(did)
    _budget(monkeypatch, limit=8.0, used=1.0)
    _needs(monkeypatch, 1.0)
    seen = _spy_reads(monkeypatch, dataset_manager)
    r = await call()
    assert r.status_code == 200, r.text
    if where == "communication":
        assert [e for e, _ in _events(r.text)][-1] == "complete", r.text
    assert seen and set(seen) == {GB}  # counted by every other request while it read
    for _ in range(50):  # a stream lets go once its worker thread has ended
        if not dataset_manager._reads:
            break
        await asyncio.sleep(0.05)
    assert dataset_manager._reads == {} and not dataset_manager._reading


@pytest.mark.parametrize("where", READERS)
async def test_a_read_that_cannot_fit_is_a_507_naming_the_dataset(client, dataset_manager, test_settings,
                                                                   readers_ready, monkeypatch, where):
    did, call = await _setup(where, client, dataset_manager, test_settings, monkeypatch)
    monkeypatch.setattr(anndata_adaptor, "BACKED_MIN_BYTES", 0)
    await dataset_manager.get_or_load_dataset(did)
    _budget(monkeypatch, limit=8.0, used=1.0)
    _needs(monkeypatch, 100.0)
    monkeypatch.setattr("scview.core.dataset_manager.REFUSAL_WAIT_SECONDS", 1)
    seen = _spy_reads(monkeypatch, dataset_manager)
    r = await call()
    assert r.status_code == 507, r.text
    detail = r.json()["detail"]
    assert f"on {dataset_manager.dataset_name(did)} needs" in detail and "100.0 GB" in detail
    assert seen == [] and dataset_manager._reads == {} and not dataset_manager._reading


def test_spatial_read_estimate_counts_the_section_only(tmp_path, monkeypatch):
    a = synthetic.visium()
    path = tmp_path / "d.h5ad"
    a.write_h5ad(path)
    monkeypatch.setattr(anndata_adaptor, "BACKED_MIN_BYTES", 0)
    adaptor = AnnDataAdaptor(str(path))
    whole = sa.read_bytes(adaptor, None, None)
    section = sa.read_bytes(adaptor, "B", None)
    assert whole > section > 0 and sa.read_bytes(adaptor, None, "counts") == 0
