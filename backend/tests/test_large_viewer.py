"""A large dataset in the viewer (core/large_mode.py, the adaptor's large paths): opened
backed at any size, genes from the gene store, counts from their sidecar, whole-matrix
reads refused, obs / uns edits written in place, lifecycle (delete, export, prune)."""

from __future__ import annotations

import threading
import time

import anndata as ad
import h5py
import numpy as np
import pytest
import scipy.sparse as sp

from scview.core import gene_store, large_layout, large_mode
from scview.core.anndata_adaptor import AnnDataAdaptor
from scview.core.dataset_manager import DatasetManager, load_peak_bytes
from tests.large_fixture import counts_adata, make_large, register, small_twin


@pytest.fixture(scope="module")
def counts():
    return counts_adata()


@pytest.fixture(scope="module")
def twin(counts, tmp_path_factory):
    """The same dataset in memory, opened by the adaptor the usual way."""
    path = tmp_path_factory.mktemp("twin") / "twin.h5ad"
    small_twin(counts).write_h5ad(path)
    return AnnDataAdaptor(str(path))


@pytest.fixture()
def large(counts, tmp_path):
    a = AnnDataAdaptor(str(make_large(tmp_path / "ds", counts)))
    yield a
    a.close()


def test_opens_backed_whatever_its_size(large):
    assert large.is_large and large.isbacked  # the file is a few hundred kB
    info = large.large_info()
    assert info["matrices_on_disk"] and info["gene_store"] == "ready" and info["counts"]
    assert large.available_expression_layers() == [
        {"key": "X", "label": "Main matrix (X)", "n_genes": 300}]
    assert large.default_expression_layer() == "X"


def test_a_small_dataset_is_not_large(twin):
    assert not twin.is_large and twin.large_info() is None


def test_expression_equals_the_in_memory_dataset(large, twin):
    genes = ["G10", "G7", "MT-1", "absent", "G10"]
    assert np.array_equal(large.get_expression(genes), twin.get_expression(genes, layer="X"))
    assert large.get_expression_for_violin("G25", "leiden") == \
        twin.get_expression_for_violin("G25", "leiden", layer="X")


def test_by_group_equals_the_in_memory_dataset(large, twin):
    genes = ["G10", "G30", "G50", "MT-2", "G299"]
    for col in ("leiden", "donor"):
        got = large.expression_by_group(genes, col)
        want = twin.expression_by_group(genes, col, layer="X")
        assert got.keys() == want.keys()
        for key in ("groups", "genes", "scale_range"):
            assert got[key] == want[key]
        for key in ("mean", "mean_expressing", "fraction"):
            np.testing.assert_allclose(got[key], want[key], atol=1.5e-4)


def test_genes_never_come_from_the_cell_major_matrix(large, monkeypatch):
    def boom(*_a, **_k):
        raise AssertionError("read X")

    # opened and checked (64 probe rows), the colour range sampled (5,000 rows): then no
    # more reads of X at all
    assert large.gene_store() is not None and large.layer_scale_range("X")
    monkeypatch.setattr(large_mode.OnDiskMatrix, "__getitem__", boom)
    assert large.get_expression(["G10"]).shape == (600, 1)
    assert large.expression_by_group(["G10"], "leiden")["genes"] == ["G10"]


def test_whole_matrix_reads_are_refused(large, twin, monkeypatch):
    X = large.adata.X
    monkeypatch.setattr(large_mode, "MAX_COLUMNS", 50)
    for read in (lambda: X.to_memory(), lambda: large.adata.to_memory(), lambda: large.to_memory(),
                 lambda: X[:, :51], lambda: large.counts_matrix()[:, [1]]):
        with pytest.raises(large_mode.LargeDatasetError, match="not available for large datasets yet"):
            read()
    # rows are fine, a view of cells too
    assert np.array_equal(X[:10].toarray(), twin.adata.X[:10].toarray())
    rows = np.array([3, 50, 599])
    assert np.array_equal(X[rows].toarray(), twin.adata.X[rows].toarray())
    mask = (large.adata.obs["leiden"] == "1").to_numpy()
    assert large.adata[mask].X.shape == (int(mask.sum()), 300)


def test_a_few_genes_of_x_come_from_the_store(large, twin, monkeypatch):
    X, T = large.adata.X, twin.adata.X
    store = large.gene_store()
    calls = []
    real = type(store).genes
    monkeypatch.setattr(type(store), "genes", lambda self, *a, **k: calls.append(a) or real(self, *a, **k))
    for got, want in [(X[:, [7, 2]], T[:, [7, 2]]), (X[:, 3], T[:, 3]),
                      (large.adata[:, ["G5", "G9"]].X, twin.adata[:, ["G5", "G9"]].X)]:
        assert np.array_equal(got.toarray(), want.toarray())
    assert len(calls) == 3
    # a sample of cells is read by rows, its genes cut in memory (an assessment's sample)
    rows = np.array([5, 7, 300])
    assert np.array_equal(X[rows, 10:14].toarray(), T[rows, 10:14].toarray())
    assert np.array_equal(X[:200, :250].toarray(), T[:200, :250].toarray())
    assert len(calls) == 3


def test_too_many_rows_at_once_are_refused(large, monkeypatch):
    monkeypatch.setattr(large_mode, "MAX_READ_BYTES", 8 * 1000)
    X = large.adata.X
    assert X[:2].shape == (2, 300)
    with pytest.raises(large_mode.LargeDatasetError, match="at most"):
        X[:]
    # a chunked pass still works: blocks of a bounded number of values
    blocks = list(X.row_blocks())
    assert len(blocks) > 1 and sum(b.shape[0] for _, b in blocks) == 600


def test_counts_come_from_the_sidecar(large, counts):
    m = large.counts_matrix()
    assert np.array_equal(m[5:9].toarray(), counts.X[5:9].toarray())
    with pytest.raises(large_mode.LargeDatasetError):
        m[:, [0]]


def test_counts_of_other_cells_are_refused(counts, tmp_path):
    h5ad = make_large(tmp_path / "ds", counts)
    counts_path, _ = large_layout.sidecar_paths(h5ad)
    large_layout.stamp_counts(counts_path, [f"x{i}" for i in range(600)], counts.var_names)
    a = AnnDataAdaptor(str(h5ad))
    with pytest.raises(large_mode.LargeDatasetError, match="holds other cells"):
        a.counts_matrix("Pseudobulk")
    assert "other cells" in a.large_info()["counts_note"]


def test_qc_comes_from_obs_or_the_counts_sidecar(counts, tmp_path):
    a = AnnDataAdaptor(str(make_large(tmp_path / "with", counts)))
    cells = a.qc_per_cell(ribo=True)
    assert not cells["computed"]
    np.testing.assert_allclose(cells["total_counts"], counts.obs["total_counts"])
    # no QC columns in obs: computed from the counts, not from the log values in X
    bare = counts.copy()
    bare.obs = bare.obs[["leiden", "donor", "condition"]]
    b = AnnDataAdaptor(str(make_large(tmp_path / "without", bare)))
    cells = b.qc_per_cell()
    assert cells["computed"]
    np.testing.assert_allclose(cells["total_counts"], counts.obs["total_counts"], rtol=1e-6)
    np.testing.assert_allclose(cells["pct_mt"], counts.obs["pct_counts_mt"], rtol=1e-5)


def test_a_missing_store_falls_back_then_is_rebuilt(counts, twin, tmp_path):
    h5ad = make_large(tmp_path / "ds", counts, gene_store_too=False)
    a = AnnDataAdaptor(str(h5ad))
    info = a.large_info()
    assert info["gene_store"] == "missing" and "missing" in info["gene_store_note"]
    # the views still answer, from one pass over the on-disk rows
    assert np.array_equal(a.get_expression(["G3", "G40"]), twin.get_expression(["G3", "G40"], layer="X"))
    assert a.store_rebuild_needed() and a.start_store_rebuild()
    a._large._rebuild.join(30)
    assert a.large_info()["gene_store"] == "ready"
    store = a.gene_store()
    assert store is not None and store.mismatch(twin.adata) is None
    assert np.array_equal(a.get_expression(["G3"]), twin.get_expression(["G3"], layer="X"))


def test_closing_stops_a_rebuild(counts, tmp_path, monkeypatch):
    h5ad = make_large(tmp_path / "ds", counts, gene_store_too=False)
    _, genes_path = large_layout.sidecar_paths(h5ad)
    a = AnnDataAdaptor(str(h5ad))
    # many small blocks: the stop lands between two of them
    monkeypatch.setattr(large_mode.OnDiskMatrix, "row_blocks",
                        lambda self, max_values=None: ((i, self[i:i + 10]) for i in range(0, 600, 10)))
    assert a.start_store_rebuild()
    a.close()
    a._large._rebuild.join(30)
    assert not genes_path.exists()
    assert not list(genes_path.parent.glob(".*build-*"))


def test_a_stale_store_is_not_used(counts, twin, tmp_path):
    h5ad = make_large(tmp_path / "ds", counts)
    _, genes_path = large_layout.sidecar_paths(h5ad)
    gene_store.build(sp.csr_matrix(counts.X), genes_path, obs_names=counts.obs_names,
                     var_names=counts.var_names, source="X")  # counts, not X's log values
    a = AnnDataAdaptor(str(h5ad))
    assert a.gene_store() is None
    assert "does not match" in a.large_info()["gene_store_note"]
    assert np.array_equal(a.get_expression(["G9"]), twin.get_expression(["G9"], layer="X"))


def test_close_keeps_the_tables_and_reads_again(large, twin):
    obs = large.adata.obs
    large.close()
    assert large.adata.obs is obs  # never re-read: a holder keeps what it had
    assert np.array_equal(large.get_expression(["G12"]), twin.get_expression(["G12"], layer="X"))
    assert large.counts_matrix()[:2].shape == (2, 300)


def test_edits_are_written_in_place(large, twin):
    path = large.h5ad_path
    with h5py.File(path, "r") as f:
        x_before = f["X"]["data"].id.get_offset()
    adata = large.adata
    adata.obs["mine"] = (adata.obs["leiden"].astype(str) + "x").astype("category")
    adata.uns["scview_note"] = {"a": 1}
    large.write_in_place(uns_keys=["scview_note"])
    with h5py.File(path, "r") as f:
        assert f["X"]["data"].id.get_offset() == x_before  # X neither rewritten nor moved
    back = ad.read_h5ad(path, backed="r")
    assert list(back.obs["mine"]) == list(adata.obs["mine"])
    assert back.uns["scview_note"]["a"] == 1 and large_layout.is_large(back)
    back.file.close()
    # the open dataset reads on
    assert np.array_equal(large.adata.X[:3].toarray(), twin.adata.X[:3].toarray())
    del adata.uns["scview_note"]
    large.write_in_place(obs=False, uns_removed=["scview_note"])
    back = ad.read_h5ad(path, backed="r")
    assert "scview_note" not in back.uns
    back.file.close()


def test_a_read_waits_for_an_in_place_write(large, monkeypatch):
    started, release = threading.Event(), threading.Event()
    real = large_mode._replace

    def slow(group, key, value):
        started.set()
        release.wait(5)
        return real(group, key, value)

    monkeypatch.setattr(large_mode, "_replace", slow)
    writer = threading.Thread(target=large.write_in_place)
    writer.start()
    assert started.wait(5)
    got = {}
    reader = threading.Thread(target=lambda: got.setdefault("rows", large.adata.X[:4].shape))
    reader.start()
    time.sleep(0.2)
    assert reader.is_alive()  # waiting for the file, not failing on a closed handle
    release.set()
    writer.join(5)
    reader.join(5)
    assert got["rows"] == (4, 300)


def test_load_estimate_counts_the_tables_not_the_matrix(counts, tmp_path):
    from scview.core import preflight

    h5ad = make_large(tmp_path / "ds", counts)
    h = preflight.read_header(h5ad)
    assert h.large and h.backed  # read_header knows the layout opens from disk
    h.backed = False  # what reading it whole would cost, for comparison
    in_memory = preflight.estimate_load(h, ingest=False).peak_bytes
    peak = load_peak_bytes(str(h5ad))
    x = next(m for m in h.matrices if m.where == "X")
    assert peak < in_memory + large_mode.held_bytes(600)
    assert in_memory - (peak - large_mode.held_bytes(600)) >= x.nbytes


async def test_delete_removes_the_sidecars(tmp_data_dir):
    dm = DatasetManager(str(tmp_data_dir))
    h5ad = register(tmp_data_dir, "big1")
    counts, genes = large_layout.sidecar_paths(h5ad)
    adaptor = await dm.load_dataset("big1")
    assert adaptor.is_large and adaptor.get_expression(["G5"]).shape == (600, 1)
    assert dm.get_dataset_info("big1")["large"]["gene_store"] == "ready"
    assert dm.remove_dataset("big1")
    assert not counts.exists() and not genes.exists() and not h5ad.exists()


async def test_prune_removes_stray_sidecars_only(tmp_data_dir):
    dm = DatasetManager(str(tmp_data_dir))
    h5ad = register(tmp_data_dir, "big2")
    folder = h5ad.parent
    stray = folder / "old.genes.h5"
    stray.write_bytes(b"x")
    part = folder / ".data.counts.h5.part"
    part.write_bytes(b"x")
    build = folder / ".data.genes.h5.build-999999-abcd"
    build.mkdir()
    dm.prune_orphans()
    assert not stray.exists() and not part.exists() and not build.exists()
    counts, genes = large_layout.sidecar_paths(h5ad)
    assert counts.exists() and genes.exists() and h5ad.exists()


async def test_persist_uns_writes_a_large_dataset_in_place(tmp_data_dir):
    dm = DatasetManager(str(tmp_data_dir))
    h5ad = register(tmp_data_dir, "big3")
    adaptor = await dm.load_dataset("big3")
    adaptor.adata.uns["scview_x"] = np.arange(3)
    assert dm.persist_uns("big3", adaptor, ["scview_x"])
    assert adaptor.h5ad_path == str(h5ad)
    with h5py.File(h5ad, "r") as f:
        assert list(f["uns"]["scview_x"][()]) == [0, 1, 2]


async def test_an_uploaded_large_layout_is_copied_before_an_edit(tmp_data_dir):
    from scview.core.obs_writes import save_edit, uns_snapshot

    dm = DatasetManager(str(tmp_data_dir))
    h5ad = register(tmp_data_dir, "big4", where="uploads")
    adaptor = await dm.load_dataset("big4")
    assert adaptor.h5ad_path == str(h5ad)
    before = uns_snapshot(adaptor.adata)
    adaptor.adata.obs["kept"] = "yes"
    target = dm.derived_h5ad_path("big4", adaptor.h5ad_path)
    save_edit(adaptor, adaptor.adata, str(target), before)
    assert adaptor.h5ad_path == str(target)
    assert "kept" not in ad.read_h5ad(h5ad, backed="r").obs  # the upload is untouched
    copy = AnnDataAdaptor(str(target))
    assert copy.is_large and "kept" in copy.adata.obs
    assert copy.large_info()["gene_store"] == "ready" and copy.large_info()["counts"]
    assert dm._resolve_h5ad("big4") == str(target)


async def test_export_is_one_self_contained_h5ad(client, tmp_data_dir, counts, twin):
    h5ad = register(tmp_data_dir, "big5", counts)
    info = (await client.get("/api/v1/datasets/big5/export/h5ad/info")).json()
    assert info["large"] and info["with_counts"]
    res = await client.get("/api/v1/datasets/big5/export/h5ad", params={"raw": True})
    assert res.status_code == 200
    out = tmp_data_dir / "dl.h5ad"
    out.write_bytes(res.content)
    got = ad.read_h5ad(out)
    assert not large_layout.is_large(got)
    assert np.array_equal(got.X.toarray(), twin.adata.X.toarray())
    assert np.array_equal(got.layers["counts"].toarray(), counts.X.toarray())
    assert got.raw is not None and np.array_equal(got.raw.X.toarray(), twin.adata.X.toarray())
    assert list(got.obs["leiden"]) == list(counts.obs["leiden"])
    assert "X_umap" in got.obsm
    # the source files are as they were
    assert large_mode.file_is_large(h5ad)
    assert abs(info["size_bytes"] - (tmp_data_dir / "cache" / "big5" / "export_raw.h5ad").stat().st_size) \
        < info["size_bytes"]


async def test_dataset_info_says_large(client, tmp_data_dir):
    register(tmp_data_dir, "big6")
    body = (await client.get("/api/v1/datasets/big6")).json()
    assert body["large"]["matrices_on_disk"] and body["large"]["gene_store"] == "ready"
    assert body["default_expression_layer"] == "X"


async def test_a_refusal_is_a_422_with_the_reason(client, tmp_data_dir):
    from scview.main import app

    @app.get("/test-large-refusal")
    async def _refuse():
        large_mode.refuse("Gene programs", "it factorises every cell's values")

    res = await client.get("/test-large-refusal")
    assert res.status_code == 422
    assert res.json() == {
        "detail": "Gene programs is not available for large datasets yet: it factorises every "
                  "cell's values.", "large_dataset": True}


async def test_a_file_too_big_to_read_in_opens_backed(tmp_data_dir, counts, monkeypatch):
    """A file under the size threshold whose in-memory peak does not fit the memory left
    with the working reserve (a streamed ingest of a million cells) opens backed."""
    import json

    from scview.core import memory_budget, preflight

    folder = tmp_data_dir / "ingested" / "ing1"
    folder.mkdir(parents=True)
    small_twin(counts).write_h5ad(folder / "data.h5ad")
    up = tmp_data_dir / "uploads" / "ing1"
    up.mkdir(parents=True)
    (up / "metadata.json").write_text(json.dumps({"id": "ing1", "name": "ing", "status": "ready"}))
    gb = 1024**3
    real = preflight.estimate_load

    def estimate(h, ingest=True, cpus=None):
        est = real(h, ingest=ingest, cpus=cpus)
        if not h.backed:  # as if its matrices took 4 GB in memory
            est.peak_bytes += 4 * gb
        return est

    def budget(limit_gb):
        return lambda *a, **k: memory_budget.MemoryBudget(limit_gb * gb, gb, True, "container limit")

    monkeypatch.setattr(preflight, "estimate_load", estimate)
    monkeypatch.setattr(memory_budget, "release_memory", lambda: None)
    monkeypatch.setattr(memory_budget, "read_budget", budget(4))
    adaptor = await DatasetManager(str(tmp_data_dir)).load_dataset("ing1")
    assert adaptor.isbacked and not adaptor.is_large
    # with room to spare it opens in memory, as before
    monkeypatch.setattr(memory_budget, "read_budget", budget(64))
    assert not (await DatasetManager(str(tmp_data_dir)).load_dataset("ing1")).isbacked
