"""Gene-major store (core/gene_store.py): round trips, index widths, staleness, atomic builds.

Every expected value is the source matrix itself (scipy's own CSC of it), never the
store's output. Small ``stage_entries`` / ``block_entries`` force several spilled row
chunks and several gene blocks with a ragged last one, so the multi-chunk placement
is what is tested, not a single-block shortcut.
"""

from __future__ import annotations

import os

import anndata as ad
import h5py
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

from scview.core import gene_store as gs


def _matrix(n_cells=300, n_genes=47, density=0.15, seed=0) -> sp.csr_matrix:
    """Log-count-like CSR with ties, a never-expressed gene, explicit zeros and
    unsorted column indices in some rows."""
    rng = np.random.default_rng(seed)
    m = sp.random(n_cells, n_genes, density=density, format="csr", random_state=rng,
                  data_rvs=lambda k: np.log1p(rng.integers(1, 6, size=k)).astype(np.float32))
    m = m.astype(np.float32).tolil()
    m[:, 5] = 0  # never expressed
    m = sp.csr_matrix(m)
    m.data[::17] = 0  # explicit zeros the store must drop
    # reverse the column order inside a few rows (still a valid CSR)
    for r in (0, 3, 7):
        a, b = m.indptr[r], m.indptr[r + 1]
        m.indices[a:b] = m.indices[a:b][::-1].copy()
        m.data[a:b] = m.data[a:b][::-1].copy()
    m.has_sorted_indices = False
    return m


def _names(n, prefix):
    return [f"{prefix}{i}" for i in range(n)]


def _reference(m) -> sp.csc_matrix:
    c = sp.csc_matrix(m, dtype=np.float32, copy=True)
    c.sum_duplicates()
    c.eliminate_zeros()
    return c


def _build(m, path, **kw):
    kw.setdefault("stage_entries", 500)
    kw.setdefault("block_entries", 300)
    return gs.build(m, path, obs_names=_names(m.shape[0], "c"), var_names=_names(m.shape[1], "g"),
                    **kw)


def _assert_store_equals(store: gs.GeneStore, ref: sp.csc_matrix) -> None:
    rng = np.random.default_rng(1)
    assert store.shape == ref.shape and store.nnz == ref.nnz
    for j in range(ref.shape[1]):
        np.testing.assert_array_equal(store.gene(j), ref[:, j].toarray().ravel())
        rows, vals = store.column(j)
        np.testing.assert_array_equal(rows, ref.indices[ref.indptr[j]:ref.indptr[j + 1]])
        np.testing.assert_array_equal(vals, ref.data[ref.indptr[j]:ref.indptr[j + 1]])
    genes = rng.choice(ref.shape[1], 20, replace=False)
    block = store.genes(genes)
    assert block.format == "csc" and block.shape == (ref.shape[0], 20)
    assert (block != ref[:, genes]).nnz == 0
    cells = rng.choice(ref.shape[0], 60, replace=False)  # unsorted subset
    assert (store.genes(genes, cells=cells) != ref[cells][:, genes]).nnz == 0
    np.testing.assert_array_equal(store.gene(int(genes[0]), cells=cells),
                                  ref[cells][:, genes[0]].toarray().ravel())
    mask = np.zeros(ref.shape[0], dtype=bool)
    mask[cells] = True
    np.testing.assert_array_equal(store.gene(int(genes[0]), cells=mask),
                                  ref[np.flatnonzero(mask), genes[0]].toarray().ravel())
    seen = []
    for g0, g1, blk in store.blocks(max_entries=250):
        assert (blk != ref[:, g0:g1]).nnz == 0
        seen.append((g0, g1))
    assert seen[0][0] == 0 and seen[-1][1] == ref.shape[1] and len(seen) > 2
    assert all(a[1] == b[0] for a, b in zip(seen, seen[1:]))


def test_round_trip_in_memory_csr(tmp_path):
    m = _matrix()
    before = (m.data.copy(), m.indices.copy(), m.indptr.copy())
    path = _build(m, tmp_path / "x.genes.h5")
    # the source keeps its explicit zeros and unsorted rows: the build works on copies
    for a, b in zip(before, (m.data, m.indices, m.indptr)):
        np.testing.assert_array_equal(a, b)
    with gs.GeneStore(path) as store:
        assert store.dtype == np.float32
        assert store.gene("g5").sum() == 0  # the never-expressed gene
        _assert_store_equals(store, _reference(m))
        np.testing.assert_array_equal(store.gene("g12"), store.gene(12))
        assert list(store.var_names) == _names(m.shape[1], "g")


def test_round_trip_dense_backed_and_h5_group_sources(tmp_path):
    m = _matrix(seed=3)
    ref = _reference(m)
    dense = m.toarray()
    with gs.GeneStore(_build(dense, tmp_path / "dense.h5")) as store:
        _assert_store_equals(store, ref)
    adata = ad.AnnData(sp.csr_matrix(m), obs=pd.DataFrame(index=_names(m.shape[0], "c")),
                       var=pd.DataFrame(index=_names(m.shape[1], "g")))
    adata.write_h5ad(tmp_path / "src.h5ad", compression="gzip")
    backed = ad.read_h5ad(tmp_path / "src.h5ad", backed="r")
    path = gs.build_from_adata(backed, tmp_path / "backed.h5", stage_entries=400, block_entries=300)
    with gs.GeneStore(path) as store:
        _assert_store_equals(store, ref)
        assert store.matches(backed)
    backed.file.close()
    with h5py.File(tmp_path / "src.h5ad", "r") as f:
        path = gs.build(f["X"], tmp_path / "group.h5", obs_names=_names(m.shape[0], "c"),
                        var_names=_names(m.shape[1], "g"), stage_entries=500, block_entries=300)
        with gs.GeneStore(path) as store:
            _assert_store_equals(store, ref)
    # a dense matrix on disk (h5py dataset), read by row slices
    with h5py.File(tmp_path / "dense_src.h5", "w") as f:
        f.create_dataset("X", data=dense, chunks=(16, 16))
    with h5py.File(tmp_path / "dense_src.h5", "r") as f:
        with gs.GeneStore(_build(f["X"], tmp_path / "dense_ds.h5")) as store:
            _assert_store_equals(store, ref)


def test_csc_source_and_float64_dtype(tmp_path):
    m = _matrix(seed=4).astype(np.float64)
    m.data += 1e-9  # not representable in float32
    with gs.GeneStore(_build(sp.csc_matrix(m), tmp_path / "csc.h5")) as store:
        _assert_store_equals(store, _reference(m))  # values cast to float32
    with gs.GeneStore(_build(m, tmp_path / "f64.h5", dtype=np.float64)) as store:
        assert store.dtype == np.float64
        ref = sp.csc_matrix(m, copy=True)
        ref.sum_duplicates()
        ref.eliminate_zeros()
        j = int(np.argmax(np.diff(ref.indptr)))
        np.testing.assert_array_equal(store.gene(j), ref[:, j].toarray().ravel())


def test_dask_source(tmp_path):
    da = pytest.importorskip("dask.array")
    m = _matrix(seed=5)
    x = da.from_array(m.toarray(), chunks=(70, m.shape[1]))
    with gs.GeneStore(_build(x, tmp_path / "dask.h5")) as store:
        _assert_store_equals(store, _reference(m))


def test_int64_indices_beyond_the_int32_limit(tmp_path, monkeypatch):
    # Not a 2^31-value matrix: the limit itself is lowered, so the int64 branches
    # (cell indices in the spill and the store, the store's indptr) are what runs.
    monkeypatch.setattr(gs, "_INDEX_LIMIT", 100)
    m = _matrix(n_cells=260, seed=6)
    path = _build(m, tmp_path / "wide.h5")
    with h5py.File(path, "r") as f:
        assert f["indptr"].dtype == np.int64
        assert f["indices"].dtype == np.int64
    with gs.GeneStore(path) as store:
        assert store.indptr[-1] == _reference(m).nnz
        _assert_store_equals(store, _reference(m))
    monkeypatch.setattr(gs, "_INDEX_LIMIT", int(np.iinfo(np.int32).max))
    with h5py.File(_build(m, tmp_path / "narrow.h5"), "r") as f:
        assert f["indptr"].dtype == np.int32 and f["indices"].dtype == np.int32


def test_empty_rows_genes_and_matrix(tmp_path):
    m = sp.csr_matrix((40, 9), dtype=np.float32)
    with gs.GeneStore(_build(m, tmp_path / "empty.h5")) as store:
        assert store.nnz == 0
        assert not store.gene(3).any()
        assert store.genes([1, 2]).nnz == 0
    m = _matrix(seed=7).tolil()
    m[10:40] = 0  # a run of empty cells across chunk boundaries
    m = m.tocsr()
    with gs.GeneStore(_build(m, tmp_path / "rows.h5")) as store:
        _assert_store_equals(store, _reference(m))


def test_unknown_genes_and_bad_masks(tmp_path):
    m = _matrix()
    with gs.GeneStore(_build(m, tmp_path / "x.h5")) as store:
        with pytest.raises(KeyError, match="nope"):
            store.gene("nope")
        with pytest.raises(KeyError):
            store.gene(m.shape[1])
        with pytest.raises(ValueError):
            store.gene(0, cells=np.ones(3, dtype=bool))


def _adata(m):
    return ad.AnnData(sp.csr_matrix(m), obs=pd.DataFrame(index=_names(m.shape[0], "c")),
                      var=pd.DataFrame(index=_names(m.shape[1], "g")))


def test_stale_store_detection(tmp_path):
    m = _matrix()
    adata = _adata(m)
    with gs.GeneStore(_build(m, tmp_path / "x.h5")) as store:
        assert store.mismatch(adata) is None and store.matches(adata, full=True)
        assert store.matches((adata.X, adata.obs_names, adata.var_names))
        # filtered cells / genes
        assert "cells" in store.mismatch(adata[1:].copy())
        assert "genes" in store.mismatch(adata[:, 1:].copy())
        # renamed cells, reordered genes
        other = adata.copy()
        other.obs_names = [f"x{i}" for i in range(adata.n_obs)]
        assert store.mismatch(other) == "the cells differ"
        other = adata[:, list(reversed(range(adata.n_vars)))].copy()
        assert store.mismatch(other) == "the genes differ"
        # a value in a probe row (row 0): the cheap check sees it
        other = adata.copy()
        other.X = other.X.copy()
        other.X.data[other.X.indptr[0]] += 1
        assert "probe" in store.mismatch(other)
        # a value in a row the probe skips: only the full check sees it
        rows = set(gs.probe_rows(adata.n_obs).tolist())
        r = next(i for i in range(adata.n_obs) if i not in rows and adata.X[i].nnz)
        other = adata.copy()
        other.X = other.X.copy()
        other.X.data[other.X.indptr[r]] += 1
        assert store.mismatch(other) is None
        assert store.mismatch(other, full=True) == "the values differ"
        # one value more (same rows otherwise)
        other = adata.copy()
        x = other.X.tolil()
        free = next(j for j in range(adata.n_vars) if j not in set(x.rows[r]))
        x[r, free] = 1.0
        other.X = x.tocsr()
        assert "stored values" in store.mismatch(other)
        # renormalised values (every value differs)
        other = adata.copy()
        other.X = other.X * 2
        assert store.mismatch(other) is not None


def test_mismatch_by_source(tmp_path):
    m = _matrix()
    adata = _adata(m)
    adata.raw = adata.copy()
    adata.X = adata.X * 3
    adata.layers["counts"] = sp.csr_matrix(m)
    path = gs.build_from_adata(adata, tmp_path / "raw.h5", source="raw")
    with gs.GeneStore(path) as store:
        assert store.source == "raw"
        assert store.mismatch(adata) is None  # checks .raw, the recorded source
        assert store.mismatch(adata, source="X") is not None
        assert store.mismatch(adata, source="layers/counts") is None
        assert store.mismatch(adata.raw) is None


class _Stop(Exception):
    pass


def _stop_on(phase):
    def progress(message, done, total):
        if phase in message:
            raise _Stop(message)
    return progress


@pytest.mark.parametrize("phase", ["reading cells", "writing genes"])
def test_interrupted_build_leaves_no_store(tmp_path, phase):
    m = _matrix()
    path = tmp_path / "x.genes.h5"
    with pytest.raises(_Stop):
        _build(m, path, progress=_stop_on(phase))
    assert not path.exists()
    assert os.listdir(tmp_path) == []  # no partial file, no spill left


def test_interrupted_rebuild_keeps_the_old_store(tmp_path):
    m = _matrix()
    path = _build(m, tmp_path / "x.genes.h5")
    with gs.GeneStore(path) as store:
        before = store.fingerprint
    changed = m * 2
    with pytest.raises(_Stop):
        _build(changed, path, progress=_stop_on("writing genes"))
    with gs.GeneStore(path) as store:
        assert store.fingerprint == before
        _assert_store_equals(store, _reference(m))
    assert sorted(os.listdir(tmp_path)) == ["x.genes.h5"]
    _build(changed, path)  # a finished rebuild replaces it
    with gs.GeneStore(path) as store:
        assert store.fingerprint["values_digest"] != before["values_digest"]
        _assert_store_equals(store, _reference(changed))


def test_writer_aborts_on_error_and_rejects_wrong_rows(tmp_path):
    m = _matrix()
    path = tmp_path / "w.h5"
    with pytest.raises(ValueError, match="rows"):
        with gs.GeneStoreWriter(path, obs_names=_names(m.shape[0], "c"),
                                var_names=_names(m.shape[1], "g")) as w:
            w.add(m[:100])
            w.finish()  # 100 of 300 rows
    assert not path.exists() and os.listdir(tmp_path) == []
    with pytest.raises(ValueError, match="columns"):
        with gs.GeneStoreWriter(path, obs_names=_names(m.shape[0], "c"),
                                var_names=_names(m.shape[1] + 1, "g")) as w:
            w.add(m)
    assert os.listdir(tmp_path) == []
    # rows pushed in uneven pieces build the same store as one build() call
    with gs.GeneStoreWriter(path, obs_names=_names(m.shape[0], "c"),
                            var_names=_names(m.shape[1], "g"), stage_entries=700,
                            block_entries=200, source_nnz=m.nnz) as w:
        for a, b in ((0, 1), (1, 50), (50, 51), (51, 300)):
            w.add(m[a:b])
        w.finish()
    with gs.GeneStore(path) as store, gs.GeneStore(_build(m, tmp_path / "b.h5")) as other:
        assert store.fingerprint == other.fingerprint
        _assert_store_equals(store, _reference(m))


def test_stale_build_dirs_of_dead_processes_are_removed(tmp_path):
    path = tmp_path / "x.genes.h5"
    dead = tmp_path / f"{gs.build_dir_prefix(path)}999999999-abcd"
    dead.mkdir()
    (dead / "rows.bin").write_bytes(b"\0" * 10)
    alive = tmp_path / f"{gs.build_dir_prefix(path)}{os.getpid()}-abcd"  # a live build
    alive.mkdir()
    _build(_matrix(), path)
    assert not dead.exists() and alive.exists()
    gs.remove(path)
    assert not path.exists() and not alive.exists()


def test_not_a_store(tmp_path):
    with pytest.raises(gs.GeneStoreError):
        gs.GeneStore(tmp_path / "missing.h5")
    with h5py.File(tmp_path / "other.h5", "w") as f:
        f.create_dataset("x", data=[1])
    with pytest.raises(gs.GeneStoreError, match="not a gene store"):
        gs.GeneStore(tmp_path / "other.h5")
    path = _build(_matrix(), tmp_path / "x.h5")
    with h5py.File(path, "a") as f:
        del f.attrs["values_digest"]
    with pytest.raises(gs.GeneStoreError, match="incomplete"):
        gs.GeneStore(path)
