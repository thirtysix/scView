"""Column-stored (CSC) matrices stored by rows (CSR) when a file comes in (``core/csr_store.py``).

Every expected matrix is scipy's own ``csc_matrix(...).tocsr()`` of the source arrays, or
the source read back with anndata: never this module's output compared with itself.
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
from anndata.io import read_elem

from scview.core import csr_store, preflight
from scview.core.dataset_manager import DatasetManager


def _random_csc(n_obs: int, n_var: int, density: float, dtype, seed: int = 0) -> sp.csc_matrix:
    rng = np.random.default_rng(seed)
    m = sp.random(n_obs, n_var, density=density, format="csc", random_state=rng, dtype=np.float64)
    m.data = np.round(m.data * 50) + 1
    return m.astype(dtype)


def _write_csc_arrays(path: Path, data, indices, indptr, shape, name: str = "X") -> None:
    """The arrays as given (unsorted rows, stored zeros and repeats included)."""
    with h5py.File(path, "a") as f:
        g = f.create_group(name)
        g.attrs["encoding-type"] = "csc_matrix"
        g.attrs["encoding-version"] = "0.1.0"
        g.attrs["shape"] = np.asarray(shape, dtype=np.int64)
        g.create_dataset("data", data=data)
        g.create_dataset("indices", data=indices)
        g.create_dataset("indptr", data=indptr)


def _convert(tmp_path: Path, m: sp.csc_matrix, **kw) -> tuple[sp.csr_matrix, list, dict]:
    src, dst = tmp_path / "in.h5", tmp_path / "out.h5"
    for p in (src, dst):
        p.unlink(missing_ok=True)
    _write_csc_arrays(src, m.data, m.indices, m.indptr, m.shape)
    with h5py.File(src, "r") as fi, h5py.File(dst, "w") as fo:
        side = csr_store.convert(fi["X"], fo, "X", **kw)
    with h5py.File(dst, "r") as fo:
        attrs = {k: (v.tolist() if hasattr(v, "tolist") else v) for k, v in fo["X"].attrs.items()}
        attrs["dtypes"] = {k: fo["X"][k].dtype for k in ("data", "indices", "indptr")}
        got = read_elem(fo["X"])
    assert not list(tmp_path.glob(".csr-*"))  # the temporary files are gone
    return got, side, attrs


def _same(got, want: sp.csr_matrix) -> None:
    assert sp.issparse(got) and got.format == "csr"
    assert got.shape == want.shape and got.dtype == want.dtype
    np.testing.assert_array_equal(got.indptr, want.indptr)
    np.testing.assert_array_equal(got.indices, want.indices)
    np.testing.assert_array_equal(got.data, want.data)


SHAPES = {
    "plain": (60, 40, 0.2),
    "one_row": (1, 30, 0.5),
    "one_column": (50, 1, 0.5),
    "no_values": (20, 15, 0.0),
    "sparse_with_empty_rows_and_columns": (80, 70, 0.01),
}


@pytest.mark.parametrize("dtype", [np.int32, np.float32, np.float64])
@pytest.mark.parametrize("shape", list(SHAPES))
@pytest.mark.parametrize("ranges", ["one", "many"])
def test_convert_equals_scipy(tmp_path, monkeypatch, dtype, shape, ranges):
    n, g, density = SHAPES[shape]
    m = _random_csc(n, g, density, dtype, seed=n + g)
    if shape == "sparse_with_empty_rows_and_columns":
        assert (np.diff(m.indptr) == 0).any() and (np.bincount(m.indices, minlength=n) == 0).any()
    monkeypatch.setattr(csr_store, "BLOCK_NNZ", 37)  # many blocks of genes
    kw = {"range_bytes": 1} if ranges == "many" else {}  # 1 byte: one cell per range
    got, _, attrs = _convert(tmp_path, m, **kw)
    _same(got, m.tocsr())
    assert got.has_sorted_indices
    assert attrs["encoding-type"] == "csr_matrix" and attrs["encoding-version"] == "0.1.0"
    assert attrs["shape"] == [n, g]
    assert attrs["dtypes"]["data"] == np.dtype(dtype)
    assert attrs["dtypes"]["indices"] == np.int32 and attrs["dtypes"]["indptr"] == np.int32


@pytest.mark.parametrize("ranges", ["one", "many"])
def test_unsorted_rows_stored_zeros_and_repeats_are_kept(tmp_path, monkeypatch, ranges):
    # Column 0: rows out of order and row 2 twice; column 1: a stored zero; column 2: empty.
    data = np.asarray([5, 1, 7, 2, 0, 3, 9], dtype=np.float32)
    indices = np.asarray([3, 2, 0, 2, 1, 3, 0], dtype=np.int64)
    indptr = np.asarray([0, 4, 6, 6, 7], dtype=np.int64)
    m = sp.csc_matrix((data, indices, indptr), shape=(4, 4))
    want = sp.csc_matrix((data, indices, indptr), shape=(4, 4)).tocsr()
    assert want.nnz == 7 and 0 in want.data  # scipy keeps both
    monkeypatch.setattr(csr_store, "BLOCK_NNZ", 2)
    got, _, _ = _convert(tmp_path, m, **({"range_bytes": 1} if ranges == "many" else {}))
    _same(got, want)


def test_ranges_follow_the_byte_budget_and_use_files(tmp_path, monkeypatch):
    m = _random_csc(300, 50, 0.3, np.float32, seed=3)
    seen = []
    real = csr_store._Ranges.add

    def spy(self, b, rows, cols, data):
        seen.append((b, self.folder))
        return real(self, b, rows, cols, data)

    monkeypatch.setattr(csr_store._Ranges, "add", spy)
    per_value = 16 + 2 * 4
    got, _, _ = _convert(tmp_path, m, range_bytes=per_value * (m.nnz // 5))
    _same(got, m.tocsr())
    assert len({b for b, _ in seen}) in (5, 6) and all(folder is not None for _, folder in seen)
    assert csr_store.temp_bytes(m.nnz, 4) == 0  # fits one range at the real budget
    assert csr_store.temp_bytes(10**9, 4) == 12 * 10**9


def test_keep_and_extract_as_the_feature_split_uses_them(tmp_path, monkeypatch):
    m = _random_csc(90, 30, 0.3, np.float32, seed=5)
    keep = np.ones(30, dtype=bool)
    keep[[0, 7, 8, 29]] = False
    a, b = np.zeros(30, dtype=bool), np.zeros(30, dtype=bool)
    a[[7, 8]], b[[29]] = True, True
    monkeypatch.setattr(csr_store, "BLOCK_NNZ", 50)
    for kw in ({}, {"range_bytes": 500}):
        got, side, _ = _convert(tmp_path, m, keep=keep, extract=[a, b], **kw)
        _same(got, m[:, keep].tocsr())
        np.testing.assert_array_equal(side[0], m[:, a].toarray())
        np.testing.assert_array_equal(side[1], m[:, b].toarray())


def test_a_row_number_outside_the_cells_is_refused(tmp_path):
    src = tmp_path / "bad.h5"
    _write_csc_arrays(src, np.ones(2, np.float32), np.asarray([0, 9]), np.asarray([0, 1, 2]), (3, 2))
    with h5py.File(src, "r") as fi, h5py.File(tmp_path / "o.h5", "w") as fo:
        with pytest.raises(csr_store.CsrStoreError, match="outside its 3 cells"):
            csr_store.convert(fi["X"], fo, "X")


# ---------------------------------------------------------------------------
# A dataset in memory, a whole file
# ---------------------------------------------------------------------------


def _dataset(fmt: str = "csc", n: int = 70, g: int = 25, seed: int = 1) -> ad.AnnData:
    X = _random_csc(n, g, 0.3, np.float32, seed)
    counts = _random_csc(n, g, 0.3, np.int32, seed + 1)
    raw = _random_csc(n, g + 5, 0.3, np.float32, seed + 2)
    as_fmt = {"csc": lambda m: m.tocsc(), "csr": lambda m: m.tocsr(), "dense": lambda m: m.toarray()}[fmt]
    a = ad.AnnData(as_fmt(raw), obs=pd.DataFrame({"group": pd.Categorical([f"g{i % 3}" for i in range(n)])},
                                                 index=[f"cell{i}" for i in range(n)]),
                   var=pd.DataFrame(index=[f"raw{i}" for i in range(g + 5)]))
    a.raw = a
    a = a[:, :g].copy()
    a.var_names = [f"gene{i}" for i in range(g)]
    a.X = as_fmt(X)
    a.layers["counts"] = as_fmt(counts)
    a.layers["dense"] = np.arange(n * g, dtype=np.float32).reshape(n, g)
    a.obsm["X_umap"] = np.random.default_rng(seed).normal(size=(n, 2)).astype(np.float32)
    a.obsp["connectivities"] = sp.random(n, n, 0.05, format="csc", random_state=seed, dtype=np.float32)
    a.uns["note"] = {"k": np.arange(3)}
    return a


def _formats(path: Path) -> dict[str, str]:
    with h5py.File(path, "r") as f:
        return {where: csr_store.encoding(node) for where, node in csr_store.matrices(f)}


def _assert_same_dataset(a: ad.AnnData, b: ad.AnnData) -> None:
    """Same values, dtypes and order everywhere; the matrix layout is not compared."""
    assert a.shape == b.shape
    pd.testing.assert_frame_equal(a.obs, b.obs)
    pd.testing.assert_frame_equal(a.var, b.var)
    pairs = [(a.X, b.X), (a.raw.X, b.raw.X)] + [(a.layers[k], b.layers[k]) for k in a.layers]
    assert set(a.layers) == set(b.layers)
    for x, y in pairs:
        assert x.dtype == y.dtype
        if sp.issparse(x):
            np.testing.assert_array_equal(x.toarray(), y.toarray())
            assert x.nnz == y.nnz
        else:
            np.testing.assert_array_equal(np.asarray(x), np.asarray(y))
    pd.testing.assert_frame_equal(a.raw.var, b.raw.var)
    np.testing.assert_array_equal(a.obsm["X_umap"], b.obsm["X_umap"])
    assert (a.obsp["connectivities"] != b.obsp["connectivities"]).nnz == 0


def test_in_memory_turns_x_layers_and_raw(tmp_path):
    a = _dataset("csc")
    want = a.copy()
    assert csr_store.csc_in_memory(a) == ["X", "layers/counts", "raw/X"]
    assert csr_store.in_memory(a) == ["X", "layers/counts", "raw/X"]
    assert a.X.format == a.layers["counts"].format == a.raw.X.format == "csr"
    assert isinstance(a.layers["dense"], np.ndarray)
    _same(a.X, want.X.tocsr())
    _same(a.raw.X, want.raw.X.tocsr())
    _assert_same_dataset(want, a)
    assert csr_store.in_memory(a) == [] and csr_store.in_memory(_dataset("dense")) == []


@pytest.mark.parametrize("ranges", ["one", "many"])
def test_rewrite_file_changes_only_the_layout(tmp_path, monkeypatch, ranges):
    if ranges == "many":
        monkeypatch.setattr(csr_store, "RANGE_BYTES", 2000)
        monkeypatch.setattr(csr_store, "BLOCK_NNZ", 100)
    a = _dataset("csc")
    src, dst = tmp_path / "in.h5ad", tmp_path / "out.h5ad"
    a.write_h5ad(src)
    before = src.read_bytes()
    assert csr_store.csc_in_file(src) == ["X", "layers/counts", "raw/X"]
    done = csr_store.rewrite_file(src, dst, **({"workdir": tmp_path} if ranges == "many" else {}))
    assert done == ["X", "layers/counts", "raw/X"]
    assert src.read_bytes() == before  # only read
    assert _formats(dst) == {"X": "csr_matrix", "layers/counts": "csr_matrix", "layers/dense": "array",
                             "raw/X": "csr_matrix"}
    assert csr_store.csc_in_file(dst) == []
    got = ad.read_h5ad(dst)
    _assert_same_dataset(ad.read_h5ad(src), got)
    _same(got.X, a.X.tocsr())
    assert got.obsp["connectivities"].format == "csc"  # a graph is not a cells x genes matrix
    assert got.uns["note"]["k"].tolist() == [0, 1, 2]
    assert not list(tmp_path.glob(".csr-*"))


def _compression(path: Path, where: str = "X") -> list:
    with h5py.File(path, "r") as f:
        return [(f[where][k].compression, f[where][k].compression_opts) for k in ("data", "indices", "indptr")]


@pytest.mark.parametrize("source", [None, "gzip", "lzf"])
def test_converted_matrices_are_compressed_as_the_source(tmp_path, source):
    src, dst = tmp_path / "in.h5ad", tmp_path / "out.h5ad"
    _dataset("csc").write_h5ad(src, compression=source)
    csr_store.rewrite_file(src, dst)
    want = _compression(src)[0]
    assert _compression(dst) == [want] * 3 and _compression(dst, "raw/X") == [want] * 3
    assert want[0] == source
    # A caller that names one gets it, whatever the source.
    csr_store.rewrite_file(src, tmp_path / "gz.h5ad", compression=csr_store.GZIP)
    assert _compression(tmp_path / "gz.h5ad") == [("gzip", 4)] * 3


def test_copy_file_is_a_plain_copy_without_csc(tmp_path):
    src = tmp_path / "in.h5ad"
    _dataset("csr").write_h5ad(src)
    src.chmod(0o444)
    assert csr_store.copy_file(src, tmp_path / "a.h5ad") == []
    assert (tmp_path / "a.h5ad").read_bytes() == src.read_bytes()
    with h5py.File(tmp_path / "a.h5ad", "a"):  # writable, whatever the source's mode
        pass
    csc = tmp_path / "csc.h5ad"
    _dataset("csc").write_h5ad(csc)
    assert csr_store.copy_file(csc, tmp_path / "b.h5ad") == ["X", "layers/counts", "raw/X"]
    assert csr_store.csc_in_file("/nonexistent.h5ad") == []


def test_an_old_layout_file_is_left_as_it_is(tmp_path):
    p = tmp_path / "old.h5ad"
    m = _random_csc(5, 4, 0.5, np.float32)
    with h5py.File(p, "w") as f:
        g = f.create_group("X")
        g.attrs["h5sparse_format"] = "csc"
        g.attrs["h5sparse_shape"] = np.asarray(m.shape)
        for k in ("data", "indices", "indptr"):
            g.create_dataset(k, data=getattr(m, k))
    assert csr_store.csc_in_file(p) == []


def test_write_h5ad_of_a_new_dataset_has_no_csc(tmp_path):
    a = _dataset("csc")
    out = tmp_path / "new.h5ad"
    x = a.X
    assert csr_store.write_h5ad(a, out) == ["X", "layers/counts", "raw/X"]
    assert a.X is x and a.X.format == "csc"  # the dataset in memory is not changed
    assert set(_formats(out).values()) == {"csr_matrix", "array"}
    assert _compression(out)[0][0] == "gzip"  # as a plain write_h5ad of it, not the temporary file's lzf
    _assert_same_dataset(a, ad.read_h5ad(out))
    assert [p.name for p in tmp_path.iterdir()] == ["new.h5ad"]
    plain = tmp_path / "plain.h5ad"
    assert csr_store.write_h5ad(_dataset("csr"), plain) == []
    assert _formats(plain)["X"] == "csr_matrix"


# ---------------------------------------------------------------------------
# A dataset already in the library
# ---------------------------------------------------------------------------


def _library_dataset(data_dir: Path, did: str, a: ad.AnnData) -> Path:
    up = data_dir / "uploads" / did
    up.mkdir(parents=True)
    path = up / "old.h5ad"
    a.write_h5ad(path)
    (up / "metadata.json").write_text(json.dumps({"id": did, "name": "old", "filename": "old.h5ad",
                                                  "extension": ".h5ad", "status": "ready"}))
    return path


async def test_rewrite_dataset_writes_the_derived_file_and_leaves_the_upload(tmp_data_dir):
    dm = DatasetManager(data_dir=str(tmp_data_dir))
    upload = _library_dataset(tmp_data_dir, "d1", _dataset("csc"))
    sent = upload.read_bytes()
    adaptor = await dm.get_or_load_dataset("d1")
    assert adaptor.adata.X.format == "csc"
    out = await csr_store.rewrite_dataset_held(dm, "d1")
    assert out["converted"] == ["X", "layers/counts", "raw/X"]
    derived = tmp_data_dir / "derived" / "d1" / "old.h5ad"
    assert out["target"] == str(derived) and out["source"] == str(upload)
    assert upload.read_bytes() == sent
    assert [p.name for p in derived.parent.iterdir()] == ["old.h5ad"]  # no partial, no temp folder
    assert dm.get_dataset("d1") is None  # the cached CSC copy is dropped
    again = await dm.get_or_load_dataset("d1")
    assert again.h5ad_path == str(derived) and again.adata.X.format == "csr"
    _assert_same_dataset(ad.read_h5ad(upload), again.adata)
    # A second call has nothing to do and writes nothing.
    stamp = derived.stat().st_mtime_ns
    assert (await csr_store.rewrite_dataset_held(dm, "d1"))["converted"] == []
    assert derived.stat().st_mtime_ns == stamp and dm.get_dataset("d1") is again


async def test_rewrite_dataset_of_a_derived_file_survives_a_failed_write(tmp_data_dir, monkeypatch):
    dm = DatasetManager(data_dir=str(tmp_data_dir))
    _library_dataset(tmp_data_dir, "d2", _dataset("csr"))
    derived = tmp_data_dir / "derived" / "d2" / "old.h5ad"
    derived.parent.mkdir(parents=True)
    _dataset("csc", seed=9).write_h5ad(derived)
    before = derived.read_bytes()

    def boom(*a, **k):
        raise OSError("disk full")

    monkeypatch.setattr(csr_store, "convert", boom)
    with pytest.raises(OSError, match="disk full"):
        await csr_store.rewrite_dataset_held(dm, "d2")
    assert derived.read_bytes() == before and [p.name for p in derived.parent.iterdir()] == ["old.h5ad"]
    monkeypatch.undo()
    out = await csr_store.rewrite_dataset_held(dm, "d2")
    assert out["source"] == out["target"] == str(derived)
    _assert_same_dataset(_dataset("csc", seed=9), ad.read_h5ad(derived))
    assert csr_store.csc_in_file(derived) == []


def test_rewrite_dataset_is_only_called_inside_an_edit():
    """Nothing in the app calls it yet; whoever does must go through ``rewrite_dataset_held``."""
    src = Path(csr_store.__file__).parents[1]
    callers = [p.relative_to(src).as_posix() for p in src.rglob("*.py")
               if "rewrite_dataset(" in p.read_text() and p.name != "csr_store.py"]
    assert callers == []


# ---------------------------------------------------------------------------
# Memory
# ---------------------------------------------------------------------------


def test_the_copy_is_what_in_memory_adds(tmp_path):
    """tracemalloc (numpy reports to it): ``in_memory`` peaks at one matrix's CSR arrays over
    what the dataset held, the estimate's term."""
    import tracemalloc

    tracemalloc.start()  # before the matrices are made: a free is only seen for what it saw allocated
    m = _random_csc(20_000, 400, 0.25, np.float32, seed=4)
    a = ad.AnnData(m)
    a.raw = ad.AnnData(m.copy())
    one = preflight.Matrix("X", 20_000, 400, True, m.nnz, "float32", "int32").nbytes
    del m  # the dataset alone holds its matrices, as an ingest does
    tracemalloc.reset_peak()
    base = tracemalloc.get_traced_memory()[0]
    csr_store.in_memory(a)
    peak = tracemalloc.get_traced_memory()[1] - base
    tracemalloc.stop()
    assert 0.8 * one < peak < 1.2 * one


def test_convert_memory_is_a_range_not_the_matrix(tmp_path, monkeypatch):
    import tracemalloc

    # 6M values, 48 MB as CSR: half of it is well over the bound below (at 3M values the two
    # limits were 14.1 and 12 MB, and CI measured 12.46)
    m = _random_csc(60_000, 200, 0.5, np.float32, seed=6)
    src, dst = tmp_path / "in.h5", tmp_path / "out.h5"
    _write_csc_arrays(src, m.data, m.indices, m.indptr, m.shape)
    monkeypatch.setattr(csr_store, "BLOCK_NNZ", 100_000)
    budget = 2 * 2**20
    with h5py.File(src, "r") as fi, h5py.File(dst, "w") as fo:
        tracemalloc.start()
        base = tracemalloc.get_traced_memory()[0]
        csr_store.convert(fi["X"], fo, "X", range_bytes=budget)
        peak = tracemalloc.get_traced_memory()[1] - base
        tracemalloc.stop()
        _same(read_elem(fo["X"]), m.tocsr())
    # One range, one block of genes (96 bytes a value) and the per-cell counts and pointers.
    assert peak < budget + 96 * 100_000 + 40 * 60_000
    assert peak < m.nnz * 8 / 2
