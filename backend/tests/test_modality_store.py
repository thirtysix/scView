"""The on-disk per-cell store: rows come back as written, by barcode, in any order."""

import threading

import numpy as np
import pytest
import scipy.sparse as sp

from scview.core import modality_store as ms


def _matrix(n_obs=50, n_vars=20, seed=0):
    return sp.random(n_obs, n_vars, density=0.2, format="csr", random_state=seed, dtype=np.float32)


def _barcodes(n=50):
    return [f"cell{i}" for i in range(n)]


def test_paths_are_per_dataset(tmp_path):
    assert ms.store_path(tmp_path, "abc", "atac") == tmp_path / "modalities" / "abc" / "atac.h5"


def test_round_trip_two_matrices(tmp_path):
    a, b = _matrix(seed=1), _matrix(seed=2)
    path = ms.write(tmp_path / "x.h5", {"spliced": a, "unspliced": b}, _barcodes(), [f"g{i}" for i in range(20)],
                    feature_attrs={"interval": [f"chr1:{i}-{i + 1}" for i in range(20)]}, attrs={"genome": "GRCh38"})
    with ms.ModalityStore(path) as s:
        assert sorted(s.matrices) == ["spliced", "unspliced"]
        assert (s.n_obs, s.n_vars) == (50, 20)
        assert s.meta == {"genome": "GRCh38"}
        assert s.feature_attr("interval")[3] == "chr1:3-4"
        assert s.nnz("spliced") == a.nnz
        assert (s.read("spliced", np.arange(50)) != a).nnz == 0
        assert (s.read("unspliced", np.arange(50)) != b).nnz == 0


def test_rows_in_any_order_and_repeated(tmp_path):
    a = _matrix()
    path = ms.write(tmp_path / "x.h5", {"m": a}, _barcodes(), [f"g{i}" for i in range(20)])
    rows = np.array([7, 3, 4, 49, 0, 3])
    with ms.ModalityStore(path) as s:
        assert (s.read("m", rows) != a[rows]).nnz == 0
        assert s.read("m", np.array([], dtype=np.int64)).shape == (0, 20)


def test_barcodes_map_to_rows_and_missing_is_minus_one(tmp_path):
    path = ms.write(tmp_path / "x.h5", {"m": _matrix()}, _barcodes(), [f"g{i}" for i in range(20)])
    with ms.ModalityStore(path) as s:
        rows = s.rows_of(["cell9", "nope", "cell0"])
        assert rows.tolist() == [9, -1, 0]
        with pytest.raises(ms.ModalityStoreError, match="out of range"):
            s.read("m", rows)


def test_blocks_cover_every_row_in_order_and_respect_the_limit(tmp_path):
    a = _matrix(n_obs=200, n_vars=40)
    path = ms.write(tmp_path / "x.h5", {"m": a}, _barcodes(200), [f"g{i}" for i in range(40)])
    rows = np.arange(199, -1, -1)
    with ms.ModalityStore(path) as s:
        blocks = list(s.iter_blocks("m", rows, max_values=100))
    assert len(blocks) > 1
    assert np.concatenate([p for p, _ in blocks]).tolist() == list(range(200))
    assert all(b.nnz <= 100 or b.shape[0] == 1 for _, b in blocks)
    assert (sp.vstack([b for _, b in blocks]).tocsr() != a[rows]).nnz == 0


def test_writer_appends_blocks(tmp_path):
    a = _matrix()
    with ms.StoreWriter(tmp_path / "x.h5", _barcodes(), [f"g{i}" for i in range(20)]) as w:
        for start in range(0, 50, 7):
            w.append("m", a[start:start + 7])
    with ms.ModalityStore(tmp_path / "x.h5") as s:
        assert (s.read("m", np.arange(50)) != a).nnz == 0


def test_a_stopped_writer_leaves_nothing_that_looks_finished(tmp_path):
    with pytest.raises(ValueError):
        with ms.StoreWriter(tmp_path / "x.h5", _barcodes(), [f"g{i}" for i in range(20)]) as w:
            w.append("m", _matrix()[:10])
            raise ValueError("stop")
    assert list(tmp_path.iterdir()) == []


def test_too_few_rows_is_refused(tmp_path):
    w = ms.StoreWriter(tmp_path / "x.h5", _barcodes(), [f"g{i}" for i in range(20)])
    w.append("m", _matrix()[:10])
    with pytest.raises(ms.ModalityStoreError, match="10 of 50"):
        w.close()
    w.abort()


def test_wrong_columns_and_unknown_matrix_say_so(tmp_path):
    with pytest.raises(ms.ModalityStoreError, match="columns"):
        ms.write(tmp_path / "x.h5", {"m": _matrix()}, _barcodes(), ["only_one"])
    path = ms.write(tmp_path / "y.h5", {"m": _matrix()}, _barcodes(), [f"g{i}" for i in range(20)])
    with ms.ModalityStore(path) as s:
        with pytest.raises(ms.ModalityStoreError, match="no matrix 'z'"):
            s.read("z", [0])
    with pytest.raises(ms.ModalityStoreError, match="missing"):
        ms.ModalityStore(tmp_path / "gone.h5")


def test_threads_read_the_same_store(tmp_path):
    a = _matrix(n_obs=300, n_vars=30)
    path = ms.write(tmp_path / "x.h5", {"m": a}, _barcodes(300), [f"g{i}" for i in range(30)])
    bad = []
    with ms.ModalityStore(path) as s:
        def worker(seed):
            rng = np.random.default_rng(seed)
            for _ in range(40):
                rows = rng.integers(0, 300, 25)
                if (s.read("m", rows) != a[rows]).nnz:
                    bad.append(seed)
        threads = [threading.Thread(target=worker, args=(i,)) for i in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
    assert not bad


def test_remove_dataset_removes_its_stores(tmp_path):
    from scview.core.dataset_manager import DatasetManager

    dm = DatasetManager(str(tmp_path))
    ms.write(ms.store_path(tmp_path, "d1", "atac"), {"m": _matrix()}, _barcodes(), [f"g{i}" for i in range(20)])
    assert dm.remove_dataset("d1")
    assert not ms.store_dir(tmp_path, "d1").exists()
