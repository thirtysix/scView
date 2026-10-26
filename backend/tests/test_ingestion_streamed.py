"""Streamed ingest (``core/ingestion/streamed.py``) against today's in-memory ingest.

Every equivalence test stages the same files in two sessions, commits one with the
``streamed`` option off (``load_unit`` + ``merge_units`` + ``write_h5ad``, as always) and
one with it on, and compares the two h5ad files: X (values, cell and gene order, dtypes),
obs and var (names, columns, dtypes), obsm, layers, .raw, obsp and uns (the provenance
without its timestamps).
"""

from __future__ import annotations

import gzip
import json
import shutil
import tracemalloc
from pathlib import Path

import anndata as ad
import h5py
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

from scview.core import memory_budget
from scview.core.ingestion import IngestCommitError, IngestOptions, IngestSessionManager
from scview.core.ingestion import streamed
from scview.core.ingestion.bundling import build_bundle


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------


def _barcode(i: int, lane: int = 1) -> str:
    bases = "ACGT"
    tail = "".join(bases[(i >> (2 * k)) & 3] for k in range(10))
    return f"AAAC{tail}-{lane}"


def _counts(n_cells: int, n_feat: int, seed: int = 0, density: float = 0.2) -> sp.csr_matrix:
    rng = np.random.default_rng(seed)
    m = sp.random(n_cells, n_feat, density=density, format="csr", random_state=rng,
                  data_rvs=lambda k: rng.integers(1, 60, k)).astype(np.int64)
    m.sort_indices()
    return m


def _features(n_genes: int, antibodies: int = 0, tags: int = 0, guides: int = 0, dup: bool = True):
    names = [f"Gene{i}" for i in range(n_genes)]
    if dup and n_genes > 4:
        names[3] = names[4] = "Dup"  # var_names_make_unique: "Dup", "Dup-1"
    ids = [f"ENSG{i:011d}" for i in range(n_genes)]
    types = ["Gene Expression"] * n_genes
    for i in range(antibodies):
        names.append(f"CD{i}_TotalSeqB")
        ids.append(f"AB{i}")
        types.append("Antibody Capture")
    for i in range(tags):
        names.append(f"CMO30{i}")
        ids.append(f"CMO30{i}")
        types.append("Multiplexing Capture")
    for i in range(guides):
        names.append(f"guide{i}")
        ids.append(f"guide{i}")
        types.append("CRISPR Guide Capture")
    return names, ids, types


def _tenx_v3(p: Path, X: sp.csr_matrix, names, ids, types, lane: int = 1) -> Path:
    """A Cell Ranger 3+ .h5: genes x cells CSC = these cells x genes CSR arrays."""
    with h5py.File(p, "w") as f:
        g = f.create_group("matrix")
        g.create_dataset("data", data=X.data.astype("int32"), compression="gzip")
        g.create_dataset("indices", data=X.indices.astype("int64"), compression="gzip")
        g.create_dataset("indptr", data=X.indptr.astype("int64"))
        g.create_dataset("shape", data=np.array([X.shape[1], X.shape[0]], dtype="int32"))
        g.create_dataset("barcodes", data=np.array([_barcode(i, lane).encode() for i in range(X.shape[0])]))
        ff = g.create_group("features")
        ff.create_dataset("id", data=np.array([s.encode() for s in ids]))
        ff.create_dataset("name", data=np.array([s.encode() for s in names]))
        ff.create_dataset("feature_type", data=np.array([s.encode() for s in types]))
        ff.create_dataset("genome", data=np.array([b"GRCh38"] * len(ids)))
        ff.create_dataset("_all_tag_keys", data=np.array([b"genome"]))
    return p


def _tenx_legacy(p: Path, X: sp.csr_matrix, genome: str = "mm10") -> Path:
    with h5py.File(p, "w") as f:
        g = f.create_group(genome)
        g.create_dataset("data", data=X.data.astype("int32"), compression="gzip", chunks=(64,))
        g.create_dataset("indices", data=X.indices.astype("int64"), compression="gzip", chunks=(64,))
        g.create_dataset("indptr", data=X.indptr.astype("int64"))
        g.create_dataset("shape", data=np.array([X.shape[1], X.shape[0]], dtype="int32"))
        g.create_dataset("barcodes", data=np.array([_barcode(i).encode() for i in range(X.shape[0])]))
        g.create_dataset("genes", data=np.array([f"ENSMUSG{i:011d}".encode() for i in range(X.shape[1])]))
        g.create_dataset("gene_names", data=np.array([f"Gene{i % (X.shape[1] - 2)}".encode()
                                                      for i in range(X.shape[1])]))
    return p


def _mex(d: Path, X: sp.csr_matrix, names, ids, types, *, order: str = "cell", gz: bool = True,
         field: str = "integer", prefix: str = "") -> list[Path]:
    """matrix.mtx (genes x cells) in ``order``: "cell" = by cell, genes descending within a
    cell (as Cell Ranger writes it), "gene" = scipy's row order, "shuffled" = any order."""
    d.mkdir(parents=True, exist_ok=True)
    coo = X.tocoo()
    cells, genes, vals = coo.row.astype(np.int64), coo.col.astype(np.int64), coo.data
    if field == "real":
        vals = vals / 7.0
    if order == "cell":
        idx = np.lexsort((-genes, cells))
    elif order == "gene":
        idx = np.lexsort((cells, genes))
    else:
        idx = np.random.default_rng(1).permutation(len(cells))
    lines = [f"%%MatrixMarket matrix coordinate {field} general", "%metadata_json: {}",
             f"{X.shape[1]} {X.shape[0]} {len(idx)}"]
    if field == "pattern":
        lines += [f"{genes[i] + 1} {cells[i] + 1}" for i in idx]
    elif field == "real":
        lines += [f"{genes[i] + 1} {cells[i] + 1} {float(vals[i])!r}" for i in idx]
    else:
        lines += [f"{genes[i] + 1} {cells[i] + 1} {int(vals[i])}" for i in idx]
    texts = {
        f"{prefix}matrix.mtx": "\n".join(lines) + "\n",
        f"{prefix}barcodes.tsv": "".join(_barcode(i) + "\n" for i in range(X.shape[0])),
        f"{prefix}features.tsv": "".join(f"{i}\t{n}\t{t}\n" for i, n, t in zip(ids, names, types)),
    }
    out = []
    for name, text in texts.items():
        p = d / (name + (".gz" if gz else ""))
        if gz:
            with gzip.open(p, "wt") as fh:
                fh.write(text)
        else:
            p.write_text(text)
        out.append(p)
    return out


def _h5ad(p: Path, X, *, layers=True, raw=True, obsp=True, var_names=None, feature_types=None,
          obsm_protein: bool = False) -> Path:
    n_obs, n_var = X.shape
    a = ad.AnnData(X=X)
    a.obs_names = [f"c{i}" for i in range(n_obs)]
    a.obs_names = list(a.obs_names[:-1]) + ["c0"]  # a duplicate: obs_names_make_unique
    a.var_names = var_names or [f"G{i}" for i in range(n_var)]
    a.obs["donor"] = pd.Categorical(np.where(np.arange(n_obs) % 2, "d1", "d2"))
    a.obs["n"] = np.arange(n_obs, dtype=np.int32)
    a.var["symbol"] = [f"sym{i}" for i in range(n_var)]
    if feature_types is not None:
        a.var["feature_types"] = feature_types
    a.obsm["X_pca"] = np.random.default_rng(0).normal(size=(n_obs, 3)).astype(np.float32)
    a.varm["loadings"] = np.ones((n_var, 2))
    a.uns["note"] = {"a": 1, "b": "x"}
    if obsm_protein:
        a.obsm["protein_expression"] = pd.DataFrame(np.arange(n_obs * 2).reshape(n_obs, 2) % 5,
                                                    index=a.obs_names, columns=["CD3", "CD4"])
    if layers:
        a.layers["counts"] = X.copy()
    if raw:
        a.raw = a
    if obsp:
        a.obsp["connectivities"] = sp.random(n_obs, n_obs, density=0.05, format="csr", random_state=0)
    a.write_h5ad(p, compression="gzip")
    return p


# ---------------------------------------------------------------------------
# Running both paths and comparing
# ---------------------------------------------------------------------------


@pytest.fixture()
def data_dir(tmp_path: Path) -> Path:
    d = tmp_path / "data"
    for sub in ("uploads", "converted", "cache"):
        (d / sub).mkdir(parents=True)
    return d


@pytest.fixture()
def calls(monkeypatch):
    """How many times the streamed writer ran."""
    seen = {"n": 0}
    real = streamed.ingest

    def spy(*a, **kw):
        seen["n"] += 1
        return real(*a, **kw)

    monkeypatch.setattr(streamed, "ingest", spy)
    return seen


def _commit(data_dir: Path, files: list[Path], **opts) -> Path:
    mgr = IngestSessionManager(str(data_dir))
    sid = mgr.create_session()
    for f in files:
        dest = mgr.files_dir(sid) / f.name
        if f.is_dir():
            shutil.copytree(f, dest)
        else:
            shutil.copyfile(f, dest)
    mgr.set_options(sid, IngestOptions(**opts))
    did = mgr._build_dataset(sid)
    return next((data_dir / "ingested" / did).glob("*.h5ad"))


def _both(data_dir: Path, files: list[Path], calls, **opts) -> tuple[Path, Path]:
    before = calls["n"]
    today = _commit(data_dir, files, streamed="off", **opts)
    assert calls["n"] == before
    new = _commit(data_dir, files, streamed="on", **opts)
    assert calls["n"] == before + 1
    return today, new


def _same_matrix(a, b, *, any_sparse_format: bool = False):
    assert a.shape == b.shape
    assert np.dtype(a.dtype) == np.dtype(b.dtype)
    if sp.issparse(a) or sp.issparse(b):
        assert sp.issparse(a) and sp.issparse(b)
        if not any_sparse_format:
            assert a.format == b.format
            assert a.indices.dtype == b.indices.dtype
            assert a.indptr.dtype == b.indptr.dtype
        assert (a != b).nnz == 0
        assert a.nnz == b.nnz
    else:
        np.testing.assert_array_equal(np.asarray(a), np.asarray(b))


def _same_value(a, b):
    if isinstance(a, pd.DataFrame):
        pd.testing.assert_frame_equal(a, b)
    elif sp.issparse(a):
        _same_matrix(a, b)
    elif isinstance(a, np.ndarray):
        np.testing.assert_array_equal(a, b)
    elif isinstance(a, dict):
        assert set(a) == set(b)
        for k in a:
            _same_value(a[k], b[k])
    else:
        assert a == b


def _provenance(a: ad.AnnData) -> dict:
    p = json.loads(a.uns["scview_provenance"])
    p["source"].pop("ingested_at", None)
    for h in p["history"]:
        for k in ("timestamp", "commit_id", "parent"):
            h.pop(k, None)
    p.get("current", {}).pop("head", None)
    return p


def assert_same_h5ad(today: Path, new: Path, *, any_sparse_format: bool = False) -> None:
    a, b = ad.read_h5ad(today), ad.read_h5ad(new)
    assert a.shape == b.shape
    _same_matrix(a.X, b.X, any_sparse_format=any_sparse_format) if a.X is not None else None
    assert (a.X is None) == (b.X is None)
    pd.testing.assert_frame_equal(a.obs, b.obs)
    pd.testing.assert_frame_equal(a.var, b.var)
    for attr in ("obsm", "layers", "obsp", "varm", "varp"):
        ea, eb = getattr(a, attr), getattr(b, attr)
        assert set(ea.keys()) == set(eb.keys()), attr
        for k in ea.keys():
            _same_value(ea[k], eb[k])
    assert (a.raw is None) == (b.raw is None)
    if a.raw is not None:
        _same_matrix(a.raw.X, b.raw.X)
        pd.testing.assert_frame_equal(a.raw.var, b.raw.var)
    assert set(a.uns.keys()) == set(b.uns.keys())
    for k in a.uns.keys():
        if k == "scview_provenance":
            assert _provenance(a) == _provenance(b)
        else:
            _same_value(a.uns[k], b.uns[k])


# ---------------------------------------------------------------------------
# 10x HDF5
# ---------------------------------------------------------------------------


def test_tenx_h5_streams_to_the_same_h5ad(tmp_path, data_dir, calls):
    names, ids, types = _features(40)
    f = _tenx_v3(tmp_path / "filtered_feature_bc_matrix.h5", _counts(300, 40), names, ids, types)
    today, new = _both(data_dir, [f], calls)
    assert_same_h5ad(today, new)
    a = ad.read_h5ad(new)
    assert list(a.var_names[3:5]) == ["Dup", "Dup-1"]
    assert a.X.dtype == np.float32 and a.X.indices.dtype == np.int32


def test_tenx_h5_cell_ranger_2_layout(tmp_path, data_dir, calls):
    f = _tenx_legacy(tmp_path / "1M_neurons_filtered_gene_bc_matrices_h5.h5", _counts(250, 30, seed=3))
    today, new = _both(data_dir, [f], calls)
    assert_same_h5ad(today, new)


def test_tenx_h5_cite_seq_split_is_the_same(tmp_path, data_dir, calls):
    # Antibodies -> obsm protein, CellPlex tags -> obsm hto_counts, guides -> obsm guide_counts.
    names, ids, types = _features(30, antibodies=4, tags=2, guides=2)
    X = _counts(200, len(names), seed=5, density=0.3)
    f = _tenx_v3(tmp_path / "cite.h5", X, names, ids, types)
    today, new = _both(data_dir, [f], calls)
    assert_same_h5ad(today, new)
    a = ad.read_h5ad(new)
    assert a.n_vars == 30
    assert list(a.obsm["protein_counts"].columns) == ["CD0", "CD1", "CD2", "CD3"]
    np.testing.assert_array_equal(a.obsm["protein_counts"].to_numpy(), X[:, 30:34].toarray())
    assert "hto_counts" in a.obsm
    assert a.uns["scview_protein"]["other_features"] == {}
    assert list(a.obsm["guide_counts"].columns) == ["guide0", "guide1"]
    np.testing.assert_array_equal(a.obsm["guide_counts"].to_numpy(), X[:, 36:38].toarray())


def test_tenx_h5_block_boundaries(tmp_path, data_dir, calls, monkeypatch):
    # Tiny blocks: every block edge (and a cell longer than a block) is crossed.
    monkeypatch.setattr(streamed, "BLOCK_NNZ", 7)
    names, ids, types = _features(25, antibodies=2)
    X = _counts(120, len(names), seed=8, density=0.4)
    f = _tenx_v3(tmp_path / "f.h5", X, names, ids, types)
    today, new = _both(data_dir, [f], calls)
    assert_same_h5ad(today, new)


def test_cells_without_values_keep_their_rows(tmp_path, data_dir, calls):
    X = _counts(50, 20, seed=2).tolil()
    X[[0, 7, 8, 49], :] = 0
    X = X.tocsr()
    X.eliminate_zeros()
    names, ids, types = _features(20)
    f = _tenx_v3(tmp_path / "f.h5", X, names, ids, types)
    today, new = _both(data_dir, [f], calls)
    assert_same_h5ad(today, new)


# ---------------------------------------------------------------------------
# 10x MEX
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("order", ["cell", "gene", "shuffled"])
def test_mex_streams_to_the_same_values_in_csr(tmp_path, data_dir, calls, order, monkeypatch):
    monkeypatch.setattr(streamed, "MEX_CHUNK_LINES", 97)  # many chunks, cells across chunk edges
    monkeypatch.setattr(streamed, "MEX_BUCKET_ENTRIES", 150)
    names, ids, types = _features(35, antibodies=3)
    X = _counts(180, len(names), seed=4)
    files = _mex(tmp_path / "mex", X, names, ids, types, order=order)
    today, new = _both(data_dir, files, calls)
    # Today's MEX reader leaves X as CSC (genes major); the streamed one stores CSR.
    assert_same_h5ad(today, new, any_sparse_format=True)
    assert ad.read_h5ad(today).X.format == "csc"
    assert ad.read_h5ad(new).X.format == "csr"
    assert ad.read_h5ad(new).X.has_sorted_indices


@pytest.mark.parametrize("field", ["real", "pattern"])
def test_mex_value_fields(tmp_path, data_dir, calls, field):
    names, ids, types = _features(12, dup=False)
    files = _mex(tmp_path / "mex", _counts(40, 12, seed=6), names, ids, types, field=field, gz=False)
    today, new = _both(data_dir, files, calls)
    assert_same_h5ad(today, new, any_sparse_format=True)


def test_mex_duplicate_entries_are_summed(tmp_path, data_dir, calls):
    d = tmp_path / "mex"
    d.mkdir()
    (d / "matrix.mtx").write_text("%%MatrixMarket matrix coordinate integer general\n3 2 4\n"
                                  "1 1 2\n3 1 1\n1 1 5\n2 2 4\n")
    (d / "barcodes.tsv").write_text(_barcode(0) + "\n" + _barcode(1) + "\n")
    (d / "features.tsv").write_text("".join(f"E{i}\tG{i}\tGene Expression\n" for i in range(3)))
    today, new = _both(data_dir, sorted(d.iterdir()), calls)
    assert_same_h5ad(today, new, any_sparse_format=True)
    assert ad.read_h5ad(new).X.toarray().tolist() == [[7, 0, 1], [0, 4, 0]]


def test_mex_output_reads_by_rows(tmp_path, data_dir, calls):
    # The chunked pipeline reads X by row blocks; today's CSC MEX output cannot be.
    from scview.core import chunked

    names, ids, types = _features(10)
    files = _mex(tmp_path / "mex", _counts(60, 10), names, ids, types)
    new = _commit(data_dir, files, streamed="on")
    X = chunked.read_lazy(new, "X", chunk_rows=16).compute(scheduler="sync")
    assert X.shape == (60, 10)


# ---------------------------------------------------------------------------
# .h5ad
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("kind", ["csr", "csc", "dense"])
def test_h5ad_with_layers_raw_and_graphs(tmp_path, data_dir, calls, kind):
    X = _counts(80, 15, seed=7).astype(np.float32)
    X = {"csr": X, "csc": X.tocsc(), "dense": X.toarray()}[kind]
    f = _h5ad(tmp_path / "data.h5ad", X)
    today, new = _both(data_dir, [f], calls)
    assert_same_h5ad(today, new)


def test_h5ad_feature_types_split(tmp_path, data_dir, calls):
    types = ["Gene Expression"] * 12 + ["Antibody Capture"] * 3
    X = _counts(70, 15, seed=9).astype(np.float32)
    f = _h5ad(tmp_path / "cite.h5ad", X, raw=False, feature_types=types)
    today, new = _both(data_dir, [f], calls)
    assert_same_h5ad(today, new)
    a = ad.read_h5ad(new)
    assert a.n_vars == 12 and a.layers["counts"].shape == (70, 12)
    assert a.obsm["protein_counts"].shape == (70, 3)


def test_h5ad_protein_from_another_tool_is_adopted(tmp_path, data_dir, calls):
    f = _h5ad(tmp_path / "totalvi.h5ad", _counts(30, 8).astype(np.float32), layers=False, raw=False,
              obsp=False, obsm_protein=True)
    today, new = _both(data_dir, [f], calls)
    assert_same_h5ad(today, new)
    assert "protein_clr" in ad.read_h5ad(new).obsm


# ---------------------------------------------------------------------------
# Several units
# ---------------------------------------------------------------------------


def test_merge_of_lanes_with_the_same_genes(tmp_path, data_dir, calls):
    names, ids, types = _features(30)
    files = [_tenx_v3(tmp_path / f"lane{k}.h5", _counts(90 + k, 30, seed=k), names, ids, types, lane=k)
             for k in (1, 2, 3)]
    today, new = _both(data_dir, files, calls)
    assert_same_h5ad(today, new)
    a = ad.read_h5ad(new)
    assert list(a.obs["sample"].cat.categories) == ["lane1", "lane2", "lane3"]


@pytest.mark.parametrize("join", ["inner", "outer"])
def test_merge_of_different_gene_lists(tmp_path, data_dir, calls, join):
    n1, i1, t1 = _features(30, dup=False)
    sel = np.random.default_rng(0).permutation(30)[:22]  # another order, 8 genes missing
    n2, i2, t2 = [n1[i] for i in sel] + ["Extra"], [i1[i] for i in sel] + ["E"], [t1[i] for i in sel] + [t1[0]]
    f1 = _tenx_v3(tmp_path / "a.h5", _counts(60, 30, seed=1), n1, i1, t1)
    f2 = _tenx_v3(tmp_path / "b.h5", _counts(70, 23, seed=2), n2, i2, t2)
    today, new = _both(data_dir, [f1, f2], calls, join=join)
    assert_same_h5ad(today, new)


def test_merge_of_cite_seq_lanes_keeps_the_shared_panel(tmp_path, data_dir, calls):
    n1, i1, t1 = _features(20, antibodies=3)
    n2, i2, t2 = _features(20, antibodies=2)
    f1 = _tenx_v3(tmp_path / "a.h5", _counts(40, len(n1), seed=1), n1, i1, t1)
    f2 = _tenx_v3(tmp_path / "b.h5", _counts(50, len(n2), seed=2), n2, i2, t2)
    today, new = _both(data_dir, [f1, f2], calls)
    assert_same_h5ad(today, new)
    assert list(ad.read_h5ad(new).obsm["protein_counts"].columns) == ["CD0", "CD1"]


def test_merge_with_reconciled_gene_ids(tmp_path, data_dir, calls):
    # A 10x h5 named by symbols and an .h5ad named by Ensembl ids: the plan re-indexes
    # the 10x sample on its gene_ids, both ways.
    names, ids, types = _features(25, dup=False)
    f1 = _tenx_v3(tmp_path / "tenx.h5", _counts(40, 25, seed=1), names, ids, types)
    f2 = _h5ad(tmp_path / "ens.h5ad", _counts(30, 25, seed=2).astype(np.float32), layers=False, raw=False,
               obsp=False, var_names=list(ids))
    today, new = _both(data_dir, [f1, f2], calls)
    assert_same_h5ad(today, new)
    assert ad.read_h5ad(new).var_names[0] == ids[0]


def test_merge_of_h5ad_with_layers_is_refused_with_what_to_do(tmp_path, data_dir):
    f1 = _h5ad(tmp_path / "a.h5ad", _counts(20, 6, seed=1).astype(np.float32))
    f2 = _h5ad(tmp_path / "b.h5ad", _counts(20, 6, seed=2).astype(np.float32))
    with pytest.raises(IngestCommitError, match="one at a time"):
        _commit(data_dir, [f1, f2], streamed="on")


def test_merge_plan_from_skeletons_is_todays_plan(tmp_path, data_dir, monkeypatch):
    names, ids, types = _features(25, dup=False)
    f1 = _tenx_v3(tmp_path / "tenx.h5", _counts(40, 25, seed=1), names, ids, types)
    f2 = _h5ad(tmp_path / "ens.h5ad", _counts(30, 25, seed=2).astype(np.float32) / 3, layers=False,
               raw=False, obsp=False, var_names=list(ids))
    mgr = IngestSessionManager(str(data_dir))
    sid = mgr.create_session()
    for f in (f1, f2):
        shutil.copyfile(f, mgr.files_dir(sid) / f.name)
    plan = mgr.merge_plan(sid)
    monkeypatch.setattr(streamed, "STREAMABLE", ())  # today's: every sample loaded
    assert mgr.merge_plan(sid) == plan
    assert plan.reconciliation is not None and plan.reconciliation.feasible
    assert any("raw counts" in w for w in plan.warnings)


# ---------------------------------------------------------------------------
# Choosing, memory, errors
# ---------------------------------------------------------------------------


def _budget(limit_gb: float, used_gb: float):
    gb = 1024**3
    return memory_budget.MemoryBudget(int(limit_gb * gb), int(used_gb * gb), True, "container limit")


def test_auto_streams_only_when_reading_whole_would_not_fit(tmp_path, data_dir, calls, monkeypatch):
    names, ids, types = _features(20)
    f = _tenx_v3(tmp_path / "f.h5", _counts(100, 20), names, ids, types)
    unit = build_bundle([f]).units[0]
    monkeypatch.setattr(memory_budget, "read_budget", lambda *a, **k: _budget(64, 1))
    assert not streamed.choose([unit]).streamed
    _commit(data_dir, [f])
    assert calls["n"] == 0
    # Only 0.5 GB free of 4 GB: under the working reserve (1 GB) alone.
    monkeypatch.setattr(memory_budget, "read_budget", lambda *a, **k: _budget(4, 3.5))
    choice = streamed.choose([unit])
    assert choice.streamed and "free" in choice.reason
    _commit(data_dir, [f])
    assert calls["n"] == 1
    assert not streamed.choose([unit], "off").streamed


def test_ingest_preflight_says_blocks_when_whole_would_not_fit(tmp_path, data_dir, monkeypatch):
    from scview.core import preflight
    from scview.core.dataset_manager import DatasetManager

    names, ids, types = _features(20)
    mgr = IngestSessionManager(str(data_dir))
    sid = mgr.create_session()
    for k in (1, 2):
        _tenx_v3(mgr.files_dir(sid) / f"lane{k}.h5", _counts(100, 20, seed=k), names, ids, types, lane=k)
    dm = DatasetManager(str(data_dir))
    monkeypatch.setattr(memory_budget, "read_budget", lambda *a, **k: _budget(64, 1))
    body = preflight.session_preflight(mgr, sid, dm)
    assert not body["units"][0]["streamed"]
    assert body["merge"]["sentence"].startswith("Merging the 2 samples takes ")
    monkeypatch.setattr(memory_budget, "read_budget", lambda *a, **k: _budget(4, 3.5))
    body = preflight.session_preflight(mgr, sid, dm)
    assert body["units"][0]["streamed"]
    assert "read and saved in blocks instead" in body["units"][0]["verdict"]["sentence"]
    assert body["merge"]["sentence"].startswith("Merging the 2 samples in blocks takes ")
    assert streamed.BLOCKS_NOTE in body["merge"]["notes"]


def test_tenx_header_indices_widen_past_int32(tmp_path):
    from scview.core import preflight

    p = tmp_path / "f.h5"
    with h5py.File(p, "w") as f:
        g = f.create_group("mm10")
        g.create_dataset("data", shape=(2**31 + 5,), dtype="int32")  # never written: no disk used
        g.create_dataset("shape", data=np.array([100, 10], dtype="int32"))
    h = preflight.read_header(p, "10x_h5")
    assert h.x.index_dtype == "int64" and h.x.nbytes > (2**31) * 12


def test_formats_not_read_in_blocks(tmp_path, data_dir):
    p = tmp_path / "t.csv"
    pd.DataFrame(np.arange(12).reshape(3, 4), index=["g1", "g2", "g3"],
                 columns=[f"c{i}" for i in range(4)]).to_csv(p)
    unit = build_bundle([p]).units[0]
    assert not streamed.choose([unit]).streamed  # read into memory, as before
    with pytest.raises(IngestCommitError, match="cannot be read in blocks"):
        _commit(data_dir, [p], streamed="on")


def test_streamed_memory_is_a_block_not_the_matrix(tmp_path, monkeypatch):
    # Python-side peak (numpy reports to tracemalloc) of the streamed write against the
    # in-memory read of the same 10x h5: blocks of 20k values, not the 1.2M-value matrix.
    from scview.core.ingestion import load_unit

    monkeypatch.setattr(streamed, "BLOCK_NNZ", 20_000)
    names, ids, types = _features(400, dup=False)
    f = _tenx_v3(tmp_path / "big.h5", _counts(6000, 400, seed=1, density=0.5), names, ids, types)
    unit = build_bundle([f]).units[0]
    tracemalloc.start()
    load_unit(unit)
    whole = tracemalloc.get_traced_memory()[1]
    tracemalloc.stop()
    tracemalloc.start()
    su = streamed.open_unit(unit, tmp_path)
    streamed.ingest([su], tmp_path / "x.part.h5", join=None, sample_label="sample", reconciliation=None)
    peak = tracemalloc.get_traced_memory()[1]
    tracemalloc.stop()
    assert peak < whole / 4, (peak, whole)


def test_unreadable_matrix_is_a_commit_error_and_leaves_nothing(tmp_path, data_dir):
    names, ids, types = _features(10)
    files = _mex(tmp_path / "mex", _counts(20, 10), names, ids, types, gz=False)
    text = files[0].read_text().splitlines()
    text.append("99 1 3")  # outside the size line (the count is wrong too)
    files[0].write_text("\n".join(text) + "\n")
    with pytest.raises(IngestCommitError, match="outside its size line"):
        _commit(data_dir, files, streamed="on")
    assert not any((data_dir / "ingested").glob("*/*.h5ad")) if (data_dir / "ingested").exists() else True
    assert not list(data_dir.rglob("*.part.h5"))


def test_a_failed_write_is_a_commit_error_and_leaves_no_part_file(tmp_path, data_dir, monkeypatch):
    from scview.core.ingestion import feature_split

    def full(*a, **k):
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(feature_split, "_append", full)
    names, ids, types = _features(10)
    f = _tenx_v3(tmp_path / "f.h5", _counts(20, 10), names, ids, types)
    with pytest.raises(IngestCommitError, match="No space left"):
        _commit(data_dir, [f], streamed="on")
    assert not list(data_dir.rglob("*.part.h5*"))
    assert not (data_dir / "ingested").exists() or not any((data_dir / "ingested").iterdir())


def test_not_enough_disk_is_said_before_writing(tmp_path, data_dir, monkeypatch):
    names, ids, types = _features(10)
    f = _tenx_v3(tmp_path / "f.h5", _counts(20, 10), names, ids, types)
    monkeypatch.setattr(shutil, "disk_usage", lambda p: shutil._ntuple_diskusage(100, 100, 10))
    with pytest.raises(IngestCommitError, match="of disk"):
        _commit(data_dir, [f], streamed="on")


def test_originals_are_linked_not_copied(tmp_path, data_dir):
    names, ids, types = _features(10)
    f = _tenx_v3(tmp_path / "f.h5", _counts(20, 10), names, ids, types)
    out = _commit(data_dir, [f], streamed="on")
    did = out.parent.name
    kept = data_dir / "uploads" / did / "f.h5"
    assert kept.read_bytes() == f.read_bytes()
    assert kept.stat().st_nlink == 2  # the staged file (removed with its session) and this
