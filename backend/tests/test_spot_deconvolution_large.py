"""Spot composition with a large reference and/or large spots (core/large_mode.py): the
counts read by row blocks from the sidecar give the in-memory path's inputs and fit.

Exact (bitwise): cell totals, S, cells per type, the spots' Y and N, the self-check's
synthetic spots. mu to relative 1e-12: a large reference's per-type sums are added
block by block (each block's cells, then the blocks), the in-memory product adds every
cell in one run, so the floating-point additions come in another order."""

from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp

from scview.core import large_mode
from scview.core import spot_deconvolution as sd
from scview.core.anndata_adaptor import AnnDataAdaptor
from tests.large_fixture import make_large
from tests.test_spot_deconvolution import _fit, _prepare, _profiles, _reference, _spots, _thin_rows

MU_RTOL = 1e-12
SHARE_ATOL = 1e-6


@pytest.fixture()
def small_blocks(monkeypatch):
    """Reads capped low (a block of ~5,000 values, ~15 blocks of the reference) and
    counted: the large path must work within the caps, in many reads, never whole."""
    monkeypatch.setattr(large_mode, "MAX_READ_BYTES", 32 * 5000)
    reads: list[int] = []
    orig = large_mode.OnDiskMatrix.__getitem__

    def counted(self, index):
        out = orig(self, index)
        reads.append(int(out.nnz))
        return out

    monkeypatch.setattr(large_mode.OnDiskMatrix, "__getitem__", counted)
    return reads


def _same_inputs(a: sd.Prepared, b: sd.Prepared) -> None:
    assert a.types == b.types and a.genes == b.genes and a.ref_genes == b.ref_genes
    np.testing.assert_array_equal(a.keep, b.keep)
    np.testing.assert_allclose(a.mu, b.mu, rtol=MU_RTOL, atol=0)
    np.testing.assert_array_equal(a.S, b.S)
    np.testing.assert_array_equal(a.n_type_cells, b.n_type_cells)
    assert (a.Y != b.Y).nnz == 0 and a.Y.dtype == b.Y.dtype
    np.testing.assert_array_equal(a.N, b.N)
    np.testing.assert_array_equal(a.libraries, b.libraries)


def _same_fit(a: dict, b: dict) -> None:
    assert a["s_e"] == b["s_e"]
    np.testing.assert_array_equal(np.isnan(a["beta"]), np.isnan(b["beta"]))
    np.testing.assert_allclose(a["beta"], b["beta"], atol=SHARE_ATOL, equal_nan=True)
    np.testing.assert_allclose(a["cells"], b["cells"], atol=SHARE_ATOL, equal_nan=True)


@pytest.fixture(scope="module")
def data():
    prof = _profiles()
    ref = _reference(prof, cells=80)
    keep = np.ones(ref.n_obs)
    keep[3] = 0  # a reference cell with no counts: left out on both paths
    X = sp.csr_matrix(sp.diags(keep.astype(np.float32)) @ ref.X, dtype=np.float32)
    X.eliminate_zeros()
    ref.X = X
    spots, _ = _spots(prof, n=120, libraries=2)
    return ref, spots


def test_a_large_reference_gives_the_in_memory_profiles_and_fit(tmp_path, data, small_blocks):
    ref, spots = data
    h5ad = make_large(tmp_path / "ref", ref)
    with sd.large_reference(str(h5ad)) as (skel, counts):
        assert skel.obs.shape[0] == ref.n_obs and "X" not in skel.layers and skel.X is None
        large = _prepare(skel, spots, ref_counts=counts)
    mem = _prepare(ref, spots)
    _same_inputs(large, mem)
    assert large.info["reference_source"] == "the counts file"
    assert large.info["n_reference_cells"] == mem.info["n_reference_cells"] == ref.n_obs - 1
    assert len(small_blocks) > 10 and max(small_blocks) * 8 <= large_mode.MAX_READ_BYTES
    for method in sd.METHODS:
        _same_fit(_fit(large, method), _fit(mem, method))


def test_a_large_reference_is_never_read_whole(tmp_path, data, monkeypatch):
    ref, spots = data
    h5ad = make_large(tmp_path / "ref", ref)
    monkeypatch.setattr(large_mode, "MAX_READ_BYTES", 8 * 2000)  # ~8 cells per read
    with sd.large_reference(str(h5ad)) as (skel, (M, _var, _src)):
        with pytest.raises(large_mode.LargeDatasetError):
            M[:, :]  # the whole matrix: refused
        with pytest.raises(large_mode.LargeDatasetError):
            M[:, [0, 1]]  # columns of the counts: refused
        p = _prepare(skel, spots, ref_counts=(M, _var, _src))  # row blocks within the cap
    assert p.mu.shape == (4, 400)


def test_large_spots_give_the_in_memory_counts_and_fit(tmp_path, data, small_blocks):
    ref, spots = data
    h5ad = make_large(tmp_path / "spots", spots)
    adaptor = AnnDataAdaptor(str(h5ad))
    try:
        assert adaptor.is_large
        large = _prepare(ref, adaptor.adata)
        assert large.info["spot_source"] == "the counts file" and len(small_blocks) > 1
    finally:
        adaptor.close()
    mem = _prepare(ref, spots)
    _same_inputs(large, mem)
    np.testing.assert_array_equal(large.mu, mem.mu)  # the same reference: bitwise
    a, b = _fit(large), _fit(mem)
    np.testing.assert_array_equal(a["beta"], b["beta"])  # the same inputs: the same fit


def test_both_sides_large(tmp_path, data, small_blocks):
    ref, spots = data
    ref_h5 = make_large(tmp_path / "ref", ref)
    spot_h5 = make_large(tmp_path / "spots", spots)
    adaptor = AnnDataAdaptor(str(spot_h5))
    try:
        with sd.large_reference(str(ref_h5)) as (skel, counts):
            large = _prepare(skel, adaptor.adata, ref_counts=counts)
    finally:
        adaptor.close()
    mem = _prepare(ref, spots)
    _same_inputs(large, mem)
    _same_fit(_fit(large), _fit(mem))


def test_the_self_check_on_a_large_reference_draws_the_same_spots(tmp_path, data, small_blocks, monkeypatch):
    monkeypatch.setattr(sd, "CHECK_SPOTS", 120)
    monkeypatch.setattr(sd, "CHECK_DEPTHS", (6000, 1500))
    ref, spots = data
    h5ad = make_large(tmp_path / "ref", ref)
    with sd.large_reference(str(h5ad)) as (skel, counts):
        large = _prepare(skel, spots, ref_counts=counts, check=True, seed=7)
    mem = _prepare(ref, spots, check=True, seed=7)
    a, b = large.check, mem.check
    np.testing.assert_allclose(a["mu"], b["mu"], rtol=MU_RTOL, atol=0)
    np.testing.assert_array_equal(a["S"], b["S"])
    np.testing.assert_array_equal(a["n"], b["n"])
    np.testing.assert_array_equal(a["truth_rna"], b["truth_rna"])
    np.testing.assert_array_equal(a["truth_cells"], b["truth_cells"])
    for x, y in zip(a["sets"], b["sets"], strict=True):
        assert (x["Y"] != y["Y"]).nnz == 0  # the drawn cells summed and thinned identically
        np.testing.assert_array_equal(x["N"], y["N"])
    ra = [e for e in sd.iter_check(large) if e["type"] == "checked"][0]["out"]["runs"]
    rb = [e for e in sd.iter_check(mem) if e["type"] == "checked"][0]["out"]["runs"]
    for x, y in zip(ra, rb, strict=True):
        assert x["method"] == y["method"] and x["s_e"] == y["s_e"]
        assert x["rmse"] == pytest.approx(y["rmse"], abs=SHARE_ATOL)


def test_a_reference_sidecar_of_other_cells_is_refused(tmp_path, data):
    ref, _spots_ = data
    h5ad = make_large(tmp_path / "ref", ref)
    other = make_large(tmp_path / "other", ref[::2].copy())
    counts = h5ad.with_name(h5ad.stem + ".counts.h5")
    counts.unlink()
    other.with_name(other.stem + ".counts.h5").rename(counts)
    with pytest.raises(sd.SpotDeconvolutionError, match="other cells"):
        with sd.large_reference(str(h5ad)):
            pass


def test_the_estimate_of_a_large_side_counts_a_block_not_the_matrix():
    from scview.core import preflight

    def header(large: bool, n: int, nnz: int) -> preflight.Header:
        return preflight.Header("h5ad", n, 30000, [preflight.Matrix("X", n, 30000, True, nnz)], large=large,
                                obs_columns=10)

    atlas = 2 * 10**9  # 1M cells: 16 GB of counts
    spots = header(False, 4000, 4000 * 5000)
    in_memory = sd.need_bytes(header(False, 10**6, atlas), spots, ref_open=False)
    large = sd.need_bytes(header(True, 10**6, atlas), spots, ref_open=False)
    assert in_memory > 2.5 * atlas * 8 and large < 2 * 2**30, (in_memory, large)
    # A block of the pass, not the counts: twice the stored values cost the same.
    assert sd.need_bytes(header(True, 10**6, 2 * atlas), spots, ref_open=False) == large
    # The self-check holds the drawn cells' rows (10,000 cells of 2,000 values).
    check = sd.need_bytes(header(True, 10**6, atlas), spots, ref_open=False, check=True)
    assert check - large == 2 * 12 * 2000 * 10_000
    # Large spots: their shared genes kept (about their counts, twice while stacked) and a
    # block, less than the in-memory path's copies of the same counts.
    ref = header(False, 1000, 10**6)
    big_spots = sd.need_bytes(ref, header(True, 40_000, 40_000 * 5000), ref_open=True)
    assert big_spots < sd.need_bytes(ref, header(False, 40_000, 40_000 * 5000), ref_open=True)
    # Seconds: a pass over the stored values (two for the check), not cells read in.
    secs = sd.estimate_seconds(10**6, 3000, 20, ref_values=atlas)
    assert secs == pytest.approx(sd.SECONDS_FIXED + sd.LARGE_SECONDS_PER_VALUE * atlas
                                 + sd.SECONDS_PER_SPOT_TYPE2 * 3000 * 400, abs=0.1)
    assert sd.estimate_seconds(10**6, 3000, 20, ref_values=atlas, check=True) > secs
    # Large spots beside a reference in memory: the reference still read in, the spots passed.
    mem = sd.estimate_seconds(10**5, 3000, 20)
    assert sd.estimate_seconds(10**5, 3000, 20, spot_values=10**8) == pytest.approx(
        mem + sd.LARGE_SECONDS_PER_VALUE * 10**8, abs=0.1)


# ---------------------------------------------------------------------------
# The endpoints: a large dataset on either side runs, and saves like one in memory


API = "/api/v1"


def _api_pair():
    prof = _profiles()
    ref = _reference(prof)
    spots, _ = _spots(prof, n=100)
    spots.obs["leiden"] = np.where(np.arange(spots.n_obs) < 50, "0", "1")
    spots.uns["scview_active_clustering"] = "leiden"
    return ref, spots


async def _run(client, did: str, ref: str, path: str = "run-stream") -> dict:
    from tests.test_hvg_sensitivity import _events

    r = await client.post(f"{API}/datasets/{did}/spot-deconvolution/{path}",
                          json={"reference_id": ref, "celltype_column": "cell_type"})
    assert r.status_code == 200, r.text
    events = _events(r.text)
    assert events[-2][0] == "result" and events[-1][0] == "complete", events[-1]
    return events[-2][1]


async def test_large_spots_run_and_save_in_place(client, dataset_manager, tmp_data_dir):
    from tests.large_fixture import register, small_twin
    from tests.test_hvg_sensitivity import _register

    ref, spots = _api_pair()
    rid = _register(dataset_manager, ref, "lref")
    h5ad = register(tmp_data_dir, "lspots", spots)
    mid = _register(dataset_manager, small_twin(spots), "mspots")
    info = (await client.get(f"{API}/datasets/lspots/spot-deconvolution")).json()
    assert info["available"] is True and info["reason"] is None and info["n_spots"] == 100
    chk = (await client.get(f"{API}/datasets/lspots/spot-deconvolution/reference/{rid}")).json()
    assert chk["ok"] is True and chk["shared_genes_by_name"] == 400 and chk["estimate"]["peak_bytes"] > 0
    big, small = await _run(client, "lspots", rid), await _run(client, mid, rid)
    assert big["types"] == small["types"] and big["s_e"] == small["s_e"] and big["n_fitted"] == small["n_fitted"]

    # Written into the large dataset's own h5ad (X untouched, still large), reopened.
    dataset_manager.evict("lspots")
    adaptor = await dataset_manager.get_or_load_dataset("lspots")
    assert adaptor.is_large and adaptor.h5ad_path == str(h5ad)
    twin = (await dataset_manager.get_or_load_dataset(mid)).adata
    for key in (sd.OBSM_RNA, sd.OBSM_CELLS):
        np.testing.assert_array_equal(np.asarray(adaptor.adata.obsm[key]), np.asarray(twin.obsm[key]))
    assert list(adaptor.adata.obs[sd.OBS_TOP]) == list(twin.obs[sd.OBS_TOP])
    assert sd.saved(adaptor.adata)["result_id"] == big["result_id"]
    r = await client.get(f"{API}/datasets/lspots/spot-deconvolution/share", params={"type": "B"})
    assert r.status_code == 200 and len(r.json()["values"]) == 100


async def test_a_large_reference_runs_and_checks(client, dataset_manager, tmp_data_dir, monkeypatch):
    from tests.large_fixture import register
    from tests.test_hvg_sensitivity import _register

    monkeypatch.setattr(sd, "CHECK_SPOTS", 80)
    monkeypatch.setattr(sd, "CHECK_DEPTHS", (5000,))
    ref, spots = _api_pair()
    register(tmp_data_dir, "bigref", ref)
    mref = _register(dataset_manager, ref, "memref")
    did = _register(dataset_manager, spots, "spotsr")
    chk = (await client.get(f"{API}/datasets/{did}/spot-deconvolution/reference/bigref")).json()
    assert chk["ok"] is True and chk["n_cells"] == 240 and chk["celltype_column"] == "cell_type"
    assert chk["estimate"]["large_reference"] is True and chk["estimate"]["peak_bytes"] > 0
    big = await _run(client, did, "bigref")
    assert big["reference"]["id"] == "bigref"
    adata = (await dataset_manager.get_or_load_dataset(did)).adata
    big_shares = np.asarray(adata.obsm[sd.OBSM_RNA]).copy()
    small = await _run(client, did, mref)
    adata = (await dataset_manager.get_or_load_dataset(did)).adata
    assert big["s_e"] == small["s_e"]
    np.testing.assert_allclose(big_shares, np.asarray(adata.obsm[sd.OBSM_RNA]), atol=SHARE_ATOL, equal_nan=True)
    # The self-check on the large reference: the same runs as on the one in memory.
    a = await _run(client, did, "bigref", "check-stream")
    b = await _run(client, did, mref, "check-stream")
    for x, y in zip(a["runs"], b["runs"], strict=True):
        assert x["method"] == y["method"] and x["rmse"] == pytest.approx(y["rmse"], abs=SHARE_ATOL)


def test_large_spots_honour_the_floor_as_the_in_memory_path_does(tmp_path, data, small_blocks):
    ref, spots = data
    spots = spots.copy()
    _thin_rows(spots, range(6))  # about 55 counts: under 100, over 40
    h5ad = make_large(tmp_path / "spots", spots)
    adaptor = AnnDataAdaptor(str(h5ad))
    try:
        assert adaptor.is_large
        large_100, large_40 = _prepare(ref, adaptor.adata), _prepare(ref, adaptor.adata, min_umi=40)
        counts = sd.counts_below(adaptor.adata, [])
        assert len(small_blocks) > 1  # read by row blocks, never whole
    finally:
        adaptor.close()
    mem_100, mem_40 = _prepare(ref, spots), _prepare(ref, spots, min_umi=40)
    assert large_100.min_umi == mem_100.min_umi == 100 and large_40.min_umi == mem_40.min_umi == 40
    a, b = _fit(large_100, "nnls"), _fit(mem_100, "nnls")
    np.testing.assert_array_equal(a["fitted"], b["fitted"])
    assert not a["fitted"][:6].any() and a["fitted"][6:].all()
    a, b = _fit(large_40, "nnls"), _fit(mem_40, "nnls")
    np.testing.assert_array_equal(a["fitted"], b["fitted"])
    assert a["fitted"].all()
    assert counts == sd.counts_below(spots, [])
    assert counts["below"][71] == 6 and counts["below"][100] == 6 and counts["below"][40] == 0
