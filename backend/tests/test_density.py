"""Expression-weighted kernel density on an embedding (core/density.py, C087) and
``GET /expression/density``."""

from __future__ import annotations

import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pyarrow as pa
import pytest
import scipy.sparse as sp
from scipy.stats import norm

from scview.core import density
from tests.large_fixture import counts_adata, register, small_twin

API = "/api/v1"


# ---------------------------------------------------------------------------
# Bandwidth
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("x, expected", [
    # ks::hpi 1.15.3 (= KernSmooth::dpik 2.23) in R 4.4.0, to 10 decimals
    (norm.ppf((np.arange(1, 501) - 0.5) / 500), 0.3282817066),
    ((np.arange(1, 301) ** 1.5) / 100, 3.2317876361),
    (np.concatenate([norm.ppf((np.arange(1, 201) - 0.5) / 200),
                     norm.ppf((np.arange(1, 101) - 0.5) / 100) + 6]), 0.4603546452),
])
def test_bandwidth_is_ks_hpi(x, expected):
    assert density.hpi(x) == pytest.approx(expected, rel=1e-6)


def test_bandwidth_of_constant_data_is_tiny_not_nan():
    assert 0 < density.hpi(np.ones(50)) < 1e-3


def test_linbin_keeps_the_weight_and_the_mean():
    x = np.array([0.0, 0.25, 1.0, 0.6])
    c = density.linbin(x, 0.0, 1.0, 5)
    assert c.sum() == pytest.approx(4.0)
    assert np.dot(c, np.linspace(0, 1, 5)) == pytest.approx(x.sum())


# ---------------------------------------------------------------------------
# Weighted density
# ---------------------------------------------------------------------------


def _blobs(n=1500, seed=0):
    rng = np.random.default_rng(seed)
    xy = np.concatenate([rng.normal(0, 1, (n // 2, 2)), rng.normal([6, 3], 1, (n - n // 2, 2))])
    return xy, rng


def _exact(xy, w, h):
    """The weighted KDE summed directly at every cell (no lattice)."""
    dx = (xy[:, None, 0] - xy[None, :, 0]) / h[0]
    dy = (xy[:, None, 1] - xy[None, :, 1]) / h[1]
    K = np.exp(-0.5 * (dx * dx + dy * dy)) / (2 * np.pi * h[0] * h[1])
    return K @ w / w.sum()


def test_binned_density_matches_the_exact_sum():
    xy, rng = _blobs()
    w = np.where(np.arange(len(xy)) < 750, rng.gamma(2.0, 1.0, len(xy)), 0.0)
    lat = density.Lattice(xy)
    got = lat.density(w)
    exact = _exact(xy, w, lat.h)
    assert np.max(np.abs(got - exact)) / exact.max() < 0.005
    assert np.corrcoef(got, exact)[0, 1] > 0.9999


def test_bandwidth_is_per_axis_hpi_times_adjust():
    xy, _ = _blobs()
    lat = density.Lattice(xy, adjust=2.0)
    assert lat.h == pytest.approx([2 * density.hpi(xy[:, 0]), 2 * density.hpi(xy[:, 1])])


def test_weights_move_the_density_to_the_expressing_cells():
    xy, _ = _blobs()
    first = np.arange(len(xy)) < 750
    d, facts = density.weighted_density(xy, first.astype(float))
    assert np.median(d[first]) > np.median(d[~first]) * 100
    assert facts["n_expressing"] == [750]
    # the scale of the weights does not matter, only their shares
    d10, _ = density.weighted_density(xy, first * 10.0)
    assert np.allclose(d, d10, rtol=1e-5)
    # equal weights everywhere: the cells' own (unweighted) density
    flat, _ = density.weighted_density(xy, np.ones(len(xy)))
    assert np.corrcoef(flat, _exact(xy, np.ones(len(xy)), density.Lattice(xy).h))[0, 1] > 0.9999


def test_density_integrates_to_about_one():
    xy, rng = _blobs()
    lat = density.Lattice(xy)
    w = rng.gamma(1.0, 1.0, len(xy))
    g = lat.grid
    binned = np.zeros(g * g)
    for idx, wt in zip(lat._idx, lat._wt):
        binned += np.bincount(idx, weights=w * wt, minlength=g * g)
    z = lat.kernels[0] @ binned.reshape(g, g) @ lat.kernels[1].T / w.sum()
    cell = (lat.axes[0][1] - lat.axes[0][0]) * (lat.axes[1][1] - lat.axes[1][0])
    assert 0.9 < z.sum() * cell <= 1.0  # some kernel mass falls outside the cells' range


def test_joint_density_is_the_product_of_peak_scaled_densities():
    xy, rng = _blobs()
    W = rng.gamma(1.0, 1.0, (len(xy), 3)) * (rng.random((len(xy), 3)) < 0.4)
    joint, facts = density.weighted_density(xy, W)
    each = [density.weighted_density(xy, W[:, j])[0].astype(np.float64) for j in range(3)]
    expected = np.prod([d / d.max() for d in each], axis=0)
    assert np.allclose(joint, expected, rtol=1e-4, atol=1e-7)
    assert joint.max() <= 1.0 and joint.dtype == np.float32
    assert facts["n_expressing"] == [int((W[:, j] > 0).sum()) for j in range(3)]


def test_no_expression_gives_zero_and_bad_values_are_handled():
    xy, _ = _blobs(200)
    zero, _ = density.weighted_density(xy, np.zeros(len(xy)))
    assert not zero.any()
    xy[3] = np.nan
    w = np.ones(len(xy))
    w[5] = -2.0
    d, facts = density.weighted_density(xy, w)
    assert d[3] == 0 and facts["n_no_coords"] == 1 and facts["n_negative"] == 1
    assert np.isfinite(d).all()
    with pytest.raises(density.DensityError):
        density.weighted_density(np.full((5, 2), np.nan), np.ones(5))


def test_kang_scale_is_fast():
    """100k cells, three genes: well under a second on two cores."""
    import time

    rng = np.random.default_rng(1)
    xy = rng.normal(size=(100_000, 2)) * [5, 3]
    W = rng.gamma(0.5, 1.0, (100_000, 3)) * (rng.random((100_000, 3)) < 0.2)
    t = time.perf_counter()
    density.weighted_density(xy, W)
    assert time.perf_counter() - t < 2.0


# ---------------------------------------------------------------------------
# Endpoint
# ---------------------------------------------------------------------------


def _register(dm, did: str = "dens1") -> tuple[str, ad.AnnData]:
    """300 cells in two blobs; X (2 genes) differs from .raw (3 genes)."""
    d = Path(dm.data_dir) / "uploads" / did
    d.mkdir(parents=True, exist_ok=True)
    (d / "metadata.json").write_text(json.dumps({"id": did, "name": did, "status": "ready"}))
    rng = np.random.default_rng(0)
    n = 300
    raw = rng.poisson(1.0, size=(n, 3)).astype(np.float32)
    raw[:150, 0] += 4  # CD3E high in the first blob
    full = ad.AnnData(sp.csr_matrix(np.log1p(raw)))
    full.var_names = ["CD3E", "MS4A1", "LYZ"]
    a = ad.AnnData(np.zeros((n, 2), dtype=np.float32))
    a.var_names = ["CD3E", "MS4A1"]
    a.obs_names = [f"c{i}" for i in range(n)]
    a.raw = full
    a.obs["cell_type"] = pd.Categorical(["T"] * 150 + ["B"] * 150)
    xy = np.concatenate([rng.normal(0, 1, (150, 2)), rng.normal(5, 1, (150, 2))])
    a.obsm["X_umap"] = xy.astype(np.float32)
    a.obsm["X_pca"] = np.column_stack([rng.normal(size=n), xy[:, 1], xy[:, 0]]).astype(np.float32)
    a.write_h5ad(d / "data.h5ad")
    return did, a


def _decode(content: bytes) -> tuple[np.ndarray, dict]:
    table = pa.ipc.open_stream(content).read_all()
    assert table.column_names == ["density"]
    return table.column(0).to_numpy(), json.loads(table.schema.metadata[b"scview"])


async def test_endpoint_returns_a_density_per_cell(client, dataset_manager):
    did, a = _register(dataset_manager)
    url = f"{API}/datasets/{did}/expression/density"
    r = await client.get(url, params={"genes": "CD3E", "embedding": "X_umap"})
    assert r.status_code == 200, r.text
    assert r.headers["content-type"].startswith("application/vnd.apache.arrow.stream")
    values, meta = _decode(r.content)
    assert values.dtype == np.float32 and values.size == 300
    assert meta["genes"] == ["CD3E"] and meta["layer"] == "raw" and meta["joint"] is False
    assert meta["embedding"] == "X_umap" and meta["dims"] == [0, 1] and meta["n_cells"] == 300
    # from the default layer (.raw), on the embedding
    expected, _ = density.weighted_density(a.obsm["X_umap"], a.raw.X.toarray()[:, 0])
    assert np.allclose(values, expected)
    assert values[:150].mean() > values[150:].mean()

    # joint, a missing gene listed, dims picking PCA columns 2 and 1 (= the UMAP's x, y)
    r = await client.get(url, params={"genes": "CD3E,NOPE,LYZ", "embedding": "X_pca", "dims": "2,1"})
    values, meta = _decode(r.content)
    assert meta["genes"] == ["CD3E", "LYZ"] and meta["missing_genes"] == ["NOPE"] and meta["joint"]
    expected, _ = density.weighted_density(a.obsm["X_umap"], a.raw.X.toarray()[:, [0, 2]])
    assert np.allclose(values, expected, rtol=1e-4, atol=1e-6)
    assert values.max() <= 1.0

    # bandwidth multiplier
    _, wide = _decode((await client.get(url, params={"genes": "CD3E", "adjust": 2})).content)
    assert wide["bandwidth"] == pytest.approx([2 * b for b in meta["bandwidth"]], rel=1e-3)


async def test_endpoint_errors(client, dataset_manager):
    did, _ = _register(dataset_manager)
    url = f"{API}/datasets/{did}/expression/density"
    r = await client.get(url, params={"genes": "NOPE"})
    assert r.status_code == 404 and "NOPE" in r.json()["detail"]
    r = await client.get(url, params={"genes": ",".join(f"G{i}" for i in range(10))})
    assert r.status_code == 400
    r = await client.get(url, params={"genes": " , "})
    assert r.status_code == 400
    r = await client.get(url, params={"genes": "CD3E", "embedding": "X_nope"})
    assert r.status_code == 404
    r = await client.get(url, params={"genes": "CD3E", "embedding": "X_pca", "dims": "1,1"})
    assert r.status_code == 400
    r = await client.get(url, params={"genes": "CD3E", "adjust": 10})
    assert r.status_code == 422
    r = await client.get(f"{API}/datasets/nope/expression/density", params={"genes": "CD3E"})
    assert r.status_code == 404


async def test_large_dataset_reads_the_gene_store(client, tmp_data_dir):
    """A large dataset (genes from its gene store) gives what its in-memory twin gives."""
    counts = counts_adata()
    register(tmp_data_dir, "bigD", counts)
    up = tmp_data_dir / "uploads" / "smallD"
    up.mkdir(parents=True)
    small_twin(counts).write_h5ad(up / "twin.h5ad")
    (up / "metadata.json").write_text(json.dumps({"id": "smallD", "name": "twin", "status": "ready"}))
    params = {"genes": "G10,G25", "embedding": "X_umap", "layer": "X"}
    a = await client.get(f"{API}/datasets/bigD/expression/density", params=params)
    b = await client.get(f"{API}/datasets/smallD/expression/density", params=params)
    assert a.status_code == 200, a.text
    assert b.status_code == 200, b.text
    va, ma = _decode(a.content)
    vb, mb = _decode(b.content)
    assert ma == mb
    assert np.allclose(va, vb, rtol=1e-5, atol=1e-7)
