"""DSB protein normalisation (core/dsb.py): the empty-droplet background, the mixture
fits against mclust, steps I and II, the dataset run, attach, endpoints and replay."""

from __future__ import annotations

import gzip
import json
import math
import subprocess
import sys
from pathlib import Path

import anndata as ad
import h5py
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

from scview.core import dsb, protein, provenance, replay_script, results_export, seq_metrics
from scview.core.anndata_adaptor import AnnDataAdaptor
from scview.core.claims import resolve_tag

API = "/api/v1"
N_GENES = 30
AB = ["CD3_TotalSeqB", "CD4_TotalSeqB", "CD19_TotalSeqB", "CD14_TotalSeqB", "IgG1_control_TotalSeqB",
      "IgG2a_control_TotalSeqB"]
SHOWN = [protein.display_name(f) for f in AB]  # CD3, ..., IgG1_control, IgG2a_control
ISO = ["IgG1_control", "IgG2a_control"]
AMBIENT = np.array([3.0, 5.0, 2.0, 4.0, 1.5, 1.0])  # mean ADT counts per empty droplet


def _bc(i: int) -> str:
    seq = "".join("ACGT"[(i >> (2 * k)) & 3] for k in range(12))
    return f"AAAA{seq}-1"


def _droplets(n_cells=300, n_empty=3000, seed=0):
    """Raw counts (features x barcodes): cells with three staining types and a per-cell
    capture factor; empty droplets with low RNA and ambient antibody counts."""
    rng = np.random.default_rng(seed)
    n = n_cells + n_empty
    dense = np.zeros((N_GENES + len(AB), n), dtype=np.int64)
    depth = rng.lognormal(0.0, 0.3, size=n_cells)  # technical factor per cell
    dense[:N_GENES, :n_cells] = rng.poisson(60 * depth, size=(N_GENES, n_cells))
    kind = np.arange(n_cells) % 3  # T, B, mono
    stain = np.zeros((len(AB), n_cells))
    stain[0, kind == 0] = 300
    stain[1, kind == 0] = 150
    stain[2, kind == 1] = 400
    stain[3, kind == 2] = 250
    dense[N_GENES:, :n_cells] = rng.poisson((AMBIENT[:, None] * 4 + stain) * depth)
    # Empty droplets: a few RNA UMIs, protein from free antibody (some above the window)
    dense[:N_GENES, n_cells:] = rng.poisson(0.5, size=(N_GENES, n_empty))
    scale = rng.choice([4.0, 12.0, 60.0], size=n_empty, p=[0.2, 0.6, 0.2])
    dense[N_GENES:, n_cells:] = rng.poisson(AMBIENT[:, None] * scale)
    return dense, [_bc(i) for i in range(n)], n_cells


def _write_h5(path: Path, dense, barcodes, antibodies=True) -> Path:
    csc = sp.csc_matrix(dense)
    names = [f"GENE{i}" for i in range(N_GENES)] + AB
    types = ["Gene Expression"] * N_GENES + ["Antibody Capture" if antibodies else "Gene Expression"] * len(AB)
    with h5py.File(path, "w") as f:
        m = f.create_group("matrix")
        m.create_dataset("data", data=csc.data.astype(np.int32))
        m.create_dataset("indices", data=csc.indices.astype(np.int64))
        m.create_dataset("indptr", data=csc.indptr.astype(np.int64))
        m.create_dataset("shape", data=np.array(dense.shape, dtype=np.int32))
        m.create_dataset("barcodes", data=np.array(barcodes, dtype="S"))
        ft = m.create_group("features")
        ft.create_dataset("id", data=np.array([f"ID{i}" for i in range(len(names))], dtype="S"))
        ft.create_dataset("name", data=np.array(names, dtype="S"))
        ft.create_dataset("feature_type", data=np.array(types, dtype="S"))
    return path


def _write_mex(d: Path, dense, barcodes) -> tuple[Path, Path, Path]:
    rows, cols = np.nonzero(dense)
    mtx = d / "matrix.mtx.gz"
    with gzip.open(mtx, "wt") as fh:
        fh.write("%%MatrixMarket matrix coordinate integer general\n")
        fh.write(f"{dense.shape[0]} {dense.shape[1]} {len(rows)}\n")
        for r, c in zip(rows, cols):
            fh.write(f"{r + 1} {c + 1} {dense[r, c]}\n")
    bcs = d / "barcodes.tsv.gz"
    with gzip.open(bcs, "wt") as fh:
        fh.write("".join(b + "\n" for b in barcodes))
    feats = d / "features.tsv.gz"
    with gzip.open(feats, "wt") as fh:
        for i in range(N_GENES):
            fh.write(f"ID{i}\tGENE{i}\tGene Expression\n")
        for j, a in enumerate(AB):
            fh.write(f"IDA{j}\t{a}\tAntibody Capture\n")
    return mtx, bcs, feats


def _direct_stats(dense, n_cells, rna_window=dsb.RNA_WINDOW, prot_window=dsb.PROT_WINDOW):
    """The window's droplets and their log(x + 10) mean / SD, straight from the matrix."""
    rna = dense[:N_GENES].sum(0).astype(float)
    prot = dense[N_GENES:].sum(0).astype(float)
    with np.errstate(divide="ignore"):
        lr, lp = np.log10(rna), np.log10(prot)
    sel = ((lr >= rna_window[0]) & (lr < rna_window[1]) & (lp >= prot_window[0]) & (lp < prot_window[1]))
    sel[:n_cells] = False
    x = np.log(dense[N_GENES:, sel].T + 10.0)
    return int(sel.sum()), x.mean(0), x.std(0, ddof=1)


def _cells_adata(dense, barcodes, n_cells) -> ad.AnnData:
    a = ad.AnnData(X=sp.csr_matrix(dense[:N_GENES, :n_cells].T.astype(np.float32)))
    a.obs_names = barcodes[:n_cells]
    a.var_names = [f"GENE{i}" for i in range(N_GENES)]
    split = protein.ProteinSplit()
    protein.attach(a, dense[N_GENES:, :n_cells].T, AB, split)
    return a


# ---------------------------------------------------------------------------
# The empty-droplet background
# ---------------------------------------------------------------------------


def test_background_grid_gives_the_window_stats_exactly(tmp_path):
    dense, bcs, n_cells = _droplets()
    raw = _write_h5(tmp_path / "raw.h5", dense, bcs)
    bg = dsb.background_from_h5(raw, bcs[:n_cells], nnz_chunk=5000)  # many blocks
    n, mean, sd = dsb.window_stats(bg, dsb.RNA_WINDOW, dsb.PROT_WINDOW)
    n0, mean0, sd0 = _direct_stats(dense, n_cells)
    assert n == n0 and n > 1000
    np.testing.assert_allclose(mean, mean0, atol=1e-10)
    np.testing.assert_allclose(sd, sd0, atol=1e-10)
    # The known ambient level: the window's droplets are the 12x and 60x ones
    assert list(bg["features"]) == AB
    # A wider protein window takes in all the 60x droplets, still exact
    n1, m1, s1 = dsb.window_stats(bg, (0.3, 2.5), (1.5, 3.5))
    n1d, m1d, s1d = _direct_stats(dense, n_cells, (0.3, 2.5), (1.5, 3.5))
    assert n1 == n1d > n
    np.testing.assert_allclose(m1, m1d, atol=1e-10)
    np.testing.assert_allclose(s1, s1d, atol=1e-10)
    # The cells were left out (all of them are on the grid: < 10^4 RNA UMIs)
    assert bg["n_cells_excluded"] == n_cells
    # The MEX triplet gives the same grid
    mex = _write_mex(tmp_path, dense, bcs)
    bg2 = dsb.background_from_mex(*mex, cells=bcs[:n_cells], rows=7000)
    for k in ("bins", "n"):
        np.testing.assert_array_equal(bg2[k], bg[k])
    np.testing.assert_allclose(bg2["sum"], bg["sum"], atol=1e-9)
    np.testing.assert_allclose(bg2["sumsq"], bg["sumsq"], atol=1e-9)


def test_cells_are_matched_by_barcode_sequence_too(tmp_path):
    dense, bcs, n_cells = _droplets(n_cells=60, n_empty=400)
    raw = _write_h5(tmp_path / "raw.h5", dense, bcs)
    merged_names = [f"{b[:-2]}:s1" for b in bcs[:n_cells]]  # "AAAA...:s1" without the -1
    bg = dsb.background_from_h5(raw, merged_names)
    assert bg["n_cells_excluded"] == n_cells


def test_matrix_without_antibodies_has_no_background(tmp_path):
    dense, bcs, n_cells = _droplets(n_cells=20, n_empty=100)
    raw = _write_h5(tmp_path / "raw.h5", dense, bcs, antibodies=False)
    assert dsb.background_from_h5(raw, bcs[:n_cells]) is None


def test_size_bins_keep_powers_of_ten_in_their_own_bin():
    b = dsb.size_bin(np.array([0, 1, 2, 9, 10, 31, 32, 999, 1000, 10**4]), 0.1, 40)
    assert b.tolist() == [-1, 0, 3, 9, 10, 14, 15, 29, 30, -1]


# ---------------------------------------------------------------------------
# Mixture fits: mclust Mclust(x, G = 2)$parameters$mean[1]
# ---------------------------------------------------------------------------

# R 4.6.1, mclust 6.1.3: Mclust(x, G = 2) and mclust:::qclass(x, 2)
MCLUST = [
    ([-0.4, 0.1, 0.3, -0.2, 0.0, 0.5, 2.9, 3.4, -0.1, 0.2, 4.1, 3.0, 0.4, -0.3, 2.5, 0.6, 0.05],
     0.095833344411, [1, 1, 2, 1, 1, 2, 2, 2, 1, 1, 2, 2, 2, 1, 2, 2, 1]),
    ([0, 0, 0, 0, 0, 1, 2, 5, 6, 0, 0.5], 0.388888667294, [1, 1, 1, 1, 1, 2, 2, 2, 2, 1, 1]),
    ([1, 1, 1, 2, 2, 2, 2, 9], 2.215620848439, [1, 1, 1, 2, 2, 2, 2, 2]),
]


@pytest.mark.parametrize("x,mean1,qclass", MCLUST)
def test_lower_means_match_mclust(x, mean1, qclass):
    assert dsb.qclass2(x).tolist() == qclass
    got, ok = dsb.lower_means(np.array([x]))
    assert ok[0]
    assert got[0] == pytest.approx(mean1, abs=1e-4)


def test_lower_means_recovers_the_background_of_each_row():
    rng = np.random.default_rng(3)
    rows = []
    for _ in range(200):
        rows.append(np.r_[rng.normal(0.0, 0.3, 12), rng.normal(4.0, 0.5, 6)])
    got, ok = dsb.lower_means(np.array(rows))
    assert ok.all()
    assert np.abs(got).max() < 0.4
    got, ok = dsb.lower_means(np.ones((2, 5)))  # constant rows: no fit, their value
    assert not ok.any() and np.allclose(got, 1.0)


def test_regress_out_is_limmas_remove_batch_effect():
    rng = np.random.default_rng(1)
    x = rng.normal(size=(50, 4))
    v = rng.normal(size=50)
    x += np.outer(v, [1.0, -2.0, 0.5, 0.0])
    out = dsb.regress_out(x, v)
    design = np.column_stack([np.ones(50), v - v.mean()])
    beta = np.linalg.lstsq(design, x, rcond=None)[0]
    np.testing.assert_allclose(out, x - np.outer(v - v.mean(), beta[1]), atol=1e-12)
    assert np.abs(np.corrcoef(out[:, 0], v)[0, 1]) < 1e-10


# ---------------------------------------------------------------------------
# Steps I and II, on a dataset
# ---------------------------------------------------------------------------


def _with_background(tmp_path, **kw):
    dense, bcs, n_cells = _droplets(**kw)
    a = _cells_adata(dense, bcs, n_cells)
    raw = _write_h5(tmp_path / "raw_feature_bc_matrix.h5", dense, bcs)
    rec = seq_metrics.build_record("s1", _files(raw), bcs[:n_cells])
    seq_metrics.store(a, [rec])
    return a, dense, bcs, n_cells, rec


def _files(raw):
    class F:
        def __init__(self, p):
            self.path, self.kind = str(p), "tenx_h5"
    return [F(raw)]


def test_attach_stores_the_background_and_remove_drops_it(tmp_path):
    dense, bcs, n_cells = _droplets()
    a = _cells_adata(dense, bcs, n_cells)
    raw = _write_h5(tmp_path / "raw_feature_bc_matrix.h5", dense, bcs)
    rec = seq_metrics.build_record("s1", _files(raw), bcs[:n_cells])
    assert rec["dsb"]["n_antibodies"] == len(AB)
    assert rec["dsb"]["n_droplets_default_window"] == _direct_stats(dense, n_cells)[0]
    seq_metrics.store(a, [rec])
    assert "s1" in dsb.read_backgrounds(a)
    assert "_dsb" not in json.loads(a.uns[seq_metrics.UNS_KEY])["samples"]["s1"]
    seq_metrics.remove(a, "s1")
    assert dsb.UNS_BACKGROUND not in a.uns


def test_step_one_and_two_on_the_dataset(tmp_path):
    dense, bcs, n_cells = _droplets()
    a = _cells_adata(dense, bcs, n_cells)
    raw = _write_h5(tmp_path / "raw_feature_bc_matrix.h5", dense, bcs)
    seq_metrics.store(a, [seq_metrics.build_record("s1", _files(raw), bcs[:n_cells])])
    counts = dense[N_GENES:, :n_cells].T.astype(float)

    # Step I alone: (log(x + 10) - mean) / sd against the window's droplets
    vals, rec = dsb.compute(a, denoise=False)
    n0, mean0, sd0 = _direct_stats(dense, n_cells)
    np.testing.assert_allclose(vals, (np.log(counts + 10) - mean0) / sd0, rtol=1e-5, atol=1e-5)
    assert rec["mode"] == "empty_droplets" and rec["n_droplets"] == n0
    assert rec["isotypes"] == ISO
    # Isotypes sit near the background, stained proteins far above it
    med = {p["protein"]: p["median"] for p in rec["proteins"]}
    assert abs(med["IgG1_control"]) < 2 and med["CD3"] > med["IgG1_control"]

    # Step II: the per-cell capture factor goes from the isotypes
    depth = np.log(counts[:, 4] + counts[:, 5] + 1)
    den, rec2 = dsb.compute(a)
    r_before = abs(np.corrcoef(vals[:, 4], depth)[0, 1])
    r_after = abs(np.corrcoef(den[:, 4], depth)[0, 1])
    assert r_after < r_before / 2
    assert rec2["technical"]["corr_background_mean"] is not None
    assert "isotypes" in rec2["method"]
    # T cells (every third cell) stay CD3-high after denoising
    kind = np.arange(n_cells) % 3
    assert den[kind == 0, 0].mean() > den[kind != 0, 0].mean() + 3


def test_without_isotypes_the_background_mean_alone(tmp_path):
    a, *_ = _with_background(tmp_path)
    vals, rec = dsb.compute(a, isotypes=[])
    assert rec["isotypes"] == []
    assert any("No isotype controls" in n for n in rec["notes"])
    s1, _ = dsb.compute(a, denoise=False)
    _, tech, bg_mean, _ = dsb.step2(s1.astype(np.float64), [])
    np.testing.assert_allclose(tech, bg_mean)
    np.testing.assert_allclose(vals, dsb.regress_out(s1.astype(np.float64), bg_mean), atol=1e-4)
    with pytest.raises(dsb.DsbError, match="Not in the protein panel"):
        dsb.compute(a, isotypes=["IgG3"])


def test_without_a_raw_matrix_the_empirical_background(tmp_path):
    dense, bcs, n_cells = _droplets()
    a = _cells_adata(dense, bcs, n_cells)
    with pytest.raises(dsb.DsbError, match="No empty droplets"):
        dsb.compute(a, mode="empty_droplets")
    vals, rec = dsb.compute(a)  # auto
    assert rec["mode"] == "model_negative" and rec["rna_window"] is None
    assert any("ModelNegativeADTnorm" in n for n in rec["notes"])
    counts = dense[N_GENES:, :n_cells].T.astype(float)
    s1, mu1, ok = dsb.step1_model_negative(counts)
    assert ok.all()
    # CD3's background is the non-T cells' level: log(4 * 3 + 1) ~ 2.6, not the T cells'
    assert mu1[0] == pytest.approx(math.log(13), abs=0.3)
    st = dsb.status(a)
    assert st["available"] and not st["empty_droplets"] and "Sequencing QC" in st["empty_droplets_reason"]


def test_window_with_too_few_droplets_is_refused(tmp_path):
    a, *_ = _with_background(tmp_path)
    with pytest.raises(dsb.DsbError, match="Widen the window"):
        dsb.compute(a, prot_window=(4.0, 4.1))
    with pytest.raises(dsb.DsbError, match="window must run"):
        dsb.compute(a, rna_window=(2.0, 1.0))


def test_run_writes_the_layer_record_and_provenance(tmp_path):
    a, *_ = _with_background(tmp_path)
    path = tmp_path / "d.h5ad"
    a.write_h5ad(path)
    adaptor = AnnDataAdaptor(str(path))
    with pytest.raises(KeyError):
        adaptor.get_expression(["CD3"], layer="protein_dsb")
    assert adaptor.protein_info()["dsb_layer"] is None
    rec = dsb.run(adaptor.adata)
    assert adaptor.protein_info()["dsb_layer"] == "protein_dsb"
    assert adaptor.protein_info()["dsb"]["isotypes"] == ISO
    got = adaptor.get_expression(["CD3"], layer="protein_dsb")[:, 0]
    np.testing.assert_allclose(got, adaptor.adata.obsm[dsb.DSB_KEY]["CD3"].to_numpy())
    (h,) = [h for h in provenance.read_provenance(adaptor.adata)["history"] if h["step"] == dsb.STEP]
    assert h["params"]["samples"][0]["raw_files"] == ["raw_feature_bc_matrix.h5"]
    assert h["params"]["features"] == AB
    # Co-pilot fact, claims tag, results table
    lines, detail = dsb.grounding_lines(adaptor.adata)
    assert "[result:dsb]" in lines[1] and "empty droplets" in detail
    assert resolve_tag("dsb", [dsb.TAG]) == [dsb.TAG]
    table = results_export.table({"kind": "dsb"}, dsb.results_value(rec))
    assert list(table["protein"]) == SHOWN and "background_mean" in table.columns
    # A new panel (merge, hashtags moved out) drops the values
    protein.renormalise(adaptor.adata)
    assert dsb.DSB_KEY not in adaptor.adata.obsm and dsb.read_record(adaptor.adata) is None


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


def _register(dm, a: ad.AnnData, did: str) -> str:
    d = Path(dm.data_dir) / "uploads" / did
    d.mkdir(parents=True)
    (d / "metadata.json").write_text(json.dumps({"id": did, "name": "cite", "status": "ready"}))
    a.write_h5ad(d / "data.h5ad")
    return did


async def test_endpoints_status_run_and_save(client, dataset_manager, tmp_path):
    a, dense, bcs, n_cells, _ = _with_background(tmp_path)
    did = _register(dataset_manager, a, "cite")
    r = await client.get(f"{API}/datasets/{did}/protein/dsb")
    assert r.status_code == 200, r.text
    st = r.json()
    assert st["empty_droplets"] and st["isotypes"] == ISO and not st["computed"]
    (s,) = st["samples"]
    assert s["n_droplets_default"] == _direct_stats(dense, n_cells)[0] and s["n_cells"] == n_cells
    assert sum(s["density"]["n"]) > s["n_droplets_default"]

    r = await client.post(f"{API}/datasets/{did}/protein/dsb", json={"prot_window": [4.0, 4.1]})
    assert r.status_code == 400 and "Widen" in r.json()["detail"]
    r = await client.post(f"{API}/datasets/{did}/protein/dsb", json={})
    assert r.status_code == 200, r.text
    assert r.json()["mode"] == "empty_droplets"
    derived = Path(dataset_manager.data_dir) / "derived" / did / "data.h5ad"
    on_disk = ad.read_h5ad(derived)
    assert list(on_disk.obsm[dsb.DSB_KEY].columns) == SHOWN
    r = await client.get(f"{API}/datasets/{did}")
    assert r.json()["protein"]["dsb_layer"] == "protein_dsb"
    r = await client.get(f"{API}/datasets/{did}/expression", params={"genes": "CD3", "layer": "protein_dsb"})
    assert r.status_code == 200


async def test_endpoint_without_protein_or_raw_matrix(client, dataset_manager, tmp_path):
    a = ad.AnnData(X=sp.csr_matrix(np.ones((5, 3), dtype=np.float32)))
    did = _register(dataset_manager, a, "plain")
    r = await client.get(f"{API}/datasets/{did}/protein/dsb")
    assert r.status_code == 200 and not r.json()["available"]
    r = await client.post(f"{API}/datasets/{did}/protein/dsb", json={})
    assert r.status_code == 404
    dense, bcs, n_cells = _droplets(n_cells=90, n_empty=10)
    did = _register(dataset_manager, _cells_adata(dense, bcs, n_cells), "noraw")
    r = await client.post(f"{API}/datasets/{did}/protein/dsb", json={"mode": "empty_droplets"})
    assert r.status_code == 400 and "No empty droplets" in r.json()["detail"]
    r = await client.post(f"{API}/datasets/{did}/protein/dsb", json={"mode": "auto"})
    assert r.status_code == 200 and r.json()["mode"] == "model_negative"


# ---------------------------------------------------------------------------
# Replay
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("with_raw", [True, False])
def test_replay_script_repeats_dsb(tmp_path, with_raw):
    if with_raw:
        a, *_ = _with_background(tmp_path, n_cells=150, n_empty=1500)
    else:
        dense, bcs, n_cells = _droplets(n_cells=150, n_empty=10)
        a = _cells_adata(dense, bcs, n_cells)
    inp = a.copy()
    dsb.run(a)
    text = replay_script.build_script(a, dataset_id="t", dataset_name="cite")
    assert "def lower_means(" in text and "DSB = " in text
    assert {d["status"] for d in replay_script.describe(a)} == {"exact"}
    src, script, dst = tmp_path / "input.h5ad", tmp_path / "replay.py", tmp_path / "replayed.h5ad"
    inp.write_h5ad(src)
    script.write_text(text)
    proc = subprocess.run([sys.executable, str(script), str(src), str(dst)], capture_output=True,
                          text=True, timeout=600, cwd=tmp_path)
    assert proc.returncode == 0, proc.stderr[-3000:]
    rep = ad.read_h5ad(dst)
    got = rep.obsm[dsb.DSB_KEY]
    got = got.to_numpy() if isinstance(got, pd.DataFrame) else np.asarray(got)
    np.testing.assert_allclose(got, a.obsm[dsb.DSB_KEY].to_numpy(), atol=1e-5)


def test_replay_without_raw_files_is_not_replayable():
    a = ad.AnnData(np.zeros((3, 2), dtype="float32"))
    provenance.record_step(a, step=dsb.STEP, tool="t", params={
        "mode": "empty_droplets", "proteins": ["CD3"], "features": ["CD3"], "samples": [{"sample": "s1"}]})
    assert {d["status"] for d in replay_script.describe(a)} == {"not_replayable"}
