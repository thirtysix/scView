"""Visium HD (core/spatial.py, ingestion/paths.py, bundling.py, spatial_autocorr.py): a bin
folder bundled, read and attached in memory and in blocks, its refusals, what stays out of
an upload, bins through Moran's I (a square grid, in memory and from the gene store)."""

from __future__ import annotations

import json
import tarfile
from pathlib import Path

import anndata as ad
import h5py
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import scipy.sparse as sp
from PIL import Image

from scview.core import preflight, spatial
from scview.core import spatial_autocorr as sa
from scview.core.anndata_adaptor import AnnDataAdaptor
from scview.core.ingestion import IngestSessionManager
from scview.core.ingestion.bundling import build_bundle
from scview.core.ingestion.detection import detect_file
from scview.core.ingestion.paths import junk_reason
from scview.core.ingestion.validation import validate_bundle
from tests.large_fixture import make_large

N_GENES = 12
SIDE = 6  # bins per side of the capture area
SF = {"spot_diameter_fullres": 29.2, "bin_size_um": 8.0, "microns_per_pixel": 0.2738,
      "tissue_hires_scalef": 0.25, "tissue_lowres_scalef": 0.025, "regist_target_img_scalef": 0.25,
      "fiducial_diameter_fullres": 1205.0}


def _barcode(row: int, col: int, um: int = 8) -> str:
    return f"s_{um:03d}um_{row:05d}_{col:05d}-1"


def _bin_folder(d: Path, um: int = 8, in_tissue=lambda r, c: r < 4) -> Path:
    """A square_NNNum folder: the filtered matrix holds the in-tissue bins, shuffled as Space
    Ranger writes them; the positions list every bin of the capture area, in grid order."""
    sp_dir = d / "spatial"
    sp_dir.mkdir(parents=True, exist_ok=True)
    rows, cols = np.meshgrid(np.arange(SIDE), np.arange(SIDE), indexing="ij")
    rows, cols = rows.ravel(), cols.ravel()
    tissue = np.array([in_tissue(r, c) for r, c in zip(rows, cols)], dtype=np.uint8)
    pq.write_table(pa.table({
        "barcode": [_barcode(r, c, um) for r, c in zip(rows, cols)],
        "in_tissue": pa.array(tissue, type=pa.uint8()),
        "array_row": pa.array(rows.astype(np.uint32)), "array_col": pa.array(cols.astype(np.uint32)),
        "pxl_row_in_fullres": rows * 29.2 + 100.0, "pxl_col_in_fullres": cols * 29.2 + 200.0,
    }), sp_dir / "tissue_positions.parquet")
    (sp_dir / "scalefactors_json.json").write_text(json.dumps({**SF, "bin_size_um": float(um)}))
    img = np.zeros((6, 8, 3), dtype=np.uint8)
    Image.fromarray(img).save(sp_dir / "tissue_hires_image.png")
    Image.fromarray(img).save(sp_dir / "tissue_lowres_image.png")
    keep = np.flatnonzero(tissue)[::-1]
    n = len(keep)
    counts = np.random.default_rng(um).poisson(2.0, size=(n, N_GENES)) + 1
    csc = sp.csc_matrix(counts.T.astype(np.int32))
    with h5py.File(d / "filtered_feature_bc_matrix.h5", "w") as f:
        g = f.create_group("matrix")
        g["data"], g["indices"], g["indptr"] = csc.data, csc.indices.astype(np.int64), csc.indptr
        g["shape"] = np.array([N_GENES, n], dtype=np.int32)
        g["barcodes"] = np.array([_barcode(rows[k], cols[k], um).encode() for k in keep])
        ft = g.create_group("features")
        ft["id"] = np.array([f"ENSMUSG{j:011d}".encode() for j in range(N_GENES)])
        ft["name"] = np.array([f"G{j}".encode() for j in range(N_GENES)])
        ft["feature_type"] = np.array([b"Gene Expression"] * N_GENES)
        ft["genome"] = np.array([b"mm10"] * N_GENES)
    return d


def _files(root: Path) -> list[Path]:
    return sorted(p for p in root.rglob("*") if p.is_file())


def _add_output_junk(d: Path) -> None:
    (d / "raw_feature_bc_matrix.h5").write_bytes(b"x")
    (d / "raw_probe_bc_matrix.h5").write_bytes(b"x")
    (d / "cloupe.cloupe").write_bytes(b"x")
    (d / "spatial" / "aligned_tissue_image.jpg").write_bytes(b"x")
    (d / "spatial" / "detected_tissue_image.jpg").write_bytes(b"x")
    (d / "spatial" / "cytassist_image.tiff").write_bytes(b"x")
    for sub in ("filtered_feature_bc_matrix", "raw_feature_bc_matrix", "analysis/pca"):
        (d / sub).mkdir(parents=True)
        (d / sub / "barcodes.tsv.gz").write_bytes(b"x")


# ---------------------------------------------------------------------------
# What is uploaded and read
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("rel,kept", [
    ("square_008um/filtered_feature_bc_matrix.h5", True),
    ("binned_outputs/square_008um/spatial/tissue_positions.parquet", True),
    ("square_008um/spatial/scalefactors_json.json", True),
    ("square_008um/spatial/tissue_hires_image.png", True),
    ("square_008um/spatial/tissue_lowres_image.png", True),
    ("square_008um/raw_feature_bc_matrix.h5", False),
    ("square_008um/raw_probe_bc_matrix.h5", False),
    ("square_008um/filtered_feature_bc_matrix/barcodes.tsv.gz", False),
    ("square_008um/raw_feature_bc_matrix/matrix.mtx.gz", False),
    ("square_008um/analysis/pca/components.csv", False),
    ("square_008um/cloupe.cloupe", False),
    ("square_008um/spatial/aligned_fiducials.jpg", False),
    ("square_008um/spatial/detected_tissue_image.jpg", False),
    ("square_008um/spatial/cytassist_image.tiff", False),
])
def test_only_the_needed_files_of_a_bin_folder_are_kept(rel, kept):
    assert (junk_reason(rel) is None) is kept


def test_a_cell_ranger_raw_matrix_is_still_a_side_file():
    assert junk_reason("pbmc/outs/raw_feature_bc_matrix.h5") is None
    assert junk_reason("pbmc/outs/raw_feature_bc_matrix/barcodes.tsv.gz") is None


def test_junk_in_a_bin_folder_is_listed_with_a_reason(tmp_path):
    d = _bin_folder(tmp_path / "up" / "square_008um")
    _add_output_junk(d)
    b = build_bundle(_files(tmp_path / "up"), root=tmp_path / "up")
    assert b.unsupported is None and b.complete and len(b.units) == 1
    assert b.units[0].label == "visium_hd_8um"
    assert {f.spatial_part for f in b.units[0].side_files} == {"positions", "scalefactors", "hires", "lowres"}
    reasons = {s.path.rsplit("/", 1)[-1]: s.reason for s in b.skipped}
    assert "raw" in reasons["raw_feature_bc_matrix.h5"]
    assert "cloupe.cloupe" in reasons and "cytassist_image.tiff" in reasons
    assert "Skipped" in b.issues[0]


def test_one_bin_size_of_a_run_default_8_um(tmp_path):
    root = tmp_path / "run" / "outs" / "binned_outputs"
    for um in (2, 8, 16):
        _bin_folder(root / f"square_{um:03d}um", um)
    (tmp_path / "run" / "outs" / "metrics_summary.csv").write_text("x\n")
    (tmp_path / "run" / "outs" / "segmented_outputs").mkdir()
    (tmp_path / "run" / "outs" / "segmented_outputs" / "cell_segmentations.geojson").write_text("{}")
    b = build_bundle(_files(tmp_path / "run"), root=tmp_path / "run")
    assert b.unsupported is None and len(b.units) == 1
    u = b.units[0]
    assert u.label == "visium_hd_8um"
    assert all("square_008um" in f.rel_path for f in u.files + u.side_files)
    assert any("another bin size" in s.reason and "square_002um" in s.path for s in b.skipped)
    assert any("rest of the Space Ranger" in s.reason and "segmented" in s.path for s in b.skipped)


def test_16_um_is_read_when_it_is_the_only_one_and_a_run_folder_names_the_dataset(tmp_path):
    root = tmp_path / "Mouse_Brain" / "binned_outputs" / "square_016um"
    _bin_folder(root, 16)
    b = build_bundle(_files(tmp_path), root=tmp_path)
    assert b.unsupported is None and b.units[0].label == "Mouse_Brain_16um"


def test_2_um_folder_is_refused_with_a_sentence(tmp_path):
    _bin_folder(tmp_path / "up" / "square_002um", 2)
    rep = validate_bundle(build_bundle(_files(tmp_path / "up"), root=tmp_path / "up"))
    assert not rep.ok
    msg = next(i.message for i in rep.issues if i.code == "spatial_unsupported")
    assert "8 or" in msg.replace("square_008um or square_016um", "8 or") and "2 um" in msg


def test_2_um_is_refused_by_its_barcodes_without_its_folder_name(tmp_path):
    d = _bin_folder(tmp_path / "up" / "folder", 2)
    b = build_bundle(_files(d))
    assert b.unsupported and "2 um" in b.unsupported
    assert detect_file(d / "filtered_feature_bc_matrix.h5").details["hd_bin_um"] == 2


def test_archive_is_refused_without_being_walked(tmp_path, monkeypatch):
    p = tmp_path / "Visium_HD_Mouse_Brain_binned_outputs.tar.gz"
    p.write_bytes(b"\x1f\x8b not read")
    monkeypatch.setattr(spatial, "_archive_parts", lambda *_: pytest.fail("walked the archive"))
    rep = validate_bundle(build_bundle([p]))
    assert not rep.ok and any("unpack it" in i.message and p.name in i.message for i in rep.issues)


def test_archive_under_another_name_is_refused_at_its_first_member(tmp_path):
    src = _bin_folder(tmp_path / "binned_outputs" / "square_016um", 16)
    p = tmp_path / "data.tar.gz"
    with tarfile.open(p, "w:gz") as tf:
        tf.add(src.parent, arcname="binned_outputs")
    rep = validate_bundle(build_bundle([p]))
    assert not rep.ok and any("unpack it" in i.message for i in rep.issues)


def test_segmented_outputs_alone_are_refused_with_their_own_sentence(tmp_path):
    d = tmp_path / "outs" / "segmented_outputs"
    d.mkdir(parents=True)
    (d / "cell_segmentations.geojson").write_text("{}")
    rep = validate_bundle(build_bundle(_files(tmp_path), root=tmp_path))
    assert not rep.ok and any("segmented_outputs" in i.message for i in rep.issues)


# ---------------------------------------------------------------------------
# The dataset
# ---------------------------------------------------------------------------


def _ingest(tmp_path, src: Path, streamed: str = "off"):
    from scview.core.ingestion.session import IngestOptions

    mgr = IngestSessionManager(str(tmp_path / "data"))
    sid = mgr.create_session()
    for f in _files(src):
        dest = mgr.files_dir(sid) / f.relative_to(src.parent)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(f.read_bytes())
    mgr.set_options(sid, IngestOptions(streamed=streamed))
    did = mgr._build_dataset(sid)
    path = next((tmp_path / "data" / "ingested" / did).glob("*.h5ad"))
    return ad.read_h5ad(path), path


@pytest.mark.parametrize("mode", ["off", "on"])
def test_ingest_reads_in_tissue_bins_in_memory_and_in_blocks(tmp_path, mode):
    src = _bin_folder(tmp_path / "up" / "square_008um")
    a, _ = _ingest(tmp_path, src, mode)
    assert a.n_obs == 4 * SIDE
    assert a.uns["scview_spatial"] == {"technology": "visium_hd", "unit": "bin", "bin_size_um": 8.0}
    lib = "visium_hd_8um"
    entry = a.uns["spatial"][lib]
    assert float(entry["scalefactors"]["bin_size_um"]) == 8.0
    assert float(entry["scalefactors"]["microns_per_pixel"]) == pytest.approx(0.2738)
    assert float(entry["scalefactors"]["spot_diameter_fullres"]) == pytest.approx(29.2)
    assert set(entry["images"]) == {"hires", "lowres"}
    assert list(a.obs["library_id"].cat.categories) == [lib]
    # Each bin at its own position, whatever the matrix's order.
    r, c = a.obs["array_row"].to_numpy(), a.obs["array_col"].to_numpy()
    assert [_barcode(x, y) for x, y in zip(r, c)] == list(a.obs_names)
    np.testing.assert_allclose(a.obsm["spatial"][:, 0], c * 29.2 + 200.0)
    np.testing.assert_allclose(a.obsm["spatial"][:, 1], r * 29.2 + 100.0)
    assert spatial.is_spot_data(a)
    assert spatial.info(a)["bin_size_um"] == 8.0


def test_a_bin_outside_the_tissue_is_the_existing_error(tmp_path):
    src = _bin_folder(tmp_path / "up" / "square_008um")
    # The positions say fewer bins are in the tissue than the matrix holds.
    t = pq.read_table(src / "spatial" / "tissue_positions.parquet").to_pandas()
    t.loc[t["array_row"] == 3, "in_tissue"] = 0
    pq.write_table(pa.Table.from_pandas(t, preserve_index=False), src / "spatial" / "tissue_positions.parquet")
    with pytest.raises(Exception, match="not among the in-tissue"):
        _ingest(tmp_path, src)


def test_in_tissue_rows_are_dropped_before_python_objects(tmp_path):
    src = _bin_folder(tmp_path / "up" / "square_008um")
    raw = (src / "spatial" / "tissue_positions.parquet").read_bytes()
    df, layout = spatial._read_positions(raw, "tissue_positions.parquet")
    assert len(df) == 4 * SIDE and "parquet" in layout and df.index[0].startswith("s_008um_")
    assert list(df.columns) == ["array_row", "array_col", "pxl_row_in_fullres", "pxl_col_in_fullres"]


def test_missing_diameter_comes_from_bin_size_and_pixel_size():
    out = spatial._with_bin_side({"bin_size_um": 8.0, "microns_per_pixel": 0.25})
    assert out["spot_diameter_fullres"] == 32.0


def test_two_runs_merge_and_two_bin_sizes_do_not(tmp_path):
    up = tmp_path / "up"
    _bin_folder(up / "A" / "binned_outputs" / "square_008um")
    _bin_folder(up / "B" / "binned_outputs" / "square_008um")
    mgr = IngestSessionManager(str(tmp_path / "data"))
    sid = mgr.create_session()
    for f in _files(up):
        dest = mgr.files_dir(sid) / f.relative_to(up)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(f.read_bytes())
    did = mgr._build_dataset(sid)
    a = ad.read_h5ad(next((tmp_path / "data" / "ingested" / did).glob("*.h5ad")))
    assert a.uns["scview_spatial"]["technology"] == "visium_hd"
    assert a.uns["scview_spatial"]["bin_size_um"] == 8.0
    assert sorted(a.obs["library_id"].cat.categories) == ["A_8um", "B_8um"]


def test_an_h5ad_with_bin_size_scale_factors_opens_as_visium_hd():
    a = ad.AnnData(np.ones((3, 2), dtype=np.float32))
    a.obsm["spatial"] = np.arange(6, dtype=np.float64).reshape(3, 2)
    a.obs["array_row"], a.obs["array_col"] = [0, 0, 1], [0, 1, 0]
    a.uns["spatial"] = {"lib": {"images": {}, "metadata": {}, "scalefactors": {
        "spot_diameter_fullres": 58.4, "bin_size_um": 16.0}}}
    assert spatial.adopt(a) is True
    assert a.uns["scview_spatial"] == {"technology": "visium_hd", "unit": "bin", "bin_size_um": 16.0}
    assert spatial.adopt(a) is False
    assert spatial.grounding_lines(a)[1].startswith("Visium HD, 16 um bins")
    assert "16 um bins" in spatial.grounding_lines(a)[0][0]


def test_export_and_reupload_comes_back_as_visium_hd(tmp_path):
    src = _bin_folder(tmp_path / "up" / "square_008um")
    a, _ = _ingest(tmp_path, src)
    out = tmp_path / "export.h5ad"
    a.write_h5ad(out)
    b = ad.read_h5ad(out)
    spatial.adopt(b)
    assert spatial.technology(b) == {"technology": "visium_hd", "unit": "bin"}
    assert spatial.bin_size_um(b) == 8.0


def test_doublet_step_costs_nothing_on_bins(tmp_path):
    a, path = _ingest(tmp_path, _bin_folder(tmp_path / "up" / "square_008um"))
    assert preflight.doublet_plan(a).channels == []
    assert preflight._file_doublet_plan(path).channels == []
    assert "not single cells" in spatial.DOUBLET_SKIP


def test_preflight_says_bins_and_counts_the_images(tmp_path):
    src = _bin_folder(tmp_path / "up" / "square_008um")
    b = build_bundle(_files(tmp_path / "up"), root=tmp_path / "up")
    h = preflight.read_unit_header(b.units[0])
    assert h.unit == "bin" and h.uns_bytes == 2 * 6 * 8 * 3 and h.n_cells == 4 * SIDE
    assert "24 bins x" in preflight._load_lead(h, preflight.estimate_load(h))
    assert src.exists()


# ---------------------------------------------------------------------------
# Moran's I on a square grid
# ---------------------------------------------------------------------------


def _grid(n=30, seed=0):
    """Bins on a slightly rotated square grid (Space Ranger's pixels are not axis-aligned),
    shuffled, with a smooth gene, a noise gene and an all-zero gene."""
    rng = np.random.default_rng(seed)
    rows, cols = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    rows, cols = rows.ravel(), cols.ravel()
    t = 0.01
    xy = np.c_[cols * np.cos(t) - rows * np.sin(t), cols * np.sin(t) + rows * np.cos(t)] * 29.2
    order = rng.permutation(len(rows))
    xy, rows, cols = xy[order], rows[order], cols[order]
    smooth = np.round(5 * (rows / n) + rng.normal(0, 0.1, len(rows)), 2).clip(0)
    noise = rng.poisson(1.0, len(rows)).astype(np.float64)
    M = sp.csr_matrix(np.c_[smooth, noise, np.zeros(len(rows))])
    a = ad.AnnData(M.astype(np.float32), var=pd.DataFrame(index=["smooth", "noise", "zero"]))
    a.obs_names = [f"b{i}" for i in range(a.n_obs)]
    a.obsm["spatial"] = xy
    a.obs["array_row"], a.obs["array_col"] = rows, cols
    a.obs["library_id"] = pd.Categorical(["L"] * a.n_obs)
    a.uns["spatial"] = {"L": {"images": {}, "metadata": {}, "scalefactors": {
        "spot_diameter_fullres": 29.2, "bin_size_um": 8.0}}}
    spatial.adopt(a)
    return a


def test_bins_are_a_grid_of_4_neighbours():
    a = _grid()
    assert sa.is_grid(a) and sa.unit_of(a) == "bin" and sa.neighbours_of(a) == 4
    r, c = sa.knn_edges(a.obsm["spatial"], 4, True)
    deg = np.bincount(r, minlength=a.n_obs)
    rows, cols = a.obs["array_row"].to_numpy(), a.obs["array_col"].to_numpy()
    edge = (rows == 0) | (rows == 29) | (cols == 0) | (cols == 29)
    corner = ((rows == 0) | (rows == 29)) & ((cols == 0) | (cols == 29))
    assert (deg[~edge] == 4).all() and (deg[edge & ~corner] == 3).all() and (deg[corner] == 2).all()
    # Only the bins sharing a side.
    assert (np.abs(rows[r] - rows[c]) + np.abs(cols[r] - cols[c]) == 1).all()


def test_moran_on_bins_finds_the_smooth_gene_and_records_its_grid(tmp_path):
    a = _grid()
    r = sa.run(a, genes="all")
    t = r.table.set_index("gene")
    assert r.neighbors == {"kind": "grid", "k": 4} and r.unit == "bin"
    assert t.loc["smooth", "I"] > 0.9 and abs(t.loc["noise", "I"]) < 0.15
    assert r.n_constant == 1 and "zero" not in t.index
    sa.record(a, r, "all", None, None)
    from scview.core import provenance

    assert provenance.recipe(a)[-1]["params"]["n_neighs"] == 4


def test_gene_blocks_equal_row_blocks():
    a = _grid()
    W = sa.spatial_weights(a.obsm["spatial"], a.obs["library_id"].astype(str).to_numpy(), True, 4)
    M = sp.csr_matrix(a.X, dtype=np.float64)
    rows = sa.moran_sums(W, M)
    genes = sa.moran_sums_by_genes(W, lambda x, y: M[:, x:y], M.shape[1], block=2)
    for x, y in zip(rows[:5], genes[:5]):
        np.testing.assert_allclose(x, y, rtol=1e-12)
    np.testing.assert_array_equal(rows[5], genes[5])


def test_large_dataset_moran_equals_in_memory(tmp_path):
    from tests.large_fixture import small_twin

    a = _grid()
    a.X = sp.csr_matrix(np.random.default_rng(1).poisson(1 + a.X.toarray()), dtype=np.float32)
    path = make_large(tmp_path / "large", a)
    adaptor = AnnDataAdaptor(str(path))
    assert adaptor.is_large
    got = sa.run(adaptor.adata, genes="all")
    want = sa.run(small_twin(a), genes="all")
    assert got.neighbors == want.neighbors == {"kind": "grid", "k": 4}
    g, w = got.table.set_index("gene"), want.table.set_index("gene")
    assert list(g.index) == list(w.index)
    np.testing.assert_allclose(g["I"], w["I"], rtol=1e-9)
    np.testing.assert_allclose(g["pval"], w["pval"], rtol=1e-6)


def test_replay_passes_the_neighbour_count():
    from scview.core import replay_script

    assert "n_neighs" in replay_script._spatial_autocorr_helper()
    snippet = replay_script._spatial_autocorr(
        {"genes": "hvg", "matrix_source": "log-normalised X", "neighbors": "grid", "n_neighs": 4, "library_id": None},
        replay_script._Ctx(adata=None))
    assert any("n_neighs=4" in line for line in snippet.code)


API = "/api/v1"


async def test_api_moran_on_a_large_dataset_saves_in_place(client, dataset_manager, tmp_data_dir):
    from tests.large_fixture import register

    a = _grid()
    a.X = sp.csr_matrix(np.random.default_rng(1).poisson(1 + a.X.toarray()), dtype=np.float32)
    h5ad = register(tmp_data_dir, "hdlarge", a)
    r = await client.post(f"{API}/datasets/hdlarge/spatial/autocorr", json={"genes": "all"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["neighbors"] == {"kind": "grid", "k": 4} and body["unit"] == "bin"
    assert body["n_obs"] == a.n_obs and body["matrix_source"] == "log-normalised X"
    dataset_manager.evict("hdlarge")
    adaptor = await dataset_manager.get_or_load_dataset("hdlarge")
    assert adaptor.is_large and adaptor.h5ad_path == str(h5ad) and "moranI" in adaptor.adata.uns


# ---------------------------------------------------------------------------
# QC: bins are not cells
# ---------------------------------------------------------------------------


def _bins_for_qc():
    """Most bins at 4% mito, a coherent layer at 15%: an outlier for MAD, not for tissue."""
    rng = np.random.default_rng(3)
    a = _grid(20)
    n = a.n_obs
    X = rng.poisson(3.0, size=(n, 40)).astype(np.float32)
    names = [f"mt-{i}" for i in range(4)] + [f"G{i}" for i in range(36)]
    layer = a.obs["array_row"].to_numpy() < 4
    share = np.where(layer, 0.15, 0.04)
    X[:, :4] = 0
    X[:, 0] = np.round(share * X[:, 4:].sum(axis=1) / (1 - share))
    b = ad.AnnData(sp.csr_matrix(X), obs=a.obs.copy(), var=pd.DataFrame(index=names), uns=dict(a.uns))
    b.obsm["spatial"] = a.obsm["spatial"]
    return b, layer


def test_mito_rules_are_off_on_bins_until_asked_for():
    from scview.core import qc_filter
    from scview.core.pipeline import PipelineParams, filter_judgement, per_cell_qc

    a, layer = _bins_for_qc()
    t = qc_filter.Thresholds.from_params(PipelineParams())
    as_cells = qc_filter.removed_by(per_cell_qc(a), t)
    assert as_cells["max_pct_mt"][layer].mean() > 0.9  # the rule as it is for cells cuts the layer out
    _, keep, _, found = filter_judgement(a, PipelineParams())
    assert keep[layer].mean() > 0.99 and found["max_pct_mt"] is None and found["qc_bin_mito"] is False
    _, keep, _, found = filter_judgement(a, PipelineParams(qc_bin_mito=True))
    assert not keep[layer].any() and found["qc_bin_mito"] is True


def test_fixed_defaults_on_bins_and_a_choice_stands():
    from scview.core import qc_filter
    from scview.core.pipeline import PipelineParams, per_cell_qc

    a, _ = _bins_for_qc()
    cells = per_cell_qc(a)
    fixed = PipelineParams(qc_threshold_mode="fixed")
    t, c = qc_filter.bin_rules(a, qc_filter.Thresholds.from_params(fixed), cells, fixed)
    assert (t.min_genes, t.min_counts, t.max_pct_mt) == (0.0, 1.0, None) and c["has_mt"] is False
    chosen = PipelineParams(qc_threshold_mode="fixed", min_genes=50, min_counts=1)
    t, _ = qc_filter.bin_rules(a, qc_filter.Thresholds.from_params(chosen), cells, chosen)
    assert (t.min_genes, t.min_counts) == (50.0, 1.0)


def test_cells_and_spots_keep_their_rules():
    from scview.core import qc_filter
    from scview.core.pipeline import PipelineParams, per_cell_qc

    a, _ = _bins_for_qc()
    del a.uns["spatial"]["L"]["scalefactors"]["bin_size_um"]
    a.uns.pop("scview_spatial")
    assert spatial.technology(a)["unit"] == "spot"
    cells = per_cell_qc(a)
    p = PipelineParams(qc_threshold_mode="fixed")
    t, c = qc_filter.bin_rules(a, qc_filter.Thresholds.from_params(p), cells, p)
    assert (t.min_genes, t.max_pct_mt) == (200.0, 20.0) and c is cells


async def test_filter_preview_says_why_mito_is_off_on_bins(client, dataset_manager):
    from tests.test_hvg_sensitivity import _register

    a, layer = _bins_for_qc()
    did = _register(dataset_manager, a, "hdqc")
    body = (await client.get(f"{API}/datasets/{did}/qc/filter-preview")).json()
    assert body["n_removed"] < 5 and body["has_mt"] is False and "capture grid" in body["bin_note"]
    body = (await client.get(f"{API}/datasets/{did}/qc/filter-preview", params={"bin_mito": True})).json()
    assert body["n_removed"] >= layer.sum() * 0.9 and "bin_note" not in body


def test_assessor_wording_for_bins():
    from scview.core.assessor import _check_filtering
    from scview.core.pipeline import per_cell_qc

    a, _ = _bins_for_qc()
    a.obs["n_genes_by_counts"] = per_cell_qc(a)["n_genes"]
    s = _check_filtering(a)
    assert s.done is False and "bin" in s.details and "200" not in s.details


def test_spot_composition_accepts_bins_and_its_refusal_names_them():
    from scview.core import spot_deconvolution as sd

    assert "Visium HD bins" in sd.NOT_SPOTS and spatial.is_spot_data(_grid())


def test_guide_and_router_know_bins():
    from scview.core import feature_guide
    from scview.core.rag.router import named_result_facets

    text = " ".join(str(v) for k, v in vars(feature_guide).items() if k.isupper())
    assert "Visium HD and Xenium bundles are not read" not in text and "square_NNNum" in text
    assert "groups" in named_result_facets("what do the 8 um bins of this Visium HD slide hold?")
