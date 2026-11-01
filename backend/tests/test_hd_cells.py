"""Space Ranger 4's segmented_outputs (core/hd_cells.py, core/spatial.py, ingestion/): a folder
bundled, read and attached in memory and in blocks, its refusals, what stays out of an upload, and
what unit "cell" changes for the readers that mean a grid."""

from __future__ import annotations

import gzip
import json
import tarfile
from pathlib import Path

import anndata as ad
import h5py
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp
from PIL import Image

from scview.core import cell_outlines, hd_cells, preflight, qc_filter, replay_script, spatial
from scview.core import spatial_autocorr as sa
from scview.core import spatial_domains as sd
from scview.core import spot_deconvolution as deconv
from scview.core.ingestion import IngestSessionManager
from scview.core.ingestion.bundling import SideKind, build_bundle
from scview.core.ingestion.detection import FileKind, detect_file
from scview.core.ingestion.paths import junk_reason
from scview.core.ingestion.session import IngestCommitError, IngestOptions
from scview.core.ingestion.validation import validate_bundle

N_GENES = 12
IDS = np.array([1, 2, 3, 5, 8, 9, 11, 14])  # Space Ranger's ids have gaps
SF = {"microns_per_pixel": 0.2738, "tissue_lowres_scalef": 0.025, "fiducial_diameter_fullres": 1205.0,
      "tissue_hires_scalef": 0.25, "regist_target_img_scalef": 0.25}


def _barcode(i: int) -> str:
    return f"cellid_{i:09d}-1"


def _square(x: float, y: float, side: float = 10.0) -> list[list[float]]:
    return [[x, y], [x + side, y], [x + side, y + side], [x, y + side], [x, y]]


def _ell(x: float, y: float) -> list[list[float]]:
    """An L: 3 squares of side 10. Area-weighted centroid (x + 25/3, y + 25/3), its vertices' mean differs."""
    return [[x, y], [x + 20, y], [x + 20, y + 10], [x + 10, y + 10], [x + 10, y + 20], [x, y + 20], [x, y]]


def _collection(rings: dict, **props) -> dict:
    return {"type": "FeatureCollection", "features": [
        {"type": "Feature", "geometry": {"type": "Polygon", "coordinates": [r]},
         "properties": {"cell_id": int(i), **(props["extra"](i) if "extra" in props else {})}}
        for i, r in rings.items()]}


def _write_matrix(path: Path, ids) -> None:
    n = len(ids)
    counts = np.random.default_rng(0).poisson(2.0, size=(n, N_GENES)) + 1
    csc = sp.csc_matrix(counts.T.astype(np.int32))
    with h5py.File(path, "w") as f:
        g = f.create_group("matrix")
        g["data"], g["indices"], g["indptr"] = csc.data, csc.indices.astype(np.int64), csc.indptr
        g["shape"] = np.array([N_GENES, n], dtype=np.int32)
        g["barcodes"] = np.array([_barcode(int(i)).encode() for i in ids])
        ft = g.create_group("features")
        ft["id"] = np.array([f"ENSMUSG{j:011d}".encode() for j in range(N_GENES)])
        ft["name"] = np.array([f"G{j}".encode() for j in range(N_GENES)])
        ft["feature_type"] = np.array([b"Gene Expression"] * N_GENES)
        ft["genome"] = np.array([b"mm10"] * N_GENES)


def _seg_folder(d: Path, ids=IDS, graphclust: bool = True, extra_outline: bool = True) -> Path:
    """A segmented_outputs folder: the matrix in id order, the outlines shuffled."""
    (d / "spatial").mkdir(parents=True, exist_ok=True)
    _write_matrix(d / "filtered_feature_cell_matrix.h5", ids)
    rings = {int(i): _ell(100.0 * int(i), 50.0 * int(i)) for i in ids[::-1]}
    if extra_outline:
        rings[99] = _square(5.0, 5.0)  # filtered out of the matrix
    (d / "cell_segmentations.geojson").write_text(json.dumps(_collection(rings)))
    nuclei = {int(i): _square(100.0 * int(i) + 2, 50.0 * int(i) + 2, 4.0) for i in ids[1:]}  # the first has none
    (d / "nucleus_segmentations.geojson").write_text(json.dumps(_collection(nuclei)))
    if graphclust:
        (d / "graphclust_annotated_cell_segmentations.geojson").write_text(json.dumps(_collection(
            rings, extra=lambda i: {"classification": {"name": f"Cluster-{10 if int(i) % 2 else 2}",
                                                       "color": [0, 255, 0]}})))
    (d / "spatial" / "scalefactors_json.json").write_text(json.dumps(SF))
    img = np.zeros((6, 8, 3), dtype=np.uint8)
    Image.fromarray(img).save(d / "spatial" / "tissue_hires_image.png")
    Image.fromarray(img).save(d / "spatial" / "tissue_lowres_image.png")
    return d


def _bin_folder(d: Path, um: int) -> None:
    from tests.test_visium_hd import _bin_folder as make

    make(d, um)


def _files(root: Path) -> list[Path]:
    return sorted(p for p in root.rglob("*") if p.is_file())


def _ingest(tmp_path, src: Path, streamed: str = "off"):
    mgr = IngestSessionManager(str(tmp_path / "data"))
    sid = mgr.create_session()
    for f in _files(src):
        dest = mgr.files_dir(sid) / f.relative_to(src.parent)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(f.read_bytes())
    mgr.set_options(sid, IngestOptions(streamed=streamed))
    did = mgr._build_dataset(sid)
    return ad.read_h5ad(next((tmp_path / "data" / "ingested" / did).glob("*.h5ad"))), did, tmp_path / "data"


@pytest.fixture
def cells(tmp_path) -> ad.AnnData:
    return _ingest(tmp_path, _seg_folder(tmp_path / "up" / "segmented_outputs"))[0]


# ---------------------------------------------------------------------------
# What is uploaded and read
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("rel,kept", [
    ("segmented_outputs/filtered_feature_cell_matrix.h5", True),
    ("run/outs/segmented_outputs/cell_segmentations.geojson", True),
    ("segmented_outputs/nucleus_segmentations.geojson", True),
    ("segmented_outputs/graphclust_annotated_cell_segmentations.geojson", True),
    ("segmented_outputs/spatial/scalefactors_json.json", True),
    ("segmented_outputs/spatial/tissue_hires_image.png", True),
    ("segmented_outputs/spatial/tissue_lowres_image.png", True),
    ("segmented_outputs/filtered_feature_cell_matrix/matrix.mtx.gz", True),
    ("segmented_outputs/raw_feature_cell_matrix.h5", False),
    ("segmented_outputs/raw_feature_cell_matrix/barcodes.tsv.gz", False),
    ("segmented_outputs/graphclust_annotated_nucleus_segmentations.geojson", False),
    ("segmented_outputs/analysis/clustering/gene_expression_graphclust/clusters.csv", False),
    ("segmented_outputs/spatial/cytassist_image.tiff", False),
    ("segmented_outputs/spatial/aligned_fiducials.jpg", False),
    ("segmented_outputs/cloupe_cell.cloupe", False),
])
def test_only_the_needed_files_of_the_folder_are_kept(rel, kept):
    assert (junk_reason(rel) is None) is kept


def test_outlines_are_recognised_by_name_without_being_read_as_a_table(tmp_path):
    d = _seg_folder(tmp_path / "segmented_outputs")
    found = detect_file(d / "cell_segmentations.geojson")
    assert found.kind == FileKind.hd_cells and found.details["part"] == "cell_segmentations"
    (tmp_path / "other.geojson").write_text("{}")
    assert detect_file(tmp_path / "other.geojson").kind != FileKind.hd_cells


def test_the_folder_bundles_as_one_unit_named_after_its_run(tmp_path):
    root = tmp_path / "up"
    _seg_folder(root / "Mouse_Brain" / "outs" / "segmented_outputs")
    b = build_bundle(_files(root), root=root)
    assert b.unsupported is None and b.complete and len(b.units) == 1
    u = b.units[0]
    assert u.label == "Mouse_Brain_cells"
    assert u.format.value == "tenx_h5"
    assert {f.spatial_part for f in u.side_files if f.side_kind == SideKind.hd_cells} == set(hd_cells.PARTS)
    assert {f.spatial_part for f in u.side_files if f.side_kind == SideKind.spatial} == {
        "scalefactors", "hires", "lowres"}
    assert validate_bundle(b).ok


def test_the_folder_alone_at_the_top_has_a_neutral_name(tmp_path):
    root = tmp_path / "up"
    _seg_folder(root / "segmented_outputs")
    assert build_bundle(_files(root), root=root).units[0].label == "visium_hd_cells"


def test_a_run_with_bins_and_cells_reads_as_cells_and_says_how_to_get_bins(tmp_path):
    root = tmp_path / "up"
    outs = root / "run" / "outs"
    _seg_folder(outs / "segmented_outputs")
    for um in (2, 8, 16):
        _bin_folder(outs / "binned_outputs" / f"square_{um:03d}um", um)
    (outs / "spatial").mkdir()
    (outs / "spatial" / "scalefactors_json.json").write_text(json.dumps(SF))
    b = build_bundle(_files(root), root=root)
    assert b.unsupported is None and len(b.units) == 1  # the 2 um folder is no refusal here
    assert b.units[0].label == "run_cells"
    assert all("segmented_outputs" in f.rel_path for f in b.units[0].files + b.units[0].side_files)
    bins = next(s for s in b.skipped if "square_016um" in s.path)
    assert "reads as cells" in bins.reason and "binned_outputs" in bins.reason
    assert any("rest of the Space Ranger" in s.reason and s.path.endswith("outs/spatial/scalefactors_json.json")
               for s in b.skipped)


def test_bins_and_cells_side_by_side_at_the_top_read_as_cells(tmp_path):
    root = tmp_path / "up"
    _seg_folder(root / "segmented_outputs")
    _bin_folder(root / "binned_outputs" / "square_016um", 16)
    b = build_bundle(_files(root), root=root)
    assert len(b.units) == 1 and b.units[0].label == "visium_hd_cells"


def test_a_run_of_bins_without_segmentation_is_unaffected(tmp_path):
    root = tmp_path / "up"
    _bin_folder(root / "run" / "outs" / "binned_outputs" / "square_016um", 16)
    b = build_bundle(_files(root), root=root)
    assert len(b.units) == 1 and b.units[0].label == "run_16um"


def test_a_binned_upload_alone_still_reads_as_bins_when_chosen_from_a_run_with_cells(tmp_path):
    root = tmp_path / "up"
    _bin_folder(root / "binned_outputs" / "square_016um", 16)
    b = build_bundle(_files(root), root=root)
    assert b.units[0].label == "visium_hd_16um"


@pytest.mark.parametrize("name", ["Visium_HD_Mouse_Brain_segmented_outputs.tar.gz", "segmented_outputs.tar"])
def test_an_archive_gets_a_sentence_without_being_walked(tmp_path, monkeypatch, name):
    p = tmp_path / name
    p.write_bytes(b"\x1f\x8b not read")
    monkeypatch.setattr(spatial, "_archive_parts", lambda *_: pytest.fail("walked the archive"))
    rep = validate_bundle(build_bundle([p]))
    assert not rep.ok and any("segmented_outputs archive" in i.message and name in i.message for i in rep.issues)


def test_an_archive_under_another_name_is_refused_at_its_first_member(tmp_path):
    src = _seg_folder(tmp_path / "x" / "segmented_outputs")
    p = tmp_path / "data.tar.gz"
    with tarfile.open(p, "w:gz") as tf:
        tf.add(src, arcname="segmented_outputs")
    rep = validate_bundle(build_bundle([p]))
    assert not rep.ok and any("segmented_outputs archive" in i.message for i in rep.issues)


@pytest.mark.parametrize("gone", ["filtered_feature_cell_matrix.h5", "cell_segmentations.geojson"])
def test_a_folder_lacking_its_matrix_or_outlines_is_refused(tmp_path, gone):
    root = tmp_path / "up"
    d = _seg_folder(root / "segmented_outputs")
    (d / gone).unlink()
    rep = validate_bundle(build_bundle(_files(root), root=root))
    assert not rep.ok and any("lacks its filtered_feature_cell_matrix.h5" in i.message for i in rep.issues)


def test_flat_upload_of_the_files_reads_too_and_needs_the_cell_outlines(tmp_path):
    d = _seg_folder(tmp_path / "segmented_outputs")
    flat = tmp_path / "flat"
    flat.mkdir()
    for f in _files(d):
        (flat / f.name).write_bytes(f.read_bytes())
    b = build_bundle(_files(flat), root=flat)
    assert len(b.units) == 1 and validate_bundle(b).ok and b.units[0].label == "visium_hd_cells"
    (flat / "cell_segmentations.geojson").unlink()
    rep = validate_bundle(build_bundle(_files(flat), root=flat))
    assert not rep.ok and any("cell_segmentations.geojson" in i.message for i in rep.issues)


def test_no_scale_factors_is_a_warning_not_a_refusal(tmp_path):
    root = tmp_path / "up"
    d = _seg_folder(root / "segmented_outputs")
    (d / "spatial" / "scalefactors_json.json").unlink()
    rep = validate_bundle(build_bundle(_files(root), root=root))
    assert rep.ok and any(i.code == "spatial_no_scalefactors" for i in rep.issues)


# ---------------------------------------------------------------------------
# Reading the outlines
# ---------------------------------------------------------------------------


def test_features_are_read_across_chunk_boundaries(tmp_path, monkeypatch):
    monkeypatch.setattr(hd_cells, "_CHUNK_CHARS", 64)
    p = tmp_path / "cell_segmentations.geojson"
    rings = {i: _square(float(i), 2.0 * i) for i in range(1, 40)}
    p.write_text(json.dumps(_collection(rings), indent=1))
    ids, offsets, xy = hd_cells.read_rings(p)
    assert ids.tolist() == list(range(1, 40))
    assert np.diff(offsets).tolist() == [4] * 39  # the closing vertex is dropped
    assert xy[4].tolist() == [2.0, 4.0]


def test_a_cut_file_and_not_geojson_get_sentences(tmp_path):
    p = tmp_path / "cell_segmentations.geojson"
    p.write_text(json.dumps(_collection({1: _square(0, 0)}))[:-40])
    with pytest.raises(hd_cells.HdCellsError, match="ends in the middle"):
        hd_cells.read_rings(p)
    p.write_text("{}")
    with pytest.raises(hd_cells.HdCellsError, match="not a GeoJSON FeatureCollection"):
        hd_cells.read_rings(p)


def test_a_cell_listed_twice_is_refused(tmp_path):
    p = tmp_path / "cell_segmentations.geojson"
    data = _collection({1: _square(0, 0)})
    data["features"] *= 2
    p.write_text(json.dumps(data))
    with pytest.raises(hd_cells.HdCellsError, match="more than once"):
        hd_cells.read_rings(p)


def test_centroid_is_area_weighted_and_exact_far_from_the_origin():
    x, y = 20_000.0, 30_000.0
    ring = np.array(_ell(x, y)[:-1])
    got = hd_cells.ring_centroids(np.array([0, len(ring)]), ring)
    np.testing.assert_allclose(got[0], [x + 25 / 3, y + 25 / 3], atol=1e-9)
    assert not np.allclose(got[0], ring.mean(axis=0))


def test_either_winding_gives_the_same_centroid():
    ring = np.array(_square(1.0, 1.0)[:-1])
    offsets = np.array([0, 4, 8])
    got = hd_cells.ring_centroids(offsets, np.concatenate([ring, ring[::-1]]))
    np.testing.assert_allclose(got, [[6, 6], [6, 6]])


def test_a_ring_with_no_area_takes_its_vertices_mean():
    ring = np.array([[0.0, 0.0], [2.0, 2.0], [4.0, 4.0]])
    np.testing.assert_allclose(hd_cells.ring_centroids(np.array([0, 3]), ring), [[2.0, 2.0]])


def test_barcode_to_cell_number():
    assert hd_cells.cell_numbers(["cellid_000000014-1", "ACGT-1", "cellid_x-1"]).tolist() == [14, -1, -1]


# ---------------------------------------------------------------------------
# The dataset
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("mode", ["off", "on"])
def test_ingest_places_the_cells_in_memory_and_in_blocks(tmp_path, mode):
    a, did, data = _ingest(tmp_path, _seg_folder(tmp_path / "up" / "segmented_outputs"), mode)
    assert a.n_obs == len(IDS) and list(a.obs_names) == [_barcode(int(i)) for i in IDS]
    assert a.uns["scview_spatial"] == {"technology": "visium_hd", "unit": "cell"}
    # Each cell at its own outline's centroid, whatever the file's order.
    np.testing.assert_allclose(a.obsm["spatial"], np.c_[100.0 * IDS + 25 / 3, 50.0 * IDS + 25 / 3])
    lib = "visium_hd_cells"
    assert list(a.obs["library_id"].cat.categories) == [lib]
    entry = a.uns["spatial"][lib]
    assert set(entry["images"]) == {"hires", "lowres"}
    assert float(entry["scalefactors"]["tissue_hires_scalef"]) == 0.25
    assert "array_row" not in a.obs.columns
    assert spatial.technology(a) == {"technology": "visium_hd", "unit": "cell"}
    info = spatial.info(a)
    assert info["unit"] == "cell" and "bin_size_um" not in info
    assert info["libraries"][0]["images"]["hires"]["scalef"] == 0.25
    assert cell_outlines.kinds_of(data, did) == ["cell", "nucleus"]


def test_outlines_are_stored_by_obs_name_without_the_closing_vertex(tmp_path):
    a, did, data = _ingest(tmp_path, _seg_folder(tmp_path / "up" / "segmented_outputs"))
    offsets, xy = cell_outlines.read_for(cell_outlines.store_path(data, did, "cell"), a.obs_names[::-1])
    assert np.diff(offsets).tolist() == [6] * len(IDS)
    np.testing.assert_allclose(xy[:6], np.array(_ell(1400.0, 700.0)[:-1], dtype=np.float32))
    # A filtered-out cell's outline is not kept; a cell without a nucleus has an empty one.
    with h5py.File(cell_outlines.store_path(data, did, "cell")) as f:
        assert len(f["ids"]) == len(IDS)
    offsets, xy = cell_outlines.read_for(cell_outlines.store_path(data, did, "nucleus"), a.obs_names)
    assert np.diff(offsets).tolist() == [0] + [4] * (len(IDS) - 1)
    assert "outlines" not in str(list(a.uns.keys())) and "cell_outlines" not in a.obsm


def test_graph_clusters_come_in_as_a_category_in_their_own_order(cells):
    s = cells.obs[hd_cells.GRAPHCLUST_COLUMN]
    assert isinstance(s.dtype, pd.CategoricalDtype) and list(s.cat.categories) == ["Cluster-2", "Cluster-10"]
    assert s.astype(str).tolist() == [f"Cluster-{10 if int(i) % 2 else 2}" for i in IDS]


def test_without_the_annotated_file_there_is_no_cluster_column(tmp_path):
    a, _, _ = _ingest(tmp_path, _seg_folder(tmp_path / "up" / "segmented_outputs", graphclust=False))
    assert hd_cells.GRAPHCLUST_COLUMN not in a.obs.columns


def test_a_cell_without_an_outline_stops_the_import_with_the_count(tmp_path):
    d = _seg_folder(tmp_path / "up" / "segmented_outputs")
    data = json.loads((d / "cell_segmentations.geojson").read_text())
    data["features"] = [f for f in data["features"] if f["properties"]["cell_id"] not in (3, 5)]
    (d / "cell_segmentations.geojson").write_text(json.dumps(data))
    with pytest.raises(IngestCommitError, match=r"2 of the 8 cells .* have no outline .*cellid_000000003-1"):
        _ingest(tmp_path, d)


def test_two_runs_merge_with_their_own_clusters_and_outlines(tmp_path):
    up = tmp_path / "up"
    _seg_folder(up / "A" / "outs" / "segmented_outputs")
    _seg_folder(up / "B" / "outs" / "segmented_outputs")
    mgr = IngestSessionManager(str(tmp_path / "data"))
    sid = mgr.create_session()
    for f in _files(up):
        dest = mgr.files_dir(sid) / f.relative_to(up)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(f.read_bytes())
    did = mgr._build_dataset(sid)
    a = ad.read_h5ad(next((tmp_path / "data" / "ingested" / did).glob("*.h5ad")))
    assert a.n_obs == 2 * len(IDS)
    assert a.uns["scview_spatial"]["unit"] == "cell"
    assert sorted(a.obs["library_id"].cat.categories) == ["A_cells", "B_cells"]
    assert set(a.obs[hd_cells.GRAPHCLUST_COLUMN].cat.categories) == {
        "A_cells:Cluster-2", "A_cells:Cluster-10", "B_cells:Cluster-2", "B_cells:Cluster-10"}
    assert np.isfinite(a.obsm["spatial"]).all()
    offsets, _ = cell_outlines.read_for(cell_outlines.store_path(tmp_path / "data", did, "cell"), a.obs_names)
    assert (np.diff(offsets) == 6).all()


def test_segmented_and_binned_runs_do_not_merge(tmp_path):
    up = tmp_path / "up"
    _seg_folder(up / "A" / "outs" / "segmented_outputs")
    _bin_folder(up / "B" / "binned_outputs" / "square_016um", 16)
    mgr = IngestSessionManager(str(tmp_path / "data"))
    sid = mgr.create_session()
    for f in _files(up):
        dest = mgr.files_dir(sid) / f.relative_to(up)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(f.read_bytes())
    with pytest.raises(IngestCommitError, match="Only some samples came with a Space Ranger cell segmentation"):
        mgr._build_dataset(sid)


def test_a_failed_build_leaves_no_outlines_behind(tmp_path):
    d = _seg_folder(tmp_path / "up" / "segmented_outputs")
    (d / "nucleus_segmentations.geojson").write_text('{"features": [')
    with pytest.raises(IngestCommitError, match="ends in the middle"):
        _ingest(tmp_path, d)
    assert not (tmp_path / "data" / "outlines").exists() or not list((tmp_path / "data" / "outlines").iterdir())


def test_mex_folder_reads_like_the_h5(tmp_path):
    root = tmp_path / "up"
    d = _seg_folder(root / "segmented_outputs")
    a_h5 = sc_read(d / "filtered_feature_cell_matrix.h5")
    mex = d / "filtered_feature_cell_matrix"
    mex.mkdir()
    with gzip.open(mex / "barcodes.tsv.gz", "wt") as fh:
        fh.write("\n".join(a_h5.obs_names) + "\n")
    with gzip.open(mex / "features.tsv.gz", "wt") as fh:
        for j in range(N_GENES):
            fh.write(f"ENSMUSG{j:011d}\tG{j}\tGene Expression\n")
    m = sp.coo_matrix(a_h5.X.T)
    with gzip.open(mex / "matrix.mtx.gz", "wt") as fh:
        fh.write(f"%%MatrixMarket matrix coordinate integer general\n{m.shape[0]} {m.shape[1]} {m.nnz}\n")
        for r, c, v in zip(m.row, m.col, m.data):
            fh.write(f"{r + 1} {c + 1} {int(v)}\n")
    # With the .h5 beside it the folder is the copy.
    b = build_bundle(_files(root), root=root)
    assert len(b.units) == 1 and b.units[0].format.value == "tenx_h5"
    assert any("the same matrix" in s.reason for s in b.skipped)
    (d / "filtered_feature_cell_matrix.h5").unlink()
    b = build_bundle(_files(root), root=root)
    assert len(b.units) == 1 and b.units[0].format.value == "tenx_mex" and b.units[0].label == "visium_hd_cells"
    a, _, _ = _ingest(tmp_path, d)
    assert a.n_obs == len(IDS) and spatial.technology(a)["unit"] == "cell"
    np.testing.assert_allclose(a.obsm["spatial"][0], [100.0 + 25 / 3, 50.0 + 25 / 3])


def sc_read(path):
    import scanpy as sc

    return sc.read_10x_h5(path)


def test_preflight_counts_cells_not_spots(tmp_path):
    root = tmp_path / "up"
    _seg_folder(root / "segmented_outputs")
    h = preflight.read_unit_header(build_bundle(_files(root), root=root).units[0])
    assert h.unit == "cell" and h.n_cells == len(IDS) and h.uns_bytes > 0
    assert h.side_seconds > 0
    h.side_seconds = 1000.0
    assert preflight.estimate_load(h, ingest=True).seconds[0] > 100


def test_the_router_finds_the_tissue_fact_for_a_question_on_segmentation():
    from scview.core.rag.router import named_result_facets

    assert "groups" in named_result_facets("how was the cell segmentation of this section made?")


def test_export_and_reupload_keeps_the_tissue_view_without_outlines(tmp_path, cells):
    out = tmp_path / "export.h5ad"
    cells.write_h5ad(out)
    b = ad.read_h5ad(out)
    assert spatial.adopt(b) is False
    assert spatial.technology(b) == {"technology": "visium_hd", "unit": "cell"}
    assert spatial.info(b)["libraries"][0]["images"]
    np.testing.assert_array_equal(b.obsm["spatial"], cells.obsm["spatial"])
    assert hd_cells.GRAPHCLUST_COLUMN in b.obs.columns


# ---------------------------------------------------------------------------
# Unit "cell" for the readers that mean a grid
# ---------------------------------------------------------------------------


def test_morans_i_uses_six_nearest_cells_not_the_grid(cells):
    assert sa.is_grid(cells) is False
    assert sa.neighbours_of(cells) == sa.N_NEIGHS
    assert sa.unit_of(cells) == "cell"


def test_bins_keep_their_grid():
    a = ad.AnnData(np.ones((3, 2), dtype=np.float32))
    a.uns["scview_spatial"] = {"technology": "visium_hd", "unit": "bin"}
    assert sa.is_grid(a) is True and sa.neighbours_of(a) == sa.HD_NEIGHS


def test_domains_use_delaunay_and_niches_are_allowed(cells):
    assert sd.grid_neighbours(cells) is None
    allowed = sd.niches_allowed(cells)
    assert allowed["ok"] is True and allowed["note"] is None


def test_qc_keeps_the_mito_rule_and_the_fixed_default(cells):
    t = qc_filter.Thresholds(mode="fixed", min_genes=qc_filter._FIXED_DEFAULT_MIN_GENES, max_genes=0, min_counts=0,
                             max_pct_mt=20.0)
    got, table = qc_filter.grid_rules(cells, t, {"has_mt": True})
    assert got is t and table == {"has_mt": True}


def test_cells_are_not_spot_data_so_the_doublet_step_and_spot_composition_follow(cells):
    from scview.api.v1.spot_deconvolution import _reason

    assert spatial.is_spot_data(cells) is False
    assert preflight._unit_of(cells) == "cell"
    assert _reason(cells) == deconv.NOT_SPOTS and "segmented Visium HD" in deconv.NOT_SPOTS


def test_segmented_cells_skip_the_doublet_step_with_their_own_reason(cells):
    from scview.core import pipeline

    assert spatial.doublet_skip(cells) == spatial.CELLS_DOUBLET_SKIP
    assert preflight._skips_doublets(cells) is True
    with pytest.raises(pipeline.StepNotApplicable, match="not droplets"):
        pipeline._run_doublet_detection(cells, pipeline.PipelineParams())


def test_the_copilot_fact_says_what_a_cell_is(cells):
    lines, detail = spatial.grounding_lines(cells)
    assert detail.startswith("Visium HD, Space Ranger cell segmentation")
    assert "8 cells" in lines[0] and "segmentation" in lines[0]
    assert "doublet" not in lines[0] and "not a segmented cell" not in lines[0]


def test_replay_rebuilds_positions_from_the_outlines_beside_the_matrix(tmp_path):
    d = _seg_folder(tmp_path / "up" / "segmented_outputs")
    a, _, _ = _ingest(tmp_path, d)
    script = replay_script.build_script(a)
    assert "replay_positions" in script and "import json" in script
    assert '"spatial" not in adata.obsm' in script
    out = tmp_path / "replayed.h5ad"
    import runpy
    import sys

    p = tmp_path / "replay.py"
    p.write_text(script)
    argv = sys.argv
    sys.argv = [str(p), str(d / "filtered_feature_cell_matrix.h5"), str(out)]
    try:
        runpy.run_path(str(p), run_name="__main__")
    finally:
        sys.argv = argv
    b = ad.read_h5ad(out)
    np.testing.assert_array_equal(b.obsm["spatial"], a.obsm["spatial"])
    assert list(b.obs["library_id"].astype(str)) == list(a.obs["library_id"].astype(str))
    assert list(b.obs["spaceranger_graphclust"].astype(str)) == list(a.obs["spaceranger_graphclust"].astype(str))
    assert list(b.obs["spaceranger_graphclust"].cat.categories) == list(a.obs["spaceranger_graphclust"].cat.categories)
    assert sa.is_grid(b) is False
