"""Spatial data (core/spatial.py): Space Ranger folders detected, bundled with their
matrix and attached to the spots at ingest (both position layouts, an archive, a
merge of two sections), .h5ad adoption, and spatial data kept through filtering,
merges and joint embeddings. The doublet skip: test_spatial_steps.py; the API:
test_spatial_api.py."""

from __future__ import annotations

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

from scview.core import provenance, spatial
from scview.core.ingestion import IngestSessionManager
from scview.core.ingestion.bundling import SideKind, build_bundle
from scview.core.ingestion.detection import FileKind, detect_file
from scview.core.ingestion.merge import merge_units
from scview.core.ingestion.validation import validate_bundle
from tests.test_ingestion_session import _barcode

N_SPOTS = 20
N_GENES = 10


def _h5(path: Path, n: int = N_SPOTS, n_genes: int = N_GENES, seed: int = 0) -> Path:
    """A Cell Ranger v3 filtered matrix (genes x spots CSC) of barcodes _barcode(0..n)."""
    counts = np.random.default_rng(seed).poisson(3.0, size=(n, n_genes)) + 1
    csc = sp.csc_matrix(counts.T.astype(np.int32))
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as f:
        g = f.create_group("matrix")
        g["data"], g["indices"], g["indptr"] = csc.data, csc.indices.astype(np.int64), csc.indptr
        g["shape"] = np.array([n_genes, n], dtype=np.int32)
        g["barcodes"] = np.array([_barcode(i).encode() for i in range(n)])
        ft = g.create_group("features")
        ft["id"] = np.array([f"ENSG{j:011d}".encode() for j in range(n_genes)])
        ft["name"] = np.array([f"G{j}".encode() for j in range(n_genes)])
        ft["feature_type"] = np.array([b"Gene Expression"] * n_genes)
        ft["genome"] = np.array([b"GRCh38"] * n_genes)
    return path


def _png(path: Path, w: int = 4, h: int = 3, alpha: bool = False) -> Path:
    a = np.zeros((h, w, 4 if alpha else 3), dtype=np.uint8)
    a[..., 0] = np.arange(w)[None, :] * 50
    a[..., 1] = np.arange(h)[:, None] * 80
    if alpha:
        a[..., 3] = 128
    Image.fromarray(a).save(path, format="PNG")
    return path


def _positions(n: int = N_SPOTS, extra_out: int = 3) -> list[tuple]:
    """(barcode, in_tissue, array_row, array_col, pxl_row, pxl_col): the n matrix spots
    in tissue, ``extra_out`` more barcodes off the tissue."""
    rows = [(_barcode(i), 1, i // 5, 2 * (i % 5) + (i // 5) % 2, 100 + 10 * (i // 5), 200 + 7 * i)
            for i in range(n)]
    rows += [(_barcode(n + k), 0, 9, k, 5, 5) for k in range(extra_out)]
    return rows


def _spatial_dir(d: Path, layout: str = "2.x", hires: bool = True, scalefactors: bool = True) -> Path:
    """A Space Ranger spatial/ folder: positions (1.x: no header, 2.x: header), scale
    factors, a 4 x 3 lowres PNG (+ hires), and files scView ignores."""
    d.mkdir(parents=True, exist_ok=True)
    rows = _positions()
    if layout == "1.x":
        (d / "tissue_positions_list.csv").write_text("".join(",".join(map(str, r)) + "\n" for r in rows))
    else:
        head = "barcode,in_tissue,array_row,array_col,pxl_row_in_fullres,pxl_col_in_fullres\n"
        (d / "tissue_positions.csv").write_text(head + "".join(",".join(map(str, r)) + "\n" for r in rows))
    if scalefactors:
        (d / "scalefactors_json.json").write_text(json.dumps({
            "spot_diameter_fullres": 89.5, "tissue_hires_scalef": 0.17, "tissue_lowres_scalef": 0.05,
            "fiducial_diameter_fullres": 144.6}))
    _png(d / "tissue_lowres_image.png", alpha=True)
    if hires:
        _png(d / "tissue_hires_image.png", w=8, h=6)
    (d / "aligned_fiducials.jpg").write_bytes(b"\xff\xd8\xff\xe0 not really a jpeg")
    (d / "spatial_enrichment.csv").write_text("Feature ID,Feature Name,Feature Type,I,P value,Adjusted p value\n"
                                              "ENSG1,G1,Gene Expression,0.8,0.0,0.0\n")
    return d


def _tar(src: Path, dest: Path) -> Path:
    with tarfile.open(dest, "w:gz") as tf:
        tf.add(src, arcname="spatial")
    return dest


def _files(d: Path) -> list[Path]:
    return sorted(p for p in d.rglob("*") if p.is_file())


# ---------------------------------------------------------------------------
# Detection and bundling
# ---------------------------------------------------------------------------


def test_spatial_files_are_detected(tmp_path):
    d = _spatial_dir(tmp_path / "spatial")
    parts = {p.name: detect_file(p) for p in _files(d)}
    assert parts["tissue_positions.csv"].kind == FileKind.spatial
    assert parts["tissue_positions.csv"].details["part"] == "positions"
    assert parts["scalefactors_json.json"].details["part"] == "scalefactors"
    assert parts["tissue_lowres_image.png"].details["part"] == "lowres"
    assert parts["tissue_hires_image.png"].details["part"] == "hires"
    old = _spatial_dir(tmp_path / "old" / "spatial", layout="1.x")
    det = detect_file(old / "tissue_positions_list.csv")
    assert det.kind == FileKind.spatial and "no header" in det.reason
    arc = detect_file(_tar(d, tmp_path / "s1_spatial.tar.gz"))
    assert arc.kind == FileKind.spatial and arc.details["part"] == "archive"
    # An unrelated PNG / tar stays unknown.
    assert detect_file(_png(tmp_path / "logo.png")).kind == FileKind.unknown


def test_spaceranger_outs_folder_bundles_into_one_sample(tmp_path):
    outs = tmp_path / "LN" / "outs"
    _h5(outs / "filtered_feature_bc_matrix.h5")
    _spatial_dir(outs / "spatial")
    (outs / "web_summary.html").write_text("<html></html>")
    bundle = build_bundle(_files(tmp_path), root=tmp_path)
    assert len(bundle.units) == 1 and bundle.complete and bundle.unsupported is None
    (u,) = bundle.units
    assert u.label == "LN"
    side = sorted((f.spatial_part, f.side_kind) for f in u.side_files)
    assert side == [(p, SideKind.spatial) for p in ("hires", "lowres", "positions", "scalefactors")]
    skipped = {Path(s.path).name for s in bundle.skipped}
    assert {"aligned_fiducials.jpg", "spatial_enrichment.csv", "web_summary.html"} <= skipped
    assert validate_bundle(bundle).ok


def test_h5_and_archive_pairs_match_by_prefix(tmp_path):
    for name in ("A", "B"):
        _h5(tmp_path / f"{name}_filtered_feature_bc_matrix.h5")
        _tar(_spatial_dir(tmp_path / f"src{name}" / "spatial"), tmp_path / f"{name}_spatial.tar.gz")
    bundle = build_bundle(sorted(tmp_path.glob("*.h5")) + sorted(tmp_path.glob("*.tar.gz")))
    assert [u.label for u in bundle.units] == ["A", "B"]
    for u in bundle.units:
        assert [f.name for f in u.side_files] == [f"{u.label}_spatial.tar.gz"]


def test_missing_positions_is_an_error_and_missing_scalefactors_a_warning(tmp_path):
    d = _spatial_dir(tmp_path / "spatial", scalefactors=False)
    h5 = _h5(tmp_path / "filtered_feature_bc_matrix.h5")
    rep = validate_bundle(build_bundle([h5, d / "tissue_positions.csv", d / "tissue_lowres_image.png"]))
    assert rep.ok and any(i.code == "spatial_no_scalefactors" for i in rep.issues)
    rep = validate_bundle(build_bundle([h5, d / "tissue_lowres_image.png"]))
    assert not rep.ok and any(i.code == "spatial_no_positions" for i in rep.issues)


def test_visium_hd_and_xenium_are_refused_with_a_sentence(tmp_path):
    hd = tmp_path / "hd" / "outs" / "binned_outputs" / "square_008um"
    _h5(hd / "filtered_feature_bc_matrix.h5")
    rep = validate_bundle(build_bundle(_files(tmp_path / "hd"), root=tmp_path / "hd"))
    assert not rep.ok and any(i.code == "spatial_unsupported" and "Visium HD" in i.message for i in rep.issues)
    xe = tmp_path / "xe"
    _h5(xe / "cell_feature_matrix.h5")
    (xe / "experiment.xenium").write_text('{"run_name": "x"}')
    (xe / "transcripts.csv.gz").write_bytes(b"")
    rep = validate_bundle(build_bundle(_files(xe), root=xe))
    assert not rep.ok and any("Xenium" in i.message for i in rep.issues)


def test_hd_barcodes_in_positions_are_refused(tmp_path):
    p = tmp_path / "tissue_positions.csv"
    p.write_text("barcode,in_tissue,array_row,array_col,pxl_row_in_fullres,pxl_col_in_fullres\n"
                 "s_008um_00000_00001-1,1,0,1,5,5\n")
    with pytest.raises(spatial.SpatialError, match="Visium HD"):
        spatial.read_spaceranger([("positions", str(p), p.name)])


# ---------------------------------------------------------------------------
# Ingest (commit)
# ---------------------------------------------------------------------------


@pytest.fixture()
def mgr(tmp_data_dir: Path) -> IngestSessionManager:
    return IngestSessionManager(data_dir=str(tmp_data_dir))


def _stage(mgr, sid: str, src: Path, root: Path) -> None:
    for p in _files(src):
        dest = mgr.files_dir(sid) / p.relative_to(root)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(p.read_bytes())


def _committed(mgr, sid: str, tmp_data_dir: Path) -> ad.AnnData:
    did = mgr._build_dataset(sid)
    return ad.read_h5ad(next((tmp_data_dir / "ingested" / did).glob("*.h5ad")))


@pytest.mark.parametrize("layout", ["1.x", "2.x"])
def test_ingest_attaches_positions_images_and_scalefactors(mgr, tmp_data_dir, tmp_path, layout):
    outs = tmp_path / "src" / "LN" / "outs"
    _h5(outs / "filtered_feature_bc_matrix.h5")
    _spatial_dir(outs / "spatial", layout=layout)
    sid = mgr.create_session()
    _stage(mgr, sid, tmp_path / "src", tmp_path / "src")
    a = _committed(mgr, sid, tmp_data_dir)
    assert a.n_obs == N_SPOTS  # off-tissue barcodes are not cells
    xy = a.obsm["spatial"]
    assert xy.shape == (N_SPOTS, 2) and xy.dtype == np.float64
    expect = {r[0]: (r[5], r[4]) for r in _positions() if r[1] == 1}  # x = pxl_col, y = pxl_row
    assert [tuple(v) for v in xy] == [expect[b] for b in a.obs_names]
    assert list(a.obs["library_id"].cat.categories) == ["LN"]
    assert a.obs["array_row"].dtype.kind == "i"
    lib = a.uns["spatial"]["LN"]
    assert lib["images"]["lowres"].shape == (3, 4, 3) and lib["images"]["lowres"].dtype == np.uint8
    assert lib["images"]["hires"].shape == (6, 8, 3)
    assert float(lib["scalefactors"]["spot_diameter_fullres"]) == 89.5
    assert lib["metadata"]["source"] == "spaceranger"
    assert dict(a.uns["scview_spatial"]) == {"technology": "visium", "unit": "spot"}
    ingest = [s for s in provenance.read_provenance(a)["history"] if s["step"] == "ingest"][0]
    sp_params = ingest["params"]["spatial"]
    assert sp_params["source"] == "spaceranger" and sp_params["spaceranger_version"] is None
    assert sp_params["libraries"] == {"LN": N_SPOTS}


def test_ingest_of_h5_and_archive(mgr, tmp_data_dir, tmp_path):
    sid = mgr.create_session()
    files = mgr.files_dir(sid)
    _h5(files / "S1_filtered_feature_bc_matrix.h5")
    _tar(_spatial_dir(tmp_path / "s" / "spatial", hires=False), files / "S1_spatial.tar.gz")
    a = _committed(mgr, sid, tmp_data_dir)
    assert set(a.uns["spatial"]) == {"S1"} and set(a.uns["spatial"]["S1"]["images"]) == {"lowres"}
    assert np.isfinite(a.obsm["spatial"]).all()


def test_a_spot_missing_from_the_positions_is_an_error_naming_the_count(mgr, tmp_path):
    sid = mgr.create_session()
    files = mgr.files_dir(sid)
    _h5(files / "filtered_feature_bc_matrix.h5", n=N_SPOTS + 2)
    d = _spatial_dir(files / "spatial")
    from scview.core.ingestion.session import IngestCommitError

    with pytest.raises(IngestCommitError, match="2 of the 22 spots"):
        mgr._build_dataset(sid)
    assert d.exists()


def test_merge_of_two_sections_keeps_each_library(mgr, tmp_data_dir, tmp_path):
    src = tmp_path / "src"
    for name, layout in (("secA", "1.x"), ("secB", "2.x")):
        _h5(src / name / "outs" / "filtered_feature_bc_matrix.h5", seed=len(name))
        _spatial_dir(src / name / "outs" / "spatial", layout=layout)
    sid = mgr.create_session()
    _stage(mgr, sid, src, src)
    a = _committed(mgr, sid, tmp_data_dir)
    assert a.n_obs == 2 * N_SPOTS
    assert list(a.obs["library_id"].cat.categories) == ["secA", "secB"] == sorted(a.uns["spatial"])
    assert (a.obs["library_id"].astype(str) == a.obs["sample"].astype(str)).all()
    assert np.isfinite(a.obsm["spatial"]).all()
    rows = (a.obs["sample"] == "secB").to_numpy()
    assert a.obsm["spatial"][rows][0].tolist() == [200.0, 100.0]


def test_merge_of_spatial_and_plain_samples_is_refused(mgr, tmp_path):
    sid = mgr.create_session()
    files = mgr.files_dir(sid)
    _h5(files / "A" / "outs" / "filtered_feature_bc_matrix.h5")
    _spatial_dir(files / "A" / "outs" / "spatial")
    _h5(files / "B" / "outs" / "filtered_feature_bc_matrix.h5")
    from scview.core.ingestion.session import IngestCommitError

    with pytest.raises(IngestCommitError, match="Only some samples"):
        mgr._build_dataset(sid)


def test_preflight_counts_the_images(tmp_path):
    from scview.core import preflight

    outs = tmp_path / "outs"
    _h5(outs / "filtered_feature_bc_matrix.h5")
    _spatial_dir(outs / "spatial")
    bundle = build_bundle(_files(tmp_path), root=tmp_path)
    plain = preflight.read_header(outs / "filtered_feature_bc_matrix.h5", "10x_h5")
    h = preflight.read_unit_header(bundle.units[0])
    assert h.uns_bytes - plain.uns_bytes == 4 * 3 * 3 + 8 * 6 * 3
    assert h.obsm_bytes - plain.obsm_bytes == 16 * N_SPOTS


# ---------------------------------------------------------------------------
# Adoption of a spatial .h5ad, merges, subsets
# ---------------------------------------------------------------------------


def _squidpy_like(n: int = 12, key: str = "spatial", lib: str | None = "V1_LN") -> ad.AnnData:
    rng = np.random.default_rng(1)
    a = ad.AnnData(X=sp.csr_matrix(rng.poisson(2.0, size=(n, 6)).astype(np.float32)))
    a.obs_names = [_barcode(i) for i in range(n)]
    a.var_names = [f"G{j}" for j in range(6)]
    a.obsm[key] = np.column_stack([np.arange(n) * 10, np.arange(n) * 5]).astype(np.int64)
    a.obs["array_row"] = np.arange(n) // 4
    a.obs["array_col"] = np.arange(n) % 4
    if lib:
        a.uns["spatial"] = {lib: {"images": {"lowres": np.full((5, 7, 3), 200, dtype=np.uint8)},
                                  "scalefactors": {"tissue_lowres_scalef": 0.1,
                                                   "spot_diameter_fullres": 50.0},
                                  "metadata": {"chemistry_description": "Spatial 3' v1"}}}
    return a


def test_adopt_adds_library_and_layout():
    a = _squidpy_like()
    assert spatial.adopt(a)
    assert a.obsm["spatial"].dtype == np.float64
    assert a.obs["library_id"].cat.categories.tolist() == ["V1_LN"]
    assert dict(a.uns["scview_spatial"]) == {"technology": "visium", "unit": "spot"}
    assert a.uns["spatial"]["V1_LN"]["metadata"] == {"chemistry_description": "Spatial 3' v1"}
    assert not spatial.adopt(a)  # already in the layout


def test_adopt_x_spatial_without_uns_is_cells():
    a = _squidpy_like(key="X_spatial", lib=None)
    del a.obs["array_row"]
    spatial.adopt(a)
    assert "spatial" in a.obsm and a.obs["library_id"].tolist() == ["section1"] * a.n_obs
    assert dict(a.uns["scview_spatial"]) == {"technology": "cells", "unit": "cell"}
    info = spatial.info(a)
    assert info["unit"] == "cell" and info["libraries"][0]["images"] == {}
    assert info["libraries"][0]["bounds"] == [0.0, 0.0, 110.0, 55.0]


def test_adaptor_adopts_on_open_and_keeps_x_spatial_out_of_embeddings(tmp_path):
    from scview.core.anndata_adaptor import AnnDataAdaptor

    a = _squidpy_like(key="X_spatial")
    a.obsm["X_umap"] = np.zeros((a.n_obs, 2))
    p = tmp_path / "v.h5ad"
    a.write_h5ad(p)
    ada = AnnDataAdaptor(str(p))
    assert [e["name"] for e in ada.available_embeddings()] == ["X_umap"]
    assert ada.get_embedding("spatial").shape == (a.n_obs, 2)
    assert ada.adata.obs["library_id"].iloc[0] == "V1_LN"


def test_merge_units_keeps_libraries_of_spatial_h5ads():
    m = merge_units({"a": _squidpy_like(), "b": _squidpy_like(lib="other")}, sample_label="sample")
    assert sorted(m.uns["spatial"]) == ["a", "b"]
    assert m.obs["library_id"].astype(str).tolist() == ["a"] * 12 + ["b"] * 12
    assert m.uns["spatial"]["b"]["images"]["lowres"].shape == (5, 7, 3)
    plain = _squidpy_like(lib=None)
    del plain.obsm["spatial"]
    m2 = merge_units({"a": _squidpy_like(), "b": plain}, sample_label="sample")
    assert "spatial" not in m2.obsm and "spatial" not in m2.uns


def test_filtering_keeps_coordinates_with_their_cells():
    from scview.core.pipeline import PipelineParams, run_pipeline

    a = _squidpy_like(n=30)
    spatial.adopt(a)
    dense = a.X.toarray()
    dense[[3, 7], :] = 0  # two spots with no counts: filtered out
    a.X = sp.csr_matrix(dense)
    before = pd.DataFrame(a.obsm["spatial"], index=a.obs_names)
    params = PipelineParams(min_genes=1, min_cells=1, max_pct_mt=100.0, qc_threshold_mode="fixed")
    out, res = run_pipeline(a, ["qc_metrics", "filtering"], params)
    assert "filtering" in res.steps_run and out.n_obs == 28
    assert np.array_equal(out.obsm["spatial"], before.loc[out.obs_names].to_numpy())
    assert out.uns["spatial"]["V1_LN"]["images"]["lowres"].shape == (5, 7, 3)
    assert out.obs["library_id"].cat.categories.tolist() == ["V1_LN"]


def test_integration_unit_carries_spatial():
    from scview.core.integration import build_unit

    a = _squidpy_like()
    spatial.adopt(a)
    u = build_unit(a, dataset_id="d", name="n", label="L", fallback_paths=[])
    assert u.adata.obsm["spatial"].shape == (12, 2) and "V1_LN" in u.adata.uns["spatial"]
