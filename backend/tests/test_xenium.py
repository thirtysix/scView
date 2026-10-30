"""Xenium output folders: bundling, the dataset, the image (core/xenium.py)."""

from __future__ import annotations

import asyncio
import gzip
import io
import json
import zipfile
from pathlib import Path
from types import SimpleNamespace

import anndata as ad
import h5py
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp
import tifffile
from PIL import Image, features

from scview.core import preflight, seq_metrics, spatial, xenium
from scview.core.ingestion.bundling import SideKind, UnitFormat, build_bundle
from scview.core.ingestion.detection import FileKind, detect_file
from scview.core.ingestion.loading import IngestLoadError, load_unit
from scview.core.ingestion.session import IngestCommitError, IngestOptions, IngestSessionManager
from scview.core.ingestion.validation import validate_bundle

N_CELLS, N_GENES, N_CONTROLS = 40, 12, 5


def _counts(n: int = N_CELLS, n_features: int = N_GENES + N_CONTROLS, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    counts = rng.poisson(4.0, size=(n, n_features)).astype(np.int32)
    counts[:, N_GENES:] = rng.binomial(1, 0.05, size=(n, n_features - N_GENES))
    return counts


def _ids(n: int, kind: str) -> list[str]:
    return [str(i + 1) if kind == "int" else f"aaaaaaa{chr(97 + i % 26)}{i:03d}-1" for i in range(n)]


def _matrix(path: Path, counts: np.ndarray, ids: list[str]) -> Path:
    """A Xenium cell_feature_matrix.h5: genes, then controls (Cell Ranger's v3 layout)."""
    n, n_f = counts.shape
    csc = sp.csc_matrix(counts.T)
    path.parent.mkdir(parents=True, exist_ok=True)
    types = [b"Gene Expression"] * N_GENES + [b"Negative Control Probe", b"Blank Codeword"] * (n_f - N_GENES)
    with h5py.File(path, "w") as f:
        g = f.create_group("matrix")
        g["data"], g["indices"], g["indptr"] = csc.data, csc.indices.astype(np.int64), csc.indptr
        g["shape"] = np.array([n_f, n], dtype=np.int32)
        g["barcodes"] = np.array([i.encode() for i in ids])
        ft = g.create_group("features")
        ft["id"] = np.array([f"ENSG{j:011d}".encode() for j in range(n_f)])
        ft["name"] = np.array([(f"G{j}" if j < N_GENES else f"NegControl_{j}").encode() for j in range(n_f)])
        ft["feature_type"] = np.array(types[:n_f])
        ft["genome"] = np.array([b"Unknown"] * n_f)
    return path


def _mex(folder: Path, counts: np.ndarray, ids: list[str]) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    n, n_f = counts.shape
    coo = sp.coo_matrix(counts.T)
    with gzip.open(folder / "matrix.mtx.gz", "wt") as fh:
        fh.write("%%MatrixMarket matrix coordinate integer general\n")
        fh.write(f"{n_f} {n} {coo.nnz}\n")
        order = np.lexsort((coo.row, coo.col))
        for r, c, v in zip(coo.row[order], coo.col[order], coo.data[order]):
            fh.write(f"{r + 1} {c + 1} {v}\n")
    with gzip.open(folder / "barcodes.tsv.gz", "wt") as fh:
        fh.write("\n".join(ids) + "\n")
    with gzip.open(folder / "features.tsv.gz", "wt") as fh:
        for j in range(n_f):
            kind = "Gene Expression" if j < N_GENES else "Negative Control Probe"
            name = f"G{j}" if j < N_GENES else f"NegControl_{j}"
            fh.write(f"ENSG{j:011d}\t{name}\t{kind}\n")


def _cells(counts: np.ndarray, ids: list[str]) -> pd.DataFrame:
    genes, controls = counts[:, :N_GENES].sum(axis=1), counts[:, N_GENES:].sum(axis=1)
    return pd.DataFrame({
        "cell_id": ids, "x_centroid": np.arange(len(ids)) * 10.0, "y_centroid": np.arange(len(ids)) * 5.0 + 1.0,
        "transcript_counts": genes, "control_probe_counts": controls, "control_codeword_counts": 0,
        "total_counts": genes + controls, "cell_area": 100.0, "nucleus_area": 30.0})


EXPERIMENT = {"run_name": "r", "region_name": "Replicate 1", "panel_name": "Breast TME", "panel_organism": "Human",
              "panel_tissue_type": "Breast", "pixel_size": 0.2125, "analysis_sw_version": "Xenium-1.0.1"}


def _bundle(root: Path, *, kind: str = "int", shuffle: bool = False, experiment: dict | None = None,
            table: str = "parquet", matrix: str = "h5", n: int = N_CELLS, seed: int = 0) -> np.ndarray:
    """A Xenium folder; returns the counts in the matrix's row order."""
    root.mkdir(parents=True, exist_ok=True)
    counts, ids = _counts(n, seed=seed), _ids(n, kind)
    (root / "experiment.xenium").write_text(json.dumps(EXPERIMENT if experiment is None else experiment))
    if matrix == "h5":
        _matrix(root / "cell_feature_matrix.h5", counts, ids)
    else:
        _mex(root / "cell_feature_matrix", counts, ids)
    cells = _cells(counts, ids)
    if shuffle:  # the table's rows in another order than the matrix's: the id says which is which
        cells = cells.iloc[np.random.default_rng(1).permutation(n)]
    if table == "parquet":
        cells.to_parquet(root / "cells.parquet")
    else:
        cells.to_csv(root / "cells.csv.gz", index=False)
    (root / "metrics_summary.csv").write_text("run_name,region_area,num_cells_detected\n,41118695.5,40\n")
    return counts


def _files(d: Path) -> list[Path]:
    return sorted(p for p in d.rglob("*") if p.is_file())


def _pyramid(path: Path, sizes=((256, 320), (128, 160), (64, 80)), seed: int = 0) -> np.ndarray:
    """An uncompressed OME-TIFF pyramid; returns level 0."""
    base = np.random.default_rng(seed).integers(1, 4000, size=sizes[0], dtype=np.uint16)
    with tifffile.TiffWriter(path, ome=True, bigtiff=True) as tif:
        tif.write(base, subifds=len(sizes) - 1, tile=(64, 64), metadata={"axes": "YX"})
        for i, _ in enumerate(sizes[1:], 1):
            step = 2**i
            tif.write(base[::step, ::step], subfiletype=1, tile=(64, 64))
    return base


# ---------------------------------------------------------------------------
# Detection and bundling
# ---------------------------------------------------------------------------


def test_files_are_detected_and_a_folder_is_one_unit(tmp_path):
    root = tmp_path / "run" / "outs"
    _bundle(root)
    assert detect_file(root / "experiment.xenium").kind == FileKind.xenium
    assert detect_file(root / "cells.parquet").details["part"] == xenium.PART_CELLS
    b = build_bundle(_files(tmp_path), root=tmp_path)
    assert len(b.units) == 1 and b.complete and not b.is_merge and b.unsupported is None
    u = b.units[0]
    assert u.format == UnitFormat.xenium and u.label == "Replicate 1"
    assert [f.name for f in u.files] == ["cell_feature_matrix.h5"]
    assert {f.spatial_part for f in u.side_files} == {"experiment", "cells", "metrics"}
    assert all(f.side_kind == SideKind.xenium for f in u.side_files)
    assert validate_bundle(b).ok


def test_a_plain_upload_of_the_files_is_one_unit_named_by_the_region(tmp_path):
    _bundle(tmp_path)
    b = build_bundle(_files(tmp_path))
    assert [u.format for u in b.units] == [UnitFormat.xenium] and b.complete


@pytest.mark.parametrize("drop,named", [
    ("experiment.xenium", "experiment.xenium"),
    ("cell_feature_matrix.h5", "cell_feature_matrix.h5"),
    ("cells.parquet", "cells.parquet"),
])
def test_a_missing_part_is_an_incomplete_unit_with_one_sentence_naming_it(tmp_path, drop, named):
    root = tmp_path / "outs"
    _bundle(root)
    (root / drop).unlink()
    b = build_bundle(_files(tmp_path), root=tmp_path)
    assert len(b.units) == 1 and not b.units[0].complete and not b.complete
    rep = validate_bundle(b)
    assert not rep.ok
    errs = [i.message for i in rep.issues if i.code == "xenium_incomplete"]
    assert len(errs) == 1 and named in errs[0] and "outs" in errs[0]


def test_heavy_files_are_skipped_with_a_reason_and_only_the_named_ones_are_read(tmp_path):
    root = tmp_path / "outs"
    _bundle(root)
    for name in ("transcripts.parquet", "transcripts.csv.gz", "transcripts.zarr.zip", "morphology.ome.tif",
                 "cell_boundaries.parquet", "cell_boundaries.csv.gz", "nucleus_boundaries.parquet",
                 "nucleus_boundaries.csv.gz", "cells.zarr.zip", "cell_feature_matrix.zarr.zip",
                 "analysis.zarr.zip", "analysis_summary.html", "gene_panel.json", "notes.txt"):
        (root / name).write_bytes(b"x")
    (root / "analysis" / "clustering").mkdir(parents=True)
    (root / "analysis" / "clustering" / "clusters.csv").write_text("Barcode,Cluster\n1,1\n")
    (root / "aux_outputs").mkdir()
    (root / "aux_outputs" / "overlay.ome.tif").write_bytes(b"x")
    b = build_bundle(_files(tmp_path), root=tmp_path)
    assert len(b.units) == 1 and b.complete and not [i for i in b.issues if "could not be recognised" in i]
    skipped = {Path(s.path).name for s in b.skipped}
    assert {"transcripts.parquet", "gene_panel.json", "clusters.csv", "overlay.ome.tif", "notes.txt"} <= skipped
    assert all(s.reason for s in b.skipped)


def test_the_mex_folder_and_csv_table_are_taken_when_there_is_no_h5_or_parquet(tmp_path):
    counts = _bundle(tmp_path / "outs", matrix="mex", table="csv", kind="str")
    b = build_bundle(_files(tmp_path), root=tmp_path)
    assert b.complete and len(b.units) == 1
    a = load_unit(b.units[0])
    assert a.n_vars == N_GENES
    assert np.array_equal(np.asarray(a.X.sum(axis=1)).ravel().astype(int), counts[:, :N_GENES].sum(axis=1))
    assert a.obs["total_counts"].iloc[0] == counts[0].sum()


def test_h5_beats_mex_and_parquet_beats_csv(tmp_path):
    root = tmp_path / "outs"
    counts = _bundle(root)
    _mex(root / "cell_feature_matrix", counts, _ids(N_CELLS, "int"))
    _cells(counts, _ids(N_CELLS, "int")).to_csv(root / "cells.csv.gz", index=False)
    b = build_bundle(_files(tmp_path), root=tmp_path)
    u = b.units[0]
    assert [f.name for f in u.files] == ["cell_feature_matrix.h5"]
    assert next(f.name for f in u.side_files if f.spatial_part == "cells") == "cells.parquet"
    assert {"matrix.mtx.gz", "cells.csv.gz"} <= {Path(s.path).name for s in b.skipped}


def test_image_preference(tmp_path):
    root = tmp_path / "outs"
    _bundle(root)
    (root / "morphology_focus").mkdir()
    for name in ("morphology_focus_0000.ome.tif", "morphology_focus_0001.ome.tif"):
        (root / "morphology_focus" / name).write_bytes(b"II*\0")
    (root / "morphology_mip.ome.tif").write_bytes(b"II*\0")
    b = build_bundle(_files(tmp_path), root=tmp_path)
    img = [f for f in b.units[0].side_files if f.spatial_part == "image"]
    assert [f.name for f in img] == ["morphology_focus_0000.ome.tif"]
    assert {"morphology_focus_0001.ome.tif", "morphology_mip.ome.tif"} <= {Path(s.path).name for s in b.skipped}


def test_a_zipped_outs_folder_is_one_sentence(tmp_path):
    z = tmp_path / "Xenium_outs.zip"
    with zipfile.ZipFile(z, "w") as zf:
        zf.writestr("outs/experiment.xenium", "{}")
    b = build_bundle([z])
    rep = validate_bundle(b)
    assert not rep.ok
    assert any(i.code == "spatial_unsupported" and "drop the folder" in i.message for i in rep.issues)
    other = tmp_path / "other.zip"
    with zipfile.ZipFile(other, "w") as zf:
        zf.writestr("a.txt", "x")
    assert build_bundle([other]).unsupported is None


def test_visium_hd_stays_refused():
    assert "Visium HD" in spatial.unsupported(["x/binned_outputs/square_008um/matrix.h5"])
    assert spatial.unsupported(["x/experiment.xenium"]) is None


def test_metrics_summary_of_xenium_is_not_cell_rangers(tmp_path):
    _bundle(tmp_path / "outs")
    b = build_bundle(_files(tmp_path), root=tmp_path)
    a = load_unit(b.units[0])
    assert seq_metrics.UNS_KEY not in a.uns
    assert a.uns["spatial"]["Replicate 1"]["metadata"]["metrics"]["num_cells_detected"] == 40


# ---------------------------------------------------------------------------
# The dataset
# ---------------------------------------------------------------------------


def _load(root: Path, tmp_path: Path) -> ad.AnnData:
    return load_unit(build_bundle(_files(tmp_path), root=tmp_path).units[0])


@pytest.mark.parametrize("kind", ["int", "str"])
def test_cells_are_matched_by_id_not_by_row(tmp_path, kind):
    counts = _bundle(tmp_path / "outs", kind=kind, shuffle=True)
    a = _load(tmp_path / "outs", tmp_path)
    assert a.shape == (N_CELLS, N_GENES)
    assert a.obs_names.tolist() == _ids(N_CELLS, kind)
    assert np.array_equal(a.obs["transcript_counts"].to_numpy(), counts[:, :N_GENES].sum(axis=1))
    assert np.array_equal(a.obsm["spatial"][:, 0], np.arange(N_CELLS) * 10.0)
    assert np.array_equal(a.obsm["spatial"][:, 1], np.arange(N_CELLS) * 5.0 + 1.0)


def test_bytes_cell_ids_of_a_parquet_file(tmp_path):
    counts = _bundle(tmp_path / "outs", kind="str")
    cells = _cells(counts, _ids(N_CELLS, "str"))
    cells["cell_id"] = [i.encode() for i in cells["cell_id"]]
    cells.to_parquet(tmp_path / "outs" / "cells.parquet")
    assert _load(tmp_path / "outs", tmp_path).n_obs == N_CELLS


def test_a_mismatch_is_an_error_with_the_count(tmp_path):
    counts = _bundle(tmp_path / "outs")
    _cells(counts, _ids(N_CELLS, "int")).iloc[:-3].to_parquet(tmp_path / "outs" / "cells.parquet")
    with pytest.raises(IngestLoadError, match="3 of the 40 cells"):
        _load(tmp_path / "outs", tmp_path)
    more = _cells(_counts(N_CELLS + 2), _ids(N_CELLS + 2, "int"))
    more.to_parquet(tmp_path / "outs" / "cells.parquet")
    with pytest.raises(IngestLoadError, match="2 of the 42 cells"):
        _load(tmp_path / "outs", tmp_path)


def test_the_dataset_holds_genes_only_with_the_panel_recorded(tmp_path):
    _bundle(tmp_path / "outs")
    a = _load(tmp_path / "outs", tmp_path)
    assert a.n_vars == N_GENES and all(g.startswith("G") for g in a.var_names)
    rec = a.uns["scview_spatial"]
    assert rec["technology"] == "xenium" and rec["unit"] == "cell" and rec["targeted_panel"] is True
    assert rec["panel_name"] == "Breast TME" and rec["n_panel_genes"] == N_GENES
    assert rec["pixel_size"] == 0.2125 and rec["organism"] == "human"
    meta = a.uns["spatial"]["Replicate 1"]["metadata"]
    assert meta["source"] == "xenium" and meta["analysis_sw_version"] == "Xenium-1.0.1"
    assert meta["files"]["cells"].endswith("cells.parquet")
    assert a.obs["library_id"].cat.categories.tolist() == ["Replicate 1"]
    assert spatial.technology(a)["technology"] == "xenium" and not spatial.is_spot_data(a)
    want = 100 * (a.obs["total_counts"] - a.obs["transcript_counts"]) / a.obs["total_counts"]
    assert np.allclose(a.obs["pct_control"], want.fillna(0))
    info = spatial.info(a)
    assert info["unit"] == "cell" and info["libraries"][0]["spot_diameter_fullres"] is None
    # A round trip keeps it all.
    a.write_h5ad(tmp_path / "x.h5ad")
    b = ad.read_h5ad(tmp_path / "x.h5ad")
    assert spatial.technology(b)["technology"] == "xenium"
    assert b.uns["scview_spatial"]["n_panel_genes"] == N_GENES


def test_newer_cell_table_columns_are_kept(tmp_path):
    counts = _bundle(tmp_path / "outs", kind="str")
    cells = _cells(counts, _ids(N_CELLS, "str"))
    cells["unassigned_codeword_counts"] = 1
    cells["deprecated_codeword_counts"] = 0
    cells["genomic_control_counts"] = 0
    cells["nucleus_count"] = 1
    cells["segmentation_method"] = "Segmented by boundary stain (ATP1A1+CD45+E-Cadherin)"
    cells["total_counts"] = cells["total_counts"] + 1
    cells.to_parquet(tmp_path / "outs" / "cells.parquet")
    a = _load(tmp_path / "outs", tmp_path)
    for col in ("unassigned_codeword_counts", "genomic_control_counts", "nucleus_count", "segmentation_method"):
        assert col in a.obs.columns
    assert isinstance(a.obs["segmentation_method"].dtype, pd.CategoricalDtype)
    assert "cell_id" not in a.obs.columns


def test_the_header_counts_genes_not_controls(tmp_path):
    _bundle(tmp_path / "outs")
    h = preflight.read_unit_header(build_bundle(_files(tmp_path), root=tmp_path).units[0])
    assert (h.n_cells, h.n_genes) == (N_CELLS, N_GENES) and h.x.n_genes == N_GENES
    _bundle(tmp_path / "mex" / "outs", matrix="mex")
    h = preflight.read_unit_header(build_bundle(_files(tmp_path / "mex"), root=tmp_path / "mex").units[0])
    assert (h.n_cells, h.n_genes) == (N_CELLS, N_GENES)


def _commit(tmp_path: Path, src: Path, **opts) -> ad.AnnData:
    mgr = IngestSessionManager(str(tmp_path / "data"))
    sid = mgr.create_session()
    for p in _files(src):
        dest = mgr.files_dir(sid) / p.relative_to(src)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(p.read_bytes())
    if opts:
        mgr.set_options(sid, IngestOptions(**opts))

    class DM:
        async def load_dataset(self, did):
            return None

    did = asyncio.run(mgr.commit(sid, DM()))
    return ad.read_h5ad(next((tmp_path / "data" / "ingested" / did).glob("*.h5ad")))


def test_commit_in_blocks_gives_what_memory_gives(tmp_path):
    _bundle(tmp_path / "src" / "outs", shuffle=True)
    whole = _commit(tmp_path / "a", tmp_path / "src", streamed="off")
    blocks = _commit(tmp_path / "b", tmp_path / "src", streamed="on")
    assert whole.shape == blocks.shape == (N_CELLS, N_GENES)
    assert whole.obs_names.tolist() == blocks.obs_names.tolist()
    assert (whole.X != blocks.X).nnz == 0
    assert np.array_equal(whole.obsm["spatial"], blocks.obsm["spatial"])
    pd.testing.assert_frame_equal(whole.obs, blocks.obs, check_categorical=False)
    assert whole.uns["scview_spatial"] == blocks.uns["scview_spatial"]
    prov = whole.uns["scview_provenance"]
    assert "xenium" in json.dumps(prov, default=str)


@pytest.mark.parametrize("mode", ["off", "on"])
def test_several_bundles_are_merged_with_libraries_from_region_or_folder(tmp_path, mode):
    src = tmp_path / "src"
    _bundle(src / "a" / "outs", seed=1)
    _bundle(src / "b" / "outs", experiment={**EXPERIMENT, "region_name": ""}, seed=2)
    b = build_bundle(_files(src), root=src)
    assert sorted(u.label for u in b.units) == ["Replicate 1", "b"] and b.is_merge
    a = _commit(tmp_path / "x", src, streamed=mode, name="two")
    assert a.n_obs == 2 * N_CELLS and a.n_vars == N_GENES
    assert sorted(a.obs["library_id"].cat.categories) == ["Replicate 1", "b"]
    rec = a.uns["scview_spatial"]
    assert rec["technology"] == "xenium" and rec["targeted_panel"] is True
    assert rec["n_panel_genes"] == N_GENES and rec["panel_name"] == "Breast TME"
    assert np.isfinite(a.obsm["spatial"]).all()
    assert sorted(a.uns["spatial"]) == ["Replicate 1", "b"]


def test_a_commit_of_an_incomplete_bundle_says_what_is_missing(tmp_path):
    _bundle(tmp_path / "src" / "outs")
    (tmp_path / "src" / "outs" / "cells.parquet").unlink()
    with pytest.raises(IngestCommitError, match="cells.parquet"):
        _commit(tmp_path / "x", tmp_path / "src")


# ---------------------------------------------------------------------------
# The image
# ---------------------------------------------------------------------------


def test_uncompressed_pyramid_levels_scale_and_stretch(tmp_path):
    base = _pyramid(tmp_path / "m.ome.tif", sizes=((2600, 3000), (1300, 1500), (650, 750), (325, 375)))
    found = xenium.read_image(tmp_path / "m.ome.tif", 0.2125)
    assert found.levels == {"hires": 1, "lowres": 3}
    assert found.images["hires"].shape == (1300, 1500, 3) and found.images["hires"].dtype == np.uint8
    assert found.images["lowres"].shape == (325, 375, 3)
    assert found.scalefactors["tissue_hires_scalef"] == pytest.approx(1 / (0.2125 * 2))
    assert found.scalefactors["tissue_lowres_scalef"] == pytest.approx(1 / (0.2125 * 8))
    # Brightest pixels saturate, the rest stay below (the 0.5 to 99.8 percentile stretch).
    level = base[::2, ::2]
    assert found.images["hires"][..., 0][level == level.max()].max() == 255
    assert (found.images["hires"][..., 0] < 255).mean() > 0.9
    assert (found.images["hires"][..., 0] == found.images["hires"][..., 2]).all()
    assert xenium.image_bytes(tmp_path / "m.ome.tif") == pytest.approx(3 * (1300 * 1500 + 325 * 375), rel=0.01)


def test_a_small_image_without_a_pyramid_still_gives_both(tmp_path):
    _pyramid(tmp_path / "m.ome.tif", sizes=((400, 700),))
    found = xenium.read_image(tmp_path / "m.ome.tif", 0.5)
    assert found.images["hires"].shape[:2] == (400, 700)
    assert max(found.images["lowres"].shape[:2]) <= xenium.LOWRES_LONG
    ratio = found.scalefactors["tissue_hires_scalef"] / found.scalefactors["tissue_lowres_scalef"]
    assert ratio == pytest.approx(700 / found.images["lowres"].shape[1])


def _j2k(tile: np.ndarray) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(tile).save(buf, format="JPEG2000", irreversible=False)
    return buf.getvalue()


@pytest.mark.skipif(not features.check("jpg_2000"), reason="Pillow without JPEG 2000")
def test_jpeg2000_tiles_are_decoded_with_pillow_in_their_places(tmp_path):
    """As the real files: tifffile would need imagecodecs for these tiles."""
    tile, h, w = 64, 128, 160
    base = (np.arange(h * w).reshape(h, w) % 4000 + 1).astype(np.uint16)
    raw, offsets, sizes = b"", [], []
    for r in range(0, h, tile):
        for c in range(0, w, tile):
            part = base[r:r + tile, c:c + tile]
            data = _j2k(np.pad(part, ((0, tile - part.shape[0]), (0, tile - part.shape[1]))))
            offsets.append(len(raw))
            sizes.append(len(data))
            raw += data
    kf = SimpleNamespace(compression=34712, imagelength=h, imagewidth=w, is_tiled=True, tilelength=tile,
                         tilewidth=tile, rowsperstrip=0, dtype=np.uint16)
    page = SimpleNamespace(keyframe=kf, dataoffsets=offsets, databytecounts=sizes)
    # tifffile cannot write these tiles without imagecodecs: the file handle and page by hand.
    assert np.array_equal(xenium._decode_page(SimpleNamespace(filehandle=io.BytesIO(raw)), page), base)


def test_a_broken_image_leaves_a_note_and_the_dataset_works(tmp_path):
    root = tmp_path / "outs"
    _bundle(root)
    (root / "morphology_focus.ome.tif").write_bytes(b"not a tiff")
    a = _load(root, tmp_path)
    entry = a.uns["spatial"]["Replicate 1"]
    assert entry["images"] == {} and "could not be read" in entry["metadata"]["image_note"]
    assert spatial.info(a)["libraries"][0]["images"] == {}


def test_the_image_reaches_the_api_helpers_and_places_cells_in_microns(tmp_path):
    root = tmp_path / "outs"
    _bundle(root)
    _pyramid(root / "morphology_focus.ome.tif")
    a = _load(root, tmp_path)
    lib = spatial.info(a)["libraries"][0]
    assert set(lib["images"]) == {"hires", "lowres"}
    assert lib["images"]["hires"]["scalef"] == pytest.approx(1 / 0.2125)
    assert spatial.image_png(a, "Replicate 1", "lowres").startswith(b"\x89PNG")
