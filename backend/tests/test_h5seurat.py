"""SeuratDisk .h5Seurat files read natively (core/ingestion/h5seurat.py), Seurat's Visium
images in the spatial layout (core/seurat_spatial.py), and the Visium grid recovered
from spot centres.

The fixtures (tests/fixtures/h5seurat/, made by make_fixtures.R with SeuratDisk's own
SaveH5Seurat): SeuratObject's pbmc_small (80 cells x 230 genes, PCA + tSNE), the same
with a 4-antibody ADT assay, and 60 spots of 10x's V1_Human_Lymph_Node as a VisiumV1
and a VisiumV2 (Seurat 5.1) object; visium_positions.csv is Space Ranger's own table
for those spots.
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

from scview.core import preflight, seurat_spatial, spatial
from scview.core.dataset_manager import DatasetManager
from scview.core.ingestion import IngestSessionManager, h5seurat
from scview.core.ingestion.bundling import IngestUnit, UnitFormat, build_bundle
from scview.core.ingestion.detection import FileKind, detect_file
from scview.core.ingestion.loading import load_unit

FIX = Path(__file__).parent / "fixtures" / "h5seurat"
PBMC = FIX / "pbmc_small.h5Seurat"
ADT = FIX / "pbmc_small_adt.h5Seurat"
V1 = FIX / "visium_v1.h5Seurat"
V2 = FIX / "visium_v2.h5Seurat"


@pytest.fixture(scope="module")
def positions() -> pd.DataFrame:
    return pd.read_csv(FIX / "visium_positions.csv", index_col="barcode")


@pytest.fixture(scope="module")
def scalefactors() -> dict:
    return json.loads((FIX / "visium_scalefactors.json").read_text())


# ---------------------------------------------------------------------------
# Detection, bundling, preflight
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("path", [PBMC, ADT, V1, V2], ids=lambda p: p.stem)
def test_detected_as_h5seurat(path):
    r = detect_file(path)
    assert r.kind == FileKind.h5seurat
    assert r.format_family == "seurat"


def test_other_hdf5_is_not_h5seurat(tmp_path):
    p = tmp_path / "x.h5"
    with h5py.File(p, "w") as f:
        f.attrs["version"] = "5.0.2"
        f.create_group("assays")  # no meta.data, no cell.names
    assert detect_file(p).kind == FileKind.unknown


def test_bundled_as_its_own_unit(tmp_path):
    bundle = build_bundle([PBMC])
    assert [u.format for u in bundle.units] == [UnitFormat.h5seurat]
    assert bundle.units[0].complete


def test_preflight_reads_the_header():
    h = preflight.read_header(PBMC)
    assert h.format == "h5seurat"
    assert (h.n_cells, h.n_genes) == (80, 230)
    assert h.x is not None and h.x.sparse and h.x.nnz == 4456
    assert h.obsm["X_pca"] == [80, 19]


# ---------------------------------------------------------------------------
# pbmc_small: what convert.R makes of a Seurat object
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def pbmc() -> ad.AnnData:
    return h5seurat.read(PBMC)


def test_counts_in_x_with_seurat_names(pbmc):
    assert pbmc.shape == (80, 230)
    assert sp.issparse(pbmc.X)
    X = pbmc.X.toarray()
    assert np.allclose(X, np.round(X)) and X.min() == 0  # counts, not the log values
    # Seurat's own per-cell totals agree with the matrix read here
    assert np.allclose(X.sum(axis=1), pbmc.obs["nCount_RNA"].to_numpy())
    assert np.array_equal((X > 0).sum(axis=1), pbmc.obs["nFeature_RNA"].to_numpy())
    assert list(pbmc.var_names[:3]) == ["MS4A1", "CD79B", "CD79A"]
    assert pbmc.obs_names[0] == "ATGCCAGAACGACT"
    assert list(pbmc.var.columns) == []  # as convert.R: gene names only


def test_meta_data_types(pbmc):
    obs = pbmc.obs
    assert list(obs.columns) == ["orig.ident", "nCount_RNA", "nFeature_RNA", "RNA_snn_res.0.8",
                                 "letter.idents", "groups", "RNA_snn_res.1", "is_high", "note"]
    assert isinstance(obs["RNA_snn_res.1"].dtype, pd.CategoricalDtype)
    assert list(obs["RNA_snn_res.1"].cat.categories) == ["0", "1", "2"]
    assert list(obs["letter.idents"].cat.categories) == ["A", "B"]
    assert obs["is_high"].dtype == bool
    assert obs["is_high"].sum() == (obs["nCount_RNA"] > obs["nCount_RNA"].median()).sum()
    assert obs["note"].isna().sum() == 16  # every 5th cell: R's NA
    assert set(obs["note"].dropna()) == {"kept"}


def test_reductions_in_obsm(pbmc):
    assert set(pbmc.obsm.keys()) == {"X_pca", "X_tsne"}
    assert pbmc.obsm["X_pca"].shape == (80, 19)
    assert pbmc.obsm["X_tsne"].shape == (80, 2)
    # not transposed: the first PC varies over the cells
    assert np.std(pbmc.obsm["X_pca"][:, 0]) > 0


def test_load_unit_keeps_it_and_writes_h5ad(tmp_path):
    unit = build_bundle([PBMC]).units[0]
    a = load_unit(unit)
    assert a.shape == (80, 230)
    a.write_h5ad(tmp_path / "out.h5ad")  # every column type writes
    back = ad.read_h5ad(tmp_path / "out.h5ad")
    assert back.obs["note"].isna().sum() == 16


def test_adt_assay_is_the_protein_modality():
    a = h5seurat.read(ADT)
    assert a.n_vars == 230  # the antibodies are not genes
    m = a.obsm["protein_counts"]
    assert list(m.columns) == ["CD3", "CD4", "CD8", "IgG1-ctrl"]
    assert np.allclose(m.sum(axis=1).to_numpy(), a.obs["nCount_ADT"].to_numpy())
    unit = build_bundle([ADT]).units[0]
    loaded = load_unit(unit)
    assert "protein_clr" in loaded.obsm  # core/protein.py takes it from there


def test_counts_missing_falls_back_to_data(tmp_path):
    p = tmp_path / "datas.h5Seurat"
    _tiny_h5seurat(p, counts=False)
    a = h5seurat.read(p)
    assert np.allclose(a.X.toarray(), [[0.5, 0], [0, 1.5], [2.5, 0]])


def test_active_antibody_assay_still_puts_genes_in_x(tmp_path):
    p = tmp_path / "adt_active.h5Seurat"
    _tiny_h5seurat(p, active="ADT")
    a = h5seurat.read(p)
    assert list(a.var_names) == ["g1", "g2"]
    assert list(a.obsm["protein_counts"].columns) == ["CD3"]


def test_mismatched_names_are_an_error(tmp_path):
    p = tmp_path / "bad.h5Seurat"
    _tiny_h5seurat(p)
    with h5py.File(p, "a") as f:
        del f["cell.names"]
        f["cell.names"] = np.array([b"a", b"b"], dtype=object)
    with pytest.raises(h5seurat.H5SeuratError, match="3 cells"):
        h5seurat.read(p)


def _tiny_h5seurat(path: Path, counts: bool = True, active: str = "RNA") -> None:
    """3 cells x 2 genes in SeuratDisk's layout, an ADT assay of one antibody."""
    s = h5py.string_dtype()
    with h5py.File(path, "w") as f:
        f.attrs["version"] = np.array(["5.0.2"], dtype=object)
        f.attrs["active.assay"] = np.array([active], dtype=object)
        f.create_dataset("cell.names", data=np.array(["c1", "c2", "c3"], dtype=object), dtype=s)
        rna = f.create_group("assays/RNA")
        rna.create_dataset("features", data=np.array(["g1", "g2"], dtype=object), dtype=s)
        for slot, vals in (("counts", [1.0, 2.0, 3.0]), ("data", [0.5, 1.5, 2.5])):
            g = rna.create_group(slot)
            if slot == "counts" and not counts:
                g.attrs["dims"] = np.array([0, 0], dtype=np.int32)
                for k in ("data", "indices"):
                    g.create_dataset(k, data=np.array([], dtype=np.float64))
                g.create_dataset("indptr", data=np.array([0], dtype=np.int32))
                continue
            g.attrs["dims"] = np.array([2, 3], dtype=np.int32)
            g.create_dataset("data", data=np.array(vals))
            g.create_dataset("indices", data=np.array([0, 1, 0], dtype=np.int32))
            g.create_dataset("indptr", data=np.array([0, 1, 2, 3], dtype=np.int32))
        adt = f.create_group("assays/ADT")
        adt.create_dataset("features", data=np.array(["CD3"], dtype=object), dtype=s)
        g = adt.create_group("counts")
        g.attrs["dims"] = np.array([1, 3], dtype=np.int32)
        g.create_dataset("data", data=np.array([5.0, 7.0, 9.0]))
        g.create_dataset("indices", data=np.array([0, 0, 0], dtype=np.int32))
        g.create_dataset("indptr", data=np.array([0, 1, 2, 3], dtype=np.int32))
        md = f.create_group("meta.data")
        md.attrs["colnames"] = np.array(["kind"], dtype=object)
        k = md.create_group("kind")
        k.create_dataset("levels", data=np.array(["a", "b"], dtype=object), dtype=s)
        k.create_dataset("values", data=np.array([1, 2, -2147483648], dtype=np.int32))


def test_factor_na_code(tmp_path):
    p = tmp_path / "na.h5Seurat"
    _tiny_h5seurat(p)
    kind = h5seurat.read(p).obs["kind"]
    assert list(kind.astype(object).iloc[:2]) == ["a", "b"] and pd.isna(kind.iloc[2])


# ---------------------------------------------------------------------------
# Visium: VisiumV1 and VisiumV2 images
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("path", [V1, V2], ids=["VisiumV1", "VisiumV2"])
def test_visium_coordinates_in_full_res_pixels(path, positions, scalefactors):
    a = h5seurat.read(path)
    pos = positions.loc[a.obs_names]
    # contract.md: x = pxl_col_in_fullres (Seurat's imagecol), y = pxl_row_in_fullres
    assert np.allclose(a.obsm["spatial"], pos[["imagecol", "imagerow"]].to_numpy())
    assert list(a.obs["library_id"].cat.categories) == ["slice1"]
    entry = a.uns["spatial"]["slice1"]
    sf = entry["scalefactors"]
    assert sf["tissue_lowres_scalef"] == pytest.approx(scalefactors["lowres"])
    assert sf["tissue_hires_scalef"] == pytest.approx(scalefactors["hires"])
    assert sf["spot_diameter_fullres"] == pytest.approx(scalefactors["spot"])
    img = entry["images"]["lowres"]
    assert img.dtype == np.uint8 and img.shape == tuple(scalefactors["image_dim"])
    # the spots land inside the image they are drawn on
    px = a.obsm["spatial"] * sf["tissue_lowres_scalef"]
    assert (px[:, 0] < img.shape[1]).all() and (px[:, 1] < img.shape[0]).all()
    assert entry["metadata"]["source"] == "seurat"
    assert a.uns["scview_spatial"] == {"technology": "visium", "unit": "spot"}
    assert spatial.info(a)["libraries"][0]["images"]["lowres"]["width"] == img.shape[1]


def test_visium_v1_keeps_the_array_grid(positions):
    a = h5seurat.read(V1)
    pos = positions.loc[a.obs_names]
    assert np.array_equal(a.obs["array_row"], pos["row"])
    assert np.array_equal(a.obs["array_col"], pos["col"])
    assert "array_grid" not in a.uns["spatial"]["slice1"]["metadata"]


def test_visium_v2_grid_recovered(positions):
    a = h5seurat.read(V2)
    pos = positions.loc[a.obs_names]
    # Space Ranger's grid less a constant (counted from the patch's first row / column)
    assert len(np.unique(a.obs["array_row"].to_numpy() - pos["row"].to_numpy())) == 1
    assert len(np.unique(a.obs["array_col"].to_numpy() - pos["col"].to_numpy())) == 1
    assert a.uns["spatial"]["slice1"]["metadata"]["array_grid"] == seurat_spatial.GRID_INFERRED
    assert list(a.obs["region"].cat.categories) == ["cortex", "medulla"]


def test_visium_through_load_unit_stays_in_layout():
    a = load_unit(build_bundle([V2]).units[0])
    assert a.uns["scview_spatial"]["unit"] == "spot"
    assert spatial.has_spatial(a)


# ---------------------------------------------------------------------------
# The grid from spot centres
# ---------------------------------------------------------------------------


def _lattice(n_rows=12, n_cols=30, d=138.0, angle=0.02, shear=0.004, noise=1.0, seed=0):
    rng = np.random.default_rng(seed)
    rows, cols = [], []
    for r in range(n_rows):
        for c in range(r % 2, n_cols, 2):
            rows.append(r)
            cols.append(c)
    rows, cols = np.array(rows), np.array(cols)
    u = cols * d / 2 + rows * d * shear
    v = rows * d * np.sqrt(3) / 2
    rot = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    xy = np.column_stack([u, v]) @ rot.T + [1300.0, 1000.0]
    xy += rng.normal(scale=noise, size=xy.shape)
    order = rng.permutation(len(rows))
    return xy[order], rows[order], cols[order]


def test_grid_recovered_from_a_rotated_sheared_lattice():
    xy, rows, cols = _lattice()
    got = seurat_spatial.visium_grid(xy)
    assert got is not None
    assert np.array_equal(got[0], rows - rows.min())
    assert np.array_equal(got[1], cols - cols.min())


def test_grid_none_off_the_lattice():
    xy, rows, cols = _lattice()
    rng = np.random.default_rng(1)
    assert seurat_spatial.visium_grid(rng.uniform(0, 3000, size=(300, 2))) is None
    assert seurat_spatial.visium_grid(xy[:2]) is None
    # A thin subset: no grid, or the true one (never a wrong one).
    got = seurat_spatial.visium_grid(xy[::7])
    if got is not None:
        assert len(np.unique(got[0] - rows[::7])) == 1
        assert len(np.unique(got[1] - cols[::7])) == 1
    # A spot halfway between lattice points
    bad = np.vstack([xy, xy[:2].mean(axis=0) + [0.0, 40.0]])
    assert seurat_spatial.visium_grid(bad) is None


def test_slice_with_cells_outside_any_image_is_not_attached():
    a = ad.AnnData(X=np.ones((3, 2)))
    a.obs_names = ["a", "b", "c"]
    s = seurat_spatial.SeuratSlice("s1", "VisiumV1", ["a", "b"], np.ones((2, 2)), None,
                                   {"lowres": 0.1}, (np.zeros(2, int), np.zeros(2, int)))
    out = seurat_spatial.attach(a, [s])
    assert "skipped" in out and "spatial" not in a.obsm


def test_complete_converted_adds_the_grid(tmp_path):
    xy, rows, cols = _lattice(n_rows=6, n_cols=16)
    a = ad.AnnData(X=sp.csr_matrix(np.ones((len(xy), 3))))
    a.obs_names = [f"s{i}" for i in range(len(xy))]
    a.obsm["spatial"] = xy
    a.obs["library_id"] = pd.Categorical(["slice1"] * len(xy))
    a.uns["spatial"] = {"slice1": {"images": {}, "scalefactors": {"spot_diameter_fullres": 89.5},
                                   "metadata": {"source": "seurat"}}}
    p = tmp_path / "conv.h5ad"
    a.write_h5ad(p)
    assert seurat_spatial.complete_converted(p) is True
    b = ad.read_h5ad(p)
    assert np.array_equal(b.obs["array_row"].to_numpy(), rows - rows.min())
    assert b.uns["scview_spatial"]["unit"] == "spot"
    assert seurat_spatial.complete_converted(p) is False  # done once


def test_complete_converted_leaves_other_files_alone(tmp_path):
    p = tmp_path / "plain.h5ad"
    ad.AnnData(X=np.ones((4, 2))).write_h5ad(p)
    before = p.stat().st_mtime_ns
    assert seurat_spatial.complete_converted(p) is False
    assert p.stat().st_mtime_ns == before


# ---------------------------------------------------------------------------
# The ingest session
# ---------------------------------------------------------------------------


@pytest.fixture()
def data_dir(tmp_path: Path) -> Path:
    for sub in ("uploads", "converted", "cache"):
        (tmp_path / sub).mkdir()
    return tmp_path


async def test_commit_h5seurat_alone_and_merged(data_dir):
    import shutil

    from scview.core import provenance

    mgr = IngestSessionManager(data_dir=str(data_dir))
    dm = DatasetManager(data_dir=str(data_dir))
    sid = mgr.create_session()
    shutil.copy(PBMC, mgr.files_dir(sid) / "pbmc.h5Seurat")
    state = mgr.get_state(sid)
    assert state["validation"]["ok"]
    did = await mgr.commit(sid, dm)
    h5 = next((data_dir / "ingested" / did).glob("*.h5ad"))
    back = ad.read_h5ad(h5)
    assert back.shape == (80, 230) and "X_pca" in back.obsm
    assert provenance.read_provenance(back)["source"]["format"] == "h5seurat"

    sid = mgr.create_session()
    shutil.copy(PBMC, mgr.files_dir(sid) / "a.h5Seurat")
    shutil.copy(ADT, mgr.files_dir(sid) / "b.h5Seurat")
    assert mgr.merge_plan(sid) is not None
    did = await mgr.commit(sid, dm)
    assert dm.get_dataset_info(did)["n_cells"] == 160


def test_unit_format_value():
    assert IngestUnit(label="x", format=UnitFormat.h5seurat).format.value == "h5seurat"
