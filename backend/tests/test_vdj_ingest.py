"""TCR/BCR contig tables in the import flow: detected, bundled beside their sample's
matrix, and joined to that sample's cells when the dataset is created."""

from __future__ import annotations

import gzip
from pathlib import Path

import anndata as ad
import pandas as pd
import pytest

from scview.core import vdj
from scview.core.ingestion import IngestCommitError, IngestOptions, IngestSessionManager
from scview.core.ingestion.bundling import FileRole, SideKind, UnitFormat, build_bundle
from scview.core.ingestion.detection import FileKind, detect_file
from tests.test_ingestion_session import _barcode, _stage_h5ad, _stage_mex
from tests.test_vdj import COLS, _airr, _row, b_rows, t_rows, write_csv


def _contigs_for(path: Path, cells: list[int], cdr3: str) -> Path:
    """A paired TRA/TRB for each of ``cells`` (the session tests' barcodes)."""
    rows = []
    for i in cells:
        for k, chain in enumerate(("TRA", "TRB")):
            r = _row(0, k, chain, f"{cdr3}{chain[-1]}", 3)
            rows.append({**r, "barcode": _barcode(i), "contig_id": f"{_barcode(i)}_{k}"})
    return write_csv(path, rows)


def test_contig_tables_are_detected(tmp_path):
    csv = write_csv(tmp_path / "x_filtered_contig_annotations.csv", t_rows())
    gz = write_csv(tmp_path / "all_contig_annotations.csv.gz", t_rows(), gz=True)
    tsv = tmp_path / "airr_rearrangement.tsv"
    _airr(t_rows()).to_csv(tsv, sep="\t", index=False)
    for p, fmt in ((csv, "10x"), (gz, "10x"), (tsv, "airr")):
        det = detect_file(p)
        assert det.kind == FileKind.vdj_contigs and det.details["vdj_format"] == fmt, p.name
    # A dense expression table is still a dense table.
    dense = tmp_path / "expr.csv"
    pd.DataFrame({"gene": ["A", "B"], "c1": [1, 2], "c2": [3, 4], "c3": [0, 1], "c4": [5, 6]}).to_csv(dense, index=False)
    assert detect_file(dense).kind == FileKind.dense_table


def _mex(d: Path, prefix: str) -> list[Path]:
    paths = []
    for name in ("matrix.mtx", "barcodes.tsv", "features.tsv"):
        p = d / f"{prefix}{name}"
        p.write_text({"matrix.mtx": "%%MatrixMarket matrix coordinate integer general\n3 2 1\n1 1 1\n",
                      "barcodes.tsv": _barcode(0) + "\n" + _barcode(1) + "\n",
                      "features.tsv": "ENSG00000000001\tA\tGene Expression\n"
                                      "ENSG00000000002\tB\tGene Expression\n"
                                      "ENSG00000000003\tC\tGene Expression\n"}[name])
        paths.append(p)
    return paths


def test_contigs_bundle_with_the_sample_of_the_same_prefix(tmp_path):
    files = _mex(tmp_path, "GSM1_") + _mex(tmp_path, "GSM2_")
    t1 = _contigs_for(tmp_path / "GSM1_t_filtered_contig_annotations.csv", [0], "CA")
    b2 = _contigs_for(tmp_path / "GSM2_vdj_b_filtered_contig_annotations.csv.gz", [1], "CB")
    bundle = build_bundle(files + [t1, b2])
    assert bundle.complete and not bundle.issues
    by = {u.label: u for u in bundle.units}
    # Side files of their sample (never one of the files loaded as its cells).
    assert [f.name for f in by["gsm1"].side_files] == [t1.name]
    assert [f.name for f in by["gsm2"].side_files] == [b2.name]
    assert all(f.role == FileRole.side and f.side_kind == SideKind.vdj
               for u in bundle.units for f in u.side_files)
    assert all(f.kind != FileKind.vdj_contigs for u in bundle.units for f in u.files)
    assert all(u.complete and u.format == UnitFormat.tenx_mex for u in bundle.units)


def test_t_and_b_tables_are_told_apart_by_their_chains(tmp_path):
    # The names say nothing; the chain column does (TR* = T-cell, IG* = B-cell receptor).
    d1, d2 = tmp_path / "one", tmp_path / "two"
    d1.mkdir(), d2.mkdir()
    t = write_csv(d1 / "filtered_contig_annotations.csv", t_rows())
    b = write_csv(d2 / "filtered_contig_annotations.csv", b_rows())
    both = write_csv(tmp_path / "x_filtered_contig_annotations.csv", t_rows()[:2] + b_rows()[:1])
    assert detect_file(t).details["receptor"] == "T"
    assert detect_file(b).details["receptor"] == "B"
    assert detect_file(both).details["receptor"] == "T+B"
    airr_b = tmp_path / "airr_rearrangement.tsv"
    _airr(b_rows()).to_csv(airr_b, sep="\t", index=False)
    assert detect_file(airr_b).details["receptor"] == "B"
    h5 = tmp_path / "pbmc.h5ad"
    ad.AnnData(pd.DataFrame([[1.0]], index=["c"], columns=["g"])).write_h5ad(h5)
    unit = build_bundle([h5, t, b]).units[0]
    assert sorted((f.path, f.receptor) for f in unit.side_files) == [(str(t), "T"), (str(b), "B")]
    # Without a readable chain, the name's arm token: sample_t_ / _vdj_b_.
    empty = tmp_path / "s1_vdj_b_filtered_contig_annotations.csv"
    empty.write_text(",".join(COLS) + "\n")
    assert build_bundle([h5, empty]).units[0].side_files[0].receptor == "B"


def test_unprefixed_contigs_go_to_the_only_sample_and_not_to_a_guess(tmp_path):
    h5 = tmp_path / "pbmc_filtered_feature_bc_matrix.h5ad"
    ad.AnnData(pd.DataFrame([[1.0]], index=["c"], columns=["g"])).write_h5ad(h5)
    bare = write_csv(tmp_path / "filtered_contig_annotations.csv", t_rows())
    one = build_bundle([h5, bare])
    assert [f.name for f in one.units[0].files] == [h5.name]
    assert [f.side_kind for f in one.units[0].side_files] == [SideKind.vdj]
    two = build_bundle(_mex(tmp_path, "GSM1_") + _mex(tmp_path, "GSM2_") + [bare])
    assert any("which sample" in i for i in two.issues)
    assert not [f for u in two.units for f in u.side_files]
    alone = build_bundle([bare])
    assert not alone.complete and any("expression matrix" in i for i in alone.issues)


@pytest.fixture()
def mgr(tmp_path: Path) -> IngestSessionManager:
    for sub in ("uploads", "converted", "cache"):
        (tmp_path / sub).mkdir()
    return IngestSessionManager(data_dir=str(tmp_path))


async def test_commit_joins_each_sample_to_its_own_cells(mgr, tmp_path):
    from scview.core.dataset_manager import DatasetManager

    dm = DatasetManager(data_dir=str(tmp_path))
    sid = mgr.create_session()
    _stage_mex(mgr, sid, prefix="GSM1_", n_cells=3)
    _stage_mex(mgr, sid, prefix="GSM2_", n_cells=3)
    d = mgr.files_dir(sid)
    # The same barcodes in both samples, different receptors: each must land in its own sample.
    _contigs_for(d / "GSM1_filtered_contig_annotations.csv", [0, 1], "CAX")
    _contigs_for(d / "GSM2_filtered_contig_annotations.csv", [0], "CAY")
    mgr.set_options(sid, IngestOptions(name="merged"))
    did = await mgr.commit(sid, dm)
    a = dm.get_dataset(did).adata
    obs = a.obs.assign(cdr3=a.obs["tra_cdr3"].astype(object))
    got = {(str(s), n.split(":")[0]): c for n, s, c in zip(a.obs_names, obs["sample"], obs["cdr3"])}
    assert got[("gsm1", _barcode(0))] == "CAXA" and got[("gsm1", _barcode(1))] == "CAXA"
    assert got[("gsm2", _barcode(0))] == "CAYA" and pd.isna(got[("gsm2", _barcode(1))])
    assert a.obs[vdj.CLONOTYPE_COL].value_counts().to_dict() == {"TCR1": 2, "TCR2": 1}
    info = vdj.summary(a)
    assert [f["sample"] for f in info["files"]] == ["gsm1", "gsm2"]
    assert info["match"]["cells_without_contigs"] == 3
    assert (tmp_path / "uploads" / did / "GSM1_filtered_contig_annotations.csv").exists()


async def test_commit_single_h5ad_with_contigs(mgr, tmp_path):
    from scview.core.dataset_manager import DatasetManager

    dm = DatasetManager(data_dir=str(tmp_path))
    sid = mgr.create_session()
    _stage_h5ad(mgr, sid, fname="pbmc.h5ad", n_obs=4)
    h5 = mgr.files_dir(sid) / "pbmc.h5ad"
    a = ad.read_h5ad(h5)
    a.obs_names = [_barcode(i) for i in range(4)]
    a.write_h5ad(h5)
    with gzip.open(mgr.files_dir(sid) / "pbmc_t_filtered_contig_annotations.csv.gz", "wt") as fh:
        pd.read_csv(_contigs_for(tmp_path / "c.csv", [2, 3], "CAZ")).to_csv(fh, index=False, columns=COLS)
    did = await mgr.commit(sid, dm)
    out = dm.get_dataset(did).adata
    assert out.obs[vdj.STATUS_COL].astype(str).tolist() == ["no contig", "no contig", "paired", "paired"]
    assert out.obs[vdj.SIZE_COL].tolist()[2:] == [2, 2]
    # A contig table is not a sequencing-QC side file: no empty Sequencing QC record.
    from scview.core import seq_metrics

    assert seq_metrics.read(out) is None


async def test_a_cut_off_contig_table_is_a_commit_error(mgr, tmp_path):
    # Cut after the part detection reads: found as contigs, unreadable at commit.
    from scview.core.dataset_manager import DatasetManager

    dm = DatasetManager(data_dir=str(tmp_path))
    sid = mgr.create_session()
    _stage_h5ad(mgr, sid, fname="pbmc.h5ad", n_obs=4)
    gz = write_csv(mgr.files_dir(sid) / "pbmc_t_filtered_contig_annotations.csv.gz", t_rows() * 400, gz=True)
    gz.write_bytes(gz.read_bytes()[: gz.stat().st_size // 2])
    assert mgr.get_state(sid)["bundle"]["units"][0]["side_files"][0]["side_kind"] == "vdj"
    with pytest.raises(IngestCommitError, match="could not be attached"):
        await mgr.commit(sid, dm)
