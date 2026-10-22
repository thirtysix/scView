"""Folder uploads: each file keeps its path inside the chosen folder, so a Cell Ranger
``outs/`` tree stages without same-named files overwriting each other, and its folders
say which file is the raw matrix, which contig table is T or B, and which sample each
belongs to. Expected unit layouts are written out by hand from the trees built here."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from scview.core import seq_metrics, vdj
from scview.core.dataset_manager import DatasetManager
from scview.core.ingestion import IngestSessionManager
from scview.core.ingestion.bundling import SideKind, UnitFormat, build_bundle
from scview.core.ingestion.paths import UploadPathError, flat_name, junk_reason, safe_relative_path
from tests.test_seq_metrics import PBMC_COUNT, _bc, _raw_counts, _write_h5, _write_mex
from tests.test_vdj import _row, write_csv

API = "/api/v1"

# ---------------------------------------------------------------------------
# The path sanitiser
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("raw, want", [
    ("outs/raw_feature_bc_matrix/barcodes.tsv.gz", "outs/raw_feature_bc_matrix/barcodes.tsv.gz"),
    ("outs\\vdj_t\\filtered_contig_annotations.csv", "outs/vdj_t/filtered_contig_annotations.csv"),
    ("échantillon/x.h5", "échantillon/x.h5"),
    ("e\u0301chantillon/x.h5", "échantillon/x.h5"),  # macOS decomposed form -> NFC: one path
    ("样本 1/matrix.mtx.gz", "样本 1/matrix.mtx.gz"),
    ("a.h5ad", "a.h5ad"),
])
def test_safe_paths(raw, want):
    assert safe_relative_path(raw) == want


@pytest.mark.parametrize("raw, why", [
    ("../etc/passwd", "'..'"),
    ("outs/../../x", "'..'"),
    ("/etc/passwd", "absolute"),
    ("C:\\Users\\me\\x.h5", "absolute"),
    ("c:x.h5", "absolute"),
    ("\\\\server\\share\\x.h5", "absolute"),
    ("outs//x.h5", "empty"),
    ("outs/./x.h5", "'.'"),
    ("outs/", "empty"),
    ("", "without a name"),
    ("   ", "without a name"),
    ("a\x00b.h5", "control"),
    ("a\nb.h5", "control"),
    ("/".join(["d"] * 17) + "/x.h5", "deep"),
    ("x" * 256 + ".h5", "too long"),
    ("é" * 128 + ".h5", "too long"),  # 2 bytes each in UTF-8: over 255 bytes
])
def test_unsafe_paths_are_refused(raw, why):
    with pytest.raises(UploadPathError, match=why):
        safe_relative_path(raw)


@pytest.mark.parametrize("name, key", [
    ("filtered_gene_bc_matrices_h5.h5", ""), ("raw_gene_bc_matrices_h5.h5", ""),
    ("pbmc4k_raw_gene_bc_matrices_h5.h5", "pbmc4k"), ("pbmc4k_filtered_gene_bc_matrices_h5.h5", "pbmc4k"),
])
def test_sample_key_of_cellranger2_h5(name, key):
    from scview.core.ingestion.bundling import sample_key

    assert sample_key(name) == key


def test_flat_name_and_junk():
    assert flat_name("../../x.h5ad") == "x.h5ad" and flat_name("C:\\dir\\y.csv") == "y.csv"
    assert flat_name(None) == "unnamed"
    for name in ("outs/possorted_genome_bam.bam", "outs/possorted_genome_bam.bam.bai", "x/.DS_Store",
                 "outs/web_summary.html", "outs/cloupe.cloupe", "vdj_t/vloupe.vloupe",
                 "outs/molecule_info.h5", "._barcodes.tsv.gz"):
        assert junk_reason(name), name
    for name in ("outs/filtered_feature_bc_matrix.h5", "outs/metrics_summary.csv", "a.h5ad",
                 "x.zarr/.zgroup"):
        assert junk_reason(name) is None, name


# ---------------------------------------------------------------------------
# Synthetic Cell Ranger trees
# ---------------------------------------------------------------------------


def _contigs(path: Path, cells: range, cdr3: str, chains=("TRA", "TRB")) -> Path:
    rows = []
    for i in cells:
        for k, chain in enumerate(chains):
            r = _row(0, k, chain, f"{cdr3}{chain[-1]}{i % 3}", 3)
            rows.append({**r, "barcode": _bc(i), "contig_id": f"{_bc(i)}_{k}"})
    path.parent.mkdir(parents=True, exist_ok=True)
    return write_csv(path, rows)


def _write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return path


def count_outs(outs: Path, *, h5: bool = True, mex: bool = True, vdj_dirs: bool = True,
               barcodes=range(200)) -> None:
    """A cellranger count ``outs/`` folder: filtered + raw matrix as .h5 and/or as MEX
    folders (same names inside), metrics, Cell Ranger's own analysis CSVs, the report, a
    BAM; with ``vdj_dirs``, ``vdj_t`` (filtered, all, AIRR, clonotypes, consensus) and
    ``vdj_b`` (filtered) as cellranger multi lays them out."""
    dense = _raw_counts()
    bcs = [_bc(i) for i in range(dense.shape[1])]
    keep = list(barcodes)
    if h5:
        _write_h5(_mk(outs / "filtered_feature_bc_matrix.h5"), dense[:, keep], [bcs[i] for i in keep])
        _write_h5(_mk(outs / "raw_feature_bc_matrix.h5"), dense, bcs)
    if mex:
        _write_mex(_mkdir(outs / "filtered_feature_bc_matrix"), "", dense[:, keep], [bcs[i] for i in keep])
        _write_mex(_mkdir(outs / "raw_feature_bc_matrix"), "", dense, bcs)
    _write(outs / "metrics_summary.csv", PBMC_COUNT)
    _write(outs / "web_summary.html", "<html></html>")
    (outs / "possorted_genome_bam.bam").write_bytes(b"BAM\x01" + b"\x00" * 64)
    proj = "Barcode,PC-1,PC-2,PC-3,PC-4,PC-5\n" + "".join(f"{bcs[i]},1,2,3,4,5\n" for i in keep[:20])
    _write(outs / "analysis" / "pca" / "gene_expression_10_components" / "projection.csv", proj)
    _write(outs / "analysis" / "tsne" / "gene_expression_2_components" / "projection.csv",
           "Barcode,TSNE-1,TSNE-2,x,y\n" + "".join(f"{bcs[i]},1,2,3,4\n" for i in keep[:20]))
    if vdj_dirs:
        t = outs / "vdj_t"
        _contigs(t / "filtered_contig_annotations.csv", range(0, 40), "CAT")
        _contigs(t / "all_contig_annotations.csv", range(0, 60), "CAT")
        _write(t / "clonotypes.csv", "clonotype_id,frequency,proportion,cdr3s_aa,cdr3s_nt\n"
               + "".join(f"clonotype{i},{i},0.1,TRA:CA,TRA:TGT\n" for i in range(5)))
        _write(t / "consensus_annotations.csv",
               "clonotype_id,consensus_id,length,chain,v_gene,d_gene,j_gene,c_gene\n"
               "clonotype1,clonotype1_consensus_1,500,TRA,TRAV1,,TRAJ1,TRAC\n")
        _write(t / "filtered_contig.fasta", ">x\nACGT\n")
        _contigs(outs / "vdj_b" / "filtered_contig_annotations.csv", range(100, 130), "CAB", ("IGH", "IGK"))


def _mk(p: Path) -> Path:
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def _mkdir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def _files(root: Path) -> list[Path]:
    return sorted(p for p in root.rglob("*") if p.is_file())


def _side(unit) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for f in unit.side_files:
        out.setdefault(f.side_kind.value, []).append(f.rel_path)
    return {k: sorted(v) for k, v in out.items()}


# ---------------------------------------------------------------------------
# Bundling a folder
# ---------------------------------------------------------------------------


def test_outs_tree_is_one_unit_with_raw_side_and_both_contig_tables(tmp_path):
    root = tmp_path / "up"
    count_outs(root / "pbmc_1k" / "outs")
    bundle = build_bundle(_files(root), root=root)
    (unit,) = bundle.units
    assert unit.label == "pbmc_1k" and unit.format == UnitFormat.tenx_h5 and unit.complete
    assert [f.rel_path for f in unit.files] == ["pbmc_1k/outs/filtered_feature_bc_matrix.h5"]
    side = _side(unit)
    assert side["metrics"] == ["pbmc_1k/outs/metrics_summary.csv"]
    assert side["raw"] == ["pbmc_1k/outs/raw_feature_bc_matrix.h5"]  # the .h5, not the MEX copy
    assert {f.rel_path: f.receptor for f in unit.side_files if f.side_kind == SideKind.vdj} == {
        "pbmc_1k/outs/vdj_t/filtered_contig_annotations.csv": "T",
        "pbmc_1k/outs/vdj_b/filtered_contig_annotations.csv": "B"}
    skipped = {s.path: s.reason for s in bundle.skipped}
    assert set(skipped) == {
        "pbmc_1k/outs/web_summary.html", "pbmc_1k/outs/possorted_genome_bam.bam",
        "pbmc_1k/outs/analysis/pca/gene_expression_10_components/projection.csv",
        "pbmc_1k/outs/analysis/tsne/gene_expression_2_components/projection.csv",
        *(f"pbmc_1k/outs/{m}/{n}" for m in ("filtered_feature_bc_matrix", "raw_feature_bc_matrix")
          for n in ("matrix.mtx.gz", "barcodes.tsv.gz", "features.tsv.gz")),
        "pbmc_1k/outs/vdj_t/all_contig_annotations.csv", "pbmc_1k/outs/vdj_t/clonotypes.csv",
        "pbmc_1k/outs/vdj_t/consensus_annotations.csv", "pbmc_1k/outs/vdj_t/filtered_contig.fasta",
    }
    assert "the same matrix as filtered_feature_bc_matrix.h5" in \
        skipped["pbmc_1k/outs/filtered_feature_bc_matrix/barcodes.tsv.gz"]
    assert "filtered_contig_annotations.csv" in skipped["pbmc_1k/outs/vdj_t/all_contig_annotations.csv"]
    # One sentence for all of it, first among the issues; nothing else to say.
    assert len(bundle.issues) == 1 and bundle.issues[0].startswith("Skipped 14 files scView does not read")
    assert "possorted_genome_bam.bam (read alignments" in bundle.issues[0]


def test_mex_only_outs_keeps_raw_and_filtered_barcodes_apart(tmp_path):
    root = tmp_path / "up"
    count_outs(root / "outs", h5=False, vdj_dirs=False)
    (unit,) = build_bundle(_files(root), root=root).units
    assert unit.format == UnitFormat.tenx_mex and unit.complete and unit.label == "sample"
    assert sorted(f.rel_path for f in unit.files) == [
        "outs/filtered_feature_bc_matrix/barcodes.tsv.gz", "outs/filtered_feature_bc_matrix/features.tsv.gz",
        "outs/filtered_feature_bc_matrix/matrix.mtx.gz"]
    assert sorted(_side(unit)["raw"]) == [
        "outs/raw_feature_bc_matrix/barcodes.tsv.gz", "outs/raw_feature_bc_matrix/features.tsv.gz",
        "outs/raw_feature_bc_matrix/matrix.mtx.gz"]
    assert _side(unit)["metrics"] == ["outs/metrics_summary.csv"]


def test_multi_sample_folders_give_one_unit_each(tmp_path):
    root = tmp_path / "up"
    count_outs(root / "runs" / "donor_A" / "outs", mex=False, barcodes=range(0, 150))
    count_outs(root / "runs" / "donor_B" / "outs", mex=False, barcodes=range(50, 200))
    bundle = build_bundle(_files(root), root=root)
    assert [u.label for u in bundle.units] == ["donor_A", "donor_B"] and bundle.is_merge
    for u in bundle.units:
        base = f"runs/{u.label}/outs"
        assert [f.rel_path for f in u.files] == [f"{base}/filtered_feature_bc_matrix.h5"]
        assert _side(u) == {"metrics": [f"{base}/metrics_summary.csv"],
                            "raw": [f"{base}/raw_feature_bc_matrix.h5"],
                            "vdj": [f"{base}/vdj_b/filtered_contig_annotations.csv",
                                    f"{base}/vdj_t/filtered_contig_annotations.csv"]}
    assert len(bundle.issues) == 1  # the skipped-files sentence only: every file found its sample


def test_cellranger_multi_per_sample_outs(tmp_path):
    root = tmp_path / "up"
    dense = _raw_counts()
    bcs = [_bc(i) for i in range(dense.shape[1])]
    outs = root / "outs"
    _write_h5(_mk(outs / "multi" / "count" / "raw_feature_bc_matrix.h5"), dense, bcs)
    _contigs(outs / "multi" / "vdj_t" / "all_contig_annotations.csv", range(0, 200), "CAM")
    for s, cells in (("PBMC_ctrl", range(0, 100)), ("PBMC_stim", range(100, 200))):
        so = outs / "per_sample_outs" / s
        keep = list(cells)
        _write_mex(_mkdir(so / "count" / "sample_filtered_feature_bc_matrix"), "", dense[:, keep],
                   [bcs[i] for i in keep])
        _write_h5(_mk(so / "count" / "sample_filtered_feature_bc_matrix.h5"), dense[:, keep],
                  [bcs[i] for i in keep])
        _write(so / "metrics_summary.csv", PBMC_COUNT)
        _write(so / "count" / "analysis" / "clustering" / "x.csv", "Barcode,Cluster,a,b,c\nA,1,2,3,4\n")
        _contigs(so / "vdj_t" / "filtered_contig_annotations.csv", cells, "CAS")
    bundle = build_bundle(_files(root), root=root)
    assert [u.label for u in bundle.units] == ["PBMC_ctrl", "PBMC_stim"]
    for u in bundle.units:
        base = f"outs/per_sample_outs/{u.label}"
        assert [f.rel_path for f in u.files] == [f"{base}/count/sample_filtered_feature_bc_matrix.h5"]
        side = _side(u)
        assert side["metrics"] == [f"{base}/metrics_summary.csv"]
        assert side["vdj"] == [f"{base}/vdj_t/filtered_contig_annotations.csv"]
    skipped = {s.path: s.reason for s in bundle.skipped}
    assert "per-sample filtered table is used" in skipped["outs/multi/vdj_t/all_contig_annotations.csv"]
    # The library's raw matrix belongs to both samples: not guessed into one of them.
    assert not any("raw" in _side(u) for u in bundle.units)
    assert any("raw_feature_bc_matrix.h5" in i and "could not be matched" in i for i in bundle.issues)


def test_cellranger2_genome_folders(tmp_path):
    root = tmp_path / "up"
    dense = _raw_counts()
    bcs = [_bc(i) for i in range(dense.shape[1])]
    outs = root / "pbmc4k" / "outs"
    _write_mex(_mkdir(outs / "filtered_gene_bc_matrices" / "GRCh38"), "", dense[:, :200], bcs[:200])
    _write_mex(_mkdir(outs / "raw_gene_bc_matrices" / "GRCh38"), "", dense, bcs)
    (unit,) = build_bundle(_files(root), root=root).units
    assert unit.label == "pbmc4k" and unit.format == UnitFormat.tenx_mex and unit.complete
    assert all("/raw_gene_bc_matrices/GRCh38/" in p for p in _side(unit)["raw"])
    # With Cell Ranger 2's .h5 files beside them, the .h5 files are read and the folders skipped.
    _write_h5(outs / "filtered_gene_bc_matrices_h5.h5", dense[:, :200], bcs[:200])
    _write_h5(outs / "raw_gene_bc_matrices_h5.h5", dense, bcs)
    bundle = build_bundle(_files(root), root=root)
    (unit,) = bundle.units
    assert unit.format == UnitFormat.tenx_h5 and unit.label == "pbmc4k"
    assert _side(unit)["raw"] == ["pbmc4k/outs/raw_gene_bc_matrices_h5.h5"]
    assert len(bundle.skipped) == 6


def test_folder_labels_never_collide_and_plain_files_keep_their_names(tmp_path):
    root = tmp_path / "up"
    dense = _raw_counts()
    bcs = [_bc(i) for i in range(dense.shape[1])]
    for run in ("run1", "run2"):
        _write_h5(_mk(root / run / "sample" / "outs" / "filtered_feature_bc_matrix.h5"), dense[:, :200], bcs[:200])
    import anndata as ad

    ad.AnnData(pd.DataFrame([[1.0]], index=["c"], columns=["g"])).write_h5ad(_mk(root / "mine" / "atlas.h5ad"))
    labels = sorted(u.label for u in build_bundle(_files(root), root=root).units)
    assert labels == ["atlas", "sample", "sample_2"]


# ---------------------------------------------------------------------------
# The upload endpoint
# ---------------------------------------------------------------------------


@pytest.fixture()
async def ingest(tmp_data_dir, client):
    from scview.dependencies import get_ingest_session_manager
    from scview.main import app

    mgr = IngestSessionManager(str(tmp_data_dir))
    app.dependency_overrides[get_ingest_session_manager] = lambda: mgr
    yield client, mgr
    app.dependency_overrides.pop(get_ingest_session_manager, None)


def _form(root: Path, files: list[Path], paths: bool = True):
    parts = [("files", (p.name, p.read_bytes(), "application/octet-stream")) for p in files]
    data = {"paths": [p.relative_to(root).as_posix() for p in files]} if paths else None
    return parts, data


async def test_folder_upload_keeps_same_named_files_apart(ingest, tmp_path):
    client, mgr = ingest
    src = tmp_path / "src"
    count_outs(src / "pbmc" / "outs", h5=False)
    sid = mgr.create_session()
    files, data = _form(src, _files(src))
    r = await client.post(f"{API}/ingest/session/{sid}/files", files=files, data=data)
    assert r.status_code == 200, r.text
    staged = mgr.files_dir(sid)
    # Both barcodes.tsv.gz files and both filtered_contig_annotations.csv are there, byte for byte.
    for rel in ("pbmc/outs/filtered_feature_bc_matrix/barcodes.tsv.gz",
                "pbmc/outs/raw_feature_bc_matrix/barcodes.tsv.gz",
                "pbmc/outs/vdj_t/filtered_contig_annotations.csv",
                "pbmc/outs/vdj_b/filtered_contig_annotations.csv"):
        assert (staged / rel).read_bytes() == (src / rel).read_bytes(), rel
    # The BAM and the report never reached the staging folder; the state says so.
    assert not (staged / "pbmc/outs/possorted_genome_bam.bam").exists()
    body = r.json()
    (unit,) = body["bundle"]["units"]
    assert unit["format"] == "tenx_mex" and unit["label"] == "pbmc" and unit["complete"]
    assert {f["side_kind"] for f in unit["side_files"]} == {"metrics", "raw", "vdj"}
    sentence = body["bundle"]["issues"][0]
    assert "possorted_genome_bam.bam (read alignments (scView starts from the count matrix), 68 bytes)" in sentence
    assert body["validation"]["ok"]


async def test_plain_upload_is_staged_by_name_as_before(ingest, tmp_path):
    client, mgr = ingest
    sid = mgr.create_session()
    r = await client.post(f"{API}/ingest/session/{sid}/files",
                          files=[("files", ("../../evil/x.csv", b"a,b\n1,2\n", "text/csv"))])
    assert r.status_code == 200, r.text
    assert [p.name for p in mgr.staged_paths(sid)] == ["x.csv"]
    assert (mgr.files_dir(sid) / "x.csv").exists()


async def test_unsafe_or_mismatched_paths_are_refused(ingest):
    client, mgr = ingest
    sid = mgr.create_session()
    for paths in (["../x.csv"], ["/abs/x.csv"], ["C:\\x.csv"], ["a//x.csv"]):
        r = await client.post(f"{API}/ingest/session/{sid}/files",
                              files=[("files", ("x.csv", b"1", "text/csv"))], data={"paths": paths})
        assert r.status_code == 400, (paths, r.text)
    r = await client.post(f"{API}/ingest/session/{sid}/files",
                          files=[("files", ("x.csv", b"1", "text/csv")), ("files", ("y.csv", b"2", "text/csv"))],
                          data={"paths": ["x.csv"]})
    assert r.status_code == 400 and "one path per file" in r.json()["detail"]
    assert mgr.staged_paths(sid) == [] and list(mgr.files_dir(sid).iterdir()) == []


async def test_two_files_on_one_path_are_a_409_and_stage_nothing(ingest):
    client, mgr = ingest
    sid = mgr.create_session()
    url = f"{API}/ingest/session/{sid}/files"
    two = [("files", ("barcodes.tsv", b"AAAA-1\n", "text/plain")),
           ("files", ("barcodes.tsv", b"CCCC-1\n", "text/plain"))]
    r = await client.post(url, files=two, data={"paths": ["s/barcodes.tsv", "s/barcodes.tsv"]})
    assert r.status_code == 409 and "same path" in r.json()["detail"]
    assert list(mgr.files_dir(sid).iterdir()) == []  # no partial left behind either
    # Staged, then the same file again: kept once. A different file there: 409, the first stays.
    one = [("files", ("barcodes.tsv", b"AAAA-1\n", "text/plain"))]
    assert (await client.post(url, files=one, data={"paths": ["s/barcodes.tsv"]})).status_code == 200
    assert (await client.post(url, files=one, data={"paths": ["s/barcodes.tsv"]})).status_code == 200
    other = [("files", ("barcodes.tsv", b"GGGG-1\n", "text/plain")),
             ("files", ("new.tsv", b"x\n", "text/plain"))]
    r = await client.post(url, files=other, data={"paths": ["s/barcodes.tsv", "s/new.tsv"]})
    assert r.status_code == 409 and "already staged as s/barcodes.tsv" in r.json()["detail"]
    assert (mgr.files_dir(sid) / "s/barcodes.tsv").read_bytes() == b"AAAA-1\n"
    assert not (mgr.files_dir(sid) / "s/new.tsv").exists()  # all or nothing
    # A file where a folder is staged, and a folder under a staged file.
    r = await client.post(url, files=[("files", ("s", b"x", "text/plain"))], data={"paths": ["s"]})
    assert r.status_code == 409
    r = await client.post(url, files=[("files", ("y", b"x", "text/plain"))],
                          data={"paths": ["s/barcodes.tsv/y"]})
    assert r.status_code == 409


# ---------------------------------------------------------------------------
# Commit of a folder upload
# ---------------------------------------------------------------------------


async def test_commit_outs_folder_sequencing_qc_and_both_contig_tables(ingest, tmp_path, tmp_data_dir,
                                                                       dataset_manager):
    client, mgr = ingest
    src = tmp_path / "src"
    count_outs(src / "pbmc" / "outs", mex=True)
    sid = mgr.create_session()
    files, data = _form(src, _files(src))
    assert (await client.post(f"{API}/ingest/session/{sid}/files", files=files, data=data)).status_code == 200
    r = await client.post(f"{API}/ingest/session/{sid}/commit")
    assert r.status_code == 200, r.text
    did = r.json()["dataset_id"]
    a = (await dataset_manager.get_or_load_dataset(did)).adata
    assert a.n_obs == 200  # the filtered matrix; the raw one is a side file
    rec = json.loads(a.uns[seq_metrics.UNS_KEY])["samples"]["pbmc"]
    assert rec["files"] == {"metrics": "metrics_summary.csv", "raw": ["raw_feature_bc_matrix.h5"]}
    assert rec["knee"]["n_cells_matched"] == 200
    info = vdj.summary(a)
    assert sorted(f["stored_as"] for f in info["files"]) == [
        "pbmc/outs/vdj_b/filtered_contig_annotations.csv", "pbmc/outs/vdj_t/filtered_contig_annotations.csv"]
    receptor = a.obs[vdj.RECEPTOR_COL].astype(str)
    assert (receptor.iloc[:40] == "TCR alpha-beta").all() and (receptor.iloc[100:130] == "BCR").all()
    assert (receptor.iloc[40:100] == "none").all()  # all_contig_annotations.csv was not read too
    # The originals are kept in their folders, the two barcodes files side by side.
    up = tmp_data_dir / "uploads" / did / "pbmc" / "outs"
    assert (up / "filtered_feature_bc_matrix.h5").exists() and (up / "raw_feature_bc_matrix.h5").exists()
    assert (up / "vdj_b" / "filtered_contig_annotations.csv").exists()
    assert not (up / "analysis").exists() and not (up / "web_summary.html").exists()


def test_preflight_sees_the_folder_units(tmp_data_dir):
    # core/preflight.py builds its own bundle from staged_paths: same units as the state.
    from scview.core.preflight import session_preflight

    mgr = IngestSessionManager(str(tmp_data_dir))
    sid = mgr.create_session()
    count_outs(mgr.files_dir(sid) / "pbmc" / "outs")  # .h5 and MEX copies of both matrices
    state_units = [u["label"] for u in mgr.get_state(sid)["bundle"]["units"]]
    body = session_preflight(mgr, sid, DatasetManager(str(tmp_data_dir)))
    assert state_units == ["pbmc"] and [u["label"] for u in body["units"]] == ["pbmc"]
    assert body["merge"] is None
