"""The standard Cell Ranger ARC outs file set uploaded together lands as ONE multiome unit.

ARC writes, each name led by the sample: filtered_feature_bc_matrix.h5, atac_fragments.tsv.gz
(+ .tbi), per_barcode_metrics.csv, summary.csv, atac_peaks.bed, atac_peak_annotation.tsv and
web_summary.html. Before v0.78 the BED and the peak annotation were each read as a dense-table
sample, so the upload became a three-sample merge without ATAC, the fragments were not attached,
and summary.csv could not be matched by name. Now the peak files and the report are skipped
with a reason, summary.csv is the sample's metrics side file, and the fragments are attached.
"""

from __future__ import annotations

from pathlib import Path

import anndata as ad
import pandas as pd

from scview.core import atac_fragments as af
from scview.core import atac_import, seq_metrics
from scview.core.ingestion import IngestSessionManager
from scview.core.ingestion.bundling import SideKind, build_bundle
from scview.core.ingestion.detection import FileKind, detect_file
from scview.core.ingestion.paths import ARC_PEAK_ANNOTATION, ARC_PEAKS, REPORT, junk_reason
from tests import multiome_fixture as mf
from tests.fragments_fixture import write_fragments

PREFIX = "5k_Mouse_Kidney_MO_gemx_"
ARC_HEADER = ("# id=5k_Mouse_Kidney_MO_gemx\n# pipeline_name=cellranger-arc\n# pipeline_version=2.2.0\n"
              "# reference_version=2024-A\n#\n")
SUMMARY_HEADER = ("Sample ID,Genome,Pipeline version,Estimated number of cells,Feature linkages detected,"
                  "ATAC Median high-quality fragments per cell,GEX Median genes per cell,GEX Mean raw reads per cell")


def _arc_outs(folder: Path) -> list[Path]:
    """Tiny stand-ins for the eight top-level ARC files, with ARC's real names."""
    folder.mkdir(parents=True, exist_ok=True)
    h5 = mf.write_tenx_h5(folder / f"{PREFIX}filtered_feature_bc_matrix.h5")
    barcodes = mf.matrices()[2]
    lines = sorted((c, i * 7, i * 7 + 60, barcodes[i % len(barcodes)], 1) for c in ("chr1", "chr2") for i in range(200))
    frag, idx = write_fragments(folder / f"{PREFIX}atac_fragments.tsv.gz", lines, header=ARC_HEADER)
    metrics = folder / f"{PREFIX}per_barcode_metrics.csv"
    pd.DataFrame({"barcode": barcodes, "is_cell": 1, "atac_raw_reads": 1000, "atac_mitochondrial_reads": 50,
                  "atac_fragments": 400, "atac_TSS_fragments": 100,
                  "atac_peak_region_fragments": 300}).to_csv(metrics, index=False)
    summary = folder / f"{PREFIX}summary.csv"
    summary.write_text(SUMMARY_HEADER + "\n5k_Mouse_Kidney_MO_gemx,GRCm39,cellranger-arc-2.2.0,80,10,15213,3525,40000\n")
    bed = folder / f"{PREFIX}atac_peaks.bed"
    bed.write_text(ARC_HEADER + "".join(f"chr1\t{i * 1000}\t{i * 1000 + 500}\n" for i in range(1, 30)))
    ann = folder / f"{PREFIX}atac_peak_annotation.tsv"
    ann.write_text("chrom\tstart\tend\tgene\tdistance\tpeak_type\n"
                   + "".join(f"chr1\t{i * 1000}\t{i * 1000 + 500}\t\t\tintergenic\n" for i in range(1, 30)))
    html = folder / f"{PREFIX}web_summary.html"
    html.write_text("<!DOCTYPE html>\n<html><head><meta charset=\"utf-8\" /></head><body>report</body></html>\n")
    return [h5, frag, idx, metrics, summary, bed, ann, html]


def test_the_peak_files_and_report_are_skipped_with_a_reason():
    assert junk_reason(f"{PREFIX}atac_peaks.bed") == ARC_PEAKS
    assert junk_reason(f"{PREFIX}atac_peak_annotation.tsv") == ARC_PEAK_ANNOTATION
    assert junk_reason(f"{PREFIX}web_summary.html") == REPORT
    assert junk_reason(f"{PREFIX}gex_molecule_info.h5") is not None
    for kept in ("filtered_feature_bc_matrix.h5", "atac_fragments.tsv.gz", "atac_fragments.tsv.gz.tbi",
                 "per_barcode_metrics.csv", "summary.csv"):
        assert junk_reason(PREFIX + kept) is None, kept


def test_arc_summary_gex_metrics_are_read(tmp_path):
    from scview.core.seq_metrics import parse_metrics

    path = _arc_outs(tmp_path / "outs")[4]
    vals = parse_metrics(path)["values"]
    assert vals["estimated_cells"] == 80 and vals["median_genes_per_cell"] == 3525
    assert vals["mean_reads_per_cell"] == 40000


def test_the_whole_outs_set_is_one_multiome_unit_with_its_side_files(tmp_path):
    files = _arc_outs(tmp_path / "outs")
    assert detect_file(tmp_path / "outs" / f"{PREFIX}summary.csv").kind == FileKind.tenx_metrics
    bundle = build_bundle(files)
    assert len(bundle.units) == 1 and not bundle.is_merge and bundle.complete
    unit = bundle.units[0]
    assert unit.label == "5k_Mouse_Kidney_MO_gemx_filtered_feature_bc" or unit.label.startswith("5k_Mouse_Kidney")
    kinds = sorted((f.side_kind.value, f.spatial_part or "") for f in unit.side_files)
    assert kinds == [("atac", "fragments"), ("atac", "index"), ("atac_metrics", ""), ("metrics", "")]
    assert {s.path for s in bundle.skipped} == {f"{PREFIX}atac_peaks.bed", f"{PREFIX}atac_peak_annotation.tsv",
                                                 f"{PREFIX}web_summary.html"}
    # The one issue is the sentence that says what was skipped and why; nothing unmatched.
    assert len(bundle.issues) == 1 and "same peaks are in the matrix" in bundle.issues[0]
    assert atac_import.has_peaks(unit)


def test_summary_matches_its_sample_among_several(tmp_path):
    a = _arc_outs(tmp_path / "a")
    b = tmp_path / "b"
    b.mkdir()
    other = mf.write_tenx_h5(b / "other_sample_filtered_feature_bc_matrix.h5", seed=1)
    bundle = build_bundle([a[0], a[4], other])
    by_label = {u.label: u for u in bundle.units}
    kidney = next(u for label, u in by_label.items() if label.lower().startswith("5k_mouse_kidney"))
    assert [f.side_kind for f in kidney.side_files] == [SideKind.metrics]
    assert not any("could not be matched" in i for i in bundle.issues)


def test_the_upload_commits_as_one_dataset_with_its_fragments_attached(tmp_path):
    m = IngestSessionManager(str(tmp_path / "data"))
    sid = m.create_session()
    for f in _arc_outs(tmp_path / "outs"):
        (m.files_dir(sid) / f.name).write_bytes(f.read_bytes())
    validation = m.validate(sid) if hasattr(m, "validate") else None
    if validation is not None:
        assert "atac_merge_left_out" not in [i.code for i in validation.issues]
    did = m._build_dataset(sid)
    a = ad.read_h5ad(next((m.data_dir / "ingested" / did).glob("*.h5ad")))
    assert atac_import.OBS_TOTAL in a.obs.columns and atac_import.OBS_FRIP in a.obs.columns
    info = atac_import.read_info(a)
    assert info is not None and info["metrics_files"] == [f"{PREFIX}per_barcode_metrics.csv"]
    st = af.status(m.data_dir, did)
    assert st["attached"] and st["barcode_share"] == 1.0
    assert seq_metrics.UNS_KEY in a.uns  # summary.csv read as the sample's metrics
