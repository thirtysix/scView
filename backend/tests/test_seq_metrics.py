"""Sequencing QC from Cell Ranger outputs (core/seq_metrics.py, api/v1/seqqc.py).

Expected values are written out by hand from the inputs (the metrics strings, the
counts each synthetic barcode is given) or come from DropletUtils' own barcodeRanks
code run in R (the knee / inflection numbers), never from this module's output.
"""

from __future__ import annotations

import gzip
import json
from pathlib import Path

import anndata as ad
import h5py
import numpy as np
import pytest
import scipy.sparse as sp

from scview.core import qc_report, seq_metrics as sm
from scview.core.claims import resolve_tag
from scview.core.ingestion import IngestSessionManager, build_bundle, detect_file
from scview.core.ingestion.bundling import FileRole, UnitFormat, sample_key
from scview.core.ingestion.detection import FileKind

API = "/api/v1"

# pbmc_1k_v3_metrics_summary.csv as 10x publishes it (Cell Ranger 3.0.0), verbatim:
# https://cf.10xgenomics.com/samples/cell-exp/3.0.0/pbmc_1k_v3/pbmc_1k_v3_metrics_summary.csv
PBMC_COUNT = (
    "Estimated Number of Cells,Mean Reads per Cell,Median Genes per Cell,Number of Reads,"
    "Valid Barcodes,Sequencing Saturation,Q30 Bases in Barcode,Q30 Bases in RNA Read,"
    "Q30 Bases in Sample Index,Q30 Bases in UMI,Reads Mapped to Genome,Reads Mapped Confidently "
    "to Genome,Reads Mapped Confidently to Intergenic Regions,Reads Mapped Confidently to "
    "Intronic Regions,Reads Mapped Confidently to Exonic Regions,Reads Mapped Confidently to "
    "Transcriptome,Reads Mapped Antisense to Gene,Fraction Reads in Cells,Total Genes Detected,"
    "Median UMI Counts per Cell\n"
    '"1,222","54,502","1,919","66,601,887",97.4%,70.8%,94.1%,90.2%,91.1%,92.7%,95.4%,92.4%,4.8%,'
    '31.1%,56.5%,53.7%,1.0%,94.9%,"18,391","6,628"\n'
)

# Rows of a real Cell Ranger multi (Flex) per-sample metrics_summary.csv, verbatim
# (github.com/kevin-lee-aifi/mm_bm_chip_organoids, EXP-01244/OR00001_1_metrics_summary.csv).
FLEX_MULTI = """Category,Library Type,Grouped By,Group Name,Metric Name,Metric Value
Cells,Gene Expression,,,Cells,"2,616"
Cells,Gene Expression,,,Confidently mapped reads in cells,75.39%
Cells,Gene Expression,,,Estimated UMIs from genomic DNA,0.13%
Cells,Gene Expression,,,Mean reads per cell,"14,468"
Cells,Gene Expression,,,Median UMI counts per cell,"4,004"
Cells,Gene Expression,,,Median genes per cell,"1,885"
Cells,Gene Expression,,,Reads confidently mapped to probe set,98.03%
Cells,Gene Expression,,,Total genes detected,"14,430"
Cells,Gene Expression,Physical library ID,GEX_1,Cells detected in this sample,"2,616 (5.29%)"
Library,Gene Expression,Fastq ID,Pool1,Number of reads,"2,011,911,001"
Library,Gene Expression,Fastq ID,Pool1,Q30 GEM barcodes,96.6%
Library,Gene Expression,Fastq ID,Pool1,Q30 RNA read,95.9%
Library,Gene Expression,Fastq ID,Pool1,Q30 UMI,96.9%
Library,Gene Expression,Fastq ID,Pool1,Q30 barcodes,96.0%
Library,Gene Expression,Physical library ID,GEX_1,Confidently mapped reads in cells,91.33%
Library,Gene Expression,Physical library ID,GEX_1,Estimated number of cells,"49,435"
Library,Gene Expression,Physical library ID,GEX_1,Mean reads per cell,"40,698"
Library,Gene Expression,Physical library ID,GEX_1,Number of reads,"2,011,911,001"
Library,Gene Expression,Physical library ID,GEX_1,Reads confidently mapped to probe set,80.50%
Library,Gene Expression,Physical library ID,GEX_1,Sequencing saturation,59.40%
Library,Gene Expression,Physical library ID,GEX_1,Valid GEM barcodes,82.00%
Library,Gene Expression,Physical library ID,GEX_1,Valid UMIs,99.97%
Library,Gene Expression,Physical library ID,GEX_1,Valid barcodes,70.31%
Library,Gene Expression,Probe barcode ID,BC005,Cells per probe barcode,"2,616 (5.29%)"
Library,Gene Expression,Probe barcode ID,BC005,Sample ID,OR00001_1
"""

# Cell Ranger 10 multi writes fractions as decimals (layout of auto_process_ngs'
# CELLPLEX_METRICS_SUMMARY_10_0_0), with an Antibody Capture row to be ignored.
CELLPLEX_10 = """Category,Library Type,Grouped By,Group Name,Metric Name,Metric Value
Cells,Gene Expression,,,Cells,1571
Cells,Gene Expression,,,Confidently mapped antisense,0.036641947005987205
Cells,Gene Expression,,,Confidently mapped to transcriptome,0.8415785517777279
Cells,Gene Expression,,,Median UMI counts per cell,6692
Cells,Gene Expression,,,Median genes per cell,2472
Library,Gene Expression,Fastq ID,SC_GE,Q30 RNA read,0.9279107008913702
Library,Gene Expression,Physical library ID,SC_GE,Cells,14680
Library,Gene Expression,Physical library ID,SC_GE,Confidently mapped reads in cells,0.9694591347685292
Library,Gene Expression,Physical library ID,SC_GE,Confidently mapped to transcriptome,0.8409228816526795
Library,Gene Expression,Physical library ID,SC_GE,Mean reads per cell,52565
Library,Gene Expression,Physical library ID,SC_GE,Sequencing saturation,0.6903867704556981
Library,Gene Expression,Physical library ID,SC_GE,Valid barcodes,0.9760066973801198
Library,Antibody Capture,Physical library ID,SC_AB,Valid barcodes,0.31
"""


# ---------------------------------------------------------------------------
# Synthetic 10x files
# ---------------------------------------------------------------------------


def _bc(i: int) -> str:
    return "".join("ACGT"[(i >> (2 * k)) & 3] for k in range(16)) + "-1"


N_GENES, N_AB = 30, 2


def _raw_counts(n_cells=200, n_empty=2800):
    """features x barcodes: cells get (i % 7 + 1) UMIs in each of the 30 genes plus 50 in
    each antibody; empties get (j % 4) UMIs in gene 0 only (so a quarter are 0)."""
    n = n_cells + n_empty
    m = np.zeros((N_GENES + N_AB, n), dtype=np.int64)
    for i in range(n_cells):
        m[:N_GENES, i] = i % 7 + 1
        m[N_GENES:, i] = 50
    for j in range(n_empty):
        m[0, n_cells + j] = j % 4
    return m


def _write_h5(path: Path, dense: np.ndarray, barcodes: list[str]) -> Path:
    csc = sp.csc_matrix(dense)
    names = [f"GENE{i}" for i in range(N_GENES)] + [f"AB{i}_TotalSeqB" for i in range(N_AB)]
    types = ["Gene Expression"] * N_GENES + ["Antibody Capture"] * N_AB
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
        ft.create_dataset("genome", data=np.array(["GRCh38"] * len(names), dtype="S"))
        ft.create_dataset("_all_tag_keys", data=np.array([b"genome"]))
    return path


def _write_mex(d: Path, prefix: str, dense: np.ndarray, barcodes: list[str]) -> list[Path]:
    rows, cols = np.nonzero(dense)
    mtx = d / f"{prefix}matrix.mtx.gz"
    with gzip.open(mtx, "wt") as fh:
        fh.write("%%MatrixMarket matrix coordinate integer general\n%metadata_json: {}\n")
        fh.write(f"{dense.shape[0]} {dense.shape[1]} {len(rows)}\n")
        for r, c in zip(rows, cols):
            fh.write(f"{r + 1} {c + 1} {dense[r, c]}\n")
    bcs = d / f"{prefix}barcodes.tsv.gz"
    with gzip.open(bcs, "wt") as fh:
        fh.write("".join(b + "\n" for b in barcodes))
    feats = d / f"{prefix}features.tsv.gz"
    with gzip.open(feats, "wt") as fh:
        for i in range(N_GENES):
            fh.write(f"ID{i}\tGENE{i}\tGene Expression\n")
        for i in range(N_AB):
            fh.write(f"IDAB{i}\tAB{i}_TotalSeqB\tAntibody Capture\n")
    return [mtx, bcs, feats]


def _gex_totals(dense: np.ndarray) -> np.ndarray:
    return dense[:N_GENES].sum(axis=0)


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("raw,kind,want", [
    ('"1,222"', "int", 1222), ("54,502", "int", 54502), ("97.4%", "frac", 0.974),
    ("0.976", "frac", 0.976), ("95.2", "frac", 0.952), ("0.13%", "frac", 0.0013),
    ('"2,616 (5.29%)"', "int", 2616), ("", "int", None), ("NA", "frac", None),
    ("---", "int", None), ("n/a", "frac", None), ("1", "frac", 1.0),
])
def test_parse_value_reads_cellranger_strings(raw, kind, want):
    got = sm.parse_value(raw, kind)
    assert got == pytest.approx(want) if want is not None else got is None


def test_count_metrics_match_the_web_summary():
    rec = sm.parse_metrics_text(PBMC_COUNT)
    v = rec["format"], rec["values"]
    assert v[0] == "count"
    vals = v[1]
    # The pbmc_1k_v3 web summary's numbers, typed in from the file above.
    assert vals["estimated_cells"] == 1222 and vals["mean_reads_per_cell"] == 54502
    assert vals["median_genes_per_cell"] == 1919 and vals["median_umi_per_cell"] == 6628
    assert vals["n_reads"] == 66601887 and vals["total_genes_detected"] == 18391
    assert vals["valid_barcodes"] == pytest.approx(0.974)
    assert vals["sequencing_saturation"] == pytest.approx(0.708)
    assert vals["frac_reads_in_cells"] == pytest.approx(0.949)
    assert vals["conf_transcriptome"] == pytest.approx(0.537)
    assert vals["q30_rna_read"] == pytest.approx(0.902)
    assert vals["antisense"] == pytest.approx(0.010)
    assert "valid_umis" not in vals  # Cell Ranger 3 had no Valid UMIs column
    assert len(rec["raw"]) == 20 and rec["raw"][0] == ["Estimated Number of Cells", "1,222"]
    # Every metric is on the good side of 10x's lines: no flags.
    assert sm.flags_for(rec) == []


def test_count_metrics_with_missing_columns_and_antibody_columns():
    text = ("Estimated Number of Cells,Mean Reads per Cell,Antibody: Valid Barcodes,Valid Barcodes,"
            'Fraction Reads in Cells\n"85",,12.0%,,62.5%\n')
    rec = sm.parse_metrics_text(text)
    assert rec["values"] == {"estimated_cells": 85, "frac_reads_in_cells": pytest.approx(0.625)}
    levels = {f["metric"]: f["level"] for f in sm.flags_for(rec)}
    assert levels == {"estimated_cells": "warn", "frac_reads_in_cells": "warn"}


def test_flex_multi_per_library_layout():
    rec = sm.parse_metrics_text(FLEX_MULTI)
    vals, scope = rec["values"], rec["scope"]
    assert rec["format"] == "multi"
    assert vals["estimated_cells"] == 2616 and scope["estimated_cells"] == "cells"
    assert vals["mean_reads_per_cell"] == 14468 and scope["mean_reads_per_cell"] == "cells"
    assert vals["median_genes_per_cell"] == 1885 and vals["median_umi_per_cell"] == 4004
    assert vals["frac_reads_in_cells"] == pytest.approx(0.7539)  # the sample's, not the library's
    assert vals["valid_barcodes"] == pytest.approx(0.7031) and vals["valid_umis"] == pytest.approx(0.9997)
    assert vals["n_reads"] == 2011911001  # the physical library's row before the FASTQ's
    assert vals["q30_barcode"] == pytest.approx(0.96) and vals["q30_umi"] == pytest.approx(0.969)
    assert vals["sequencing_saturation"] == pytest.approx(0.594)
    assert vals["conf_probe_set"] == pytest.approx(0.805)
    flags = {f["metric"]: f for f in sm.flags_for(rec)}
    assert set(flags) == {"valid_barcodes", "mean_reads_per_cell"}
    assert flags["valid_barcodes"]["level"] == "warn" and flags["valid_barcodes"]["threshold"] == 0.75
    assert flags["mean_reads_per_cell"]["level"] == "info"
    assert "70.3%" in flags["valid_barcodes"]["message"] and "75.0%" in flags["valid_barcodes"]["message"]


def test_cellranger10_multi_decimals_and_other_library_types_ignored():
    rec = sm.parse_metrics_text(CELLPLEX_10)
    vals = rec["values"]
    assert vals["estimated_cells"] == 1571  # the sample's cells, not the library's 14,680
    assert vals["valid_barcodes"] == pytest.approx(0.976007, abs=1e-6)  # not the antibody's 0.31
    assert vals["frac_reads_in_cells"] == pytest.approx(0.969459, abs=1e-6)
    assert vals["conf_transcriptome"] == pytest.approx(0.840923, abs=1e-6)  # library first
    assert vals["antisense"] == pytest.approx(0.036642, abs=1e-6)
    assert vals["q30_rna_read"] == pytest.approx(0.927911, abs=1e-6)
    assert sm.flags_for(rec) == []


def test_old_multi_header_and_multiple_libraries():
    text = ("Library or Sample,Library Type,Grouped By,Group Name,Metric Name,Metric Value\n"
            "Sample,Gene Expression,,,Cells,5\n"
            "Library,Gene Expression,Physical library ID,GEX_1,Valid barcodes,40.0%\n"
            "Library,Gene Expression,Physical library ID,GEX_2,Valid barcodes,97.0%\n")
    rec = sm.parse_metrics_text(text)
    assert rec["values"]["estimated_cells"] == 5
    assert rec["values"]["valid_barcodes"] == pytest.approx(0.40)  # the first library's
    assert any("2 Gene Expression libraries" in n for n in rec["notes"])
    flags = {f["metric"]: f for f in sm.flags_for(rec)}
    # Multiplexed sample: 10x's per-sample line is 10 cells; valid barcodes < 50% is an error.
    assert flags["estimated_cells"]["level"] == "warn" and flags["estimated_cells"]["threshold"] == 10
    assert flags["valid_barcodes"]["level"] == "error" and flags["valid_barcodes"]["threshold"] == 0.5


def test_not_a_metrics_file():
    with pytest.raises(sm.SeqQcError):
        sm.parse_metrics_text("gene,cell1,cell2\nA,1,2\n")
    with pytest.raises(sm.SeqQcError):
        sm.parse_metrics_text("Estimated Number of Cells,Mean Reads per Cell\n")


def test_flag_boundaries_and_direction():
    def lv(**vals):
        return {f["metric"]: f["level"] for f in sm.flags_for({"format": "count", "values": vals})}

    assert lv(frac_reads_in_cells=0.70) == {}  # exactly on the line: no alert ("< 0.70")
    assert lv(frac_reads_in_cells=0.6999) == {"frac_reads_in_cells": "warn"}
    assert lv(frac_reads_in_cells=0.4999) == {"frac_reads_in_cells": "error"}
    assert lv(antisense=0.10) == {} and lv(antisense=0.11) == {"antisense": "warn"}
    assert lv(antisense=0.31) == {"antisense": "error"}
    assert lv(estimated_cells=0) == {"estimated_cells": "error"}
    assert lv(estimated_cells=99) == {"estimated_cells": "warn"}
    assert lv(q30_umi=0.70, conf_transcriptome=0.19) == {"q30_umi": "warn", "conf_transcriptome": "error"}
    order = [f["level"] for f in sm.flags_for(
        {"values": {"mean_reads_per_cell": 100, "q30_umi": 0.70, "conf_transcriptome": 0.19}})]
    assert order == ["error", "warn", "info"]


# ---------------------------------------------------------------------------
# Knee / inflection: DropletUtils barcodeRanks
# ---------------------------------------------------------------------------


def _cliff():
    cells = np.floor(10 ** (4 - 0.3 * np.arange(1000) / 1000))
    drop = np.floor(10 ** (3.7 - 1.7 * np.arange(200) / 200))
    empty = np.floor(10 ** (2 - 2 * np.arange(100_000) / 100_000))
    return np.concatenate([cells, drop, empty])


def _two_cliffs():
    a = np.floor(10 ** (4.3 - 0.1 * np.arange(500) / 500))
    d1 = np.floor(10 ** (4.2 - 0.7 * np.arange(100) / 100))
    b = np.floor(10 ** (3.5 - 0.2 * np.arange(3000) / 3000))
    d2 = np.floor(10 ** (3.3 - 1.6 * np.arange(300) / 300))
    empty = np.floor(10 ** (1.7 - 1.7 * np.arange(50_000) / 50_000))
    return np.concatenate([a, d1, b, d2, empty])


def _ties():
    return np.repeat(np.array([9000, 8000, 7000, 3000, 900, 400, 300, 200, 150, 120, 110, 105, 50, 3]),
                     [300, 300, 400, 50, 30, 200, 400, 800, 900, 1000, 3000, 5000, 9000, 20000])


# Expected: DropletUtils 1.33.2's barcodeRanks code (R/barcodeRanks.R, commit 8f18cce)
# run in R 4.4.0 on these exact totals (written out from the functions above).
@pytest.mark.parametrize("curve,knee,inflection", [
    (_cliff, 5486.3025157376, 324.8440931154),
    (_two_cliffs, 16012.7875490585, 9899.0423378997),  # the first knee + elbow are reported
    (_ties, 4085.0847692947, 2293.8166812512),  # runs of ties share their mid-rank
    (lambda: np.floor(10 ** (3 - np.arange(80) / 80)) + 101, 203.0, 203.0),  # no window fits
])
def test_barcode_ranks_matches_dropletutils(curve, knee, inflection):
    got = sm.barcode_ranks(curve())
    assert got["knee"] == pytest.approx(knee, rel=1e-9)
    assert got["inflection"] == pytest.approx(inflection, rel=1e-9)


def test_barcode_ranks_known_knee_and_too_few_points():
    t = _cliff()
    got = sm.barcode_ranks(t)
    # The cells' plateau is 10,000 -> 5,012 UMIs and the empties are <= 100: the knee sits at
    # the plateau's end, the inflection inside the cliff between the two.
    assert 5012 <= got["knee"] <= 10_000 and 100 < got["inflection"] < 5012
    assert np.isclose(sm.barcode_ranks(np.r_[t, np.zeros(10_000)])["knee"], got["knee"])  # zeros don't move it
    with pytest.raises(ValueError):
        sm.barcode_ranks(np.r_[np.full(60, 5000.0), np.full(500, 50.0)])  # 1 point above 100


# ---------------------------------------------------------------------------
# Raw matrices
# ---------------------------------------------------------------------------


def test_raw_h5_totals_count_gene_expression_only_in_blocks(tmp_path, monkeypatch):
    dense = _raw_counts()
    bcs = [_bc(i) for i in range(dense.shape[1])]
    path = _write_h5(tmp_path / "raw.h5", dense, bcs)
    monkeypatch.setattr(sm, "_NNZ_CHUNK", 37)  # many blocks, boundaries inside empties
    monkeypatch.setattr(sm, "_BC_CHUNK", 101)
    got_bc, got_tot, info = sm.raw_totals_h5(path)
    want = _gex_totals(dense)
    keep = want > 0
    assert list(got_bc) == [b for b, k in zip(bcs, keep) if k]
    assert got_tot.tolist() == want[keep].tolist()
    assert info["n_barcodes_in_file"] == 3000 and info["gex_features"] == N_GENES
    # Cells: 30 genes x (i % 7 + 1); the antibodies' 100 UMIs are not counted.
    assert got_tot[0] == 30 and got_tot[6] == 210


def test_raw_mex_totals_equal_the_h5_totals(tmp_path):
    dense = _raw_counts()
    bcs = [_bc(i) for i in range(dense.shape[1])]
    h5_bc, h5_tot, _ = sm.raw_totals_h5(_write_h5(tmp_path / "raw.h5", dense, bcs))
    mtx, bfile, ffile = _write_mex(tmp_path, "s1_raw_", dense, bcs)
    mex_bc, mex_tot, info = sm.raw_totals_mex(mtx, bfile, ffile)
    assert list(mex_bc) == list(h5_bc) and mex_tot.tolist() == h5_tot.tolist()
    assert info["gex_features"] == N_GENES


def test_knee_summary_against_the_filtered_cells():
    dense = _raw_counts()
    bcs = np.array([_bc(i) for i in range(dense.shape[1])])
    tot = _gex_totals(dense)
    keep = tot > 0
    cells = list(bcs[:200])
    s = sm.knee_summary(bcs[keep], tot[keep], cells)
    assert s["n_barcodes"] == int(keep.sum()) == 200 + 2100  # 700 empties have 0 UMIs
    assert s["n_cells"] == s["n_cells_matched"] == 200 and s["match"] == "barcode"
    assert s["cells_umi_min"] == 30 and s["cells_umi_median"] == 120.0  # 30 x (i % 7 + 1)
    assert s["frac_umis_in_cells"] == pytest.approx(tot[:200].sum() / tot.sum(), abs=1e-4)
    curve = s["curve"]
    assert len(curve["rank"]) <= sm.MAX_CURVE_POINTS and curve["rank"][0] == 1
    assert curve["rank"][-1] == s["n_barcodes"] and curve["umi"][0] == 210
    assert sum(curve["kept"]) >= 1 and curve["kept"][0] == 1 and curve["kept"][-1] == 0


def test_match_cells_falls_back_to_the_barcode_sequence():
    raw = np.array([_bc(i) for i in range(50)])
    merged_names = [f"{_bc(i)}:s1" for i in range(10)]  # merge_units' index_unique
    mask, found, how = sm.match_cells(raw, merged_names)
    assert how == "sequence" and found == 10 and mask[:10].all() and not mask[10:].any()
    prefixed = [f"s1_{_bc(i)[:-2]}" for i in range(5)]
    mask, found, how = sm.match_cells(raw, prefixed)
    assert how == "sequence" and found == 5
    mask, found, how = sm.match_cells(raw, ["nothing", "here"])
    assert found == 0 and not mask.any()


# ---------------------------------------------------------------------------
# Detection + bundling of side files
# ---------------------------------------------------------------------------


def _pbmc_like_files(d: Path, prefix: str = "s1_") -> dict[str, Path]:
    dense = _raw_counts()
    bcs = [_bc(i) for i in range(dense.shape[1])]
    return {
        "raw": _write_h5(d / f"{prefix}raw_feature_bc_matrix.h5", dense, bcs),
        "filtered": _write_h5(d / f"{prefix}filtered_feature_bc_matrix.h5", dense[:, :200], bcs[:200]),
        "metrics": _write_text(d / f"{prefix}metrics_summary.csv", PBMC_COUNT),
    }


def _write_text(p: Path, text: str) -> Path:
    p.write_text(text)
    return p


def test_detection_of_metrics_and_raw_matrices(tmp_path):
    assert detect_file(_write_text(tmp_path / "a.csv", PBMC_COUNT)).kind == FileKind.tenx_metrics
    assert detect_file(_write_text(tmp_path / "b.csv", FLEX_MULTI)).kind == FileKind.tenx_metrics
    files = _pbmc_like_files(tmp_path)
    raw = detect_file(files["raw"])
    assert raw.kind == FileKind.tenx_h5 and "name" in raw.details["raw_matrix"]
    assert "raw_matrix" not in detect_file(files["filtered"]).details
    # By content: whitelist-sized, nearly every barcode empty, a neutral name.
    n = 120_000
    dense = np.zeros((N_GENES + N_AB, n), dtype=np.int64)
    dense[:, :50] = 3
    big = _write_h5(tmp_path / "GSM1.h5", dense, [_bc(i) for i in range(n)])
    det = detect_file(big)
    assert "median of 0 genes" in det.details["raw_matrix"]
    big.rename(tmp_path / "GSM1_filtered.h5")
    assert "raw_matrix" not in detect_file(tmp_path / "GSM1_filtered.h5").details


@pytest.mark.parametrize("name,key", [
    ("pbmc_1k_v3_filtered_feature_bc_matrix.h5", "pbmc_1k_v3"),
    ("pbmc_1k_v3_raw_feature_bc_matrix.h5", "pbmc_1k_v3"),
    ("pbmc_1k_v3_metrics_summary.csv", "pbmc_1k_v3"),
    ("metrics_summary.csv", ""), ("filtered_feature_bc_matrix.h5", ""),
    ("sample_filtered_feature_bc_matrix.h5", ""), ("GSM1_raw.h5", "gsm1"),
    ("s1_raw_feature_bc_matrix_barcodes.tsv.gz", "s1"), ("S2.h5ad", "s2"),
])
def test_sample_key(name, key):
    assert sample_key(name) == key


def test_bundle_attaches_side_files_by_prefix(tmp_path):
    a = _pbmc_like_files(tmp_path, "s1_")
    b = _pbmc_like_files(tmp_path, "s2_")
    bundle = build_bundle([*a.values(), *b.values()])
    assert bundle.is_merge and [u.format for u in bundle.units] == [UnitFormat.tenx_h5] * 2
    for u, files in zip(bundle.units, (a, b)):
        assert [f.name for f in u.files] == [files["filtered"].name]
        assert sorted(f.name for f in u.side_files) == sorted([files["raw"].name, files["metrics"].name])
        assert all(f.role == FileRole.side for f in u.side_files)
    assert bundle.issues == []


def test_bundle_bare_names_and_unmatched_and_raw_only(tmp_path):
    d1 = tmp_path / "one"
    d1.mkdir()
    files = _pbmc_like_files(d1, "")
    bundle = build_bundle(list(files.values()))
    (u,) = bundle.units
    assert len(u.side_files) == 2 and not bundle.is_merge

    d2 = tmp_path / "two"
    d2.mkdir()
    a = _pbmc_like_files(d2, "s1_")
    _pbmc_like_files(d2, "s2_")
    stray = _write_text(d2 / "run3_metrics_summary.csv", PBMC_COUNT)
    bundle = build_bundle(sorted(d2.iterdir()))
    assert all(stray.name not in [f.name for f in x.side_files] for x in bundle.units)
    assert any("run3_metrics_summary.csv" in i and "could not be matched" in i for i in bundle.issues)

    d3 = tmp_path / "three"
    d3.mkdir()
    raw = Path(a["raw"]).rename(d3 / "s1_raw_feature_bc_matrix.h5")
    bundle = build_bundle([raw])
    assert bundle.units == [] and not bundle.complete
    assert "never loaded as the data" in bundle.issues[0]


def test_bundle_raw_mex_and_duplicate_metrics(tmp_path):
    dense = _raw_counts()
    bcs = [_bc(i) for i in range(dense.shape[1])]
    raw = _write_mex(tmp_path, "s1_raw_feature_bc_matrix_", dense, bcs)
    flt = _write_mex(tmp_path, "s1_filtered_feature_bc_matrix_", dense[:, :200], bcs[:200])
    m1 = _write_text(tmp_path / "s1_metrics_summary.csv", PBMC_COUNT)
    m2 = _write_text(tmp_path / "metrics_summary.csv", PBMC_COUNT)
    bundle = build_bundle([*raw, *flt, m1, m2])
    (u,) = bundle.units  # the raw triplet is not a second sample
    assert u.format == UnitFormat.tenx_mex and {f.name for f in u.files} == {p.name for p in flt}
    assert {f.name for f in u.side_files} == {p.name for p in raw} | {m1.name}
    assert any("already has a Cell Ranger metrics summary" in i for i in bundle.issues)


# ---------------------------------------------------------------------------
# Ingest commit + endpoints
# ---------------------------------------------------------------------------


def _stage(mgr: IngestSessionManager, files: dict[str, Path]) -> str:
    sid = mgr.create_session()
    for p in files.values():
        Path(p).rename(mgr.files_dir(sid) / Path(p).name)
    return sid


async def test_ingest_attaches_metrics_and_knee(client, dataset_manager, tmp_path):
    mgr = IngestSessionManager(data_dir=str(dataset_manager.data_dir))
    sid = _stage(mgr, _pbmc_like_files(tmp_path, "s1_"))
    did = await mgr.commit(sid, dataset_manager)
    adaptor = await dataset_manager.get_or_load_dataset(did)
    data = json.loads(adaptor.adata.uns[sm.UNS_KEY])
    (rec,) = data["samples"].values()
    assert rec["files"]["metrics"] == "s1_metrics_summary.csv"
    assert rec["files"]["raw"] == ["s1_raw_feature_bc_matrix.h5"] and rec["cells_source"] == "ingest"
    assert rec["knee"]["n_cells"] == rec["knee"]["n_cells_matched"] == 200
    assert rec["knee"]["cells_umi_min"] == 30
    assert adaptor.n_cells() == 200  # the raw matrix was not loaded as data

    r = await client.get(f"{API}/datasets/{did}/seq-metrics")
    assert r.status_code == 200, r.text
    body = r.json()
    (s,) = body["samples"]
    assert s["flags"] == [] and "No Cell Ranger alert fires" in s["summary"]
    assert s["metrics"]["values"]["estimated_cells"] == 1222
    assert {m["key"] for m in body["metrics_info"]} >= {"frac_reads_in_cells", "valid_barcodes"}


async def test_ingest_merged_samples_one_record_each(dataset_manager, tmp_path):
    mgr = IngestSessionManager(data_dir=str(dataset_manager.data_dir))
    files = {**{f"a{k}": v for k, v in _pbmc_like_files(tmp_path, "s1_").items()},
             **{f"b{k}": v for k, v in _pbmc_like_files(tmp_path, "s2_").items()}}
    sid = _stage(mgr, files)
    did = await mgr.commit(sid, dataset_manager)
    adaptor = await dataset_manager.get_or_load_dataset(did)
    samples = json.loads(adaptor.adata.uns[sm.UNS_KEY])["samples"]
    assert len(samples) == 2
    for rec in samples.values():
        assert rec["column"] == "sample"
        assert rec["knee"]["n_cells"] == rec["knee"]["n_cells_matched"] == 200  # ":label" stripped
        assert rec["knee"]["match"] == "barcode"


def _register(dm, a: ad.AnnData, did: str = "sq") -> str:
    d = Path(dm.data_dir) / "uploads" / did
    d.mkdir(parents=True)
    (d / "metadata.json").write_text(json.dumps({"id": did, "name": "my data", "status": "ready"}))
    a.write_h5ad(d / "data.h5ad")
    return did


def _cells_adata(names: list[str], sample: list[str] | None = None) -> ad.AnnData:
    a = ad.AnnData(X=sp.csr_matrix(np.ones((len(names), 3), dtype=np.float32)))
    a.obs_names = names
    if sample is not None:
        a.obs["sample"] = sample
    return a


async def test_attach_endpoint(client, dataset_manager, tmp_path):
    dense = _raw_counts()
    bcs = [_bc(i) for i in range(dense.shape[1])]
    did = _register(dataset_manager, _cells_adata(bcs[:150]))
    original = Path(dataset_manager.data_dir) / "uploads" / did / "data.h5ad"
    original.chmod(0o444)  # uploads are read-only
    r = await client.get(f"{API}/datasets/{did}/seq-metrics")
    assert r.status_code == 404
    raw = _write_h5(tmp_path / "raw_feature_bc_matrix.h5", dense, bcs)
    with open(raw, "rb") as fr:
        r = await client.post(
            f"{API}/datasets/{did}/seq-metrics",
            files=[("files", ("metrics_summary.csv", FLEX_MULTI.encode(), "text/csv")),
                   ("files", ("raw_feature_bc_matrix.h5", fr.read(), "application/octet-stream"))])
    assert r.status_code == 200, r.text
    (s,) = r.json()["samples"]
    assert s["sample"] == "my data" and s["cells_source"] == "dataset"
    assert s["knee"]["n_cells"] == 150 and s["knee"]["n_cells_matched"] == 150
    assert {f["metric"] for f in s["flags"]} == {"valid_barcodes", "mean_reads_per_cell"}
    derived = Path(dataset_manager.data_dir) / "derived" / did / "data.h5ad"
    on_disk = ad.read_h5ad(derived)
    assert json.loads(on_disk.uns[sm.UNS_KEY])["samples"]["my data"]["files"]["metrics"] == "metrics_summary.csv"
    # Only the uns entry was written: the matrix and cells are the upload's.
    assert list(on_disk.obs_names) == bcs[:150] and on_disk.X.sum() == 150 * 3
    assert sm.UNS_KEY not in ad.read_h5ad(original).uns  # the original is untouched
    r = await client.get(f"{API}/datasets/{did}/seq-metrics")
    assert r.status_code == 200 and r.json()["n_samples"] == 1

    r = await client.post(f"{API}/datasets/{did}/seq-metrics",
                          files=[("files", ("genes.csv", b"gene,a,b\nX,1,2\nY,3,4\n", "text/csv"))])
    assert r.status_code == 400 and "genes.csv" in r.json()["detail"]
    r = await client.post(f"{API}/datasets/ghost/seq-metrics",
                          files=[("files", ("metrics_summary.csv", PBMC_COUNT.encode(), "text/csv"))])
    assert r.status_code == 404


async def test_attach_to_one_sample_of_a_merged_dataset(client, dataset_manager):
    names = [f"{_bc(i)}:s1" for i in range(5)] + [f"{_bc(i)}:s2" for i in range(5)]
    did = _register(dataset_manager, _cells_adata(names, ["s1"] * 5 + ["s2"] * 5), "merged")
    up = [("files", ("metrics_summary.csv", PBMC_COUNT.encode(), "text/csv"))]
    r = await client.post(f"{API}/datasets/{did}/seq-metrics", files=up,
                          data={"column": "sample", "sample": "s2"})
    assert r.status_code == 200, r.text
    (s,) = r.json()["samples"]
    assert s["sample"] == "s2" and s["column"] == "sample" and s["knee"] is None
    r = await client.post(f"{API}/datasets/{did}/seq-metrics", files=up,
                          data={"column": "sample", "sample": "s9"})
    assert r.status_code == 404
    r = await client.post(f"{API}/datasets/{did}/seq-metrics", files=up, data={"column": "nope"})
    assert r.status_code == 404


# ---------------------------------------------------------------------------
# QC report + co-pilot
# ---------------------------------------------------------------------------


def test_report_and_copilot_facts():
    a = _cells_adata([_bc(i) for i in range(20)])
    assert sm.grounding_lines(a) == ([], "") and sm.report_facts(a) == ([], [])
    text = PBMC_COUNT.replace("94.9%", "62.0%")  # fraction reads in cells below 70%
    rec = {"sample": "pbmc", "metrics": sm.parse_metrics_text(text), "knee": None,
           "cells_source": "ingest"}
    sm.store(a, [rec])
    lines, detail = sm.grounding_lines(a)
    assert lines[1].startswith("- pbmc: ") and lines[1].endswith(f"[{sm.TAG}]")
    assert "62.0% reads in cells" in lines[1] and "Low fraction reads in cells" in detail
    facts, concerns = sm.report_facts(a)
    assert "1,222 estimated cells" in facts[0]
    assert concerns == ["Sample pbmc: low fraction reads in cells (62.0%; Cell Ranger's warn line "
                        "is 70.0%)."]


class _Adaptor:
    def __init__(self, adata):
        self.adata = adata

    def qc_per_cell(self, ribo=False):
        from scview.core.anndata_adaptor import per_cell_qc
        return per_cell_qc(self.adata, ribo=ribo)

    def obs_columns_info(self):
        return []


def test_qc_report_carries_the_sequencing_fact_and_concern():
    a = _cells_adata([_bc(i) for i in range(20)])
    rec = {"sample": "pbmc", "metrics": sm.parse_metrics_text(PBMC_COUNT.replace("97.4%", "61.0%")),
           "knee": None, "cells_source": "ingest"}
    sm.store(a, [rec])
    facts = qc_report.build_facts(_Adaptor(a))
    seq = [s for s in facts.quality if s.ref == sm.REPORT_TAG]
    assert len(seq) == 1 and "Sample pbmc" in seq[0].detail
    assert any(c.ref == sm.REPORT_TAG and "valid barcodes" in c.detail for c in facts.concerns)


def test_claims_resolve_the_seqqc_tag():
    refs = ["result:seqqc", "result:pca"]
    assert resolve_tag("result:seqqc", refs) == ["result:seqqc"]
    assert resolve_tag("seqqc", refs) == ["result:seqqc"]
    assert resolve_tag("seqqc:pbmc", refs) == ["result:seqqc"]
