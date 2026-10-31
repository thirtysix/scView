"""10x Multiome import (core/atac_import.py): peaks to the store, genes to X, every source."""

from __future__ import annotations

import gzip
import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

from scview.core import atac_import, modality_store, preflight
from scview.core.ingestion.bundling import UnitFormat, build_bundle
from scview.core.ingestion.detection import FileKind, detect_file
from scview.core.ingestion.feature_split import split_h5ad
from scview.core.ingestion.loading import IngestLoadError, load_unit
from scview.core.ingestion.session import IngestCommitError, IngestOptions, IngestSessionManager
from scview.core.ingestion.validation import validate_bundle
from tests import multiome_fixture as mf


def _session(tmp_path: Path, *files: Path, options: IngestOptions | None = None) -> tuple[IngestSessionManager, str]:
    m = IngestSessionManager(str(tmp_path / "data"))
    sid = m.create_session()
    for f in files:
        if f.is_dir():
            for g in f.iterdir():
                (m.files_dir(sid) / g.name).write_bytes(g.read_bytes())
        else:
            (m.files_dir(sid) / f.name).write_bytes(f.read_bytes())
    if options is not None:
        m.set_options(sid, options)
    return m, sid


def _dataset(m: IngestSessionManager, did: str) -> ad.AnnData:
    return ad.read_h5ad(next((m.data_dir / "ingested" / did).glob("*.h5ad")))


def _store(m: IngestSessionManager, did: str) -> modality_store.ModalityStore:
    return modality_store.ModalityStore(modality_store.store_path(m.data_dir, did, atac_import.STORE))


def _check(a: ad.AnnData, s: modality_store.ModalityStore, n_per: int = 40, seed: int = 0) -> None:
    """Genes in X, peaks in the store, exactly the matrix the file was made from."""
    g, p, barcodes, table = mf.matrices(n_per, seed)
    assert list(a.var_names) == [x[0] for x in mf.GENES]
    assert not any(str(v).startswith("chr") for v in a.var_names)
    assert np.array_equal(np.asarray(sp.csr_matrix(a.X).todense()), g.toarray())
    assert list(a.obs_names) == barcodes
    assert list(s.barcodes) == barcodes and list(s.features) == table["name"].tolist()
    got = s.read(atac_import.MATRIX, np.arange(s.n_obs))
    assert np.array_equal(got.toarray(), p.toarray())
    assert np.allclose(a.obs[atac_import.OBS_TOTAL], np.asarray(p.sum(axis=1)).ravel())
    assert np.array_equal(a.obs[atac_import.OBS_N_PEAKS], np.diff(p.indptr))
    assert list(s.feature_attr("chrom")) == table["chrom"].tolist()
    assert list(s.feature_attr("start")) == table["start"].astype(str).tolist()
    assert list(s.feature_attr("end")) == table["end"].astype(str).tolist()
    assert s.meta["genome"] == "GRCh38"
    info = atac_import.read_info(a)
    assert info["n_peaks"] == len(table) and info["genome"] == "GRCh38" and info["nnz"] == p.nnz
    assert "peak counts" in info["computed"]


@pytest.mark.parametrize("mode", ["auto", "on", "off"])
def test_tenx_h5_splits_peaks_whichever_way_the_genes_are_read(tmp_path, mode):
    h5 = mf.write_tenx_h5(tmp_path / "pbmc_filtered_feature_bc_matrix.h5")
    m, sid = _session(tmp_path, h5, options=IngestOptions(streamed=mode))
    did = m._build_dataset(sid)
    a = _dataset(m, did)
    with _store(m, did) as s:
        _check(a, s)
    assert "scview_protein" not in a.uns  # no other_features record: peaks are not "dropped"
    step = a.uns["scview_provenance"]["history"][-1] if isinstance(a.uns["scview_provenance"], dict) else None
    if step is not None:
        assert step["params"]["atac"]["n_peaks"] == 72


def test_mex_folder(tmp_path):
    folder = mf.write_mex(tmp_path / "mex")
    m, sid = _session(tmp_path, folder)
    did = m._build_dataset(sid)
    with _store(m, did) as s:
        _check(_dataset(m, did), s)


def test_mex_out_of_cell_order_is_read_again_by_buckets(tmp_path):
    folder = mf.write_mex(tmp_path / "mex")
    with gzip.open(folder / "matrix.mtx.gz", "rt") as fh:
        lines = fh.readlines()
    body = lines[2:]
    rng = np.random.default_rng(1)
    with gzip.open(folder / "matrix.mtx.gz", "wt") as fh:
        fh.writelines(lines[:2])
        fh.writelines(list(np.asarray(body, dtype=object)[rng.permutation(len(body))]))
    m, sid = _session(tmp_path, folder)
    did = m._build_dataset(sid)
    with _store(m, did) as s:
        _check(_dataset(m, did), s)


def test_h5ad_through_the_wizard(tmp_path):
    path = tmp_path / "multiome.h5ad"
    mf.as_anndata().write_h5ad(path)
    m, sid = _session(tmp_path, path)
    did = m._build_dataset(sid)
    with _store(m, did) as s:
        _check(_dataset(m, did), s)


@pytest.mark.parametrize("layout", ["csr", "csc", "dense"])
def test_plain_h5ad_upload_split_writes_the_store(tmp_path, layout):
    a = mf.as_anndata()
    if layout == "csc":
        a.X = a.X.tocsc()
    elif layout == "dense":
        a.X = a.X.toarray()
    src, dst, store = tmp_path / "m.h5ad", tmp_path / "out.h5ad", tmp_path / "atac.h5"
    a.write_h5ad(src)
    summary = split_h5ad(src, dst, original_filename="m.h5ad", atac_store=store)
    assert summary["n_peaks"] == 72 and summary["n_genes"] == 4 and summary["other_features"] == {}
    out = ad.read_h5ad(dst)
    with modality_store.ModalityStore(store) as s:
        _check(out, s)


def test_plain_h5ad_upload_without_a_store_path_only_leaves_peaks_out(tmp_path):
    src, dst = tmp_path / "m.h5ad", tmp_path / "out.h5ad"
    mf.as_anndata().write_h5ad(src)
    summary = split_h5ad(src, dst, original_filename="m.h5ad")
    assert summary["n_genes"] == 4 and "n_peaks" not in summary
    assert atac_import.UNS_KEY not in ad.read_h5ad(dst).uns


def test_h5mu_counts_in_x(tmp_path):
    path = mf.write_h5mu(tmp_path / "pbmc.h5mu")
    det = detect_file(path)
    assert det.kind == FileKind.h5mu and det.details["modalities"] == ["atac", "rna"]
    assert preflight.sniff_format(path) == "h5mu"
    m, sid = _session(tmp_path, path)
    bundle = m._bundle(sid)
    assert bundle.units[0].format == UnitFormat.h5mu
    did = m._build_dataset(sid)
    with _store(m, did) as s:
        _check(_dataset(m, did), s)


def test_h5mu_counts_in_a_layer(tmp_path):
    path = mf.write_h5mu(tmp_path / "pbmc.h5mu", atac_x="tfidf")
    m, sid = _session(tmp_path, path)
    did = m._build_dataset(sid)
    with _store(m, did) as s:
        _check(_dataset(m, did), s)


def test_h5mu_without_counts_leaves_atac_out_with_a_sentence(tmp_path):
    path = mf.write_h5mu(tmp_path / "pbmc.h5mu", atac_x="tfidf_only")
    m, sid = _session(tmp_path, path)
    did = m._build_dataset(sid)
    a = _dataset(m, did)
    assert list(a.var_names) == [x[0] for x in mf.GENES]
    assert atac_import.UNS_KEY not in a.uns
    assert not modality_store.store_path(m.data_dir, did, atac_import.STORE).exists()
    from scview.core import provenance

    assert "are not counts" in provenance.read_provenance(a)["history"][-1]["note"]


def test_only_peaks_is_refused_never_read_as_genes(tmp_path):
    h5 = mf.write_tenx_h5(tmp_path / "atac_only.h5", peaks_only=True)
    m, sid = _session(tmp_path, h5)
    report = validate_bundle(m._bundle(sid))
    assert not report.ok and report.issues[0].code == "atac_only"
    assert "ATAC-only data is not supported" in report.issues[0].message
    with pytest.raises(IngestCommitError, match="ATAC-only"):
        m._build_dataset(sid)


def test_only_peaks_in_memory_reader_is_refused(tmp_path):
    h5 = mf.write_tenx_h5(tmp_path / "atac_only.h5", peaks_only=True)
    bundle = build_bundle([h5])
    with pytest.raises(IngestLoadError, match="ATAC-only"):
        load_unit(bundle.units[0])


def test_metrics_side_file_adds_fractions(tmp_path):
    h5 = mf.write_tenx_h5(tmp_path / "pbmc_filtered_feature_bc_matrix.h5")
    barcodes = mf.matrices()[2]
    csv = tmp_path / "pbmc_per_barcode_metrics.csv"
    rows = {"barcode": ["AAAAAAAAAAAA-1", *barcodes], "is_cell": 1,
            "atac_raw_reads": 1000, "atac_mitochondrial_reads": 50, "atac_fragments": 400,
            "atac_TSS_fragments": 100, "atac_peak_region_fragments": 300}
    pd.DataFrame(rows).to_csv(csv, index=False)
    assert detect_file(csv).kind == FileKind.atac_metrics
    m, sid = _session(tmp_path, h5, csv)
    unit = m._bundle(sid).units[0]
    assert [f.side_kind.value for f in unit.side_files] == ["atac_metrics"]
    did = m._build_dataset(sid)
    a = _dataset(m, did)
    assert np.allclose(a.obs[atac_import.OBS_FRIP], 0.75)
    assert np.allclose(a.obs[atac_import.OBS_TSS], 0.25)
    assert np.allclose(a.obs[atac_import.OBS_MITO], 0.05)
    info = atac_import.read_info(a)
    assert "per-barcode metrics" in info["computed"] and info["metrics_files"] == [csv.name]


def test_metrics_of_another_sample_stop_the_commit_with_a_sentence(tmp_path):
    h5 = mf.write_tenx_h5(tmp_path / "pbmc_filtered_feature_bc_matrix.h5")
    csv = tmp_path / "pbmc_per_barcode_metrics.csv"
    pd.DataFrame({"barcode": ["TTTT-1"], "is_cell": [1], "atac_raw_reads": [1], "atac_mitochondrial_reads": [0],
                  "atac_fragments": [1], "atac_TSS_fragments": [0], "atac_peak_region_fragments": [1]}).to_csv(csv, index=False)
    m, sid = _session(tmp_path, h5, csv)
    with pytest.raises(IngestCommitError, match="None of the 80 cells"):
        m._build_dataset(sid)


def test_metrics_attached_later_to_a_merged_sample_only(tmp_path):
    a = ad.AnnData(X=sp.csr_matrix((3, 2), dtype=np.float32),
                   obs=pd.DataFrame({"sample": ["s1", "s1", "s2"]}, index=["A-1:s1", "B-1:s1", "A-1:s2"]))
    csv = tmp_path / "m.csv"
    pd.DataFrame({"barcode": ["A-1", "B-1"], "is_cell": [1, 1], "atac_raw_reads": [10, 10],
                  "atac_mitochondrial_reads": [1, 2], "atac_fragments": [4, 8], "atac_TSS_fragments": [1, 4],
                  "atac_peak_region_fragments": [2, 2]}).to_csv(csv, index=False)
    out = atac_import.attach_metrics(a, csv, sample_column="sample", sample="s1", suffix=":s1")
    assert out == {"n_cells": 2, "n_matched": 2, "file": "m.csv"}
    assert np.allclose(a.obs[atac_import.OBS_FRIP].to_numpy()[:2], [0.5, 0.25])
    assert np.isnan(a.obs[atac_import.OBS_FRIP].to_numpy()[2])


def test_merge_of_one_peak_set_keeps_a_store_addressed_by_merged_names(tmp_path):
    a = mf.write_tenx_h5(tmp_path / "s1_filtered_feature_bc_matrix.h5", seed=0)
    b = mf.write_tenx_h5(tmp_path / "s2_filtered_feature_bc_matrix.h5", seed=1)
    m, sid = _session(tmp_path, a, b)
    assert not [i for i in validate_bundle(m._bundle(sid)).issues if i.code == "atac_merge_left_out"]
    did = m._build_dataset(sid)
    merged = _dataset(m, did)
    with _store(m, did) as s:
        rows = s.rows_of(merged.obs_names)
        assert (rows >= 0).all() and s.n_obs == 160
        totals = np.asarray(s.read(atac_import.MATRIX, rows).sum(axis=1)).ravel()
    assert np.allclose(merged.obs[atac_import.OBS_TOTAL], totals)


def test_merge_of_different_peak_sets_leaves_atac_out_and_says_so(tmp_path):
    a = mf.write_tenx_h5(tmp_path / "s1_filtered_feature_bc_matrix.h5", seed=0)
    b = mf.write_tenx_h5(tmp_path / "s2_filtered_feature_bc_matrix.h5", seed=1, extra_peak=True)
    m, sid = _session(tmp_path, a, b)
    warn = [i for i in validate_bundle(m._bundle(sid)).issues if i.code == "atac_merge_left_out"]
    assert warn and "different peaks" in warn[0].message
    did = m._build_dataset(sid)
    merged = _dataset(m, did)
    assert atac_import.UNS_KEY not in merged.uns and merged.n_obs == 160 and merged.n_vars == 4
    assert not modality_store.store_path(m.data_dir, did, atac_import.STORE).exists()


def test_rna_only_datasets_are_untouched(tmp_path):
    a = mf.as_anndata()
    genes = a[:, a.var["feature_types"] == "Gene Expression"].copy()
    path = tmp_path / "rna.h5ad"
    genes.write_h5ad(path)
    m, sid = _session(tmp_path, path)
    assert validate_bundle(m._bundle(sid)).ok
    did = m._build_dataset(sid)
    out = _dataset(m, did)
    assert atac_import.UNS_KEY not in out.uns
    assert not any(c.startswith("atac_") for c in out.obs.columns)
    assert not modality_store.store_dir(m.data_dir, did).exists()


def test_preflight_counts_genes_in_memory_and_adds_a_store_line(tmp_path):
    h5 = mf.write_tenx_h5(tmp_path / "pbmc_filtered_feature_bc_matrix.h5")
    h = preflight.read_header(h5)
    g, p, *_ = mf.matrices()
    assert h.n_genes == 4 and h.n_cells == 80
    assert h.atac["n_peaks"] == 72 and abs(h.atac["nnz"] - p.nnz) < 0.05 * p.nnz  # from a sample of cells
    assert abs(h.matrices[0].nnz - g.nnz) < 0.2 * g.nnz
    est = preflight.estimate_load(h)
    assert any("72 peaks" in n and "stay on disk" in n for n in est.notes)
    plain = preflight.read_header(_rna_h5(tmp_path))
    assert plain.atac is None and not preflight.estimate_load(plain).notes


def _rna_h5(tmp_path: Path) -> Path:
    a = mf.as_anndata()
    genes = a[:, a.var["feature_types"] == "Gene Expression"].copy()
    path = tmp_path / "rna_only.h5ad"
    genes.write_h5ad(path)
    return path


def test_preflight_reads_h5mu_and_h5ad_headers(tmp_path):
    mu = preflight.read_header(mf.write_h5mu(tmp_path / "x.h5mu"))
    assert mu.format == "h5mu" and mu.n_genes == 4 and mu.atac["n_peaks"] == 72
    path = tmp_path / "m.h5ad"
    mf.as_anndata().write_h5ad(path)
    h = preflight.read_header(path)
    assert h.n_genes == 4 and h.x.n_genes == 4 and h.atac["n_peaks"] == 72


def test_interval_forms():
    assert atac_import.parse_interval("chr1:10-20") == ("chr1", 10, 20)
    assert atac_import.parse_interval("chr1-10-20") == ("chr1", 10, 20)
    assert atac_import.parse_interval("KI270713.1:31340-32243") == ("KI270713.1", 31340, 32243)
    assert atac_import.parse_interval("GL000218.1_5_9") == ("GL000218.1", 5, 9)
    assert atac_import.parse_interval("not a peak") is None


def test_info_round_trip_replaces_the_entry():
    a = ad.AnnData(X=np.zeros((1, 1), dtype=np.float32))
    assert atac_import.read_info(a) is None
    atac_import.write_info(a, {"n_peaks": 3})
    assert json.loads(a.uns[atac_import.UNS_KEY]) == {"n_peaks": 3}
