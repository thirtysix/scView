"""Cluster QC cautions found by the v0.75 re-analysis round (core/cluster_qc.py): mito RNA in
nuclei (``nuclei_mito``), a cluster with another's identity but less RNA (``low_content``), and
marker lists led by non-coding or sex genes (``noncoding_markers``, ``sex_markers``). None of
them is a damage flag: AI annotation still asks the AI about these clusters."""

from __future__ import annotations

import anndata as ad
import numpy as np
import pandas as pd
import pytest

from scview.core import cluster_evidence, cluster_qc, marker_selection
from scview.core.pipeline import PipelineParams, decide_clusters, run_pipeline

N = 120


def _obs_data(genes: dict[str, float], counts: dict[str, float], mito: dict[str, tuple[float, float]],
              seed: int = 0) -> ad.AnnData:
    """QC numbers in obs, per cluster: median genes, counts, and (share of cells with mito reads,
    their % mito)."""
    rng = np.random.default_rng(seed)
    groups = list(genes)
    a = ad.AnnData(np.zeros((N * len(groups), 3), dtype="float32"))
    a.var_names = ["G0", "MT-CO1", "G2"]
    lab = np.repeat(groups, N)
    a.obs["cluster"] = pd.Categorical(lab, categories=groups)
    a.obs["n_genes_by_counts"] = np.concatenate([rng.normal(genes[g], genes[g] * 0.05, N) for g in groups])
    a.obs["total_counts"] = np.concatenate([rng.normal(counts[g], counts[g] * 0.05, N) for g in groups])
    pct = []
    for g in groups:
        share, level = mito[g]
        on = rng.random(N) < share
        pct.append(np.where(on, rng.gamma(2.0, level / 2.0, N), 0.0))
    a.obs["pct_counts_mt"] = np.concatenate(pct)
    return a


def _by(report):
    return {c.group: c for c in report.clusters}


NUCLEI = {"0": (0.05, 0.1), "1": (0.08, 0.1), "2": (0.04, 0.1), "3": (0.06, 0.1), "4": (0.95, 1.2)}


def test_mito_in_nuclei_is_seen_where_the_median_cell_has_none():
    g = {k: 2000.0 for k in NUCLEI}
    a = _obs_data(g, {k: 5000.0 for k in NUCLEI}, NUCLEI)
    report = cluster_qc.cluster_qc(a, "cluster")
    rows = _by(report)
    assert report.nuclei_like_mito and "mt_detected" in report.metrics
    assert report.typical["median_pct_mt"] == 0.0  # the old view: 0% everywhere
    assert rows["4"].flags == ["nuclei_mito"]
    assert rows["4"].mt_detected > 0.9 and report.typical["mt_detected"] < 0.1
    assert "of its cells have mito reads vs" in rows["4"].reasons[0]
    for k in "0123":
        assert rows[k].flags == []
    # The evidence line says how many cells have mito reads, not "0.0% vs 0.0%"
    line = cluster_evidence.qc_line(rows["4"], report.typical)
    qc = line.text
    assert "mito reads in" in qc and "0.0% vs 0.0%" not in qc and "mito RNA in nuclei" in qc
    assert line.fired  # the evidence card opens on it
    assert not cluster_evidence.qc_line(rows["0"], report.typical).fired
    lines, _ = cluster_qc.grounding_block(report)
    assert any("Mitochondrial RNA in nuclei" in ln for ln in lines)
    # Not a damage flag: the annotator does not label it Low-quality cells itself
    decided, _ = decide_clusters(type("C", (), {"best": {}, "flagged": []})(), report, {k: ["X"] for k in NUCLEI})
    assert decided == {}


def test_mito_in_cells_is_left_to_the_percent_flags():
    # Cell data: every cell has mito reads, a high-mito type (hepatocytes) is real biology
    cells = {"0": (1.0, 5.0), "1": (1.0, 6.0), "2": (1.0, 5.5), "3": (1.0, 5.0), "4": (1.0, 7.0)}
    a = _obs_data({k: 2000.0 for k in cells}, {k: 5000.0 for k in cells}, cells)
    report = cluster_qc.cluster_qc(a, "cluster")
    assert not report.nuclei_like_mito and "mt_detected" not in report.metrics
    assert not any("nuclei_mito" in c.flags for c in report.clusters)
    assert "mito reads in" not in cluster_evidence.qc_line(report.clusters[0], report.typical).text


def test_mito_is_printed_with_the_digits_it_needs():
    assert [cluster_qc.mito_text(v) for v in (12.34, 0.228, 0.0201, 0.0004, 0.0, 0.0999, 0.99996)] == \
        ["12.3%", "0.23%", "0.020%", "<0.001%", "0%", "0.10%", "1.0%"]


# --- marker make-up: the stored markers are stubbed per test --------------------------------

def _stub_markers(monkeypatch, named: dict[str, list[str]], full: dict[str, list[str]] | None = None):
    def fake(adata, column, *, n=25, skip_unnamed=True, **_k):
        src = named if skip_unnamed or full is None else full
        return {g: v[:n] for g, v in src.items()}
    monkeypatch.setattr(marker_selection, "group_markers", fake)


def _lists(k: int = 25) -> dict[str, list[str]]:
    return {g: [f"{g}_gene{i}" for i in range(k)] for g in ("A", "B", "C", "D", "E")}


def _plain(groups=("A", "B", "C", "D", "E"), counts=None, genes=None) -> ad.AnnData:
    counts = counts or {}
    genes = genes or {}
    flat = {g: (0.9, 3.0) for g in groups}
    return _obs_data({g: genes.get(g, 2000.0) for g in groups}, {g: counts.get(g, 5000.0) for g in groups}, flat)


def test_sex_genes_leading_the_markers_are_named(monkeypatch):
    m = _lists()
    m["B"] = ["Xist", "Tsix"] + m["B"][:23]  # Emont 2022 fat Leiden 1
    m["C"] = ["Dcn", "Xist"] + m["C"][:23]
    m["D"] = m["D"][:5] + ["Xist"] + m["D"][5:24]  # 6th: not leading
    _stub_markers(monkeypatch, m)
    rows = _by(cluster_qc.cluster_qc(_plain(), "cluster"))
    assert rows["B"].flags == ["sex_markers"] and rows["B"].sex_markers == ["Xist", "Tsix"]
    assert rows["C"].sex_markers == ["Xist"] and "sex_markers" not in rows["D"].flags
    line = cluster_evidence.makeup_line(rows["B"])
    assert line is not None and line.kind == "makeup" and "Xist, Tsix" in line.text


def test_sex_genes_under_ensembl_ids_are_recognised(monkeypatch):
    a = _plain()
    a.var["feature_name"] = ["G0", "XIST", "G2"]
    a.var_names = ["ENSG00000000001", "ENSG00000229807", "ENSG00000000003"]
    m = _lists()
    m["A"] = ["ENSG00000229807"] + m["A"][:24]
    _stub_markers(monkeypatch, m)
    rows = _by(cluster_qc.cluster_qc(a, "cluster"))
    assert rows["A"].sex_markers == ["XIST"]


def test_small_rnas_and_pseudogenes_leading_the_markers(monkeypatch):
    m = _lists()
    # Tabula Sapiens liver Leiden 14 (Smart-seq2 cells), as the annotator sees it
    m["E"] = ["CST5", "SLX1B", "MIR1244-2", "POTEE", "SNORA12", "RN7SL3", "FTH1P20", "MTND5P11", "RNU4ATAC",
              "SCARNA9", "MIR1244-3", "POTEJ", "H3P16"] + m["E"][:12]
    m["D"] = m["D"][:21] + ["CES1P1", "FYTTD1P1", "MIR4435-2HG", "CASP3"]  # hepatocyte-like tail: fine
    _stub_markers(monkeypatch, m)
    rows = _by(cluster_qc.cluster_qc(_plain(), "cluster"))
    assert rows["E"].flags == ["noncoding_markers"] and len(rows["E"].noncoding_markers) == 11
    assert "11 of its top 25 markers" in rows["E"].reasons[0]
    assert rows["D"].flags == []
    pattern = cluster_qc.NONCODING_RE
    assert not any(pattern.match(g) for g in ("CASP3", "MAP4", "VAMP2", "MIR4435-2HG", "SNHG5", "TRAP1", "Cyp2e1"))
    assert all(pattern.match(g) for g in ("Mir205", "Snord118", "Rpl13-ps3", "MTRNR2L12", "EEF1A1P5"))


def test_unnamed_and_malat1_genes_leading_the_markers(monkeypatch):
    m, full = _lists(), _lists()
    full["C"] = ["Malat1", "Gm48099", "4931431B13Rik", "ENSMUSG00000095041", "Gm15581", "A630012P03Rik",
                 "Mageb18", "Acaca", "Pde3b", "Fasn"] + full["C"][:15]
    full["B"] = ["Gm48099", "Camk1d", "Gsn", "1700028E10Rik", "Cyp2e1"] + full["B"][:20]  # 2 of 10: fine
    _stub_markers(monkeypatch, m, full)
    rows = _by(cluster_qc.cluster_qc(_plain(), "cluster"))
    assert rows["C"].flags == ["noncoding_markers"] and rows["C"].noncoding_markers[:2] == ["Malat1", "Gm48099"]
    assert "6 of its top 10" in rows["C"].reasons[0]
    assert rows["B"].flags == []


def test_a_cluster_with_another_identity_and_less_rna(monkeypatch):
    m = _lists()
    # Wilson 2022 kidney: Seurat 0 is PT (cluster 1) with a third of its RNA
    m["E"] = m["A"][:13] + [f"E_gene{i}" for i in range(12)]
    m["D"] = m["A"][:13] + [f"D_gene{i}" for i in range(12)]  # also A's identity, but its RNA is A's
    _stub_markers(monkeypatch, m)
    a = _plain(counts={"E": 1718.0, "D": 4800.0, "A": 4816.0}, genes={"E": 1175.0, "D": 2300.0, "A": 2449.0})
    report = cluster_qc.cluster_qc(a, "cluster")
    rows = _by(report)
    assert rows["E"].flags == ["low_content"] and rows["E"].low_content_of in ("A", "D")
    assert rows["E"].low_content_shared == [13, 25]
    assert "Low RNA for its identity: 13 of its top 25 markers" in rows["E"].reasons[0]
    for g in "ABCD":
        assert rows[g].flags == [], (g, rows[g].reasons)
    line = cluster_evidence.content_line(rows["E"], rows)
    assert line is not None and line.fired and "with less RNA" in line.text
    # A real small-RNA type with its own markers is not one (Kang B cells vs DCs)
    m["E"] = m["A"][:7] + [f"E_gene{i}" for i in range(18)]
    assert _by(cluster_qc.cluster_qc(a, "cluster"))["E"].flags == []


def test_low_content_end_to_end_on_counts():
    """Five types plus a cluster of type A's cells with a fifth of the RNA (the run's own markers)."""
    rng = np.random.default_rng(3)
    g = 2000
    base = rng.gamma(0.6, 1.0, g)
    profiles = {}
    for k, t in enumerate("ABCDE"):
        p = base.copy()
        p[60 * k: 60 * k + 60] *= rng.uniform(8, 16, 60)
        profiles[t] = p / p.sum()

    def cells(p, n, lib):
        return rng.poisson(p * rng.lognormal(np.log(lib), 0.15, n)[:, None]).astype(np.float32)

    blocks = {t: cells(profiles[t], 200, 3000.0) for t in "ABCDE"}
    blocks["A_low"] = cells(profiles["A"], 120, 600.0)
    a = ad.AnnData(np.vstack(list(blocks.values())))
    a.var_names = [f"G{i}" for i in range(g)]
    a.obs_names = [f"c{i}" for i in range(a.n_obs)]
    a.obs["cluster"] = pd.Categorical(np.concatenate([[t] * len(v) for t, v in blocks.items()]),
                                      categories=list(blocks))
    a, res = run_pipeline(a, ["qc_metrics", "normalization", "log_transform", "marker_genes"],
                          PipelineParams(marker_columns=["cluster"]))
    assert res.errors == {}
    rows = _by(cluster_qc.cluster_qc(a, "cluster"))
    assert rows["A_low"].low_content_of == "A", rows["A_low"].reasons
    assert all("low_content" not in rows[t].flags for t in "ABCDE")


@pytest.mark.parametrize("flag", ["nuclei_mito", "low_content", "noncoding_markers", "sex_markers"])
def test_new_flags_have_help_words_and_no_damage(flag):
    from scview.core import pipeline

    assert flag in cluster_qc.FLAG_HELP and flag in cluster_evidence.FLAG_WORDS
    assert flag not in pipeline._DAMAGE_FLAGS and flag not in pipeline._QC_NOTE_FLAGS
