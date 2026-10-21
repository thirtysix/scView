"""Cluster and gene reports (core/report*.py, C064): section order and statuses on
sparse and dense data, with and without donors / conditions / mito genes; the
not-computed notes; stored results cited (marker databases, ORA, the removal
record, label confidence, the composition test); HTML escaping; the endpoints."""

from __future__ import annotations

import csv
import gzip
import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import scanpy as sc
import scipy.sparse as sp

from scview.core import annotation_confidence as conf
from scview.core import composition, provenance, qc_filter, report, report_html
from scview.core.anndata_adaptor import AnnDataAdaptor
from scview.core.pipeline import PipelineParams

API = "/api/v1"
EVIL = "<script>alert(1)</script>"
CLUSTER_IDS = ["identity", "labels", "markers", "enrichment", "qc", "composition", "methods"]
GENE_IDS = ["expression", "markers", "databases", "per_donor", "coexpression", "hvg", "gene_sets",
            "protein", "methods"]


def _adata(*, dense: bool = False, donors: bool = True, mito: bool = False, markers: bool = True,
           evil: bool = False) -> ad.AnnData:
    """240 cells x 64 genes: three cell types with their own up genes (G0-9, G10-19,
    G20-29), counts in layers, log-normalised X and .raw; 4 donors x 2 conditions."""
    rng = np.random.default_rng(0)
    n, g = 240, 60
    types = np.repeat(["A", "B", EVIL if evil else "C"], n // 3)
    counts = rng.poisson(0.5, (n, g)).astype(np.float32)
    for k in range(3):
        rows = slice(k * n // 3, (k + 1) * n // 3)
        counts[rows, k * 10:(k + 1) * 10] += rng.poisson(6.0, (n // 3, 10))
    names = [f"G{i}" for i in range(g)]
    extra = ["CD14", "RPL3", "RPS6", "MALAT1"]
    counts = np.hstack([counts, rng.poisson(1.0, (n, len(extra))).astype(np.float32)])
    names += extra
    if mito:
        counts = np.hstack([counts, rng.poisson(2.0, (n, 3)).astype(np.float32)])
        names += ["MT-CO1", "MT-ND1", "MT-ATP6"]
    lognorm = np.log1p(counts / counts.sum(axis=1, keepdims=True) * 1e4).astype(np.float32)
    a = ad.AnnData(lognorm if dense else sp.csr_matrix(lognorm))
    a.var_names = names
    a.obs_names = [f"c{i}" for i in range(n)]
    a.layers["counts"] = counts if dense else sp.csr_matrix(counts)
    a.raw = a.copy()
    a.obs["cell_type"] = pd.Categorical(types)
    a.obs["leiden"] = pd.Categorical([str(i % 4) for i in range(n)])
    a.uns["scview_active_clustering"] = "leiden"
    if donors:
        a.obs["donor"] = pd.Categorical([f"D{i % 4}" for i in range(n)])
        a.obs["condition"] = pd.Categorical(np.where((np.arange(n) // 4) % 2 == 0, "ctrl", "stim"))
    hv = np.zeros(a.n_vars, dtype=bool)
    hv[:30] = True
    a.var["highly_variable"] = hv
    a.var["means"] = np.asarray(lognorm.mean(axis=0))
    a.var["dispersions_norm"] = np.linspace(-1, 3, a.n_vars)
    if markers:
        sc.tl.rank_genes_groups(a, "cell_type", method="wilcoxon", pts=True,
                                key_added="rank_genes_groups__cell_type")
    provenance.record_step(a, step="normalization", tool="scanpy.pp.normalize_total",
                           params={"target_sum": 1e4})
    return a


def _Adaptor(a: ad.AnnData) -> AnnDataAdaptor:
    """A real adaptor over an in-memory AnnData (nothing is read from disk)."""
    adaptor = AnnDataAdaptor("unused.h5ad")
    adaptor._adata = a
    return adaptor


def _sec(rep: dict, sid: str) -> dict:
    return next(s for s in rep["sections"] if s["id"] == sid)


class _Loader:
    """An MSigDB loader holding a tiny HALLMARK collection."""

    def __init__(self, tmp: Path):
        self.dir = tmp
        self.sets = {"HALLMARK_A_UP": [f"G{i}" for i in range(10)] + ["X1", "X2"],
                     "HALLMARK_B_UP": [f"G{i}" for i in range(10, 20)],
                     "HALLMARK_OTHER": ["G40", "G41", "G42", "CD14"]}

    def get_collection_as_dict(self, cid):
        return dict(self.sets) if cid == "h.all" else {}

    def _find_file(self, cid):
        return self.dir / "h.all.json" if cid == "h.all" else None


def _marker_db(tmp: Path) -> Path:
    """A one-source marker database on disk (PanglaoDB-shaped table): type
    "A cells" lists G0-G9, "B cells" G10-G19, "Monocytes" CD14 and G40-G44."""
    dest = tmp / "marker_db"
    dest.mkdir()
    with gzip.open(dest / "panglao.tsv.gz", "wt", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["species", "cell_type", "cl_id", "gene", "weight"])
        for t, genes in (("A cells", [f"G{i}" for i in range(10)]),
                         ("B cells", [f"G{i}" for i in range(10, 20)]),
                         ("Monocytes", ["CD14"] + [f"G{i}" for i in range(40, 45)])):
            for i, gname in enumerate(genes):
                w.writerow(["human", t, "", gname, 10 - i * 0.5])
    return dest


# ---------------------------------------------------------------------------
# Cluster report
# ---------------------------------------------------------------------------


def test_cluster_report_sections_with_donors_sparse(tmp_path):
    a = _adata()
    rep = report.cluster_report(_Adaptor(a), "cell_type", "A", dataset_name="toy",
                                msigdb_loader=_Loader(tmp_path), marker_db_dir=_marker_db(tmp_path))
    assert [s["id"] for s in rep["sections"]] == CLUSTER_IDS
    assert rep["title"] == "toy - cluster A report" and rep["n_cells"] == 80
    assert rep["design"] == {"condition": "condition", "donor": "donor"}
    for s in rep["sections"]:
        assert set(s) == {"id", "title", "caption", "method", "status", "note", "data"}
        assert s["status"] in (report.OK, report.NOT_COMPUTED, report.NOT_APPLICABLE)

    ident = _sec(rep, "identity")
    assert ident["status"] == "ok" and ident["data"]["n_cells"] == 80 and ident["data"]["n_total"] == 240
    assert "80 of 240 cells (33.3%)" in ident["caption"] and "4 donors" in ident["caption"]
    assert sum(r["n_cells"] for r in ident["data"]["by_condition"]) == 80
    assert [r["donor"] for r in ident["data"]["by_donor"]] == ["D0", "D1", "D2", "D3"]

    markers = _sec(rep, "markers")
    genes = [g["gene"] for g in markers["data"]["genes"]]
    assert markers["status"] == "ok" and genes and set(genes) <= {f"G{i}" for i in range(10)}
    first = markers["data"]["genes"][0]
    assert first["logfc"] > 0 and first["pct_in"] > first["pct_out"] and first["padj"] < 0.05
    assert "adjusted p under 0.05" in markers["caption"] and str(markers["data"]["n_passing"]) in markers["caption"]
    db = markers["data"]["database"]
    assert db["status"] == "ok" and db["suggestions"][0]["cell_type"] == "A cells"
    assert set(db["suggestions"][0]["overlap"]) <= {f"G{i}" for i in range(10)}
    assert db["verdict"] == "supported" and db["db_type"] == "A cells"  # "A" matched by name

    enr = _sec(rep, "enrichment")
    assert enr["status"] == "ok" and enr["data"]["terms"][0]["term"] == "HALLMARK_A_UP"
    assert enr["data"]["n_sets"] == 3 and "of 3" in enr["caption"]

    comp = _sec(rep, "composition")
    by = {r["condition"]: r for r in comp["data"]["by_condition"]}
    assert by["ctrl"]["n_total"] == 120 and by["ctrl"]["share"] == pytest.approx(by["ctrl"]["n_cells"] / 120, abs=1e-4)
    assert len(comp["data"]["samples"]) == 8  # 4 donors x 2 conditions, one point each
    assert comp["data"]["paired"]["every_donor"] and "paired design" in comp["note"]
    assert "Compare > Composition" in comp["note"]  # no stored test to cite

    methods = _sec(rep, "methods")
    assert any("normalization" in s for s in methods["data"]["steps"])
    assert any(s.startswith("Marker genes:") for s in methods["data"]["sections"])


def test_cluster_report_dense_without_donors_or_mito():
    a = _adata(dense=True, donors=False)
    rep = report.cluster_report(_Adaptor(a), "cell_type", "B")
    assert [s["id"] for s in rep["sections"]] == CLUSTER_IDS
    assert rep["design"] == {"condition": None, "donor": None}
    comp = _sec(rep, "composition")
    assert comp["status"] == "not_applicable" and "No condition or donor column" in comp["note"]
    qc = _sec(rep, "qc")
    mito = next(m for m in qc["data"]["medians"] if m["metric"] == "pct_mt")
    assert mito["cluster"] is None and mito["note"] == "unmeasurable: no mitochondrial genes"
    assert qc["data"]["doublets"]["status"] == "not_computed"
    assert "doublet detection" in qc["data"]["doublets"]["note"]
    # MSigDB and the marker databases are absent: said, never silently missing
    enr = _sec(rep, "enrichment")
    assert enr["status"] == "not_computed" and "MSigDB" in enr["note"]
    db = _sec(rep, "markers")["data"]["database"]
    assert db["status"] == "not_computed" and "never downloads" in db["note"]
    labels = _sec(rep, "labels")
    assert labels["status"] == "ok" and labels["data"]["columns"][0]["column"] == "leiden"
    assert labels["data"]["confidence"]["status"] == "not_computed"


def test_cluster_report_measures_mito_when_genes_exist():
    a = _adata(mito=True)
    qc = _sec(report.cluster_report(_Adaptor(a), "cell_type", "A"), "qc")
    mito = next(m for m in qc["data"]["medians"] if m["metric"] == "pct_mt")
    assert qc["data"]["has_mt"] and mito["cluster"] > 0 and mito["note"] is None


def test_markers_not_computed_say_where_to_run_them():
    a = _adata(markers=False)
    rep = report.cluster_report(_Adaptor(a), "cell_type", "A")
    for sid in ("markers", "enrichment"):
        s = _sec(rep, sid)
        assert s["status"] == "not_computed" and "Unified View > Markers" in s["note"]
    # a column relabelled since its markers ran no longer matches them
    b = _adata()
    b.obs["cell_type"] = b.obs["cell_type"].cat.rename_categories({"C": "C2"})
    s = _sec(report.cluster_report(_Adaptor(b), "cell_type", "A"), "markers")
    assert s["status"] == "not_computed" and "changed since" in s["note"]


def test_unknown_column_value_and_per_cell_columns_are_refused():
    ada = _Adaptor(_adata())
    with pytest.raises(report.ReportError) as e:
        report.cluster_report(ada, "nope", "A")
    assert e.value.status == 404
    with pytest.raises(report.ReportError) as e:
        report.cluster_report(ada, "cell_type", "Z")
    assert e.value.status == 404
    ada.adata.obs["barcode"] = ada.adata.obs_names.astype(str)
    with pytest.raises(report.ReportError) as e:
        report.cluster_report(ada, "barcode", "c1")
    assert e.value.status == 422


def test_removal_record_and_qc_states_are_cited():
    a = _adata(mito=True)
    ada = _Adaptor(a)
    cells = ada.qc_per_cell(ribo=True)
    drop = np.zeros(a.n_obs, dtype=bool)
    drop[:10] = True  # ten cells of type A
    drop[100:103] = True  # three of B
    t = qc_filter.Thresholds.from_params(PipelineParams(qc_threshold_mode="fixed"))
    rec = qc_filter.removal_record(a, cells, {"min_genes": drop}, drop, t, "donor")
    kept = a[~drop].copy()
    kept.uns[qc_filter.RECORD_KEY] = rec
    kept.obs["qc_state"] = pd.Categorical(["main"] * (kept.n_obs - 5) + ["low-RP high-mito"] * 5)
    qc = _sec(report.cluster_report(_Adaptor(kept), "cell_type", "A"), "qc")
    removed = qc["data"]["removed"]
    assert removed["status"] == "ok" and removed["n_removed"] == 10 and removed["n_kept"] == 70
    assert removed["fraction_lost"] == pytest.approx(10 / 80, abs=1e-4)
    assert removed["by_criterion"] == {"min_genes": 10}
    states = qc["data"]["states"]
    assert states["status"] == "ok" and states["column"] == "qc_state"
    assert sum(s["n_cells"] for s in states["states"]) == 70
    # a record that no longer adds up is not used
    stale = kept[5:].copy()
    stale.uns[qc_filter.RECORD_KEY] = rec
    gone = _sec(report.cluster_report(_Adaptor(stale), "cell_type", "A"), "qc")["data"]["removed"]
    assert gone["status"] == "not_applicable" and "no longer adds up" in gone["note"]


def test_label_confidence_and_stored_composition_test_are_cited():
    a = _adata()
    a.obs["scview_conf_cell_type_flag"] = pd.Categorical(
        np.where(np.arange(a.n_obs) % 5 == 0, conf.UNSURE, conf.CONFIDENT))
    entry = {"columns": {"flag": "scview_conf_cell_type_flag"}, "alpha": 0.1, "classifier": "logreg",
             "labels": [{"label": "A", "n_cells": 80, "confident": 0.8}],
             "fingerprint": conf.fingerprint(conf._label_values(a, "cell_type"))}
    a.uns[conf.REGISTRY_KEY] = json.dumps({"cell_type": entry})
    ada = _Adaptor(a)
    res = composition.composition_test(a.obs, "condition", "donor", "cell_type")
    composition.remember(ada, res)
    rep = report.cluster_report(ada, "cell_type", "A")
    check = _sec(rep, "labels")["data"]["confidence"]["checks"][0]
    assert check["column"] == "cell_type" and check["counts"][conf.UNSURE] == 16
    assert check["own_label"]["label"] == "A" and check["stale"] is False
    stored = _sec(rep, "composition")["data"]["stored_test"]
    assert stored is not None and stored["condition_key"] == "condition" and set(stored["props"]) == {"ctrl", "stim"}


def test_a_failing_section_is_reported_not_raised(monkeypatch):
    from scview.core import report_cluster

    def boom(ctx):
        raise RuntimeError("bad stored result")

    monkeypatch.setattr(report_cluster, "_qc", boom)
    rep = report.cluster_report(_Adaptor(_adata()), "cell_type", "A")
    qc = _sec(rep, "qc")
    assert qc["status"] == "not_computed" and "bad stored result" in qc["note"]
    assert _sec(rep, "markers")["status"] == "ok"


# ---------------------------------------------------------------------------
# Gene report
# ---------------------------------------------------------------------------


def test_gene_report_sections_with_donors(tmp_path):
    a = _adata()
    rep = report.gene_report(_Adaptor(a), "G3", "cell_type", dataset_name="toy",
                             msigdb_loader=_Loader(tmp_path), marker_db_dir=_marker_db(tmp_path))
    assert [s["id"] for s in rep["sections"]] == GENE_IDS
    assert rep["title"] == "toy - gene G3 report" and rep["groupby"] == "cell_type"
    expr = _sec(rep, "expression")
    by = {g["group"]: g for g in expr["data"]["groups"]}
    assert by["A"]["mean"] > by["B"]["mean"] and by["A"]["n_cells"] == 80
    assert expr["data"]["layer"] == "raw" and expr["data"]["overall"]["n_cells"] == 240
    mk = _sec(rep, "markers")
    assert [g["group"] for g in mk["data"]["groups"]] == ["A"] and mk["data"]["groups"][0]["padj"] < 0.05
    db = _sec(rep, "databases")
    assert db["status"] == "ok" and [t["cell_type"] for t in db["data"]["types"]] == ["A cells"]
    assert db["data"]["types"][0]["rank"] == 4
    donor = _sec(rep, "per_donor")
    assert donor["status"] == "ok" and donor["data"]["design"]["paired"]
    assert {p["sample"] for p in donor["data"]["points"]} == {"D0", "D1", "D2", "D3"}
    co = _sec(rep, "coexpression")
    assert co["status"] == "ok" and co["data"]["positive"] and not co["data"]["subsampled"]
    assert {g["gene"] for g in co["data"]["positive"][:5]} & {f"G{i}" for i in range(10)}
    hvg = _sec(rep, "hvg")
    assert hvg["data"]["is_hvg"] and hvg["data"]["n_hvg"] == 30
    assert {s["field"] for s in hvg["data"]["stats"]} == {"means", "dispersions_norm"}
    gs = _sec(rep, "gene_sets")
    assert gs["data"]["n_sets"] == 1 and gs["data"]["sets"][0]["name"] == "HALLMARK_A_UP"
    prot = _sec(rep, "protein")
    assert prot["status"] == "not_applicable" and "no protein" in prot["note"]


def test_gene_report_dense_no_donors_default_groupby_and_not_hvg():
    a = _adata(dense=True, donors=False)
    rep = report.gene_report(_Adaptor(a), "G45")
    assert rep["groupby"] == "leiden"  # the active clustering
    assert _sec(rep, "per_donor")["status"] == "not_applicable"
    assert _sec(rep, "markers")["status"] == "not_computed"  # none stored for leiden
    assert _sec(rep, "gene_sets")["status"] == "not_computed"
    assert _sec(rep, "databases")["status"] == "not_computed"
    assert _sec(rep, "hvg")["data"]["is_hvg"] is False


def test_unknown_gene_is_a_404_with_a_close_match():
    ada = _Adaptor(_adata())
    with pytest.raises(report.ReportError) as e:
        report.gene_report(ada, "cd14", "cell_type")
    assert e.value.status == 404 and e.value.suggestion == "CD14"
    with pytest.raises(report.ReportError) as e:
        report.gene_report(ada, "QQQQQQ", "cell_type")
    assert e.value.status == 404 and e.value.suggestion is None


def test_protein_counterpart_from_the_panel():
    from scview.core import protein

    a = _adata()
    rng = np.random.default_rng(1)
    a.obsm[protein.CLR_KEY] = pd.DataFrame({"CD14": rng.normal(size=a.n_obs), "CD3": rng.normal(size=a.n_obs)},
                                           index=a.obs_names)
    prot = _sec(report.gene_report(_Adaptor(a), "CD14", "cell_type"), "protein")
    assert prot["status"] == "ok" and [p["protein"] for p in prot["data"]["proteins"]] == ["CD14"]
    assert len(prot["data"]["proteins"][0]["groups"]) == 3


# ---------------------------------------------------------------------------
# HTML
# ---------------------------------------------------------------------------


def test_html_escapes_every_dataset_string(tmp_path):
    a = _adata(evil=True)
    a.obs["cell_type"] = a.obs["cell_type"].cat.add_categories(["x"])
    rep = report.cluster_report(_Adaptor(a), "cell_type", EVIL, dataset_name=f"ds {EVIL}",
                                msigdb_loader=_Loader(tmp_path), marker_db_dir=_marker_db(tmp_path))
    page = report_html.render(rep)
    assert EVIL not in page and "<script" not in page.lower()
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in page
    assert page.startswith("<!doctype html>") and "@media print" in page
    assert f"<title>ds {EVIL}".replace("<script>", "&lt;script&gt;")[:20] in page
    gene_page = report_html.render(report.gene_report(_Adaptor(a), "G25", "cell_type"))
    assert EVIL not in gene_page and "&lt;script&gt;" in gene_page  # the group name in tables and SVG
    assert "http" not in page.replace("http://www.w3.org/2000/svg", "")  # nothing external


def test_html_renders_not_computed_sections_and_footer():
    rep = report.cluster_report(_Adaptor(_adata(markers=False, donors=False)), "cell_type", "A",
                                dataset_name="toy")
    page = report_html.render(rep)
    assert "Not computed" in page and "Unified View &gt; Markers" in page
    assert "Not applicable" in page and "Made with scView" in page
    assert "<svg" not in page  # no donors, no composition plot


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


def _register(dm, a: ad.AnnData, did: str = "rep1", name: str = "My data (v2)") -> str:
    d = Path(dm.data_dir) / "uploads" / did
    d.mkdir(parents=True, exist_ok=True)
    (d / "metadata.json").write_text(json.dumps({"id": did, "name": name, "status": "ready"}))
    a.write_h5ad(d / "data.h5ad")
    return did


async def test_report_endpoints(client, dataset_manager):
    did = _register(dataset_manager, _adata(evil=True))
    r = await client.get(f"{API}/datasets/{did}/report/cluster", params={"column": "cell_type", "value": "A"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["kind"] == "cluster" and [s["id"] for s in body["sections"]] == CLUSTER_IDS
    assert body["dataset"]["name"] == "My data (v2)"

    r = await client.get(f"{API}/datasets/{did}/report/cluster.html", params={"column": "cell_type", "value": EVIL})
    assert r.status_code == 200 and r.headers["content-type"].startswith("text/html")
    disp = r.headers["content-disposition"]
    assert disp.startswith("attachment; filename=") and "<" not in disp and " " not in disp.split("=", 1)[1]
    assert disp == 'attachment; filename="My_data_v2_cluster_script_alert_1_script_report.html"'
    assert EVIL not in r.text

    r = await client.get(f"{API}/datasets/{did}/report/gene", params={"gene": "G3", "groupby": "cell_type"})
    assert r.status_code == 200 and r.json()["gene"] == "G3"
    r = await client.get(f"{API}/datasets/{did}/report/gene.html", params={"gene": "G3"})
    assert r.status_code == 200 and "G3" in r.text
    assert r.headers["content-disposition"] == 'attachment; filename="My_data_v2_gene_G3_report.html"'

    r = await client.get(f"{API}/datasets/{did}/report/gene", params={"gene": "cd14"})
    assert r.status_code == 404 and r.json()["detail"] == {
        "message": "Gene 'cd14' not found in this dataset. Did you mean CD14?", "suggestion": "CD14"}
    r = await client.get(f"{API}/datasets/{did}/report/cluster", params={"column": "nope", "value": "A"})
    assert r.status_code == 404 and "nope" in r.json()["detail"]
    r = await client.get(f"{API}/datasets/ghost/report/gene", params={"gene": "G3"})
    assert r.status_code == 404


def test_copilot_can_open_the_report_tab():
    from scview.core.assistant import _coerce_actions

    acts = _coerce_actions('[{"type":"set_subtab","subtab":"report"}]', columns=["cell_type"],
                           embeddings=["X_umap"], genes_upper={}, n_cells=10, active_clustering="leiden")
    assert [(a.type, a.subtab) for a in acts] == [("set_subtab", "report")]
