"""Marker off-target expression from the Human Protein Atlas (core/offtarget.py)."""

from __future__ import annotations

import dataclasses
import hashlib
import io
import json
import zipfile
from pathlib import Path

import anndata as ad
import httpx
import numpy as np
import pandas as pd
import pytest
import scanpy as sc
import scipy.sparse as sp

from scview.core import offtarget as ot
from scview.core.uns_keys import markers_key
from tests.large_fixture import counts_adata, register

API = "/api/v1"

TISSUES = ["liver", "pancreas", "kidney", "lymph node", "spleen", "tonsil", "cerebellum",
           "cerebral cortex", "skin", "lung"]
CELLS = ["hepatocytes", "b-cells", "t-cells", "neutrophils", "astrocytes", "fibroblasts"]
# Positive controls (shapes as in HPA v25.1) and two made-up genes.
GENES = {
    "ALB": ("ENSG00000163631", [5000, 8, 6, 1, 1, 1, 0, 0, 2, 3], [4000, 1, 1, 1, 0, 2]),
    "PTPRC": ("ENSG00000081237", [5, 4, 3, 300, 280, 250, 2, 2, 6, 40], [1, 500, 600, 400, 3, 2]),
    "ACTB": ("ENSG00000075624", [2500, 2000, 3000, 2800, 2600, 2700, 1800, 1900, 3200, 3100],
             [900, 1100, 1000, 1200, 800, 1300]),
    "MS4A1": ("ENSG00000156738", [0.5, 0.2, 0.3, 100, 80, 120, 0, 0, 0.4, 2], [0.1, 900, 2, 1, 0, 0]),
    "GFAP": ("ENSG00000131095", [1, 1, 1, 1, 1, 1, 200, 900, 2, 1], [0, 0, 0, 0, 700, 3]),
    "QUIET1": ("ENSG00000999991", [0.1] * 10, [0.2] * 6),
}


def _zip(kind: str) -> bytes:
    spec = ot.FILES[kind]
    rows = ["\t".join(["Gene", "Gene name", spec.label_column, spec.value_column])]
    for sym, (ensg, tv, cv) in GENES.items():
        labels, vals = (TISSUES, tv) if kind == "tissue" else (CELLS, cv)
        rows += [f"{ensg}\t{sym}\t{lab}\t{v}" for lab, v in zip(labels, vals)]
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr(spec.filename.removesuffix(".zip"), "\n".join(rows) + "\n")
    return buf.getvalue()


@pytest.fixture()
def fixture_files(monkeypatch):
    """The two fixture zips, with FILES pinned to their sha256."""
    raw = {k: _zip(k) for k in ot.FILES}
    files = {k: dataclasses.replace(s, sha256=hashlib.sha256(raw[k]).hexdigest()) for k, s in ot.FILES.items()}
    monkeypatch.setattr(ot, "FILES", files)
    return raw


def _place(dest: Path, raw: dict[str, bytes]) -> Path:
    (dest / "raw").mkdir(parents=True, exist_ok=True)
    for k, b in raw.items():
        (dest / "raw" / ot.FILES[k].filename).write_bytes(b)
    return dest


# --- categories ---------------------------------------------------------------------------


def test_classify_follows_hpas_rules():
    assert ot.classify(np.array([0.5, 0.9, 0.0]), 5) == ("not_detected", [])
    assert ot.classify(np.array([40, 10, 1, 1]), 5) == ("enriched", [0])  # exactly 4x the next
    assert ot.classify(np.array([39, 10, 1, 1]), 5)[0] != "enriched"
    # group of two: mean 30 >= 4 x 5
    assert ot.classify(np.array([1, 30, 30, 5, 1, 1, 1]), 5) == ("group_enriched", [1, 2])
    # a group of six exceeds the tissue limit (5) but not the cell-type one (10)
    six = np.array([100] * 6 + [10] * 10, dtype=float)
    assert ot.classify(six, 5)[0] != "group_enriched"
    assert ot.classify(six, 10) == ("group_enriched", list(range(6)))
    # enhanced: 4x the mean of the others, but not 4x the next one
    assert ot.classify(np.array([60, 25, 15, 15, 10, 10, 10, 10, 10, 10]), 5) == ("enhanced", [0])
    assert ot.classify(np.array([10, 9, 11, 10, 12, 8]), 5) == ("low_specificity", [])
    # an elevated tissue must itself be detected (>= 1)
    assert ot.classify(np.array([1.2, 0.2, 0.0, 0.0]), 5) == ("enriched", [0])
    assert ot.classify(np.array([0.8, 0.0, 1.0, 0.9]), 5)[0] == "low_specificity"


def test_parse_keeps_regions_and_levels_merge_by_their_maximum(fixture_files):
    raw = ot.parse_zip(fixture_files["tissue"], ot.FILES["tissue"])
    assert sorted(raw.labels) == sorted(TISSUES)
    assert ot.at_level(raw, "region") is raw
    t = ot.at_level(raw, "tissue")
    labels = list(t.labels)
    assert "brain" in labels and "cerebellum" not in labels and len(labels) == 9
    i = t.find("GFAP")
    assert t.values[i, labels.index("brain")] == 900
    assert ot.at_level(raw, "tissue") is t  # cached
    o = ot.at_level(raw, "organ")
    assert sorted(o.labels) == sorted(["liver & gallbladder", "pancreas", "kidney & urinary bladder",
                                       "bone marrow & lymphoid tissues", "brain", "skin", "respiratory system"])
    j = o.find("PTPRC")
    assert o.values[j, list(o.labels).index("bone marrow & lymphoid tissues")] == 300
    with pytest.raises(ValueError):
        ot.at_level(raw, "planet")
    c = ot.parse_zip(fixture_files["cell_type"], ot.FILES["cell_type"])
    assert list(c.labels) == sorted(CELLS)


def test_every_hpa_tissue_has_an_organ():
    hpa_v25 = {"adipose tissue", "adrenal gland", "amygdala", "appendix", "basal ganglia", "blood vessel",
               "bone marrow", "breast", "cerebellum", "cerebral cortex", "cervix", "choroid plexus", "colon",
               "duodenum", "endometrium", "epididymis", "esophagus", "fallopian tube", "gallbladder",
               "heart muscle", "hippocampal formation", "hypothalamus", "kidney", "liver", "lung", "lymph node",
               "midbrain", "ovary", "pancreas", "parathyroid gland", "pituitary gland", "placenta", "prostate",
               "rectum", "retina", "salivary gland", "seminal vesicle", "skeletal muscle", "skin",
               "small intestine", "smooth muscle", "spinal cord", "spleen", "stomach", "testis", "thymus",
               "thyroid gland", "tongue", "tonsil", "urinary bladder", "vagina"}
    assert hpa_v25 == set(ot.ORGAN_OF)


def test_the_level_changes_the_category(tmp_path, fixture_files):
    _place(tmp_path, fixture_files)
    by = {lvl: {x["gene"]: x for x in ot.lookup(["GFAP", "PTPRC"], dest=tmp_path, auto_download=False,
                                                 level=lvl)["genes"]} for lvl in ot.TISSUE_LEVELS}
    assert by["tissue"]["GFAP"]["tissue"]["category"] == "enriched"
    # cerebral cortex 900 vs cerebellum 200: 4.5x, still enriched, now in one region
    assert by["region"]["GFAP"]["tissue"]["elevated"] == ["cerebral cortex"]
    # three lymphoid tissues become one organ: group enriched -> enriched
    assert by["tissue"]["PTPRC"]["tissue"]["category"] == "group_enriched"
    assert by["organ"]["PTPRC"]["tissue"]["category"] == "enriched"
    assert by["organ"]["PTPRC"]["tissue"]["elevated"] == ["bone marrow & lymphoid tissues"]
    r = ot.lookup(["GFAP"], dest=tmp_path, auto_download=False, level="region")
    assert r["level"] == "region" and len(r["heatmap"]["tissues"]) == 10
    with pytest.raises(ValueError):
        ot.lookup(["GFAP"], dest=tmp_path, auto_download=False, level="planet")


def test_an_old_merged_store_is_rebuilt_from_the_zip(tmp_path, fixture_files):
    _place(tmp_path, fixture_files)
    np.savez(tmp_path / "hpa_tissue.npz", labels=np.array(["brain"]))  # v0.39 store, ignored
    t = ot.load_table("tissue", tmp_path, auto_download=False)
    assert "cerebellum" in list(t.labels)


# --- fetch and store -------------------------------------------------------------------------


def test_install_downloads_checks_the_sha_and_stores_a_plain_npz(tmp_path, fixture_files):
    calls = []

    def handler(req: httpx.Request) -> httpx.Response:
        calls.append(str(req.url))
        kind = "tissue" if "tissue" in str(req.url) else "cell_type"
        return httpx.Response(200, content=fixture_files[kind])

    client = httpx.Client(transport=httpx.MockTransport(handler))
    t = ot.load_table("tissue", tmp_path, auto_download=True, client=client)
    assert calls == [ot.FILES["tissue"].url] and t.find("ALB") is not None
    with np.load(tmp_path / "hpa_tissue.v2.npz", allow_pickle=False) as z:  # no object arrays
        assert z["values"].dtype == np.float32
    meta = json.loads((tmp_path / "hpa_tissue.v2.json").read_text())
    assert meta["version"] == ot.HPA_VERSION and meta["licence"] == "CC BY-SA 4.0"
    assert meta["sha256"] == ot.FILES["tissue"].sha256 and meta["n_labels"] == 10  # regions kept
    ot.load_table("tissue", tmp_path, auto_download=True, client=client)
    assert len(calls) == 1  # on disk now


def test_a_wrong_download_is_refused_and_leaves_nothing(tmp_path, fixture_files):
    client = httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(200, content=b"<html>")))
    with pytest.raises(ot.HpaUnavailable, match="sha256"):
        ot.load_table("tissue", tmp_path, auto_download=True, client=client)
    assert not (tmp_path / "hpa_tissue.v2.npz").exists()
    assert not any((tmp_path / "raw").iterdir())


def test_offline_and_disabled_say_what_to_do(tmp_path, fixture_files):
    def down(req):
        raise httpx.ConnectError("no network")

    with pytest.raises(ot.HpaUnavailable, match="internet"):
        ot.load_table("tissue", tmp_path, auto_download=True, client=httpx.Client(transport=httpx.MockTransport(down)))
    with pytest.raises(ot.HpaUnavailable, match="HPA_AUTO_DOWNLOAD is off"):
        ot.load_table("cell_type", tmp_path, auto_download=False)


def test_a_hand_placed_zip_is_used_without_download(tmp_path, fixture_files):
    _place(tmp_path, fixture_files)
    t = ot.load_table("cell_type", tmp_path, auto_download=False)
    assert t.meta["served_from"] == "local file"
    (tmp_path / "raw" / ot.FILES["tissue"].filename).write_bytes(b"not the release")
    with pytest.raises(ot.HpaUnavailable, match="not HPA"):
        ot.load_table("tissue", tmp_path, auto_download=False)


# --- lookups --------------------------------------------------------------------------------------


def test_positive_controls(tmp_path, fixture_files):
    _place(tmp_path, fixture_files)
    r = ot.lookup(["ALB", "PTPRC", "ACTB", "MS4A1", "GFAP", "QUIET1", "NOTAGENE"], dest=tmp_path,
                  auto_download=False)
    g = {x["gene"]: x for x in r["genes"]}
    assert g["ALB"]["tissue"]["category"] == "enriched" and g["ALB"]["tissue"]["elevated"] == ["liver"]
    assert g["ALB"]["verdict"] == "restricted" and g["ALB"]["tissue"]["tau"] > 0.7
    assert g["ALB"]["url"] == "https://www.proteinatlas.org/ENSG00000163631"
    assert g["PTPRC"]["tissue"]["category"] == "group_enriched"
    assert set(g["PTPRC"]["cell_type"]["elevated"]) == {"b-cells", "t-cells", "neutrophils"}
    assert g["ACTB"]["tissue"]["category"] == "low_specificity" and g["ACTB"]["verdict"] == "broad"
    assert g["ACTB"]["tissue"]["tau"] < 0.2
    assert g["MS4A1"]["cell_type"]["category"] == "enriched"
    assert g["MS4A1"]["cell_type"]["top"][0] == {"name": "b-cells", "value": 900.0}
    assert g["GFAP"]["tissue"]["elevated"] == ["brain"]
    assert g["QUIET1"]["verdict"] == "not_detected"
    assert g["NOTAGENE"]["verdict"] == "not_in_hpa" and not g["NOTAGENE"]["found"]
    assert r["summary"]["counts"]["restricted"] == 4 and r["summary"]["counts"]["broad"] == 1
    hm = r["heatmap"]
    assert hm["genes"] == ["ALB", "PTPRC", "ACTB", "MS4A1", "GFAP", "QUIET1"] and len(hm["tissues"]) == 9
    assert hm["log2_ntpm"][0][hm["tissues"].index("liver")] == pytest.approx(np.log2(5001), abs=1e-3)
    assert "Human Protein Atlas" in r["attribution"] and r["note"] is None


def test_mouse_symbols_ensembl_ids_and_the_flag(tmp_path, fixture_files):
    _place(tmp_path, fixture_files)
    r = ot.lookup(["Alb", "Actb"], species="mouse", dest=tmp_path, auto_download=False)
    assert [x["hpa_symbol"] for x in r["genes"]] == ["ALB", "ACTB"] and "upper-case" in r["note"]
    r = ot.lookup(["ENSG00000156738.12"], dest=tmp_path, auto_download=False)
    assert r["genes"][0]["hpa_symbol"] == "MS4A1"
    # broad in HPA: flagged when specific in the dataset (or unknown there), not when not
    r = ot.lookup(["ACTB", "ACTB ", "ALB"], dest=tmp_path, auto_download=False, in_dataset_tau=[0.8, 0.1])
    assert [x["flag"] for x in r["genes"]] == [True, False]  # the repeat is dropped
    assert ot.lookup(["ACTB"], dest=tmp_path, auto_download=False)["genes"][0]["flag"] is True
    r = ot.lookup(["ACTB"], dest=tmp_path, auto_download=False, in_dataset_tau=[0.2])
    assert r["genes"][0]["flag"] is False and r["summary"]["n_flagged"] == 0
    with pytest.raises(ValueError):
        ot.lookup([" "], dest=tmp_path, auto_download=False)


# --- dataset groups and endpoints ---------------------------------------------------------------


def _data() -> ad.AnnData:
    """Two groups: "B" up in MS4A1 and ACTB, "L" up in ALB."""
    rng = np.random.default_rng(0)
    genes = ["MS4A1", "ACTB", "ALB", "PTPRC"] + [f"X{i}" for i in range(26)]
    labels = np.repeat(["B", "L"], 100)
    rate = np.full((200, len(genes)), 0.5)
    rate[labels == "B", 0:2] = 8.0
    rate[labels == "L", 2] = 8.0
    a = ad.AnnData(X=sp.csr_matrix(rng.poisson(rate).astype(np.float32)),
                   obs=pd.DataFrame({"type": pd.Categorical(labels)}, index=[f"c{i}" for i in range(200)]),
                   var=pd.DataFrame(index=genes))
    sc.pp.normalize_total(a, target_sum=1e4)
    sc.pp.log1p(a)
    sc.tl.rank_genes_groups(a, "type", method="wilcoxon", pts=True, key_added=markers_key("type"))
    return a


def test_for_group_takes_the_one_marker_choice(tmp_path, fixture_files):
    from scview.core.marker_selection import group_markers

    _place(tmp_path, fixture_files)
    a = _data()
    res = ot.for_group(a, "type", "B", 5, dest=tmp_path, auto_download=False)
    assert [g["gene"] for g in res["genes"]] == group_markers(a, "type", n=5)["B"]
    assert {"MS4A1", "ACTB"} <= {g["gene"] for g in res["genes"]}
    with pytest.raises(KeyError):
        ot.for_group(a, "type", "nope", dest=tmp_path, auto_download=False)
    a.obs["other"] = a.obs["type"]
    with pytest.raises(LookupError, match="Markers table"):
        ot.for_group(a, "other", "B", dest=tmp_path, auto_download=False)


def _register(dm, a: ad.AnnData, did: str = "off") -> str:
    d = Path(dm.data_dir) / "uploads" / did
    d.mkdir(parents=True)
    (d / "metadata.json").write_text(json.dumps({"id": did, "name": did, "status": "ready"}))
    a.write_h5ad(d / "data.h5ad")
    return did


async def test_endpoints_and_the_copilot_fact(client, dataset_manager, tmp_data_dir, test_settings,
                                             fixture_files):
    from scview.core.assistant import build_grounding_context

    test_settings.HPA_AUTO_DOWNLOAD = False  # never the network in tests

    did = _register(dataset_manager, _data())
    url = f"{API}/datasets/{did}/markers/offtarget"
    r = await client.get(url, params={"group_col": "type", "group": "B"})
    assert r.status_code == 503 and "HPA_AUTO_DOWNLOAD is off" in r.json()["detail"]
    _place(tmp_data_dir / "hpa", fixture_files)
    r = await client.get(url, params={"group_col": "type", "group": "B", "n": 10})
    assert r.status_code == 200, r.text
    body = r.json()
    g = {x["gene"]: x for x in body["genes"]}
    assert g["MS4A1"]["verdict"] == "restricted" and g["ACTB"]["verdict"] == "broad"
    assert g["ACTB"]["in_dataset_tau"] is not None and body["group"] == "B"
    assert (await client.get(url, params={"group_col": "nope", "group": "B"})).status_code == 404
    assert (await client.get(url, params={"group_col": "type", "group": "Z"})).status_code == 404
    assert (await client.get(url, params={"group_col": "type", "group": "B", "n": 51})).status_code == 422
    r = await client.get(url, params={"group_col": "type", "group": "B", "level": "organ"})
    assert r.status_code == 200 and r.json()["level"] == "organ"
    assert "bone marrow & lymphoid tissues" in r.json()["heatmap"]["tissues"]
    assert (await client.get(url, params={"group_col": "type", "group": "B", "level": "x"})).status_code == 422

    r = await client.post(f"{API}/offtarget/genes", json={"genes": ["ALB", "Ptprc"], "species": "mouse"})
    assert r.status_code == 200 and r.json()["genes"][1]["hpa_symbol"] == "PTPRC"
    assert (await client.post(f"{API}/offtarget/genes", json={"genes": []})).status_code == 422
    r = await client.post(f"{API}/offtarget/genes", json={"genes": ["GFAP"], "level": "region"})
    assert r.status_code == 200 and "cerebellum" in r.json()["heatmap"]["tissues"]
    st = (await client.get(f"{API}/offtarget/status")).json()
    assert st["tables"]["tissue"]["version"] == ot.HPA_VERSION

    context, sources = build_grounding_context(await dataset_manager.get_or_load_dataset(did), {"markers"})
    fact = next(s for s in sources if s.ref == ot.TAG)
    assert "off-target (HPA): type B markers:" in fact.evidence and "broadly expressed (e.g. ACTB" in fact.evidence
    assert f"[{ot.TAG}]" in context


def test_the_tag_resolves_for_claims():
    from scview.core.claims import resolve_tag

    assert resolve_tag("offtarget", [ot.TAG, "result:merge_check"]) == [ot.TAG]
    assert resolve_tag("result:offtarget", [ot.TAG]) == [ot.TAG]


async def test_large_dataset_needs_only_the_stored_markers(client, tmp_data_dir, test_settings, fixture_files):
    test_settings.HPA_AUTO_DOWNLOAD = False
    register(tmp_data_dir, "bigO", counts_adata())
    _place(tmp_data_dir / "hpa", fixture_files)
    r = await client.get(f"{API}/datasets/bigO/markers", params={"groupby_column": "leiden", "format": "json"})
    assert r.status_code == 200, r.text
    r = await client.get(f"{API}/datasets/bigO/markers/offtarget", params={"group_col": "leiden", "group": "0"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["genes"] and all(g["verdict"] == "not_in_hpa" for g in body["genes"])  # synthetic names


def test_cell_type_enrichment_needs_specificity():
    """A gene "group enriched" among cell types at low tau (FTL) is not restricted."""
    tissue = {"category": "enhanced", "tau": 0.19}
    assert ot._verdict(tissue, {"category": "group_enriched", "tau": 0.43}) == "enhanced"
    assert ot._verdict(tissue, {"category": "enriched", "tau": 0.8}) == "restricted"
    assert ot._verdict({"category": "group_enriched", "tau": 0.7}, {"category": "enhanced", "tau": 0.3}) == "restricted"
    assert ot._verdict(None, {"category": "enriched", "tau": 0.9}) == "restricted"
