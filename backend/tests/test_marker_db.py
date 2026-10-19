"""Marker databases (core/marker_db.py, core/marker_names.py): fetch, parse, name
matching, group markers, the support verdict and suggestions, the API and the
marker-score annotator.

The fixtures in tests/fixtures/marker_db are invented rows in the PanglaoDB TSV,
CellMarker 2.0 (the xlsx is written from the TSV here), CellMarker 3.0 (zipped here)
and ACT per-tissue (Latin-1) layouts, with ACT's per-file sha256 list, and a
nine-term Cell Ontology. ``_serve`` leaves CellMarker 3.0 out unless asked
(``cm3=True``), so most tests run the CellMarker 2.0 fallback path; downloads go through httpx.MockTransport, so no test touches the
internet. The real numbers (Kang lookup 0.615, paul15 0.579, name coverage 1.00)
were measured on the downloaded databases and are in the batch-8 report.
"""

from __future__ import annotations

import dataclasses
import gzip
import hashlib
import json
import threading
from pathlib import Path
from urllib.parse import quote

import anndata as ad
import httpx
import numpy as np
import pandas as pd
import pytest

from scview.core import marker_db as mdb
from scview.core import marker_names as mn

API = "/api/v1"
FIX = Path(__file__).parent / "fixtures" / "marker_db"
URLS = {"panglao": ("https://first.test/pg.tsv.gz", "https://mirror.test/pg.tsv.gz"),
        "cellmarker2": ("http://cm.test/Cell_marker_All.xlsx",), "act": ("http://act.test/download.jsp",)}
CM3_BASE = "https://cm3.test/file/"
CM3_ZIPS = {"human_cell_marker.zip": "human_cell_marker.txt", "mouse_cell_marker.zip": "mouse_cell_marker.txt"}
CM3_URLS = [CM3_BASE + f for f in CM3_ZIPS]
URLS["cellmarker3"] = (CM3_BASE,)
ACT_BASE = "http://act.test/download/"
ACT_FILES = ["Human/Human_Blood.txt", "Human/Human_Bone marrow.txt", "Mouse/Mouse_Blood.txt"]
ACT_URLS = [ACT_BASE + quote(f) for f in ACT_FILES]


def _panglao_bytes() -> bytes:
    return gzip.compress((FIX / "panglao_sample.tsv").read_bytes(), mtime=0)


def _cellmarker_bytes(tmp_path: Path) -> bytes:
    import openpyxl

    wb = openpyxl.Workbook()
    ws = wb.active
    for line in (FIX / "cellmarker_sample.tsv").read_text().splitlines():
        cells = line.split("\t")
        ws.append([int(c) if c.isdigit() and len(c) == 4 else c for c in cells])  # PMIDs as numbers, as in the real file
    p = tmp_path / "cm_fixture.xlsx"
    wb.save(p)
    return p.read_bytes()


def _cm3_zips() -> dict[str, bytes]:
    """The CellMarker 3.0 fixture split by species into the two zips, as the site does."""
    import io
    import zipfile

    head, *rows = (FIX / "cellmarker3_sample.tsv").read_text(encoding="utf-8").splitlines()
    out = {}
    for zname, member in CM3_ZIPS.items():
        keep = [r for r in rows if (r.split("\t")[0] == "Mouse") == zname.startswith("mouse")]
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
            info = zipfile.ZipInfo(member, date_time=(2026, 7, 22, 0, 0, 0))
            z.writestr(info, "\n".join([head, *keep]) + "\n")
        out[zname] = buf.getvalue()
    return out


@pytest.fixture()
def raw(tmp_path) -> dict[str, bytes]:
    return {"panglao": _panglao_bytes(), "cellmarker2": _cellmarker_bytes(tmp_path),
            "ontology": (FIX / "cl_sample.obo").read_bytes(),
            "act": {f: (FIX / "act" / f).read_bytes() for f in ACT_FILES},
            "cellmarker3": _cm3_zips()}


@pytest.fixture(autouse=True)
def small_sources(monkeypatch, raw):
    """Fixture-sized sources pinned to the fixtures' sha256, and no real network."""
    for name, spec in list(mdb.SOURCES.items()):
        pinned = None if name in ("act", "cellmarker3") else hashlib.sha256(raw[name]).hexdigest()
        monkeypatch.setitem(mdb.SOURCES, name, dataclasses.replace(spec, urls=URLS[name], sha256=pinned, min_rows=1))
    monkeypatch.setattr(mdb, "CM3_BASE", CM3_BASE)
    monkeypatch.setattr(mdb, "CM3_FILES", tuple(
        (f, hashlib.sha256(b).hexdigest(), len(b)) for f, b in raw["cellmarker3"].items()))
    monkeypatch.setattr(mdb, "ACT_BASE", ACT_BASE)
    monkeypatch.setattr(mdb, "ACT_SUMS", FIX / "act_sample.sha256")
    monkeypatch.setattr(mdb, "ACT_DELAY", 0.0)
    monkeypatch.setattr(mdb, "ONTOLOGY_URL", "https://cl.test/cl-basic.obo")
    monkeypatch.setattr(mdb, "ONTOLOGY_MIN_TERMS", 5)

    def no_network(client):
        if client is None:
            raise AssertionError("a test reached for the real network")
        return client, False

    monkeypatch.setattr(mdb, "_client", no_network)
    mdb._loaded.clear()


def _serve(raw: dict[str, bytes], fail: set[str] = frozenset(), calls: list | None = None,
           cm3: bool = False, on_request=None) -> httpx.Client:
    by_url = {URLS["panglao"][0]: raw["panglao"], URLS["panglao"][1]: raw["panglao"],
              URLS["cellmarker2"][0]: raw["cellmarker2"], "https://cl.test/cl-basic.obo": raw["ontology"],
              **{u: raw["act"][f] for u, f in zip(ACT_URLS, ACT_FILES)},
              **({u: raw["cellmarker3"][f] for u, f in zip(CM3_URLS, CM3_ZIPS)} if cm3 else {})}

    def handler(req: httpx.Request) -> httpx.Response:
        url = str(req.url)
        if calls is not None:
            calls.append(url)
        if on_request:
            on_request(req)
        if url in fail or url not in by_url:
            return httpx.Response(504)
        body = by_url[url]
        rng = req.headers.get("Range")
        if rng:  # "bytes=N-": the rest of the file
            start = int(rng.split("=")[1].rstrip("-"))
            return httpx.Response(206, content=body[start:])
        return httpx.Response(200, content=body)

    return httpx.Client(transport=httpx.MockTransport(handler))


@pytest.fixture()
def installed(tmp_path, raw) -> Path:
    dest = tmp_path / "marker_db"
    st = mdb.ensure(dest, auto_download=True, client=_serve(raw))
    assert st["state"] == "ready", st
    return dest


# --- parsers ---------------------------------------------------------------------


def test_panglao_weights_canonical_and_sensitivity_per_species(tmp_path, raw):
    p = tmp_path / "pg.tsv.gz"
    p.write_bytes(raw["panglao"])
    rows = mdb._parse_panglao(p)
    w = {(sp, ct, g): wt for sp, ct, _cl, g, wt in rows}
    assert w[("human", "B cells", "CD79A")] == pytest.approx(2.9)  # canonical (2) + sensitivity 0.9
    assert w[("human", "B cells", "CD19")] == pytest.approx(0.7)
    assert w[("human", "NK cells", "GZMB")] == 0.0  # sensitivity "NA"
    # Mouse rows: the same marker in mouse case; human-only rows stay human
    assert ("mouse", "B cells", "Cd79a") in w and ("mouse", "NK cells", "Humanonly1") not in w
    assert ("human", "Monocytes", "LY6C2") not in w and ("mouse", "Monocytes", "Ly6c2") in w
    assert not any(ct == "BROKEN" or sp not in ("human", "mouse") for sp, ct, *_ in rows)


def test_cellmarker_counts_distinct_papers_for_normal_human_and_mouse_cells(tmp_path, raw):
    p = tmp_path / "cm.xlsx"
    p.write_bytes(raw["cellmarker2"])
    rows = {(sp, ct, g): (cl, wt) for sp, ct, cl, g, wt in mdb._parse_cellmarker(p)}
    # CD79A: papers 1001 and 1002 (1002 twice); 1004 is a cancer row
    assert rows[("human", "B cell", "CD79A")] == ("CL:0000236", 2.0)
    assert rows[("human", "B cell", "CD22")][1] == 1.0 and ("human", "B cell", "MS4A1") in rows  # list, brackets
    assert rows[("human", "Rare cell", "XYZ1")][0] == ""  # "NA" CL id
    assert ("mouse", "Monocyte", "Ly6c2") in rows
    assert not any(g == "csf1ra" for _sp, _ct, g in rows)  # zebrafish


def test_act_pools_tissues_and_reads_latin_1():
    rows = {(sp, ct, g): w for sp, ct, _cl, g, w in mdb._parse_act(FIX / "act")}
    # LILRA4: listed first in two tissue rows -> 2 + rank bonus 1/10; IRF7 4th of 5 in one row
    assert rows[("human", "Plasmacytoid dendritic cell", "LILRA4")] == pytest.approx(2.1)
    assert rows[("human", "Plasmacytoid dendritic cell", "IRF7")] == pytest.approx(1 + 0.4 / 10)
    assert ("human", "Monocyte (CD14é)", "CD14") in rows  # Latin-1 decoded
    assert ("mouse", "B cell", "Cd79a") in rows


def test_obo_keeps_live_cl_terms_with_exact_synonyms_and_parents():
    terms = mn.parse_obo((FIX / "cl_sample.obo").read_text())
    assert "CL:9999999" not in terms and "part_of" not in terms
    assert terms["CL:0000236"] == ("B cell", ["B lymphocyte"], ["CL:0000542"])  # BROAD synonym left out


# --- fetch and store -------------------------------------------------------------


def test_download_stores_tables_and_manifest_and_falls_back_to_the_mirror(tmp_path, raw):
    dest = tmp_path / "db"
    calls: list[str] = []
    st = mdb.ensure(dest, auto_download=True, client=_serve(raw, fail={URLS["panglao"][0]}, calls=calls))
    assert st["state"] == "ready" and st["ontology"]["downloaded"]
    pg = next(s for s in st["sources"] if s["name"] == "panglao")
    assert pg["served_from"] == URLS["panglao"][1]  # the official site timed out
    assert pg["sha256"] == hashlib.sha256(raw["panglao"]).hexdigest() and pg["fetched_at"]
    assert pg["types"]["human"] == 5 and pg["types"]["mouse"] == 4
    act = next(s for s in st["sources"] if s["name"] == "act")
    assert act["served_from"] == ACT_BASE and act["types"] == {"human": 3, "mouse": 1}
    assert act["sha256"] == hashlib.sha256((FIX / "act_sample.sha256").read_bytes()).hexdigest()
    m = json.loads((dest / "manifest.json").read_text())
    assert set(m["sources"]) == {"panglao", "cellmarker2", "act", "ontology"}
    assert set(m["failures"]) == {"cellmarker3"} and st["cellmarker"] == "CellMarker 2.0"  # 3.0 not served here
    assert st["ontology"]["version"] == "cl/releases/2026-06-08/cl-basic.owl"
    assert (dest / "raw" / mdb.SOURCES["cellmarker2"].filename).is_file()
    assert not list(dest.rglob("*.part*"))
    with gzip.open(dest / "panglao.tsv.gz", "rt") as f:
        genes = {line.split("\t")[3] for line in f}
    assert "MT-CO1" not in genes and "RPL3" not in genes  # mito and ribosomal dropped
    # Already there: nothing is fetched again
    calls.clear()
    mdb.ensure(dest, auto_download=True, client=_serve(raw, calls=calls))
    assert calls == []


def test_a_file_that_is_not_the_pinned_release_is_refused(tmp_path, raw):
    dest = tmp_path / "db"
    bad = dict(raw, cellmarker2=b"PK not the release")
    st = mdb.ensure(dest, auto_download=True, client=_serve(bad))
    cm = next(s for s in st["sources"] if s["name"] == "cellmarker2")
    assert not cm["downloaded"] and "sha256" in cm["error"] and cm["failed_at"]
    assert st["state"] == "partial"  # PanglaoDB alone still serves
    assert not (dest / "raw" / mdb.SOURCES["cellmarker2"].filename).exists()
    assert not list(dest.rglob("*.part*"))
    db = mdb.load(dest)
    assert db.sources == ["panglao", "act"]


def test_an_act_file_that_differs_from_its_pinned_hash_keeps_act_out(tmp_path, raw):
    dest = tmp_path / "db"
    bad = dict(raw, act={**raw["act"], ACT_FILES[1]: b"Species\tTissue\tCellType\tMarker\nHuman\tX\tEvil\tFOO\n"})
    st = mdb.ensure(dest, auto_download=True, client=_serve(bad))
    act = next(s for s in st["sources"] if s["name"] == "act")
    assert not act["downloaded"] and "1 of 3 ACT files failed" in act["error"] and "sha256" in act["error"]
    assert st["state"] == "partial" and mdb.load(dest).sources == ["panglao", "cellmarker2"]
    assert not (dest / "raw" / "ACT" / ACT_FILES[1]).exists()  # never kept
    assert (dest / "raw" / "ACT" / ACT_FILES[0]).is_file()  # verified files stay for a retry
    calls: list[str] = []
    st = mdb.ensure(dest, auto_download=True, retry=True, client=_serve(raw, calls=calls))
    assert calls == [CM3_URLS[0], ACT_URLS[1]] and st["state"] == "ready"  # retry tries CellMarker 3.0 again too


def test_a_failed_source_waits_for_the_cooldown_unless_retried(tmp_path, raw):
    dest = tmp_path / "db"
    dead = {URLS["cellmarker2"][0]}
    mdb.ensure(dest, auto_download=True, client=_serve(raw, fail=dead))
    calls: list[str] = []
    mdb.ensure(dest, auto_download=True, client=_serve(raw, fail=dead, calls=calls))
    assert calls == []  # failed a moment ago: not tried on every check
    st = mdb.ensure(dest, auto_download=True, retry=True, client=_serve(raw, calls=calls))
    assert calls == [CM3_URLS[0], URLS["cellmarker2"][0]] and st["state"] == "ready"
    assert next(s for s in st["sources"] if s["name"] == "cellmarker2")["error"] is None


def test_nothing_is_fetched_with_the_download_off_but_files_put_in_raw_are_used(tmp_path, raw):
    dest = tmp_path / "db"
    calls: list[str] = []
    st = mdb.ensure(dest, auto_download=False, client=_serve(raw, calls=calls))
    assert st["state"] == "disabled" and calls == []
    with pytest.raises(mdb.MarkerDBUnavailable, match="MARKER_DB_AUTO_DOWNLOAD"):
        mdb.get_db(dest, auto_download=False, client=_serve(raw, calls=calls))
    (dest / "raw").mkdir(parents=True, exist_ok=True)
    (dest / "raw" / mdb.SOURCES["panglao"].filename).write_bytes(raw["panglao"])
    st = mdb.ensure(dest, auto_download=False, client=_serve(raw, calls=calls))
    assert calls == [] and st["state"] == "partial"
    assert next(s for s in st["sources"] if s["name"] == "panglao")["served_from"] == "local file"


def test_unreachable_sites_say_so(tmp_path, raw):
    every = {*URLS["panglao"], *URLS["cellmarker2"], *CM3_URLS, "https://cl.test/cl-basic.obo", *ACT_URLS}
    with pytest.raises(mdb.MarkerDBUnavailable, match="could not be downloaded.*PanglaoDB.*504"):
        mdb.get_db(tmp_path / "db", auto_download=True, client=_serve(raw, fail=every))
    assert mdb.status(tmp_path / "db", auto_download=True)["state"] == "failed"


def test_two_requests_at_once_fetch_once(tmp_path, raw):
    dest = tmp_path / "db"
    calls: list[str] = []
    client = _serve(raw, calls=calls)
    threads = [threading.Thread(target=mdb.ensure, args=(dest,), kwargs={"auto_download": True, "client": client})
               for _ in range(3)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert sorted(calls) == sorted([URLS["panglao"][0], CM3_URLS[0], URLS["cellmarker2"][0],
                                    "https://cl.test/cl-basic.obo", *ACT_URLS])


# --- merged database and names ---------------------------------------------------


def test_sources_merge_by_name_and_greek_letters_keep_types_apart(installed):
    sdb = mdb.load(installed).species("human")
    b = sdb.types["B cells"]  # PanglaoDB "B cells" + CellMarker "B cell"
    assert b["sources"] == ["panglao", "cellmarker2", "act"] and b["cl"] == "CL:0000236"
    assert b["genes"][0] == "CD79A"  # top of both lists
    assert {"CD22", "BANK1"} <= set(b["genes"])
    assert "γδ T cell" in sdb.types and "TRDC" not in sdb.types["T cells"]["genes"]


def test_name_matching_modes(installed):
    m = mdb.load(installed).species("human").matcher
    assert m.match("T-cells") == (["T cells"], "exact")  # not the γδ T cells
    assert m.match("gamma delta T cells") == (["γδ T cell"], "exact")
    assert m.match("B lymphocytes") == (["B cells"], "cl")  # CL EXACT synonym
    assert m.match("Intestinal goblet cells") == (["Goblet cells"], "subset")  # head noun, not "Intestinal cell"
    assert m.match("Plasmablasts") == (["B cells"], "cl-ancestor")  # nearest CL parent with database rows
    assert m.match("Zorblax") == ([], "none")
    # Abbreviations and leading cluster numbers only in mode C
    assert m.match("12NK", "C")[0] == ["NK cells"] and m.match("12NK", "A")[0] == []
    mm = mdb.load(installed).species("mouse").matcher
    assert mm.match("14Mo")[0] == ["Monocytes"]  # PanglaoDB "Monocytes" + CellMarker "Monocyte"


def test_without_the_ontology_names_still_match(installed):
    (installed / "cl.json.gz").unlink()
    db = mdb.load(installed)
    assert db.ontology is None
    m = db.species("human").matcher
    assert m.match("B cells") == (["B cells"], "exact") and m.match("Plasmablasts") == ([], "none")


# --- group markers ---------------------------------------------------------------


def _rgg(groups: dict[str, list[tuple[str, float, float, float]]]):
    """A stored rank_genes_groups: per group (gene, logFC, padj, pts) in score order."""
    n = max(len(v) for v in groups.values())
    names = np.rec.fromarrays([[r[0] for r in v] for v in groups.values()], names=list(groups))
    lfc = np.rec.fromarrays([[r[1] for r in v] for v in groups.values()], names=list(groups))
    padj = np.rec.fromarrays([[r[2] for r in v] for v in groups.values()], names=list(groups))
    genes = sorted({r[0] for v in groups.values() for r in v})
    pts = pd.DataFrame(0.0, index=genes, columns=list(groups))
    for g, v in groups.items():
        for r in v:
            pts.loc[r[0], g] = r[3]
    assert n
    return {"names": names, "logfoldchanges": lfc, "pvals_adj": padj, "pts": pts,
            "params": {"groupby": "lab"}}


def test_markers_are_detected_up_genes_by_fold_change():
    rows = [("RPL3", 9.0, 1e-9, 0.9), ("MT-ND1", 8.0, 1e-9, 0.9), ("RPLP1", 7.5, 1e-9, 0.9),
            ("RPSA", 7.4, 1e-9, 0.9), ("LOWPTS", 7.0, 1e-9, 0.1),
            ("DOWN", -2.0, 1e-9, 0.9), ("MID", 2.0, 1e-9, 0.5), ("TOP", 5.0, 1e-3, 0.3)]
    rows += [(f"G{i:02d}", 1.0 - i / 100, 1e-6, 0.4) for i in range(30)]
    other = [(r[0], 0.1, 0.9, 0.1) for r in rows]
    a = ad.AnnData(np.zeros((4, 1), dtype=np.float32))
    a.obs["lab"] = pd.Categorical(["x", "x", "y", "y"])
    a.uns["rank_genes_groups__lab"] = _rgg({"x": rows, "y": other})
    out = mdb._markers_for(a, "lab")
    # logFC order, not score order; ribosomal/mito/low-detection/down left out; 25 kept
    assert out["x"][:3] == ["TOP", "MID", "G00"] and len(out["x"]) == 25
    assert not {"RPL3", "MT-ND1", "RPLP1", "RPSA", "LOWPTS", "DOWN"} & set(out["x"])
    assert out["y"] == []


def test_markers_of_a_relabelled_column_are_computed_again():
    a = _cells()
    a.uns["rank_genes_groups__lab"] = _rgg({"old": [("CD3D", 3.0, 1e-9, 0.9)], "other": [("CD3E", 3.0, 1e-9, 0.9)]})
    out = mdb._markers_for(a, "lab")
    assert set(out) == {"B", "T-cells", "Zorblax", "Rare cell"} and "CD79A" in out["B"]
    assert set(a.uns["rank_genes_groups__lab"]["names"].dtype.names) == {"old", "other"}  # untouched


def _cells(seed: int = 0, extra: dict[str, list[str]] | None = None) -> ad.AnnData:
    """Four (or more) labelled groups of 30 cells; each group expresses one database type's genes."""
    rng = np.random.default_rng(seed)
    programs = {
        "B": ["CD79A", "MS4A1", "CD19", "BANK1", "PAX5", "CD79B", "CD22"],
        "T-cells": ["NKG7", "GNLY", "KLRD1", "NCAM1", "PRF1", "GZMB"],  # labelled T, expresses NK genes
        "Zorblax": ["CD3D", "CD3E", "CD3G", "TRAC", "CD2", "LCK"],
        "Rare cell": ["XYZ1", "XYZ2", "TRDC"],
        **(extra or {}),
    }
    genes = sorted({g for v in programs.values() for g in v}) + [f"FILL{i}" for i in range(60)]
    n = 30 * len(programs)
    X = rng.poisson(0.2, size=(n, len(genes))).astype(np.float32)
    labels = []
    for k, (lab, prog) in enumerate(programs.items()):
        rows = slice(30 * k, 30 * (k + 1))
        for g in prog:
            X[rows, genes.index(g)] += rng.poisson(6, size=30)
        labels += [lab] * 30
    a = ad.AnnData(np.log1p(X))
    a.var_names = genes
    a.obs_names = [f"c{i}" for i in range(n)]
    a.obs["lab"] = pd.Categorical(labels, categories=list(programs))
    return a


def test_markers_are_computed_without_touching_the_dataset():
    a = _cells()
    out = mdb._markers_for(a, "lab")
    assert set(out["B"][:7]) == {"CD79A", "MS4A1", "CD19", "BANK1", "PAX5", "CD79B", "CD22"}
    assert not any(k.startswith("rank_genes_groups") for k in a.uns)


# --- verdicts and suggestions ----------------------------------------------------


def test_check_gives_each_label_its_verdict_and_suggestions(installed):
    a = _cells()
    res = mdb.check(a, "lab", "human", mdb.load(installed))
    by = {r["label"]: r for r in res["labels"]}
    assert [r["label"] for r in res["labels"]] == ["B", "T-cells", "Zorblax", "Rare cell"]  # category order
    # "B" is the database's B cells and its markers are B-cell markers
    assert by["B"]["verdict"] == "supported" and by["B"]["db_type"] == "B cells"
    assert {"CD79A", "MS4A1"} <= set(by["B"]["overlap"]) and by["B"]["p"] < 1e-3
    assert by["B"]["match"]["how"] == "exact" and by["B"]["sources"] == ["PanglaoDB", "CellMarker 2.0", "ACT"]
    # Labelled T cells, expressing NK genes: none of the T-cell database markers is among its markers
    assert by["T-cells"]["verdict"] == "no_support" and by["T-cells"]["overlap"] == []
    assert by["T-cells"]["verdict_text"] == "no database support"
    assert by["T-cells"]["suggestions"][0]["cell_type"] == "NK cells"
    assert "NKG7" in by["T-cells"]["suggestions"][0]["overlap"]
    # A name the databases do not have; its cells still get suggestions
    assert by["Zorblax"]["verdict"] == "not_in_db" and by["Zorblax"]["match"]["types"] == []
    assert by["Zorblax"]["suggestions"][0]["cell_type"] == "T cells"
    # "Rare cell" has 3 database genes: too few to judge, whatever the overlap
    assert by["Rare cell"]["verdict"] == "too_few" and by["Rare cell"]["n_db_genes"] == 3
    assert res["verdict_counts"] == {"supported": 1, "no_support": 1, "too_few": 1, "not_in_db": 1, "no_markers": 0}
    assert all(len(r["suggestions"]) <= 3 for r in res["labels"])
    assert [s["label"] for s in res["sources"]] == ["PanglaoDB", "CellMarker 2.0", "ACT"] and res["ontology"]
    assert res["verdict_sources"] == ["PanglaoDB", "CellMarker 2.0", "ACT"]
    assert res["suggestion_sources"] == ["PanglaoDB", "CellMarker 2.0"]


def test_act_types_judge_labels_but_are_not_suggested(installed):
    a = _cells(extra={"pDC": ["LILRA4", "CLEC4C", "IL3RA", "IRF7", "TCF4"]})
    res = mdb.check(a, "lab", "human", mdb.load(installed))
    pdc = next(r for r in res["labels"] if r["label"] == "pDC")
    assert pdc["db_type"] == "Plasmacytoid dendritic cell" and pdc["sources"] == ["ACT"]
    assert pdc["verdict"] == "supported" and {"LILRA4", "IRF7"} <= set(pdc["overlap"])
    assert all(s["cell_type"] != "Plasmacytoid dendritic cell" for r in res["labels"] for s in r["suggestions"])


def test_mouse_data_is_checked_against_mouse_rows(installed):
    a = _cells()
    a.var_names = [g[:1] + g[1:].lower() for g in a.var_names]
    res = mdb.check(a, "lab", None, mdb.load(installed))  # species from the gene names
    assert res["species"] == "mouse"
    b = next(r for r in res["labels"] if r["label"] == "B")
    assert b["verdict"] == "supported" and "Cd79a" in b["overlap"]  # the dataset's own spelling
    assert next(r for r in res["labels"] if r["label"] == "Rare cell")["verdict"] == "not_in_db"  # human-only type


def test_check_refuses_columns_it_cannot_use(installed):
    a = _cells()
    a.obs["one"] = "x"
    with pytest.raises(ValueError, match="2 to"):
        mdb.check(a, "one", "human", mdb.load(installed))


# --- CellMarker 3.0 ------------------------------------------------------------------


@pytest.fixture()
def installed_cm3(tmp_path, raw) -> Path:
    dest = tmp_path / "marker_db3"
    calls: list[str] = []
    st = mdb.ensure(dest, auto_download=True, client=_serve(raw, cm3=True, calls=calls))
    assert st["state"] == "ready" and st["cellmarker"] == "CellMarker 3.0", st
    assert URLS["cellmarker2"][0] not in calls  # the fallback is not fetched
    return dest


def test_cellmarker3_weights_all_and_curated_rows_from_the_zips(tmp_path, raw):
    root = tmp_path / "CellMarker3"
    root.mkdir()
    for f, b in raw["cellmarker3"].items():
        (root / f).write_bytes(b)
    rows = {(sp, ct, g): (cl, w, wc) for sp, ct, cl, g, w, wc in mdb._parse_cellmarker3(root)}
    # CD79A: papers 1001 (as "1001.0" and "1001") and 1002 -> 2, plus 2 of 9 supports / 10
    assert rows[("human", "B cell", "CD79A")] == ("CL:0000236", pytest.approx(2 + 0.2 / 9), pytest.approx(2 + 0.2 / 9))
    # CD22 comes from a computational row only (no PMID: its dataset id counts as the paper)
    assert rows[("human", "B cell", "CD22")][1:] == (pytest.approx(1 + 0.1 / 9), 0.0)
    assert rows[("human", "Methodcyte", "MTH1")][2] == 0.0
    assert ("human", "B cell", "PAX5") not in rows  # a disease row
    assert not any(sp not in ("human", "mouse") for sp, _ct, _g in rows)  # zebrafish
    assert not any(g.startswith("RPL") for _sp, _ct, g in rows)


def test_cellmarker3_replaces_2_and_the_manifest_says_so(installed_cm3):
    st = mdb.status(installed_cm3, auto_download=True)
    cm3 = next(s for s in st["sources"] if s["name"] == "cellmarker3")
    cm2 = next(s for s in st["sources"] if s["name"] == "cellmarker2")
    assert cm3["downloaded"] and cm3["release"] == "2026-07-22 release, unpublished"
    assert "36300619" in cm3["paper"]  # cite 2.0 until 3.0 is published
    assert not cm2["downloaded"] and not cm2["needed"] and cm2["error"] is None
    m = json.loads((installed_cm3 / "manifest.json").read_text())
    assert m["sources"]["cellmarker3"]["curated_rows"] > 0 and "cellmarker2" not in m["sources"]
    assert not list(installed_cm3.rglob("*.part*"))
    assert sorted(p.name for p in (installed_cm3 / "raw" / "CellMarker3").iterdir()) == sorted(CM3_ZIPS)  # no TSV written out


def test_a_changed_cellmarker3_file_fails_closed_and_2_is_used(tmp_path, raw):
    dest = tmp_path / "db"
    zips = dict(raw["cellmarker3"])
    zips["mouse_cell_marker.zip"] = zips["mouse_cell_marker.zip"][:-10] + b"0123456789"
    st = mdb.ensure(dest, auto_download=True, client=_serve(dict(raw, cellmarker3=zips), cm3=True))
    cm3 = next(s for s in st["sources"] if s["name"] == "cellmarker3")
    assert not cm3["downloaded"] and "sha256" in cm3["error"]
    assert st["cellmarker"] == "CellMarker 2.0" and st["state"] == "ready"
    assert not (dest / "raw" / "CellMarker3" / "mouse_cell_marker.zip").exists()
    db = mdb.load(dest)
    assert db.cellmarker == "cellmarker2" and db.view("suggest", "human") == ["panglao", "cellmarker2"]


def test_a_cut_cellmarker3_download_resumes_with_a_range_request(tmp_path, raw):
    dest = tmp_path / "db"
    root = dest / "raw" / "CellMarker3"
    root.mkdir(parents=True)
    human = raw["cellmarker3"]["human_cell_marker.zip"]
    (root / "human_cell_marker.part.zip").write_bytes(human[:100])
    ranges, progress = {}, []

    def seen(req):
        ranges[str(req.url)] = req.headers.get("Range")
        if str(req.url) == CM3_URLS[1]:
            progress.append(mdb.status(dest, auto_download=True)["sources"][1]["progress"])

    st = mdb.ensure(dest, auto_download=True, client=_serve(raw, cm3=True, on_request=seen))
    assert ranges[CM3_URLS[0]] == "bytes=100-" and ranges[CM3_URLS[1]] is None
    assert (root / "human_cell_marker.zip").read_bytes() == human and st["cellmarker"] == "CellMarker 3.0"
    total = sum(len(b) for b in raw["cellmarker3"].values())
    assert progress == [{"done": len(human), "total": total}]  # the human zip done when the mouse one starts


def test_cellmarker3_jobs_use_their_measured_parts(installed_cm3):
    db = mdb.load(installed_cm3)
    assert db.view("verdict", "human") == ["panglao", "cellmarker3", "act"]
    assert db.view("verdict", "mouse") == ["cellmarker3"]
    assert db.view("suggest", "human") == db.view("annotate", "mouse") == ["panglao", "cellmarker3_curated"]
    a = _cells(extra={"Methodcyte": ["MTH1", "MTH2", "MTH3", "MTH4", "MTH5"]})
    res = mdb.check(a, "lab", "human", db)
    assert res["cellmarker"] == "CellMarker 3.0"
    assert res["verdict_sources"] == ["PanglaoDB", "CellMarker 3.0", "ACT"]
    assert res["suggestion_sources"] == ["PanglaoDB", "CellMarker 3.0 (curated)"]
    by = {r["label"]: r for r in res["labels"]}
    # A type known only from computational rows judges its label ...
    assert by["Methodcyte"]["verdict"] == "supported" and by["Methodcyte"]["sources"] == ["CellMarker 3.0"]
    # ... but is never suggested
    assert all(s["cell_type"] != "Methodcyte" for r in res["labels"] for s in r["suggestions"])
    assert by["B"]["verdict"] == "supported" and "CD22" in by["B"]["overlap"]  # a computational marker counts
    assert by["B"]["suggestions"][0]["cell_type"] == "B cells"


def test_mouse_verdicts_use_cellmarker3_mouse_rows_alone(installed_cm3):
    a = _cells()
    a.var_names = [g[:1] + g[1:].lower() for g in a.var_names]
    res = mdb.check(a, "lab", "mouse", mdb.load(installed_cm3))
    assert res["verdict_sources"] == ["CellMarker 3.0"]
    b = next(r for r in res["labels"] if r["label"] == "B")
    assert b["verdict"] == "supported" and b["sources"] == ["CellMarker 3.0"] and "Bank1" in b["overlap"]
    # T cells is a PanglaoDB type: not in the mouse verdict
    assert next(r for r in res["labels"] if r["label"] == "T-cells")["verdict"] == "not_in_db"


# --- endpoints -------------------------------------------------------------------


def _register(dm, a: ad.AnnData, did: str) -> str:
    d = Path(dm.data_dir) / "uploads" / did
    d.mkdir(parents=True)
    (d / "metadata.json").write_text(json.dumps({"id": did, "name": did, "status": "ready"}))
    a.write_h5ad(d / "data.h5ad")
    return did


async def test_status_and_check_endpoints(client, dataset_manager, test_settings, tmp_data_dir, raw):
    test_settings.MARKER_DB_AUTO_DOWNLOAD = False
    did = _register(dataset_manager, _cells(), "mdb")
    r = await client.get(f"{API}/datasets/{did}/annotation/marker-db/status")
    assert r.status_code == 200 and r.json()["state"] == "disabled" and r.json()["dataset_species"] == "human"
    r = await client.post(f"{API}/datasets/{did}/annotation/marker-db/check", json={"column": "lab"})
    assert r.status_code == 503 and "MARKER_DB_AUTO_DOWNLOAD" in r.json()["detail"]
    # The files put in the volume by hand
    rawdir = tmp_data_dir / "marker_db" / "raw"
    rawdir.mkdir(parents=True)
    for name, spec in mdb.SOURCES.items():
        if name not in ("act", "cellmarker3"):
            (rawdir / spec.filename).write_bytes(raw[name])
    (rawdir / mdb.ONTOLOGY_FILE).write_bytes(raw["ontology"])
    for f, data in raw["act"].items():
        (rawdir / "ACT" / f).parent.mkdir(parents=True, exist_ok=True)
        (rawdir / "ACT" / f).write_bytes(data)
    r = await client.post(f"{API}/datasets/{did}/annotation/marker-db/check", json={"column": "lab"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["verdict_counts"]["supported"] == 1 and body["species"] == "human"
    st = (await client.get(f"{API}/datasets/{did}/annotation/marker-db/status")).json()
    assert st["state"] == "ready" and st["cellmarker"] == "CellMarker 2.0"  # 3.0 not in the volume
    assert {s["name"] for s in st["sources"] if s["downloaded"]} == {"panglao", "cellmarker2", "act"}
    r = await client.post(f"{API}/datasets/{did}/annotation/marker-db/check", json={"column": "nope"})
    assert r.status_code == 404
