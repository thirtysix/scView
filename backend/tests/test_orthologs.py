"""Human -> mouse orthologs (core/orthologs.py) and the mouse CollecTRI / PROGENy.

The HCOP fixture (tests/fixtures/orthologs/hcop_sample.tsv) is invented, in the
real file's 15-column layout: G<i> -> Gene<i> for the activity fixtures' genes,
plus the cases the rules are about (ONE2TWO, MANY over the cap, WEAK under the
evidence cut, DUP listed twice, MERGEA / MERGEB onto one mouse gene, "-" rows).
No test touches the internet. Agreement with decoupler 2.3.0 on the real
networks and paul15 is in the batch 24 report.
"""

from __future__ import annotations

import dataclasses
import gzip
import hashlib
import json
from pathlib import Path

import httpx
import numpy as np
import pandas as pd
import pytest

from scview.core import activity_nets, orthologs

FIXTURES = Path(__file__).parent / "fixtures"
RAW = gzip.compress((FIXTURES / "orthologs" / "hcop_sample.tsv").read_bytes(), mtime=0)
NET_RAW = {
    "collectri": (FIXTURES / "activity" / "collectri_sample.csv").read_bytes(),
    "progeny": (FIXTURES / "activity" / "progeny_sample.tsv").read_bytes(),
}


@pytest.fixture(autouse=True)
def small_tables(monkeypatch):
    monkeypatch.setattr(orthologs, "MIN_PAIRS", 100)
    for name, s in list(activity_nets.NETWORKS.items()):
        monkeypatch.setitem(activity_nets.NETWORKS, name, dataclasses.replace(s, min_edges=10, min_sources=3))
    orthologs._loaded.clear()
    orthologs._maps.clear()


def _client(handler) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(handler))


def _serve(urls: list[str] | None = None) -> httpx.Client:
    """HCOP and both networks from the fixtures, by URL."""
    def handler(req: httpx.Request) -> httpx.Response:
        url = str(req.url)
        if urls is not None:
            urls.append(url)
        if url == orthologs.URL:
            return httpx.Response(200, content=RAW, headers={"last-modified": "Thu, 01 Oct 2026 12:45:06 GMT"})
        for name, s in activity_nets.NETWORKS.items():
            if url == s.url:
                return httpx.Response(200, content=NET_RAW[name])
        return httpx.Response(404)
    return _client(handler)


@pytest.fixture()
def table() -> pd.DataFrame:
    return orthologs.parse(RAW)


# --- the table --------------------------------------------------------------------


def test_parse_keeps_symbol_pairs_and_counts_the_supporting_resources(table):
    assert list(table.columns) == ["human", "mouse", "evidence"]
    assert "-" not in set(table["human"]) and "-" not in set(table["mouse"])  # HCOP's empty field
    ev = {(h, m): e for h, m, e in table.itertuples(index=False)}
    assert ev[("STAT1", "Stat1")] == 5 and ev[("WEAK", "Weak")] == 2 and ev[("LOWWEAK", "Lowweak")] == 1
    assert (table["human"] == "DUP").sum() == 2  # rows are kept as listed


@pytest.mark.parametrize("bad, why", [
    (b"not gzip", "not a gzipped"),
    (gzip.compress(b"human_symbol\tmouse_symbol\nA\tB\n"), "lacks columns"),
    (gzip.compress(gzip.decompress(RAW).replace(b"TP53\t", b"TP99\t")), "no TP53 -> Trp53"),
])
def test_a_wrong_file_is_refused(bad, why):
    with pytest.raises(ValueError, match=why):
        orthologs.parse(bad)


def test_too_few_pairs_is_refused(monkeypatch):
    monkeypatch.setattr(orthologs, "MIN_PAIRS", 10_000)
    with pytest.raises(ValueError, match="only .* pairs"):
        orthologs.parse(RAW)


# --- the mapping rules (decoupler 2.x translate defaults) ------------------------------


def test_one_to_one_one_to_many_cap_and_evidence_cut(table):
    m = orthologs.human_to_mouse(["STAT1", "ONE2TWO", "MANY", "WEAK", "DUP", "MERGEA", "MERGEB",
                                  "LOWWEAK", "NOMOUSE", "ABSENT"], table=table)
    assert m["STAT1"] == ["Stat1"]
    assert m["ONE2TWO"] == ["Otwo1", "Otwo2"]  # every ortholog within the cap
    assert m["MANY"] == []  # 6 orthologs > 5: dropped, not truncated
    assert m["WEAK"] == [] and m["LOWWEAK"] == []  # fewer than 3 resources
    assert m["DUP"] == ["Dup"]  # listed twice in HCOP, returned once
    assert m["MERGEA"] == m["MERGEB"] == ["Merged"]  # many-to-one is the caller's call
    assert m["NOMOUSE"] == [] and m["ABSENT"] == []

    looser = orthologs.human_to_mouse(["MANY", "WEAK"], table=table, min_evidence=2, one_to_many=6)
    assert len(looser["MANY"]) == 6 and looser["WEAK"] == ["Weak"]
    # decoupler counts HCOP rows against the cap, so DUP's two rows exceed a cap of 1.
    assert orthologs.human_to_mouse(["DUP", "STAT1"], table=table, one_to_many=1) == {"DUP": [], "STAT1": ["Stat1"]}


def _decoupler_translate(net: pd.DataFrame, columns, table: pd.DataFrame, min_evidence=3, one_to_many=5):
    """decoupler 2.3.0's op.translate, restated from its documented behaviour: a dict
    of HCOP rows per human symbol, a gene over the cap mapped to nothing, complexes
    by subunit, a left merge and a dropna."""
    from itertools import product

    kept = table[table["evidence"] >= min_evidence]
    d = kept.groupby("human")["mouse"].apply(list).to_dict()
    for col in columns:
        rows = []
        for v in net[col].drop_duplicates():
            subs = [d.get(p, []) for p in v.split("_")]
            subs = [s if len(s) <= one_to_many else [] for s in subs]
            if all(subs):
                rows += [(v, "_".join(c)) for c in product(*subs)]
        tr = pd.DataFrame(rows, columns=["o", "t"]).set_index("o")
        net = net.merge(tr, left_on=col, right_index=True, how="left")
        net[col] = net.pop("t")
        net = net.dropna(subset=[col])
    return net.reset_index(drop=True)


def test_translate_net_matches_decouplers_rules(table):
    net = pd.DataFrame({
        "source": ["STAT1", "STAT1", "STAT1_IRF1", "STAT1_NFKB", "MYC", "MYC", "ONE2TWO", "MANY", "WEAK"],
        "target": ["G0", "G290", "G1", "G2", "MERGEA", "MERGEB", "G3", "G4", "G5"],
        "weight": [1.0, -1.0, 1.0, 1.0, 1.0, -1.0, 1.0, 1.0, 1.0],
    })
    out = orthologs.translate_net(net, ["source", "target"], table=table)
    assert list(zip(out["source"], out["target"], out["weight"])) == [
        ("Stat1", "Gene0", 1.0),
        ("Stat1", "Gene290", -1.0), ("Stat1", "Gene290b", -1.0),  # one-to-many: both
        ("Stat1_Irf1", "Gene1", 1.0),  # a complex maps per subunit
        ("Myc", "Merged", 1.0), ("Myc", "Merged", -1.0),  # many-to-one: both rows, caller dedups
        ("Otwo1", "Gene3", 1.0), ("Otwo2", "Gene3", 1.0),
    ]  # STAT1_NFKB (NFKB unmapped), MANY (over the cap), WEAK (evidence) are gone
    ref = _decoupler_translate(net, ["source", "target"], table)
    pd.testing.assert_frame_equal(out.drop_duplicates(), ref.drop_duplicates().reset_index(drop=True))
    with pytest.raises(ValueError, match="None of"):
        orthologs.translate_net(net, ["genesymbol"], table=table)


def test_map_gene_sets_reports_coverage(table):
    sets, cov = orthologs.map_gene_sets(
        {"IFN": ["STAT1", "IRF1", "ONE2TWO", "MANY", "STAT1"], "NONE": ["WEAK"]}, table=table)
    assert sets == {"IFN": ["Stat1", "Irf1", "Otwo1", "Otwo2"], "NONE": []}
    assert cov == {"IFN": {"n_genes": 4, "n_mapped": 3, "n_mouse": 4},
                   "NONE": {"n_genes": 1, "n_mapped": 0, "n_mouse": 0}}


# --- fetching -------------------------------------------------------------------------


def test_download_records_whether_it_is_the_pinned_release(tmp_path, monkeypatch):
    urls: list[str] = []
    orthologs.download(tmp_path, client=_serve(urls))
    assert urls == [orthologs.URL]
    meta = orthologs.read_meta(tmp_path)
    assert meta["pinned"] is False and "newer HCOP release" in meta["note"]  # not the real file
    assert meta["licence"].startswith("CC0") and meta["last_modified"].startswith("Thu, 01 Oct 2026")
    assert meta["n_pairs_min_evidence"] == int((orthologs.parse(RAW)["evidence"] >= 3).sum())
    assert not list(tmp_path.glob("*.part"))

    monkeypatch.setattr(orthologs, "SHA256", hashlib.sha256(RAW).hexdigest())
    orthologs.download(tmp_path, client=_serve())
    meta = orthologs.read_meta(tmp_path)
    assert meta["pinned"] is True and "note" not in meta
    t = orthologs.load(tmp_path, client=_client(lambda r: pytest.fail("no fetch when on disk")))
    assert len(t) == meta["n_rows"]


def test_offline_switched_off_and_altered_tables_say_what_to_do(tmp_path):
    def down(req):
        raise httpx.ConnectError("no route to host")

    with pytest.raises(orthologs.OrthologsUnavailable, match="check the internet connection"):
        orthologs.load(tmp_path, auto_download=True, client=_client(down))
    assert orthologs.read_meta(tmp_path) is None and not (tmp_path / orthologs.TABLE).exists()
    with pytest.raises(orthologs.OrthologsUnavailable, match="ORTHOLOGS_AUTO_DOWNLOAD is off"):
        orthologs.load(tmp_path, auto_download=False)
    bad = gzip.compress(b"<html>captive portal</html>")
    with pytest.raises(orthologs.OrthologsUnavailable, match="lacks columns"):
        orthologs.load(tmp_path, auto_download=True, client=_client(lambda r: httpx.Response(200, content=bad)))
    assert orthologs.read_meta(tmp_path) is None

    orthologs.load(tmp_path, auto_download=True, client=_serve())
    orthologs._loaded.clear()
    t = pd.read_parquet(tmp_path / orthologs.TABLE)
    t.iloc[:5].to_parquet(tmp_path / orthologs.TABLE, index=False)
    with pytest.raises(orthologs.OrthologsUnavailable, match="does not match the sha256"):
        orthologs.load(tmp_path, auto_download=False)


# --- mouse CollecTRI and PROGENy --------------------------------------------------------


def test_mouse_networks_are_mapped_as_decoupler_maps_them(tmp_path, table):
    for name, cols in (("collectri", ["source", "target"]), ("progeny", ["target"])):
        human = activity_nets.build_from_bytes(name, NET_RAW[name])
        mouse = activity_nets.translate_network(name, human, table)
        ref = _decoupler_translate(human, cols, table).drop_duplicates(["source", "target"])
        pd.testing.assert_frame_equal(mouse, ref.reset_index(drop=True))
        assert not mouse.duplicated(["source", "target"]).any()
    ct = activity_nets.translate_network("collectri", activity_nets.build_from_bytes("collectri", NET_RAW["collectri"]),
                                         table)
    assert {"Stat1", "Irf1", "Trp53", "Myc"} <= set(ct["source"]) and "NFKB" not in set(ct["source"])
    pg = activity_nets.translate_network("progeny", activity_nets.build_from_bytes("progeny", NET_RAW["progeny"]),
                                         table)
    assert "JAK-STAT" in set(pg["source"]) and set(pg["target"]) <= {f"Gene{i}" for i in range(300)} | {"Gene290b"}


def test_the_mouse_table_is_cached_and_rebuilt_when_an_input_changes(tmp_path):
    urls: list[str] = []
    net = activity_nets.load_mouse_network("collectri", tmp_path, client=_serve(urls))
    assert sorted(urls) == sorted([orthologs.URL, activity_nets.NETWORKS["collectri"].url])
    meta = activity_nets.read_mouse_meta(tmp_path, "collectri")
    assert meta["n_edges"] == len(net) and meta["n_human_edges"] > meta["n_edges"]
    assert meta["n_targets_mapped"] <= meta["n_human_targets"] and meta["min_evidence"] == 3
    built = meta["built_at"]

    again = activity_nets.load_mouse_network("collectri", tmp_path, client=_client(lambda r: pytest.fail("fetch")))
    pd.testing.assert_frame_equal(again, net)
    assert activity_nets.read_mouse_meta(tmp_path, "collectri")["built_at"] == built

    hmeta = json.loads((tmp_path / orthologs.META).read_text())
    hmeta["table_sha256"] = None  # a refetched HCOP table: the mouse network is rebuilt
    (tmp_path / orthologs.META).write_text(json.dumps(hmeta))
    activity_nets.load_mouse_network("collectri", tmp_path, client=_serve())
    assert activity_nets.read_mouse_meta(tmp_path, "collectri")["hcop_sha256"] is None

    # The human path never touches HCOP: the same table as before this item.
    urls = []
    human = activity_nets.load_for("progeny", tmp_path / "h", "human", client=_serve(urls))
    assert urls == [activity_nets.NETWORKS["progeny"].url] and not (tmp_path / "h" / orthologs.TABLE).exists()
    pd.testing.assert_frame_equal(human, activity_nets.build_from_bytes("progeny", NET_RAW["progeny"]),
                                  check_dtype=False, check_names=False)
    assert np.isfinite(human["weight"]).all()


def test_no_hcop_means_a_503_style_network_error(tmp_path):
    def handler(req):
        if str(req.url) == orthologs.URL:
            raise httpx.ConnectError("offline")
        return httpx.Response(200, content=NET_RAW["progeny"])

    with pytest.raises(activity_nets.NetworkUnavailable, match="HCOP ortholog table"):
        activity_nets.load_mouse_network("progeny", tmp_path, client=_client(handler))
