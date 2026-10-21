"""Overlap of several kept DE results (UpSet): core/de_overlap.py, the labels
de_results keeps, and the /de/results + /de/overlap endpoints."""

from __future__ import annotations

import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

from scview.api.v1.de import DECompareRequest, DERequest, differential_expression, differential_expression_compare
from scview.core import de_overlap, de_results
from scview.core.de_results import StoredResult

API = "/api/v1"


@pytest.fixture(autouse=True)
def _fresh():
    de_results.clear()
    yield
    de_results.clear()


def _res(rows: dict[str, tuple[float, float]], *, method="wilcoxon", label="", raw_only=False,
         stats: dict[str, float] | None = None) -> StoredResult:
    """A stored result from {gene: (log2FC, adjusted p)}."""
    genes = list(rows)
    lfc = np.array([rows[g][0] for g in genes])
    p = np.array([rows[g][1] for g in genes])
    st = np.array([(stats or {}).get(g, np.nan) for g in genes])
    if method == "logreg":
        return StoredResult("d", genes, lfc, np.full(len(genes), np.nan), st, method="logreg",
                            pvals_adj=np.full(len(genes), np.nan), label=label)
    return StoredResult("d", genes, lfc, p, st, method=method, label=label,
                        pvals_adj=None if raw_only else p)


SIG, NS = 0.001, 0.5


def _three() -> list[tuple[str, StoredResult]]:
    """ISG15/IFI6 up in all three; MONO1 up in A only; T1 up in B only; FLIP up in
    A, down in B (discordant); DOWN1 down in A and C; WEAK below the fold-change cut."""
    a = _res({"ISG15": (3, SIG), "IFI6": (2, SIG), "MONO1": (1.5, SIG), "T1": (0.1, NS),
              "FLIP": (1, SIG), "DOWN1": (-1, SIG), "WEAK": (0.1, SIG)}, label="A")
    b = _res({"ISG15": (2.5, SIG), "IFI6": (1.8, SIG), "MONO1": (0.2, NS), "T1": (1.2, SIG),
              "FLIP": (-1, SIG), "DOWN1": (-0.1, NS), "WEAK": (0.1, SIG)}, label="B")
    c = _res({"ISG15": (2, SIG), "IFI6": (1.1, SIG), "MONO1": (0, NS), "T1": (0, NS),
              "FLIP": (0, NS), "DOWN1": (-2, SIG), "WEAK": (0, NS)}, label="C")
    return [("a", a), ("b", b), ("c", c)]


def _inter(out, sets) -> dict | None:
    return next((x for x in out["intersections"] if x["sets"] == sets), None)


def test_exclusive_intersections_add_up_to_the_union():
    out = de_overlap.overlap(_three(), direction="both")
    total = sum(x["n"] for x in out["intersections"]) + out["hidden_genes"] + out["discordant"]["n"]
    assert total == out["n_union"] == 6  # ISG15 IFI6 MONO1 T1 FLIP DOWN1; WEAK is under the cut
    every = [g["gene"] for x in out["intersections"] for g in x["genes"]]
    assert len(every) == len(set(every))  # each gene in exactly one combination
    assert {g["gene"] for g in _inter(out, [0, 1, 2])["genes"]} == {"ISG15", "IFI6"}
    assert [g["gene"] for g in _inter(out, [0])["genes"]] == ["MONO1"]
    assert [g["gene"] for g in _inter(out, [1])["genes"]] == ["T1"]
    assert _inter(out, [0, 2])["n_down"] == 1 and _inter(out, [0, 2])["n_up"] == 0
    # Largest first; the set sizes count both directions.
    assert out["intersections"][0]["sets"] == [0, 1, 2]
    assert [s["n"] for s in out["sets"]] == [5, 4, 3]
    # ISG15 before IFI6 (larger mean |log2FC|); every result's fold change listed.
    assert _inter(out, [0, 1, 2])["genes"][0] == {"gene": "ISG15", "lfc": [3.0, 2.5, 2.0]}


def test_both_keeps_the_sign_a_flip_is_discordant_not_shared():
    out = de_overlap.overlap(_three(), direction="both")
    assert _inter(out, [0, 1]) is None  # FLIP is not a shared hit of A and B
    assert out["discordant"]["n"] == 1
    assert out["discordant"]["combinations"] == [{"sets": [0, 1], "n": 1}]
    assert out["discordant"]["genes"] == [{"gene": "FLIP", "lfc": [1.0, -1.0, 0.0]}]
    ab = next(p for p in out["pairs"] if (p["a"], p["b"]) == (0, 1))
    # A = {ISG15 IFI6 MONO1 FLIP DOWN1}, B = {ISG15 IFI6 T1 FLIP}: 2 shared, 1 discordant of 6.
    assert (ab["shared"], ab["discordant"], ab["union"]) == (2, 1, 6)
    assert ab["jaccard"] == round(2 / 6, 4)


def test_up_and_down_directions():
    up = de_overlap.overlap(_three(), direction="up")
    assert [s["n"] for s in up["sets"]] == [4, 3, 2]  # FLIP counts as up in A only
    assert {g["gene"] for g in _inter(up, [0])["genes"]} == {"MONO1", "FLIP"}
    assert up["discordant"]["n"] == 0
    down = de_overlap.overlap(_three(), direction="down")
    assert [g["gene"] for g in _inter(down, [0, 2])["genes"]] == ["DOWN1"]
    assert [g["gene"] for g in _inter(down, [1])["genes"]] == ["FLIP"]


def test_thresholds_move_genes_in_and_out():
    loose = de_overlap.overlap(_three(), direction="up", lfc_min=0.05)
    assert "WEAK" in {g["gene"] for g in _inter(loose, [0, 1])["genes"]}
    strict = de_overlap.overlap(_three(), direction="up", padj_max=0.0001)
    assert strict["n_union"] == 0 and strict["intersections"] == []


def test_logistic_regression_takes_the_top_n_coefficients(monkeypatch):
    monkeypatch.setattr(de_overlap, "LOGREG_TOP_N", 2)
    lr = _res({"ISG15": (3, 0), "IFI6": (2, 0), "MONO1": (1, 0), "OPP": (1, 0), "DOWN1": (-1, 0)},
              method="logreg", label="LR",
              stats={"ISG15": 0.9, "IFI6": 0.5, "MONO1": 0.7, "OPP": -0.8, "DOWN1": -0.4})
    out = de_overlap.overlap([("w", _three()[0][1]), ("l", lr)], direction="both")
    s = out["sets"][1]
    assert s["rule"] == "top_n"
    # Up: ISG15 0.9 and MONO1 0.7 (IFI6 0.5 is third); OPP's coefficient disagrees
    # with its fold change; down: DOWN1.
    assert (s["n_up"], s["n_down"]) == (2, 1)
    assert {g["gene"] for g in _inter(out, [0, 1])["genes"]} == {"ISG15", "MONO1", "DOWN1"}
    assert any("no p-values" in n and "top 2" in n for n in out["notes"])


def test_raw_p_only_is_adjusted_here_and_said():
    rows = {f"G{i}": (1.0, 0.004 * (i + 1)) for i in range(20)}  # raw p 0.004 .. 0.08
    out = de_overlap.overlap([("r", _res(rows, raw_only=True, label="raw")),
                              ("s", _res(rows, label="adj"))], direction="up")
    raw, adj = out["sets"]
    assert raw["rule"] == "padj_bh_here" and adj["rule"] == "padj"
    # BH over 20 genes: p_(k) * 20 / k = 0.08 for all, past 0.05; the stored
    # (already adjusted) copy keeps the 12 genes at 0.004 .. 0.048.
    assert raw["n"] == 0 and adj["n"] == 12
    assert any("adjusted here" in n for n in out["notes"])


def test_restrict_to_common_tested():
    a = _res({"ISG15": (2, SIG), "RARE": (2, SIG), "B1": (1, SIG)}, label="A")
    b = _res({"ISG15": (2, SIG), "B1": (0.1, NS)}, label="B")  # RARE not tested in B
    free = de_overlap.overlap([("a", a), ("b", b)], direction="up")
    assert {g["gene"] for g in _inter(free, [0])["genes"]} == {"RARE", "B1"}
    assert free["n_common_tested"] == 2
    assert any("may not have been tested" in n for n in free["notes"])
    common = de_overlap.overlap([("a", a), ("b", b)], direction="up", restrict_to_common_tested=True)
    assert [g["gene"] for g in _inter(common, [0])["genes"]] == ["B1"]
    assert common["sets"][0]["n"] == 2 and common["sets"][0]["n_tested"] == 3
    rare = next(g for g in _inter(free, [0])["genes"] if g["gene"] == "RARE")
    assert rare["lfc"] == [2.0, None]  # not tested in B: no fold change, not 0


def test_an_extra_gene_list_is_a_set_without_direction():
    extra = [{"name": "ISGs", "genes": ["isg15", "IFI6", "NOTAGENE", "ifi6"]}]
    out = de_overlap.overlap(_three()[:2], direction="both", extra_sets=extra)
    e = out["sets"][2]
    assert e["kind"] == "extra" and e["n"] == 3 and e["n_not_tested"] == 1
    # Matched ignoring case to the results' gene names.
    assert {g["gene"] for g in _inter(out, [0, 1, 2])["genes"]} == {"ISG15", "IFI6"}
    assert [g["gene"] for g in _inter(out, [2])["genes"]] == ["NOTAGENE"]
    assert _inter(out, [2])["n_up"] == 0 and _inter(out, [2])["n_down"] == 0
    assert any("tested in no result" in n for n in out["notes"])
    common = de_overlap.overlap(_three()[:2], extra_sets=extra, restrict_to_common_tested=True)
    assert common["sets"][2]["n"] == 2  # NOTAGENE dropped


def test_listed_genes_are_capped_with_the_total(monkeypatch):
    monkeypatch.setattr(de_overlap, "MAX_LISTED", 3)
    rows = {f"G{i}": (1.0 + i, SIG) for i in range(10)}
    out = de_overlap.overlap([("a", _res(rows, label="A")), ("b", _res(rows, label="B"))])
    x = out["intersections"][0]
    assert x["n"] == 10 and [g["gene"] for g in x["genes"]] == ["G9", "G8", "G7"]


def test_duplicate_labels_are_told_apart():
    out = de_overlap.overlap([("a", _res({"X": (1, SIG)}, label="Same")),
                              ("b", _res({"X": (1, SIG)}, label="Same"))])
    assert [s["name"] for s in out["sets"]] == ["Same", "Same (2)"]


# --- labels and the list kept by de_results ---------------------------------


class G:
    def __init__(self, gene, lfc=1.0, p=0.01, padj=0.02, stat=1.0):
        self.gene, self.logfoldchange, self.pval, self.pval_adj, self.stat = gene, lfc, p, padj, stat


def test_describe_names_each_de_mode():
    d = de_results.describe
    assert d({"mode": "selection", "label": "selection", "n_a": 412, "method": "t-test"}) == \
        "selection (412 cells) vs rest (t-test)"
    assert d({"mode": "groups", "group_a": "STIM", "group_b": "CTRL", "within_column": "cell",
              "within_value": "CD14 Mono", "method": "wilcoxon"}) == "CD14 Mono: STIM vs CTRL (Wilcoxon)"
    assert d({"mode": "groups", "group_a": "STIM", "group_b": "CTRL", "method": "pseudobulk"}) == \
        "All cells: STIM vs CTRL (pseudobulk DESeq2)"
    assert d({"mode": "selection", "label": "B", "n_a": 1234, "method": "logreg"}) == \
        "B (1,234 cells) vs rest (logistic regression)"
    assert d({"method": "wilcoxon"}) == "DE result (Wilcoxon)"


def test_remember_keeps_label_adjusted_p_and_comparison():
    rid = de_results.remember("d", [G("A", padj=0.03)], method="t-test",
                              comparison={"mode": "selection", "label": "sel", "n_a": 5})
    r = de_results.recall("d", rid)
    assert r.label == "sel (5 cells) vs rest (t-test)"
    assert r.pvals_adj.tolist() == [0.03] and r.comparison["method"] == "t-test"
    assert r.created_at > 0


def test_list_results_is_per_dataset_newest_first():
    first = de_results.remember("d", [G("A")])
    other = de_results.remember("e", [G("A")])
    second = de_results.remember("d", [G("A")])
    assert [rid for rid, _ in de_results.list_results("d")] == [second, first]
    assert [rid for rid, _ in de_results.list_results("e")] == [other]
    de_results.recall("d", first)  # used again: the listing order stays by creation
    assert [rid for rid, _ in de_results.list_results("d")] == [second, first]


# --- the DE endpoints keep the labels ---------------------------------------


class _DM:
    def __init__(self, adata):
        self._a = type("A", (), {"adata": adata})()

    async def get_or_load_dataset(self, _id):
        return self._a


def _cells(n: int = 120, g: int = 40, seed: int = 0) -> ad.AnnData:
    """Counts for cell types X/Y in conditions A/B over donors d1-d4: G0-G4 up in
    A in both types, G5-G9 up in A in X only."""
    rng = np.random.default_rng(seed)
    X = rng.poisson(3.0, (n, g)).astype("float32")
    cond = np.array(["A", "B"] * (n // 2))
    ct = np.array(["X"] * (n // 2) + ["Y"] * (n // 2))
    X[cond == "A", :5] += 12
    X[(cond == "A") & (ct == "X"), 5:10] += 12
    a = ad.AnnData(sp.csr_matrix(X))
    a.var_names = [f"G{i}" for i in range(g)]
    a.obs_names = [f"c{i}" for i in range(n)]
    a.obs["cond"] = pd.Categorical(cond)
    a.obs["ct"] = pd.Categorical(ct)
    a.obs["donor"] = pd.Categorical(np.tile(np.repeat(["d1", "d2", "d3", "d4"], 2), n // 8))
    return a


async def test_endpoint_labels_for_each_mode():
    a = _cells()
    dm = _DM(a)
    sel = await differential_expression("x", DERequest(indices=list(range(30)), label="mine",
                                                       method="t-test"), dm)
    within = await differential_expression_compare(
        "x", DECompareRequest(split_by="cond", group_a="A", group_b="B", within_column="ct",
                              within_value="X"), dm)
    everything = await differential_expression_compare(
        "x", DECompareRequest(split_by="cond", group_a="A", group_b="B", method="logreg"), dm)
    labels = {rid: r.label for rid, r in de_results.list_results("x")}
    assert labels[sel.result_id] == "mine (30 cells) vs rest (t-test)"
    assert labels[within.result_id] == "X: A vs B (Wilcoxon)"
    assert labels[everything.result_id] == "All cells: A vs B (logistic regression)"
    c = de_results.recall("x", within.result_id).comparison
    assert c["column"] == "cond" and c["within_column"] == "ct" and c["replicate_key"] == "donor"
    assert c["n_a"] == 30 and c["min_pct"] == pytest.approx(within.min_pct)


async def test_endpoint_label_for_pseudobulk():
    rng = np.random.default_rng(3)
    rows, cond, donor = [], [], []
    for c in ("A", "B"):
        for d in ("d1", "d2", "d3"):
            mu = np.full((15, 30), 4.0)
            if c == "A":
                mu[:, 0] *= 4
            rows.append(rng.poisson(mu))
            cond += [c] * 15
            donor += [d] * 15
    a = ad.AnnData(sp.csr_matrix(np.vstack(rows).astype("float32")))
    a.var_names = [f"G{i}" for i in range(30)]
    a.obs_names = [f"c{i}" for i in range(a.n_obs)]
    a.obs["cond"] = pd.Categorical(cond)
    a.obs["donor"] = pd.Categorical(donor)
    res = await differential_expression_compare(
        "x", DECompareRequest(split_by="cond", group_a="A", group_b="B", method="pseudobulk",
                              replicate_key="donor"), _DM(a))
    r = de_results.recall("x", res.result_id)
    assert r.label == "All cells: A vs B (pseudobulk DESeq2)"
    assert r.comparison["replicate_key"] == "donor" and r.pvals_adj is not None


# --- /de/results and /de/overlap over HTTP ----------------------------------


def _register(dm, did: str = "ov") -> str:
    d = Path(dm.data_dir) / "uploads" / did
    d.mkdir(parents=True)
    (d / "metadata.json").write_text(json.dumps({"id": did, "name": did, "status": "ready"}))
    _cells().write_h5ad(d / "data.h5ad")
    return did


async def _compare(client, did, within):
    r = await client.post(f"{API}/datasets/{did}/de/compare",
                          json={"split_by": "cond", "group_a": "A", "group_b": "B",
                                "within_column": "ct", "within_value": within, "min_pct": 0})
    assert r.status_code == 200, r.text
    return r.json()["result_id"]


async def test_results_list_and_overlap_over_http(client, dataset_manager):
    did = _register(dataset_manager)
    x = await _compare(client, did, "X")
    y = await _compare(client, did, "Y")

    r = await client.get(f"{API}/datasets/{did}/de/results")
    assert r.status_code == 200, r.text
    listing = r.json()
    assert listing["keep"] == de_results.KEEP and listing["padj_max"] == 0.05
    assert [it["result_id"] for it in listing["results"]] == [y, x]
    item = listing["results"][1]
    assert item["label"] == "X: A vs B (Wilcoxon)" and item["n_tested"] == 40
    assert item["has_pvalues"] and item["n_sig_up"] >= 10 and item["created_at"].endswith("+00:00")

    r = await client.post(f"{API}/datasets/{did}/de/overlap",
                          json={"result_ids": [x, y], "direction": "up"})
    assert r.status_code == 200, r.text
    out = r.json()
    shared = {g["gene"] for g in next(i for i in out["intersections"] if i["sets"] == [0, 1])["genes"]}
    only_x = {g["gene"] for g in next(i for i in out["intersections"] if i["sets"] == [0])["genes"]}
    assert {f"G{i}" for i in range(5)} <= shared
    assert {f"G{i}" for i in range(5, 10)} <= only_x
    assert sum(i["n"] for i in out["intersections"]) + out["hidden_genes"] == out["n_union"]

    # Another dataset's listing is empty; its overlap of these ids is a 404.
    assert (await client.get(f"{API}/datasets/other/de/results")).json()["results"] == []


async def test_overlap_rejects_gone_duplicate_and_too_few(client, dataset_manager):
    did = _register(dataset_manager)
    x = await _compare(client, did, "X")
    r = await client.post(f"{API}/datasets/{did}/de/overlap", json={"result_ids": [x, "f" * 32]})
    assert r.status_code == 404
    assert "no longer kept" in r.json()["detail"] and f"last {de_results.KEEP}" in r.json()["detail"]
    r = await client.post(f"{API}/datasets/other/de/overlap", json={"result_ids": [x, x + "0"]})
    assert r.status_code == 404
    r = await client.post(f"{API}/datasets/{did}/de/overlap", json={"result_ids": [x, x]})
    assert r.status_code == 400 and "once" in r.json()["detail"]
    r = await client.post(f"{API}/datasets/{did}/de/overlap", json={"result_ids": [x]})
    assert r.status_code == 400 and "Pick 2 to" in r.json()["detail"]


async def test_overlap_with_a_gene_list_over_http(client, dataset_manager):
    did = _register(dataset_manager)
    x = await _compare(client, did, "X")
    y = await _compare(client, did, "Y")
    r = await client.post(f"{API}/datasets/{did}/de/overlap", json={
        "result_ids": [x, y], "direction": "both", "restrict_to_common_tested": True,
        "extra_sets": [{"name": "mine", "genes": ["g0", "G1", "NOPE"]}]})
    assert r.status_code == 200, r.text
    out = r.json()
    assert out["sets"][2] == {"index": 2, "name": "mine", "kind": "extra", "result_id": None,
                              "method": None, "rule": None, "n": 2, "n_up": 0, "n_down": 0,
                              "n_tested": None, "n_not_tested": 1}
    assert {g["gene"] for g in next(i for i in out["intersections"] if i["sets"] == [0, 1, 2])["genes"]} \
        == {"G0", "G1"}
