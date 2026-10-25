"""Cell-cell communication compared between two conditions (core/ccc_compare.py,
``POST /communication/compare``).

A small design with a planted change: monocytes express LUP (to TREC on T
cells) only in STIM, B cells express LLOST (to TREC) only in CTRL, and LSAME
(B cells) -> MREC (monocytes) is the same in both. Four donors, each sampled in
both conditions.
"""

from __future__ import annotations

import io
import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

from scview.core import ccc, ccc_compare

API = "/api/v1"
GENES = ["LUP", "LLOST", "LSAME", "TREC", "MREC", "CDA", "CDB"] + [f"F{i}" for i in range(20)]
# The planted pairs, and ten broadly expressed filler pairs: ranks need a crowd.
RESOURCE = pd.DataFrame({"ligand": ["LUP", "LLOST", "LSAME", "LSAME", "LUP"] + [f"F{i}" for i in range(10)],
                         "receptor": ["TREC", "TREC", "MREC", "CDA_CDB", "MREC"]
                         + [f"F{i + 10}" for i in range(10)]})


def toy(n_per: int = 25, donors: int = 4, seed: int = 0, stim_mono_extra: int = 0):
    """(log-normalised X, obs) with the planted signals; ``stim_mono_extra`` adds
    monocytes to STIM only (a composition change)."""
    rng = np.random.default_rng(seed)
    rows, obs = [], []
    for d in range(donors):
        for cond in ("CTRL", "STIM"):
            for grp in ("Mono", "T", "B", "tiny"):
                n = 2 if grp == "tiny" and cond == "STIM" else n_per
                if grp == "Mono" and cond == "STIM":
                    n += stim_mono_extra
                lam = np.full(len(GENES), 0.3)
                lam[GENES.index("TREC")] = 3 if grp == "T" else 0.05
                lam[GENES.index("MREC")] = 3 if grp == "Mono" else 0.05
                lam[GENES.index("LSAME")] = 3 if grp == "B" else 0.05
                lam[GENES.index("LLOST")] = 3 if (grp == "B" and cond == "CTRL") else 0.0
                # Donor-specific level, so the paired test has a spread to work with.
                lam[GENES.index("LUP")] = (2 + 0.5 * d) if (grp == "Mono" and cond == "STIM") else 0.0
                rows.append(np.log1p(rng.poisson(lam, size=(n, len(GENES))).astype(np.float64)))
                obs += [{"cell_type": grp, "stim": cond, "donor": f"d{d}"}] * n
    X = sp.csr_matrix(np.vstack(rows))
    obs = pd.DataFrame(obs, index=[f"c{i}" for i in range(X.shape[0])])
    obs["cell_type"] = pd.Categorical(obs["cell_type"], categories=["Mono", "T", "B", "tiny"])
    return X, obs


def _row(tab: pd.DataFrame, s: str, t: str, lig: str, rec: str) -> pd.Series:
    hit = tab[(tab.source == s) & (tab.target == t) & (tab.ligand_complex == lig) & (tab.receptor_complex == rec)]
    assert len(hit) == 1, (s, t, lig, rec)
    return hit.iloc[0]


@pytest.mark.parametrize("dense", [False, True])
def test_planted_gains_and_losses_are_called_with_a_paired_donor_test(dense):
    X, obs = toy()
    if dense:
        X = X.toarray()
    out = ccc_compare.compare(X, GENES, obs, "cell_type", "stim", "CTRL", "STIM", RESOURCE, n_perms=50)
    tab = out["table"]
    assert out["groups"] == ["Mono", "T", "B"]
    assert out["dropped_groups"] == {"tiny": {"a": 100, "b": 8}}
    up = _row(tab, "Mono", "T", "LUP", "TREC")
    assert up["call"] == "gained" and up["lr_means_a"] == 0 and up["delta"] > 0
    assert np.isnan(up["specificity_rank_a"])  # silent in CTRL: never passes the filter there
    lost = _row(tab, "B", "T", "LLOST", "TREC")
    assert lost["call"] == "lost" and lost["delta"] < 0
    same = _row(tab, "B", "Mono", "LSAME", "MREC")
    assert same["call"] == "unchanged"
    rep = out["replicates"]
    assert rep["sample_key"] == "donor" and rep["paired"] and rep["n_pairs"] == 4
    assert rep["test"].startswith("Wilcoxon") and "descriptive" in rep["caution"]
    # Four pairs all up: the exact signed-rank minimum, 2 / 2**4.
    assert up["p_value"] == pytest.approx(0.125) and up["n_donors"] == 4
    assert np.isnan(same["p_value"]) or same["p_value"] > 0.1
    assert np.all(np.isfinite(tab["p_adj"]) == np.isfinite(tab["p_value"]))
    # Changed rows first, biggest change first.
    calls = tab["call"].tolist()
    assert calls.index("unchanged") > max(i for i, c in enumerate(calls) if c != "unchanged")
    # Pair summary: B and A tables counted per sender x receiver.
    i, j = out["groups"].index("Mono"), out["groups"].index("T")
    assert out["counts_b"][i, j] > out["counts_a"][i, j]


def test_each_condition_scores_as_a_run_on_its_own_cells():
    # Same groups and genes for both, yet each condition's table is exactly the
    # single run on that condition's cells (the extra, silent pairs are never scored).
    X, obs = toy()
    keep = obs["cell_type"].isin(["Mono", "T", "B"]).to_numpy()
    lab = pd.Series(pd.Categorical(obs["cell_type"].astype(str)[keep], categories=["Mono", "T", "B"]))
    in_a = (obs["stim"] == "CTRL").to_numpy()[keep]
    pa, pb = ccc_compare.split_prepared(X[keep], GENES, lab, in_a, RESOURCE)
    assert pa.stats.genes == pb.stats.genes and len(pa.interactions) == len(pb.interactions)
    for prep, mask in ((pa, in_a), (pb, ~in_a)):
        single = ccc.prepare(X[keep][mask], GENES, lab[mask].reset_index(drop=True), RESOURCE)
        got = ccc.score(prep, n_perms=40)
        want = ccc.score(single, n_perms=40)
        cols = ["source", "target", "ligand_complex", "receptor_complex", "lr_means", "lrscore",
                "cellphone_pvals", "magnitude_rank", "specificity_rank"]
        pd.testing.assert_frame_equal(got[cols], want[cols])


def test_no_donor_column_means_no_p_values_and_a_reason():
    X, obs = toy()
    obs = obs.drop(columns="donor")
    out = ccc_compare.compare(X, GENES, obs, "cell_type", "stim", "CTRL", "STIM", RESOURCE, n_perms=0)
    assert out["replicates"]["test"] is None and "no test" in out["replicates"]["reason"]
    assert out["table"]["p_value"].isna().all() and out["donor_values"] == {}


def test_unpaired_donors_use_welch():
    X, obs = toy(donors=4)
    # Donors d0, d1 only in CTRL and d2, d3 only in STIM: not paired.
    keep = ((obs["stim"] == "CTRL") & obs["donor"].isin(["d0", "d1"])) | \
           ((obs["stim"] == "STIM") & obs["donor"].isin(["d2", "d3"]))
    k = keep.to_numpy()
    out = ccc_compare.compare(X[k], GENES, obs[k], "cell_type", "stim", "CTRL", "STIM", RESOURCE, n_perms=0)
    rep = out["replicates"]
    assert rep["paired"] is False and rep["test"].startswith("Welch")
    up = _row(out["table"], "Mono", "T", "LUP", "TREC")
    assert up["n_donors"] == 4 and np.isfinite(up["p_value"])


def test_composition_change_is_noted():
    X, obs = toy(stim_mono_extra=150)
    out = ccc_compare.compare(X, GENES, obs, "cell_type", "stim", "CTRL", "STIM", RESOURCE, n_perms=0)
    comp = {c["group"]: c for c in out["composition"]}
    assert comp["Mono"]["flagged"] and comp["Mono"]["fold"] > 2
    tab = out["table"]
    assert tab.loc[(tab.source == "Mono") | (tab.target == "Mono"), "composition_note"].all()


def test_call_rules():
    spec_a = np.array([False, True, True, True, False, True])
    spec_b = np.array([True, False, True, True, True, False])
    lr_a = np.array([0.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    lr_b = np.array([1.0, 0.2, 2.0, 1.1, 1.05, 0.95])
    assert ccc_compare.call_rows(spec_a, spec_b, lr_a, lr_b).tolist() == [
        "gained", "lost", "changed", "unchanged", "unchanged", "unchanged"]


def test_bad_inputs():
    X, obs = toy()
    with pytest.raises(ccc_compare.CompareError):
        ccc_compare.compare(X, GENES, obs, "stim", "stim", "CTRL", "STIM", RESOURCE)
    with pytest.raises(ccc_compare.CompareError):
        ccc_compare.compare(X, GENES, obs, "cell_type", "stim", "CTRL", "CTRL", RESOURCE)
    with pytest.raises(ccc_compare.CompareError, match="Fewer than 2"):
        ccc_compare.compare(X, GENES, obs, "cell_type", "stim", "CTRL", "STIM", RESOURCE, min_cells=500)


# --- endpoint -------------------------------------------------------------------------


def _events(text: str) -> list[tuple[str, dict]]:
    out = []
    for block in text.strip().split("\n\n"):
        if block.startswith(":"):
            continue
        lines = dict(line.split(": ", 1) for line in block.splitlines())
        out.append((lines["event"], json.loads(lines["data"])))
    return out


@pytest.fixture()
def toy_resource(tmp_data_dir: Path) -> Path:
    dest = tmp_data_dir / "networks"
    dest.mkdir()
    table = pd.concat([RESOURCE.assign(resource="consensus"), RESOURCE.assign(resource="mouseconsensus")])
    table[["resource", "ligand", "receptor"]].to_csv(dest / "liana_lr_resource.tsv", sep="\t", index=False)
    (dest / "liana_lr_resource.json").write_text(json.dumps({"fetched_at": "2026-10-25"}))
    return dest


def _dataset(dm, did: str = "cmp") -> str:
    X, obs = toy()
    a = ad.AnnData(X=X, obs=obs, var=pd.DataFrame(index=GENES))
    d = Path(dm.data_dir) / "uploads" / did
    d.mkdir(parents=True)
    (d / "metadata.json").write_text(json.dumps({"id": did, "name": did, "status": "ready"}))
    a.write_h5ad(d / "data.h5ad")
    return did


async def test_compare_streams_result_csv_donors_and_copilot(client, dataset_manager, toy_resource):
    from scview.core.assistant import build_grounding_context

    did = _dataset(dataset_manager)
    r = await client.post(f"{API}/datasets/{did}/communication/compare",
                          json={"groupby": "cell_type", "condition": "stim", "a": "CTRL", "b": "STIM",
                                "n_perms": 30, "resource": "consensus"})
    assert r.status_code == 200, r.text
    events = _events(r.text)
    names = [e for e, _ in events]
    assert names[0] == "start" and names[-1] == "complete", events[-1]
    prog = [d for e, d in events if e == "progress"]
    assert {p["condition"] for p in prog} == {"CTRL", "STIM"}
    res = dict(events)["result"]
    assert res["groups"] == ["Mono", "T", "B"] and "tiny" in res["dropped_groups"]
    assert res["calls"]["gained"] >= 1 and res["calls"]["lost"] >= 1
    assert res["replicates"]["paired"] is True
    cols = res["columns"]
    assert cols["call"][0] != "unchanged" and len(cols["source"]) == res["rows_shown"]
    assert len(res["counts_a"]) == 3 and len(res["strength_b"][0]) == 3

    csv = await client.get(f"{API}/datasets/{did}/communication/compare/{res['result_id']}.csv")
    assert csv.status_code == 200 and "STIM_vs_CTRL" in csv.headers["content-disposition"]
    tab = pd.read_csv(io.StringIO(csv.text))
    assert len(tab) == res["n_rows"] and "_int" not in tab.columns and "p_adj" in tab.columns

    i = next(k for k in range(len(cols["source"])) if cols["ligand_complex"][k] == "LUP"
             and cols["source"][k] == "Mono" and cols["target"][k] == "T")
    don = (await client.get(f"{API}/datasets/{did}/communication/compare/{res['result_id']}/donors/{i}")).json()
    assert don["available"] and don["paired"] and len(don["points"]) == 8
    assert {p["condition"] for p in don["points"]} == {"CTRL", "STIM"}
    assert all(p["n_cells"] == 25 for p in don["points"])
    assert don["paired_test"]["n_pairs"] == 4 and don["paired_test"]["wilcoxon_p"] == pytest.approx(0.125)
    assert (await client.get(f"{API}/datasets/{did}/communication/compare/{res['result_id']}/donors/999")
            ).status_code == 404
    assert (await client.get(f"{API}/datasets/{did}/communication/compare/nothere.csv")).status_code == 404

    adaptor = await dataset_manager.get_or_load_dataset(did)
    context, sources = build_grounding_context(adaptor, {"groups"})
    assert "compared between conditions" in context and "gained: Mono -> T: LUP -> TREC" in context
    assert any(s.ref == "result:communication_compare" for s in sources)


async def test_compare_refuses_bad_requests(client, dataset_manager, toy_resource):
    did = _dataset(dataset_manager)
    url = f"{API}/datasets/{did}/communication/compare"
    base = {"groupby": "cell_type", "condition": "stim", "a": "CTRL", "b": "STIM"}
    assert (await client.post(url, json={**base, "groupby": "nope"})).status_code == 404
    assert (await client.post(url, json={**base, "b": "NONE"})).status_code == 400
    assert (await client.post(url, json={**base, "b": "CTRL"})).status_code == 400
    assert (await client.post(url, json={**base, "groupby": "stim"})).status_code == 400
    assert (await client.post(url, json={**base, "sample_key": "nope"})).status_code == 404
    assert (await client.post(url, json={**base, "resource": "x"})).status_code == 400
