"""Cell-cell communication (core/ccc.py, core/lr_resource.py, api/v1/communication.py).

The fixture in ``fixtures/ccc`` is a small dataset (4 groups, one below the
5-cell minimum, 40 consensus interactions of which 10 have a complex) with
liana-py 1.10.0's ``li.mt.rank_aggregate`` output on it (200 permutations,
seed 1337): every method score, the p-values and the specificity rank must
match it, and the magnitude rank must match it with ``liana_compat``.
"""

from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path

import anndata as ad
import httpx
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp
from scipy import stats

from scview.core import ccc, lr_resource

FIX = Path(__file__).parent / "fixtures" / "ccc"
KEYS = ["source", "target", "ligand_complex", "receptor_complex"]
API = "/api/v1"


def _fixture():
    X = sp.load_npz(FIX / "X.npz").tocsr()
    genes = pd.read_csv(FIX / "genes.txt", header=None)[0].astype(str).tolist()
    labels = pd.Series(pd.Categorical(pd.read_csv(FIX / "labels.txt", header=None)[0].astype(str)))
    res = pd.read_csv(FIX / "resource.tsv", sep="\t", dtype=str)
    lia = pd.read_csv(FIX / "liana_rank_aggregate.tsv", sep="\t")
    return X, genes, labels, res, lia


# --- agreement with liana-py ------------------------------------------------------


@pytest.mark.parametrize("dense", [False, True])
def test_scores_match_liana_rank_aggregate(dense):
    X, genes, labels, res, lia = _fixture()
    prep = ccc.prepare(X.toarray() if dense else X, genes, labels, res)
    assert prep.dropped_groups == {"D": 3}
    out = ccc.score(prep, n_perms=200, liana_compat=True)
    m = lia.merge(out, on=KEYS, suffixes=("_li", ""), how="outer", indicator=True)
    assert (m["_merge"] == "both").all() and len(out) == len(lia)
    for col in ("lr_means", "expr_prod", "scaled_weight", "lr_logfc", "spec_weight", "lrscore",
                "cellphone_pvals", "specificity_rank", "magnitude_rank"):
        np.testing.assert_allclose(m[col], m[f"{col}_li"], rtol=1e-4, atol=1e-7, err_msg=col)


def test_magnitude_counts_the_shared_product_once_unlike_liana():
    """liana-py 1.10 re-ranks the column Connectome and NATMI share, which turns it
    upside down; by default it is ranked once, high product = strong."""
    X, genes, labels, res, _ = _fixture()
    prep = ccc.prepare(X, genes, labels, res)
    ours = ccc.score(prep, n_perms=0)
    compat = ccc.score(prep, n_perms=0, liana_compat=True).set_index(KEYS).loc[ours.set_index(KEYS).index]
    prod = ours["expr_prod"].to_numpy()
    # By default a higher product never makes the aggregate weaker, all else equal.
    r_ours = stats.spearmanr(prod, ours["magnitude_rank"]).statistic
    r_compat = stats.spearmanr(prod, compat["magnitude_rank"]).statistic
    assert r_ours < r_compat
    np.testing.assert_allclose(ours["specificity_rank"], compat["specificity_rank"].to_numpy())


# --- the pieces, against direct formulas -------------------------------------------


def _toy(seed: int = 0):
    """3 groups; L1 high in group a, R1 high in group b; a complex receptor R2_R3."""
    rng = np.random.default_rng(seed)
    sizes = {"a": 40, "b": 30, "c": 20}
    genes = ["L1", "R1", "R2", "R3", "L2", "N1", "N2"]
    rows = []
    for g, n in sizes.items():
        base = rng.gamma(1.5, 0.4, size=(n, len(genes))) * (rng.random((n, len(genes))) < 0.5)
        if g == "a":
            base[:, 0] += 3.0
        if g == "b":
            base[:, 1] += 3.0
        rows.append(base)
    X = np.log1p(np.vstack(rows)).astype(np.float32)
    X[:, genes.index("N2")] = 0
    labels = pd.Series(np.concatenate([[g] * n for g, n in sizes.items()]))
    res = pd.DataFrame({"ligand": ["L1", "L2", "L1", "L1"], "receptor": ["R1", "R2_R3", "R1", "N9"]})
    return X, genes, labels, res


def test_method_scores_follow_their_definitions():
    X, genes, labels, res = _toy()
    prep = ccc.prepare(sp.csr_matrix(X), genes, labels, res)
    assert prep.n_resource == 3  # the duplicate pair counts once; L1-N9 is not scorable
    out = ccc.score(prep, n_perms=0).set_index(KEYS)
    lab = labels.to_numpy()
    n = len(lab)
    expr = (X != 0)
    mean = {g: X[lab == g].mean(axis=0) for g in "abc"}
    gi = {g: i for i, g in enumerate(genes)}
    row = out.loc[("a", "b", "L1", "R1")]
    lm, rm = mean["a"][gi["L1"]], mean["b"][gi["R1"]]
    assert row["lr_means"] == pytest.approx((lm + rm) / 2, rel=1e-5)
    assert row["expr_prod"] == pytest.approx(lm * rm, rel=1e-5)
    sum_l = sum(mean[g][gi["L1"]] for g in "abc")
    sum_r = sum(mean[g][gi["R1"]] for g in "abc")
    assert row["spec_weight"] == pytest.approx(lm / sum_l * rm / sum_r, rel=1e-5)
    mat_mean = X.sum() / (n * 6)  # N2 is empty: not counted
    assert row["lrscore"] == pytest.approx(np.sqrt(lm * rm) / (np.sqrt(lm * rm) + mat_mean), rel=1e-5)
    mu, sd = X.mean(axis=0), X.std(axis=0, ddof=1)
    z = lambda g, gene: (mean[g][gi[gene]] - mu[gi[gene]]) / sd[gi[gene]]  # noqa: E731
    assert row["scaled_weight"] == pytest.approx((z("a", "L1") + z("b", "R1")) / 2, rel=1e-4)
    E = np.expm1(X.astype(np.float64))
    lfc = lambda g, gene: (np.log2(E[lab == g, gi[gene]].mean() + 1)  # noqa: E731
                           - np.log2(E[lab != g, gi[gene]].mean() + 1))
    assert row["lr_logfc"] == pytest.approx((lfc("a", "L1") + lfc("b", "R1")) / 2, rel=1e-5)
    assert row["ligand_props"] == pytest.approx(expr[lab == "a", gi["L1"]].mean())
    # The planted pair is the most specific a -> b interaction.
    assert out.xs(("a", "b"), level=("source", "target"))["specificity_rank"].idxmin() == ("L1", "R1")


def test_a_complex_is_scored_by_its_least_expressed_and_filtered_by_its_least_detected_subunit():
    X, genes, labels, res = _toy()
    gi = {g: i for i, g in enumerate(genes)}
    X[:, gi["R3"]] *= 0.25  # R3 lower than R2 everywhere it is expressed
    lab = labels.to_numpy()
    X[lab == "c", gi["R3"]] = 0  # ... and absent in c
    prep = ccc.prepare(X, genes, labels, res)
    out = ccc.score(prep, n_perms=0)
    cplx = out[out["receptor_complex"] == "R2_R3"]
    assert not cplx.empty and "c" not in set(cplx["target"])
    assert set(cplx["receptor"]) == {"R3"}
    t = cplx.iloc[0]["target"]
    assert cplx.iloc[0]["receptor_means"] == pytest.approx(X[lab == t, gi["R3"]].mean(), rel=1e-5)


def test_expr_prop_filter_needs_both_genes_detected():
    X, genes, labels, res = _toy()
    prep = ccc.prepare(X, genes, labels, res)
    lo = ccc.score(prep, n_perms=0, expr_prop=0.0)
    hi = ccc.score(prep, n_perms=0, expr_prop=0.9)
    assert len(hi) < len(lo)
    assert (hi["ligand_props"] >= 0.9).all() and (hi["receptor_props"] >= 0.9).all()


def _naive_counts(X, codes, n_groups, rows, observed, n_perms, seed):
    """LIANA's own loop: shuffled row order, mean over each label's positions."""
    rng = np.random.default_rng(seed=seed)
    idx = np.arange(X.shape[0])
    counts = np.zeros(len(rows), dtype=int)
    for _ in range(n_perms):
        P = X[rng.permutation(idx)]
        means = np.array([P[codes == g].mean(axis=0) for g in range(n_groups)])
        perm = (means[rows["s"], rows["lig"]] + means[rows["t"], rows["rec"]]) / 2
        counts += perm >= observed
    return counts


def test_permutations_are_liana_shuffles_in_batches(monkeypatch):
    X, genes, labels, res = _toy()
    prep = ccc.prepare(X, genes, labels, res)
    rows = ccc.candidate_rows(prep)
    st = prep.stats
    obs = (st.means[rows["s"], rows["lig"]] + st.means[rows["t"], rows["rec"]]) / 2
    want = _naive_counts(prep.X.toarray().astype(np.float64), prep.codes, 3, rows, obs, 30, 5)
    monkeypatch.setattr(ccc, "_PERM_BATCH", 7)  # batches that do not divide 30
    seen = []
    got = ccc.permutation_counts(prep.X, prep.codes, 3, rows, obs, n_perms=30, seed=5,
                                 progress=lambda d, n: seen.append((d, n)))
    np.testing.assert_array_equal(got, want)
    assert seen[-1] == (30, 30) and len(seen) == 5


def test_a_stop_request_ends_the_permutations():
    X, genes, labels, res = _toy()
    prep = ccc.prepare(X, genes, labels, res)
    with pytest.raises(InterruptedError):
        ccc.score(prep, n_perms=100, stop=lambda: True)


def test_robust_rank_aggregate_is_the_beta_order_statistic():
    rmat = np.array([[1.0, 2.0, 1.0], [3.0, 1.0, 3.0], [2.0, 3.0, 2.0]])
    got = ccc.robust_rank_aggregate(rmat)
    r = np.sort(rmat / rmat.max(axis=0), axis=1)
    k = 3
    want = [min(1.0, k * min(stats.beta.cdf(r[i, j], j + 1, k - j) for j in range(k))) for i in range(3)]
    np.testing.assert_allclose(got, want)


def test_unlabelled_cells_small_groups_and_wrong_species_are_handled():
    X, genes, labels, res = _toy()
    labels = labels.copy().astype(object)
    labels.iloc[:3] = None
    labels.iloc[-4:] = "tiny"
    prep = ccc.prepare(X, genes, labels, res)
    assert prep.dropped_groups == {"tiny": 4}
    assert prep.X.shape[0] == len(labels) - 7 and prep.stats.groups == ["a", "b", "c"]
    mouse = pd.DataFrame({"ligand": ["Cd4"], "receptor": ["Cd8a"]})
    with pytest.raises(ccc.CCCError, match="right species"):
        ccc.prepare(X, genes, labels, mouse)
    with pytest.raises(ccc.CCCError, match="at least 2 groups"):
        ccc.prepare(X, genes, pd.Series(["a"] * len(labels)), res)


def test_pair_summary_and_roles_count_specific_interactions():
    X, genes, labels, res = _toy()
    prep = ccc.prepare(X, genes, labels, res)
    tab = ccc.score(prep, n_perms=50)
    r = ccc.Result(id="x", dataset_id="d", groupby="g", subset=None, resource="consensus",
                   species="human", matrix_source="X", groups=prep.stats.groups,
                   group_sizes=prep.stats.sizes.tolist(), dropped_groups={}, expr_prop=0.1,
                   n_perms=50, n_resource=3, table=tab, elapsed_seconds=0.1)
    spec = tab[tab["specificity_rank"] <= ccc.SPECIFIC]
    assert r.counts.sum() == len(spec)
    roles = r.roles().set_index("group")
    for g in r.groups:
        assert roles.loc[g, "outgoing"] == (spec["source"] == g).sum()
        assert roles.loc[g, "incoming"] == (spec["target"] == g).sum()

    class Adaptor:
        pass

    a = Adaptor()
    obs = pd.DataFrame({"g": labels})
    ccc.remember(a, r)
    lines, detail = ccc.grounding_lines(a, obs)
    assert detail and "between 3 'g' groups, all cells" in detail
    pairs = [x for x in lines if " -> " in x and "(magnitude rank" in x]
    assert pairs and len(pairs) == len({x.split(": ", 1)[1].split(" (")[0] for x in pairs})  # one line per LR pair
    # Re-clustered since: the result no longer describes these groups.
    lines, detail = ccc.grounding_lines(a, pd.DataFrame({"g": ["x"] * len(labels)}))
    assert detail is None and lines == []


# --- resource -----------------------------------------------------------------------


def _resource_csv(n: int = 1100) -> bytes:
    rows = []
    for i in range(n):
        rows.append((f"P{i}", f"Q{i}", f"LIG{i}", f"REC{i % 50}" if i % 7 else f"RA{i}_RB{i}", "consensus"))
        rows.append((f"P{i}", f"Q{i}", f"Lig{i}", f"Rec{i % 50}", "mouseconsensus"))
        rows.append((f"P{i}", f"Q{i}", f"X{i}", f"Y{i}", "cellphonedb"))
    df = pd.DataFrame(rows, columns=["source", "target", "source_genesymbol", "target_genesymbol", "resource"])
    buf = io.StringIO()
    df.to_csv(buf)
    return buf.getvalue().encode()


def _client(handler) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(handler))


def test_resource_keeps_the_two_consensus_tables():
    t = lr_resource.parse(_resource_csv())
    assert set(t["resource"]) == {"consensus", "mouseconsensus"}
    assert lr_resource.select(t, "consensus").iloc[0].tolist() == ["LIG0", "RA0_RB0"]
    with pytest.raises(KeyError):
        lr_resource.select(t, "cellphonedb")
    with pytest.raises(ValueError, match="lacks columns"):
        lr_resource.parse(b"a,b\n1,2\n")
    with pytest.raises(ValueError, match="'mouseconsensus' rows"):
        lr_resource.parse(_resource_csv(1100).replace(b"mouseconsensus", b"other"))


def test_download_checks_the_pinned_sha256(tmp_path):
    raw = _resource_csv()
    good = hashlib.sha256(raw).hexdigest()
    with pytest.raises(ValueError, match="sha256"):
        lr_resource.download(tmp_path, client=_client(lambda r: httpx.Response(200, content=raw)))
    assert not list(tmp_path.glob("liana_lr_resource*"))
    t = lr_resource.download(tmp_path, client=_client(lambda r: httpx.Response(200, content=raw)), sha256=good)
    assert len(t) == 2200
    meta = lr_resource.read_meta(tmp_path)
    assert meta["sha256"] == good and meta["n_interactions"] == {"consensus": 1100, "mouseconsensus": 1100}


def test_an_offline_first_use_says_what_to_do(tmp_path):
    def boom(request):
        raise httpx.ConnectError("no network")

    with pytest.raises(lr_resource.ResourceUnavailable, match="check the internet connection"):
        lr_resource.load(tmp_path / "networks", client=_client(boom))


# --- endpoints ----------------------------------------------------------------------


def _events(text: str) -> list[tuple[str, dict]]:
    out = []
    for block in text.strip().split("\n\n"):
        if block.startswith(":"):
            continue
        lines = dict(line.split(": ", 1) for line in block.splitlines())
        out.append((lines["event"], json.loads(lines["data"])))
    return out


@pytest.fixture()
def resource_on_disk(tmp_data_dir: Path, monkeypatch) -> Path:
    """The fixture's interactions as the 'downloaded' human consensus resource."""
    _, _, _, res, _ = _fixture()
    dest = tmp_data_dir / "networks"
    dest.mkdir()
    table = pd.concat([res.assign(resource="consensus"),
                       res.assign(resource="mouseconsensus",
                                  ligand=res["ligand"].str.capitalize(),
                                  receptor=res["receptor"].str.capitalize())])
    table[["resource", "ligand", "receptor"]].to_csv(dest / "liana_lr_resource.tsv", sep="\t", index=False)
    (dest / "liana_lr_resource.json").write_text(json.dumps({"fetched_at": "2026-10-21"}))
    return dest


def _dataset(dm, did: str = "ccc") -> str:
    X, genes, labels, _, _ = _fixture()
    n = X.shape[0]
    obs = pd.DataFrame({"group": labels.to_numpy(),
                        "stim": pd.Categorical(np.where(np.arange(n) % 2, "STIM", "CTRL"))},
                       index=[f"c{i}" for i in range(n)])
    a = ad.AnnData(X=X, obs=obs, var=pd.DataFrame(index=genes))
    d = Path(dm.data_dir) / "uploads" / did
    d.mkdir(parents=True)
    (d / "metadata.json").write_text(json.dumps({"id": did, "name": did, "status": "ready"}))
    a.write_h5ad(d / "data.h5ad")
    return did


async def test_options_list_conditions_and_estimate(client, dataset_manager, resource_on_disk):
    did = _dataset(dataset_manager)
    r = await client.get(f"{API}/datasets/{did}/communication/options",
                         params={"groupby": "group", "subset_column": "stim", "subset_value": "STIM"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert {"column": "stim", "values": ["CTRL", "STIM"]} in body["conditions"]
    assert body["n_groups"] == 4 and body["n_cells"] == 76 and body["estimate_seconds"] > 0
    status = (await client.get(f"{API}/communication/resource")).json()
    assert status["downloaded"] is True and "v1.10.0" in status["url"]


async def test_run_streams_progress_and_result_then_csv_and_copilot(client, dataset_manager, resource_on_disk):
    from scview.core.assistant import build_grounding_context

    did = _dataset(dataset_manager)
    r = await client.post(f"{API}/datasets/{did}/communication/run",
                          json={"groupby": "group", "n_perms": 60, "resource": "consensus"})
    assert r.status_code == 200, r.text
    events = _events(r.text)
    names = [e for e, _ in events]
    assert names[0] == "start" and names[-1] == "complete" and "progress" in names
    res = dict(events)["result"]
    assert res["groups"] == ["A", "B", "C"] and res["dropped_groups"] == {"D": 3}
    assert res["n_rows"] == res["rows_shown"] == len(res["columns"]["source"])
    assert len(res["pair_counts"]) == 3 and sum(map(sum, res["pair_counts"])) == res["n_specific"]
    assert res["columns"]["magnitude_rank"] == sorted(res["columns"]["magnitude_rank"])

    csv = await client.get(f"{API}/datasets/{did}/communication/results/{res['result_id']}.csv")
    assert csv.status_code == 200 and "attachment" in csv.headers["content-disposition"]
    tab = pd.read_csv(io.StringIO(csv.text))
    assert len(tab) == res["n_rows"] and {"cellphone_pvals", "specificity_rank"} <= set(tab.columns)
    gone = await client.get(f"{API}/datasets/{did}/communication/results/nothere.csv")
    assert gone.status_code == 404

    adaptor = await dataset_manager.get_or_load_dataset(did)
    context, sources = build_grounding_context(adaptor, {"groups"})
    assert "Cell-cell communication (last run" in context
    assert any(s.ref == "result:communication" for s in sources)


async def test_run_on_one_condition_only(client, dataset_manager, resource_on_disk):
    did = _dataset(dataset_manager)
    r = await client.post(f"{API}/datasets/{did}/communication/run",
                          json={"groupby": "group", "subset_column": "stim", "subset_value": "CTRL",
                                "n_perms": 0, "resource": "consensus"})
    res = dict(_events(r.text))["result"]
    assert res["subset"] == {"column": "stim", "value": "CTRL"}
    assert res["n_cells"] == 75  # 77 CTRL cells (even indices) less D's 2
    assert all(p is None for p in res["columns"]["cellphone_pvals"])


async def test_bad_requests_are_refused(client, dataset_manager, resource_on_disk):
    did = _dataset(dataset_manager)
    url = f"{API}/datasets/{did}/communication/run"
    assert (await client.post(url, json={"groupby": "nope"})).status_code == 404
    assert (await client.post(url, json={"groupby": "group", "subset_column": "stim",
                                         "subset_value": "NONE"})).status_code == 400
    assert (await client.post(url, json={"groupby": "group", "resource": "x"})).status_code == 400
    assert (await client.post(url, json={"groupby": "group", "n_perms": 99999})).status_code == 422
    # Mouse resource on human genes: nothing to score, said plainly.
    r = await client.post(url, json={"groupby": "group", "resource": "mouseconsensus", "n_perms": 0})
    assert "right species" in dict(_events(r.text))["error"]["error"]
