"""DE audit (core/de_audit.py, POST /de/audit): donor-label shuffles and
pseudobulk concordance for a cell-level comparison.

Kang 2018 (CD14 Mono, STIM vs CTRL within the cell type, Donor, Wilcoxon, 1%
prefilter, 20 balanced shuffles, seed 0): 1,836 genes at cell level, 10-663 in
the shuffles (median 310), empirical p 0.048; pseudobulk DESeq2 (~Donor +
condition) 2,265 of 3,454, 1,565 shared, Jaccard 0.62, Spearman of log fold
changes 0.995; 271 genes significant at cell level only (MYL6, TMSB4X, PSAP, FTL,
RPS19 first). Their top-donor shares are low (0.04-0.17) and their pseudobulk
fold changes a fifth of the cell-level ones: not one donor, but power and
library-size normalisation under the large interferon response (DESeq2's
median-of-ratios is not moved by it). 12 s on 2 cores.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

from scview.api.v1.de import _prepare
from scview.core import de_audit, pseudobulk

MAIN = threading.main_thread()


def synthetic(donors_a, donors_b, *, cells: int = 60, g: int = 80, effect: float = 1.0,
              effect_genes: int = 10, donor_sd: float = 0.1, outlier: str | None = None,
              outlier_fold: float = 8.0, seed: int = 0) -> ad.AnnData:
    """Negative-binomial counts for donor x condition samples. Genes 0..effect_genes-1
    are ``effect``-fold higher in every donor, the first half in condition A, the
    rest in B; with ``outlier``, that
    donor alone has genes 20-39 ``outlier_fold``-fold higher (on either side)."""
    rng = np.random.default_rng(seed)
    base = rng.gamma(2.0, 1.0, size=g) + 1.0
    shift = {d: rng.lognormal(0, donor_sd, size=g) for d in sorted(set(donors_a) | set(donors_b))}
    rows, cond, donor = [], [], []
    for c, ds in (("A", donors_a), ("B", donors_b)):
        for d in ds:
            mu = np.tile(base * shift[d], (cells, 1))
            # Half the genes up in A, half up in B: equal library sizes, so normalisation
            # shifts no other gene.
            half = effect_genes // 2
            mu[:, :half] *= effect if c == "A" else 1.0
            mu[:, half:effect_genes] *= effect if c == "B" else 1.0
            if d == outlier:
                mu[:, 20:40] *= outlier_fold
            lam = rng.gamma(1 / 0.2, mu * 0.2)
            rows.append(rng.poisson(lam))
            cond += [c] * cells
            donor += [d] * cells
    a = ad.AnnData(sp.csr_matrix(np.vstack(rows).astype("float32")))
    a.var_names = [f"G{i}" for i in range(g)]
    a.obs_names = [f"c{i}" for i in range(a.n_obs)]
    a.obs["cond"] = pd.Categorical(cond)
    a.obs["donor"] = pd.Categorical(donor)
    a.obs["ct"] = pd.Categorical(["X"] * a.n_obs)
    return a


def _audit(a: ad.AnnData, k: int = 10, **kw) -> de_audit.AuditResult:
    in_a = (a.obs["cond"] == "A").to_numpy()
    test, _ = _prepare(a, in_a, ~in_a, 0.01)

    def pb():
        return pseudobulk.pseudobulk_de(a, in_a, a.obs["donor"], label_a="A", label_b="B",
                                        donor_key="donor")

    return de_audit.run(test, in_a, a.obs["donor"].astype(str).to_numpy(), n_permutations=k,
                        pseudobulk=pb, donor_key="donor", label="A vs B", **kw)


# --- the shuffles -----------------------------------------------------------------


def _design(da: list[str], db: list[str]) -> de_audit.Design:
    donors = np.array(da + db, dtype=object)
    in_a = np.array([True] * len(da) + [False] * len(db))
    return de_audit.design(donors, in_a)


def test_design_names_the_scheme_and_counts_balanced_shuffles():
    d = _design(list("abc"), list("abc"))
    assert d.scheme == "paired" and (d.n_donors_a, d.n_donors_b) == (3, 3)
    assert d.n_possible() == 3  # swap 1 of 3 donors (2 of 3 is the same, sides swapped)
    d = _design([f"d{i}" for i in range(8)], [f"d{i}" for i in range(8)])
    assert d.n_possible() == 35  # C(8, 4) / 2
    d = _design(list("abc"), list("def"))
    assert d.scheme == "donors" and d.n_possible() == 9  # every other labelling of 3 + 3
    d = _design(list("abc"), list("defgh"))
    assert d.n_possible() == 45  # 1 or 2 of A's 3 donors stay on A
    d = _design(list("abc"), list("bcd"))
    assert d.scheme == "samples"
    assert not _design(list("ab"), list("cde")).enough


@pytest.mark.parametrize("da,db", [(list("abcdef"), list("abcdef")), (list("abc"), list("def")),
                                   (list("abcd"), list("cdef")), (list("abcde"), list("fghijklm"))])
def test_shuffles_are_distinct_balanced_and_never_the_observed(da, db):
    d = _design(da, db)
    labs, exhaustive = de_audit.shuffles(d, 1000)
    assert exhaustive and len(labs) == d.n_possible()
    keys = {de_audit._canonical(x) for x in labs}
    assert len(keys) == len(labs) and de_audit._canonical(d.sample_is_a) not in keys
    some, ex = de_audit.shuffles(d, 4, seed=3)
    assert len(some) == min(4, d.n_possible()) and ex == (d.n_possible() <= 4)
    _, _, allowed = d.balance()
    for lab in labs:
        if d.scheme == "paired":  # half the donors swapped
            assert int((lab != d.sample_is_a).sum()) // 2 in allowed
        else:  # A's samples split as evenly as the sizes allow
            assert int((lab & d.sample_is_a).sum()) in allowed
            assert lab.sum() == d.sample_is_a.sum()


def test_qq_spreads_points_over_the_tail():
    rng = np.random.default_rng(0)
    obs = np.concatenate([rng.uniform(size=900), np.full(100, 1e-20), [0.0]])
    q = de_audit.qq(obs, [rng.uniform(size=1001) for _ in range(5)])
    assert len(q["expected"]) == len(q["observed"]) == len(q["permuted"]) <= de_audit.QQ_POINTS
    assert q["expected"] == sorted(q["expected"], reverse=True)
    assert q["observed"][0] == 300.0  # p = 0 capped
    assert q["observed"][0] > 10 > q["permuted"][0]
    assert de_audit.qq(np.array([0.5]), [])["permuted"] is None


def test_top_donor_share_is_even_or_one_donor():
    donors = np.repeat(np.array(["a", "b", "c", "d"], dtype=object), 10)
    in_a = np.repeat([True, True, False, False], 10)
    d = de_audit.design(donors, in_a)
    X = np.zeros((40, 2))
    X[in_a, 0] = 1.0  # both A donors carry it
    X[:10, 1] = 1.0  # donor a alone
    shares = de_audit.donor_shares(X, np.array([0, 1]), d.sample_of_cell, d)
    assert shares[0][0] == pytest.approx(0.0)  # leaving any one donor out keeps all of it
    assert shares[1] == (pytest.approx(1.0), "a")  # without a, nothing is left


def test_verdict_names_its_rule():
    v = de_audit.verdict(900, [850, 700, 990], {"n_shared": 2, "n_sig_pseudobulk": 3}, "DESeq2", None)
    assert v.startswith("900 genes at cell level (adj. p < 0.05); 700-990 in 3 donor shuffles, median 850")
    assert "half the observed count or more" in v and "trust the pseudobulk list" in v
    v = de_audit.verdict(1240, [0, 12, 3], {"n_shared": 310, "n_sig_pseudobulk": 320}, "DESeq2", None)
    assert "0-12 in 3 donor shuffles" in v and "310 also in pseudobulk (DESeq2, 320 there)" in v
    assert "under half" in v and v.endswith(".")
    v = de_audit.verdict(40, [], {"n_shared": 0, "n_sig_pseudobulk": 0}, "DESeq2", "too few")
    assert "no donor shuffles (too few donors)" in v and "candidates, not findings" in v
    assert de_audit.verdict(0, [0, 0], None, "DESeq2", None).endswith("nothing to audit.")


# --- synthetic comparisons ------------------------------------------------------------


def test_a_condition_that_is_one_donor_fails_the_audit():
    """Condition A's donor d0 has 20 genes 8x and no gene differs by condition: cell
    level calls them (and the genes d0's library shift moves), every shuffle does too
    (d0 is always on one side), pseudobulk (d0 is one of four) calls nothing."""
    a = synthetic(["d0", "d1", "d2", "d3"], ["d4", "d5", "d6", "d7"], effect=1.0, outlier="d0")
    r = _audit(a)
    assert r.scheme == "donors" and (r.n_donors_a, r.n_donors_b) == (4, 4)
    assert r.observed >= 20 and len(r.null_counts) == 10
    assert np.median(r.null_counts) >= de_audit.NULL_SHARE * r.observed
    assert r.pseudobulk["n_sig_pseudobulk"] == 0 and r.n_cell_only == r.observed
    planted = [g for g in r.cell_only if 20 <= int(g["gene"][1:]) < 40]
    assert len(planted) >= 15 and all(g["top_donor"] == "d0" for g in planted)
    assert np.median([g["top_donor_share"] for g in planted]) > 0.5
    # DESeq2 flags d0 as an outlier sample on these genes: no p-value there.
    assert all(g["pb_padj"] is None for g in planted if not g["pb_tested"])
    assert "most cell-level genes are donor differences: trust the pseudobulk list" in r.verdict
    json.dumps(r.to_dict())  # JSON-safe


def test_a_clean_effect_passes_the_audit():
    """Ten genes 4x apart in each of 6 paired donors: no shuffle finds anything, and
    pseudobulk finds those ten, all also at cell level."""
    donors = [f"d{i}" for i in range(6)]
    a = synthetic(donors, donors, effect=4.0, donor_sd=0.05)
    r = _audit(a)
    assert r.scheme == "paired" and r.exhaustive is True and r.n_possible == 10
    assert r.observed >= 10 and max(r.null_counts) <= 2
    assert r.empirical_p == pytest.approx(1 / 11, abs=1e-4)  # the smallest 10 shuffles allow
    pb = r.pseudobulk
    assert pb["n_sig_pseudobulk"] == pb["n_shared"] == 10
    assert not any(int(g["gene"][1:]) < 10 for g in r.cell_only)
    assert pb["spearman_lfc"] > 0.5
    assert "trust the pseudobulk list; the shuffles' median stays under half" in r.verdict
    assert r.qq["observed"][0] > r.qq["permuted"][0]


def test_too_few_donors_says_why_and_still_compares_with_pseudobulk():
    a = synthetic(["d0", "d1"], ["d0", "d1"], effect=4.0)
    r = _audit(a)
    assert r.null_counts == [] and r.empirical_p is None and r.qq["permuted"] is None
    assert "at least 3 donors on each side ('donor': 2 and 2)" in r.permutation_note
    assert r.pseudobulk is not None and "no donor shuffles (too few donors)" in r.verdict


def test_pseudobulk_failure_is_a_note_and_stop_ends_without_a_result():
    a = synthetic(list("abc"), list("def"), cells=5, effect=4.0)  # under 10 cells per sample
    r = _audit(a, k=3)
    assert r.pseudobulk is None and "at least 2 donors in each group" in r.pseudobulk_note
    assert len(r.null_counts) == 3
    in_a = (a.obs["cond"] == "A").to_numpy()
    test, _ = _prepare(a, in_a, ~in_a, 0.01)
    calls = iter([False, True])
    items = list(de_audit.iter_audit(test, in_a, a.obs["donor"].astype(str).to_numpy(),
                                     n_permutations=5, stop=lambda: next(calls, True)))
    assert not any("result" in i for i in items)
    with pytest.raises(de_audit.AuditError):
        de_audit.run(test, in_a, a.obs["donor"].astype(str).to_numpy(), stop=lambda: True)


def test_time_limit_stops_the_shuffles(monkeypatch):
    a = synthetic(list("abcd"), list("efgh"), effect=4.0)
    clock = iter(np.arange(0, 1000, 7.0))
    monkeypatch.setattr(de_audit.time, "perf_counter", lambda: float(next(clock)))
    in_a = (a.obs["cond"] == "A").to_numpy()
    test, _ = _prepare(a, in_a, ~in_a, 0.01)
    r = de_audit.run(test, in_a, a.obs["donor"].astype(str).to_numpy(), n_permutations=10, max_seconds=10)
    assert 0 < len(r.null_counts) < 10 and "s limit" in r.stopped


# --- endpoint, results store, co-pilot ---------------------------------------------------


def _dataset(root: Path, a: ad.AnnData, did: str = "ds") -> None:
    d = root / "uploads" / did
    d.mkdir(parents=True)
    (d / "metadata.json").write_text(json.dumps({"id": did, "name": did, "status": "ready"}))
    a.write_h5ad(d / "data.h5ad")


def _events(text: str) -> list[tuple[str, dict]]:
    out = []
    for block in text.split("\n\n"):
        lines = [x for x in block.splitlines() if x and not x.startswith(":")]
        if len(lines) == 2 and lines[0].startswith("event: "):
            out.append((lines[0][7:], json.loads(lines[1][6:])))
    return out


async def test_audit_endpoint_streams_a_kept_result_off_the_loop(client, dataset_manager, monkeypatch):
    from scview.core import claims, de_methods, results_export, results_store
    from scview.core.assistant import build_grounding_context

    donors = [f"d{i}" for i in range(4)]
    _dataset(Path(dataset_manager.data_dir), synthetic(donors, donors, effect=4.0))
    r = await client.post("/api/v1/datasets/ds/de/compare",
                          json={"split_by": "cond", "group_a": "A", "group_b": "B"})
    assert r.status_code == 200, r.text
    rid = r.json()["result_id"]

    threads: list[threading.Thread] = []
    real = de_methods.run

    def spy(*args, **kwargs):
        threads.append(threading.current_thread())
        return real(*args, **kwargs)

    monkeypatch.setattr(de_methods, "run", spy)
    r = await client.post("/api/v1/datasets/ds/de/audit", json={"result_id": rid, "n_permutations": 5})
    assert r.status_code == 200, r.text
    events = _events(r.text)
    kinds = [e for e, _ in events]
    assert kinds[0] == "start" and kinds[-1] == "complete" and "result" in kinds
    start = events[0][1]
    assert start["donor_key"] == "donor" and start["scheme"] == "paired" and start["n_permutations"] == 3
    steps = [d["step"] for e, d in events if e == "progress"]
    assert steps == ["observed", "shuffle", "shuffle", "shuffle", "pseudobulk"]
    res = next(d for e, d in events if e == "result")
    assert res["label"] == "All cells: A vs B (Wilcoxon)" and res["observed"] >= 10
    assert len(res["null_counts"]) == 3 and res["exhaustive"] is True and res["pseudobulk"]["n_shared"] >= 10
    assert threads and all(t is not MAIN for t in threads)

    adaptor = await dataset_manager.get_or_load_dataset("ds")
    context, sources = build_grounding_context(adaptor, {"groups"})
    assert "## DE audit (last run" in context and "donors from 'donor' (4 vs 4)" in context
    assert any(s.ref == de_audit.TAG for s in sources)
    assert claims.resolve_tag("deaudit", [de_audit.TAG, "result:pseudobulk"]) == [de_audit.TAG]

    results_store.flush()
    entries = results_store.entries("ds", de_audit.KIND)
    assert len(entries) == 1 and entries[0]["label"] == "DE audit: All cells: A vs B (Wilcoxon)"
    _, value = results_store.load("ds", entries[0]["id"])
    df = results_export.table(entries[0], value)
    assert list(df.columns)[:3] == ["gene", "cell_lfc", "cell_padj"] or df.empty
    # The in-memory copy gone (a restart): the co-pilot reads the stored one.
    with de_audit._last_lock:
        de_audit._last.clear()
    assert de_audit.last_result(adaptor).observed == res["observed"]


async def test_audit_endpoint_by_scope_and_its_refusals(client, dataset_manager):
    donors = [f"d{i}" for i in range(4)]
    a = synthetic(donors, donors, effect=4.0)
    a.obs["sample"] = a.obs["cond"]
    _dataset(Path(dataset_manager.data_dir), a)
    body = {"split_by": "cond", "group_a": "A", "group_b": "B", "within_column": "ct",
            "within_value": "X", "n_permutations": 2, "method": "t-test"}
    r = await client.post("/api/v1/datasets/ds/de/audit", json=body)
    assert r.status_code == 200
    res = next(d for e, d in _events(r.text) if e == "result")
    assert res["label"] == "X: A vs B (t-test)" and res["method"] == "t-test" and len(res["null_counts"]) == 2

    r = await client.post("/api/v1/datasets/ds/de/audit", json={"split_by": "cond"})
    assert r.status_code == 400
    r = await client.post("/api/v1/datasets/ds/de/audit", json={**body, "replicate_key": "cond"})
    assert r.status_code == 400 and "must differ" in r.json()["detail"]
    r = await client.post("/api/v1/datasets/ds/de/audit", json={"result_id": "f" * 32})
    assert r.status_code == 404
    r = await client.post("/api/v1/datasets/ds/de", json={"indices": list(range(50))})
    r = await client.post("/api/v1/datasets/ds/de/audit", json={"result_id": r.json()["result_id"]})
    assert r.status_code == 400 and "between-groups" in r.json()["detail"]
    r = await client.post("/api/v1/datasets/ds/de/compare",
                          json={"split_by": "cond", "group_a": "A", "group_b": "B", "method": "logreg"})
    r = await client.post("/api/v1/datasets/ds/de/audit", json={"result_id": r.json()["result_id"]})
    assert r.status_code == 400 and "no p-values" in r.json()["detail"]


async def test_audit_needs_a_donor_column(client, dataset_manager):
    a = synthetic(["d0", "d1", "d2"], ["d0", "d1", "d2"], effect=4.0)
    a.obs = a.obs.rename(columns={"donor": "who"})
    _dataset(Path(dataset_manager.data_dir), a)
    r = await client.post("/api/v1/datasets/ds/de/audit", json={"split_by": "cond", "group_a": "A",
                                                                  "group_b": "B"})
    assert r.status_code == 400 and "donor or sample column" in r.json()["detail"]
    r = await client.post("/api/v1/datasets/ds/de/audit",
                          json={"split_by": "cond", "group_a": "A", "group_b": "B", "replicate_key": "who",
                                "n_permutations": 1})
    assert r.status_code == 200 and "result" in [e for e, _ in _events(r.text)]


async def test_a_large_dataset_is_refused(client, tmp_data_dir):
    from tests.large_fixture import register

    register(tmp_data_dir, "big")
    r = await client.post("/api/v1/datasets/big/de/audit",
                          json={"split_by": "condition", "group_a": "stim", "group_b": "ctrl"})
    assert r.status_code == 422 and "DE audit" in r.json()["detail"]
