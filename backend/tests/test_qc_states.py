"""QC states: a low-RP / high-mito second state per cell type (core/qc_states.py)."""

from __future__ import annotations

import json
import threading
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest

from scview.core import provenance, qc_states, replay_script
from scview.core.anndata_adaptor import per_cell_qc

API = "/api/v1"


def _logit_normal(rng, n: int, mean: float, sd: float) -> np.ndarray:
    z = rng.normal(np.log(mean / (1 - mean)), sd, n)
    return 1 / (1 + np.exp(-z))


def _cells(rng, n: int, rp: float, mt: float, sd: float = 0.25) -> tuple[np.ndarray, np.ndarray]:
    """(RP share of non-mito counts, mito fraction) for ``n`` cells of one state."""
    return _logit_normal(rng, n, rp, sd), _logit_normal(rng, n, mt, sd)


def _adata(groups: dict[str, tuple[np.ndarray, np.ndarray]], *, mito: bool = True,
           condition: dict[str, np.ndarray] | None = None, seed: int = 0) -> ad.AnnData:
    """Cells with QC metrics in obs (what per_cell_qc reads), one group per key."""
    rng = np.random.default_rng(seed)
    rp = np.concatenate([g[0] for g in groups.values()])
    mt = np.concatenate([g[1] for g in groups.values()])
    labels = np.concatenate([[k] * g[0].size for k, g in groups.items()])
    n = rp.size
    tc = rng.integers(3000, 8000, n).astype(np.float64)
    if not mito:
        mt = np.zeros(n)
    obs = pd.DataFrame({
        "cell_type": pd.Categorical(labels),
        "total_counts": tc,
        "n_genes_by_counts": tc / 3,
        "pct_counts_mt": mt * 100,
        "pct_counts_ribo": rp * (1 - mt) * 100,  # of all counts
    }, index=[f"c{i}" for i in range(n)])
    if condition is not None:
        obs["treatment"] = pd.Categorical(np.concatenate([condition[k] for k in groups]))
    genes = ["RPL3", "RPS6", "ACTB", "GAPDH"] + (["MT-CO1"] if mito else [])
    X = rng.poisson(1.0, (n, len(genes))).astype(np.float32)
    return ad.AnnData(X=X, obs=obs, var=pd.DataFrame(index=genes))


def _two_state(rng, n: int = 1500, low: float = 0.3) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Main state (30% RP, 4% mito) + a low-RP / high-mito state (10% RP, 20% mito);
    returns rp, mt and the truth (True = low state)."""
    k = int(n * low)
    rp_a, mt_a = _cells(rng, n - k, 0.30, 0.04)
    rp_b, mt_b = _cells(rng, k, 0.10, 0.20)
    truth = np.r_[np.zeros(n - k, bool), np.ones(k, bool)]
    return np.r_[rp_a, rp_b], np.r_[mt_a, mt_b], truth


class _Loaded:
    """A loaded dataset without the adaptor's per-cell QC cache (weak-referenceable,
    as the adaptor is)."""

    def __init__(self, adata: ad.AnnData) -> None:
        self.adata = adata


def _run(a: ad.AnnData, column: str = "cell_type") -> tuple[dict, np.ndarray]:
    return qc_states.compute(a, column, per_cell_qc(a, ribo=True))


def _group(result: dict, name: str) -> dict:
    return next(g for g in result["groups"] if g["name"] == name)


# --- the fit ----------------------------------------------------------------


def test_two_states_are_recovered_with_their_share_and_per_condition():
    rng = np.random.default_rng(1)
    rp, mt, truth = _two_state(rng)
    # The low state is 15% of control cells and 45% of drug cells.
    cond = np.where(truth, np.where(rng.random(truth.size) < 0.75, "drug", "ctrl"),
                    np.where(rng.random(truth.size) < 0.36, "drug", "ctrl"))
    a = _adata({"LNCaP": (rp, mt)}, condition={"LNCaP": cond})
    result, states = _run(a)
    g = _group(result, "LNCaP")
    assert result["fit"] == "rp_mito" and result["has_mt"] and not result["note"]
    assert g["n_states"] == 2 and g["flagged_label"] == qc_states.LOW_RP_HIGH_MT
    assert g["delta_bic"] > 100 and g["separation"] >= 2
    assert g["flagged_share"] == pytest.approx(0.3, abs=0.03)
    main, low = g["components"]
    assert (main["label"], low["label"]) == ("main", qc_states.LOW_RP_HIGH_MT)
    assert low["mean_rp_share"] < main["mean_rp_share"] and low["mean_pct_mt"] > main["mean_pct_mt"]
    assert main["n_cells"] + low["n_cells"] == 1500
    # Each cell's state matches the truth almost everywhere.
    assert ((states == 1) == truth).mean() > 0.97
    shares = {k: v["share"] for k, v in g["per_condition"].items()}
    expected = {lvl: truth[cond == lvl].mean() for lvl in ("ctrl", "drug")}
    assert shares["ctrl"] == pytest.approx(expected["ctrl"], abs=0.03)
    assert shares["drug"] == pytest.approx(expected["drug"], abs=0.03)
    assert result["condition"] == {"column": "treatment", "kind": "condition", "levels": ["ctrl", "drug"]}
    assert "Two states" in g["reason"] and "D = " in g["reason"]


def test_one_state_stays_one_even_when_skewed():
    rng = np.random.default_rng(2)
    n = 3000
    gaussian = _cells(rng, n, 0.25, 0.05)
    # Skewed shares: beta RP share and gamma mito, one population.
    skewed = (rng.beta(6, 18, n), np.clip(rng.gamma(2.0, 0.02, n), 0.001, 0.6))
    result, states = _run(_adata({"G": gaussian, "S": skewed}))
    for name in ("G", "S"):
        g = _group(result, name)
        assert g["fitted"] and g["n_states"] == 1, g
        assert [c["label"] for c in g["components"]] == ["main"]
    assert result["two_state_groups"] == [] and result["overall"]["n_flagged"] == 0
    assert set(np.unique(states)) == {0}


def test_overlapping_components_that_the_bic_favours_are_one_broad_state():
    rng = np.random.default_rng(3)
    n = 6000
    # Heavy tails (t, 3 df) on the logit scale: a second Gaussian improves the BIC
    # a lot, but the two overlap.
    z_rp = np.log(0.25 / 0.75) + 0.3 * rng.standard_t(3, n)
    z_mt = np.log(0.05 / 0.95) + 0.3 * rng.standard_t(3, n)
    tailed = (1 / (1 + np.exp(-z_rp)), 1 / (1 + np.exp(-z_mt)))
    g = _group(_run(_adata({"T": tailed}))[0], "T")
    assert g["delta_bic"] > qc_states.MIN_DELTA_BIC and g["separation"] < qc_states.MIN_SEPARATION
    assert g["n_states"] == 1 and "One broad state" in g["reason"]


def test_small_groups_are_not_fitted_and_say_why():
    rng = np.random.default_rng(4)
    rp, mt, _ = _two_state(rng, n=1000)
    tiny = _two_state(rng, n=60)
    result, states = _run(_adata({"big": (rp, mt), "tiny": tiny[:2]}))
    t = _group(result, "tiny")
    assert not t["fitted"] and t["n_states"] == 1 and t["delta_bic"] is None
    assert "under 100" in t["reason"]
    assert (states[-60:] == -1).all()
    assert result["overall"]["n_fitted"] == 1000


def test_without_mito_the_fit_is_on_rp_share_alone():
    rng = np.random.default_rng(5)
    a, b = _cells(rng, 900, 0.30, 0.05)[0], _cells(rng, 600, 0.08, 0.05)[0]
    a_ = _adata({"Mono": (np.r_[a, b], np.zeros(1500))}, mito=False)
    cells = per_cell_qc(a_, ribo=True)
    assert cells["has_mt"] is False  # all-zero mito and no MT- genes: unmeasurable
    result, _ = qc_states.compute(a_, "cell_type", cells)
    g = _group(result, "Mono")
    assert result["fit"] == "rp_only" and not result["has_mt"]
    assert "can't be measured" in result["note"]
    assert g["n_states"] == 2 and g["flagged_label"] == qc_states.LOW_RP
    assert g["flagged_share"] == pytest.approx(0.4, abs=0.03)
    assert all(c["mean_pct_mt"] is None for c in g["components"])
    assert result["scatter"]["y"] is None and result["scatter"]["y_label"] is None
    lines, _ = qc_states.grounding_lines(result)
    assert "no MT- genes" in lines[0] and "low-RP/high-mito" not in "\n".join(lines)


def test_a_split_without_more_mito_is_a_second_state_not_low_rp_high_mito():
    rng = np.random.default_rng(6)
    a = _cells(rng, 1000, 0.30, 0.15)
    b = _cells(rng, 400, 0.10, 0.03)  # lower RP AND lower mito
    g = _group(_run(_adata({"X": (np.r_[a[0], b[0]], np.r_[a[1], b[1]])}))[0], "X")
    assert g["n_states"] == 2 and g["flagged_label"] == qc_states.OTHER
    assert g["flagged_share"] == pytest.approx(400 / 1400, abs=0.03)


def test_rp_share_is_of_the_non_mito_counts():
    a = _adata({"A": (np.full(200, 0.30), np.full(200, 0.40))})
    f = qc_states.features(per_cell_qc(a, ribo=True))
    # 30% of non-mito = 18% of all counts at 40% mito; the share is read back as 30%.
    np.testing.assert_allclose(f["rp_pct"], 30.0, rtol=1e-9)
    np.testing.assert_allclose(f["mt_pct"], 40.0)


def test_no_ribosomal_share_is_refused():
    with pytest.raises(qc_states.QcStatesError, match="ribosomal"):
        qc_states.features({"total_counts": np.ones(3), "pct_mt": np.zeros(3), "pct_ribo": None})


def test_the_scatter_is_capped_per_group_and_seeded(monkeypatch):
    monkeypatch.setattr(qc_states, "N_SCATTER", 300)
    rng = np.random.default_rng(7)
    rp, mt, _ = _two_state(rng, n=1200)
    a = _adata({"A": (rp[:600], mt[:600]), "B": (rp[600:], mt[600:])})
    r1, _ = _run(a)
    r2, _ = _run(a)
    sc = r1["scatter"]
    assert sc["n_shown"] == 600 and np.bincount(sc["group"]).tolist() == [300, 300]
    assert len(sc["x"]) == len(sc["y"]) == len(sc["state"]) == 600
    assert sc == r2["scatter"]


def test_too_many_groups_and_a_missing_column_are_refused():
    rng = np.random.default_rng(8)
    groups = {f"g{i}": _cells(rng, 3, 0.3, 0.05) for i in range(qc_states.MAX_GROUPS + 1)}
    a = _adata(groups)
    with pytest.raises(qc_states.QcStatesError, match="groups"):
        _run(a)
    with pytest.raises(KeyError):
        _run(a, "nope")


def test_the_default_column_is_the_cell_type_then_the_clustering():
    rng = np.random.default_rng(9)
    a = _adata({"A": _cells(rng, 50, 0.3, 0.05), "B": _cells(rng, 50, 0.3, 0.05)})
    assert qc_states.default_column(a) == ("cell_type", "celltype")
    a.obs = a.obs.rename(columns={"cell_type": "leiden"})
    assert qc_states.default_column(a) == ("leiden", "clustering")


def test_the_result_is_cached_until_the_column_changes():
    rng = np.random.default_rng(10)
    rp, mt, _ = _two_state(rng, n=600)
    a = _adata({"A": (rp[:300], mt[:300]), "B": (rp[300:], mt[300:])})
    adaptor = _Loaded(a)  # computes its own per-cell QC
    calls = []
    real = qc_states.compute

    def spy(*args, **kwargs):
        calls.append(1)
        return real(*args, **kwargs)

    qc_states.compute = spy
    try:
        r1, _ = qc_states.for_adaptor(adaptor, "cell_type")
        r2, _ = qc_states.for_adaptor(adaptor, None)  # the default is the same column
        a.obs["cell_type"] = pd.Categorical(np.where(np.arange(600) < 200, "X", "Y"))
        r3, _ = qc_states.for_adaptor(adaptor, "cell_type")
    finally:
        qc_states.compute = real
    assert len(calls) == 2  # the default is the cached column; a relabel refits
    assert r1["reason"] == "requested" and r2["reason"] == "celltype"
    assert r2["groups"] == r1["groups"]
    assert [g["name"] for g in r3["groups"]] == ["X", "Y"]
    assert qc_states.cached(adaptor, "cell_type") is r3


# --- the label --------------------------------------------------------------


def test_label_writes_main_and_the_flagged_state():
    rng = np.random.default_rng(11)
    rp, mt, truth = _two_state(rng, n=1000)
    a = _adata({"LNCaP": (rp, mt), "CAF": _cells(rng, 800, 0.25, 0.05), "tiny": _cells(rng, 30, 0.1, 0.3)})
    result, states = _run(a)
    name = qc_states.check_name(a, "cell_type", None)
    saved = qc_states.label(a, result, states, name)
    col = a.obs["qc_state"]
    assert name == "qc_state" and list(col.cat.categories) == ["main", qc_states.LOW_RP_HIGH_MT]
    assert saved["counts"]["main"] + saved["counts"][qc_states.LOW_RP_HIGH_MT] == a.n_obs
    lncap = (a.obs["cell_type"] == "LNCaP").to_numpy()
    assert ((col[lncap] == qc_states.LOW_RP_HIGH_MT).to_numpy() == truth).mean() > 0.97
    assert (col[~lncap] == "main").all()  # one-state and unfitted groups
    assert qc_states.written_columns(a) == ["qc_state"]
    step = provenance.read_provenance(a)["history"][-1]
    assert step["step"] == "qc_state_label" and step["params"]["two_state_groups"] == ["LNCaP"]
    # Writing again replaces the column it wrote; a foreign column is protected.
    qc_states.check_name(a, "cell_type", "qc_state")
    with pytest.raises(qc_states.QcStatesError) as e:
        qc_states.check_name(a, "cell_type", "total_counts")
    assert e.value.status == 409
    for bad in ("cell_type", "scview_x", "1abc", "a/b", ""):
        if bad == "":
            assert qc_states.check_name(a, "cell_type", bad) == qc_states.DEFAULT_NAME
            continue
        with pytest.raises(qc_states.QcStatesError):
            qc_states.check_name(a, "cell_type", bad)


def test_the_label_step_is_not_replayed_but_says_how_to_join_it():
    a = ad.AnnData(np.zeros((2, 2), dtype=np.float32))
    evil = "x\nimport os; os.system('touch /tmp/pwned')\n"
    provenance.record_step(a, step="qc_state_label", tool="t", params={"column": "ct", "output": "qc_state"})
    provenance.record_step(a, step="qc_state_label", tool="t", params={"column": evil, "output": evil})
    script = replay_script.build_script(a)
    compile(script, "replay.py", "exec")
    assert [d["status"] for d in replay_script.describe(a)] == ["not_replayable"] * 2
    assert "meta.loc[adata.obs_names, 'qc_state']" in script
    assert not any(line.lstrip().startswith("import os") for line in script.splitlines())


# --- the co-pilot fact ------------------------------------------------------


def test_the_fact_line_gives_the_share_and_the_conditions():
    rng = np.random.default_rng(12)
    rp, mt, truth = _two_state(rng, n=1200, low=0.4)
    cond = np.where(truth, np.where(rng.random(truth.size) < 0.8, "drug", "control"), "control")
    a = _adata({"LNCaP": (rp, mt), "CAF": _cells(rng, 600, 0.25, 0.05)},
               condition={"LNCaP": cond, "CAF": np.array(["control"] * 600)})
    result, _ = _run(a)
    lines, detail = qc_states.grounding_lines(result)
    text = "\n".join(lines)
    g = _group(result, "LNCaP")
    shares = {k: v["share"] for k, v in g["per_condition"].items()}
    assert f"two QC states (BIC +{g['delta_bic']:,.0f}" in text
    assert f"{g['flagged_share']:.0%} of 1,200 cells low-RP/high-mito" in text
    assert f"{shares['control']:.0%} in control vs {shares['drug']:.0%} in drug (treatment)" in text
    assert qc_states.TAG in text and detail.startswith("cell_type: two QC states in 1 of 2")


def test_the_copilot_uses_the_cache_and_skips_large_uncached_datasets(monkeypatch):
    rng = np.random.default_rng(13)
    a = _adata({"A": _cells(rng, 300, 0.3, 0.05), "B": _cells(rng, 300, 0.3, 0.05)})
    adaptor = _Loaded(a)
    monkeypatch.setattr(qc_states, "FACT_MAX_CELLS", 100)
    assert qc_states.grounding_for(adaptor) == ([], "")
    qc_states.for_adaptor(adaptor, "cell_type")  # the card was opened
    lines, _ = qc_states.grounding_for(adaptor)
    assert "No cell_type group shows a second QC state" in "\n".join(lines)


def test_the_copilot_context_carries_the_fact(tmp_path):
    from scview.core.anndata_adaptor import AnnDataAdaptor
    from scview.core.assistant import build_grounding_context

    rng = np.random.default_rng(14)
    rp, mt, _ = _two_state(rng, n=800)
    p = tmp_path / "d.h5ad"
    _adata({"LNCaP": (rp, mt), "CAF": _cells(rng, 400, 0.25, 0.05)}).write_h5ad(p)
    context, sources = build_grounding_context(AnnDataAdaptor(str(p)))
    assert "QC states per cell_type" in context
    src = next(s for s in sources if s.ref == qc_states.TAG)
    assert "LNCaP" in src.evidence and "low-RP/high-mito" in src.evidence


# --- the endpoints ----------------------------------------------------------


def _register(dm, a: ad.AnnData, did: str = "qcs") -> str:
    d = Path(dm.data_dir) / "uploads" / did
    d.mkdir(parents=True)
    (d / "metadata.json").write_text(json.dumps({"id": did, "name": did, "status": "ready"}))
    a.write_h5ad(d / "data.h5ad")
    return did


def _fixture() -> ad.AnnData:
    rng = np.random.default_rng(15)
    rp, mt, _ = _two_state(rng, n=600)
    return _adata({"LNCaP": (rp, mt), "CAF": _cells(rng, 300, 0.25, 0.05)})


async def test_get_fits_off_the_loop(client, dataset_manager, monkeypatch):
    did = _register(dataset_manager, _fixture())
    threads: list[threading.Thread] = []
    real = qc_states.compute

    def spy(*args, **kwargs):
        threads.append(threading.current_thread())
        return real(*args, **kwargs)

    monkeypatch.setattr(qc_states, "compute", spy)
    r = await client.get(f"{API}/datasets/{did}/qc/states")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["column"] == "cell_type" and body["reason"] == "celltype"
    assert body["two_state_groups"] == ["LNCaP"] and body["fit"] == "rp_mito"
    assert len(threads) == 1 and threads[0] is not threading.main_thread()
    r = await client.get(f"{API}/datasets/{did}/qc/states", params={"column": "nope"})
    assert r.status_code == 404
    r = await client.get(f"{API}/datasets/{did}/qc/states", params={"column": "total_counts"})
    assert r.status_code == 400  # a numeric column has thousands of "groups"
    r = await client.get(f"{API}/datasets/ghost/qc/states")
    assert r.status_code == 404


async def test_post_saves_the_label_and_the_library_lists_it(client, dataset_manager):
    did = _register(dataset_manager, _fixture())
    r = await client.post(f"{API}/datasets/{did}/qc/states/label", json={"column": "cell_type"})
    assert r.status_code == 200, r.text
    saved = r.json()
    assert saved["name"] == "qc_state" and saved["two_state_groups"] == ["LNCaP"]
    derived = Path(dataset_manager.data_dir) / "derived" / did / "data.h5ad"
    on_disk = ad.read_h5ad(derived)
    col = on_disk.obs["qc_state"]
    assert list(col.cat.categories) == ["main", qc_states.LOW_RP_HIGH_MT]
    assert int((col == qc_states.LOW_RP_HIGH_MT).sum()) == saved["counts"][qc_states.LOW_RP_HIGH_MT]
    assert (col[on_disk.obs["cell_type"] == "CAF"] == "main").all()
    info = await client.get(f"{API}/datasets/{did}")
    assert "qc_state" in [c["name"] for c in info.json()["obs_columns"]]
    # Again under the same name: replaced, not refused.
    r = await client.post(f"{API}/datasets/{did}/qc/states/label", json={})
    assert r.status_code == 200, r.text
    r = await client.post(f"{API}/datasets/{did}/qc/states/label", json={"name": "total_counts"})
    assert r.status_code == 409
    r = await client.post(f"{API}/datasets/{did}/qc/states/label", json={"name": "bad/name"})
    assert r.status_code == 400
    r = await client.post(f"{API}/datasets/{did}/qc/states/label", json={"column": "nope"})
    assert r.status_code == 404
