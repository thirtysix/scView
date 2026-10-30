"""Learned gates (core/learned_gate.py): training on a planted programme, the negatives
choice under a planted condition shift, refusals, orthologs, the gate file."""

from __future__ import annotations

import json

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

from scview.core import learned_gate as lg
from scview.core.label_transfer import lt_lognorm

N_GENES = 300
PROGRAM = {"A": range(0, 10), "B": range(10, 20), "C": range(20, 30)}
ISG = range(100, 120)


def _cells(seed: int = 0, per: int = 40, stim_factor: float = 10.0, prefix: str = "c") -> ad.AnnData:
    """4 donors x CTRL/STIM, types A/B/C with 10 programme genes each over a Poisson
    background; STIM multiplies 20 'interferon' genes in every cell."""
    rng = np.random.default_rng(seed)
    rows, obs = [], []
    for d in range(4):
        for cond in ("CTRL", "STIM"):
            for t, genes in PROGRAM.items():
                lam = np.full(N_GENES, 0.3)
                lam[list(genes)] = 5.0
                if cond == "STIM":
                    lam[list(ISG)] *= stim_factor
                size = rng.uniform(0.6, 1.6, size=(per, 1))
                rows.append(rng.poisson(lam * size))
                obs += [{"donor": f"d{d}", "condition": cond, "cluster": t}] * per
    a = ad.AnnData(sp.csr_matrix(np.vstack(rows).astype(np.float32)))
    a.var_names = [f"G{i}" for i in range(N_GENES)]
    a.obs_names = [f"{prefix}{i}" for i in range(a.n_obs)]
    o = pd.DataFrame(obs, index=a.obs_names)
    for c in o.columns:
        a.obs[c] = pd.Categorical(o[c])
    return a


def _expr(a, rows=None):  # every cell's values: prepare takes the training rows
    return lt_lognorm(a.X), a.var.copy(), "counts"


def _sel(a, mask) -> dict:
    return {"n_cells": int(a.n_obs), "indices": [int(i) for i in np.flatnonzero(mask)]}


def _train(a, mask, **kw):
    inp = lg.prepare(a, _sel(a, mask), expression=_expr, **kw)
    return lg.train(inp, name="A cells")


def _apply(a, gate, **kw):
    X, var, src = _expr(a)
    return lg.apply(a, gate, X=X, var=var, source=src, dataset_id="ds", **kw)


def _pr(pred, truth):
    tp = int((pred & truth).sum())
    return tp / max(int(pred.sum()), 1), tp / max(int(truth.sum()), 1)


def test_a_gate_on_a_planted_programme_recovers_it():
    a = _cells()
    is_a = (a.obs["cluster"] == "A").to_numpy()
    ctrl = (a.obs["condition"] == "CTRL").to_numpy()
    gate, report = _train(a, is_a & ctrl)
    assert report["verdict"]["level"] == "good"
    assert gate.cv.auroc > 0.99 and gate.cv.f1 > 0.95
    assert report["sample_column"] == "donor" and report["condition_column"] == "condition"
    # negatives come from the selection's own samples: CTRL libraries only
    assert all(s.endswith("CTRL") for s in report["samples"]) and len(report["samples"]) == 4
    assert gate.cv.grouped and gate.cv.folds >= 3
    assert len(gate.genes) <= lg.MAX_GENES
    planted = {f"G{i}" for i in PROGRAM["A"]}
    assert {g.gene for g in gate.top_for[:5]} <= planted
    assert all(g.weight < 0 for g in gate.top_against)
    assert 0 < gate.threshold < 1
    assert report["disagree"]["cells"]["n_cells"] == a.n_obs


def test_negatives_from_own_samples_beat_all_cells_under_a_condition_shift():
    a = _cells()
    is_a = (a.obs["cluster"] == "A").to_numpy()
    ctrl = (a.obs["condition"] == "CTRL").to_numpy()
    stim = ~ctrl
    own, _ = _train(a, is_a & ctrl)
    every, rep_all = _train(a, is_a & ctrl, negatives="all")
    assert rep_all["samples"] == []
    b = a.copy()
    _apply(b, own)
    col, _ = lg.column_names(own.name)
    pred_own = (b.obs[col] == "in").to_numpy()
    c = a.copy()
    _apply(c, every)
    pred_all = (c.obs[col] == "in").to_numpy()
    p_own, r_own = _pr(pred_own[stim], is_a[stim])
    p_all, r_all = _pr(pred_all[stim], is_a[stim])
    assert p_own >= 0.9 and r_own >= 0.9
    assert r_all < r_own - 0.3  # STIM A cells were negatives: interferon genes count against
    assert any(g.gene in {f"G{i}" for i in ISG} for g in every.top_against)


def test_apply_writes_columns_counts_per_sample_and_records_the_step():
    from scview.core import provenance

    a = _cells()
    gate, _ = _train(a, ((a.obs["cluster"] == "A") & (a.obs["condition"] == "CTRL")).to_numpy())
    res = _apply(a, gate)
    col, score_col = lg.column_names(gate.name)
    assert col == "gate_a_cells" and res["column"] == col
    assert list(a.obs[col].cat.categories) == ["in", "out"]
    assert a.obs[score_col].dtype == np.float32
    assert res["n_in"] == int((a.obs[col] == "in").sum())
    assert res["sample_column"] == "donor" and len(res["per_sample"]) == 4
    assert sum(r["n_cells"] for r in res["per_sample"]) == a.n_obs
    assert res["coverage"]["gene_share"] == 1.0 and not res["replaced"]
    steps = [h["step"] for h in provenance.read_provenance(a)["history"]]
    assert steps[-1] == lg.STEP_APPLY
    again = _apply(a, gate)
    assert again["replaced"]
    # a column of the same name that no gate wrote is not overwritten
    b = _cells()
    b.obs[col] = "x"
    with pytest.raises(lg.LearnedGateError) as e:
        _apply(b, gate)
    assert e.value.status == 409


def test_refusals_tiny_selection_whole_dataset_and_no_contrast():
    a = _cells()
    with pytest.raises(lg.LearnedGateError, match="at least 20"):
        lg.prepare(a, _sel(a, np.arange(a.n_obs) < 10), expression=_expr)
    with pytest.raises(lg.LearnedGateError, match="at most 90%"):
        lg.prepare(a, _sel(a, np.arange(a.n_obs) < a.n_obs - 40), expression=_expr, sample_column="")
    with pytest.raises(lg.LearnedGateError, match="now has"):
        lg.prepare(a, {"n_cells": 5, "indices": [1]}, expression=_expr)
    small = lg.prepare(a, _sel(a, np.arange(a.n_obs) < 30), expression=_expr, sample_column="")
    assert small.warnings and "uncertain" in small.warnings[0]


def test_excluded_genes_ribo_mito_hb_and_cell_cycle_toggle():
    names = pd.Index(["RPL3", "MT-CO1", "HBB", "MKI67", "FOS", "CD3E"])
    assert lg.excluded_genes(names, "human", True).tolist() == [True, True, True, True, True, False]
    assert lg.excluded_genes(names, "human", False).tolist() == [True, True, True, False, False, False]


def test_low_overlap_is_refused_and_partial_overlap_warns():
    a = _cells()
    gate, _ = _train(a, ((a.obs["cluster"] == "A") & (a.obs["condition"] == "CTRL")).to_numpy())
    b = _cells(seed=4)
    w = np.abs(np.asarray(gate.coef))
    top = set(np.asarray(gate.genes)[np.argsort(-w)][: max(1, len(gate.genes) // 2 + 1)])
    b.var_names = [f"X{g}" if g in top else g for g in b.var_names]
    rep = lg.check(gate, b, dataset_id="other")
    assert not rep["ok"] and "carry" in rep["refusal"]
    with pytest.raises(lg.LearnedGateError):
        _apply(b, gate)
    c = _cells(seed=5)
    smallest = np.asarray(gate.genes)[np.argsort(w)][0]
    c.var_names = [f"X{g}" if g == smallest else g for g in c.var_names]
    rep = lg.check(gate, c, dataset_id="other")
    assert rep["ok"] and any("missing" in x for x in rep["warnings"])


def test_a_shift_warning_when_scores_bunch_in_the_middle():
    a = _cells()
    gate, _ = _train(a, ((a.obs["cluster"] == "A") & (a.obs["condition"] == "CTRL")).to_numpy())
    assert not any("undecided" in w for w in _apply(a.copy(), gate)["warnings"])
    # a gate whose scores sit near 0.5 everywhere, trained with none in the middle
    flat = _gate(["G1", "G2"]).model_copy(update={"coef": [0.01, 0.01], "intercept": 0.0})
    res = _apply(_cells(seed=8), flat)
    assert res["mid_share"] > 0.9 and any("undecided" in w for w in res["warnings"])


def test_isg_heavy_gates_warn_on_other_data():
    a = _cells()
    gate, _ = _train(a, ((a.obs["cluster"] == "A") & (a.obs["condition"] == "CTRL")).to_numpy())
    isg = set(gate.genes[:3])
    share = lg.isg_share(gate, isg)
    assert share is not None and share > 0
    rep = lg.check(gate.model_copy(update={"isg_weight_share": 0.5}), a, dataset_id="other")
    assert any("interferon" in w for w in rep["warnings"])
    rep = lg.check(gate.model_copy(update={"isg_weight_share": 0.1}), a, dataset_id="other")
    assert not any("interferon" in w for w in rep["warnings"])


def _gate(genes, species="human", **kw) -> lg.LearnedGate:
    n = len(genes)
    return lg.LearnedGate(
        format=lg.FORMAT, version=1, id="g1", name="T cells", species=species, genes=list(genes),
        coef=[1.0] * n, intercept=-1.0, mean=[0.5] * n, std=[1.0] * n, threshold=0.5,
        cv=lg.GateCV(auroc=0.99, precision=0.95, recall=0.95, f1=0.95, n_pos=100, n_neg=200, folds=5, C=0.1),
        **kw)


def test_ortholog_mapping_is_one_to_one_only():
    table = pd.DataFrame({
        "human": ["AAA1", "BBB2", "BBB2", "CCC3", "DDD4", "EEE5"],
        "mouse": ["Aaa1", "Bbb2", "Bbb2b", "Ccc3", "Ccc3", "Eee5"],
        "evidence": [5, 5, 5, 5, 5, 1],
    })
    gate = _gate(["AAA1", "BBB2", "CCC3", "DDD4", "EEE5"])
    var = pd.DataFrame(index=["Aaa1", "Bbb2", "Ccc3", "Eee5", "Zzz9"])
    gm = lg.map_genes(gate, var, "mouse", ortholog_table=table)
    # one-to-many (BBB2), two gate genes on one (CCC3, DDD4), low evidence (EEE5): missing
    assert gm.target == ["Aaa1", None, None, None, None]
    assert gm.via.startswith("one-to-one orthologs, human -> mouse")
    same = lg.map_genes(gate, pd.DataFrame(index=["AAA1", "CCC3"]), "human", ortholog_table=table)
    assert same.target == ["AAA1", None, "CCC3", None, None] and same.via == "name"


def test_ortholog_apply_scores_a_mouse_dataset():
    rng = np.random.default_rng(1)
    human = [f"HG{i}" for i in range(60)]
    mouse = [f"Hg{i}" for i in range(60)]
    table = pd.DataFrame({"human": human, "mouse": mouse, "evidence": 5})
    gate = _gate(human[:10])
    b = ad.AnnData(sp.csr_matrix(rng.poisson(1.0, size=(40, 60)).astype(np.float32)))
    b.var_names = mouse
    rep = lg.check(gate, b, dataset_id="m", ortholog_table=table)
    assert rep["species"] == "mouse" and rep["ok"], rep
    assert rep["coverage"]["n_found"] == 10
    assert any("orthologs" in w for w in rep["warnings"])
    X, var, src = _expr(b)
    res = lg.apply(b, gate, X=X, var=var, source=src, dataset_id="m", ortholog_table=table)
    assert res["n_cells"] == 40 and res["via"].startswith("one-to-one")


def test_missing_genes_take_the_training_mean():
    gate = _gate(["G1", "G2"])
    X = np.array([[np.nan, 0.5], [0.5, 0.5]], dtype=np.float32)
    s, _ = lg.lg_apply(X, lg.gate_dict(gate))
    assert s[0] == pytest.approx(s[1])


def test_gate_json_round_trip_and_import_caps(tmp_path):
    a = _cells()
    gate, _ = _train(a, ((a.obs["cluster"] == "A") & (a.obs["condition"] == "CTRL")).to_numpy())
    lg.save_gate(tmp_path, gate)
    back = lg.load_gate(tmp_path, gate.id)
    assert back.model_dump() == gate.model_dump()
    text = (tmp_path / "learned_gates" / f"{gate.id}.json").read_text()
    assert len(text) < 20_000 and json.loads(text)["format"] == lg.FORMAT
    assert [g["id"] for g in lg.list_gates(tmp_path)] == [gate.id]
    # the same file imported again keeps its id; other content under that id gets a new one
    assert lg.import_gate(tmp_path, text).id == gate.id
    other = json.loads(text)
    other["name"] = "renamed"
    assert lg.import_gate(tmp_path, other).id != gate.id
    base = json.loads(text)
    bad = [
        ({**base, "format": "other"}, "format"),
        ({**base, "version": 2}, "version"),
        ({**base, "coef": base["coef"][:-1]}, "equal lengths"),
        ({**base, "std": [0.0] * len(base["std"])}, "positive"),
        ({**base, "threshold": 1.5}, "threshold"),
        ({**base, "id": "../x"}, "id"),
        ({**base, "genes": [f"G{i}" for i in range(201)], "coef": [1.0] * 201, "mean": [0.0] * 201,
          "std": [1.0] * 201}, "valid"),
    ]
    for doc, why in bad:
        with pytest.raises(lg.LearnedGateError) as e:
            lg.parse_gate(doc)
        assert why in str(e.value) or "valid" in str(e.value), (why, str(e.value))
    nan_doc = text.replace(f'"intercept": {json.dumps(base["intercept"])}', '"intercept": NaN')
    with pytest.raises(lg.LearnedGateError):
        lg.parse_gate(nan_doc)
    with pytest.raises(lg.LearnedGateError) as e:
        lg.parse_gate(b" " * (lg.MAX_FILE_BYTES + 1))
    assert e.value.status == 413
    with pytest.raises(lg.LearnedGateError):
        lg.parse_gate("not json")
    assert lg.delete_gate(tmp_path, gate.id) and not lg.delete_gate(tmp_path, gate.id)
    with pytest.raises(lg.LearnedGateError) as e:
        lg.load_gate(tmp_path, "../../etc")
    assert e.value.status == 404


def test_cluster_overlap_note():
    a = _cells()
    is_a = (a.obs["cluster"] == "A").to_numpy()
    full = lg.cluster_overlap(a.obs, np.flatnonzero(is_a), a.n_obs)
    assert full["is_cluster"] and full["group"] == "A" and "reference mapping" in full["note"]
    half = lg.cluster_overlap(a.obs, np.flatnonzero(is_a & (a.obs["condition"] == "CTRL").to_numpy()), a.n_obs)
    assert not half["is_cluster"] and half["note"] == ""


def test_training_stops_when_asked():
    a = _cells()
    inp = lg.prepare(a, _sel(a, ((a.obs["cluster"] == "A") & (a.obs["condition"] == "CTRL")).to_numpy()),
                     expression=_expr)
    calls = {"n": 0}

    def stop():
        calls["n"] += 1
        return calls["n"] > 3

    gen = lg.iter_train(inp, name="x", stop=stop)
    with pytest.raises(lg.LearnedGateCancelled):
        for _ in gen:
            pass


def test_replay_functions_match_the_app():
    a = _cells()
    gate, _ = _train(a, ((a.obs["cluster"] == "A") & (a.obs["condition"] == "CTRL")).to_numpy())
    X, var, src = _expr(a)
    gm = lg.map_genes(gate, var, "unknown")
    s_app, in_app = lg.score_cells(gate, X, pd.Index(var.index), gm)
    Xr = lg.lg_expression(a, gate.genes)
    s_rep, in_rep = lg.lg_apply(Xr, lg.gate_dict(gate))
    np.testing.assert_allclose(s_app, s_rep, atol=1e-5)
    assert (in_app == in_rep).all()
    assert np.isnan(lg.lg_expression(a, ["G1", "nope"])[:, 1]).all()


def test_grounding_lines_from_the_last_apply(monkeypatch):
    res = {"name": "A cells", "n_in": 10, "n_cells": 100, "share": 0.1, "column": "gate_a_cells",
           "cv": {"auroc": 0.99, "f1": 0.95}, "verdict": {"level": "good"}, "source": {"name": "Kang"},
           "coverage": {"n_found": 9, "n_genes": 10, "weight_share": 0.97}, "via": "name",
           "per_sample": [{"sample": "d1", "share": 0.1}], "sample_column": "donor", "warnings": ["w"]}
    monkeypatch.setattr(lg, "last_apply", lambda adaptor: res)
    lines, detail = lg.grounding_lines(object())
    text = "\n".join(lines)
    assert lg.TAG in text and "10 of 100" in detail and "d1 10.0%" in text and "not a frequency" in text
    monkeypatch.setattr(lg, "last_apply", lambda adaptor: None)
    assert lg.grounding_lines(object()) == ([], None)


def test_the_replay_script_applies_the_gate_inline(tmp_path):
    import subprocess
    import sys

    from scview.core import replay_script

    a = _cells()
    inp = a.copy()
    gate, report = _train(a, ((a.obs["cluster"] == "A") & (a.obs["condition"] == "CTRL")).to_numpy())
    lg.record_training(a, gate, report)
    res = _apply(a, gate)
    script = replay_script.build_script(a, dataset_id="t", dataset_name="test")
    assert "LEARNED_GATE = " in script and "def lg_apply" in script and "import scview" not in script
    statuses = [s["status"] for s in replay_script.describe(a)[-2:]]
    assert statuses == ["recorded", "exact"]
    src, path, dst = tmp_path / "in.h5ad", tmp_path / "replay.py", tmp_path / "out.h5ad"
    inp.write_h5ad(src)
    path.write_text(script)
    proc = subprocess.run([sys.executable, str(path), str(src), str(dst)], capture_output=True, text=True,
                          timeout=300)
    assert proc.returncode == 0, proc.stderr[-3000:]
    out = ad.read_h5ad(dst)
    assert (out.obs[res["column"]].astype(str).to_numpy() == a.obs[res["column"]].astype(str).to_numpy()).all()
    np.testing.assert_allclose(out.obs[res["score_column"]].to_numpy(), a.obs[res["score_column"]].to_numpy(),
                               atol=1e-5)
