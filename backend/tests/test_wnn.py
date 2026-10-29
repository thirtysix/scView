"""WNN joint RNA + protein neighbours (core/wnn.py)."""

from __future__ import annotations

import json
import math

import anndata as ad
import numpy as np
import pandas as pd
import pytest

from scview.core import protein, provenance
from scview.core import wnn

GROUPS = 4
PER_GROUP = 90
CONTROLS = ["IgG1_control", "IgG2a_control"]


def _cite(n_per: int = PER_GROUP, n_prot: int = 8, seed: int = 0, zero_cells: int = 0) -> ad.AnnData:
    """Four groups: the RNA tells 0+1 from 2 from 3, the proteins tell 0 from 1 from
    2+3, so only the two together tell all four apart."""
    rng = np.random.default_rng(seed)
    n = GROUPS * n_per
    g = np.repeat(np.arange(GROUPS), n_per)
    rna_means = np.zeros((GROUPS, 10))
    rna_means[0, 0] = rna_means[1, 0] = 6.0
    rna_means[2, 1] = 6.0
    rna_means[3, 2] = 6.0
    rna = rna_means[g] + rng.normal(size=(n, 10))
    lam = np.full((n, n_prot), 30.0)
    lam[g == 0, 0:2] = 400.0
    lam[g == 1, 2:4] = 400.0
    counts = rng.poisson(lam).astype(np.float64)
    ctrl = rng.poisson(15.0, size=(n, len(CONTROLS))).astype(np.float64)
    allc = np.concatenate([counts, ctrl], axis=1)
    if zero_cells:
        allc[:zero_cells] = 0
    names = [f"P{i}" for i in range(n_prot)] + CONTROLS
    a = ad.AnnData(X=np.zeros((n, 3), dtype=np.float32),
                   obs=pd.DataFrame({"group": pd.Categorical([f"g{i}" for i in g])},
                                    index=[f"c{i}" for i in range(n)]))
    split = protein.ProteinSplit()
    protein.attach(a, allc, names, split)
    a.obsm["X_pca"] = rna.astype(np.float32)
    a.uns["neighbors"] = {"connectivities_key": "connectivities", "distances_key": "distances",
                          "params": {"n_neighbors": 15, "method": "umap", "use_rep": "X_pca"}}
    return a


def _purity(idx: np.ndarray, labels: np.ndarray) -> float:
    return float((labels[idx] == labels[:, None]).mean())


# ---------------------------------------------------------------------------
# The pieces against literal (loop) versions of Seurat's code
# ---------------------------------------------------------------------------


def _ref_sigma(X, nn, nearest, k):
    """SNN_SmallestNonzero_Dist after ComputeSNN(prune = 0), as the C++ loops it."""
    n, kk = nn.shape
    sets = [set(r) for r in nn]
    out = []
    for i in range(n):
        nonzero = []
        for j in range(n):  # ascending cell index, as Eigen's column iterator
            s = len(sets[i] & sets[j])
            if s:
                nonzero.append((s / (2 * kk - s), j))
        n_i = min(k, len(nonzero))
        nonzero.sort(key=lambda t: t[0])  # stable
        dists = []
        for val, j in nonzero:
            if len(dists) < n_i or val == nonzero[n_i - 1][0]:
                d = float(np.linalg.norm(X[j] - X[i]))
                if nearest[i] > 0:
                    d = max(d - nearest[i], 0.0)
                dists.append(d)
            else:
                break
        if len(dists) > n_i:
            dists.sort(reverse=True)
            out.append(sum(dists[:n_i]) / n_i)
        else:
            out.append(sum(dists) / len(dists))
    return np.array(out)


def _ref_joint(Xs, cands, nearests, sigmas, weights, k):
    """MultiModalNN: union of the candidates (minus the cell), weighted kernel, top k."""
    n = cands[0].shape[0]
    idx, dist = [], []
    for i in range(n):
        union: list[int] = []
        for c in cands:
            for j in c[i, 1:]:
                if j not in union:
                    union.append(int(j))
        union = [j for j in union if j != i]
        sim = []
        for j in union:
            s = 0.0
            for X, c, sg, w in zip(Xs, nearests, sigmas, weights):
                d = max(float(np.linalg.norm(X[i] - X[j])) - c[i], 0.0)
                s += w[i] * math.exp(-d / sg[i])
            sim.append(s)
        order = sorted(range(len(union)), key=lambda t: -sim[t])[:k]
        idx.append([union[t] for t in order])
        dist.append([math.sqrt(min(max((1 - sim[t]) / 2, 0), 1)) for t in order])
    return np.array(idx), np.array(dist)


def test_bandwidth_scores_and_joint_neighbours_match_seurat_s_loops():
    rng = np.random.default_rng(3)
    R = wnn.l2_rows(rng.normal(size=(40, 6)))
    P = wnn.l2_rows(rng.normal(size=(40, 4)))
    k, K = 5, 12
    nn_r, d_r = wnn.knn(R, K)
    nn_p, d_p = wnn.knn(P, K)
    assert (nn_r[:, 0] == np.arange(40)).all() and (d_r[:, 0] == 0).all()
    c_r, c_p = d_r[:, 1], d_p[:, 1]
    sig_r = wnn.snn_bandwidth(R, nn_r[:, :k], c_r, k, chunk=7)
    np.testing.assert_allclose(sig_r, _ref_sigma(R, nn_r[:, :k], c_r, k), rtol=1e-10)
    sig_p = wnn.snn_bandwidth(P, nn_p[:, :k], c_p, k)
    np.testing.assert_allclose(sig_p, _ref_sigma(P, nn_p[:, :k], c_p, k), rtol=1e-10)

    s_r = wnn.modality_scores(R, nn_r[:, :k], nn_p[:, :k], c_r, sig_r)
    # By hand for cell 0: means of the four other neighbours, distances minus c.
    within = R[nn_r[0, 1:k]].mean(axis=0)
    cross = R[nn_p[0, 1:k]].mean(axis=0)
    dw = max(np.linalg.norm(R[0] - within) - c_r[0], 0)
    dc = max(np.linalg.norm(R[0] - cross) - c_r[0], 0)
    want = min(max(math.exp(-dw / sig_r[0]) / (math.exp(-dc / sig_r[0]) + 1e-4), 0), 200)
    assert s_r[0] == pytest.approx(want, rel=1e-12)
    s_p = wnn.modality_scores(P, nn_p[:, :k], nn_r[:, :k], c_p, sig_p)
    w_r, w_p = wnn.modality_weights(s_r, s_p)
    np.testing.assert_allclose(w_r, np.exp(s_r) / (np.exp(s_r) + np.exp(s_p)), rtol=1e-12)
    np.testing.assert_allclose(w_r + w_p, 1.0)

    idx, dist = wnn.joint_knn((R, P), (nn_r, nn_p), (c_r, c_p), (sig_r, sig_p), (w_r, w_p), k, chunk=9)
    ref_idx, ref_dist = _ref_joint((R, P), (nn_r, nn_p), (c_r, c_p), (sig_r, sig_p), (w_r, w_p), k)
    assert (idx == ref_idx).all()
    np.testing.assert_allclose(dist, ref_dist, rtol=1e-10, atol=1e-12)


def test_a_tiny_case_by_hand():
    # Three cells on a line (1D, not normalised): 0 at 0, 1 at 1, 2 at 3. k = 2.
    X = np.array([[0.0], [1.0], [3.0]])
    nn, d = wnn.knn(X, 3)
    assert nn.tolist() == [[0, 1, 2], [1, 0, 2], [2, 1, 0]]
    c = d[:, 1]  # 1, 1, 2
    # Each 2-NN set shares a cell with every other: Jaccard 1/3 (one shared of 2+2-1),
    # itself 1. Cell 0's two smallest: cells 1 and 2 (1/3 each); distances minus
    # c = 1: 0 and 2 -> sigma 1. Cell 2: cells 0 and 1 (ties: 1/3), 3 - 2 and 2 - 2.
    sigma = wnn.snn_bandwidth(X, nn[:, :2], c, 2)
    assert sigma.tolist() == [1.0, pytest.approx(0.5), pytest.approx(0.5)]
    # Cell 0 predicted by its one other neighbour (cell 1, at 1): distance 1 - c = 0.
    # By the other modality's neighbour cell 2 (at 3): 3 - 1 = 2 -> e^-2 / sigma 1.
    other = np.array([[0, 2, 1], [1, 2, 0], [2, 0, 1]])
    s = wnn.modality_scores(X, nn[:, :2], other[:, :2], c, sigma)
    assert s[0] == pytest.approx(1.0 / (math.exp(-2.0) + 1e-4))
    w_a, w_b = wnn.modality_weights(np.array([2.0]), np.array([0.0]))
    assert w_a[0] == pytest.approx(math.e ** 2 / (math.e ** 2 + 1)) and w_b[0] == pytest.approx(1 / (math.e ** 2 + 1))
    # Scores are clipped at 200; softmax of two clipped scores stays finite.
    w_a, w_b = wnn.modality_weights(np.array([200.0]), np.array([0.0]))
    assert np.isfinite(w_a).all() and w_b[0] < 1e-80


def test_duplicate_cells_keep_themselves_first_and_identical_rows_give_no_nan():
    X = np.zeros((30, 3))
    X[:10] = 1.0  # ten identical cells
    X[10:] = 10 + np.random.default_rng(0).normal(size=(20, 3))  # far from them
    nn, d = wnn.knn(X, 8)
    assert (nn[:, 0] == np.arange(30)).all() and (d[:, 0] == 0).all()
    sigma = wnn.snn_bandwidth(X, nn[:, :5], d[:, 1], 5)
    assert np.isfinite(sigma).all() and (sigma[:10] == 0).all()  # floored where it is used
    s = wnn.modality_scores(X, nn[:, :5], nn[:, :5], d[:, 1], sigma)
    assert np.isfinite(s).all()


def test_l2_scale_and_pcs():
    assert wnn.l2_rows(np.array([[3.0, 4.0], [0.0, 0.0]])).tolist() == [[0.6, 0.8], [0.0, 0.0]]
    M = np.array([[1.0, 5.0], [2.0, 5.0], [3.0, 5.0], [100.0, 5.0]])
    Z = wnn.scale_proteins(M, 1.0)
    assert Z[:, 1].tolist() == [0, 0, 0, 0] and Z[:, 0].max() == 1.0  # constant -> 0, clipped above
    assert Z[0, 0] == pytest.approx((1 - 26.5) / np.std(M[:, 0], ddof=1))
    pcs = wnn.protein_pcs(np.random.default_rng(0).normal(size=(50, 6)), 3)
    assert pcs.shape == (50, 3)


# ---------------------------------------------------------------------------
# The method on two modalities
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def ran():
    a = _cite(zero_cells=3)
    p = wnn.plan(a)
    inp = wnn.prepare(a, p)
    events = list(wnn.iter_wnn(inp))
    return a, p, inp, events


def test_weights_favour_the_modality_that_tells_a_group_apart(ran):
    a, p, inp, events = ran
    res = events[-1]["result"]
    g = a.obs["group"].to_numpy().astype(str)
    w = res.rna_weight
    assert np.isfinite(w).all() and np.isfinite(res.protein_weight).all()
    np.testing.assert_allclose(w + res.protein_weight, 1.0, atol=1e-6)
    mean = {k: float(w[g == k].mean()) for k in ("g0", "g1", "g2", "g3")}
    # Proteins tell g0 from g1; RNA tells g2 from g3.
    assert mean["g0"] < 0.4 and mean["g1"] < 0.4, mean
    assert mean["g2"] > 0.6 and mean["g3"] > 0.6, mean


def test_joint_neighbours_tell_apart_what_neither_modality_does_alone(ran):
    a, p, inp, events = ran
    res = events[-1]["result"]
    g = a.obs["group"].cat.codes.to_numpy()
    rows = np.arange(a.n_obs)
    joint = res.distances.indices.reshape(a.n_obs, 20)
    nn_r, _ = wnn.knn(wnn.l2_rows(inp.rna), 21)
    nn_p, _ = wnn.knn(wnn.l2_rows(res.protein_pca), 21)
    assert _purity(joint, g) > 0.95
    assert _purity(nn_r[:, 1:], g) < 0.8 and _purity(nn_p[:, 1:], g) < 0.8
    assert not (joint == rows[:, None]).any()
    # The clusters on the joint graph find the four groups.
    from sklearn.metrics import adjusted_rand_score

    assert adjusted_rand_score(g, np.asarray(res.clusters).astype(int)) > 0.9


def test_stages_come_in_order_with_a_result(ran):
    *_, events = ran
    stages = [e["stage"] for e in events if e["type"] == "stage"]
    assert stages == [s for s, _ in wnn.STAGES]
    assert events[-1]["type"] == "result"
    res = events[-1]["result"]
    assert res.umap_wnn.shape == (GROUPS * PER_GROUP, 2) and res.umap_protein.shape == (GROUPS * PER_GROUP, 2)
    assert set(res.stage_seconds) == {s for s, _ in wnn.STAGES}


def test_the_plan_drops_controls_and_counts_cells_without_antibodies(ran):
    a, p, *_ = ran
    assert p.proteins == [f"P{i}" for i in range(8)] and p.controls == CONTROLS
    assert p.d_protein == 7 and p.d_rna == 10 and p.rep == "X_pca" and p.exact is True
    assert p.n_zero_protein == 3 and any("3 cells have no antibody counts" in w for w in p.warnings)
    assert any("Only 8 proteins" in w for w in p.warnings)
    d = p.to_dict()
    assert d["estimate_seconds"] > 0 and d["n_proteins"] == 8


def test_too_few_proteins_or_no_protein_is_refused():
    a = _cite(n_prot=4)  # 4 + 2 controls
    with pytest.raises(wnn.WnnError, match="at least 5 proteins besides isotype controls; this panel has 4"):
        wnn.plan(a)
    b = ad.AnnData(np.zeros((60, 3), dtype=np.float32))
    with pytest.raises(wnn.WnnError, match="no protein"):
        wnn.plan(b)
    assert wnn.runnable(b)["ok"] is False
    c = _cite()
    with pytest.raises(wnn.WnnError, match="between 2 and 7"):
        wnn.plan(c, protein_pcs=8)
    del c.obsm["X_pca"]
    with pytest.raises(wnn.WnnError, match="no X_pca"):
        wnn.plan(c)


def test_harmony_rna_gets_harmony_protein_pcs(monkeypatch):
    a = _cite()
    a.obs["batch"] = pd.Categorical(np.tile(["b1", "b2"], a.n_obs // 2))
    a.obsm["X_pca_harmony"] = a.obsm["X_pca"].copy()
    a.uns["neighbors"]["params"]["use_rep"] = "X_pca_harmony"
    provenance.record_step(a, step="batch_correction", tool="t", params={"batch_key": "batch",
                                                                        "batch_method": "harmony"})
    p = wnn.plan(a)
    assert p.rep == "X_pca_harmony" and p.batch_key == "batch"
    assert any("corrected on it too" in n for n in p.notes)
    seen = {}

    def fake(X, obs, key):
        seen["key"], seen["shape"] = key, X.shape
        return X + 1.0

    from scview.core import batch_methods

    monkeypatch.setattr(batch_methods, "harmony_correct", fake)
    pcs = wnn._protein_pca(wnn.prepare(a, p))
    assert seen == {"key": "batch", "shape": (a.n_obs, 7)}
    plain = wnn.protein_pcs(wnn.scale_proteins(wnn.prepare(a, p).clr), 7)
    np.testing.assert_allclose(pcs, plain + 1.0, rtol=1e-5)


def test_save_stores_everything_and_a_later_change_makes_it_stale(ran):
    a0, p, inp, events = ran
    a = a0.copy()
    out = wnn.save(a, inp, events[-1]["result"], resources={"seconds": 1.0}, cells_fingerprint="fp")
    assert out["available"] is True and out["stale"] is False
    for key in wnn.OBS_KEYS:
        assert key in a.obs.columns
    for key in wnn.OBSM_KEYS:
        assert key in a.obsm
    assert a.obsp[wnn.CONN_KEY].shape == (a.n_obs, a.n_obs)
    assert np.diff(a.obsp[wnn.DIST_KEY].indptr).tolist() == [20] * a.n_obs
    assert a.uns["wnn"]["params"]["method"] == "umap" and "neighbors" in a.uns
    step = provenance.read_provenance(a)["history"][-1]
    assert step["step"] == "wnn" and step["params"]["proteins"] == p.proteins
    assert step["params"]["panel"] == p.proteins + CONTROLS
    info = json.loads(a.uns[wnn.UNS_KEY])
    assert info["plan"]["d_protein"] == 7 and "Hao et al. 2021" in info["method"]

    s = wnn.summary(a, "group", cells_fingerprint="fp")
    assert [r["group"] for r in s["by_group"]["rows"]] == ["g0", "g1", "g2", "g3"]
    assert s["embeddings"] == ["X_umap_protein", "X_umap_wnn"]
    assert wnn.summary(a, cells_fingerprint="other")["stale"] is True
    b = a.copy()
    b.obsm["X_pca"] = b.obsm["X_pca"] * 2
    assert "changed" in wnn.summary(b)["stale_reason"]
    assert set(wnn.violin(a, "group")) == {"g0", "g1", "g2", "g3"}

    a.obs["cell_type"] = a.obs["group"]  # the author labels the fact groups by
    lines, detail = wnn.grounding_lines(a)
    assert lines[0] == "\n## Joint RNA + protein (WNN)" and f"[{wnn.TAG}]" in lines[1]
    assert "Mean modality weight per cell_type, highest RNA weight first: g" in lines[1]
    assert "8 proteins" in detail
    assert wnn.grounding_lines(b) == ([], "")  # stale: no fact

    wnn.drop(a)
    assert wnn.stored(a) is None and wnn.OBSM_WNN not in a.obsm and wnn.CONN_KEY not in a.obsp


def test_a_merge_drops_the_result(ran):
    a0, p, inp, events = ran
    a, b = a0.copy(), a0.copy()
    wnn.save(a, inp, events[-1]["result"])
    protein.common_panel([a, b])
    assert wnn.stored(a) is None and wnn.OBSM_PCA not in a.obsm and wnn.OBS_RNA not in a.obs.columns


def test_cancel_stops_before_the_next_stage():
    a = _cite(n_per=20)
    inp = wnn.prepare(a, wnn.plan(a))
    calls = {"n": 0}

    def stop():
        calls["n"] += 1
        return calls["n"] > 2

    gen = wnn.iter_wnn(inp, stop=stop)
    assert next(gen)["stage"] == "protein_pca"
    assert next(gen)["stage"] == "knn_rna"
    with pytest.raises(wnn.WnnCancelled):
        next(gen)


def test_the_replay_helpers_compute_the_same_weights(ran):
    a, p, inp, events = ran
    res = events[-1]["result"]
    out = wnn.wnn_neighbors(inp.rna, res.protein_pca)
    np.testing.assert_allclose(out["rna_weight"], res.rna_weight, rtol=1e-6)
