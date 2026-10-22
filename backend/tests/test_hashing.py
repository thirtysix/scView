"""Cell hashing demultiplexing (core/hashing.py, api/v1/hashing.py).

References: R 4.4.0 with Seurat 5.1.0 / cluster 2.1.6 / DropletUtils 1.24.0 on the
same inputs (fixtures/hashing, made by make_reference.R), and expectations worked
out independently (plain numpy / scipy) on small examples.
"""

from __future__ import annotations

import json
import math
import subprocess
import sys
from pathlib import Path

import anndata as ad
import h5py
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp
from scipy import optimize, stats

from scview.core import cluster_qc, hashing, protein, provenance, replay_script

FIX = Path(__file__).parent / "fixtures" / "hashing"
API = "/api/v1"


# ---------------------------------------------------------------------------
# The maths, by hand and against R
# ---------------------------------------------------------------------------


def test_clr_is_seurats_per_tag_formula():
    x = np.array([[0, 5, 0], [10, 5, 0], [100, 5, 0]], dtype=float)
    # Seurat margin = 1: log1p(x / exp(sum(log1p(x[x > 0])) / length(x))) per tag.
    expected = np.zeros_like(x)
    for j in range(2):
        col = x[:, j]
        expected[:, j] = np.log1p(col / np.exp(np.log1p(col[col > 0]).sum() / len(col)))
    assert np.allclose(hashing.clr_tags(x), expected, atol=1e-14)
    assert (hashing.clr_tags(x)[:, 2] == 0).all()  # a tag with no counts stays 0
    # The divisor is per tag, so a tag's CLR is a monotone function of its counts.
    assert hashing.clr_denominators(x)[0] == pytest.approx(math.exp((math.log1p(10) + math.log1p(100)) / 3))


@pytest.mark.parametrize("case", ["A", "B", "C"])
def test_clara_is_r_clara(case):
    # A: 300 points (the usual case); B: 60 points, the sample is more than half the
    # data (clara draws the half left out); C: 40 points, the sample is all of it.
    pts = pd.read_csv(FIX / "clara_points.csv")
    ref = pd.read_csv(FIX / "clara_reference.tsv", sep="\t")
    d = pts[pts["case"] == case]
    x = d[["x1", "x2", "x3", "x4"]].to_numpy(dtype=float)
    x = x[:, (x != 0).any(axis=0)]
    got = hashing.clara(x, int(d["k"].iloc[0]), samples=100)
    assert got.tolist() == ref[ref["case"] == case]["cluster"].tolist()


def test_the_nb_fit_is_the_maximum_likelihood():
    rng = np.random.default_rng(3)
    v = rng.negative_binomial(1.5, 1.5 / (1.5 + 20), size=3000)
    size, mu = hashing.nb_fit(v)
    assert mu == pytest.approx(v.mean(), rel=1e-12)  # the NB MLE of mu is the mean

    def nll(p):
        r, m = np.exp(p)
        return -stats.nbinom.logpmf(v, r, r / (r + m)).sum()

    best = optimize.minimize(nll, x0=[0.0, math.log(v.mean())], method="Nelder-Mead",
                             options={"xatol": 1e-10, "fatol": 1e-10, "maxiter": 5000})
    assert size == pytest.approx(math.exp(best.x[0]), rel=1e-5)
    assert nll([math.log(size), math.log(mu)]) <= best.fun + 1e-8
    # Not over-dispersed: the Poisson limit.
    assert hashing.nb_fit(np.array([2, 2, 3, 3, 2, 3]))[0] == math.inf
    assert hashing.nb_fit(np.zeros(10)) == (math.inf, 0.0)


def test_the_quantile_is_rs_qnbinom():
    ref = pd.read_csv(FIX / "qnbinom_reference.tsv", sep="\t")
    for r in ref.itertuples():
        assert hashing.nb_quantile(r.size, r.mu, 0.99) == r.q99
        assert hashing.nb_quantile(r.size, r.mu, 0.90) == r.q90
    assert hashing.nb_quantile(math.inf, 4.0, 0.99) == int(stats.poisson(4.0).ppf(0.99))
    assert hashing.nb_quantile(1.0, 0.0) == 0


def _tiny() -> tuple[np.ndarray, list[str]]:
    """20 cells high for A, 20 high for B, 20 empty droplets, one A+B doublet."""
    a_hi = np.arange(150, 250, 5)
    b_hi = np.arange(160, 260, 5)
    low = np.array([0, 0, 0, 0, 0, 1, 1, 2, 3, 8, 0, 0, 1, 0, 5, 0, 2, 0, 1, 9])
    a = np.concatenate([a_hi, low[::-1] % 5 + 2, low, [200]])
    b = np.concatenate([low % 4 + 2, b_hi, low[::-1], [210]])
    return np.column_stack([a, b]).astype(float), ["A", "B"]


def test_the_cutoff_is_the_backgrounds_99th_percentile_by_hand():
    x, tags = _tiny()
    call = hashing.htodemux(x, tags)
    for j, t in enumerate(tags):
        p = call.per_tag[j]
        # The background of each tag is the empty droplets: the group with the lowest A
        # (resp. B), which has 20 of them (rows 40-59) and maybe the doublet's partner.
        bg = x[40:60, j]
        assert p["background_cells"] == 20
        r = optimize.minimize_scalar(
            lambda lr: -stats.nbinom.logpmf(bg, math.exp(lr), math.exp(lr) / (math.exp(lr) + bg.mean())).sum(),
            bounds=(-10, 10), method="bounded", options={"xatol": 1e-10})
        size = math.exp(r.x)
        expected = int(stats.nbinom(size, size / (size + bg.mean())).ppf(0.99))
        assert p["cutoff"] == expected
        assert p["n_positive"] == int((x[:, j] > expected).sum())
    assert call.cls[:20].tolist() == ["Singlet"] * 20 and call.call[:20].tolist() == ["A"] * 20
    assert call.call[20:40].tolist() == ["B"] * 20
    assert set(call.cls[40:60]) == {"Negative"}
    assert call.cls[60] == "Doublet"


def _simulate(rng, n: int = 3000, n_tags: int = 8, doublet: float = 0.10, negative: float = 0.08):
    """Cells with one sample's tag (singlets), two samples' tags (doublets) or
    background only (empty droplets), every tag with its own NB background."""
    kind = rng.choice(np.array(["S", "D", "N"]), p=[1 - doublet - negative, doublet, negative], size=n)
    first = rng.integers(0, n_tags, n)
    other = (first + rng.integers(1, n_tags, n)) % n_tags
    bg = rng.uniform(5, 25, n_tags)
    x = rng.negative_binomial(4, 4 / (4 + bg), size=(n, n_tags)).astype(float)
    sig = rng.uniform(300, 700, n_tags)
    depth = rng.lognormal(0, 0.25, n)
    rows = np.arange(n)
    has = kind != "N"
    x[rows[has], first[has]] += rng.negative_binomial(10, 10 / (10 + sig[first[has]] * depth[has]))
    d = kind == "D"
    x[rows[d], other[d]] += rng.negative_binomial(10, 10 / (10 + sig[other[d]] * depth[d]))
    return x, [f"HTO-{i + 1}" for i in range(n_tags)], kind, first


@pytest.mark.parametrize("method", [hashing.HTODEMUX, hashing.HASHEDDROPS])
def test_simulated_hashing_is_recovered(method):
    x, tags, kind, first = _simulate(np.random.default_rng(118))
    call = hashing.run(x, tags, method)
    cls = call.cls
    singlet, doublet, empty = kind == "S", kind == "D", kind == "N"
    called = singlet & (cls == "Singlet")
    assert (call.best[called] == first[called]).mean() > 0.995  # the right tag
    assert ((cls == "Doublet") & doublet).sum() / doublet.sum() > 0.95
    assert ((cls == "Negative") & empty).sum() / empty.sum() > 0.85
    false_doublets = ((cls == "Doublet") & singlet).sum() / singlet.sum()
    if method == hashing.HTODEMUX:
        # Each tag is called at its background's 99th percentile, so a singlet has
        # ~1 - 0.99^7 = 7% odds of one background tag above its cutoff: a "doublet".
        assert called.sum() / singlet.sum() > 0.88
        assert 0.03 < false_doublets < 0.12
    else:
        assert called.sum() / singlet.sum() > 0.97
        assert false_doublets < 0.02


def _reference():
    counts = pd.read_csv(FIX / "pbmc_subset_counts.csv", index_col=0)
    ref = pd.read_csv(FIX / "reference_calls.tsv", sep="\t")
    return counts, ref


def test_real_hashing_data_agrees_with_seurat_htodemux():
    # 1,000 PBMC from Stoeckius et al. 2018 (8 hashtags), called by Seurat 5.1.0.
    counts, ref = _reference()
    tags = list(counts.columns)
    call = hashing.htodemux(counts.to_numpy(float), tags)
    assert (call.cls == ref["global"].to_numpy()).all()
    names = np.asarray(tags)
    assert (names[call.best] == ref["maxID"].to_numpy()).all()
    assert (names[call.second] == ref["secondID"].to_numpy()).all()
    assert np.allclose(call.margin, ref["margin"].to_numpy(), atol=1e-8)
    assert dict(zip(*np.unique(call.cls, return_counts=True))) == {"Doublet": 173, "Negative": 80, "Singlet": 747}


def test_real_hashing_data_agrees_with_dropletutils_hasheddrops():
    counts, ref = _reference()
    x = counts.to_numpy(float)
    st = hashing.hashed_drops_stats(x)
    assert (st["best"] + 1 == ref["hd_best"].to_numpy()).all()
    assert (st["second"] + 1 == ref["hd_second"].to_numpy()).all()
    assert np.allclose(st["lfc"], ref["hd_lfc"].to_numpy(), atol=1e-8)
    assert np.allclose(st["lfc2"], ref["hd_lfc2"].to_numpy(), atol=1e-8)
    call = hashing.hashed_drops(x, list(counts.columns))
    expected = np.where(ref["hd_doublet"], "Doublet", np.where(ref["hd_confident"], "Singlet", "Negative"))
    assert (call.cls == expected).all()


def test_an_empty_background_group_is_refused_with_a_reason():
    x, tags = _tiny()
    x = np.column_stack([x, np.zeros(len(x))])  # a tag nobody used
    with pytest.raises(hashing.HashingError, match="no C counts"):
        hashing.htodemux(x, tags + ["C"])
    for bad, msg in ((x[:, :1], "Pick 2"), (x[:10, :2], "At least 50"), (-x[:, :2], "not negative")):
        with pytest.raises(hashing.HashingError, match=msg):
            hashing.run(bad, tags[: bad.shape[1]] if bad.shape[1] < 2 else tags)
    with pytest.raises(hashing.HashingError, match="Unknown method"):
        hashing.run(x[:, :2], tags, "kmeans")


@pytest.mark.parametrize("name,tag", [
    ("Hashtag1", True), ("HTO_A", True), ("HTO-12", True), ("B0251_anti-human_Hashtag_1", True),
    ("C0256", True), ("TotalSeq-A0260", True), ("CMO301", True), ("CD3", False), ("CD45RA", False),
    ("PHTOA", False), ("A0252x9", False), ("C0300", False), ("IgG1_control", False),
])
def test_hashtag_names_are_suggested(name, tag):
    assert hashing.looks_like_tag(name) is tag


# ---------------------------------------------------------------------------
# The dataset: loading, moving, saving
# ---------------------------------------------------------------------------


def _cite_hashed(n: int = 600, seed: int = 0) -> ad.AnnData:
    """Genes, three CITE-seq antibodies and four hashtags in one 10x feature matrix."""
    rng = np.random.default_rng(seed)
    x, tags, kind, first = _simulate(rng, n=n, n_tags=4)
    genes = rng.poisson(3, size=(n, 30)).astype(np.float32)
    adt = rng.poisson(40, size=(n, 3)).astype(np.float32)
    X = np.hstack([genes, adt, x.astype(np.float32)])
    names = [f"G{i}" for i in range(30)] + ["CD3_TotalSeqB", "CD19_TotalSeqB", "IgG1_control_TotalSeqB"] + [
        f"{t}_TotalSeqB" for t in tags]
    types = ["Gene Expression"] * 30 + ["Antibody Capture"] * 7
    a = ad.AnnData(sp.csr_matrix(X), var=pd.DataFrame({"feature_types": types}, index=names))
    a.obs_names = [f"cell{i}" for i in range(n)]
    a.uns["truth"] = json.dumps({"kind": kind.tolist(), "first": first.tolist()})
    return a


def test_cellplex_tags_are_kept_at_load_and_the_cite_seq_split_still_works():
    a = _cite_hashed(n=80)
    types = a.var["feature_types"].astype(str).to_numpy()
    types[-4:] = hashing.CMO_TYPE  # the hashtags as CellPlex reads them
    a.var["feature_types"] = types
    a.var_names = list(a.var_names[:-4]) + ["CMO301", "CMO302", "CMO303", "CMO304"]
    genes, split = protein.split_features(a)
    assert genes.n_vars == 30
    assert list(genes.obsm[hashing.HTO_KEY].columns) == ["CMO301", "CMO302", "CMO303", "CMO304"]
    assert np.array_equal(genes.obsm[hashing.HTO_KEY].to_numpy(), a.X[:, -4:].toarray())
    assert protein.protein_names(genes) == ["CD3", "CD19", "IgG1_control"]
    assert split.other_features == {}  # not dropped any more
    assert hashing.read_state(genes)["sources"]["CMO301"] == hashing.CMO_TYPE
    cand = hashing.candidates(genes)
    assert cand["tags"] == ["CMO301", "CMO302", "CMO303", "CMO304"] and cand["suggested"] == []


def test_hashtags_read_as_antibodies_are_suggested_then_moved_out_of_the_protein_modality():
    a, _ = protein.split_features(_cite_hashed())
    cand = hashing.candidates(a)
    assert cand["tags"] == [] and cand["suggested"] == ["HTO-1", "HTO-2", "HTO-3", "HTO-4"]
    before = a.obsm[protein.COUNTS_KEY].copy()
    out = hashing.demux(a, ["HTO-1", "HTO-2", "HTO-3", "HTO-4"])
    assert protein.protein_names(a) == ["CD3", "CD19", "IgG1_control"]
    # The protein CLR is per cell across proteins: made again without the hashtags.
    rest = before[["CD3", "CD19", "IgG1_control"]].to_numpy()
    assert np.allclose(a.obsm[protein.CLR_KEY].to_numpy(), protein.clr(rest), atol=1e-6)
    assert a.uns[protein.UNS_KEY]["names"] == ["CD3", "CD19", "IgG1_control"]
    hto = a.obsm[hashing.HTO_KEY]
    assert list(hto.columns) == ["HTO-1", "HTO-2", "HTO-3", "HTO-4"]
    assert np.array_equal(hto.to_numpy(), before[list(hto.columns)].to_numpy())
    assert out["call"]["tags"] == ["HTO-1", "HTO-2", "HTO-3", "HTO-4"]
    assert out["moved_from_protein"] == ["HTO-1", "HTO-2", "HTO-3", "HTO-4"]
    assert hashing.read_state(a)["original"]["HTO-1"] == "HTO-1_TotalSeqB"
    truth = json.loads(a.uns["truth"])
    called = a.obs[hashing.CLASS].astype(str).to_numpy()
    assert ((called == "Doublet") == (np.array(truth["kind"]) == "D")).mean() > 0.95
    for col in hashing.OBS_COLUMNS:
        assert col in a.obs.columns
    assert list(a.obs[hashing.CALL].cat.categories) == ["HTO-1", "HTO-2", "HTO-3", "HTO-4", "Doublet", "Negative"]
    # Again with three tags: the fourth goes back to the protein modality.
    hashing.demux(a, ["HTO-1", "HTO-2", "HTO-3"], hashing.HASHEDDROPS)
    assert protein.protein_names(a) == ["CD3", "CD19", "IgG1_control", "HTO-4"]
    assert list(a.obsm[hashing.HTO_KEY].columns) == ["HTO-1", "HTO-2", "HTO-3"]
    assert np.allclose(a.obsm[protein.COUNTS_KEY]["HTO-4"].to_numpy(), before["HTO-4"].to_numpy())
    hist = [h["step"] for h in provenance.read_provenance(a)["history"]]
    assert hist.count("hashing_demux") == 2


def test_a_panel_of_hashtags_only_leaves_no_protein_modality():
    a = _cite_hashed()
    a = a[:, [i for i, n in enumerate(a.var_names) if not n.startswith(("CD", "IgG"))]].copy()
    a, _ = protein.split_features(a)
    hashing.demux(a, ["HTO-1", "HTO-2", "HTO-3", "HTO-4"])
    assert not protein.has_protein(a) and protein.COUNTS_KEY not in a.obsm
    assert a.uns[protein.UNS_KEY]["n_proteins"] == 0


def test_a_hashtag_table_another_tool_left_in_obsm_is_used_and_adopted():
    x, tags, kind, _ = _simulate(np.random.default_rng(7), n=300, n_tags=3)
    a = ad.AnnData(np.ones((300, 2), dtype=np.float32))
    a.obsm["HTO"] = pd.DataFrame(x, index=a.obs_names, columns=["h1", "h2", "h3"])
    a.obsm["hto"] = pd.DataFrame(x / 2.5, index=a.obs_names, columns=["n1", "n2", "n3"])  # not counts
    cand = hashing.candidates(a)
    assert cand["tags"] == ["h1", "h2", "h3"] and cand["tag_sources"]["h1"] == "obsm HTO"
    out = hashing.demux(a, ["h1", "h2", "h3"])
    assert np.array_equal(a.obsm[hashing.HTO_KEY].to_numpy(), x.astype(np.float32))
    assert out["tag_sources"] == {"h1": "obsm HTO", "h2": "obsm HTO", "h3": "obsm HTO"}
    step = provenance.read_provenance(a)["history"][-1]
    assert step["params"]["sources"]["h1"] == "obsm HTO"


def test_a_failed_call_changes_nothing():
    a, _ = protein.split_features(_cite_hashed())
    obsm, obs = set(a.obsm.keys()), list(a.obs.columns)
    for tags in (["HTO-1", "nope"], ["HTO-1"], ["HTO-1", "HTO-1"], "HTO-1"):
        with pytest.raises(hashing.HashingError):
            hashing.demux(a, tags)  # type: ignore[arg-type]
    with pytest.raises(hashing.HashingError):
        hashing.demux(a, ["HTO-1", "HTO-2"], method="nope")
    assert set(a.obsm.keys()) == obsm and list(a.obs.columns) == obs
    assert hashing.UNS_KEY not in a.uns and protein.protein_names(a)[-1] == "HTO-4"


def test_the_sample_column_names_singlets_and_leaves_doublets_empty():
    a, _ = protein.split_features(_cite_hashed())
    with pytest.raises(hashing.HashingError) as e:
        hashing.sample_column(a, {})
    assert e.value.status == 409
    hashing.demux(a, ["HTO-1", "HTO-2", "HTO-3", "HTO-4"])
    out = hashing.sample_column(a, {"HTO-1": "donor A", "HTO-2": "donor B", "HTO-3": ""}, "donor")
    col, call = a.obs["donor"], a.obs[hashing.CALL].astype(str)
    assert list(col.cat.categories) == ["donor A", "donor B", "HTO-3", "HTO-4"]
    assert (col[call == "HTO-1"] == "donor A").all() and (col[call == "HTO-3"] == "HTO-3").all()
    assert col[call.isin(["Doublet", "Negative"])].isna().all()
    assert out["n_unassigned"] == int(call.isin(["Doublet", "Negative"]).sum())
    # Renamed: the old column goes; an existing column of another kind is refused.
    hashing.sample_column(a, {"HTO-1": "donor A"}, "sample_id")
    assert "donor" not in a.obs.columns and "sample_id" in a.obs.columns
    with pytest.raises(hashing.HashingError) as e:
        hashing.sample_column(a, {}, hashing.CLASS)
    assert e.value.status == 409
    for bad in ({"nope": "x"}, {"HTO-1": "a\x00b"}):
        with pytest.raises(hashing.HashingError):
            hashing.sample_column(a, bad, "sample_id")
    # A new call keeps the sample column in step (same names, new tags keep theirs).
    hashing.demux(a, ["HTO-1", "HTO-2"], hashing.HASHEDDROPS)
    col, call = a.obs["sample_id"], a.obs[hashing.CALL].astype(str)
    assert (col[call == "HTO-1"] == "donor A").all() and col[call == "Doublet"].isna().all()
    assert set(col.dropna().unique()) <= {"donor A", "HTO-2"}


def test_hashing_doublets_feed_the_doublet_views():
    obs = pd.DataFrame({hashing.CLASS: pd.Categorical(["Singlet", "Doublet", "Negative"])})
    assert cluster_qc.doublet_calls(obs).tolist() == [False, True, False]
    obs["predicted_doublet"] = [True, False, False]  # Scrublet's call is read first
    assert cluster_qc.doublet_calls(obs).tolist() == [True, False, False]


def test_summary_counts_the_current_cells_and_compares_with_scrublet():
    a, _ = protein.split_features(_cite_hashed())
    hashing.demux(a, ["HTO-1", "HTO-2", "HTO-3", "HTO-4"])
    a.obs["predicted_doublet"] = a.obs[hashing.CLASS].astype(str).eq("Doublet").to_numpy()
    sub = a[a.obs[hashing.CLASS] != "Negative"].copy()  # filtered after the call
    s = hashing.summary(sub)
    assert s["call"]["counts"]["Negative"] == 0 and s["call"]["n_cells_at_call"] == a.n_obs
    assert s["scrublet"]["Doublet"]["scrublet_singlet"] == 0
    assert s["doublet_source"] == "predicted_doublet"
    assert sum(s["call"]["per_call"].values()) == sub.n_obs


def test_plot_data_draws_each_cutoff_in_clr_units():
    a, _ = protein.split_features(_cite_hashed())
    pre = hashing.plot_data(a)
    assert pre["available"] is False  # tags still in the protein modality, no call
    hashing.demux(a, ["HTO-1", "HTO-2", "HTO-3", "HTO-4"])
    pd_ = hashing.plot_data(a)
    counts = a.obsm[hashing.HTO_KEY].to_numpy(float)
    clr = hashing.clr_tags(counts)
    for j, p in enumerate(pd_["per_tag"]):
        assert sum(sum(v) for v in p["counts"].values()) == a.n_obs
        above = counts[:, j] > p["cutoff"]
        # The CLR line separates exactly the cells above the count cutoff.
        assert (clr[above, j] > p["cutoff_clr"]).all() and (clr[~above, j] < p["cutoff_clr"]).all()
        assert p["n_above"] == int(above.sum())
    hm = pd_["heatmap"]
    j = hm["calls"].index("HTO-2")
    assert hm["n"][j] == int((a.obs[hashing.CALL] == "HTO-2").sum())
    assert np.argmax([row[j] for row in hm["mean_clr"]]) == 1  # HTO-2's singlets: HTO-2 highest
    hashing.demux(a, ["HTO-1", "HTO-2", "HTO-3", "HTO-4"], hashing.HASHEDDROPS)
    lfc = hashing.plot_data(a)["lfc"]
    assert lfc["lfc2"]["threshold"] == hashing.read_state(a)["call"]["thresholds"]["doublet_lfc2"]
    assert sum(lfc["lfc"]["counts"]) == a.n_obs


def test_the_copilot_fact_gives_the_counts_and_the_cutoffs():
    a, _ = protein.split_features(_cite_hashed())
    lines, _ = hashing.grounding_lines(a)
    assert "not demultiplexed yet" in lines[0] and "HTO-1" in lines[0]
    hashing.demux(a, ["HTO-1", "HTO-2", "HTO-3", "HTO-4"])
    lines, detail = hashing.grounding_lines(a)
    c = hashing.read_state(a)["call"]["counts"]
    assert f"{c['Doublet']:,} doublets" in lines[0] and "[result:hashing]" in lines[0]
    assert "count cutoffs: HTO-1" in lines[0] and "the doublet views use these hashing calls" in lines[0]
    assert f"{c['Singlet']:,} singlets" in detail
    assert hashing.grounding_lines(ad.AnnData(np.ones((3, 2), dtype=np.float32))) == ([], "")


def test_the_copilot_context_carries_the_fact(tmp_path):
    from scview.core.anndata_adaptor import AnnDataAdaptor
    from scview.core.assistant import build_grounding_context
    from scview.core.claims import resolve_tag

    a, _ = protein.split_features(_cite_hashed())
    hashing.demux(a, ["HTO-1", "HTO-2", "HTO-3", "HTO-4"])
    a.write_h5ad(tmp_path / "d.h5ad")
    text, sources = build_grounding_context(AnnDataAdaptor(str(tmp_path / "d.h5ad")))
    assert "[result:hashing]" in text
    refs = [s.ref for s in sources]
    assert hashing.TAG in refs
    assert resolve_tag("hashing", refs) == [hashing.TAG]


# ---------------------------------------------------------------------------
# Replay: the script computes the same call
# ---------------------------------------------------------------------------


def _replay(tmp_path: Path, src: Path, out: ad.AnnData) -> ad.AnnData:
    script, dst = tmp_path / "replay.py", tmp_path / "replayed.h5ad"
    script.write_text(replay_script.build_script(out, dataset_id="t", dataset_name="test"))
    proc = subprocess.run([sys.executable, str(script), str(src), str(dst)],
                          capture_output=True, text=True, timeout=600)
    assert proc.returncode == 0, proc.stderr[-3000:]
    return ad.read_h5ad(dst)


def _same_call(a: ad.AnnData, b: ad.AnnData, extra: tuple[str, ...] = ()) -> None:
    b = b[a.obs_names]
    for col in (hashing.CALL, hashing.CLASS, hashing.MAX, hashing.SECOND, *extra):
        assert a.obs[col].astype(str).tolist() == b.obs[col].astype(str).tolist(), col
        if hasattr(a.obs[col], "cat"):
            assert list(a.obs[col].cat.categories) == list(b.obs[col].cat.categories), col
    assert np.array_equal(a.obs[hashing.MARGIN].to_numpy(), b.obs[hashing.MARGIN].to_numpy())


def test_the_replay_script_repeats_an_htodemux_call_on_hashtags_in_an_h5ad(tmp_path):
    raw = _cite_hashed(n=400)
    src = tmp_path / "input.h5ad"
    raw.write_h5ad(src)
    a, _ = protein.split_features(raw.copy())
    hashing.demux(a, ["HTO-1", "HTO-2", "HTO-3", "HTO-4"])
    hashing.sample_column(a, {"HTO-1": "d1", "HTO-2": "d2"}, "donor")
    desc = replay_script.describe(a)
    assert [d["status"] for d in desc] == ["exact", "exact"]
    _same_call(a, _replay(tmp_path, src, a), extra=("donor",))


def _tenx_cmo_h5(p: Path, a: ad.AnnData) -> None:
    """A 10x .h5 with the hashtags as CellPlex 'Multiplexing Capture' features."""
    X = a.X.toarray().astype(np.int32)
    keep = [i for i, t in enumerate(a.var["feature_types"]) if t == "Gene Expression"] + list(range(33, 37))
    m = sp.csc_matrix(X[:, keep].T)
    names = [a.var_names[i] for i in keep[:-4]] + ["CMO301", "CMO302", "CMO303", "CMO304"]
    with h5py.File(p, "w") as f:
        g = f.create_group("matrix")
        g.create_dataset("data", data=m.data)
        g.create_dataset("indices", data=m.indices.astype("int64"))
        g.create_dataset("indptr", data=m.indptr.astype("int64"))
        g.create_dataset("shape", data=np.array(m.shape, dtype="int32"))
        g.create_dataset("barcodes", data=np.array([f"{c}-1".encode() for c in a.obs_names]))
        ff = g.create_group("features")
        ff.create_dataset("id", data=np.array([n.encode() for n in names]))
        ff.create_dataset("name", data=np.array([n.encode() for n in names]))
        ff.create_dataset("feature_type", data=np.array([b"Gene Expression"] * 30 + [b"Multiplexing Capture"] * 4))
        ff.create_dataset("genome", data=np.array([b"GRCh38"] * len(names)))
        ff.create_dataset("_all_tag_keys", data=np.array([b"genome"]))


def test_cellplex_in_a_10x_h5_loads_as_tags_and_its_hasheddrops_call_replays(tmp_path):
    from scview.core.ingestion import build_bundle, load_unit

    src = tmp_path / "multi.h5"
    _tenx_cmo_h5(src, _cite_hashed(n=400))
    bundle = build_bundle([src])
    a = load_unit(bundle.units[0])
    assert a.n_vars == 30 and list(a.obsm[hashing.HTO_KEY].columns) == ["CMO301", "CMO302", "CMO303", "CMO304"]
    assert not protein.has_protein(a)
    hashing.demux(a, ["CMO301", "CMO302", "CMO303", "CMO304"], hashing.HASHEDDROPS)
    _same_call(a, _replay(tmp_path, src, a))


def test_an_unrecorded_call_is_not_replayed_and_nothing_is_injected():
    a = ad.AnnData(np.ones((3, 2), dtype=np.float32))
    evil = "x'] ; import os; os.system('touch /tmp/pwned') #"
    provenance.record_step(a, step="hashing_demux", tool="t", params={"tags": [evil], "method": "htodemux"})
    provenance.record_step(a, step="hashing_samples", tool="t", params={"column": evil, "mapping": {evil: evil}})
    script = replay_script.build_script(a)
    assert [d["status"] for d in replay_script.describe(a)] == ["not_replayable", "exact"]
    assert not any(line.lstrip().startswith("import os") for line in script.splitlines())
    compile(script, "replay.py", "exec")


# ---------------------------------------------------------------------------
# The endpoints
# ---------------------------------------------------------------------------


def _register(dm, a: ad.AnnData, did: str = "hto") -> str:
    d = Path(dm.data_dir) / "uploads" / did
    d.mkdir(parents=True)
    (d / "metadata.json").write_text(json.dumps({"id": did, "name": did, "status": "ready"}))
    a.write_h5ad(d / "data.h5ad")
    return did


async def test_a_dataset_without_tags_says_so(client, dataset_manager):
    a = ad.AnnData(np.ones((60, 3), dtype=np.float32))
    did = _register(dataset_manager, a, "plain")
    r = await client.get(f"{API}/datasets/{did}/hashing")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["available"] is False and body["tags"] == [] and body["call"] is None
    r = await client.get(f"{API}/datasets/{did}/hashing/plot-data")
    assert r.json()["available"] is False
    r = await client.post(f"{API}/datasets/{did}/hashing/demux", json={"tags": ["a", "b"]})
    assert r.status_code == 400
    r = await client.get(f"{API}/datasets/ghost/hashing")
    assert r.status_code == 404


async def test_demux_saves_the_call_and_moves_the_hashtags_on_disk(client, dataset_manager):
    a, _ = protein.split_features(_cite_hashed())
    did = _register(dataset_manager, a)
    r = await client.get(f"{API}/datasets/{did}/hashing")
    body = r.json()
    assert body["available"] and body["suggested"] == ["HTO-1", "HTO-2", "HTO-3", "HTO-4"]
    r = await client.post(f"{API}/datasets/{did}/hashing/samples", json={"mapping": {}})
    assert r.status_code == 409  # nothing called yet
    tags = ["HTO-1", "HTO-2", "HTO-3", "HTO-4"]
    r = await client.post(f"{API}/datasets/{did}/hashing/demux", json={"tags": tags})
    assert r.status_code == 200, r.text
    saved = r.json()
    assert saved["call"]["method"] == "htodemux" and sum(saved["call"]["counts"].values()) == a.n_obs
    derived = Path(dataset_manager.data_dir) / "derived" / did / "data.h5ad"
    disk = ad.read_h5ad(derived)
    assert list(disk.obsm[hashing.HTO_KEY].columns) == tags
    assert list(disk.obsm[protein.COUNTS_KEY].columns) == ["CD3", "CD19", "IgG1_control"]
    assert list(disk.obsm[protein.CLR_KEY].columns) == ["CD3", "CD19", "IgG1_control"]
    assert disk.obs[hashing.CLASS].astype(str).value_counts().to_dict() == {
        k: v for k, v in saved["call"]["counts"].items() if v}
    r = await client.post(f"{API}/datasets/{did}/hashing/samples",
                          json={"mapping": {"HTO-1": "P1", "HTO-2": "P2"}, "column": "donor"})
    assert r.status_code == 200, r.text
    assert r.json()["samples"]["column"] == "donor"
    disk = ad.read_h5ad(derived)
    assert set(disk.obs["donor"].dropna().unique()) == {"P1", "P2", "HTO-3", "HTO-4"}
    info = await client.get(f"{API}/datasets/{did}")
    assert {"donor", hashing.CALL, hashing.CLASS} <= {c["name"] for c in info.json()["obs_columns"]}
    r = await client.get(f"{API}/datasets/{did}/hashing/plot-data")
    plot = r.json()
    assert plot["available"] and [p["tag"] for p in plot["per_tag"]] == tags
    assert plot["heatmap"]["tags"] == tags
    r = await client.post(f"{API}/datasets/{did}/hashing/samples", json={"mapping": {}, "column": "bad/name"})
    assert r.status_code == 400
    r = await client.post(f"{API}/datasets/{did}/hashing/demux", json={"tags": tags, "method": "nope"})
    assert r.status_code == 400
    r = await client.post(f"{API}/datasets/{did}/hashing/demux", json={"tags": tags, "method": "hashedDrops"})
    assert r.status_code == 200 and r.json()["call"]["method"] == "hashedDrops"
