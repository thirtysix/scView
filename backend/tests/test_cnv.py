"""Copy-number inference from expression (core/cnv.py)."""

from __future__ import annotations

import base64
import gzip
import hashlib
import json

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp
from sklearn.metrics import adjusted_rand_score

from scview.core import claims, cnv, preflight, provenance, replay_script
from scview.core.anndata_adaptor import NOT_EMBEDDINGS

PER_CHROM = 80
CHROMS = [f"chr{i}" for i in range(1, 23)] + ["chrX"]
WINDOW, STEP = 51, 5


def _genes(seed: int = 0) -> pd.DataFrame:
    """PER_CHROM real human genes from each chromosome (shipped table), shuffled."""
    t = cnv.load_positions("human")
    t = t[(t["symbol"] != "") & ~t["symbol"].duplicated(keep=False)]
    rng = np.random.default_rng(seed)
    parts = [t[t["chromosome"] == c].sample(PER_CHROM, random_state=int(rng.integers(1 << 30)))
             for c in CHROMS]
    return pd.concat(parts).sample(frac=1.0, random_state=seed).reset_index(drop=True)


def _tumour(n_t: int = 120, n_b: int = 120, n_epi: int = 160, seed: int = 0, layout: str = "raw",
            gain: str | None = "chr7", loss: str | None = "chr10", ids: bool = False):
    """Counts for two normal types (T, B, each with its own marker genes) and an
    epithelial tumour carrying a gain (x2) of one chromosome and a loss (x0.5) of
    another. ``layout``: raw = scaled X + log .raw (scView's pipeline); x = log X;
    counts = counts only. ``ids``: symbols in var_names, versioned Ensembl ids in
    var['gene_ids']."""
    rng = np.random.default_rng(seed)
    genes = _genes(seed)
    g = len(genes)
    base = rng.lognormal(mean=-0.3, sigma=0.8, size=g)
    groups = np.array(["T"] * n_t + ["B"] * n_b + ["Epi"] * n_epi)
    rate = np.tile(base, (len(groups), 1))
    markers = rng.choice(g, 60, replace=False)
    rate[np.ix_(groups == "T", markers[:30])] *= 4
    rate[np.ix_(groups == "B", markers[30:])] *= 4
    if gain:
        rate[np.ix_(groups == "Epi", np.flatnonzero(genes["chromosome"] == gain))] *= 2.0
    if loss:
        rate[np.ix_(groups == "Epi", np.flatnonzero(genes["chromosome"] == loss))] *= 0.5
    lib = rng.uniform(0.7, 1.4, size=(len(groups), 1))
    counts = rng.poisson(rate * lib * 6).astype(np.float32)
    lognorm = np.log1p(counts / np.maximum(counts.sum(1, keepdims=True), 1) * 1e4).astype(np.float32)
    names = genes["symbol"].tolist()
    var = pd.DataFrame(index=names)
    if ids:
        var["gene_ids"] = [f"{x}.{i % 7 + 1}" for i, x in enumerate(genes["gene_id"])]
    obs = pd.DataFrame({"cell_type": pd.Categorical(groups, categories=["T", "B", "Epi"])},
                       index=[f"c{i}" for i in range(len(groups))])
    if layout == "raw":
        a = ad.AnnData(X=sp.csr_matrix(lognorm), obs=obs, var=var)
        a.raw = a.copy()
        a.X = ((lognorm - lognorm.mean(0)) / (lognorm.std(0) + 1e-9)).astype(np.float32)
    elif layout == "x":
        a = ad.AnnData(X=lognorm, obs=obs, var=var)
    else:
        a = ad.AnnData(X=sp.csr_matrix(counts), obs=obs, var=var)
    return a


def _run(a, refs=("T", "B"), **kw):
    kw.setdefault("window", WINDOW)
    kw.setdefault("step", STEP)
    p = cnv.plan(a, "cell_type", list(refs), **kw)
    return p, cnv.run(p)


# ---------------------------------------------------------------------------
# Gene positions
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("genome", ["human", "mouse"])
def test_shipped_tables_match_their_pinned_hash_and_shape(genome):
    path = cnv.RESOURCES / cnv.GENOMES[genome]["file"]
    assert hashlib.sha256(path.read_bytes()).hexdigest() == cnv.GENOMES[genome]["sha256"]
    with gzip.open(path, "rt") as f:
        assert f"Ensembl release {cnv.ENSEMBL_RELEASE} {cnv.GENOMES[genome]['assembly']}" in f.readline()
    t = cnv.load_positions(genome)
    assert list(t.columns) == ["symbol", "gene_id", "chromosome", "start", "end"]
    assert len(t) > 50_000 and set(t["chromosome"]) == set(cnv.chromosome_order(genome))
    assert (t["end"] >= t["start"]).all()
    known = {"human": ("EGFR", "chr7"), "mouse": ("Egfr", "chr11")}[genome]
    assert t.loc[t["symbol"] == known[0], "chromosome"].iloc[0] == known[1]


def test_genes_are_placed_by_ensembl_id_else_symbol_and_sorted():
    names = pd.Index(["EGFR", "egfr_alias", "PTEN", "NOTAGENE", "SRY", "MT-CO1", "Tp53"])
    ids = np.array([None, "ENSG00000146648", None, None, None, None, None], dtype=object)
    genes, match = cnv.place_genes(names, ids, "human", exclude=("chrY", "chrMT"))
    # egfr_alias found by its id; Tp53 by symbol ignoring case; SRY (chrY) and MT-CO1 excluded.
    assert match["by_id"] == 1 and match["by_symbol"] == 5 and match["unplaced"] == 1
    assert match["excluded"] == 2
    assert genes["gene"].tolist() == ["EGFR", "egfr_alias", "PTEN", "Tp53"]
    assert genes["chromosome"].tolist() == ["chr7", "chr7", "chr10", "chr17"]


def test_ensembl_ids_come_from_the_names_or_an_id_column():
    a = _tumour(ids=True, n_t=20, n_b=20, n_epi=20)
    ids = cnv.ensembl_ids(a, pd.Index(a.var_names))
    assert all(i is not None and "." not in i for i in ids)
    b = ad.AnnData(X=np.zeros((2, 2)), var=pd.DataFrame(index=["ENSG00000146648.18", "PTEN"]))
    assert cnv.ensembl_ids(b, pd.Index(b.var_names)).tolist() == ["ENSG00000146648", None]


# ---------------------------------------------------------------------------
# The method's parts
# ---------------------------------------------------------------------------


def test_window_weights_are_pyramids_per_chromosome():
    chroms = np.array(["chr1"] * 10 + ["chr2"] * 3, dtype=object)
    W, win = cnv.window_weights(chroms, 4, 2)
    W = W.toarray()
    # chr1: windows start at 0, 2, 4, 6 (whole windows only); chr2 is shorter than the window.
    assert win["chromosome"].tolist() == ["chr1"] * 4 + ["chr2"]
    assert win["first"].tolist() == [0, 2, 4, 6, 10] and win["last"].tolist() == [3, 5, 7, 9, 12]
    assert np.allclose(W.sum(axis=0), 1)
    assert np.allclose(W[0:4, 0], np.array([1, 2, 2, 1]) / 6)
    assert np.allclose(W[10:13, 4], 1 / 3)
    W5, _ = cnv.window_weights(np.array(["chr1"] * 5, dtype=object), 5 - 2, 1)
    assert np.allclose(W5.toarray()[0:3, 0], np.array([1, 2, 1]) / 4)


def test_relative_uses_the_bounded_difference_with_several_references():
    block = np.array([[1.0, 5.0, 0.0, 9.0]], dtype=np.float32)
    one = cnv.relative(block.copy(), np.array([[1.0, 1.0, 1.0, 1.0]], dtype=np.float32))
    assert one.tolist() == [[0.0, 3.0, -1.0, 3.0]]  # clipped at +-3
    ref = np.array([[0.5, 1.0, 1.0, 2.0], [1.5, 2.0, 0.5, 2.0]], dtype=np.float32)
    two = cnv.relative(block.copy(), ref)
    # 1.0 lies between 0.5 and 1.5: 0; 5 is 3 above 2; 0 is 0.5 below 0.5; 9 clipped.
    assert two.tolist() == [[0.0, 3.0, -0.5, 3.0]]


# ---------------------------------------------------------------------------
# Whole runs
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def tumour_run():
    a = _tumour()
    p, r = _run(a)
    return a, p, r


def test_planted_gain_and_loss_are_recovered(tumour_run):
    a, p, r = tumour_run
    epi = (a.obs["cell_type"] == "Epi").to_numpy()
    assert r.X.shape == (a.n_obs, p.W.shape[1]) and r.X.dtype == np.float32
    assert r.score[epi].mean() > 4 * r.score[~epi].mean()
    win_chr = p.windows["chromosome"].to_numpy()
    assert r.X[np.ix_(epi, win_chr == "chr7")].mean() > 0.1
    assert r.X[np.ix_(epi, win_chr == "chr10")].mean() < -0.1
    assert abs(r.X[np.ix_(epi, win_chr == "chr3")].mean()) < 0.03
    # The tumour cells form CNV clusters of their own.
    lab = np.asarray(r.labels)
    epi_clusters = set(lab[epi])
    assert not epi_clusters & set(lab[~epi])
    assert p.src.text == "log-normalised .raw"
    assert p.genome == "human" and p.match["by_symbol"] == len(a.var_names)
    assert "chrX" in set(win_chr) and "chrY" not in set(win_chr)


def test_no_tumour_gives_uniformly_low_scores():
    a = _tumour(gain=None, loss=None)
    _, r = _run(a)
    epi = (a.obs["cell_type"] == "Epi").to_numpy()
    assert r.score[epi].mean() < 1.5 * r.score[~epi].mean()


def test_blocks_of_cells_and_input_layouts_give_the_same_profile(tumour_run):
    a, p, r = tumour_run
    small = cnv.run(p, chunk=37)
    assert np.allclose(small.X, r.X, atol=1e-6) and np.allclose(small.score, r.score, atol=1e-7)
    for layout, text in (("x", "log-normalised X"), ("counts", "log1p of counts per 10,000 from X")):
        b = _tumour(layout=layout)
        pb, rb = _run(b)
        assert pb.src.text == text
        # Float32 rounding of the log values can move a value across the denoising cut.
        assert np.mean(np.abs(rb.X - r.X) > 1e-4) < 1e-3
        assert np.allclose(rb.score, r.score, atol=1e-4)


def test_threshold_off_keeps_every_value(tumour_run):
    a, _, r = tumour_run
    _, r0 = _run(a, threshold=None)
    assert r0.cut is None and np.count_nonzero(r0.X) > np.count_nonzero(r.X)
    assert np.isclose(r.cut, cnv.DEFAULT_THRESHOLD * r0.noise_sd, rtol=1e-4)


def test_one_reference_group_subtracts_its_mean():
    a = _tumour()
    _, r = _run(a, refs=("T",))
    epi = (a.obs["cell_type"] == "Epi").to_numpy()
    t = (a.obs["cell_type"] == "T").to_numpy()
    assert r.score[epi].mean() > 3 * r.score[t].mean()


def test_genes_detected_in_few_cells_are_dropped():
    a = _tumour()
    p0 = cnv.plan(a, "cell_type", ["T", "B"], window=WINDOW, step=STEP, min_frac=0.0)
    p1 = cnv.plan(a, "cell_type", ["T", "B"], window=WINDOW, step=STEP, min_frac=0.5)
    assert p0.n_low == 0 and p1.n_low > 0 and len(p1.genes) == len(p0.genes) - p1.n_low


@pytest.mark.parametrize("kw,msg", [
    ({"ref_column": "nope"}, "Choose the column"),
    ({"ref_groups": []}, "at least one reference group"),
    ({"ref_groups": ["Q"]}, "Not in 'cell_type': Q"),
    ({"ref_groups": ["T", "B", "Epi"]}, "Every cell is in the reference"),
    ({"window": 5}, "The window must be"),
    ({"step": 0}, "The step must be"),
    ({"exclude": ["chr99"]}, "Unknown chromosome"),
    ({"genome": "yeast"}, "Unknown genome"),
])
def test_bad_requests_say_why(kw, msg):
    a = _tumour(n_t=30, n_b=30, n_epi=30)
    args = {"ref_column": "cell_type", "ref_groups": ["T", "B"], **kw}
    col, groups = args.pop("ref_column"), args.pop("ref_groups")
    with pytest.raises(cnv.CnvError, match=msg):
        cnv.plan(a, col, groups, **args)


def test_small_reference_and_unknown_species_are_refused():
    a = _tumour(n_t=5, n_b=60, n_epi=60)
    with pytest.raises(cnv.CnvError, match="at least 10 cells each: T \\(5\\)"):
        cnv.plan(a, "cell_type", ["T", "B"])
    b = _tumour(n_t=30, n_b=30, n_epi=30)
    b.var_names = [f"g{i}" for i in range(b.n_vars)]
    b.raw = None
    with pytest.raises(cnv.CnvError, match="species"):
        cnv.plan(b, "cell_type", ["T", "B"])
    assert cnv.runnable(b)["ok"] is False


def test_mouse_genome_is_picked_from_the_symbols():
    a = _tumour(n_t=30, n_b=30, n_epi=30, layout="x")
    a.var_names = [s.capitalize() for s in a.var_names]
    genome, _ = cnv.pick_genome(a)
    assert genome == "mouse"
    assert cnv.pick_genome(a, "human")[0] == "human"


# ---------------------------------------------------------------------------
# Saved result, heatmap, co-pilot, replay
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def saved(tumour_run):
    a0, p, r = tumour_run
    a = a0.copy()
    cnv.save(a, p, r, fingerprint="fp")
    return a, p, r


def test_save_stores_obsm_obs_uns_and_provenance(saved):
    a, p, r = saved
    assert a.obsm[cnv.OBSM_KEY].shape == r.X.shape and cnv.OBSM_KEY in NOT_EMBEDDINGS
    assert a.obs[cnv.SCORE_COL].dtype == np.float32
    assert isinstance(a.obs[cnv.CLUSTER_COL].dtype, pd.CategoricalDtype)
    info = cnv.stored(a)
    assert info["reference_groups"] == ["T", "B"] and info["n_windows"] == p.W.shape[1]
    assert info["chromosomes"][0] == {"chromosome": "chr1", "start": 0,
                                      "end": int((p.windows["chromosome"] == "chr1").sum())}
    assert len(a.uns[cnv.UNS_KEY]["window_start"]) == info["n_windows"]
    json.loads(a.uns[cnv.UNS_KEY]["info"])
    step = provenance.read_provenance(a)["history"][-1]
    assert step["step"] == "cnv" and step["params"]["genes"] == p.genes["gene"].tolist()


def test_summary_marks_high_groups_and_clusters(saved):
    a, _, _ = saved
    s = cnv.summary(a)
    rows = {g["group"]: g for g in s["by_group"]["groups"]}
    assert rows["T"]["reference"] and not rows["Epi"]["reference"]
    assert rows["Epi"]["ratio"] > 4 and rows["Epi"]["q1"] <= rows["Epi"]["median"] <= rows["Epi"]["q3"]
    high = [c for c in s["clusters"] if c["high"]]
    assert high and all("chr7" in c["gains"] and "chr10" in c["losses"] for c in high)
    assert all(c["reference_share"] == 0 for c in high)
    assert cnv.summary(a, "nope")["by_group"] is None
    assert cnv.summary(ad.AnnData(X=np.zeros((3, 3)))) == {"available": False}


def test_heatmap_decodes_to_sampled_cells_by_genome(saved):
    a, _, _ = saved
    h = cnv.heatmap(a, max_cells=200, max_cols=60)
    codes = np.frombuffer(base64.b64decode(h["values"]), dtype=np.int8).reshape(h["n_rows"], h["n_cols"])
    assert h["n_cols"] <= 60 + len(CHROMS) and h["n_rows"] == sum(g["n_sampled"] for g in h["groups"])
    assert [g["group"] for g in h["groups"]] == ["T", "B", "Epi"]
    assert sum(g["n_total"] for g in h["groups"]) == a.n_obs and h["n_rows"] <= 220
    spans = h["chromosomes"]
    assert spans[0]["start"] == 0 and spans[-1]["end"] == h["n_cols"]
    assert all(x["end"] == y["start"] for x, y in zip(spans, spans[1:]))
    epi_rows = slice(h["n_rows"] - h["groups"][2]["n_sampled"], h["n_rows"])
    c7 = next(s for s in spans if s["chromosome"] == "chr7")
    assert codes[epi_rows, c7["start"]:c7["end"]].mean() > 20
    assert len(h["clusters"]) == h["n_rows"]
    with pytest.raises(cnv.CnvError):
        cnv.heatmap(ad.AnnData(X=np.zeros((3, 3))))


def test_grounding_line_names_the_reference_groups_and_pattern(saved):
    a, _, _ = saved
    lines, detail = cnv.grounding_lines(a)
    text = lines[1]
    assert "ref: T/B in 'cell_type'" in text and "Epi (mean" in text
    assert "chr7 gain" in text and "chr10 loss" in text and text.endswith(f"[{cnv.TAG}]")
    assert "not DNA" in text and "ref: T/B" in detail
    assert claims.resolve_tag("cnv", [cnv.TAG, "result:programs"]) == [cnv.TAG]
    assert cnv.grounding_lines(ad.AnnData(X=np.zeros((3, 3)))) == ([], "")


def test_replay_script_reproduces_the_saved_profile(saved):
    a, p, r = saved
    script = replay_script.build_script(a)
    assert "def infer_cnv(" in script and "cnv_x, cnv_score, cnv_leiden = infer_cnv(" in script
    assert replay_script.describe(a)[-1]["status"] == replay_script.EXACT
    ns: dict = {}
    head = script.split("adata = read_input(INPUT)")[0]
    exec(compile(head.replace("INPUT = sys.argv[1]", "INPUT = 'x' #"), "replay", "exec"), ns)  # noqa: S102
    X, score, labels = ns["infer_cnv"](a, p.genes["gene"].tolist(), p.genes["chromosome"].tolist(),
                                       "cell_type", ["T", "B"], window=WINDOW, step=STEP, source="raw")
    assert np.abs(X - r.X).max() < 1e-5 and np.allclose(score, r.score, atol=1e-6)
    assert adjusted_rand_score(labels, np.asarray(r.labels)) == 1.0
    snip = replay_script._cnv({"window": 100}, replay_script._Ctx(adata=a))
    assert snip.status == replay_script.NOT_REPLAYABLE


def test_preflight_need_covers_the_result_and_a_block():
    need = preflight.cnv_need(100_000, 15_000, 1_500)
    assert need > 100_000 * 1_500 * 4
    assert preflight.cnv_need(200_000, 15_000, 1_500) > need
    assert preflight.memory_level(10, 100) == "fits" and preflight.memory_level(10, None) == "unknown"
    assert cnv.estimate_seconds(10_000, 15_000, 1_500) > cnv.estimate_seconds(1_000, 15_000, 1_500)


def test_copilot_context_carries_the_fact(saved, tmp_path):
    from scview.core.anndata_adaptor import AnnDataAdaptor
    from scview.core.assistant import build_grounding_context

    a, _, _ = saved
    path = tmp_path / "cnv.h5ad"
    a.write_h5ad(path)
    context, sources = build_grounding_context(AnnDataAdaptor(str(path)), {"groups"})
    assert "## Copy number (inferred)" in context and "chr7 gain" in context
    assert any(s.ref == cnv.TAG for s in sources)
