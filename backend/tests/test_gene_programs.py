"""Gene programs: consensus NMF of the variable genes (core/gene_programs.py)."""

from __future__ import annotations

import json

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

from scview.core import gene_programs as gp
from scview.core import provenance, replay_script
from scview.core.anndata_adaptor import NOT_EMBEDDINGS

N_PROG, PER_PROG = 3, 25


def _planted(n: int = 600, g: int = 300, seed: int = 0, dense: bool = False, layout: str = "raw"):
    """Counts from 3 planted programs on disjoint gene sets, with known usages.

    Each cell's usages come from a Dirichlet(0.4) (so most cells mix programs); a
    program raises its 25 genes; the other genes are background noise. Returns the
    dataset, the true usage shares (cells x 3) and the planted gene sets.
    ``layout``: "raw" = scaled X + log values in .raw + counts layer (scView's
    pipeline); "x" = log values in X; "counts" = counts only; "scaled" = scaled X only."""
    rng = np.random.default_rng(seed)
    usage = rng.dirichlet(np.full(N_PROG, 0.4), size=n)
    genes = [f"G{i}" for i in range(g)]
    sets = [genes[i * PER_PROG:(i + 1) * PER_PROG] for i in range(N_PROG)]
    spectra = np.zeros((N_PROG, g))
    for p in range(N_PROG):
        spectra[p, p * PER_PROG:(p + 1) * PER_PROG] = rng.uniform(2.0, 4.0, PER_PROG)
    rate = 0.3 + usage @ spectra
    lib = rng.uniform(0.7, 1.4, size=(n, 1))
    counts = rng.poisson(rate * lib).astype(np.float32)
    totals = counts.sum(axis=1, keepdims=True)
    lognorm = np.log1p(counts / totals * 1e4).astype(np.float32)
    var = pd.DataFrame(index=genes)
    hvg = np.zeros(g, dtype=bool)
    hvg[: N_PROG * PER_PROG] = True
    hvg[rng.choice(np.arange(N_PROG * PER_PROG, g), 100, replace=False)] = True
    var["highly_variable"] = hvg
    var["dispersions_norm"] = np.where(hvg, rng.uniform(1, 5, g), np.nan)
    M = (lambda a: a) if dense else sp.csr_matrix
    obs = pd.DataFrame(index=[f"c{i}" for i in range(n)])
    obs["group"] = pd.Categorical(np.where(usage.argmax(axis=1) == 0, "a", np.where(usage.argmax(axis=1) == 1, "b", "c")))
    if layout == "raw":
        a = ad.AnnData(X=M(lognorm), obs=obs, var=var)
        a.layers["counts"] = M(counts)
        a.raw = a.copy()
        z = (lognorm - lognorm.mean(0)) / (lognorm.std(0) + 1e-9)
        a.X = z.astype(np.float32)
    elif layout == "x":
        a = ad.AnnData(X=M(lognorm), obs=obs, var=var)
    elif layout == "counts":
        a = ad.AnnData(X=M(counts), obs=obs, var=var)
    else:  # scaled only
        z = (lognorm - lognorm.mean(0)) / (lognorm.std(0) + 1e-9)
        a = ad.AnnData(X=z.astype(np.float32), obs=obs, var=var)
    return a, usage, sets


def _match(fit: gp.KFit, genes: list[str], sets: list[list[str]]):
    """For each planted program, the recovered program whose top genes overlap it most:
    (index, Jaccard)."""
    tops = gp.top_genes(fit.scores, genes, n=PER_PROG)
    out = []
    for s in sets:
        best = max(range(fit.k), key=lambda j: len(set(s) & {t["gene"] for t in tops[j]}))
        got = {t["gene"] for t in tops[best]}
        out.append((best, len(set(s) & got) / len(set(s) | got)))
    return out


@pytest.mark.parametrize("dense", [False, True])
def test_planted_programs_are_recovered_with_their_usages(dense):
    a, usage, sets = _planted(dense=dense)
    inp = gp.prepare_input(a)
    assert inp.source == "log-normalised .raw" and len(inp.genes) == 175 and inp.n_dropped == 0
    fit = gp.fit_k(inp, 3, [0, 1, 2, 3, 4])
    matches = _match(fit, inp.genes, sets)
    assert len({j for j, _ in matches}) == 3
    for (j, jac), p in zip(matches, range(N_PROG)):
        assert jac > 0.8, (p, jac)
        r = np.corrcoef(fit.usage[:, j], usage[:, p])[0, 1]
        assert r > 0.9, (p, r)
    # Shares sum to 1 per cell; every planted program is found in all 5 runs.
    assert np.allclose(fit.usage.sum(axis=1), 1.0, atol=1e-6)
    assert all(p["stable"] and p["runs_found"] == 5 and p["correlation"] > 0.95 for p in fit.programs)
    assert fit.silhouette is not None and fit.silhouette > 0.8
    assert 0 < fit.error < 1


def test_sparse_and_dense_inputs_give_the_same_programs():
    a_s, _, _ = _planted(dense=False)
    a_d, _, _ = _planted(dense=True)
    fs = gp.fit_k(gp.prepare_input(a_s), 3, [0, 1])
    fd = gp.fit_k(gp.prepare_input(a_d), 3, [0, 1])
    assert np.allclose(fs.usage, fd.usage, atol=1e-4)
    assert np.allclose(fs.spectra, fd.spectra, atol=1e-5)


def test_too_many_programs_split_one_and_are_flagged_unstable():
    """With k above the planted 3, the extra programs are not found the same way in
    every run: at least one is not stable, and the sweep's silhouette drops."""
    a, _, _ = _planted()
    inp = gp.prepare_input(a)
    k3 = gp.fit_k(inp, 3, list(range(5)))
    k7 = gp.fit_k(inp, 7, list(range(5)))
    assert k3.stats()["n_stable"] == 3
    assert k7.stats()["n_stable"] < 7
    assert k7.silhouette < k3.silhouette


def test_input_sources_and_the_scaled_x_refusal():
    a, _, _ = _planted(layout="x")
    assert gp.prepare_input(a).source == "log-normalised X"
    a, _, _ = _planted(layout="counts")
    inp = gp.prepare_input(a)
    assert inp.source.startswith("shifted log of the counts in X")
    # The shifted log by hand: log1p(counts / total over ALL genes * 1e4), then / sd.
    C = a.X.toarray()
    L = np.log1p(C / C.sum(1, keepdims=True) * 1e4)[:, a.var_names.get_indexer(inp.genes)]
    expect = L / L.std(axis=0, ddof=1)
    assert np.allclose(inp.X.toarray(), expect, atol=1e-4)
    a, _, _ = _planted(layout="scaled")
    with pytest.raises(gp.ProgramsError, match="scaled"):
        gp.prepare_input(a)
    assert gp.runnable(a)["ok"] is False and "scaled" in gp.runnable(a)["reason"]
    assert gp.runnable(_planted()[0]) == {"ok": True, "reason": None, "n_flagged": 175,
                                          "source": "log-normalised .raw"}


def test_no_flagged_genes_and_n_hvg_ranking():
    a, _, _ = _planted()
    names, n_flagged, notes = gp.select_genes(a, 60)
    ranked = a.var.loc[a.var["highly_variable"], "dispersions_norm"].sort_values(ascending=False)
    assert n_flagged == 175 and set(names) == set(ranked.index[:60])
    assert names == [g for g in a.var_names if g in set(names)]  # var order
    assert "dispersions_norm" in notes[0]
    _, _, notes = gp.select_genes(a, 500)
    assert "175" in notes[0]
    with pytest.raises(gp.ProgramsError, match="at least 50"):
        gp.select_genes(a, 10)
    b = a.copy()
    del b.var["highly_variable"]
    with pytest.raises(gp.ProgramsError, match="highly variable genes step"):
        gp.prepare_input(b)


def test_genes_without_variance_are_left_out_and_empty_cells_get_zero_usage():
    a, _, _ = _planted(layout="x")
    X = a.X.tolil()
    X[:, 0] = 0  # a flagged gene never expressed
    X[5, :] = 0  # a cell with nothing
    a.X = X.tocsr()
    inp = gp.prepare_input(a)
    assert inp.n_dropped == 1 and "G0" not in inp.genes and inp.n_empty_cells == 1
    fit = gp.fit_k(inp, 3, [0])
    assert np.all(fit.usage[5] == 0)
    assert fit.programs[0]["stable"] is False  # one run: no stability claim
    assert fit.silhouette is None


def test_gene_scores_match_least_squares_on_the_z_scores():
    rng = np.random.default_rng(3)
    X = sp.random(80, 12, density=0.4, random_state=4, format="csr", dtype=np.float64)
    U = gp.share_usage(rng.uniform(0, 1, size=(80, 3)))
    U[7] = 0  # no usage: left out of the fit
    keep = U.sum(1) > 0
    D = X.toarray()[keep]
    Z = (D - D.mean(0)) / D.std(0, ddof=1)
    beta, *_ = np.linalg.lstsq(U[keep], Z, rcond=None)
    assert np.allclose(gp.gene_scores(X, U), beta, atol=1e-8)


def test_relative_error_matches_the_dense_norm():
    rng = np.random.default_rng(5)
    X = sp.random(50, 20, density=0.3, random_state=6, format="csr")
    W, H = rng.uniform(0, 1, (50, 4)), rng.uniform(0, 1, (4, 20))
    D = X.toarray()
    expect = ((D - W @ H) ** 2).sum() / (D ** 2).sum()
    assert gp.relative_error(X, W, H) == pytest.approx(expect, rel=1e-10)


def test_consensus_takes_the_median_of_each_group_and_scores_it():
    base = np.array([[1.0, 0, 0, 1], [0, 2.0, 2.0, 0]])
    runs = [base, base[::-1] * 1.1, base + np.array([[0.1, 0, 0, 0], [0, 0, 0, 0.1]])]
    c = gp.consensus(runs, 2)
    # Each group has one spectrum from every run; its median (of unit-length rows),
    # scaled to sum 1, is the consensus.
    unit = lambda v: v / np.linalg.norm(v)  # noqa: E731
    first = np.median([unit(base[0]), unit(base[0] * 1.1), unit(base[0] + [0.1, 0, 0, 0])], axis=0)
    first /= first.sum()
    got = c.spectra[np.argmax(c.spectra[:, 0])]
    assert np.allclose(got, first)
    assert all(p["runs_found"] == 3 and p["n_members"] == 3 for p in c.programs)
    assert all(p["correlation"] > 0.99 for p in c.programs)


def test_clean_request_and_estimate():
    assert gp.clean_request(None, None, None) == (10, [0, 1, 2, 3, 4], [10])
    assert gp.clean_request(10, 3, [20, 5, 10, 15]) == (10, [0, 1, 2], [5, 15, 20, 10])
    for bad in ((1, 5, None), (40, 5, None), (10, 0, None), (10, 5, [1]), (10, 5, list(range(2, 12)))):
        with pytest.raises(gp.ProgramsError):
            gp.clean_request(*bad)
    X = sp.random(100, 100, density=0.1, format="csr")
    assert gp.estimate_seconds(X, [10], 5) > 0


def test_iter_programs_streams_progress_then_the_sweep_and_the_fit():
    a, _, _ = _planted()
    inp = gp.prepare_input(a)
    k, seeds, ks = gp.clean_request(3, 2, [2, 4])
    events = list(gp.iter_programs(inp, k, seeds, ks))
    kinds = [e["type"] for e in events]
    assert kinds.count("progress") == 6 and kinds.count("k_result") == 3 and kinds[-1] == "fit"
    assert [e["done"] for e in events if e["type"] == "progress"] == [1, 2, 3, 4, 5, 6]
    last = events[-1]
    assert last["fit"].k == 3 and [s["k"] for s in last["sweep"]] == [2, 3, 4]
    stop = iter([False, False, True])
    with pytest.raises(gp.ProgramsCancelled):
        list(gp.iter_programs(inp, k, seeds, ks, stop=lambda: next(stop)))


def _saved(a=None, k=3):
    a = a if a is not None else _planted()[0]
    inp = gp.prepare_input(a)
    fit = gp.fit_k(inp, k, [0, 1, 2])
    out = gp.save(a, fit, inp, sweep=[fit.stats()], fingerprint="fp")
    return a, fit, inp, out


def test_save_stores_usage_programs_and_provenance(tmp_path):
    a, fit, inp, out = _saved()
    assert out["available"] and out["k"] == 3 and [p["name"] for p in out["programs"]] == ["P1", "P2", "P3"]
    assert a.obsm[gp.OBSM_KEY].shape == (a.n_obs, 3) and a.obsm[gp.OBSM_KEY].dtype == np.float32
    assert set(NOT_EMBEDDINGS) == {gp.OBSM_KEY, gp.OBSM_RAW_KEY}
    p1 = out["programs"][0]
    assert len(p1["top_genes"]) <= gp.N_TOP and p1["top_genes"][0]["score"] > 0
    assert p1["mean_usage"] >= out["programs"][1]["mean_usage"]  # most used first
    assert sum(p["pct_top"] for p in out["programs"]) == pytest.approx(1.0, abs=1e-3)
    hist = provenance.read_provenance(a)["history"]
    assert hist[-1]["step"] == "gene_programs" and hist[-1]["params"]["genes"] == inp.genes
    # Round trip through h5ad.
    path = tmp_path / "p.h5ad"
    a.write_h5ad(path)
    b = ad.read_h5ad(path)
    assert gp.stored(b)["run_id"] == out["run_id"]
    assert np.allclose(gp.program_usage(b, 2), fit.usage[:, 1], atol=1e-6)
    assert gp.background_genes(b) == inp.genes
    with pytest.raises(gp.ProgramsError, match="no P4"):
        gp.program_usage(b, 4)


def test_summary_by_group_and_violin():
    a, fit, _, _ = _saved()
    s = gp.summary(a, "group")
    bg = s["by_group"]
    assert bg["groups"] == ["a", "b", "c"]
    labels = a.obs["group"].to_numpy()
    for j in range(3):
        for gi, g in enumerate(bg["groups"]):
            assert bg["mean"][j][gi] == pytest.approx(fit.usage[labels == g, j].mean(), abs=1e-4)
    assert bg["n_cells"] == [int((labels == g).sum()) for g in "abc"]
    assert gp.summary(a, "nope")["by_group"] is None
    v = gp.violin(a, fit.usage[:, 0], "group")
    assert sorted(v) == ["a", "b", "c"] and len(v["a"]) == int((labels == "a").sum())
    assert gp.summary(_planted()[0]) == {"available": False}


def test_stored_ignores_usage_that_no_longer_fits():
    a, _, _, _ = _saved()
    a.obsm[gp.OBSM_KEY] = a.obsm[gp.OBSM_KEY][:, :2]
    assert gp.stored(a) is None


def test_enrichment_cache_belongs_to_its_run():
    a, fit, inp, _ = _saved()
    rec = [{"term": "HALLMARK_X", "adjusted_pvalue": 0.001, "overlap_count": 5, "gene_count": 50},
           {"term": "HALLMARK_Y", "adjusted_pvalue": 0.5}]
    gp.remember_enrichment(a, 1, ["h.all"], rec)
    cache = gp.enrichment_cache(a)
    assert [t["term"] for t in cache["1"]["top"]] == ["HALLMARK_X"]  # adj. p < 0.05 only
    assert gp.summary(a)["programs"][0]["top_term"]["term"] == "HALLMARK_X"
    gp.save(a, fit, inp)  # a new run: its programs have no enrichment yet
    assert gp.enrichment_cache(a) == {}


def test_grounding_names_stable_programs_with_genes_and_terms():
    a, _, _, out = _saved()
    gp.remember_enrichment(a, 1, ["h.all"], [{"term": "HALLMARK_X", "adjusted_pvalue": 1e-4}])
    lines, detail = gp.grounding_lines(a)
    assert lines[0] == "\n## Gene programs (NMF)"
    top5 = ", ".join(g["gene"] for g in out["programs"][0]["top_genes"][:5])
    assert f"P1 ({top5};" in lines[1] and "HALLMARK_X" in lines[1] and "[result:programs]" in lines[1]
    assert "3 of 3 are stable" in detail
    assert gp.grounding_lines(_planted()[0]) == ([], "")


def test_replay_script_reproduces_the_saved_usage():
    a, fit, inp, _ = _saved()
    script = replay_script.build_script(a)
    assert "gene_programs(adata, PROGRAM_GENES, 3, [0, 1, 2], source='raw'" in script
    assert "exact reproduction is not guaranteed" in script
    assert replay_script.describe(a)[-1]["status"] == replay_script.EXACT
    # Run the script's helper on the same dataset: the same usages.
    ns: dict = {}
    head = script.split("adata = read_input(INPUT)")[0]
    exec(compile(head.replace("INPUT = sys.argv[1]", "INPUT = 'x' #"), "replay", "exec"), ns)  # noqa: S102
    share, usage, spectra = ns["gene_programs"](a, inp.genes, 3, [0, 1, 2], source="raw")
    assert np.allclose(share, fit.usage, atol=1e-5)
    assert np.allclose(spectra, fit.spectra, atol=1e-6)


def test_replay_of_counts_source_and_missing_record():
    a, _, _ = _planted(layout="counts")
    inp = gp.prepare_input(a)
    fit = gp.fit_k(inp, 3, [0, 1])
    gp.save(a, fit, inp)
    script = replay_script.build_script(a)
    assert "source='counts'" in script and "def find_counts(adata):" in script
    ns: dict = {}
    head = script.split("adata = read_input(INPUT)")[0]
    exec(compile(head.replace("INPUT = sys.argv[1]", "INPUT = 'x' #"), "replay", "exec"), ns)  # noqa: S102
    share, _, _ = ns["gene_programs"](a, inp.genes, 3, [0, 1], source="counts")
    assert np.allclose(share, fit.usage, atol=1e-5)
    snip = replay_script._gene_programs({"k": 3}, replay_script._Ctx(adata=a))
    assert snip.status == replay_script.NOT_REPLAYABLE


def test_info_json_is_plain_json():
    a, _, _, _ = _saved()
    info = json.loads(a.uns[gp.UNS_KEY]["info"])
    assert info["params"]["solver"] == gp.SOLVER and info["method"].startswith("Consensus NMF")


def test_write_derived_fast_path_rewrites_the_named_obsm_entries(tmp_path):
    from scview.core.obs_writes import uns_snapshot, write_derived

    a, _, _ = _planted(layout="x")
    path = str(tmp_path / "d.h5ad")
    a.obsm["X_umap"] = np.zeros((a.n_obs, 2), dtype=np.float32)
    a.write_h5ad(path)
    b = ad.read_h5ad(path)
    before = uns_snapshot(b)
    inp = gp.prepare_input(b)
    fit = gp.fit_k(inp, 3, [0])
    gp.save(b, fit, inp)
    write_derived(b, path, path, before, obsm_keys=(gp.OBSM_KEY, gp.OBSM_RAW_KEY))
    c = ad.read_h5ad(path)
    assert np.allclose(c.obsm[gp.OBSM_KEY], fit.usage, atol=1e-6) and "X_umap" in c.obsm
    assert gp.stored(c) is not None
    del c.obsm[gp.OBSM_RAW_KEY]
    write_derived(c, path, path, uns_snapshot(c), obsm_keys=(gp.OBSM_RAW_KEY,))
    assert gp.OBSM_RAW_KEY not in ad.read_h5ad(path).obsm


def test_write_derived_drops_an_index_name_that_clashes_with_a_column(tmp_path):
    """An uploaded h5ad (ovary) names its obs index "cell_id" and has a different
    "cell_id" column: anndata refuses to write that, so the first edit failed."""
    from scview.core.obs_writes import write_derived

    a, _, _ = _planted(layout="x")
    a.obs["cell_id"] = [f"other{i}" for i in range(a.n_obs)]
    a.obs.index.name = "cell_id"
    path = str(tmp_path / "e.h5ad")
    write_derived(a, path)
    b = ad.read_h5ad(path)
    assert list(b.obs_names) == list(a.obs_names) and b.obs["cell_id"].iloc[0] == "other0"
