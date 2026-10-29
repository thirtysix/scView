"""Spot composition (core/spot_deconvolution.py): an RCTD-style fit of Visium spots to a
single-cell reference, on synthetic spots drawn from known profiles."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

from scview.core import provenance, replay_script
from scview.core import spot_deconvolution as sd

TYPES = ["B", "T", "Mono", "Stroma"]


def _profiles(n_genes: int = 400, n_types: int = 4, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    base = rng.gamma(0.6, 1.0, size=n_genes)
    prof = np.stack([base * np.exp(rng.normal(0, 1.2, n_genes) * (rng.random(n_genes) < 0.3))
                     for _ in range(n_types)])
    return prof / prof.sum(axis=1, keepdims=True)


def _reference(prof: np.ndarray, types=TYPES, cells: int = 60, seed: int = 1, names=None) -> ad.AnnData:
    rng = np.random.default_rng(seed)
    X, labels = [], []
    for k, t in enumerate(types):
        size = rng.lognormal(np.log(rng.uniform(2000, 5000)), 0.3, cells)
        X.append(rng.poisson(prof[k][None] * size[:, None]))
        labels += [t] * cells
    a = ad.AnnData(sp.csr_matrix(np.vstack(X).astype(np.float32)))
    a.obs_names = [f"c{i}" for i in range(a.n_obs)]
    a.var_names = names or [f"GENE{j}" for j in range(prof.shape[1])]
    a.obs["cell_type"] = pd.Categorical(labels)
    return a


def _spots(prof: np.ndarray, n: int = 150, seed: int = 2, depth: float = 3000, s_e: float = 0.3,
           platform: float = 0.0, libraries: int = 1) -> tuple[ad.AnnData, np.ndarray]:
    """Visium-like spots: Dirichlet(0.5) RNA shares of the profiles, Poisson-lognormal counts."""
    rng = np.random.default_rng(seed)
    truth = rng.dirichlet(np.full(prof.shape[0], 0.5), size=n)
    gamma = rng.normal(0, platform, prof.shape[1]) if platform else np.zeros(prof.shape[1])
    lam = depth * (truth @ prof) * np.exp(gamma)[None] * np.exp(rng.normal(0, s_e, (n, prof.shape[1])))
    a = ad.AnnData(sp.csr_matrix(rng.poisson(lam).astype(np.float32)))
    a.obs_names = [f"s{i}" for i in range(n)]
    a.var_names = [f"GENE{j}" for j in range(prof.shape[1])]
    side = int(np.ceil(np.sqrt(n)))
    a.obs["array_row"] = np.arange(n) // side
    a.obs["array_col"] = np.arange(n) % side
    libs = [f"lib{i % libraries + 1}" for i in range(n)]
    a.obs["library_id"] = pd.Categorical(libs)
    a.obsm["spatial"] = np.column_stack([a.obs["array_col"] * 100.0, a.obs["array_row"] * 100.0])
    a.uns["spatial"] = {lib: {"scalefactors": {"spot_diameter_fullres": 89.0}} for lib in sorted(set(libs))}
    a.uns["scview_spatial"] = {"technology": "visium", "unit": "spot"}
    return a, truth


def _prepare(ref, spots, **kw) -> sd.Prepared:
    return sd.prepare(ref, spots, celltype_column="cell_type", ref_info={"id": "ref", "name": "Ref"},
                      ref_fallback=[], spot_fallback=[], cluster_column=kw.pop("cluster_column", None), **kw)


def _fit(p: sd.Prepared, method: str = "rctd") -> dict:
    out = None
    for item in sd.iter_fit(p, method=method):
        if item["type"] == "fitted":
            out = item["out"]
    assert out is not None
    return out


@pytest.fixture(scope="module")
def clean():
    prof = _profiles()
    ref = _reference(prof)
    spots, truth = _spots(prof)
    p = _prepare(ref, spots)
    return prof, ref, spots, truth, p, {m: _fit(p, m) for m in sd.METHODS}


def test_recovers_the_shares_and_beats_nnls(clean):
    _prof, _ref, _spots_, truth, p, fits = clean
    assert p.types == sorted(TYPES, key=lambda t: t.lower()) or set(p.types) == set(TYPES)
    order = [p.types.index(t) for t in TYPES]
    m = {meth: sd.metrics(truth, f["beta"][:, order], TYPES) for meth, f in fits.items()}
    assert m["rctd"]["rmse"] < 0.05, m["rctd"]
    assert m["rctd"]["rmse"] < 0.8 * m["nnls"]["rmse"] and m["rctd"]["jsd"] < m["nnls"]["jsd"], m
    assert all(r["r"] > 0.95 for r in m["rctd"]["per_type"]), m["rctd"]["per_type"]
    assert 0.2 <= fits["rctd"]["s_e"] <= 0.4  # the spots' noise was 0.3
    assert fits["nnls"]["s_e"] is None
    beta = fits["rctd"]["beta"]
    assert np.allclose(beta.sum(axis=1), 1) and (beta >= 0).all()
    assert np.allclose(fits["rctd"]["cells"].sum(axis=1), 1)


def test_cell_shares_divide_by_the_reference_cell_sizes(clean):
    _prof, _ref, _spots_, _truth, p, fits = clean
    beta, cells = fits["rctd"]["beta"], fits["rctd"]["cells"]
    expect = beta / p.S[None, :]
    assert np.allclose(cells, expect / expect.sum(axis=1, keepdims=True))


def test_gene_sets_and_their_fallback(clean):
    _prof, _ref, _spots_, _truth, p, fits = clean
    out = fits["rctd"]
    assert set(out["fit"]) <= set(out["platform"]) and 50 <= out["fit"].size <= out["platform"].size
    # Excluded genes (mitochondrial, ribosomal) never enter.
    prof = _profiles()
    names = [f"GENE{j}" for j in range(prof.shape[1])]
    names[:20] = [f"MT-X{j}" for j in range(10)] + [f"RPL{j}" for j in range(10)]
    spots, _ = _spots(prof)
    spots.var_names = names
    p2 = _prepare(_reference(prof, names=names), spots)
    out2 = _fit(p2, "nnls")
    assert not (set(out2["platform"].tolist()) & set(range(20)))


def test_a_platform_effect_is_estimated_and_corrected():
    prof = _profiles(n_genes=1000, seed=3)
    ref = _reference(prof, seed=4)
    rmse = {}
    for name, plat in (("clean", 0.0), ("platform", 0.7)):
        spots, truth = _spots(prof, n=150, seed=5, depth=4000, platform=plat)
        p = _prepare(ref, spots)
        order = [p.types.index(t) for t in TYPES]
        for meth in sd.METHODS:
            out = _fit(p, meth)
            rmse[name, meth] = sd.metrics(truth, out["beta"][:, order], TYPES)["rmse"]
        if plat:
            s_g = out["s_g"]["lib1"]
            assert 0.55 < s_g < 0.85, s_g  # the genes' effect was N(0, 0.7^2)
        else:
            assert out["s_g"]["lib1"] < 0.2
    # Measured 1.5-1.7x on several seeds (an exact gamma gives 1.1x): the section-wide
    # bulk fit's error is what remains. The fit still beats NNLS under the effect.
    assert rmse["platform", "rctd"] <= 2.0 * rmse["clean", "rctd"], rmse
    assert rmse["platform", "rctd"] < rmse["platform", "nnls"], rmse


def test_without_the_platform_step_the_effect_costs_more(monkeypatch):
    prof = _profiles(n_genes=1000, seed=3)
    ref = _reference(prof, seed=4)
    spots, truth = _spots(prof, n=150, seed=5, depth=4000, platform=0.7)
    p = _prepare(ref, spots)
    order = [p.types.index(t) for t in TYPES]
    with_step = sd.metrics(truth, _fit(p)["beta"][:, order], TYPES)["rmse"]
    monkeypatch.setattr(sd, "sd_platform_effect", lambda mu, y, n: (np.zeros(mu.shape[1]), None, None))
    without = sd.metrics(truth, _fit(p)["beta"][:, order], TYPES)["rmse"]
    assert with_step < without, (with_step, without)


def test_a_type_missing_from_the_reference_goes_to_its_nearest_type():
    rng = np.random.default_rng(7)
    prof = _profiles(seed=8)
    # "Plasma": mostly B's RNA, the rest its own genes; the reference lacks it.
    own = rng.gamma(0.6, 1.0, prof.shape[1]) * (rng.random(prof.shape[1]) < 0.15)
    plasma = 0.65 * prof[0] + 0.35 * own / own.sum()
    full = np.vstack([prof, plasma])
    ref = _reference(prof, seed=9)
    spots, truth = _spots(full, n=240, seed=10)
    p = _prepare(ref, spots)
    out = _fit(p)
    beta = out["beta"]
    b = p.types.index("B")
    rich = truth[:, 4] > 0.6
    assert rich.sum() >= 10
    # Plasma-rich spots are read mostly as B, its nearest type.
    assert (beta[rich, b] - truth[rich, 0]).mean() > 0.3
    assert (beta[rich].argmax(axis=1) == b).mean() > 0.8
    # The shares of the present types keep their order (Spotless: RCTD keeps relative proportions).
    rest = [p.types.index(t) for t in ("T", "Mono", "Stroma")]
    r = np.corrcoef(truth[:, 1:4].ravel(), beta[:, rest].ravel())[0, 1]
    assert r > 0.9, r
    # The top share does not reliably fall (here 0.81 against 0.83 in pure spots): the
    # missing type's own genes are not in the reference's fit set, so its spots look like B.


def test_low_count_spots_are_not_fitted():
    prof = _profiles()
    spots, _ = _spots(prof, n=60)
    X = spots.X.tolil()
    for i in range(5):
        X[i, :] = 0
        X[i, i] = 50  # 50 counts: under MIN_SPOT_UMI
    spots.X = sp.csr_matrix(X)
    p = _prepare(_reference(prof), spots)
    for meth in sd.METHODS:
        out = _fit(p, meth)
        assert np.isnan(out["beta"][:5]).all() and np.isfinite(out["beta"][5:]).all()
        arr = sd.outputs(p, out)
        assert list(arr["top"][:5]) == [sd.LOW_COUNTS] * 5 and np.isnan(arr["top_share"][:5]).all()
        s = sd.summarize(p, out, method=meth)
        assert s["n_skipped"] == 5 and s["n_fitted"] == 55
        assert any("fewer than 100 counts" in n for n in s["notes"])


def test_genes_match_between_ensembl_ids_and_symbols():
    prof = _profiles()
    spots, _ = _spots(prof)
    ids = [f"ENSG{j:011d}" for j in range(prof.shape[1])]
    spots.var["gene_ids"] = ids
    ref = _reference(prof, names=[f"{g}.7" for g in ids])  # Ensembl ids with a version suffix
    p = _prepare(ref, spots)
    assert len(p.genes) == prof.shape[1] and p.genes[0] == "GENE0" and p.ref_genes[0] == "ENSG00000000000.7"
    assert p.info["renamed"] == "this dataset by its var 'gene_ids'"


def test_duplicate_symbols_keep_the_first():
    prof = _profiles()
    spots, _ = _spots(prof)
    names = list(spots.var_names)
    names[1] = names[0]
    spots.var_names = names
    p = _prepare(_reference(prof), spots)
    assert len(p.genes) == len(set(p.genes)) == prof.shape[1] - 1


def test_refusals():
    prof = _profiles()
    ref = _reference(prof)
    spots, _ = _spots(prof)
    # Another species by gene-name case.
    mouse = _reference(prof, names=[f"Gene{j}" for j in range(prof.shape[1])])
    with pytest.raises(sd.SpotDeconvolutionError, match="looks mouse and this dataset human"):
        _prepare(mouse, spots)
    # Not spots.
    cells = spots.copy()
    del cells.uns["scview_spatial"]
    del cells.obs["array_row"]
    with pytest.raises(sd.SpotDeconvolutionError, match="needs Visium spots"):
        _prepare(ref, cells)
    # Too few cells per type: one type left, nothing to split into.
    small = ref[np.r_[0:60, 60:70]].copy()
    with pytest.raises(sd.SpotDeconvolutionError, match="fewer than 2 types"):
        _prepare(small, spots)
    # Log values, no counts.
    logged = ref.copy()
    logged.X = sp.csr_matrix(np.log1p(logged.X.toarray()) / 3)
    with pytest.raises(sd.SpotDeconvolutionError, match="no raw counts"):
        _prepare(logged, spots)
    with pytest.raises(sd.SpotDeconvolutionError, match="no column"):
        sd.prepare(ref, spots, celltype_column="nope", ref_info={}, ref_fallback=[], spot_fallback=[],
                   cluster_column=None)
    with pytest.raises(sd.SpotDeconvolutionError, match="Unknown method"):
        next(sd.iter_fit(_prepare(ref, spots), method="magic"))


def test_small_types_are_left_out_and_listed():
    prof = _profiles()
    ref = _reference(prof)
    keep = np.r_[0:180, 180:190]  # Stroma: 10 cells
    p = _prepare(ref[keep].copy(), _spots(prof)[0])
    assert "Stroma" not in p.types and p.small_types == ["Stroma"]
    s = sd.summarize(p, _fit(p, "nnls"), method="nnls")
    assert "Left out of the reference" in s["notes"][0] and s["small_types"] == ["Stroma"]


def test_sections_get_their_own_platform_effect():
    prof = _profiles()
    spots, _ = _spots(prof, libraries=2)
    p = _prepare(_reference(prof), spots)
    out = _fit(p, "nnls")
    s = sd.summarize(p, out, method="nnls")
    assert [lib["id"] for lib in s["libraries"]] == ["lib1", "lib2"]
    assert all(lib["s_g"] is not None and len(lib["bulk"]) == 4 for lib in s["libraries"])


def test_save_writes_the_arrays_columns_summary_and_step(clean):
    _prof, ref, spots, _truth, _p, _fits = clean
    spots = spots.copy()
    spots.obs["cluster"] = pd.Categorical(np.where(np.arange(spots.n_obs) % 2, "1", "0"))
    p = _prepare(ref, spots, cluster_column="cluster")
    out = _fit(p)
    summary = sd.summarize(p, out, method="rctd")
    sd.save(spots, p, out, summary)
    assert spots.obsm[sd.OBSM_RNA].shape == (spots.n_obs, 4) and spots.obsm[sd.OBSM_RNA].dtype == np.float32
    assert set(spots.obs[sd.OBS_TOP].cat.categories) <= set(TYPES)
    assert np.allclose(spots.obs[sd.OBS_TOP_SHARE], spots.obsm[sd.OBSM_RNA].max(axis=1))
    s = sd.saved(spots)
    assert s is not None and s["types"] == p.types and set(s["clusters"]["rna"]) == {"0", "1"}
    assert len(s["clusters"]["rna"]["0"]) == 4
    assert json.loads(spots.uns[sd.UNS_KEY])["method"] == "rctd"
    step = provenance.read_provenance(spots)["history"][-1]
    assert step["step"] == sd.STEP and step["params"]["types"] == p.types and step["params"]["s_e"] == out["s_e"]
    assert len(step["params"]["genes"]) == out["platform"].size == len(step["params"]["ref_genes"])
    vals = sd.share_values(spots, p.types[0], "cell")
    assert np.allclose(vals, spots.obsm[sd.OBSM_CELLS][:, 0])
    with pytest.raises(sd.SpotDeconvolutionError, match="not one of"):
        sd.share_values(spots, "Nope")
    # The co-pilot's fact names the dominant types.
    class _A:
        adata = spots
    lines, detail = sd.grounding_lines(_A())
    assert lines[0].endswith(f"[{sd.TAG}]") and "Spots by dominant type" in "\n".join(lines)
    assert "4 types" in detail


def test_self_check_scores_both_methods(monkeypatch):
    monkeypatch.setattr(sd, "CHECK_SPOTS", 120)
    monkeypatch.setattr(sd, "CHECK_DEPTHS", (6000,))
    prof = _profiles()
    ref = _reference(prof, cells=80)
    p = _prepare(ref, _spots(prof)[0], check=True)
    events = list(sd.iter_check(p))
    assert [e["type"] for e in events][-1] == "checked"
    out = events[-1]["out"]
    runs = {r["method"]: r for r in out["runs"]}
    assert set(runs) == set(sd.METHODS) and out["n_spots"] == 120
    assert runs["rctd"]["rmse"] < runs["nnls"]["rmse"] and runs["rctd"]["jsd"] < runs["nnls"]["jsd"], runs
    assert runs["rctd"]["rmse"] < 0.1 and runs["rctd"]["aupr"] > 0.8
    assert len(runs["rctd"]["per_type"]) == 4 and runs["rctd"]["rmse_cells"] is not None


def test_metrics_on_known_values():
    truth = np.array([[0.5, 0.5, 0.0], [1.0, 0.0, 0.0], [0.0, 0.2, 0.8]])
    m = sd.metrics(truth, truth.copy(), ["a", "b", "c"])
    assert m["rmse"] == 0 and m["jsd"] == pytest.approx(0) and m["aupr"] == pytest.approx(1)
    est = np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.2, 0.8]])
    assert sd._jsd(truth[:1], est[:1])[0] == pytest.approx(0.311278, abs=1e-5)


def test_the_likelihood_table_matches_direct_evaluation():
    yvals = np.array([0, 1, 5, 40])
    tab = sd.sd_table(0.5, yvals)
    t = np.array([-3.0, 0.0, 1.7, 3.6])
    rows = np.arange(4)
    ll, lt, ltt = sd.sd_lookup(tab, rows, t, (0, 1, 2))
    assert np.allclose(ll, sd.sd_loglik_direct(yvals, t, 0.5), atol=1e-4)
    h = 1e-4  # derivatives: central differences of the direct value
    num = (sd.sd_loglik_direct(yvals, t + h, 0.5) - sd.sd_loglik_direct(yvals, t - h, 0.5)) / (2 * h)
    assert np.allclose(lt, num, atol=1e-3)
    num2 = (sd.sd_loglik_direct(yvals, t + 1e-3, 0.5) - 2 * sd.sd_loglik_direct(yvals, t, 0.5)
            + sd.sd_loglik_direct(yvals, t - 1e-3, 0.5)) / 1e-6
    assert np.allclose(ltt, num2, rtol=1e-2, atol=1e-2)


def test_replay_code_is_exact_and_runs(tmp_path, clean):
    _prof, ref, spots, _truth, _p, _fits = clean
    spots = spots.copy()
    p = _prepare(ref, spots)
    out = _fit(p)
    summary = sd.summarize(p, out, method="rctd")
    summary["reference"]["path"] = str(tmp_path / "reference.h5ad")
    sd.save(spots, p, out, summary)
    assert replay_script.describe(spots)[-1]["status"] == replay_script.EXACT
    inp = spots.copy()
    for key in (sd.OBSM_RNA, sd.OBSM_CELLS):
        del inp.obsm[key]
    del inp.obs[sd.OBS_TOP], inp.obs[sd.OBS_TOP_SHARE]
    inp.write_h5ad(tmp_path / "input.h5ad")
    ref.write_h5ad(tmp_path / "reference.h5ad")
    script = tmp_path / "replay.py"
    script.write_text(replay_script.build_script(spots, dataset_id="t", dataset_name="test"))
    r = subprocess.run([sys.executable, str(script), str(tmp_path / "input.h5ad"), str(tmp_path / "out.h5ad")],
                       cwd=tmp_path, capture_output=True, text=True, timeout=600)
    assert r.returncode == 0, r.stderr[-3000:]
    got = ad.read_h5ad(tmp_path / "out.h5ad")
    assert np.allclose(got.obsm[sd.OBSM_RNA], spots.obsm[sd.OBSM_RNA], atol=1e-6, equal_nan=True)
    assert (got.obs[sd.OBS_TOP].astype(str).to_numpy() == spots.obs[sd.OBS_TOP].astype(str).to_numpy()).all()


def test_replay_of_an_unreadable_record_is_a_comment():
    snip = replay_script._spot_deconvolution({"method": "rctd"}, replay_script._Ctx(adata=None))
    assert snip.status == replay_script.NOT_REPLAYABLE
    assert Path(sd.__file__).name == "spot_deconvolution.py"
