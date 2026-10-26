"""Mixscape (core/mixscape.py): against pertpy on a Papalexi 2021 slice, and on
planted data."""

from __future__ import annotations

import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

from scview.core import mixscape, provenance, results_export

FIXTURES = Path(__file__).parent / "fixtures" / "mixscape"


def planted(seed: int = 0, n_nt: int = 150, n_a: int = 120, ko_share: float = 0.7, n_b: int = 80,
            n_genes: int = 60) -> tuple[ad.AnnData, np.ndarray]:
    """Log-normalised cells: target A knocks down genes 0-9 in ``ko_share`` of its
    cells (the rest look like controls), target B does nothing. Two replicates.
    Returns the data and the planted KO mask."""
    rng = np.random.default_rng(seed)
    n = n_nt + n_a + n_b
    labels = np.array(["NT"] * n_nt + ["A"] * n_a + ["B"] * n_b)
    base = rng.gamma(2.0, 1.0, n_genes)
    counts = rng.poisson(base * 4, (n, n_genes)).astype(float)
    ko = np.zeros(n, dtype=bool)
    a_idx = np.flatnonzero(labels == "A")
    ko[a_idx[: int(ko_share * n_a)]] = True
    counts[np.ix_(ko, np.arange(10))] = rng.poisson(0.3, (ko.sum(), 10))
    X = np.log1p(counts / counts.sum(1, keepdims=True) * 1e4).astype(np.float32)
    a = ad.AnnData(X=sp.csr_matrix(X), obs=pd.DataFrame({
        "perturbation": pd.Categorical(labels), "rep": pd.Categorical(np.where(np.arange(n) % 2, "r1", "r2"))},
        index=[f"c{i}" for i in range(n)]), var=pd.DataFrame(index=[f"G{i}" for i in range(n_genes)]))
    from sklearn.decomposition import PCA

    a.obsm["X_pca"] = PCA(15, random_state=0).fit_transform(X - X.mean(0)).astype(np.float32)
    return a, ko


def test_against_pertpy_on_a_papalexi_slice():
    """pertpy 1.0.3 on the same 510 cells (tests/fixtures/mixscape; exact neighbours,
    the control variance held in its E-step too): the same class for >= 99% of cells."""
    a = ad.read_h5ad(FIXTURES / "papalexi_subset.h5ad")
    exp = pd.read_csv(FIXTURES / "expected_pertpy.csv", index_col=0)
    res, cols = mixscape.run(a, mixscape.Params(labels="gene_target", control="NT", split_by="replicate"))
    names = a.obs_names[cols["rows"]]
    ref = exp.loc[names, "mixscape_class_global"].astype(str).to_numpy()
    assert (ref == cols[mixscape.GLOBAL]).mean() >= 0.99
    by = {t["target"]: t for t in res.targets}
    for t in ("IFNGR1", "JAK2", "CD86"):
        ref_ko = (exp.loc[exp["gene_target"] == t, "mixscape_class_global"] == "KO").mean()
        assert abs(by[t]["ko_fraction"] - ref_ko) <= 0.03, t
    assert by["CD86"]["n_ko"] == 0 and by["IFNGR1"]["n_de_genes"] >= 5
    p_ref = exp.loc[names, "mixscape_class_p_ko"].to_numpy()
    tgt = cols[mixscape.CLASS] != mixscape.NT
    assert np.corrcoef(p_ref[tgt], cols[mixscape.P_KO][tgt])[0, 1] > 0.98
    assert res.n_cells == 510 and res.n_control == 200 and res.lda is not None


def test_planted_ko_cells_are_found():
    a, ko = planted()
    res, cols = mixscape.run(a, mixscape.Params(split_by="rep"))
    glob = np.full(a.n_obs, "", dtype=object)
    glob[cols["rows"]] = cols[mixscape.GLOBAL]
    is_a = a.obs["perturbation"].to_numpy() == "A"
    assert (glob[ko] == "KO").all()
    assert (glob[is_a & ~ko] == "NP").mean() >= 0.95
    by = {t["target"]: t for t in res.targets}
    assert by["B"]["n_ko"] == 0 and "DE genes" in by["B"]["note"]
    assert by["A"]["n_de_genes"] >= 10 and {g["gene"] for g in by["A"]["top_de_genes"]} <= {f"G{i}" for i in range(60)}
    assert sum(by["A"]["posterior_hist"]["ko"]) == by["A"]["n_ko"]
    assert (glob[a.obs["perturbation"].to_numpy() == "NT"] == "NT").all()


def test_fixed_component_mixture():
    rng = np.random.default_rng(1)
    x = np.concatenate([rng.normal(0, 1, 400), rng.normal(5, 0.5, 200)])
    post, fit = mixscape.fit_mixture(x, 0.0, 1.0, 3.0, 4.0)
    assert fit["means"][0] == 0.0 and fit["sds"][0] == 1.0  # held
    assert fit["means"][1] == pytest.approx(5, abs=0.15) and fit["weights"][1] == pytest.approx(1 / 3, abs=0.03)
    assert (post[:400] < 0.5).mean() > 0.99 and (post[400:] > 0.5).all()


def test_save_writes_columns_and_record():
    a, _ = planted()
    a.obs["perturbation"] = a.obs["perturbation"].cat.add_categories(["multiple"])
    a.obs.loc["c0", "perturbation"] = "multiple"
    res, cols = mixscape.run(a, mixscape.Params(lda=False))
    out = mixscape.save(a, res, cols)
    assert out["summary"]["n_ko"] == res.summary()["n_ko"]
    cls = a.obs[mixscape.CLASS]
    assert pd.isna(cls.iloc[0]) and np.isnan(a.obs[mixscape.P_KO].iloc[0])  # left out
    assert list(a.obs[mixscape.GLOBAL].cat.categories) == ["KO", "NP", "NT"]
    assert set(cls.dropna().unique()) == {"NT", "A KO", "A NP", "B NP"}
    rec = json.loads(a.uns[mixscape.UNS_KEY])
    assert rec["labels"] == "perturbation" and "posterior_hist" not in rec["targets"][0]
    assert provenance.read_provenance(a)["history"][-1]["step"] == "mixscape"
    assert mixscape.saved_record(a)["n_targets"] == 2
    small = a[:-5].copy()
    with pytest.raises(mixscape.MixscapeError):
        mixscape.save(small, res, cols)
    df = results_export.table({"kind": "mixscape"}, res)
    assert list(df["target"]) == ["A", "B"] and "top_de_genes" in df.columns


def test_plan_refusals():
    a, _ = planted()
    with pytest.raises(mixscape.MixscapeError, match="not found"):
        mixscape.plan(a, mixscape.Params(labels="nope"))
    with pytest.raises(mixscape.MixscapeError, match="control"):
        mixscape.plan(a, mixscape.Params(control="zzz"))
    with pytest.raises(mixscape.MixscapeError, match="at least 200"):
        mixscape.plan(a, mixscape.Params(n_neighbors=200))
    b = a.copy()
    del b.obsm["X_pca"]
    with pytest.raises(mixscape.MixscapeError, match="PCA") as e:
        mixscape.plan(b, mixscape.Params())
    assert e.value.status == 422
    c = a.copy()
    c.obs["perturbation"] = c.obs["perturbation"].cat.add_categories(["tiny"])
    c.obs.loc[["c0", "c1"], "perturbation"] = "tiny"
    pl = mixscape.plan(c, mixscape.Params())
    assert pl.skipped == {"tiny": 2} and pl.targets == ["A", "B"]


def test_cancel_stops_between_steps():
    a, _ = planted()
    pl = mixscape.plan(a, mixscape.Params())
    calls = {"n": 0}

    def stop():
        calls["n"] += 1
        return calls["n"] > 1

    with pytest.raises(mixscape.MixscapeCancelled):
        for _ in mixscape.iter_mixscape(a, pl, stop=stop):
            pass


def test_screen_and_de_questions_route_to_the_groups_facet():
    # The Mixscape, pseudobulk and neighbourhood-DE facts sit under the "groups" facet.
    from scview.core.rag.router import heuristic_intent

    for q in ("Which knockouts worked?", "did the CRISPR screen work?",
              "which pseudobulk method did I use?", "which genes change in only part of the monocytes?"):
        intent = heuristic_intent(q)
        assert "data" in intent.sources and "groups" in intent.data_facets, q
