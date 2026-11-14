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


def test_de_genes_all_finds_a_target_whose_genes_are_not_variable():
    """Papalexi MYC (QA 2026-11-14): 12 DE genes among the 2,000 variable genes and no KO
    cells; Seurat tests every gene (de.assay RNA) and gives 34 and KO cells. Here A's
    knocked-down genes (0-9) are left out of the variable genes."""
    a, ko = planted()
    a.var["highly_variable"] = np.arange(a.n_vars) >= 10
    res_v, cols_v = mixscape.run(a, mixscape.Params(de_genes="variable", lda=False))
    res_a, cols_a = mixscape.run(a, mixscape.Params(lda=False))  # default: all
    by_v = {t["target"]: t for t in res_v.targets}
    by_a = {t["target"]: t for t in res_a.targets}
    assert res_v.n_genes == 50 and res_a.n_genes == 60 and res_a.params["de_genes"] == "all"
    assert by_v["A"]["n_ko"] < by_a["A"]["n_ko"]
    top = [g["gene"] for g in by_a["A"]["top_de_genes"]]
    assert sum(g in {f"G{i}" for i in range(10)} for g in top) >= 8  # the knocked-down genes lead
    glob = np.full(a.n_obs, "", dtype=object)
    glob[cols_a["rows"]] = cols_a[mixscape.GLOBAL]
    assert (glob[ko] == "KO").all() and "Seurat" in res_a.gene_source
    with pytest.raises(mixscape.MixscapeError, match="de_genes") as e:
        mixscape.plan(a, mixscape.Params(de_genes="hvg"))
    assert e.value.status == 422


def test_seurat_rule_min_pct_fold_change_and_bonferroni():
    """TopDEGenesMixscape: FindMarkers(min.pct = 0.1, logfc.threshold) and p_val_adj =
    Bonferroni over every gene of the assay."""
    rng = np.random.default_rng(3)
    n_c, n_t = 200, 100
    labels = np.array(["NT"] * n_c + ["A"] * n_t)
    X = np.zeros((n_c + n_t, 4))
    X[:, 0] = rng.poisson(5, n_c + n_t)  # up in A, detected everywhere
    X[n_c:, 0] += 10
    X[n_c:n_c + 8, 1] = 50  # up in 8% of A only: under min.pct
    X[:, 2] = rng.poisson(5, n_c + n_t)  # no change
    X[n_c:, 3] = rng.poisson(1.2, n_t)  # A only, low level
    L = np.log1p(X)
    pct, lmean = mixscape.group_stats(sp.csr_matrix(L), labels, ["NT", "A"])
    assert pct[1, 1] == pytest.approx(0.08) and pct[0, 3] == 0
    assert lmean[1, 0] == pytest.approx(np.log2(X[n_c:, 0].mean() + 1))
    names = np.array(["G0", "G1", "G2", "G3"])
    de = mixscape.de_genes(sp.csr_matrix(L, dtype=np.float32), labels, ["A"], "NT", names, 0.25, 0.05, 1,
                           rule=mixscape.DE_ALL, n_genes_total=4)["A"]
    assert set(de["gene"]) == {"G0", "G3"}  # G1: under min.pct; G2: no fold change
    assert de.set_index("gene").loc["G0", "logfc"] == pytest.approx(lmean[1, 0] - lmean[0, 0])
    dense = mixscape.de_genes(L.astype(np.float32), labels, ["A"], "NT", names, 0.25, 0.05, 1,
                              rule=mixscape.DE_ALL, n_genes_total=4)["A"]
    assert list(dense["gene"]) == list(de["gene"])
    # Bonferroni over every gene of the dataset (n_genes_total), not only those tested.
    more = mixscape.de_genes(sp.csr_matrix(L, dtype=np.float32), labels, ["A"], "NT", names, 0.25, 0.05, 1,
                             rule=mixscape.DE_ALL, n_genes_total=8)["A"].set_index("gene")
    for g, p in de.set_index("gene")["pval_adj"].items():
        assert more.loc[g, "pval_adj"] == pytest.approx(min(2 * p, 1.0))


def test_ko_share_per_guide_and_per_replicate():
    """Papalexi Fig. 3c (IRF1 guides 39-92% KO) needed a crosstab by hand (QA 2026-11-14)."""
    a, ko = planted()
    lab = a.obs["perturbation"].astype(str).to_numpy()
    idx = np.arange(a.n_obs)
    guide = np.where(lab == "NT", "NTg1", np.char.add(lab.astype(str), np.where(idx % 3 == 0, "g1", "g2")))
    a.obs["guide_ID"] = pd.Categorical(guide)
    res, cols = mixscape.run(a, mixscape.Params(split_by="rep", lda=False))
    assert res.guide_column == "guide_ID" and res.replicate_column == "rep"
    glob = np.full(a.n_obs, "", dtype=object)
    glob[cols["rows"]] = cols[mixscape.GLOBAL]
    rows = {(r["target"], r["guide"]): r for r in res.per_guide}
    assert set(rows) == {("A", "Ag1"), ("A", "Ag2"), ("B", "Bg1"), ("B", "Bg2")}
    m = guide == "Ag1"
    assert rows[("A", "Ag1")]["n_cells"] == m.sum() and rows[("A", "Ag1")]["n_ko"] == (glob[m] == "KO").sum()
    assert sum(r["n_ko"] for r in res.per_guide if r["target"] == "A") == {t["target"]: t for t in res.targets}["A"]["n_ko"]
    reps = {(r["target"], r["replicate"]): r["ko_fraction"] for r in res.per_replicate}
    m = (lab == "A") & (a.obs["rep"].astype(str).to_numpy() == "r1")
    assert reps[("A", "r1")] == pytest.approx((glob[m] == "KO").mean())
    # No guide column found and no split: no tables
    b, _ = planted()
    res_b, _ = mixscape.run(b, mixscape.Params(lda=False))
    assert res_b.guide_column is None and res_b.per_guide == [] and res_b.per_replicate == []
    # An explicit replicate column without a split
    res_c, _ = mixscape.run(b, mixscape.Params(lda=False, replicate_column="rep"))
    assert {r["replicate"] for r in res_c.per_replicate} == {"r1", "r2"}
    with pytest.raises(mixscape.MixscapeError, match="not found"):
        mixscape.plan(b, mixscape.Params(guide_column="nope"))


def test_guide_column_from_the_saved_assignment():
    from scview.core import crispr

    a, _ = planted()
    a.obs["guide_call"] = pd.Categorical(a.obs["perturbation"].astype(str) + "_1")
    crispr._write_state(a, {"call": {"columns": {"perturbation": "perturbation", "guide_call": "guide_call"}}})
    rows = np.arange(a.n_obs)
    assert mixscape.guide_column_for(a, "perturbation", rows) == "guide_call"
    assert mixscape.guide_column_for(a, "rep", rows) is None  # not this assignment's targets


def test_filler_and_control_labels_are_not_targets():
    """set_labels' "Unassigned" filler (18,430 Papalexi cells) ran as a target (QA 2026-11-14)."""
    a, _ = planted()
    lab = a.obs["perturbation"].astype(str).to_numpy().copy()
    lab[:20] = "NT"
    lab[150:180] = "Unassigned"
    lab[180:190] = "non-targeting"
    lab[190:196] = "Doublet"
    a.obs["perturbation"] = pd.Categorical(lab)
    pl = mixscape.plan(a, mixscape.Params())
    assert pl.targets == ["A", "B"]
    assert pl.excluded == {"Unassigned": {"n_cells": 30, "reason": "no guide"},
                           "non-targeting": {"n_cells": 10, "reason": "a control label, not 'NT'"},
                           "Doublet": {"n_cells": 6, "reason": "several guides"}}
    assert "Unassigned" not in set(pl.labels) and pl.to_dict()["excluded"] == pl.excluded
    res, _ = mixscape.run(a, mixscape.Params(lda=False))
    assert [t["target"] for t in res.targets] == ["A", "B"] and res.excluded == pl.excluded
    assert any(w.startswith("Not targets, left out: Unassigned (30 cells, no guide)") for w in res.warnings)


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


class _StoreAdaptor:
    """What the results store reads from an adaptor: a derived path, adata and cells hash."""

    def __init__(self, root: Path, adata: ad.AnnData) -> None:
        self.h5ad_path = str(root / "derived" / "dsmx" / "data.h5ad")
        self.adata = adata

    def cells_fingerprint(self) -> str:
        return str(self.adata.n_obs)


def test_reruns_stale_only_results_that_named_a_changed_column(tmp_path):
    """v0.79: a guide assignment and a Mixscape run are label steps. A new run stales a DE by
    ``mixscape_class`` when the classes changed, never a DE by another column; a new
    guide assignment stales the Mixscape run that read its target column."""
    from scview.core import crispr, results_store

    a, _ = planted()
    crispr.assign(a, "column", "perturbation", control="NT")
    labels = crispr.saved_call(a)["columns"]["perturbation"]
    assert labels == "perturbation_scview"
    ad_ = _StoreAdaptor(tmp_path, a)
    res, cols = mixscape.run(a, mixscape.Params(labels=labels, lda=False))
    mixscape.save(a, res, cols)
    mixscape.remember(ad_, res)
    results_store.keep(ad_, "de", {"v": 1}, params={"groupby": mixscape.CLASS, "group_a": "A KO"}).result()
    results_store.keep(ad_, "de", {"v": 2}, params={"groupby": "rep", "group_a": "r1"}, key="rep").result()

    def state():
        return [(e["kind"], e["key"], e["stale"], e.get("stale_reason"))
                for e in results_store.entries("dsmx", adaptor=ad_) if e["status"] == "primary"]

    assert all(not s for _, _, s, _ in state())
    # The same run again: the same classes, nothing stale.
    res2, cols2 = mixscape.run(a, mixscape.Params(labels=labels, lda=False))
    mixscape.save(a, res2, cols2)
    assert all(not s for _, _, s, _ in state())
    hist = [h["step"] for h in provenance.read_provenance(a)["history"]]
    assert hist.count("mixscape") == 2 and "crispr_assign" in hist
    # Classes changed (every target NP): the DE by mixscape_class and the earlier run go stale.
    res3, cols3 = mixscape.run(a, mixscape.Params(labels=labels, lda=False, min_de_genes=1000))
    mixscape.save(a, res3, cols3)
    s = {(k, key): (st, why) for k, key, st, why in state()}
    assert s[("de", "")] == (True, f"labels of '{mixscape.CLASS}' changed after this result")
    assert s[("de", "rep")] == (False, None)
    assert s[("mixscape", f"{labels}|NT")][0] is True
    # Re-run on the new classes, then re-assign the guides with B merged into A: the
    # Mixscape run read the target column, the DE by replicate did not.
    mixscape.remember(ad_, res3)
    results_store.flush()
    assert s[("de", "rep")] == (False, None)
    crispr.assign(a, "column", "perturbation", control="NT", mapping={"B": "A"})
    s = {(k, key): (st, why) for k, key, st, why in state()}
    assert s[("mixscape", f"{labels}|NT")] == (True, f"labels of '{labels}' changed after this result")
    assert s[("de", "rep")] == (False, None)
    # A data step still stales them all.
    provenance.record_step(a, step="normalization", tool="scanpy")
    assert all(st for _, _, st, _ in state())
