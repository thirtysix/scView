"""Donor-aware pseudobulk DE (pydeseq2) for between-groups comparisons."""

from __future__ import annotations

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp
from fastapi import HTTPException

from scview.api.v1.de import DECompareRequest, _detect_replicate_key, differential_expression_compare
from scview.core import pseudobulk


class _DM:
    def __init__(self, adata):
        self._a = type("A", (), {"adata": adata})()

    async def get_or_load_dataset(self, _id):
        return self._a


def _donor_counts(
    donors_a: list[str],
    donors_b: list[str],
    cells: int = 30,
    g: int = 120,
    donor_col: str = "donor",
    seed: int = 0,
) -> ad.AnnData:
    """Negative-binomial counts for donor x condition samples, with a per-donor
    baseline. G0 is 4x higher in condition A; G1 is 4x higher in B."""
    rng = np.random.default_rng(seed)
    base = rng.gamma(2.0, 1.0, size=g) + 0.5  # per-gene mean count per cell
    rows, cond, donor = [], [], []
    for c, ds in (("A", donors_a), ("B", donors_b)):
        for d in ds:
            shift = rng.lognormal(0, 0.3, size=g)  # donor-specific baseline
            mu = np.tile(base * shift, (cells, 1))
            if c == "A":
                mu[:, 0] *= 4.0
            else:
                mu[:, 1] *= 4.0
            # NB(mean mu, dispersion 0.2) via gamma-Poisson.
            lam = rng.gamma(1 / 0.2, mu * 0.2)
            rows.append(rng.poisson(lam))
            cond += [c] * cells
            donor += [d] * cells
    a = ad.AnnData(sp.csr_matrix(np.vstack(rows).astype("float32")))
    a.var_names = [f"G{i}" for i in range(g)]
    a.obs_names = [f"c{i}" for i in range(a.n_obs)]
    a.obs["cond"] = pd.Categorical(cond)
    a.obs[donor_col] = pd.Categorical(donor)
    a.obs["ct"] = pd.Categorical(["X" if i % 2 else "Y" for i in range(a.n_obs)])
    return a


def _req(**kw) -> DECompareRequest:
    return DECompareRequest(split_by="cond", group_a="A", group_b="B", method="pseudobulk", **kw)


DONORS = [f"d{i}" for i in range(5)]


@pytest.mark.asyncio
async def test_pseudobulk_finds_planted_genes_paired():
    a = _donor_counts(DONORS, DONORS)
    res = await differential_expression_compare("x", _req(replicate_key="donor"), _DM(a))
    assert res.method == "pseudobulk"
    assert (res.n_samples_a, res.n_samples_b) == (5, 5)
    assert (res.n_selected, res.n_rest) == (150, 150)
    assert res.design == "~donor + condition"
    assert res.counts_source == "X"
    assert 0 < res.n_genes_tested <= a.n_vars
    by = {g.gene: g for g in res.genes}
    assert by["G0"].pval_adj < 1e-3 and by["G0"].logfoldchange == pytest.approx(2.0, abs=0.5)
    assert by["G1"].pval_adj < 1e-3 and by["G1"].logfoldchange == pytest.approx(-2.0, abs=0.5)
    others = [g for g in res.genes if g.gene not in ("G0", "G1")]
    assert sum(g.pval_adj < 0.05 for g in others) <= 3


@pytest.mark.asyncio
async def test_pseudobulk_unpaired_when_donors_nested_in_condition():
    a = _donor_counts(["a0", "a1", "a2"], ["b0", "b1", "b2"])
    res = await differential_expression_compare("x", _req(replicate_key="donor"), _DM(a))
    assert res.design == "~condition"
    by = {g.gene: g for g in res.genes}
    assert by["G0"].logfoldchange > 1 and by["G0"].pval_adj < 0.05


@pytest.mark.asyncio
async def test_pseudobulk_detects_ind_column_and_scopes_within():
    a = _donor_counts(DONORS, DONORS, donor_col="ind")
    assert _detect_replicate_key(a.obs, {"cond"}) == "ind"
    for name in ("orig.ident", "orig_ident"):  # Seurat's per-sample column, either spelling
        assert _detect_replicate_key(a.obs.rename(columns={"ind": name}), {"cond"}) == name
    res = await differential_expression_compare(
        "x", _req(within_column="ct", within_value="X"), _DM(a)
    )
    assert res.replicate_key == "ind"
    assert (res.n_selected, res.n_rest) == (75, 75)
    assert res.label == "A (X)"


@pytest.mark.asyncio
async def test_pseudobulk_rejects_too_few_replicates():
    a = _donor_counts(["d0"], ["d0", "d1"])
    with pytest.raises(HTTPException) as e:
        await differential_expression_compare("x", _req(replicate_key="donor"), _DM(a))
    assert e.value.status_code == 400
    assert "at least 2 donors" in e.value.detail and "A: 1, B: 2" in e.value.detail


@pytest.mark.asyncio
async def test_pseudobulk_drops_samples_with_few_cells():
    a = _donor_counts(DONORS, DONORS)
    # d4 keeps only 4 of its A cells: too few to be a sample.
    drop = (a.obs["donor"] == "d4") & (a.obs["cond"] == "A")
    keep = ~drop.to_numpy()
    keep[np.flatnonzero(drop.to_numpy())[:4]] = True
    res = await differential_expression_compare("x", _req(replicate_key="donor"), _DM(a[keep].copy()))
    assert (res.n_samples_a, res.n_samples_b) == (4, 5)
    assert res.dropped_samples == ["d4 (A, 4 cells)"]
    assert res.design == "~condition"  # d4 is no longer in both groups


@pytest.mark.asyncio
async def test_pseudobulk_needs_counts_and_a_donor_column():
    a = _donor_counts(DONORS, DONORS)
    lognorm = a.copy()
    lognorm.X = np.log1p(lognorm.X.toarray() / 3.0)
    with pytest.raises(HTTPException) as e:
        await differential_expression_compare("x", _req(replicate_key="donor"), _DM(lognorm))
    assert e.value.status_code == 400 and "raw counts" in e.value.detail

    # Counts kept in a layer are found even when X is normalised.
    lognorm.layers["counts"] = a.X.copy()
    res = await differential_expression_compare("x", _req(replicate_key="donor"), _DM(lognorm))
    assert res.counts_source == "layers['counts']"

    nodonor = a.copy()
    del nodonor.obs["donor"]
    with pytest.raises(HTTPException) as e:
        await differential_expression_compare("x", _req(), _DM(nodonor))
    assert e.value.status_code == 400 and "donor" in e.value.detail
    with pytest.raises(HTTPException) as e:
        await differential_expression_compare("x", _req(replicate_key="cond"), _DM(a))
    assert e.value.status_code == 400


@pytest.mark.asyncio
async def test_pseudobulk_caps_genes_tested(monkeypatch):
    monkeypatch.setattr(pseudobulk, "MAX_GENES", 40)
    a = _donor_counts(DONORS, DONORS)
    res = await differential_expression_compare("x", _req(replicate_key="donor"), _DM(a))
    assert res.n_genes_tested == 40
    assert len(res.genes) <= 40


def test_gene_filter_needs_counts_in_enough_samples():
    P = sp.csr_matrix(np.array([[50, 9, 10], [50, 9, 0], [50, 30, 10], [50, 30, 0]]))
    # Gene 1 has >=10 counts in 2 samples; gene 2 in 2 as well; gene 0 in all 4.
    assert list(pseudobulk._filter_genes(P, 2)) == [0, 1, 2]
    assert list(pseudobulk._filter_genes(P, 3)) == [0]


def test_detect_replicate_key_skips_a_sample_column_that_is_the_condition():
    # Kang's layout: orig_ident (first in obs) is CTRL/STIM, Donor holds the people.
    import pandas as pd

    obs = pd.DataFrame({
        "orig_ident": ["CTRL"] * 4 + ["STIM"] * 4,
        "stim": ["CTRL"] * 4 + ["STIM"] * 4,
        "Donor": ["d1", "d2", "d3", "d4"] * 2,
    })
    a = (obs["stim"] == "STIM").to_numpy()
    b = (obs["stim"] == "CTRL").to_numpy()
    assert _detect_replicate_key(obs, {"stim"}) == "Donor"
    assert _detect_replicate_key(obs, {"stim"}, a, b) == "Donor"
    # A better-named column with one donor per side loses to one with two per side.
    obs2 = obs.assign(Donor=["d1"] * 4 + ["d2"] * 4, sample=["s1", "s2"] * 4)
    assert _detect_replicate_key(obs2, {"stim"}) == "Donor"
    assert _detect_replicate_key(obs2, {"stim"}, a, b) == "sample"
