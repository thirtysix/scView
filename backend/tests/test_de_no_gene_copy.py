"""Cell-level DE reads the prefilter's genes from the matrix as it is (api/v1/de.py
``_prepare`` + ``de_methods.TESTED``): cutting them out copied the tested matrix,
+1.4 GiB at 100k cells. The genes are cut from each block instead
(``matrix_genes.GeneSubset``), with the same results, bit for bit.
"""

from __future__ import annotations

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import scanpy as sc
import scipy.sparse as sp

from scview.api.v1 import de as de_api
from scview.core import de_methods, matrix_genes, rank_genes


def _lognorm(n: int = 300, g: int = 200, seed: int = 0) -> ad.AnnData:
    rng = np.random.default_rng(seed)
    mu = rng.gamma(0.3, 1.0, g)[None, :] * np.ones((n, 1))
    mu[: n // 3, :20] *= 6
    X = rng.poisson(mu).astype(np.float32)
    X[:, 150:] = 0
    X[rng.choice(n, 2, replace=False), 150:160] = 1  # rare genes, under the 10% cut
    a = ad.AnnData(sp.csr_matrix(X))
    a.var_names = [f"G{i}" for i in range(g)]
    sc.pp.normalize_total(a, target_sum=1e4)
    sc.pp.log1p(a)
    return a


def test_the_tested_matrix_is_the_datasets_own():
    a = _lognorm()
    sel = np.arange(a.n_obs) < 100
    test, pre = de_api._prepare(a, sel, ~sel, 0.1)
    assert not pre.keep.all() and pre.keep.any()
    assert test.X is a.X  # every cell: no row copy, and no column copy either
    assert test.var[de_methods.TESTED].to_numpy().tolist() == pre.keep.tolist()


@pytest.mark.parametrize("method", ["wilcoxon", "t-test", "logreg"])
def test_results_equal_testing_the_cut_out_genes(method):
    a = _lognorm()
    sel = np.arange(a.n_obs) < 100
    test, pre = de_api._prepare(a, sel, ~sel, 0.1)
    got = de_methods.run(test, method)
    cut = ad.AnnData(X=a.X[:, pre.keep].copy(), obs=test.obs.copy(),
                     var=pd.DataFrame(index=a.var_names[pre.keep]), uns=dict(test.uns))
    ref = de_methods.run(cut, method)
    assert got.genes.tolist() == ref.genes.tolist() == a.var_names[pre.keep].tolist()
    for f in ("scores", "logfoldchanges", "pvals", "pvals_adj"):
        g, r = getattr(got, f), getattr(ref, f)
        if r is None:
            assert g is None
        else:
            np.testing.assert_array_equal(g, r, err_msg=f)


def test_a_gene_mask_is_scanpys_mask_var_bit_for_bit():
    a = _lognorm()
    a.obs["g"] = pd.Categorical(np.where(np.arange(a.n_obs) < 100, "a", "b"))
    mask = np.zeros(a.n_vars, dtype=bool)
    mask[::3] = True
    ours, theirs = a.copy(), a.copy()
    assert rank_genes.rank_genes_groups(ours, "g", method="wilcoxon", mask_var=mask, pts=True) \
        == rank_genes.SPARSE
    sc.tl.rank_genes_groups(theirs, "g", method="wilcoxon", mask_var=mask, pts=True)
    r1, r2 = ours.uns["rank_genes_groups"], theirs.uns["rank_genes_groups"]
    for f in ("names", "scores", "pvals", "pvals_adj", "logfoldchanges"):
        for grp in ("a", "b"):
            np.testing.assert_array_equal(np.asarray(r1[f][grp]), np.asarray(r2[f][grp]), err_msg=f)
    pd.testing.assert_frame_equal(r1["pts"], r2["pts"])


def test_gene_subset_blocks_follow_the_kept_genes():
    M = sp.random(50, 40, density=0.3, format="csr", dtype=np.float32, random_state=1)
    keep = np.zeros(40, dtype=bool)
    keep[[1, 5, 6, 30, 39]] = True
    sub = matrix_genes.GeneSubset(matrix_genes.MatrixGenes(M, [f"G{i}" for i in range(40)]), keep)
    assert sub.n_genes == 5 and list(sub.var_names) == ["G1", "G5", "G6", "G30", "G39"]
    blocks = list(sub.blocks(max_entries=60))
    assert blocks[0][0] == 0 and blocks[-1][1] == 5
    assert all(b[1] == c[0] for b, c in zip(blocks, blocks[1:]))
    got = sp.hstack([b for _, _, b in blocks]).toarray()
    np.testing.assert_array_equal(got, M[:, keep].toarray())
    with pytest.raises(ValueError):
        matrix_genes.GeneSubset(matrix_genes.MatrixGenes(M, [f"G{i}" for i in range(40)]), keep[:10])
