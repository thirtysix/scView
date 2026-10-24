"""PCA that would get no component says what to change (chunked and in memory).

Before, large-dataset mode said "PCA needs at least 2 cells and 2 variable genes (n x k)"
and the in-memory run passed n_comps 0 on to scanpy.
"""

from __future__ import annotations

import anndata as ad
import dask.array as da
import numpy as np
import pytest
import scipy.sparse as sp

from scview.core import chunked
from scview.core.pipeline import PipelineParams, run_pipeline


def test_the_message_names_the_setting_to_change():
    assert chunked.pca_size_problem(50, 1000, 2000) is None
    assert chunked.pca_size_problem(50, 2, 2) is None  # one component still fits
    assert "set N components (PCA step) to 1 or more" in chunked.pca_size_problem(0, 1000, 2000)
    assert "loosen the filtering thresholds" in chunked.pca_size_problem(50, 1, 2000)
    one = chunked.pca_size_problem(50, 1000, 1)
    assert "there is 1: run Variable Genes again (N top genes 2 or more)" in one
    assert "there are 0" in chunked.pca_size_problem(50, 1000, 0)


def test_chunked_pca_raises_it():
    X = da.from_array(sp.random(40, 6, density=0.5, format="csr", dtype=np.float32, random_state=0),
                      chunks=(10, 6), asarray=False)
    hvg = np.zeros(6, dtype=bool)
    hvg[2] = True
    with pytest.raises(ValueError, match="run Variable Genes again"):
        chunked.pca_scaled(X, hvg, n_comps=20)
    with pytest.raises(ValueError, match="set N components"):
        chunked.pca_scaled(X, np.ones(6, dtype=bool), n_comps=0)


def test_the_in_memory_step_says_it_too():
    rng = np.random.default_rng(0)
    a = ad.AnnData(sp.csr_matrix(rng.poisson(2.0, (60, 30)).astype(np.float32)))
    a.var_names = [f"G{i}" for i in range(30)]
    a.var["highly_variable"] = np.arange(30) == 4
    _, res = run_pipeline(a, ["pca"], PipelineParams(n_comps=20))
    assert "there is 1: run Variable Genes again" in res.errors["pca"]
