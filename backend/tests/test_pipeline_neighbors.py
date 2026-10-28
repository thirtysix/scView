"""The neighbour graph above scanpy's exact-kNN size (8,192 cells), where it uses pynndescent.

pynndescent 0.7.0 returned csr_array graphs that scanpy 1.11.5 cannot read (no .getnnz):
the step failed on every real dataset while small test data, under 8,192 cells, passed."""

import anndata as ad
import numpy as np

from scview.core import pipeline


def test_neighbors_above_exact_knn_size():
    rng = np.random.default_rng(0)
    n = 9000
    a = ad.AnnData(np.zeros((n, 1), dtype=np.float32))
    a.obsm["X_pca"] = rng.normal(size=(n, 10)).astype(np.float32)
    pipeline._run_neighbors(a, pipeline.PipelineParams())
    conn = a.obsp["connectivities"]
    assert conn.shape == (n, n)
    assert conn.nnz > 0
