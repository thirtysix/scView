"""Scrublet on the counts alone: the scores ``sc.pp.scrublet`` gives, without its copies.

``sc.pp.scrublet`` copies the whole AnnData it is given (X, layers, .raw, obsm, obsp),
and then the copy's layers again each time it subsets genes, though it reads only X:
ovary_raw (37k cells; after a reset, a dense 3 GB X beside its dense 3 GB counts layer)
was killed at 18 GiB in doublet detection. ``scrublet_counts`` runs the steps of
scanpy 1.11's ``sc.pp.scrublet`` in its order (filter genes and cells, normalise, log,
variable genes, simulate doublets from the counts, normalise both, then
``sc.pp.scrublet(adata_sim=...)`` for the scoring), with its seed, on a sparse matrix of
the counts and nothing else:

- dense counts are made CSR a block of rows at a time (``dense_rows_to_csr``), in their
  own dtype, so no second dense copy is ever made;
- once the variable genes are chosen, the observed cells' matrix (cells x variable
  genes, small) is made dense again when the counts were dense: the second
  normalisation sums non-integer values, and numpy's dense sum rounds differently from
  a sparse one (paul15: scores up to 9e-4 apart, the same calls). The steps before it
  sum integers, or add in row order in float64, which is exact either way.

So the scores, calls, threshold and simulated scores are bitwise the ones
``sc.pp.scrublet`` gives on the whole dataset (``tests/test_scrublet_lean.py`` compares
them on dense, sparse and integer counts, so a scanpy upgrade that changes its steps
fails there). The code uses numpy, pandas, scipy, anndata and scanpy only: the replay
script copies its source (``REPLAY_CONSTANTS``, ``REPLAY_CODE``).
"""

from __future__ import annotations

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp

# Rows of a dense counts matrix made sparse at a time (80 MB of float32 at 20,000 genes).
CSR_ROWS = 1024


def dense_rows_to_csr(M, rows=CSR_ROWS):
    """A dense matrix (in memory or on disk) as CSR in its own dtype, ``rows`` rows at a time.

    ``sp.csr_matrix(M)`` holds a mask and int64 coordinates of every non-zero at once;
    this holds one block of rows besides the result. The same values and order (sorted
    indices, no stored zeros) as ``sp.csr_matrix(M)``."""
    n, m = M.shape
    per_row = np.zeros(n + 1, dtype=np.int64)
    for i in range(0, n, rows):
        per_row[i + 1:i + 1 + rows] = np.count_nonzero(np.asarray(M[i:i + rows]), axis=1)
    indptr = np.cumsum(per_row)
    nnz = int(indptr[-1])
    idx = np.int32 if max(nnz, m) <= np.iinfo(np.int32).max else np.int64
    data = np.empty(nnz, dtype=M.dtype)
    indices = np.empty(nnz, dtype=idx)
    for i in range(0, n, rows):
        block = np.asarray(M[i:i + rows])
        r, c = np.nonzero(block)
        a, b = indptr[i], indptr[min(i + rows, n)]
        data[a:b] = block[r, c]
        indices[a:b] = c
    return sp.csr_matrix((data, indices, indptr.astype(idx)), shape=(n, m))


def scrublet_counts(counts, expected_doublet_rate=0.05, use_approx_neighbors=None, owned=False):
    """Scrublet on ``counts`` (cells x genes; sparse, dense or an on-disk matrix).

    Returns (doublet_score, predicted_doublet, uns) with one value per cell in order, as
    ``sc.pp.scrublet`` writes them on an AnnData holding these counts (a cell it filters
    out for fewer than 3 genes: NaN), and its ``uns["scrublet"]``. ``owned``: a sparse
    ``counts`` nobody else holds, worked on instead of copied (Scrublet normalises it in
    place)."""
    dense = not sp.issparse(counts)
    if not dense:
        X = counts if owned else counts.copy()
    elif callable(getattr(counts, "to_memory", None)):  # an on-disk sparse matrix
        X, dense = counts.to_memory(), False
    else:
        X = dense_rows_to_csr(counts)
    n, m = X.shape
    a = ad.AnnData(X=X, obs=pd.DataFrame({"row": np.arange(n)}, index=pd.RangeIndex(n).astype(str)),
                   var=pd.DataFrame(index=pd.RangeIndex(m).astype(str)))
    del X  # freed by the first subset below
    sc.pp.filter_genes(a, min_cells=3)
    sc.pp.filter_cells(a, min_genes=3)
    # Doublets are simulated from the counts, on the genes picked from the normalised data.
    a.layers["raw"] = a.X.copy()
    sc.pp.normalize_total(a)
    a.layers["log1p"] = a.X.copy()
    sc.pp.log1p(a, layer="log1p")
    sc.pp.highly_variable_genes(a, layer="log1p")
    del a.layers["log1p"]
    a = a[:, a.var["highly_variable"].to_numpy()].copy()
    sim = sc.pp.scrublet_simulate_doublets(a, layer="raw", sim_doublet_ratio=2.0,
                                           synthetic_doublet_umi_subsampling=1.0, random_seed=0)
    del a.layers["raw"]
    if dense:  # sums of non-integer values from here on: as sc.pp.scrublet had them
        a.X = a.X.toarray()
    sc.pp.normalize_total(a, target_sum=1e6)
    sc.pp.normalize_total(sim, target_sum=1e6)
    sc.pp.scrublet(a, adata_sim=sim, expected_doublet_rate=expected_doublet_rate,
                   use_approx_neighbors=use_approx_neighbors, random_state=0)
    rows = a.obs["row"].to_numpy()
    score = pd.Series(a.obs["doublet_score"].to_numpy(), index=rows).reindex(range(n))
    call = pd.Series(a.obs["predicted_doublet"].to_numpy(), index=rows).reindex(range(n))
    return score.to_numpy(), call.to_numpy(), a.uns["scrublet"]


REPLAY_CONSTANTS = ("CSR_ROWS",)
REPLAY_CODE = (dense_rows_to_csr, scrublet_counts)
