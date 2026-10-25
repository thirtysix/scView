"""float16 uploads are widened to float32 before a pipeline run (core/pipeline.widen_half)."""

from __future__ import annotations

import anndata as ad
import numpy as np
import pandas as pd

from scview.core import pipeline


def _logged_half(n=120, m=60, seed=0):
    rng = np.random.default_rng(seed)
    counts = rng.poisson(3.0, size=(n, m)).astype(np.float32)
    counts[:, 0] += 200  # a gene whose log1p reaches ~5.3: expm1 then needs float32
    logged = np.log1p(counts / counts.sum(1, keepdims=True) * 1e4)
    return ad.AnnData(X=logged.astype(np.float16), obs=pd.DataFrame(index=[f"c{i}" for i in range(n)]),
                      var=pd.DataFrame(index=[f"G{i}" for i in range(m)]))


def test_widen_half_x_layers_raw():
    a = _logged_half()
    a.layers["extra"] = a.X.copy()
    a.raw = a.copy()
    pipeline.widen_half(a)
    assert a.X.dtype == np.float32 and a.layers["extra"].dtype == np.float32 and a.raw.X.dtype == np.float32


def test_hvg_runs_on_a_float16_upload():
    """Highly variable genes raised "infinity" on float16 log values (expm1 overflow)."""
    a = _logged_half()
    out, res = pipeline.run_pipeline(a, ["highly_variable_genes"])
    assert not res.errors, res.errors
    assert out.X.dtype == np.float32 and "highly_variable" in out.var
