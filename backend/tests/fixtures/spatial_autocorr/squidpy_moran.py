"""Regenerate squidpy_moran.csv: squidpy 1.6.5's defaults on synthetic.visium() and
synthetic.cells(). Needs squidpy (not part of the image), e.g. a throwaway venv:

    python -m venv /tmp/sq && /tmp/sq/bin/pip install squidpy==1.6.5 "setuptools<81"
    /tmp/sq/bin/python tests/fixtures/spatial_autocorr/squidpy_moran.py   # from backend/

Run: ``sq.gr.spatial_neighbors(adata, library_key="library_id")`` (coord_type from
squidpy's rule: grid when uns["spatial"] exists, else generic; n_neighs 6) then
``sq.gr.spatial_autocorr(adata, mode="moran")`` (genes: highly_variable when marked,
else all; analytic p, one-tailed, fdr_bh). Also ``connectivities`` counts per spot, to
check the graph itself.
"""

import importlib.util
from pathlib import Path

import pandas as pd
import squidpy as sq

here = Path(__file__).parent
spec = importlib.util.spec_from_file_location("synthetic", here / "synthetic.py")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

frames, degrees = [], []
for name, adata in (("visium", mod.visium()), ("cells", mod.cells())):
    sq.gr.spatial_neighbors(adata, library_key="library_id")
    df = sq.gr.spatial_autocorr(adata, mode="moran", copy=True)
    frames.append(pd.DataFrame({"dataset": name, "gene": df.index, "I": df["I"].to_numpy(),
                                "pval": df["pval_norm"].to_numpy(), "var_I": df["var_norm"].to_numpy(),
                                "pval_adj": df["pval_norm_fdr_bh"].to_numpy()}))
    g = adata.obsp["spatial_connectivities"]
    degrees.append(pd.DataFrame({"dataset": name, "spot": adata.obs_names,
                                 "degree": (g != 0).sum(axis=1).A1, "lib_cross": [
                                     int((adata.obs["library_id"].to_numpy()[g[i].indices]
                                          != adata.obs["library_id"].to_numpy()[i]).sum())
                                     for i in range(adata.n_obs)]}))
pd.concat(frames).to_csv(here / "squidpy_moran.csv", index=False, float_format="%.17g")
pd.concat(degrees).to_csv(here / "squidpy_degrees.csv", index=False)
print("written", here / "squidpy_moran.csv")
