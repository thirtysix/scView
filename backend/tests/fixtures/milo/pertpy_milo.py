"""Regenerate pertpy_milo.tsv: pertpy 1.0.3 Milo (pydeseq2 solver) on
test_milo.synthetic(). Needs pertpy installed (not part of the image):

    python tests/fixtures/milo/pertpy_milo.py   # from backend/, src on PYTHONPATH
"""

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pertpy as pt

here = Path(__file__).parent
spec = importlib.util.spec_from_file_location("test_milo", here.parent.parent / "test_milo.py")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

adata = mod.synthetic()
adata.obs["sample"] = adata.obs["cond"].astype(str) + "__" + adata.obs["donor"].astype(str)
for c in ("cond", "donor"):
    adata.obs[c] = adata.obs[c].astype(str)
m = pt.tl.Milo()
mdata = m.load(adata)
m.make_nhoods(mdata["rna"], prop=0.1, seed=0)
mdata = m.count_nhoods(mdata, sample_col="sample")
m.da_nhoods(mdata, design="~donor+cond", solver="pydeseq2")
m.annotate_nhoods(mdata, anno_col="type")
var = mdata["milo"].var
pd.DataFrame({
    "index_cell": var["index_cell"].astype(str).values,
    "label": var["nhood_annotation"].astype(str).values,
    "label_fraction": np.round(var["nhood_annotation_frac"].values, 6),
    "logFC": var["logFC"].values, "PValue": var["PValue"].values,
    "SpatialFDR": var["SpatialFDR"].values,
}).to_csv(here / "pertpy_milo.tsv", sep="\t", index=False, float_format="%.8g")
