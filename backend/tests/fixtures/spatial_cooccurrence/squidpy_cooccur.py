"""Writes cooccur_squidpy.json: squidpy 1.6.5's co_occurrence on the synthetic tissue, the
reference test_spatial_cooccurrence.py compares with. Run with squidpy==1.6.5 (not in the
image), from this folder: python squidpy_cooccur.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import squidpy as sq

sys.path.insert(0, str(Path(__file__).parent.parent / "spatial_domains"))
from synthetic import tissue  # noqa: E402

out = {}
for name, tech, n_splits in (("cells", "cells", 1), ("visium", "visium", 1), ("cells_splits", "cells", 3)):
    a = tissue(tech, n_side=20, seed=3)
    occ, interval = sq.gr.co_occurrence(a, cluster_key="stripe", interval=20, copy=True, n_splits=n_splits,
                                        n_jobs=1, show_progress_bar=False)
    out[name] = {"occ": np.round(occ.astype(float), 6).tolist(), "interval": [float(v) for v in interval]}
Path(__file__).with_name("cooccur_squidpy.json").write_text(json.dumps(out))
