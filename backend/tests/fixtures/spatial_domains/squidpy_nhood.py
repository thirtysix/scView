"""Writes nhood_squidpy.json: squidpy 1.6.5's nhood_enrichment on the synthetic tissue, the
reference test_spatial_domains.py compares with. Run in a venv with squidpy==1.6.5 (not in the
image): python squidpy_nhood.py
"""

import json
from pathlib import Path

import numpy as np
import squidpy as sq
from synthetic import tissue

out = {}
for tech, grid in (("cells", None), ("visium", 6)):
    a = tissue(tech, n_side=30, seed=3, sections=2)
    if grid:
        sq.gr.spatial_neighbors(a, library_key="library_id", coord_type="grid", n_neighs=grid)
    else:
        sq.gr.spatial_neighbors(a, library_key="library_id", coord_type="generic", delaunay=True)
    z, count = sq.gr.nhood_enrichment(a, cluster_key="stripe", library_key="library_id", n_perms=1000, seed=0,
                                      copy=True, n_jobs=2, show_progress_bar=False)
    out[tech] = {"zscore": np.asarray(z).tolist(), "count": np.asarray(count).astype(int).tolist()}
Path(__file__).with_name("nhood_squidpy.json").write_text(json.dumps(out))
