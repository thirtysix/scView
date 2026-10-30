"""Small synthetic tissues for test_spatial_domains.py and squidpy_nhood.py (numpy, pandas,
anndata only, so squidpy's venv and the test image build the same data): stripes of tissue
with their own profile, noisy enough that one observation alone says little.

- ``tissue(tech)``: "visium" (hexagonal grid), "visium_hd" (square grid, ``bin_um``),
  "xenium" and "cells" (random positions), each with obsm ``X_pca`` and ``spatial``, obs
  ``library_id``, ``stripe`` (the truth) and ``array_row`` / ``array_col`` on a grid.
"""

from __future__ import annotations

import anndata as ad
import numpy as np
import pandas as pd

STRIPES = 4


def _positions(tech: str, n_side: int, rng) -> tuple[np.ndarray, pd.DataFrame]:
    obs = pd.DataFrame()
    if tech == "visium":
        rows, cols = np.meshgrid(np.arange(n_side), np.arange(n_side), indexing="ij")
        x = (cols + 0.5 * (rows % 2)).ravel() * 100.0
        y = rows.ravel() * 100.0 * np.sqrt(3) / 2
        obs = pd.DataFrame({"array_row": rows.ravel(), "array_col": cols.ravel()})
    elif tech == "visium_hd":
        rows, cols = np.meshgrid(np.arange(n_side), np.arange(n_side), indexing="ij")
        x, y = cols.ravel() * 30.0, rows.ravel() * 30.0
        obs = pd.DataFrame({"array_row": rows.ravel(), "array_col": cols.ravel()})
    else:
        x, y = rng.uniform(0, 1000, n_side * n_side), rng.uniform(0, 1000, n_side * n_side)
    return np.column_stack([x, y]).astype(np.float64), obs


def tissue(tech: str = "visium", n_side: int = 30, noise: float = 3.0, seed: int = 0, sections: int = 1,
           bin_um: float = 8.0, n_pcs: int = 12) -> ad.AnnData:
    rng = np.random.default_rng(seed)
    coords, grid, libs = [], [], []
    for s in range(sections):
        c, g = _positions(tech, n_side, rng)
        coords.append(c)
        grid.append(g)
        libs += [f"s{s}"] * len(c)
    xy = np.concatenate(coords)
    n = len(xy)
    span = np.ptp(xy[:, 0]) or 1.0
    stripe = np.minimum((xy[:, 0] - xy[:, 0].min()) / span * STRIPES, STRIPES - 1).astype(int)
    centres = rng.normal(0, 1.0, (STRIPES, n_pcs))
    pcs = centres[stripe] + rng.normal(0, noise, (n, n_pcs))
    obs = pd.concat(grid, ignore_index=True) if len(grid[0].columns) else pd.DataFrame(index=range(n))
    obs.index = [f"o{i}" for i in range(n)]
    obs["library_id"] = pd.Categorical(libs)
    obs["stripe"] = pd.Categorical([f"t{i}" for i in stripe])
    a = ad.AnnData(X=np.zeros((n, 3), dtype=np.float32), obs=obs,
                   var=pd.DataFrame(index=["G0", "G1", "G2"]))
    a.obsm["spatial"] = xy
    a.obsm["X_pca"] = pcs.astype(np.float32)
    if tech in ("visium", "visium_hd"):
        sf = {"spot_diameter_fullres": 20.0, "tissue_hires_scalef": 0.1, "tissue_lowres_scalef": 0.05}
        if tech == "visium_hd":
            sf["bin_size_um"] = bin_um
        a.uns["spatial"] = {f"s{s}": {"images": {}, "scalefactors": sf, "metadata": {"source": "spaceranger"}}
                            for s in range(sections)}
    if tech == "xenium":
        a.uns["scview_spatial"] = {"technology": "xenium", "unit": "cell"}
    return a
