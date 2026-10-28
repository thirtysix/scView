"""Synthetic spatial datasets for test_spatial_autocorr.py and squidpy_moran.py
(numpy, pandas, anndata only, so squidpy's venv and the test image build the same data).

- ``visium()``: two sections of a Visium-like hexagonal grid (full-res pixels rounded
  to integers, a slight rotation, ~10% of spots missing), the second laid over the
  first (same place, a few pixels off) so a pooled graph would join them, plus one
  spot far from the rest in section A (no neighbour). 30 genes: gradients, blobs,
  a stripe pattern, noise; 20 marked highly variable; ``CONST`` is 0 everywhere.
- ``cells()``: imaging-like cells at random float positions in two sections, no
  ``uns["spatial"]`` (squidpy's generic kNN); 24 genes, none marked variable.
"""

from __future__ import annotations

import anndata as ad
import numpy as np
import pandas as pd


def _values(rng, coords, n_genes):
    """Log-normalised-looking values with zeros: Poisson counts on a spatial rate."""
    x = (coords[:, 0] - coords[:, 0].min()) / max(np.ptp(coords[:, 0]), 1.0)
    y = (coords[:, 1] - coords[:, 1].min()) / max(np.ptp(coords[:, 1]), 1.0)
    out = np.zeros((len(coords), n_genes))
    for g in range(n_genes):
        kind = g % 6
        if kind == 0:
            rate = 0.2 + 3 * x
        elif kind == 1:
            rate = 0.2 + 3 * y
        elif kind == 2:
            cx, cy = rng.uniform(0.2, 0.8, 2)
            rate = 0.1 + 4 * np.exp(-((x - cx) ** 2 + (y - cy) ** 2) / 0.02)
        elif kind == 3:
            rate = 0.3 + 2 * (np.sin(x * 25) > 0)
        else:
            rate = np.full(len(coords), rng.uniform(0.2, 2.0))
        counts = rng.poisson(rate * rng.uniform(0.5, 1.5))
        out[:, g] = np.log1p(counts * 2.0)
    return out


def visium(seed: int = 0) -> ad.AnnData:
    rng = np.random.default_rng(seed)
    spacing = 138.0
    theta = np.deg2rad(0.6)
    parts, rows, cols, libs = [], [], [], []
    for li, (lib, off) in enumerate((("A", (0.0, 0.0)), ("B", (7.0, -5.0)))):
        rr, cc = [], []
        for r in range(16):
            for c in range(r % 2, 40, 2):
                rr.append(r)
                cc.append(c)
        rr, cc = np.array(rr), np.array(cc)
        keep = rng.random(len(rr)) > 0.1
        rr, cc = rr[keep], cc[keep]
        gx, gy = cc * spacing / 2, rr * spacing * np.sqrt(3) / 2
        px = 1000 + off[0] + gx * np.cos(theta) - gy * np.sin(theta)
        py = 1500 + off[1] + gx * np.sin(theta) + gy * np.cos(theta)
        xy = np.rint(np.c_[px, py])
        if lib == "A":
            xy = np.vstack([xy, [9000.0, 9000.0]])
            rr, cc = np.r_[rr, 99], np.r_[cc, 99]
        parts.append(xy)
        rows.append(rr)
        cols.append(cc)
        libs += [lib] * len(xy)
    coords = np.vstack(parts)
    n_genes = 30
    X = _values(rng, coords, n_genes)
    names = [f"G{i:02d}" for i in range(n_genes - 1)] + ["CONST"]
    X[:, -1] = 0.0
    obs = pd.DataFrame({"library_id": pd.Categorical(libs, categories=["A", "B"]),
                        "array_row": np.concatenate(rows), "array_col": np.concatenate(cols)},
                       index=[f"spot{i}" for i in range(len(coords))])
    var = pd.DataFrame(index=names)
    hv = np.zeros(n_genes, dtype=bool)
    hv[rng.choice(n_genes - 1, 20, replace=False)] = True
    var["highly_variable"] = hv
    adata = ad.AnnData(X=X, obs=obs, var=var)
    adata.obsm["spatial"] = coords
    adata.uns["spatial"] = {lib: {"scalefactors": {"spot_diameter_fullres": 89.0}, "metadata": {}}
                            for lib in ("A", "B")}
    adata.uns["scview_spatial"] = {"technology": "visium", "unit": "spot"}
    return adata


def cells(seed: int = 1) -> ad.AnnData:
    rng = np.random.default_rng(seed)
    coords = np.vstack([rng.uniform(0, 500, (180, 2)), rng.uniform(0, 500, (150, 2))])
    libs = ["C"] * 180 + ["D"] * 150
    X = _values(rng, coords, 24)
    obs = pd.DataFrame({"library_id": pd.Categorical(libs, categories=["C", "D"])},
                       index=[f"cell{i}" for i in range(len(coords))])
    adata = ad.AnnData(X=X, obs=obs, var=pd.DataFrame(index=[f"H{i:02d}" for i in range(24)]))
    adata.obsm["spatial"] = coords
    return adata
