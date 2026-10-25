"""Expression-weighted kernel density on an embedding (C087), after Nebulosa
(Alquicira-Hernandez & Powell 2021, Bioinformatics 37:2485). Written from the
paper's method, not from the package (GPL-3).

A gene's density at a point is a 2D Gaussian kernel density of the cells'
embedding coordinates, each cell weighted by its expression of the gene:

    f(p) = sum_i w_i K_hx(p_x - x_i) K_hy(p_y - y_i) / sum_i w_i

so a region of many weakly expressing cells, or of a few strong ones among
non-expressing neighbours, shows up where a dropout-ridden colouring looks like
noise. Several genes' joint density is the product of their densities (each
divided by its own peak first, so the product is 0..1 and cannot underflow).

Bandwidth, one per axis from the coordinates alone (the same for every gene,
and so for every panel of a split view): the two-stage direct plug-in of Wand
& Jones (1994, Kernel Smoothing, §3.6), which is what ``ks::hpi`` gives for
one variable (via ``KernSmooth::dpik``: binned on 401 points with the maximum
truncated, scale the smaller of the SD and IQR/1.349). ``adjust`` multiplies
it, as Nebulosa's.

Evaluation is binned: the weights are linearly binned on a ``grid`` x ``grid``
lattice spanning the coordinates' range, convolved with the two 1D kernels
(dense G x G matrices: no wrap-around, no truncation), and each cell reads the
lattice by bilinear interpolation (within 0.3% of the exact sum at each cell
on a 256 lattice). Nebulosa 1.22.0 evaluates the sum exactly on a 100-point
lattice, gives each cell its lower-left lattice value, and its code applies
each weight twice (it sums w_i^2 K K); on Kang's tSNE the per-cell densities
here correlate with its ``calculate_density`` at r = 0.990-0.996 over six
marker genes, and the bandwidths are ``ks::hpi`` 1.15.3's to 1e-6.
About 25 ms for 100k cells and 0.4 s for a million (two cores).
"""

from __future__ import annotations

import json

import numpy as np
import pyarrow as pa

METADATA_KEY = b"scview"
MAX_GENES = 9
GRID = 256
# KernSmooth::dpik's binning grid.
HPI_GRID = 401


class DensityError(ValueError):
    """A request that cannot be served; ``status`` is the HTTP code."""

    def __init__(self, message: str, status: int = 400) -> None:
        super().__init__(message)
        self.status = status


# ---------------------------------------------------------------------------
# Bandwidth: Wand & Jones two-stage direct plug-in
# ---------------------------------------------------------------------------


def linbin(x: np.ndarray, a: float, b: float, m: int, *, truncate: bool = False) -> np.ndarray:
    """Linear binning of ``x`` on ``m`` equally spaced points from a to b: each
    value splits its unit weight between its two neighbouring points.
    ``truncate``: drop values outside [a, b), as KernSmooth's ``linbin`` does
    (which drops the maximum itself when b is the data's maximum)."""
    counts = np.zeros(m, dtype=np.float64)
    if b <= a:
        counts[0] = x.size
        return counts
    pos = (np.asarray(x, dtype=np.float64) - a) / ((b - a) / (m - 1))
    if truncate:
        pos = pos[(pos >= 0) & (pos < m - 1)]
    lo = np.clip(np.floor(pos).astype(np.int64), 0, m - 2)
    frac = pos - lo
    np.add.at(counts, lo, 1.0 - frac)
    np.add.at(counts, lo + 1, frac)
    return counts


def _hermite(r: int, u: np.ndarray) -> np.ndarray:
    """Probabilists' Hermite polynomial He_r(u): phi^(r)(u) = (-1)^r He_r(u) phi(u)."""
    h0, h1 = np.ones_like(u), u
    if r == 0:
        return h0
    for i in range(2, r + 1):
        h0, h1 = h1, u * h1 - (i - 1) * h0
    return h1


def bkfe(counts: np.ndarray, drv: int, h: float, a: float, b: float) -> float:
    """Binned estimate of the density functional psi_r = E f^(r)(X) (r = ``drv``,
    even): (1/n^2) sum_k sum_l c_k c_l phi_h^(r)(g_k - g_l), the kernel cut at
    (4 + r) bandwidths as KernSmooth's ``bkfe``."""
    m = counts.size
    n = counts.sum()
    delta = (b - a) / (m - 1)
    L = min(int(np.floor((4 + drv) * h / delta)), m)
    u = np.arange(L + 1) * delta / h
    kappa = _hermite(drv, u) * np.exp(-0.5 * u * u) / np.sqrt(2 * np.pi) / h ** (drv + 1)
    full = np.concatenate([kappa[:0:-1], kappa])  # lags -L..L
    conv = np.convolve(counts, full)[L:L + m]
    return float(np.dot(counts, conv) / (n * n))


def hpi(x: np.ndarray) -> float:
    """Two-stage plug-in bandwidth of a Gaussian kernel for ``x`` (``ks::hpi``,
    ``KernSmooth::dpik(level=2)``). Falls back to a tiny positive width for
    constant data."""
    x = np.asarray(x, dtype=np.float64)
    n = x.size
    sd = float(np.std(x, ddof=1)) if n > 1 else 0.0
    q1, q3 = np.quantile(x, [0.25, 0.75]) if n else (0.0, 0.0)
    iqr = (q3 - q1) / 1.3489795003921634  # qnorm(3/4) - qnorm(1/4)
    scale = min(sd, iqr) if iqr > 0 else sd
    if not np.isfinite(scale) or scale <= 0:
        return 1e-6
    mean = x.mean()
    sx = (x - mean) / scale
    sa, sb = float(sx.min()), float(sx.max())
    counts = linbin(sx, sa, sb, HPI_GRID, truncate=True)
    alpha = (2 * np.sqrt(2) ** 9 / (7 * n)) ** (1 / 9)
    psi6 = bkfe(counts, 6, alpha, sa, sb)
    alpha = (-3 * np.sqrt(2 / np.pi) / (psi6 * n)) ** (1 / 7)
    psi4 = bkfe(counts, 4, alpha, sa, sb)
    if not psi4 > 0:  # degenerate (e.g. two points): a normal-scale rule
        return scale * 1.06 * n ** (-0.2)
    return float(scale * (1 / (2 * np.sqrt(np.pi) * psi4 * n)) ** 0.2)


# ---------------------------------------------------------------------------
# Binned weighted 2D KDE
# ---------------------------------------------------------------------------


class Lattice:
    """The evaluation lattice of one embedding: its axes, the bandwidths, the
    two kernel matrices and each cell's bilinear position. Built once per
    request and shared by every gene."""

    def __init__(self, coords: np.ndarray, *, grid: int = GRID, adjust: float = 1.0) -> None:
        xy = np.asarray(coords, dtype=np.float64)[:, :2]
        self.ok = np.isfinite(xy).all(axis=1)
        pts = xy[self.ok]
        if pts.shape[0] < 2:
            raise DensityError("The embedding needs at least two cells with coordinates.")
        self.grid = grid
        self.h = np.array([hpi(pts[:, 0]), hpi(pts[:, 1])]) * adjust
        self.axes = []
        self.kernels = []
        self.lo, self.frac = [], []
        for d in range(2):
            a, b = float(pts[:, d].min()), float(pts[:, d].max())
            if b <= a:
                b = a + 1.0
            axis = np.linspace(a, b, grid)
            u = (axis[:, None] - axis[None, :]) / self.h[d]
            self.kernels.append(np.exp(-0.5 * u * u) / (np.sqrt(2 * np.pi) * self.h[d]))
            pos = (pts[:, d] - a) / ((b - a) / (grid - 1))
            lo = np.clip(np.floor(pos).astype(np.int64), 0, grid - 2)
            self.axes.append(axis)
            self.lo.append(lo)
            self.frac.append(pos - lo)
        ix, iy = self.lo
        fx, fy = self.frac
        g = grid
        self._idx = [ix * g + iy, (ix + 1) * g + iy, ix * g + iy + 1, (ix + 1) * g + iy + 1]
        self._wt = [(1 - fx) * (1 - fy), fx * (1 - fy), (1 - fx) * fy, fx * fy]

    def density(self, weights: np.ndarray) -> np.ndarray:
        """Per-cell weighted density (cells without coordinates: 0). Weights are
        the cells' values in the dataset's order; all zero gives all zero."""
        w = np.asarray(weights, dtype=np.float64)[self.ok]
        total = w.sum()
        out = np.zeros(self.ok.size, dtype=np.float64)
        if not total > 0:
            return out
        g = self.grid
        binned = np.zeros(g * g, dtype=np.float64)
        for idx, wt in zip(self._idx, self._wt):
            binned += np.bincount(idx, weights=w * wt, minlength=g * g)
        Kx, Ky = self.kernels
        z = (Kx @ binned.reshape(g, g) @ Ky.T) / total
        flat = z.ravel()
        vals = sum(flat[idx] * wt for idx, wt in zip(self._idx, self._wt))
        out[self.ok] = np.maximum(vals, 0.0)
        return out


def weighted_density(coords: np.ndarray, weights: np.ndarray, *, grid: int = GRID,
                     adjust: float = 1.0) -> tuple[np.ndarray, dict]:
    """Density per cell of each weight column (cells x genes) and their joint
    (the product of each gene's density over its peak; the gene's own density
    when there is one). Returns (joint, facts)."""
    W = np.asarray(weights, dtype=np.float64)
    if W.ndim == 1:
        W = W[:, None]
    lat = Lattice(coords, grid=grid, adjust=adjust)
    n_negative = int((W < 0).sum())
    W = np.where(np.isfinite(W) & (W > 0), W, 0.0)
    per_gene = [lat.density(W[:, j]) for j in range(W.shape[1])]
    if len(per_gene) == 1:
        joint = per_gene[0]
    else:
        joint = np.ones(W.shape[0], dtype=np.float64)
        for d in per_gene:
            peak = d.max()
            joint *= d / peak if peak > 0 else 0.0
    facts = {
        "bandwidth": [float(v) for v in lat.h],
        "grid": grid,
        "adjust": float(adjust),
        "n_expressing": [int((W[:, j] > 0).sum()) for j in range(W.shape[1])],
        "n_negative": n_negative,
        "n_no_coords": int((~lat.ok).sum()),
    }
    return joint.astype(np.float32), facts


def to_arrow_ipc(values: np.ndarray, meta: dict) -> bytes:
    """Arrow IPC stream of one float32 column ``density`` (one value per cell, in
    the dataset's order), the JSON ``meta`` in the schema metadata under ``scview``."""
    batch = pa.RecordBatch.from_arrays([pa.array(np.asarray(values, dtype=np.float32))], names=["density"])
    batch = batch.replace_schema_metadata({METADATA_KEY: json.dumps(meta).encode("utf-8")})
    sink = pa.BufferOutputStream()
    with pa.ipc.new_stream(sink, batch.schema) as writer:
        writer.write_batch(batch)
    return sink.getvalue().to_pybytes()
