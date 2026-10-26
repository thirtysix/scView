"""Unwanted-variation factors from negative-control genes (RUVg).

Risso et al. 2014, Nature Biotechnology 32:896 (RUVSeq), written from the paper:
genes that should not respond to the condition (here: the genes a first pass
found least associated with it) still vary between samples, and that shared
variation is technical or otherwise unwanted. Its leading left singular vectors
over the samples, ``W``, go into the design as covariates.

Matrices are samples x genes. With ``nuisance`` None the control genes are
centred per gene, as RUVSeq's RUVg(center = TRUE) does; with a nuisance design
(e.g. the intercept plus donor columns of a paired design) they are first
residualised on it, so the factors describe variation the design does not
already hold rather than repeating the donor effect.
"""

from __future__ import annotations

import numpy as np

# Default number of empirical control genes and the share of tested genes they
# may take at most (a first pass on few genes keeps some signal out of them).
N_CONTROLS = 1000
MAX_CONTROL_SHARE = 0.5


def empirical_controls(pvals: np.ndarray, n: int = N_CONTROLS) -> np.ndarray:
    """Indices of the genes with the largest p-values: at most ``n``, and at most
    half of the genes."""
    p = np.where(np.isfinite(pvals), pvals, -1.0)
    n = int(min(n, np.floor(len(p) * MAX_CONTROL_SHARE)))
    if n < 2:
        return np.array([], dtype=int)
    return np.sort(np.argsort(-p, kind="stable")[:n])


def log_normalised(counts: np.ndarray, lib_size: np.ndarray) -> np.ndarray:
    """log(count scaled to the mean library size + 1), natural log as RUVSeq."""
    lib = np.asarray(lib_size, dtype=np.float64)
    return np.log(np.asarray(counts, dtype=np.float64) / lib[:, None] * lib.mean() + 1.0)


def ruvg_factors(
    log_counts: np.ndarray,
    controls: np.ndarray,
    k: int,
    nuisance: np.ndarray | None = None,
    tol: float = 1e-8,
) -> np.ndarray:
    """The ``k`` unwanted factors (samples x k) from ``log_counts`` (samples x
    genes) of the ``controls``; fewer when the controls hold fewer independent
    directions. Each factor's sign is fixed so its largest entry is positive."""
    Y = np.asarray(log_counts, dtype=np.float64)[:, controls]
    if nuisance is None:
        Yc = Y - Y.mean(axis=0, keepdims=True)
    else:
        N = np.asarray(nuisance, dtype=np.float64)
        beta, *_ = np.linalg.lstsq(N, Y, rcond=None)
        Yc = Y - N @ beta
    if Yc.size == 0:
        return np.zeros((Y.shape[0], 0))
    U, d, _ = np.linalg.svd(Yc, full_matrices=False)
    k = int(min(k, int((d > tol * max(d[0], 1.0)).sum())))
    W = U[:, :k]
    for j in range(k):
        if W[np.argmax(np.abs(W[:, j])), j] < 0:
            W[:, j] = -W[:, j]
    return W
