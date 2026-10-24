"""A second doublet detector: an scDblFinder-style classifier, written from the paper.

Germain, Lun, Garcia Meixide, Macnair and Robinson (2021), "Doublet identification in
single-cell sequencing data using scDblFinder", F1000Research 10:979
(doi:10.12688/f1000research.73600.2). scDblFinder itself is GPL-3 R code; this module
follows the method as the paper describes it and copies none of it:

1. a quick clustering of the cells (k-means on the principal components of the
   log-normalised counts);
2. artificial doublets: the summed counts of two random cells, two thirds of them of
   DIFFERENT clusters (heterotypic doublets, the ones that look like a cell state that
   is not there; two cells of one type look like one cell of it, and no method on
   expression tells them apart well), a third of any two;
3. the real cells and the artificial doublets together on the most expressed genes of
   each cluster, log-normalised, principal components, and each one's nearest
   neighbours there;
4. per cell, features: the share of artificial doublets among its 3 / 10 / 25 / 50
   nearest neighbours (and a share weighted by rank), the distance to the nearest
   artificial doublet and to the nearest real cell, its library size and genes
   detected, its principal components, and a co-expression score (cxds, Bais & Kostka
   2020: genes seldom detected in one cell together, detected together);
5. a gradient-boosted classifier (scikit-learn's histogram gradient boosting, BSD;
   scDblFinder uses xgboost) learns artificial doublet vs real cell; each cell's score
   comes from a model that did not see it (3-fold cross-fitting). Real cells it scores
   as doublets are left out of the next round's training (they are probably real
   doublets, labelled singlets), three rounds in all;
6. the threshold: the score that best separates the artificial doublets from the real
   cells while calling close to the expected doublet rate (``threshold``).

Run per capture channel (``classify_by_channel``), as Scrublet is: doublets form inside
one channel, and each channel gets its own rate (the same 10x rule of thumb, 0.8 % per
1,000 cells, ``doublets.rate_for_cells``) and threshold. A channel under
``MIN_CHANNEL_CELLS`` is not scored (NaN score, no call), as for Scrublet.

Measured against R scDblFinder 1.26.9 (2 CPUs): on 10x's hgmm_10k_v3 human + mouse mix
(11,456 cells; 422 cross-species doublets of >= 1,500 UMIs as the truth) AUC 0.948 vs
scDblFinder's 0.951 and Scrublet's 0.938, in 7 s vs 232 s (Scrublet 90 s).

Deterministic for a given input (seed 0). The code uses numpy, pandas, scipy and
scikit-learn only: the replay script copies its source (``REPLAY_CONSTANTS``,
``REPLAY_CODE``).
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp

# Genes the classifier's space is built on: the most expressed in each cluster, in turns
# (scDblFinder's default is 1,352).
N_FEATURES = 1352
N_PCS = 20  # principal components of real + artificial cells, all given to the classifier
N_PC_FEATURES = 20
KS = (3, 10, 25, 50)  # neighbourhood sizes for the share of artificial neighbours
# Artificial doublets: as many as cells, within these bounds; a third of them from any two
# cells (of one cluster or two), the rest from two cells of different clusters.
MIN_ARTIFICIAL = 1500
MAX_ARTIFICIAL = 25_000
RANDOM_SHARE = 1 / 3
CXDS_GENES = 500  # genes of the co-expression score
N_ROUNDS = 3  # training rounds (real cells called doublets leave the next round's training)
N_FOLDS = 3  # cross-fitting: each cell's score comes from a model that did not see it
# Tolerated deviation from the expected rate before the threshold pays for it, as a share
# of that rate (the paper's "dbr.sd").
RATE_SD_SHARE = 0.4
MIN_CHANNEL_CELLS = 100  # as Scrublet (core/scrublet_lean.py): smaller channels not scored
MIN_GENES = 3  # a cell with fewer detected genes is not scored
SEED = 0


def _csr(counts):
    """``counts`` (sparse, dense or on disk) as float32 CSR, without a dense copy."""
    if sp.issparse(counts):
        X = sp.csr_matrix(counts, dtype=np.float32)
        X.eliminate_zeros()
        return X
    if callable(getattr(counts, "to_memory", None)):  # an on-disk sparse matrix
        X = sp.csr_matrix(counts.to_memory(), dtype=np.float32)
        X.eliminate_zeros()
        return X
    n = counts.shape[0]
    blocks = [sp.csr_matrix(np.asarray(counts[i:i + 1024], dtype=np.float32)) for i in range(0, n, 1024)]
    return sp.vstack(blocks, format="csr") if blocks else sp.csr_matrix(counts.shape, dtype=np.float32)


def _lognorm(M, lib, scale):
    """log1p of counts scaled to ``scale`` per cell (``lib``: each row's total)."""
    f = np.where(lib > 0, scale / np.maximum(lib, 1e-12), 0.0).astype(np.float32)
    out = sp.diags(f) @ M
    out.data = np.log1p(out.data)
    return out


def _pcs(M, n_comps, seed):
    """Principal components (scores) of a sparse matrix, centred without densifying it
    more than one dense cells x genes copy of the (few) selected genes."""
    from sklearn.utils.extmath import randomized_svd

    D = M.toarray() if sp.issparse(M) else np.asarray(M, dtype=np.float32)
    D -= D.mean(axis=0, keepdims=True)
    k = int(max(1, min(n_comps, min(D.shape) - 1)))
    U, S, _ = randomized_svd(D, n_components=k, n_iter=4, random_state=seed)
    return (U * S).astype(np.float32)


def _clusters(pcs, seed):
    """A quick k-means clustering of the cells on their principal components."""
    from sklearn.cluster import KMeans

    n = pcs.shape[0]
    k = int(np.clip(round(np.sqrt(n) / 4), 4, 30))
    k = min(k, max(1, n // 10))
    if k < 2:
        return np.zeros(n, dtype=np.int64)
    return KMeans(n_clusters=k, n_init=3, random_state=seed).fit_predict(pcs).astype(np.int64)


def _features_genes(L, clusters, n_features):
    """The most expressed genes of each cluster, taken in turns, ``n_features`` in all
    (fewer when the data has fewer genes)."""
    g = L.shape[1]
    if g <= n_features:
        return np.arange(g)
    ks = np.unique(clusters)
    means = np.vstack([np.asarray(L[clusters == c].mean(axis=0)).ravel() for c in ks])
    order = np.argsort(-means, axis=1, kind="stable")
    chosen: list[int] = []
    seen = np.zeros(g, dtype=bool)
    for rank in range(g):
        for j in range(len(ks)):
            gene = int(order[j, rank])
            if not seen[gene]:
                seen[gene] = True
                chosen.append(gene)
                if len(chosen) >= n_features:
                    return np.sort(np.asarray(chosen))
    return np.sort(np.asarray(chosen))


def _pairs(clusters, n_art, rng):
    """``n_art`` pairs of cells (indices into the cells), drawn at random, so cluster pairs
    come up as often as their cells would meet in a droplet: the first ``RANDOM_SHARE`` of
    any two cells, the rest of two of different clusters (when there is more than one)."""
    n = clusters.size
    a = rng.integers(0, n, size=n_art)
    b = rng.integers(0, n, size=n_art)
    if np.unique(clusters).size > 1:
        free = np.arange(n_art) < int(RANDOM_SHARE * n_art)
        for _ in range(50):
            same = (clusters[a] == clusters[b]) & ~free
            if not same.any():
                break
            b[same] = rng.integers(0, n, size=int(same.sum()))
    return a, b


def _neighbour_features(pcs, is_art, ks):
    """Per point: the share of artificial doublets among its k nearest neighbours for each
    k in ``ks``, a rank-weighted share over the largest k, and the distances to the
    nearest artificial doublet and the nearest real cell (scaled by the median distance
    to the nearest neighbour)."""
    from sklearn.neighbors import NearestNeighbors

    n = pcs.shape[0]
    kmax = int(min(max(ks), n - 1))
    nn = NearestNeighbors(n_neighbors=kmax + 1).fit(pcs)
    dist, idx = nn.kneighbors(pcs)
    dist, idx = dist[:, 1:], idx[:, 1:]  # each point is its own nearest
    art = is_art[idx]
    cols = [art[:, :min(k, kmax)].mean(axis=1) for k in ks]
    w = 1.0 / np.arange(1, kmax + 1)
    cols.append((art * w).sum(axis=1) / w.sum())
    unit = float(np.median(dist[:, 0])) or 1.0
    far = dist[:, -1] / unit * 2.0  # none within the neighbours: past the farthest
    d_art = np.where(art.any(axis=1), dist[np.arange(n), np.argmax(art, axis=1)] / unit, far)
    d_real = np.where((~art).any(axis=1), dist[np.arange(n), np.argmax(~art, axis=1)] / unit, far)
    cols += [d_art, d_real]
    return np.column_stack(cols).astype(np.float32)


def _cxds(B_real, B_all):
    """A co-expression score per point (cxds, Bais & Kostka 2020, as scDblFinder adds
    it): pairs of genes detected together in fewer real cells than independence predicts
    weigh the most (-log of that binomial tail, capped at 100); a point's score sums the
    weights of the pairs it has both genes of, scaled to a maximum of 1. ``B_real`` and
    ``B_all``: detected (1) or not (0), cells x the ``CXDS_GENES`` genes."""
    from scipy.stats import binom

    n = B_real.shape[0]
    p = np.asarray(B_real.mean(axis=0)).ravel()
    obs = (B_real.T @ B_real).toarray()
    expect = np.outer(p, p)
    with np.errstate(divide="ignore"):
        w = -binom.logcdf(obs, n, np.clip(expect, 1e-12, 1 - 1e-12))
    w = np.clip(np.nan_to_num(w, nan=0.0, posinf=100.0), 0.0, 100.0)
    np.fill_diagonal(w, 0.0)
    s = 0.5 * np.asarray(sp.csr_matrix(B_all).multiply(B_all @ w).sum(axis=1)).ravel()
    return s / max(float(np.max(s)), 1e-12)


def _model(seed):
    from sklearn.ensemble import HistGradientBoostingClassifier

    return HistGradientBoostingClassifier(max_iter=100, learning_rate=0.1, max_depth=4,
                                          early_stopping=False, random_state=seed)


def threshold(real, art, rate, sd_share=RATE_SD_SHARE):
    """The score above which real cells are called doublets: the one that minimises the
    share of artificial doublets under it, plus the share of real cells above it, plus
    the squared deviation of the share called from ``rate`` beyond ``sd_share`` of it
    (as a share of the rate). Candidates: the real cells' own scores (each one called
    or not), so the calls at it are a cut of the scores."""
    real = np.sort(np.asarray(real, dtype=np.float64))
    art = np.sort(np.asarray(art, dtype=np.float64))
    n = real.size
    if not n:
        return 1.0
    cand = np.unique(np.concatenate([real, [1.0]]))
    above = n - np.searchsorted(real, cand, side="right")  # real cells strictly above
    art_under = np.searchsorted(art, cand, side="right") / max(art.size, 1)
    share = above / n
    sd = sd_share * rate
    dev = np.maximum(0.0, np.abs(share - rate) - sd) / max(rate, 1e-9)
    cost = art_under + share + dev ** 2
    return float(cand[int(np.argmin(cost))])


def classify_counts(counts, expected_doublet_rate=0.05, seed=SEED):
    """The doublet classifier on ``counts`` (cells x genes; sparse, dense or on disk).

    Returns (score, call, info): each cell's doublet score in [0, 1] (NaN for a cell not
    scored: under ``MIN_GENES`` genes), its call (score above the threshold) and what was
    done (threshold, clusters, artificial doublets, genes, the artificial doublets'
    scores' median)."""
    X = _csr(counts)
    n = X.shape[0]
    score = np.full(n, np.nan)
    call = np.zeros(n, dtype=bool)
    ok = np.flatnonzero(np.diff(X.indptr) >= MIN_GENES)
    if ok.size < MIN_CHANNEL_CELLS:
        raise ValueError(f"only {ok.size} cells with {MIN_GENES} or more genes")
    X = X[ok]
    keep_genes = np.flatnonzero(np.asarray((X > 0).sum(axis=0)).ravel() >= 3)
    X = X[:, keep_genes]
    rng = np.random.default_rng(seed)
    lib = np.asarray(X.sum(axis=1)).ravel().astype(np.float64)
    scale = float(np.median(lib))
    L = _lognorm(X, lib, scale)

    # 1. quick clusters on the genes most expressed overall
    top = np.argsort(-np.asarray(L.mean(axis=0)).ravel(), kind="stable")[:min(N_FEATURES, L.shape[1])]
    clusters = _clusters(_pcs(L[:, np.sort(top)], N_PCS, seed), seed)
    # 2. artificial doublets: two cells added together, most of them of different clusters
    feats = _features_genes(L, clusters, N_FEATURES)
    m = int(np.clip(ok.size, MIN_ARTIFICIAL, MAX_ARTIFICIAL))
    a, b = _pairs(clusters, m, rng)
    F = X[:, feats]
    D = F[a] + F[b]
    lib_all = np.concatenate([lib, lib[a] + lib[b]])
    M = sp.vstack([F, D], format="csr")
    del D
    # 3. real + artificial: log-normalised, principal components, neighbours
    pcs = _pcs(_lognorm(M, lib_all, scale), N_PCS, seed)
    is_art = np.r_[np.zeros(ok.size, dtype=bool), np.ones(m, dtype=bool)]
    genes = np.diff(M.indptr).astype(np.float32)
    del M
    # 4. features
    # cxds on the genes detected in closest to half the cells (the most informative)
    B = (F > 0).astype(np.float32)
    top = np.sort(np.argsort(np.abs(np.asarray(B.mean(axis=0)).ravel() - 0.5), kind="stable")[:CXDS_GENES])
    B = B[:, top]
    cxds = _cxds(B, sp.vstack([B, ((B[a] + B[b]) > 0).astype(np.float32)], format="csr"))
    del B, F
    feat = np.column_stack([_neighbour_features(pcs, is_art, KS),
                            np.log1p(lib_all).astype(np.float32), np.log1p(genes),
                            pcs[:, :N_PC_FEATURES], cxds.astype(np.float32)])
    del pcs
    y = is_art.astype(np.int64)
    # 5. cross-fitted gradient boosting, real doublets left out of later rounds' training
    train = np.ones(y.size, dtype=bool)
    fold = rng.permutation(np.arange(y.size) % N_FOLDS)
    pred = np.zeros(y.size)
    t = 1.0
    for _ in range(N_ROUNDS):
        for f in range(N_FOLDS):
            test = fold == f
            fit = train & ~test
            pred[test] = _model(seed).fit(feat[fit], y[fit]).predict_proba(feat[test])[:, 1]
        t = threshold(pred[:ok.size], pred[ok.size:], expected_doublet_rate)
        train = np.r_[pred[:ok.size] <= t, np.ones(m, dtype=bool)]
    # 6. scores and calls at the threshold
    real = pred[:ok.size]
    score[ok] = real
    call[ok] = real > t
    info = {"threshold": float(t), "n_clusters": int(np.unique(clusters).size),
            "n_artificial": int(m), "n_genes": int(feats.size),
            "artificial_median": float(np.median(pred[ok.size:]))}
    return score, call, info


def classify_by_channel(counts, codes, rates, min_cells=MIN_CHANNEL_CELLS, rows=None):
    """The classifier on each capture channel's cells alone, one channel after another.

    ``codes``: each cell's channel, an index into ``rates`` (-1: none, not scored).
    ``rows(counts, idx)`` reads a channel's rows (default: ``counts[idx]``; the doublet
    step passes ``scrublet_lean.channel_rows`` for on-disk counts). Returns (score, call,
    runs) in cell order, NaN and False for a cell not scored, and per channel a dict:
    status (scored | too_small | failed), cells, cells scored, threshold, cells called."""
    codes = np.asarray(codes)
    n = counts.shape[0]
    score = np.full(n, np.nan)
    call = np.zeros(n, dtype=bool)
    order = np.argsort(codes, kind="stable")
    bounds = np.searchsorted(codes[order], np.arange(len(rates) + 1))
    runs = []
    for j, rate in enumerate(rates):
        idx = order[bounds[j]:bounds[j + 1]]
        run = {"status": "too_small", "n_cells": int(idx.size), "n_scored": 0, "threshold": None,
               "n_called": 0}
        runs.append(run)
        if idx.size < min_cells:
            continue
        sub = counts if idx.size == n else (rows(counts, idx) if rows else counts[idx])
        try:
            s, c, info = classify_counts(sub, expected_doublet_rate=float(rate))
        except ValueError as e:  # too few usable cells or genes
            run.update(status="failed", error=str(e)[:200])
            continue
        finally:
            del sub
        score[idx] = s
        call[idx] = c
        run.update(status="scored", n_scored=int(np.isfinite(s).sum()), n_called=int(c.sum()),
                   threshold=info["threshold"])
    return score, call, runs


REPLAY_CONSTANTS = ("N_FEATURES", "N_PCS", "N_PC_FEATURES", "KS", "MIN_ARTIFICIAL", "MAX_ARTIFICIAL",
                    "RANDOM_SHARE", "CXDS_GENES", "N_ROUNDS", "N_FOLDS", "RATE_SD_SHARE",
                    "MIN_CHANNEL_CELLS", "MIN_GENES", "SEED")
REPLAY_CODE = (_csr, _lognorm, _pcs, _clusters, _features_genes, _pairs, _neighbour_features, _cxds,
               _model, threshold, classify_counts, classify_by_channel)
