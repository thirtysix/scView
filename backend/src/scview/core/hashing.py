"""Cell hashing: each cell's sample from its hashtag counts (C118).

Hashtag oligos (HTOs: TotalSeq-A/B/C hashtag antibodies, or 10x CellPlex
multiplexing oligos, CMOs) label each sample's cells before they are pooled
into one lane. A cell's tags say which sample it came from, and a droplet with
two samples' tags is a doublet. The counts live in ``obsm["hto_counts"]``
(cells x tags, a DataFrame):

- CellPlex "Multiplexing Capture" features go there at load
  (``core/protein.split_features``);
- hashtags read as "Antibody Capture" features stay in the protein modality
  until the user confirms them (``suggest_tags`` proposes names that look like
  hashtags: Hashtag, HTO, TotalSeq hashtag IDs A0251-C0265, CMO3xx): a panel's
  names alone cannot say which antibodies were the hashtags. The confirmed ones
  are moved into ``hto_counts`` when the call is saved, and the protein CLR is
  made again without them.

Two calls, the user's choice (both deterministic, so the replay is exact):

``htodemux`` (default): Seurat's HTODemux (Stoeckius et al. 2018, Genome Biol
19:224, PMID 30567574; ported from Seurat 5.1.0's ``HTODemux`` source). Each tag's counts are
CLR-normalised across cells, as ``NormalizeData(method = "CLR", margin = 1)``:
``log1p(x / exp(sum(log1p(x)) / n))`` per tag. The cells are clustered on those
values into (tags + 1) groups by CLARA k-medoids (``cluster::clara``, 100
samples, its own built-in random generator, which ``set.seed`` does not touch:
ported line by line, so the groups are R's). For each tag the group with the
lowest mean ``expm1(CLR)`` is its background; a negative binomial is fitted to
that group's counts by maximum likelihood and a cell is positive for the tag
when its count is above the fit's 0.99 quantile. One positive tag: singlet,
called to the tag with the highest CLR (as Seurat does); two or more: doublet;
none: negative. Seurat's fit uses ``fitdistrplus`` (Nelder-Mead); the exact
maximum is used here, which moves the fitted mean by ~0.01% and leaves the
integer cutoffs unchanged on the data checked.

``hashedDrops``: DropletUtils' ``hashedDrops`` (A. Lun; DropletUtils, Lun et al.
2019, Genome Biol 20:63, PMID 30902100; the cell hashing chapter of OSCA,
Amezquita et al. 2020, Nat Methods 17:137, PMID 31792435; ported from
DropletUtils 1.24 ``hashedDrops.R`` + ``hashed_deltas.cpp``), on log-fold changes
of the top two tags. The ambient profile is each tag's mean count over its lower mode (a
two-centre k-means on ``log1p`` counts started from the 5% and 95% quantiles,
``ambientProfileBimodal``; R uses Hartigan-Wong, Lloyd's iterations here). Per
cell, the ambient is scaled to the cell (median count/ambient ratio over tags;
the 3rd largest with 3-4 tags) and subtracted; ``LogFC`` = log2 of the top tag
over the second, ``LogFC2`` = log2 of the second over the ambient level. A
doublet has ``LogFC2`` more than 3 MADs above the median (and above 2); a
confident singlet, among the rest, ``LogFC`` not more than 3 MADs below the
median (and above 2). hashedDrops calls no negatives: a cell that is neither
is stored as "Negative" here (not assigned).

Checked against R on GSE108313 (8-HTO PBMC, 20,238 cells; 12-HTO cell lines,
8,410 cells): the same class and tag for every cell as Seurat 5.1.0 and
DropletUtils 1.24 (tests/fixtures/hashing has a 1,000-cell reference).

Outputs (obs): ``hash_call`` (tag / "Doublet" / "Negative"), ``hash_class``
(Singlet / Doublet / Negative), ``hash_max`` and ``hash_second`` (top and
second tag), ``hash_margin`` (the top tag's lead: CLR difference, or LogFC for
hashedDrops). ``sample_column`` writes a sample column from a tag -> sample
mapping: singlets get their sample's name, doublets and negatives are left
empty. ``hash_class`` feeds the doublet views (``cluster_qc.doublet_calls``).
"""

from __future__ import annotations

import json
import logging
import math
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd
import scipy.sparse as sp

logger = logging.getLogger(__name__)

HTO_KEY = "hto_counts"
UNS_KEY = "scview_hashing"  # JSON string: sources, moved features, the saved call, the mapping
CMO_TYPE = "Multiplexing Capture"
ANTIBODY_SOURCE = "Antibody Capture"
# obsm keys other tools leave hashtag counts under (DataFrames with tag names).
FOREIGN_KEYS = ("HTO", "hto", "hashtags", "hashtag_counts", "cmo_counts", "CMO")

CALL, CLASS, MAX, SECOND, MARGIN = "hash_call", "hash_class", "hash_max", "hash_second", "hash_margin"
OBS_COLUMNS = (CALL, CLASS, MAX, SECOND, MARGIN)
SINGLET, DOUBLET, NEGATIVE = "Singlet", "Doublet", "Negative"
CLASSES = (SINGLET, DOUBLET, NEGATIVE)
DEFAULT_SAMPLE_COLUMN = "hash_sample"

HTODEMUX, HASHEDDROPS = "htodemux", "hashedDrops"
METHODS = (HTODEMUX, HASHEDDROPS)
METHOD_LABELS = {HTODEMUX: "HTODemux (Seurat)", HASHEDDROPS: "hashedDrops (DropletUtils)"}
POSITIVE_QUANTILE = 0.99
CLARA_SAMPLES = 100
# hashedDrops defaults (DropletUtils 1.24).
MIN_PROP = 0.05
PSEUDO_COUNT = 5
DOUBLET_NMADS, DOUBLET_MIN = 3.0, 2.0
CONFIDENT_NMADS, CONFIDENT_MIN = 3.0, 2.0

MIN_TAGS = 2
MAX_TAGS = 50  # more is not a hashing panel
MIN_CELLS = 50
HIST_BINS = 40
TAG = "result:hashing"

# TotalSeq-A/B/C hashtag antibodies are A0251-A0265, B0251-..., C0251-...;
# CellPlex CMOs are CMO301-CMO312; MULTI-seq and custom panels say HTO / Hashtag.
_TAG_NAME = re.compile(
    r"hashtag|(^|[^a-z])hto([^a-z]|$)|(^|[^a-z0-9])[abc]0?2(5[1-9]|6[0-5])([^0-9a-z]|$)|(^|[^a-z])cmo\d{3}",
    re.I,
)


class HashingError(ValueError):
    """A request this cannot serve; nothing was changed. ``status`` is the HTTP code."""

    def __init__(self, message: str, status: int = 400) -> None:
        super().__init__(message)
        self.status = status


# ---------------------------------------------------------------------------
# Which features are tags
# ---------------------------------------------------------------------------


def looks_like_tag(name: str) -> bool:
    return bool(_TAG_NAME.search(str(name)))


def suggest_tags(names) -> list[str]:
    """The names that look like hashtags (to be confirmed by the user)."""
    return [str(n) for n in names if looks_like_tag(n)]


def _uniq(names: list[str]) -> list[str]:
    seen: dict[str, int] = {}
    out = []
    for n in names:
        k = seen.get(n, 0)
        out.append(n if k == 0 else f"{n}-{k}")
        seen[n] = k + 1
    return out


def read_state(adata) -> dict:
    """The stored hashing record (``uns[UNS_KEY]``), or an empty one."""
    raw = adata.uns.get(UNS_KEY) if hasattr(adata, "uns") else None
    try:
        state = json.loads(raw) if isinstance(raw, str) else {}
    except ValueError:
        logger.warning("Ignoring an unreadable %s", UNS_KEY)
        state = {}
    return state if isinstance(state, dict) else {}


def _write_state(adata, state: dict) -> None:
    # A new string each time: write_derived saves replaced uns entries.
    adata.uns[UNS_KEY] = json.dumps(state, default=str)


def attach_counts(adata, counts, features: list[str], source: str) -> list[str]:
    """Add tag counts (cells x ``features``) to ``obsm[HTO_KEY]``, beside any
    already there, and note where they came from. Returns the column names."""
    from scview.core.protein import display_name

    dense = counts.toarray() if sp.issparse(counts) else np.asarray(counts)
    old = adata.obsm.get(HTO_KEY)
    taken = [str(c) for c in old.columns] if isinstance(old, pd.DataFrame) else []
    names = _uniq(taken + [display_name(str(f)) for f in features])[len(taken):]
    frame = pd.DataFrame(np.asarray(dense, dtype=np.float32), index=adata.obs_names, columns=names)
    adata.obsm[HTO_KEY] = pd.concat([old, frame], axis=1) if taken else frame
    state = read_state(adata)
    state.setdefault("sources", {}).update({n: source for n in names})
    state.setdefault("original", {}).update({n: str(f) for n, f in zip(names, features) if n != str(f)})
    _write_state(adata, state)
    return names


def _tag_key(adata) -> str | None:
    """The obsm key holding tag counts: scView's, else a hashtag table another
    tool left under a known name (a DataFrame of counts)."""
    if not hasattr(adata, "obsm"):
        return None
    m = adata.obsm.get(HTO_KEY)
    if isinstance(m, pd.DataFrame) and m.shape[1] > 0:
        return HTO_KEY
    for key in FOREIGN_KEYS:
        m = adata.obsm.get(key)
        if isinstance(m, pd.DataFrame) and 0 < m.shape[1] <= MAX_TAGS:
            v = m.to_numpy(dtype=np.float64)
            if np.isfinite(v).all() and v.min() >= 0 and np.allclose(v, np.round(v)):
                return key
    return None


def tag_counts_frame(adata) -> pd.DataFrame | None:
    """The tag counts (cells x tags), wherever ``_tag_key`` finds them."""
    key = _tag_key(adata)
    return None if key is None else adata.obsm[key]


def candidates(adata) -> dict[str, Any]:
    """What could be demultiplexed: tag counts already kept, protein features
    that look like hashtags, and every protein feature (one can be marked)."""
    from scview.core import protein

    key = _tag_key(adata)
    frame = None if key is None else adata.obsm[key]
    state = read_state(adata)
    tags = [str(c) for c in frame.columns] if frame is not None else []
    proteins = protein.protein_names(adata)
    sources = state.get("sources", {})
    return {
        "tags": tags,
        "tag_sources": {t: (f"obsm {key}" if key != HTO_KEY else sources.get(t, "obsm")) for t in tags},
        "proteins": proteins,
        "suggested": suggest_tags(proteins),
        "moved_from_protein": [t for t in state.get("moved_from_protein", []) if t in tags],
    }


# ---------------------------------------------------------------------------
# HTODemux (Seurat 5.1.0), with cluster::clara ported exactly
# ---------------------------------------------------------------------------


def clr_denominators(counts) -> np.ndarray:
    """Each tag's CLR divisor ``exp(sum(log1p(x[x > 0])) / n)`` (0 for a tag
    with no counts). The sum is accumulated in long double, in order, and log1p
    / exp are the C library's, as R's ``sum`` and maths functions are."""
    x = np.asarray(counts, dtype=np.float64)
    n = x.shape[0]
    log1p = np.frompyfunc(math.log1p, 1, 1)
    out = np.zeros(x.shape[1])
    for j in range(x.shape[1]):
        pos = x[x[:, j] > 0, j]
        if pos.size:
            out[j] = math.exp(float(np.cumsum(log1p(pos).astype(np.longdouble))[-1]) / n)
    return out


def clr_tags(counts) -> np.ndarray:
    """Seurat's CLR across cells, one tag (column) at a time (margin = 1):
    ``log1p(x / exp(sum(log1p(x[x > 0])) / n))``."""
    x = np.asarray(counts, dtype=np.float64)
    out = np.zeros_like(x)
    log1p = np.frompyfunc(math.log1p, 1, 1)
    for j, denom in enumerate(clr_denominators(x)):
        if denom > 0:
            out[:, j] = log1p(x[:, j] / denom).astype(np.float64)
    return out


def _randm(nrun: int) -> tuple[int, float]:
    """clara's built-in generator (period 65536)."""
    nrun = (nrun * 5761 + 999) & 0o177777
    return nrun, nrun / 65536.0


def _sq_dist(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Squared Euclidean distances (rows of ``a`` x rows of ``b``), summed over
    the features in order, as clara's C loops do (no pairwise summation)."""
    acc = np.zeros((a.shape[0], b.shape[0]))
    for j in range(a.shape[1]):
        d = a[:, j][:, None] - b[:, j][None, :]
        acc += d * d
    return acc


def _seqsum(v: np.ndarray, axis: int = -1) -> np.ndarray:
    """Sums accumulated in order along ``axis`` (cumsum is sequential)."""
    if v.shape[axis] == 0:
        return np.zeros(np.delete(v.shape, axis))
    return np.take(np.cumsum(v, axis=axis), -1, axis=axis)


def _bswap2(k: int, dys: np.ndarray, s: float) -> np.ndarray:
    """clara's BUILD + SWAP on one sample's distance matrix (``pam_like = FALSE``).
    Returns the medoid flags over the sample."""
    n = dys.shape[0]
    s = s * 1.1 + 1.0
    nrepr = np.zeros(n, dtype=bool)
    dysma = np.full(n, s)
    for _ in range(k):
        cmd = dysma[None, :] - dys
        beter = _seqsum(np.where(cmd > 0.0, cmd, 0.0), axis=1)
        nmax, ammax = -1, 0.0
        for i in np.flatnonzero(~nrepr):  # ties: the last of the maxima (<=)
            if ammax <= beter[i]:
                ammax, nmax = beter[i], i
        nrepr[nmax] = True
        dysma = np.minimum(dysma, dys[nmax])
    if k == 1:
        return nrepr
    while True:
        reps = np.flatnonzero(nrepr)
        d = dys[reps]  # reps x n
        srt = np.sort(d, axis=0)
        dysma = srt[0]
        dysmb = srt[1] if len(reps) > 1 else np.full(n, s)
        hs = np.flatnonzero(~nrepr)
        dij = d[None, :, :]  # 1 x reps x n
        dhj = dys[hs][:, None, :]  # hs x 1 x n
        small = np.where(dysmb[None, None, :] > dij, dhj, dysmb[None, None, :])
        term = np.where(dij == dysma, small - dysma,
                        np.where(dhj < dysma, dhj - dysma, 0.0))
        dz = _seqsum(term, axis=2)  # hs x reps, loops h then i
        flat = int(np.argmin(dz))  # the first of the minima, as `dzsky > dz`
        best = dz.flat[flat]
        if not best < 0.0 or not best < 1.0:
            return nrepr
        h, i = divmod(flat, len(reps))
        nrepr[hs[h]] = True
        nrepr[reps[i]] = False


def clara(x: np.ndarray, k: int, samples: int = CLARA_SAMPLES, sampsize: int | None = None) -> np.ndarray:
    """``cluster::clara(x, k, samples)`` with Euclidean distances, ``rngR =
    FALSE`` and ``pamLike = FALSE`` (HTODemux's call): the same sub-samples, the
    same medoids and the same cluster numbers 1..k as R. ``x`` has no NaN."""
    x = np.ascontiguousarray(x, dtype=np.float64)
    n = x.shape[0]
    nsam = min(n, 40 + 2 * k) if sampsize is None else int(sampsize)
    if not 1 <= k <= n - 1:
        raise HashingError(f"Cannot make {k} groups from {n} cells.")
    full, lrg = n == nsam, n < 2 * nsam
    n_sam = n - nsam if lrg else nsam
    nrun, kall, zba = 0, False, -1.0
    nrx: list[int] = []  # best medoids so far (1-based), in cluster order

    def draw() -> int:
        nonlocal nrun
        nrun, r = _randm(nrun)
        return min(1 + int(n * r), n)

    for jran in range(1, samples + 1):
        if full:
            nsel = list(range(1, n + 1))
        else:
            if kall and jran != 1 and not lrg:  # the best medoids join every later sample
                nsel = sorted(nrx)
            else:
                while True:
                    rk = draw()
                    if not (kall and rk in nrx):
                        break
                nsel = [rk]
            while len(nsel) < n_sam:
                rk = draw()
                if kall and lrg and rk in nrx:
                    continue
                pos = int(np.searchsorted(nsel, rk))
                if pos < len(nsel) and nsel[pos] == rk:
                    continue
                nsel.insert(pos, rk)
            if lrg:  # the draw was the half left out
                chosen = set(nsel)
                nsel = [j for j in range(1, n + 1) if j not in chosen][:nsam]
        sample = x[np.asarray(nsel) - 1]
        dys = np.sqrt(_sq_dist(sample, sample))
        np.fill_diagonal(dys, 0.0)
        s = float(dys[np.tril_indices(len(nsel), -1)].max()) if len(nsel) > 1 else 0.0
        nrepr = _bswap2(k, dys, s)
        nr = [nsel[j] for j in np.flatnonzero(nrepr)]
        # selec(): every object to its nearest medoid (the first of equals).
        sq = _sq_dist(x, x[np.asarray(nr) - 1])
        jk = np.argmin(sq, axis=1)
        for c, m in enumerate(nr):  # a medoid is in its own cluster, even beside a duplicate
            sq[m - 1, c] = 0.0
            jk[m - 1] = c
        zb = float(_seqsum(np.sqrt(sq[np.arange(n), jk])))
        first = list(dict.fromkeys(jk.tolist()))  # cluster order: first appearance
        first += [c for c in range(len(nr)) if c not in first]
        if not kall or zba > zb:
            kall, zba = True, zb
            nrx = [nr[c] for c in first]
        if full:
            break
    # resul(): the whole data set to the best medoids (distances, first of equals).
    d = np.sqrt(_sq_dist(x, x[np.asarray(nrx) - 1]))
    out = np.argmin(d, axis=1) + 1
    out[np.asarray(nrx) - 1] = np.arange(1, len(nrx) + 1)
    return out


def nb_fit(values: np.ndarray) -> tuple[float, float]:
    """Negative binomial ``(size, mu)`` by maximum likelihood. ``mu`` is the
    mean; ``size`` solves the profile score equation
    ``sum_i [digamma(x_i + r) - digamma(r)] = n log(1 + mu / r)``, with the
    digamma difference written as ``sum_{k < x_i} 1 / (r + k)`` (integer counts:
    exact, and no cancellation at large r). ``inf`` when the counts are not
    over-dispersed (the Poisson limit)."""
    from scipy.optimize import brentq

    v = np.rint(np.asarray(values, dtype=np.float64)).astype(np.int64)
    m = float(v.mean()) if v.size else 0.0
    if v.size == 0 or m == 0.0 or float(v.var()) <= m:
        return math.inf, m
    n = v.size
    # above[k] = how many counts exceed k, for k = 0 .. max - 1
    above = n - np.cumsum(np.bincount(v, minlength=int(v.max()) + 1))[:-1]
    ks = np.arange(above.size, dtype=np.float64)

    def score(log_r: float) -> float:
        r = math.exp(log_r)
        return float((above / (r + ks)).sum() - n * math.log1p(m / r))

    lo, hi = -30.0, 30.0
    if score(hi) > 0:  # still rising at size = e^30: Poisson for all purposes
        return math.inf, m
    while score(lo) < 0 and lo > -200:
        lo -= 30
    return math.exp(brentq(score, lo, hi, xtol=1e-14, rtol=1e-14)), m


def nb_quantile(size: float, mu: float, q: float = POSITIVE_QUANTILE) -> int:
    """R's ``qnbinom(q, size, mu = mu)``: the smallest count whose CDF reaches
    ``q`` (with R's 64-epsilon fuzz); Poisson when ``size`` is infinite."""
    from scipy import stats

    if mu <= 0:
        return 0
    target = q * (1 - 64 * np.finfo(float).eps)
    dist = stats.poisson(mu) if math.isinf(size) else stats.nbinom(size, size / (size + mu))
    hi = max(16, int(dist.ppf(q)) + 2)
    while dist.cdf(hi) < target:
        hi *= 2
    ks = np.arange(hi + 1)
    return int(ks[np.argmax(dist.cdf(ks) >= target)])


@dataclass
class Call:
    """One demultiplexing of a cells x tags count matrix."""

    method: str
    tags: list[str]
    best: np.ndarray  # tag index per cell (hash_max)
    second: np.ndarray  # tag index (hash_second)
    margin: np.ndarray  # float
    cls: np.ndarray  # "Singlet" / "Doublet" / "Negative"
    per_tag: list[dict] = field(default_factory=list)
    thresholds: dict = field(default_factory=dict)
    params: dict = field(default_factory=dict)

    @property
    def call(self) -> np.ndarray:
        names = np.asarray(self.tags, dtype=object)
        out = np.where(self.cls == SINGLET, names[self.best], self.cls)
        return out.astype(object)


def htodemux(counts: np.ndarray, tags: list[str], quantile: float = POSITIVE_QUANTILE,
             samples: int = CLARA_SAMPLES) -> Call:
    """Seurat's HTODemux on ``counts`` (cells x tags)."""
    counts = np.asarray(counts, dtype=np.float64)
    n, t = counts.shape
    data = clr_tags(counts)
    groups = clara(data, t + 1, samples=samples)
    levels = np.unique(groups)
    sizes = np.array([(groups == g).sum() for g in levels], dtype=np.float64)
    discrete = np.zeros((n, t), dtype=bool)
    per_tag = []
    for j, name in enumerate(tags):
        e = np.expm1(data[:, j])
        avg = np.array([e[groups == g].sum() for g in levels]) / sizes
        if (avg == 0).any():
            empty = int(levels[np.flatnonzero(avg == 0)[0]])
            raise HashingError(
                f"A group of {int(sizes[levels == empty][0]):,} cells has no {name} counts at all, so "
                "HTODemux cannot fit its background (Seurat stops here too: 'Cells with zero counts "
                "exist as a cluster'). Leave out tags that were not used, or try hashedDrops.")
        bg = int(levels[int(np.argmin(avg))])
        vals = counts[groups == bg, j]
        size, mu = nb_fit(vals)
        cutoff = nb_quantile(size, mu, quantile)
        discrete[:, j] = counts[:, j] > cutoff
        per_tag.append({"tag": name, "cutoff": cutoff, "background_group": bg,
                        "background_cells": int(vals.size), "background_mean": round(float(mu), 4),
                        "nb_size": None if math.isinf(size) else round(float(size), 6),
                        "n_positive": int(discrete[:, j].sum())})
    npos = discrete.sum(axis=1)
    cls = np.where(npos == 0, NEGATIVE, np.where(npos == 1, SINGLET, DOUBLET)).astype(object)
    hmax = data.max(axis=1)
    best = np.argmax(data == hmax[:, None], axis=1)  # the first tag at the maximum
    # Seurat's MaxN(N = 2): the second-largest value, ties with the maximum included;
    # its tag is the first one holding that value.
    second_val = np.sort(data, axis=1)[:, -2] if t > 1 else hmax
    second = np.argmax(data == second_val[:, None], axis=1)
    return Call(HTODEMUX, list(tags), best, second, hmax - second_val, cls, per_tag,
                {"positive_quantile": quantile},
                {"positive_quantile": quantile, "clara_samples": samples, "k": t + 1,
                 "clustering": "cluster::clara, built-in generator (no seed)"})


# ---------------------------------------------------------------------------
# hashedDrops (DropletUtils 1.24)
# ---------------------------------------------------------------------------


def _lower_mode(x: np.ndarray, p: float) -> np.ndarray:
    """DropletUtils' ``.get_lower_dist``: the lower of two k-means groups started
    from the ``p`` and ``1 - p`` quantiles (Lloyd's iterations here)."""
    q1, q2 = np.quantile(x, [p, 1 - p])
    if q1 == q2:
        return x <= q2
    c = np.array([q1, q2], dtype=np.float64)
    lower = np.abs(x - c[0]) <= np.abs(x - c[1])
    for _ in range(100):
        if lower.any():
            c[0] = x[lower].mean()
        if (~lower).any():
            c[1] = x[~lower].mean()
        new = np.abs(x - c[0]) <= np.abs(x - c[1])
        if (new == lower).all():
            break
        lower = new
    return lower if c[0] <= c[1] else ~lower


def ambient_profile(counts: np.ndarray, min_prop: float = MIN_PROP) -> np.ndarray:
    """``ambientProfileBimodal``: per tag, the mean count of its lower mode."""
    counts = np.asarray(counts, dtype=np.float64)
    out = np.zeros(counts.shape[1])
    for j in range(counts.shape[1]):
        cur = counts[:, j]
        out[j] = cur[_lower_mode(np.log1p(cur), min_prop)].mean()
    return out


def _mad(v: np.ndarray, center: float) -> float:
    return float(1.4826 * np.median(np.abs(v - center)))


def hashed_drops_stats(counts: np.ndarray, ambient: np.ndarray | None = None) -> dict[str, Any]:
    """hashedDrops' per-cell statistics (``hashed_deltas``): the top and second
    tag after ambient removal, ``lfc`` (log2 top / second) and ``lfc2`` (log2
    second / ambient level; NaN with 2 tags or fewer)."""
    counts = np.asarray(counts, dtype=np.float64)
    n = counts.shape[0]
    amb = ambient_profile(counts) if ambient is None else np.asarray(ambient, dtype=np.float64)
    keep = np.flatnonzero(amb != 0)  # a tag with no ambient is left out, as in R
    x, prop = counts[:, keep], amb[keep]
    nr = len(keep)
    if nr < 2:
        raise HashingError("hashedDrops needs at least two tags with an ambient level.")
    half, fallback = nr // 2, min(nr - 1, 2)
    srt = np.sort(x / prop, axis=1)[:, ::-1]  # count / ambient ratios, descending
    if half > fallback:
        scaling = (srt[:, half - 1] + srt[:, half]) / 2 if nr % 2 == 0 else srt[:, half]
    else:
        scaling = srt[:, fallback]
    pseudo = np.maximum(float(int(PSEUDO_COUNT)), scaling * (float(_seqsum(prop)) / nr))
    corr = np.maximum(0.0, x - scaling[:, None] * prop[None, :]) + pseudo[:, None]
    # std::greater on (value, index) pairs: descending, ties to the larger index.
    order = np.lexsort((-np.broadcast_to(np.arange(nr), corr.shape), -corr), axis=1)
    b, s2 = order[:, 0], order[:, 1]
    rows = np.arange(n)
    top, sec = corr[rows, b], corr[rows, s2]
    return {"ambient": amb, "best": keep[b], "second": keep[s2], "lfc": np.log2(top / sec),
            "lfc2": np.log2(sec / pseudo) if nr > 2 else np.full(n, np.nan)}


def hashed_drops(counts: np.ndarray, tags: list[str], ambient: np.ndarray | None = None) -> Call:
    """DropletUtils' hashedDrops on ``counts`` (cells x tags), default settings."""
    st = hashed_drops_stats(counts, ambient)
    lfc, lfc2, amb = st["lfc"], st["lfc2"], st["ambient"]
    n = lfc.size
    if np.isfinite(lfc2).any():
        med2 = float(np.median(lfc2))
        upper = max(med2 + DOUBLET_NMADS * _mad(lfc2, med2), DOUBLET_MIN)
        doublet = lfc2 > upper
    else:  # 2 tags: no doublet calls (R sets them NA)
        upper, doublet = None, np.zeros(n, dtype=bool)
    single = lfc[~doublet]
    if single.size:
        med = float(np.median(single))
        lower = max(med - CONFIDENT_NMADS * _mad(single, med), CONFIDENT_MIN)
    else:
        lower = CONFIDENT_MIN
    confident = (lfc > lower) & ~doublet
    cls = np.where(doublet, DOUBLET, np.where(confident, SINGLET, NEGATIVE)).astype(object)
    per_tag = [{"tag": t, "ambient": round(float(amb[i]), 4),
                "n_singlets": int(((cls == SINGLET) & (st["best"] == i)).sum())}
               for i, t in enumerate(tags)]
    thresholds = {"doublet_lfc2": None if upper is None else round(float(upper), 4),
                  "confident_lfc": round(float(lower), 4)}
    return Call(HASHEDDROPS, list(tags), st["best"], st["second"], lfc, cls, per_tag, thresholds,
                {"min_prop": MIN_PROP, "pseudo_count": PSEUDO_COUNT, "doublet_nmads": DOUBLET_NMADS,
                 "doublet_min": DOUBLET_MIN, "confident_nmads": CONFIDENT_NMADS,
                 "confident_min": CONFIDENT_MIN, "ambient": "ambientProfileBimodal (Lloyd)"})


def call_columns(call: Call) -> dict[str, Any]:
    """The obs columns of a call: ``hash_call``, ``hash_class``, ``hash_max``,
    ``hash_second`` (categoricals, tags in the call's order) and ``hash_margin``."""
    names = np.asarray(call.tags, dtype=object)
    tags = list(dict.fromkeys(call.tags))
    return {
        CALL: pd.Categorical(call.call, categories=list(dict.fromkeys(tags + [DOUBLET, NEGATIVE]))),
        CLASS: pd.Categorical(call.cls, categories=list(CLASSES)),
        MAX: pd.Categorical(names[call.best], categories=tags),
        SECOND: pd.Categorical(names[call.second], categories=tags),
        MARGIN: np.asarray(call.margin, dtype=np.float32),
    }


def run(counts: np.ndarray, tags: list[str], method: str = HTODEMUX, quantile: float = POSITIVE_QUANTILE) -> Call:
    """The chosen call on ``counts`` (cells x tags), validated first."""
    if method not in METHODS:
        raise HashingError(f"Unknown method '{method}' (htodemux or hashedDrops).")
    counts = np.asarray(counts, dtype=np.float64)
    if counts.ndim != 2 or counts.shape[1] != len(tags):
        raise HashingError("The counts do not match the tags.")
    if not MIN_TAGS <= len(tags) <= MAX_TAGS:
        raise HashingError(f"Pick {MIN_TAGS} to {MAX_TAGS} tags (got {len(tags)}).")
    if counts.shape[0] < MIN_CELLS:
        raise HashingError(f"At least {MIN_CELLS} cells are needed (got {counts.shape[0]}).")
    if not np.isfinite(counts).all() or counts.min() < 0:
        raise HashingError("Tag counts must be finite and not negative.")
    if not 0.5 < quantile < 1:
        raise HashingError("The positive quantile must be between 0.5 and 1.")
    if method == HTODEMUX:
        return htodemux(counts, tags, quantile)
    return hashed_drops(counts, tags)


# ---------------------------------------------------------------------------
# The dataset: gathering the counts, saving a call, the sample column
# ---------------------------------------------------------------------------


def _protein_counts(adata) -> pd.DataFrame | None:
    from scview.core import protein

    m = adata.obsm.get(protein.COUNTS_KEY) if hasattr(adata, "obsm") else None
    return m if isinstance(m, pd.DataFrame) and m.shape[1] > 0 else None


def gather(adata, tags: list[str]) -> tuple[np.ndarray, list[str]]:
    """The counts (cells x ``tags``) and, per tag, where they are: "tags" (the
    tag counts) or "protein" (an antibody feature still in the protein
    modality). Raises HashingError naming any tag found nowhere."""
    if not isinstance(tags, list) or not all(isinstance(t, str) for t in tags):
        raise HashingError("Tags must be a list of feature names.")
    if len(set(tags)) != len(tags):
        raise HashingError("A tag is listed twice.")
    frame, prot = tag_counts_frame(adata), _protein_counts(adata)
    cols, where, missing = [], [], []
    for t in tags:
        if frame is not None and t in frame.columns:
            cols.append(frame[t].to_numpy(dtype=np.float64))
            where.append("tags")
        elif prot is not None and t in prot.columns:
            cols.append(prot[t].to_numpy(dtype=np.float64))
            where.append("protein")
        else:
            missing.append(t)
    if missing:
        raise HashingError(f"Not a tag or antibody feature of this dataset: {', '.join(missing[:5])}.")
    counts = np.column_stack(cols) if cols else np.zeros((adata.n_obs, 0))
    return counts, where


def _move_to_tags(adata, names: list[str]) -> None:
    """Antibody features out of the protein modality into the tag counts; the
    protein CLR is made again without them (it is per cell across proteins)."""
    prot = _protein_counts(adata)
    if not names or prot is None:
        return
    from scview.core import protein

    taken = adata.obsm.get(HTO_KEY)
    add = prot[names].astype(np.float32)
    adata.obsm[HTO_KEY] = pd.concat([taken, add], axis=1) if isinstance(taken, pd.DataFrame) else add
    info = adata.uns.get(protein.UNS_KEY)
    renamed = (info.get("original") or {}) if isinstance(info, dict) else {}
    rest = prot.drop(columns=names)
    _set_protein(adata, rest)
    state = read_state(adata)
    state.setdefault("sources", {}).update({n: ANTIBODY_SOURCE for n in names})
    state.setdefault("original", {}).update({n: str(renamed[n]) for n in names if n in renamed})
    moved = [m for m in state.get("moved_from_protein", []) if m not in names] + list(names)
    state["moved_from_protein"] = moved
    _write_state(adata, state)


def _set_protein(adata, frame: pd.DataFrame) -> None:
    from scview.core import protein

    info = adata.uns.get(protein.UNS_KEY)
    if frame.shape[1]:
        adata.obsm[protein.COUNTS_KEY] = frame
        protein.renormalise(adata)
    else:  # every antibody was a hashtag: no protein modality left
        adata.obsm.pop(protein.COUNTS_KEY, None)
        adata.obsm.pop(protein.CLR_KEY, None)
    if isinstance(info, dict):
        new = dict(info)
        new["names"] = [str(c) for c in frame.columns]
        new["n_proteins"] = int(frame.shape[1])
        new["controls"] = [c for c in info.get("controls", []) if c in frame.columns]
        adata.uns[protein.UNS_KEY] = new


def _move_back(adata, names: list[str]) -> None:
    """Features moved here from the protein modality earlier, back to it."""
    frame = adata.obsm.get(HTO_KEY)
    if not names or not isinstance(frame, pd.DataFrame):
        return
    prot = _protein_counts(adata)
    back = frame[names].astype(np.float32)
    _set_protein(adata, pd.concat([prot, back], axis=1) if prot is not None else back)
    rest = frame.drop(columns=names)
    if rest.shape[1]:
        adata.obsm[HTO_KEY] = rest
    else:
        adata.obsm.pop(HTO_KEY, None)
    state = read_state(adata)
    state["moved_from_protein"] = [m for m in state.get("moved_from_protein", []) if m not in names]
    for n in names:
        state.get("sources", {}).pop(n, None)
    _write_state(adata, state)


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def demux(adata, tags: list[str], method: str = HTODEMUX, quantile: float = POSITIVE_QUANTILE) -> dict:
    """Call every cell's tag and save it: the ``hash_*`` obs columns, the record
    in ``uns`` and provenance. Antibody features among ``tags`` are moved into
    the tag counts; features moved earlier but no longer listed go back to the
    protein modality. Everything is checked (and the call computed) before the
    dataset changes, so a HashingError leaves it as it was. Returns ``summary``."""
    from scview.core import provenance
    from scview.core.obs_writes import forget_column_results

    counts, where = gather(adata, tags)
    call = run(counts, tags, method, quantile)  # raises before any change

    key = _tag_key(adata)
    if key not in (None, HTO_KEY):  # another tool's table: copied under scView's key
        frame = adata.obsm[key]
        adata.obsm[HTO_KEY] = frame.astype(np.float32)
        state = read_state(adata)
        state.setdefault("sources", {}).update({str(c): f"obsm {key}" for c in frame.columns})
        _write_state(adata, state)
    back = [m for m in read_state(adata).get("moved_from_protein", []) if m not in tags]
    _move_back(adata, back)
    moved = [t for t, w in zip(tags, where) if w == "protein"]
    _move_to_tags(adata, moved)

    obs = adata.obs
    for col, values in call_columns(call).items():
        obs[col] = values
    n_cls = {c: int((call.cls == c).sum()) for c in CLASSES}
    state = read_state(adata)
    sources = state.get("sources", {})
    features = {t: str(state.get("original", {}).get(t, t)) for t in tags}
    record = {
        "method": call.method, "label": METHOD_LABELS[call.method], "tags": list(tags),
        "params": call.params, "thresholds": call.thresholds, "per_tag": call.per_tag,
        "n_cells": int(adata.n_obs), "counts": n_cls, "when": _now(),
    }
    state["call"] = record
    _write_state(adata, state)
    for col in OBS_COLUMNS:
        forget_column_results(adata, col)
    samples = state.get("samples")
    redone = None
    if isinstance(samples, dict) and samples.get("column") in obs.columns:
        # The sample column follows the new call (same mapping; new tags keep their names).
        mapping = {t: (samples.get("mapping") or {}).get(t, t) for t in tags}
        _write_samples(adata, mapping, samples["column"], record_step=False)
        redone = {"column": samples["column"], "mapping": mapping}
    try:
        provenance.record_step(adata, step="hashing_demux", tool=f"scview hashing ({METHOD_LABELS[call.method]})",
                               params={
                                   "method": call.method, "tags": list(tags),
                                   "features": features,
                                   "sources": {t: sources.get(t, "") for t in tags},
                                   "moved_from_protein": moved, "moved_back_to_protein": back,
                                   **call.params,
                                   "cutoffs": {p["tag"]: p["cutoff"] for p in call.per_tag if "cutoff" in p},
                                   "thresholds": call.thresholds, "counts": n_cls,
                                   "sample_column": redone,
                               })
    except Exception as e:  # provenance must never block the result
        logger.warning("Could not record the hashing call in provenance: %s", e)
    return summary(adata)


def _clean_mapping(mapping: dict, tags: list[str]) -> dict[str, str]:
    from scview.core.user_labels import LabelError, clean_label

    if not isinstance(mapping, dict):
        raise HashingError("The mapping must be an object: tag -> sample name.")
    unknown = [k for k in mapping if k not in tags]
    if unknown:
        raise HashingError(f"Not a tag of the saved call: {', '.join(map(str, unknown[:5]))}.")
    out = {}
    for t in tags:
        v = mapping.get(t)
        if v is None or (isinstance(v, str) and not v.strip()):
            out[t] = t
            continue
        try:
            out[t] = clean_label(v)
        except LabelError as e:
            raise HashingError(f"{t}: {e}") from None
    return out


def _write_samples(adata, mapping: dict[str, str], column: str, record_step: bool = True) -> dict:
    from scview.core import provenance
    from scview.core.obs_writes import forget_column_results

    calls = adata.obs[CALL].astype(object).to_numpy()
    values = np.array([mapping.get(c) if c in mapping else None for c in calls], dtype=object)
    order = list(dict.fromkeys(mapping[t] for t in mapping))
    adata.obs[column] = pd.Categorical(values, categories=order)
    state = read_state(adata)
    state["samples"] = {"column": column, "mapping": mapping}
    _write_state(adata, state)
    forget_column_results(adata, column)
    counts = adata.obs[column].value_counts(dropna=False)
    n_missing = int(adata.obs[column].isna().sum())
    if record_step:
        try:
            provenance.record_step(adata, step="hashing_samples", tool="scview hashing (tag -> sample)",
                                   params={"column": column, "mapping": mapping, "source": CALL,
                                           "n_unassigned": n_missing})
        except Exception as e:
            logger.warning("Could not record the sample column in provenance: %s", e)
    return {"column": column, "mapping": mapping,
            "counts": {str(k): int(v) for k, v in counts.items() if not pd.isna(k)},
            "n_unassigned": n_missing}


def sample_column(adata, mapping: dict, column: str = DEFAULT_SAMPLE_COLUMN) -> dict:
    """Write ``column``: each singlet's sample name from ``mapping`` (tag ->
    name; a tag left out keeps its own name), doublets and negatives empty. A
    column this module wrote before is replaced; any other existing column is
    not (409)."""
    from scview.core.user_labels import LabelError, clean_column_name

    state = read_state(adata)
    call = state.get("call")
    if not isinstance(call, dict) or CALL not in adata.obs.columns:
        raise HashingError("Demultiplex first: there is no hashing call to name samples from.", status=409)
    tags = [str(t) for t in call.get("tags", [])]
    clean = _clean_mapping(mapping, tags)
    ours = (state.get("samples") or {}).get("column")
    name = column.strip() if isinstance(column, str) else ""
    if name != ours:
        try:
            name = clean_column_name(name, adata.obs.columns)
        except LabelError as e:
            raise HashingError(str(e), status=e.status) from None
    if name in OBS_COLUMNS:
        raise HashingError(f"'{name}' is one of the hashing call's own columns.", status=409)
    if ours and ours != name and ours in adata.obs.columns:
        del adata.obs[ours]  # renamed: the old copy goes
    return _write_samples(adata, clean, name)


# ---------------------------------------------------------------------------
# What the card shows
# ---------------------------------------------------------------------------


def _doublet_source(obs: pd.DataFrame) -> str | None:
    from scview.core import cluster_qc

    if "predicted_doublet" in obs.columns:
        return "predicted_doublet"
    names = [c for c in cluster_qc._DOUBLET_CLASS_COLS if c in obs.columns]
    names += [c for c in obs.columns if str(c).startswith(cluster_qc._DOUBLET_CLASS_PREFIX)]
    return str(names[0]) if names else None


def summary(adata) -> dict:
    """Everything the hashing card shows: candidate tags, the saved call (counts
    from the current cells), the sample column and how hashing doublets compare
    with Scrublet's calls."""
    cand = candidates(adata)
    state = read_state(adata)
    obs = adata.obs
    out: dict[str, Any] = {
        "n_cells": int(adata.n_obs),
        "available": bool(cand["tags"] or cand["proteins"]),
        **cand,
        "methods": [{"id": m, "label": METHOD_LABELS[m]} for m in METHODS],
        "defaults": {"method": HTODEMUX, "quantile": POSITIVE_QUANTILE},
        "limits": {"min_tags": MIN_TAGS, "max_tags": MAX_TAGS, "min_cells": MIN_CELLS},
        "default_sample_column": DEFAULT_SAMPLE_COLUMN,
        "call": None,
        "samples": None,
        "scrublet": None,
        "doublet_source": _doublet_source(obs),
    }
    call = state.get("call")
    if isinstance(call, dict) and CALL in obs.columns and CLASS in obs.columns:
        cls = obs[CLASS].astype(object)
        per_call = obs[CALL].astype(object).value_counts()
        out["call"] = {
            **{k: call.get(k) for k in ("method", "label", "tags", "params", "thresholds", "per_tag", "when")},
            "n_cells_at_call": call.get("n_cells"),
            "counts": {c: int((cls == c).sum()) for c in CLASSES},
            "per_call": {str(k): int(v) for k, v in per_call.items()},
            "n_uncalled": int(cls.isna().sum()),
        }
        if "predicted_doublet" in obs.columns:
            from scview.core import cluster_qc

            scr = cluster_qc.doublet_calls(obs[["predicted_doublet"]])
            if scr is not None:
                out["scrublet"] = {c: {"scrublet_doublet": int(((cls == c).to_numpy() & scr).sum()),
                                       "scrublet_singlet": int(((cls == c).to_numpy() & ~scr).sum())}
                                   for c in CLASSES}
    samples = state.get("samples")
    if isinstance(samples, dict) and samples.get("column") in obs.columns:
        col = obs[samples["column"]]
        out["samples"] = {"column": samples["column"], "mapping": samples.get("mapping") or {},
                          "counts": {str(k): int(v) for k, v in col.value_counts().items()},
                          "n_unassigned": int(col.isna().sum())}
    return out


def _hist(values: np.ndarray, edges: np.ndarray, groups: dict[str, np.ndarray]) -> dict[str, list[int]]:
    return {g: np.histogram(values[m], bins=edges)[0].tolist() for g, m in groups.items()}


def plot_data(adata) -> dict:
    """Per tag: a CLR histogram split by the call (this tag's singlets, all
    doublets, the rest) with the cutoff drawn in CLR units; the tag x call
    heatmap (mean CLR of each tag in each call group); for hashedDrops the
    LogFC / LogFC2 histograms with their thresholds."""
    state = read_state(adata)
    call = state.get("call") if isinstance(state.get("call"), dict) else None
    frame = tag_counts_frame(adata)
    if call and CALL in adata.obs.columns:
        tags = [t for t in call.get("tags", []) if frame is not None and t in frame.columns]
    else:
        call = None
        tags = [str(c) for c in frame.columns] if frame is not None else []
    if not tags:
        return {"available": False, "tags": [], "n_cells": int(adata.n_obs)}
    counts = frame[tags].to_numpy(dtype=np.float64)
    denom = clr_denominators(counts)
    data = clr_tags(counts)
    called = adata.obs[CALL].astype(object).to_numpy() if call else np.full(adata.n_obs, None, dtype=object)
    doublet = called == DOUBLET
    cutoffs = {p["tag"]: p.get("cutoff") for p in (call or {}).get("per_tag", [])}
    per_tag = []
    for j, t in enumerate(tags):
        top = float(data[:, j].max()) if data.shape[0] else 1.0
        edges = np.linspace(0.0, max(top, 1e-6), HIST_BINS + 1)
        this = called == t
        groups = {"tag": this, "doublet": doublet, "other": ~this & ~doublet} if call else {
            "all": np.ones(len(called), dtype=bool)}
        c = cutoffs.get(t)
        thr = (math.log1p((c + 0.5) / denom[j]) if c is not None and denom[j] > 0 else None)
        per_tag.append({
            "tag": t, "bin_edges": np.round(edges, 4).tolist(), "counts": _hist(data[:, j], edges, groups),
            "cutoff": c, "cutoff_clr": None if thr is None else round(thr, 4),
            "n_above": None if c is None else int((counts[:, j] > c).sum()),
            "median_count": float(np.median(counts[:, j])),
        })
    out: dict[str, Any] = {"available": True, "tags": tags, "n_cells": int(adata.n_obs), "per_tag": per_tag,
                           "method": call.get("method") if call else None, "heatmap": None, "lfc": None}
    if call:
        cols = [c for c in list(tags) + [DOUBLET, NEGATIVE] if (called == c).any()]
        out["heatmap"] = {
            "tags": tags, "calls": cols,
            "n": [int((called == c).sum()) for c in cols],
            "mean_clr": [[round(float(data[called == c, j].mean()), 4) for c in cols] for j in range(len(tags))],
        }
        if call.get("method") == HASHEDDROPS:
            out["lfc"] = _lfc_hists(counts, call.get("thresholds") or {})
    return out


def _lfc_hists(counts: np.ndarray, thresholds: dict) -> dict:
    """LogFC (top over second tag) and LogFC2 (second over ambient) per cell,
    on the current cells, binned, with the saved call's thresholds."""
    stats = hashed_drops_stats(counts)
    out = {}
    for key, values, thr in (("lfc", stats["lfc"], thresholds.get("confident_lfc")),
                             ("lfc2", stats["lfc2"], thresholds.get("doublet_lfc2"))):
        v = values[np.isfinite(values)]
        if not v.size:
            out[key] = None
            continue
        edges = np.linspace(float(min(v.min(), 0.0)), float(max(v.max(), 1e-6)), HIST_BINS + 1)
        out[key] = {"bin_edges": np.round(edges, 4).tolist(), "counts": np.histogram(v, bins=edges)[0].tolist(),
                    "threshold": thr}
    return out


# What the replay script copies from this module (core/replay_script.py): the
# calls themselves, so the script computes exactly what scView did.
REPLAY_CONSTANTS = ("CALL", "CLASS", "MAX", "SECOND", "MARGIN", "SINGLET", "DOUBLET", "NEGATIVE", "CLASSES",
                    "HTODEMUX", "HASHEDDROPS", "POSITIVE_QUANTILE", "CLARA_SAMPLES", "MIN_PROP",
                    "PSEUDO_COUNT", "DOUBLET_NMADS", "DOUBLET_MIN", "CONFIDENT_NMADS", "CONFIDENT_MIN")
REPLAY_CODE = (HashingError, clr_denominators, clr_tags, _randm, _sq_dist, _seqsum, _bswap2, clara, nb_fit,
               nb_quantile, Call, htodemux, _lower_mode, ambient_profile, _mad, hashed_drops_stats,
               hashed_drops, call_columns)


def grounding_lines(adata) -> tuple[list[str], str]:
    """The co-pilot's fact: the saved hashing call on the current cells (method,
    tags, singlets / doublets / negatives, singlets per sample, the cutoffs), or
    the tags found when nothing is called yet. ``(lines, detail)``."""
    s = summary(adata)
    call = s["call"]
    if not call:
        found = s["tags"] or s["suggested"]
        if not found:
            return [], ""
        where = "tag counts" if s["tags"] else "antibody features that look like hashtags"
        line = (f"- Cell hashing: {len(found)} {where} ({', '.join(found[:12])}), not demultiplexed yet "
                f"(Data Assessment > Cell hashing) [{TAG}]")
        return [line], f"{len(found)} tags, no call"
    c = call["counts"]
    n = sum(c.values())
    pct = (lambda k: f"{c[k] / n:.1%}") if n else (lambda k: "0%")
    tags = call.get("tags") or []
    samples = s.get("samples")
    if samples:
        per = ", ".join(f"{k} {v:,}" for k, v in samples["counts"].items())
        per_text = f"cells per sample ('{samples['column']}'): {per}"
    else:
        per = ", ".join(f"{t} {call['per_call'].get(t, 0):,}" for t in tags)
        per_text = f"singlets per tag: {per}"
    if call.get("method") == HTODEMUX:
        cuts = ", ".join(f"{p['tag']} {p['cutoff']}" for p in call.get("per_tag") or [] if "cutoff" in p)
        rule = f"positive above the background fit's {call['params'].get('positive_quantile', 0.99)} quantile; " \
               f"count cutoffs: {cuts}"
    else:
        th = call.get("thresholds") or {}
        rule = (f"doublet when LogFC2 > {th.get('doublet_lfc2')}, confident singlet when LogFC > "
                f"{th.get('confident_lfc')}; 'Negative' = not confidently assigned")
    used = s.get("doublet_source")
    views = ("the doublet views use these hashing calls" if used == CLASS
             else f"the doublet views use '{used}'" if used else "")
    line = (f"- Cell hashing ({call.get('label')}, {len(tags)} tags): of {n:,} cells, {c[SINGLET]:,} singlets "
            f"({pct(SINGLET)}), {c[DOUBLET]:,} doublets ({pct(DOUBLET)}), {c[NEGATIVE]:,} negatives "
            f"({pct(NEGATIVE)}); {per_text}; {rule}" + (f"; {views}" if views else "") + f" [{TAG}]")
    return [line], f"{call.get('label')}: {c[SINGLET]:,} singlets, {c[DOUBLET]:,} doublets, {c[NEGATIVE]:,} negatives"
