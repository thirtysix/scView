"""Co-occurrence pair counting (numpy, numba): the distances, in a child process.

The score and the thresholds are squidpy 1.6.5's ``sq.gr.co_occurrence``
(``gr/_ppatterns.py``, BSD 3-Clause License, Copyright (c) 2021, Theislab); see
``core/spatial_cooccurrence.py`` for what they mean and where this differs from it. Here is
only the arithmetic: every pair within the largest distance is visited once, its distance
put in a bin per pair of labels, and the bins added up afterwards, so memory is one table of
labels x labels x thresholds per thread, never cells x cells.

Run as ``python -m scview.core.cooccurrence_kernel`` it is the child: arrays in on stdin as
.npz bytes, one JSON progress line per finished block on stdout, then ``result <bytes>`` and
the arrays. numba's compile and the distances never touch the server's GIL, Cancel ends the
child at once, and the child imports only numpy and numba.

``REPLAY_CODE`` is copied into replay scripts: numba comes with scanpy.
"""

from __future__ import annotations

import io
import json
import queue
import sys
from concurrent.futures import ThreadPoolExecutor

import numpy as np

SUM = "sum"  # every pair counted once, then one score (= squidpy with n_splits=1)
SQUIDPY_SPLITS = "squidpy_splits"  # squidpy's automatic n_splits above SPLIT_ABOVE_MB: mean of the blocks' scores
SPLIT_ABOVE_MB = 2000  # squidpy: n_obs ** 2 * 4 bytes
SPLIT_SIZE = 2048  # squidpy: at most this many observations a split
BLOCK_PAIRS = 200_000_000  # pairs between a progress line and the next


def distance32(a, b):
    """A distance as ``sklearn.pairwise_distances`` gives it for float32 input: the squared
    distance in float64, rounded to float32, then its float32 square root."""
    d = np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)
    return np.sqrt(np.float32(np.sum(d * d)))


def find_min_max(spatial):
    """squidpy's ``_find_min_max``: the lowest threshold is the distance between the two
    observations with the smallest x + y, the highest half the distance from the first of
    those to the one with the largest x + y."""
    coord_sum = np.sum(spatial, axis=1)
    min_idx, min_idx2 = np.argpartition(coord_sum, 2)[:2]
    max_idx = np.argmax(coord_sum)
    return distance32(spatial[min_idx], spatial[min_idx2]), distance32(spatial[min_idx], spatial[max_idx]) / 2.0


def default_interval(spatial, sections, n):
    """``n`` thresholds as squidpy's ``np.linspace(min, max, num=n, dtype=float32)``. With
    several sections: from the smallest lowest to the largest highest of the sections (of 3
    observations or more)."""
    bounds = [find_min_max(spatial[sections == s]) for s in np.unique(sections) if (sections == s).sum() >= 3]
    return np.linspace(min(b[0] for b in bounds), max(b[1] for b in bounds), num=n, dtype=np.float32)


def sample_rows(codes, sections, cap, seed):
    """Rows of a seeded sample of about ``cap``: the same share of each label in each section
    (rounded up, so a rare label keeps a cell). All rows, in order, at or under ``cap``."""
    n = codes.shape[0]
    if n <= cap:
        return np.arange(n)
    rng = np.random.default_rng(seed)
    picked = []
    groups = sections.astype(np.int64) * (int(codes.max()) + 1) + codes
    for g in np.unique(groups):
        rows = np.flatnonzero(groups == g)
        picked.append(rng.choice(rows, size=min(rows.size, int(np.ceil(rows.size * cap / n))), replace=False))
    return np.sort(np.concatenate(picked))


def bin_lookup(edges, size=4096):
    """(table, cells per unit): a distance's table cell gives a bin at or before its own."""
    per_unit = size / float(edges[-1]) if edges[-1] > 0 else 1.0
    table = np.searchsorted(edges, (np.arange(size + 2, dtype=np.float64) - 1.0) / per_unit, side="left")
    return np.minimum(table, len(edges) - 1).astype(np.int32), per_unit


def count_triangle(x, y, codes, lo, hi, edges, table, per_unit, n_labels, hist):
    """Pairs (i, j), lo <= i < hi, i < j, of observations sorted by x: ``hist`` gets one per
    pair in its labels' row, in the first bin whose upper edge holds the distance. A pair
    at distance 0 is not counted (squidpy's ``pw_dist > 0``). Compiled with numba."""
    n = x.shape[0]
    n_bins = edges.shape[0]
    top = edges[n_bins - 1]
    reach = float(top) * (1.0 + 1e-6)
    for i in range(lo, hi):
        xi = float(x[i])
        yi = float(y[i])
        row = int(codes[i]) * n_labels
        for j in range(i + 1, n):
            dx = float(x[j]) - xi
            if dx > reach:
                break  # sorted by x: all the others are further
            dy = float(y[j]) - yi
            d = np.sqrt(np.float32(dx * dx + dy * dy))
            if d > top or d == 0:
                continue
            b = table[int(d * per_unit)]
            while d > edges[b]:
                b += 1
            hist[(row + int(codes[j])) * n_bins + b] += 1


def count_rectangle(xa, ya, ca, xb, yb, cb, edges, table, per_unit, n_labels, hist):
    """As ``count_triangle`` for every pair of two different sets. Compiled with numba."""
    n_bins = edges.shape[0]
    top = edges[n_bins - 1]
    for i in range(xa.shape[0]):
        xi = float(xa[i])
        yi = float(ya[i])
        row = int(ca[i]) * n_labels
        for j in range(xb.shape[0]):
            dx = float(xb[j]) - xi
            dy = float(yb[j]) - yi
            d = np.sqrt(np.float32(dx * dx + dy * dy))
            if d > top or d == 0:
                continue
            b = table[int(d * per_unit)]
            while d > edges[b]:
                b += 1
            hist[(row + int(cb[j])) * n_bins + b] += 1


def cooccurrence_scores(count):
    """P(b | a) / P(b) from cumulative ordered pair counts (labels x labels x thresholds),
    each probability from the pairs alone; 0 where a label has no pair, as squidpy."""
    count = np.asarray(count, dtype=np.float64)
    row = count.sum(axis=1, keepdims=True)
    column = count.sum(axis=0, keepdims=True)
    total = count.sum(axis=(0, 1), keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = (count / row) / (column / total)
    out[~np.isfinite(out)] = 0.0
    return out


def visited_pairs(x, top):
    """The pairs ``count_triangle`` visits per observation, for x sorted."""
    reach = np.searchsorted(x, x.astype(np.float64) + float(top) * (1.0 + 1e-6), side="right")
    return reach - np.arange(x.shape[0], dtype=np.int64) - 1


def _cumulative(hist, n_labels, n_bins, order):
    """Binned unordered pairs -> cumulative ordered pairs at the thresholds' own order."""
    table = hist.reshape(n_labels, n_labels, n_bins)
    table = table + table.transpose(1, 0, 2)  # (a, b) and (b, a), so a pair within a label is 2
    return np.cumsum(table, axis=2)[:, :, order]


def _edges(interval):
    """The thresholds in use (squidpy reads every entry but the first as an upper bound),
    sorted for the bins, and how to give the counts back in the order asked."""
    upper = np.asarray(interval[1:], dtype=np.float32)
    order = np.argsort(upper, kind="stable")
    inverse = np.empty_like(order)
    inverse[order] = np.arange(order.size)
    return upper[order], inverse


def pair_counts(spatial, codes, n_labels, sections, interval, threads, kernels, progress=None):
    """Cumulative ordered pairs (labels x labels x thresholds), section by section."""
    edges, inverse = _edges(interval)
    table, per_unit = bin_lookup(edges)
    triangle = kernels[0]
    jobs, total = [], 0
    for s in np.unique(sections):
        rows = np.flatnonzero(sections == s)
        rows = rows[np.argsort(spatial[rows, 0], kind="stable")]
        x, y, c = spatial[rows, 0], spatial[rows, 1], codes[rows]
        visits = visited_pairs(x, edges[-1])
        total += int(visits.sum())
        ends = np.cumsum(visits)
        lo = 0
        while lo < rows.size:
            hi = int(np.searchsorted(ends, (ends[lo - 1] if lo else 0) + BLOCK_PAIRS, side="left")) + 1
            hi = min(hi, rows.size)
            jobs.append((x, y, c, lo, hi, int(ends[hi - 1] - (ends[lo - 1] if lo else 0))))
            lo = hi
    size = n_labels * n_labels * edges.size
    free: queue.SimpleQueue = queue.SimpleQueue()
    hists = [np.zeros(size, dtype=np.int64) for _ in range(max(1, threads))]
    for h in hists:
        free.put(h)

    def work(job):
        x, y, c, lo, hi, n_pairs = job
        hist = free.get()
        triangle(x, y, c, lo, hi, edges, table, per_unit, n_labels, hist)
        free.put(hist)
        return n_pairs

    done = 0
    with ThreadPoolExecutor(max_workers=len(hists)) as pool:
        for n_pairs in pool.map(work, jobs):
            done += n_pairs
            if progress is not None:
                progress(done, total)
    return _cumulative(np.sum(hists, axis=0), n_labels, edges.size, inverse), total


def n_splits_of(n_obs):
    """squidpy's automatic ``n_splits``."""
    if n_obs**2 * 4 / 1024 / 1024 <= SPLIT_ABOVE_MB:
        return 1
    n_splits = 1
    while n_obs / n_splits > SPLIT_SIZE:
        n_splits += 1
    return n_splits


def split_scores(spatial, codes, n_labels, interval, threads, kernels, progress=None, n_splits=None):
    """squidpy's default above ``SPLIT_ABOVE_MB``: the observations cut by ROW into
    ``n_splits``, one score per pair of splits (itself included), and their plain mean,
    where a pair of splits with no pair of observations in reach adds zeros.
    Returns (score, summed cumulative pairs, pairs visited)."""
    edges, inverse = _edges(interval)
    table, per_unit = bin_lookup(edges)
    triangle, rectangle = kernels
    n_obs = spatial.shape[0]
    n_splits = n_splits or n_splits_of(n_obs)
    parts = []
    for rows in np.array_split(np.arange(n_obs), n_splits):
        rows = rows[np.argsort(spatial[rows, 0], kind="stable")]
        parts.append((spatial[rows, 0], spatial[rows, 1], codes[rows]))
    pairs = list(zip(*np.triu_indices(n_splits), strict=True))
    size = n_labels * n_labels * edges.size
    free: queue.SimpleQueue = queue.SimpleQueue()
    states = [(np.zeros(size, dtype=np.int64), np.zeros((n_labels, n_labels, edges.size)),
               np.zeros((n_labels, n_labels, edges.size), dtype=np.int64)) for _ in range(max(1, threads))]
    for s in states:
        free.put(s)

    def work(pair):
        a, b = pair
        hist, scores, counts = free.get()
        hist[:] = 0
        xa, ya, ca = parts[a]
        if a == b:
            triangle(xa, ya, ca, 0, xa.size, edges, table, per_unit, n_labels, hist)
        else:
            rectangle(xa, ya, ca, *parts[b], edges, table, per_unit, n_labels, hist)
        count = _cumulative(hist, n_labels, edges.size, inverse)
        scores += cooccurrence_scores(count)
        counts += count
        free.put((hist, scores, counts))

    with ThreadPoolExecutor(max_workers=len(states)) as pool:
        for i, _ in enumerate(pool.map(work, pairs), start=1):
            if progress is not None and (i % 50 == 0 or i == len(pairs)):
                progress(i, len(pairs))
    return (sum(s[1] for s in states) / len(pairs), sum(s[2] for s in states), n_obs * (n_obs - 1) // 2)


def compiled_kernels():
    import numba

    return numba.njit(nogil=True)(count_triangle), numba.njit(nogil=True)(count_rectangle)


def compute(spatial, codes, n_labels, sections, interval, combine=SUM, threads=2, progress=None, kernels=None):
    """(score, cumulative ordered pairs, pairs visited) at ``interval[1:]``."""
    kernels = kernels or compiled_kernels()
    if combine == SQUIDPY_SPLITS and n_splits_of(spatial.shape[0]) > 1:
        return split_scores(spatial, codes, n_labels, interval, threads, kernels, progress)
    count, visited = pair_counts(spatial, codes, n_labels, sections, interval, threads, kernels, progress)
    return cooccurrence_scores(count), count, visited


REPLAY_CODE = (distance32, find_min_max, default_interval, sample_rows, bin_lookup, count_triangle, count_rectangle,
               cooccurrence_scores, visited_pairs, _cumulative, _edges, pair_counts, n_splits_of, split_scores,
               compiled_kernels, compute)


def _peak_rss_kb() -> int:
    """This process's own high-water mark: ``VmHWM`` restarts at exec, while ``ru_maxrss``
    keeps what the server held when it forked this child."""
    with open("/proc/self/status") as f:
        for line in f:
            if line.startswith("VmHWM:"):
                return int(line.split()[1])
    return 0  # pragma: no cover - no procfs


def _child() -> None:
    out = sys.stdout.buffer

    def progress(done: int, total: int) -> None:
        out.write(json.dumps({"done": int(done), "total": int(total)}).encode() + b"\n")
        out.flush()

    with np.load(io.BytesIO(sys.stdin.buffer.read())) as d:
        score, count, visited = compute(d["spatial"], d["codes"], int(d["n_labels"]), d["sections"], d["interval"],
                                        str(d["combine"]), int(d["threads"]), progress)
    payload = io.BytesIO()
    np.savez(payload, score=score.astype(np.float32), count=count.astype(np.int64), visited=visited,
             peak_rss_kb=_peak_rss_kb())
    out.write(f"result {payload.getbuffer().nbytes}\n".encode())
    out.write(payload.getvalue())
    out.flush()


if __name__ == "__main__":
    _child()
