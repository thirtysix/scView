"""chromVAR per-cell motif activity on the peak counts (chromVAR 1.34.1, as Signac's RunChromVAR
called it before Signac 1.17 removed that wrapper).

For each motif, a cell's deviation is how far its counts in the motif's peaks are from what its
depth predicts, against 50 sets of background peaks matched on total counts and GC:

- Peaks with no count in these cells, and peaks with no bases (``core/genome_seq.py``), are left
  out first, as ``RunChromVAR`` left out the former and ``AddMotifs`` users the latter.
- ``background_peaks``: ``getBackgroundPeaks`` with its defaults. (log10 total, GC fraction)
  whitened with the Cholesky factor of their covariance, a 50 x 50 grid over the range of each,
  each peak in its nearest grid point; a peak's 50 draws come, with replacement, from all peaks,
  each with probability dnorm(grid distance, 0, 0.1) / (peaks in that peak's grid point). Drawing a
  grid point first and then a peak in it is the same law. Seeded; R's stream is not reproduced.
- ``deviations``: ``compute_deviations_single``. observed = the motif's counts in the cell; expected =
  the motif's share of all counts x the cell's total; a set's deviation is
  (sampled - sampled expected) / sampled expected, a peak counted as often as it was drawn;
  deviation = (observed - expected) / expected - the sets' mean; z = deviation / the sets' sd.
  NaN when expected < 1 count.

The product is cells x (motifs x 50 sets), about 6e11 additions on PBMC 10k, so ``deviation_kernel``
is numba at 2 threads in a child process, a cell at a time, walking each stored count through the
peaks that drew its peak and on to their motifs. Results are stored by barcode
(``DATA_DIR/modalities/<id>/atac_chromvar.h5``), never rewritten by a run; a dataset whose cells have
changed reads them as stale. ``group_stats`` turns z a motif at a time from a by-motif file.
"""

from __future__ import annotations

import json
import os
import sys
import threading
import time
import uuid
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numba
import numpy as np
import scipy.sparse as sp
from scipy.special import ndtr
from scipy.stats import chi2, rankdata

from scview.core import atac_motifs, memory_budget, modality_store, tf_motifs
from scview.core.modality_store import ModalityStore

STORE = "atac_chromvar"
Z = "z"
DEVIATIONS = "deviations"
BY_MOTIF = "atac_chromvar_z_by_motif.npy"
SUMMARY = "atac_chromvar_summary.json"
N_ITERATIONS = 50
WIDTH = 0.1
N_BINS = 50
SEED = 0
MIN_EXPECTED = 1.0
BLOCK_VALUES = 4_000_000
THREADS = 2
STEP = "atac_chromvar"
TAG = "result:chromvar"
HEATMAP_MOTIFS = 40  # the most variable motifs in the groups x motifs view
TOP_PER_GROUP = 50
MATRIX = "counts"  # atac_import.MATRIX
STAGES = ("totals", "background", "deviations")
CAVEATS = ("A chromVAR score is relative: accessibility at a motif's peaks against matched peaks across these cells, not "
           "a measure of binding or of a TF's amount. Paralogous TFs share motifs, so their scores move together.")
METHOD = ("chromVAR computeDeviations: {iterations} sets of background peaks matched on log10 total counts and GC "
          "(getBackgroundPeaks, w = {w}, bs = {bs}, seed {seed}); z = (the motif's deviation - the sets' mean) / the sets' sd, "
          "per cell; NaN where a cell's expected count is under 1.")
CITATION = "chromVAR (Schep et al. 2017, Nat Methods, PMID 28825706)"


class ChromVarError(ValueError):
    """A run that cannot start or a read that cannot be answered: the sentence says what to do."""

    def __init__(self, message: str, status: int = 400) -> None:
        super().__init__(message)
        self.status = status


# ---------------------------------------------------------------------------
# Background peaks
# ---------------------------------------------------------------------------


def background_peaks(fragments, bias, *, iterations=N_ITERATIONS, w=WIDTH, bs=N_BINS, seed=SEED):
    """getBackgroundPeaks: peaks x iterations of peak numbers (module docstring)."""
    fragments = np.asarray(fragments, dtype=np.float64)
    if fragments.size == 0 or fragments.min() <= 0:
        raise ValueError("every peak needs at least one count")
    rng = np.random.default_rng(seed)
    norm = np.column_stack([np.log10(fragments), np.asarray(bias, dtype=np.float64)])
    lower = np.linalg.cholesky(np.cov(norm, rowvar=False))  # R's t(chol(cov))
    trans = np.linalg.solve(lower, norm.T).T
    lo, hi = trans.min(axis=0), trans.max(axis=0)
    axes = [np.linspace(lo[i], hi[i], bs) for i in range(2)]
    step = np.where(hi > lo, (hi - lo) / (bs - 1), 1.0)
    cell = np.rint((trans - lo) / step).astype(np.int64)  # the nearest grid point of each axis
    member = cell[:, 0] * bs + cell[:, 1]
    density = np.bincount(member, minlength=bs * bs)
    occupied = np.flatnonzero(density)
    gx, gy = np.meshgrid(axes[0], axes[1], indexing="ij")
    grid = np.column_stack([gx.ravel(), gy.ravel()])
    by_bin = np.argsort(member, kind="stable")
    first = np.concatenate([[0], np.cumsum(density)])
    out = np.zeros((fragments.size, iterations), dtype=np.int32)
    for b in occupied:
        d = np.sqrt(((grid[occupied] - grid[b]) ** 2).sum(axis=1))
        p = np.exp(-0.5 * (d / w) ** 2)
        cum = np.cumsum(p)
        peaks = by_bin[first[b]:first[b + 1]]
        m = peaks.size * iterations
        chosen = occupied[np.searchsorted(cum, rng.random(m) * cum[-1], side="right").clip(max=occupied.size - 1)]
        drawn = by_bin[first[chosen] + (rng.random(m) * density[chosen]).astype(np.int64)]
        out[peaks] = drawn.reshape(peaks.size, iterations)
    return out


# ---------------------------------------------------------------------------
# Deviations
# ---------------------------------------------------------------------------


@numba.njit(parallel=True, nogil=True)
def deviation_kernel(indptr, indices, data, hit_ptr, hit_idx, inv_ptr, inv_peak, inv_iter, e_obs, e_bg, n_motifs, n_iter, min_expected, dev, z):
    """One block of cells (CSR over the kept peaks): fills ``dev`` and ``z``, cells x motifs."""
    for c in numba.prange(indptr.size - 1):
        sampled = np.zeros(n_motifs * n_iter, dtype=np.float64)
        observed = np.zeros(n_motifs, dtype=np.float64)
        total = 0.0
        for q in range(indptr[c], indptr[c + 1]):
            j = indices[q]
            v = data[q]
            total += v
            for h in range(hit_ptr[j], hit_ptr[j + 1]):
                observed[hit_idx[h]] += v
            for e in range(inv_ptr[j], inv_ptr[j + 1]):
                p = inv_peak[e]
                t = inv_iter[e]
                for h in range(hit_ptr[p], hit_ptr[p + 1]):
                    sampled[hit_idx[h] * n_iter + t] += v
        for m in range(n_motifs):
            expected = e_obs[m] * total
            if not expected >= min_expected:
                dev[c, m] = np.nan
                z[c, m] = np.nan
                continue
            mean = 0.0
            for t in range(n_iter):
                sampled_expected = e_bg[m, t] * total
                mean += (sampled[m * n_iter + t] - sampled_expected) / sampled_expected
            mean /= n_iter
            ss = 0.0
            for t in range(n_iter):
                sampled_expected = e_bg[m, t] * total
                d = (sampled[m * n_iter + t] - sampled_expected) / sampled_expected - mean
                ss += d * d
            sd = np.sqrt(ss / (n_iter - 1))
            d = (observed[m] - expected) / expected - mean
            dev[c, m] = d
            z[c, m] = d / sd if sd > 0 else np.nan


@dataclass
class Model:
    """What the kernel needs, over the kept peaks."""

    hit_ptr: np.ndarray
    hit_idx: np.ndarray
    inv_ptr: np.ndarray
    inv_peak: np.ndarray
    inv_iter: np.ndarray
    e_obs: np.ndarray
    e_bg: np.ndarray
    n_motifs: int
    n_iter: int


def make_model(hit_ptr, hit_idx, n_motifs, fragments, bg):
    """Expected shares of every motif and background set, and who drew each peak."""
    expectation = fragments / fragments.sum()
    n_iter = bg.shape[1]
    per_peak = np.diff(hit_ptr)
    e_obs = np.bincount(hit_idx, weights=np.repeat(expectation, per_peak), minlength=n_motifs)
    e_bg = np.zeros((n_motifs, n_iter), dtype=np.float64)
    for t in range(n_iter):
        e_bg[:, t] = np.bincount(hit_idx, weights=np.repeat(expectation[bg[:, t]], per_peak), minlength=n_motifs)
    flat = bg.ravel()
    order = np.argsort(flat, kind="stable")
    inv_ptr = np.zeros(fragments.size + 1, dtype=np.int64)
    inv_ptr[1:] = np.cumsum(np.bincount(flat, minlength=fragments.size))
    return Model(hit_ptr.astype(np.int64), hit_idx.astype(np.int32), inv_ptr, (order // n_iter).astype(np.int32),
                 (order % n_iter).astype(np.int32), e_obs, e_bg, n_motifs, n_iter)


def deviations(model, block):
    """(deviations, z), cells x motifs, for a block of cells over the kept peaks."""
    n = block.shape[0]
    dev = np.empty((n, model.n_motifs), dtype=np.float32)
    z = np.empty((n, model.n_motifs), dtype=np.float32)
    deviation_kernel(block.indptr.astype(np.int64), block.indices.astype(np.int32), block.data.astype(np.float64),
                     model.hit_ptr, model.hit_idx, model.inv_ptr, model.inv_peak, model.inv_iter, model.e_obs, model.e_bg,
                     model.n_motifs, model.n_iter, MIN_EXPECTED, dev, z)
    return dev, z


def hit_lists(table: atac_motifs.PeakMotifTable, peaks: np.ndarray, block: int = 16_384) -> tuple[np.ndarray, np.ndarray]:
    """The hits of these peaks as CSR pointers and motif numbers, a block of peaks at a time."""
    ptr = np.zeros(peaks.size + 1, dtype=np.int64)
    parts = []
    for start in range(0, peaks.size, block):
        r, c = np.nonzero(table.hits(peaks[start:start + block]))
        ptr[start + 1:start + 1 + min(block, peaks.size - start)] = np.bincount(r, minlength=min(block, peaks.size - start))
        parts.append(c.astype(np.int32))
    return np.cumsum(ptr), np.concatenate(parts) if parts else np.zeros(0, np.int32)


def peak_totals(store: ModalityStore, rows: np.ndarray, stop: Callable[[], bool]) -> tuple[np.ndarray, int]:
    total = np.zeros(store.n_vars, dtype=np.float64)
    for _pos, block in store.iter_blocks(MATRIX, rows, BLOCK_VALUES):
        if stop():
            raise atac_motifs.ScanCancelled
        total += np.bincount(block.indices, weights=block.data, minlength=store.n_vars)
    return total, int(rows.size)


def variability(z_row: np.ndarray, n_cells: int) -> tuple[float, float]:
    """computeVariability: sd of a motif's z over the cells, and its chi-squared p."""
    ok = z_row[np.isfinite(z_row)]
    if ok.size < 2:
        return float("nan"), float("nan")
    sd = float(ok.std(ddof=1))
    return sd, float(chi2.sf((n_cells - 1) * sd * sd, n_cells - 1))


def run(store_path: Path, rows: np.ndarray, table: atac_motifs.PeakMotifTable, out_dir: Path, *, seed: int = SEED,
        iterations: int = N_ITERATIONS, w: float = WIDTH, bs: int = N_BINS, say: Callable[[dict], None] = lambda _e: None,
        stop: Callable[[], bool] = lambda: False) -> dict[str, Any]:
    """Everything, written into ``out_dir``; ``publish`` moves it beside the dataset's store."""
    t0 = time.time()
    out_dir.mkdir(parents=True, exist_ok=True)
    with ModalityStore(store_path) as store:
        say({"stage": "totals", "message": "Adding up each peak's counts"})
        total, _ = peak_totals(store, rows, stop)
        kept = np.flatnonzero((total > 0) & table.known)
        if kept.size < 2:
            raise ChromVarError("Fewer than 2 peaks have both counts and known bases, so nothing can be matched.", 422)
        where = np.full(store.n_vars, -1, dtype=np.int32)
        where[kept] = np.arange(kept.size, dtype=np.int32)
        fragments = total[kept]
        say({"stage": "background", "message": f"Drawing {iterations} sets of background peaks for {kept.size:,} peaks"})
        bg = background_peaks(fragments, table.gc[kept] / 100.0, iterations=iterations, w=w, bs=bs, seed=seed)
        ptr, idx = hit_lists(table, kept)
        model = make_model(ptr, idx, table.n_motifs, fragments, bg)
        del bg
        by_motif = np.lib.format.open_memmap(out_dir / BY_MOTIF, mode="w+", dtype=np.float32, shape=(table.n_motifs, rows.size))
        barcodes = store.barcodes[rows]
        meta = {"seed": seed, "iterations": iterations, "w": w, "bs": bs, "genome": table.genome, "table": table.key}
        attrs = {"name": table.motif_names.tolist()}
        done = 0
        with modality_store.StoreWriter(out_dir / f"{STORE}.h5", barcodes, table.motif_ids.tolist(), attrs, meta) as writer:
            for pos, block in store.iter_blocks(MATRIX, rows, BLOCK_VALUES):
                if stop():
                    raise atac_motifs.ScanCancelled
                mapped = where[block.indices]
                keep = mapped >= 0
                per_cell = np.repeat(np.arange(block.shape[0]), np.diff(block.indptr))
                lengths = np.bincount(per_cell[keep], minlength=block.shape[0])
                sub = sp.csr_matrix((block.data[keep], mapped[keep], np.concatenate([[0], np.cumsum(lengths)])),
                                    shape=(block.shape[0], kept.size))
                dev, z = deviations(model, sub)
                writer.append(Z, sp.csr_matrix(z))
                writer.append(DEVIATIONS, sp.csr_matrix(dev))
                by_motif[:, pos] = z.T
                done += block.shape[0]
                say({"stage": "deviations", "done": done, "total": int(rows.size), "seconds": round(time.time() - t0, 1),
                     "message": f"Deviations of {done:,} of {rows.size:,} cells"})
        by_motif.flush()
    sd = np.full(table.n_motifs, np.nan)
    p = np.full(table.n_motifs, np.nan)
    n_nan = np.zeros(table.n_motifs, dtype=np.int64)
    for m in range(table.n_motifs):
        row = np.asarray(by_motif[m])
        sd[m], p[m] = variability(row, int(rows.size))
        n_nan[m] = int(np.isnan(row).sum())
    tested = np.isfinite(p)
    padj = np.full(p.size, np.nan)
    if tested.any():
        padj[tested] = tf_motifs.bh(p[tested])
    return {
        "run_id": uuid.uuid4().hex[:12], "n_cells": int(rows.size), "n_peaks_used": int(kept.size),
        "n_peaks_no_counts": int(((total <= 0) & table.known).sum()), "n_peaks_left_out": int((~table.known).sum()),
        "seed": seed, "iterations": iterations, "w": w, "bs": bs, "genome": table.genome, "table": table.key,
        "motif_ids": table.motif_ids.tolist(), "motif_names": table.motif_names.tolist(),
        "variability": [None if not np.isfinite(x) else round(float(x), 5) for x in sd],
        "p": [None if not np.isfinite(x) else float(x) for x in p],
        "padj": [None if not np.isfinite(x) else float(x) for x in padj],
        "n_hits": np.bincount(idx, minlength=table.n_motifs).tolist(), "n_nan": n_nan.tolist(),
        "gc_rich": table.gc_rich.tolist(), "seconds": round(time.time() - t0, 1),
    }


REPLAY_CONSTANTS = ("MIN_EXPECTED", "N_ITERATIONS", "WIDTH", "N_BINS", "SEED")
REPLAY_CODE = (background_peaks, deviation_kernel, Model, make_model, deviations)


# ---------------------------------------------------------------------------
# Time and memory
# ---------------------------------------------------------------------------

_OPS_SECONDS = 0.92e-9  # per addition at 2 threads: PBMC 10k 591 s (98.9M values), mouse E18 240 s (41.7M)
_PREP_SECONDS = 30.0
_BASE_BYTES = 300 * 1024**2  # the child: numpy, scipy, h5py, numba and its compiled kernel
_PER_DRAW_BYTES = 4 + 8 + 4 + 4  # the draws and their sort, then who drew each peak
_PER_HIT_BYTES = 4 + 4  # the hit lists


def estimate_seconds(nnz: int, n_iterations: int, mean_hits: float) -> float:
    return _PREP_SECONDS + _OPS_SECONDS * float(nnz) * n_iterations * mean_hits


def estimate_bytes(n_cells: int, n_peaks: int, n_motifs: int, n_iterations: int, mean_hits: float) -> int:
    block_cells = min(n_cells, 1500)
    return int(_BASE_BYTES + n_peaks * n_iterations * _PER_DRAW_BYTES + n_peaks * mean_hits * _PER_HIT_BYTES
               + block_cells * n_motifs * 16 + BLOCK_VALUES * 16 + THREADS * n_motifs * n_iterations * 8)


def estimate_disk_bytes(n_cells: int, n_motifs: int) -> int:
    return int(n_cells * n_motifs * 15)


# ---------------------------------------------------------------------------
# The child
# ---------------------------------------------------------------------------


def child_cmd(store: Path, rows: Path, table: Path, out_dir: Path, seed: int) -> list[str]:
    return [sys.executable, "-m", "scview.core.atac_chromvar", "--store", str(store), "--rows", str(rows), "--table", str(table),
            "--out", str(out_dir), "--seed", str(seed)]


def _child(argv: list[str]) -> None:
    import argparse

    ap = argparse.ArgumentParser()
    for a in ("--store", "--rows", "--table", "--out"):
        ap.add_argument(a, required=True)
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args(argv)
    numba.set_num_threads(min(THREADS, numba.config.NUMBA_NUM_THREADS))
    last = [0.0]

    def say(e: dict) -> None:
        now = time.time()
        if e.get("stage") == "deviations" and e.get("done") != e.get("total") and now - last[0] < 1.0:
            return
        last[0] = now
        sys.stdout.write(json.dumps({"type": "progress", **e}) + "\n")
        sys.stdout.flush()

    table = atac_motifs.read_table_file(Path(args.table))
    summary = run(Path(args.store), np.load(args.rows), table, Path(args.out), seed=args.seed, say=say)
    summary["peak_mb"] = memory_budget.own_peak_mb()
    (Path(args.out) / SUMMARY).write_text(json.dumps(summary))
    sys.stdout.write(json.dumps({"type": "done", "seconds": summary["seconds"], "peak_mb": summary["peak_mb"]}) + "\n")
    sys.stdout.flush()


def iter_run(store: Path, rows: np.ndarray, table_file: Path, out_dir: Path, *, seed: int = SEED,
             stop: Callable[[], bool] = lambda: False) -> Iterator[dict[str, Any]]:
    """The child's events (``stage`` ones), then a ``result`` event with its summary."""
    out_dir.mkdir(parents=True, exist_ok=True)
    rows_file = out_dir / "rows.npy"
    np.save(rows_file, rows)
    t0 = time.time()
    for event in atac_motifs.iter_child(child_cmd(store, rows_file, table_file, out_dir, seed), stop, THREADS, "chromVAR"):
        if event.get("type") == "done":
            summary = json.loads((out_dir / SUMMARY).read_text())
            yield {"type": "result", "summary": {**summary, "peak_mb": event.get("peak_mb")}}
            return
        stage = str(event.get("stage"))
        yield {"type": "stage", "stage": stage, "label": event.get("message"), "index": STAGES.index(stage) + 1 if stage in STAGES else 1,
               "total": len(STAGES), "elapsed_s": round(time.time() - t0, 1)}


# ---------------------------------------------------------------------------
# Saved runs
# ---------------------------------------------------------------------------


def _folder(data_dir: str | Path, dataset_id: str) -> Path:
    return modality_store.store_dir(data_dir, dataset_id)


def staging(data_dir: str | Path, dataset_id: str) -> Path:
    return _folder(data_dir, dataset_id) / f".chromvar-{uuid.uuid4().hex[:8]}"


def publish(out_dir: Path, data_dir: str | Path, dataset_id: str, summary: dict[str, Any]) -> None:
    """Moves a finished run beside the store; the summary last, which is what names a run as there."""
    folder = _folder(data_dir, dataset_id)
    (out_dir / SUMMARY).write_text(json.dumps(summary))
    for name in (f"{STORE}.h5", BY_MOTIF, SUMMARY):
        os.replace(out_dir / name, folder / name)
    with _cache_lock:
        for k in [k for k in _stats_cache if k[0] == dataset_id]:
            del _stats_cache[k]


def read_summary(data_dir: str | Path, dataset_id: str) -> dict[str, Any] | None:
    path = _folder(data_dir, dataset_id) / SUMMARY
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


def _alignment(data_dir: str | Path, dataset_id: str, obs_names: Iterable) -> np.ndarray:
    """The run's position of each of the dataset's cells; ``ChromVarError`` when one is not in the run."""
    with ModalityStore(_folder(data_dir, dataset_id) / f"{STORE}.h5") as store:
        where = store.rows_of(obs_names)
    if (where < 0).any():
        raise ChromVarError(f"{int((where < 0).sum()):,} of this dataset's cells were not in the chromVAR run: run it again.", 409)
    return where


def motif_values(data_dir: str | Path, dataset_id: str, summary: dict[str, Any], motif_id: str, obs_names: Iterable) -> np.ndarray:
    """One motif's z per cell of the dataset, NaN where the run has none."""
    try:
        m = summary["motif_ids"].index(motif_id)
    except ValueError:
        raise ChromVarError(f"'{motif_id}' is not in the chromVAR run.", 404) from None
    z = np.load(_folder(data_dir, dataset_id) / BY_MOTIF, mmap_mode="r")
    return np.asarray(z[m])[_alignment(data_dir, dataset_id, obs_names)]


def wilcoxon_by_group(values: np.ndarray, codes: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Each group against the other labelled cells on one motif's finite z: (mean, rest mean, p),
    as ``atac.wilcoxon_sparse`` (normal approximation, tie correction, no continuity correction)."""
    ok = np.isfinite(values) & (codes >= 0)
    v, c = values[ok].astype(np.float64), codes[ok]
    n = v.size
    means, rest, p = np.full(k, np.nan), np.full(k, np.nan), np.ones(k)
    if n < 2:
        return means, rest, p
    ranks = rankdata(v)
    sizes = np.bincount(c, minlength=k).astype(np.float64)
    sums = np.bincount(c, weights=v, minlength=k)
    total = sums.sum()
    rank_sum = np.bincount(c, weights=ranks, minlength=k)
    _, tie_counts = np.unique(v, return_counts=True)
    ties = float(((tie_counts.astype(np.float64) ** 3) - tie_counts).sum())
    with np.errstate(divide="ignore", invalid="ignore"):
        means = np.where(sizes > 0, sums / sizes, np.nan)
        rest = np.where(n - sizes > 0, (total - sums) / (n - sizes), np.nan)
        u = rank_sum - sizes * (sizes + 1) / 2.0
        n2 = n - sizes
        sigma2 = sizes * n2 / 12.0 * ((n + 1) - ties / (n * (n - 1.0)))
        z = np.where(sigma2 > 0, (u - sizes * n2 / 2.0) / np.sqrt(np.maximum(sigma2, 1e-300)), 0.0)
    return means, rest, 2 * ndtr(-np.abs(z))


_stats_cache: dict[tuple[str, str, str], dict[str, Any]] = {}
_cache_lock = threading.Lock()


def group_stats(data_dir: str | Path, dataset_id: str, summary: dict[str, Any], column: str, codes: np.ndarray, groups: list[str],
                obs_names: Iterable, *, n_heatmap: int = HEATMAP_MOTIFS, n_rows: int = TOP_PER_GROUP,
                stop: Callable[[], bool] = lambda: False) -> dict[str, Any]:
    """The most variable motifs x groups (mean z), and each group's motifs against the rest
    (mean difference and Wilcoxon, BH across motifs within the group). A motif at a time."""
    key = (dataset_id, str(summary.get("run_id")), column)
    with _cache_lock:
        hit = _stats_cache.get(key)
    if hit is not None:
        return hit
    align = _alignment(data_dir, dataset_id, obs_names)
    by_motif = np.load(_folder(data_dir, dataset_id) / BY_MOTIF, mmap_mode="r")
    m, k = by_motif.shape[0], len(groups)
    mean = np.full((m, k), np.nan)
    diff = np.full((m, k), np.nan)
    p = np.ones((m, k))
    for j in range(m):
        if stop():
            raise atac_motifs.ScanCancelled
        means, rest, pv = wilcoxon_by_group(np.asarray(by_motif[j])[align], codes, k)
        mean[j], diff[j], p[j] = means, means - rest, pv
    sd = np.array([np.nan if x is None else x for x in summary["variability"]], dtype=np.float64)
    order = np.argsort(-np.nan_to_num(sd, nan=-1.0), kind="stable")
    top = order[:n_heatmap]
    names, ids = summary["motif_names"], summary["motif_ids"]
    tested = np.isfinite(diff)
    rows: dict[str, list[dict]] = {}
    gc_rich = summary.get("gc_rich") or [False] * m
    for g, group in enumerate(groups):
        ok = tested[:, g]
        padj = np.ones(m)
        if ok.any():
            padj[ok] = tf_motifs.bh(p[ok, g])
        keep = np.flatnonzero(ok)
        keep = keep[np.lexsort((p[keep, g], -diff[keep, g]))][:n_rows]
        rows[group] = [{"motif_id": ids[j], "motif": names[j], "tfs": tf_motifs.tf_members(names[j]),
                        "mean_z": round(float(mean[j, g]), 4), "mean_diff": round(float(diff[j, g]), 4), "pval": float(p[j, g]),
                        "pval_adj": float(padj[j]), "gc_rich": bool(gc_rich[j])} for j in keep]
    out = {"column": column, "groups": groups, "n_cells": int(align.size),
           "heatmap": {"motif_ids": [ids[j] for j in top], "motifs": [names[j] for j in top],
                       "variability": [None if not np.isfinite(sd[j]) else round(float(sd[j]), 4) for j in top],
                       "mean_z": [[None if not np.isfinite(x) else round(float(x), 3) for x in mean[j]] for j in top]},
           "rows": rows, "n_rows": n_rows}
    with _cache_lock:
        while len(_stats_cache) >= 6:
            _stats_cache.pop(next(iter(_stats_cache)))
        _stats_cache[key] = out
    return out


def ranking(summary: dict[str, Any], n: int = 100) -> list[dict[str, Any]]:
    """Motifs by variability, highest first."""
    sd = summary["variability"]
    order = sorted((j for j, x in enumerate(sd) if x is not None), key=lambda j: -sd[j])[:n]
    return [{"motif_id": summary["motif_ids"][j], "motif": summary["motif_names"][j], "variability": sd[j],
             "p": summary["p"][j], "padj": summary["padj"][j], "n_hits": summary["n_hits"][j], "n_nan": summary["n_nan"][j],
             "gc_rich": bool((summary.get("gc_rich") or [False] * len(sd))[j])} for j in order]


RESULT_KIND = "atac_chromvar"


def keep(adaptor, summary: dict[str, Any]) -> None:
    """The run's motif ranking in the results store, so it lists, reads and exports as a table
    like the other results (the per-cell z stay beside the peak store)."""
    from scview.core import results_store

    n = len(summary["motif_ids"])
    params = {k: summary.get(k) for k in ("seed", "iterations", "w", "bs", "genome", "table", "n_cells", "n_peaks_used",
                                          "n_peaks_left_out")}
    results_store.keep(adaptor, RESULT_KIND, {"motifs": ranking(summary, n), "method": summary.get("method"), **params},
                       result_id=str(summary["run_id"]), key="chromvar", params=params,
                       label=f"chromVAR motif variability ({summary['n_cells']:,} cells, {n:,} motifs)")


def record(adata, summary: dict[str, Any], resources: dict | None = None) -> None:
    from scview.core import provenance

    provenance.record_step(adata, step=STEP, tool="scview.atac_chromvar (chromVAR computeDeviations)",
                           params={k: summary.get(k) for k in ("seed", "iterations", "w", "bs", "genome", "table", "run_id",
                                                               "n_cells", "n_peaks_used")}, resources=resources)


def grounding_lines(data_dir: str | Path, dataset_id: str, fingerprint: str | None) -> tuple[list[str], str]:
    """The saved run on these cells as facts: the most variable motifs (nothing when stale)."""
    summary = read_summary(data_dir, dataset_id)
    if summary is None or (fingerprint is not None and summary.get("cells_fingerprint") not in (None, fingerprint)):
        return [], ""
    top = ranking(summary, 8)
    detail = f"chromVAR on {summary['n_cells']:,} cells, {summary['n_peaks_used']:,} peaks, {summary['iterations']} background sets"
    lines = ["\n## Motif activity per cell (chromVAR)", f"- {detail}; most variable motifs by sd of z: "
             + "; ".join(f"{r['motif']} ({r['motif_id']}) {r['variability']:.2f}" for r in top) + f" [{TAG}]",
             "- Caveat: a chromVAR score is relative to matched peaks across these cells, not binding or an amount of TF; "
             "paralogues share motifs and their scores move together."]
    return lines, detail


if __name__ == "__main__":
    _child(sys.argv[1:])
