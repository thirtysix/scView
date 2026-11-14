"""Spatial co-occurrence: squidpy 1.6.5's ``sq.gr.co_occurrence`` (``gr/_ppatterns.py``, BSD
3-Clause License, Copyright (c) 2021, Theislab).

For each pair of labels (a, b) and each distance t, score = P(b | a) / P(b), both read from
the ordered pairs of observations no further apart than t: above 1, b turns up within t of a
more often than among all pairs at that range. It says nothing about contact or signalling,
and it is not a test. What squidpy does, kept here as it is:

- Thresholds are cumulative radii, not rings: a pair at 40 um counts at every t >= 40. The
  first entry of the interval is never a bound, so 50 entries give 49 scores.
- Default thresholds (``find_min_max``): ``np.linspace(low, high, 50)``, where low is the
  distance between the two observations with the smallest x + y (an accident of the shape,
  not a neighbour distance) and high is half the distance from one of them to the largest
  x + y. On a large section the first step is far above a cell's size (about 100 um on the
  Xenium breast section), so ``distances`` can ask for a closer range.
- P(b) comes from the pairs, not from the share of cells, and all of it is symmetric: the
  score of (a, b) equals that of (b, a).
- Pairs at distance 0 are left out (a second cell at the same place too). Distances and
  thresholds are float32, as squidpy's.

Where this differs from squidpy, each measured against it (``tests/fixtures/spatial_cooccurrence``):

- Above 22,897 observations squidpy cuts them by ROW into splits of at most 2,048
  (``n_splits``), scores each pair of splits apart and takes the plain mean, so the result
  depends on the order of the rows and a pair of far-apart splits adds zeros. On the Xenium
  breast section (82 splits) that takes the mean same-label score within 111 um from 3.94 to
  0.22 and the mean score over all pairs and distances from 1.02 to 0.47, and it correlates
  at r = -0.15 with counting all pairs (with the rows shuffled, the same mean gives r = 0.999:
  the file's rows are in spatial order). Here all pairs are counted and scored once (``SUM``),
  which is squidpy's own ``n_splits=1``, and needs no distance matrix. ``SQUIDPY_SPLITS`` is
  squidpy's mean, for one section, to reproduce it (max difference 7.6e-05 on that section).
- squidpy adds up its counts in float32, where 16,777,216 + 1 is 16,777,216: from about 4,097
  observations a large label's count can stop growing at long range. Counts here are int64.
- Several sections (``obs["library_id"]``): squidpy pools the coordinates, which counts pairs
  across sections. Here each section's pairs are counted in that section, and each section is
  scored against its OWN label shares; the sections are combined weighted by the pairs of
  (a, b) each would have with its labels placed at random (``within_section_scores``). Summed
  counts against pooled shares mixed slide composition into the score: on 8 Kuppe 2022 slides
  Fibroblast|Myeloid rose from 1.40 at 180 um to 1.71 at 4 mm because both sit on the same
  slides, while neighbourhood enrichment (shuffled within section) gave z -16.8. Each
  section's own curves are kept (uns ``occ_by_section``). The default thresholds run from the
  sections' smallest low to their largest high. Sections with different pixel sizes are
  brought to the first one's scale.
- Observations without a label are left out (squidpy cannot take them).

Scale: cells are sorted by x, so a pair beyond the largest distance in x is never visited.
Every pair within reach is visited, in a child process (``core/cooccurrence_kernel.py``),
in blocks with one table per thread: memory does not grow with the number of pairs. Above
``MAX_FULL_OBS`` a seeded per-label sample within each section is used, and the result says so.
"""

from __future__ import annotations

import contextlib
import io
import json
import logging
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable, Iterator, Mapping
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any

import numpy as np
from scipy.spatial import cKDTree

from scview.core import cooccurrence_kernel as kernel
from scview.core import off_tissue, spatial
from scview.core import spatial_domains as sd
from scview.core.preflight import cpu_quota

logger = logging.getLogger(__name__)

TAG = "result:cooccurrence"
INFO_KEY = "scview_co_occurrence"  # uns, JSON: the runs, one per column
STEP = "co_occurrence"
VERSION = 1

SUM = kernel.SUM
SQUIDPY_SPLITS = kernel.SQUIDPY_SPLITS
COMBINES = (SUM, SQUIDPY_SPLITS)
DEFAULT_INTERVAL = 50  # squidpy's
MIN_INTERVAL = 2
MAX_INTERVAL = 200
MAX_FULL_OBS = 500_000
MIN_FULL_OBS = 1_000
SPLITS_ABOVE_OBS = 22_897  # squidpy's automatic n_splits starts above this (n ** 2 * 4 bytes > 2,000 MB)
SEED = 0
MAX_THREADS = 4
VISIUM_SPOT_SPACING_UM = 100.0  # 10x: spot centres on the capture grid
# A score from fewer pairs of observations at a distance is too few to read: the heatmap greys it,
# the curves leave it out, the co-pilot and the MCP skip it. The stored result keeps every score.
# Measured (v0.78): k random cells of a common label relabelled as a rare one, 40 draws per k
# (3-160 cells), on DLPFC 151673 layers and lymph node clusters, the rare label's score against
# the whole label's at the same distance: at 100-199 pairs 5% are off by more than 2x (9 in 10
# within 1.7x); at 50-99, 12.5% (within 2.1x); at 30-49, 15.5%; at 20-29, 29% (within 2.8x).
# The pair count, not the label's cells, carries it: at a given count, 3 cells or 160 differ little.
MIN_PAIRS = 100
SECTIONS_COMBINED = "within_sections"  # each section against its own shares, weighted by its expected pairs

# Measured on the 0.53.0 image at 2 CPUs (Xenium breast, Visium HD 8 um; see the tests).
_SECONDS_STARTUP = 4.0  # the child's imports and numba's compile
_SECONDS_PER_PAIR = 2.6e-9  # at 2 threads, per pair within reach in x
_SECONDS_PER_SPLIT_PAIR = 1.75e-9
_CHILD_BYTES = 170 * 1024**2  # python, numpy, numba and LLVM in the child
_BYTES_PER_OBS = 170  # the arrays passed on and sorted, in both processes


STAGES = (("pairs", "Counting pairs at each distance"),)


@dataclass
class CoOccurrencePlan:
    n_obs: int
    n_labelled: int
    n_used: int
    unit: str
    column: str
    labels: list[str]
    interval: int | None  # the number of default thresholds; None when ``distances`` were given
    thresholds: list[float]  # the interval, in the positions' units: the first is never a bound
    combine: str
    n_splits: int
    seed: int
    max_obs: int
    libraries: list[str]
    distance_unit: str
    um_per_unit: float | None
    distance_source: str
    scales: dict[str, float] | None
    threads: int
    visited_pairs: int
    notes: list[str] = field(default_factory=list)
    estimate_seconds: float = 0.0
    estimate_bytes: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class CoOccurrenceInputs:
    plan: CoOccurrencePlan
    spatial: np.ndarray  # float32, the rows in use
    codes: np.ndarray
    sections: np.ndarray
    fingerprint: str  # which observation has which label, over all rows


@dataclass
class CoOccurrenceResult:
    occ: np.ndarray  # labels x labels x thresholds, float32
    count: np.ndarray  # the ordered pairs behind it, int64
    seconds: float
    peak_rss_bytes: int
    by_section: np.ndarray | None = None  # sections x labels x labels x thresholds pairs (several sections)


def threads() -> int:
    """Threads the child counts with: the container's CPU quota, at most ``MAX_THREADS``."""
    return max(1, min(int(cpu_quota()), MAX_THREADS))


def _scale_factor(entry: Any, name: str) -> float | None:
    sf = entry.get("scalefactors") if isinstance(entry, Mapping) else None
    v = sf.get(name) if isinstance(sf, Mapping) else None
    try:
        v = float(np.asarray(v).ravel()[0]) if v is not None else None
    except (TypeError, ValueError, IndexError):
        return None
    return v if v is not None and np.isfinite(v) and v > 0 else None


def _grid_spacing(coords: np.ndarray) -> float | None:
    """The median distance to the nearest other observation (a sample of 5,000 asked)."""
    if coords.shape[0] < 2:
        return None
    step = max(1, coords.shape[0] // 5000)
    nearest = cKDTree(coords).query(coords[::step], k=2)[0][:, 1]
    spacing = float(np.median(nearest))
    return spacing if spacing > 0 else None


def _um_per_unit(entry: Any, tech: str, coords: np.ndarray) -> tuple[float | None, str]:
    """(microns per position unit of one library, where that came from): Space Ranger's
    ``microns_per_pixel`` where it is given, a Visium HD bin's size, else on Visium the 100 um
    between spot centres (``spot_diameter_fullres`` is no ruler: 96.4 pixels on DLPFC 151673
    and 89.5 on the lymph node, with the same 137 pixels between centres)."""
    per_px = _scale_factor(entry, "microns_per_pixel")
    if per_px:
        return per_px, "Space Ranger's pixel size"
    diameter = _scale_factor(entry, "spot_diameter_fullres")
    bin_um = _scale_factor(entry, "bin_size_um")
    if diameter and bin_um:
        return bin_um / diameter, "the bin size"
    if tech == "visium":
        spacing = _grid_spacing(coords)
        if spacing:
            return VISIUM_SPOT_SPACING_UM / spacing, f"the {VISIUM_SPOT_SPACING_UM:g} um between spot centres"
    return None, ""


def distance_units(adata) -> tuple[str, float | None, dict[str, float] | None, str]:
    """(what the distance axis says, microns per position unit, each section's factor to the
    first section's scale when their pixel sizes differ by over 1%, where the size came from).
    "um" when it is known, else the positions' own "position units"."""
    tech = spatial.technology(adata)["technology"]
    if tech == "xenium":
        return "um", 1.0, None, "the positions, which are in microns"
    uns = adata.uns.get(spatial.KEY)
    libs = [str(x) for x in uns] if isinstance(uns, Mapping) else []
    if not libs:
        return "position units", None, None, ""
    rows = sd._libraries(adata)
    coords = np.asarray(adata.obsm[spatial.KEY])[:, :2]
    found = {lib: _um_per_unit(uns[lib], tech, coords[rows == lib]) for lib in libs}
    per = {lib: v for lib, (v, _) in found.items() if v}
    if len(per) < len(found):
        return "position units", None, None, ""
    first = next(iter(per.values()))
    scales = {lib: v / first for lib, v in per.items()} if any(abs(v / first - 1) > 0.01 for v in per.values()) else None
    return "um", first, scales, next(iter(found.values()))[1]


def _float_positions(adata, keep: np.ndarray, scales: dict[str, float] | None, sections: np.ndarray) -> np.ndarray:
    coords = sd._coords(adata)[keep]
    if scales:
        coords = coords.copy()
        for lib, factor in scales.items():
            coords[sections == lib] *= factor
    return coords.astype(np.float32)


def even_distances(max_distance: float, n: int) -> list[float]:
    """Evenly spaced thresholds up to ``max_distance``: 0 (never a bound), then ``n`` scores."""
    return [float(v) for v in np.linspace(0.0, float(max_distance), num=int(n) + 1)]


def check_distances(distances: list[float] | None) -> np.ndarray | None:
    """An interval of the caller's as squidpy reads one: sorted, the first entry not a bound."""
    if distances is None:
        return None
    a = np.asarray(sorted(float(d) for d in distances), dtype=np.float32)
    if a.size < MIN_INTERVAL or a.size > MAX_INTERVAL + 1 or not np.isfinite(a).all() or a[0] < 0 or a[-1] <= 0:
        raise sd.SpatialDomainsError(f"Distances must be {MIN_INTERVAL} to {MAX_INTERVAL + 1} values, none negative, "
                                     "the last above 0: fix the list, or leave it out for squidpy's default.")
    return a


def plan_cooccurrence(adata, cluster_key: str, *, interval: int = DEFAULT_INTERVAL, distances: list[float] | None = None,
                      combine: str = SUM, seed: int = SEED, max_obs: int = MAX_FULL_OBS) -> CoOccurrencePlan:
    sd._need_positions(adata)
    if combine not in COMBINES:
        raise sd.SpatialDomainsError(f"Unknown way to combine pairs '{combine}': choose {' or '.join(COMBINES)}.")
    if not MIN_INTERVAL <= int(interval) <= MAX_INTERVAL:
        raise sd.SpatialDomainsError(f"The number of distances must be between {MIN_INTERVAL} and {MAX_INTERVAL}.")
    if not MIN_FULL_OBS <= int(max_obs):
        raise sd.SpatialDomainsError(f"The largest sample must be at least {MIN_FULL_OBS:,} observations.")
    asked = check_distances(distances)
    codes, labels = sd._labels_of(adata, cluster_key)
    keep = codes >= 0
    n_labelled = int(keep.sum())
    unit = spatial.technology(adata)["unit"]
    if n_labelled < sd.MIN_OBS:
        raise sd.SpatialDomainsError(f"Only {n_labelled:,} {unit}(s) have a label in '{cluster_key}': this needs at "
                                     f"least {sd.MIN_OBS}.")
    sections = sd._libraries(adata)[keep]
    libs = sorted({str(s) for s in sections})
    if combine == SQUIDPY_SPLITS and len(libs) > 1:
        raise sd.SpatialDomainsError(f"squidpy's split mean is defined for one section and this dataset has {len(libs)}: "
                                     f"use '{SUM}', which counts each section's pairs and sums them.")
    distance_unit, um_per_unit, scales, distance_source = distance_units(adata)
    points = _float_positions(adata, keep, scales, sections)
    if asked is None and max(np.unique(sections, return_counts=True)[1]) < 3:
        raise sd.SpatialDomainsError("No section has 3 labelled observations: pick another label column.")
    thresholds = asked if asked is not None else kernel.default_interval(points, sections, int(interval))
    n_used = min(n_labelled, int(max_obs))
    n_splits = kernel.n_splits_of(n_used) if combine == SQUIDPY_SPLITS else 1
    visited = _pairs_in_reach(points, sections, float(thresholds.max()), n_used / n_labelled, n_splits)
    n_threads = threads()
    notes = []
    if n_labelled < adata.n_obs:
        notes.append(f"{adata.n_obs - n_labelled:,} {unit}(s) without a label are left out.")
    if n_used < n_labelled:
        notes.append(f"Above {int(max_obs):,} {unit}s: a sample of about {n_used:,} of {n_labelled:,}, the same share of "
                     f"each label in each section (seed {int(seed)}).")
    if len(libs) > 1:
        notes.append(f"{len(libs)} sections: pairs counted within each section, never between sections, and each "
                     "section scored against its own label shares; the sections combined weighted by the pairs of "
                     "each two labels a section would have at random. A label common on one section only does not "
                     "raise its score elsewhere.")
    if n_splits > 1:
        notes.append(f"squidpy's split mean: {n_splits} splits by row order, so the result depends on the row order and "
                     "distant splits add zeros; below 1 here does not mean less than expected.")
    if scales:
        notes.append("The sections' pixel sizes differ: positions were brought to the first section's scale.")
    off = off_tissue.note(adata, unit)
    if off:  # imported before off-tissue spots were left out at ingest
        notes.append(off)
    return CoOccurrencePlan(
        n_obs=int(adata.n_obs), n_labelled=n_labelled, n_used=n_used, unit=unit, column=cluster_key, labels=labels,
        interval=None if asked is not None else int(interval), thresholds=[float(t) for t in thresholds],
        combine=combine, n_splits=n_splits, seed=int(seed), max_obs=int(max_obs), libraries=libs,
        distance_unit=distance_unit, um_per_unit=um_per_unit, distance_source=distance_source, scales=scales,
        threads=n_threads, visited_pairs=visited,
        notes=notes, estimate_seconds=estimate_seconds(visited, n_threads, n_splits),
        estimate_bytes=estimate_bytes(n_used, len(labels), thresholds.size - 1, n_threads, len(libs)))


def _pairs_in_reach(points: np.ndarray, sections: np.ndarray, top: float, share: float, n_splits: int) -> int:
    """How many pairs a run visits: all with squidpy's splits, else those within ``top`` in x."""
    if n_splits > 1:
        n = points.shape[0] * share
        return int(n * (n - 1) / 2)
    total = 0
    for s in np.unique(sections):
        x = np.sort(points[sections == s, 0])
        total += int(kernel.visited_pairs(x, top).sum())
    return int(total * share * share)


def estimate_seconds(visited: int, n_threads: int = 2, n_splits: int = 1) -> float:
    per_pair = _SECONDS_PER_SPLIT_PAIR if n_splits > 1 else _SECONDS_PER_PAIR
    return _SECONDS_STARTUP + visited * per_pair * 2 / max(1, n_threads)


def estimate_bytes(n_used: int, n_labels: int, n_thresholds: int, n_threads: int = 2, n_sections: int = 1) -> int:
    """The child's peak: each thread's table, one per section, and the cumulative copies."""
    return int(_CHILD_BYTES + n_used * _BYTES_PER_OBS
               + (3 * n_threads + 2) * max(1, n_sections) * n_labels**2 * n_thresholds * 8)


def prepare_cooccurrence(adata, p: CoOccurrencePlan) -> CoOccurrenceInputs:
    codes, labels = sd._labels_of(adata, p.column)
    keep = codes >= 0
    sections = sd._libraries(adata)[keep]
    points = _float_positions(adata, keep, p.scales, sections)
    kept = codes[keep]
    section_codes = np.unique(sections, return_inverse=True)[1].astype(np.int32)
    rows = kernel.sample_rows(kept, section_codes, p.max_obs, p.seed)
    return CoOccurrenceInputs(plan=p, spatial=points[rows], codes=kept[rows].astype(np.int32),
                              sections=section_codes[rows], fingerprint=sd._labels_key(codes, labels))


def iter_cooccurrence(inp: CoOccurrenceInputs, stop: Callable[[], bool] = lambda: False) -> Iterator[dict[str, Any]]:
    """Run: ``stage``, ``progress`` per block of pairs, then the result. The work is a child
    process, so the server's event loop never waits on it and Cancel ends it at once."""
    from scview.core.run_watchdog import current

    p = inp.plan
    stages = sd._Stages(STAGES, stop)
    yield stages.begin("pairs")
    payload = io.BytesIO()
    np.savez(payload, spatial=inp.spatial, codes=inp.codes, n_labels=len(p.labels), sections=inp.sections,
             interval=np.asarray(p.thresholds, dtype=np.float32), combine=p.combine, threads=p.threads)
    with tempfile.TemporaryFile() as err:
        proc = subprocess.Popen([sys.executable, "-m", "scview.core.cooccurrence_kernel"],
                                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=err)
        try:
            control = current()
            with control.child(proc) if control is not None else contextlib.nullcontext():
                proc.stdin.write(payload.getvalue())
                proc.stdin.close()
            del payload
            result = None
            while result is None:
                if stop():
                    raise sd.SpatialDomainsCancelled()
                control = current()
                with control.child(proc) if control is not None else contextlib.nullcontext():
                    line = proc.stdout.readline()
                if line.startswith(b"result "):
                    result = np.load(io.BytesIO(proc.stdout.read(int(line.split()[1]))))
                elif line:
                    done = json.loads(line)
                    yield {"type": "progress", "done": done["done"], "total": done["total"],
                           "elapsed_s": round(time.monotonic() - stages.t0, 1)}
                else:
                    if stop() or (control is not None and control.cancelled):
                        raise sd.SpatialDomainsCancelled()
                    proc.wait()
                    err.seek(0)
                    tail = " ".join(err.read().decode(errors="replace").strip().splitlines()[-3:])
                    raise RuntimeError(f"the counting process ended early (exit {proc.returncode}): {tail}")
            proc.wait()
        except BrokenPipeError:
            if stop():
                raise sd.SpatialDomainsCancelled() from None
            raise
        finally:
            if proc.poll() is None:
                proc.kill()
                proc.wait()
            with contextlib.suppress(OSError):
                proc.stdin.close()
            proc.stdout.close()
    seconds = stages.close()
    yield {"type": "result", "result": CoOccurrenceResult(
        occ=result["score"], count=result["count"], seconds=seconds,
        peak_rss_bytes=int(result["peak_rss_kb"]) * 1024,
        by_section=result["by_section"] if "by_section" in result.files else None)}


def run_cooccurrence(inp: CoOccurrenceInputs) -> CoOccurrenceResult:
    for item in iter_cooccurrence(inp):
        if item["type"] == "result":
            return item["result"]
    raise sd.SpatialDomainsError("The run ended without a result.")  # pragma: no cover


def squidpy_key(column: str) -> str:
    """squidpy's uns key for the result (no "/": it would nest in the h5ad)."""
    return f"{column.replace('/', '_')}_co_occurrence"


def save_cooccurrence(adata, inp: CoOccurrenceInputs, res: CoOccurrenceResult, *, resources: dict | None = None,
                      cells_fingerprint: str | None = None) -> dict[str, Any]:
    """Stored where squidpy stores it (uns ``<column>_co_occurrence``: occ, interval), so an
    exported file reads in squidpy as its own, with the pair counts and settings beside it."""
    from scview.core import provenance

    p = inp.plan
    stored = {"occ": res.occ.astype(np.float32), "interval": np.asarray(p.thresholds, dtype=np.float32),
              "count": res.count.astype(np.int64)}
    if res.by_section is not None:  # each section's own pairs and score (the sections in plan.libraries order)
        stored["count_by_section"] = res.by_section.astype(np.int64)
        stored["occ_by_section"] = np.stack([kernel.cooccurrence_scores(c) for c in res.by_section]).astype(np.float32)
    adata.uns[squidpy_key(p.column)] = stored
    block = sd._load(adata, INFO_KEY)
    info = {"version": VERSION, "created_at": datetime.now(timezone.utc).isoformat(), "column": p.column,
            "labels": p.labels, "sizes": np.bincount(inp.codes, minlength=len(p.labels)).astype(int).tolist(),
            "n_obs": p.n_obs, "n_labelled": p.n_labelled, "n_used": p.n_used, "unit": p.unit, "interval": p.interval,
            "combine": p.combine, "n_splits": p.n_splits, "seed": p.seed, "max_obs": p.max_obs, "libraries": p.libraries,
            "distance_unit": p.distance_unit, "um_per_unit": p.um_per_unit, "distance_source": p.distance_source,
            "notes": p.notes, "sections_combined": SECTIONS_COMBINED if res.by_section is not None else None,
            "visited_pairs": p.visited_pairs, "threads": p.threads, "seconds": round(res.seconds, 1),
            "peak_rss_gb": round(res.peak_rss_bytes / 1024**3, 2), "cells_fingerprint": cells_fingerprint,
            "labels_fingerprint": inp.fingerprint}
    adata.uns[INFO_KEY] = json.dumps({"latest": p.column, "runs": {**block["runs"], p.column: info}})
    provenance.record_step(adata, step=STEP, tool="scview.spatial_cooccurrence (= squidpy 1.6 co_occurrence)",
                           params={"column": p.column, "labels": p.labels, "interval": p.interval,
                                   "distances": None if p.interval is not None else p.thresholds, "combine": p.combine,
                                   "max_obs": p.max_obs, "seed": p.seed, "scales": p.scales}, resources=resources)
    return cooccurrence_summary(adata, p.column, cells_fingerprint)


def distances_shown(thresholds: np.ndarray, um_per_unit: float | None) -> list[float]:
    """Each score's distance on the plot's axis: the interval's upper bounds, in microns when
    the pixel size is known."""
    upper = np.asarray(thresholds, dtype=np.float64)[1:] * (um_per_unit or 1.0)
    return [round(float(v), 2) for v in upper]


def section_pairs(stored: Any, libraries: list[str]) -> list[dict[str, Any]]:
    """Each section's pairs at the longest distance (what weighs it), when the run kept them."""
    if "count_by_section" not in stored:
        return []
    counts = np.asarray(stored["count_by_section"])
    return [{"library_id": lib, "pairs": int(counts[k, :, :, -1].sum())} for k, lib in enumerate(libraries)]


def cooccurrence_summary(adata, column: str | None = None, cells_fingerprint: str | None = None,
                         section: str | None = None) -> dict[str, Any]:
    """A saved run with its arrays (``column``'s, else the latest); ``available`` False without one.
    ``section``: that section's own score and pairs in ``occ`` and ``count`` (``section`` says
    which; None: the sections combined)."""
    block = sd._load(adata, INFO_KEY)
    runs = {c: i for c, i in block["runs"].items() if isinstance(i, dict) and squidpy_key(c) in adata.uns}
    want = column if column in runs else (block.get("latest") if block.get("latest") in runs else next(iter(runs), None))
    if want is None:
        return {"available": False, "columns": []}
    stored = adata.uns[squidpy_key(want)]
    info = runs[want]
    libraries = [str(x) for x in info.get("libraries") or []]
    reason = sd._labels_changed(adata, info, cells_fingerprint)
    occ, count = stored["occ"], stored.get("count")  # an older or squidpy-made result may lack the counts
    shown = None
    if section is not None and "occ_by_section" in stored:
        if section not in libraries:
            raise sd.SpatialDomainsError(f"No section '{section}' in this run: choose one of {', '.join(libraries)}.")
        k = libraries.index(section)
        occ, count, shown = stored["occ_by_section"][k], stored["count_by_section"][k], section
    occ = np.asarray(occ, dtype=np.float64)
    counted = count is not None
    return {"available": True, **info, "occ": np.round(occ, 4).tolist(),
            "count": np.asarray(count).astype(np.int64).tolist() if counted else None,
            "min_pairs": MIN_PAIRS if counted else None,
            "too_few": too_few_by_distance(count) if counted else None, "section": shown,
            "sections": section_pairs(stored, libraries),
            "distances": distances_shown(stored["interval"], info.get("um_per_unit")),
            "stale": reason is not None, "stale_reason": reason, "columns": list(runs)}


def too_few(count) -> np.ndarray:
    """Where a score rests on fewer than ``MIN_PAIRS`` pairs (but some): too few to read."""
    count = np.asarray(count, dtype=np.int64)
    return (count > 0) & (count < MIN_PAIRS)


def too_few_by_distance(count) -> list[int]:
    """At each distance, how many pairs of labels (each two once, a label with itself included)
    have a score from too few pairs."""
    thin = too_few(count)
    upper = np.triu(np.ones(thin.shape[:2], dtype=bool))
    return [int((thin[:, :, t] & upper).sum()) for t in range(thin.shape[2])]


def first_readable(count, n_labels: int) -> int:
    """The heatmap's first view and the co-pilot's fact: the shortest distance where most pairs
    of labels reach ``MIN_PAIRS`` (on Visium at 28 labels the old rule opened at 183 um with 660
    of 784 cells grey); else the shortest where more than the diagonal does; else the last."""
    count = np.asarray(count, dtype=np.int64)
    readable = [int((count[:, :, t] >= MIN_PAIRS).sum()) for t in range(count.shape[2])]
    most = [t for t, n in enumerate(readable) if 2 * n > n_labels * n_labels]
    usable = [t for t, n in enumerate(readable) if n > n_labels]
    return most[0] if most else usable[0] if usable else count.shape[2] - 1


def _top_pairs(s: dict[str, Any], n: int = 3) -> tuple[list[str], list[str], int, int]:
    """The highest and lowest pairs of different labels at the shortest distance with at least
    ``MIN_PAIRS`` pairs for each: as text, with the index of that distance and how many pairs of
    different labels were left out there for too few pairs."""
    occ = np.asarray(s["occ"], dtype=np.float64)
    labels = s["labels"]
    if s.get("count") is None:  # saved without counts: every score as it is
        count = np.full(occ.shape, MIN_PAIRS, dtype=np.int64)
    else:
        count = np.asarray(s["count"], dtype=np.int64)
    t = first_readable(count, len(labels))
    off = [(i, j) for i in range(len(labels)) for j in range(i + 1, len(labels))]
    pairs = [(float(occ[i, j, t]), labels[i], labels[j]) for i, j in off if count[i, j, t] >= MIN_PAIRS]
    left_out = sum(1 for i, j in off if 0 < count[i, j, t] < MIN_PAIRS)
    high = sorted((x for x in pairs if x[0] > 1), key=lambda x: -x[0])[:n]
    low = sorted((x for x in pairs if x[0] < 1), key=lambda x: x[0])[:n]

    def text(rows: list) -> list[str]:
        return [f"{a} with {b} (score {v:.2f})" for v, a, b in rows]

    return text(high), text(low), t, left_out


def grounding_lines(adata, cells_fingerprint: str | None = None) -> tuple[list[str], str]:
    """The saved run as one fact: the pairs that co-occur most at the shortest distance with
    enough pairs. Nothing when it no longer fits the dataset."""
    s = cooccurrence_summary(adata, None, cells_fingerprint)
    if not s.get("available") or s.get("stale"):
        return [], ""
    high, low, t, left_out = _top_pairs(s)
    at = f"{s['distances'][t]:g} {s['distance_unit']}"
    detail = f"Co-occurrence of '{s['column']}' on {s['n_used']:,} {s['unit']}s, within {at}"
    sampled = f" (a sample of {s['n_used']:,} of {s['n_labelled']:,})" if s["n_used"] < s["n_labelled"] else ""
    splits = s.get("combine") == SQUIDPY_SPLITS  # low scores there come from far-apart splits, not from the tissue
    parts = ([f"most together: {', '.join(high)}"] if high else []) + (
        [f"most apart: {', '.join(low)}"] if low and not splits else [])
    within = (f" ({len(s['libraries'])} sections, each scored against its own label shares, then combined)"
              if s.get("sections_combined") == SECTIONS_COMBINED else "")
    lines = ["\n## Spatial co-occurrence (last run)",
             f"- {detail}{sampled}{within}: {'; '.join(parts) if parts else 'no pair with enough observations'} "
             f"[{TAG}].",
             "- The score is P(b | a) / P(b) among pairs within that distance (squidpy's): above 1, b is near a more often "
             "than among all pairs at that range. It is symmetric: a near b equals b near a. Nearby does not mean "
             "interacting."]
    if left_out:
        lines.append(f"- {left_out} pair(s) of labels have fewer than {MIN_PAIRS} pairs of {s['unit']}s within {at}: too "
                     "few to read, so left out here (a rare label can give a very high or low score by chance).")
    elif s.get("count") is None:
        lines.append("- This run was saved without its pair counts, so a score from very few pairs cannot be told apart.")
    if splits:
        lines.append("- This run reproduces squidpy's mean over row-order splits: it depends on the row order and scores run "
                     "low, so a score below 1 does not mean apart. Counting all pairs is the default.")
    return lines, detail
