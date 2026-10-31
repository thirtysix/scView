"""Xenium transcripts beside a dataset: DATA_DIR/transcripts/<dataset id>/transcripts.h5.

XOA's ``transcripts.parquet`` (0.67 GB for 43 M rows on the breast Rep1 run) is in neither gene
nor spatial order, so looking at three genes in a view would read the whole file. The store
holds, gene by gene and each gene by coarse tile, what the plot needs:

    genes         UTF-8, sorted
    gene_offsets  int64, n_genes + 1: a gene's points are ``gene_offsets[g]:gene_offsets[g + 1]``
    tile_offsets  uint32, (n_genes, TILES**2 + 1), from the gene's start; row-major tiles
    x, y          float32, in the units of ``obsm["spatial"]`` (microns), contiguous on disk
    qv            uint8, Phred quality rounded down
    assigned      int64, per gene: points with a cell id, for the check against the matrix
    attrs         summary (JSON: what was read, dropped, the check), bounds (the tile grid)

What is kept: q >= ``MIN_QV`` (20, 10x's own default) and genes only. Control features leave by
name (``CONTROL_PREFIXES``) or, in newer XOA, by the ``is_gene`` column. Both counts are in the
summary. Coordinates only, never the cell a transcript belongs to: an h5ad export carries none of
this, and filtering cells does not change what is drawn.

Built in two passes over the row groups, so memory is bound by one row group (plus the tile table):
the first counts points per gene and tile, the second puts each row group's points in their
places through a map of the file, opened afresh per row group so the pages it touches are not
held. Written through a part file: a stopped build leaves nothing that looks finished.

Check (``check_against_matrix``): per gene, the points with q >= 20 and a cell id against that
gene's counts in ``cell_feature_matrix.h5``, which XOA builds from the same rule.
"""

from __future__ import annotations

import json
import logging
import shutil
import time
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np

logger = logging.getLogger(__name__)

SUBDIR = "transcripts"
STORE = "transcripts.h5"
UPLOAD = "transcripts.parquet"
STEP = "index_transcripts"
MIN_QV = 20.0  # 10x's default: the matrix counts these
TILES = 64  # tiles per side of the grid over the section
DEFAULT_LIMIT = 300_000
MAX_LIMIT = 2_000_000
MAX_GENES = 8
# Not genes (XOA 1.0.1 calls some negative control probes antisense_*).
CONTROL_PREFIXES = ("NegControl", "BLANK", "Unassigned", "Deprecated", "Intergenic", "antisense_")
UNASSIGNED = ("UNASSIGNED", "")  # cell_id of a transcript in no cell, string ids (int ids: -1)
REQUIRED = ("cell_id", "feature_name", "x_location", "y_location", "qv")
PARQUET_MAGIC = b"PAR1"
_MATRIX_NAME = "cell_feature_matrix.h5"
_MATRIX_BLOCK = 2_000_000  # stored values of the matrix read at once
_MERGE_GAP = 16_384  # points: a smaller gap between tile runs of a view is read through
_STR = h5py.string_dtype("utf-8")

# Measured on the breast Rep1 run (43.0 M rows, 0.67 GB, two cores: see the report of batch 35).
_BYTES_PER_ROW_FILE = 15.6  # of transcripts.parquet, for an estimate before the upload
_KEPT_BYTES_PER_ROW = 7.3  # store on disk per row read (9 bytes a kept point, about 4 in 5 kept)
_SECONDS_PER_MILLION_ROWS = 0.33  # both passes
_MEMORY_PER_ROW = 60.0  # process memory, bytes per row of the largest row group (58 MB at 1 M rows)
_MEMORY_PER_TILE = 12  # the tile tables, bytes per gene and tile (15 MB at 313 genes)
# The store's pages count as used until the kernel writes them out (it can always drop them):
# 390 MB over the start on this run, 73 MB of it process memory.
_FILE_CACHE_CAP = 320 * 1024**2
_TIME_BAND = (0.7, 2.5)  # a file not yet in the page cache takes longer


class TranscriptsError(RuntimeError):
    """A sentence naming the thing and what to do; ``status`` is the HTTP code."""

    def __init__(self, message: str, status: int = 422):
        super().__init__(message)
        self.status = status


class TranscriptsCancelled(Exception):
    """The client went away: nothing was kept."""


def store_dir(data_dir: str | Path, dataset_id: str) -> Path:
    return Path(data_dir) / SUBDIR / dataset_id


def store_path(data_dir: str | Path, dataset_id: str) -> Path:
    return store_dir(data_dir, dataset_id) / STORE


def upload_path(data_dir: str | Path, dataset_id: str) -> Path:
    return store_dir(data_dir, dataset_id) / UPLOAD


def has_store(data_dir: str | Path, dataset_id: str) -> bool:
    return store_path(data_dir, dataset_id).is_file()


def remove(data_dir: str | Path, dataset_id: str) -> None:
    shutil.rmtree(store_dir(data_dir, dataset_id), ignore_errors=True)


# ---------------------------------------------------------------------------
# What a build needs, said before it starts
# ---------------------------------------------------------------------------


def estimate(rows: int, largest_row_group: int | None = None, n_genes: int = 500) -> dict[str, Any]:
    """Disk, memory and time of a build over ``rows`` rows. A row group XOA writes holds 1 M rows."""
    group = largest_row_group if largest_row_group is not None else min(rows, 1_000_000)
    seconds = rows / 1e6 * _SECONDS_PER_MILLION_ROWS
    disk = int(rows * _KEPT_BYTES_PER_ROW)
    peak = int(group * _MEMORY_PER_ROW + n_genes * TILES**2 * _MEMORY_PER_TILE + min(disk, _FILE_CACHE_CAP))
    return {"rows": int(rows), "disk_bytes": disk, "peak_bytes": peak, "seconds": round(seconds, 1),
            "seconds_range": [round(seconds * _TIME_BAND[0], 1), round(seconds * _TIME_BAND[1], 1)]}


def estimate_for_size(size_bytes: int, n_genes: int = 500) -> dict[str, Any]:
    """The estimate from a file's size alone, before anything is sent."""
    return {**estimate(int(size_bytes / _BYTES_PER_ROW_FILE), n_genes=n_genes), "upload_bytes": int(size_bytes)}


def read_plan(path: str | Path) -> dict[str, Any]:
    """Rows and row groups from the parquet footer. TranscriptsError when it is no transcript
    table. The row groups are added up: XOA 1.0.1's footer says num_rows = 1,000,000 of 42 M."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    try:
        pf = pq.ParquetFile(path)
    except (pa.ArrowException, OSError) as e:
        raise TranscriptsError(f"That file could not be read as parquet ({e}): send transcripts.parquet from "
                               "the Xenium output folder.") from e
    missing = [c for c in REQUIRED if c not in pf.schema_arrow.names]
    if missing:
        raise TranscriptsError(f"That file lacks the column(s) {', '.join(missing)} of Xenium's transcripts.parquet: "
                               "send that file from the output folder.")
    sizes = [pf.metadata.row_group(i).num_rows for i in range(pf.num_row_groups)]
    if not sizes or not sum(sizes):
        raise TranscriptsError("That file holds no transcripts: send transcripts.parquet from the output folder.")
    return {"rows": int(sum(sizes)), "row_groups": len(sizes), "largest_row_group": int(max(sizes)),
            "is_gene": "is_gene" in pf.schema_arrow.names}


def disk_free(path: str | Path) -> int:
    probe = Path(path)
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent
    return shutil.disk_usage(probe).free


# ---------------------------------------------------------------------------
# Reading a row group
# ---------------------------------------------------------------------------


def _text(value) -> str:
    return value.decode("utf-8", "replace") if isinstance(value, (bytes, bytearray)) else str(value)


def _is_control(name: str) -> bool:
    return name.startswith(CONTROL_PREFIXES)


def _codes_of(column, code_of: Callable[[str], int]) -> np.ndarray:
    """A gene code per row, -1 for a control feature or none; each name looked at once."""
    import pyarrow as pa

    encoded = column if pa.types.is_dictionary(column.type) else column.dictionary_encode()
    lookup = np.array([-1 if _is_control(n) else code_of(n)
                       for n in map(_text, encoded.dictionary.to_pylist())] + [-1], dtype=np.int32)
    indices = encoded.indices.fill_null(len(lookup) - 1)
    return lookup[indices.to_numpy(zero_copy_only=False)]


def _assigned(column) -> np.ndarray:
    """Whether each row has a cell id: an int up to XOA 1.2 (-1 for none), then text."""
    import pyarrow as pa
    import pyarrow.compute as pc

    if pa.types.is_integer(column.type):
        return column.fill_null(-1).to_numpy(zero_copy_only=False) >= 0
    if pa.types.is_binary(column.type) or pa.types.is_large_binary(column.type):
        column = column.cast(pa.string())
    elif pa.types.is_dictionary(column.type):
        column = column.dictionary_decode().cast(pa.string())
    return ~pc.is_in(column.fill_null(""), value_set=pa.array(UNASSIGNED)).to_numpy(zero_copy_only=False)


@dataclass
class _Block:
    codes: np.ndarray  # gene code per kept row
    x: np.ndarray
    y: np.ndarray
    qv: np.ndarray
    assigned: np.ndarray
    n_rows: int
    n_control: int
    n_low_quality: int


def _read_block(pf, i: int, columns: Sequence[str], code_of: Callable[[str], int]) -> _Block:
    """One row group's kept rows; never more than that row group in memory."""
    t = pf.read_row_group(i, columns=list(columns), use_threads=False)
    codes = _codes_of(t["feature_name"].combine_chunks(), code_of)
    if "is_gene" in columns:
        codes[~t["is_gene"].combine_chunks().fill_null(False).to_numpy(zero_copy_only=False)] = -1
    x = t["x_location"].to_numpy().astype(np.float32, copy=False)
    y = t["y_location"].to_numpy().astype(np.float32, copy=False)
    qv = t["qv"].to_numpy().astype(np.float32, copy=False)
    assigned = _assigned(t["cell_id"].combine_chunks())
    n_rows = t.num_rows
    del t
    gene = codes >= 0
    keep = gene & (qv >= MIN_QV) & np.isfinite(x) & np.isfinite(y)
    return _Block(codes[keep], x[keep], y[keep], qv[keep], assigned[keep], n_rows,
                  int(n_rows - gene.sum()), int(gene.sum() - keep.sum()))


class _Grid:
    """The coarse tiles over the section; a point outside goes to the nearest edge tile."""

    def __init__(self, bounds: Sequence[float]):
        self.x0, self.y0, x1, y1 = (float(b) for b in bounds)
        self.dx = max(x1 - self.x0, 1e-6) / TILES
        self.dy = max(y1 - self.y0, 1e-6) / TILES

    def tile(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        tx = np.clip(((x - self.x0) / self.dx).astype(np.int64), 0, TILES - 1)
        ty = np.clip(((y - self.y0) / self.dy).astype(np.int64), 0, TILES - 1)
        return ty * TILES + tx


# ---------------------------------------------------------------------------
# The build
# ---------------------------------------------------------------------------


def iter_index(source: str | Path, dest: str | Path, bounds: Sequence[float], *, stop: Callable[[], bool] | None = None,
               matrix: str | Path | None = None, source_name: str = UPLOAD) -> Iterator[dict[str, Any]]:
    """Build the store ``dest`` from ``source``, one row group at a time. Yields
    ``{"type": "progress", "phase", "done", "total", "message"}`` per row group, then
    ``{"type": "result", "summary"}``. ``bounds``: x0, y0, x1, y1 of the section, for the tiles.
    A stop between row groups raises TranscriptsCancelled and leaves no part file."""
    import pyarrow.parquet as pq

    t0 = time.monotonic()
    plan = read_plan(source)
    total = plan["rows"] * 2
    pf = pq.ParquetFile(source)
    columns = [*REQUIRED, *(["is_gene"] if plan["is_gene"] else [])]
    grid = _Grid(bounds)
    part = Path(dest).with_name(Path(dest).name + ".part")
    part.parent.mkdir(parents=True, exist_ok=True)

    def check_stop():
        if stop is not None and stop():
            raise TranscriptsCancelled

    try:
        # Pass 1: how many points of each gene in each tile.
        seen: dict[str, int] = {}
        counts = np.zeros((0, TILES**2), dtype=np.uint32)
        assigned = np.zeros(0, dtype=np.int64)
        n_rows = n_control = n_low_quality = 0
        for i in range(pf.num_row_groups):
            check_stop()
            block = _read_block(pf, i, columns, lambda n: seen.setdefault(n, len(seen)))
            n_rows += block.n_rows
            n_control += block.n_control
            n_low_quality += block.n_low_quality
            if len(seen) > counts.shape[0]:
                grown = len(seen) - counts.shape[0]
                counts = np.vstack((counts, np.zeros((grown, TILES**2), dtype=np.uint32)))
                assigned = np.concatenate((assigned, np.zeros(grown, dtype=np.int64)))
            keys, n = np.unique(block.codes.astype(np.int64) * TILES**2 + grid.tile(block.x, block.y), return_counts=True)
            counts[keys // TILES**2, keys % TILES**2] += n.astype(np.uint32)
            assigned[:len(seen)] += np.bincount(block.codes[block.assigned], minlength=len(seen))
            yield {"type": "progress", "phase": "count", "done": n_rows, "total": total,
                   "message": f"Counting: {n_rows:,} of {plan['rows']:,} transcripts read"}
        if not counts.any():
            raise TranscriptsError("That file holds no gene transcripts with q >= 20: check it is the run's "
                                   "transcripts.parquet.")

        # In name order, and each gene's places.
        names = sorted(seen)
        old = np.array([seen[n] for n in names], dtype=np.int64)
        counts = counts[old]
        assigned = assigned[old]
        final = {n: g for g, n in enumerate(names)}
        tile_offsets = np.zeros((len(names), TILES**2 + 1), dtype=np.uint32)
        tile_offsets[:, 1:] = np.cumsum(counts, axis=1, dtype=np.uint64).astype(np.uint32)
        per_gene = counts.sum(axis=1, dtype=np.int64)
        del counts
        gene_offsets = np.concatenate(([0], np.cumsum(per_gene)))
        n_kept = int(gene_offsets[-1])
        places = _create_store(part, n_kept)

        # Pass 2: each row group's points into their places.
        cursor = tile_offsets[:, :-1].copy()
        low, high = np.array([np.inf, np.inf]), np.array([-np.inf, -np.inf])
        done = 0
        for i in range(pf.num_row_groups):
            check_stop()
            block = _read_block(pf, i, columns, final.__getitem__)
            key = block.codes.astype(np.int64) * TILES**2 + grid.tile(block.x, block.y)
            order = np.argsort(key, kind="stable")
            keys, starts, n = np.unique(key[order], return_index=True, return_counts=True)
            genes, tiles = keys // TILES**2, keys % TILES**2
            base = gene_offsets[genes] + cursor[genes, tiles]
            rows = np.repeat(base, n) + (np.arange(len(order), dtype=np.int64) - np.repeat(starts, n))
            cursor[genes, tiles] += n.astype(np.uint32)
            _write_places(part, places, n_kept, rows, block.x[order], block.y[order], block.qv[order])
            if block.x.size:
                low = np.minimum(low, [block.x.min(), block.y.min()])
                high = np.maximum(high, [block.x.max(), block.y.max()])
            done += block.n_rows
            yield {"type": "progress", "phase": "write", "done": n_rows + done, "total": total,
                   "message": f"Writing the index: {done:,} of {plan['rows']:,} transcripts"}
        del cursor

        check = check_against_matrix(names, assigned, matrix)
        summary = {
            "source": source_name, "n_rows": int(n_rows), "n_kept": n_kept, "n_low_quality": int(n_low_quality),
            "n_control": int(n_control), "min_qv": MIN_QV, "n_genes": len(names),
            "n_assigned": int(assigned.sum()), "bounds": [float(low[0]), float(low[1]), float(high[0]), float(high[1])],
            "check": check, "seconds": round(time.monotonic() - t0, 1),
        }
        _finish_store(part, names, gene_offsets, tile_offsets, assigned, bounds, summary)
        part.replace(dest)
        yield {"type": "result", "summary": summary}
    finally:
        part.unlink(missing_ok=True)


def _create_store(part: Path, n: int) -> dict[str, tuple[int, str]]:
    """The file with its point arrays allocated on disk; (byte offset, type) of each, to write
    through a map. Writing the last value makes HDF5 place the whole array."""
    places = {}
    with h5py.File(part, "w") as f:
        for name, dtype in (("x", "<f4"), ("y", "<f4"), ("qv", "u1")):
            ds = f.create_dataset(name, shape=(n,), dtype=dtype)
            ds[n - 1] = 0
            offset = ds.id.get_offset()
            if offset is None:
                raise TranscriptsError("The transcript store could not be allocated on disk: free some space and "
                                       "build it again.", 507)
            places[name] = (int(offset), dtype)
    return places


def _write_places(part: Path, places: dict[str, tuple[int, str]], n: int, rows: np.ndarray, x: np.ndarray, y: np.ndarray,
                  qv: np.ndarray) -> None:
    """One row group's points into their places; the maps closed again, so the pages they touched
    stay file cache (the kernel's to drop) and not this process's memory."""
    for name, values in (("x", x), ("y", y), ("qv", np.minimum(qv, 255).astype(np.uint8))):
        offset, dtype = places[name]
        mapped = np.memmap(part, dtype=dtype, mode="r+", offset=offset, shape=(n,))
        mapped[rows] = values
        del mapped


def _finish_store(part: Path, names: list[str], gene_offsets: np.ndarray, tile_offsets: np.ndarray, assigned: np.ndarray,
                  bounds: Sequence[float], summary: dict) -> None:
    with h5py.File(part, "r+") as f:
        f.create_dataset("genes", data=np.asarray(names, dtype=object), dtype=_STR)
        f.create_dataset("gene_offsets", data=gene_offsets, dtype=np.int64)
        f.create_dataset("tile_offsets", data=tile_offsets, chunks=(1, TILES**2 + 1), compression="gzip", compression_opts=1)
        f.create_dataset("assigned", data=assigned, dtype=np.int64)
        f.attrs["bounds"] = np.asarray(bounds, dtype=np.float64)
        f.attrs["summary"] = json.dumps(summary)


# ---------------------------------------------------------------------------
# The check against the matrix
# ---------------------------------------------------------------------------


def find_matrix(data_dir: str | Path, dataset_id: str) -> Path | None:
    """The run's cell_feature_matrix.h5 among the dataset's originals."""
    up = Path(data_dir) / "uploads" / dataset_id
    return next(iter(sorted(up.rglob(_MATRIX_NAME))), None) if up.is_dir() else None


def matrix_gene_counts(path: str | Path) -> dict[str, int]:
    """Each gene's total in a 10x matrix (features as rows), a block of values at a time."""
    with h5py.File(path, "r") as f:
        m = f["matrix"]
        names = [_text(n) for n in m["features"]["name"][:]]
        kinds = [_text(k) for k in m["features"]["feature_type"][:]]
        totals = np.zeros(len(names), dtype=np.float64)
        n_values = m["indices"].shape[0]
        for start in range(0, n_values, _MATRIX_BLOCK):
            stop = min(start + _MATRIX_BLOCK, n_values)
            totals += np.bincount(m["indices"][start:stop], weights=m["data"][start:stop], minlength=len(names))
    return {n: int(t) for n, k, t in zip(names, kinds, totals) if k == "Gene Expression"}


def check_against_matrix(names: Sequence[str], assigned: np.ndarray, matrix: str | Path | None) -> dict[str, Any]:
    """Per gene, points with q >= 20 and a cell id against the gene's counts in the matrix: r,
    the median and largest absolute difference, and genes on one side only."""
    if matrix is None:
        return {"available": False, "note": f"No {_MATRIX_NAME} among this dataset's originals to check against."}
    try:
        counts = matrix_gene_counts(matrix)
    except (OSError, KeyError, ValueError) as e:
        logger.warning("Transcript check could not read %s: %s", matrix, e)
        return {"available": False, "note": f"{_MATRIX_NAME} could not be read to check against ({e})."}
    points = dict(zip(names, map(int, assigned)))
    shared = sorted(set(points) & set(counts))
    if not shared:
        return {"available": False, "note": f"No gene of {_MATRIX_NAME} is among the transcripts: "
                                            "is this the same run?", "n_genes": 0, "r": None}
    a = np.array([points[g] for g in shared], dtype=np.float64)
    b = np.array([counts[g] for g in shared], dtype=np.float64)
    diff = np.abs(a - b)
    r = float(np.corrcoef(a, b)[0, 1]) if len(shared) > 1 and a.std() > 0 and b.std() > 0 else None
    worst = int(diff.argmax())
    return {"available": True, "n_genes": len(shared), "r": r, "median_abs_diff": float(np.median(diff)),
            "max_abs_diff": float(diff[worst]), "max_diff_gene": shared[worst],
            "n_equal": int((diff == 0).sum()), "total_points": int(a.sum()), "total_counts": int(b.sum()),
            "only_in_transcripts": sorted(set(points) - set(counts))[:20],
            "only_in_matrix": sorted(set(counts) - set(points))[:20]}


# ---------------------------------------------------------------------------
# Reading the store
# ---------------------------------------------------------------------------


def read_summary(path: str | Path) -> dict[str, Any]:
    with h5py.File(path, "r") as f:
        return json.loads(f.attrs["summary"])


def status(data_dir: str | Path, dataset_id: str, n_genes: int = 500) -> dict[str, Any]:
    """What is attached, for the card: the store's summary and genes, or the upload waiting."""
    path = store_path(data_dir, dataset_id)
    out: dict[str, Any] = {"attached": False, "uploaded": False, "max_genes": MAX_GENES, "default_limit": DEFAULT_LIMIT,
                           "min_qv": MIN_QV}
    if path.is_file():
        with h5py.File(path, "r") as f:
            names = f["genes"].asstr()[:]
            offsets = f["gene_offsets"][:]
            out.update(attached=True, summary=json.loads(f.attrs["summary"]),
                       genes=[{"name": n, "n": int(c)} for n, c in zip(names, np.diff(offsets))])
        out["disk_bytes"] = path.stat().st_size
    upload = upload_path(data_dir, dataset_id)
    if upload.is_file():
        try:
            plan = read_plan(upload)
        except TranscriptsError:
            return out
        out.update(uploaded=True, upload_bytes=upload.stat().st_size,
                   estimate=estimate(plan["rows"], plan["largest_row_group"], n_genes))
    return out


@dataclass
class View:
    gene: np.ndarray  # uint8: index into the genes asked for
    x: np.ndarray
    y: np.ndarray
    total: int  # points in the view, before thinning
    per_gene: list[int]
    seconds: float


def _spans(row: np.ndarray, x_tiles: tuple[int, int], y_tiles: tuple[int, int]) -> list[tuple[int, int]]:
    """Ranges of a gene's points (from its start) holding the tiles of a view; close ranges joined."""
    out: list[tuple[int, int]] = []
    for ty in range(y_tiles[0], y_tiles[1] + 1):
        a = int(row[ty * TILES + x_tiles[0]])
        b = int(row[ty * TILES + x_tiles[1] + 1])
        if b <= a:
            continue
        if out and a - out[-1][1] < _MERGE_GAP:
            out[-1] = (out[-1][0], b)
        else:
            out.append((a, b))
    return out


def query(path: str | Path, genes: Sequence[str], box: Sequence[float] | None = None, limit: int = DEFAULT_LIMIT) -> View:
    """The points of up to ``MAX_GENES`` genes in ``box`` (x0, y0, x1, y1; None = everywhere). Over
    ``limit`` every gene is thinned by the same share, evenly along its spatial order, so the
    colours keep their relative density."""
    t0 = time.monotonic()
    if not genes or len(genes) > MAX_GENES:
        raise TranscriptsError(f"Ask for 1 to {MAX_GENES} genes at a time.")
    with h5py.File(path, "r") as f:
        names = f["genes"].asstr()[:]
        at = {n: i for i, n in enumerate(names)}
        unknown = [g for g in genes if g not in at]
        if unknown:
            raise TranscriptsError(f"{', '.join(unknown[:3])} {'is' if len(unknown) == 1 else 'are'} not among this "
                                   "run's transcripts: pick genes of its panel.", 404)
        offsets = f["gene_offsets"]
        grid = _Grid(f.attrs["bounds"])
        if box is None:
            xt, yt = (0, TILES - 1), (0, TILES - 1)
        else:
            x0, y0, x1, y1 = box
            # Points outside the grid sit in its edge tiles, so the clamped range still holds them.
            xt = (int(np.clip((x0 - grid.x0) // grid.dx, 0, TILES - 1)), int(np.clip((x1 - grid.x0) // grid.dx, 0, TILES - 1)))
            yt = (int(np.clip((y0 - grid.y0) // grid.dy, 0, TILES - 1)), int(np.clip((y1 - grid.y0) // grid.dy, 0, TILES - 1)))
        xs, ys = [], []
        for g in genes:
            i = at[g]
            start = int(offsets[i])
            spans = _spans(f["tile_offsets"][i], xt, yt)
            px = [f["x"][start + a:start + b] for a, b in spans]
            py = [f["y"][start + a:start + b] for a, b in spans]
            gx = np.concatenate(px) if px else np.zeros(0, np.float32)
            gy = np.concatenate(py) if py else np.zeros(0, np.float32)
            if box is not None:
                inside = (gx >= x0) & (gx <= x1) & (gy >= y0) & (gy <= y1)
                gx, gy = gx[inside], gy[inside]
            xs.append(gx)
            ys.append(gy)
    per_gene = [len(a) for a in xs]
    total = sum(per_gene)
    share = min(1.0, limit / total) if total else 1.0
    if share < 1.0:
        for j, n in enumerate(per_gene):
            k = max(1, int(n * share)) if n else 0
            picks = np.floor(np.arange(k) * (n / max(k, 1))).astype(np.int64)
            xs[j], ys[j] = xs[j][picks], ys[j][picks]
    gene = np.repeat(np.arange(len(genes), dtype=np.uint8), [len(a) for a in xs])
    empty = np.zeros(0, np.float32)
    return View(gene, np.concatenate(xs) if xs else empty, np.concatenate(ys) if ys else empty, total, per_gene,
                time.monotonic() - t0)


def encode(view: View) -> bytes:
    """The wire format of ``GET .../spatial/transcripts``, little-endian: ``int32 n, uint8 gene[n]``,
    zero padding to a multiple of 4, ``float32 x[n], float32 y[n]``."""
    n = len(view.gene)
    pad = (4 - (4 + n) % 4) % 4
    return (np.array([n], dtype="<i4").tobytes() + view.gene.tobytes() + b"\0" * pad
            + view.x.astype("<f4").tobytes() + view.y.astype("<f4").tobytes())


# ---------------------------------------------------------------------------
# The co-pilot's fact
# ---------------------------------------------------------------------------

TAG = "result:transcripts"


def grounding_lines(data_dir: str | Path, dataset_id: str, outlines: Sequence[str]) -> tuple[list[str], str | None]:
    """The outlines and transcripts attached to a dataset, for the co-pilot; nothing without them."""
    has = has_store(data_dir, dataset_id)
    if not outlines and not has:
        return [], None
    shape = f"{' and '.join(outlines)} outlines" if outlines else "no outlines"
    lines = [f"\n## Imaging data beside this dataset [{TAG}]",
             f"- Attached: {shape}, {'transcripts' if has else 'no transcripts'}; drawn on the tissue plot, "
             f"never in an h5ad export [{TAG}]."]
    detail = f"Imaging data attached: {shape}"
    if has:
        s = read_summary(store_path(data_dir, dataset_id))
        detail += f", {s['n_kept']:,} transcripts"
        lines.append(f"- Transcripts: {s['n_kept']:,} kept of {s['n_rows']:,} read ({s['n_low_quality']:,} under q "
                     f"{s['min_qv']:.0f}, {s['n_control']:,} control features left out), {s['n_genes']:,} genes [{TAG}].")
        check = s.get("check") or {}
        if check.get("available") and check.get("r") is not None:
            lines.append(f"- Per gene, transcripts with q >= {s['min_qv']:.0f} and a cell id against the count matrix: "
                         f"r = {check['r']:.4f}, median difference {check['median_abs_diff']:.0f}, largest "
                         f"{check['max_abs_diff']:.0f} ({check['max_diff_gene']}) [{TAG}].")
    return lines, detail
