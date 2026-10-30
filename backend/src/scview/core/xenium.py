"""Xenium Onboard Analysis (XOA) output folders: which files count, and reading them.

A bundle is ``experiment.xenium`` + ``cell_feature_matrix.h5`` (or the folder
``cell_feature_matrix/``, MEX) + ``cells.parquet`` (or ``cells.csv.gz``), with an optional
image: ``morphology_focus.ome.tif``, ``morphology_focus/morphology_focus_0000.ome.tif``
(XOA 2+, channel 0 = DAPI), else ``morphology_mip.ome.tif``. The transcripts, boundaries,
Explorer stores and XOA's own analysis are never read: the 10 GB folder sends about 30 MB
plus the image (``paths.py`` and ``lib/ingestUpload.ts`` leave them out).

The dataset (the format history follows spatialdata-io's Xenium reader, BSD-3-Clause,
Copyright (c) 2022, scverse; nothing is copied from it):

* X: the ``Gene Expression`` features only (``protein.split_features`` at load). Blank,
  negative-control, deprecated, unassigned and genomic-control features leave X; their per
  cell totals are columns of ``cells.parquet`` already.
* Cells are matched by id, never by row: ``cell_id`` is an int up to XOA 1.2, then a string
  (bytes in some parquet files). A mismatch either way is an error with the count.
* obs: every numeric column of the cells table, ``segmentation_method`` when present, and
  ``pct_control`` = 100 x (the control and other non-gene counts) / ``total_counts``.
* obsm["spatial"]: x_centroid, y_centroid in MICRONS, so the images' scale factors are image
  pixels per micron: ``tissue_{hires,lowres}_scalef`` = 1 / (pixel_size x 2^level).
* uns["scview_spatial"]: technology xenium, unit cell, targeted_panel, panel_name,
  n_panel_genes, pixel_size, organism; uns["spatial"][library]["metadata"]: source xenium,
  analysis_sw_version, files, metrics (``metrics_summary.csv``, which is not Cell Ranger's
  metrics file: provenance only, ``seq_metrics`` never sees it).

The images are JPEG 2000 tiles that tifffile cannot decode without imagecodecs (not in the
image, not to be added): each tile's bytes are read with ``TiffFile.filehandle`` and decoded
with Pillow. A pyramid's level 4 (1611 x 2213) took 0.5 s on the Xenium breast Rep1 image.
"""

from __future__ import annotations

import csv
import gzip
import io
import json
import logging
import zipfile
import zlib
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath

import numpy as np
import pandas as pd

from scview.core import spatial

logger = logging.getLogger(__name__)

TECHNOLOGY = "xenium"
EXPERIMENT = "experiment.xenium"
GENE_TYPE = "Gene Expression"

# The parts of a bundle (``BundleFile.spatial_part``).
PART_EXPERIMENT, PART_CELLS, PART_IMAGE, PART_METRICS = "experiment", "cells", "image", "metrics"

H5_NAME = "cell_feature_matrix.h5"
MEX_DIR = "cell_feature_matrix"
CELLS_NAMES = ("cells.parquet", "cells.csv.gz")  # in order of preference
METRICS_NAME = "metrics_summary.csv"
FOCUS_DIR = "morphology_focus"
# Paths inside a bundle, in order of preference.
IMAGE_PATHS = ("morphology_focus.ome.tif", "morphology_focus/morphology_focus_0000.ome.tif",
               "morphology_mip.ome.tif")
MEX_FILES = {"matrix.mtx": "matrix", "barcodes.tsv": "barcodes", "features.tsv": "features"}

EXTRA = "not used by scView's Xenium import"
HIRES_LONG, LOWRES_LONG = 2500, 650  # the first pyramid level with its long side under these
MAX_DECIMATED_PIXELS = 200_000_000  # no pyramid: decoded whole and thinned up to this size
PCT_LOW, PCT_HIGH = 0.5, 99.8  # the stretch to uint8, of the non-zero pixels
DEFAULT_PIXEL_SIZE = 0.2125  # microns per pixel at level 0, every Xenium instrument so far
# The per-cell counts that are not genes (every XOA version's columns).
CONTROL_COLUMNS = ("control_probe_counts", "control_codeword_counts", "genomic_control_counts",
                   "unassigned_codeword_counts", "deprecated_codeword_counts")
CELL_TABLE_COLUMNS = 10  # obs columns a cell table adds, for the preflight (XOA 1.0.1: 9 + pct_control)
_TEXT_COLUMNS = ("segmentation_method",)
_CELL_ID = "cell_id"
_ORGANISMS = {"human": "human", "homo sapiens": "human", "mouse": "mouse", "mus musculus": "mouse"}
_PARQUET_MAGIC = b"PAR1"
_ZIP_MAGIC = b"PK\x03\x04"
_MAX_METRICS = 200
# TIFF compression tags of the JPEG 2000 and JPEG codecs (Pillow decodes a tile's bytes).
_PILLOW_CODECS = {7, 33003, 33004, 33005, 34712, 34713}


class XeniumError(spatial.SpatialError):
    """A Xenium bundle that cannot be read; the message names the file and what to do."""


# ---------------------------------------------------------------------------
# Detection and grouping (stages [1] and [2])
# ---------------------------------------------------------------------------


def sniff(path: Path, head: bytes) -> dict | None:
    """``{"part", "reason"}`` for a file of a Xenium bundle, by its name and first bytes;
    ``{"zip": True, ...}`` for an outs archive; None for anything else."""
    name = path.name.lower()
    if head.startswith(_ZIP_MAGIC):
        return {"zip": True, "reason": "a zip archive of a Xenium output folder"} if _zip_is_xenium(path) else None
    if name == EXPERIMENT:
        return {"part": PART_EXPERIMENT, "reason": "Xenium experiment file (experiment.xenium)"}
    if name == CELLS_NAMES[0] and head.startswith(_PARQUET_MAGIC):
        return {"part": PART_CELLS, "reason": "Xenium cell table (cells.parquet)"}
    if name == CELLS_NAMES[1] and _cells_csv_header(path):
        return {"part": PART_CELLS, "reason": "Xenium cell table (cells.csv.gz)"}
    if any(name == PurePosixPath(p).name for p in IMAGE_PATHS):
        return {"part": PART_IMAGE, "reason": "Xenium morphology image (OME-TIFF)"}
    return None


def _cells_csv_header(path: Path) -> bool:
    try:
        with gzip.open(path, "rt", encoding="utf-8", errors="replace") as fh:
            header = fh.readline().lower()
    except (OSError, EOFError, zlib.error):
        return False
    return _CELL_ID in header and "x_centroid" in header


def _zip_is_xenium(path: Path) -> bool:
    """Whether a zip holds a Xenium run, from its member list (the file's end only)."""
    try:
        with zipfile.ZipFile(path) as zf:
            return any(PurePosixPath(n).name.lower() == EXPERIMENT for n in zf.namelist())
    except (zipfile.BadZipFile, OSError):
        return "xenium" in path.name.lower()


@dataclass
class Group:
    """One bundle's files by their paths in the upload. ``mex``: role -> path of the
    ``cell_feature_matrix/`` files, used when there is no h5; ``skipped``: (path, reason)."""

    root: str
    experiment: str | None = None
    h5: str | None = None
    mex: dict[str, str] = field(default_factory=dict)
    cells: str | None = None
    image: str | None = None
    metrics: str | None = None
    skipped: list[tuple[str, str]] = field(default_factory=list)

    def missing(self) -> list[str]:
        """The required files this bundle lacks, by the names to tell the user."""
        out = []
        if self.experiment is None:
            out.append(EXPERIMENT)
        if self.h5 is None and len(self.mex) < len(MEX_FILES):
            out.append(f"{H5_NAME} (or the {MEX_DIR}/ folder)")
        if self.cells is None:
            out.append(f"{CELLS_NAMES[0]} (or {CELLS_NAMES[1]})")
        return out

    def taken(self) -> set[str]:
        files = [self.experiment, self.h5, self.cells, self.image, self.metrics, *self.mex.values()]
        return {f for f in files if f} | {p for p, _ in self.skipped}


def _dir(rel: str) -> str:
    parent = PurePosixPath(rel).parent.as_posix()
    return "" if parent == "." else parent


def _inside(rel: str, root: str) -> str | None:
    """``rel`` relative to the folder ``root``, or None when it is not under it."""
    if not root:
        return rel
    return rel[len(root) + 1:] if rel.startswith(root + "/") else None


def _roots(rels: list[str], parts: Mapping[str, str]) -> list[str]:
    """The bundles' folders: where an ``experiment.xenium`` is, and where a file only a
    Xenium run has is without one (an incomplete bundle, told what it lacks)."""
    roots = {_dir(r) for r in rels if PurePosixPath(r).name.lower() == EXPERIMENT}
    for r, part in parts.items():
        if part == PART_EXPERIMENT:
            continue
        d = _dir(r)
        if part == PART_IMAGE and PurePosixPath(d).name.lower() == FOCUS_DIR:
            d = _dir(d)
        if not any(_inside(r, root) is not None and root for root in roots):
            roots.add(d)
    return sorted(roots, key=len, reverse=True)


def _place(group: Group, inner: str, rel: str) -> bool:
    """File ``rel`` (``inner`` inside its bundle) into ``group``; False when it is no part
    of a bundle."""
    low = inner.lower()
    if low == EXPERIMENT:
        group.experiment = rel
    elif low == H5_NAME:
        group.h5 = rel
    elif low in CELLS_NAMES:
        other = group.cells
        if other is None or low == CELLS_NAMES[0]:
            group.cells = rel
        if other is not None:
            kept, dropped = (rel, other) if low == CELLS_NAMES[0] else (other, rel)
            group.skipped.append((dropped, f"the same cell table as {PurePosixPath(kept).name}, which is read"))
    elif low == METRICS_NAME:
        group.metrics = rel
    elif low in IMAGE_PATHS:
        if group.image is None or IMAGE_PATHS.index(low) < _image_rank(group):
            if group.image is not None:
                group.skipped.append((group.image, f"another morphology image is used ({PurePosixPath(rel).name})"))
            group.image = rel
        else:
            group.skipped.append((rel, f"another morphology image is used ({PurePosixPath(group.image).name})"))
    elif low.startswith(MEX_DIR + "/"):
        name = low[len(MEX_DIR) + 1:].removesuffix(".gz")
        if name not in MEX_FILES:
            return False
        group.mex[MEX_FILES[name]] = rel
    else:
        return False
    return True


def _image_rank(group: Group) -> int:
    low = _inside(group.image, group.root).lower()
    return IMAGE_PATHS.index(low)


def groups(rels: list[str], parts: Mapping[str, str]) -> list[Group]:
    """The Xenium bundles among the upload's paths, one per folder with an
    ``experiment.xenium`` (a plain upload of the files is one, at the top). ``parts``: path ->
    part for the files ``sniff`` recognised. Every file under a bundle's own folder is placed
    or skipped with a reason; at the top of a plain upload only a bundle's own names count, so
    another sample uploaded beside them stays a sample."""
    roots = _roots(rels, parts)
    if not roots:
        return []
    out = {root: Group(root) for root in roots}
    for rel in rels:
        for root in roots:  # deepest first
            inner = _inside(rel, root)
            if inner is None:
                continue
            if not _place(out[root], inner, rel) and root:
                out[root].skipped.append((rel, EXTRA))
            break
    for g in out.values():
        if g.h5 is not None and g.mex:
            g.skipped += [(p, f"the same matrix as {H5_NAME}, which is read") for p in g.mex.values()]
            g.mex = {}
    return [out[root] for root in sorted(out)]


def skip_reasons(rels: list[str]) -> dict[str, str]:
    """Files of a bundle's folder (one with an ``experiment.xenium``) that are never read, so
    nothing need open them: XOA's own analysis, auxiliary outputs and the other stains. The
    frontend leaves out the same (``lib/ingestUpload.ts``)."""
    roots = [_dir(r) for r in rels if PurePosixPath(r).name.lower() == EXPERIMENT]
    out: dict[str, str] = {}
    for rel in rels:
        for root in roots:
            inner = _inside(rel, root)
            if not inner or not root:
                continue
            low = inner.lower()
            if low.split("/", 1)[0] in ("analysis", "aux_outputs") or (
                    low.endswith(".ome.tif") and low not in IMAGE_PATHS):
                out[rel] = EXTRA
            break
    return out


# ---------------------------------------------------------------------------
# Reading the files
# ---------------------------------------------------------------------------


def read_experiment(path: str | Path, shown: str = EXPERIMENT) -> dict:
    """experiment.xenium as a dict; XeniumError naming the file."""
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, ValueError) as e:
        raise XeniumError(f"{shown} could not be read as JSON ({e}): upload the file Xenium wrote.") from e
    if not isinstance(data, dict):
        raise XeniumError(f"{shown} is not a Xenium experiment file: upload the one Xenium wrote.")
    return data


def region_name(path: str | Path) -> str:
    """A bundle's region name, for its library; "" when there is none."""
    try:
        return str(read_experiment(path).get("region_name") or "").strip()
    except XeniumError:
        return ""


def _cell_ids(values) -> np.ndarray:
    """Cell ids as text: an int, a string, or the bytes some parquet files hold."""
    return np.array([v.decode("utf-8", "replace") if isinstance(v, (bytes, bytearray)) else str(v)
                     for v in values], dtype=object)


def read_cells(path: str | Path, shown: str | None = None) -> pd.DataFrame:
    """The cell table indexed by cell id as text. XeniumError for an unreadable table, no
    ``cell_id``, no centroids or a repeated id."""
    path = Path(path)
    shown = shown or path.name
    try:
        with open(path, "rb") as fh:
            is_parquet = fh.read(4) == _PARQUET_MAGIC
        df = pd.read_parquet(path) if is_parquet else pd.read_csv(path, dtype={_CELL_ID: str})
    except Exception as e:  # noqa: BLE001 - pyarrow's and pandas' own errors
        raise XeniumError(f"{shown} could not be read as a cell table ({e}): upload the file Xenium wrote.") from e
    missing = [c for c in (_CELL_ID, "x_centroid", "y_centroid") if c not in df.columns]
    if missing:
        raise XeniumError(f"{shown} lacks the column(s) {', '.join(missing)} of a Xenium cell table: "
                          "upload the file Xenium wrote.")
    ids = _cell_ids(df[_CELL_ID].to_numpy())
    if len(set(ids)) < len(ids):
        raise XeniumError(f"{shown} lists a cell id more than once: upload the file Xenium wrote.")
    df = df.drop(columns=[_CELL_ID])
    df.index = pd.Index(ids, name=None)
    return df


def _aligned(cells: pd.DataFrame, names: pd.Index, cells_name: str, matrix_name: str) -> pd.DataFrame:
    """The cell table in the matrix's order (the id says which row is which cell, the row
    number never does). Raises XeniumError with the count when the two differ."""
    not_in_table = int((~names.isin(cells.index)).sum())
    if not_in_table:
        example = ", ".join(map(str, names[~names.isin(cells.index)][:3]))
        raise XeniumError(f"{not_in_table:,} of the {len(names):,} cells in {matrix_name} are not in "
                          f"{cells_name} (e.g. {example}): upload the files of one Xenium run together.")
    not_in_matrix = len(cells) - len(names)
    if not_in_matrix:
        raise XeniumError(f"{not_in_matrix:,} of the {len(cells):,} cells in {cells_name} are not in "
                          f"{matrix_name}: upload the files of one Xenium run together.")
    return cells.loc[names]


def pct_control(cells: pd.DataFrame) -> np.ndarray:
    """100 x the counts that are not genes / all counts, per cell (0 for a cell with none).
    Control counts: ``total_counts`` - ``transcript_counts`` when both are there (they take in
    the blank codewords, which XOA 1.0 has no column for), else the control columns added up."""
    n = len(cells)
    transcripts = cells["transcript_counts"].to_numpy(dtype=np.float64) if "transcript_counts" in cells else None
    columns = [c for c in CONTROL_COLUMNS if c in cells.columns]
    summed = cells[columns].to_numpy(dtype=np.float64).sum(axis=1) if columns else np.zeros(n)
    if "total_counts" in cells.columns:
        total = cells["total_counts"].to_numpy(dtype=np.float64)
        control = total - transcripts if transcripts is not None else summed
    else:
        total = (transcripts + summed) if transcripts is not None else np.full(n, np.nan)
        control = summed
    return np.divide(100.0 * control, total, out=np.zeros(n), where=np.nan_to_num(total) > 0)


def read_metrics(path: str | Path) -> dict:
    """The first row of XOA's metrics_summary.csv, non-empty values only; {} when it cannot
    be read (it is provenance, never a reason to refuse a dataset)."""
    try:
        with open(path, newline="", encoding="utf-8", errors="replace") as fh:
            rows = list(csv.reader(fh))
    except OSError:
        return {}
    if len(rows) < 2:
        return {}
    out: dict = {}
    for key, value in list(zip(rows[0], rows[1]))[:_MAX_METRICS]:
        value = value.strip()
        if not value:
            continue
        try:
            out[key.strip()] = float(value)
        except ValueError:
            out[key.strip()] = value
    return out


# ---------------------------------------------------------------------------
# The morphology image
# ---------------------------------------------------------------------------


def _levels(tf) -> list:
    """The first page of each pyramid level, largest first: channel 0 in a multi-channel
    series (DAPI in XOA 2+)."""
    series = tf.series[0]
    pages = [lvl.pages[0] for lvl in series.levels]
    return [p for p in pages if p is not None]


def _size(page) -> tuple[int, int]:
    kf = getattr(page, "keyframe", page)
    return int(kf.imagelength), int(kf.imagewidth)


def _decode_page(tf, page) -> np.ndarray:
    """A page's pixels as a 2-D array: tile by tile through Pillow when it is compressed
    with a codec tifffile cannot decode alone (JPEG 2000), else tifffile's own read."""
    from PIL import Image

    kf = getattr(page, "keyframe", page)
    if int(kf.compression) not in _PILLOW_CODECS:
        a = np.asarray(page.asarray())
        return a if a.ndim == 2 else a.reshape(_size(page) + (-1,)).mean(axis=2)
    h, w = _size(page)
    th = int(kf.tilelength) if kf.is_tiled else int(kf.rowsperstrip)
    tw = int(kf.tilewidth) if kf.is_tiled else w
    out = np.zeros((h, w), dtype=kf.dtype if kf.dtype is not None else np.uint16)
    fh = tf.filehandle
    across = max(1, -(-w // tw))
    for i, (offset, nbytes) in enumerate(zip(page.dataoffsets, page.databytecounts)):
        if not nbytes:
            continue
        fh.seek(offset)
        with Image.open(io.BytesIO(fh.read(nbytes))) as tile:
            a = np.asarray(tile)
        if a.ndim == 3:
            a = a[..., 0]
        r, c = (i // across) * th, (i % across) * tw
        rows, cols = min(a.shape[0], h - r), min(a.shape[1], w - c)
        if rows > 0 and cols > 0:
            out[r:r + rows, c:c + cols] = a[:rows, :cols]
    return out


def _stretch(a: np.ndarray, low: float, high: float) -> np.ndarray:
    """uint8 HxWx3 from grey values between ``low`` and ``high``."""
    scaled = np.clip((a.astype(np.float32) - low) / max(high - low, 1e-6), 0.0, 1.0) * 255.0
    grey = scaled.astype(np.uint8)
    return np.ascontiguousarray(np.repeat(grey[..., None], 3, axis=2))


def _limits(a: np.ndarray) -> tuple[float, float]:
    nz = a[a > 0]
    if nz.size == 0:
        return 0.0, 1.0
    low, high = np.percentile(nz, [PCT_LOW, PCT_HIGH])
    return float(low), float(high)


@dataclass
class Morphology:
    images: dict[str, np.ndarray] = field(default_factory=dict)
    scalefactors: dict[str, float] = field(default_factory=dict)
    levels: dict[str, int] = field(default_factory=dict)  # resolution -> pyramid level


def read_image(path: str | Path, pixel_size: float, shown: str | None = None) -> Morphology:
    """hires and lowres of a morphology OME-TIFF: the first pyramid level with its long side
    under ``HIRES_LONG`` and ``LOWRES_LONG``, stretched to uint8 from the percentiles of the
    hires non-zero pixels, with scale factors so that an image pixel / scalef = microns. A
    file without such levels has its smallest level thinned. XeniumError naming the file."""
    import tifffile

    shown = shown or Path(path).name
    try:
        # _multifile False: an XOA 2+ image's series also names its sibling stains' files.
        with tifffile.TiffFile(str(path), _multifile=False) as tf:
            pages = _levels(tf)
            if not pages:
                raise XeniumError(f"{shown} holds no image.")
            w0 = _size(pages[0])[1]
            scale = [_level_scale(w0, _size(p)[1], i) for i, p in enumerate(pages)]
            hires_at = _first_under(pages, HIRES_LONG)
            lowres_at = _first_under(pages, LOWRES_LONG)
            at = hires_at if hires_at is not None else len(pages) - 1
            if hires_at is None and np.prod(_size(pages[at])) > MAX_DECIMATED_PIXELS:
                raise XeniumError(f"{shown} has no pyramid level small enough to read.")
            hires = _decode_page(tf, pages[at])
            step = max(1, -(-max(hires.shape) // HIRES_LONG))
            hires = hires[::step, ::step]
            low, high = _limits(hires)
            out = Morphology()
            out.images["hires"] = _stretch(hires, low, high)
            out.scalefactors["tissue_hires_scalef"] = 1.0 / (pixel_size * scale[at] * step)
            out.levels["hires"] = at
            if lowres_at is not None and lowres_at != at:
                small = _decode_page(tf, pages[lowres_at])
                out.images["lowres"] = _stretch(small, low, high)
                out.scalefactors["tissue_lowres_scalef"] = 1.0 / (pixel_size * scale[lowres_at])
                out.levels["lowres"] = lowres_at
            else:
                thin = max(1, -(-max(hires.shape) // LOWRES_LONG))
                out.images["lowres"] = _stretch(hires[::thin, ::thin], low, high)
                out.scalefactors["tissue_lowres_scalef"] = 1.0 / (pixel_size * scale[at] * step * thin)
                out.levels["lowres"] = at
            return out
    except XeniumError:
        raise
    except Exception as e:  # noqa: BLE001 - tifffile's and Pillow's own errors
        raise XeniumError(f"{shown} could not be read as a morphology image ({e}).") from e


def _level_scale(w0: int, w: int, level: int) -> float:
    """How many level-0 pixels a pixel of this level spans: 2^level, unless the pyramid does
    not halve (then from its widths)."""
    ratio = w0 / max(w, 1)
    return float(2**level) if abs(ratio / 2**level - 1.0) < 0.02 else ratio


def _first_under(pages: list, limit: int) -> int | None:
    return next((i for i, p in enumerate(pages) if max(_size(p)) <= limit), None)


def image_bytes(path: str | Path) -> int:
    """The memory the decoded hires and lowres take (uint8 RGB), from the file's header
    alone; 0 when it cannot be read. For the preflight."""
    import tifffile

    try:
        with tifffile.TiffFile(str(path), _multifile=False) as tf:
            pages = _levels(tf)
            total = 0
            for limit in (HIRES_LONG, LOWRES_LONG):
                at = _first_under(pages, limit)
                h, w = _size(pages[at if at is not None else -1])
                long_side = max(h, w)
                ratio = min(1.0, limit / long_side) if long_side else 1.0
                total += int(3 * h * w * ratio * ratio)
            return total
    except Exception:  # noqa: BLE001 - the load says what is wrong with it
        return 0


# ---------------------------------------------------------------------------
# The dataset
# ---------------------------------------------------------------------------


def organism_of(experiment: Mapping) -> str:
    """``human``, ``mouse``, else the panel's own word in lower case; "" when not stated."""
    raw = str(experiment.get("panel_organism") or "").strip().lower()
    return _ORGANISMS.get(raw, raw)


def attach(adata, files: Mapping[str, tuple[str, str]], library_id: str, matrix_name: str = H5_NAME) -> dict:
    """Put a bundle's cell table, coordinates, panel and image on the cells of ``adata`` (the
    matrix just read, every feature in it). ``files``: part -> (path, shown name). Returns
    the summary for the ingest history. XeniumError names the file and what to do; only the
    image is optional, and an image that cannot be read leaves a note instead."""
    exp_path, exp_name = files[PART_EXPERIMENT]
    cells_path, cells_name = files[PART_CELLS]
    experiment = read_experiment(exp_path, exp_name)
    cells = _aligned(read_cells(cells_path, cells_name), adata.obs_names.astype(str), cells_name, matrix_name)
    numeric = [c for c in cells.columns if pd.api.types.is_numeric_dtype(cells[c])]
    for col in numeric:
        adata.obs[col] = cells[col].to_numpy()
    for col in _TEXT_COLUMNS:
        if col in cells.columns and col not in numeric:
            adata.obs[col] = pd.Categorical(cells[col].astype(str).to_numpy())
    adata.obs["pct_control"] = pct_control(cells).astype(np.float32)
    adata.obsm[spatial.KEY] = cells[["x_centroid", "y_centroid"]].to_numpy(dtype=np.float64)
    adata.obs[spatial.LIBRARY_COLUMN] = pd.Categorical([library_id] * adata.n_obs, categories=[library_id])

    types = adata.var.get("feature_types")
    is_gene = (types.astype(str) == GENE_TYPE).to_numpy() if types is not None else np.ones(adata.n_vars, bool)
    pixel_size = _positive(experiment.get("pixel_size"), DEFAULT_PIXEL_SIZE)
    meta: dict = {"source": TECHNOLOGY, "files": {p: shown for p, (_, shown) in files.items()}}
    for key in ("analysis_sw_version", "run_name", "region_name", "preservation_method", "panel_tissue_type"):
        if experiment.get(key):
            meta[key] = str(experiment[key])
    if PART_METRICS in files:
        meta["metrics"] = read_metrics(files[PART_METRICS][0])

    images, scalefactors, note = {}, {}, None
    if PART_IMAGE in files:
        img_path, img_name = files[PART_IMAGE]
        try:
            found = read_image(img_path, pixel_size, img_name)
            images, scalefactors = found.images, found.scalefactors
            meta["image_levels"] = found.levels
        except XeniumError as e:
            note = str(e)
            logger.warning("Xenium image left out: %s", e)
    if note:
        meta["image_note"] = note
    adata.uns[spatial.KEY] = {library_id: {"images": images, "scalefactors": scalefactors, "metadata": meta}}
    adata.uns[spatial.INFO_KEY] = {
        "technology": TECHNOLOGY, "unit": spatial.TECHNOLOGIES[TECHNOLOGY], "targeted_panel": True,
        "panel_name": str(experiment.get("panel_name") or ""), "n_panel_genes": int(is_gene.sum()),
        "pixel_size": float(pixel_size), "organism": organism_of(experiment),
    }
    return summary(adata)


def _positive(value, default: float) -> float:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return default
    return v if np.isfinite(v) and v > 0 else default


def summary(adata) -> dict:
    """What the ingest history records about a Xenium dataset (the merge of several too)."""
    info = dict(adata.uns.get(spatial.INFO_KEY) or {})
    libs = adata.uns.get(spatial.KEY) or {}
    counts = adata.obs[spatial.LIBRARY_COLUMN].value_counts() if spatial.LIBRARY_COLUMN in adata.obs else {}
    out = {"source": TECHNOLOGY, **{k: _plain(v) for k, v in info.items()},
           "libraries": {str(lib): int(counts.get(lib, 0)) for lib in libs},
           "images": {str(lib): sorted(libs[lib].get("images", {})) for lib in libs}}
    notes = [libs[lib]["metadata"]["image_note"] for lib in libs if libs[lib].get("metadata", {}).get("image_note")]
    if notes:
        out["image_notes"] = notes
    return out


def _plain(v):
    return v.item() if isinstance(v, np.generic) else v


def after_merge(parts: Mapping, merged) -> None:
    """``spatial.after_merge`` keeps the technology and unit; a merge of Xenium runs also keeps
    what the runs share (panel, organism, pixel size), and n_panel_genes = the merged genes."""
    if not parts or spatial.INFO_KEY not in merged.uns:
        return
    infos = [spatial._recorded(a) for a in parts.values()]
    if not all(i.get("technology") == TECHNOLOGY for i in infos):
        return
    shared = {k: v for k, v in infos[0].items()
              if k not in ("technology", "unit", "n_panel_genes") and all(i.get(k) == v for i in infos)}
    merged.uns[spatial.INFO_KEY] = {**merged.uns[spatial.INFO_KEY], **shared, "targeted_panel": True,
                                    "n_panel_genes": int(merged.n_vars)}


def n_gene_features(h5_path: str | None = None, features_path: str | None = None) -> int | None:
    """How many features of the matrix are genes (the header of a Xenium matrix counts its
    controls too), from the h5's feature types or the features.tsv; None when unreadable."""
    try:
        if h5_path:
            import h5py

            with h5py.File(h5_path, "r") as f:
                kinds = f["matrix"]["features"]["feature_type"][:]
            return int(sum(k == GENE_TYPE.encode() or k == GENE_TYPE for k in kinds))
        if features_path:
            opener = gzip.open if str(features_path).endswith(".gz") else open
            with opener(features_path, "rt", encoding="utf-8", errors="replace") as fh:
                return sum(ln.rstrip("\n").split("\t")[2:3] == [GENE_TYPE] for ln in fh)
    except (OSError, KeyError, EOFError, zlib.error, ValueError) as e:
        logger.debug("Could not count the genes of a Xenium matrix: %s", e)
    return None
