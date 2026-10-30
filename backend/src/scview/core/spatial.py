"""Spatial transcriptomics: Space Ranger import, adoption of spatial .h5ad files, and
what the API serves about them.

Storage follows scanpy / squidpy, so their h5ad files open as they are:

* ``obsm["spatial"]``: (n_obs, 2) float, x, y in full-resolution image pixels (Space
  Ranger: x = pxl_col_in_fullres, y = pxl_row_in_fullres; y grows downwards).
* ``uns["spatial"][library_id]``: ``images`` (``hires`` / ``lowres``, uint8 HxWx3),
  ``scalefactors`` (``tissue_hires_scalef``, ``tissue_lowres_scalef``,
  ``spot_diameter_fullres``...) and ``metadata`` (``source``: spaceranger | h5ad).
  An image pixel (r, c) at a resolution sits at full-res (c / scalef, r / scalef).
* ``obs["library_id"]``: categorical, its categories the keys of ``uns["spatial"]``;
  ``obs["array_row"]`` / ``obs["array_col"]``: the Visium grid, when known.
* ``uns["scview_spatial"]``: ``{"technology", "unit"}`` from ``TECHNOLOGIES``, plus any
  technology's own keys. A Xenium dataset's coordinates are microns, not pixels, so its
  scale factors are image pixels per micron; the rule above still places its images.

A Visium sample is its filtered matrix plus the ``spatial/`` folder (loose files or
``spatial.tar.gz``): the ingest bundler attaches those as side files of the matrix
(``SideKind.spatial``) and ``session._attach_spatial`` reads them here. Visium HD is refused
with a sentence, not read. A Xenium output folder is read by ``core/xenium.py``.
"""

from __future__ import annotations

import gzip
import io
import json
import logging
import re
import tarfile
import zlib
import threading
import weakref
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

KEY = "spatial"  # obsm and uns
ALT_KEY = "X_spatial"  # some tools' obsm name: copied to obsm["spatial"]
LIBRARY_COLUMN = "library_id"
INFO_KEY = "scview_spatial"
DEFAULT_LIBRARY = "section1"
RESOLUTIONS = ("hires", "lowres")
TAG = "result:tissue"  # the co-pilot's fact (claims.py family "tissue")
# technology -> what one observation is. Several cells share a spot, and a bin is a square
# of the capture grid that need not match a cell: both count as spot data.
TECHNOLOGIES = {"visium": "spot", "visium_hd": "bin", "xenium": "cell", "cells": "cell"}
TECHNOLOGY_NAMES = {"visium": "Visium", "visium_hd": "Visium HD", "xenium": "Xenium",
                    "cells": "imaging-based or other (cell coordinates)"}
SPOT_UNITS = ("spot", "bin")

DOUBLET_SKIP = "Visium spots hold several cells, so doublet detection does not apply"
# A spatial dataset of cells with no more genes than this is a panel (CosMx, MERSCOPE, an
# h5ad written by spatialdata-io): the commercial imaging panels go up to about 6,000 genes.
PANEL_MAX_GENES = 6000
HD_MESSAGE = ("Visium HD (binned outputs) is not supported yet. scView reads standard Visium "
              "(Space Ranger's filtered matrix with its spatial folder) and .h5ad files with "
              "obsm['spatial'].")
XENIUM_MESSAGE = ("A zipped Xenium output folder is not read (the transcripts in it alone are "
                  "gigabytes): unzip it and drop the folder instead; only the files scView needs "
                  "are uploaded.")

_POS_COLS = ["barcode", "in_tissue", "array_row", "array_col", "pxl_row_in_fullres",
             "pxl_col_in_fullres"]
_PNG_MAGIC = b"\x89PNG\r\n\x1a\n"
_GZIP_MAGIC = b"\x1f\x8b"
_PARQUET_MAGIC = b"PAR1"
_BARCODE_RE = re.compile(r"^[ACGTN]{6,}([-.]\d+)?$", re.IGNORECASE)
_HD_BARCODE_RE = re.compile(r"^s_\d{3}um_")
_INT_RE = re.compile(r"^-?\d+$")
_MAX_MEMBER_BYTES = 512 * 1024**2  # one file of a spatial archive

# The parts of a Space Ranger spatial folder scView reads.
POSITIONS, SCALEFACTORS, ARCHIVE = "positions", "scalefactors", "archive"
PARTS = (POSITIONS, SCALEFACTORS, "hires", "lowres", ARCHIVE)
_PART_TOKENS = ("tissue_positions_list", "tissue_positions", "scalefactors_json", "scalefactors",
                "tissue_hires_image", "tissue_lowres_image", "spatial")
_PART_EXTS = (".gz", ".tgz", ".tar", ".csv", ".json", ".png", ".parquet")


class SpatialError(ValueError):
    """Spatial files that cannot be read or matched to the cells; the message says why."""


# ---------------------------------------------------------------------------
# Detection (stage [1]): which part of a spatial folder a file is
# ---------------------------------------------------------------------------


def sniff_binary(path: Path, head: bytes) -> dict | None:
    """``{"part", "reason"}`` for a tissue image, a parquet positions file or an archive of
    a spatial folder; None for anything else. ``head``: the file's first bytes."""
    name = path.name.lower()
    if head.startswith(_GZIP_MAGIC):
        try:
            with gzip.open(path, "rb") as fh:
                inner = fh.read(512)
        except (OSError, EOFError, zlib.error):
            return None
    else:
        inner = _read(path, 512)
    if inner.startswith(_PNG_MAGIC):
        for res in RESOLUTIONS:
            if res in name:
                return {"part": res, "reason": f"Space Ranger tissue image ({res})"}
        return None
    if inner.startswith(_PARQUET_MAGIC) and "tissue_positions" in name:
        out = {"part": POSITIONS, "reason": "Space Ranger tissue positions (parquet)"}
        if _parquet_is_hd(path):
            out["hd"] = True
        return out
    if len(inner) >= 262 and inner[257:262] == b"ustar":
        members = _archive_parts(path)
        if members is None:
            return None
        if POSITIONS not in members:
            return {"part": None, "reason": "an archive without Space Ranger tissue positions"}
        return {"part": ARCHIVE, "members": sorted(members),
                "reason": "Space Ranger spatial folder (archive)"}
    return None


def sniff_text(text: str) -> dict | None:
    """``{"part", "reason"}`` for a tissue positions CSV (Space Ranger 1.x without a
    header, 2.x+ with one) or a scalefactors JSON; None for other text."""
    stripped = text.lstrip()
    if stripped.startswith("{"):
        try:
            data = json.loads(stripped)
        except ValueError:
            return None
        if isinstance(data, dict) and any(k in data for k in (
                "tissue_hires_scalef", "tissue_lowres_scalef", "spot_diameter_fullres")):
            return {"part": SCALEFACTORS, "reason": "Space Ranger scale factors (scalefactors_json.json)"}
        return None
    lines = [ln for ln in stripped.splitlines() if ln.strip()][:5]
    if not lines:
        return None
    first = [c.strip().strip('"').lower() for c in lines[0].split(",")]
    if first == _POS_COLS:
        return {"part": POSITIONS, "reason": "Space Ranger tissue positions (tissue_positions.csv)"}
    rows = [[c.strip().strip('"') for c in ln.split(",")] for ln in lines]
    if all(len(r) == 6 and _BARCODE_RE.match(r[0]) and r[1] in ("0", "1")
           and all(_INT_RE.match(c) for c in r[2:]) for r in rows):
        return {"part": POSITIONS,
                "reason": "Space Ranger tissue positions (tissue_positions_list.csv, no header)"}
    return None


def name_key(name: str) -> str:
    """The sample prefix of a spatial file's name, as ``bundling.sample_key`` gives a
    matrix's: ``V1_Human_Lymph_Node_spatial.tar.gz`` and
    ``GSM1_tissue_positions_list.csv.gz`` give ``v1_human_lymph_node`` and ``gsm1``."""
    s = name.lower()
    changed = True
    while changed:
        changed = False
        for ext in _PART_EXTS:
            if s.endswith(ext):
                s = s[: -len(ext)]
                changed = True
    for tok in _PART_TOKENS:
        if s.endswith(tok):
            s = s[: -len(tok)]
            break
    return s.strip(" _-.")


_HD_DIR_RE = re.compile(r"^square_\d{3}um$", re.IGNORECASE)


def unsupported(rels: list[str]) -> str | None:
    """The sentence for an upload of a spatial format scView does not read yet (Visium
    HD binned outputs), from its file paths; None otherwise."""
    for r in rels:
        dirs = [d.lower() for d in PurePosixPath(r).parent.parts]
        if "binned_outputs" in dirs or any(_HD_DIR_RE.match(d) for d in dirs):
            return HD_MESSAGE
    return None


def _read(path: Path, n: int) -> bytes:
    try:
        with open(path, "rb") as fh:
            return fh.read(n)
    except OSError:
        return b""


def _part_of(name: str) -> str | None:
    """Which part a member of a spatial archive is, by its name."""
    low = PurePosixPath(name).name.lower()
    if low.startswith("tissue_positions") and low.endswith((".csv", ".parquet")):
        return POSITIONS
    if low.startswith("scalefactors") and low.endswith(".json"):
        return SCALEFACTORS
    for res in RESOLUTIONS:
        if low == f"tissue_{res}_image.png":
            return res
    return None


def _archive_parts(path: Path) -> dict[str, str] | None:
    """part -> member name of a tar(.gz) archive's spatial files; None when it is no
    readable archive."""
    try:
        with tarfile.open(path, "r:*") as tf:
            out: dict[str, str] = {}
            for m in tf:
                part = _part_of(m.name) if m.isfile() else None
                if part and (part not in out or m.name.endswith("tissue_positions.csv")):
                    out[part] = m.name
            return out
    except (tarfile.TarError, OSError, EOFError, zlib.error):
        return None


def _parquet_is_hd(path: Path) -> bool:
    try:
        import pyarrow.parquet as pq

        batch = next(pq.ParquetFile(path).iter_batches(batch_size=5, columns=["barcode"]), None)
    except Exception:  # noqa: BLE001 - not readable: the load says so
        return False
    if batch is None:
        return False
    return any(_HD_BARCODE_RE.match(str(b)) for b in batch.column(0).to_pylist())


# ---------------------------------------------------------------------------
# Reading a Space Ranger spatial folder
# ---------------------------------------------------------------------------


@dataclass
class SpaceRanger:
    """One sample's spatial folder, read: in-tissue positions by barcode, scale
    factors, images, and where they came from."""

    positions: pd.DataFrame  # index barcode; array_row, array_col, pxl_row_in_fullres, pxl_col_in_fullres
    scalefactors: dict[str, float] = field(default_factory=dict)
    images: dict[str, np.ndarray] = field(default_factory=dict)
    files: dict[str, str] = field(default_factory=dict)  # part -> file name
    layout: str = ""  # which positions file layout


def read_spaceranger(parts: list[tuple[str, str, str]]) -> SpaceRanger:
    """Read a sample's spatial files: ``(part, path, shown name)`` each, loose files or an
    archive (``part`` "archive") of the folder. Raises SpatialError naming the file."""
    data: dict[str, tuple[bytes, str]] = {}
    for part, path, shown in parts:
        if part == ARCHIVE:
            for p, (b, name) in _read_archive(Path(path), shown).items():
                data.setdefault(p, (b, name))
        elif part in PARTS:
            if part in data:
                raise SpatialError(f"Two {part} files ({data[part][1]}, {shown}): keep one.")
            data[part] = (_gunzip(Path(path).read_bytes()), shown)
    if POSITIONS not in data:
        raise SpatialError("No tissue positions file (tissue_positions.csv or "
                           "tissue_positions_list.csv) came with the spatial files.")
    raw, pos_name = data[POSITIONS]
    positions, layout = _read_positions(raw, pos_name)
    sr = SpaceRanger(positions=positions, layout=layout, files={POSITIONS: pos_name})
    if SCALEFACTORS in data:
        raw, name = data[SCALEFACTORS]
        try:
            sf = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, ValueError) as e:
            raise SpatialError(f"{name} is not readable JSON: {e}") from e
        sr.scalefactors = {str(k): float(v) for k, v in sf.items()
                           if isinstance(v, (int, float)) and not isinstance(v, bool)}
        sr.files[SCALEFACTORS] = name
    for res in RESOLUTIONS:
        if res in data:
            raw, name = data[res]
            sr.images[res] = read_png(raw, name)
            sr.files[res] = name
    return sr


def _gunzip(raw: bytes) -> bytes:
    return gzip.decompress(raw) if raw.startswith(_GZIP_MAGIC) else raw


def _read_archive(path: Path, shown: str) -> dict[str, tuple[bytes, str]]:
    """part -> (bytes, "archive/member") of the spatial files in a tar(.gz), read in
    memory (nothing is extracted to disk)."""
    out: dict[str, tuple[bytes, str]] = {}
    try:
        with tarfile.open(path, "r:*") as tf:
            for m in tf:
                part = _part_of(m.name) if m.isfile() else None
                if part is None or (part in out and not m.name.endswith("tissue_positions.csv")):
                    continue
                if m.size > _MAX_MEMBER_BYTES:
                    raise SpatialError(f"{m.name} in {shown} is {m.size / 1024**2:.0f} MB, larger "
                                       "than a Space Ranger spatial file.")
                fh = tf.extractfile(m)
                if fh is not None:
                    out[part] = (_gunzip(fh.read()), f"{shown}/{m.name}")
    except (tarfile.TarError, OSError, EOFError, zlib.error) as e:
        raise SpatialError(f"{shown} could not be read as an archive: {e}") from e
    return out


def _read_positions(raw: bytes, name: str) -> tuple[pd.DataFrame, str]:
    if raw.startswith(_PARQUET_MAGIC):
        try:
            df = pd.read_parquet(io.BytesIO(raw))
        except Exception as e:  # noqa: BLE001 - pyarrow's own errors
            raise SpatialError(f"{name} could not be read as parquet: {e}") from e
        layout = "parquet (Space Ranger 3.x)"
    else:
        text = raw.decode("utf-8", errors="replace")
        first = text.lstrip().split("\n", 1)[0]
        header = first.split(",")[0].strip().strip('"').lower() == "barcode"
        try:
            df = pd.read_csv(io.StringIO(text), header=0 if header else None,
                             names=None if header else _POS_COLS, dtype={"barcode": str})
        except Exception as e:  # noqa: BLE001 - pandas' parser errors
            raise SpatialError(f"{name} could not be read as a CSV table: {e}") from e
        layout = ("tissue_positions.csv with a header (Space Ranger 2.x)" if header
                  else "tissue_positions_list.csv without a header (Space Ranger 1.x)")
    df.columns = [str(c).strip().lower() for c in df.columns]
    missing = [c for c in _POS_COLS if c not in df.columns]
    if missing:
        raise SpatialError(f"{name} lacks the column(s) {', '.join(missing)} of a Space Ranger "
                           "tissue positions file.")
    df["barcode"] = df["barcode"].astype(str)
    if df["barcode"].head(20).map(lambda b: bool(_HD_BARCODE_RE.match(b))).any():
        raise SpatialError(HD_MESSAGE)
    for c in _POS_COLS[1:]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    if df[_POS_COLS[1:]].isna().any().any():
        raise SpatialError(f"{name} has values that are not numbers in its position columns.")
    df = df[df["in_tissue"] == 1]
    if df.empty:
        raise SpatialError(f"{name} marks no spot as in the tissue (in_tissue = 1).")
    if df["barcode"].duplicated().any():
        raise SpatialError(f"{name} lists a barcode more than once.")
    return df.set_index("barcode")[_POS_COLS[2:]], layout


def read_png(raw: bytes, name: str = "image") -> np.ndarray:
    """A PNG as uint8 HxWx3 (RGB; alpha dropped, grey made RGB, 16-bit scaled to 8)."""
    try:
        from PIL import Image

        with Image.open(io.BytesIO(raw)) as img:
            img.load()
            if img.mode in ("I;16", "I;16B", "I;16L", "I"):
                a = np.asarray(img, dtype=np.float64)
                top = float(a.max()) or 1.0
                a = np.clip(a / (65535.0 if top > 255 else 255.0) * 255.0, 0, 255).astype(np.uint8)
                return np.ascontiguousarray(np.repeat(a[..., None], 3, axis=2))
            return np.ascontiguousarray(np.asarray(img.convert("RGB"), dtype=np.uint8))
    except SpatialError:
        raise
    except Exception as e:  # noqa: BLE001 - PIL's errors
        raise SpatialError(f"{name} could not be read as a PNG image: {e}") from e


def png_size(raw: bytes) -> tuple[int, int] | None:
    """(width, height) of an (uncompressed) PNG header."""
    if len(raw) < 24 or not raw.startswith(_PNG_MAGIC):
        return None
    return int.from_bytes(raw[16:20], "big"), int.from_bytes(raw[20:24], "big")


def side_bytes(parts: list[tuple[str, str]]) -> int:
    """The memory the decoded tissue images of ``(part, path)`` side files take (uint8
    RGB), read from their PNG headers (inside an archive too). For the preflight."""
    total = 0
    for part, path in parts:
        try:
            if part in RESOLUTIONS:
                with open(path, "rb") as fh:
                    head = fh.read(65536)
                if head.startswith(_GZIP_MAGIC):
                    with gzip.open(path, "rb") as fh:
                        head = fh.read(32)
                size = png_size(head)
                total += size[0] * size[1] * 3 if size else 0
            elif part == ARCHIVE:
                with tarfile.open(path, "r:*") as tf:
                    for m in tf:
                        if m.isfile() and _part_of(m.name) in RESOLUTIONS:
                            fh = tf.extractfile(m)
                            size = png_size(fh.read(32)) if fh is not None else None
                            total += size[0] * size[1] * 3 if size else 0
        except (OSError, EOFError, zlib.error, tarfile.TarError):
            continue
    return total


# ---------------------------------------------------------------------------
# Attaching a sample's positions to its cells
# ---------------------------------------------------------------------------


def attach(adata, sr: SpaceRanger, library_id: str, *, sample_column: str | None = None,
           sample: str | None = None, label: str | None = None) -> int:
    """Put ``sr``'s spots on the cells of ``adata`` (those with ``sample_column`` =
    ``sample`` in a merge, whose names carry ":<sample>"): obsm["spatial"] (x, y),
    obs library_id / array_row / array_col, uns["spatial"][library_id]. Every cell must
    be an in-tissue spot of the positions file (SpatialError names the count). Returns
    the number of spots."""
    n = adata.n_obs
    names = adata.obs_names.astype(str)
    if sample_column and sample is not None:
        rows = (adata.obs[sample_column].astype(str) == str(sample)).to_numpy()
        keys = pd.Index([s.rsplit(":", 1)[0] if s.endswith(f":{sample}") else s
                         for s in names[rows]])
    else:
        rows = np.ones(n, dtype=bool)
        keys = pd.Index(names)
    pos = sr.positions.reindex(keys)
    missing = int(pos["pxl_col_in_fullres"].isna().sum())
    who = label or library_id
    if missing:
        example = ", ".join(map(str, keys[pos["pxl_col_in_fullres"].isna().to_numpy()][:3]))
        raise SpatialError(
            f"{missing:,} of the {len(keys):,} spots in the matrix of {who} are not among the in-tissue "
            f"spots of {sr.files.get(POSITIONS, 'the positions file')} (e.g. {example}). Upload the "
            "spatial folder of the same Space Ranger run as the matrix.")
    coords = np.full((n, 2), np.nan, dtype=np.float64)
    old = adata.obsm.get(KEY) if KEY in adata.obsm else None
    if old is not None and np.shape(old) == (n, 2):
        coords[:] = np.asarray(old, dtype=np.float64)
    coords[rows, 0] = pos["pxl_col_in_fullres"].to_numpy(dtype=np.float64)
    coords[rows, 1] = pos["pxl_row_in_fullres"].to_numpy(dtype=np.float64)
    adata.obsm[KEY] = coords
    obs = adata.obs
    libs = (obs[LIBRARY_COLUMN].astype(object).to_numpy() if LIBRARY_COLUMN in obs.columns
            else np.full(n, None, dtype=object))
    libs[rows] = library_id
    obs[LIBRARY_COLUMN] = libs
    for col in ("array_row", "array_col"):
        vals = (obs[col].to_numpy(dtype=np.int64, na_value=-1) if col in obs.columns
                else np.full(n, -1, dtype=np.int64))
        vals[rows] = pos[col].to_numpy(dtype=np.int64)
        obs[col] = vals
    meta = {"source": "spaceranger", "positions_file": sr.files.get(POSITIONS, ""),
            "positions_layout": sr.layout}
    meta.update({f"{p}_file": f for p, f in sr.files.items() if p != POSITIONS})
    uns = adata.uns.get(KEY)
    uns = dict(uns) if isinstance(uns, Mapping) else {}
    uns[library_id] = {"images": dict(sr.images), "scalefactors": dict(sr.scalefactors),
                       "metadata": meta}
    adata.uns[KEY] = uns
    return int(rows.sum())


def finish(adata) -> dict:
    """After every library is attached: library_id categorical (its categories the
    libraries), uns["scview_spatial"]. Returns the provenance summary."""
    obs = adata.obs
    libs = [str(k) for k in adata.uns[KEY].keys()]
    obs[LIBRARY_COLUMN] = pd.Categorical(obs[LIBRARY_COLUMN].astype(object), categories=libs)
    tech = technology(adata)
    adata.uns[INFO_KEY] = {**_recorded(adata), **tech}
    counts = obs[LIBRARY_COLUMN].value_counts()
    return {"source": "spaceranger", "spaceranger_version": None, **tech,
            "libraries": {lib: int(counts.get(lib, 0)) for lib in libs},
            "images": {lib: sorted(adata.uns[KEY][lib].get("images", {})) for lib in libs}}


def technology(adata) -> dict:
    """``{"technology", "unit"}``: Xenium when an import recorded it; Visium HD bins when
    the scale factors give a bin size; Visium spots when the grid (array_row, array_col)
    and every library's spot diameter are known; else cells."""
    if _recorded(adata).get("technology") == "xenium":
        return _tech("xenium")
    obs = adata.obs
    uns = adata.uns.get(KEY)
    libs = _libraries(adata)
    grid = ("array_row" in obs.columns and "array_col" in obs.columns and bool(libs)
            and isinstance(uns, Mapping)
            and all(_spot_diameter(uns.get(lib)) is not None for lib in libs))
    if not grid:
        return _tech("cells")
    hd = any(_bin_size(uns.get(lib)) is not None for lib in libs)
    return _tech("visium_hd" if hd else "visium")


def _tech(name: str) -> dict:
    return {"technology": name, "unit": TECHNOLOGIES[name]}


def _recorded(adata) -> dict:
    """``uns["scview_spatial"]`` as a plain dict, empty when there is none."""
    rec = adata.uns.get(INFO_KEY) if hasattr(adata, "uns") else None
    return dict(rec) if isinstance(rec, Mapping) else {}


def _bin_size(entry) -> float | None:
    """Visium HD's ``bin_size_um`` scale factor; None for anything else."""
    sf = entry.get("scalefactors") if isinstance(entry, Mapping) else None
    v = sf.get("bin_size_um") if isinstance(sf, Mapping) else None
    try:
        v = float(np.asarray(v).ravel()[0]) if v is not None else None
    except (TypeError, ValueError, IndexError):
        return None
    return v if v is not None and np.isfinite(v) and v > 0 else None


def _spot_diameter(entry) -> float | None:
    sf = entry.get("scalefactors") if isinstance(entry, Mapping) else None
    v = sf.get("spot_diameter_fullres") if isinstance(sf, Mapping) else None
    try:
        v = float(np.asarray(v).ravel()[0]) if v is not None else None
    except (TypeError, ValueError, IndexError):
        return None
    return v if v is not None and np.isfinite(v) and v > 0 else None


def _libraries(adata) -> list[str]:
    obs = adata.obs
    if LIBRARY_COLUMN in obs.columns:
        s = obs[LIBRARY_COLUMN]
        cats = [str(c) for c in s.cat.categories] if isinstance(s.dtype, pd.CategoricalDtype) \
            else sorted({str(v) for v in s.dropna().unique()})
        present = set(s.dropna().astype(str).unique())
        return [c for c in cats if c in present]
    uns = adata.uns.get(KEY)
    return [str(k) for k in uns.keys()][:1] if isinstance(uns, Mapping) and len(uns) else []


# ---------------------------------------------------------------------------
# Adoption of a spatial .h5ad on open
# ---------------------------------------------------------------------------


def has_spatial(adata) -> bool:
    obsm = getattr(adata, "obsm", None)
    if obsm is None:
        return False
    m = obsm.get(KEY) if KEY in obsm else None
    return m is not None and np.ndim(m) == 2 and np.shape(m)[1] >= 2


def adopt(adata) -> bool:
    """Make a spatial .h5ad (scanpy / squidpy / spatialdata-io) follow scView's layout,
    in memory: obsm "X_spatial" copied to "spatial", coordinates as float (x, y; a third
    column kept apart in obsm "spatial_full"), obs library_id (the uns["spatial"] key,
    else "section1"), an uns["spatial"] entry per library and uns["scview_spatial"].
    Returns whether it changed anything; a file already in this layout is left alone."""
    obsm = adata.obsm
    changed = False
    if KEY not in obsm and ALT_KEY in obsm and np.ndim(obsm[ALT_KEY]) == 2:
        obsm[KEY] = np.asarray(obsm[ALT_KEY], dtype=np.float64)
        changed = True
    if not has_spatial(adata):
        return changed
    m = obsm[KEY]
    arr = m.to_numpy() if isinstance(m, pd.DataFrame) else np.asarray(m)
    if arr.shape[1] > 2:
        obsm["spatial_full"] = arr
        arr = arr[:, :2]
    if isinstance(m, pd.DataFrame) or arr is not m or not np.issubdtype(arr.dtype, np.floating):
        obsm[KEY] = np.asarray(arr, dtype=np.float64)
        changed = True
    uns = adata.uns.get(KEY)
    uns = uns if isinstance(uns, Mapping) else {}
    obs = adata.obs
    if LIBRARY_COLUMN not in obs.columns:
        keys = [str(k) for k in uns.keys()]
        column = _column_of(obs, keys) if len(keys) > 1 else None
        if column is not None:
            obs[LIBRARY_COLUMN] = pd.Categorical(obs[column].astype(str), categories=keys)
        else:
            lib = keys[0] if keys else DEFAULT_LIBRARY
            if len(keys) > 1:
                logger.info("Spatial h5ad with %d libraries and no library_id column: all "
                            "cells taken as %s", len(keys), lib)
            obs[LIBRARY_COLUMN] = pd.Categorical([lib] * adata.n_obs, categories=[lib])
        changed = True
    elif not isinstance(obs[LIBRARY_COLUMN].dtype, pd.CategoricalDtype):
        obs[LIBRARY_COLUMN] = obs[LIBRARY_COLUMN].astype(str).astype("category")
        changed = True
    libs = [str(c) for c in obs[LIBRARY_COLUMN].cat.categories]
    missing = [lib for lib in libs if not isinstance(uns.get(lib), Mapping)]
    if missing or not isinstance(adata.uns.get(KEY), Mapping):
        new = dict(uns)
        for lib in missing:
            new[lib] = {"images": {}, "scalefactors": {}, "metadata": {"source": "h5ad"}}
        adata.uns[KEY] = new
        changed = True
    tech = technology(adata)
    recorded = _recorded(adata)
    if any(recorded.get(k) != v for k, v in tech.items()):
        adata.uns[INFO_KEY] = {**recorded, **tech}
        changed = True
    return changed


def _column_of(obs: pd.DataFrame, keys: list[str]) -> str | None:
    """An obs column whose values are exactly the library keys (squidpy's ``library_id``
    under another name: ``sample``, ``section``...)."""
    want = set(keys)
    for col in obs.columns:
        s = obs[col]
        if isinstance(s.dtype, pd.CategoricalDtype) or pd.api.types.is_object_dtype(s) \
                or pd.api.types.is_string_dtype(s):
            if set(s.dropna().astype(str).unique()) == want:
                return str(col)
    return None


# ---------------------------------------------------------------------------
# Merges (ingest of several samples, joint embeddings)
# ---------------------------------------------------------------------------


# Samples whose positions a merge left out (only some samples were spatial), for the
# ingest step's note; removed from uns when the note is written.
DROPPED_KEY = "scview_spatial_dropped"


def after_merge(parts: Mapping, merged, sample_label: str) -> None:
    """Spatial data across a merge of ``parts`` (label -> AnnData) into ``merged``:
    ``ad.concat`` drops uns. When every part is spatial, each library becomes
    ``<label>`` (or ``<label>_<library>`` for a part of several), obs library_id and
    uns["spatial"] follow; when only some are, the coordinates are dropped (they would
    hold no position for the other cells)."""
    flags = {lab: has_spatial(a) for lab, a in parts.items()}
    if not any(flags.values()):
        return
    if not all(flags.values()):
        merged.obsm.pop(KEY, None)
        kept_out = [str(k) for k, v in flags.items() if v]
        # Picked up into the ingest step's History note (ingestion/session.py).
        merged.uns[DROPPED_KEY] = kept_out
        logger.info("Spatial coordinates dropped from the merge: only %s have them", ", ".join(kept_out))
        return
    labels = merged.obs[sample_label].astype(str).to_numpy()
    libs = np.empty(merged.n_obs, dtype=object)
    uns: dict = {}
    for lab, a in parts.items():
        adopt(a)
        own = a.obs[LIBRARY_COLUMN].astype(str).to_numpy()
        names = [str(c) for c in a.obs[LIBRARY_COLUMN].cat.categories]
        rename = {n: str(lab) if len(names) == 1 else f"{lab}_{n}" for n in names}
        rows = labels == str(lab)
        if rows.sum() != len(own):
            raise ValueError(f"The merged rows of {lab} do not match its cells.")
        libs[rows] = [rename[x] for x in own]
        src = a.uns.get(KEY) or {}
        for n, new in rename.items():
            uns[new] = src.get(n) if isinstance(src.get(n), Mapping) else {
                "images": {}, "scalefactors": {}, "metadata": {"source": "h5ad"}}
    merged.obsm[KEY] = np.asarray(merged.obsm[KEY], dtype=np.float64)
    merged.obs[LIBRARY_COLUMN] = pd.Categorical(libs, categories=list(uns))
    merged.uns[KEY] = uns
    # ad.concat dropped uns: a recorded technology survives when every part shares it.
    recorded = {str(_recorded(a).get("technology")) for a in parts.values()}
    if len(recorded) == 1:
        merged.uns[INFO_KEY] = {"technology": recorded.pop()}
    merged.uns[INFO_KEY] = technology(merged)


# ---------------------------------------------------------------------------
# What the API serves
# ---------------------------------------------------------------------------


def info(adata) -> dict | None:
    """SpatialInfo (contract.md) of a dataset, or None without obsm["spatial"]."""
    if not has_spatial(adata):
        return None
    tech = technology(adata)
    coords = np.asarray(adata.obsm[KEY])[:, :2].astype(np.float64, copy=False)
    uns = adata.uns.get(KEY)
    uns = uns if isinstance(uns, Mapping) else {}
    obs = adata.obs
    libs = _libraries(adata) or [DEFAULT_LIBRARY]
    lib_of = (obs[LIBRARY_COLUMN].astype(str).to_numpy() if LIBRARY_COLUMN in obs.columns
              else np.full(adata.n_obs, libs[0], dtype=object))
    out = []
    for lib in libs:
        rows = lib_of == lib
        xy = coords[rows]
        xy = xy[np.isfinite(xy).all(axis=1)]
        bounds = ([float(xy[:, 0].min()), float(xy[:, 1].min()), float(xy[:, 0].max()),
                   float(xy[:, 1].max())] if len(xy) else [0.0, 0.0, 0.0, 0.0])
        entry = uns.get(lib) if isinstance(uns.get(lib), Mapping) else {}
        out.append({"id": lib, "n_obs": int(rows.sum()), "images": _image_info(entry),
                    "spot_diameter_fullres": _spot_diameter(entry), "bounds": bounds})
    return {**tech, "library_column": LIBRARY_COLUMN, "libraries": out}


def _image_info(entry: Mapping) -> dict:
    images = entry.get("images") if isinstance(entry.get("images"), Mapping) else {}
    sf = entry.get("scalefactors") if isinstance(entry.get("scalefactors"), Mapping) else {}
    out = {}
    for res in RESOLUTIONS:
        img = images.get(res)
        scalef = sf.get(f"tissue_{res}_scalef")
        if img is None or scalef is None or np.ndim(img) < 2:
            continue
        try:
            scalef = float(np.asarray(scalef).ravel()[0])
        except (TypeError, ValueError, IndexError):
            continue
        if not np.isfinite(scalef) or scalef <= 0:
            continue
        h, w = np.shape(img)[:2]
        out[res] = {"width": int(w), "height": int(h), "scalef": scalef}
    return out


_png_cache: "weakref.WeakKeyDictionary[object, dict]" = weakref.WeakKeyDictionary()
_png_lock = threading.Lock()


def image_array(adata, library_id: str, res: str) -> np.ndarray:
    """A library's tissue image as stored; KeyError naming what is missing."""
    if res not in RESOLUTIONS:
        raise KeyError(f"Unknown resolution '{res}' (hires or lowres).")
    uns = adata.uns.get(KEY)
    entry = uns.get(library_id) if isinstance(uns, Mapping) else None
    if not isinstance(entry, Mapping):
        raise KeyError(f"No spatial library '{library_id}'.")
    images = entry.get("images") if isinstance(entry.get("images"), Mapping) else {}
    if images.get(res) is None:
        raise KeyError(f"Library '{library_id}' has no {res} image.")
    return np.asarray(images[res])


def image_png(adata, library_id: str, res: str) -> bytes:
    """A library's tissue image as PNG bytes, encoded once per open dataset (cached
    against the AnnData object, so a reprocessed dataset encodes again). KeyError when
    the library or resolution is not there."""
    img = image_array(adata, library_id, res)
    with _png_lock:
        try:
            cache = _png_cache.setdefault(adata, {})
        except TypeError:  # not weak-referenceable: no cache
            cache = {}
        hit = cache.get((library_id, res))
    if hit is not None:
        return hit
    data = encode_png(img)
    with _png_lock:
        cache[(library_id, res)] = data
    return data


def encode_png(img: np.ndarray) -> bytes:
    """uint8 PNG bytes of an image array (HxW, HxWx3 or HxWx4; float 0..1 scaled)."""
    from PIL import Image

    a = np.asarray(img)
    if np.issubdtype(a.dtype, np.floating):
        top = float(np.nanmax(a)) if a.size else 1.0
        a = np.nan_to_num(a) * (255.0 if top <= 1.0 else 1.0)
    a = np.clip(a, 0, 255).astype(np.uint8)
    if a.ndim == 3 and a.shape[2] == 4:
        a = a[..., :3]
    buf = io.BytesIO()
    Image.fromarray(np.ascontiguousarray(a)).save(buf, format="PNG", compress_level=6)
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Spot data: steps that do not apply, the co-pilot's fact
# ---------------------------------------------------------------------------


def is_spot_data(adata) -> bool:
    """Whether the observations are capture spots or bins, not cells."""
    recorded = _recorded(adata)
    if recorded.get("unit"):
        return str(recorded.get("unit")) in SPOT_UNITS
    return has_spatial(adata) and technology(adata)["unit"] in SPOT_UNITS


def targeted_panel(adata) -> bool:
    """Whether the genes are a targeted panel (core/targeted_panel.py): the flag an import
    recorded, else spatial cells with at most ``PANEL_MAX_GENES`` genes measured."""
    recorded = _recorded(adata)
    if "targeted_panel" in recorded:
        return bool(np.asarray(recorded["targeted_panel"]).ravel()[0])
    if not has_spatial(adata) or technology(adata)["technology"] != "cells":
        return False
    raw = getattr(adata, "raw", None)
    return max(adata.n_vars, raw.n_vars if raw is not None else 0) <= PANEL_MAX_GENES


def grounding_lines(adata) -> tuple[list[str], str]:
    """The co-pilot's fact about the spatial data: technology, sections, spots or cells,
    images, a targeted panel's size. ``(lines, detail)``; empty without spatial data."""
    sp = info(adata)
    if sp is None:
        return [], ""
    libs = sp["libraries"]
    unit = sp["unit"]
    n = sum(lib["n_obs"] for lib in libs)
    tech = TECHNOLOGY_NAMES[sp["technology"]]
    sections = f"{len(libs)} section{'s' if len(libs) != 1 else ''}"
    per = ", ".join(f"{lib['id']} {lib['n_obs']:,}" for lib in libs[:8])
    images = sorted({res for lib in libs for res in lib["images"]})
    line = (f"- Spatial: {tech}, {sections}, {n:,} {unit}s ({per}); tissue images: "
            f"{', '.join(images) if images else 'none'}.")
    if unit == "spot":
        line += (" Each Visium spot holds several cells, so values are per spot, not per cell, and "
                 "doublet detection does not apply.")
    panel = ""
    if targeted_panel(adata):
        from scview.core import targeted_panel as tp

        panel = f"targeted panel of {tp.n_genes(adata):,} genes"
        line += (f" A {panel}: normalised to the cells' median total counts, doublet detection does "
                 "not apply, and enrichment is against the panel's genes only.")
    line += f" [{TAG}]"
    return [line], f"{tech}, {sections}, {n:,} {unit}s" + (f", {panel}" if panel else "")
