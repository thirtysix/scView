"""Space Ranger 4's cell segmentation of a Visium HD run, read as a dataset of cells.

``segmented_outputs/`` holds ``filtered_feature_cell_matrix.h5`` (or that folder; barcodes
``cellid_000000001-1``), ``spatial/`` (scale factors and tissue images, no positions) and GeoJSON
outlines in full-resolution pixels (``cell_segmentations``, ``nucleus_segmentations``: Polygon,
properties ``cell_id`` an integer, each ring closed). ``graphclust_annotated_cell_segmentations``
repeats the cell outlines with Space Ranger's clusters in ``classification.name``.

* obsm["spatial"]: the area-weighted centroid of each cell's outline, in full-resolution pixels,
  so the tissue images and scale factors work as for bins.
* obs ``library_id``, and ``spaceranger_graphclust`` when that file came with the folder.
* uns["scview_spatial"]: technology visium_hd, unit cell. These cells are on no grid: readers that
  mean a grid ask the unit (``spatial.SPOT_UNITS``), not the technology.
* outlines: ``core/cell_outlines.py``, never in the h5ad, so a re-uploaded export has points only.

A cell with no outline has no position, so an upload with any is refused with the count. Outlines
with no cell in the filtered matrix are left out of the store and counted in the ingest record.
"""

from __future__ import annotations

import json
import os
import re
from array import array
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

import numpy as np
import pandas as pd

from scview.core import cell_outlines, spatial

TECHNOLOGY = "visium_hd"
UNIT = "cell"
PART_CELLS = "cell_segmentations"
PART_NUCLEI = "nucleus_segmentations"
PART_GRAPHCLUST = "graphclust_annotated_cell_segmentations"
PARTS = (PART_CELLS, PART_NUCLEI, PART_GRAPHCLUST)
GRAPHCLUST_COLUMN = "spaceranger_graphclust"
KIND_OF_PART = {PART_CELLS: "cell", PART_NUCLEI: "nucleus"}
_CHUNK_CHARS = 4 * 1024 * 1024  # read at a time: a feature is about 1 KB
_BARCODE_RE = r"^cellid_0*(\d+)-\d+$"
_SPACE = re.compile(r"[\s,]*")
_NUMBER = re.compile(r"(\d+)$")


class HdCellsError(ValueError):
    """Segmentation files that cannot be read or matched to the cells; the message says why."""


def sniff(path: Path, head: bytes) -> dict | None:
    """``{"part", "reason"}`` for an outline file of a segmented_outputs folder, by its name;
    None for any other file."""
    name = path.name.lower()
    for part in PARTS:
        if name == f"{part}.geojson" and head.lstrip()[:1] == b"{":
            return {"part": part, "reason": "Space Ranger cell segmentation (GeoJSON)"}
    return None


def iter_features(path, shown: str | None = None) -> Iterator[dict]:
    """The features of a GeoJSON FeatureCollection, decoded one at a time, so a file of hundreds
    of megabytes takes a few megabytes to read (json.load of the same file: its size ten times over)."""
    shown = shown or Path(path).name
    decoder = json.JSONDecoder()
    with open(path, encoding="utf-8", errors="replace") as fh:
        buf = fh.read(_CHUNK_CHARS)
        at = buf.find('"features"')
        at = buf.find("[", at) if at >= 0 else -1
        if at < 0:
            raise HdCellsError(f"{shown} is not a GeoJSON FeatureCollection: upload the file Space Ranger wrote.")
        pos = at + 1
        while True:
            pos = _SPACE.match(buf, pos).end()
            if pos < len(buf) and buf[pos] == "]":
                return
            try:
                feature, pos = decoder.raw_decode(buf, pos)
            except json.JSONDecodeError as e:
                more = fh.read(_CHUNK_CHARS)
                if not more:
                    raise HdCellsError(f"{shown} ends in the middle of a feature: upload it again.") from e
                buf, pos = buf[pos:] + more, 0
                continue
            if isinstance(feature, dict):
                yield feature


def _outer_ring(geometry) -> list | None:
    """The outline of a feature: a Polygon's outer ring, a MultiPolygon's longest outer ring."""
    kind = geometry.get("type") if isinstance(geometry, dict) else None
    coords = geometry.get("coordinates") if kind else None
    if not coords:
        return None
    if kind == "Polygon":
        return coords[0]
    if kind == "MultiPolygon":
        return max((p[0] for p in coords if p), key=len, default=None)
    return None


def _cell_number(value) -> int:
    """A feature's cell_id as its number: an integer in Space Ranger 4.0's files, a barcode's
    digits if it is written as a barcode."""
    if isinstance(value, str):
        m = _NUMBER.search(value.split("-", 1)[0])
        if m is None:
            raise ValueError(value)
        return int(m.group(1))
    return int(value)


def read_rings(path, shown: str | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(cell numbers, offsets, xy): the outer ring of each feature as an open ring (the closing
    vertex dropped), float64. A feature with under 3 distinct vertices is not an outline."""
    shown = shown or Path(path).name
    numbers, lengths, flat = array("q"), array("q"), array("d")
    for feature in iter_features(path, shown):
        ring = _outer_ring(feature.get("geometry"))
        if ring is not None and len(ring) > 1 and ring[0][:2] == ring[-1][:2]:
            ring = ring[:-1]
        if ring is None or len(ring) < 3:
            continue
        try:
            number = _cell_number(feature["properties"]["cell_id"])
            for v in ring:
                flat.append(v[0])
                flat.append(v[1])
        except (KeyError, TypeError, ValueError, IndexError) as e:
            raise HdCellsError(f"{shown} has a feature without a cell_id and x, y vertices: upload the file "
                               "Space Ranger wrote.") from e
        numbers.append(number)
        lengths.append(len(ring))
    ids = np.frombuffer(numbers, dtype=np.int64) if len(numbers) else np.zeros(0, dtype=np.int64)
    if ids.size == 0:
        raise HdCellsError(f"{shown} holds no cell outline: upload the file Space Ranger wrote.")
    if np.unique(ids).size != ids.size:
        raise HdCellsError(f"{shown} lists a cell more than once: upload the file Space Ranger wrote.")
    offsets = np.zeros(ids.size + 1, dtype=np.int64)
    offsets[1:] = np.cumsum(np.frombuffer(lengths, dtype=np.int64))
    return ids.copy(), offsets, np.frombuffer(flat, dtype=np.float64).reshape(-1, 2).copy()


def ring_centroids(offsets, xy):
    """The area-weighted centroid of each open ring (x, y); a ring with no area: its vertices' mean."""
    n = len(offsets) - 1
    starts, lengths = offsets[:-1], np.diff(offsets)
    following = np.arange(len(xy)) + 1
    following[offsets[1:] - 1] = starts
    origin = np.repeat(xy[starts], lengths, axis=0)  # against the ring's own first vertex: no rounding at 1e4 px
    p, q = xy - origin, xy[following] - origin
    cross = p[:, 0] * q[:, 1] - q[:, 0] * p[:, 1]
    area = np.add.reduceat(cross, starts)  # twice the signed area
    out = np.empty((n, 2), dtype=np.float64)
    means = np.add.reduceat(xy, starts, axis=0) / lengths[:, None]
    for axis in (0, 1):
        moment = np.add.reduceat((p[:, axis] + q[:, axis]) * cross, starts)
        with np.errstate(divide="ignore", invalid="ignore"):
            out[:, axis] = xy[starts, axis] + moment / (3.0 * area)
    flat = np.abs(area) < 1e-9
    out[flat] = means[flat]
    return out


def read_clusters(path, shown: str | None = None) -> dict[int, str]:
    """Cell number -> Space Ranger's graph-based cluster, from the annotated outlines."""
    out: dict[int, str] = {}
    for feature in iter_features(path, shown):
        props = feature.get("properties") or {}
        name = (props.get("classification") or {}).get("name")
        if name is not None and props.get("cell_id") is not None:
            out[_cell_number(props["cell_id"])] = str(name)
    return out


def cell_numbers(names) -> np.ndarray:
    """Each barcode's cell number (``cellid_000000001-1`` -> 1), -1 for any other name."""
    found = pd.Series(pd.Index(names).astype(str)).str.extract(_BARCODE_RE, expand=False)
    return pd.to_numeric(found, errors="coerce").fillna(-1).to_numpy(dtype=np.int64)


def run_label(rel: str, prefix: str = "") -> str:
    """A dataset's name: the sample prefix of its file, else the folder above ``segmented_outputs``
    other than Space Ranger's own (``Mouse_Brain_cells``), else ``visium_hd_cells``."""
    run = spatial.hd_seg_folder(rel) or ""
    own = next((d for d in reversed(PurePosixPath(run).parts) if d.lower() != "outs"), None) if run else None
    return f"{prefix or own or 'visium_hd'}_cells"


@dataclass
class Outlines:
    """One sample's outlines of one kind, in the order of its cells; a cell may have none."""

    ids: list[str]
    offsets: np.ndarray
    xy: np.ndarray
    source: str = ""


def _aligned(numbers, offsets, xy, wanted) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(row per cell or -1, offsets, xy) of the rings of the cells ``wanted``, in that order."""
    rows = pd.Index(numbers).get_indexer(wanted)
    lengths = np.where(rows >= 0, offsets[rows + 1] - offsets[rows], 0)
    out = np.zeros(len(rows) + 1, dtype=np.int64)
    out[1:] = np.cumsum(lengths)
    kept = rows >= 0
    starts = np.repeat(offsets[rows[kept]], lengths[kept])
    within = np.arange(out[-1], dtype=np.int64) - np.repeat(out[:-1][kept], lengths[kept])
    return rows, out, xy[starts + within]


def attach(adata, files: Mapping[str, tuple[str, str]], sr: spatial.SpaceRanger | None, library_id: str, *,
           sample_column: str | None = None, sample: str | None = None,
           label: str | None = None) -> tuple[dict[str, Outlines], dict]:
    """Put one sample's segmentation on its cells of ``adata``. ``files``: part -> (path, shown
    name), ``sr``: its scale factors and images. Returns ({kind: its outlines}, what the ingest
    record keeps). HdCellsError names the file, the count and what to do."""
    who = label or library_id
    if PART_CELLS not in files:
        raise HdCellsError(f"The segmentation of {who} has no {PART_CELLS}.geojson: upload the whole "
                           "segmented_outputs folder.")
    rows, keys = spatial.sample_rows(adata, sample_column, sample)
    wanted = cell_numbers(keys)
    path, shown = files[PART_CELLS]
    numbers, offsets, xy = read_rings(path, shown)
    found, cell_offsets, cell_xy = _aligned(numbers, offsets, xy, wanted)
    missing = found < 0
    if missing.any():
        example = ", ".join(map(str, keys[missing][:3]))
        raise HdCellsError(
            f"{int(missing.sum()):,} of the {len(keys):,} cells in the matrix of {who} have no outline in {shown} "
            f"(e.g. {example}), so they have no position. Upload the segmented_outputs folder of one Space "
            "Ranger run whole.")
    centroids = ring_centroids(cell_offsets, cell_xy)
    n = adata.n_obs
    coords = np.full((n, 2), np.nan, dtype=np.float64)
    old = adata.obsm.get(spatial.KEY) if spatial.KEY in adata.obsm else None
    if old is not None and np.shape(old) == (n, 2):
        coords[:] = np.asarray(old, dtype=np.float64)
    coords[rows] = centroids
    adata.obsm[spatial.KEY] = coords
    obs = adata.obs
    libs = (obs[spatial.LIBRARY_COLUMN].astype(object).to_numpy() if spatial.LIBRARY_COLUMN in obs.columns
            else np.full(n, None, dtype=object))
    libs[rows] = library_id
    obs[spatial.LIBRARY_COLUMN] = libs
    names = list(adata.obs_names.astype(str)[rows])
    store = {"cell": Outlines(names, cell_offsets, cell_xy, shown)}
    record: dict = {"outlines": len(keys), "outlines_without_a_cell": int(numbers.size - found.size)}
    if PART_NUCLEI in files:
        path, shown = files[PART_NUCLEI]
        numbers, offsets, xy = read_rings(path, shown)
        found, nucleus_offsets, nucleus_xy = _aligned(numbers, offsets, xy, wanted)
        store["nucleus"] = Outlines(names, nucleus_offsets, nucleus_xy, shown)
        record["nuclei"] = int((found >= 0).sum())
    if PART_GRAPHCLUST in files:
        clusters = read_clusters(*files[PART_GRAPHCLUST])
        values = (obs[GRAPHCLUST_COLUMN].astype(object).to_numpy() if GRAPHCLUST_COLUMN in obs.columns
                  else np.full(n, None, dtype=object))
        # A merge's samples were clustered apart: their "Cluster-1"s are not one group.
        prefix = f"{sample}:" if sample_column and sample is not None else ""
        values[rows] = [prefix + clusters[c] if c in clusters else None for c in wanted.tolist()]
        obs[GRAPHCLUST_COLUMN] = values
    meta = {"source": "spaceranger", "segmentation": files[PART_CELLS][1]}
    meta.update({f"{p}_file": f for p, f in (sr.files if sr else {}).items()})
    uns = adata.uns.get(spatial.KEY)
    uns = dict(uns) if isinstance(uns, Mapping) else {}
    uns[library_id] = {"images": dict(sr.images) if sr else {}, "scalefactors": dict(sr.scalefactors) if sr else {},
                       "metadata": meta}
    adata.uns[spatial.KEY] = uns
    adata.uns[spatial.INFO_KEY] = {"technology": TECHNOLOGY, "unit": UNIT}
    return store, record


def finish(adata, records: list[dict]) -> dict:
    """After every sample is attached: the graph clusters categorical in their own order, the
    library column and uns as ``spatial.finish`` leaves them. Returns the ingest record."""
    if GRAPHCLUST_COLUMN in adata.obs.columns:
        values = adata.obs[GRAPHCLUST_COLUMN].astype(object)
        cats = sorted(values.dropna().unique(), key=_cluster_order)
        adata.obs[GRAPHCLUST_COLUMN] = pd.Categorical(values.where(values.notna(), None), categories=cats)
    out = spatial.finish(adata)
    for key in ("outlines", "outlines_without_a_cell", "nuclei"):
        if any(key in r for r in records):
            out[key] = int(sum(r.get(key, 0) for r in records))
    return out


def _cluster_order(name: str) -> tuple:
    """Cluster-2 before Cluster-10, within each sample of a merge."""
    m = _NUMBER.search(str(name))
    return (str(name)[: m.start()], int(m.group(1))) if m else (str(name), -1)


def write_store(folder: Path, pieces: list[dict[str, Outlines]]) -> dict[str, int]:
    """Write the outlines of every sample as one store per kind (a sample without nuclei has none
    to give). Returns kind -> outlines with vertices. The session moves ``folder`` beside the dataset."""
    written: dict[str, int] = {}
    for kind in cell_outlines.KINDS:
        have = [p[kind] for p in pieces if kind in p]
        if not have:
            continue
        ids = [i for o in have for i in o.ids]
        lengths = np.concatenate([np.diff(o.offsets) for o in have])
        offsets = np.zeros(len(ids) + 1, dtype=np.int64)
        offsets[1:] = np.cumsum(lengths)
        xy = np.concatenate([o.xy for o in have]).astype(np.float32)
        cell_outlines.write(folder / f"{kind}.h5", ids, offsets, xy, source=", ".join(o.source for o in have))
        written[kind] = int((lengths > 0).sum())
    return written


def replay_positions(adata, folder, library_id):
    """obsm["spatial"], obs library_id and spaceranger_graphclust from the GeoJSON files beside
    the matrix, as scView's import made them: a replay from the upload, not from its stored .h5ad."""
    cells = os.path.join(folder, "cell_segmentations.geojson")
    if not os.path.isfile(cells):
        print(f"No {cells}: this replay has no tissue positions, and a step that needs them will fail.")
        return
    wanted = adata.obs_names.astype(str).str.extract(r"^cellid_0*(\d+)-\d+$", expand=False).astype("int64")
    with open(cells, encoding="utf-8") as fh:
        features = json.load(fh)["features"]
    numbers, offsets, flat = [], [0], []
    for feature in features:
        ring = feature["geometry"]["coordinates"][0]
        ring = ring[:-1] if ring[0][:2] == ring[-1][:2] else ring
        numbers.append(int(feature["properties"]["cell_id"]))
        offsets.append(offsets[-1] + len(ring))
        flat.extend(v[:2] for v in ring)
    offsets, flat = np.asarray(offsets, dtype=np.int64), np.asarray(flat, dtype=np.float64)
    rows = pd.Index(numbers).get_indexer(wanted)
    adata.obsm["spatial"] = ring_centroids(offsets, flat)[rows]
    adata.obs["library_id"] = pd.Categorical([library_id] * adata.n_obs, categories=[library_id])
    adata.uns["scview_spatial"] = {"technology": "visium_hd", "unit": "cell"}
    annotated = os.path.join(folder, "graphclust_annotated_cell_segmentations.geojson")
    if os.path.isfile(annotated):
        with open(annotated, encoding="utf-8") as fh:
            clusters = {int(f["properties"]["cell_id"]): f["properties"]["classification"]["name"]
                        for f in json.load(fh)["features"]}
        values = [clusters.get(c) for c in wanted.tolist()]
        adata.obs["spaceranger_graphclust"] = pd.Categorical(
            values, categories=sorted(set(clusters.values()), key=lambda s: int(s.rsplit("-", 1)[-1])))


REPLAY_CODE = (ring_centroids, replay_positions)
