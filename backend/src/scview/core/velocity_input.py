"""Spliced and unspliced counts: getting them into a dataset's modality store.

They reach scView three ways, and all three end in ``DATA_DIR/modalities/<id>/velocity.h5``
(``core/modality_store.py``; matrices ``spliced``, ``unspliced``, ``ambiguous`` if present,
addressed by the dataset's own cell names), never in ``layers``, where they would be read on
every open and rewritten on every save:

- an import-wizard upload (h5ad, loom) with both layers: ``adopt_anndata`` at commit, or, when
  the file is ingested in blocks, ``adopt_file`` from the original;
- a plain ``POST /datasets/upload`` of an h5ad with both layers: ``adopt_upload`` writes the
  dataset's working source to ``ingested/{id}/`` without them (the upload stays as sent);
- ``POST /datasets/{id}/velocity/attach``: a velocyto ``.loom``, an h5ad with the layers, or a
  STARsolo ``Velocyto/filtered/`` folder (``spliced.mtx``, ``unspliced.mtx``, ``ambiguous.mtx``,
  ``barcodes.tsv``, ``features.tsv``; genes x cells, may be gzipped): ``attach``. velocyto names a
  cell ``sample:BARCODEx`` where the dataset says ``BARCODE-1``, so cells are matched by the
  barcode inside each name (``match_cells``); the sample part is used only when a barcode occurs
  more than once. A file that matches no clear majority is refused. Only the matched cells are
  stored, under the dataset's names.

``uns["scview_velocity"]`` (a JSON string, shared with ``core/velocity.py``'s run) holds
``source``: the file, how many cells and genes matched.

No file here is read whole when a block will do: an h5ad's CSR layer and a loom's chunks are read
by cell blocks. The exceptions are a CSC layer (rare; converted in memory, one matrix at a time)
and a Matrix Market file not sorted by cell (bucketed on disk by ``streamed._MexRows``).
"""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pandas as pd
import scipy.sparse as sp

from scview.core import modality_store as ms

logger = logging.getLogger(__name__)

STORE = "velocity"
UNS_KEY = "scview_velocity"
REQUIRED = ("spliced", "unspliced")
OPTIONAL = ("ambiguous",)
LAYERS = (*REQUIRED, *OPTIONAL)
BLOCK_CELLS = 512  # a loom's dense columns read at once (about 120 MB at 30,000 genes, float64)
BLOCK_VALUES = 4_000_000  # stored values of an h5ad's CSR read at once
MIN_DATASET_SHARE = 0.5  # matched / the dataset's cells: a clear majority, or
MIN_FILE_SHARE = 0.9  # matched / the file's cells: this file is one sample of a merged dataset
MAX_UPLOAD_BYTES = 200 * 1024**3
ID_COLUMNS = ("gene_ids", "gene_id", "Accession", "accession", "ensembl_id", "ensembl_ids", "gene_ensembl", "ENSEMBL")
SOURCE_HELP = ("Bring a velocyto .loom, an .h5ad with layers 'spliced' and 'unspliced', or a STARsolo "
               "Velocyto/filtered folder.")


class VelocityInputError(RuntimeError):
    """A file that cannot give velocity counts; the message says what to do."""

    def __init__(self, message: str, status: int = 422) -> None:
        super().__init__(message)
        self.status = status


def store_path(data_dir: str | Path, dataset_id: str) -> Path:
    return ms.store_path(data_dir, dataset_id, STORE)


def has_store(data_dir: str | Path, dataset_id: str) -> bool:
    return store_path(data_dir, dataset_id).exists()


def read_record(adata) -> dict[str, Any]:
    """``uns["scview_velocity"]`` as a dict ({} when absent or unreadable)."""
    raw = adata.uns.get(UNS_KEY) if hasattr(adata, "uns") else None
    if raw is None:
        return {}
    try:
        data = json.loads(raw) if isinstance(raw, str) else dict(raw)
    except (TypeError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def write_source(adata, source: dict[str, Any]) -> None:
    """Record where the counts came from; replaced, never mutated in place. Counts from another
    file make an earlier run's result meaningless, so it goes too (``core/velocity.py`` saves it
    in the same entry)."""
    adata.uns[UNS_KEY] = json.dumps({"source": source})


# ---------------------------------------------------------------------------
# Matching a file's cells to the dataset's
# ---------------------------------------------------------------------------

_BARCODE = re.compile(r"[ACGTN]{6,}")


def barcode_of(name: str) -> tuple[str, str]:
    """(barcode, what is left of the name): ``sample:ACGTx`` -> (``ACGT``, ``sample``),
    ``ACGT-1`` -> (``ACGT``, ``1``), ``ACGT-1:s2`` -> (``ACGT``, ``1 s2``). The longest run of
    bases is the barcode; a name without one is its own barcode."""
    text = str(name)
    runs = list(_BARCODE.finditer(text))
    if not runs:
        return text, ""
    best = max(runs, key=lambda m: m.end() - m.start())
    rest = (text[:best.start()] + " " + text[best.end():]).strip()
    rest = re.sub(r"(^|\s)x($|\s)", " ", rest)  # velocyto's trailing x
    return best.group(0), " ".join(t for t in re.split(r"[\s:_\-.]+", rest) if t)


@dataclass
class CellMatch:
    """``file_rows[i]`` of the file is the dataset's ``dataset_names[i]``. ``samples``: each
    sample of the file (the name around the barcode, e.g. velocyto's ``sample:``), the dataset's
    sample it was found to be (None: not in this dataset), and its cells; ``sample_key`` says
    where the dataset's samples were read (``cell names`` or an obs column)."""

    file_rows: np.ndarray
    dataset_names: list[str]
    n_dataset: int
    n_file: int
    n_ambiguous: int = 0
    used_sample: bool = False
    samples: list[dict[str, Any]] = field(default_factory=list)
    sample_key: str | None = None
    n_other_sample: int = 0  # the file's cells whose barcode is in the dataset, but under another sample
    samples_without: list[dict[str, Any]] = field(default_factory=list)  # the dataset's samples no file sample is
    n_dataset_in_samples: int | None = None  # the dataset's cells in the samples the file has

    @property
    def n_matched(self) -> int:
        return len(self.dataset_names)


MIN_SAMPLE_SHARE = 0.8  # of a file sample's barcode-unique pairs that must fall in one dataset sample
MIN_SAMPLE_CELLS = 0.02  # of a file sample's cells paired, at least: below, chance barcode collisions
MAX_SAMPLE_COLUMNS = 30  # obs columns tried as the dataset's sample


def sample_columns(obs) -> dict[str, list[str]]:
    """Label columns of the dataset that could name its samples: 2 to 200 values, each held by
    many cells (at most a tenth of the cells as values)."""
    out: dict[str, list[str]] = {}
    n = len(obs)
    for c in obs.columns:
        s = obs[c]
        if not (isinstance(s.dtype, pd.CategoricalDtype) or s.dtype == object or pd.api.types.is_string_dtype(s)):
            continue
        k = s.nunique(dropna=True)
        if 2 <= k <= 200 and k <= max(2, n // 10):
            out[str(c)] = s.astype(str).where(s.notna(), "").tolist()
        if len(out) >= MAX_SAMPLE_COLUMNS:
            break
    return out


def _learn_samples(tentative: list[tuple[int, int]], file_sample: list[str], keys: dict[str, list[str]],
                   file_sizes: dict[str, int], ds_codes: list[str], file_codes: set[str]
                   ) -> tuple[str, dict[str, str]] | None:
    """Which of the dataset's sample keys (the name around the barcode, or an obs column) tells
    the file's samples apart, from the cells whose barcode is unique on both sides: (key, file
    sample -> dataset sample). A file sample is placed when most of its unique pairs fall in one
    dataset sample and they are more than chance collisions. A key is no sample key when a value
    a file sample took has most of its barcodes outside the file, or one no file sample took has
    most inside it (a QC flag or a cluster cuts across samples). The key placing the most distinct
    samples, then splitting the dataset, then the most pairs, wins. None when nothing has two
    samples to tell apart."""
    best = None
    for key, values in keys.items():
        table: dict[str, dict[str, int]] = {}
        for i, d in tentative:
            row = table.setdefault(file_sample[i], {})
            row[values[d]] = row.get(values[d], 0) + 1
        mapping, agreeing = {}, 0
        for f, row in table.items():
            v, top = max(row.items(), key=lambda x: x[1])
            if top >= MIN_SAMPLE_SHARE * sum(row.values()) and top >= max(3, MIN_SAMPLE_CELLS * file_sizes[f]) and v != "":
                mapping[f] = v
                agreeing += top
        if not mapping:
            continue
        taken = set(mapping.values())
        cells: dict[str, list[int]] = {}
        for d, v in enumerate(values):
            c = cells.setdefault(v, [0, 0])
            c[0] += 1
            c[1] += ds_codes[d] in file_codes
        # A sample's cells are (nearly) all in its file; another sample's only by chance collisions.
        if any((hit < 0.5 * n) if v in taken else (hit > 0.5 * n) for v, (n, hit) in cells.items()):
            continue
        score = (len(taken), len(set(values) - {""}) > 1, agreeing)  # "": a name or cell with no sample
        if best is None or score > best[0]:
            best = (score, key, mapping)
    if best is None:
        return None
    (n_values, splits, _), key, mapping = best
    if n_values < 2 and len(file_sizes) < 2 and not splits:
        return None  # one sample each side: the barcode alone decides
    return key, mapping


def match_cells(file_names, dataset_names, dataset_columns: dict[str, list[str]] | None = None) -> CellMatch:
    """Pair the file's cells with the dataset's: exact names first, then by the barcode inside
    the name. When either side has several samples (velocyto's ``sample:`` prefix, a suffix such
    as ``-1`` / ``-2`` or ``_day0``, or a dataset's sample column, ``dataset_columns``), which
    file sample is which dataset sample is learned from the barcodes found once on each side
    (``_learn_samples``), and every cell is then paired within its own sample: a file sample not
    in the dataset pairs with nothing, even where its barcodes recur there. Without that, a
    barcode held by several cells is paired by the other parts of the name. What stays ambiguous
    is left out and counted."""
    file_names = [str(n) for n in file_names]
    dataset_names = [str(n) for n in dataset_names]
    exact = {n: i for i, n in enumerate(dataset_names)}
    pairs: dict[int, int] = {i: exact[n] for i, n in enumerate(file_names) if n in exact}
    if len(pairs) == len(dataset_names) or len(pairs) == len(file_names):
        return _match(pairs, file_names, dataset_names, 0, False)

    ds_parsed = [barcode_of(n) for n in dataset_names]
    file_parsed = [barcode_of(n) for n in file_names]
    by_ds: dict[str, list[int]] = {}
    for i, (code, _rest) in enumerate(ds_parsed):
        by_ds.setdefault(code, []).append(i)
    by_file: dict[str, list[int]] = {}
    for i, (code, _rest) in enumerate(file_parsed):
        by_file.setdefault(code, []).append(i)
    file_sample = [rest for _code, rest in file_parsed]
    file_sizes: dict[str, int] = {}
    for f in file_sample:
        file_sizes[f] = file_sizes.get(f, 0) + 1
    tentative = [(cells[0], by_ds[code][0]) for code, cells in by_file.items()
                 if len(cells) == 1 and len(by_ds.get(code, ())) == 1]
    keys = {"cell names": [rest for _code, rest in ds_parsed], **(dataset_columns or {})}
    learned = _learn_samples(tentative, file_sample, keys, file_sizes, [code for code, _rest in ds_parsed], set(by_file))
    if learned is not None:
        return _match_by_sample(learned, keys, by_file, by_ds, file_sample, file_names, dataset_names, file_sizes)

    pairs = {}
    ambiguous = 0
    used_sample = False
    for code, cells in by_file.items():
        ds = by_ds.get(code)
        if not ds:
            continue
        if len(cells) == 1 and len(ds) == 1:
            pairs[cells[0]] = ds[0]
            continue
        used_sample = True
        taken: set[int] = set()
        for i in cells:
            parts = set(file_sample[i].split())
            hits = [d for d in ds if d not in taken and parts & set(ds_parsed[d][1].split())]
            if len(hits) == 1:
                pairs[i] = hits[0]
                taken.add(hits[0])
            else:
                ambiguous += 1
    return _match(pairs, file_names, dataset_names, ambiguous, used_sample)


def _match_by_sample(learned: tuple[str, dict[str, str]], keys: dict[str, list[str]], by_file: dict[str, list[int]],
                     by_ds: dict[str, list[int]], file_sample: list[str], file_names: list[str], dataset_names: list[str],
                     file_sizes: dict[str, int]) -> CellMatch:
    key, mapping = learned
    values = keys[key]
    pairs: dict[int, int] = {}
    ambiguous = other = 0
    for code, cells in by_file.items():
        ds = by_ds.get(code)
        if not ds:
            continue
        wanted: dict[str, list[int]] = {}
        for i in cells:
            target = mapping.get(file_sample[i])
            if target is None:
                other += 1  # a sample the dataset does not have: its barcode there is another cell
            else:
                wanted.setdefault(target, []).append(i)
        for target, mine in wanted.items():
            hits = [d for d in ds if values[d] == target]
            if len(mine) == 1 and len(hits) == 1:
                pairs[mine[0]] = hits[0]
            elif hits:
                ambiguous += len(mine)  # one barcode twice in a sample, or two file samples in one
            else:
                other += len(mine)  # its own copy was filtered out; the dataset's is another sample's cell
    matched: dict[str, int] = {}
    for i in pairs:
        matched[file_sample[i]] = matched.get(file_sample[i], 0) + 1
    samples = [{"file_sample": f or "(none)", "dataset_sample": mapping.get(f), "cells_in_file": n,
                "cells_matched": matched.get(f, 0)} for f, n in sorted(file_sizes.items())]
    sizes: dict[str, int] = {}
    for v in values:
        sizes[v] = sizes.get(v, 0) + 1
    taken = set(mapping.values())
    out = _match(pairs, file_names, dataset_names, ambiguous, True)
    out.samples, out.sample_key, out.n_other_sample = samples, key, other
    out.samples_without = [{"dataset_sample": v or "(none)", "cells": n} for v, n in sorted(sizes.items()) if v not in taken]
    out.n_dataset_in_samples = sum(n for v, n in sizes.items() if v in taken)
    return out


def _match(pairs: dict[int, int], file_names: list[str], dataset_names: list[str], ambiguous: int,
           used_sample: bool) -> CellMatch:
    rows = np.asarray(sorted(pairs), dtype=np.int64)
    return CellMatch(rows, [dataset_names[pairs[int(i)]] for i in rows], len(dataset_names), len(file_names),
                     ambiguous, used_sample)


def check_majority(m: CellMatch, file_name: str) -> None:
    """Refuse a file that matches no clear majority: of the dataset's cells, of the file's, or,
    once its samples were placed, of the dataset's cells in those samples (one sample's loom
    of a merged dataset, where the author kept a third of its barcodes)."""
    ds_share = m.n_matched / max(m.n_dataset, 1)
    file_share = m.n_matched / max(m.n_file, 1)
    own_share = m.n_matched / max(m.n_dataset_in_samples, 1) if m.n_dataset_in_samples else 0.0
    if ds_share >= MIN_DATASET_SHARE or file_share >= MIN_FILE_SHARE or own_share >= MIN_DATASET_SHARE:
        return
    extra = f"; {m.n_ambiguous:,} barcodes occur in several samples and could not be told apart" if m.n_ambiguous else ""
    raise VelocityInputError(
        f"{file_name} matches only {m.n_matched:,} of this dataset's {m.n_dataset:,} cells "
        f"({m.n_file:,} cells in the file{extra}): check that it comes from the same samples.")


# ---------------------------------------------------------------------------
# Reading a file's matrices by cell blocks
# ---------------------------------------------------------------------------


@dataclass
class Source:
    """A file (or folder) of velocity counts: its names and each matrix as CSR cell blocks in
    file order (``rows(name)``)."""

    kind: str
    label: str
    barcodes: list[str]
    genes: list[str]
    gene_ids: list[str] | None
    matrices: list[str]
    rows: Callable[[str], Iterator[sp.csr_matrix]]
    close: Callable[[], None] = lambda: None
    files: list[str] = field(default_factory=list)


def _text(values) -> list[str]:
    return [v.decode("utf-8") if isinstance(v, bytes) else str(v) for v in values]


def _h5_frame_index(node) -> list[str]:
    """obs / var names of an h5ad: a modern group, or a pre-0.7 compound table."""
    if isinstance(node, h5py.Dataset):
        return _text(node["index"] if "index" in (node.dtype.names or ()) else node[node.dtype.names[0]])
    idx = node.attrs.get("_index", "_index")
    if isinstance(idx, bytes):
        idx = idx.decode()
    from anndata.io import read_elem

    return _text(np.asarray(read_elem(node[idx])))


def _h5_frame_column(node, names: tuple[str, ...]) -> list[str] | None:
    """The first of ``names`` that the table has, as text; None when it has none."""
    if isinstance(node, h5py.Dataset):
        have = node.dtype.names or ()
        for n in names:
            if n in have:
                return _text(node[n])
        return None
    from anndata.io import read_elem

    for n in names:
        if n in node:
            return _text(pd.Series(read_elem(node[n])).astype(str).to_numpy())
    return None


def _enc(node) -> str:
    enc = node.attrs.get("encoding-type") or node.attrs.get("h5sparse_format") or ""
    if isinstance(enc, bytes):
        enc = enc.decode()
    return str(enc).replace("_matrix", "")


def _matrix_shape(node) -> tuple[int, int]:
    shape = node.attrs.get("shape") if "shape" in node.attrs else node.attrs.get("h5sparse_shape")
    return int(shape[0]), int(shape[1])


def h5ad_blocks(node) -> Iterator[sp.csr_matrix]:
    """A stored matrix (cells x genes) of an h5ad as CSR cell blocks: CSR in any anndata layout
    and dense read by blocks, CSC converted in memory."""
    if isinstance(node, h5py.Dataset):
        n = node.shape[0]
        step = max(1, min(BLOCK_CELLS, BLOCK_VALUES // max(node.shape[1], 1)))
        for start in range(0, n, step):
            yield sp.csr_matrix(node[start:start + step], dtype=np.float32)
        return
    enc = _enc(node)
    shape = _matrix_shape(node)
    if enc == "csc":
        whole = sp.csc_matrix((node["data"][:].astype(np.float32), node["indices"][:], node["indptr"][:]), shape=shape)
        whole = whole.tocsr()
        for start in range(0, shape[0], BLOCK_CELLS):
            yield whole[start:start + BLOCK_CELLS]
        return
    if enc != "csr":
        raise VelocityInputError(f"A matrix is stored as {enc or 'an unknown layout'}: save the file with CSR layers.")
    indptr = node["indptr"][:].astype(np.int64)
    start = 0
    while start < shape[0]:
        end = int(np.searchsorted(indptr, indptr[start] + BLOCK_VALUES, side="right")) - 1
        end = min(shape[0], max(end, start + 1))
        lo, hi = int(indptr[start]), int(indptr[end])
        yield sp.csr_matrix((node["data"][lo:hi].astype(np.float32), node["indices"][lo:hi], indptr[start:end + 1] - lo),
                            shape=(end - start, shape[1]))
        start = end


def open_h5ad(path: Path, label: str | None = None) -> Source:
    f = h5py.File(path, "r")
    try:
        layers = f["layers"] if "layers" in f else {}
        have = [k for k in LAYERS if k in layers]
        if not all(k in have for k in REQUIRED):
            raise VelocityInputError(f"{label or path.name} has no layers 'spliced' and 'unspliced' (it has "
                                     f"{sorted(layers.keys()) if layers else 'no layers'}). {SOURCE_HELP}")
        barcodes = _h5_frame_index(f["obs"])
        genes = _h5_frame_index(f["var"])
        ids = _h5_frame_column(f["var"], ID_COLUMNS)
    except Exception:
        f.close()
        raise
    return Source("h5ad", label or path.name, barcodes, genes, ids, have, lambda m: h5ad_blocks(f["layers"][m]),
                  f.close, [path.name])


def open_loom(path: Path, label: str | None = None) -> Source:
    """A loom by velocyto's conventions: genes x cells, ``row_attrs/Gene``, ``row_attrs/Accession``,
    ``col_attrs/CellID``; layers spliced, unspliced, ambiguous (main matrix = spliced)."""
    f = h5py.File(path, "r")
    try:
        layers = f["layers"] if "layers" in f else {}
        have = [k for k in LAYERS if k in layers]
        if not all(k in have for k in REQUIRED):
            raise VelocityInputError(f"{label or path.name} has no layers 'spliced' and 'unspliced'. {SOURCE_HELP}")
        cols, row_attrs = f["col_attrs"], f["row_attrs"]
        cell_key = next((k for k in ("CellID", "obs_names", "cell_names") if k in cols), None)
        gene_key = next((k for k in ("Gene", "var_names", "gene_names") if k in row_attrs), None)
        if cell_key is None or gene_key is None:
            raise VelocityInputError(f"{label or path.name} has no col_attrs/CellID or row_attrs/Gene: is it a "
                                     "velocyto loom?")
        barcodes = _text(cols[cell_key][:])
        genes = _text(row_attrs[gene_key][:])
        id_key = next((k for k in ("Accession", "gene_ids", "gene_id") if k in row_attrs), None)
        ids = _text(row_attrs[id_key][:]) if id_key else None
    except Exception:
        f.close()
        raise

    def rows(matrix: str) -> Iterator[sp.csr_matrix]:
        ds = f["layers"][matrix]
        n = ds.shape[1]
        for start in range(0, n, BLOCK_CELLS):
            yield sp.csr_matrix(ds[:, start:start + BLOCK_CELLS].T, dtype=np.float32)

    return Source("loom", label or path.name, barcodes, genes, ids, have, rows, f.close, [path.name])


def _read_lines(path: Path) -> list[list[str]]:
    from scview.core.ingestion.streamed import _open_text

    with _open_text(path) as fh:
        return [line.rstrip("\n").split("\t") for line in fh if line.strip()]


def _starsolo_files(paths: list[Path]) -> dict[str, Path]:
    """{spliced, unspliced, ambiguous, barcodes, features} from their names, each maybe .gz."""
    found: dict[str, Path] = {}
    for p in paths:
        name = re.sub(r"\.gz$", "", p.name.lower())
        # An upload's stored name may carry a hash in front: the end of the name decides.
        for key, ending in (("spliced", "spliced.mtx"), ("unspliced", "unspliced.mtx"), ("ambiguous", "ambiguous.mtx"),
                            ("barcodes", "barcodes.tsv"), ("features", "features.tsv"), ("features", "genes.tsv")):
            if name == ending or name.endswith("_" + ending):
                found[key] = p
    return found


def open_starsolo(paths: list[Path], workdir: Path, label: str = "STARsolo Velocyto") -> Source:
    from scview.core.ingestion.streamed import StreamUnsupported, _MexRows, _Unsorted

    files = _starsolo_files(paths)
    missing = [k for k in ("spliced", "unspliced", "barcodes", "features") if k not in files]
    if missing:
        raise VelocityInputError(f"The STARsolo folder lacks {', '.join(missing)}: bring spliced.mtx, unspliced.mtx, "
                                 "barcodes.tsv and features.tsv from Solo.out/Velocyto/filtered.")
    barcodes = [r[0] for r in _read_lines(files["barcodes"])]
    feats = _read_lines(files["features"])
    # STARsolo's features.tsv: id, name, type (older genes.tsv: id, name).
    genes = [r[1] if len(r) > 1 else r[0] for r in feats]
    ids = [r[0] for r in feats] if feats and len(feats[0]) > 1 else None
    have = [k for k in LAYERS if k in files]
    mats: dict[str, Any] = {}
    try:
        for k in have:
            mats[k] = _MexRows(files[k], workdir)
            if mats[k].n_obs != len(barcodes) or mats[k].n_vars != len(genes):
                raise VelocityInputError(f"{files[k].name} is {mats[k].n_vars:,} genes x {mats[k].n_obs:,} cells, but the "
                                         f"folder lists {len(genes):,} features and {len(barcodes):,} barcodes.")
    except StreamUnsupported as e:
        raise VelocityInputError(f"The STARsolo matrices could not be read: {e}.") from e

    def rows(matrix: str) -> Iterator[sp.csr_matrix]:
        m = mats[matrix]
        done = 0  # a file found unsorted partway restarts bucketed: rows given already are skipped
        while True:
            seen = 0
            try:
                for indptr, indices, data in m.blocks():
                    n = len(indptr) - 1
                    if seen + n > done:
                        block = sp.csr_matrix((data, indices, indptr), shape=(n, m.n_vars))
                        yield block[done - seen:] if done > seen else block
                        done = seen + n
                    seen += n
                return
            except _Unsorted:
                continue

    return Source("starsolo", label, barcodes, genes, ids, have, rows, files=sorted(p.name for p in files.values()))


def open_several(paths: list[Path], labels: list[str] | None = None) -> Source:
    """Several .loom / .h5ad files as one source: one file per sample (velocyto writes one loom
    per sample). Their cells one after another; a name without velocyto's ``sample:`` gets its
    file's name in front, so a barcode in two files stays two cells. They must list the same
    genes (made with one annotation); the matrices are those every file has."""
    labels = labels if labels and len(labels) == len(paths) else [p.name for p in paths]
    sources: list[Source] = []
    try:
        for p, name in zip(paths, labels, strict=True):
            sources.append(open_loom(p, name) if p.suffix.lower() == ".loom" else open_h5ad(p, name))
        genes = sources[0].genes
        for s in sources[1:]:
            if s.genes != genes:
                raise VelocityInputError(f"{s.label} lists other genes than {sources[0].label} ({len(s.genes):,} vs "
                                         f"{len(genes):,}): attach files made with the same gene annotation.")
    except Exception:
        for s in sources:
            s.close()
        raise
    barcodes: list[str] = []
    for s in sources:
        stem = re.sub(r"\.(loom|h5ad)$", "", s.label, flags=re.I)
        barcodes += [b if ":" in b else f"{stem}:{b}" for b in s.barcodes]
    have = [m for m in LAYERS if all(m in s.matrices for s in sources)]

    def rows(matrix: str) -> Iterator[sp.csr_matrix]:
        for s in sources:
            yield from s.rows(matrix)

    def close() -> None:
        for s in sources:
            s.close()

    kinds = sorted({s.kind for s in sources})
    return Source("+".join(kinds), " + ".join(s.label for s in sources), barcodes, genes, sources[0].gene_ids, have,
                  rows, close, [f for s in sources for f in s.files])


@contextmanager
def open_source(paths: list[Path], workdir: Path, label: str | None = None, labels: list[str] | None = None):
    """The source these uploaded files make: one .h5ad or .loom, several (one per sample), or a
    STARsolo folder's files. ``label``: the name to report one file by (an upload's stored name
    carries a hash); ``labels``: each file's, for several."""
    if not paths:
        raise VelocityInputError(f"No file was sent. {SOURCE_HELP}")
    if len(paths) == 1 and paths[0].suffix.lower() == ".h5ad":
        src = open_h5ad(paths[0], label)
    elif len(paths) == 1 and paths[0].suffix.lower() == ".loom":
        src = open_loom(paths[0], label)
    elif len(paths) > 1 and all(p.suffix.lower() in (".loom", ".h5ad") for p in paths):
        src = open_several(paths, labels)
    elif any(re.search(r"\.mtx(\.gz)?$", p.name.lower()) for p in paths):
        src = open_starsolo(paths, workdir)
    else:
        raise VelocityInputError(f"{label or paths[0].name} is not a velocity file. {SOURCE_HELP}")
    try:
        yield src
    finally:
        src.close()


# ---------------------------------------------------------------------------
# Into the store
# ---------------------------------------------------------------------------


def _unique(names: list[str]) -> list[str]:
    from anndata.utils import make_index_unique

    return list(map(str, make_index_unique(pd.Index(names))))


def write_store(src: Source, path: Path, cell_rows: np.ndarray | None = None, names: list[str] | None = None,
                genes: list[str] | None = None, meta: dict | None = None,
                stop: Callable[[], bool] | None = None) -> None:
    """Copy the source's matrices to a store at ``path``, by blocks, in file order. ``cell_rows``:
    only these rows (ascending), stored under ``names``."""
    barcodes = list(src.barcodes) if names is None else list(names)
    features = _unique(list(src.genes) if genes is None else list(genes))
    attrs = {"gene_id": src.gene_ids} if src.gene_ids is not None else None
    with ms.StoreWriter(path, barcodes, features, attrs, meta or {}) as w:
        for matrix in src.matrices:
            start = 0
            for block in src.rows(matrix):
                if stop is not None and stop():
                    raise VelocityInputError("Cancelled.", 499)
                n = block.shape[0]
                if cell_rows is None:
                    w.append(matrix, block)
                else:
                    lo, hi = np.searchsorted(cell_rows, [start, start + n])
                    if hi > lo:
                        w.append(matrix, block[cell_rows[lo:hi] - start])
                start += n


def _totals(path: Path) -> dict[str, Any]:
    """Stored values per matrix, and the file's overall unspliced share."""
    with ms.ModalityStore(path) as st:
        return {"n_cells": int(st.n_obs), "n_genes": int(st.n_vars),
                "matrices": list(st.matrices), "nnz": {m: st.nnz(m) for m in st.matrices}}


def adopt_anndata(adata, dataset_id: str, data_dir: str | Path, *, file_name: str | None = None) -> dict | None:
    """An upload read into memory: its spliced / unspliced layers moved to the store, out of
    ``layers``. None (nothing changes) unless the dataset has both."""
    if not all(k in adata.layers for k in REQUIRED):
        return None
    path = store_path(data_dir, dataset_id)
    have = [k for k in LAYERS if k in adata.layers]
    ids = next((adata.var[c].astype(str).tolist() for c in ID_COLUMNS if c in adata.var.columns), None)
    with ms.StoreWriter(path, adata.obs_names, adata.var_names, {"gene_id": ids} if ids is not None else None,
                        {"source": file_name or "upload"}) as w:
        for k in have:
            m = sp.csr_matrix(adata.layers[k], dtype=np.float32)
            for start in range(0, m.shape[0], 8192):
                w.append(k, m[start:start + 8192])
    for k in have:
        del adata.layers[k]
    source = {"kind": "upload", "file": file_name, "matrices": have, **_totals(path),
              "cells_matched": int(adata.n_obs), "cells_in_dataset": int(adata.n_obs), "cells_without": 0,
              "genes_in_file": int(adata.n_vars), "genes_by_id": 0, "genes_by_name": int(adata.n_vars)}
    write_source(adata, source)
    logger.info("Moved %s of %s to the velocity store (%d cells)", have, dataset_id, adata.n_obs)
    return source


def file_has_layers(path: str | Path) -> bool:
    """Whether an .h5ad holds both layers (nothing else is read)."""
    try:
        with h5py.File(path, "r") as f:
            layers = f.get("layers")
            return isinstance(layers, h5py.Group) and all(k in layers for k in REQUIRED)
    except OSError:
        return False


def adopt_file(path: str | Path, dataset_id: str, data_dir: str | Path, *, file_name: str | None = None) -> dict:
    """The layers of an .h5ad on disk into the store, by blocks; the file is left as it is.
    Returns the source record."""
    path = Path(path)
    store = store_path(data_dir, dataset_id)
    src = open_h5ad(path, file_name)
    try:
        write_store(src, store, meta={"source": file_name or path.name})
        n_cells, n_genes = len(src.barcodes), len(src.genes)
        have = list(src.matrices)
    finally:
        src.close()
    return {"kind": "upload", "file": file_name or path.name, "matrices": have, **_totals(store),
            "cells_matched": n_cells, "cells_in_dataset": n_cells, "cells_without": 0,
            "genes_in_file": n_genes, "genes_by_id": 0, "genes_by_name": n_genes}


def adopt_upload(src: Path, out: Path, dataset_id: str, data_dir: str | Path, *, file_name: str) -> dict:
    """A plain .h5ad upload with both layers: the store, and ``out`` = ``src`` without the
    velocity layers (each other entry copied as stored, so a legacy layout stays legacy), with the
    source record in uns. ``src`` is never changed. The caller makes ``out`` read-only."""
    record = adopt_file(src, dataset_id, data_dir, file_name=file_name)
    partial = out.with_name(f".{out.name}.velocity.partial")
    from anndata.io import read_elem, write_elem

    from scview.core import h5_copy

    try:
        with h5py.File(src, "r") as fi, h5py.File(partial, "w") as fo:
            for k, v in fi.attrs.items():
                fo.attrs[k] = v
            for key in fi.keys():
                if key != "layers":
                    # Re-pointed references: a legacy categorical's ``categories`` (core/h5_copy.py).
                    h5_copy.copy(fi[key], fo, key)
            keep = [k for k in fi["layers"].keys() if k not in LAYERS]
            g = fo.create_group("layers")
            for k, v in fi["layers"].attrs.items():
                g.attrs[k] = v
            for k in keep:
                h5_copy.copy(fi["layers"][k], g, k)
            if "uns" not in fo:
                write_elem(fo, "uns", {})
            if UNS_KEY in fo["uns"]:
                del fo["uns"][UNS_KEY]
            write_elem(fo["uns"], UNS_KEY, json.dumps({"source": record}))
            for key in ("obs", "var"):  # the copy must open: fail here, not on the dataset's first read
                if key in fo:
                    read_elem(fo[key])
        os.replace(partial, out)
    except BaseException:
        partial.unlink(missing_ok=True)
        raise
    logger.info("Moved the velocity layers of %s to the store (%d cells)", file_name, record["n_cells"])
    return record


def match_genes(src: Source, var_names, var_ids: list[str] | None) -> tuple[list[str], int, int]:
    """The file's genes under the dataset's names where they match: by gene id when both sides
    have ids, else by name. (names for the store, matched by id, matched by name)."""
    names = [str(n) for n in var_names]
    by_name = set(names)
    out = list(src.genes)
    n_id = n_name = 0
    id_to_name = {}
    if src.gene_ids is not None and var_ids is not None:
        id_to_name = {v.split(".")[0]: n for v, n in zip(var_ids, names)}
    for i, g in enumerate(src.genes):
        if id_to_name and src.gene_ids[i].split(".")[0] in id_to_name:
            out[i] = id_to_name[src.gene_ids[i].split(".")[0]]
            n_id += 1
        elif g in by_name:
            n_name += 1
    return out, n_id, n_name


def attach(adata, dataset_id: str, data_dir: str | Path, paths: list[Path], stored_as: list[str] | None = None,
           label: str | None = None, labels: list[str] | None = None) -> dict[str, Any]:
    """Match the file's cells and genes to the dataset, write the store (replacing an earlier one,
    only once the new one is complete), and record the source in uns. Raises VelocityInputError
    before anything changes."""
    store = store_path(data_dir, dataset_id)
    store.parent.mkdir(parents=True, exist_ok=True)
    tmp = store.with_name(f"{STORE}.attach.h5")
    tmp.unlink(missing_ok=True)
    workdir = store.parent / ".velocity-attach"
    workdir.mkdir(exist_ok=True)
    try:
        with open_source(paths, workdir, label, labels) as src:
            match = match_cells(src.barcodes, adata.obs_names, sample_columns(adata.obs))
            check_majority(match, src.label)
            var_ids = next((adata.var[c].astype(str).tolist() for c in ID_COLUMNS if c in adata.var.columns), None)
            genes, n_id, n_name = match_genes(src, adata.var_names, var_ids)
            if n_id + n_name == 0:
                sample = ", ".join(src.genes[:3])
                raise VelocityInputError(f"None of {src.label}'s {len(src.genes):,} genes (e.g. {sample}) is in this "
                                         "dataset by id or name: check that both use the same genome.")
            write_store(src, tmp, match.file_rows, match.dataset_names, genes, {"source": src.label})
            record = {
                "kind": src.kind, "file": src.label, "files": src.files, "stored_as": stored_as or [],
                "cells_in_file": match.n_file, "cells_matched": match.n_matched, "cells_in_dataset": match.n_dataset,
                "cells_without": match.n_dataset - match.n_matched, "cells_ambiguous": match.n_ambiguous,
                "sample_used": match.used_sample, "genes_in_file": len(src.genes), "genes_by_id": n_id,
                "genes_by_name": n_name, "genes_in_dataset": int(adata.n_vars),
                # The file's cells left out: not in the dataset (filtered out, or another sample's).
                "cells_in_file_unmatched": match.n_file - match.n_matched,
                "cells_other_sample": match.n_other_sample, "samples": match.samples, "sample_key": match.sample_key,
                "samples_without": match.samples_without,
            }
        record.update(_totals(tmp))
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    finally:
        shutil.rmtree(workdir, ignore_errors=True)
    from scview.core import velocity

    velocity.drop_run_files(data_dir, dataset_id)
    os.replace(tmp, store)
    velocity.clear_saved(adata)
    write_source(adata, record)
    return record
