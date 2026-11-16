"""A new dataset made of some of a dataset's cells (``POST /datasets/{id}/subset``).

The cells are chosen by one of: a cell filter (``core/cell_filter.py``, the grammar of
``POST /cells/filter``), groups of an obs column, or the packed bitmask ``/cells/filter``
returns (a lasso or any other selection). The parent is never changed.

The new dataset holds, for those cells:

- the counts, found as a joint embedding finds them (``integration.recover_counts``:
  ``layers['counts']``, ``.raw``, ``X``, each checked to hold counts, else the dataset's
  source files by barcode), after the counts from before an ambient RNA correction when
  there are some (``layers['counts_uncorrected']``, which ``reset_to_counts`` restores first);
- obs, unused categories dropped. Author labels stay as they are; scView's own columns
  (clusters, annotations, scores: ``scview_*``, which a run clears) become ``parent_*`` so
  they can be compared with the subset's own clusters once it is processed;
- the parent's UMAP as ``X_umap_parent`` (kept by runs, as the file's own UMAP is: the
  subset can be looked at before it is processed, and compared after) and the file's own
  ``X_umap_original``. PCA, the neighbour graph and other layouts are not carried: they
  describe the parent's cells and genes, and processing computes them again;
- per-cell modalities in obsm (protein, hashtags, guides, TCR/BCR chains, spatial positions)
  sliced by row; tissue images for the sections that keep cells; the input files kept
  beside a dataset and read by barcode (ATAC peaks and fragments, spliced/unspliced
  counts, cell outlines), linked or copied under the new id.

Results are not carried (markers, DE, enrichment, trajectories, velocity fits, saved
scores): they were computed on the parent. ``plan`` lists what goes and what does not, and
the new dataset's source (``subset_of``) and first recorded step (``subset``) say how it
was made. The replay script reproduces it from the parent (``replay_script._subset``): the
cells' barcodes are kept in ``uns['scview_subset_cells']``.
"""

from __future__ import annotations

import base64
import json
import logging
import os
import shutil
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated, Any

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp
from pydantic import BaseModel, ConfigDict, Field, model_validator

from scview.core import provenance
from scview.core.cell_filter import (
    MAX_NAME,
    MAX_VALUES,
    CategoryCondition,
    CellFilter,
    ConditionGroup,
    FilterError,
    GeneCondition,
    _category_mask,
    evaluate,
)

logger = logging.getLogger(__name__)

STEP = "subset"
SOURCE = "subset"
# Fewer cells than this make no dataset worth processing: the default run's per-sample QC
# bounds need 100 cells a sample, 15 neighbours and 50 PCs on a few dozen cells give
# clusters of noise. (Sub-clustering one group inside a dataset, with the dataset's own
# genes and QC, allows 30.)
MIN_CELLS = 50
# The cells' barcodes, in the parent's order (a string array: h5-native, not JSON).
CELLS_KEY = "scview_subset_cells"
PARENT_PREFIX = "parent_"
# A mask is at most this long (base64 text): 4M cells.
MAX_MASK_CHARS = 700_000
# Memory over the counts read: the rows as read, the float32 CSR copy, the h5ad written
# (gzip buffers, obs). Measured: see tests and the v0.81 notes.
COUNTS_COPIES = 3.0
FIXED_BYTES = 64 * 1024**2

# obsm entries that are results of an analysis of the parent, not data: left behind.
_RESULT_OBSM = {
    "protein_pca": "the WNN protein PCA",
    "protein_dsb": "DSB-normalised protein (run DSB again)",
    "spot_deconv": "spot composition",
    "spot_deconv_cells": "spot composition",
    "atac_gene_activity": "ATAC gene activity",
    "atac_chromvar_z": "chromVAR scores",
    "atac_chromvar_deviations": "chromVAR scores",
}
# obsm entries remade from the carried ones when the subset is opened (protein.adopt_obsm).
_REMADE_OBSM = ("protein_clr",)
# uns entries describing a modality's input (where it came from, its panel), carried as they are.
_INPUT_UNS = ("scview_spatial", "scview_protein", "scview_hashing", "scview_vdj", "scview_crispr",
              "scview_atac", "scview_seq_metrics", "scview_ambient_soup")


class SubsetError(ValueError):
    """A subset that cannot be made: ``status`` is the HTTP code (404 for a column or
    value the dataset does not have, 400 otherwise)."""

    def __init__(self, message: str, status: int = 400) -> None:
        super().__init__(message)
        self.status = status


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Groups(_Strict):
    column: str = Field(min_length=1, max_length=MAX_NAME)
    values: list[Annotated[str, Field(max_length=MAX_NAME)]] = Field(min_length=1, max_length=MAX_VALUES)


class SubsetRequest(_Strict):
    """Exactly one of ``filter``, ``groups`` or ``mask`` chooses the cells."""

    filter: CellFilter | None = None
    groups: Groups | None = None
    # Packed bitmask, base64, as /cells/filter returns it: cell i is bit i % 8 (least
    # significant first) of byte i // 8, over the dataset's cells in their current order.
    mask: str | None = Field(default=None, max_length=MAX_MASK_CHARS)
    name: str | None = Field(default=None, max_length=200)
    # True: say what the subset would hold, make nothing.
    dry_run: bool = False

    @model_validator(mode="after")
    def _one_selector(self) -> SubsetRequest:
        given = [k for k in ("filter", "groups", "mask") if getattr(self, k) is not None]
        if len(given) != 1:
            raise ValueError("Choose the cells with exactly one of filter, groups or mask.")
        return self


# ---------------------------------------------------------------------------
# Choosing the cells
# ---------------------------------------------------------------------------


@dataclass
class Selection:
    keep: np.ndarray  # bool, one per parent cell
    how: dict[str, Any]  # the request's selector, as JSON
    described: str  # in words

    @property
    def n(self) -> int:
        return int(self.keep.sum())


def _values_text(values: list[str], limit: int = 4) -> str:
    shown = ", ".join(values[:limit])
    return shown + (f" and {len(values) - limit} more" if len(values) > limit else "")


def _condition_text(c) -> str:
    if isinstance(c, GeneCondition):
        return f"{c.gene} {c.op} {c.value:g}"
    if isinstance(c, CategoryCondition):
        return f"{c.column} in {{{_values_text(c.values)}}}"
    lo = "" if c.min is None else f"{c.min:g} <= "
    hi = "" if c.max is None else f" <= {c.max:g}"
    return f"{lo}{c.column}{hi}"


def describe_filter(spec: CellFilter) -> str:
    parts = []
    for item in spec.items:
        if isinstance(item, ConditionGroup):
            parts.append("(" + f" {item.op.upper()} ".join(_condition_text(c) for c in item.conditions) + ")")
        else:
            parts.append(_condition_text(item))
    return f" {spec.op.upper()} ".join(parts)


def decode_mask(text: str, n_cells: int) -> np.ndarray:
    """The bool mask a packed base64 bitmask stands for; SubsetError when it is not one
    over ``n_cells`` cells (a selection made before the dataset's cells changed)."""
    try:
        raw = base64.b64decode(text, validate=True)
    except (ValueError, TypeError):
        raise SubsetError("The mask is not base64 text.") from None
    if len(raw) != (n_cells + 7) // 8:
        raise SubsetError(f"The mask covers {len(raw) * 8:,} cells' bits and this dataset has {n_cells:,} "
                          "cells: was the selection made before its cells changed? Select the cells again.")
    bits = np.unpackbits(np.frombuffer(raw, dtype=np.uint8), bitorder="little")
    if bits[n_cells:].any():
        raise SubsetError(f"The mask marks cells past the dataset's {n_cells:,}: select the cells again.")
    return bits[:n_cells].astype(bool)


def select(adaptor, req: SubsetRequest) -> Selection:
    """The parent's cells the request chooses. Reads the dataset (a gene condition reads
    its expression): run it off the event loop. Refuses fewer than ``MIN_CELLS`` and every cell."""
    obs = adaptor.adata.obs
    n_all = int(adaptor.adata.n_obs)
    if req.filter is not None:
        try:
            keep = evaluate(adaptor, req.filter)
        except FilterError as e:
            raise SubsetError(str(e), e.status) from None
        sel = Selection(keep, {"filter": req.filter.model_dump(mode="json")}, describe_filter(req.filter))
    elif req.groups is not None:
        g = req.groups
        if g.column not in obs.columns:
            raise SubsetError(f"Column '{g.column}' not found.", 404)
        series = obs[g.column]
        present = (set(map(str, series.cat.categories)) if isinstance(series.dtype, pd.CategoricalDtype)
                   else set(series.dropna().astype(str).unique()))
        missing = [v for v in g.values if v not in present]
        if missing:
            raise SubsetError(f"Column '{g.column}' has no group '{missing[0]}'.", 404)
        keep = _category_mask(series, list(g.values))
        sel = Selection(keep, {"groups": g.model_dump()}, f"{g.column} in {{{_values_text(list(g.values))}}}")
    else:
        keep = decode_mask(req.mask or "", n_all)
        sel = Selection(keep, {"mask": True}, f"a selection of {int(keep.sum()):,} cells")
    n = sel.n
    if n == 0:
        raise SubsetError("No cells match: nothing to make a dataset of.")
    if n < MIN_CELLS:
        raise SubsetError(f"Only {n:,} cells match; a new dataset needs at least {MIN_CELLS}. Choose more cells.")
    if n == n_all:
        raise SubsetError(f"All {n_all:,} cells match: that is the whole dataset. Choose fewer cells.")
    return sel


def default_name(parent_name: str, sel: Selection) -> str:
    """'<parent> - <what>': the groups, else 'filtered' / 'selected', with the count."""
    groups = sel.how.get("groups")
    if groups:
        what = ", ".join(groups["values"][:3]) + (" +" if len(groups["values"]) > 3 else "")
    elif "filter" in sel.how:
        what = f"{sel.n:,} filtered cells"
    else:
        what = f"{sel.n:,} selected cells"
    from scview.core.dataset_manager import DATASET_NAME_MAX

    name = f"{parent_name} - {what}"
    return name if len(name) <= DATASET_NAME_MAX else name[: DATASET_NAME_MAX - 1] + "…"


# ---------------------------------------------------------------------------
# What goes into the subset
# ---------------------------------------------------------------------------


def _parent_name(column: str, taken: set[str]) -> str:
    base = PARENT_PREFIX + column[len("scview_"):]
    name, i = base, 2
    while name in taken:
        name, i = f"{base}_{i}", i + 1
    return name


def obs_names_map(columns) -> dict[str, str]:
    """Each obs column's name in the subset: scView's own ``scview_*`` columns (cleared by a
    run) as ``parent_*``, except the cell types a per-type % mito filter reads; the rest as
    they are."""
    cols = [str(c) for c in columns]
    taken = {c for c in cols if not c.startswith("scview_")}
    out = {}
    for c in cols:
        if c.startswith("scview_") and c != "scview_mito_type":
            new = _parent_name(c, taken)
            taken.add(new)
            out[c] = new
        else:
            out[c] = c
    return out


def subset_obs(obs: pd.DataFrame, keep: np.ndarray) -> pd.DataFrame:
    """The kept rows, renamed (``obs_names_map``), unused categories dropped."""
    out = obs.loc[keep].copy()
    out = out.rename(columns=obs_names_map(out.columns))
    for c in out.columns:
        if isinstance(out[c].dtype, pd.CategoricalDtype):
            out[c] = out[c].cat.remove_unused_categories()
    out.index = out.index.astype(str)
    return out


def _rows(value, keep: np.ndarray):
    if isinstance(value, pd.DataFrame):
        return value.loc[keep].copy()
    if sp.issparse(value):
        return sp.csr_matrix(value)[keep]
    return np.asarray(value)[keep]


def _umap_key(adata) -> str | None:
    from scview.core.pipeline import file_umap_key

    return file_umap_key(adata.obsm.keys())


@dataclass
class Contents:
    """What the subset holds and leaves out, in words (the reply and the dataset's notes)."""

    carried: list[str] = field(default_factory=list)
    not_carried: list[str] = field(default_factory=list)
    obsm: list[str] = field(default_factory=list)  # obsm keys taken (source names)
    stores: list[tuple[Path, str]] = field(default_factory=list)  # (file, path under the new id's folder)
    renamed: dict[str, str] = field(default_factory=dict)


def _store_files(data_dir: Path, parent_id: str) -> tuple[list[tuple[Path, str]], list[str], list[str]]:
    """Input files kept beside the parent and read by barcode: (file, its subdir/name), what
    they are, and what is beside the parent but not carried."""
    from scview.core import atac_fragments, atac_import, cell_outlines, modality_store, transcripts
    from scview.core import velocity_input

    files: list[tuple[Path, str]] = []
    said: list[str] = []
    left: list[str] = []
    mod = modality_store.store_dir(data_dir, parent_id)
    peaks = mod / f"{atac_import.STORE}.h5"
    if peaks.is_file():
        files.append((peaks, f"{modality_store.SUBDIR}/{peaks.name}"))
        said.append("ATAC peak counts (read by barcode)")
    frag = [mod / n for n in (atac_fragments.FRAGMENTS_FILE, atac_fragments.INDEX_FILE, atac_fragments.STATE_FILE)]
    if frag[0].is_file():
        files += [(p, f"{modality_store.SUBDIR}/{p.name}") for p in frag if p.is_file()]
        said.append("ATAC fragments file")
    vel = mod / f"{velocity_input.STORE}.h5"
    if vel.is_file():
        files.append((vel, f"{modality_store.SUBDIR}/{vel.name}"))
        said.append("spliced/unspliced counts (velocity input)")
    kinds = cell_outlines.kinds_of(data_dir, parent_id)
    for k in kinds:
        p = cell_outlines.store_path(data_dir, parent_id, k)
        files.append((p, f"{cell_outlines.SUBDIR}/{p.name}"))
    if kinds:
        said.append(f"{' and '.join(kinds)} outlines")
    if transcripts.has_store(data_dir, parent_id):
        left.append("Xenium transcripts (they cover the whole section, not the chosen cells)")
    return files, said, left


def contents(adata, data_dir: Path | None, parent_id: str) -> Contents:
    """What a subset of ``adata`` holds and leaves out (any cells: the same for each)."""
    from scview.core import ambient, protein, spatial

    out = Contents()
    out.renamed = {c: n for c, n in obs_names_map(adata.obs.columns).items() if c != n}
    n_cols = len(adata.obs.columns)
    out.carried.append(f"{n_cols} obs column{'s' if n_cols != 1 else ''}"
                       + (f" ({len(out.renamed)} of scView's renamed parent_*, e.g. "
                          f"{next(iter(out.renamed.values()))})" if out.renamed else ""))
    if ambient.ORIGINAL_LAYER in adata.layers:
        out.carried.append("the counts from before the ambient RNA correction")
    umap = _umap_key(adata)
    if umap is not None:
        out.carried.append("the parent's UMAP, as 'UMAP (parent)'")
    from scview.core.pipeline import ORIGINAL_UMAP_KEY

    if ORIGINAL_UMAP_KEY in adata.obsm:
        out.carried.append("the file's own UMAP ('UMAP (original)')")
    left_x = sorted(k for k in adata.obsm if str(k).startswith("X_") and k not in (umap, ORIGINAL_UMAP_KEY))
    graphs = len(adata.obsp) > 0
    if left_x or graphs:
        named = ", ".join(left_x[:5]) + (f" and {len(left_x) - 5} more" if len(left_x) > 5 else "")
        out.not_carried.append("PCA, the neighbour graph and other layouts"
                               + (f" ({named})" if named else "") + ": processing computes them again")
    for k in adata.obsm:
        k = str(k)
        if k.startswith("X_"):
            continue
        if k in _REMADE_OBSM:
            continue
        if k in _RESULT_OBSM:
            out.not_carried.append(_RESULT_OBSM[k])
            continue
        out.obsm.append(k)
    if protein.has_protein(adata):
        n = adata.obsm[protein.COUNTS_KEY].shape[1]
        out.carried.append(f"protein counts ({n} antibodies)")
    if "hto_counts" in adata.obsm:
        out.carried.append("hashtag counts")
    if "guide_counts" in adata.obsm:
        out.carried.append("guide counts")
    if "vdj_chains" in adata.obsm:
        out.carried.append("TCR/BCR chains")
    if spatial.has_spatial(adata):
        out.carried.append("spatial positions and the tissue images of the sections kept")
    if data_dir is not None:
        out.stores, said, left = _store_files(Path(data_dir), parent_id)
        out.carried += said
        out.not_carried += left
    out.not_carried = list(dict.fromkeys(out.not_carried))
    out.not_carried.append("stored results (markers, DE, enrichment, trajectories, velocity, scores): "
                           "run them on the subset")
    return out


def _spatial_uns(adata, obs: pd.DataFrame) -> dict | None:
    """uns['spatial'] with the sections that keep cells, their shown images only."""
    from scview.core import spatial
    from scview.core.integration import _shown_images_only

    libs = adata.uns.get(spatial.KEY)
    if not isinstance(libs, dict) and not hasattr(libs, "items"):
        return None
    kept = None
    if spatial.LIBRARY_COLUMN in obs.columns:
        kept = set(map(str, obs[spatial.LIBRARY_COLUMN].astype(str).unique()))
    out = {}
    for lib, entry in dict(libs).items():
        if hasattr(entry, "items") and kept is not None and str(lib) not in kept:
            continue
        out[lib] = entry
    return _shown_images_only(out)


def _velocity_uns(adata) -> str | None:
    """The velocity entry's ``source`` only: a run's result goes (its stores are not carried)."""
    from scview.core import velocity_input

    raw = adata.uns.get(velocity_input.UNS_KEY)
    if raw is None:
        return None
    try:
        data = json.loads(raw) if isinstance(raw, str) else dict(raw)
    except (TypeError, ValueError):
        return None
    return json.dumps({"source": data["source"]}) if isinstance(data, dict) and data.get("source") else None


def counts_rows(adata, keep: np.ndarray, fallback_paths: list[str]):
    """The chosen cells' counts as ``reset_to_counts`` then ``integration.recover_counts``
    find them: the counts before an ambient RNA correction first, else the first of
    layers['counts'], .raw and X holding counts, else the source files by barcode. Only the
    chosen rows are read from an on-disk matrix."""
    from scview.core import ambient
    from scview.core.integration import RecoveredCounts, recover_counts, rows_csr

    from scview.core.integration import IntegrationError

    if ambient.ORIGINAL_LAYER in adata.layers:
        return RecoveredCounts(rows_csr(adata.layers[ambient.ORIGINAL_LAYER], keep), adata.var.copy(),
                               f"layers['{ambient.ORIGINAL_LAYER}'] (before the ambient RNA correction)")
    try:
        return recover_counts(adata, fallback_paths, keep=keep)
    except IntegrationError:
        raise SubsetError("This dataset holds no raw counts (checked layers['counts'], .raw and X, and its "
                          "source files), and a subset is processed from counts.", 422) from None


def build(adata, keep: np.ndarray, fallback_paths: list[str], info: Contents) -> tuple[ad.AnnData, str]:
    """The subset's AnnData (counts in X) and where its counts came from."""
    from scview.core import spatial

    counts = counts_rows(adata, keep, fallback_paths)
    var = counts.var
    var.index = var.index.astype(str)
    obs = subset_obs(adata.obs, keep)
    sub = ad.AnnData(X=counts.X, obs=obs, var=var)
    umap = _umap_key(adata)
    from scview.core.pipeline import ORIGINAL_UMAP_KEY, PARENT_UMAP_KEY

    if umap is not None:
        xy = np.asarray(adata.obsm[umap])
        if xy.ndim == 2 and xy.shape[1] >= 2:
            sub.obsm[PARENT_UMAP_KEY] = np.array(xy[keep, :2], dtype=np.float32)
    if ORIGINAL_UMAP_KEY in adata.obsm:
        sub.obsm[ORIGINAL_UMAP_KEY] = np.asarray(adata.obsm[ORIGINAL_UMAP_KEY])[keep]
    for k in info.obsm:
        value = _rows(adata.obsm[k], keep)
        if isinstance(value, pd.DataFrame):
            value.index = obs.index
        sub.obsm[k] = value
    uns: dict[str, Any] = {}
    for k in _INPUT_UNS:
        if k in adata.uns:
            uns[k] = adata.uns[k]
    if spatial.KEY in adata.uns:
        libs = _spatial_uns(adata, obs)
        if libs:
            uns[spatial.KEY] = libs
    vel = _velocity_uns(adata)
    if vel is not None:
        from scview.core import velocity_input

        uns[velocity_input.UNS_KEY] = vel
    uns[CELLS_KEY] = np.asarray(obs.index, dtype=object)
    sub.uns.update(uns)
    return sub, counts.source


def record(sub: ad.AnnData, *, parent: dict[str, Any], sel: Selection, n_parent: int,
           counts_source: str, info: Contents) -> None:
    """The source block (origin subset, the parent named) and the ``subset`` step."""
    provenance.init_source(sub, origin=SOURCE, original_filename=str(parent.get("filename") or "input.h5ad"),
                           fmt="subset", overwrite=True)
    data = provenance.read_provenance(sub)
    data["source"]["subset_of"] = subset_of(parent, sel, n_parent)
    data["history"] = []
    provenance._write(sub, data)
    provenance.record_step(
        sub, step=STEP, tool="scview.subset",
        params={"parent_id": parent["id"], "parent_name": parent.get("name"), **sel.how,
                "described": sel.described, "n_cells": sel.n, "n_parent_cells": n_parent,
                "counts_source": counts_source, "renamed": info.renamed,
                "parent_source": parent.get("source") or {}},
        note=f"{sel.n:,} of {n_parent:,} cells of '{parent.get('name')}': {sel.described}",
    )


def subset_of(parent: dict[str, Any], sel: Selection, n_parent: int) -> dict[str, Any]:
    return {"id": parent["id"], "name": parent.get("name"), "how": sel.how, "described": sel.described,
            "n_cells": sel.n, "n_parent_cells": n_parent}


def _link_or_copy(src: Path, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(src, dest)  # the same bytes: read-only inputs, a delete of either keeps the other
    except OSError:
        shutil.copy2(src, dest)


def persist(data_dir: Path, dataset_id: str, name: str, sub: ad.AnnData, *, parent: dict[str, Any],
            sel: Selection, n_parent: int, info: Contents) -> Path:
    """Write the subset (read-only, ``ingested/{id}/``), its input files and the metadata
    sidecar the picker reads. A failure part-way removes what was written."""
    from scview.core.ingestion.session import _make_readonly, _plain_strings, _safe_name

    folders = [data_dir / d / dataset_id for d in ("ingested", "uploads")]
    folders += list(dict.fromkeys(data_dir / rel.split("/")[0] / dataset_id for _, rel in info.stores))
    try:
        ing = data_dir / "ingested" / dataset_id
        ing.mkdir(parents=True, exist_ok=True)
        path = ing / f"{_safe_name(name)}.h5ad"
        _plain_strings(sub)
        sub.write_h5ad(path, compression="gzip")
        _make_readonly(path)
        for src, rel in info.stores:
            sub_dir, fname = rel.split("/", 1)
            _link_or_copy(src, data_dir / sub_dir / dataset_id / fname)
        up = data_dir / "uploads" / dataset_id
        up.mkdir(parents=True, exist_ok=True)
        meta = {
            "id": dataset_id,
            "name": name,
            "filename": path.name,
            "extension": ".h5ad",
            "size_bytes": path.stat().st_size,
            "status": "ready",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "n_cells": int(sub.n_obs),
            "n_genes": int(sub.n_vars),
            "available_embeddings": [],
            "obs_columns": [],
            "source": SOURCE,
            "subset_of": {**subset_of(parent, sel, n_parent), "carried": info.carried,
                          "not_carried": info.not_carried},
        }
        with open(up / "metadata.json", "w") as fh:
            json.dump(meta, fh, indent=2)
        return path
    except BaseException:
        for d in folders:
            shutil.rmtree(d, ignore_errors=True)
        raise


# ---------------------------------------------------------------------------
# Memory
# ---------------------------------------------------------------------------


def read_need(h5ad_path: str | None, n_keep: int, n_all: int) -> int:
    """Peak memory of making the subset, from the parent file's header: the chosen rows of
    its counts matrix (``preflight._lt_counts_bytes``, float32 CSR) ``COUNTS_COPIES`` times,
    plus a fixed share. 0 when the header cannot be read (never blocks the work)."""
    from scview.core import preflight

    if not h5ad_path or n_all <= 0:
        return 0
    try:
        h = preflight.read_header(h5ad_path, "h5ad")
        counts = preflight._lt_counts_bytes(h)
    except Exception as e:  # noqa: BLE001
        logger.warning("No subset estimate for %s: %s", h5ad_path, e)
        return 0
    return int(COUNTS_COPIES * counts * n_keep / n_all + FIXED_BYTES)
