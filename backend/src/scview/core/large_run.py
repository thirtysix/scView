"""The pipeline on a dataset whose matrices stay on disk (large-dataset mode).

A run is made in memory (``pipeline.run_pipeline``) unless ``PipelineParams.chunked``
says otherwise: "on" always runs it here, "off" never, and "auto" (the default) runs it
here when the in-memory run would not fit the memory budget (``choose_mode``: its
estimated need, a backed dataset's read-in included, with ``LOAD_MARGIN`` and the working
reserve, against what is free with the other open datasets closed: the rule
``DatasetManager.free_memory_for`` uses). A dataset already in this layout always runs
here, "off" or not: it is never read into memory. ``core/processing.py`` makes that choice
before the run and, for this mode, never reads the dataset into memory. Work that needs
only obs, obsm, obsp and uns (PAGA, pseudotime) gets the dataset without its matrices
(``read_light``).

The run has the in-memory pipeline's step names, parameters and stored results. It keeps
a copy of the dataset without matrices (obs, var, obsm, obsp, varm, uns: "light") and a
model of the matrices the in-memory run would hold (X, ``layers['counts']``, ``.raw``),
each a lazy row-block matrix on disk (``core/chunked.py``):

    step                    here                                         equal to in memory
    reset_to_counts         X := the counts layer / .raw / counts sidecar  same
    qc_metrics              chunked.qc_metrics                           bitwise
    doublet_detection       doublets.run_doublets on the stored counts,  bitwise
                            one capture channel's rows at a time
    filtering               the step's own masks from obs, gene counts   same cells, genes
                            by blocks, lazy cuts
    normalization, log      lazy, written once (below)                   bitwise
    highly_variable_genes   chunked.highly_variable_genes / pearson_hvg  same genes
    scaling                 the same plan, statistics by blocks          float precision
    pca                     chunked.pca_scaled / pca_pearson             float precision
    neighbours .. UMAP      the pipeline's own steps on light            same code
    marker_genes            sparse_markers from the gene store           bitwise (given labels)
    enrichment              the pipeline's step (reads the markers)      same code
    cell_cycle, stress      large_scores, from the gene store + blocks   bitwise
    cell_type_annotation    CellTypist by blocks + its majority vote;    see large_scores
                            LLM / marker score on the stored markers

The log-normalised X is written once, at the first step that reads it (variable genes,
usually), in one pass that also writes the filtered counts and builds the gene store
(``core/gene_store.py``); later steps read those files. The result is the layout of
``core/large_layout.py``: the derived h5ad (X log-normalised CSR float32, no .raw, no
layers) and its ``.counts.h5`` and ``.genes.h5`` sidecars, all written into a staging
folder beside it and moved into place only at the end (sidecars first, then the h5ad that
names them), so a cancelled, stopped or failed run leaves the dataset as it was.
"""

from __future__ import annotations

import contextlib
import contextvars
import logging
import os
import queue
import secrets
import shutil
import threading
import time
from collections.abc import Callable, Generator, Iterator
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import anndata as ad
import dask
import h5py
import numpy as np
import pandas as pd
import scipy.sparse as sp

from scview.core import (
    chunked,
    doublets,
    gene_store,
    large_layout,
    large_scores,
    normalization,
    provenance,
    qc_filter,
    scaling,
    sparse_markers,
    step_failures,
)
from scview.core.pipeline import (
    ALL_STEPS,
    HVG_FLAVORS,
    SAVED_STATUSES,
    PipelineParams,
    PipelineResult,
    StepSkipped,
    _cancelled_during,
    _end_status,
    _finalize_current,
    _halted,
    _new_outcome,
    _record_errors,
    _record_pipeline_step,
    _REGRESS_PEARSON,
    _skip_if_blocked,
    _step_failed,
    _STEP_RUNNERS,
    _unknown_requested,
    _var_from_hvg_frame,
    check_chunked,
    check_umap_3d,
    clear_processing,
    filter_judgement,
    large_backend,
    looks_like_counts,
    regress_covariates,
    resolve_umap_3d,
    run_pipeline,
    run_pipeline_streamed,
    skipped_event,
    subset_lean,
)
from scview.core.run_watchdog import RunCancelled, RunControl, cancelled_reason, default_before_step
from scview.core.uns_keys import BATCH_MIXING_KEY, repair_slash_keys

# The h5ad is created with HDF5's free-space manager kept in the file, so the in-place
# edits of obs and uns the viewer makes reuse the space of what they replace.
H5AD_CREATE = {"fs_strategy": "fsm", "fs_persist": True}

logger = logging.getLogger(__name__)

GB = 1024**3
STAGE_PREFIX = ".large-run-"
# Steps that run the pipeline's own function on the matrix-free copy, as they are...
_LIGHT_STEPS = ("uncorrected_embedding", "batch_correction", "neighbors", "clustering",
                "embeddings", "enrichment")
# ...and with the matrices read by this run (pipeline.large_backend).
_BACKED_STEPS = ("marker_genes", "cell_cycle", "dissociation_stress", "cell_type_annotation")
_SUBSTEP_RUNNERS = ("marker_genes", "enrichment")  # take the substep callback
# Progress events of a pass over the blocks: at most one per this many seconds.
PROGRESS_SECONDS = 1.0


class LargeRunError(ValueError):
    """A run large-dataset mode cannot make (no raw counts, an X that would not end up
    log-normalised, a stale sidecar)."""


# Steps this mode does not run: (what, why), refused before the run starts (a 422).
REFUSED_STEPS = {
    "ambient_correction": (
        "Ambient RNA correction",
        "it rewrites every cell's counts and keeps the originals beside them, a second "
        "copy of the counts this mode does not hold. Run it on a smaller dataset, or with "
        "large-dataset mode off when the memory allows"),
}


def refused(steps: list[str]) -> str | None:
    """Why a large-dataset run of ``steps`` is refused (the first such step), or None."""
    from scview.core.large_mode import LargeDatasetError

    for step in steps:
        if step in REFUSED_STEPS:
            return str(LargeDatasetError(*REFUSED_STEPS[step]))
    return None


# ---------------------------------------------------------------------------
# Which mode
# ---------------------------------------------------------------------------


@dataclass
class ModeChoice:
    """How a run works: ``chunked`` (large-dataset mode) or in memory, and why."""

    chunked: bool
    requested: str = "auto"
    reason: str = ""
    need_bytes: int | None = None  # what the in-memory run needs beyond what is held now
    run_bytes: int | None = None  # of that, the run itself (beyond the dataset in memory)
    free_bytes: int | None = None  # memory free with the other open datasets closed
    limit_bytes: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {"chunked": self.chunked, "requested": self.requested, "reason": self.reason,
                "need_bytes": self.need_bytes, "free_bytes": self.free_bytes,
                "limit_bytes": self.limit_bytes}


def large_info(h5ad_path) -> dict | None:
    """``uns["scview_large"]`` of an h5ad on disk (read alone), or None: not a large dataset."""
    from anndata.io import read_elem

    if not h5ad_path:
        return None
    try:
        with h5py.File(h5ad_path, "r") as f:
            key = f"uns/{large_layout.KEY}"
            return read_elem(f[key]) if key in f else None
    except OSError:
        return None


def is_legacy_layout(path: str | Path) -> bool:
    """Whether the h5ad at ``path`` was written by anndata < 0.7 (obs as a compound
    dataset, raw as ``raw.X``), as Kang 2018 was published."""
    with h5py.File(str(path), "r") as f:
        return isinstance(f.get("obs"), h5py.Dataset)


def read_light(path: str | Path) -> ad.AnnData:
    """The dataset at ``path`` without its matrices (obs, var, obsm, obsp, varm, varp,
    uns), as an in-memory open leaves them: for work that needs only those (PAGA,
    pseudotime) on a large dataset, and for the large-dataset run itself."""
    from anndata.io import read_elem

    from scview.core import protein

    with h5py.File(str(path), "r") as f:
        def get(key, default=None):
            return read_elem(f[key]) if key in f else default

        light = ad.AnnData(obs=get("obs"), var=get("var"), obsm=get("obsm", {}),
                           obsp=get("obsp", {}), varm=get("varm", {}), varp=get("varp", {}),
                           uns=get("uns", {}))
    if repair_slash_keys(light.uns):
        logger.info("Re-keyed cached results saved under '/' keys")
    protein.adopt_obsm(light)  # as an in-memory open does
    if light.obs.index.name is not None and light.obs.index.name in light.obs.columns:
        light.obs.index.name = None  # as run_pipeline does
    return light


def is_chunked(control: RunControl | None) -> bool:
    """Whether ``core/processing.py`` chose large-dataset mode for this run."""
    return bool(getattr(getattr(control, "mode", None), "chunked", False))


def in_memory_need(adaptor, steps: list[str], params) -> tuple[int, int]:
    """(memory the in-memory run needs beyond what the dataset holds now, the run's own
    part of it): reading a backed dataset into memory (``working_copy.copy_bytes``), then
    the run's peak over that (``preflight.run_need``; a backed dataset's doublet channels
    from its file, never by reading its matrix)."""
    from scview.core import preflight, working_copy

    adata = adaptor.adata
    copy = working_copy.copy_bytes(adaptor)
    if not getattr(adata, "isbacked", False):
        run = preflight.run_need(adata, steps, params)
        return copy + run, run
    plan = (preflight._file_doublet_plan(adaptor.h5ad_path, params)
            if "doublet_detection" in steps else None)
    est = preflight.estimate_run(preflight.in_memory_header(adata), steps=steps, params=params,
                                 warm=preflight.process_warm(), doublets=plan)
    run = max(0, int(est.peak_bytes - est.start_bytes))
    return copy + run, run


def choose_mode(dm, dataset_id: str, adaptor, steps: list[str], params) -> ModeChoice:
    """Large-dataset mode or in memory for a run of ``steps`` (see the module docstring).
    Raises ``ValueError`` for a ``chunked`` value that is not auto / on / off."""
    from scview.core import memory_budget, preflight
    from scview.core.dataset_manager import LOAD_MARGIN, working_reserve

    requested = check_chunked(getattr(params, "chunked", "auto"))
    if large_info(getattr(adaptor, "h5ad_path", None)) is not None:
        return ModeChoice(True, requested, "This dataset is in large-dataset layout (its counts "
                          "and gene-major copy are on disk beside it), so it runs in "
                          "large-dataset mode: it is never read into memory.")
    if requested == "on":
        return ModeChoice(True, requested, "Large-dataset mode was chosen for this run.")
    if requested == "off":
        return ModeChoice(False, requested, "In memory was chosen for this run.")
    budget = memory_budget.read_budget()
    if budget.limit_bytes is None or budget.used_bytes is None:
        return ModeChoice(False, requested, "No memory limit is known: the run works in memory.")
    need, run = in_memory_need(adaptor, steps, params)
    others = preflight.cached_others(dm, dataset_id).bytes
    free = max(0, budget.limit_bytes - budget.used_bytes) + others
    target = need * LOAD_MARGIN + working_reserve(budget.limit_bytes)
    fits = target <= free
    reason = (f"In memory the run needs about {need / GB:.1f} GB more at its peak; scView has "
              f"{free / GB:.1f} GB free (with the other open datasets closed) and keeps "
              f"{working_reserve(budget.limit_bytes) / GB:.1f} GB in reserve, so ")
    reason += ("it works in memory." if fits else
               "it runs in large-dataset mode: the matrices stay on disk, read in blocks.")
    return ModeChoice(not fits, requested, reason, need, run, free, budget.limit_bytes)


# ---------------------------------------------------------------------------
# Matrices the in-memory run would hold, read lazily
# ---------------------------------------------------------------------------


@dataclass
class _Mat:
    """One matrix (cells x ``names``) as a lazy dask array of row blocks."""

    X: Any
    names: pd.Index
    kind: str = "unknown"  # counts | normalized | lognorm | logged | scaled | unknown
    stored: tuple[str, str] | None = None  # (file, key) while X is that stored matrix as is
    index_dtype: np.dtype = field(default_factory=lambda: np.dtype(np.int32))
    dense: bool = False  # held dense in memory (the dense counts it came from)
    var: pd.DataFrame | None = None  # .raw's own var (and varm)
    varm: dict | None = None

    def sample(self):
        """The first rows, as ``pipeline.looks_like_counts`` judges an on-disk matrix."""
        n = min(self.X.shape[0], 2000)
        return self.X[:n].compute(scheduler="sync")


def _judge(m: _Mat) -> str:
    """What ``m`` holds: its known kind, else counts / scaled / lognorm from its first
    rows (counts: non-negative integers; scaled: values below 0)."""
    if m.kind != "unknown":
        return m.kind
    s = m.sample()
    if looks_like_counts(s):
        m.kind = "counts"
    elif scaling.holds_scaled(s):
        m.kind = "scaled"
    else:
        m.kind = "lognorm"
    return m.kind


def _holds_counts(m: _Mat | None) -> bool:
    return m is not None and _judge(m) == "counts"


def _index_dtype(dense: bool, nnz: int, n_vars: int) -> np.dtype:
    """The index dtype a matrix read into memory has (what ``n_genes_by_counts`` takes):
    scipy keeps int64 indices only when the values need them (anndata writes a streamed
    CSR with int64 indices); a dense matrix's count is int64."""
    if dense or max(nnz, n_vars) > np.iinfo(np.int32).max:
        return np.dtype(np.int64)
    return np.dtype(np.int32)


def _f32_csr(b) -> sp.csr_matrix:
    """A block of X as written: CSR float32 in canonical order (each row's genes sorted,
    duplicates summed), as the in-memory scaling step leaves X (``scaling.gene_stats``
    sorts it in place): the gene scores add each cell's values in this order."""
    m = sp.csr_matrix(b) if sp.issparse(b) else sp.csr_matrix(np.asarray(b))
    m = m if m.dtype == np.float32 else m.astype(np.float32)
    if not m.has_canonical_format:
        m = m.copy()  # the block may share its arrays with the matrix it came from
        m.sum_duplicates()
    return m


def _csr(b) -> sp.csr_matrix:
    return sp.csr_matrix(b) if sp.issparse(b) else sp.csr_matrix(np.asarray(b))


class _DaskRows:
    """Row reads of a lazy matrix for Scrublet (``scrublet_lean.channel_rows``): ``M[a:b]``
    or ``M[idx]`` computed, CSR (or dense when the matrix is), and ``to_memory``."""

    def __init__(self, X):
        self.X = X
        self.shape = tuple(int(s) for s in X.shape)
        self.dtype = X.dtype
        self._sparse = sp.issparse(X._meta)

    def __getitem__(self, idx):
        with chunked.scheduler():
            got = self.X[idx].compute()
        return _csr(got) if self._sparse else np.asarray(got)

    def to_memory(self):
        with chunked.scheduler():
            return _csr(self.X.compute())


def _remove_stale_stages(folder: Path) -> None:
    """Staging folders a run left behind when its process died (its pid is gone)."""
    if not folder.is_dir():
        return
    for entry in folder.iterdir():
        if not entry.name.startswith(STAGE_PREFIX) or not entry.is_dir():
            continue
        try:
            pid = int(entry.name[len(STAGE_PREFIX):].split("-")[0])
        except ValueError:
            continue
        if pid != os.getpid() and not gene_store._pid_alive(pid):
            shutil.rmtree(entry, ignore_errors=True)


# ---------------------------------------------------------------------------
# The run
# ---------------------------------------------------------------------------


class LargeRun:
    """The state of one large-dataset run: the matrix-free copy, the matrices' model,
    the staging folder. Steps are its methods (``step``)."""

    def __init__(self, path: str | Path, output_path: str | Path | None, params: PipelineParams,
                 control: RunControl | None = None) -> None:
        self.path = Path(path)
        self.output = Path(output_path) if output_path else None
        self.params = params
        self.control = control
        self.light: ad.AnnData | None = None
        self.x: _Mat | None = None
        self.layer: _Mat | None = None
        self.raw: _Mat | None = None
        self.large_in = False
        self.dense = False
        self.rows = 0
        self.nnz = 0
        self._input_x: _Mat | None = None  # a large input's own X and counts, unchanged
        self._input_counts: _Mat | None = None
        self._input_sidecars: tuple[Path, Path] | None = None  # (counts, genes) of a large input
        self._shape0 = (0, 0)  # cells x genes when opened
        self._stage: Path | None = None
        self._written = False  # X, counts and the gene store as the result will hold them
        self._x_file: Path | None = None  # staged h5ad holding the new X (None: the input's)
        self._counts_file: Path | None = None  # staged counts sidecar (None: the input's)
        self._store_file: Path | None = None  # the gene store of X (staged or the input's)
        self._store: gene_store.GeneStore | None = None
        self._means: np.ndarray | None = None
        self._post: Callable[[dict], None] | None = None  # substep events of the step running
        self._label = ""
        self._last_post = 0.0
        self.saved = False

    # -- opening ------------------------------------------------------------

    def open(self) -> None:
        from anndata.io import read_elem

        if is_legacy_layout(self.path):
            self.path = self._modernised()
        p = str(self.path)
        light = read_light(p)
        with h5py.File(p, "r") as f:
            def get(key, default=None):
                return read_elem(f[key]) if key in f else default

            info = {}
            for key in ("X", "layers/counts", "raw/X"):
                if key not in f:
                    continue
                elem = f[key]
                dense = isinstance(elem, h5py.Dataset)
                shape = chunked._elem_shape(elem)
                nnz = shape[0] * shape[1] if dense else int(elem["indptr"][-1])
                info[key] = (dense, shape, nnz, _index_dtype(dense, nnz, shape[1]))
            raw_var = get("raw/var") if "raw/X" in info else None
            raw_varm = get("raw/varm", {}) if "raw/X" in info else None
        self.light = light
        self.large_in = large_layout.is_large(light.uns)
        self._shape0 = (light.n_obs, light.n_vars)
        if "X" not in info:
            raise LargeRunError(f"{self.path.name} has no matrix X.")
        n = light.n_obs
        per_row = max(nnz / max(n, 1) for _, _, nnz, _ in info.values())
        widest = max(shape[1] for _, shape, _, _ in info.values())
        any_dense = any(d for d, _, _, _ in info.values())
        self.rows = chunked.rows_per_block(widest, nnz_per_row=per_row, dense=any_dense,
                                           n_hvg=int(self.params.n_top_genes))
        self.nnz = info["X"][2]

        def mat(key, names, kind="unknown", file=p, **kw) -> _Mat:
            dense, _, _, idx = info[key]
            return _Mat(chunked.read_lazy(file, key, chunk_rows=self.rows), pd.Index(names), kind,
                        (file, key), idx, dense, **kw)

        if self.large_in:
            self.x = mat("X", light.var_names, "lognorm")
            counts = large_layout.counts_path(self.path, light.uns)
            self._input_sidecars = (counts, large_layout.genes_path(self.path, light.uns))
            why = large_layout.counts_mismatch(counts, light.obs_names, light.var_names)
            if why is not None:
                raise LargeRunError(f"This large dataset's counts can't be used ({why}): "
                                    "upload the data again to reprocess it.")
            with h5py.File(counts, "r") as f:
                cnnz = int(f[large_layout.COUNTS_KEY]["indptr"][-1])
                cidx = _index_dtype(False, cnnz, light.n_vars)
                self.nnz = max(self.nnz, cnnz)
            self.layer = _Mat(chunked.read_lazy(counts, large_layout.COUNTS_KEY,
                                                chunk_rows=self.rows),
                              pd.Index(light.var_names), "counts",
                              (str(counts), large_layout.COUNTS_KEY), cidx)
            self._input_x, self._input_counts = self.x, self.layer
        else:
            self.x = mat("X", light.var_names)
            if "layers/counts" in info:
                self.layer = mat("layers/counts", light.var_names)
            if "raw/X" in info:
                self.raw = mat("raw/X", raw_var.index, var=raw_var, varm=raw_varm)
        self.dense = self.x.dense

    def _modernised(self) -> Path:
        """A copy of a pre-0.7 anndata file in today's layout, staged: the blocks the run
        reads (``X``, ``raw/X``, ``obs`` as a group) do not exist in the old one. Such files
        date from before 2020 and are small, so it is read whole."""
        target = self.stage() / "input.h5ad"
        ad.read_h5ad(str(self.path)).write_h5ad(target)
        logger.info("%s is in the anndata < 0.7 layout: rewrote it for the run", self.path.name)
        return target

    # -- staging ------------------------------------------------------------

    def stage(self) -> Path:
        if self._stage is None:
            folder = (self.output or self.path).parent
            folder.mkdir(parents=True, exist_ok=True)
            _remove_stale_stages(folder)
            self._stage = folder / f"{STAGE_PREFIX}{os.getpid()}-{secrets.token_hex(4)}"
            self._stage.mkdir()
        return self._stage

    def cleanup(self) -> None:
        """Close the gene store and remove what the run staged (moved files are gone)."""
        self._close_store()
        if self._stage is not None:
            shutil.rmtree(self._stage, ignore_errors=True)
            self._stage = None

    def _close_store(self) -> None:
        if self._store is not None:
            self._store.close()
            self._store = None

    # -- progress and cancel -------------------------------------------------

    def tick(self, done: int, total: int) -> None:
        """Between blocks: stop a cancelled run, send a progress event now and then."""
        c = self.control
        if c is not None and c.cancelled:
            raise RunCancelled()
        if self._post is None or total <= 0:
            return
        now = time.monotonic()
        if done < total and now - self._last_post < PROGRESS_SECONDS:
            return
        self._last_post = now
        self._post({"message": f"{self._label}: {done:,} of {total:,}", "current": int(done),
                    "total": int(total)})

    def _store_tick(self, message: str, done: int, total: int) -> None:
        self._label = message
        self.tick(done, total)

    # -- the in-memory slots ----------------------------------------------

    def _cut(self, obs=None, var=None) -> None:
        """Cut X, the counts layer and (rows only) .raw, keeping which slots share a matrix."""
        done: dict[int, _Mat] = {}

        def cut(m: _Mat | None, cols: bool) -> _Mat | None:
            if m is None:
                return None
            if id(m) in done:
                return done[id(m)]
            v = var if cols else None
            if obs is None and v is None:
                return m
            new = replace(m, X=chunked.subset(m.X, obs=obs, var=v), stored=None,
                          names=m.names if v is None else m.names[np.asarray(v, dtype=bool)])
            done[id(m)] = new
            return new

        self.x = cut(self.x, True)
        self.layer = cut(self.layer, True)
        if self.raw is not None and obs is not None:
            self.raw = replace(self.raw, X=chunked.subset(self.raw.X, obs=obs), stored=None)

    def _keep_raw_counts(self) -> None:
        """``pipeline._keep_raw_counts``: counts held only in .raw become the layer."""
        if self.layer is not None or self.raw is None or not _holds_counts(self.raw):
            return
        names = self.light.var_names
        if len(names.difference(self.raw.names)):
            logger.warning("adata.raw lacks some genes in X; raw counts not copied to layers")
            return
        X = chunked._take_columns(self.raw.X, self.raw.names.get_indexer(names))
        self.layer = replace(self.raw, X=X, names=pd.Index(names), stored=None, var=None,
                             varm=None)

    def find_counts(self) -> tuple[_Mat, str]:
        """``normalization.find_counts``: the layer, .raw (every gene of X), else X."""
        if _holds_counts(self.layer):
            return self.layer, "layers['counts']"
        names = self.light.var_names
        if self.raw is not None and not len(names.difference(self.raw.names)) \
                and _holds_counts(self.raw):
            X = chunked._take_columns(self.raw.X, self.raw.names.get_indexer(names))
            return replace(self.raw, X=X, names=pd.Index(names), stored=None), ".raw"
        if _holds_counts(self.x):
            return self.x, "X"
        raise ValueError("Pearson residuals need raw counts (layers['counts'], .raw or X), and "
                         "this dataset has none: use the shifted log instead.")

    # -- the one write of X, the counts and the gene store ------------------

    def lognorm(self):
        """X, log-normalised, as the result holds it (written first if need be)."""
        self.write_matrices()
        return self.x.X

    def write_matrices(self) -> None:
        """Write X (CSR float32), the filtered counts and the gene store of X in one pass,
        unless a large input's own files still hold them; later steps read them back."""
        if self._written:
            return
        kind = _judge(self.x)
        if kind == "counts":
            raise LargeRunError(
                "Large-dataset mode stores X log-normalised, and this run leaves raw counts "
                "in X: include normalization and log transform (or reprocess from scratch).")
        if kind == "normalized":
            raise LargeRunError("Large-dataset mode stores X log-normalised: include the log "
                                "transform after normalization.")
        if kind == "scaled":
            raise LargeRunError("X holds scaled values (an older run): reprocess from scratch "
                                "(reset to counts) in large-dataset mode.")
        self._keep_raw_counts()
        try:
            counts, _ = self.find_counts()
        except ValueError:
            raise LargeRunError(
                "Large-dataset mode keeps the raw counts beside the log values, and this "
                "dataset has none (layers['counts'], .raw or X).") from None
        new_x = self.x is not self._input_x
        new_counts = counts is not self._input_counts
        light = self.light
        if new_x or new_counts:
            stage = self.stage()
            self._label = "Writing the matrices"
            if new_x:
                self._x_file, self._store_file = stage / "x.h5ad", stage / "genes.h5"
            if new_counts:
                self._counts_file = stage / "counts.h5"
            t = time.time()
            self._write_pass(self.x if new_x else None, counts if new_counts else None)
            logger.info("Large run: wrote %s%s in %.1f s", "X + gene store" if new_x else "",
                        " + counts" if new_counts else "", time.time() - t)
        if not new_x:
            self._store_file = self._input_sidecars[1]
        x_file = self._x_file or self.path
        self.x = _Mat(chunked.read_lazy(str(x_file), "X", chunk_rows=self.rows),
                      pd.Index(light.var_names), "lognorm", (str(x_file), "X"),
                      np.dtype(np.int64), self.dense)
        if new_counts:
            self.layer = _Mat(chunked.read_lazy(str(self._counts_file), large_layout.COUNTS_KEY,
                                                chunk_rows=self.rows),
                              pd.Index(light.var_names), "counts",
                              (str(self._counts_file), large_layout.COUNTS_KEY),
                              np.dtype(np.int64), counts.dense)
        else:
            self.layer = counts
        self.raw = None  # the counts are in the layer; the result has no .raw
        if new_x:
            self._input_x = None
        if new_counts:
            self._input_counts = None
        self._written = True

    def _write_pass(self, x: _Mat | None, counts: _Mat | None) -> None:
        light = self.light
        xd = list(x.X.to_delayed().ravel()) if x is not None else None
        cd = list(counts.X.to_delayed().ravel()) if counts is not None else None
        n_blocks = len(xd if xd is not None else cd)
        with contextlib.ExitStack() as es:
            xa = gw = ca = None
            if x is not None:
                fx = es.enter_context(h5py.File(self._x_file, "w", **H5AD_CREATE))
                xa = chunked.CsrAppender(fx, "X")
                gw = es.enter_context(gene_store.GeneStoreWriter(
                    self._store_file, obs_names=light.obs_names, var_names=light.var_names,
                    source="X", progress=self._store_tick))
            if counts is not None:
                fc = es.enter_context(h5py.File(self._counts_file, "w"))
                ca = chunked.CsrAppender(fc, large_layout.COUNTS_KEY)
            with chunked.scheduler() as w:
                for start in range(0, n_blocks, w):
                    self._label = "Writing the matrices"
                    self.tick(start, n_blocks)
                    idx = range(start, min(n_blocks, start + w))
                    todo = []
                    for i in idx:
                        if xd is not None:
                            todo.append(xd[i])
                        if cd is not None:
                            todo.append(cd[i])
                    got = iter(dask.compute(*todo))
                    for _ in idx:
                        if xa is not None:
                            b = _f32_csr(next(got))
                            xa.add(b)
                            gw.add(b)
                        if ca is not None:
                            ca.add(_csr(next(got)))
                self.tick(n_blocks, n_blocks)
            if xa is not None:
                xa.finish(light.n_vars, np.float32)
                gw.finish()
            if ca is not None:
                ca.finish(light.n_vars, counts.X.dtype)
        if counts is not None:
            large_layout.stamp_counts(self._counts_file, light.obs_names, light.var_names)

    def store(self) -> gene_store.GeneStore:
        """The gene store of X (checked against X when it is the input's)."""
        self.write_matrices()
        if self._store is None:
            st = gene_store.GeneStore(self._store_file)
            if self._x_file is None:  # the input's: it must hold this X
                why = st.mismatch((self.x.X, self.light.obs_names, self.light.var_names),
                                  source="X")
                if why is not None:
                    st.close()
                    raise LargeRunError(f"This large dataset's gene store is stale ({why}): "
                                        "reprocess from scratch.")
            self._store = st
        return self._store

    def gene_means(self) -> np.ndarray:
        if self._means is None:
            self._label = "Gene means"
            self._means = large_scores.gene_means(self.store(), dense=self.dense)
        return self._means

    # -- steps ----------------------------------------------------------------

    def step(self, name: str, cb=None) -> Any:
        """Run one step (the pipeline's name); returns what the step records, as its
        in-memory runner does."""
        self._label = name.replace("_", " ")
        if name in _LIGHT_STEPS:
            runner = _STEP_RUNNERS[name]
            return runner(self.light, self.params, cb) if name in _SUBSTEP_RUNNERS and cb \
                else runner(self.light, self.params)
        if name in _BACKED_STEPS:
            runner = _STEP_RUNNERS[name]
            with large_backend(_Backend(self)):
                return runner(self.light, self.params, cb) if name in _SUBSTEP_RUNNERS and cb \
                    else runner(self.light, self.params)
        return getattr(self, f"_{name}")()

    def _reset_to_counts(self) -> None:
        light = self.light
        if self.layer is not None:
            self.x = self.layer
        elif self.raw is not None:  # adata.raw.to_adata(): .raw's genes
            raw = self.raw
            self.light = ad.AnnData(obs=light.obs.copy(), var=raw.var.copy(),
                                    varm=dict(raw.varm or {}), obsm=light.obsm.copy(),
                                    obsp=light.obsp.copy(), uns=light.uns.copy())
            self.x = replace(raw, var=None, varm=None)
        else:
            logger.warning("No counts layer or raw found; proceeding with current X")
            return
        self.raw = None
        clear_processing(self.light)
        logger.info("Reset to counts (%d cells, %d genes)", self.light.n_obs, self.light.n_vars)

    def _qc_metrics(self) -> None:
        light = self.light
        lc = chunked.LazyCounts(X=self.x.X, obs=light.obs, var=light.var, source="X",
                                path=str(self.path), index_dtype=self.x.index_dtype)
        self._label = "QC metrics"
        obs, var = chunked.qc_metrics(lc)
        for c in obs.columns:
            light.obs[c] = obs[c].to_numpy()
        for c in var.columns:
            light.var[c] = var[c].to_numpy()
        logger.info("QC metrics: median genes/cell=%d, median counts/cell=%d",
                    int(light.obs["n_genes_by_counts"].median()),
                    int(light.obs["total_counts"].median()))

    @contextlib.contextmanager
    def _rows(self, m: _Mat) -> Iterator[Any]:
        """``m`` for row reads: the stored matrix itself (anndata's backed CSR, or the h5py
        dataset of a dense one), else its lazy blocks computed."""
        if m.stored is None:
            yield _DaskRows(m.X)
            return
        file, key = m.stored
        with h5py.File(file, "r") as f:
            elem = f[key]
            yield elem if isinstance(elem, h5py.Dataset) else ad.io.sparse_dataset(elem)

    def _doublet_detection(self) -> dict[str, Any]:
        params = self.params
        method = doublets.method_of(params.doublet_method)  # an unknown one fails here
        # pipeline._doublet_matrix: X unless it holds processed values, then the counts.
        where = None
        m = self.x
        if not _holds_counts(m):
            if _holds_counts(self.layer):
                m, where = self.layer, "layers['counts']"
            elif _holds_counts(self.raw):
                m, where = self.raw, ".raw"
            else:
                raise StepSkipped("Doublet detection needs raw counts: X holds processed values "
                                  "and there is no counts layer or counts in .raw.")
        with self._rows(m) as counts:
            found = doublets.run_doublets(self.light, counts, method, params.expected_doublet_rate,
                                          params.doublet_channel_key or None)
        if where is not None:
            found["input"] = where
        return found

    def _filtering(self) -> dict[str, Any]:
        from scview.core.anndata_adaptor import _QC_OBS_NAMES

        light, params = self.light, self.params
        missing = [k for k in ("total_counts", "n_genes", "pct_mt")
                   if not any(c in light.obs.columns for c in _QC_OBS_NAMES[k])]
        if missing:
            raise LargeRunError(
                "Filtering in large-dataset mode judges the cells by the QC metrics in the "
                "cell table: run QC metrics first (in this run or before).")
        n_before = light.n_obs
        t, keep, record, found = filter_judgement(light, params)
        if not keep.all():
            light = subset_lean(light, obs=keep)
            self._cut(obs=keep)
        light.uns[qc_filter.RECORD_KEY] = record
        found.update(qc_filter.after_filter(light, t, int((~keep).sum())))
        self._label = "Genes per cell count"
        number = chunked.gene_detection(self.x.X)
        light.var["n_cells"] = number
        genes = number >= params.min_cells
        if not genes.all():
            light = subset_lean(light, var=genes)
            self._cut(var=genes)
        self.light = light
        logger.info("Filtering (%s): %d -> %d cells", t.mode, n_before, light.n_obs)
        return found

    def _normalization(self) -> None:
        normalization.check_method(self.params.normalization_method)
        if self.layer is None:
            if self.raw is None or _holds_counts(self.x):
                self.layer = self.x
            else:
                self._keep_raw_counts()
        x = self.x
        self._label = "Normalisation"
        X = chunked.normalize_total(x.X, target_sum=self.params.target_sum)
        self.x = replace(x, X=X, kind="normalized", stored=None)
        scaling.drop_plan(self.light)
        logger.info("Normalized to target sum %.0f", self.params.target_sum)

    def _log_transform(self) -> None:
        x = self.x
        kind = "lognorm" if x.kind == "normalized" else "logged"
        self.x = replace(x, X=chunked.log1p(x.X), kind=kind, stored=None)
        self.light.uns["log1p"] = {"base": None}
        scaling.drop_plan(self.light)
        logger.info("Applied log1p transformation")

    def _batch(self, key: str | None):
        return self.light.obs[key] if key else None

    def _highly_variable_genes(self) -> dict[str, Any] | None:
        params, light = self.params, self.light
        method = normalization.check_method(params.normalization_method)
        if method == "log" and params.hvg_flavor not in HVG_FLAVORS:
            raise ValueError(f"Unsupported HVG flavor '{params.hvg_flavor}'; choose from "
                             f"{', '.join(HVG_FLAVORS)}.")
        batch_key = params.hvg_batch_key or None
        if batch_key and batch_key not in light.obs.columns:
            raise ValueError(f"HVG batch key '{batch_key}' not found in obs columns.")
        self.write_matrices()
        n_top = min(params.n_top_genes, light.n_vars)
        self._label = "Variable genes"
        if method == "pearson_residuals":
            counts, source = self.find_counts()
            df = chunked.pearson_hvg(counts.X, light.var_names, n_top_genes=n_top,
                                     batch=self._batch(batch_key), theta=normalization.THETA)
            _var_from_hvg_frame(light, df)
            return {"hvg_flavor": "pearson_residuals", "theta": normalization.THETA,
                    "counts_source": source}
        if params.hvg_flavor == "seurat_v3":
            try:
                counts, source = self.find_counts()
            except ValueError:
                raise ValueError(
                    "The Seurat v3 flavour ranks raw counts (layers['counts'], .raw or X), and "
                    "this dataset has none: choose the Seurat or Cell Ranger flavour.") from None
            df = chunked.highly_variable_genes(counts.X, light.var_names, flavor="seurat_v3",
                                               n_top_genes=n_top, batch=self._batch(batch_key))
            _var_from_hvg_frame(light, df)
            return {"counts_source": source}
        df = chunked.highly_variable_genes(self.lognorm(), light.var_names,
                                           flavor=params.hvg_flavor,
                                           n_top_genes=params.n_top_genes,
                                           batch=self._batch(batch_key))
        for c in df.columns:
            light.var[c] = df[c].to_numpy()
        light.uns["hvg"] = {"flavor": params.hvg_flavor}
        logger.info("Selected %d highly variable genes (flavor=%s, batch_key=%s)",
                    int(light.var["highly_variable"].sum()), params.hvg_flavor, batch_key)
        return None

    def _scaling(self) -> dict[str, Any]:
        method = normalization.check_method(self.params.normalization_method)
        keys = scaling.check_regress_out(self.params.regress_out)
        if keys and method == "pearson_residuals":
            raise ValueError(_REGRESS_PEARSON)
        self.write_matrices()
        light = self.light
        self._label = "Scaling statistics"
        if method == "pearson_residuals":
            _, source = self.find_counts()
            clip = normalization.default_clip(light.n_obs)
            found = scaling.plan_residuals(light, normalization.THETA, clip, source)
            return {"theta": normalization.THETA, "clip": round(clip, 3), "counts_source": source,
                    **found}
        X = self.lognorm()
        scored = {}
        if keys:
            with large_backend(_Backend(self)):
                scored = regress_covariates(light, self.params, keys)
            self._label = "Regress-out"

        def regress(idx, C):
            B = chunked.regression_coef(X, idx, C)
            return (B, *chunked.regressed_stats(X, idx, C, B))

        found = scaling.plan_zscores(light, max_value=scaling.MAX_VALUE,
                                     stats=lambda idx: chunked.scaled_stats(X, idx),
                                     regress_out=keys, regress=regress)
        return {**found, **scored}

    def _pca(self) -> dict[str, Any] | None:
        light, params = self.light, self.params
        # chunked caps it at cells - 1 and variable genes - 1, or says what to change
        n_comps = params.n_comps
        plan = scaling.read_plan(light)
        mask = scaling.gene_mask(light)
        hvg = mask if mask is not None else np.ones(light.n_vars, dtype=bool)
        cols = np.flatnonzero(hvg)
        self.write_matrices()
        self._label = "PCA"
        found = None
        if plan is not None and plan["method"] == scaling.PEARSON:
            counts, _ = self.find_counts()
            res = chunked.pca_pearson(counts.X, hvg, n_comps=n_comps,
                                      theta=float(plan.get("theta", normalization.THETA)),
                                      clip=float(plan["clip"]))
        elif plan is not None:
            X = self.lognorm()
            mean = scaling._var_column(light, scaling.MEAN, cols)
            std = scaling._var_column(light, scaling.STD, cols)
            fit = scaling.plan_regression(light, plan, cols,
                                          fit=lambda c, C: chunked.regression_coef(X, c, C))
            missing = np.flatnonzero(~np.isfinite(mean) | ~np.isfinite(std))
            if fit is not None:
                missing = np.union1d(missing, fit[2])
            if missing.size:  # variable since scaling: worked out now, as pca_matrix does
                mean[missing], std[missing] = (
                    chunked.scaled_stats(X, cols[missing]) if fit is None else
                    chunked.regressed_stats(X, cols[missing], fit[0], fit[1][:, missing]))
            res = chunked.pca_scaled(X, hvg, n_comps=n_comps,
                                     max_value=float(plan.get("max_value", scaling.MAX_VALUE)),
                                     stats=(mean, std),
                                     regress=None if fit is None else fit[:2])
        else:  # sc.tl.pca on X as it is: centred, not scaled
            res = chunked.pca_scaled(self.lognorm(), hvg, n_comps=n_comps, max_value=None,
                                     stats=(np.zeros(cols.size), np.ones(cols.size)))
        if plan is not None:
            found = {"pca_input": plan["method"],
                     "pca_genes": int(mask.sum()) if mask is not None else light.n_vars}
        light.obsm["X_pca"] = res["X_pca"]
        light.varm["PCs"] = res["PCs"]
        light.uns["pca"] = {
            "params": {"zero_center": True, "use_highly_variable": mask is not None,
                       "mask_var": "highly_variable" if mask is not None else None},
            "variance": res["variance"],
            "variance_ratio": res["variance_ratio"],
        }
        light.obsm.pop("X_pca_harmony", None)
        light.uns.pop(BATCH_MIXING_KEY, None)
        logger.info("PCA: %d components", res["n_comps"])
        return found

    # -- the result ------------------------------------------------------------

    def save(self) -> None:
        """Write the derived h5ad beside the staged matrices and move everything in place:
        the sidecars first, then the h5ad whose ``uns["scview_large"]`` names them."""
        if self.output is None:
            return
        self.write_matrices()
        self._close_store()
        light = self.light
        stage = self.stage()
        out = self.output
        counts_out, genes_out = large_layout.sidecar_paths(out)
        large_layout.record(light.uns, out)
        light.strings_to_categoricals()
        target = self._x_file or stage / "x.h5ad"
        kw = {"compression": "gzip"}
        self._label = "Saving"
        if self._x_file is None:  # X unchanged: the input's, copied as stored
            with h5py.File(target, "w", **H5AD_CREATE) as f, h5py.File(self.path, "r") as src:
                src.copy(src["X"], f, name="X")
        with h5py.File(target, "a") as f:
            for key in [k for k in f.keys() if k != "X"]:
                del f[key]
            f.attrs.setdefault("encoding-type", "anndata")
            f.attrs.setdefault("encoding-version", "0.1.0")
            for key, value in (("obs", light.obs), ("var", light.var), ("obsm", dict(light.obsm)),
                               ("varm", dict(light.varm)), ("obsp", dict(light.obsp)),
                               ("varp", dict(light.varp)), ("layers", {}),
                               ("uns", dict(light.uns))):
                ad.io.write_elem(f, key, value, dataset_kwargs=kw)
        moves = []
        inputs = self._input_sidecars or (None, None)
        for staged, input_file, final in ((self._counts_file, inputs[0], counts_out),
                                          (self._store_file if self._x_file is not None else None,
                                           inputs[1], genes_out)):
            if staged is not None:
                moves.append((staged, final))
            elif input_file is not None and Path(input_file) != final:
                copy = stage / final.name
                shutil.copy2(input_file, copy)  # another folder's: this dataset gets its own
                moves.append((copy, final))
        for src, dst in moves:
            os.replace(src, dst)
        os.replace(target, out)
        gene_store._sync_dir(out.parent)
        self.saved = True
        logger.info("Saved large dataset to %s (+ %s, %s)", out, counts_out.name, genes_out.name)

    # -- the pre-step memory check ---------------------------------------------

    def halt_before(self, step: str) -> tuple[str, str] | None:
        c = self.control
        if c is None:
            return None
        if c.cancelled:
            return "cancelled", cancelled_reason("before", step)
        if c.before_step is None:
            return None
        if c.before_step is not default_before_step:  # a check plugged in: give it light
            return c.halt_before(step, self.light, self.params)
        try:
            message = self.memory_check(step)
        except Exception:  # noqa: BLE001 — a broken check must not stop a run
            logger.warning("The pre-step check failed before %s; running it", step, exc_info=True)
            return None
        return ("stopped", message) if message else None

    def memory_check(self, step: str) -> str | None:
        """``run_watchdog.default_before_step`` with this mode's estimate of the step."""
        from scview.core import preflight

        avail = preflight.available_memory()
        if not avail.limited or avail.available_bytes is None or avail.limit_bytes is None:
            return None
        write = not self._written and step in preflight._CHUNKED_READS_X
        need = preflight.chunked_step_peak(step, self.shape_for_estimate(), self.params,
                                           write=write, doublets=self._doublet_plan(step))
        if need * (1 - preflight.ERROR_BAND) <= avail.available_bytes:
            return None
        label = step.replace("_", " ")
        return (f"Stopped before {label}: in large-dataset mode it needs about {need / GB:.1f} GB "
                f"more at its peak, and scView has {avail.available_bytes / GB:.1f} GB free of its "
                f"{avail.limit_bytes / GB:.1f} GB limit, so the system would have stopped the "
                "backend. Nothing was saved: the dataset is as it was before the run. Raise "
                "SCVIEW_MEMORY_GB in .env (then restart scView), or leave this step out.")

    def shape_for_estimate(self):
        """The ``preflight.ChunkedShape`` of the run now (filtering cuts the stored values
        with the cells and genes)."""
        from scview.core import preflight

        light = self.light
        n0, m0 = self._shape0
        share = (light.n_obs * light.n_vars) / max(1, n0 * m0)
        other = (sum(_array_bytes(v) for v in light.obsm.values())
                 + sum(_array_bytes(v) for v in light.obsp.values())
                 + sum(_array_bytes(v) for v in light.varm.values())
                 + light.n_obs * (64 + 8 * light.obs.shape[1])
                 + light.n_vars * (64 + 8 * light.var.shape[1])
                 + light.n_obs * preflight.LOAD_PER_CELL + preflight.LOAD_FIXED)
        return preflight.ChunkedShape(light.n_obs, light.n_vars, int(self.nnz * min(1.0, share)),
                                      self.dense, int(other), _n_groups(light))

    def _doublet_plan(self, step: str):
        if step != "doublet_detection":
            return None
        from scview.core import preflight

        try:
            ch = doublets.channels(self.light, self.params.doublet_channel_key or None)
            return preflight.DoubletPlan([int(k) for k in ch.sizes])
        except Exception:  # noqa: BLE001 — an estimate never fails a step
            return None


def _array_bytes(m) -> int:
    if sp.issparse(m):
        return int(m.data.nbytes + m.indices.nbytes + m.indptr.nbytes)
    if isinstance(m, pd.DataFrame):
        return int(m.memory_usage(deep=False).sum())
    return int(getattr(m, "nbytes", 0))


def _n_groups(light: ad.AnnData) -> int | None:
    col = light.uns.get("scview_active_clustering")
    if col and col in light.obs.columns:
        return int(light.obs[col].nunique())
    return None


class _Backend:
    """What ``pipeline.large_backend`` asks for: the matrix reads of the steps that run
    the pipeline's own functions on the matrix-free copy."""

    def __init__(self, run: LargeRun) -> None:
        self.run = run

    def rank_genes_groups(self, adata, groupby: str, key_added: str | None = None) -> None:
        run = self.run
        store = run.store()
        run._label = f"Marker genes for {groupby}"
        sparse_markers.rank_genes_groups(
            adata, groupby, store, method="wilcoxon", pts=True, use_raw=False,
            key_added=key_added, specificity=True, check_store=False,
            progress=run._store_tick)

    def score_genes(self, adata, genes, *, score_name: str, ctrl_size: int,
                    random_state: int = 0) -> None:
        run = self.run
        means = run.gene_means()
        run._label = f"Score {score_name}"
        large_scores.score_genes(adata, run.lognorm(), means, genes, score_name=score_name,
                                 ctrl_size=ctrl_size, random_state=random_state, dense=run.dense)

    def score_genes_cell_cycle(self, adata, *, s_genes, g2m_genes) -> None:
        run = self.run
        means = run.gene_means()
        run._label = "Cell cycle scores"
        large_scores.score_genes_cell_cycle(adata, run.lognorm(), means, s_genes=s_genes,
                                            g2m_genes=g2m_genes, dense=run.dense)

    def celltypist(self, adata, model, groupby: str):
        run = self.run
        run.write_matrices()
        counts, _ = run.find_counts()
        run._label = "CellTypist"
        return large_scores.celltypist_annotate(model, counts.X, adata.obs, adata.var_names,
                                                groupby, progress=run.tick)


# ---------------------------------------------------------------------------
# Running the steps
# ---------------------------------------------------------------------------

_SENTINEL = object()


def _disk_check(run: LargeRun, steps: list[str]) -> str | None:
    """A sentence when the data folder lacks the disk the run writes, else None."""
    from scview.core import preflight

    if run.output is None:
        return None
    rewrite = not run.large_in or any(s in steps for s in preflight.CHUNKED_WRITES)
    x_bytes = 0 if rewrite else _stored_bytes(run.path, "X")
    need = preflight.chunked_disk_bytes(run.shape_for_estimate(), rewrite=rewrite,
                                        x_file_bytes=x_bytes)
    folder = run.output.parent
    folder.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(folder).free
    if need <= free:
        return None
    return (f"Not enough disk for a large-dataset run: it writes about {need / GB:.1f} GB (the "
            f"log-normalised matrix, the counts, the gene-major copy and its temporary files) "
            f"and the data folder has {free / GB:.1f} GB free. Free some disk (delete datasets "
            "you no longer need) and run again. Nothing was changed.")


def _plan_check(run: LargeRun, steps: list[str]) -> str | None:
    """Why this mode cannot make the run (it stores X log-normalised, beside the counts),
    before any step: None when it can."""
    why = refused(steps)
    if why is not None:
        return f"{why} Nothing was changed."
    if not (_holds_counts(run.layer) or _holds_counts(run.raw) or _holds_counts(run.x)):
        return ("Large-dataset mode keeps the raw counts beside the log values, and this "
                "dataset has none (layers['counts'], .raw or X). Nothing was changed.")
    resets = "reset_to_counts" in steps and (run.layer is not None or run.raw is not None)
    kind = "counts" if resets else _judge(run.x)
    if "normalization" in steps:
        kind = "normalized"
    if "log_transform" in steps:
        kind = "lognorm" if kind == "normalized" else "logged"
    if kind == "counts":
        return ("Large-dataset mode stores X log-normalised, and these steps leave raw counts "
                "in X: include normalization and log transform (or reprocess from scratch). "
                "Nothing was changed.")
    if kind == "normalized":
        return ("Large-dataset mode stores X log-normalised: include the log transform after "
                "normalization. Nothing was changed.")
    if kind == "scaled":
        return ("X holds scaled values (an older run): reprocess from scratch (reset to counts) "
                "in large-dataset mode. Nothing was changed.")
    return None


def _stored_bytes(path: Path, key: str) -> int:
    """Bytes a stored matrix takes on disk (its datasets' storage)."""
    total = 0
    with h5py.File(path, "r") as f:
        elem = f[key]
        items = [elem] if isinstance(elem, h5py.Dataset) else [elem[k] for k in elem]
        for d in items:
            if isinstance(d, h5py.Dataset):
                total += d.id.get_storage_size()
    return int(total)


def run_need(path: str | Path, steps: list[str], params=None) -> int:
    """Memory a large-dataset run of ``steps`` on the h5ad at ``path`` needs at its peak
    (nothing of the dataset is held when it starts: its cached copy is dropped)."""
    from scview.core import preflight

    h = preflight.read_header(path, "h5ad")
    plan = (preflight._file_doublet_plan(path, params) if "doublet_detection" in steps
            else None)
    large = large_info(path) is not None
    est = preflight.estimate_chunked_run(h, steps, params, large=large, doublets=plan)
    return int(est.peak_bytes)


def run_streamed(
    path: str | Path,
    steps: list[str],
    params: PipelineParams | None = None,
    output_path: str | None = None,
    control: RunControl | None = None,
) -> Generator[tuple[str, dict[str, Any]], None, PipelineResult]:
    """``pipeline.run_pipeline_streamed`` in large-dataset mode on the h5ad at ``path``:
    the same events (step_start, step_done, step_error, step_skipped, substep, complete;
    substeps also from the passes over the blocks) and result (``chunked`` true). Saved to
    ``output_path`` only when the run reaches its end."""
    params = params or PipelineParams()
    if check_umap_3d(params.umap_3d) == "auto":  # no 3D UMAP in this mode unless asked for
        params = replace(params, umap_3d=resolve_umap_3d("auto", large=True))
    result = PipelineResult(chunked=True)
    t0 = time.time()
    ordered = [s for s in ALL_STEPS if s in steps]
    total = len(ordered)
    result.unknown_steps = _unknown_requested(steps)
    run = LargeRun(path, output_path, params, control)
    try:
        run.open()
        why = _plan_check(run, ordered) or _disk_check(run, ordered)
        if why is not None:
            _halted(result, ("stopped", why))
        # A failed step skips the steps that need it, as in memory (core/step_failures.py).
        outcome = _new_outcome(run.light, ordered, params)
        for idx, step in enumerate(ordered if result.status == "complete" else []):
            because = _skip_if_blocked(run.light, result, outcome, step)
            if because is not None:
                yield ("step_skipped", skipped_event(step, idx, total, because))
                continue
            halt = run.halt_before(step)
            if halt is not None:
                _halted(result, halt)
                break
            yield ("step_start", {"step": step, "index": idx, "total": total})
            step_t0 = time.time()
            try:
                scope = control.step(step) if control is not None else contextlib.nullcontext()
                with scope:
                    ret = yield from _in_thread(run, step, control)
                result.steps_run.append(step)
                outcome.succeeded.append(step)
                recorded = dict(ret) if isinstance(ret, dict) else {}
                recorded["chunked"] = True
                _record_pipeline_step(run.light, step, params, recorded,
                                      control.watchdog.steps.get(step) if control else None)
                yield ("step_done", {"step": step, "index": idx,
                                     "elapsed": round(time.time() - step_t0, 2)})
            except StepSkipped as e:
                logger.info("Step '%s' skipped: %s", step, e)
                result.notes[step] = str(e)
                yield ("step_done", {"step": step, "index": idx, "skipped": True, "note": str(e),
                                     "elapsed": round(time.time() - step_t0, 2)})
            except Exception as e:  # noqa: BLE001 — the step's error, as in memory
                if control is not None and control.cancelled:
                    _cancelled_during(result, step)
                    break
                _step_failed(run.light, result, outcome, step, str(e))  # logs it
                yield ("step_error", {"step": step, "index": idx, "error": str(e)})
        _finish(run, params, result, t0, control, outcome)
    finally:
        run.cleanup()
    yield ("complete", result.to_dict())
    return result


def _in_thread(run: LargeRun, step: str, control: RunControl | None):
    """Run one step in a thread of its own (the run's context, so a Leiden child stays
    cancellable), yielding its substep events as they come; returns what it returns."""
    q: queue.Queue = queue.Queue()
    holder: dict[str, Any] = {}

    def cb(msg: str, current: int, sub_total: int) -> None:  # the markers' / enrichment's
        if control is not None and control.cancelled:
            raise RunCancelled()
        q.put({"message": msg, "current": current, "total": sub_total})

    def target() -> None:
        run._post = q.put
        try:
            with chunked.on_blocks(run.tick):
                holder["ret"] = run.step(step, cb)
        except BaseException as e:  # noqa: BLE001 — handed to the generator
            holder["exc"] = e
        finally:
            run._post = None
            q.put(_SENTINEL)

    ctx = contextvars.copy_context()
    t = threading.Thread(target=ctx.run, args=(target,), daemon=True, name=f"large-{step}")
    t.start()
    try:
        while True:
            item = q.get()
            if item is _SENTINEL:
                break
            yield ("substep", {"step": step, **item})
    finally:
        t.join()
    if "exc" in holder:
        raise holder["exc"]
    return holder.get("ret")


def _finish(run: LargeRun, params: PipelineParams, result: PipelineResult, t0: float,
            control: RunControl | None, outcome=None) -> None:
    """``pipeline._finish_run`` for this mode: save only a run that reached its end (also
    one that finished with errors). This mode saves X log-normalised only: when a failed
    step (reset, normalisation, log transform) leaves it otherwise, the run is stopped
    with nothing saved, the dataset as it was."""
    if outcome is not None:
        step_failures.settle_unrun(run.light, outcome)  # markers of a column clustered again
        _end_status(result, outcome)
    if result.status in SAVED_STATUSES and control is not None and control.cancelled:
        _halted(result, ("cancelled", cancelled_reason("before", "saving")))
    save = result.status in SAVED_STATUSES
    if save:
        _record_errors(run.light, result, outcome)
        _finalize_current(run.light, params, result.steps_run, result)
        provenance.set_current(run.light, anchors=["counts", "lognorm"], large=True)
    result.elapsed_seconds = round(time.time() - t0, 2)
    write_s = None
    if run.output is not None and save:
        tw = time.time()
        try:
            run.save()
        except RunCancelled:  # cancelled while the matrices were written for the save
            _halted(result, ("cancelled", cancelled_reason("during", "saving")))
        except LargeRunError as e:
            if not result.errors:
                raise
            failed = ", ".join(s.replace("_", " ") for s in result.errors)
            _halted(result, ("stopped", f"Nothing was saved: {failed} failed. {e} The dataset "
                                        "is as it was before the run."))
        else:
            write_s = round(time.time() - tw, 2)
            result.output_path = str(run.output)
    if control is not None:
        result.resources = {**control.watchdog.summary(), "write_s": write_s}


def run(path: str | Path, steps: list[str], params: PipelineParams | None = None,
        output_path: str | None = None, control: RunControl | None = None) -> PipelineResult:
    """``run_streamed`` to its end; the result."""
    gen = run_streamed(path, steps, params, output_path, control)
    while True:
        try:
            next(gen)
        except StopIteration as stop:
            return stop.value


# ---------------------------------------------------------------------------
# Dispatch: the run endpoints call these with the adaptor processing.py hands them
# ---------------------------------------------------------------------------


def run_for(adaptor, steps: list[str], params: PipelineParams, output_path: str,
            control: RunControl | None) -> PipelineResult:
    """The run in the mode ``core/processing.py`` chose (``control.mode``)."""
    mode = getattr(control, "mode", None)
    if is_chunked(control):
        result = run(adaptor.h5ad_path, steps, params, output_path, control)
    else:
        result = run_pipeline(adaptor.adata, steps, params, output_path, control)[1]
    result.mode = mode.to_dict() if mode is not None else None
    return result


def run_streamed_for(adaptor, steps: list[str], params: PipelineParams, output_path: str,
                     control: RunControl | None):
    """The streamed run in the mode ``core/processing.py`` chose (``control.mode``)."""
    mode = getattr(control, "mode", None)
    if is_chunked(control):
        gen = run_streamed(adaptor.h5ad_path, steps, params, output_path, control)
    else:
        gen = run_pipeline_streamed(adaptor.adata, steps, params, output_path, control)
    for event, data in gen:
        if event == "complete":
            data = {**data, "mode": mode.to_dict() if mode is not None else None}
        yield event, data
