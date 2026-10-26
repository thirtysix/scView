"""A dataset as an R object: a Seurat (SeuratObject v5) or SingleCellExperiment .rds.

The R converter sidecar builds the object (``converter/export_rds.R``, ``POST /export``);
this module decides what goes in and hands it over in a staging h5ad on the shared
data volume, laid out so the R side reads it without guessing:

- ``X``: the log-normalised values (Seurat ``data`` layer / SCE ``logcounts``): X, or
  its ``.raw`` when X holds an older scView's scaled values (``cellxgene_export.matrices``,
  the same choice as the CELLxGENE export). Scaled values never go in: when they are all
  there is, ``X`` is left out and the report says so (Seurat recomputes them with
  ScaleData). A dataset that is not normalised yet has no ``X``.
- ``layers/counts``: the raw counts (``layers['counts']``, a counts ``.raw``, X itself
  when it holds counts, or a large dataset's counts sidecar).
- ``obs`` and ``var``: cell and gene tables (names made unique, symbols as gene names);
  ``obsm``: every layout (``X_pca``, ``X_umap`` ...) with all its columns; ``varm/PCs``:
  the PCA loadings.
- ``uns/scview_rexport``: one JSON string with the provenance (scView version, dataset
  name, recorded steps), colours, the active clustering column, PCA variance, protein
  names and what was left out on this side; ``uns/protein_counts`` / ``protein_clr``:
  the CITE-seq antibody counts and their CLR (an "ADT" assay / altExp).

The export runs as a background job (``start``): the staging file is written while the
dataset's edit lock is held (one consistent state), then the converter is asked and its
stages are read from a progress file, as an import is (``core/conversion.py``). The
finished .rds is sent once and removed, or removed an hour after it was written.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import shutil
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import h5py
import httpx
import numpy as np
import pandas as pd
import scipy.sparse as sp

from scview.core import cell_ontology as co
from scview.core import cellxgene_export as cx
from scview.core import conversion

logger = logging.getLogger(__name__)

FORMATS: dict[str, dict[str, str]] = {
    "seurat": {"label": "Seurat object", "suffix": "_seurat.rds", "needs": "SeuratObject >= 5"},
    "sce": {"label": "SingleCellExperiment", "suffix": "_sce.rds", "needs": "SingleCellExperiment"},
}
UNS_KEY = "scview_rexport"
STAGING = "input.h5ad"
OUTPUT = "output.rds"
PROGRESS = "progress.json"
KEEP_SECONDS = 3600  # a finished .rds nobody downloaded is removed after an hour

# The converter's peak memory: R and Python with their packages, plus per stored value
# (the R object keeps 12 bytes a value per matrix, an 8-byte double and a 4-byte row
# index; while one is read, Python holds it too, and R's conversion of the vectors makes
# a transient copy). Container peaks measured (cgroup memory.peak, page cache included):
# Kang 19.3M values 1.18 GB (Seurat) / 1.12 GB (SCE); paul15, dense, 4.8M values
# 0.60 / 0.84 GB. The estimate below gives 1.47 and 0.89 GB: on the safe side.
CONVERTER_BASE_BYTES = 700 * 1024 ** 2
CONVERTER_BYTES_PER_VALUE = 40
# A dgCMatrix stores its row indices and column pointers as 32-bit integers.
MAX_VALUES = 2 ** 31 - 1

_JOBS: dict[str, "Job"] = {}


class ExportError(ValueError):
    """Nothing to export, or an export that cannot be made: a 400/422 with this sentence."""


# ---------------------------------------------------------------------------
# What goes in


@dataclass
class Plan:
    obs: pd.DataFrame
    var: pd.DataFrame
    obsm: dict[str, np.ndarray]
    x: cx.Matrix | None  # log-normalised values (Seurat data / logcounts)
    counts: cx.Matrix | None
    pcs: np.ndarray | None
    protein: dict[str, Any] = field(default_factory=dict)  # names, counts, clr
    meta: dict[str, Any] = field(default_factory=dict)  # into uns scview_rexport
    notes: list[str] = field(default_factory=list)
    dropped: dict[str, list[str]] = field(default_factory=dict)
    n_values: int = 0  # stored values over both matrices


def _n_values(m: cx.Matrix | None) -> int:
    """Stored (non-zero) values of a matrix in memory or in a file."""
    if m is None:
        return 0
    if m.value is not None:
        v = m.value
        if sp.issparse(v):
            return int(v.nnz)
        if isinstance(v, np.ndarray):
            return int(np.count_nonzero(v))
    try:
        with h5py.File(m.path, "r") as f:
            node = f[m.key]
            if isinstance(node, h5py.Group):
                return int(node["data"].shape[0])
            return int(np.prod(node.shape))  # dense on disk: every value, at most
    except (OSError, KeyError, TypeError):
        return int(np.prod(m.shape))


def _r_obs(adata) -> tuple[pd.DataFrame, list[str], int]:
    """Cell table for R: categoricals keep their level order (factors), text becomes a
    factor, true/false stays logical, numbers stay numbers."""
    obs, notes, _left, renamed = cx.clean_obs(adata.obs, None)
    for col in adata.obs.columns:
        s = adata.obs[col]
        if pd.api.types.is_bool_dtype(s) and not s.isna().any():
            obs[str(col).replace("/", "_")] = s.to_numpy(dtype=bool)
        elif pd.api.types.is_extension_array_dtype(s) and pd.api.types.is_numeric_dtype(s):
            obs[str(col).replace("/", "_")] = s.to_numpy(dtype=np.float64, na_value=np.nan)
    return obs, notes, renamed


def _ontology_columns(adata, obs: pd.DataFrame) -> list[str]:
    """Each label column's Cell Ontology mapping (uns scview_label_terms) as a per-cell
    ``<column>_ontology_term_id`` column in ``obs`` (CELLxGENE's naming); "unknown"
    stays as written, labels without a term are NA. Returns the columns added."""
    added = []
    for col in co.mapped_columns(adata):
        name = f"{str(col).replace('/', '_')}_ontology_term_id"
        if name in obs.columns:
            continue
        ids = adata.obs[col].astype(str).map(co.stored(adata, col))
        obs[name] = pd.Categorical(ids.to_numpy(), categories=sorted(set(ids.dropna())))
        added.append(name)
    return added


def _protein(adata) -> dict[str, Any]:
    from scview.core import protein as prot

    val = adata.obsm.get(prot.COUNTS_KEY) if prot.COUNTS_KEY in adata.obsm else None
    if val is None:
        return {}
    if isinstance(val, pd.DataFrame):
        names = [str(c) for c in val.columns]
        counts = val.to_numpy(dtype=np.float64)
    else:
        counts = sp.csr_matrix(val).toarray().astype(np.float64) if sp.issparse(val) else np.asarray(
            val, dtype=np.float64)
        names = [f"protein{i + 1}" for i in range(counts.shape[1])]
    out: dict[str, Any] = {"names": names, "counts": counts}
    clr = adata.obsm.get(prot.CLR_KEY) if prot.CLR_KEY in adata.obsm else None
    if clr is not None:
        clr = clr.to_numpy(dtype=np.float64) if isinstance(clr, pd.DataFrame) else np.asarray(clr, np.float64)
        if clr.shape == counts.shape:
            out["clr"] = clr
    return out


def _pca_meta(adata) -> dict[str, Any]:
    pca = adata.uns.get("pca")
    out: dict[str, Any] = {}
    if isinstance(pca, dict):
        for key in ("variance", "variance_ratio"):
            if key in pca:
                vals = np.asarray(pca[key], dtype=np.float64).ravel()
                if vals.size and np.isfinite(vals).all():
                    out[f"pca_{key}"] = [float(v) for v in vals]
    return out


def _recipe(adata) -> list[str]:
    from scview.core.provenance import recipe

    try:
        return [str(s["step"]) for s in recipe(adata)]
    except Exception:  # noqa: BLE001 - a malformed record never stops the export
        return []


def plan(adaptor, fmt: str, dataset_name: str = "", dataset_id: str = "") -> Plan:
    """What the R object will hold and what is left out, without writing anything."""
    from scview.core.provenance import UNS_KEY as PROV_KEY
    from scview.core.scaling import holds_scaled

    if fmt not in FORMATS:
        raise ExportError(f"Unknown R format '{fmt}': use seurat or sce.")
    adata = adaptor.adata
    x, counts, notes = cx.matrices(adaptor)
    notes = list(notes)
    if x is not None and x.is_counts:
        if counts is None:
            counts = x
        x = None
        notes.append("The dataset is not normalised yet, so the object holds raw counts only "
                     + ("(run NormalizeData() in R)." if fmt == "seurat" else "(no logcounts)."))
    elif x is not None and x.value is not None and holds_scaled(x.value):
        x = None
        notes.append("The expression values are scaled (z-scores) and there is no log-normalised "
                     "copy, so they are left out: scaled values are not log-normalised data.")
    if x is None and counts is None:
        raise ExportError("This dataset has neither raw counts nor log-normalised values to export.")
    var_src = adata.raw.var if (x is not None and x.where == ".raw" and adata.raw is not None) else adata.var
    var, vnotes, _ = cx.symbol_var(var_src)
    notes += vnotes
    obs, onotes, _ = _r_obs(adata)
    notes += onotes
    terms = _ontology_columns(adata, obs)
    if terms:
        notes.append("Cell Ontology terms you chose travel as cell metadata: "
                     + ", ".join(terms) + " (labels without a term are NA).")

    choices = cx.embedding_choices(adata, max_dims=10 ** 6)
    obsm = {c["export_name"]: np.ascontiguousarray(np.asarray(adata.obsm[c["key"]]), dtype=np.float64)
            for c in choices}
    pcs = None
    if "X_pca" in obsm and "PCs" in adata.varm:
        loadings = np.asarray(adata.varm["PCs"], dtype=np.float64)
        if loadings.shape == (var.shape[0], obsm["X_pca"].shape[1]):
            pcs = loadings
        else:
            notes.append("The PCA loadings do not match the PCA's genes and components; left out.")

    protein = _protein(adata)
    colors = cx._colors(adata.uns, obs, {str(c).replace("/", "_"): str(c) for c in adata.obs.columns})
    cluster = adaptor.active_clustering_column() if hasattr(adaptor, "active_clustering_column") else None
    hv = ([str(g) for g, keep in zip(var.index, var_src["highly_variable"].to_numpy()) if keep]
          if "highly_variable" in var_src.columns and var_src["highly_variable"].dtype == bool else [])

    kept_obsm = {c["key"] for c in choices}
    if protein:
        kept_obsm |= {"protein_counts", "protein_clr"} if "clr" in protein else {"protein_counts"}
    pca = _pca_meta(adata)
    used_uns = ({PROV_KEY, *(f"{c}_colors" for c in adata.obs.columns)} | ({"pca"} if pca else set())
                | ({co.TERMS_KEY} if terms else set()))
    dropped = {
        "obsm": [str(k) for k in adata.obsm.keys() if k not in kept_obsm],
        "layers": [str(k) for k in adata.layers.keys()
                   if not (counts is not None and counts.key == f"layers/{k}")],
        "obsp": [str(k) for k in adata.obsp.keys()],
        "varm": [str(k) for k in adata.varm.keys() if not (k == "PCs" and pcs is not None)],
        "uns": sorted(str(k) for k in adata.uns.keys() if k not in used_uns),
    }
    reasons = {
        "obsp": "neighbour graphs (rebuild with FindNeighbors / scran::buildSNNGraph)",
        "uns": "scView's stored results (download them as 'Stored results (.zip)')",
    }
    meta: dict[str, Any] = {
        "format": fmt,
        "dataset_name": dataset_name,
        "dataset_id": dataset_id,
        "x_from": x.where if x is not None else None,
        "counts_from": counts.where if counts is not None else None,
        "cluster_column": cluster if cluster in obs.columns else None,
        "variable_genes": hv,
        "colors": colors,
        "recipe": _recipe(adata),
        "provenance": adata.uns.get(PROV_KEY) if isinstance(adata.uns.get(PROV_KEY), str) else None,
        "protein_names": protein.get("names", []),
        "left_out": [f"{where}: {', '.join(keys)}" + (f" - {reasons[where]}" if where in reasons else "")
                     for where, keys in dropped.items() if keys],
        **pca,
    }
    p = Plan(obs=obs, var=var, obsm=obsm, x=x, counts=counts, pcs=pcs, protein=protein, meta=meta,
             notes=notes, dropped={k: v for k, v in dropped.items() if v})
    p.n_values = _n_values(x) + (_n_values(counts) if counts is not x else 0)
    for m in (x, counts):
        if m is not None and _n_values(m) > MAX_VALUES:
            raise ExportError(f"The matrix from {m.where} has {_n_values(m):,} stored values; an R sparse "
                              f"matrix holds at most {MAX_VALUES:,}. Subset the dataset first.")
    return p


def estimate_bytes(p: Plan) -> int:
    """About the converter's peak memory while it builds and saves the object."""
    dense = sum(int(a.size) for a in p.obsm.values()) * 16
    if p.protein:
        dense += int(p.protein["counts"].size) * 24
    return CONVERTER_BASE_BYTES + CONVERTER_BYTES_PER_VALUE * p.n_values + dense


def memory_check(p: Plan) -> dict[str, Any]:
    """The estimate against the memory limit (the converter runs under the same
    SCVIEW_MEMORY_GB as the backend: docker-compose.yml)."""
    from scview.core.memory_budget import read_budget

    need = estimate_bytes(p)
    budget = read_budget()
    limit = budget.limit_bytes
    return {"needs_bytes": need, "limit_bytes": limit, "limit_source": budget.source,
            "fits": limit is None or need <= limit}


def summary(p: Plan, fmt: str) -> dict[str, Any]:
    """For the dialog: what goes in, what is left out, and whether it fits."""
    return {
        "format": fmt,
        "label": FORMATS[fmt]["label"],
        "needs": FORMATS[fmt]["needs"],
        "n_cells": int(p.obs.shape[0]),
        "n_genes": int(p.var.shape[0]),
        "data_from": p.x.where if p.x is not None else None,
        "counts_from": p.counts.where if p.counts is not None else None,
        "embeddings": list(p.obsm),
        "n_proteins": len(p.protein.get("names", [])),
        "cluster_column": p.meta.get("cluster_column"),
        "notes": p.notes,
        "dropped": p.dropped,
        "n_values": p.n_values,
        **memory_check(p),
    }


# ---------------------------------------------------------------------------
# The staging h5ad


def write_staging(p: Plan, out: Path, note: dict[str, Any],
                  progress: Callable[[str], None] = lambda _m: None) -> Path:
    """Write the plan as an h5ad the R side reads element by element."""
    import anndata as ad

    out = Path(out)
    tmp = out.with_name(f".{out.name}.part")
    meta = dict(p.meta)
    meta.update(note)
    meta["notes"] = p.notes
    uns: dict[str, Any] = {UNS_KEY: json.dumps(meta)}
    if p.protein:
        uns["protein_counts"] = p.protein["counts"]
        if "clr" in p.protein:
            uns["protein_clr"] = p.protein["clr"]
    varm = {"PCs": p.pcs} if p.pcs is not None else {}
    try:
        progress("Writing cell and gene tables")
        ad.AnnData(obs=p.obs, var=p.var, obsm=p.obsm, varm=varm, uns=uns).write_h5ad(tmp)
        with h5py.File(tmp, "r+") as f:
            if "X" in f:
                del f["X"]
            if p.x is not None:
                progress(f"Writing the log-normalised values (from {p.x.where})")
                cx._write_matrix(f, "X", p.x)
            if p.counts is not None:
                progress(f"Writing the raw counts (from {p.counts.where})")
                layers = f.require_group("layers")
                layers.attrs.setdefault("encoding-type", "dict")
                layers.attrs.setdefault("encoding-version", "0.1.0")
                cx._write_matrix(layers, "counts", p.counts)
        os.replace(tmp, out)
    finally:
        tmp.unlink(missing_ok=True)
    return out


# ---------------------------------------------------------------------------
# The converter


class ConverterDown(RuntimeError):
    """The converter container is not running: a 503."""


CONVERTER_DOWN = ("The R converter is not running, so .rds files cannot be written. Start it "
                  "(docker compose up -d converter) and try again.")


async def converter_state(converter_url: str, transport: httpx.AsyncBaseTransport | None = None) -> str:
    """ok | busy (it answers nothing while it converts) | down (nothing listens)."""
    timeout = httpx.Timeout(connect=3.0, read=5.0, write=5.0, pool=3.0)
    try:
        async with httpx.AsyncClient(timeout=timeout, transport=transport) as client:
            r = await client.get(f"{converter_url}/health")
        return "ok" if r.status_code == 200 else "down"
    except httpx.ReadTimeout:
        return "busy"
    except httpx.TransportError:
        return "down"


async def trigger_export(input_path: str, output_path: str, fmt: str, converter_url: str,
                         progress_path: str | None = None,
                         transport: httpx.AsyncBaseTransport | None = None) -> dict:
    """POST /export to the converter; its body (status, message, report), or
    ``conversion.ConversionError`` with its own sentence."""
    body = {"input_path": input_path, "output_path": output_path, "format": fmt}
    if progress_path:
        body["progress_path"] = progress_path
    try:
        async with httpx.AsyncClient(timeout=conversion.TIMEOUT, transport=transport) as client:
            resp = await client.post(f"{converter_url}/export", json=body)
    except httpx.TimeoutException as e:
        raise conversion.ConversionError(
            f"The R converter did not finish within {conversion.READ_TIMEOUT_S / 3600:.0f} hours.") from e
    except httpx.TransportError as e:
        raise conversion.ConversionError(CONVERTER_DOWN) from e
    try:
        data = resp.json()
    except ValueError:
        data = None
    if not isinstance(data, dict):
        data = {}
    if resp.status_code == 404 and not data.get("error", "").startswith("Input"):
        raise conversion.ConversionError(
            "This R converter cannot write .rds files yet: update scView so the converter "
            "image matches the backend.")
    if resp.status_code >= 400 or data.get("status") != "ok":
        reason = data.get("error")
        raise conversion.ConversionError(
            str(reason) if reason else f"The R converter answered {resp.status_code} without saying why.")
    return data


# ---------------------------------------------------------------------------
# Jobs


@dataclass
class Job:
    id: str
    dataset_id: str
    fmt: str
    folder: Path
    filename: str
    status: str = "writing"  # writing | converting | done | error
    message: str = "Preparing the data"
    error: str | None = None
    report: dict[str, Any] | None = None
    size_bytes: int | None = None
    created: float = field(default_factory=time.time)
    finished: float | None = None
    task: asyncio.Task | None = None

    @property
    def output(self) -> Path:
        return self.folder / OUTPUT

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {"job": self.id, "format": self.fmt, "status": self.status,
                               "message": self.message, "filename": self.filename}
        if self.status == "converting":
            prog = conversion.read_progress(self.folder / PROGRESS)
            if prog and prog.get("message"):
                out["message"] = str(prog["message"])
                out["stage"] = prog.get("stage")
        if self.error:
            out["error"] = self.error
        if self.report is not None:
            out["report"] = self.report
        if self.size_bytes is not None:
            out["size_bytes"] = self.size_bytes
        return out


def jobs_root(data_dir: Path) -> Path:
    return Path(data_dir) / "cache" / "rexport"


def get_job(job_id: str) -> Job | None:
    return _JOBS.get(job_id)


def running_for(dataset_id: str, fmt: str) -> Job | None:
    for job in _JOBS.values():
        if job.dataset_id == dataset_id and job.fmt == fmt and job.status in ("writing", "converting"):
            return job
    return None


def remove(job: Job) -> None:
    _JOBS.pop(job.id, None)
    shutil.rmtree(job.folder, ignore_errors=True)


def sweep(data_dir: Path, now: float | None = None, keep: float = KEEP_SECONDS) -> int:
    """Remove finished jobs and folders older than ``keep`` seconds (also those left by
    a restart, which no job refers to); the number of folders removed."""
    now = time.time() if now is None else now
    live = {j.folder for j in _JOBS.values() if j.status in ("writing", "converting")}
    for job in list(_JOBS.values()):
        if job.finished is not None and now - job.finished > keep:
            _JOBS.pop(job.id, None)
    root = jobs_root(data_dir)
    removed = 0
    if root.is_dir():
        for d in root.iterdir():
            try:
                age = now - d.stat().st_mtime
            except OSError:
                continue
            if d not in live and age > keep:
                shutil.rmtree(d, ignore_errors=True)
                removed += 1
    return removed


def new_job(data_dir: Path, dataset_id: str, fmt: str, filename: str) -> Job:
    job_id = uuid.uuid4().hex
    folder = jobs_root(data_dir) / job_id
    folder.mkdir(parents=True, exist_ok=True)
    job = Job(job_id, dataset_id, fmt, folder, filename)
    _JOBS[job_id] = job
    return job


def _report(result: dict) -> dict[str, Any]:
    """The converter's report; kept / skipped always lists (a single item comes as a string)."""
    rep = result.get("report") if isinstance(result.get("report"), dict) else {}
    for key in ("kept", "skipped"):
        val = rep.get(key, [])
        rep[key] = [val] if isinstance(val, str) else [str(v) for v in (val or [])]
    return rep


async def run(job: Job, write: Callable[[], Any], converter_url: str,
              trigger: Callable[..., Any] | None = None) -> None:
    """Write the staging file (``write``: awaitable, under the dataset's edit lock), ask
    the converter, then record the outcome. The staging file is removed either way."""
    staging = job.folder / STAGING
    try:
        await write()
        job.status, job.message = "converting", "Starting R"
        result = await (trigger or trigger_export)(
            str(staging), str(job.output), job.fmt, converter_url, progress_path=str(job.folder / PROGRESS))
        if not job.output.is_file():
            raise conversion.ConversionError("The R converter answered but wrote no file.")
        job.report = _report(result)
        job.size_bytes = job.output.stat().st_size
        job.status, job.message = "done", str(result.get("message") or "Written")
    except asyncio.CancelledError:
        job.status, job.error = "error", "The export was stopped."
        raise
    except Exception as e:  # noqa: BLE001 - any failure becomes the job's error
        logger.warning("R export %s of %s failed: %s", job.fmt, job.dataset_id, e)
        job.status, job.error = "error", str(e)
        job.message = "Failed"
    finally:
        job.finished = time.time()
        staging.unlink(missing_ok=True)
        if job.status != "done":
            job.output.unlink(missing_ok=True)
        try:
            asyncio.get_running_loop().call_later(KEEP_SECONDS, _expire, job)
        except RuntimeError:
            pass


def _expire(job: Job) -> None:
    if _JOBS.get(job.id) is job:
        remove(job)


def start(job: Job, coro) -> asyncio.Task:
    job.task = asyncio.get_running_loop().create_task(coro)
    return job.task
