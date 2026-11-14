"""Reset to counts from the cells as uploaded (``PipelineParams.reset_from_original``).

``reset_to_counts`` restores the counts of the cells the dataset has now, so a dataset filtered
once can only lose more cells: on the 10x mouse kidney multiome, a re-run with looser QC kept the
4,012 cells the first filtering left and trimmed 17 more, and the only way back was a second
upload. With ``reset_from_original`` the run starts from the stored original instead (the import,
conversion or upload under ``uploads/`` / ``ingested/``, never written to; the file
``mito_refilter`` re-filters from), so the new filtering judges every cell uploaded.

What comes along: the dataset's history (the run's steps are recorded after it), and the ATAC QC
columns fragment QC wrote (``atac_*``) for the cells that had them; the others are blank until the
fragment QC runs again. What does not: anything made in scView that is not a pipeline step
(labels, gates, annotations), as with the mito re-filter. A dataset without a stored original
(a joint embedding, a subset made in scView) is refused with a sentence, never reset from the
cells it has. Large datasets are refused too: the original is read into memory.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Generator
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd

from scview.core import provenance, qc_filter

logger = logging.getLogger(__name__)

MARK = "scview_reset_from_original"  # uns, read and removed by the reset_to_counts step
CARRY_PREFIXES = ("atac_",)  # obs columns of the dataset carried onto the original's cells
NO_ORIGINAL = ("No original upload is stored for this dataset (it was made in scView, e.g. a joint "
               "embedding or a subset), so it can't be reset to the cells as uploaded: reset to counts "
               "keeps the cells it has.")
NEEDS_RESET = "reset_from_original starts the reset_to_counts step from the original upload: include that step."


class ResetError(ValueError):
    """A reset from the original that cannot run; ``status`` is the HTTP status."""

    def __init__(self, message: str, status: int = 409) -> None:
        super().__init__(message)
        self.status = status


def source(dm, dataset_id: str) -> str:
    """The stored original (``mito_refilter.source_file``); ResetError without one."""
    from scview.core import mito_refilter

    src = mito_refilter.source_file(dm, dataset_id)
    if src is None:
        raise ResetError(NO_ORIGINAL)
    return src


def check(dm, dataset_id: str, steps: list[str]) -> str:
    """The original's path, after the checks made before anything is read."""
    if "reset_to_counts" not in steps:
        raise ResetError(NEEDS_RESET, status=422)
    return source(dm, dataset_id)


def read_need(dm, dataset_id: str) -> int:
    """Memory to read the original whole (``working_copy.file_bytes``); 0 without an estimate."""
    from scview.core import mito_refilter

    return mito_refilter.read_need(dm, dataset_id)


def step_needs(dm, dataset_id: str, steps: list[str], params) -> dict[str, int]:
    """Per step, what the run needs beyond the original held in memory (``preflight.estimate_run``
    on the original's header): reserved as any run's growth (core/processing.py)."""
    from scview.core import preflight

    h = preflight.read_header(source(dm, dataset_id), "h5ad")
    if hasattr(h, "backed"):
        h.backed = False  # read whole, whatever its size
    est = preflight.estimate_run(h, steps=steps, params=params, warm=preflight.process_warm())
    return {e.step: max(0, int(e.peak_bytes - est.start_bytes)) for e in est.steps}


def carry_columns(cur: ad.AnnData, orig: ad.AnnData) -> list[str]:
    """The dataset's ATAC QC columns (``CARRY_PREFIXES``, numbers) the original lacks, put on the
    original's cells by name (NaN for a cell the dataset no longer has); their names."""
    carried: list[str] = []
    idx = cur.obs_names.astype(str).get_indexer(orig.obs_names.astype(str))
    hit = idx >= 0
    for col in cur.obs.columns:
        if str(col).startswith(CARRY_PREFIXES) and col not in orig.obs.columns \
                and pd.api.types.is_numeric_dtype(cur.obs[col].dtype):
            values = np.full(orig.n_obs, np.nan)
            values[hit] = cur.obs[col].to_numpy(dtype=np.float64)[idx[hit]]
            orig.obs[col] = values
            carried.append(str(col))
    return carried


def load(dm, dataset_id: str, adaptor) -> ad.AnnData:
    """The original upload, ready for the run: the dataset's history, the ATAC QC columns the
    dataset has, no removal record (the filtering starts afresh), and the mark the
    ``reset_to_counts`` step records (cells before and as uploaded)."""
    src = source(dm, dataset_id)
    cur = adaptor.adata
    orig = ad.read_h5ad(src)
    orig.obs_names = orig.obs_names.astype(str)
    carried = carry_columns(cur, orig)
    hit = cur.obs_names.astype(str).get_indexer(orig.obs_names) >= 0
    prov = provenance.read_provenance(cur)
    prov["current"] = {k: v for k, v in prov.get("current", {}).items() if k == "head"}
    orig.uns[provenance.UNS_KEY] = json.dumps(prov)
    orig.uns.pop(qc_filter.RECORD_KEY, None)
    mark = {"from_original": True, "source_file": Path(src).name, "n_original": int(orig.n_obs),
            "n_before": int(cur.n_obs), "n_in_dataset": int(hit.sum())}
    if carried:
        mark.update(carried_columns=carried, n_carried=int(hit.sum()))
    orig.uns[MARK] = json.dumps(mark)
    return orig


def take_mark(adata) -> dict | None:
    """The mark ``load`` left (removed from ``adata``), or None for an ordinary reset."""
    raw = adata.uns.pop(MARK, None)
    if raw is None:
        return None
    try:
        mark = json.loads(str(raw))
    except (TypeError, ValueError):
        return None
    if not isinstance(mark, dict):
        return None
    if mark.get("carried_columns"):
        mark["note"] = (f"{', '.join(mark['carried_columns'])} came along for the {mark['n_carried']:,} cells "
                        "that had them; the others are blank until the fragment QC runs again.")
    return mark


def run_events(dm, dataset_id: str, adaptor, steps: list[str], params, output_path: str,
               control=None) -> Generator[tuple[str, dict], None, None]:
    """The run from the original upload, as ``run_pipeline_streamed``'s events after one of its
    own (``reset``: cells now and as uploaded)."""
    from scview.core.pipeline import run_pipeline_streamed

    A = load(dm, dataset_id, adaptor)
    mark = json.loads(A.uns[MARK])
    yield ("reset", {"stage": "load", "message": "Read the original upload",
                     "n_original": mark["n_original"], "n_before": mark["n_before"]})
    yield from run_pipeline_streamed(A, steps, params, output_path, control=control)


def run(dm, dataset_id: str, adaptor, steps: list[str], params, output_path: str, control=None):
    """The run from the original upload, not streamed: its ``PipelineResult``."""
    from scview.core.pipeline import run_pipeline

    A = load(dm, dataset_id, adaptor)
    return run_pipeline(A, steps, params, output_path, control)[1]
