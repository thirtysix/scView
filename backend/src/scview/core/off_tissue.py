"""Spots outside the tissue (obs ``in_tissue`` 0) in a spatial .h5ad.

Space Ranger marks each spot of the capture grid ``in_tissue`` 1 or 0, and its reader
here keeps the tissue spots only (``core/spatial.py``). A CELLxGENE Visium file holds the
whole grid instead: on Kuppe 2022's control_P1, 713 of 4,992 spots are off the tissue,
with no expression and NaN in every author column. Left in, they make housekeeping genes
look spatially variable (MALAT1 Moran's I 0.88: the empty block against the tissue), and
a joint embedding normalises and clusters them.

At ingest (a plain .h5ad upload, ``api/v1/datasets.py``; a public import,
``core/public_import.py``) the working copy in ``ingested/{id}/`` leaves them out
(``drop_in_file``): the upload or download stays as sent, and the ingest step records
how many (provenance ``ingest``, ``off_tissue``; the dataset's metadata). A dataset
imported before keeps its cell count: the Data Assessment and the spatial analyses
say how many off-tissue spots it holds (``note``) instead of dropping them on open.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

COLUMN = "in_tissue"
_SPATIAL_KEYS = ("spatial", "X_spatial")
_OFF = {"0", "0.0", "false"}
_ON = {"1", "1.0", "true"}
# The rewrite reads the tissue spots into memory: only when that is well inside what is
# free (a Visium slide: 5k spots, about 0.2 GB); a larger file keeps them, and says so.
FITS_SHARE = 0.5


def off_mask(values: Any) -> np.ndarray | None:
    """True for each off-tissue row of an ``in_tissue`` column; None unless every value
    is 0 or 1 (ints, floats, booleans or their text)."""
    s = pd.Series(np.asarray(values)).astype(str).str.strip().str.lower()
    if not s.isin(_OFF | _ON).all():
        return None
    return s.isin(_OFF).to_numpy()


def count(adata) -> int:
    """Off-tissue rows of a spatial dataset (0 when it has none, or is not spatial)."""
    from scview.core import spatial

    if COLUMN not in adata.obs.columns or not spatial.has_spatial(adata):
        return 0
    mask = off_mask(adata.obs[COLUMN])
    return int(mask.sum()) if mask is not None else 0


def note(adata, unit: str = "spot") -> str | None:
    """What a result on this dataset says about its off-tissue rows, or None."""
    n = count(adata)
    if not n:
        return None
    return (f"{n:,} {unit}{'' if n == 1 else 's'} marked off-tissue (in_tissue 0) are included: they lie "
            f"outside the tissue and hold little or no expression, so they can make genes look spatially "
            f"patterned. Filtering removes the empty ones; importing the file again leaves them all out.")


def file_count(path: Path) -> tuple[int, int] | None:
    """(off-tissue rows, all rows) of an .h5ad with spatial coordinates and an
    ``in_tissue`` column of 0s and 1s, read from its obs alone; None otherwise."""
    import h5py
    from anndata.io import read_elem

    try:
        with h5py.File(path, "r") as f:
            obsm = f.get("obsm")
            if not isinstance(obsm, h5py.Group) or not any(k in obsm for k in _SPATIAL_KEYS):
                return None
            obs = f.get("obs")
            if not isinstance(obs, h5py.Group) or COLUMN not in obs:
                return None
            mask = off_mask(read_elem(obs[COLUMN]))
    except (OSError, KeyError, ValueError, TypeError) as e:
        logger.warning("Could not read %s of %s: %s", COLUMN, path, e)
        return None
    if mask is None:
        return None
    return int(mask.sum()), int(mask.size)


def _fits(path: Path) -> bool:
    from scview.core import preflight

    avail = preflight.available_memory()
    if avail.available_bytes is None:
        return True
    try:
        peak = preflight.estimate_load(preflight.read_header(path, "h5ad"), ingest=False).peak_bytes
    except Exception as e:  # noqa: BLE001 - no estimate: the file keeps its spots
        logger.warning("No load estimate for %s: %s", path, e)
        return False
    return peak <= FITS_SHARE * avail.available_bytes


def drop_in_file(src: Path, dst: Path, *, original_filename: str) -> dict | None:
    """Write ``dst``: ``src`` without its off-tissue rows, with an ingest step that says
    how many (``off_tissue``: removed, kept, column). None when there are none (``dst`` not
    written). A file too large to rewrite in the memory free keeps them: the summary says
    so (``kept_reason``) and ``dst`` is not written. ``src`` is only read."""
    import anndata as ad

    from scview.core import provenance

    counted = file_count(src)
    if counted is None or counted[0] == 0:
        return None
    n_off, n = counted
    if n_off == n:
        return {"removed": 0, "kept": n, "column": COLUMN,
                "kept_reason": "every row is marked off-tissue, so none was left out"}
    if not _fits(src):
        return {"removed": 0, "kept": n, "column": COLUMN, "off_tissue": n_off,
                "kept_reason": "the file is too large to rewrite in the memory free"}
    a = ad.read_h5ad(src, backed="r")
    try:
        keep = np.flatnonzero(~off_mask(a.obs[COLUMN]))
        sub = a[keep].to_memory()
    finally:
        a.file.close()
    summary = {"removed": n_off, "kept": int(keep.size), "column": COLUMN}
    provenance.init_source(sub, origin="uploaded", original_filename=original_filename, fmt="h5ad")
    provenance.record_step(sub, step="ingest", tool="scview.off_tissue",
                           params={"formats": ["h5ad"], "off_tissue": summary})
    sub.write_h5ad(dst, compression="gzip")
    logger.info("Left out %d off-tissue rows of %d (%s)", n_off, n, original_filename)
    return summary
