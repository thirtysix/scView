"""Ingestion engine — stage [4] LOAD.

Read a validated, complete :class:`IngestUnit` into a canonical
:class:`anndata.AnnData` using the appropriate scanpy reader. The result is the
cells×genes (obs×var) AnnData the rest of scView expects; persistence into the
immutable ``ingested/{id}/`` layer and registration with the DatasetManager are
done by the ingest endpoints (stage [5]). Merging several units is stage merge
(§3a). See ``docs/INGESTION_ENGINE.md``.

10x MEX files may be renamed (GEO prefixes) or gzipped inconsistently, so they
are staged into a temp dir with the canonical names scanpy's ``read_10x_mtx``
expects, normalising the features file to the 3-column v3 layout on the way.

Dependency note: loom needs the ``loompy`` package and zarr a compatible
``zarr`` install — neither is currently in the backend image. Those branches are
implemented but will raise an informative error until the deps are added.
"""

from __future__ import annotations

import gzip
import logging
import shutil
import tempfile
from pathlib import Path

import anndata as ad
import pandas as pd
import scanpy as sc

from scview.core.ingestion.bundling import FileRole, IngestUnit, UnitFormat

logger = logging.getLogger(__name__)


class IngestLoadError(RuntimeError):
    """Raised when a unit cannot be read into an AnnData."""


def load_unit(unit: IngestUnit, options: dict | None = None) -> ad.AnnData:
    """Read one complete unit into an AnnData (cells × genes)."""
    options = options or {}
    if not unit.complete:
        raise IngestLoadError(
            f"Unit '{unit.label}' is not complete; resolve validation issues before loading."
        )

    loaders = {
        UnitFormat.tenx_mex: _load_mex,
        UnitFormat.tenx_h5: _load_tenx_h5,
        UnitFormat.anndata: _load_anndata,
        UnitFormat.loom: _load_loom,
        UnitFormat.zarr: _load_zarr,
        UnitFormat.dense_table: _load_dense,
    }
    loader = loaders.get(unit.format)
    if loader is None:
        if unit.format == UnitFormat.seurat:
            raise IngestLoadError(
                "Seurat .rds files are converted by the R converter service, not this loader."
            )
        raise IngestLoadError(f"No loader for unit format '{unit.format}'.")

    try:
        adata = loader(unit, options)
    except IngestLoadError:
        raise
    except Exception as e:  # surface a clean error to the ingest layer
        logger.exception("Failed to load unit '%s' (%s)", unit.label, unit.format)
        raise IngestLoadError(f"Could not read this {unit.format.value} dataset: {e}") from e

    adata.var_names_make_unique()
    adata.obs_names_make_unique()
    return adata


# ---------------------------------------------------------------------------
# Per-format loaders
# ---------------------------------------------------------------------------


def _load_mex(unit: IngestUnit, options: dict) -> ad.AnnData:
    matrix = _path_for_role(unit, FileRole.matrix)
    barcodes = _path_for_role(unit, FileRole.barcodes)
    features = _path_for_role(unit, FileRole.features)
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        _stage_gzipped(matrix, d / "matrix.mtx.gz")
        _stage_gzipped(barcodes, d / "barcodes.tsv.gz")
        _stage_features(features, d / "features.tsv.gz")
        return sc.read_10x_mtx(d, var_names="gene_symbols", gex_only=False)


def _load_tenx_h5(unit: IngestUnit, options: dict) -> ad.AnnData:
    return sc.read_10x_h5(_single(unit))


def _load_anndata(unit: IngestUnit, options: dict) -> ad.AnnData:
    return sc.read_h5ad(_single(unit))


def _load_loom(unit: IngestUnit, options: dict) -> ad.AnnData:
    try:
        return sc.read_loom(_single(unit))
    except ImportError as e:
        raise IngestLoadError(
            "Reading .loom files requires the 'loompy' package, which isn't installed."
        ) from e


def _load_zarr(unit: IngestUnit, options: dict) -> ad.AnnData:
    return ad.read_zarr(_single(unit))


def _load_dense(unit: IngestUnit, options: dict) -> ad.AnnData:
    """Read a dense table; orient to cells × genes.

    ``options['genes_in_rows']`` (default True — most expression CSVs are
    genes × cells) controls transposition. The wizard resolves this with the
    user; the default keeps the common case working headlessly.
    """
    path = Path(_single(unit))
    # sep=None + python engine auto-detects comma/tab; gz handled by extension.
    df = pd.read_csv(path, sep=None, engine="python", index_col=0)
    if options.get("genes_in_rows", True):
        df = df.T  # genes × cells → cells × genes
    adata = ad.AnnData(df.to_numpy(dtype="float32"))
    adata.obs_names = df.index.astype(str)
    adata.var_names = df.columns.astype(str)
    return adata


# ---------------------------------------------------------------------------
# Staging helpers (10x MEX → canonical names)
# ---------------------------------------------------------------------------


def _stage_gzipped(src: Path, dst: Path) -> None:
    """Copy src to dst as gzip (compressing if the source isn't already gzipped)."""
    if src.suffix.lower() == ".gz":
        shutil.copyfile(src, dst)
    else:
        with open(src, "rb") as fi, gzip.open(dst, "wb") as fo:
            shutil.copyfileobj(fi, fo)


def _stage_features(src: Path, dst: Path) -> None:
    """Normalise a features/genes file to the 3-column v3 layout, gzipped."""
    with _open_text(src) as fi:
        rows = [ln.rstrip("\n").split("\t") for ln in fi if ln.strip()]
    with gzip.open(dst, "wt", encoding="utf-8") as fo:
        for r in rows:
            if len(r) >= 3:
                gid, sym, ftype = r[0], r[1], r[2]
            elif len(r) == 2:
                gid, sym, ftype = r[0], r[1], "Gene Expression"
            else:  # 1-column gene list: id == symbol
                gid = sym = r[0]
                ftype = "Gene Expression"
            fo.write(f"{gid}\t{sym}\t{ftype}\n")


def _open_text(path: Path):
    if path.suffix.lower() == ".gz":
        return gzip.open(path, "rt", encoding="utf-8", errors="replace")
    return open(path, "rt", encoding="utf-8", errors="replace")


# ---------------------------------------------------------------------------
# Unit file lookups
# ---------------------------------------------------------------------------


def _path_for_role(unit: IngestUnit, role: FileRole) -> Path:
    bf = next((f for f in unit.files if f.role == role), None)
    if bf is None:
        raise IngestLoadError(f"Unit '{unit.label}' has no {role.value} file.")
    return Path(bf.path)


def _single(unit: IngestUnit) -> str:
    if not unit.files:
        raise IngestLoadError(f"Unit '{unit.label}' has no file to load.")
    return unit.files[0].path
