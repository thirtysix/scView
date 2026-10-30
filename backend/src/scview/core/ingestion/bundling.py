"""Ingestion engine — stage [2] BUNDLE.

Group detected files into *units* and a *bundle*:

* **Unit** = one experiment/sample. A 10x MEX unit is up to three files
  (matrix + barcodes + features/genes); every other format is a single file or
  store directory.
* **Bundle** = the output of one ingest session: a single unit, or several units
  to be merged (a ``sample`` label is added at merge time — see §3a of the design).

The bundler runs detection (stage [1]) on each path, assigns a role, groups the
10x MEX parts by GEO-style prefix / parent directory, and reports completeness
— precisely which companion files a partial unit is still missing. It performs
no loading and has no side effects beyond reading. See ``docs/INGESTION_ENGINE.md``.
"""

from __future__ import annotations

import logging
import re
from enum import Enum
from pathlib import Path, PurePosixPath

from pydantic import BaseModel, Field

from scview.core import spatial, xenium
from scview.core.ingestion.detection import DetectionResult, FileKind, detect_file
from scview.core.ingestion.paths import junk_reason, skipped_sentence

logger = logging.getLogger(__name__)


class UnitFormat(str, Enum):
    """The recipe by which a unit will be loaded."""

    tenx_mex = "tenx_mex"
    tenx_h5 = "tenx_h5"
    anndata = "anndata"
    loom = "loom"
    zarr = "zarr"
    dense_table = "dense_table"
    seurat = "seurat"  # an R object: converted by the R converter at commit (session.py)
    h5seurat = "h5seurat"  # SeuratDisk .h5Seurat, read natively (h5seurat.py)
    xenium = "xenium"  # a Xenium output folder: its matrix + side files (core/xenium.py)
    unknown = "unknown"


class FileRole(str, Enum):
    """A file's role within its unit."""

    matrix = "matrix"
    barcodes = "barcodes"
    features = "features"
    data = "data"  # the single file of a single-file format
    side = "side"  # attached to a unit, never loaded as its cells (``side_kind`` says what)
    unknown = "unknown"


class SideKind(str, Enum):
    """What a side file of a unit is."""

    metrics = "metrics"  # Cell Ranger metrics_summary.csv (core/seq_metrics.py)
    raw = "raw"  # the raw (unfiltered) matrix: a .h5 or MEX parts (core/seq_metrics.py)
    vdj = "vdj"  # a TCR/BCR contig table, joined to the unit's cells (core/vdj.py)
    crispr = "crispr"  # Cell Ranger's guide calls per cell, joined to the unit's cells (core/crispr.py)
    spatial = "spatial"  # a Space Ranger spatial file (positions, scale factors, images; core/spatial.py)
    xenium = "xenium"  # experiment.xenium, cell table, image, its metrics (core/xenium.py)


# A 10x MEX unit needs exactly one of each of these.
_MEX_REQUIRED: tuple[FileRole, ...] = (FileRole.matrix, FileRole.barcodes, FileRole.features)

# FileKind → (UnitFormat for a single-file unit, FileRole). MEX parts map to a
# role but no standalone format (they only exist inside a tenx_mex unit).
_KIND_MAP: dict[FileKind, tuple[UnitFormat | None, FileRole]] = {
    FileKind.anndata_h5ad: (UnitFormat.anndata, FileRole.data),
    FileKind.tenx_h5: (UnitFormat.tenx_h5, FileRole.data),
    FileKind.loom: (UnitFormat.loom, FileRole.data),
    FileKind.zarr_dir: (UnitFormat.zarr, FileRole.data),
    FileKind.dense_table: (UnitFormat.dense_table, FileRole.data),
    FileKind.seurat_rds: (UnitFormat.seurat, FileRole.data),
    FileKind.h5seurat: (UnitFormat.h5seurat, FileRole.data),
    FileKind.mtx_matrix: (None, FileRole.matrix),
    FileKind.tenx_barcodes: (None, FileRole.barcodes),
    FileKind.tenx_features: (None, FileRole.features),
    FileKind.vdj_contigs: (None, FileRole.side),
    FileKind.crispr_calls: (None, FileRole.side),
    FileKind.spatial: (None, FileRole.side),
    FileKind.xenium: (None, FileRole.side),
    FileKind.unknown: (None, FileRole.unknown),
}

# Tokens stripped when deriving a 10x MEX grouping prefix from a filename.
_STRIP_EXTS = (".gz", ".mtx", ".tsv", ".csv", ".txt")
_ROLE_TOKENS = ("matrix", "barcodes", "features", "genes")

# Directory names that are NOT sample-specific (CellRanger conventions + our
# upload-staging dir), so they make poor default labels for a prefix-less MEX.
_GENERIC_MEX_DIRS = {
    "files", "filtered_feature_bc_matrix", "raw_feature_bc_matrix", "outs", "matrix",
}


class BundleFile(BaseModel):
    """One file placed into a unit."""

    path: str
    name: str
    kind: FileKind
    role: FileRole
    confidence: float = 0.0
    side_kind: SideKind | None = None  # a side file's kind (role == side)
    receptor: str | None = None  # contig tables: "T", "B" or "T+B" (their chains, else the name)
    # Space Ranger spatial files: positions | scalefactors | hires | lowres | archive;
    # a Xenium bundle's side files: experiment | cells | image | metrics.
    spatial_part: str | None = None
    # The path inside the uploaded folder (``outs/raw_feature_bc_matrix/barcodes.tsv.gz``);
    # the name alone for a plain upload.
    rel_path: str = ""


class SkippedFile(BaseModel):
    """A staged or sent file scView does not read, and why."""

    path: str
    reason: str
    bytes: int | None = None


class IngestUnit(BaseModel):
    """One experiment/sample, possibly still incomplete."""

    label: str
    format: UnitFormat
    files: list[BundleFile] = Field(default_factory=list)
    complete: bool = False
    missing_roles: list[FileRole] = Field(default_factory=list)
    issues: list[str] = Field(default_factory=list)
    # Never loaded as cells: metrics_summary.csv and the raw matrix (sequencing QC,
    # core/seq_metrics.py) and TCR/BCR contig tables (core/vdj.py); ``side_kind`` on each.
    side_files: list[BundleFile] = Field(default_factory=list)


class Bundle(BaseModel):
    """The set of units staged in one ingest session."""

    units: list[IngestUnit] = Field(default_factory=list)
    is_merge: bool = False  # more than one loadable unit
    complete: bool = False  # every (non-unknown) unit is complete
    issues: list[str] = Field(default_factory=list)
    skipped: list[SkippedFile] = Field(default_factory=list)  # also one sentence in issues
    # A spatial upload scView does not read (a Xenium .zip, Visium HD's 2 um bins, its archive,
    # its segmented outputs): an error at validation.
    unsupported: str | None = None


# nf-core/scrnaseq's mtx_conversions/ emits "<sample>_{raw,filtered,cellbender_filter}_matrix.h5ad"
# (+ "combined_matrix.h5ad" for the all-samples concat). Strip these so the dataset
# label is the sample name rather than the verbose nf-core filename.
_NFCORE_MATRIX_SUFFIXES = (
    "_raw_matrix",
    "_filtered_matrix",
    "_cellbender_filter_matrix",
    "_unfiltered_matrix",
    "_matrix",
)


def _clean_unit_label(stem: str) -> str:
    """Best label for a single-file unit, cleaning nf-core matrix-export suffixes."""
    if stem == "combined_matrix":
        return "combined"
    for suffix in _NFCORE_MATRIX_SUFFIXES:
        if stem.endswith(suffix) and len(stem) > len(suffix):
            return stem[: -len(suffix)]
    return stem


def build_bundle(
    paths: list[Path], root: Path | None = None, skipped: list[dict] | None = None
) -> Bundle:
    """Detect, role-assign and group a set of staged paths into a Bundle.

    ``root``: the staging folder, so each file keeps its path inside an uploaded
    folder (``BundleFile.rel_path``), whose subfolders then group the files; without
    it every file counts by its name alone (unless ``paths`` is the session's
    ``staged_paths``, which carries its folder). ``skipped``: files left out at
    upload, listed in the bundle's one sentence about what was skipped."""
    if root is None:
        root = getattr(paths, "root", None)
    paths = [Path(p) for p in paths]
    rels = {p: _rel(p, root) for p in paths}
    skipped = list(skipped or [])
    detected = []
    extras = xenium.skip_reasons(list(rels.values()))
    for p in paths:
        reason = junk_reason(rels[p]) or extras.get(rels[p])
        if reason:
            skipped.append({"path": rels[p], "reason": reason})
            continue
        detected.append((p, detect_file(p)))
    return bundle_from_detections(detected, rels, skipped)


def _rel(path: Path, root: Path | None) -> str:
    if root is not None:
        try:
            return path.relative_to(root).as_posix()
        except ValueError:
            pass
    return path.name


def bundle_from_detections(
    detected: list[tuple[Path, DetectionResult]],
    rels: dict[Path, str] | None = None,
    skipped: list[dict] | None = None,
) -> Bundle:
    """Group already-detected files (split out for testing / reuse). ``rels``: each
    path's place in an uploaded folder (default: its name)."""
    rels = rels or {}
    skipped = list(skipped or [])

    def rel(p: Path) -> str:
        return rels.get(p) or p.name

    zipped = any(det.details.get("zip") for _, det in detected)
    detected, xenium_units = _take_xenium(detected, rel, skipped)
    refusal = spatial.unsupported([rel(p) for p, _ in detected])
    if zipped:
        refusal = refusal or spatial.XENIUM_MESSAGE
    detected = _select_hd_bins(detected, rel, skipped)
    detected = _drop_tree_extras(detected, rel, skipped)
    single_files: list[BundleFile] = []
    mex_parts: list[tuple[Path, BundleFile]] = []
    unknown_files: list[BundleFile] = []

    side_files: list[BundleFile] = []
    vdj_files: list[BundleFile] = []
    crispr_files: list[BundleFile] = []
    spatial_files: list[BundleFile] = []

    for path, det in detected:
        refusal = refusal or _hd_refusal(path, det)
        if _is_side_file(det):
            kind = SideKind.metrics if det.kind == FileKind.tenx_metrics else SideKind.raw
            side_files.append(BundleFile(path=str(path), name=path.name, kind=det.kind,
                                         role=FileRole.side, confidence=det.confidence,
                                         side_kind=kind, rel_path=rel(path)))
            continue
        fmt, role = _KIND_MAP.get(det.kind, (None, FileRole.unknown))
        bf = BundleFile(
            path=str(path), name=path.name, kind=det.kind, role=role, confidence=det.confidence,
            rel_path=rel(path),
        )
        if role == FileRole.data and fmt is not None:
            single_files.append(bf)
        elif role in _MEX_REQUIRED:
            mex_parts.append((path, bf))
        elif det.kind == FileKind.vdj_contigs:
            vdj_files.append(bf.model_copy(update={
                "side_kind": SideKind.vdj,
                "receptor": det.details.get("receptor") or _vdj_dir_arm(bf) or _vdj_arm(path.name)}))
        elif det.kind == FileKind.crispr_calls:
            crispr_files.append(bf.model_copy(update={"side_kind": SideKind.crispr}))
        elif det.kind == FileKind.spatial:
            spatial_files.append(bf.model_copy(update={
                "side_kind": SideKind.spatial, "spatial_part": det.details.get("part")}))
        else:
            unknown_files.append(bf)

    units: list[IngestUnit] = list(xenium_units)
    for bf in single_files:
        fmt, _ = _KIND_MAP[bf.kind]
        units.append(
            IngestUnit(
                label=_single_label(bf),
                format=fmt,  # type: ignore[arg-type]
                files=[bf],
                complete=True,
            )
        )

    units.extend(_group_mex_units(mex_parts))
    units = _drop_mex_copies(units, side_files, skipped)
    units, raw_mex = _split_raw_mex(units)

    bundle_issues: list[str] = _attach_side_files(units, side_files, raw_mex)
    if unknown_files:
        names = ", ".join(f.rel_path or f.name for f in unknown_files)
        bundle_issues.append(
            f"{len(unknown_files)} file(s) could not be recognised and were ignored: {names}"
        )

    bundle_issues.extend(_attach_vdj_files(units, vdj_files))
    bundle_issues.extend(_attach_crispr_files(units, crispr_files))
    bundle_issues.extend(_attach_spatial_files(units, spatial_files))
    _unique_labels(units)
    if skipped:
        bundle_issues.insert(0, skipped_sentence(skipped))

    loadable = [u for u in units if u.format != UnitFormat.unknown]
    return Bundle(
        units=sorted(units, key=lambda u: u.label),
        is_merge=len(loadable) > 1,
        complete=bool(loadable) and all(u.complete for u in loadable),
        issues=bundle_issues,
        skipped=[SkippedFile(**s) for s in skipped],
        unsupported=refusal,
    )


def _hd_refusal(path: Path, det: DetectionResult) -> str | None:
    """A Visium HD file whose content (not its folder) says it cannot be read: an archive
    of the binned outputs, or bins finer than ``HD_MIN_BIN_UM``."""
    if det.details.get("hd_archive"):
        return spatial.HD_ARCHIVE.format(name=path.name)
    um = det.details.get("hd_bin_um")
    return spatial.too_fine(um) if um is not None and um < spatial.HD_MIN_BIN_UM else None


def _select_hd_bins(detected, rel, skipped: list[dict]):
    """Visium HD: one bin size of each run is read (``spatial.hd_chosen_bins``; the wizard
    sends only the one the user chose). The run's other bin sizes and the rest of its output
    folder (``outs/spatial``, ``segmented_outputs``...) are recorded in ``skipped``."""
    chosen = spatial.hd_chosen_bins(rel(p) for p, _ in detected)
    if not chosen:
        return detected
    prefixes = [run + "/" for run in chosen if run]
    keep = []
    for p, det in detected:
        hit = spatial.hd_folder(rel(p))
        if hit is not None:
            um = chosen[hit[0]]
            if hit[1] == um or um < spatial.HD_MIN_BIN_UM:
                keep.append((p, det))
            else:
                skipped.append({"path": rel(p), "reason": f"another bin size of the run: one dataset "
                                f"per bin size, the {um} um bins are read"})
        elif any(rel(p).startswith(pre) for pre in prefixes):
            skipped.append({"path": rel(p), "reason": "the rest of the Space Ranger output folder, "
                            "not part of the bins read"})
        else:
            keep.append((p, det))
    return keep


# ---------------------------------------------------------------------------
# Xenium output folders: one unit each (core/xenium.py)
# ---------------------------------------------------------------------------


def _take_xenium(detected, rel, skipped: list[dict]):
    """The Xenium bundles among the files, each one unit, and the files left for the rest
    of the bundler. Taken before anything else looks at them: their matrix is no sample of
    its own, and their ``metrics_summary.csv`` is not Cell Ranger's."""
    by_rel = {rel(p): (p, det) for p, det in detected}
    parts = {r: det.details["part"] for r, (_, det) in by_rel.items()
             if det.kind == FileKind.xenium and det.details.get("part")}
    groups = xenium.groups(list(by_rel), parts)
    if not groups:
        return detected, []
    taken: set[str] = set()
    units = []
    for g in groups:
        taken |= g.taken()
        skipped.extend({"path": path, "reason": reason} for path, reason in g.skipped)
        units.append(_xenium_unit(g, by_rel))
    return [(p, det) for p, det in detected if rel(p) not in taken], units


def _xenium_unit(g: xenium.Group, by_rel: dict) -> IngestUnit:
    def bf(path: str, **kw) -> BundleFile:
        p, det = by_rel[path]
        return BundleFile(path=str(p), name=p.name, kind=det.kind, confidence=det.confidence,
                          rel_path=path, **kw)

    if g.h5:
        files = [bf(g.h5, role=FileRole.data)]
    else:
        files = sorted((bf(path, role=FileRole(role)) for role, path in g.mex.items()),
                       key=lambda f: f.role.value)
    side = [bf(path, role=FileRole.side, side_kind=SideKind.xenium, spatial_part=part)
            for part, path in ((xenium.PART_EXPERIMENT, g.experiment), (xenium.PART_CELLS, g.cells),
                               (xenium.PART_IMAGE, g.image), (xenium.PART_METRICS, g.metrics)) if path]
    label = xenium.region_name(by_rel[g.experiment][0]) if g.experiment else ""
    label = label or _dir_label(f"{g.root}/x" if g.root else "x") or "xenium"
    missing = g.missing()
    issues = []
    if missing:
        where = f" in {g.root}" if g.root else ""
        issues.append(f"This Xenium output{where} is missing {' and '.join(missing)}: upload the run's "
                      "folder (outs/) whole.")
    return IngestUnit(label=label, format=UnitFormat.xenium, files=files, side_files=side,
                      complete=not missing, issues=issues)


# ---------------------------------------------------------------------------
# Folders: what a subfolder says about its files (folder uploads only; a flat
# upload has no folders, so none of this applies to it)
# ---------------------------------------------------------------------------

# Cell Ranger's own folder names: they say what a file is, never which sample.
_GENERIC_DIRS = {
    "outs", "count", "counts", "multi", "per_sample_outs", "matrix", "gex", "gene_expression",
    "filtered_feature_bc_matrix", "raw_feature_bc_matrix", "sample_filtered_feature_bc_matrix",
    "sample_raw_feature_bc_matrix", "filtered_gene_bc_matrices", "raw_gene_bc_matrices",
    "vdj_t", "vdj_b", "vdj_t_gd", "vdj", "spatial",
}
# Folders that hold one sample's matrix / contigs, below that sample's own folder.
_CONTAINER_DIRS = {
    "count", "filtered_feature_bc_matrix", "raw_feature_bc_matrix", "crispr_analysis",
    "sample_filtered_feature_bc_matrix", "sample_raw_feature_bc_matrix",
    "filtered_gene_bc_matrices", "raw_gene_bc_matrices", "vdj_t", "vdj_b", "vdj_t_gd",
    "spatial",  # Space Ranger's spatial/ folder: positions, scale factors, images
}
_GENOME_PARENTS = {"filtered_gene_bc_matrices", "raw_gene_bc_matrices"}  # Cell Ranger 2: <dir>/GRCh38/
_VDJ_DIRS = {"vdj_t": "T", "vdj_t_gd": "T", "vdj_b": "B"}
# Cell Ranger's secondary analysis and reference folders: recomputed by scView, and their
# CSVs (PCA, t-SNE, DE, tag calls) would read as expression tables, i.e. fake samples.
_ANALYSIS_DIRS = {"multiplexing_analysis", "antibody_analysis", "crispr_analysis", "vdj_reference"}
_CR_CONTIGS = ("filtered_contig_annotations", "airr_rearrangement", "all_contig_annotations")

ANALYSIS = "Cell Ranger's own analysis (clustering, PCA, t-SNE/UMAP, DE): scView computes these itself"
VDJ_EXTRA = "Cell Ranger V(D)J outputs other than the contig table"


def _dirs(rel: str) -> tuple[str, ...]:
    return PurePosixPath(rel).parent.parts


def _is_analysis(dirs: tuple[str, ...]) -> bool:
    low = [d.lower() for d in dirs]
    return any(d in _ANALYSIS_DIRS or (d == "analysis" and i > 0 and low[i - 1] in ("outs", "count"))
               for i, d in enumerate(low))


def _drop_tree_extras(detected, rel, skipped: list[dict]):
    """Leave out what a Cell Ranger output folder holds beside its matrices and contigs:
    its own analysis (``outs/analysis/``), and in a V(D)J folder (``vdj_t``, ``vdj_b``,
    or a cellranger vdj ``outs`` with no matrix) everything but one contig table: the
    filtered one, else the AIRR one, else all contigs. A Cell Ranger multi run's
    library-level ``multi/vdj_*`` tables give way to its per-sample ones. Each is
    recorded in ``skipped``."""
    rels = {p: rel(p) for p, _ in detected}
    has_per_sample = any("per_sample_outs" in _dirs(r) for r in rels.values())
    matrix_kinds = {FileKind.tenx_h5, FileKind.anndata_h5ad, FileKind.loom, FileKind.mtx_matrix,
                    FileKind.h5seurat,
                    FileKind.dense_table}
    matrix_dirs = {_dirs(rels[p]) for p, det in detected if det.kind in matrix_kinds}
    vdj_dirs = set()
    for p, det in detected:
        d = _dirs(rels[p])
        if not d:
            continue
        cr_contig = det.kind == FileKind.vdj_contigs and Path(p).name.lower().startswith(_CR_CONTIGS)
        if d[-1].lower() in _VDJ_DIRS or (cr_contig and d not in matrix_dirs):
            vdj_dirs.add(d)

    keep, contigs = [], {}
    for p, det in detected:
        d = _dirs(rels[p])
        if not d or det.kind == FileKind.crispr_calls:  # guide calls: kept from crispr_analysis/
            keep.append((p, det))
        elif _is_analysis(d):
            skipped.append({"path": rels[p], "reason": ANALYSIS})
        elif any(d[:k] in vdj_dirs for k in range(1, len(d) + 1)):
            if det.kind != FileKind.vdj_contigs or d not in vdj_dirs:
                skipped.append({"path": rels[p], "reason": VDJ_EXTRA})
            elif has_per_sample and "multi" in (x.lower() for x in d):
                skipped.append({"path": rels[p], "reason": "a Cell Ranger multi run's library-level "
                                "table: the per-sample filtered table is used"})
            else:
                contigs.setdefault(d, []).append((p, det))
        else:
            keep.append((p, det))
    for items in contigs.values():
        items.sort(key=lambda it: _contig_rank(Path(it[0]).name))
        keep.append(items[0])
        for p, _ in items[1:]:
            skipped.append({"path": rels[p], "reason": "another contig table of the same folder is "
                            f"used ({Path(items[0][0]).name})"})
    return keep


def _contig_rank(name: str) -> int:
    low = name.lower()
    return next((i for i, tok in enumerate(_CR_CONTIGS) if tok in low), len(_CR_CONTIGS))


def _dir_label(rel: str) -> str | None:
    """The sample a folder upload names for a file: the nearest folder above it that is
    not one of Cell Ranger's own (``pbmc/outs/filtered_feature_bc_matrix/x`` -> ``pbmc``,
    ``per_sample_outs/s1/count/...`` -> ``s1``), or None."""
    dirs = _dirs(rel)
    for i in range(len(dirs) - 1, -1, -1):
        name = dirs[i]
        if name.lower() in _GENERIC_DIRS:
            continue
        if i > 0 and dirs[i - 1].lower() in _GENOME_PARENTS:  # the genome folder of Cell Ranger 2
            continue
        return name
    return None


def _sample_root(rel: str) -> str:
    """The folder of the sample a file belongs to: its folder without Cell Ranger's
    containers below it (``pbmc/outs/raw_feature_bc_matrix/x`` -> ``pbmc/outs``,
    ``per_sample_outs/s1/count/x.h5`` and ``per_sample_outs/s1/vdj_t/x.csv`` ->
    ``per_sample_outs/s1``); "" for a file at the top (every file of a flat upload)."""
    dirs = list(_dirs(rel))
    while dirs:
        last = dirs[-1].lower()
        genome = len(dirs) > 1 and dirs[-2].lower() in _GENOME_PARENTS
        if last in _CONTAINER_DIRS or genome:
            dirs.pop()
            continue
        break
    return "/".join(dirs)


def _single_label(bf: BundleFile) -> str:
    """A single-file unit's label: from its name, as always; in a folder, a Cell Ranger
    name (``filtered_feature_bc_matrix.h5``) gives its sample prefix, else the sample
    folder, else "sample". A Visium HD bin folder: the run's folder and the bin size
    (``Mouse_Brain_8um``, ``visium_hd_16um``)."""
    stem = Path(bf.name).stem
    hd = spatial.hd_folder(bf.rel_path)
    if hd is not None:
        run = next((d for d in reversed(PurePosixPath(hd[0]).parts) if d.lower() not in _GENERIC_DIRS), None)
        return f"{run or 'visium_hd'}_{hd[1]}um"
    if not _dirs(bf.rel_path) or not _strip_tenx(stem.lower().removesuffix("_h5"))[1]:
        return _clean_unit_label(stem)
    return sample_key(bf.name) or _dir_label(bf.rel_path) or "sample"


def _drop_mex_copies(units: list[IngestUnit], side_files: list[BundleFile],
                     skipped: list[dict]) -> list[IngestUnit]:
    """Cell Ranger writes each matrix twice, as ``X.h5`` and as the folder ``X/``
    (Cell Ranger 2: ``X_h5.h5`` and ``X/<genome>/``): keep the .h5 (one file, read
    faster) and skip the folder's three files."""
    h5s = {(_dirs(f.rel_path), Path(f.name).stem.lower()): f.name
           for u in units if u.format == UnitFormat.tenx_h5 for f in u.files}
    h5s.update({(_dirs(f.rel_path), Path(f.name).stem.lower()): f.name
                for f in side_files if f.kind == FileKind.tenx_h5})
    out = []
    for u in units:
        d = _dirs(u.files[0].rel_path) if u.format == UnitFormat.tenx_mex and u.files else ()
        twin = None
        if d:
            base, name = d[:-1], d[-1].lower()
            if len(d) > 1 and d[-2].lower() in _GENOME_PARENTS:  # Cell Ranger 2: <dir>/GRCh38/
                base, name = d[:-2], d[-2].lower()
            twin = h5s.get((base, name)) or h5s.get((base, name + "_h5"))
        if twin is None:
            out.append(u)
            continue
        for f in u.files:
            skipped.append({"path": f.rel_path, "reason": f"the same matrix as {twin}, which is read"})
    return out


def _unique_labels(units: list[IngestUnit]) -> None:
    """Two samples never share a label (a merge keys samples by it): later ones get _2, _3."""
    seen: dict[str, int] = {}
    for u in sorted(units, key=lambda x: (x.label, x.files[0].rel_path if x.files else "")):
        n = seen.get(u.label.lower(), 0)
        seen[u.label.lower()] = n + 1
        if n:
            u.label = f"{u.label}_{n + 1}"


# ---------------------------------------------------------------------------
# 10x MEX grouping
# ---------------------------------------------------------------------------


def _group_mex_units(mex_parts: list[tuple[Path, BundleFile]]) -> list[IngestUnit]:
    """Group matrix/barcodes/features files into MEX units by parent dir + prefix."""
    if not mex_parts:
        return []

    groups: dict[str, list[BundleFile]] = {}
    labels: dict[str, str] = {}
    for path, bf in mex_parts:
        prefix = _mex_prefix(path.name)
        key = f"{path.parent}|{prefix}"
        groups.setdefault(key, []).append(bf)
        if _dirs(bf.rel_path):  # a folder upload: the folders name the sample
            labels.setdefault(key, (sample_key(prefix) if prefix else "")
                              or _dir_label(bf.rel_path) or "sample")
        else:
            labels.setdefault(key, _mex_label(prefix, path.parent.name))

    units: list[IngestUnit] = []
    for key, files in groups.items():
        units.append(_assemble_mex_unit(labels[key], files))
    return units


def _assemble_mex_unit(label: str, files: list[BundleFile]) -> IngestUnit:
    present = {f.role for f in files}
    missing = [r for r in _MEX_REQUIRED if r not in present]

    issues: list[str] = []
    # Duplicate roles (e.g. two matrix files grouped together) — flag, don't guess.
    for role in _MEX_REQUIRED:
        if sum(1 for f in files if f.role == role) > 1:
            issues.append(f"More than one {role.value} file in this sample — keep only one.")
    if missing:
        issues.append(
            "Missing " + ", ".join(r.value for r in missing) + " for this 10x sample."
        )

    return IngestUnit(
        label=label,
        format=UnitFormat.tenx_mex,
        files=sorted(files, key=lambda f: f.role.value),
        complete=not missing and not issues,
        missing_roles=missing,
        issues=issues,
    )


def _mex_label(prefix: str, parent_name: str) -> str:
    """Best label for a MEX unit: the filename prefix, else a meaningful parent
    directory, else a neutral default (the user renames it in the wizard)."""
    if prefix:
        return prefix
    if parent_name and parent_name.lower() not in _GENERIC_MEX_DIRS:
        return parent_name
    return "sample"


def _mex_prefix(name: str) -> str:
    """Derive a grouping prefix by stripping extensions and the role token.

    ``GSM4711_barcodes.tsv.gz`` → ``gsm4711``; a bare ``barcodes.tsv`` → ``""``
    (so a standard triplet in one folder groups together).
    """
    s = name.lower()
    changed = True
    while changed:
        changed = False
        for ext in _STRIP_EXTS:
            if s.endswith(ext):
                s = s[: -len(ext)]
                changed = True
    for tok in _ROLE_TOKENS:
        if s.endswith(tok):
            s = s[: -len(tok)]
            break
    return s.strip(" _-.")


# ---------------------------------------------------------------------------
# Side files: Cell Ranger metrics_summary.csv and the raw (unfiltered) matrix
# ---------------------------------------------------------------------------

_RAW_TOKEN_RE = re.compile(r"(^|[_.\-])raw([_.\-]|$)")
_RAW_MEX_DIRS = ("raw_feature_bc_matrix", "raw_gene_bc_matrices")
_SIDE_EXTS = (".gz", ".h5", ".h5ad", ".csv", ".mtx", ".tsv", ".txt")
# Cell Ranger's own file stems, stripped to leave the sample prefix ("pbmc_1k_v3").
_TENX_TOKENS = (
    "sample_filtered_feature_bc_matrix", "sample_raw_feature_bc_matrix",
    "filtered_feature_bc_matrix", "raw_feature_bc_matrix",
    "filtered_gene_bc_matrices", "raw_gene_bc_matrices", "metrics_summary",
)


def _is_side_file(det: DetectionResult) -> bool:
    return det.kind == FileKind.tenx_metrics or (
        det.kind == FileKind.tenx_h5 and bool(det.details.get("raw_matrix"))
    )


def _strip_tenx(s: str) -> tuple[str, bool]:
    for tok in _TENX_TOKENS:
        if s.endswith(tok):
            return s[: -len(tok)], True
    return s, False


def sample_key(name: str) -> str:
    """The sample prefix of a Cell Ranger file name, shared by its metrics, raw and
    filtered files: ``pbmc_1k_v3_metrics_summary.csv``, ``pbmc_1k_v3_raw_feature_bc_matrix.h5``
    and ``pbmc_1k_v3_filtered_feature_bc_matrix.h5`` all give ``pbmc_1k_v3``; a bare
    ``metrics_summary.csv`` gives ""."""
    s = name.lower()
    changed = True
    while changed:
        changed = False
        for ext in _SIDE_EXTS:
            if s.endswith(ext):
                s = s[: -len(ext)]
                changed = True
    s = s.removesuffix("_h5")  # Cell Ranger 2: filtered_gene_bc_matrices_h5.h5
    s, hit = _strip_tenx(s)
    if not hit:  # a MEX part: "<sample>_raw_feature_bc_matrix_barcodes"
        for tok in _ROLE_TOKENS:
            if s.endswith(tok) and len(s) > len(tok):
                s = s[: -len(tok)].rstrip(" _-.")
                break
        s, _ = _strip_tenx(s)
    s = s.rstrip(" _-.")
    for tok in ("raw", "filtered"):
        if s == tok or (s.endswith(tok) and s[-len(tok) - 1] in " _-."):
            s = s[: -len(tok)]
            break
    return s.strip(" _-.")


def _is_raw_mex(unit: IngestUnit) -> bool:
    if unit.format != UnitFormat.tenx_mex or not unit.files:
        return False
    prefix = _mex_prefix(unit.files[0].name)
    parent = Path(unit.files[0].path).parent.name.lower()
    named_raw = bool(_RAW_TOKEN_RE.search(prefix)) and "filtered" not in prefix
    # Any folder of a folder upload (Cell Ranger 2: raw_gene_bc_matrices/GRCh38/).
    folders = [parent] + [d.lower() for d in _dirs(unit.files[0].rel_path)]
    return named_raw or any(r in d for d in folders for r in _RAW_MEX_DIRS)


def _split_raw_mex(units: list[IngestUnit]) -> tuple[list[IngestUnit], list[IngestUnit]]:
    """Raw (unfiltered) MEX triplets, named so, leave the data units: every barcode of
    the whitelist is in them, nearly all empty droplets."""
    raw = [u for u in units if _is_raw_mex(u)]
    return [u for u in units if u not in raw], raw


def _unit_key(unit: IngestUnit) -> str:
    if not unit.files:
        return unit.label.lower()
    first = unit.files[0]
    name = _mex_prefix(first.name) if unit.format == UnitFormat.tenx_mex else first.name
    return f"{_sample_root(first.rel_path)}|{sample_key(name)}"


def _side_key(bf: BundleFile) -> str:
    return f"{_sample_root(bf.rel_path)}|{sample_key(bf.name)}"


def _only_in_root(units: list[IngestUnit], key: str) -> IngestUnit | None:
    """The one unit in the sample folder of ``key`` (a folder upload), else None."""
    root = key.split("|", 1)[0]
    if not root:
        return None
    hits = [u for u in units if u.files and _sample_root(u.files[0].rel_path) == root]
    return hits[0] if len(hits) == 1 else None


def _attach_side_files(
    units: list[IngestUnit], side_files: list[BundleFile], raw_mex: list[IngestUnit]
) -> list[str]:
    """Attach each metrics file / raw matrix to the unit with the same sample prefix
    (or to the only unit); return sentences for what could not be attached."""
    issues: list[str] = []
    # One item per metrics file or raw matrix (a raw MEX triplet is one item).
    items: list[tuple[str, str, list[BundleFile]]] = []  # (key, "metrics" | "raw", files)
    for bf in side_files:
        items.append((_side_key(bf), "metrics" if bf.kind == FileKind.tenx_metrics else "raw", [bf]))
    for u in raw_mex:
        names = ", ".join(f.name for f in u.files)
        if u.missing_roles:
            missing = ", ".join(r.value for r in u.missing_roles)
            issues.append(f"The raw 10x matrix files ({names}) are missing their {missing} file "
                          "and were not attached.")
            continue
        parts = [f.model_copy(update={"role": FileRole.side, "side_kind": SideKind.raw})
                 for f in u.files]
        key = f"{_sample_root(u.files[0].rel_path)}|{sample_key(_mex_prefix(u.files[0].name))}"
        items.append((key, "raw", parts))

    loadable = [u for u in units if u.format != UnitFormat.unknown]
    by_key: dict[str, IngestUnit] = {}
    for u in loadable:
        by_key.setdefault(_unit_key(u), u)
    for key, what, files in items:
        names = ", ".join(f.name for f in files)
        noun = "metrics summary" if what == "metrics" else "raw (unfiltered) matrix"
        unit = (by_key.get(key) or _only_in_root(loadable, key)
                or (loadable[0] if len(loadable) == 1 else None))
        if unit is None:
            if not loadable:
                issues.append(
                    f"{names}: a Cell Ranger {noun} is attached to a sample for sequencing QC, "
                    "never loaded as the data. Add the filtered matrix "
                    "(filtered_feature_bc_matrix) as the dataset."
                )
            else:
                example = ("metrics_summary.csv" if what == "metrics"
                           else "raw_feature_bc_matrix.h5")
                issues.append(
                    f"{names} (a Cell Ranger {noun}) could not be matched to a sample by name and "
                    f"was not attached. Name it after its sample, e.g. "
                    f"{loadable[0].label}_{example}."
                )
            continue
        is_metrics = [f.side_kind == SideKind.metrics for f in unit.side_files
                      if f.side_kind != SideKind.vdj]
        if (what == "metrics" and any(is_metrics)) or (what == "raw" and not all(is_metrics)):
            issues.append(f"{names}: sample {unit.label} already has a Cell Ranger {noun}; this "
                          "one was not attached.")
            continue
        unit.side_files.extend(files)
    return issues


# ---------------------------------------------------------------------------
# TCR/BCR contig tables: side files of a unit
# ---------------------------------------------------------------------------

_VDJ_TOKENS = ("filtered_contig_annotations", "all_contig_annotations", "contig_annotations",
               "airr_rearrangement", "airr")
_VDJ_ARM_TOKENS = ("vdj_t", "vdj_b", "tcr", "bcr", "vdj", "t", "b")
_MATRIX_TOKENS = ("filtered_feature_bc_matrix", "raw_feature_bc_matrix", "feature_bc_matrix",
                  "filtered_gene_bc_matrices", "gene_bc_matrices", "matrix", "count", "counts", "gex")


def _stem(name: str, tokens: tuple[str, ...]) -> str:
    s = name.lower()
    for ext in (".gz", ".csv", ".tsv", ".txt", ".h5ad", ".h5", ".loom", ".mtx"):
        if s.endswith(ext):
            s = s[: -len(ext)]
    for tok in tokens:
        if s.endswith(tok):
            s = s[: -len(tok)].rstrip(" _-.")
            break
    return s.strip(" _-.")


def _vdj_prefix(name: str) -> str:
    """``sample1_t_filtered_contig_annotations.csv`` -> ``sample1``."""
    s = _stem(name, _VDJ_TOKENS)
    for tok in _VDJ_ARM_TOKENS:
        if s == tok:
            return ""
        if s.endswith("_" + tok) or s.endswith("-" + tok):
            return s[: -len(tok) - 1].strip(" _-.")
    return s


def _unit_prefix(unit: IngestUnit) -> str:
    if unit.format == UnitFormat.tenx_mex:
        return unit.label.lower()
    return _stem(unit.files[0].name, _MATRIX_TOKENS) if unit.files else ""


def _vdj_dir_arm(bf: BundleFile) -> str | None:
    """"T" / "B" from the V(D)J folder a contig table is in (``vdj_t``, ``vdj_b``)."""
    return next((_VDJ_DIRS[d.lower()] for d in reversed(_dirs(bf.rel_path))
                 if d.lower() in _VDJ_DIRS), None)


def _vdj_arm(name: str) -> str | None:
    """"T" / "B" from a contig table's name (``sample_t_filtered_contig_annotations.csv``),
    or None; the chains in the file decide when detection could read them."""
    s = _stem(name, _VDJ_TOKENS)
    for tok, arm in (("vdj_t", "T"), ("vdj_b", "B"), ("tcr", "T"), ("bcr", "B"), ("t", "T"),
                     ("b", "B")):
        if s == tok or s.endswith("_" + tok) or s.endswith("-" + tok):
            return arm
    return None


def _attach_vdj_files(units: list[IngestUnit], vdj_files: list[BundleFile]) -> list[str]:
    """Add each contig table to the side files of the unit it belongs to: the one whose
    matrix shares its filename prefix, else the only unit. Returns bundle issues."""
    if not vdj_files:
        return []
    loadable = [u for u in units if u.format != UnitFormat.unknown]
    if not loadable:
        return ["TCR/BCR contig tables need the expression matrix of the same sample: add it, or "
                "attach the table to an existing dataset in Unified View > Clonotypes."]
    issues: list[str] = []
    for bf in vdj_files:
        prefix = _vdj_prefix(bf.name)
        hits = [u for u in loadable if prefix and _unit_prefix(u) == prefix]
        unit = (hits[0] if len(hits) == 1 else _only_in_root(loadable, _sample_root(bf.rel_path))
                or (loadable[0] if len(loadable) == 1 else None))
        if unit is None:
            issues.append(f"Could not tell which sample the contig table {bf.name} belongs to; give it "
                          "the sample's filename prefix, or attach it after import in Unified View > "
                          "Clonotypes.")
            continue
        unit.side_files.append(bf)
    return issues


def _attach_crispr_files(units: list[IngestUnit], files: list[BundleFile]) -> list[str]:
    """Add Cell Ranger's guide calls (``crispr_analysis/protospacer_calls_per_cell.csv``)
    to the unit of their sample folder, else the only unit. Returns bundle issues."""
    if not files:
        return []
    loadable = [u for u in units if u.format != UnitFormat.unknown]
    issues: list[str] = []
    for bf in files:
        root = _sample_root(bf.rel_path)
        unit = _only_in_root(loadable, f"{root}|") or (loadable[0] if len(loadable) == 1 else None)
        if unit is None:
            issues.append(f"Could not tell which sample the guide calls {bf.rel_path or bf.name} belong to: "
                          "upload them in the sample's Cell Ranger folder, or with one sample at a time.")
            continue
        if any(f.side_kind == SideKind.crispr for f in unit.side_files):
            issues.append(f"{bf.name}: sample {unit.label} already has guide calls; this file was not attached.")
            continue
        unit.side_files.append(bf)
    return issues


# ---------------------------------------------------------------------------
# Space Ranger spatial files: side files of a unit
# ---------------------------------------------------------------------------


def _attach_spatial_files(units: list[IngestUnit], files: list[BundleFile]) -> list[str]:
    """Add each sample's spatial files (``spatial/`` loose or as an archive) to the unit
    of the same sample: the same folder and name prefix (``X_spatial.tar.gz`` beside
    ``X_filtered_feature_bc_matrix.h5``), else the only unit of its folder, else the only
    unit. A part a unit already has is not attached twice. Returns bundle issues."""
    if not files:
        return []
    loadable = [u for u in units if u.format != UnitFormat.unknown]
    if not loadable:
        return ["Space Ranger spatial files (tissue positions, scale factors, images) need the "
                "sample's matrix (filtered_feature_bc_matrix.h5 or its folder): add it."]
    by_key: dict[str, IngestUnit] = {}
    for u in loadable:
        by_key.setdefault(_unit_key(u), u)
    issues: list[str] = []
    for bf in sorted(files, key=lambda f: (f.spatial_part != spatial.ARCHIVE, f.rel_path or f.name)):
        key = f"{_sample_root(bf.rel_path)}|{spatial.name_key(bf.name)}"
        unit = (by_key.get(key) or _only_in_root(loadable, key)
                or (loadable[0] if len(loadable) == 1 else None))
        if unit is None:
            issues.append(f"Could not tell which sample the spatial file {bf.rel_path or bf.name} belongs "
                          "to: give it the sample's file name prefix, or upload each Space Ranger "
                          "folder (outs/) whole.")
            continue
        have = {f.spatial_part for f in unit.side_files if f.side_kind == SideKind.spatial}
        if spatial.ARCHIVE in have:
            have |= set(spatial.PARTS)
        if bf.spatial_part in have:
            issues.append(f"{bf.name}: sample {unit.label} already has its {bf.spatial_part} "
                          "spatial file; this one was not attached.")
            continue
        unit.side_files.append(bf)
        _spatial_label(unit)
    return issues


def _spatial_label(unit: IngestUnit) -> None:
    """A spatial sample uploaded as plain files is named by its Space Ranger prefix
    (``V1_Lymph_Node_filtered_feature_bc_matrix.h5`` -> ``V1_Lymph_Node``): the label is
    its section's library id."""
    if unit.format != UnitFormat.tenx_h5 or not unit.files or _dirs(unit.files[0].rel_path):
        return
    name = unit.files[0].name
    stem = name[:-3] if name.lower().endswith(".h5") else name
    low, hit = _strip_tenx(stem.lower().removesuffix("_h5"))
    prefix = stem[:len(low)].rstrip(" _-.") if hit else ""
    if prefix:
        unit.label = prefix
