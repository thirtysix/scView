##
## export_rds.R
## The reverse of convert.R: an h5ad scView staged for export -> a Seurat
## (SeuratObject v5) or SingleCellExperiment object saved as .rds.
##
## The backend (core/rds_export.py) writes the staging h5ad with a fixed layout:
##   X              log-normalised values (absent when there are none)
##   layers/counts  raw counts (absent when there are none)
##   obs, var       cell and gene tables (unique names; categoricals in level order)
##   obsm/X_*       layouts with all their columns; varm/PCs the PCA loadings
##   uns/scview_rexport  JSON: provenance, colours, cluster column, PCA variance,
##                  protein names, what the backend left out
##   uns/protein_counts, uns/protein_clr  CITE-seq antibodies (cells x proteins)
##
## What goes where:
##   Seurat  assay "RNA" (Assay5): layers counts + data (never scaled values),
##           feature metadata = var, VariableFeatures = var highly_variable;
##           meta.data = obs (factors keep their level order); reductions from
##           obsm (pca key PC_, umap umap_, tsne tSNE_, others <name>_) with the
##           PCA loadings and stdev; assay "ADT" from the antibody counts (data =
##           CLR); Idents = scView's active clustering; Misc(obj, "scview").
##           Seurat forbids "_" in feature names: replaced by "-" (reported).
##   SCE     assays counts / logcounts, colData = obs, rowData = var,
##           reducedDims PCA / UMAP / TSNE / ... (PCA rotation + percentVar),
##           altExp "ADT", metadata$scview.
## Matrices are read one at a time as CSR (cells x genes), which is a
## dgCMatrix of genes x cells as it stands: no transpose, no dense copy.
##

library(Matrix)

EXPORT_PY <- "
import json
import h5py
import numpy as np
import pandas as pd
import scipy.sparse as sp
try:
    from anndata.io import read_elem as _read_elem
except ImportError:
    from anndata.experimental import read_elem as _read_elem


def scv_has(path, key):
    with h5py.File(path, 'r') as f:
        return key in f


def scv_matrix(path, key):
    with h5py.File(path, 'r') as f:
        m = _read_elem(f[key])
    if not sp.issparse(m):
        m = sp.csr_matrix(np.asarray(m))
    m = sp.csr_matrix(m)
    m.sum_duplicates()
    m.sort_indices()
    if m.nnz > 2**31 - 1:
        raise ValueError('%s has %d stored values; an R sparse matrix holds at most 2^31 - 1' % (key, m.nnz))
    return {'p': m.indptr.astype(np.int32), 'i': m.indices.astype(np.int32), 'x': m.data,
            'n_rows': int(m.shape[0]), 'n_cols': int(m.shape[1]), 'nnz': int(m.nnz)}


def scv_dense(path, key):
    with h5py.File(path, 'r') as f:
        return np.ascontiguousarray(np.asarray(f[key][()], dtype=np.float64))


def scv_keys(path, group):
    with h5py.File(path, 'r') as f:
        return list(f[group].keys()) if group in f else []


def scv_text(path, key):
    with h5py.File(path, 'r') as f:
        v = f[key][()]
    return v.decode() if isinstance(v, bytes) else str(v)


def scv_frame(path, key):
    with h5py.File(path, 'r') as f:
        df = _read_elem(f[key])
    cols = []
    for c in df.columns:
        s = df[c]
        if isinstance(s.dtype, pd.CategoricalDtype):
            cols.append({'name': str(c), 'kind': 'factor',
                         'codes': s.cat.codes.to_numpy().astype(np.int32),
                         'levels': np.array([str(v) for v in s.cat.categories], dtype=str),
                         'ordered': bool(s.cat.ordered)})
        elif pd.api.types.is_bool_dtype(s) and not s.isna().any():
            cols.append({'name': str(c), 'kind': 'logical', 'values': s.to_numpy(dtype=bool)})
        elif (pd.api.types.is_integer_dtype(s) and not s.isna().any() and len(s)
                and s.abs().max() < 2**31 - 1):
            cols.append({'name': str(c), 'kind': 'integer', 'values': s.to_numpy(dtype=np.int32)})
        elif pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s):
            cols.append({'name': str(c), 'kind': 'numeric',
                         'values': s.to_numpy(dtype=np.float64, na_value=np.nan)})
        else:
            na = s.isna().to_numpy()
            vals = np.array(['' if m else str(v) for v, m in zip(s.tolist(), na)], dtype=str)
            cols.append({'name': str(c), 'kind': 'character', 'values': vals, 'na': na})
    return {'index': np.array([str(v) for v in df.index], dtype=str), 'columns': cols}
"

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

#' Build a Seurat or SingleCellExperiment object from a staging h5ad and save it.
#'
#' @param input_path    The staging .h5ad (core/rds_export.py).
#' @param output_path   Where the .rds goes (written beside it, then renamed).
#' @param format        "seurat" or "sce".
#' @param progress_path Optional JSON file each stage is written to.
#' @return list(success, message | error, report = list(kept, skipped, ...)).
export_h5ad_to_rds <- function(input_path, output_path, format = "seurat", progress_path = NULL) {
  progress <- progress_writer(progress_path)
  if (!file.exists(input_path)) {
    return(finish(progress, list(success = FALSE, error = paste0("Input file not found: ", input_path))))
  }
  if (!format %in% c("seurat", "sce")) {
    return(finish(progress, list(success = FALSE,
                                 error = sprintf("Unknown format '%s': use seurat or sce.", format))))
  }
  result <- tryCatch(
    build_and_save(input_path, output_path, format, progress),
    error = function(e) list(success = FALSE, error = paste0("Export failed: ", conditionMessage(e)))
  )
  finish(progress, result)
}

build_and_save <- function(input_path, output_path, format, progress) {
  pkg <- if (format == "seurat") "SeuratObject" else "SingleCellExperiment"
  if (!suppressMessages(requireNamespace(pkg, quietly = TRUE))) {
    stop(sprintf("the %s package is not installed in the converter", pkg))
  }
  py <- export_py()
  rpt <- new.env()
  rpt$kept <- character(0)
  rpt$skipped <- character(0)

  meta <- jsonlite::fromJSON(py_chr(py$scv_text(input_path, "uns/scview_rexport")), simplifyVector = TRUE)
  for (s in meta$left_out %||% character(0)) note(rpt, "skipped", "%s", s)

  progress("reading", "Reading the cell and gene tables", format = format)
  obs <- read_frame(py, input_path, "obs")
  var <- read_frame(py, input_path, "var")
  cells <- rownames(obs)
  genes <- rownames(var)
  progress("reading", sprintf("Reading the cell and gene tables (%s cells x %s genes)",
                              fmt_n(length(cells)), fmt_n(length(genes))),
           n_cells = length(cells), n_genes = length(genes))

  counts <- NULL
  data <- NULL
  if (py_lgl(py$scv_has(input_path, "layers/counts"))) {
    progress("reading", "Reading the raw counts")
    counts <- read_csr(py, input_path, "layers/counts", genes, cells)
  }
  if (py_lgl(py$scv_has(input_path, "X"))) {
    progress("reading", "Reading the log-normalised values")
    data <- read_csr(py, input_path, "X", genes, cells)
  }
  if (is.null(counts) && is.null(data)) stop("the staging file holds no expression matrix")

  emb <- list()
  for (key in py_chr(py$scv_keys(input_path, "obsm"))) {
    m <- py_num(py$scv_dense(input_path, paste0("obsm/", key)))
    if (length(dim(m)) != 2 || nrow(m) != length(cells)) next
    emb[[key]] <- m
  }
  pcs <- NULL
  if (py_lgl(py$scv_has(input_path, "varm/PCs"))) {
    pcs <- py_num(py$scv_dense(input_path, "varm/PCs"))
    rownames(pcs) <- genes
  }
  protein <- NULL
  if (py_lgl(py$scv_has(input_path, "uns/protein_counts"))) {
    protein <- list(names = as.character(meta$protein_names),
                    counts = py_num(py$scv_dense(input_path, "uns/protein_counts")))
    if (py_lgl(py$scv_has(input_path, "uns/protein_clr"))) {
      protein$clr <- py_num(py$scv_dense(input_path, "uns/protein_clr"))
    }
  }

  progress("building", if (format == "seurat") "Building the Seurat object" else
    "Building the SingleCellExperiment")
  obj <- if (format == "seurat") {
    build_seurat(counts, data, obs, var, emb, pcs, protein, meta, rpt)
  } else {
    build_sce(counts, data, obs, var, emb, pcs, protein, meta, rpt)
  }
  rm(counts, data)

  progress("writing", "Writing the .rds file (compressed)")
  tmp <- paste0(output_path, ".part")
  on.exit(unlink(tmp), add = TRUE)
  saveRDS(obj, tmp)
  if (!file.rename(tmp, output_path)) stop("could not move the .rds into place")
  what <- if (format == "seurat") "Seurat object" else "SingleCellExperiment"
  list(
    success = TRUE,
    message = sprintf("Wrote a %s (%s cells x %s genes)", what, fmt_n(length(cells)), fmt_n(length(genes))),
    report = list(
      format = format,
      target_class = if (format == "seurat") "Seurat" else "SingleCellExperiment",
      n_cells = length(cells), n_genes = length(genes),
      renamed_features = rpt$renamed %||% 0L,
      r_packages = as.list(pkg_versions(format)),
      kept = as.list(rpt$kept), skipped = as.list(rpt$skipped),
      file_bytes = file.size(output_path)
    )
  )
}

# ---------------------------------------------------------------------------
# Reading
# ---------------------------------------------------------------------------

#' The Python helpers above, defined in Python's __main__ (functions only).
export_py <- function() {
  suppressMessages(library(reticulate))
  reticulate::py_run_string(EXPORT_PY, local = FALSE, convert = FALSE)
  reticulate::import_main(convert = FALSE)
}

py_chr <- function(x) as.character(reticulate::py_to_r(x))
py_lgl <- function(x) isTRUE(reticulate::py_to_r(x))
py_num <- function(x) reticulate::py_to_r(x)

#' A CSR (cells x genes) element as a dgCMatrix of genes x cells.
read_csr <- function(py, path, key, genes, cells) {
  m <- py$scv_matrix(path, key)
  get <- function(k) reticulate::py_to_r(reticulate::py_get_item(m, k))
  n_cells <- get("n_rows")
  n_genes <- get("n_cols")
  if (n_cells != length(cells) || n_genes != length(genes)) {
    stop(sprintf("%s is %d x %d, the tables %d x %d", key, n_cells, n_genes, length(cells), length(genes)))
  }
  p <- get("p")
  i <- get("i")
  x <- as.double(get("x"))
  rm(m)
  out <- methods::new("dgCMatrix", p = as.integer(p), i = as.integer(i), x = x,
                      Dim = c(as.integer(n_genes), as.integer(n_cells)),
                      Dimnames = list(genes, cells))
  rm(p, i, x)
  invisible(gc(verbose = FALSE))
  out
}

#' obs / var as an R data.frame: factors keep their level order (and NA codes),
#' logical, integer, numeric and character columns as they are.
read_frame <- function(py, path, key) {
  fr <- py$scv_frame(path, key)
  index <- py_chr(reticulate::py_get_item(fr, "index"))
  cols <- reticulate::py_get_item(fr, "columns")
  df <- data.frame(row.names = index, check.names = FALSE)
  n <- reticulate::py_to_r(reticulate::py_len(cols))
  for (k in seq_len(n) - 1L) {
    col <- reticulate::py_get_item(cols, k)
    g <- function(name) reticulate::py_to_r(reticulate::py_get_item(col, name))
    name <- g("name")
    kind <- g("kind")
    val <- switch(kind,
      factor = {
        codes <- as.integer(g("codes")) + 1L
        codes[codes < 1L] <- NA_integer_
        levels <- as.character(g("levels"))
        structure(codes, levels = levels,
                  class = if (isTRUE(g("ordered"))) c("ordered", "factor") else "factor")
      },
      logical = as.logical(g("values")),
      integer = as.integer(g("values")),
      numeric = as.numeric(g("values")),
      character = {
        v <- as.character(g("values"))
        v[as.logical(g("na"))] <- NA_character_
        v
      }
    )
    df[[name]] <- val
  }
  df
}

# ---------------------------------------------------------------------------
# Seurat
# ---------------------------------------------------------------------------

#' Seurat reduction name and key for an obsm key: X_pca -> pca / PC_.
seurat_reduction <- function(key) {
  name <- tolower(sub("^X_", "", key))
  name <- gsub("[^a-z0-9_.]", "_", name)
  known <- c(pca = "PC_", umap = "umap_", tsne = "tSNE_")
  k <- if (name %in% names(known)) known[[name]] else paste0(gsub("[^a-zA-Z0-9]", "", name), "_")
  if (!grepl("^[a-zA-Z]", k)) k <- paste0("dim", k)
  list(name = name, key = k)
}

#' Feature names Seurat accepts: "_" -> "-", made unique again.
seurat_features <- function(genes, rpt, what = "gene") {
  fixed <- make.unique(gsub("_", "-", genes, fixed = TRUE))
  n <- sum(fixed != genes)
  if (n > 0) {
    note(rpt, "kept", "%s changed: '_' replaced by '-' (Seurat does not allow '_' in feature names)%s",
         plural(n, paste(what, "name")),
         if (anyDuplicated(gsub("_", "-", genes, fixed = TRUE))) ", clashes made unique ('.1' added)" else "")
  }
  list(names = fixed, n = n)
}

build_seurat <- function(counts, data, obs, var, emb, pcs, protein, meta, rpt) {
  suppressMessages(library(SeuratObject))
  original <- rownames(var)
  feats <- seurat_features(original, rpt)
  rpt$renamed <- feats$n
  if (feats$n > 0) {
    if (!is.null(counts)) rownames(counts) <- feats$names
    if (!is.null(data)) rownames(data) <- feats$names
    if (!is.null(pcs)) rownames(pcs) <- feats$names
    rownames(var) <- feats$names
  }
  assay <- SeuratObject::CreateAssay5Object(counts = counts, data = data)
  if (!is.null(counts)) note(rpt, "kept", "raw counts (from %s) as the RNA counts layer", meta$counts_from %||% "the dataset")
  if (!is.null(data)) note(rpt, "kept", "log-normalised values (from %s) as the RNA data layer", meta$x_from %||% "X")
  if (is.null(data)) note(rpt, "skipped", "no log-normalised values: run NormalizeData() in R")
  note(rpt, "skipped", "scale.data: not exported (run ScaleData() in R; scaled values are never put in data)")
  if (ncol(var) > 0) {
    assay <- SeuratObject::AddMetaData(assay, metadata = var)
    note(rpt, "kept", "gene metadata (var): %s as feature metadata", plural(ncol(var), "column"))
  }

  obj <- SeuratObject::CreateSeuratObject(counts = assay, assay = "RNA",
                                          project = project_name(meta))
  had_ident <- "orig.ident" %in% colnames(obs)
  obj <- SeuratObject::AddMetaData(obj, metadata = obs)
  if (!had_ident) obj$orig.ident <- factor(rep(project_name(meta), ncol(obj)))
  note(rpt, "kept", "cell metadata (obs): %s as meta.data (categories as factors, level order kept)",
       plural(ncol(obs), "column"))

  hv <- match(as.character(meta$variable_genes %||% character(0)), original)
  hv <- feats$names[hv[!is.na(hv)]]
  if (length(hv)) {
    SeuratObject::VariableFeatures(obj) <- hv
    note(rpt, "kept", "%s as VariableFeatures", plural(length(hv), "highly variable gene"))
  }

  for (key in names(emb)) {
    r <- seurat_reduction(key)
    m <- emb[[key]]
    dimnames(m) <- list(colnames(obj), paste0(r$key, seq_len(ncol(m))))
    loadings <- NULL
    stdev <- numeric(0)
    if (r$name == "pca") {
      if (!is.null(pcs) && ncol(pcs) == ncol(m)) {
        keep <- rowSums(abs(pcs)) > 0 & is.finite(rowSums(pcs))
        loadings <- pcs[keep, , drop = FALSE]
        colnames(loadings) <- colnames(m)
      }
      if (length(meta$pca_variance) == ncol(m)) stdev <- sqrt(as.numeric(meta$pca_variance))
    }
    obj[[r$name]] <- SeuratObject::CreateDimReducObject(
      embeddings = m, loadings = if (is.null(loadings)) new("matrix") else loadings,
      stdev = stdev, key = r$key, assay = "RNA")
    note(rpt, "kept", "%s as reduction '%s' (%s%s)", key, r$name, plural(ncol(m), "dimension"),
         if (!is.null(loadings)) sprintf(", loadings for %s", plural(nrow(loadings), "gene")) else "")
  }

  if (!is.null(protein)) {
    adt <- protein_matrix(protein$counts, protein$names, colnames(obj), rpt, seurat = TRUE)
    clr <- if (!is.null(protein$clr)) protein_matrix(protein$clr, protein$names, colnames(obj), NULL, seurat = TRUE)
    a <- SeuratObject::CreateAssay5Object(counts = adt)
    if (!is.null(clr)) SeuratObject::LayerData(a, layer = "data") <- clr
    obj[["ADT"]] <- a
    note(rpt, "kept", "antibody counts: %s as assay ADT%s", plural(nrow(adt), "protein"),
         if (!is.null(clr)) " (data = CLR per cell, as NormalizeData(method = 'CLR', margin = 2))" else "")
  }

  cl <- meta$cluster_column
  if (!is.null(cl) && length(cl) == 1 && cl %in% colnames(obj[[]])) {
    SeuratObject::Idents(obj) <- cl
    note(rpt, "kept", "Idents set to the clustering '%s'", cl)
  } else {
    SeuratObject::Idents(obj) <- "orig.ident"
  }
  SeuratObject::Misc(obj, slot = "scview") <- scview_misc(meta)
  note(rpt, "kept", "provenance (scView version, dataset, recorded steps) in Misc(obj, 'scview')")
  obj
}

# ---------------------------------------------------------------------------
# SingleCellExperiment
# ---------------------------------------------------------------------------

sce_reduced_name <- function(key) toupper(sub("^X_", "", key))

build_sce <- function(counts, data, obs, var, emb, pcs, protein, meta, rpt) {
  suppressMessages({
    library(SingleCellExperiment)
    library(SummarizedExperiment)
  })
  assays <- list()
  if (!is.null(counts)) {
    assays$counts <- counts
    note(rpt, "kept", "raw counts (from %s) as assay counts", meta$counts_from %||% "the dataset")
  }
  if (!is.null(data)) {
    assays$logcounts <- data
    note(rpt, "kept", "log-normalised values (from %s) as assay logcounts", meta$x_from %||% "X")
  } else {
    note(rpt, "skipped", "no log-normalised values: run scuttle::logNormCounts() in R")
  }
  sce <- SingleCellExperiment::SingleCellExperiment(
    assays = assays,
    colData = S4Vectors::DataFrame(obs, check.names = FALSE),
    rowData = S4Vectors::DataFrame(var, check.names = FALSE)
  )
  note(rpt, "kept", "cell metadata (obs): %s as colData (categories as factors, level order kept)",
       plural(ncol(obs), "column"))
  note(rpt, "kept", "gene metadata (var): %s as rowData", plural(ncol(var), "column"))

  for (key in names(emb)) {
    name <- sce_reduced_name(key)
    m <- emb[[key]]
    dimnames(m) <- list(colnames(sce), paste0(name, seq_len(ncol(m))))
    extra <- ""
    if (name == "PCA") {
      if (!is.null(pcs) && ncol(pcs) == ncol(m)) {
        rot <- pcs
        colnames(rot) <- colnames(m)
        attr(m, "rotation") <- rot
        extra <- ", rotation"
      }
      if (length(meta$pca_variance_ratio) == ncol(m)) {
        attr(m, "percentVar") <- 100 * as.numeric(meta$pca_variance_ratio)
        extra <- paste0(extra, ", percentVar")
      }
    }
    SingleCellExperiment::reducedDim(sce, name) <- m
    note(rpt, "kept", "%s as reducedDim '%s' (%s%s)", key, name, plural(ncol(m), "dimension"), extra)
  }

  if (!is.null(protein)) {
    adt <- protein_matrix(protein$counts, protein$names, colnames(sce), rpt, seurat = FALSE)
    a <- list(counts = adt)
    if (!is.null(protein$clr)) a$clr <- protein_matrix(protein$clr, protein$names, colnames(sce), NULL, seurat = FALSE)
    SingleCellExperiment::altExp(sce, "ADT") <- SummarizedExperiment::SummarizedExperiment(assays = a)
    note(rpt, "kept", "antibody counts: %s as altExp 'ADT'%s", plural(nrow(adt), "protein"),
         if (!is.null(protein$clr)) " (assays counts, clr)" else "")
  }
  cl <- meta$cluster_column
  if (!is.null(cl) && length(cl) == 1 && cl %in% colnames(obs)) {
    note(rpt, "kept", "scView's active clustering is colData column '%s'", cl)
  }
  S4Vectors::metadata(sce)$scview <- scview_misc(meta)
  note(rpt, "kept", "provenance (scView version, dataset, recorded steps) in metadata(sce)$scview")
  sce
}

# ---------------------------------------------------------------------------
# Shared pieces
# ---------------------------------------------------------------------------

#' cells x proteins (dense, from Python) as a sparse proteins x cells matrix.
protein_matrix <- function(m, names, cells, rpt, seurat) {
  if (length(names) != ncol(m)) names <- sprintf("protein%d", seq_len(ncol(m)))
  if (seurat) {
    fixed <- make.unique(gsub("_", "-", names, fixed = TRUE))
    if (!is.null(rpt) && any(fixed != names)) {
      note(rpt, "kept", "%s: '_' replaced by '-'", plural(sum(fixed != names), "protein name"))
    }
    names <- fixed
  }
  out <- methods::as(t(m), "CsparseMatrix")
  dimnames(out) <- list(names, cells)
  out
}

project_name <- function(meta) {
  nm <- meta$dataset_name %||% ""
  if (!nzchar(nm)) "scView" else nm
}

scview_misc <- function(meta) {
  list(
    scview_version = meta$scview_version %||% NA_character_,
    dataset_name = meta$dataset_name %||% NA_character_,
    dataset_id = meta$dataset_id %||% NA_character_,
    exported_at = meta$exported_at %||% format(Sys.time(), "%Y-%m-%dT%H:%M:%S%z"),
    recipe = as.character(meta$recipe %||% character(0)),
    provenance_json = meta$provenance %||% NA_character_,
    colors = meta$colors %||% list(),
    cluster_column = meta$cluster_column %||% NA_character_,
    notes = as.character(meta$notes %||% character(0)),
    left_out = as.character(meta$left_out %||% character(0))
  )
}

pkg_versions <- function(format) {
  pkgs <- if (format == "seurat") c("SeuratObject", "Matrix") else
    c("SingleCellExperiment", "SummarizedExperiment", "Matrix")
  vapply(pkgs, function(p) as.character(utils::packageVersion(p)), character(1))
}
