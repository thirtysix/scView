##
## convert_sce.R
## SingleCellExperiment / SummarizedExperiment -> AnnData h5ad.
##
## What goes where:
##   assays       X = logcounts (else counts, else the first assay); every
##                other assay a layer (counts included when X is logcounts).
##                Sparse stays sparse (CSR); dense goes over in blocks of cells.
##   colData      obs (factors -> categories in level order; S4 columns skipped,
##                nested tables flattened as a.b); sizeFactor and label
##                (sizeFactors, colLabels) are colData columns, so they come too.
##   rowData      var (Ensembl row names with a symbol column: symbols become
##                the gene names, the IDs var gene_ids); names made unique.
##   reducedDims  obsm X_<lower-case name>; PCA's percentVar / varExplained /
##                rotation -> uns pca + varm PCs.
##   altExps      antibody (ADT) -> obsm protein_counts (the app's protein
##                modality), hashtags (HTO) -> obsm hto_counts, others noted.
##   metadata     simple values -> uns r_metadata.
## The h5ad is written element by element (tables first, then one matrix at a
## time), so Python never holds more than one matrix. Everything kept or left
## out goes in the report: returned, and stored in uns scview_conversion.
##

# Values per dense block: about 64 MB of doubles in R. Tests set
# SCVIEW_CONVERT_BLOCK_VALUES small to get many blocks out of a small matrix.
block_values <- function() {
  v <- suppressWarnings(as.numeric(Sys.getenv("SCVIEW_CONVERT_BLOCK_VALUES", "8388608")))
  if (is.na(v) || v < 1) 8388608 else v
}

MAX_PROTEINS <- 1000  # as the backend's core/protein.py: more is not an antibody panel

ADT_NAMES <- c("adt", "adts", "antibody capture", "antibody", "antibodies", "cite",
               "citeseq", "cite-seq", "protein", "proteins")
HTO_NAMES <- c("hto", "htos", "hashtag", "hashtags", "multiplexing capture", "cmo", "cmos")
FEATURE_TYPE_COLUMNS <- c("Type", "type", "feature_type", "feature_types", "FeatureType")
SYMBOL_COLUMNS <- c("Symbol", "symbol", "SYMBOL", "gene_name", "gene_names", "gene_symbol",
                    "gene_symbols", "GeneSymbol", "feature_name")
ENSEMBL_RE <- "^ENS[A-Z]*G[0-9]+(\\.[0-9]+)?$"
# BioLegend TotalSeq / BD AbSeq suffixes, stripped as the backend's protein.py
# display_name does for a 10x import ("CD3_TotalSeqB" -> "CD3").
ANTIBODY_SUFFIX_RE <- "[_.-](TotalSeq[-_]?[ABCD]|TotalA|TotalB|TotalC|AbSeq|ADT)$"

fmt_n <- function(x) format(x, big.mark = ",", scientific = FALSE, trim = TRUE)
plural <- function(n, word) sprintf("%s %s%s", fmt_n(n), word, if (n == 1) "" else "s")

note <- function(rpt, field, fmt, ...) {
  assign(field, c(get(field, envir = rpt), sprintf(fmt, ...)), envir = rpt)
}

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

#' Convert a SingleCellExperiment (kind = "SingleCellExperiment") or any other
#' SummarizedExperiment (kind = "SummarizedExperiment") to an h5ad file.
convert_se_to_h5ad <- function(obj, output_path, kind, progress, object_name = NULL) {

  for (pkg in c("SummarizedExperiment", if (kind == "SingleCellExperiment") "SingleCellExperiment")) {
    if (!suppressMessages(requireNamespace(pkg, quietly = TRUE))) {
      stop(sprintf("the %s package is not installed in the converter", pkg))
    }
  }
  is_sce <- kind == "SingleCellExperiment"

  rpt <- new.env()
  rpt$kept <- character(0)
  rpt$skipped <- character(0)

  source_class <- class(obj)[1]
  source_pkg   <- attr(class(obj), "package") %||% ""
  obj <- as_known_class(obj, kind, rpt)

  # ---- which assays can be read, and which one becomes X -----------------
  # Unreadable assays leave the object first: every SingleCellExperiment
  # accessor runs updateObject() over all assays, and one that cannot be read
  # would break them all.
  assays <- plan_assays(obj, rpt)
  obj <- assays$obj
  x_i <- pick_x(assays)
  obj <- tryCatch(suppressMessages(BiocGenerics::updateObject(obj)), error = function(e) obj)

  n_cells <- ncol(obj)
  if (n_cells == 0 || nrow(obj) == 0) {
    stop(sprintf("the %s has %d cells and %d features: nothing to convert",
                 source_class, n_cells, nrow(obj)))
  }
  progress("converting",
           sprintf("Converting a %s (%s cells x %s features)", source_class,
                   fmt_n(n_cells), fmt_n(nrow(obj))),
           source_class = source_class, n_cells = n_cells, n_genes = nrow(obj))

  py <- py_modules()

  # ---- features: genes vs antibody / hashtag rows of the main experiment ---
  row_df <- SummarizedExperiment::rowData(obj)
  alt <- if (is_sce) alt_exp_roles(obj) else list()
  split <- split_feature_types(row_df, nrow(obj), assays, alt, rpt)
  gene_rows <- split$gene_rows
  n_genes <- length(gene_rows)

  # ---- obs / var ----------------------------------------------------------
  cells <- unique_names(colnames(obj), n_cells, "cell", rpt, "cell names")
  genes <- gene_names(rownames(obj), row_df, gene_rows, rpt)

  col_df <- SummarizedExperiment::colData(obj)
  obs <- frame_to_py(col_df, cells, "colData", rpt, py)
  note(rpt, "kept", "cell metadata (colData): %d of %d columns as obs", obs$n, obs$total)
  if ("sizeFactor" %in% colnames(col_df)) note(rpt, "kept", "size factors (sizeFactors) as obs sizeFactor")
  if ("label" %in% colnames(col_df)) note(rpt, "kept", "cell labels (colLabels) as obs label")

  var_cols <- row_df[gene_rows, , drop = FALSE]
  var <- frame_to_py(var_cols, genes$names, "rowData", rpt, py,
                     extra = if (!is.null(genes$ids) && !("gene_ids" %in% colnames(var_cols)))
                       list(gene_ids = genes$ids))
  note(rpt, "kept", "gene metadata (rowData): %d of %d columns as var", var$n, var$total)
  rr <- tryCatch(SummarizedExperiment::rowRanges(obj), error = function(e) NULL)
  if (methods::is(rr, "GRanges") && length(rr) > 0) {
    note(rpt, "skipped", "gene coordinates (rowRanges): not converted")
  }

  # ---- obsm / varm / uns --------------------------------------------------
  obsm <- list()
  uns <- list()
  varm_pcs <- NULL
  if (is_sce) {
    rd <- convert_reduced_dims(obj, n_cells, rownames(obj)[gene_rows], rpt, py)
    obsm <- rd$obsm
    if (length(rd$pca)) uns$pca <- reticulate::py_dict(names(rd$pca), unname(rd$pca), convert = FALSE)
    varm_pcs <- rd$pcs
    obsm <- c(obsm, convert_alt_exps(obj, alt, cells, rpt, py))
    obsm <- c(obsm, spatial_coords(obj, rpt, py))
    skip_graphs(obj, rpt)
  }
  obsm <- c(obsm, split$obsm(obj, cells, py))

  md <- S4Vectors::metadata(obj)
  if (length(md) > 0) {
    meta_py <- simple_value(md, "metadata", rpt, py, depth = 0)
    if (!is.null(meta_py)) {
      uns$r_metadata <- meta_py
      n_kept <- reticulate::py_len(meta_py)
      note(rpt, "kept", "metadata: %d of %d entries in uns r_metadata", n_kept, length(md))
    }
  }

  # ---- the report (complete before anything is written) -------------------
  x_kind <- assays$kind[x_i]
  layer_is <- setdiff(which(assays$ok), x_i)
  x_note <- sprintf("X: the %s assay (%s)", assays$name[x_i], x_kind)
  layer_notes <- if (length(layer_is)) {
    sprintf("layers: %s", paste(sprintf("%s (%s)", assays$name[layer_is], assays$kind[layer_is]),
                                collapse = ", "))
  }
  rpt$kept <- c(x_note, layer_notes, rpt$kept)

  report <- list(
    source_class   = source_class,
    source_package = source_pkg,
    read_as        = kind,
    object_name    = object_name %||% "",
    n_cells        = n_cells,
    n_genes        = n_genes,
    x              = assays$name[x_i],
    layers         = as.list(assays$name[layer_is]),
    obsm           = as.list(names(obsm)),
    kept           = as.list(rpt$kept),
    skipped        = as.list(rpt$skipped),
    converter      = sprintf("scView converter: R %s, SingleCellExperiment %s",
                             getRversion(), utils::packageVersion("SingleCellExperiment"))
  )
  uns$scview_conversion <- report_to_py(report, py)

  # ---- write: tables and embeddings, then one matrix at a time ------------
  partial <- paste0(output_path, ".partial")
  if (file.exists(partial)) file.remove(partial)
  ok <- FALSE
  f <- NULL
  on.exit(if (!ok) {
    if (!is.null(f)) try(f$close(), silent = TRUE)
    if (file.exists(partial)) file.remove(partial)
  }, add = TRUE)

  adata <- py$ad$AnnData(obs = obs$df, var = var$df)
  for (key in names(obsm)) reticulate::py_set_item(adata$obsm, key, obsm[[key]])
  if (!is.null(varm_pcs)) reticulate::py_set_item(adata$varm, "PCs", varm_pcs)
  for (key in names(uns)) reticulate::py_set_item(adata$uns, key, uns[[key]])
  progress("writing", "Writing cell and gene tables")
  adata$write_h5ad(partial)
  rm(adata, obs, var, obsm, uns)

  f <- py$h5py$File(partial, "r+")

  rows <- if (n_genes == nrow(obj)) NULL else gene_rows
  progress("writing", sprintf("Writing X (the %s assay)", assays$name[x_i]))
  write_assay(f, "X", assays$get(x_i), rows, n_cells, py)
  layers <- require_dict_group(f, "layers")
  for (i in layer_is) {
    progress("writing", sprintf("Writing the %s assay", assays$name[i]))
    write_assay(layers, assays$name[i], assays$get(i), rows, n_cells, py)
    invisible(gc(verbose = FALSE))
  }
  f$close()
  f <- NULL

  if (file.exists(output_path)) file.remove(output_path)
  if (!file.rename(partial, output_path)) stop("could not move the h5ad into place")
  ok <- TRUE

  list(
    success = TRUE,
    message = sprintf("Converted a %s (%s cells x %s genes)", source_class,
                      fmt_n(n_cells), fmt_n(n_genes)),
    report  = report
  )
}

#' A class from a package that is not installed here (SpatialExperiment, ...)
#' read as the class it is built on: the slots are the same, only the extra
#' methods are missing.
as_known_class <- function(obj, kind, rpt) {
  if (isTRUE(tryCatch(methods::is(obj, kind), error = function(e) FALSE))) return(obj)
  pkg <- if (kind == "SingleCellExperiment") "SingleCellExperiment" else "SummarizedExperiment"
  note(rpt, "kept", "a %s (its package is not installed here) read as a %s",
       class(obj)[1], kind)
  attr(obj, "class") <- structure(kind, package = pkg)
  obj
}

# ---------------------------------------------------------------------------
# Python side
# ---------------------------------------------------------------------------

py_modules <- function() {
  suppressMessages(library(reticulate))
  ad <- tryCatch(import("anndata", convert = FALSE), error = function(e) {
    stop("Python anndata package is not available: ", conditionMessage(e))
  })
  write_elem <- tryCatch(import("anndata.io", convert = FALSE)$write_elem,
                         error = function(e) import("anndata.experimental", convert = FALSE)$write_elem)
  list(
    ad         = ad,
    np         = import("numpy", convert = FALSE),
    pd         = import("pandas", convert = FALSE),
    sp         = import("scipy.sparse", convert = FALSE),
    h5py       = import("h5py", convert = FALSE),
    builtins   = import_builtins(convert = FALSE),
    write_elem = write_elem
  )
}

#' An R vector as a 1-d NumPy array (a plain R vector would become a list).
np_array <- function(x) {
  if (is.character(x)) x <- enc2utf8(x)
  reticulate::r_to_py(array(x, dim = length(x)), convert = FALSE)
}

#' An R matrix (any numeric type) as a float NumPy array, same shape.
np_matrix <- function(m, py, dtype = "float64") {
  m <- as.matrix(m)
  storage.mode(m) <- "double"
  py$np$ascontiguousarray(reticulate::r_to_py(unname(m), convert = FALSE), dtype = dtype)
}

#' The report as a dict for uns: every list an array of strings, so it reads
#' back the same however many entries it has.
report_to_py <- function(report, py) {
  vals <- lapply(report, function(v) {
    if (is.list(v)) np_array(as.character(unlist(v))) else reticulate::r_to_py(v, convert = FALSE)
  })
  reticulate::py_dict(names(report), unname(vals), convert = FALSE)
}

#' An h5ad group of encoding "dict" (layers, uns), created if anndata left it out.
require_dict_group <- function(f, name) {
  g <- f$require_group(name)
  attrs <- g$attrs
  if (!reticulate::py_to_r(attrs$`__contains__`("encoding-type"))) {
    reticulate::py_set_item(attrs, "encoding-type", "dict")
    reticulate::py_set_item(attrs, "encoding-version", "0.1.0")
  }
  g
}

# ---------------------------------------------------------------------------
# Assays
# ---------------------------------------------------------------------------

#' Names, readability and storage of every assay; get(i) returns assay i.
plan_assays <- function(obj, rpt) {
  all <- SummarizedExperiment::assays(obj, withDimnames = FALSE)
  n <- length(all)
  if (n == 0) stop("the object has no assays: there are no expression values to convert")

  nms <- names(all)
  if (is.null(nms)) nms <- rep("", n)
  unnamed <- is.na(nms) | !nzchar(nms)
  nms[unnamed] <- paste0("assay", which(unnamed))
  if (any(unnamed)) note(rpt, "kept", "%s named assay1, assay2, ...", plural(sum(unnamed), "unnamed assay"))
  nms <- gsub("/", "_", make.unique(nms, sep = "_"), fixed = TRUE)

  ok <- logical(n)
  kind <- character(n)
  for (i in seq_len(n)) {
    m <- all[[i]]
    why <- tryCatch({
      corner <- as.matrix(m[seq_len(min(2L, nrow(m))), seq_len(min(2L, ncol(m))), drop = FALSE])
      if (is.numeric(corner) || is.logical(corner)) NULL else sprintf("its values are %s", typeof(corner))
    }, error = function(e) {
      sprintf("its values could not be read (%s; an HDF5Array or TileDB assay keeps them in a separate file, which is not part of the .rds)",
              conditionMessage(e))
    })
    if (is.null(why)) {
      ok[i] <- TRUE
      kind[i] <- if (is_sparse_assay(m)) "sparse" else "dense"
    } else {
      note(rpt, "skipped", "assay %s (%s): %s", nms[i], class(m)[1], why)
    }
  }
  if (!any(ok)) stop("none of the assays could be read: ", paste(rpt$skipped, collapse = "; "))
  if (!all(ok)) {
    all <- all[ok]
    SummarizedExperiment::assays(obj, withDimnames = FALSE) <- all
  }

  list(obj = obj, name = nms[ok], ok = rep(TRUE, sum(ok)), kind = kind[ok], n = sum(ok),
       get = function(i) all[[i]])
}

#' X: logcounts when there is one, else counts, else the first readable assay.
pick_x <- function(assays) {
  for (want in c("logcounts", "counts")) {
    i <- match(want, assays$name)
    if (!is.na(i) && assays$ok[i]) return(i)
  }
  which(assays$ok)[1]
}

is_sparse_assay <- function(m) {
  methods::is(m, "sparseMatrix") ||
    methods::is(m, "SparseArray") ||
    (methods::is(m, "DelayedArray") && isTRUE(tryCatch(DelayedArray::is_sparse(m), error = function(e) FALSE)))
}

#' Any sparse assay as a dgCMatrix (genes x cells, numeric).
to_dgc <- function(m) {
  if (inherits(m, "dgCMatrix")) return(m)
  if (methods::is(m, "DelayedArray") || methods::is(m, "SparseArray")) {
    m <- methods::as(m, "dgCMatrix")
    return(m)
  }
  m <- methods::as(m, "CsparseMatrix")
  m <- methods::as(m, "generalMatrix")
  methods::as(m, "dMatrix")
}

#' Write one assay (genes x cells) under `key` of an h5py group as cells x
#' genes, keeping only `rows` (genes) when given.
write_assay <- function(group, key, m, rows, n_cells, py) {
  if (is_sparse_assay(m)) {
    m <- to_dgc(m)
    if (!is.null(rows)) m <- m[rows, , drop = FALSE]
    write_csr(group, key, m, n_cells, py)
  } else {
    write_dense(group, key, m, rows, n_cells, py)
  }
}

#' A genes x cells dgCMatrix is, array for array, the CSR form of the cells x
#' genes matrix: column pointers = row pointers, row indices = column indices.
#' No transpose, no copy in R.
write_csr <- function(group, key, m, n_cells, py) {
  csr <- py$sp$csr_matrix(
    reticulate::tuple(
      py$np$asarray(np_array(m@x), dtype = "float32"),
      np_array(m@i),
      np_array(m@p)
    ),
    shape = reticulate::tuple(as.integer(n_cells), as.integer(nrow(m)))
  )
  py$write_elem(group, key, csr)
}

#' A dense assay, block of cells by block: only one block is ever realised
#' (a DelayedArray is computed block by block) or in Python.
write_dense <- function(group, key, m, rows, n_cells, py) {
  n_out <- if (is.null(rows)) nrow(m) else length(rows)
  ds <- group$create_dataset(key, shape = reticulate::tuple(as.integer(n_cells), as.integer(n_out)),
                             dtype = "float32")
  reticulate::py_set_item(ds$attrs, "encoding-type", "array")
  reticulate::py_set_item(ds$attrs, "encoding-version", "0.2.0")

  step <- max(1L, as.integer(floor(block_values() / max(1, nrow(m)))))
  for (start in seq(1L, n_cells, by = step)) {
    end <- min(start + step - 1L, n_cells)
    blk <- as.matrix(m[, start:end, drop = FALSE])
    if (!is.null(rows)) blk <- blk[rows, , drop = FALSE]
    storage.mode(blk) <- "double"
    # genes x cells in R's column order is cells x genes in C order: .T, no copy
    arr <- py$np$ascontiguousarray(reticulate::r_to_py(unname(blk), convert = FALSE)$T,
                                   dtype = "float32")
    reticulate::py_set_item(ds, py$builtins$slice(as.integer(start - 1L), as.integer(end)), arr)
  }
  invisible(NULL)
}

# ---------------------------------------------------------------------------
# Cells, genes and the two tables
# ---------------------------------------------------------------------------

#' Names that anndata can use: missing ones numbered, duplicates made unique
#' the way scanpy's var_names_make_unique does (A, A-1, A-2).
unique_names <- function(nms, n, stem, rpt, what) {
  if (is.null(nms)) {
    note(rpt, "kept", "%s: none in the object, numbered %s_1 to %s_%d", what, stem, stem, n)
    return(paste0(stem, "_", seq_len(n)))
  }
  nms <- enc2utf8(as.character(nms))
  bad <- is.na(nms) | !nzchar(nms)
  if (any(bad)) {
    nms[bad] <- paste0(stem, "_", which(bad))
    note(rpt, "kept", "%s: %s missing, numbered %s_<position>", what, fmt_n(sum(bad)), stem)
  }
  dup <- duplicated(nms)
  if (any(dup)) {
    example <- nms[dup][1]
    nms <- make.unique(nms, sep = "-")
    note(rpt, "kept", "%s: %s made unique with a -1, -2 suffix (e.g. %s-1)",
         what, plural(sum(dup), "duplicate"), example)
  }
  nms
}

#' Gene names for the kept rows: the row names, unless they are Ensembl IDs and
#' rowData has a symbol column (then symbols, with the IDs as gene_ids).
gene_names <- function(rn, row_df, gene_rows, rpt) {
  n <- length(gene_rows)
  sym_col <- base::intersect(SYMBOL_COLUMNS, colnames(row_df))[1]
  ids <- NULL
  nms <- if (is.null(rn)) NULL else as.character(rn)[gene_rows]

  if (!is.na(sym_col)) {
    sym <- as.character(row_df[[sym_col]])[gene_rows]
    has <- !is.na(sym) & nzchar(sym)
    ensembl <- !is.null(nms) && mean(grepl(ENSEMBL_RE, nms)) >= 0.9
    if ((is.null(nms) || ensembl) && mean(has) >= 0.9) {
      ids <- nms
      nms <- ifelse(has, sym, if (is.null(ids)) NA_character_ else ids)
      note(rpt, "kept", "gene names from the rowData column %s%s", sym_col,
           if (is.null(ids)) "" else " (the row names are Ensembl IDs: kept as var gene_ids)")
    }
  }
  list(names = unique_names(nms, n, "gene", rpt, "gene names"), ids = ids)
}

#' Columns of a (possibly nested) DataFrame as a flat named list, nested
#' tables as parent.child.
flatten_frame <- function(df, prefix = "") {
  vals <- list()
  nms <- character(0)
  cn <- colnames(df)
  for (i in seq_along(cn)) {
    v <- df[[i]]
    nm <- if (nzchar(prefix)) paste0(prefix, ".", cn[i]) else cn[i]
    if (methods::is(v, "DataFrame") || is.data.frame(v)) {
      inner <- flatten_frame(v, nm)
      vals <- c(vals, unname(inner))
      nms <- c(nms, names(inner))
    } else {
      vals[[length(vals) + 1L]] <- v
      nms <- c(nms, nm)
    }
  }
  names(vals) <- nms
  vals
}

#' One column as a NumPy array / pandas Categorical, or a sentence saying why
#' it cannot be one.
py_column <- function(v, n, py) {
  if (methods::is(v, "Rle")) v <- S4Vectors::decode(v)
  if (!is.null(dim(v)) && length(dim(v)) > 1) return("a matrix in one column")
  if (isS4(v)) return(sprintf("a %s, not a plain column", class(v)[1]))
  if (is.list(v)) return("a list, not a plain column")
  if (length(v) != n) return("its length does not match")
  if (inherits(v, c("Date", "POSIXt"))) v <- as.character(v)
  if (inherits(v, "difftime")) v <- as.numeric(v)

  if (is.factor(v)) return(categorical(v, py))
  if (is.character(v)) return(categorical(factor(v), py))
  if (is.logical(v)) {
    if (anyNA(v)) {
      return(categorical(factor(ifelse(v, "TRUE", "FALSE"), levels = c("FALSE", "TRUE")), py))
    }
    return(np_array(v))
  }
  if (is.integer(v)) return(np_array(if (anyNA(v)) as.numeric(v) else v))
  if (is.double(v)) return(np_array(v))
  sprintf("of type %s", typeof(v))
}

#' A factor as a pandas Categorical: its levels in order, NA missing.
categorical <- function(f, py) {
  codes <- as.integer(f) - 1L
  codes[is.na(codes)] <- -1L
  py$pd$Categorical$from_codes(np_array(codes),
                               categories = py$pd$Index(np_array(levels(f)), dtype = "object"),
                               ordered = is.ordered(f))
}

#' colData / rowData as a pandas DataFrame indexed by `index`; `extra` adds
#' plain columns. Returns list(df, n = columns kept).
frame_to_py <- function(df, index, what, rpt, py, extra = NULL) {
  cols <- c(flatten_frame(df), extra)
  keys <- character(0)
  vals <- list()
  for (i in seq_along(cols)) {
    nm <- names(cols)[i]
    col <- py_column(cols[[i]], length(index), py)
    if (is.character(col)) {
      note(rpt, "skipped", "%s column %s: %s", what, nm, col)
      next
    }
    keys <- c(keys, nm)
    vals[[length(vals) + 1L]] <- col
  }
  # anndata reserves _index and nests names at "/"
  keys[is.na(keys) | !nzchar(keys)] <- paste0("column_", which(is.na(keys) | !nzchar(keys)))
  keys[keys == "_index"] <- "_index_"
  keys <- make.unique(gsub("/", "_", keys, fixed = TRUE), sep = "_")

  idx <- py$pd$Index(np_array(index), dtype = "object")
  data <- reticulate::py_dict(keys, vals, convert = FALSE)
  out <- py$pd$DataFrame(data, index = idx)
  list(df = out, n = length(keys) - length(extra), total = length(cols) - length(extra))
}

# ---------------------------------------------------------------------------
# Antibody and hashtag rows of the main experiment
# ---------------------------------------------------------------------------

#' Rows whose feature type (rowData Type, as DropletUtils::read10xCounts
#' writes it) is Antibody Capture or Multiplexing Capture leave the genes for
#' obsm protein_counts / hto_counts, when an altExp does not already hold them
#' and there is a counts assay to take them from.
split_feature_types <- function(row_df, n_rows, assays, alt, rpt) {
  none <- list(gene_rows = seq_len(n_rows), obsm = function(obj, cells, py) list())
  ft_col <- base::intersect(FEATURE_TYPE_COLUMNS, colnames(row_df))[1]
  if (is.na(ft_col)) return(none)
  ft <- as.character(row_df[[ft_col]])
  ab  <- which(ft == "Antibody Capture")
  hto <- which(ft == "Multiplexing Capture")
  if (any(vapply(alt, function(a) identical(a$role, "protein"), logical(1)))) ab <- integer(0)
  if (any(vapply(alt, function(a) identical(a$role, "hashtag"), logical(1)))) hto <- integer(0)
  if (length(ab) + length(hto) == 0) return(none)

  counts_i <- match("counts", assays$name)
  gene_rows <- setdiff(seq_len(n_rows), c(ab, hto))
  if (is.na(counts_i) || !assays$ok[counts_i] || length(gene_rows) == 0) {
    note(rpt, "skipped", "%s antibody / hashtag rows (rowData %s) stay with the genes: %s",
         fmt_n(length(ab) + length(hto)), ft_col,
         if (length(gene_rows) == 0) "there are no other rows" else "there is no counts assay")
    return(none)
  }
  if (length(ab) > MAX_PROTEINS) {
    note(rpt, "skipped", "%s Antibody Capture rows stay with the genes: more than %d is not an antibody panel",
         fmt_n(length(ab)), MAX_PROTEINS)
    gene_rows <- sort(c(gene_rows, ab))
    ab <- integer(0)
  }

  obsm <- function(obj, cells, py) {
    out <- list()
    counts <- assays$get(counts_i)
    if (length(ab)) {
      out$protein_counts <- counts_frame(counts[ab, , drop = FALSE],
                                         antibody_names(row_df[ab, , drop = FALSE], rownames(obj)[ab]),
                                         cells, py)
      note(rpt, "kept", "%d Antibody Capture rows (rowData %s) as the protein modality (obsm protein_counts)",
           length(ab), ft_col)
    }
    if (length(hto)) {
      out$hto_counts <- counts_frame(counts[hto, , drop = FALSE],
                                     antibody_names(row_df[hto, , drop = FALSE], rownames(obj)[hto]),
                                     cells, py)
      note(rpt, "kept", "%d Multiplexing Capture rows (rowData %s) as hashtag counts (obsm hto_counts)",
           length(hto), ft_col)
    }
    out
  }
  list(gene_rows = gene_rows, obsm = obsm)
}

#' Antibody / hashtag names as a direct 10x import shows them: the feature
#' name (rowData Symbol, as read10xCounts stores it; the row names hold the
#' 10x IDs) without the TotalSeq suffix, so "IgG1_control_TotalSeqB" reads
#' "IgG1_control" and is recognised as an isotype control.
antibody_names <- function(row_df, rn) {
  sym_col <- base::intersect(SYMBOL_COLUMNS, colnames(row_df))[1]
  nms <- if (!is.na(sym_col)) as.character(row_df[[sym_col]]) else NULL
  if (is.null(nms) || any(is.na(nms) | !nzchar(nms))) nms <- if (is.null(rn)) NULL else as.character(rn)
  if (is.null(nms)) return(NULL)
  short <- sub(ANTIBODY_SUFFIX_RE, "", nms, ignore.case = TRUE)
  ifelse(nzchar(short), short, nms)
}

#' A small features x cells count matrix as a cells x features DataFrame.
counts_frame <- function(m, feature_names, cells, py) {
  dm <- as.matrix(m)
  rep_local <- new.env()
  rep_local$kept <- character(0)
  feats <- unique_names(feature_names, nrow(dm), "feature", rep_local, "features")
  py$pd$DataFrame(np_matrix(base::t(dm), py), index = py$pd$Index(np_array(cells), dtype = "object"),
                  columns = py$pd$Index(np_array(feats), dtype = "object"))
}

# ---------------------------------------------------------------------------
# reducedDims, altExps, spatial coordinates, graphs
# ---------------------------------------------------------------------------

#' obsm key for a reducedDim name: PCA -> X_pca, UMAP.HARMONY -> X_umap_harmony.
obsm_key <- function(nm, used) {
  base <- gsub("_+", "_", gsub("[^a-z0-9]", "_", tolower(nm)))
  base <- gsub("^_|_$", "", base)
  if (!nzchar(base)) base <- "reduction"
  key <- paste0("X_", base)
  k <- 2L
  while (key %in% used) {
    key <- sprintf("X_%s_%d", base, k)
    k <- k + 1L
  }
  key
}

convert_reduced_dims <- function(obj, n_cells, gene_ids, rpt, py) {
  nms <- SingleCellExperiment::reducedDimNames(obj)
  out <- list(obsm = list(), pca = list(), pcs = NULL)
  if (length(nms) == 0) return(out)
  kept <- character(0)
  for (i in seq_along(nms)) {
    m <- SingleCellExperiment::reducedDim(obj, i, withDimnames = FALSE)
    nm <- if (is.na(nms[i]) || !nzchar(nms[i])) sprintf("reduction%d", i) else nms[i]
    if (methods::is(m, "LinearEmbeddingMatrix")) m <- SingleCellExperiment::sampleFactors(m)
    if (methods::is(m, "DataFrame") || is.data.frame(m)) m <- as.matrix(as.data.frame(m))
    mat <- tryCatch(as.matrix(m), error = function(e) NULL)
    if (is.null(mat) || !(is.numeric(mat) || is.logical(mat)) || nrow(mat) != n_cells) {
      note(rpt, "skipped", "reducedDim %s (%s): not a numeric cells x dimensions matrix", nm, class(m)[1])
      next
    }
    key <- obsm_key(nm, names(out$obsm))
    out$obsm[[key]] <- np_matrix(mat, py)
    kept <- c(kept, sprintf("%s as %s", nm, key))

    if (key == "X_pca") {
      k <- ncol(mat)
      pv <- attr(m, "percentVar")
      ve <- attr(m, "varExplained")
      rot <- attr(m, "rotation")
      if (is.numeric(pv) && length(pv) == k) out$pca$variance_ratio <- np_array(as.numeric(pv) / 100)
      if (is.numeric(ve) && length(ve) == k) out$pca$variance <- np_array(as.numeric(ve))
      if (is.matrix(rot) && ncol(rot) == k) {
        full <- matrix(0, nrow = length(gene_ids), ncol = k)
        at <- if (!is.null(rownames(rot)) && !is.null(gene_ids)) match(rownames(rot), gene_ids)
              else if (nrow(rot) == length(gene_ids)) seq_len(nrow(rot))
        if (!is.null(at) && sum(!is.na(at)) > 0) {
          full[at[!is.na(at)], ] <- rot[!is.na(at), , drop = FALSE]
          out$pcs <- np_matrix(full, py)
        }
      }
      extras <- c(if (!is.null(out$pca$variance_ratio)) "percentVar",
                  if (!is.null(out$pca$variance)) "varExplained",
                  if (!is.null(out$pcs)) "rotation")
      if (length(extras)) {
        note(rpt, "kept", "PCA %s as uns pca%s", paste(extras, collapse = ", "),
             if (!is.null(out$pcs)) " and varm PCs" else "")
      }
    }
  }
  if (length(kept)) note(rpt, "kept", "reduced dimensions (reducedDims): %s", paste(kept, collapse = ", "))
  out
}

#' What each altExp is for: list(name, role = "protein" | "hashtag" | NA).
alt_exp_roles <- function(obj) {
  nms <- SingleCellExperiment::altExpNames(obj)
  lapply(seq_along(nms), function(i) {
    ae <- SingleCellExperiment::altExp(obj, i, withDimnames = FALSE)
    role <- NA_character_
    low <- tolower(nms[i])
    ft_col <- base::intersect(FEATURE_TYPE_COLUMNS, colnames(SummarizedExperiment::rowData(ae)))[1]
    ft <- if (is.na(ft_col)) character(0) else unique(as.character(SummarizedExperiment::rowData(ae)[[ft_col]]))
    if (low %in% ADT_NAMES || identical(ft, "Antibody Capture")) role <- "protein"
    if (low %in% HTO_NAMES || identical(ft, "Multiplexing Capture")) role <- "hashtag"
    list(index = i, name = nms[i], role = role)
  })
}

#' The counts of an altExp: its counts assay, or a lone assay of whole
#' non-negative numbers. NULL when it has neither.
alt_counts <- function(ae) {
  an <- SummarizedExperiment::assayNames(ae)
  if ("counts" %in% an) return(SummarizedExperiment::assay(ae, "counts", withDimnames = FALSE))
  if (length(SummarizedExperiment::assays(ae)) == 1) {
    m <- SummarizedExperiment::assay(ae, 1, withDimnames = FALSE)
    v <- as.matrix(m)
    if (is.numeric(v) && all(v >= 0, na.rm = TRUE) && all(v == round(v), na.rm = TRUE)) return(m)
  }
  NULL
}

convert_alt_exps <- function(obj, alt, cells, rpt, py) {
  out <- list()
  for (a in alt) {
    ae <- SingleCellExperiment::altExp(obj, a$index, withDimnames = FALSE)
    nf <- nrow(ae)
    if (is.na(a$role)) {
      note(rpt, "skipped", "altExp %s (%s features): not converted (only antibody and hashtag altExps are)",
           a$name, fmt_n(nf))
      next
    }
    key <- if (a$role == "protein") "protein_counts" else "hto_counts"
    what <- if (a$role == "protein") "the protein modality" else "hashtag counts"
    if (!is.null(out[[key]])) {
      note(rpt, "skipped", "altExp %s (%s features): %s already came from another altExp",
           a$name, fmt_n(nf), what)
      next
    }
    if (nf == 0 || nf > MAX_PROTEINS) {
      note(rpt, "skipped", "altExp %s (%s features): too %s for %s", a$name, fmt_n(nf),
           if (nf == 0) "few" else "many", what)
      next
    }
    counts <- alt_counts(ae)
    if (is.null(counts)) {
      note(rpt, "skipped", "altExp %s: no counts assay, so not %s (it needs raw counts)", a$name, what)
      next
    }
    out[[key]] <- counts_frame(counts, antibody_names(SummarizedExperiment::rowData(ae), rownames(ae)),
                               cells, py)
    note(rpt, "kept", "altExp %s (%s features) as %s (obsm %s)", a$name, fmt_n(nf), what, key)
  }
  out
}

#' SpatialExperiment keeps spot coordinates in int_colData: obsm spatial (the
#' squidpy key; not shown as an embedding). Images are not converted.
spatial_coords <- function(obj, rpt, py) {
  out <- list()
  icd <- tryCatch(SingleCellExperiment::int_colData(obj), error = function(e) NULL)
  if (!is.null(icd) && "spatialCoords" %in% colnames(icd)) {
    sc <- tryCatch(as.matrix(icd[["spatialCoords"]]), error = function(e) NULL)
    if (!is.null(sc) && is.numeric(sc) && ncol(sc) > 0) {
      out$spatial <- np_matrix(sc, py)
      note(rpt, "kept", "spatial coordinates (spatialCoords) as obsm spatial")
    }
  }
  imd <- tryCatch(SingleCellExperiment::int_metadata(obj)$imgData, error = function(e) NULL)
  if (!is.null(imd) && NROW(imd) > 0) note(rpt, "skipped", "images (imgData): not converted")
  out
}

skip_graphs <- function(obj, rpt) {
  cp <- tryCatch(SingleCellExperiment::colPairNames(obj), error = function(e) character(0))
  if (length(cp)) {
    note(rpt, "skipped", "cell graphs (colPairs %s): not converted; scView builds its own neighbour graph",
         paste(cp, collapse = ", "))
  }
  rp <- tryCatch(SingleCellExperiment::rowPairNames(obj), error = function(e) character(0))
  if (length(rp)) note(rpt, "skipped", "gene graphs (rowPairs %s): not converted", paste(rp, collapse = ", "))
}

# ---------------------------------------------------------------------------
# metadata -> uns
# ---------------------------------------------------------------------------

#' A metadata value as something uns can hold (scalars, arrays, tables,
#' nested dicts), or NULL with a note saying what was left out.
simple_value <- function(v, path, rpt, py, depth) {
  if (is.null(v)) return(NULL)
  if (is.factor(v)) v <- as.character(v)
  if (is.atomic(v) && is.null(dim(v)) && (is.numeric(v) || is.character(v) || is.logical(v))) {
    if (length(v) > 1e6) {
      note(rpt, "skipped", "%s: a vector of %s values (too long for uns)", path, fmt_n(length(v)))
      return(NULL)
    }
    if (is.character(v)) v[is.na(v)] <- "NA"
    if (is.integer(v) && anyNA(v)) v <- as.numeric(v)
    if (is.logical(v) && anyNA(v)) v <- ifelse(is.na(v), "NA", ifelse(v, "TRUE", "FALSE"))
    if (length(v) == 1) return(reticulate::r_to_py(if (is.character(v)) enc2utf8(v) else v, convert = FALSE))
    return(np_array(v))
  }
  if (is.matrix(v) && (is.numeric(v) || is.logical(v)) && length(v) <= 1e7) return(np_matrix(v, py))
  if ((is.data.frame(v) || methods::is(v, "DataFrame")) && nrow(v) <= 1e6) {
    idx <- rownames(v)
    if (is.null(idx)) idx <- as.character(seq_len(nrow(v)))
    return(frame_to_py(v, make.unique(as.character(idx)), path, rpt, py)$df)
  }
  if (is.list(v) && !isS4(v) && depth < 5) {
    nms <- names(v)
    if (is.null(nms)) nms <- rep("", length(v))
    nms[is.na(nms) | !nzchar(nms)] <- as.character(which(is.na(nms) | !nzchar(nms)))
    nms <- make.unique(gsub("/", "_", nms, fixed = TRUE), sep = "_")
    keys <- character(0)
    vals <- list()
    for (i in seq_along(v)) {
      pv <- simple_value(v[[i]], paste0(path, "$", nms[i]), rpt, py, depth + 1)
      if (is.null(pv)) next
      keys <- c(keys, nms[i])
      vals[[length(vals) + 1L]] <- pv
    }
    if (length(keys) == 0) return(NULL)
    return(reticulate::py_dict(keys, vals, convert = FALSE))
  }
  note(rpt, "skipped", "%s (%s): not a simple value", path, class(v)[1])
  NULL
}
