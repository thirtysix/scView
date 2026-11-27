##
## convert.R
## Core conversion logic: R objects (RDS / RData) -> AnnData h5ad.
## Seurat objects: SeuratObject + reticulate + Python anndata (no sceasy
## dependency). SingleCellExperiment and SummarizedExperiment objects (and
## classes built on them): convert_sce.R. Anything else is refused, naming its
## class.
##

library(Matrix)

# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

#' Convert the R object stored in an RDS (or RData) file to an h5ad file.
#'
#' @param input_path    Path to the .rds or .RData file.
#' @param output_path   Path where the .h5ad file should be written.
#' @param progress_path Optional JSON file each stage is written to, so the
#'                      backend can say what is being converted meanwhile.
#' @return A list with \code{success} (logical), \code{message} or
#'         \code{error}, and \code{report} (the class, what was kept and what
#'         was left out).
convert_rds_to_h5ad <- function(input_path, output_path, progress_path = NULL) {

  progress <- progress_writer(progress_path)

  if (!file.exists(input_path)) {
    return(finish(progress, list(success = FALSE, error = paste0("Input file not found: ", input_path))))
  }

  # ------------------------------------------------------------------
  # 1. Load the object and decide what it is
  # ------------------------------------------------------------------
  progress("reading", "Reading the R file")
  loaded <- load_r_object(input_path)
  if (!is.null(loaded$error)) {
    return(finish(progress, list(success = FALSE, error = loaded$error)))
  }
  obj  <- loaded$obj
  kind <- object_kind(obj)

  # ------------------------------------------------------------------
  # 2. Seurat: the original path, unchanged
  # ------------------------------------------------------------------
  if (kind == "Seurat") {
    # kept in the progress file: after an out-of-memory kill the backend says DietSeurat
    # only when it would drop something
    progress("converting", "Converting a Seurat object", source_class = "Seurat",
             droppable_layers = seurat_droppable_layers(obj))
    version_info <- detect_seurat_version(obj)
    cat(sprintf("[convert] Detected Seurat version: %s\n", version_info))

    # Handed over in an environment, so manual_convert holds the only reference and
    # can free the object (its data / scale.data layers, graphs) once it is read.
    object_name <- loaded$name
    box <- new.env(parent = emptyenv())
    box$obj <- obj
    rm(obj, loaded)
    result <- tryCatch(
      manual_convert(box, output_path, version_info, progress),
      error = function(e) list(success = FALSE, error = conversion_error(e))
    )
    result$report <- list(
      source_class = "Seurat", detail = paste("Seurat", version_info),
      object_name = object_name %||% "", kept = as.list(result$kept %||% character(0)),
      skipped = as.list(result$skipped %||% character(0))
    )
    result$kept <- NULL
    result$skipped <- NULL
    return(finish(progress, result))
  }

  # ------------------------------------------------------------------
  # 3. SingleCellExperiment / SummarizedExperiment
  # ------------------------------------------------------------------
  if (kind %in% c("SingleCellExperiment", "SummarizedExperiment")) {
    result <- tryCatch(
      convert_se_to_h5ad(obj, output_path, kind, progress, object_name = loaded$name),
      error = function(e) list(success = FALSE, error = conversion_error(e))
    )
    return(finish(progress, result))
  }

  finish(progress, list(success = FALSE, error = unsupported_message(obj)))
}

#' The error a failed conversion answers with, without reticulate's advice to run
#' py_last_error() (nobody using scView can).
conversion_error <- function(e) {
  msg <- sub("\\s*Run `reticulate::py_last_error\\(\\)` for details\\.?\\s*$", "", conditionMessage(e))
  paste0("Conversion failed: ", msg)
}

#' The previous entry point (Seurat only), kept for callers that still use it.
convert_seurat_to_h5ad <- function(input_path, output_path) {
  convert_rds_to_h5ad(input_path, output_path)
}

# ---------------------------------------------------------------------------
# Progress file (read by the backend while the conversion runs)
# ---------------------------------------------------------------------------

#' A function that records a stage: progress(stage, message, key = value...).
#' Keys given once (the class, the size) stay in every later write. Written
#' to a temporary file and renamed, so a reader never sees half a file.
progress_writer <- function(path) {
  state <- new.env()
  state$info <- list()
  function(stage, message, ...) {
    extra <- list(...)
    for (k in names(extra)) state$info[[k]] <- extra[[k]]
    cat(sprintf("[convert] %s\n", message))
    if (is.null(path) || !nzchar(path)) return(invisible(NULL))
    body <- c(list(stage = stage, message = message,
                   updated_at = format(Sys.time(), "%Y-%m-%dT%H:%M:%S%z")),
              state$info)
    tmp <- paste0(path, ".tmp")
    tryCatch({
      writeLines(jsonlite::toJSON(body, auto_unbox = TRUE), tmp)
      file.rename(tmp, path)
    }, error = function(e) NULL)
    invisible(NULL)
  }
}

#' Record the outcome in the progress file and return the result unchanged.
finish <- function(progress, result) {
  if (isTRUE(result$success)) {
    progress("done", result$message %||% "Converted")
  } else {
    progress("failed", result$error %||% "Conversion failed")
  }
  result
}

# ---------------------------------------------------------------------------
# Load helpers
# ---------------------------------------------------------------------------

#' Load the R object from an RDS or RData file: list(obj, name) or
#' list(error). From an RData file: the first Seurat object, as before; else
#' the first SingleCellExperiment, then SummarizedExperiment.
load_r_object <- function(path) {

  ext <- tolower(tools::file_ext(path))

  if (ext == "rds") {
    cat("[convert] Loading RDS file ...\n")
    # In a list: inherits() on an S4 object whose package is not installed fails
    read <- tryCatch(list(obj = readRDS(path)), error = function(e) list(error = conditionMessage(e)))
    if (!is.null(read$error)) {
      return(list(error = paste0("Could not read the .rds file: ", read$error)))
    }
    return(list(obj = read$obj, name = NULL))
  }

  if (ext %in% c("rdata", "rda")) {
    cat("[convert] Loading RData file ...\n")
    tmp_env <- new.env(parent = emptyenv())
    loaded  <- tryCatch(load(path, envir = tmp_env), error = function(e) e)
    if (!is.character(loaded)) {
      return(list(error = paste0("Could not read the .RData file: ", conditionMessage(loaded))))
    }
    if (length(loaded) == 0) {
      return(list(error = "The .RData file holds no objects."))
    }

    kinds <- vapply(loaded, function(nm) object_kind(get(nm, envir = tmp_env)), character(1))
    for (want in c("Seurat", "SingleCellExperiment", "SummarizedExperiment")) {
      hit <- loaded[kinds == want]
      if (length(hit) > 0) {
        cat(sprintf("[convert] Found %s object '%s' in RData file.\n", want, hit[1]))
        return(list(obj = get(hit[1], envir = tmp_env), name = hit[1]))
      }
    }

    found <- vapply(loaded, function(nm) class(get(nm, envir = tmp_env))[1], character(1))
    return(list(error = paste0(
      "The .RData file holds no object scView can convert (",
      paste(sprintf("%s: %s", loaded, found), collapse = ", "),
      "). It converts Seurat, SingleCellExperiment and SummarizedExperiment objects."
    )))
  }

  list(error = sprintf("Unsupported file extension: .%s", ext))
}

# Slots (attributes) every SummarizedExperiment / SingleCellExperiment has.
SE_SLOTS  <- c("colData", "assays", "NAMES", "elementMetadata", "metadata")
SCE_SLOTS <- c(SE_SLOTS, "int_elementMetadata", "int_colData", "int_metadata")

#' "Seurat", "SingleCellExperiment", "SummarizedExperiment" or "other".
#' Bioconductor is loaded only for S4 objects that are not Seurat, so a Seurat
#' conversion never touches it.
object_kind <- function(obj) {
  # inherits() looks the class up, and fails when its package is not installed
  if (isTRUE(tryCatch(inherits(obj, "Seurat"), error = function(e) FALSE))) return("Seurat")
  if (!isS4(obj)) return("other")

  # The class's own package, when installed here, tells methods::is() what it
  # extends (a subclass of SingleCellExperiment from another package).
  pkg <- attr(class(obj), "package")
  if (!is.null(pkg)) suppressMessages(suppressWarnings(requireNamespace(pkg, quietly = TRUE)))
  if (suppressMessages(requireNamespace("SingleCellExperiment", quietly = TRUE))) {
    is_a <- function(cls) isTRUE(tryCatch(methods::is(obj, cls), error = function(e) FALSE))
    if (is_a("SingleCellExperiment")) return("SingleCellExperiment")
    if (is_a("SummarizedExperiment")) return("SummarizedExperiment")
  }

  # A class from a package that is not installed here (SpatialExperiment,
  # ...): its slots say what it is built on.
  slots <- names(attributes(obj))
  if (all(SCE_SLOTS %in% slots)) return("SingleCellExperiment")
  if (all(SE_SLOTS %in% slots)) return("SummarizedExperiment")
  "other"
}

#' The error for an object scView cannot convert, naming its class.
unsupported_message <- function(obj) {
  cls  <- class(obj)
  pkg  <- attr(cls, "package")
  what <- paste(cls, collapse = ", ")
  if (!is.null(pkg)) what <- sprintf("%s (from the %s package)", what, pkg)

  hint <- paste("scView converts Seurat, SingleCellExperiment and SummarizedExperiment",
                "objects, and classes built on them.")
  if (identical(cls[1], "seurat")) {
    hint <- paste("This is a Seurat v2 object: update it in R with",
                  "Seurat::UpdateSeuratObject() and save it again.")
  } else if (is.matrix(obj) || is.data.frame(obj) ||
             isTRUE(tryCatch(inherits(obj, "Matrix"), error = function(e) FALSE))) {
    hint <- paste("A bare matrix has no cell or gene metadata: wrap it in R with",
                  "SingleCellExperiment(assays = list(counts = m)) and save that,",
                  "or export it as a 10x matrix.")
  }
  sprintf("This file holds a %s object, which scView cannot convert. %s", what, hint)
}

# ---------------------------------------------------------------------------
# Version detection
# ---------------------------------------------------------------------------

#' Detect the Seurat version flavour of a loaded Seurat object.
#' Returns a human-readable string such as "v3", "v4", or "v5".
detect_seurat_version <- function(obj) {

  # Seurat v5 objects expose "layers" inside assays
  if (!is.null(tryCatch(obj[["RNA"]]@layers, error = function(e) NULL))) {
    return("v5")
  }

  # Seurat v5 alternative check: class name contains "Assay5"
  rna_assay <- tryCatch(obj[["RNA"]], error = function(e) NULL)
  if (!is.null(rna_assay) && any(grepl("Assay5", class(rna_assay)))) {
    return("v5")
  }

  # Seurat v3/v4 both use the Assay class.  Distinguish by package version
  # stored in the object (if available) or by slot structure.
  seurat_ver <- tryCatch(obj@version, error = function(e) NULL)

  if (!is.null(seurat_ver)) {
    major <- as.integer(substr(as.character(seurat_ver), 1, 1))
    if (!is.na(major)) {
      if (major >= 5) return("v5")
      if (major >= 4) return("v4")
      if (major >= 3) return("v3")
    }
  }

  # Fallback: check for the older Assays slot (Seurat v3 had @assays as a
  # named list of Assay objects)
  if (!is.null(tryCatch(slot(obj, "assays"), error = function(e) NULL))) {
    return("v3/v4")
  }

  return("unknown")
}
# ---------------------------------------------------------------------------
# Seurat: the parts that are written, then the matrix in blocks of cells
# ---------------------------------------------------------------------------
#
# Memory: the object as read is the floor (readRDS holds all of it). The rest stays
# near it: the counts layer is taken as it is stored (no dimnames copy, no transpose),
# everything else the h5ad keeps is small, and then the object goes, so its data /
# scale.data layers, graphs and other assays are freed before anything is written.
# X goes to the h5ad's csr_matrix group in blocks of cells: a genes x cells dgCMatrix
# is, array for array, the CSR form of the cells x genes matrix (as write_csr in
# convert_sce.R). Values and their type (float64) are the ones 0.81.0 wrote, which
# stored X as csc_matrix; CSR is what the backend's chunked reader needs.

#' Whether a Seurat object holds layers that DietSeurat(obj, layers = "counts") drops
#' (data, scale.data or others, in any assay). A v3 / v4 assay always has a data slot.
seurat_droppable_layers <- function(obj) {
  for (a in tryCatch(SeuratObject::Assays(obj), error = function(e) character(0))) {
    assay <- tryCatch(obj[[a]], error = function(e) NULL)
    if (inherits(assay, "Assay5")) {
      if (!all(grepl("^counts(\\.|$)", names(methods::slot(assay, "layers"))))) return(TRUE)
    } else if (inherits(assay, "Assay")) {
      return(TRUE)
    }
  }
  FALSE
}

#' Join the per-sample layers (counts.1, counts.2, data.1, ...) of one Seurat v5 assay.
#' An assay that cannot be joined comes back as it is.
join_assay_layers <- function(assay, name) {
  if (!inherits(assay, "Assay5")) return(assay)
  lyr <- tryCatch(SeuratObject::Layers(assay), error = function(e) character(0))
  if (!any(grepl("^(counts|data)\\.", lyr))) return(assay)
  tryCatch({
    joined <- SeuratObject::JoinLayers(assay)
    cat(sprintf("[convert] Joined %d split layers of assay %s\n", length(lyr), name))
    joined
  }, error = function(e) {
    cat(sprintf("[convert] Could not join the layers of %s: %s\n", name, conditionMessage(e)))
    assay
  })
}

#' The stored pieces of one layer of an assay: the layer itself ("counts"), else its
#' per-sample layers ("counts.1", ...). Each piece is list(m, rows, cells): m as
#' stored (genes x cells), rows its genes' positions in `genes` (NULL when they are
#' all of them, in order). NULL when the assay has no such layer, or it is empty.
layer_pieces <- function(assay, layer) {
  if (inherits(assay, "Assay5")) {
    stored <- methods::slot(assay, "layers")
    names_ <- if (layer %in% names(stored)) layer else grep(paste0("^", layer, "\\."), names(stored), value = TRUE)
    if (length(names_) == 0) return(NULL)
    feats <- lapply(names_, function(l) SeuratObject::Features(assay, layer = l))
    cells <- lapply(names_, function(l) SeuratObject::Cells(assay, layer = l))
    all_feats <- SeuratObject::Features(assay)
    genes <- all_feats[all_feats %in% unique(unlist(feats))]
    pieces <- lapply(seq_along(names_), function(k) {
      rows <- match(feats[[k]], genes)
      if (identical(rows, seq_along(genes))) rows <- NULL
      list(m = stored[[names_[k]]], rows = rows, cells = cells[[k]])
    })
  } else {
    m <- tryCatch(methods::slot(assay, layer), error = function(e) NULL)
    if (is.null(m)) return(NULL)
    genes <- rownames(m)
    pieces <- list(list(m = m, rows = NULL, cells = colnames(m)))
  }
  if (length(genes) == 0 || sum(vapply(pieces, function(p) ncol(p$m), numeric(1))) == 0) return(NULL)
  list(layer = layer, genes = genes, pieces = pieces)
}

#' X of a Seurat object: the main assay's counts (joined across samples), else its
#' data, as stored pieces plus the cell order to write them in (the object's, which
#' is the metadata's), or NULL.
seurat_x <- function(obj, main_assay) {
  assay <- tryCatch(obj[[main_assay]], error = function(e) NULL)
  if (is.null(assay)) return(NULL)
  x <- layer_pieces(assay, "counts")
  if (is.null(x)) x <- layer_pieces(assay, "data")
  if (is.null(x)) return(NULL)
  stored <- unlist(lapply(x$pieces, `[[`, "cells"))
  cells <- colnames(obj)
  x$cells <- if (!identical(stored, cells) && setequal(stored, cells)) cells else stored
  dense <- vapply(x$pieces, function(p) is.matrix(p$m), logical(1))
  x$dense <- all(dense)
  for (k in seq_along(x$pieces)) {
    p <- x$pieces[[k]]
    if (!x$dense && !inherits(p$m, "dgCMatrix")) p$m <- to_dgc(p$m)
    if (!is.null(p$rows) && is.unsorted(p$rows)) {
      o <- order(p$rows)
      p$m <- p$m[o, , drop = FALSE]
      p$rows <- p$rows[o]
    }
    x$pieces[k] <- list(p)
  }
  x
}

#' Where each output cell is stored: piece number and column, in output order.
x_locations <- function(x) {
  piece <- rep(seq_along(x$pieces), vapply(x$pieces, function(p) ncol(p$m), numeric(1)))
  col <- unlist(lapply(x$pieces, function(p) seq_len(ncol(p$m))))
  at <- match(x$cells, unlist(lapply(x$pieces, `[[`, "cells")))
  list(piece = piece[at], col = col[at])
}

#' Write X (cells x genes) into the open h5py file `f`.
write_seurat_x <- function(f, x, py) {
  loc <- x_locations(x)
  if (x$dense) write_seurat_dense(f, x, loc, py) else write_seurat_csr(f, x, loc, py)
}

write_seurat_csr <- function(f, x, loc, py) {
  n_cells <- length(x$cells)
  n_genes <- length(x$genes)
  # entries per output cell, straight from the column pointers
  per_cell <- numeric(n_cells)
  for (k in seq_along(x$pieces)) {
    sel <- which(loc$piece == k)
    per_cell[sel] <- diff(x$pieces[[k]]$m@p)[loc$col[sel]]
  }
  indptr <- c(0, cumsum(per_cell))
  nnz <- indptr[n_cells + 1]
  ptr_type <- if (nnz > .Machine$integer.max) "int64" else "int32"

  g <- f$create_group("X")
  reticulate::py_set_item(g$attrs, "encoding-type", "csr_matrix")
  reticulate::py_set_item(g$attrs, "encoding-version", "0.1.0")
  reticulate::py_set_item(g$attrs, "shape", py$np$array(c(n_cells, n_genes), dtype = "int64"))
  int <- py$builtins$int  # a Python int: nnz and offsets can pass R's integer range
  shape <- reticulate::tuple(int(nnz))
  if (nnz > 0) {
    data <- g$create_dataset("data", shape = shape, dtype = "float64", chunks = TRUE)
    indices <- g$create_dataset("indices", shape = shape, dtype = "int32", chunks = TRUE)
  } else {
    data <- g$create_dataset("data", shape = shape, dtype = "float64")
    indices <- g$create_dataset("indices", shape = shape, dtype = "int32")
  }

  step <- block_values()
  a <- 1L
  while (a <= n_cells) {
    b <- max(a, findInterval(indptr[a] + step, indptr[-1]))
    b <- min(b, n_cells)
    blk <- csr_block(x, loc, a, b)
    if (length(blk$x) > 0) {
      at <- py$builtins$slice(int(indptr[a]), int(indptr[b + 1]))
      reticulate::py_set_item(data, at, py$np$asarray(np_array(blk$x), dtype = "float64"))
      reticulate::py_set_item(indices, at, py$np$asarray(np_array(blk$i), dtype = "int32"))
    }
    rm(blk)
    release_chunk()
    a <- b + 1L
  }
  g$create_dataset("indptr", data = py$np$asarray(np_array(indptr), dtype = ptr_type))
  invisible(NULL)
}

#' Output cells a..b as CSR rows: list(x, i), entries in output order, genes as
#' 0-based positions in x$genes. Gathered from the stored arrays (no `[` copy of
#' the piece, explicit zeros kept as 0.81.0's transpose kept them).
csr_block <- function(x, loc, a, b) {
  pk <- loc$piece[a:b]
  cl <- loc$col[a:b]
  ks <- unique(pk)
  out_x <- vector("list", length(ks))
  out_i <- vector("list", length(ks))
  lens <- numeric(length(pk))
  for (n in seq_along(ks)) {
    k <- ks[n]
    p <- x$pieces[[k]]
    sel <- which(pk == k)
    from <- p$m@p[cl[sel]]
    len <- p$m@p[cl[sel] + 1L] - from
    lens[sel] <- len
    idx <- sequence(len, from = from + 1L)
    out_x[[n]] <- p$m@x[idx]
    i <- p$m@i[idx]
    out_i[[n]] <- if (is.null(p$rows)) i else p$rows[i + 1L] - 1L
  }
  if (length(ks) == 1L) return(list(x = out_x[[1]], i = out_i[[1]]))
  # several pieces in this block: their segments back into output order
  vx <- unlist(out_x); vi <- unlist(out_i)
  start <- numeric(length(pk))
  off <- 0
  for (n in seq_along(ks)) {
    sel <- which(pk == ks[n])
    start[sel] <- off + c(0, cumsum(lens[sel]))[seq_along(sel)]
    off <- off + sum(lens[sel])
  }
  idx <- sequence(lens, from = start + 1)
  list(x = vx[idx], i = vi[idx])
}

#' Dense stored counts (a plain matrix): written dense as 0.81.0 did, integer
#' storage as int32, else float64, in blocks of cells.
write_seurat_dense <- function(f, x, loc, py) {
  n_cells <- length(x$cells)
  n_genes <- length(x$genes)
  int <- all(vapply(x$pieces, function(p) is.integer(p$m), logical(1)))
  dtype <- if (int) "int32" else "float64"
  ds <- f$create_dataset("X", shape = reticulate::tuple(as.integer(n_cells), as.integer(n_genes)), dtype = dtype)
  reticulate::py_set_item(ds$attrs, "encoding-type", "array")
  reticulate::py_set_item(ds$attrs, "encoding-version", "0.2.0")
  step <- max(1L, as.integer(floor(block_values() / max(1, n_genes))))
  for (a in seq(1L, n_cells, by = step)) {
    b <- min(a + step - 1L, n_cells)
    blk <- matrix(if (int) 0L else 0, n_genes, b - a + 1L)
    pk <- loc$piece[a:b]; cl <- loc$col[a:b]
    for (k in unique(pk)) {
      sel <- which(pk == k)
      p <- x$pieces[[k]]
      rows <- p$rows %||% seq_len(n_genes)
      blk[rows, sel] <- p$m[, cl[sel], drop = FALSE]
    }
    arr <- py$np$ascontiguousarray(reticulate::r_to_py(unname(blk), convert = FALSE)$T, dtype = dtype)
    reticulate::py_set_item(ds, py$builtins$slice(as.integer(a - 1L), as.integer(b)), arr)
    rm(blk, arr)
    release_chunk()
  }
  invisible(NULL)
}

#' A Seurat object to h5ad. `obj` is the object, or an environment holding it as
#' `obj` (the entry point passes one, so the object can be freed here once its
#' parts are read: nothing else refers to it then).
manual_convert <- function(obj, output_path, version_info, progress = NULL) {

  if (is.environment(obj)) {
    box <- obj
    obj <- box$obj
    rm("obj", envir = box)
  }
  say <- function(stage, message) if (is.null(progress)) cat(sprintf("[convert] %s\n", message)) else progress(stage, message)

  suppressMessages(library(reticulate))

  ad <- tryCatch(import("anndata"), error = function(e) {
    stop("Python anndata package is not available: ", conditionMessage(e))
  })
  np <- tryCatch(import("numpy"), error = function(e) NULL)
  py <- py_modules()

  # ---- expression matrix ---------------------------------------------------
  cat("[convert] Extracting expression matrix ...\n")

  # A chromatin assay is not read: said in the report, and never taken for the genes.
  chromatin <- seurat_chromatin_assays(obj)
  if (seurat_main_assay(obj) %in% chromatin) {
    stop("This Seurat object holds ATAC peaks (", seurat_main_assay(obj), ") and no RNA assay: ATAC-only data ",
         "is not supported yet. Upload the object with its RNA assay.")
  }
  chromatin_note <- if (length(chromatin) > 0) {
    sprintf(paste0("the chromatin assay %s: ATAC inside a Seurat object is not read yet; upload the 10x ",
                   "filtered_feature_bc_matrix.h5 or an .h5mu for its peaks"), paste(chromatin, collapse = ", "))
  } else character(0)

  # RNA as before; a Visium object's genes are in its Spatial assay. Seurat v5 keeps one
  # layer per sample after split() / merge() (counts.1, counts.2, ...): they are read
  # piece by piece, in the object's cell order, as JoinLayers() would join them.
  main_assay <- seurat_main_assay(obj)
  x <- seurat_x(obj, main_assay)
  if (is.null(x)) {
    return(list(success = FALSE, error = "Could not extract expression matrix from Seurat object"))
  }
  cells <- x$cells
  cat(sprintf("[convert]   X: the %s layer of %s (%d cells x %d genes)\n",
              x$layer, main_assay, length(cells), length(x$genes)))

  # ---- metadata (obs) ------------------------------------------------------
  cat("[convert] Extracting cell metadata ...\n")
  meta <- seurat_obs(tryCatch(obj@meta.data, error = function(e) NULL), cells)
  obs_df <- meta$df

  # ---- var (gene names) -----------------------------------------------------
  var_df <- data.frame(row.names = x$genes)

  # ---- dimensional reductions (obsm) ----------------------------------------
  cat("[convert] Extracting dimensional reductions ...\n")
  obsm <- list()
  reductions <- tryCatch(names(obj@reductions), error = function(e) character(0))

  for (red_name in reductions) {
    emb <- tryCatch(
      {
        SeuratObject::Embeddings(obj, reduction = red_name)
      },
      error = function(e) NULL
    )
    if (!is.null(emb)) {
      key <- paste0("X_", red_name)
      obsm[[key]] <- r_to_py(emb)
      cat(sprintf("[convert]   added reduction: %s (%d x %d)\n", key, nrow(emb), ncol(emb)))
    }
  }

  # ---- CITE-seq antibody counts (an ADT / Protein assay) ---------------------
  # scView reads them from obsm["protein_counts"] (cells x proteins) and adds the
  # CLR itself; a panel is small, so it goes over as a dense table.
  protein_df <- NULL
  assays <- tryCatch(SeuratObject::Assays(obj), error = function(e) character(0))
  adt_name <- setdiff(intersect(ADT_ASSAYS, assays), main_assay)[1]
  if (!is.na(adt_name)) {
    adt <- tryCatch(
      SeuratObject::GetAssayData(join_assay_layers(obj[[adt_name]], adt_name), layer = "counts"),
      error = function(e) NULL
    )
    if (!is.null(adt) && !identical(colnames(adt), cells) && setequal(colnames(adt), cells)) {
      adt <- adt[, cells, drop = FALSE]
    }
    if (!is.null(adt) && nrow(adt) > 0 && nrow(adt) <= 1000 && identical(colnames(adt), cells)) {
      protein_df <- as.data.frame(as.matrix(Matrix::t(adt)))
      cat(sprintf("[convert]   added %s assay: %d proteins\n", adt_name, ncol(protein_df)))
    }
    rm(adt)
  }

  # ---- spatial: Visium images (VisiumV1, VisiumV2) ---------------------------
  tissue <- tryCatch(seurat_images(obj, cells, np),
                     error = function(e) list(skipped = paste0("tissue images: ", conditionMessage(e))))
  if (!is.null(tissue$obs)) {
    for (col in names(tissue$obs)) obs_df[[col]] <- tissue$obs[[col]]
  }

  # Everything written is read: the object goes, with the layers and graphs it holds.
  rm(obj)
  invisible(gc(verbose = FALSE))

  # ---- write: the tables, then X in blocks ---------------------------------
  say("writing", "Writing cell and gene tables")
  partial <- paste0(output_path, ".partial")
  if (file.exists(partial)) file.remove(partial)
  ok <- FALSE
  f <- NULL
  on.exit(if (!ok) {
    if (!is.null(f)) try(f$close(), silent = TRUE)
    if (file.exists(partial)) file.remove(partial)
  }, add = TRUE)

  adata <- ad$AnnData(
    obs = r_to_py(obs_df),
    var = r_to_py(var_df)
  )

  # Attach reductions
  if (length(obsm) > 0) {
    for (key in names(obsm)) {
      adata$obsm[key] <- obsm[[key]]
    }
  }
  if (!is.null(protein_df)) {
    adata$obsm["protein_counts"] <- r_to_py(protein_df)
  }
  if (!is.null(tissue$xy)) {
    adata$obsm["spatial"] <- np$asarray(tissue$xy, dtype = "float64")
    # py_set_item on the Python uns (adata$uns comes back as an R list; `[<-` takes a
    # one-element R list for its element, the slice's dict)
    reticulate::py_set_item(reticulate::py_get_attr(adata, "uns"), "spatial",
                            reticulate::r_to_py(tissue$uns))
  }
  adata$write_h5ad(partial)
  rm(adata, obsm, protein_df, obs_df, var_df)
  invisible(gc(verbose = FALSE))

  say("writing", sprintf("Writing X (the %s layer, %s cells x %s genes)", x$layer,
                         fmt_n(length(cells)), fmt_n(length(x$genes))))
  cat(sprintf("[convert] Writing h5ad to %s ...\n", output_path))
  f <- py$h5py$File(partial, "r+")
  write_seurat_x(f, x, py)
  f$close()
  f <- NULL
  rm(x)
  invisible(gc(verbose = FALSE))

  if (file.exists(output_path)) file.remove(output_path)
  if (!file.rename(partial, output_path)) stop("could not move the h5ad into place")
  ok <- TRUE

  list(
    success = TRUE,
    message = paste0("Converted via manual extraction (", version_info, ")"),
    kept = c(meta$kept, tissue$kept), skipped = c(meta$skipped, tissue$skipped, chromatin_note)
  )
}

# ---------------------------------------------------------------------------
# Cell metadata (Seurat meta.data)
# ---------------------------------------------------------------------------
#
# Shared with convert_sce.R (colData, rowData), which is always sourced after this file.
# meta.data went to pandas as it was, so one odd column failed the whole conversion
# (a list column: "Mixing dicts with non-Series"; dates, nested tables, matrix columns,
# a column named _index) or came out wrong (integer NA as -2147483648, logical NA as
# FALSE). Every column now becomes a plain one pandas and anndata write as they are:
# number, integer, TRUE / FALSE, factor or text. What cannot be one is left out and
# said in the report.

MAX_MATRIX_COLUMN <- 50  # a matrix column wider than this is not split into columns

#' meta.data as list(df, kept, skipped): every column a plain one, in order, row names
#' the cells. Nested tables and matrix columns come as parent.child columns. `kept` says
#' what was changed on the way (a list column joined as text), `skipped` what was left
#' out and why.
seurat_obs <- function(md, cells) {
  if (is.null(md) || !is.data.frame(md) || nrow(md) != length(cells)) {
    skipped <- if (is.null(md)) character(0) else "cell metadata (meta.data): its rows are not the object's cells"
    return(list(df = data.frame(row.names = cells), kept = character(0), skipped = skipped))
  }
  # a "/" in a name stays, as it always has for a Seurat object
  plain <- plain_columns(md, nrow(md), "cell metadata", keep_slash = TRUE)
  df <- data.frame(row.names = rownames(md))
  for (k in seq_along(plain$cols)) df[[names(plain$cols)[k]]] <- plain$cols[[k]]
  list(df = df, kept = plain$kept, skipped = plain$skipped)
}

#' The columns of a table with one row per cell or gene (Seurat meta.data; colData,
#' rowData and metadata tables of a SingleCellExperiment, convert_sce.R) as plain ones:
#' list(cols = named list of plain vectors, in order, with names anndata takes,
#' kept, skipped, n = columns of the table kept, total = columns it has). `what` names
#' the table in the report ("cell metadata", "colData"), `unit` what a row is.
plain_columns <- function(df, n, what, unit = "cell", keep_slash = FALSE) {
  cols <- list()
  kept <- character(0)
  skipped <- character(0)
  cn <- names(df)
  n_kept <- 0L
  for (i in seq_along(cn)) {
    out <- obs_column(df[[i]], cn[i], n, unit)
    if (is.character(out$why)) {
      skipped <- c(skipped, sprintf("%s column %s: %s", what, cn[i], out$why))
      next
    }
    n_kept <- n_kept + 1L
    cols <- c(cols, out$cols)
    if (length(out$note)) kept <- c(kept, sprintf("%s column %s: %s", what, cn[i], paste(out$note, collapse = "; ")))
    if (length(out$left_out)) {
      skipped <- c(skipped, sprintf("%s column %s: %s", what, cn[i], paste(out$left_out, collapse = "; ")))
    }
  }
  names(cols) <- column_keys(names(cols), keep_slash)
  list(cols = cols, kept = kept, skipped = skipped, n = n_kept, total = length(cn))
}

#' Column names anndata can store: it reserves _index and nests a name at "/" (made
#' "_" unless `keep_slash`); an empty or repeated name is not a column name. A "/"
#' that stays is stored escaped (`h5_escape`), as scView stores it.
column_keys <- function(keys, keep_slash = FALSE) {
  if (length(keys) == 0) return(character(0))
  keys[is.na(keys) | !nzchar(keys)] <- paste0("column_", which(is.na(keys) | !nzchar(keys)))
  keys[keys == "_index"] <- "_index_"
  if (!keep_slash) keys <- gsub("/", "_", keys, fixed = TRUE)
  h5_escape(make.unique(keys, sep = "_"))
}

#' The key a name is stored under in an h5ad, as scView's backend writes it
#' (backend/src/scview/core/h5_names.py `escape`; the two are tested against one list,
#' tests/fixtures/h5_names.json): HDF5 reads "/" as a path, so "CD4/CD8" is stored
#' "CD4%2fCD8", a "%" that would read as an escape "%25", "." "%2e". Before this a
#' meta.data column "a/b" beside a column "a" failed the conversion.
h5_escape <- function(names) {
  out <- gsub("%(?=25|2f|2e|00)", "%25", names, perl = TRUE)
  out <- gsub("/", "%2f", out, fixed = TRUE)
  out[!is.na(names) & names == "."] <- "%2e"
  out[!is.na(names) & names == ""] <- "%00"
  out
}

#' The name a stored key stands for (`h5_escape` undone; h5_names.py `unescape`): what
#' export_rds.R gives the columns of a staging file, so a name reaches R as typed.
h5_unescape <- function(keys) {
  out <- gsub("%2f", "/", keys, fixed = TRUE)
  out <- gsub("%25(?=25|2f|2e|00)", "%", out, perl = TRUE)
  out[!is.na(keys) & keys == "%2e"] <- "."
  out[!is.na(keys) & keys == "%00"] <- ""
  out
}

#' One column as list(cols = named list of plain vectors, note = what was changed,
#' left_out = parts of a nested table left out), or list(why = the reason the whole
#' column is left out). `n` rows, each one `unit` (a cell, a gene).
obs_column <- function(v, nm, n, unit = "cell") {
  one <- function(x, note = NULL) list(cols = stats::setNames(list(x), nm), note = note)
  units <- paste0(unit, "s")

  # Bioconductor's containers (colData / rowData): a run-length column as its values,
  # a list of vectors (CharacterList, IntegerList, SimpleList) as a list column
  if (methods::is(v, "Rle")) v <- S4Vectors::decode(v)
  if (methods::is(v, "AtomicList") || identical(class(v)[1], "SimpleList")) v <- as.list(v)

  if (is.data.frame(v) || methods::is(v, "DataFrame")) {  # a nested table: its columns as nm.child
    if (nrow(v) != n) return(list(why = sprintf("a nested table whose rows are not the %s", units)))
    out <- list(); notes <- character(0); left_out <- character(0)
    for (j in seq_along(names(v))) {
      sub_nm <- paste0(nm, ".", names(v)[j])
      sub <- obs_column(v[[j]], sub_nm, n, unit)
      if (is.character(sub$why)) {
        left_out <- c(left_out, sprintf("%s left out (%s)", sub_nm, sub$why))
        next
      }
      out <- c(out, sub$cols)
      if (length(sub$note)) notes <- c(notes, sprintf("%s %s", sub_nm, sub$note))
      left_out <- c(left_out, sub$left_out)
    }
    if (length(out) == 0) return(list(why = "a nested table with no column scView can read"))
    return(list(cols = out, note = c(sprintf("a nested table, kept as %s", paste(names(out), collapse = ", ")), notes),
                left_out = left_out))
  }
  if (!is.null(dim(v)) && length(dim(v)) == 2) {  # a matrix column: one column per matrix column
    if (nrow(v) != n) return(list(why = sprintf("a matrix whose rows are not the %s", units)))
    if (ncol(v) > MAX_MATRIX_COLUMN) {
      return(list(why = sprintf("a matrix of %s columns in one column (more than %d)", fmt_n(ncol(v)), MAX_MATRIX_COLUMN)))
    }
    m <- unclass(as.matrix(v))
    sub_names <- colnames(m)
    if (is.null(sub_names) || anyNA(sub_names) || !all(nzchar(sub_names))) sub_names <- as.character(seq_len(ncol(m)))
    df <- stats::setNames(lapply(seq_len(ncol(m)), function(j) m[, j]), sub_names)
    res <- obs_column(structure(df, class = "data.frame", row.names = seq_len(n)), nm, n, unit)
    if (!is.null(res$note)) res$note[1] <- sprintf("a matrix in one column, kept as %s", paste(names(res$cols), collapse = ", "))
    return(res)
  }
  if (!is.null(dim(v))) return(list(why = sprintf("an array of %d dimensions in one column", length(dim(v)))))
  if (length(v) != n) return(list(why = sprintf("its length does not match the %s", units)))
  # before the list check: a POSIXlt date is a list
  if (inherits(v, c("Date", "POSIXt"))) return(one(as.character(v)))
  if (inherits(v, "difftime")) return(one(as.numeric(v)))
  if (isS4(v)) return(list(why = sprintf("a %s, not a plain column", class(v)[1])))
  if (is.list(v)) return(obs_list_column(v, nm, n, unit))

  if (is.factor(v)) {
    lv <- levels(v)
    # an NA level (factor(x, exclude = NULL)): its cells are missing values
    if (anyNA(lv)) v <- factor(as.character(v), levels = lv[!is.na(lv)], ordered = is.ordered(v))
    return(one(v))
  }
  if (is.logical(v)) {
    if (!anyNA(v)) return(one(as.vector(v)))
    # pandas has no missing TRUE / FALSE: a category
    return(one(factor(ifelse(v, "TRUE", "FALSE"), levels = c("FALSE", "TRUE"))))
  }
  if (is.integer(v)) return(one(if (anyNA(v)) as.numeric(v) else as.vector(v)))
  if (is.double(v)) return(one(as.vector(v)))
  if (is.character(v)) return(one(enc2utf8(as.vector(v))))
  if (is.complex(v)) return(one(as.character(v)))
  if (is.raw(v)) return(one(as.integer(v)))
  list(why = sprintf("of type %s", typeof(v)))
}

#' A list column. One value or none per cell: those values as a plain column, empty
#' cells missing. Several values per cell: joined with "; ". Anything else, or no value
#' in any cell, is left out.
obs_list_column <- function(v, nm, n, unit = "cell") {
  lens <- lengths(v)
  if (all(lens == 0)) return(list(why = sprintf("a list column with no value in any %s", unit)))
  plain <- vapply(v, function(e) is.null(e) || (is.atomic(e) && is.null(dim(e)) && !isS4(e)), logical(1))
  if (!all(plain)) {
    return(list(why = sprintf("a list of %s values, not one value per %s", class(v[!plain][[1]])[1], unit)))
  }
  text <- function(e) as.character(e)  # a factor's labels, not its codes
  if (all(lens <= 1)) {
    has <- lens == 1
    vals <- v[has]
    if (all(vapply(vals, function(e) is.numeric(e) && !is.factor(e), logical(1)))) {
      out <- rep(NA_real_, n)
      out[has] <- as.numeric(unlist(vals, use.names = FALSE))
    } else if (all(vapply(vals, is.logical, logical(1)))) {
      out <- rep(NA, n)
      out[has] <- unlist(vals, use.names = FALSE)
    } else {
      out <- rep(NA_character_, n)
      out[has] <- vapply(vals, text, character(1))
    }
    res <- obs_column(out, nm, n, unit)
    res$note <- sprintf("a list column with one value per %s, kept as a plain column", unit)
    return(res)
  }
  joined <- vapply(v, function(e) if (length(e) == 0) NA_character_ else paste(text(e), collapse = "; "),
                   character(1))
  res <- obs_column(unname(joined), nm, n, unit)
  res$note <- sprintf("a list column with up to %d values per %s, kept as text joined with \"; \"", max(lens), unit)
  res
}

# ---------------------------------------------------------------------------
# Assay choice and Visium images (Seurat)
# ---------------------------------------------------------------------------

# A CITE-seq antibody assay's names, in order of preference.
ADT_ASSAYS <- c("ADT", "Protein", "protein", "adt", "CITE", "Antibody")

# Signac's usual names for a chromatin assay; any assay of its class counts too.
CHROMATIN_ASSAYS <- c("ATAC", "atac", "peaks", "Peaks", "chromatin")

#' The chromatin assays of a Seurat object (none on a plain RNA object).
seurat_chromatin_assays <- function(obj) {
  assays <- tryCatch(SeuratObject::Assays(obj), error = function(e) character(0))
  Filter(function(a) {
    a %in% CHROMATIN_ASSAYS || inherits(tryCatch(obj[[a]], error = function(e) NULL), "ChromatinAssay")
  }, assays)
}

#' The assay whose genes go in X: RNA (as before), else Spatial (a Visium object),
#' else the default assay, else the first that is not antibodies.
seurat_main_assay <- function(obj) {
  assays <- tryCatch(SeuratObject::Assays(obj), error = function(e) character(0))
  for (a in c("RNA", "Spatial")) if (a %in% assays) return(a)
  active <- tryCatch(SeuratObject::DefaultAssay(obj), error = function(e) NA_character_)
  if (!is.na(active) && active %in% assays && !(active %in% ADT_ASSAYS)) return(active)
  rest <- setdiff(assays, ADT_ASSAYS)
  if (length(rest) > 0) rest[1] else "RNA"
}

#' The tissue positions and images of a Seurat object's slices, in scView's layout
#' (scanpy's): list(xy, obs, uns, kept, skipped), or NULL without images.
#'   VisiumV1 (Seurat 3.2 - 5.0): @coordinates holds row / col (the array grid) and
#'     imagerow / imagecol in FULL-resolution pixels (Space Ranger's
#'     pxl_row_in_fullres / pxl_col_in_fullres; Seurat multiplies them by the lowres
#'     scale factor only to plot).
#'   VisiumV2 (Seurat >= 5.1): @boundaries$centroids holds x = imagerow, y = imagecol,
#'     full-res too (checked against tissue_positions_list.csv with Seurat 5.1.0); no
#'     array grid (the backend recovers it from the centres).
#'   Another FOV (Xenium, CosMx via Seurat): its centroids as they are (x, y), no image.
#' xy is (imagecol, imagerow): x across, y down. Seurat keeps one image, the low-res
#' one it places spots with (scale.factors$lowres), stored as uns images "lowres".
#' Slots are read with attr(): the Seurat package (where VisiumV1 / V2 are defined) is
#' not installed here, only SeuratObject.
seurat_images <- function(obj, cells, np) {
  images <- tryCatch(obj@images, error = function(e) list())
  if (length(images) == 0) return(NULL)
  n <- length(cells)
  xy <- matrix(NA_real_, n, 2)
  lib <- rep(NA_character_, n)
  arow <- rep(NA_integer_, n)
  acol <- rep(NA_integer_, n)
  all_grid <- TRUE
  uns <- list(); kept <- character(0); skipped <- character(0)

  for (name in names(images)) {
    img <- images[[name]]
    cls <- class(img)[1]
    coords <- attr(img, "coordinates")
    g_row <- NULL; g_col <- NULL
    if (is.data.frame(coords) && all(c("imagerow", "imagecol") %in% names(coords))) {
      spots <- rownames(coords)
      x <- as.numeric(coords$imagecol); y <- as.numeric(coords$imagerow)
      if (all(c("row", "col") %in% names(coords))) {
        g_row <- as.integer(coords$row); g_col <- as.integer(coords$col)
      }
    } else {
      cent <- tryCatch(attr(img, "boundaries")$centroids, error = function(e) NULL)
      m <- if (!is.null(cent)) attr(cent, "coords") else NULL
      if (is.null(m) || ncol(m) < 2) {
        skipped <- c(skipped, sprintf("image %s (%s): no spot positions scView reads", name, cls))
        next
      }
      spots <- attr(cent, "cells")
      if (cls == "VisiumV2") {
        x <- as.numeric(m[, 2]); y <- as.numeric(m[, 1])
      } else {
        x <- as.numeric(m[, 1]); y <- as.numeric(m[, 2])
      }
    }
    idx <- match(spots, cells)
    ok <- !is.na(idx)
    xy[idx[ok], 1] <- x[ok]
    xy[idx[ok], 2] <- y[ok]
    lib[idx[ok]] <- name
    if (is.null(g_row)) {
      all_grid <- FALSE
    } else {
      arow[idx[ok]] <- g_row[ok]
      acol[idx[ok]] <- g_col[ok]
    }

    visium <- cls %in% c("VisiumV1", "VisiumV2")
    sf <- if (visium) unclass(attr(img, "scale.factors")) else list()
    num <- function(v) if (length(v) == 1 && is.finite(as.numeric(v))) as.numeric(v) else NULL
    scalefactors <- Filter(Negate(is.null), list(
      tissue_lowres_scalef = num(sf$lowres), tissue_hires_scalef = num(sf$hires),
      spot_diameter_fullres = num(sf$spot), fiducial_diameter_fullres = num(sf$fiducial)))
    image <- if (visium) attr(img, "image") else NULL
    imgs <- reticulate::dict()
    if (!is.null(image) && length(dim(image)) == 3 && dim(image)[3] >= 3 &&
        !is.null(scalefactors$tissue_lowres_scalef)) {
      v <- if (max(image, na.rm = TRUE) <= 1) image * 255 else image
      v <- array(as.integer(round(pmin(pmax(v, 0), 255))), dim = dim(image))
      # a NumPy uint8 array that stays one (not converted back to an R integer array)
      npc <- reticulate::import("numpy", convert = FALSE)
      imgs <- reticulate::dict(lowres = npc$asarray(reticulate::r_to_py(v[, , 1:3, drop = FALSE]),
                                                    dtype = "uint8"))
    }
    uns[[name]] <- list(
      images = imgs,
      scalefactors = if (length(scalefactors)) scalefactors else reticulate::dict(),
      metadata = list(source = "seurat", seurat_image_class = cls)
    )
    kept <- c(kept, sprintf("tissue positions%s of %s (%s, %d spots)",
                            if (length(imgs)) " and image" else "", name, cls, sum(ok)))
  }

  if (length(uns) == 0) return(list(skipped = skipped))
  if (any(is.na(lib))) {
    return(list(skipped = c(skipped, sprintf(
      "tissue positions: %d of %d cells are in none of the images, so they were left out",
      sum(is.na(lib)), n))))
  }
  obs <- data.frame(library_id = factor(lib, levels = names(uns)), row.names = cells)
  if (all_grid) {
    obs$array_row <- arow
    obs$array_col <- acol
  }
  list(xy = xy, obs = obs, uns = uns, kept = kept, skipped = skipped)
}
