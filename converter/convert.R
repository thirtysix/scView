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
    progress("converting", "Converting a Seurat object", source_class = "Seurat")
    version_info <- detect_seurat_version(obj)
    cat(sprintf("[convert] Detected Seurat version: %s\n", version_info))

    result <- tryCatch(
      manual_convert(obj, output_path, version_info),
      error = function(e) {
        list(success = FALSE, error = paste0("Conversion failed: ", conditionMessage(e)))
      }
    )
    result$report <- list(
      source_class = "Seurat", detail = paste("Seurat", version_info),
      object_name = loaded$name %||% "", kept = list(), skipped = list()
    )
    return(finish(progress, result))
  }

  # ------------------------------------------------------------------
  # 3. SingleCellExperiment / SummarizedExperiment
  # ------------------------------------------------------------------
  if (kind %in% c("SingleCellExperiment", "SummarizedExperiment")) {
    result <- tryCatch(
      convert_se_to_h5ad(obj, output_path, kind, progress, object_name = loaded$name),
      error = function(e) {
        list(success = FALSE, error = paste0("Conversion failed: ", conditionMessage(e)))
      }
    )
    return(finish(progress, result))
  }

  finish(progress, list(success = FALSE, error = unsupported_message(obj)))
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
# Manual conversion via reticulate + Python anndata
# ---------------------------------------------------------------------------

#' Extract Seurat slots and write an h5ad file using the Python
#' anndata package through reticulate.
manual_convert <- function(obj, output_path, version_info) {

  suppressMessages(library(reticulate))

  ad <- tryCatch(import("anndata"), error = function(e) {
    stop("Python anndata package is not available: ", conditionMessage(e))
  })
  np  <- tryCatch(import("numpy"),   error = function(e) NULL)
  sp  <- tryCatch(import("scipy.sparse", convert = FALSE), error = function(e) NULL)

  # ---- expression matrix ---------------------------------------------------
  cat("[convert] Extracting expression matrix ...\n")

  # SeuratObject >= 5.0.0 uses `layer` instead of the defunct `slot` argument
  counts <- tryCatch(
    {
      SeuratObject::GetAssayData(obj, assay = "RNA", layer = "counts")
    },
    error = function(e) {
      cat(sprintf("[convert] Could not get counts: %s\n", conditionMessage(e)))
      NULL
    }
  )

  if (is.null(counts)) {
    # Try data layer as fallback
    counts <- tryCatch(
      SeuratObject::GetAssayData(obj, assay = "RNA", layer = "data"),
      error = function(e) NULL
    )
  }

  if (is.null(counts)) {
    return(list(success = FALSE, error = "Could not extract expression matrix from Seurat object"))
  }

  # anndata expects cells x genes (obs x var)
  counts_t <- Matrix::t(counts)

  # Convert sparse matrix to Python scipy sparse via its components
  # dgCMatrix is Compressed Sparse Column (CSC) format, so use csc_matrix
  cat("[convert] Converting matrix to Python format ...\n")
  if (inherits(counts_t, "dgCMatrix")) {
    X <- sp$csc_matrix(
      tuple(
        r_to_py(as.vector(counts_t@x)),
        r_to_py(as.integer(counts_t@i)),
        r_to_py(as.integer(counts_t@p))
      ),
      shape = tuple(
        as.integer(nrow(counts_t)),
        as.integer(ncol(counts_t))
      )
    )
  } else {
    # Dense fallback
    X <- r_to_py(as.matrix(counts_t))
  }

  # ---- metadata (obs) ------------------------------------------------------
  cat("[convert] Extracting cell metadata ...\n")
  obs_df <- tryCatch(obj@meta.data, error = function(e) data.frame(row.names = colnames(counts)))

  # ---- var (gene names) -----------------------------------------------------
  var_df <- data.frame(row.names = rownames(counts))

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
  adt_name <- intersect(c("ADT", "Protein", "protein", "adt", "CITE", "Antibody"), assays)[1]
  if (!is.na(adt_name)) {
    adt <- tryCatch(
      SeuratObject::GetAssayData(obj, assay = adt_name, layer = "counts"),
      error = function(e) NULL
    )
    if (!is.null(adt) && nrow(adt) > 0 && nrow(adt) <= 1000 &&
        identical(colnames(adt), colnames(counts))) {
      protein_df <- as.data.frame(as.matrix(Matrix::t(adt)))
      cat(sprintf("[convert]   added %s assay: %d proteins\n", adt_name, ncol(protein_df)))
    }
  }

  # ---- build AnnData object ------------------------------------------------
  cat("[convert] Building AnnData object ...\n")

  adata <- ad$AnnData(
    X   = X,
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

  # ---- write ---------------------------------------------------------------
  cat(sprintf("[convert] Writing h5ad to %s ...\n", output_path))
  adata$write_h5ad(output_path)

  if (file.exists(output_path)) {
    return(list(
      success = TRUE,
      message = paste0("Converted via manual extraction (", version_info, ")")
    ))
  } else {
    return(list(success = FALSE, error = "h5ad file was not created"))
  }
}
