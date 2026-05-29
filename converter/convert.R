##
## convert.R
## Core conversion logic: Seurat (RDS / RData) -> AnnData h5ad
## Uses SeuratObject + reticulate + Python anndata (no sceasy dependency).
##

library(Matrix)

# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

#' Convert a Seurat object stored as RDS (or RData) to an h5ad file.
#'
#' @param input_path  Path to the .rds or .RData file.
#' @param output_path Path where the .h5ad file should be written.
#' @return A list with \code{success} (logical) and either
#'         \code{message} or \code{error}.
convert_seurat_to_h5ad <- function(input_path, output_path) {

  if (!file.exists(input_path)) {
    return(list(success = FALSE, error = paste0("Input file not found: ", input_path)))
  }

  # ------------------------------------------------------------------
  # 1. Load the object
  # ------------------------------------------------------------------
  obj <- load_seurat_object(input_path)

  if (is.null(obj)) {
    return(list(success = FALSE, error = "Could not load a Seurat object from the input file"))
  }

  if (!inherits(obj, "Seurat")) {
    return(list(success = FALSE, error = paste0(
      "Loaded object is not a Seurat object (class: ",
      paste(class(obj), collapse = ", "), ")"
    )))
  }

  version_info <- detect_seurat_version(obj)
  cat(sprintf("[convert] Detected Seurat version: %s\n", version_info))

  # ------------------------------------------------------------------
  # 2. Extract data via reticulate + anndata
  # ------------------------------------------------------------------
  result <- tryCatch(
    manual_convert(obj, output_path, version_info),
    error = function(e) {
      list(success = FALSE, error = paste0("Conversion failed: ", conditionMessage(e)))
    }
  )

  return(result)
}

# ---------------------------------------------------------------------------
# Load helpers
# ---------------------------------------------------------------------------

#' Load a Seurat object from an RDS or RData file.
#' For RData files the function searches the loaded environment for an
#' object that inherits from "Seurat".
load_seurat_object <- function(path) {

  ext <- tolower(tools::file_ext(path))

  if (ext == "rds") {
    cat("[convert] Loading RDS file ...\n")
    obj <- tryCatch(readRDS(path), error = function(e) NULL)
    return(obj)
  }

  if (ext %in% c("rdata", "rda")) {
    cat("[convert] Loading RData file ...\n")
    tmp_env <- new.env(parent = emptyenv())
    loaded  <- tryCatch(load(path, envir = tmp_env), error = function(e) NULL)

    if (is.null(loaded)) return(NULL)

    # Search for a Seurat object among loaded names
    for (nm in loaded) {
      candidate <- get(nm, envir = tmp_env)
      if (inherits(candidate, "Seurat")) {
        cat(sprintf("[convert] Found Seurat object '%s' in RData file.\n", nm))
        return(candidate)
      }
    }

    cat("[convert] No Seurat object found in RData file.\n")
    return(NULL)
  }

  cat(sprintf("[convert] Unsupported file extension: %s\n", ext))
  return(NULL)
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
