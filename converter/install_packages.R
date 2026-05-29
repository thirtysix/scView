##
## install_packages.R
## Install all R packages required by the scView converter service.
## Each installation is wrapped in tryCatch so that a single failure
## does not abort the entire build.
##

cat("=== Installing R packages for scView converter ===\n")

# ---------- helper ----------------------------------------------------------
safe_install <- function(expr_str, label) {
  cat(sprintf("[install] %s ...\n", label))
  tryCatch(
    {
      eval(parse(text = expr_str))
      cat(sprintf("[install] %s  OK\n", label))
    },
    error = function(e) {
      cat(sprintf("[install] %s  FAILED: %s\n", label, conditionMessage(e)))
    }
  )
}

# ---------- CRAN packages ---------------------------------------------------
# SeuratObject provides GetAssayData/Embeddings for Seurat v3/v4/v5 objects
cran_pkgs <- c("SeuratObject", "httpuv", "jsonlite", "Matrix", "reticulate")

for (pkg in cran_pkgs) {
  safe_install(
    sprintf('install.packages("%s", repos = "https://cloud.r-project.org")', pkg),
    paste0("CRAN: ", pkg)
  )
}

cat("=== Package installation complete ===\n")

# Print a quick summary of what is available
installed <- rownames(installed.packages())
missing   <- setdiff(cran_pkgs, installed)

if (length(missing) == 0) {
  cat("All expected packages are installed.\n")
} else {
  cat(sprintf("WARNING - missing packages: %s\n", paste(missing, collapse = ", ")))
}
