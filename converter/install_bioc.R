##
## install_bioc.R
## Bioconductor packages for SingleCellExperiment / SummarizedExperiment
## objects. Bioconductor 3.19 is the release for this image's R (4.4.0) and is
## frozen, so it always installs the versions below; the build fails if any
## differs. There are no Bioconductor binaries for rocker/r-ver: the 16
## Bioconductor packages build from source on amd64 and arm64 alike (about 2 min
## on 2 cores); their 9 small CRAN dependencies come from the image's CRAN
## snapshot (binaries where it has them, else source).
##

cat("=== Installing Bioconductor packages for scView converter ===\n")

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

# Straight from the 3.19 repositories, not BiocManager: BiocManager takes 3.20 as
# current for R 4.4 and would first downgrade the CRAN packages installed above.
# install.packages only adds what is missing, so SeuratObject, Matrix and the
# rest stay exactly as they are and the Seurat conversion does not change.
bioc_version <- "3.19"
bioc_repos   <- c(
  BioCsoft = sprintf("https://bioconductor.org/packages/%s/bioc", bioc_version),
  BioCann  = sprintf("https://bioconductor.org/packages/%s/data/annotation", bioc_version),
  CRAN     = getOption("repos")[["CRAN"]]
)
bioc_pkgs <- c("SingleCellExperiment", "SummarizedExperiment", "S4Vectors",
               "DelayedArray")
safe_install(
  sprintf(
    'install.packages(c(%s), repos = bioc_repos, Ncpus = 2L)',
    paste0('"', bioc_pkgs, '"', collapse = ", ")
  ),
  paste0("Bioconductor ", bioc_version, ": ", paste(bioc_pkgs, collapse = ", "))
)

cat("=== Bioconductor installation complete ===\n")

# Every version Bioconductor 3.19 resolves for these four packages (with their
# Bioconductor dependencies). A mismatch fails the build: then the release
# changed or a different one was used, and conversions could differ.
expected <- c(
  SingleCellExperiment = "1.26.0",
  SummarizedExperiment = "1.34.0",
  S4Vectors            = "0.42.1",
  DelayedArray         = "0.30.1",
  BiocGenerics         = "0.50.0",
  Biobase              = "2.64.0",
  IRanges              = "2.38.1",
  GenomicRanges        = "1.56.2",
  GenomeInfoDb         = "1.40.1",
  GenomeInfoDbData     = "1.2.12",
  UCSC.utils           = "1.0.0",
  MatrixGenerics       = "1.16.0",
  S4Arrays             = "1.4.1",
  SparseArray          = "1.4.8",
  XVector              = "0.44.0",
  zlibbioc             = "1.50.0"
)

installed <- installed.packages()[, "Version"]
missing   <- setdiff(names(expected), names(installed))
if (length(missing) > 0) {
  cat(sprintf("ERROR - missing packages: %s\n", paste(missing, collapse = ", ")))
  quit(status = 1)
}
found <- installed[names(expected)]
wrong <- names(expected)[found != expected]
if (length(wrong) > 0) {
  cat(sprintf("ERROR - unexpected versions: %s\n",
              paste(sprintf("%s %s (expected %s)", wrong, found[wrong], expected[wrong]),
                    collapse = ", ")))
  quit(status = 1)
}
cat(sprintf("Bioconductor %s packages: %s\n", bioc_version,
            paste(sprintf("%s %s", names(found), found), collapse = ", ")))
