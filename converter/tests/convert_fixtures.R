##
## convert_fixtures.R <out_dir>
## Converts every fixture make_fixtures.R saved, with the converter's own
## code, and writes results.json (success, error, report per fixture).
##

args <- commandArgs(trailingOnly = TRUE)
out <- args[1]
if (is.na(out)) stop("usage: convert_fixtures.R <out_dir>")

file_arg <- sub("^--file=", "", grep("^--file=", commandArgs(FALSE), value = TRUE))
here <- dirname(normalizePath(file_arg))
source(file.path(here, "..", "convert.R"))
source(file.path(here, "..", "convert_sce.R"))

inputs <- list.files(out, pattern = "\\.(rds|RData)$", full.names = TRUE)
results <- list()
for (path in inputs) {
  name <- tools::file_path_sans_ext(basename(path))
  h5ad <- file.path(out, paste0(name, ".h5ad"))
  progress <- file.path(out, paste0(name, ".progress.json"))
  if (file.exists(h5ad)) file.remove(h5ad)
  t0 <- Sys.time()
  res <- convert_rds_to_h5ad(path, h5ad, progress)
  res$seconds <- as.numeric(difftime(Sys.time(), t0, units = "secs"))
  res$progress <- jsonlite::fromJSON(progress)
  results[[name]] <- res
  cat(sprintf("[fixtures] %-20s %s\n", name, if (isTRUE(res$success)) res$message else paste("ERROR:", res$error)))
}

writeLines(jsonlite::toJSON(results, auto_unbox = TRUE, digits = NA, null = "null"),
           file.path(out, "results.json"))
