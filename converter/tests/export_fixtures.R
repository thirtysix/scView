##
## export_fixtures.R <out_dir>
## Exports the staging fixtures make_export_fixtures.py wrote to Seurat and
## SingleCellExperiment .rds files with export_rds.R, reads them back, and checks
## them against export_expected.json (computed from the arrays in Python). The
## Seurat object also goes back through the import (convert_rds_to_h5ad).
## Exit status 1 when any check fails.
##

suppressPackageStartupMessages({
  library(Matrix)
  library(SeuratObject)
  library(SingleCellExperiment)
})

out <- commandArgs(trailingOnly = TRUE)[1]
if (is.na(out)) stop("usage: export_fixtures.R <out_dir>")
file_arg <- sub("^--file=", "", grep("^--file=", commandArgs(FALSE), value = TRUE))
here <- dirname(normalizePath(file_arg))
source(file.path(here, "..", "convert.R"))
source(file.path(here, "..", "convert_sce.R"))
source(file.path(here, "..", "export_rds.R"))

exp <- jsonlite::fromJSON(file.path(out, "export_expected.json"))
failures <- character(0)
check <- function(ok, what) {
  ok <- isTRUE(ok)
  cat(sprintf("[export] %s %s\n", if (ok) "PASS" else "FAIL", what))
  if (!ok) failures <<- c(failures, what)
}
near <- function(a, b) isTRUE(abs(a - b) <= 1e-6 * max(1, abs(b)))

run <- function(input, fmt) {
  rds <- file.path(out, sprintf("%s_%s.rds", tools::file_path_sans_ext(basename(input)), fmt))
  prog <- paste0(rds, ".progress.json")
  res <- export_h5ad_to_rds(file.path(out, input), rds, fmt, prog)
  check(isTRUE(res$success), sprintf("%s -> %s: %s", input, fmt, res$message %||% res$error))
  check(jsonlite::fromJSON(prog)$stage == "done", sprintf("%s -> %s: progress file ends 'done'", input, fmt))
  list(res = res, rds = rds)
}

# ---- Seurat, full ------------------------------------------------------------
s <- run("export_full.h5ad", "seurat")
obj <- readRDS(s$rds)
check(inherits(obj, "Seurat") && validObject(obj), "Seurat: a valid Seurat object")
check(inherits(obj[["RNA"]], "Assay5"), "Seurat: RNA is an Assay5")
check(all(dim(obj) == c(exp$n_genes, exp$n_cells)), "Seurat: genes x cells")
cnt <- LayerData(obj, assay = "RNA", layer = "counts")
dat <- LayerData(obj, assay = "RNA", layer = "data")
check(inherits(cnt, "dgCMatrix") && inherits(dat, "dgCMatrix"), "Seurat: counts and data stay sparse")
check(near(sum(cnt), exp$counts_sum) && length(cnt@x) == exp$counts_nnz, "Seurat: counts sum and nnz")
check(cnt[8, 4] == exp$count_cell3_gene7, "Seurat: counts[gene 7, cell 3] (genes x cells)")
check(near(sum(dat), exp$logn_sum), "Seurat: data = the log-normalised values")
check(!"scale.data" %in% Layers(obj, assay = "RNA"), "Seurat: no scale.data")
check(sum(rownames(obj) != c(sprintf("Gene_%d", 0:3), sprintf("G%d", 4:29))) == exp$n_renamed,
      "Seurat: gene names changed ('_' -> '-') as reported")
check(rownames(obj)[6] == exp$gene_5_seurat, "Seurat: a clash made unique")
check(s$res$report$renamed_features == exp$n_renamed, "Seurat: report counts the changed names")
check(!any(grepl("_", rownames(obj))), "Seurat: no '_' in feature names")
md <- obj[[]]
check(identical(levels(md$leiden), exp$leiden_levels), "Seurat: leiden level order kept")
check(identical(as.character(md$leiden[1:5]), exp$leiden_first), "Seurat: leiden values")
check(is.ordered(md$stage) && identical(levels(md$stage), exp$stage_levels), "Seurat: ordered factor")
check(is.na(md$donor[exp$donor_na + 1]) && sum(is.na(md$donor)) == 1, "Seurat: factor NA kept")
check(is.logical(md$is_stim) && identical(md$is_stim[1:5], exp$is_stim_first), "Seurat: logical column")
check(is.integer(md$n_genes) && identical(md$n_genes[1:5], as.integer(exp$n_genes_first)), "Seurat: integer column")
check(is.na(md$score[exp$score_na + 1]), "Seurat: numeric NA kept")
check(identical(sort(Reductions(obj)), c("pca", "umap", "umap_3d")), "Seurat: reductions pca, umap, umap_3d")
check(Key(obj[["pca"]]) == "PC_" && Key(obj[["umap"]]) == "umap_" && Key(obj[["umap_3d"]]) == "umap3d_",
      "Seurat: reduction keys")
emb <- Embeddings(obj, "pca")
check(near(emb[1, 1], exp$pca_00) && near(emb[8, 4], exp$pca_cell7_dim3), "Seurat: PCA embedding values")
check(ncol(Embeddings(obj, "umap_3d")) == 3, "Seurat: 3-D layout keeps its 3 columns")
check(nrow(Loadings(obj, "pca")) == 12 && ncol(Loadings(obj, "pca")) == 5, "Seurat: PCA loadings (12 genes)")
check(near(Stdev(obj, "pca")[1], sqrt(5)), "Seurat: PCA stdev = sqrt(variance)")
check(length(VariableFeatures(obj)) == 12 && "Gene-0" %in% VariableFeatures(obj), "Seurat: VariableFeatures")
check(identical(levels(Idents(obj)), exp$leiden_levels), "Seurat: Idents = the clustering")
check("ADT" %in% names(obj@assays) && nrow(obj[["ADT"]]) == 3, "Seurat: ADT assay")
check(near(sum(LayerData(obj, assay = "ADT", layer = "counts")), exp$protein_sum) &&
        near(sum(LayerData(obj, assay = "ADT", layer = "data")), exp$clr_sum), "Seurat: ADT counts + CLR")
check(rownames(obj[["ADT"]])[1] == "CD3-TotalSeqB", "Seurat: protein names without '_'")
m <- Misc(obj, "scview")
check(m$scview_version == "0.0.test" && identical(m$recipe, c("normalize", "pca")), "Seurat: Misc provenance")
check(any(grepl("connectivities", unlist(s$res$report$skipped))), "Seurat: report lists what was left out")
fm <- obj[["RNA"]][[]]
check("highly_variable" %in% colnames(fm), "Seurat: var columns as feature metadata")

# ---- the same object back through the import -------------------------------
back <- convert_rds_to_h5ad(s$rds, file.path(out, "export_full_seurat_back.h5ad"))
check(isTRUE(back$success), "Seurat -> import: converts back to h5ad")

# ---- SingleCellExperiment, full ----------------------------------------------
x <- run("export_full.h5ad", "sce")
sce <- readRDS(x$rds)
check(is(sce, "SingleCellExperiment") && validObject(sce), "SCE: a valid SingleCellExperiment")
check(identical(assayNames(sce), c("counts", "logcounts")), "SCE: assays counts, logcounts")
check(near(sum(counts(sce)), exp$counts_sum) && near(sum(logcounts(sce)), exp$logn_sum), "SCE: assay sums")
check(identical(rownames(sce)[1], "Gene_0"), "SCE: gene names unchanged")
check(identical(levels(sce$leiden), exp$leiden_levels) && is.logical(sce$is_stim), "SCE: colData kinds")
check(identical(reducedDimNames(sce), c("PCA", "UMAP", "UMAP_3D")), "SCE: reducedDims PCA, UMAP, UMAP_3D")
pc <- reducedDim(sce, "PCA")
check(near(pc[8, 4], exp$pca_cell7_dim3) && near(attr(pc, "percentVar")[1], 30) &&
        identical(dim(attr(pc, "rotation")), c(30L, 5L)), "SCE: PCA with percentVar and rotation")
check(identical(altExpNames(sce), "ADT") && near(sum(assay(altExp(sce, "ADT"), "counts")), exp$protein_sum),
      "SCE: altExp ADT")
check(metadata(sce)$scview$dataset_name == "Fixture set", "SCE: metadata$scview")

# ---- counts only (not normalised; dense counts) -----------------------------
s2 <- run("export_counts.h5ad", "seurat")
o2 <- readRDS(s2$rds)
check(identical(Layers(o2, assay = "RNA"), "counts") && near(sum(LayerData(o2, layer = "counts")), exp$counts_sum),
      "counts only -> Seurat: one counts layer, no data")
check(any(grepl("NormalizeData", unlist(s2$res$report$skipped))), "counts only -> Seurat: report says NormalizeData")
check(identical(levels(Idents(o2)), "Fixture set"), "counts only -> Seurat: Idents = orig.ident (no clustering)")
x2 <- run("export_counts.h5ad", "sce")
check(identical(assayNames(readRDS(x2$rds)), "counts"), "counts only -> SCE: counts assay only")

bad <- export_h5ad_to_rds(file.path(out, "export_full.h5ad"), file.path(out, "x.rds"), "loom")
check(!isTRUE(bad$success) && grepl("Unknown format", bad$error), "unknown format refused")
missing <- export_h5ad_to_rds(file.path(out, "nope.h5ad"), file.path(out, "x.rds"), "seurat")
check(!isTRUE(missing$success) && grepl("not found", missing$error), "missing input refused")

cat(sprintf("[export] %d failed\n", length(failures)))
if (length(failures)) quit(status = 1)
