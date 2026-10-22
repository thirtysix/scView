##
## make_fixtures.R <out_dir>
## Small R objects for the converter tests, saved as .rds / .RData, and
## expected.json: what each conversion must produce, computed here from the R
## objects themselves (sums, shapes, levels), never from the converter.
##

suppressPackageStartupMessages({
  library(Matrix)
  library(SingleCellExperiment)
  library(DelayedArray)
})

out <- commandArgs(trailingOnly = TRUE)[1]
if (is.na(out)) stop("usage: make_fixtures.R <out_dir>")
dir.create(out, showWarnings = FALSE, recursive = TRUE)
set.seed(1)

expected <- list()
rds <- function(obj, name) saveRDS(obj, file.path(out, paste0(name, ".rds")))

sparse_counts <- function(n_genes, n_cells, lambda = 0.6) {
  m <- matrix(rpois(n_genes * n_cells, lambda), n_genes, n_cells)
  methods::as(m, "CsparseMatrix")
}
lognorm <- function(counts) {
  sf <- Matrix::colSums(counts) / mean(Matrix::colSums(counts))
  m <- counts %*% Diagonal(x = 1 / sf)
  m@x <- log1p(m@x)
  methods::as(m, "CsparseMatrix")
}

# ---- 1. a full SingleCellExperiment ----------------------------------------
n_c <- 60; n_g <- 40
counts <- sparse_counts(n_g, n_c)
genes <- sprintf("Gene%02d", seq_len(n_g))
genes[2] <- genes[1]                       # a duplicate gene name
cells <- sprintf("AAAC%03d-1", seq_len(n_c))
dimnames(counts) <- list(genes, cells)
logc <- lognorm(counts)
dense <- matrix(rnorm(n_g * n_c), n_g, n_c, dimnames = list(genes, cells))
scaled <- DelayedArray(dense) * 2          # dense, delayed: realised block by block

sce <- SingleCellExperiment(assays = list(counts = counts, logcounts = logc, scaled = scaled))
cd <- colData(sce)
cd$cluster  <- factor(sample(c("b", "a", "c"), n_c, TRUE), levels = c("b", "a", "c"))
cd$stage    <- factor(sample(c("early", "late"), n_c, TRUE), levels = c("early", "late"), ordered = TRUE)
cd$donor    <- sample(c("d2", "d1"), n_c, TRUE); cd$donor[3] <- NA
cd$is_stim  <- rep(c(TRUE, FALSE), length.out = n_c)
cd$flag     <- rep(c(TRUE, FALSE, NA), length.out = n_c)
cd$n_umi    <- as.integer(Matrix::colSums(counts))
cd$n_int_na <- seq_len(n_c); cd$n_int_na[5] <- NA
cd$score    <- rnorm(n_c); cd$score[7] <- NA
cd$ranges   <- IRanges::IRanges(start = seq_len(n_c), width = 10)
cd$qc       <- S4Vectors::DataFrame(sum = Matrix::colSums(counts), detected = Matrix::colSums(counts > 0))
cd$batch_rle <- S4Vectors::Rle(factor(rep(c("b1", "b2"), each = n_c / 2)))
cd$listcol  <- I(lapply(seq_len(n_c), function(i) i))
colData(sce) <- cd
sizeFactors(sce) <- Matrix::colSums(counts) / mean(Matrix::colSums(counts))
colLabels(sce) <- factor(sample(c("T", "B"), n_c, TRUE))

rd <- rowData(sce)
rd$Symbol <- genes
rd$biotype <- sample(c("protein_coding", "lncRNA"), n_g, TRUE)
rd[["a/b"]] <- seq_len(n_g)
rd[["_index"]] <- rev(seq_len(n_g))
rowData(sce) <- rd

pca <- matrix(rnorm(n_c * 5), n_c, 5)
attr(pca, "percentVar") <- c(20, 10, 5, 3, 2)
attr(pca, "varExplained") <- c(8, 4, 2, 1.2, 0.8)
rot <- matrix(rnorm(10 * 5), 10, 5, dimnames = list(genes[c(3, 5, 7, 9, 11, 13, 15, 17, 19, 21)], NULL))
attr(pca, "rotation") <- rot
reducedDim(sce, "PCA") <- pca
reducedDim(sce, "UMAP") <- matrix(rnorm(n_c * 2), n_c, 2)
reducedDim(sce, "UMAP.HARMONY") <- matrix(rnorm(n_c * 2), n_c, 2)
reducedDim(sce, "DF_EMB") <- data.frame(a = rnorm(n_c), b = rnorm(n_c), row.names = cells)

adt <- sparse_counts(5, n_c, 20); dimnames(adt) <- list(c("CD3", "CD4", "CD8", "CD14", "IgG1"), cells)
hto <- sparse_counts(3, n_c, 30); dimnames(hto) <- list(paste0("Hashtag", 1:3), cells)
ercc <- sparse_counts(4, n_c, 2); dimnames(ercc) <- list(paste0("ERCC-0000", 1:4), cells)
# as read10xCounts: the row names are the 10x IDs, Symbol the feature names
altExp(sce, "ADT") <- SingleCellExperiment(
  assays = list(counts = adt),
  rowData = S4Vectors::DataFrame(Symbol = paste0(c("CD3", "CD4", "CD8", "CD14", "IgG1_control"), "_TotalSeqB"))
)
altExp(sce, "HTO") <- SingleCellExperiment(assays = list(counts = hto))
altExp(sce, "ERCC") <- SingleCellExperiment(assays = list(counts = ercc))
colPair(sce, "SNN") <- S4Vectors::SelfHits(from = 1:(n_c - 1), to = 2:n_c, nnode = n_c)
metadata(sce) <- list(
  study = "fixture",
  params = list(k = 10L, method = "louvain", cutoffs = c(1.5, 2)),
  table = data.frame(a = 1:3, b = c("x", "y", "z")),
  fn = function(x) x
)
rds(sce, "sce_full")

expected$sce_full <- list(
  n_obs = n_c, n_vars = n_g, x = "logcounts", x_sum = sum(logc),
  x_format = "csr_matrix",
  layers = list(counts = sum(counts), scaled = sum(dense * 2)),
  layer_formats = list(counts = "csr_matrix", scaled = "array"),
  obs_names_head = cells[1:3],
  var_names_head = c("Gene01", "Gene01-1", "Gene03"),
  obsm = list(X_pca = c(n_c, 5), X_umap = c(n_c, 2), X_umap_harmony = c(n_c, 2), X_df_emb = c(n_c, 2)),
  obsm_sums = list(X_pca = sum(pca), X_umap = sum(reducedDim(sce, "UMAP"))),
  protein = list(columns = c("CD3", "CD4", "CD8", "CD14", "IgG1_control"), sum = sum(adt)),
  hto = list(columns = rownames(hto), sum = sum(hto)),
  pca_variance_ratio = c(20, 10, 5, 3, 2) / 100,
  pca_variance = c(8, 4, 2, 1.2, 0.8),
  pcs_rows = c(3, 5, 7, 9, 11, 13, 15, 17, 19, 21) - 1,
  pcs_sum = sum(rot),
  obs_dtypes = list(cluster = "category", stage = "category", donor = "category",
                    is_stim = "bool", flag = "category", n_umi = "int32", n_int_na = "float64",
                    score = "float64", qc.sum = "float64", qc.detected = "int32",
                    batch_rle = "category", sizeFactor = "float64", label = "category"),
  categories = list(cluster = c("b", "a", "c"), stage = c("early", "late"), donor = c("d1", "d2"),
                    flag = c("FALSE", "TRUE"), batch_rle = c("b1", "b2")),
  ordered = list("stage"),
  n_missing = list(donor = 1, flag = sum(is.na(cd$flag)), n_int_na = 1, score = 1),
  obs_absent = list("ranges", "listcol"),
  var_columns = list("Symbol", "biotype", "a_b", "_index_"),
  uns_metadata = list("study", "params", "table"),
  report_skipped = list("ranges", "listcol", "ERCC", "colPairs SNN", "metadata$fn"),
  report_kept = list("protein modality", "hashtag counts", "sizeFactor", "colLabels", "percentVar")
)

# ---- 2. counts only, dense integers, no cell names, Ensembl row names ------
n_c2 <- 30; n_g2 <- 20
ids <- sprintf("ENSG%011d", seq_len(n_g2))
m2 <- matrix(rpois(n_g2 * n_c2, 2L), n_g2, n_c2, dimnames = list(ids, NULL))
storage.mode(m2) <- "integer"
sce2 <- SingleCellExperiment(assays = list(counts = m2))
sym <- sprintf("SYM%d", seq_len(n_g2)); sym[4] <- sym[3]; sym[6] <- NA
rowData(sce2)$Symbol <- sym
rds(sce2, "sce_counts_only")
expected$sce_counts_only <- list(
  n_obs = n_c2, n_vars = n_g2, x = "counts", x_sum = sum(m2), x_format = "array",
  layers = structure(list(), names = character(0)),
  obs_names_head = c("cell_1", "cell_2", "cell_3"),
  var_names_head = c("SYM1", "SYM2", "SYM3", "SYM3-1", "SYM5", ids[6]),
  gene_ids_head = ids[1:3],
  report_kept = list("Ensembl IDs", "numbered cell_1")
)

# ---- 3. antibody and hashtag rows in the main experiment (read10xCounts) ---
n_c3 <- 50
m3 <- sparse_counts(36, n_c3)
rn3 <- c(sprintf("G%02d", 1:30), paste0("AB", 1:4), paste0("HTO", 1:2))
dimnames(m3) <- list(rn3, sprintf("C%02d", seq_len(n_c3)))
sce3 <- SingleCellExperiment(assays = list(counts = m3))
rowData(sce3)$ID <- rn3
rowData(sce3)$Type <- c(rep("Gene Expression", 30), rep("Antibody Capture", 4), rep("Multiplexing Capture", 2))
rds(sce3, "sce_feature_types")
expected$sce_feature_types <- list(
  n_obs = n_c3, n_vars = 30, x = "counts", x_sum = sum(m3[1:30, ]), x_format = "csr_matrix",
  layers = structure(list(), names = character(0)),
  var_names_head = c("G01", "G02"),
  protein = list(columns = paste0("AB", 1:4), sum = sum(m3[31:34, ])),
  hto = list(columns = paste0("HTO", 1:2), sum = sum(m3[35:36, ])),
  report_kept = list("Antibody Capture rows", "Multiplexing Capture rows")
)

# ---- 4. a plain (Ranged)SummarizedExperiment ------------------------------
n_c4 <- 12; n_g4 <- 8
ex <- matrix(runif(n_g4 * n_c4), n_g4, n_c4)
ct <- sparse_counts(n_g4, n_c4, 3)
gr <- GenomicRanges::GRanges("chr1", IRanges::IRanges(start = seq(1, by = 100, length.out = n_g4), width = 50))
se <- SummarizedExperiment::SummarizedExperiment(
  assays = list(exprs = ex, counts = ct), rowRanges = gr,
  colData = S4Vectors::DataFrame(group = factor(rep(c("x", "y"), 6)), row.names = paste0("s", 1:n_c4))
)
rownames(se) <- paste0("g", 1:n_g4)
rds(se, "se_plain")
expected$se_plain <- list(
  n_obs = n_c4, n_vars = n_g4, x = "counts", x_sum = sum(ct), x_format = "csr_matrix",
  layers = list(exprs = sum(ex)), layer_formats = list(exprs = "array"),
  obs_dtypes = list(group = "category"),
  report_skipped = list("rowRanges")
)

# ---- 5. a subclass from a package not installed (SpatialExperiment) --------
n_c5 <- 20
m5 <- sparse_counts(10, n_c5); dimnames(m5) <- list(paste0("g", 1:10), paste0("spot", 1:n_c5))
spe <- SingleCellExperiment(assays = list(counts = m5))
reducedDim(spe, "UMAP") <- matrix(rnorm(n_c5 * 2), n_c5, 2)
xy <- matrix(runif(n_c5 * 2) * 1000, n_c5, 2, dimnames = list(NULL, c("x", "y")))
int_colData(spe)$spatialCoords <- xy
attr(spe, "class") <- structure("SpatialExperiment", package = "SpatialExperiment")
rds(spe, "spe_like")
expected$spe_like <- list(
  n_obs = n_c5, n_vars = 10, x = "counts", x_sum = sum(m5), x_format = "csr_matrix",
  layers = structure(list(), names = character(0)),
  obsm = list(X_umap = c(n_c5, 2), spatial = c(n_c5, 2)),
  obsm_sums = list(spatial = sum(xy)),
  report_kept = list("SpatialExperiment", "spatialCoords")
)

# ---- 6. an assay that cannot be read (its values live outside the .rds) ----
setClass("ExternalSeed", representation(dim = "integer"))
setMethod("dim", "ExternalSeed", function(x) x@dim)
setMethod("dimnames", "ExternalSeed", function(x) list(NULL, NULL))
setMethod(DelayedArray::extract_array, "ExternalSeed", function(x, index) {
  d <- vapply(seq_along(index), function(k) if (is.null(index[[k]])) x@dim[k] else length(index[[k]]), 1L)
  array(0, dim = d)
})
n_c6 <- 15
m6 <- sparse_counts(9, n_c6); dimnames(m6) <- list(paste0("g", 1:9), paste0("c", 1:n_c6))
ext <- DelayedArray(new("ExternalSeed", dim = c(9L, as.integer(n_c6))))
sce6 <- SingleCellExperiment(assays = list(counts = m6, logcounts = ext))
rds(sce6, "sce_unreadable")
expected$sce_unreadable <- list(
  n_obs = n_c6, n_vars = 9, x = "counts", x_sum = sum(m6), x_format = "csr_matrix",
  layers = structure(list(), names = character(0)),
  report_skipped = list("assay logcounts")
)

# ---- 7. delayed sparse logcounts + a dense delayed layer, unnamed altExp ---
n_c7 <- 25
m7 <- sparse_counts(12, n_c7); dimnames(m7) <- list(paste0("g", 1:12), paste0("c", 1:n_c7))
d7 <- matrix(rnorm(12 * n_c7), 12, n_c7)
sce7 <- SingleCellExperiment(assays = list(counts = m7, logcounts = log1p(DelayedArray(m7)),
                                           other = DelayedArray(d7) * 3 + 1))
rds(sce7, "sce_delayed")
expected$sce_delayed <- list(
  n_obs = n_c7, n_vars = 12, x = "logcounts", x_sum = sum(log1p(as.matrix(m7))), x_format = "csr_matrix",
  layers = list(counts = sum(m7), other = sum(d7 * 3 + 1)),
  layer_formats = list(counts = "csr_matrix", other = "array")
)

# ---- 8. unnamed assays ----------------------------------------------------
m8a <- sparse_counts(6, 10); m8b <- matrix(runif(60), 6, 10)
se8 <- SummarizedExperiment::SummarizedExperiment(assays = list(m8a, m8b))
rds(se8, "se_unnamed")
expected$se_unnamed <- list(
  n_obs = 10, n_vars = 6, x = "assay1", x_sum = sum(m8a), x_format = "csr_matrix",
  layers = list(assay2 = sum(m8b)), layer_formats = list(assay2 = "array"),
  obs_names_head = c("cell_1", "cell_2"), var_names_head = c("gene_1", "gene_2")
)

# ---- 9. an RData file with a table and an SCE -----------------------------
sce_small <- SingleCellExperiment(assays = list(counts = m7))
some_table <- data.frame(a = 1:3)
save(some_table, sce_small, file = file.path(out, "objects.RData"))
expected$objects <- list(
  n_obs = n_c7, n_vars = 12, x = "counts", x_sum = sum(m7), x_format = "csr_matrix",
  layers = structure(list(), names = character(0)), object_name = "sce_small"
)

# ---- 10. objects scView cannot convert ------------------------------------
rds(list(a = 1, b = "x"), "a_list")
rds(m7, "bare_matrix")
expected$a_list <- list(error = list("holds a list object"))
expected$bare_matrix <- list(error = list("dgCMatrix (from the Matrix package)", "A bare matrix"))

# ---- 11. a Seurat object (the original path) ------------------------------
if (requireNamespace("SeuratObject", quietly = TRUE)) {
  m11 <- sparse_counts(30, 40, 1); dimnames(m11) <- list(paste0("g", 1:30), paste0("c", 1:40))
  so <- suppressWarnings(SeuratObject::CreateSeuratObject(counts = m11))
  so$group <- rep(c("a", "b"), 20)
  emb <- matrix(rnorm(40 * 4), 40, 4, dimnames = list(colnames(m11), paste0("PC_", 1:4)))
  so[["pca"]] <- SeuratObject::CreateDimReducObject(embeddings = emb, key = "PC_", assay = "RNA")
  um <- matrix(rnorm(40 * 2), 40, 2, dimnames = list(colnames(m11), paste0("UMAP_", 1:2)))
  so[["umap"]] <- SeuratObject::CreateDimReducObject(embeddings = um, key = "UMAP_", assay = "RNA")
  adt11 <- sparse_counts(3, 40, 10); dimnames(adt11) <- list(c("CD3", "CD4", "CD8"), colnames(m11))
  so[["ADT"]] <- suppressWarnings(SeuratObject::CreateAssay5Object(counts = adt11))
  rds(so, "seurat")
  expected$seurat <- list(
    n_obs = 40, n_vars = 30, x_sum = sum(m11), x_format = "csc_matrix",
    layers = structure(list(), names = character(0)),
    obsm = list(X_pca = c(40, 4), X_umap = c(40, 2)),
    protein = list(columns = c("CD3", "CD4", "CD8"), sum = sum(adt11)),
    source_class = "Seurat"
  )
}

writeLines(jsonlite::toJSON(expected, auto_unbox = TRUE, digits = NA), file.path(out, "expected.json"))
cat(sprintf("[fixtures] %d fixtures in %s\n", length(expected), out))
