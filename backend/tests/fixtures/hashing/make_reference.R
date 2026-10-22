# References for tests/test_hashing.py, made with R 4.4.0, Seurat 5.1.0
# (SeuratObject 5.0.2), cluster 2.1.6, fitdistrplus 1.1-11, DropletUtils 1.24.0:
#   Rscript make_reference.R   (in this folder)
# pbmc_subset_counts.csv: 1,000 cells (seeded sample) of the GSE108313 PBMC hashtag
# counts (Stoeckius et al. 2018; GEO GSM2895283, BatchA-H renamed HTO-A..HTO-H as in
# Seurat's hashing vignette).
suppressMessages({library(Seurat); library(cluster); library(DropletUtils)})
pts <- read.csv("clara_points.csv")
out <- do.call(rbind, lapply(split(pts, pts$case), function(d) {
  x <- as.matrix(d[, c("x1", "x2", "x3", "x4")])
  x <- x[, colSums(x != 0) > 0, drop = FALSE]
  data.frame(case = d$case, i = d$i, cluster = clara(x, k = d$k[1], samples = 100)$clustering)
}))
write.table(out, "clara_reference.tsv", sep = "\t", row.names = FALSE, quote = FALSE)
counts <- read.csv("pbmc_subset_counts.csv", row.names = 1, check.names = FALSE)
m <- t(as.matrix(counts))
obj <- CreateSeuratObject(counts = as(m, "dgCMatrix"), assay = "HTO")
obj <- NormalizeData(obj, assay = "HTO", normalization.method = "CLR", verbose = FALSE)
obj <- HTODemux(obj, assay = "HTO", positive.quantile = 0.99, verbose = FALSE)
md <- obj[[]]
hd <- hashedDrops(m)
# Rows in the order of pbmc_subset_counts.csv.
write.table(data.frame(maxID = md$HTO_maxID, secondID = md$HTO_secondID,
                       margin = sprintf("%.10g", md$HTO_margin), global = md$HTO_classification.global,
                       hd_best = hd$Best, hd_second = hd$Second, hd_lfc = sprintf("%.10g", hd$LogFC),
                       hd_lfc2 = sprintf("%.10g", hd$LogFC2), hd_doublet = hd$Doublet,
                       hd_confident = hd$Confident),
            "reference_calls.tsv", sep = "\t", row.names = FALSE, quote = FALSE)
# NB background fits as R makes them (qnbinom of the fitted size / mu).
q <- data.frame(size = c(0.813261, 1.28277, 1.64008, 3.30842), mu = c(16.8135, 21.1512, 23.301, 2.68283))
q$q99 <- qnbinom(0.99, size = q$size, mu = q$mu)
q$q90 <- qnbinom(0.90, size = q$size, mu = q$mu)
write.table(q, "qnbinom_reference.tsv", sep = "\t", row.names = FALSE, quote = FALSE)
