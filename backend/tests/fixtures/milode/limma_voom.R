# Reference for test_milode.py::test_engine_matches_edger_limma_voom.
# Run in this folder: Rscript limma_voom.R  (edgeR >= 4, limma >= 3.60)
suppressPackageStartupMessages({ library(edgeR); library(limma) })
counts <- as.matrix(read.delim("counts.tsv", row.names = 1, check.names = FALSE))
design <- as.matrix(read.delim("design.tsv", row.names = 1, check.names = FALSE))
lib <- read.delim("lib.tsv", row.names = 1)$lib
y <- DGEList(counts, lib.size = lib)
# As miloDE calls it: each sample its own group, min.total.count round(1.5 * 3).
keep <- filterByExpr(y, group = colnames(y), min.count = 3, min.total.count = 4)
y <- y[keep, , keep.lib.sizes = TRUE]
y <- calcNormFactors(y)
v <- voom(y, design)
fit <- eBayes(lmFit(v, design))
tt <- topTable(fit, coef = ncol(design), number = Inf, sort.by = "none")
write.table(signif(tt[, c("logFC", "P.Value")], 10), "limma_voom.tsv", sep = "\t",
            quote = FALSE, col.names = NA)
cat("edgeR", as.character(packageVersion("edgeR")), "limma", as.character(packageVersion("limma")),
    "genes kept", sum(keep), "\n")
