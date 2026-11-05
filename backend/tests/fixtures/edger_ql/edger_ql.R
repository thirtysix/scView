# Reference numbers for tests/test_edger_ql.py: R edgeR 4.6.3 / limma 3.64.3 (Bioconductor 3.21) on
# small simulated pseudobulk-like data. Run from this directory:
#   Rscript edger_ql.R
# Writes <case>.counts.tsv (genes x samples), <case>.design.tsv, <case>.genes.tsv (per gene),
# <case>.fit.tsv (per fit). Only edgeR's outputs are used; no edgeR code is copied anywhere.
suppressPackageStartupMessages({ library(edgeR); library(limma) })
set.seed(20261104)
nb <- function(mu, phi) matrix(rnbinom(length(mu), mu = mu, size = 1 / phi), nrow(mu))

write_case <- function(name, C, X, coef, robust = FALSE, data = TRUE) {
  keep <- rowSums(C) > 0
  C <- C[keep, , drop = FALSE]; dimnames(C) <- NULL
  y <- calcNormFactors(DGEList(C))
  fit <- glmQLFit(y, X, robust = robust)
  res <- glmQLFTest(fit, coef = coef)
  tt <- topTags(res, n = Inf, sort.by = "none")$table
  if (data) {
    write.table(C, paste0(name, ".counts.tsv"), sep = "\t", row.names = FALSE, col.names = FALSE)
    write.table(X, paste0(name, ".design.tsv"), sep = "\t", row.names = FALSE, col.names = FALSE)
  }
  g <- data.frame(AveLogCPM = fit$AveLogCPM, logFC = tt$logFC, F = tt$F, PValue = tt$PValue, FDR = tt$FDR,
                  s2_prior = fit$s2.prior, s2_post = fit$s2.post, df_prior = rep_len(fit$df.prior, nrow(C)),
                  df_residual_adj = fit$df.residual.adj, deviance_adj = fit$deviance.adj)
  write.table(format(g, digits = 12), paste0(name, ".genes.tsv"), sep = "\t", row.names = FALSE, quote = FALSE)
  f <- data.frame(coef = coef, robust = robust, dispersion = fit$dispersion[1], average_ql = fit$average.ql.dispersion,
                  norm_factors = paste(format(y$samples$norm.factors, digits = 15), collapse = ","))
  write.table(f, paste0(name, ".fit.tsv"), sep = "\t", row.names = FALSE, quote = FALSE)
}

# two groups 4 v 4: means from 0.05 to ~3000, gene-wise biological variation, 10% DE,
# 8% of genes silent in group b
G <- 400; n <- 8; g <- rep(0:1, each = 4)
mu0 <- exp(rnorm(G, 1.5, 2)); lfc <- ifelse(runif(G) < 0.1, rnorm(G, 0, 1.5), 0)
lib <- exp(rnorm(n, 0, 0.3)); phi <- 0.12 * exp(rnorm(G, 0, 0.5))
M <- outer(mu0, lib) * exp(outer(lfc, g))
C <- nb(M, phi)
C[runif(G) < 0.08, g == 1] <- 0
write_case("two_group", C, model.matrix(~factor(g)), 2)
write_case("two_group_robust", C, model.matrix(~factor(g)), 2, robust = TRUE, data = FALSE)

# very low counts, 3 v 3
G <- 300; n <- 6; g <- rep(0:1, each = 3)
mu0 <- exp(rnorm(G, -0.5, 1.2)); lib <- exp(rnorm(n, 0, 0.3))
C <- nb(outer(mu0, lib), 0.3)
write_case("low", C, model.matrix(~factor(g)), 2)

# paired: 4 donors x 2 conditions (not a one-way layout)
G <- 400; donor <- factor(rep(1:4, each = 2)); cond <- rep(0:1, 4)
mu0 <- exp(rnorm(G, 2, 1.8)); deff <- exp(matrix(rnorm(G * 4, 0, 0.5), G))[, as.integer(donor)]
lfc <- ifelse(runif(G) < 0.1, rnorm(G, 0, 1.2), 0)
C <- nb(mu0 * deff * exp(outer(lfc, cond)) * rep(exp(rnorm(8, 0, 0.3)), each = G), 0.05)
write_case("paired", C, model.matrix(~donor + cond), 5)

# group + continuous covariate, testing the covariate
G <- 400; n <- 10; g <- rep(0:1, each = 5); z <- rnorm(n)
mu0 <- exp(rnorm(G, 2, 1.8)); b <- ifelse(runif(G) < 0.1, rnorm(G, 0, 0.8), 0)
C <- nb(mu0 * exp(outer(b, z)) * rep(exp(rnorm(n, 0, 0.3)), each = G), 0.1)
write_case("covariate", C, model.matrix(~factor(g) + z), 3)
cat(R.version.string, "edgeR", as.character(packageVersion("edgeR")), "limma", as.character(packageVersion("limma")), "\n")
