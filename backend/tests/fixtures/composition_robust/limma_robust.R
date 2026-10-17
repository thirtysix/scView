# Reference for test_composition.py robust tests: Rscript limma_robust.R in this folder
# (limma 3.68.1). The tables are simulated counts, samples x groups; t2 and t3 carry
# one group with extra sample-to-sample variance (g3, g5).
library(limma)
out <- list()
for (tag in c("t2", "t3", "t2b")) {
  M <- as.matrix(read.delim(paste0(tag, ".tsv"), row.names = 1))
  cond <- factor(sub("__.*", "", rownames(M))); donor <- factor(sub(".*__", "", rownames(M)))
  for (tr in c("logit", "asin")) {
    if (tr == "logit") { p <- (M + 0.5) / rowSums(M + 0.5); y <- t(log(p / (1 - p))) }
    else y <- t(asin(sqrt(M / rowSums(M))))
    for (des in c("paired", "unpaired")) {
      X <- if (des == "paired") model.matrix(~cond + donor) else model.matrix(~cond)
      fit <- lmFit(y, X); k <- 2:nlevels(cond)
      if (length(k) == 1) { f <- eBayes(fit, robust = TRUE); st <- f$t[, 2]; pv <- f$p.value[, 2] }
      else { f <- eBayes(contrasts.fit(fit, coefficients = k), robust = TRUE); st <- f$F; pv <- f$F.p.value }
      out[[paste(tag, tr, des)]] <- data.frame(tag, tr, des, g = rownames(y), stat = st, p = pv, dfp = f$df.prior)
    }
  }
}
write.table(do.call(rbind, out), "limma_robust.tsv", sep = "\t", quote = FALSE, row.names = FALSE)
