import type { PseudobulkQc } from "@/api/sampleQc";

/** A Kang-like pseudobulk QC result for the tests: 4 profiles, one outlier. */
export function result(extra: Partial<PseudobulkQc> = {}): PseudobulkQc {
  const profiles = [
    { id: "D1 | CTRL", sample: "D1", split: "CTRL", n_cells: 300, total_counts: 5e5, pcs: [-30, 2, 0.1], distance: 1.2, outlier: false },
    { id: "D1 | STIM", sample: "D1", split: "STIM", n_cells: 310, total_counts: 5e5, pcs: [31, 1, 0.2], distance: 0.9, outlier: false },
    { id: "D2 | CTRL", sample: "D2", split: "CTRL", n_cells: 90, total_counts: 2e5, pcs: [-28, -9, 4], distance: 5.1, outlier: true },
    { id: "D2 | STIM", sample: "D2", split: "STIM", n_cells: 280, total_counts: 4e5, pcs: [29, 3, 0.3], distance: 1.0, outlier: false },
  ];
  return {
    sample_column: "Donor", reason: "donor", split_by: "stim", celltype_column: "cluster", celltype: null,
    celltype_levels: [{ name: "CD14 Mono", n_cells: 400 }, { name: "B", n_cells: 100 }],
    n_cells: 1000, n_cells_used: 980, min_cells: 10, profiles,
    dropped: [{ sample: "D3", split: "STIM", n_cells: 4 }], n_genes: 500, n_top: 500,
    explained_variance: [0.75, 0.049, 0.03],
    covariates: [
      { name: "stim", kind: "categorical", derived: false, same_as: ["orig_ident"], values: ["CTRL", "STIM", "CTRL", "STIM"] },
      { name: "Donor", kind: "categorical", derived: false, same_as: [], values: ["D1", "D1", "D2", "D2"] },
      { name: "n_cells", kind: "numeric", derived: true, same_as: [], values: [300, 310, 90, 280] },
    ],
    association: [
      { covariate: "stim", kind: "categorical", same_as: ["orig_ident"], test: "ANOVA", pcs: [
        { pc: 1, r2: 0.98, p: 2.6e-13, p_rank: 1e-4 }, { pc: 2, r2: 0.01, p: 0.9, p_rank: 0.9 }, { pc: 3, r2: 0.0, p: 0.99, p_rank: 0.99 }] },
      { covariate: "Donor", kind: "categorical", same_as: [], test: "ANOVA", pcs: [
        { pc: 1, r2: null, p: null, p_rank: null }, { pc: 2, r2: 0.93, p: 0.0006, p_rank: 0.01 }, { pc: 3, r2: 0.7, p: 0.07, p_rank: 0.1 }] },
      { covariate: "n_cells", kind: "numeric", same_as: [], test: "Spearman", pcs: [
        { pc: 1, r2: 0.03, p: 0.5, p_rank: null, rho: 0.17 }, { pc: 2, r2: 0.55, p: 0.001, p_rank: null, rho: 0.74 }, { pc: 3, r2: 0.0, p: 0.8, p_rank: null }] },
    ],
    drivers: { PC1: { covariate: "stim", r2: 0.98, p: 2.6e-13 }, PC2: { covariate: "Donor", r2: 0.93, p: 0.0006 } },
    loadings: { PC1: { positive: ["IFIT1", "CXCL10", "ISG15", "MX1"], negative: ["IL1B", "VCAN", "CLEC5A", "MMP9"] } },
    outliers: [{ id: "D2 | CTRL", distance: 5.1, cutoff: 4.03, pc: 3, few_cells: true,
      reason: "D2 | CTRL (90 cells) sits 5.1 SDs from the others (cut-off 4.0), mostly along PC3." }],
    outlier_rule: "Distance on PC1-3: leave one out.", counts_source: "layers['counts']",
    notes: ["1 profile(s) from fewer than 10 cells were left out: their sums are mostly noise."], message: null,
    ...extra,
  };
}
