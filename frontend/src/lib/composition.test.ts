import { describe, it, expect } from "vitest";
import type { CompositionResult } from "@/api/composition";
import type { ObsColumnInfo } from "@/api/types";
import {
  compositionCsv,
  conditionColumns,
  describeTest,
  formatRatio,
  isShifted,
  sortedGroups,
  suggestConditionColumn,
} from "./composition";

const col = (name: string, n_unique: number, dtype = "category"): ObsColumnInfo => ({ name, dtype, n_unique });

// Kang: stim CTRL/STIM, Donor 8, cluster 13 cell types (logit, paired).
const KANG: CompositionResult = {
  condition_key: "stim",
  sample_key: "Donor",
  sample_key_detected: true,
  group_key: "cluster",
  conditions: ["CTRL", "STIM"],
  reference: "CTRL",
  transform: "logit",
  robust: true,
  n_outliers: 0,
  tested: true,
  design: "paired",
  test: "moderated t",
  df_residual: 7,
  df_prior: 2.061,
  n_samples: { CTRL: 8, STIM: 8 },
  n_cells: { CTRL: 6573, STIM: 7263 },
  n_significant: 1,
  fdr_threshold: 0.05,
  n_significant_cell_level: 4,
  groups: [
    { group: "B", props: { CTRL: 0.0641, STIM: 0.0724 }, n_cells: { CTRL: 420, STIM: 530 },
      fold_change: 1.128, stat: 1.868, pval: 0.0944, fdr: 0.336, df_prior: 2.061, paired_test: null },
    { group: "CD14 Mono", props: { CTRL: 0.3496, STIM: 0.3003 }, n_cells: { CTRL: 2300, STIM: 1925 },
      fold_change: 0.859, stat: -3.817, pval: 0.00406, fdr: 0.0427, df_prior: 2.061, paired_test: null },
    { group: "Eryth", props: { CTRL: 0.0042, STIM: 0.004 }, n_cells: { CTRL: 30, STIM: 30 },
      fold_change: null, stat: null, pval: null, fdr: null, df_prior: null, paired_test: null },
  ],
  samples: [],
  warnings: [],
  notes: [],
  dropped_samples: [],
  pairing: { paired: true, n_pairs: 8, conditions: ["CTRL", "STIM"] },
};

describe("condition columns", () => {
  it("keeps categorical columns with 2-20 values and suggests a condition-like one", () => {
    const cols = [
      col("Donor", 8),
      col("orig_ident", 2),
      col("stim", 2),
      col("n_counts", 3000, "float64"),
      col("cluster", 13),
      col("barcode", 13836, "object"),
    ];
    const opts = conditionColumns(cols, ["cluster"]);
    expect(opts.map((c) => c.name)).toEqual(["Donor", "orig_ident", "stim"]);
    expect(suggestConditionColumn(opts)).toBe("stim");
  });

  it("falls back to a joint dataset's 'dataset' column, else nothing", () => {
    expect(suggestConditionColumn([col("leiden", 9), col("dataset", 2)])).toBe("dataset");
    expect(suggestConditionColumn([col("leiden", 9)])).toBe("");
    // A condition-like name with few values beats one with many.
    expect(suggestConditionColumn([col("group_id", 15), col("treatment", 3)])).toBe("treatment");
  });
});

describe("result helpers", () => {
  it("sorts tested groups by p (untested last) and flags the shifted ones", () => {
    expect(sortedGroups(KANG).map((g) => g.group)).toEqual(["CD14 Mono", "B", "Eryth"]);
    expect(KANG.groups.filter((g) => isShifted(g, KANG)).map((g) => g.group)).toEqual(["CD14 Mono"]);
    const untested = { ...KANG, tested: false };
    expect(sortedGroups(untested).map((g) => g.group)).toEqual(["CD14 Mono", "B", "Eryth"]);
    expect(untested.groups.some((g) => isShifted(g, untested))).toBe(false);
  });

  it("formats ratios without rounding small ones to zero", () => {
    expect(formatRatio(1.2057)).toBe("×1.21");
    expect(formatRatio(275.789)).toBe("×275.79");
    expect(formatRatio(0.0026)).toBe("×0.0026");
    expect(formatRatio(0)).toBe("×0.00");
    expect(formatRatio(null)).toBe("–");
  });

  it("describes the test in words", () => {
    const d = describeTest(KANG);
    expect(d).toContain("Moderated t-test on logit-transformed per-sample shares");
    expect(d).toContain("paired by Donor (CTRL 8, STIM 8 samples)");
    expect(d).toContain("prior df 2.1");
    expect(describeTest({ ...KANG, df_prior: null })).toContain("fully pooled");
    expect(d).toContain("prior df 2.1; robust)");
    expect(describeTest({ ...KANG, n_outliers: 2 })).toContain("robust: 2 outlying groups shrunk less");
    expect(describeTest({ ...KANG, robust: false })).not.toContain("robust");
    expect(describeTest({ ...KANG, tested: false })).toMatch(/not tested/);
  });

  it("exports one CSV row per group with shares in percent", () => {
    const { headers, rows } = compositionCsv(KANG);
    expect(headers).toEqual([
      "cluster", "pct_CTRL", "pct_STIM", "fold_change", "log2_fold_change", "t", "pval", "fdr",
      "df_prior", "n_cells_CTRL", "n_cells_STIM", "n_samples_CTRL", "n_samples_STIM",
    ]);
    expect(rows[0]![0]).toBe("CD14 Mono");
    expect(rows[0]![1]).toBeCloseTo(34.96, 6);
    expect(rows[0]![4]).toBeCloseTo(Math.log2(0.859), 5);
    expect(rows[0]!.slice(-4)).toEqual([2300, 1925, 8, 8]);
    expect(rows[2]!.slice(3, 9)).toEqual(["", "", "", "", "", ""]); // missing values stay empty
    expect(rows[0]![8]).toBe(2.061);
    expect(compositionCsv({ ...KANG, groups: [{ ...KANG.groups[1]!, df_prior: null }] }).rows[0]![8]).toBe("Inf");
    const three = compositionCsv({ ...KANG, conditions: ["A", "B", "C"], test: "moderated F" });
    expect(three.headers).toContain("F");
    expect(three.headers).not.toContain("fold_change");
  });
});
