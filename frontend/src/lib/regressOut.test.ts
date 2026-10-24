import { describe, it, expect } from "vitest";
import { regressLabel, regressOptions, regressOutText } from "./regressOut";
import { paramText } from "./provenanceParams";
import { flattenStepParams } from "./pipelineParams";

describe("regress-out", () => {
  it("offers the standard covariates, then the dataset's other numeric columns", () => {
    const keys = regressOptions([
      { name: "total_counts", dtype: "float32", n_unique: 900 },
      { name: "n_genes_by_counts", dtype: "int64", n_unique: 800 },
      { name: "doublet_score", dtype: "float64", n_unique: 900 },
      { name: "cell_type", dtype: "category", n_unique: 9 },
      { name: "constant", dtype: "float64", n_unique: 1 },
    ]).map((o) => o.key);
    expect(keys).toEqual(["total_counts", "pct_counts_mt", "S_score", "G2M_score", "cc_difference",
      "n_genes_by_counts"]);
    expect(regressLabel("cc_difference")).toBe("S − G2M (cell-cycle difference)");
    expect(regressLabel("drive")).toBe("drive");
  });

  it("the History chip says what was regressed out", () => {
    expect(paramText("regress_out", ["total_counts", "cc_difference"])).toBe(
      "regressed out: Total counts, S − G2M (cell-cycle difference)");
    expect(regressOutText([])).toBe("regress-out: off");
  });

  it("is sent with the scaling step, and not with Pearson residuals", () => {
    const params = { scaling: { regress_out: ["total_counts"] } };
    expect(flattenStepParams(["scaling", "pca"], params)).toEqual({ regress_out: ["total_counts"] });
    expect(flattenStepParams(["normalization", "scaling"], {
      ...params, normalization: { normalization_method: "pearson_residuals" },
    })).toEqual({ normalization_method: "pearson_residuals" });
  });
});
