import { describe, expect, it } from "vitest";
import { flattenStepParams, normalizationMethod } from "./pipelineParams";

describe("flattenStepParams", () => {
  const stepParams = {
    normalization: { target_sum: 5000, normalization_method: "pearson_residuals" },
    highly_variable_genes: { n_top_genes: 3000 },
    clustering: { clustering_resolution: 1 },
  };

  it("merges the params of the steps being run only", () => {
    expect(flattenStepParams(["normalization", "clustering"], stepParams)).toEqual({
      target_sum: 5000,
      normalization_method: "pearson_residuals",
      clustering_resolution: 1,
    });
  });

  it("carries the normalisation method to a run of the variable-gene or scaling step alone", () => {
    expect(flattenStepParams(["highly_variable_genes"], stepParams)).toEqual({
      n_top_genes: 3000,
      normalization_method: "pearson_residuals",
    });
    expect(flattenStepParams(["scaling"], stepParams)).toEqual({
      normalization_method: "pearson_residuals",
    });
  });

  it("leaves it out of steps it does not steer, and returns null for nothing", () => {
    expect(flattenStepParams(["clustering"], stepParams)).toEqual({ clustering_resolution: 1 });
    expect(flattenStepParams(["pca"], stepParams)).toBeNull();
    expect(flattenStepParams(["scaling"], {})).toBeNull();
  });
});

describe("normalizationMethod", () => {
  it("prefers the user's choice, then the recorded method, then the shifted log", () => {
    expect(normalizationMethod({ normalization: { normalization_method: "log" } }, "pearson_residuals")).toBe("log");
    expect(normalizationMethod({}, "pearson_residuals")).toBe("pearson_residuals");
    expect(normalizationMethod({}, null)).toBe("log");
  });
});
