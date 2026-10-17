/**
 * The params a Data Assessment run sends: each step's own, flattened into one
 * PipelineParams dict, plus the few params that steer more steps than the one
 * they are set on.
 */

export type StepParams = Record<string, Record<string, unknown>>;

/** Param -> the steps that act on it (it is set on the first one's card). */
export const SHARED_PARAMS: Record<string, readonly string[]> = {
  // Chosen on Normalization; with Pearson residuals the variable-gene and scaling
  // steps run differently, so a run of either alone must still carry the choice.
  normalization_method: ["normalization", "highly_variable_genes", "scaling"],
};

export const NORMALIZATION_LABELS: Record<string, string> = {
  log: "Shifted log (default)",
  pearson_residuals: "Analytic Pearson residuals",
};

export const HVG_FLAVOR_OPTION_LABELS: Record<string, string> = {
  seurat: "Seurat",
  cell_ranger: "Cell Ranger",
  seurat_v3: "Seurat v3 (ranks raw counts)",
};

/** Flattened params for running `steps`, or null when there are none. */
export function flattenStepParams(steps: string[], stepParams: StepParams): Record<string, unknown> | null {
  const flat: Record<string, unknown> = {};
  for (const step of steps) {
    const sp = stepParams[step];
    if (sp) Object.assign(flat, sp);
  }
  for (const [param, users] of Object.entries(SHARED_PARAMS)) {
    if (param in flat || !steps.some((s) => users.includes(s))) continue;
    const home = users[0]!;
    const value = stepParams[home]?.[param];
    if (value !== undefined) flat[param] = value;
  }
  return Object.keys(flat).length > 0 ? flat : null;
}

/** The normalisation method in force: the user's choice, else the dataset's recorded one. */
export function normalizationMethod(stepParams: StepParams, recorded?: string | null): string {
  const chosen = stepParams.normalization?.normalization_method;
  return typeof chosen === "string" ? chosen : recorded || "log";
}
