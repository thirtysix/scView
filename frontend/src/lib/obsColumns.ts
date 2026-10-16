/** Obs column helpers shared by the scatter's pickers, tooltip and labels. */

/** A column the scatter colours by category (codes), not by value. */
export function isCategoricalDtype(dtype: string | undefined): boolean {
  return dtype === "category" || dtype === "object" || dtype === "bool";
}
