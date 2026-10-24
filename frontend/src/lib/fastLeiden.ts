/**
 * Fast Leiden: the pipeline's `leiden_iterations` (backend core/leiden.py). -1 repeats
 * Leiden's passes until one improves nothing (scanpy's default); 2 stops after two.
 */

export const FULL_LEIDEN = -1;
export const FAST_LEIDEN = 2;

/** The clustering step's select: value -> label. */
export const LEIDEN_ITERATION_LABELS: Record<string, string> = {
  "-1": "Full optimisation (default)",
  "2": "Fast Leiden (2 passes)",
};

/** What choosing fast Leiden means, for its tooltips (measured on Roihu, 2026-10-23). */
export const FAST_LEIDEN_POINTS: readonly string[] = [
  "The same Leiden algorithm with the same seed, stopped after two passes instead of repeating them until a pass improves nothing.",
  "On a mouse brain dataset (89,000 to 1.2 million cells) its clusters agreed with the full optimisation's at ARI 0.93-0.97, with the same number of clusters: more closely than two full runs with different seeds agree with each other (ARI 0.46-0.60).",
  "Its modularity score (the quality Leiden maximises) comes out slightly lower: 0.1-0.35 %.",
  "17 s instead of 4 min at 89,000 cells; 7 min instead of an estimated 18 h at 1.2 million.",
  "Recorded in the History and the replay script (sc.tl.leiden(..., n_iterations=2)); the resolution sweep, the variable-gene and merge checks and sub-clustering of this dataset follow it.",
];

/** A recorded `leiden_iterations` in words, for the History chip. */
export function leidenIterationsText(value: unknown): string {
  const n = Number(value);
  if (n === FULL_LEIDEN) return "Leiden: full optimisation";
  if (Number.isInteger(n) && n > 0) return n === FAST_LEIDEN ? "Leiden: fast (2 passes)" : `Leiden: ${n} passes`;
  return `leiden_iterations=${String(value)}`;
}
