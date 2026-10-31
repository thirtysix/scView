/**
 * Spatial co-occurrence (backend core/spatial_cooccurrence.py): the card's words and the
 * slices of the score array its plots draw.
 */
import type { CoOccurrencePlan, CoOccurrenceResult } from "@/api/spatialDomains";
import type { CsvCell } from "@/lib/csv";

/** The distance axis's unit as shown: "µm" or "position units". */
export function unitText(unit: string | undefined): string {
  return unit === "um" ? "µm" : (unit ?? "position units");
}

/** What the distances are measured in, and where that came from. */
export function distanceSource(r: Pick<CoOccurrenceResult, "distance_unit" | "um_per_unit" | "distance_source">): string {
  if (r.distance_unit !== "um" || r.um_per_unit == null) {
    return "Distances are in the positions' own units: the pixel size is not known here.";
  }
  if (r.um_per_unit === 1) return "Distances are in microns.";
  const from = r.distance_source ? `, from ${r.distance_source.replace(/\bum\b/g, "µm")}` : "";
  return `Distances are in microns, converted from full-resolution pixels (1 pixel = ${r.um_per_unit.toPrecision(3)} µm${from}).`;
}

/** Scores of `label` against every label, one line each, over the distances. */
export function curvesOf(
  result: CoOccurrenceResult,
  label: string,
): { label: string; distances: number[]; scores: number[]; pairs: number[] }[] {
  const labels = result.labels ?? [];
  const i = labels.indexOf(label);
  if (i < 0) return [];
  const row = result.occ?.[i] ?? [];
  const counts = result.count?.[i] ?? [];
  return labels.map((other, j) => ({
    label: other,
    distances: result.distances ?? [],
    scores: row[j] ?? [],
    pairs: counts[j] ?? [],
  }));
}

/** Labels x labels at one distance; null where a pair of labels has no pair of observations (squidpy writes 0). */
export function matrixAt(result: CoOccurrenceResult, t: number): (number | null)[][] {
  const occ = result.occ ?? [];
  const count = result.count ?? [];
  return occ.map((row, i) => row.map((scores, j) => ((count[i]?.[j]?.[t] ?? 0) > 0 ? (scores[t] ?? null) : null)));
}

/**
 * The heatmap's colours: a score is a ratio, so from 0 to the largest score (at least 2),
 * blue below 1, the neutral colour at 1, red above.
 */
export function scoreScale(matrix: (number | null)[][]): { zmin: number; zmax: number; colorscale: [number, string][] } {
  let top = 2;
  for (const row of matrix) {
    for (const v of row) if (v != null && v > top) top = v;
  }
  return {
    zmin: 0,
    zmax: top,
    colorscale: [
      [0, "#2166ac"],
      [1 / top, "#f7f7f7"],
      [1, "#b2182b"],
    ],
  };
}

/**
 * The shortest distance where more than the diagonal has at least 30 pairs of observations
 * (backend `_top_pairs`, the co-pilot's fact): at a few cells' range the scores are a few
 * pairs each. The last distance when none has.
 */
export function firstUsable(result: CoOccurrenceResult): number {
  const count = result.count ?? [];
  const labels = result.labels?.length ?? 0;
  const n = result.distances?.length ?? 0;
  for (let t = 0; t < n; t++) {
    let enough = 0;
    for (const row of count) for (const cell of row) if ((cell[t] ?? 0) >= 30) enough++;
    if (enough > labels) return t;
  }
  return Math.max(0, n - 1);
}

/** Every score, long: one row per pair of labels and distance. */
export function csvRows(result: CoOccurrenceResult): CsvCell[][] {
  const labels = result.labels ?? [];
  const distances = result.distances ?? [];
  const rows: CsvCell[][] = [];
  labels.forEach((a, i) => {
    labels.forEach((b, j) => {
      distances.forEach((d, t) => {
        rows.push([a, b, d, result.occ?.[i]?.[j]?.[t] ?? null, result.count?.[i]?.[j]?.[t] ?? null]);
      });
    });
  });
  return rows;
}

export function csvHeaders(result: CoOccurrenceResult): string[] {
  return ["label", "other_label", `distance_${result.distance_unit === "um" ? "um" : "units"}`, "score", "ordered_pairs"];
}

/** The largest default distance shown, for the "up to" field's hint. */
export function reachOf(plan: CoOccurrencePlan): number {
  const top = plan.thresholds.reduce((a, b) => Math.max(a, b), 0);
  return top * (plan.um_per_unit ?? 1);
}
