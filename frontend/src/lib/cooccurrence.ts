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

/**
 * The backend's floor (core/spatial_cooccurrence.MIN_PAIRS): a score from fewer pairs of
 * observations at a distance is too few to read. null: no floor (a run saved without counts).
 */
export function minPairs(result: CoOccurrenceResult): number | null {
  return result.count && result.min_pairs != null ? result.min_pairs : null;
}

/** Whether a score rests on some pairs, but fewer than the floor. */
export function tooFew(pairs: number | undefined, floor: number | null): boolean {
  return floor != null && (pairs ?? 0) > 0 && (pairs ?? 0) < floor;
}

/** Scores of `label` against every label, one line each, over the distances; null (a gap) where too few pairs. */
export function curvesOf(
  result: CoOccurrenceResult,
  label: string,
): { label: string; distances: number[]; scores: (number | null)[]; pairs: number[] }[] {
  const labels = result.labels ?? [];
  const i = labels.indexOf(label);
  if (i < 0) return [];
  const row = result.occ?.[i] ?? [];
  const counts = result.count?.[i] ?? [];
  const floor = minPairs(result);
  return labels.map((other, j) => ({
    label: other,
    distances: result.distances ?? [],
    scores: (row[j] ?? []).map((v, t) => (tooFew(counts[j]?.[t], floor) ? null : v)),
    pairs: counts[j] ?? [],
  }));
}

/** How many points the curves of `label` leave out for too few pairs. */
export function curvePointsHidden(result: CoOccurrenceResult, label: string): number {
  const i = (result.labels ?? []).indexOf(label);
  const floor = minPairs(result);
  if (i < 0 || floor == null) return 0;
  return (result.count?.[i] ?? []).reduce((n, pairs) => n + pairs.filter((p) => tooFew(p, floor)).length, 0);
}

/**
 * Labels x labels at one distance; null where a pair of labels has no pair of observations
 * (squidpy writes 0) or too few to read (`tooFewAt` greys those).
 */
export function matrixAt(result: CoOccurrenceResult, t: number): (number | null)[][] {
  const occ = result.occ ?? [];
  const count = result.count;
  const floor = minPairs(result);
  return occ.map((row, i) =>
    row.map((scores, j) => {
      const pairs = count ? (count[i]?.[j]?.[t] ?? 0) : 1; // no counts saved: every score as it is
      return pairs > 0 && !tooFew(pairs, floor) ? (scores[t] ?? null) : null;
    }),
  );
}

/** The pair counts of the cells too few to read at one distance (null elsewhere): the heatmap's grey layer. */
export function tooFewAt(result: CoOccurrenceResult, t: number): (number | null)[][] {
  const count = result.count ?? [];
  const floor = minPairs(result);
  return (result.occ ?? []).map((row, i) =>
    row.map((_s, j) => {
      const pairs = count[i]?.[j]?.[t];
      return tooFew(pairs, floor) ? (pairs ?? null) : null;
    }),
  );
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
 * The shortest distance where more than the diagonal has at least the floor's pairs of
 * observations (backend `first_readable`, the co-pilot's fact; 30 for a result that does not
 * give its floor): at a few cells' range the scores are a few pairs each. The last distance
 * when none has.
 */
export function firstUsable(result: CoOccurrenceResult): number {
  const count = result.count ?? [];
  const labels = result.labels?.length ?? 0;
  const n = result.distances?.length ?? 0;
  const floor = result.min_pairs ?? 30;
  for (let t = 0; t < n; t++) {
    let enough = 0;
    for (const row of count) for (const cell of row) if ((cell[t] ?? 0) >= floor) enough++;
    if (enough > labels) return t;
  }
  return Math.max(0, n - 1);
}

/** Every score, long: one row per pair of labels and distance, with whether it is from too few pairs. */
export function csvRows(result: CoOccurrenceResult): CsvCell[][] {
  const labels = result.labels ?? [];
  const distances = result.distances ?? [];
  const floor = minPairs(result);
  const rows: CsvCell[][] = [];
  labels.forEach((a, i) => {
    labels.forEach((b, j) => {
      distances.forEach((d, t) => {
        const pairs = result.count?.[i]?.[j]?.[t];
        rows.push([a, b, d, result.occ?.[i]?.[j]?.[t] ?? null, pairs ?? null, floor == null ? null : String(tooFew(pairs, floor))]);
      });
    });
  });
  return rows;
}

export function csvHeaders(result: CoOccurrenceResult): string[] {
  return [
    "label",
    "other_label",
    `distance_${result.distance_unit === "um" ? "um" : "units"}`,
    "score",
    "ordered_pairs",
    "too_few_pairs",
  ];
}

/** The largest default distance shown, for the "up to" field's hint. */
export function reachOf(plan: CoOccurrencePlan): number {
  const top = plan.thresholds.reduce((a, b) => Math.max(a, b), 0);
  return top * (plan.um_per_unit ?? 1);
}
