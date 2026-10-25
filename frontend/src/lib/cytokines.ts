/**
 * Helpers for the cytokine-response view: the heatmap's rows and colour range, and
 * the plain caption of what was compared.
 */
import type { CytokineResult } from "@/api/cytokines";

/** Heatmap data: cytokines as rows (highest max |z| on top), groups as columns, a range symmetric about 0. */
export function cytokineHeatmap(r: CytokineResult, n: number = r.cytokines.length) {
  if (!r.groups.length) return null;
  const strength = r.cytokines.map((c, j) => ({
    c,
    j,
    s: Math.max(0, ...r.z.map((row) => Math.abs(row[j] ?? 0))),
  }));
  const rows = strength.sort((a, b) => b.s - a.s || a.c.localeCompare(b.c)).slice(0, n);
  const z = rows.map(({ j }) => r.z.map((row) => row[j] ?? 0));
  const zmax = Math.max(1e-9, ...z.flat().map(Math.abs));
  // Plotly draws the first row at the bottom: reverse so the strongest is on top.
  return { x: r.groups, y: rows.map((x) => x.c).reverse(), z: z.reverse(), zmax };
}

/** "STIM vs CTRL within each cell_type group", "each leiden group vs the rest". */
export function comparisonText(r: CytokineResult): string {
  if (r.mode === "conditions") {
    return `${r.a} vs ${r.b}${r.groupby ? ` within each ${r.groupby} group` : ", all cells"}`;
  }
  return `each ${r.groupby} group vs ${r.reference === "rest" ? "the rest" : "all cells"}`;
}
