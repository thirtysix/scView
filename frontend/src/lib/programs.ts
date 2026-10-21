import type { GeneProgram, ProgramsByGroup, ProgramsKStats } from "@/api/programs";

/** "P3 (ISG15, IFI6, MX1)": the program and its first top genes. */
export function programLabel(p: Pick<GeneProgram, "name" | "top_genes">, nGenes = 3): string {
  const genes = p.top_genes.slice(0, nGenes).map((g) => g.gene);
  return genes.length ? `${p.name} (${genes.join(", ")})` : p.name;
}

/** The overlay's legend label for a program's usage. */
export function usageLabel(name: string, topGenes: string[]): string {
  const genes = topGenes.slice(0, 3).join(", ");
  return genes ? `${name} usage share (${genes})` : `${name} usage share`;
}

/** How stable a program is, in words (the row's tooltip). */
export function stabilityText(p: GeneProgram, nSeeds: number, threshold: number): string {
  if (nSeeds < 2) return "One run: stability needs at least 2 seeds.";
  const r = p.correlation === null ? "n/a" : p.correlation.toFixed(2);
  const sil = p.silhouette === null ? "n/a" : p.silhouette.toFixed(2);
  const verdict = p.stable
    ? "stable"
    : p.runs_found < nSeeds
      ? `not stable: missing from ${nSeeds - p.runs_found} of the ${nSeeds} runs`
      : `not stable: its runs' spectra correlate below ${threshold}`;
  return (
    `Found in ${p.runs_found} of ${nSeeds} runs (${p.n_members} spectra grouped into it); ` +
    `mean correlation between them ${r}; silhouette ${sil}: ${verdict}.`
  );
}

/** The badge under a program's name: "stable", the runs it was found in, or its
 *  runs' correlation when every run found it but they disagree. */
export function stabilityBadge(p: GeneProgram, nSeeds: number): string {
  if (nSeeds < 2) return "1 run";
  if (p.stable) return "stable";
  if (p.runs_found < nSeeds) return `${p.runs_found}/${nSeeds} runs`;
  return p.correlation === null ? "unstable" : `r = ${p.correlation.toFixed(2)}`;
}

export interface ProgramHeatmap {
  x: string[];
  y: string[];
  z: number[][];
  zmax: number;
  /** The program index (1-based) of each row, in row order. */
  index: number[];
}

/** Programs (rows, P1 at the top) x groups (columns): mean usage share. */
export function heatmapData(byGroup: ProgramsByGroup, programs: GeneProgram[]): ProgramHeatmap {
  const rows = programs.map((p) => ({ label: programLabel(p, 2), index: p.index, z: byGroup.mean[p.index - 1] ?? [] }));
  const z = rows.map((r) => r.z.map((v) => (Number.isFinite(v) ? v : 0)));
  const zmax = Math.max(1e-6, ...z.flat());
  return { x: byGroup.groups, y: rows.map((r) => r.label), z, zmax, index: rows.map((r) => r.index) };
}

/** The group where each program's mean usage is highest, with that mean. */
export function topGroup(byGroup: ProgramsByGroup, index: number): { group: string; mean: number } | null {
  const row = byGroup.mean[index - 1];
  if (!row?.length) return null;
  let best = 0;
  for (let i = 1; i < row.length; i++) if (row[i]! > row[best]!) best = i;
  return { group: byGroup.groups[best]!, mean: row[best]! };
}

/** "HALLMARK_INTERFERON_ALPHA_RESPONSE" -> "Interferon alpha response". */
export function prettyTerm(term: string): string {
  const t = term.replace(/^HALLMARK_/, "").replace(/_/g, " ").toLowerCase();
  return t.charAt(0).toUpperCase() + t.slice(1);
}

/**
 * The sweep's k values from a text field ("5, 10, 15, 20"): whole numbers in
 * [minK, maxK], sorted, at most `max`. Returns an error sentence instead when
 * the text cannot be read.
 */
export function parseSweep(text: string, minK: number, maxK: number, max: number): number[] | string {
  const parts = text.split(/[\s,;]+/).filter(Boolean);
  const ks: number[] = [];
  for (const p of parts) {
    const v = Number(p);
    if (!Number.isInteger(v) || v < minK || v > maxK) return `Each k must be a whole number from ${minK} to ${maxK}.`;
    if (!ks.includes(v)) ks.push(v);
  }
  if (ks.length > max) return `A sweep takes at most ${max} values of k.`;
  return ks.sort((a, b) => a - b);
}

/** The sweep's chart: silhouette (left axis) and relative error (right axis) by k. */
export function sweepTraces(sweep: ProgramsKStats[]) {
  const ks = sweep.map((s) => s.k);
  return [
    {
      type: "scatter" as const,
      mode: "lines+markers" as const,
      name: "Silhouette",
      x: ks,
      y: sweep.map((s) => s.silhouette),
      line: { color: "#2563eb" },
      hovertemplate: "k = %{x}<br>silhouette %{y:.3f}<extra></extra>",
    },
    {
      type: "scatter" as const,
      mode: "lines+markers" as const,
      name: "Relative error",
      x: ks,
      y: sweep.map((s) => s.error),
      yaxis: "y2" as const,
      line: { color: "#94a3b8", dash: "dot" as const },
      hovertemplate: "k = %{x}<br>relative error %{y:.4f}<extra></extra>",
    },
  ];
}

/** The server's sentence out of an `apiFetch` error ("API error 400: {detail}"). */
export function errorText(e: unknown): string {
  const msg = e instanceof Error ? e.message : String(e);
  const body = msg.replace(/^API error \d+: /, "");
  try {
    const detail = JSON.parse(body).detail;
    if (typeof detail === "string") return detail;
  } catch {
    /* not JSON */
  }
  return msg;
}

const pct = (v: number) => `${Math.round(v * 100)}%`;


export { pct as pctText };
