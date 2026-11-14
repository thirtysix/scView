import type { Data } from "plotly.js";
import type { DESettings } from "@/stores/deSettingsStore";
import type {
  KoShareRow,
  MixscapeDeGenes,
  MixscapeResult,
  MixscapeTarget,
  PosteriorHist,
  ProteinCheck,
  ProteinCheckRow,
} from "@/api/perturbation";

/** obs columns Mixscape writes (backend core/mixscape.py). */
export const MIXSCAPE_CLASS = "mixscape_class";
export const MIXSCAPE_GLOBAL = "mixscape_class_global";
export const MIXSCAPE_P_KO = "mixscape_p_ko";

export type TargetSortKey = "target" | "n_cells" | "ko_fraction" | "n_de_genes";

/** The per-target table in the chosen order (ties by target name). */
export function sortTargets(rows: MixscapeTarget[], key: TargetSortKey, desc: boolean): MixscapeTarget[] {
  const out = [...rows];
  out.sort((a, b) => {
    const d = key === "target" ? a.target.localeCompare(b.target) : (a[key] as number) - (b[key] as number);
    return (desc ? -d : d) || a.target.localeCompare(b.target);
  });
  return out;
}

export const pctText = (v: number): string => `${Math.round(v * 100)}%`;

/** Stacked bars of the posterior P(KO) over a target's cells, split by the call. */
export function posteriorTraces(hist: PosteriorHist): Data[] {
  const mids = hist.edges.slice(0, -1).map((e, i) => (e + (hist.edges[i + 1] ?? e)) / 2);
  const width = (hist.edges[1] ?? 1) - (hist.edges[0] ?? 0);
  return [
    { type: "bar", x: mids, y: hist.np, width, name: "NP", marker: { color: "#94a3b8" },
      hovertemplate: "P(KO) %{x:.2f}: %{y} cells (NP)<extra></extra>" },
    { type: "bar", x: mids, y: hist.ko, width, name: "KO", marker: { color: "#dc2626" },
      hovertemplate: "P(KO) %{x:.2f}: %{y} cells (KO)<extra></extra>" },
  ];
}

/** The LDA view: one trace per label, the control in grey. */
export function ldaTraces(lda: NonNullable<MixscapeResult["lda"]>): Data[] {
  const groups = new Map<string, number[]>();
  lda.label.forEach((l, i) => {
    const g = groups.get(l) ?? [];
    g.push(i);
    groups.set(l, g);
  });
  return [...groups.entries()]
    .sort(([a], [b]) => (a === "NT" ? -1 : b === "NT" ? 1 : a.localeCompare(b)))
    .map(([label, idx]) => ({
      type: "scattergl",
      mode: "markers",
      name: label === "NT" ? "NT (control)" : `${label} KO`,
      x: idx.map((i) => lda.x[i] ?? 0),
      y: idx.map((i) => lda.y[i] ?? 0),
      marker: { size: 3, opacity: 0.7, ...(label === "NT" ? { color: "#cbd5e1" } : {}) },
      hovertemplate: `${label}<extra></extra>`,
    }));
}

/** The DE subtab's settings for "this target's KO cells vs the controls". */
export function deHandoff(target: string): Partial<DESettings> {
  return {
    mode: "groups",
    splitBy: MIXSCAPE_CLASS,
    groupA: `${target} KO`,
    groupB: "NT",
    withinCol: "",
    withinVal: "",
  };
}

/** One target's rows of a KO-share table (per guide or per replicate), as [value, row]. */
export function koRowsFor(rows: KoShareRow[] | undefined, target: string, key: "guide" | "replicate"): [string, KoShareRow][] {
  return (rows ?? []).filter((r) => r.target === target).map((r) => [String(r[key] ?? ""), r]);
}

/** A run's DE gene choice: "all" (Seurat's rule) unless it ran on the variable genes;
 *  runs before 0.78 have no choice recorded and used the variable genes. */
export function deGenesOf(params: Record<string, unknown> | undefined): MixscapeDeGenes {
  if (!params) return "all";
  return params.de_genes === "all" ? "all" : "variable";
}

/** Why the preview's target count leaves some mapped targets out: their guides are in
 *  the guide list but no cell carries one at these settings (a vector guide, pilot guides). */
export function unusedTargetsText(unused: { target: string; guides: string[] }[]): string {
  if (!unused.length) return "";
  const names = unused.slice(0, 6).map((u) => u.target).join(", ") + (unused.length > 6 ? ", …" : "");
  const n = unused.length;
  return ` Not counted: ${n} more target${n > 1 ? "s" : ""} (${names}) whose guides no cell has at these settings.`;
}

/** Guide -> target entries the user changed from the automatic mapping. */
export function mappingChanges(mapping: Record<string, string>, auto: Record<string, string>): Record<string, string> {
  const out: Record<string, string> = {};
  for (const [g, t] of Object.entries(mapping)) {
    const v = t.trim();
    if (v && auto[g] !== v) out[g] = v;
  }
  return out;
}

/** One protein's rows of the protein check, by its shift in all guide cells (most negative
 *  first; targets with too few cells last). */
export function proteinRowsFor(check: ProteinCheck | null | undefined, protein: string): ProteinCheckRow[] {
  const rows = (check?.rows ?? []).filter((r) => r.protein === protein);
  return rows.sort((a, b) => (a.all?.diff ?? Infinity) - (b.all?.diff ?? Infinity) || a.target.localeCompare(b.target));
}

/** A signed shift as text: "+0.21", "-0.19" ("–" without one). */
export const shiftText = (v: number | null | undefined): string =>
  v == null || !Number.isFinite(v) ? "–" : `${v > 0 ? "+" : ""}${v.toFixed(2)}`;

/** The replicate agreement in words: "3/3" (empty without replicates). */
export const replicateText = (r: ProteinCheckRow): string =>
  r.n_replicates ? `${r.same_sign ?? 0}/${r.n_replicates}` : "";

const FLAG = "#d97706";
const SIG = "#2563eb"; // not a neutral: the dark theme flips only one-colour markers (lib/theme.ts)
const NOT_SIG = "#94a3b8";

/** The dot plot of one protein's shift per target: all guide cells (blue when the shift
 *  counts, amber when it has no RNA KO class, open when it does not count), KO and NP
 *  cells, and each replicate's shift. Targets in the order of ``rows`` (layout
 *  ``categoryarray``). */
export function proteinShiftTraces(rows: ProteinCheckRow[], protein: string): Data[] {
  const x = rows.map((r) => r.target);
  const group = (key: "ko" | "np", name: string, color: string): Data => ({
    type: "scatter",
    mode: "markers",
    name,
    x,
    y: rows.map((r) => r[key]?.diff ?? null),
    marker: { color, size: 6, symbol: key === "ko" ? "triangle-up" : "diamond" },
    customdata: rows.map((r) => [r[key]?.n ?? 0]),
    hovertemplate: `%{x} ${name}: ${protein} %{y:+.2f} (%{customdata[0]} cells)<extra></extra>`,
  });
  const reps = rows.flatMap((r) => (r.replicates ?? []).map((p) => ({ t: r.target, ...p })));
  const traces: Data[] = [
    group("ko", "KO cells", "#dc2626"),
    group("np", "NP cells", "#64748b"),
  ];
  if (reps.length)
    traces.push({
      type: "scatter",
      mode: "markers",
      name: "Each replicate (all cells)",
      x: reps.map((p) => p.t),
      y: reps.map((p) => p.diff),
      marker: { color: "#64748b", size: 4, symbol: "circle-open" },
      customdata: reps.map((p) => [p.replicate, p.n]),
      hovertemplate: `%{x}, replicate %{customdata[0]}: %{y:+.2f} (%{customdata[1]} cells)<extra></extra>`,
    });
  // All guide cells, one trace per kind so the legend says what each colour means.
  const kinds: [string, (r: ProteinCheckRow) => boolean, string, string][] = [
    ["All cells: shift counts", (r) => r.significant && !r.flag, SIG, "circle"],
    ["All cells: no RNA KO class", (r) => r.flag, FLAG, "circle"],
    ["All cells: does not count", (r) => !r.significant, NOT_SIG, "circle-open"],
  ];
  for (const [name, pick, color, symbol] of kinds) {
    const sel = rows.filter(pick);
    if (!sel.length) continue;
    traces.push({
      type: "scatter",
      mode: "markers",
      name,
      x: sel.map((r) => r.target),
      y: sel.map((r) => r.all?.diff ?? null),
      marker: { size: 9, color, symbol, line: { width: 1.5, color } },
      customdata: sel.map((r) => [r.all?.n ?? 0, pctText(r.ko_fraction)]),
      hovertemplate: `%{x}: ${protein} %{y:+.2f} in all %{customdata[0]} guide cells, %{customdata[1]} KO<extra>${name}</extra>`,
    });
  }
  return traces;
}
