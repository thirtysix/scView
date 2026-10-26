import type { Data } from "plotly.js";
import type { DESettings } from "@/stores/deSettingsStore";
import type { MixscapeResult, MixscapeTarget, PosteriorHist } from "@/api/perturbation";

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

/** Guide -> target entries the user changed from the automatic mapping. */
export function mappingChanges(mapping: Record<string, string>, auto: Record<string, string>): Record<string, string> {
  const out: Record<string, string> = {};
  for (const [g, t] of Object.entries(mapping)) {
    const v = t.trim();
    if (v && auto[g] !== v) out[g] = v;
  }
  return out;
}
