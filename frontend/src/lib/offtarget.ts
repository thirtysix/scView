import type { HpaSide, OffTargetGene, OffTargetResult, OffTargetVerdict, TissueLevel } from "@/api/offtarget";

/** Display helpers for the "Elsewhere in the body" panel (components/unified/OffTargetPanel.tsx). */

export const VERDICT_STYLE: Record<OffTargetVerdict, string> = {
  restricted: "bg-emerald-50 text-emerald-800",
  enhanced: "bg-sky-50 text-sky-800",
  broad: "bg-orange-50 text-orange-800",
  not_detected: "bg-slate-100 text-slate-600",
  not_in_hpa: "bg-slate-100 text-slate-500",
};

/** The tissue-level menu: label, and the unit for the heatmap caption. */
export const TISSUE_LEVELS: { value: TissueLevel; label: string; unit: string }[] = [
  { value: "tissue", label: "Tissues (brain as one)", unit: "tissues" },
  { value: "region", label: "Tissues + brain regions", unit: "tissues" },
  { value: "organ", label: "Organ groups", unit: "organ groups" },
];

/** "liver 5,000, kidney 6" (the top tissues or cell types), or a dash. */
export function topText(side: HpaSide | null, k = 3): string {
  if (!side || side.top.length === 0) return "—";
  return side.top
    .slice(0, k)
    .map((t) => `${t.name} ${t.value.toLocaleString(undefined, { maximumFractionDigits: t.value < 10 ? 1 : 0 })}`)
    .join(", ");
}

/** The HPA category of both tables, for a tooltip. */
export function categoryTitle(g: OffTargetGene): string {
  if (!g.found) return "Not in the Human Protein Atlas under this symbol";
  const side = (name: string, s: HpaSide | null) =>
    s ? `${name}: ${s.category_text}${s.elevated.length ? ` (${s.elevated.slice(0, 5).join(", ")})` : ""}` +
      (s.tau != null ? `, tau ${s.tau.toFixed(2)}` : "")
      : `${name}: not in this table`;
  return `${side("Tissues", g.tissue)}. ${side("Cell types", g.cell_type)}.`;
}

/** One line summing up a result: "6 of 20 markers broadly expressed; 3 specific here but broad in HPA". */
export function summaryText(r: OffTargetResult): string {
  const { counts, n_genes, n_flagged } = r.summary;
  if (n_genes === 0) return "This group has no markers that pass the marker filters";
  const parts = [
    `${counts.restricted} of ${n_genes} restricted`,
    `${counts.enhanced} enhanced`,
    `${counts.broad} broad`,
  ];
  if (counts.not_in_hpa) parts.push(`${counts.not_in_hpa} not in HPA`);
  const flag = n_flagged ? `; ${n_flagged} specific here but broad in the body` : "";
  return parts.join(", ") + flag;
}

/** The heatmap's height: rows of ~16 px plus room for the tissue names. */
export function heatmapHeight(nGenes: number): number {
  return Math.max(160, 110 + 16 * nGenes);
}
