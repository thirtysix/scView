import type { PpiNode, PpiResult } from "@/api/ppi";

/** The community palette: validated for colourblind separation at four hues (as
 *  the term network's); communities past the fourth stay grey and unlabelled. */
export const COMMUNITY_SERIES = ["#2a78d6", "#008300", "#e87ba4", "#eda100"];
export const COMMUNITY_OTHER = "#aab2bd";

export type PpiColourBy = "community" | "logfc";

/** "p < 0.001" when no random set reached the observed count, else "p = 0.042". */
export function ppiPText(p: number | null, nDraws: number): string {
  if (p === null) return "p n/a";
  return p <= 1 / nDraws ? `p < ${1 / nDraws}` : `p = ${p.toFixed(3)}`;
}

/** The summary line: "84 proteins, 61 interactions (22 expected by chance), p < 0.001". */
export function ppiSummary(r: PpiResult): string {
  const exp = r.expected_links < 10 ? r.expected_links.toFixed(1) : Math.round(r.expected_links).toLocaleString();
  return (
    `${r.n_proteins.toLocaleString()} proteins, ${r.n_links.toLocaleString()} interaction${r.n_links === 1 ? "" : "s"} ` +
    `(${exp} expected by chance), ${ppiPText(r.p_value, r.n_draws)}`
  );
}

/** What was left out of the list: genes past the cap and genes STRING does not name. */
export function ppiInputNote(r: PpiResult): string | null {
  const parts: string[] = [];
  if (r.truncated) parts.push(`only the first ${r.max_genes} of ${r.n_given.toLocaleString()} genes are used`);
  if (r.n_unmapped > 0) {
    const shown = r.unmapped.slice(0, 8).join(", ");
    const more = r.n_unmapped > 8 ? ` and ${r.n_unmapped - 8} more` : "";
    parts.push(`${r.n_unmapped} gene${r.n_unmapped === 1 ? "" : "s"} not in STRING (${shown}${more})`);
  }
  return parts.length ? `${parts.join("; ")}.` : null;
}

export function communityColour(community: number | null): string {
  if (community === null || community < 1 || community > COMMUNITY_SERIES.length) return COMMUNITY_OTHER;
  return COMMUNITY_SERIES[community - 1]!;
}

/** Blue (down) - light grey (0) - red (up), symmetric about 0 to +-`maxAbs`. */
export function logfcColour(lfc: number | null, maxAbs: number): string {
  if (lfc === null || !Number.isFinite(lfc)) return COMMUNITY_OTHER;
  const t = Math.max(-1, Math.min(1, maxAbs > 0 ? lfc / maxAbs : 0));
  const mid = [226, 232, 240];
  const end = t >= 0 ? [202, 0, 32] : [5, 113, 176];
  const a = Math.abs(t);
  const c = mid.map((m, i) => Math.round(m + (end[i]! - m) * a));
  return `rgb(${c[0]},${c[1]},${c[2]})`;
}

/** The largest |logFC| among the nodes (the colour scale's end), at least 0.5. */
export function maxAbsLogfc(nodes: PpiNode[]): number {
  let m = 0;
  for (const n of nodes) if (n.logfc !== null && Number.isFinite(n.logfc)) m = Math.max(m, Math.abs(n.logfc));
  return Math.max(m, 0.5);
}

/** Node radius in px by degree: 3 px without partners, growing with sqrt(degree). */
export function nodeRadius(degree: number): number {
  return 3 + 2.2 * Math.sqrt(Math.max(0, degree));
}

export type HubSortField = "gene" | "degree" | "betweenness" | "community" | "logfc";

/** The table's rows: by the chosen field (nulls last), ties by degree then name. */
export function sortNodes(nodes: PpiNode[], field: HubSortField, dir: "asc" | "desc"): PpiNode[] {
  const sign = dir === "asc" ? 1 : -1;
  return [...nodes].sort((a, b) => {
    if (field === "gene") return sign * a.gene.localeCompare(b.gene);
    const va = a[field], vb = b[field];
    if (va === null && vb !== null) return 1;
    if (vb === null && va !== null) return -1;
    if (va !== null && vb !== null && va !== vb) return sign * (va - vb);
    return b.degree - a.degree || a.gene.localeCompare(b.gene);
  });
}

/** CSV rows of the proteins table. */
export const PPI_CSV_HEADERS = ["gene", "string_name", "string_id", "degree", "betweenness", "community", "logfc", "hub"];
export function ppiCsvRow(n: PpiNode): string[] {
  return [
    n.gene, n.name, n.string_id, String(n.degree), n.betweenness.toFixed(4),
    n.community === null ? "" : String(n.community), n.logfc === null ? "" : n.logfc.toFixed(3), n.hub ? "yes" : "",
  ];
}
