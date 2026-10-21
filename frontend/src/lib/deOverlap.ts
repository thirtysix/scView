import type {
  DEOverlapIntersection,
  DEOverlapResponse,
  DEOverlapSet,
  DEResultSummary,
  OverlapDirection,
} from "@/api/deOverlap";
import { intersectionName } from "@/lib/upset";

/** Columns the plot draws; the backend sends up to 64, largest first. */
export const MAX_COLUMNS = 40;
/** Results checked on first sight of the list. */
export const DEFAULT_CHECKED = 4;
/** Gene lists comparable beside the DE results (backend DEOverlapRequest). */
export const MAX_EXTRA_SETS = 4;

/** Results in the order they were run (oldest first): the plot's rows follow it. */
export function runOrder(results: readonly DEResultSummary[]): DEResultSummary[] {
  return [...results].sort((a, b) => a.created_at.localeCompare(b.created_at));
}

/** The newest few, checked when the list first arrives. */
export function defaultChecked(results: readonly DEResultSummary[], n = DEFAULT_CHECKED): string[] {
  return runOrder(results)
    .slice(-n)
    .map((r) => r.result_id);
}

export function directionText(d: OverlapDirection): string {
  return d === "up" ? "higher in group A" : d === "down" ? "higher in group B" : "either way (sign kept)";
}

/** How a set was chosen, for its tooltip. */
export function setRuleText(s: DEOverlapSet, res: Pick<DEOverlapResponse, "padj_max" | "lfc_min" | "logreg_top_n">): string {
  if (s.kind === "extra") return "a gene list (no direction)";
  if (s.rule === "top_n")
    return `no p-values (logistic regression): top ${res.logreg_top_n} genes per direction by |coefficient| past |log2FC| ${res.lfc_min}`;
  const adj = s.rule === "padj_bh_here" ? " (raw p adjusted here, BH)" : "";
  return `adjusted p ≤ ${res.padj_max}${adj} and |log2FC| ≥ ${res.lfc_min}`;
}

export function setTitle(s: DEOverlapSet, res: DEOverlapResponse): string {
  const counts =
    s.kind === "de"
      ? `${s.n.toLocaleString()} genes (${s.n_up.toLocaleString()} up, ${s.n_down.toLocaleString()} down) of ${(s.n_tested ?? 0).toLocaleString()} tested`
      : `${s.n.toLocaleString()} genes${s.n_not_tested ? `, ${s.n_not_tested.toLocaleString()} tested in no result` : ""}`;
  return `${s.name}: ${counts}; ${setRuleText(s, res)}`;
}

/** Tooltip and accessible name of one intersection column. */
export function columnTitle(x: DEOverlapIntersection, sets: readonly DEOverlapSet[]): string {
  const where = intersectionName(
    sets.map((s) => s.name),
    x.sets,
  );
  const signed = x.n_up + x.n_down > 0 ? ` (${x.n_up.toLocaleString()} up, ${x.n_down.toLocaleString()} down)` : "";
  return `${x.n.toLocaleString()} gene${x.n === 1 ? "" : "s"} in ${where}${signed}`;
}

/** The caption's sentence about how the sets were made, with n per set. */
export function overlapSummary(res: DEOverlapResponse): string {
  const per = res.sets.map((s) => `${s.name} ${s.n.toLocaleString()}`).join("; ");
  return (
    `Each set: genes significant ${directionText(res.direction)} in one result (adjusted p ≤ ${res.padj_max}, ` +
    `|log2FC| ≥ ${res.lfc_min}). Genes per set: ${per}. ${res.n_union.toLocaleString()} genes are in at least one set.`
  );
}

/** Genes as text for the clipboard: one per line. */
export function genesText(genes: readonly { gene: string }[]): string {
  return genes.map((g) => g.gene).join("\n");
}

export function lfcText(v: number | null | undefined): string {
  return v == null ? "–" : v.toFixed(2);
}
