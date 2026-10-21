import type { DEComposition, DECompositionGroup } from "@/api/de";

/** A share as the composition note writes it: "18%", "0.4%", "0%". */
export function formatShare(share: number): string {
  const p = share * 100;
  if (p === 0) return "0%";
  return p < 1 ? `${p.toFixed(1)}%` : `${Math.round(p)}%`;
}

/** "+24 points" / "−5 points" (one decimal under 1). */
export function formatPoints(diff: number): string {
  const a = Math.abs(diff);
  const v = a < 1 ? a.toFixed(1) : String(Math.round(a));
  return `${diff > 0 ? "+" : diff < 0 ? "−" : ""}${v} points`;
}

/** "18% of CTRL vs 42% of STIM (−24 points)". */
export function shareLine(g: DECompositionGroup, labelA: string, labelB: string): string {
  return `${formatShare(g.share_a)} of ${labelA} vs ${formatShare(g.share_b)} of ${labelB} (${formatPoints(g.diff_points)})`;
}

/** The groups past the rule, largest difference first. */
export function flaggedGroups(c: DEComposition): DECompositionGroup[] {
  return c.groups.filter((g) => g.flagged);
}

/** What the rule is, for the tooltip. */
export function compositionRule(c: Pick<DEComposition, "threshold_points" | "min_cells">): string {
  return (
    `Each side's share of every group, among the cells the DE compares (a group's cells over that ` +
    `side's cells, so unequal side sizes do not count). A group is flagged when its two shares differ ` +
    `by more than ${c.threshold_points} percentage points, it has at least ${c.min_cells} cells on ` +
    `one side, and Fisher's exact test (its cells vs the rest, one side vs the other) gives p < 0.05, ` +
    `so a few cells on a small side do not count: genes specific to it then look like a difference between the sides, cell-level or ` +
    `pseudobulk. "Nearly absent from one side" (80% or more of its share on one side) can also be a ` +
    `state the comparison itself creates.`
  );
}

/** One plain sentence when nothing is flagged: the largest difference, for scale. */
export function balancedText(c: DEComposition): string | null {
  const top = c.groups[0];
  if (!c.column || !top || c.nested) return null;
  return (
    `No ${c.column} group differs by more than ${c.threshold_points} points between the sides; ` +
    `the largest is ${top.group}, ${shareLine(top, c.label_a, c.label_b)}.`
  );
}

/** The composition caution the co-pilot reads with a DE result, or null when there is none. */
export function compositionCautionForCopilot(c: DEComposition | null | undefined): string | null {
  return c?.warn && c.message ? c.message : null;
}
