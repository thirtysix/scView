/** Helpers for the study planner (Compare > Plan a study): power curves and the
 *  sentence answers, from a plan the backend simulated. */
import type { CompositionPlan, CompositionPoint, DEPlan, DEPoint } from "@/api/planner";
import type { ObsColumnInfo } from "@/api/types";

/** A planner plans per cell type: put columns whose labels are mostly names ("CD14 Mono") before
 *  numbered clusterings, keeping the ranking's order otherwise. Kang's author cell types sit in a
 *  column called "cluster", which the shared ranking files under clusterings, after Leiden. */
export function plannerGroupOrder(ranked: readonly string[], cols: readonly ObsColumnInfo[]): string[] {
  const named = (name: string) => {
    const values = cols.find((c) => c.name === name)?.values ?? [];
    return values.length > 0 && values.filter((v) => !/^\d+$/.test(String(v).trim())).length > values.length / 2;
  };
  return [...ranked.filter(named), ...ranked.filter((n) => !named(n))];
}

export const DEFAULT_DONORS = [3, 4, 6, 8, 12, 16];
export const DEFAULT_CELLS = [500, 2000, 5000];
export const DEFAULT_FOLD_CHANGES = [1.5, 2, 3];
export const DE_DEFAULT_CELLS = [50, 200, 1000];
export const DE_DEFAULT_LOG2FC = [0.5, 1];

interface Pt {
  donors: number;
  power: number | null;
}

/**
 * Donors per group where a power curve first reaches `target`, linearly
 * interpolated between grid values and rounded up (as the backend's
 * donors_needed); null when the largest grid value falls short.
 */
export function donorsNeeded(points: readonly Pt[], target: number): number | null {
  const pts = points.filter((p) => p.power != null).sort((a, b) => a.donors - b.donors);
  let prev: Pt | null = null;
  for (const p of pts) {
    if ((p.power as number) >= target) {
      if (!prev || (prev.power as number) >= target) return p.donors;
      const frac = (target - (prev.power as number)) / Math.max((p.power as number) - (prev.power as number), 1e-9);
      // Rounded first: 8 + 0.5 * 4 can come out as 10.000000000000007.
      return Math.ceil(Math.round((prev.donors + frac * (p.donors - prev.donors)) * 1e6) / 1e6);
    }
    prev = p;
  }
  return null;
}

/** A group as a sentence names it: a bare number is a cluster ("cluster 8"). */
export function groupName(group: string): string {
  return /^\d+$/.test(group) ? `cluster ${group}` : group;
}

/** "a 2-fold increase", "a 1.5-fold decrease" (0.67 = 1 / 1.5). */
export function foldText(fc: number): string {
  if (fc === 1) return "no change";
  const f = fc >= 1 ? fc : 1 / fc;
  const shown = Number.isInteger(f) ? String(f) : f.toFixed(f < 10 ? 2 : 1).replace(/\.?0+$/, "");
  return `a ${shown}-fold ${fc > 1 ? "increase" : "decrease"}`;
}

export const pctText = (v: number | null | undefined) => (v == null ? "–" : `${Math.round(v * 100)}%`);

/** A share as "35.0%", small ones to two significant digits ("0.42%"). */
export function shareText(v: number): string {
  const p = v * 100;
  return p >= 1 ? `${p.toFixed(1)}%` : `${p.toPrecision(2)}%`;
}

export interface Curve {
  cells: number;
  pessimistic: boolean;
  donors: number[];
  power: (number | null)[];
  se: (number | null)[];
}

/** One curve per cells value (and the dashed pessimistic one) for a type and fold change. */
export function compositionCurves(plan: CompositionPlan, group: string, fc: number): Curve[] {
  const pick = (pess: boolean, cells?: number) =>
    plan.points
      .filter(
        (p) =>
          p.group === group && p.fold_change === fc && p.pessimistic === pess && (cells == null || p.cells === cells),
      )
      .sort((a, b) => a.donors - b.donors);
  const curve = (pts: CompositionPoint[], cells: number, pessimistic: boolean): Curve => ({
    cells,
    pessimistic,
    donors: pts.map((p) => p.donors),
    power: pts.map((p) => p.power),
    se: pts.map((p) => p.se),
  });
  const out = plan.cells.map((m) => curve(pick(false, m), m, false));
  const pess = pick(true);
  if (pess.length) out.push(curve(pess, plan.middle_cells, true));
  return out;
}

export function deCurves(plan: DEPlan, log2fc: number): Curve[] {
  return plan.cells.map((m) => {
    const pts = plan.points.filter((p) => p.log2fc === log2fc && p.cells === m).sort((a, b) => a.donors - b.donors);
    return {
      cells: m,
      pessimistic: false,
      donors: pts.map((p) => p.donors),
      power: pts.map((p) => p.power),
      se: pts.map((p) => p.se),
    };
  });
}

interface CellsNeed {
  cells: number;
  need: number | null;
}

/** Per cells value, the donors per group needed at `target` (from the curves). */
export function neededByCells(curves: Curve[], target: number): CellsNeed[] {
  return curves
    .filter((c) => !c.pessimistic)
    .map((c) => ({ cells: c.cells, need: donorsNeeded(c.donors.map((d, i) => ({ donors: d, power: c.power[i] ?? null })), target) }));
}

const fmt = (n: number) => n.toLocaleString();

/**
 * The cells clause: where adding cells per donor stops lowering the donors
 * needed. `best` is the fewest donors any cells value needs.
 */
function cellsClause(needs: CellsNeed[], unit: string): string {
  const reached = needs.filter((n) => n.need != null) as { cells: number; need: number }[];
  if (needs.length < 2 || !reached.length) return "";
  const best = Math.min(...reached.map((n) => n.need));
  const first = reached.find((n) => n.need === best)!;
  const all = needs.map((n) => n.need);
  if (all.every((v) => v === best)) {
    return `; the ${unit} (${fmt(needs[0]!.cells)} to ${fmt(needs[needs.length - 1]!.cells)}) barely change this`;
  }
  if (first.cells === needs[needs.length - 1]!.cells) {
    const prev = needs[needs.length - 2]!;
    return `; more cells still help (${prev.need == null ? "not reached" : `${prev.need}`} at ${fmt(prev.cells)} ${unit})`;
  }
  return `; adding cells beyond ${fmt(first.cells)} per donor barely helps`;
}

/**
 * The answer for one cell type and fold change, e.g. "To detect a 2-fold
 * increase in CD14 Mono's share (35.0% of the cells) with 80% power you need
 * about 6 donors per group at 2,000 cells per donor; adding cells beyond 2,000
 * per donor barely helps."
 */
export function compositionSentence(plan: CompositionPlan, group: string, fc: number, target: number): string {
  const type = plan.types.find((t) => t.group === group);
  const where = type ? ` (${shareText(type.share)} of the cells${plan.reference ? ` in ${plan.reference}` : ""})` : "";
  const curves = compositionCurves(plan, group, fc);
  const main = curves.filter((c) => !c.pessimistic);
  if (!main.some((c) => c.power.some((p) => p != null))) {
    return `${capital(foldText(fc))} is not possible for ${groupName(group)}: it would take its share past 98% of the cells.`;
  }
  const needs = neededByCells(curves, target);
  const reached = needs.filter((n) => n.need != null) as { cells: number; need: number }[];
  const powerPct = `${Math.round(target * 100)}%`;
  const what = `${foldText(fc)} in ${groupName(group)}'s share${where}`;
  if (!reached.length) {
    const maxN = Math.max(...plan.donors);
    const top = Math.max(...main.flatMap((c) => c.power.filter((p): p is number => p != null)));
    return (
      `Even ${maxN} donors per group give at most ${pctText(top)} power to detect ${what}; ` +
      "a study of this size can only find larger changes."
    );
  }
  const best = Math.min(...reached.map((n) => n.need));
  const at = reached.find((n) => n.need === best)!;
  const minN = Math.min(...plan.donors);
  const head =
    best <= minN
      ? `Even ${minN} donors per group at ${fmt(at.cells)} cells per donor give ${powerPct} power to detect ${what}`
      : `To detect ${what} with ${powerPct} power you need about ${best} donors per group at ${fmt(at.cells)} cells per donor`;
  let out = head + cellsClause(needs, "cells per donor") + ".";
  const pess = curves.find((c) => c.pessimistic);
  if (pess) {
    const n = donorsNeeded(pess.donors.map((d, i) => ({ donors: d, power: pess.power[i] ?? null })), target);
    out +=
      ` If the donors vary as much as this pilot allows (the upper end of its 95% interval), ` +
      (n == null ? `more than ${Math.max(...plan.donors)}` : `about ${n}`) +
      ` (at ${fmt(pess.cells)} cells).`;
  }
  return out;
}

/** "log2FC 1 (2-fold)". */
export function log2fcText(L: number): string {
  const f = 2 ** L;
  return `log2FC ${L} (${Number.isInteger(f) ? f : f.toFixed(2)}-fold)`;
}

/** The answer for one log2 fold change of a DE plan. */
export function deSentence(plan: DEPlan, L: number, target: number): string {
  const curves = deCurves(plan, L);
  const needs = neededByCells(curves, target);
  const reached = needs.filter((n) => n.need != null) as { cells: number; need: number }[];
  const share = `${Math.round(target * 100)}%`;
  const what = `${share} of the genes changed by ${log2fcText(L)} in ${groupName(plan.group)}`;
  const unit = `${groupName(plan.group)} cells per donor`;
  if (!reached.length) {
    const maxN = Math.max(...plan.donors);
    const top = Math.max(...plan.points.filter((p) => p.log2fc === L).map((p) => p.power));
    return `Even ${maxN} donors per group find at most ${pctText(top)} of the genes changed by ${log2fcText(L)} in ${groupName(plan.group)} (FDR < ${plan.alpha}); finding ${share} needs more donors or larger effects.`;
  }
  const best = Math.min(...reached.map((n) => n.need));
  const at = reached.find((n) => n.need === best)!;
  const minN = Math.min(...plan.donors);
  const head =
    best <= minN
      ? `Even ${minN} donors per group find ${what} (FDR < ${plan.alpha}) at ${fmt(at.cells)} ${unit}`
      : `Finding ${what} (FDR < ${plan.alpha}) takes about ${best} donors per group at ${fmt(at.cells)} ${unit}`;
  return head + cellsClause(needs, unit) + ".";
}

/** At the fewest cells: how many changed genes even reach the app's count filter. */
export function lostGenesNote(plan: DEPlan, L: number): string | null {
  const m = Math.min(...plan.cells);
  const pts = plan.points.filter((p) => p.log2fc === L && p.cells === m);
  if (!pts.length) return null;
  const tested = Math.max(...pts.map((p) => p.tested_share));
  if (tested >= 0.9) return null;
  return (
    `With ${fmt(m)} ${groupName(plan.group)} cells per donor only ${pctText(tested)} of the changed genes have enough ` +
    "counts to be tested at all (10+ counts in as many samples as a group has): more cells per donor raise the counts."
  );
}

/** The DE point nearest the headline (needed donors at the middle cells value), for the by-expression line. */
export function headlineDEPoint(plan: DEPlan, L: number, target: number): DEPoint | null {
  const m = plan.cells[Math.floor(plan.cells.length / 2)]!;
  const pts = plan.points.filter((p) => p.log2fc === L && p.cells === m).sort((a, b) => a.donors - b.donors);
  if (!pts.length) return null;
  const n = donorsNeeded(pts, target);
  return pts.find((p) => p.donors >= (n ?? Infinity)) ?? pts[pts.length - 1]!;
}

/** Comma- or space-separated positive numbers, sorted and de-duplicated; null when any is bad. */
export function parseNumbers(text: string, opts: { integer?: boolean; min?: number; max?: number } = {}): number[] | null {
  const parts = text.split(/[\s,;]+/).filter(Boolean);
  if (!parts.length) return null;
  const vals: number[] = [];
  for (const p of parts) {
    const v = Number(p);
    if (!Number.isFinite(v)) return null;
    if (opts.integer && !Number.isInteger(v)) return null;
    if (opts.min != null && v < opts.min) return null;
    if (opts.max != null && v > opts.max) return null;
    vals.push(v);
  }
  return [...new Set(vals)].sort((a, b) => a - b);
}

function capital(s: string): string {
  return s.charAt(0).toUpperCase() + s.slice(1);
}

/** A question for the co-pilot about the plan. */
export function plannerQuestion(plan: CompositionPlan | DEPlan): string {
  if (plan.kind === "composition") {
    const types = plan.types.filter((t) => t.targeted).map((t) => t.group).join(", ");
    return (
      `The study planner simulated a ${plan.design} follow-up from this dataset's donors for ${types} ` +
      `[result:planner]. How many donors per group should the follow-up have, what drives that, and how ` +
      "far can these estimates be trusted given the pilot's own size?"
    );
  }
  return (
    `The study planner simulated pseudobulk DE in ${plan.group} for a ${plan.design} follow-up [result:planner]. ` +
    "How many donors and cells per donor should it have, and how far can these estimates be trusted?"
  );
}
