/** Helpers for QC thresholds set on a histogram (components/plots/ThresholdHistogram). */

import type { FilterThresholds, QcThresholdMode } from "@/api/qc";

/** The mode a set of thresholds is in (none set: the fixed cut-offs). */
export function thresholdMode(t: Pick<FilterThresholds, "qc_threshold_mode">): QcThresholdMode {
  return t.qc_threshold_mode === "mad" ? "mad" : "fixed";
}

/** A count bound for display: whole numbers with separators; null = none. */
export function bound(v: number | null | undefined, digits = 0): string {
  if (v === null || v === undefined || !isFinite(v)) return "—";
  return digits > 0 ? v.toFixed(digits) : Math.round(v).toLocaleString();
}

/**
 * Where a dragged threshold line ended up, from a Plotly relayout(ing) event:
 * `shapes[i].x0` / `x1`. Dragging the line moves both ends; dragging one end
 * moves only that one (the line then snaps back upright at that position).
 * Null when the event is not about shape `index`.
 */
export function relayoutLineValue(event: Readonly<Record<string, unknown>>, index: number): number | null {
  const x0 = event[`shapes[${index}].x0`];
  const x1 = event[`shapes[${index}].x1`];
  const a = typeof x0 === "number" && isFinite(x0) ? x0 : null;
  const b = typeof x1 === "number" && isFinite(x1) ? x1 : null;
  if (a !== null && b !== null) return (a + b) / 2;
  return a ?? b;
}

/** `value` to the nearest multiple of `step` (step <= 0: unchanged), never below 0. */
export function roundTo(value: number, step: number): number {
  if (!(step > 0)) return Math.max(0, value);
  const decimals = Math.max(0, -Math.floor(Math.log10(step)));
  return Math.max(0, Number((Math.round(value / step) * step).toFixed(decimals)));
}

/** A number input's text as a threshold: empty, 0 or not a number = off (null). */
export function parseThreshold(text: string): number | null {
  const v = Number(text);
  return text.trim() === "" || !isFinite(v) || v <= 0 ? null : v;
}

/** A share as a percentage for display: 0.0312 -> "3.1%"; under 0.1 % keeps one more digit. */
export function pct(fraction: number | null | undefined): string {
  if (fraction === null || fraction === undefined || !isFinite(fraction)) return "—";
  const p = fraction * 100;
  if (p > 0 && p < 0.1) return `${p.toFixed(2)}%`;
  return `${p.toFixed(1)}%`;
}
