/**
 * Traces for the mean-variance diagnostic: each gene's mean count against its
 * variance before normalisation (counts, or after the shifted log) and after
 * (Pearson residuals), on log axes, with the current variable genes in colour.
 */
import type { Dash } from "plotly.js";
import type { MeanVariance } from "@/api/normalization";

export type BeforeMode = "counts" | "log";

export const HVG_COLOR = "#dc2626";
export const OTHER_COLOR = "#94a3b8";
const LINE_POINTS = 60;

interface Trace {
  type: "scattergl" | "scatter";
  mode: "markers" | "lines";
  name: string;
  x: number[];
  y: number[];
  text?: string[];
  marker?: { size: number; color: string; opacity: number };
  line?: { color: string; width: number; dash: Dash };
  hovertemplate?: string;
  hoverinfo?: "skip";
}

/** Log-spaced points between the smallest and largest positive value. */
export function logSpace(values: number[], n = LINE_POINTS): number[] {
  const pos = values.filter((v) => v > 0);
  if (pos.length === 0) return [];
  const lo = Math.log10(Math.min(...pos));
  const hi = Math.log10(Math.max(...pos));
  if (hi <= lo) return [10 ** lo];
  return Array.from({ length: n }, (_, i) => 10 ** (lo + ((hi - lo) * i) / (n - 1)));
}

/** Gene indices split into the variable genes and the rest, dropping what a log
 *  axis cannot show (zero or missing values). */
export function splitGenes(mean: number[], y: number[], hvg: boolean[] | null) {
  const hv: number[] = [];
  const other: number[] = [];
  let hidden = 0;
  for (let i = 0; i < mean.length; i++) {
    if (!(mean[i]! > 0 && y[i]! > 0)) {
      hidden++;
      continue;
    }
    (hvg?.[i] ? hv : other).push(i);
  }
  return { hvg: hv, other, hidden };
}

function pointTrace(d: MeanVariance, y: number[], idx: number[], name: string, color: string, yLabel: string): Trace {
  return {
    type: "scattergl",
    mode: "markers",
    name,
    x: idx.map((i) => d.mean[i]!),
    y: idx.map((i) => y[i]!),
    text: idx.map((i) => d.genes[i]!),
    marker: { size: 3, color, opacity: color === HVG_COLOR ? 0.8 : 0.45 },
    hovertemplate: `%{text}<br>mean count %{x:.3g}<br>${yLabel} %{y:.3g}<extra></extra>`,
  };
}

function lineTrace(name: string, x: number[], y: number[], color: string, dash: Dash): Trace {
  return { type: "scatter", mode: "lines", name, x, y, line: { color, width: 1.5, dash }, hoverinfo: "skip" };
}

/** Plotly traces for one panel: "before" (counts or shifted log) or "after" (residuals). */
export function meanVarianceTraces(d: MeanVariance, panel: "before" | "after", before: BeforeMode = "counts"): Trace[] {
  const y = panel === "after" ? d.var_residuals : before === "counts" ? d.var_counts : d.var_log;
  const yLabel = panel === "after" ? "residual variance" : before === "counts" ? "variance" : "log variance";
  const { hvg, other } = splitGenes(d.mean, y, d.hvg);
  const traces: Trace[] = [pointTrace(d, y, other, d.hvg ? "Other genes" : "Genes", OTHER_COLOR, yLabel)];
  if (hvg.length > 0) traces.push(pointTrace(d, y, hvg, "Variable genes", HVG_COLOR, yLabel));
  const xs = logSpace(d.mean);
  if (xs.length === 0) return traces;
  if (panel === "after") {
    // A gene that varies by sampling alone has residual variance about 1, at any mean.
    traces.push(lineTrace("Sampling noise (1)", xs, xs.map(() => 1), "#0f172a", "dash"));
  } else if (before === "counts") {
    traces.push(lineTrace("Poisson (var = mean)", xs, xs, "#0f172a", "dot"));
    traces.push(
      lineTrace(`NB, θ = ${d.theta}`, xs, xs.map((m) => m + (m * m) / d.theta), "#2563eb", "dash"),
    );
  }
  return traces;
}
