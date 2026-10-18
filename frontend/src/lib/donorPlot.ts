// Box + strip plots with one point per donor (or sample), split by condition:
// cell-type shares per donor (composition test) and a gene's pseudobulk value
// per donor (the Unified View violin's "Per donor" mode).

/** One donor (x condition) sample's value in one group. */
export interface DonorPoint {
  group: string;
  /** "" when the plot is not split by a condition. */
  condition: string;
  sample: string;
  n_cells: number;
  value: number;
}

/** Fewer donors than this in a condition and the box is a range, not a distribution. */
export const FEW_DONORS = 3;

/** Distinct donors per condition among the points. */
export function donorsPerCondition(points: readonly DonorPoint[], conditions: readonly string[]): Record<string, number> {
  return Object.fromEntries(
    conditions.map((k) => [k, new Set(points.filter((p) => p.condition === k).map((p) => p.sample)).size]),
  );
}

/** "n = 8: CTRL 8, STIM 8" (the most in any condition, then each), or "n = 8" unsplit. */
export function donorCaption(nDonors: Record<string, number>, conditions: readonly string[]): string {
  if (conditions.length === 1 && conditions[0] === "") return `n = ${nDonors[""] ?? 0}`;
  const all = Math.max(0, ...conditions.map((k) => nDonors[k] ?? 0));
  return `n = ${all}: ${conditions.map((k) => `${k} ${nDonors[k] ?? 0}`).join(", ")}`;
}

/** The caution for conditions with fewer than FEW_DONORS donors, or null. */
export function fewDonorsCaution(nDonors: Record<string, number>, conditions: readonly string[]): string | null {
  const few = conditions.filter((k) => (nDonors[k] ?? 0) < FEW_DONORS);
  if (few.length === 0) return null;
  const where = few.map((k) => `${k || "all cells"}: ${nDonors[k] ?? 0}`).join(", ");
  return (
    `Fewer than ${FEW_DONORS} donors (${where}). A box of so few points shows their range, ` +
    `not a distribution, and cannot tell a condition effect from those donors' own levels: read the points.`
  );
}

interface TraceOpts {
  groups: readonly string[];
  conditions: readonly string[];
  /** CSS colour of a condition (or, unsplit, of a group). */
  colorOf: (category: string, kind: "condition" | "group") => string;
  /** One small plot per group, each with its own y axis (rare types stay readable). */
  facet: boolean;
  /** Hover text of a value, e.g. "12.3%". */
  format: (v: number) => string;
  /** Facets per row. */
  columns?: number;
}

type Trace = Record<string, unknown>;

function box(points: readonly DonorPoint[], x: string[], name: string, color: string, format: TraceOpts["format"], showlegend: boolean): Trace {
  return {
    type: "box",
    name,
    x,
    y: points.map((p) => p.value),
    text: points.map((p) => `${p.sample}: ${format(p.value)} (${p.n_cells.toLocaleString()} cells)`),
    boxpoints: "all",
    jitter: 0.4,
    pointpos: 0,
    marker: { color, size: 5 },
    line: { width: 1, color },
    hovertemplate: "%{x}<br>%{text}<extra>" + (name || "") + "</extra>",
    legendgroup: name,
    showlegend,
  };
}

/**
 * Plotly traces and layout additions for the points: one box (with every
 * point drawn) per condition in each group, grouped on one axis, or one panel
 * per group. Unsplit (conditions [""]) each group's box takes the group colour.
 */
export function donorBoxFigure(points: readonly DonorPoint[], o: TraceOpts): { data: Trace[]; layout: Record<string, unknown>; height: number } {
  const split = !(o.conditions.length === 1 && o.conditions[0] === "");
  if (!o.facet) {
    const data = split
      ? o.conditions.map((k) => {
          const pts = points.filter((p) => p.condition === k);
          return box(pts, pts.map((p) => p.group), k, o.colorOf(k, "condition"), o.format, true);
        })
      : o.groups.map((g) => {
          const pts = points.filter((p) => p.group === g);
          return box(pts, pts.map((p) => p.group), g, o.colorOf(g, "group"), o.format, false);
        });
    return {
      data,
      layout: {
        boxmode: split ? "group" : "overlay",
        xaxis: { type: "category", categoryorder: "array", categoryarray: [...o.groups], tickangle: -45 },
      },
      height: 0,
    };
  }

  const groups = o.groups.filter((g) => points.some((p) => p.group === g));
  const columns = Math.max(1, Math.min(o.columns ?? 4, groups.length));
  const rows = Math.max(1, Math.ceil(groups.length / columns));
  const data: Trace[] = [];
  const layout: Record<string, unknown> = {
    grid: { rows, columns, pattern: "independent", xgap: 0.25, ygap: 0.45 },
    showlegend: split,
  };
  const annotations: Trace[] = [];
  groups.forEach((g, i) => {
    const suffix = i === 0 ? "" : String(i + 1);
    const conds = split ? o.conditions : [""];
    conds.forEach((k) => {
      const pts = points.filter((p) => p.group === g && p.condition === k);
      if (pts.length === 0) return;
      const t = split
        ? box(pts, pts.map(() => k), k, o.colorOf(k, "condition"), o.format, i === 0)
        : box(pts, pts.map(() => g), g, o.colorOf(g, "group"), o.format, false);
      data.push({ ...t, xaxis: `x${suffix}`, yaxis: `y${suffix}` });
    });
    layout[`xaxis${suffix}`] = { type: "category", showticklabels: split, tickfont: { size: 9 } };
    layout[`yaxis${suffix}`] = { tickfont: { size: 9 }, gridcolor: "#f1f5f9", zeroline: false };
    annotations.push({
      text: g,
      xref: `x${suffix} domain`,
      yref: `y${suffix} domain`,
      x: 0.5,
      y: 1.02,
      xanchor: "center",
      yanchor: "bottom",
      showarrow: false,
      font: { size: 10, color: "#334155" },
    });
  });
  layout.annotations = annotations;
  return { data, layout, height: rows * 150 + 60 };
}
