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

// ---------------------------------------------------------------------------
// Paired designs: the same donors in more than one condition (Kang: every
// donor in CTRL and STIM). A slope chart joins each donor's values, so the
// change within a person, which the boxes hide, is what the plot shows.
// ---------------------------------------------------------------------------

/** Whether the donors recur across conditions (backend `design` / `pairing`). */
export interface PairedDesign {
  paired: boolean;
  /** Donors present in every condition. */
  n_pairs: number;
  /** The conditions, in the plotted order. */
  conditions: string[];
}

/** Second condition minus the first over the donors with a value in both (core/paired_design.py). */
export interface PairedTest {
  n_pairs: number;
  mean_diff: number | null;
  /** Wilcoxon signed-rank, two-sided; null with fewer than 2 pairs or no change. */
  wilcoxon_p: number | null;
  /** Paired t-test, two-sided. */
  ttest_p: number | null;
  /** Set below 5 pairs (descriptive) or when there is nothing to test. */
  caution: string | null;
}

export const SLOPE_LINE = "#94a3b8";
export const SLOPE_MEAN_LINE = "#334155";

/** The distinct donors among the points, and how many of them have a point in every condition. */
export function donorPairing(points: readonly DonorPoint[], conditions: readonly string[]): { donors: number; inAll: number } {
  const by = new Map<string, Set<string>>();
  for (const p of points) {
    if (!by.has(p.sample)) by.set(p.sample, new Set());
    by.get(p.sample)!.add(p.condition);
  }
  let inAll = 0;
  for (const s of by.values()) if (conditions.every((k) => s.has(k))) inAll += 1;
  return { donors: by.size, inAll };
}

/** "n = 8 donors, each in CTRL and STIM" or "n = 8 donors: 6 in both CTRL and STIM, 2 in one". */
export function pairedCaption(points: readonly DonorPoint[], conditions: readonly string[]): string {
  const { donors, inAll } = donorPairing(points, conditions);
  const list = conditions.length === 2 ? conditions.join(" and ") : `${conditions.slice(0, -1).join(", ")} and ${conditions[conditions.length - 1]}`;
  const noun = `donor${donors === 1 ? "" : "s"}`;
  if (inAll === donors) return `n = ${donors} ${noun}, each in ${list}`;
  const both = conditions.length === 2 ? "both" : "all of";
  return `n = ${donors} ${noun}: ${inAll} in ${both} ${list}, ${donors - inAll} in fewer`;
}

/** "+1.20 log2 CPM" / "-0.35 percentage points". */
export function formatChange(d: number | null, format: (v: number) => string, unit: string): string {
  if (d == null || !Number.isFinite(d)) return "–";
  return `${d >= 0 ? "+" : "-"}${format(Math.abs(d))}${unit ? ` ${unit}` : ""}`;
}

interface SlopeOpts {
  groups: readonly string[];
  conditions: readonly string[];
  colorOf: (category: string, kind: "condition" | "group") => string;
  format: (v: number) => string;
  /** Per group, the test to print next to its name (two conditions). */
  tests?: Readonly<Record<string, PairedTest | null | undefined>>;
  columns?: number;
}

/**
 * One panel per group: a grey line per donor through its values over the
 * conditions (hover: the donor and every value), and the mean per condition as
 * a heavier line with condition-coloured markers. With a test for the group,
 * its Wilcoxon p is in the panel title.
 */
export function donorSlopeFigure(points: readonly DonorPoint[], o: SlopeOpts): { data: Trace[]; layout: Record<string, unknown>; height: number } {
  const groups = o.groups.filter((g) => points.some((p) => p.group === g));
  const columns = Math.max(1, Math.min(o.columns ?? 4, groups.length));
  const rows = Math.max(1, Math.ceil(groups.length / columns));
  const data: Trace[] = [];
  const annotations: Trace[] = [];
  const layout: Record<string, unknown> = {
    grid: { rows, columns, pattern: "independent", xgap: 0.25, ygap: 0.5 },
    showlegend: false,
  };
  const pos = new Map(o.conditions.map((k, i) => [k, i]));
  groups.forEach((g, i) => {
    const suffix = i === 0 ? "" : String(i + 1);
    const axes = { xaxis: `x${suffix}`, yaxis: `y${suffix}` };
    const byDonor = new Map<string, DonorPoint[]>();
    for (const p of points) {
      if (p.group !== g) continue;
      if (!byDonor.has(p.sample)) byDonor.set(p.sample, []);
      byDonor.get(p.sample)!.push(p);
    }
    for (const [sample, pts] of byDonor) {
      pts.sort((a, b) => (pos.get(a.condition) ?? 0) - (pos.get(b.condition) ?? 0));
      const trail = pts.map((p) => `${p.condition} ${o.format(p.value)}`).join(", ");
      data.push({
        ...axes,
        type: "scatter",
        mode: "lines+markers",
        name: sample,
        x: pts.map((p) => p.condition),
        y: pts.map((p) => p.value),
        text: pts.map((p) => `${sample}: ${trail} (${p.n_cells.toLocaleString()} cells here)`),
        hovertemplate: "%{text}<extra></extra>",
        line: { color: SLOPE_LINE, width: 1 },
        marker: { color: SLOPE_LINE, size: 5 },
        legendgroup: "donors",
        showlegend: false,
      });
    }
    const means = o.conditions.map((k) => {
      const v = points.filter((p) => p.group === g && p.condition === k).map((p) => p.value);
      return v.length ? v.reduce((a, b) => a + b, 0) / v.length : null;
    });
    data.push({
      ...axes,
      type: "scatter",
      mode: "lines+markers",
      name: "mean",
      x: [...o.conditions],
      y: means,
      text: means.map((m, j) => `${o.conditions[j]} mean: ${m == null ? "–" : o.format(m)}`),
      hovertemplate: "%{text}<extra></extra>",
      line: { color: SLOPE_MEAN_LINE, width: 2.5 },
      marker: {
        color: o.conditions.map((k) => o.colorOf(k, "condition")),
        size: 10,
        line: { color: SLOPE_MEAN_LINE, width: 1.5 },
      },
      showlegend: false,
    });
    layout[`xaxis${suffix}`] = {
      type: "category",
      categoryorder: "array",
      categoryarray: [...o.conditions],
      tickfont: { size: 9 },
      range: [-0.4, o.conditions.length - 0.6],
    };
    layout[`yaxis${suffix}`] = { tickfont: { size: 9 }, gridcolor: "#f1f5f9", zeroline: false };
    const t = o.tests?.[g];
    const p = t?.wilcoxon_p;
    const title = p == null ? g : `${g} · p = ${formatP(p)}${t && t.n_pairs < 5 ? ` (n = ${t.n_pairs})` : ""}`;
    annotations.push({
      text: title,
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
  return { data, layout, height: rows * 170 + 60 };
}

function formatP(p: number): string {
  if (p < 0.001) return p.toExponential(1);
  return p < 0.01 ? p.toFixed(3) : p.toFixed(2);
}
