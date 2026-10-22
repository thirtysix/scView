import type {
  HashClass,
  HashingCall,
  HashingLfcHist,
  HashingPlotData,
  HashingSummary,
  HashingTagHist,
} from "@/api/hashing";

// Doublets and the rest in greys: the tags' own colours come from the categorical palette.
export const DOUBLET_COLOR = "#334155";
export const OTHER_COLOR = "#cbd5e1";
export const CUTOFF_COLOR = "#dc2626";
/** The call's own obs columns: a sample column cannot take these names. */
export const HASH_COLUMNS = ["hash_call", "hash_class", "hash_max", "hash_second", "hash_margin"];
/** Per-tag histograms drawn; a larger panel is cut with a caption. */
export const MAX_TAG_PLOTS = 24;
const CLASSES: HashClass[] = ["Singlet", "Doublet", "Negative"];

/** Same rule as the backend's column names (core/user_labels.py clean_column_name). */
export function validColumnName(name: string): boolean {
  const n = name.trim();
  return (
    n.length > 0 &&
    n.length <= 64 &&
    /^[A-Za-z][A-Za-z0-9 _.-]*$/.test(n) &&
    !n.toLowerCase().startsWith("scview_") &&
    !HASH_COLUMNS.includes(n)
  );
}

/** Which features start checked: the saved call's tags, else every tag count and the
 *  antibody features whose names look like hashtags. */
export function initialSelection(s: HashingSummary): Set<string> {
  if (s.call) return new Set(s.call.tags.filter((t) => s.tags.includes(t) || s.proteins.includes(t)));
  return new Set([...s.tags, ...s.suggested]);
}

/** Selected names in display order: tag counts first, then antibody features. */
export function orderedSelection(s: HashingSummary, picked: Set<string>): string[] {
  return [...s.tags, ...s.proteins].filter((n) => picked.has(n));
}

export function pct(n: number, total: number): string {
  return total > 0 ? `${((100 * n) / total).toFixed(1)}%` : "0%";
}

/** "15,223 singlets (75.2%), 3,157 doublets (15.6%) and 1,858 negatives (9.2%)". */
export function classSentence(counts: Record<HashClass, number>): string {
  const total = CLASSES.reduce((a, c) => a + (counts[c] ?? 0), 0);
  const word = { Singlet: "singlets", Doublet: "doublets", Negative: "negatives" } as const;
  const parts = CLASSES.map((c) => `${(counts[c] ?? 0).toLocaleString()} ${word[c]} (${pct(counts[c] ?? 0, total)})`);
  return `${parts[0]}, ${parts[1]} and ${parts[2]}`;
}

/** What "Negative" means for the method (hashedDrops calls no negatives). */
export function negativeMeaning(call: HashingCall): string {
  return call.method === "hashedDrops"
    ? "Negative: not confidently assigned (hashedDrops does not tell empty droplets from ambiguous ones)."
    : "Negative: no tag above its cutoff (an empty droplet or a cell with too few tag counts).";
}

/** The doublet views read one source: say which. */
export function doubletSourceText(source: string | null): string {
  if (source === "hash_class")
    return (
      "Hashing is the primary doublet call: cluster QC, the QC report and the filtering step's 'drop doublets' " +
      "(source auto) use these hashing doublets; Scrublet stays as the second opinion (Doublet score card)."
    );
  if (source === "predicted_doublet") return "Cluster QC and 'drop doublets' use Scrublet's calls (predicted_doublet).";
  if (source) return `Cluster QC and 'drop doublets' use the column '${source}'.`;
  return "";
}

/** Hashing doublets that Scrublet also calls (only inter-sample doublets are visible to hashing). */
export function scrubletText(s: HashingSummary["scrublet"]): string | null {
  if (!s) return null;
  const d = s.Doublet;
  const n = d.scrublet_doublet + d.scrublet_singlet;
  const scrSinglets = s.Singlet.scrublet_doublet;
  return (
    `Of ${n.toLocaleString()} hashing doublets, Scrublet also calls ${d.scrublet_doublet.toLocaleString()} ` +
    `(${pct(d.scrublet_doublet, n)}); it calls ${scrSinglets.toLocaleString()} of the hashing singlets doublets ` +
    "(same-sample doublets, which hashing cannot see, or false calls)."
  );
}

/** Mapping rows start from the saved mapping, else the tag names. */
export function initialMapping(call: HashingCall, saved: Record<string, string> | undefined): Record<string, string> {
  return Object.fromEntries(call.tags.map((t) => [t, saved?.[t] ?? t]));
}

// ---------------------------------------------------------------------------
// Figures (Plotly traces + layout, built here so they can be tested)
// ---------------------------------------------------------------------------

interface Axis {
  domain: [number, number];
  anchor: string;
  [k: string]: unknown;
}

function centers(edges: number[]): { x: number[]; width: number } {
  const x = edges.slice(0, -1).map((e, i) => (e + (edges[i + 1] ?? e)) / 2);
  return { x, width: edges.length > 1 ? (edges[1] ?? 1) - (edges[0] ?? 0) : 1 };
}

/** One small histogram per tag (CLR across cells), one overlaid histogram per group of
 *  the call (this tag's singlets, all doublets, the other cells: each cell is in one),
 *  on a log count axis so the small positive mode shows beside the background; a red
 *  line at the tag's cutoff. */
export function tagHistFigure(
  hists: HashingTagHist[],
  colorOf: (tag: string) => string,
): { data: Record<string, unknown>[]; layout: Record<string, unknown>; height: number } {
  const shown = hists.slice(0, MAX_TAG_PLOTS);
  const cols = shown.length > 1 ? 2 : 1;
  const rows = Math.max(1, Math.ceil(shown.length / cols));
  const rowH = 120;
  const top = 56;
  const height = rows * rowH + top + 34;
  const gapX = 0.1;
  const gapY = rows > 1 ? 0.5 / rows : 0;
  const data: Record<string, unknown>[] = [];
  const layout: Record<string, unknown> = {
    barmode: "overlay",
    bargap: 0,
    margin: { l: 44, r: 10, t: top, b: 34 },
    showlegend: true,
    // Above the first row's titles: paper units are the plot area (height - margins).
    legend: { orientation: "h", x: 0, y: 1 + 30 / (height - top - 34), yanchor: "bottom", font: { size: 9 } },
    annotations: [] as unknown[],
    shapes: [] as unknown[],
  };
  shown.forEach((h, i) => {
    const r = Math.floor(i / cols);
    const c = i % cols;
    const n = i === 0 ? "" : String(i + 1);
    const w = (1 - gapX * (cols - 1)) / cols;
    const hgt = (1 - gapY * (rows - 1)) / rows;
    const x0 = c * (w + gapX);
    const y1 = 1 - r * (hgt + gapY);
    const xa: Axis = { domain: [x0, x0 + w], anchor: `y${n}`, tickfont: { size: 8 }, zeroline: false };
    const ya: Axis = { domain: [Math.max(0, y1 - hgt), y1], anchor: `x${n}`, tickfont: { size: 8 }, type: "log",
      dtick: 1, exponentformat: "power" };
    if (r === rows - 1 || i + cols >= shown.length) xa.title = { text: "CLR", font: { size: 9 }, standoff: 2 };
    layout[`xaxis${n}`] = xa;
    layout[`yaxis${n}`] = ya;
    const { x, width } = centers(h.bin_edges);
    const groups: [string, string, string][] = h.counts.all
      ? [["all", "cells", OTHER_COLOR]]
      : [ // drawn in this order: the tag's own singlets on top
          ["other", "other cells", OTHER_COLOR],
          ["doublet", "doublets", DOUBLET_COLOR],
          ["tag", "this tag's singlets", "__tag__"],
        ];
    for (const [key, name, color] of groups) {
      const y = h.counts[key];
      if (!y) continue;
      data.push({
        type: "bar",
        x,
        y,
        width,
        name,
        legendgroup: key,
        showlegend: i === 0,
        marker: { color: color === "__tag__" ? colorOf(h.tag) : color, line: { width: 0 } },
        opacity: key === "other" || key === "all" ? 1 : 0.85,
        xaxis: `x${n}`,
        yaxis: `y${n}`,
        hovertemplate: `${h.tag}, ${name}<br>CLR %{x:.2f}: %{y} cells<extra></extra>`,
      });
    }
    (layout.annotations as unknown[]).push({
      text: `<b>${h.tag}</b>${h.cutoff != null ? `  cutoff ${h.cutoff} counts, ${(h.n_above ?? 0).toLocaleString()} cells above` : ""}`,
      xref: `x${n} domain`,
      yref: `y${n} domain`,
      x: 0,
      y: 1.02,
      xanchor: "left",
      yanchor: "bottom",
      showarrow: false,
      font: { size: 9, color: "#334155" },
    });
    if (h.cutoff_clr != null) {
      (layout.shapes as unknown[]).push({
        type: "line",
        xref: `x${n}`,
        yref: `y${n} domain`,
        x0: h.cutoff_clr,
        x1: h.cutoff_clr,
        y0: 0,
        y1: 1,
        line: { color: CUTOFF_COLOR, width: 1.5, dash: "dash" },
      });
    }
  });
  return { data, layout, height };
}

/** The tag x call heatmap: mean CLR of each tag (rows) in the cells of each call (columns). */
export function heatmapTrace(
  hm: NonNullable<HashingPlotData["heatmap"]>,
  colorscale: [number, string][],
): Record<string, unknown> {
  const x = hm.calls.map((c, j) => `${c} (${(hm.n[j] ?? 0).toLocaleString()})`);
  return {
    type: "heatmap",
    x,
    y: hm.tags,
    z: hm.mean_clr,
    colorscale,
    zmin: 0,
    colorbar: { title: { text: "mean CLR", side: "right", font: { size: 9 } }, thickness: 10, tickfont: { size: 8 } },
    hovertemplate: "tag %{y}<br>cells called %{x}<br>mean CLR %{z:.2f}<extra></extra>",
    xgap: 1,
    ygap: 1,
  };
}

/** hashedDrops' two statistics with their thresholds. */
export function lfcFigure(lfc: { lfc: HashingLfcHist | null; lfc2: HashingLfcHist | null }): {
  data: Record<string, unknown>[];
  layout: Record<string, unknown>;
} {
  const panels: [HashingLfcHist | null, string, string][] = [
    [lfc.lfc, "LogFC: top tag over second (log2)", "confident singlet above"],
    [lfc.lfc2, "LogFC2: second tag over ambient (log2)", "doublet above"],
  ];
  const data: Record<string, unknown>[] = [];
  const layout: Record<string, unknown> = {
    margin: { l: 44, r: 10, t: 26, b: 40 },
    showlegend: false,
    shapes: [] as unknown[],
    annotations: [] as unknown[],
    bargap: 0,
  };
  panels.forEach(([h, title, rule], i) => {
    const n = i === 0 ? "" : "2";
    const d: [number, number] = i === 0 ? [0, 0.45] : [0.55, 1];
    layout[`xaxis${n}`] = { domain: d, anchor: `y${n}`, title: { text: title, font: { size: 9 } }, tickfont: { size: 8 } };
    layout[`yaxis${n}`] = { domain: [0, 1], anchor: `x${n}`, tickfont: { size: 8 } };
    if (!h) return;
    const { x, width } = centers(h.bin_edges);
    data.push({ type: "bar", x, y: h.counts, width, marker: { color: OTHER_COLOR }, xaxis: `x${n}`, yaxis: `y${n}`,
      hovertemplate: `${title}<br>%{x:.2f}: %{y} cells<extra></extra>` });
    if (h.threshold != null) {
      (layout.shapes as unknown[]).push({ type: "line", xref: `x${n}`, yref: `y${n} domain`, x0: h.threshold,
        x1: h.threshold, y0: 0, y1: 1, line: { color: CUTOFF_COLOR, width: 1.5, dash: "dash" } });
      (layout.annotations as unknown[]).push({ text: `${rule} ${h.threshold.toFixed(2)}`, xref: `x${n}`,
        yref: `y${n} domain`, x: h.threshold, y: 1, xanchor: "left", yanchor: "bottom", showarrow: false,
        font: { size: 9, color: CUTOFF_COLOR } });
    }
  });
  return { data, layout };
}
