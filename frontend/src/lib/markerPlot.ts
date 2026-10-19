/**
 * Colour and dot size for the marker dot plot and heatmap (C016).
 *
 * Scanpy-style plots scale each gene to 0-1 across the groups shown, so the
 * same colour means something different in every plot. The "fixed" scale
 * instead maps the layer's own range (`scale_range` from
 * GET /expression/by-group: 0 to the 99th percentile of its non-zero values,
 * the same whatever genes and groups are asked for) onto the colours, and the
 * dot size follows one mapping for every plot, with a legend.
 *
 * The mean over all cells is roughly fraction x level, so with the dot size
 * showing the fraction, colour and size say much the same thing. "Mean in
 * expressing cells" shows the level alone; groups where too few cells express
 * a gene for that level to mean much get no colour.
 */

export interface MarkerRow {
  gene: string;
  logfoldchange: number | null;
}

/** The top `perGroup` up-regulated markers of each group, in group order, each gene once. */
export function pickMarkerGenes(
  groups: string[],
  markers: Record<string, MarkerRow[]>,
  perGroup: number,
): { gene: string; group: string }[] {
  const seen = new Set<string>();
  const out: { gene: string; group: string }[] = [];
  for (const group of groups) {
    let n = 0;
    for (const m of markers[group] ?? []) {
      if (n >= perGroup) break;
      if (!((m.logfoldchange ?? 0) > 0) || seen.has(m.gene)) continue;
      seen.add(m.gene);
      out.push({ gene: m.gene, group });
      n++;
    }
  }
  return out;
}

/** "all": mean over every cell of the group; "expressing": over cells above 0 only. */
export type MeanOf = "all" | "expressing";
/** "gene": each gene scaled to 0-1 across the groups shown; "fixed": the layer's range. */
export type ColourScale = "gene" | "fixed";

/** Below this share of expressing cells a dot is hollow and has no colour. */
export const MIN_FRACTION = 0.05;
/** Dot diameter in px for a group where every cell expresses the gene. */
export const DOT_MAX_PX = 24;
/** Shares shown in the dot-size legend. */
export const DOT_LEGEND = [0.2, 0.4, 0.6, 0.8, 1] as const;

/** Dot diameter (px) for a share of expressing cells: area proportional to
 *  the share, the same mapping in every plot. */
export function dotSize(fraction: number): number {
  const f = Math.min(1, Math.max(0, Number.isFinite(fraction) ? fraction : 0));
  return Math.max(DOT_MAX_PX * Math.sqrt(f), 2);
}

/** Scale each gene (column) to 0-1 across groups, as scanpy's
 *  standard_scale="var". Missing values (null) stay missing and are left out of
 *  the gene's range; a flat gene maps to 0. */
export function scalePerGene<T extends number | null>(values: T[][]): T[][] {
  const nGenes = values[0]?.length ?? 0;
  const lo: number[] = [];
  const hi: number[] = [];
  for (let j = 0; j < nGenes; j++) {
    const col = values.map((row) => row[j]).filter((v): v is NonNullable<T> => v !== null && v !== undefined);
    lo.push(col.length ? Math.min(...col) : 0);
    hi.push(col.length ? Math.max(...col) : 0);
  }
  return values.map((row) =>
    row.map((v, j) => {
      if (v === null) return v;
      const range = hi[j]! - lo[j]!;
      return (range > 0 ? (v - lo[j]!) / range : 0) as T;
    }),
  );
}

export interface MarkerColours {
  /** [row][gene]; null: no colour (too few expressing cells for a level) */
  z: (number | null)[][];
  cmin: number;
  cmax: number;
  /** Colourbar title (Plotly HTML) */
  title: string;
  /** One-sentence caption */
  caption: string;
}

/** Round to at most two decimals for labels. */
function fmt(v: number): string {
  return String(Math.round(v * 100) / 100);
}

/** The values to colour by and their colour range. */
export function markerColours(
  rows: { mean: number[][]; meanExpressing: number[][] | null; fraction: number[][] },
  meanOf: MeanOf,
  scale: ColourScale,
  range: [number, number],
  noun = "gene",
): MarkerColours {
  const expressing = meanOf === "expressing" && rows.meanExpressing !== null;
  const values: (number | null)[][] = expressing
    ? rows.meanExpressing!.map((row, r) =>
        row.map((v, c) => ((rows.fraction[r]?.[c] ?? 0) < MIN_FRACTION ? null : v)),
      )
    : rows.mean;
  const what = expressing ? "Mean in<br>expressing cells" : "Mean";
  const whatText = expressing ? "mean in the cells that express it" : "mean over all cells";
  if (scale === "gene") {
    return {
      z: scalePerGene(values),
      cmin: 0,
      cmax: 1,
      title: `${what}<br>(scaled per ${noun})`,
      caption: `Colour: ${whatText}, scaled to 0-1 per ${noun} (not comparable between plots).`,
    };
  }
  const [lo, hi] = range;
  return {
    z: values,
    cmin: lo,
    cmax: hi > lo ? hi : lo + 1,
    title: `${what}<br>(fixed ${fmt(lo)}-${fmt(hi)})`,
    caption: `Colour: ${whatText}, on a fixed ${fmt(lo)} to ${fmt(hi)} scale (higher values show as ${fmt(hi)}), the same in every plot of this layer.`,
  };
}
