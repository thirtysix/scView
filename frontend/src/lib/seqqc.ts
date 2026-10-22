import type { KneeSummary, SeqFlag, SeqMetricInfo, SeqMetricKey, SeqSample } from "@/api/seqqc";
import { rgbCss } from "@/lib/categoryColors";
import { CATEGORICAL_COLORS } from "@/lib/colors";

/** The per-sample table's columns, in order, with short headers. */
export const TABLE_METRICS: { key: SeqMetricKey; short: string }[] = [
  { key: "estimated_cells", short: "Cells" },
  { key: "mean_reads_per_cell", short: "Reads/cell" },
  { key: "median_genes_per_cell", short: "Genes" },
  { key: "median_umi_per_cell", short: "UMIs" },
  { key: "sequencing_saturation", short: "Saturation" },
  { key: "valid_barcodes", short: "Valid BC" },
  { key: "conf_transcriptome", short: "Mapped" },
  { key: "q30_rna_read", short: "Q30 RNA" },
  { key: "frac_reads_in_cells", short: "In cells" },
];

/** Cells in the dataset (blue) vs the raw matrix's other barcodes (slate). */
export const CELL_COLOR = rgbCss(CATEGORICAL_COLORS[4]!);
export const OTHER_COLOR = "#94a3b8";
export const KNEE_COLOR = rgbCss(CATEGORICAL_COLORS[0]!);
export const INFLECTION_COLOR = rgbCss(CATEGORICAL_COLORS[2]!);

export const LEVEL_STYLE: Record<SeqFlag["level"], string> = {
  error: "border-red-200 bg-red-50 text-red-700",
  warn: "border-amber-200 bg-amber-50 text-amber-800",
  info: "border-sky-200 bg-sky-50 text-sky-800",
};
export const LEVEL_WORD: Record<SeqFlag["level"], string> = {
  error: "Error",
  warn: "Warning",
  info: "Note",
};

const FRACTION_KEYS = new Set<SeqMetricKey>([
  "valid_barcodes", "valid_umis", "sequencing_saturation", "q30_barcode", "q30_rna_read", "q30_umi",
  "mapped_genome", "conf_genome", "conf_exonic", "conf_intronic", "conf_intergenic", "conf_transcriptome",
  "conf_probe_set", "antisense", "frac_reads_in_cells",
]);

export function isFraction(key: SeqMetricKey): boolean {
  return FRACTION_KEYS.has(key);
}

/** "54,502" or "70.8%"; "—" when absent. */
export function formatMetric(key: SeqMetricKey, v: number | null | undefined): string {
  if (v == null || !Number.isFinite(v)) return "—";
  return isFraction(key) ? `${(v * 100).toFixed(1)}%` : Math.round(v).toLocaleString();
}

export function umiText(v: number | null | undefined): string {
  return v == null || !Number.isFinite(v) ? "—" : Math.round(v).toLocaleString();
}

/** The worst flag on one metric, if any. */
export function flagFor(flags: SeqFlag[], key: SeqMetricKey): SeqFlag | undefined {
  const rank = { error: 0, warn: 1, info: 2 } as const;
  return flags.filter((f) => f.metric === key).sort((a, b) => rank[a.level] - rank[b.level])[0];
}

/** Tooltip of a flag: what crossed which line, both lines, and the source. */
export function flagTip(f: SeqFlag): string {
  const side = f.direction === "below" ? "below" : "above";
  const lines =
    f.error == null
      ? `flagged ${side} ${formatMetric(f.metric, f.warn)}`
      : `warning ${side} ${formatMetric(f.metric, f.warn)}, error ${side} ${formatMetric(f.metric, f.error)}`;
  return `${f.title}: ${formatMetric(f.metric, f.value)} (${lines}; ${f.ideal}). Source: ${f.source}.`;
}

/** Header tooltip: the backend's description of the metric. */
export function metricTip(info: SeqMetricInfo[] | undefined, key: SeqMetricKey): string {
  return info?.find((m) => m.key === key)?.tip ?? "";
}

export function metricLabel(info: SeqMetricInfo[] | undefined, key: SeqMetricKey): string {
  return info?.find((m) => m.key === key)?.label ?? key;
}

interface RankTrace {
  type: "scatter";
  mode: "markers";
  name: string;
  x: number[];
  y: number[];
  marker: { color: string; size: number };
  hovertemplate: string;
}

/** Two traces of the downsampled rank curve: the dataset's cells and the other barcodes. */
export function rankTraces(k: KneeSummary, cellsLabel = "Cells in the dataset"): RankTrace[] {
  const cells = { x: [] as number[], y: [] as number[] };
  const other = { x: [] as number[], y: [] as number[] };
  k.curve.rank.forEach((r, i) => {
    const into = k.curve.kept[i] ? cells : other;
    into.x.push(r);
    into.y.push(k.curve.umi[i]!);
  });
  const hover = "rank %{x:,}<br>%{y:,} UMIs<extra>%{fullData.name}</extra>";
  return [
    { type: "scatter", mode: "markers", name: "Other barcodes", ...other, marker: { color: OTHER_COLOR, size: 4 }, hovertemplate: hover },
    { type: "scatter", mode: "markers", name: cellsLabel, ...cells, marker: { color: CELL_COLOR, size: 4 }, hovertemplate: hover },
  ];
}

/** Dashed horizontal lines at the knee and inflection, labelled with their UMI totals. */
export function kneeShapes(k: KneeSummary): { shapes: object[]; annotations: object[] } {
  const shapes: object[] = [];
  const annotations: object[] = [];
  const line = (y: number, color: string, text: string) => {
    shapes.push({ type: "line", xref: "paper", x0: 0, x1: 1, yref: "y", y0: y, y1: y, line: { color, width: 1.5, dash: "dash" } });
    annotations.push({ xref: "paper", x: 1, xanchor: "right", yref: "y", y: Math.log10(y), yanchor: "bottom", text, showarrow: false, font: { size: 10, color } });
  };
  if (k.knee != null) line(k.knee, KNEE_COLOR, `knee ${umiText(k.knee)} UMIs`);
  if (k.inflection != null) line(k.inflection, INFLECTION_COLOR, `inflection ${umiText(k.inflection)} UMIs`);
  return { shapes, annotations };
}

/** One plain sentence on how the cells sit on the curve. */
export function kneeSentence(s: SeqSample): string {
  const k = s.knee;
  if (!k) return "";
  const who = s.cells_source === "ingest" ? "The filtered matrix kept" : "The dataset holds";
  const parts: string[] = [];
  parts.push(`${who} ${k.n_cells.toLocaleString()} cells; ${k.n_cells_matched.toLocaleString()} are in the raw matrix`);
  if (k.knee != null && k.n_above_knee != null && k.n_above_inflection != null) {
    parts.push(
      `${k.n_above_knee.toLocaleString()} barcodes lie above the knee and ${k.n_above_inflection.toLocaleString()} above the inflection`,
    );
  }
  let text = parts.join(", ") + ".";
  if (k.cells_umi_min != null) {
    text += ` The cells' UMI floor is ${umiText(k.cells_umi_min)} (median ${umiText(k.cells_umi_median)}), and they hold ${
      k.frac_umis_in_cells != null ? `${(k.frac_umis_in_cells * 100).toFixed(1)}%` : "—"
    } of the matrix's UMIs.`;
  }
  if (k.n_cells_below_knee) {
    text += ` ${k.n_cells_below_knee.toLocaleString()} cells are below the knee: Cell Ranger's caller (EmptyDrops-style since 3.0) keeps low-RNA cells the knee alone would drop.`;
  }
  if (k.knee_note) text += ` ${k.knee_note}`;
  return text;
}

/** The plot caption (without the n). */
export function curveCaption(k: KneeSummary): string {
  return (
    `Each point is a barcode of the raw matrix, ranked by its UMI total (${k.feature_types ?? "gene expression"}), on log-log axes; ` +
    `${k.curve.rank.length.toLocaleString()} of ${k.n_barcodes.toLocaleString()} barcodes with UMIs are drawn ` +
    `(per rank bin, the first cell and the first other barcode). Dashed lines: knee and inflection, as DropletUtils barcodeRanks computes them.`
  );
}

/** Columns a merged dataset's samples are told apart by: categorical, 2-50 values. */
export function sampleColumns(cols: { name: string; dtype: string; n_unique: number; values?: string[] }[] | undefined) {
  return (cols ?? []).filter(
    (c) => (c.dtype === "category" || c.dtype === "object") && c.n_unique >= 2 && c.n_unique <= 50 && (c.values?.length ?? 0) > 0,
  );
}

/** Files the attach form accepts, for the picker. */
export const ATTACH_ACCEPT = ".csv,.csv.gz,.h5,.mtx,.mtx.gz,.tsv,.tsv.gz,.gz";
