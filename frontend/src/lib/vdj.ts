/** Wording and plot data for the Clonotypes subtab (backend core/vdj.py). */
import {
  SUFFIX_COLUMN,
  type ClonotypeChain,
  type Diversity,
  type DiversityGroup,
  type DiversityMetric,
  type Overlap,
  type SampleChoice,
  type Usage,
  type VdjFile,
  type VdjSummary,
} from "@/api/vdj";

export const METRICS: { id: DiversityMetric; label: string; info: string }[] = [
  {
    id: "richness",
    label: "Richness",
    info: "The number of distinct clonotypes among the group's cells with a clonotype. It grows with the number of cells sampled.",
  },
  {
    id: "shannon",
    label: "Shannon",
    info: "Shannon entropy, -sum p ln p (natural log), over each clonotype's share of the group's cells. Higher is more diverse; it also grows with the number of cells sampled.",
  },
  {
    id: "simpson",
    label: "Gini-Simpson",
    info: "1 - sum p^2: the chance that two cells drawn at random (with replacement) belong to different clonotypes. Near 1 for a diverse repertoire.",
  },
  {
    id: "clonality",
    label: "Clonality",
    info: "1 - Shannon / ln(richness), one minus Pielou's evenness: 0 when every clonotype has the same number of cells, towards 1 when a few dominate. Needs 2 or more clonotypes.",
  },
  {
    id: "gini",
    label: "Gini",
    info: "Gini coefficient of the clone sizes: 0 when all clonotypes have the same number of cells, towards 1 when a few hold most of them.",
  },
];

export const STATUS_INFO: Record<string, string> = {
  paired: "One VJ chain (TRA, TRG, IGK or IGL) and one VDJ chain (TRB, TRD or IGH)",
  "paired + extra chain": "A pair plus a second VJ and/or VDJ chain (dual TCR-alpha is common)",
  "single chain": "Chains of one arm only (an orphan alpha, beta, heavy or light chain)",
  multichain: "More than two chains of one arm: likely a doublet; no clonotype",
  ambiguous: "T- and B-cell chains in one cell: likely a doublet; no clonotype",
  "no usable chain": "Contigs, but none productive, high-confidence, in a called cell and with a CDR3",
  "no contig": "Its sample's contigs were attached; this barcode had none",
  "not profiled": "No contig table is attached for this cell's sample",
};

/** "barcode suffix" for the suffix choice, else the column name. */
export function choiceLabel(choice: SampleChoice): string {
  return choice.name === SUFFIX_COLUMN ? (choice.label ?? "barcode suffix") : choice.name;
}

/** A barcode suffix tag ("*-1:s2") as the reader sees it ("…-1:s2"). */
export function valueLabel(column: string, value: string): string {
  return column === SUFFIX_COLUMN ? value.replace("*", "…") : value;
}

/** True when the cells come from several libraries: a sample must be named. */
export function needsSample(summary: VdjSummary): boolean {
  return summary.sample_choices.some((c) => c.name === SUFFIX_COLUMN);
}

const n = (v: number) => v.toLocaleString();

/** The join both ways, in plain sentences. */
export function matchSentences(s: VdjSummary): string[] {
  const m = s.match;
  if (!m) return [];
  const out = [
    `${n(m.cells_with_chain)} of ${n(s.n_cells)} cells have a usable TCR or BCR chain.`,
    `${n(m.cells_without_contigs)} profiled cells had no contig` +
      (m.cells_not_profiled > 0 ? `; ${n(m.cells_not_profiled)} cells are in samples with no contig table attached.` : "."),
  ];
  if (m.barcodes_without_cells > 0)
    out.push(
      `${n(m.barcodes_without_cells)} contig barcodes match no cell of the dataset ` +
        `(${n(m.barcodes_without_cells_usable)} of them with a usable chain): cells the expression data did not keep.`,
    );
  if (m.ambiguous_barcodes > 0)
    out.push(`${n(m.ambiguous_barcodes)} barcodes matched more than one cell and were left out.`);
  return out;
}

/** One attached file: what it held, what was kept and what matched. */
export function fileSentence(f: VdjFile): string {
  const where = f.sample_column
    ? ` for ${f.sample_column === SUFFIX_COLUMN ? "barcode suffix" : f.sample_column} = ${valueLabel(f.sample_column, f.sample ?? "")}`
    : "";
  const how = f.matched_core > 0 ? ` (${n(f.matched_core)} by barcode core in decorated cell names)` : "";
  return (
    `${f.name}${where}: ${n(f.n_kept_chains)} of ${n(f.n_contigs)} contigs kept; ` +
    `${n(f.matched)} of ${n(f.n_barcodes)} barcodes matched cells${how}, ${n(f.unmatched)} did not` +
    (f.ambiguous ? `, ${n(f.ambiguous)} ambiguous` : "") +
    "."
  );
}

const DROP_LABELS: Record<string, string> = {
  duplicate: "repeated",
  no_barcode: "without a barcode",
  not_cell: "not in a called cell",
  low_confidence: "low confidence",
  non_productive: "non-productive",
  unknown_locus: "no IMGT locus",
  no_cdr3: "without a CDR3",
};

/** "12 non-productive, 3 low confidence", or null when nothing was dropped. */
export function droppedText(dropped: Record<string, number>): string | null {
  const parts = Object.entries(dropped)
    .filter(([, v]) => v > 0)
    .map(([k, v]) => `${n(v)} ${DROP_LABELS[k] ?? k}`);
  return parts.length ? parts.join(", ") : null;
}

/** "TRA CAVSDLEPNSSASKIIF / TRB CASSLGQAYEQYF". */
export function chainText(chains: readonly ClonotypeChain[]): string {
  return chains.map((c) => `${c.locus} ${c.cdr3}`).join(" / ");
}

/** The V and J genes of a clonotype's chains, for its tooltip. */
export function genesText(chains: readonly ClonotypeChain[]): string {
  return chains.map((c) => `${c.locus}: ${[c.v, c.j].filter(Boolean).join(" + ") || "genes not given"}`).join("; ");
}

/** A group's value of one metric: rarefied mean (and SD) when rarefied. */
export function metricValue(
  g: DiversityGroup,
  metric: DiversityMetric,
  rarefied: boolean,
): { value: number | null; sd: number | null } {
  if (rarefied && g.rarefied) return { value: g.rarefied[metric] ?? null, sd: g.rarefied[`${metric}_sd`] ?? null };
  return { value: g[metric], sd: null };
}

/** Bars of one metric per group: the groups with a value, their n in the hover. */
export function diversityBars(d: Diversity, metric: DiversityMetric) {
  const label = METRICS.find((m) => m.id === metric)?.label ?? metric;
  const rows = d.groups
    .map((g) => ({ g, ...metricValue(g, metric, d.rarefy) }))
    .filter((r) => r.value !== null);
  return {
    x: rows.map((r) => r.g.name),
    y: rows.map((r) => r.value as number),
    sd: rows.map((r) => r.sd ?? 0),
    hover: rows.map(
      (r) =>
        `<b>${r.g.name}</b><br>${label} ${fmt(r.value as number)}` +
        (r.sd !== null ? ` (SD ${fmt(r.sd)} over ${d.n_draws} draws)` : "") +
        `<br>${n(r.g.n_with_clonotype)} cells with a clonotype of ${n(r.g.n_cells)}`,
    ),
    left_out: d.groups.filter((g) => metricValue(g, metric, d.rarefy).value === null).map((g) => g.name),
  };
}

function fmt(v: number): string {
  return Math.abs(v) >= 100 ? v.toFixed(0) : v.toFixed(3);
}

/** Hover text per overlap cell: the measure, shared clonotypes and the two n. */
export function overlapHover(o: Overlap): string[][] {
  const label = o.metric === "morisita_horn" ? "Morisita-Horn" : "Jaccard";
  return o.matrix.map((row, r) =>
    row.map(
      (v, c) =>
        `<b>${o.groups[r]}</b> and <b>${o.groups[c]}</b><br>${label} ${v === null ? "n/a" : v.toFixed(3)}` +
        `<br>${n(o.shared[r]![c]!)} shared clonotypes<br>` +
        `${n(o.n_clonotypes[r]!)} and ${n(o.n_clonotypes[c]!)} clonotypes; ${n(o.n_cells[r]!)} and ${n(o.n_cells[c]!)} cells`,
    ),
  );
}

/** Hover text per usage cell: share and count of the group's clonotypes or cells. */
export function usageHover(u: Usage): string[][] {
  return u.share.map((row, r) =>
    row.map(
      (v, c) =>
        `<b>${u.groups[r]}</b>, ${u.genes[c]}<br>${(v * 100).toFixed(1)}% of ${n(u.n[r]!)} ${u.count}` +
        ` (${n(u.counts[r]![c]!)})`,
    ),
  );
}
