/**
 * "Plot reliability" (C075): per cell, the share of its graph neighbours (the kNN
 * graph the clustering used) that are also among its k nearest cells on the plot
 * (backend core/embedding_reliability.py). The helpers the toolbar control, the
 * scatter legend and session restore share.
 */
import {
  getEmbeddingReliability,
  type ReliabilityGroup,
  type ReliabilityResponse,
  type ReliabilitySummary,
} from "@/api/reliability";
import { prettyEmbeddingLabel } from "@/lib/formatting";
import { useUnifiedViewStore } from "@/stores/unifiedViewStore";

/** What a reliability overlay shows: enough to fetch it again and caption it. */
export interface ReliabilityRef {
  embedding: string;
  /** The PCA pair plotted, when not the first two */
  dims?: [number, number];
  k: number;
  chance: number;
  /** Mean score after one seeded shuffle of the coordinates */
  random: number | null;
}

/** "Neighbour preservation (UMAP)", "(UMAP 3D)", "(PCA, PC3 vs PC4)". */
export function reliabilityName(embedding: string, nDims = 2, dims?: [number, number]): string {
  let label = prettyEmbeddingLabel(embedding);
  if (nDims === 3) label += " 3D";
  if (dims) label += `, PC${dims[0] + 1} vs PC${dims[1] + 1}`;
  return `Neighbour preservation (${label})`;
}

/** Two decimals; two significant digits below 0.1, so a chance of 0.0011 is not "0.00". */
export function fmtScore(v: number | null | undefined): string {
  if (v == null || !Number.isFinite(v)) return "n/a";
  if (v === 0) return "0";
  return Math.abs(v) >= 0.1 ? v.toFixed(2) : String(Number(v.toPrecision(2)));
}

export function fmtShare(v: number | null | undefined): string {
  if (v == null || !Number.isFinite(v)) return "n/a";
  const p = v * 100;
  if (p > 0 && p < 1) return "<1%";
  return `${Math.round(p)}%`;
}

/** The per-cell values as the scatter takes them (NaN = not scored, drawn grey). */
export function toOverlayValues(scores: (number | null)[]): Float32Array {
  const out = new Float32Array(scores.length);
  for (let i = 0; i < scores.length; i++) {
    const v = scores[i];
    out[i] = v == null ? NaN : v;
  }
  return out;
}

export function refOf(summary: ReliabilitySummary, dims?: [number, number]): ReliabilityRef {
  return {
    embedding: summary.embedding,
    ...(dims ? { dims } : {}),
    k: summary.k,
    chance: summary.chance,
    random: summary.shuffled.mean,
  };
}

/** Same embedding and the same plotted pair (undefined = the default first two). */
export function sameView(ref: ReliabilityRef, embedding: string, dims?: [number, number]): boolean {
  const a = ref.dims;
  return ref.embedding === embedding && (a?.[0] ?? 0) === (dims?.[0] ?? 0) && (a?.[1] ?? 1) === (dims?.[1] ?? 1);
}

/** The response without its per-cell values (what the table keeps). */
export function summaryOf(res: ReliabilityResponse): ReliabilitySummary {
  const out: Partial<ReliabilityResponse> = { ...res };
  delete out.scores;
  return out as ReliabilitySummary;
}

/** Whether a summary describes the overlay on screen (same plot, same k). */
export function summaryMatches(ref: ReliabilityRef, s: ReliabilitySummary): boolean {
  if (s.embedding !== ref.embedding || s.k !== ref.k) return false;
  if (s.n_dims === 3) return true;
  const want = ref.dims ?? [0, 1];
  return s.dims[0] === want[0] && s.dims[1] === want[1];
}

/** Put a fetched result on the main scatter as a score overlay. */
export function applyReliability(datasetId: string, res: ReliabilityResponse, dims?: [number, number]): void {
  const store = useUnifiedViewStore.getState();
  const name = reliabilityName(res.embedding, res.n_dims, dims);
  store.setOverlayValues(toOverlayValues(res.scores));
  store.setOverlayLabel(name);
  store.setScatterOverlay({ type: "score", name, reliability: refOf(res, dims) });
  store.setOverlayDatasetId(datasetId);
}

/** Fetch the scores for one embedding and show them (session restore, the toolbar). */
export async function showReliability(
  datasetId: string,
  embedding: string,
  dims?: [number, number],
  groupBy?: string,
): Promise<ReliabilityResponse> {
  const res = await getEmbeddingReliability(datasetId, embedding, { dims, groupBy });
  applyReliability(datasetId, res, dims);
  return res;
}

/** The legend's line under the colour bar: the scale is fixed, and what chance is. */
export function reliabilityLegendNote(ref: ReliabilityRef): string {
  const random = ref.random != null ? `; random placement about ${fmtScore(ref.random)}` : "";
  return (
    `Fixed 0 to 1 scale, not clipped: 1 = all ${ref.k} graph neighbours are beside the cell on the plot. ` +
    `Chance ${fmtScore(ref.chance)}${random}.`
  );
}

/** What the numbers mean, in one paragraph, with the n. */
export function reliabilityCaption(s: ReliabilitySummary): string {
  const plot = s.n_dims === 3 ? "this 3D plot" : "this plot";
  const graph =
    s.graph.source === "stored"
      ? "the kNN graph the clustering used"
      : `an exact kNN in ${s.graph.use_rep ?? "the PCA"} (no graph is stored)`;
  const scored = s.sampled
    ? `${s.n_scored.toLocaleString()} of ${s.n_cells.toLocaleString()} cells (a seeded sample; the rest are grey)`
    : `${s.n_cells.toLocaleString()} cells`;
  const o = s.overall;
  const random = s.shuffled.mean != null ? `; random placement would score about ${fmtScore(s.shuffled.mean)}` : "";
  return (
    `Each cell: the share of its ${s.k} graph neighbours (${graph}) that are also among its ${s.k} nearest cells on ${plot}. ` +
    `Median ${fmtScore(o.median)} over ${scored}; ${fmtShare(o.share_low)} are low (${s.low_reason}). ` +
    `Chance is ${fmtScore(s.chance)}${random}. ` +
    `A low value means the cells around this one on the plot are mostly not its real neighbours: don't read its position closely.`
  );
}

export type GroupSortKey = "group" | "n" | "median" | "share_low";

/** Sorted rows; "worst" is the server's order (lowest median first). */
export function sortGroups(groups: ReliabilityGroup[], key: GroupSortKey | null, desc: boolean): ReliabilityGroup[] {
  if (!key) return groups;
  const rows = [...groups];
  rows.sort((a, b) => {
    const c =
      key === "group"
        ? a.group.localeCompare(b.group, undefined, { numeric: true })
        : (a[key] as number) - (b[key] as number);
    return desc ? -c : c;
  });
  return rows;
}

/** Indices of the cells whose code is `value`'s position in `categories`. */
export function groupCells(codes: ArrayLike<number>, categories: string[], value: string): Set<number> {
  const code = categories.indexOf(value);
  const out = new Set<number>();
  if (code < 0) return out;
  for (let i = 0; i < codes.length; i++) if (codes[i] === code) out.add(i);
  return out;
}

/** The error text of an `apiFetch` failure: the server's `detail` when it sent one. */
export function errorText(e: unknown): string {
  const msg = e instanceof Error ? e.message.replace(/^API error \d+: /, "") : String(e);
  try {
    const detail = JSON.parse(msg).detail;
    return typeof detail === "string" ? detail : msg;
  } catch {
    return msg;
  }
}
