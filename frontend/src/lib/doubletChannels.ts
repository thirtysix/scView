import type { DoubletChannelCalls, DoubletRateSuggestion, DoubletSummary } from "@/api/qc";
import { pct } from "@/lib/qcThresholds";

/** Where the channels came from, for a sentence ("the -N suffix of the cell barcodes"). */
const REASON_WORDS: Record<string, string> = {
  requested: "the column chosen in the doublet step",
  requested_all: "the doublet step's choice",
  channel_column: "the column naming the capture channel",
  barcode_prefix: "the prefix of the cell barcodes",
  barcode_suffix: "the -N suffix of the cell barcodes",
  barcode_prefix_suffix: "the prefix and -N suffix of the cell barcodes",
  donor: "the donor / sample column",
  batch: "the batch column",
};

const plural = (n: number, word: string) => `${n.toLocaleString()} ${word}${n === 1 ? "" : "s"}`;

/** "0.5-0.7%" (or one value) over the rates given. */
export function rateRange(rates: number[]): string {
  if (!rates.length) return "—";
  const lo = Math.min(...rates);
  const hi = Math.max(...rates);
  return pct(lo) === pct(hi) ? pct(lo) : `${pct(lo).replace("%", "")}-${pct(hi)}`;
}

/** Which channels Scrublet runs in: "133 capture channels, from gem_well", "2 capture
 *  channels (CTRL, STIM), from the prefix of the cell barcodes", "all cells as one
 *  channel (no capture channel found)". */
export function channelsText(s: DoubletRateSuggestion): string {
  if (!s.column) {
    return s.reason === "requested_all"
      ? "all cells as one channel, as chosen"
      : "all cells as one channel (no capture channel found)";
  }
  const n = s.n_channels ?? s.channels.length;
  const names = n <= 4 ? ` (${s.channels.map((c) => c.name).join(", ")})` : "";
  const from = s.reason === "channel_column" || s.reason === "requested"
    ? s.column
    : (REASON_WORDS[s.reason ?? ""] ?? s.column);
  return `${plural(n, "capture channel")}${names}, from ${from}`;
}

/** Under the expected-rate input: what the doublet step does without a rate. */
export function suggestionText(s: DoubletRateSuggestion): string {
  const rule = "a rule of thumb, ~0.8 % per 1,000 cells captured in one channel.";
  if (!s.column) return `Default ${pct(s.rate)}, from ${channelsText(s)}: ${rule}`;
  const rates = s.channels.filter((c) => c.scored).map((c) => c.rate);
  return `Default: each channel's own, ${rateRange(rates)}, in ${channelsText(s)}: ${rule} A rate typed here applies to every channel.`;
}

/** Cells left out of Scrublet, when any: why, and how to score them. */
export function unscoredText(n: number, small: number, minCells: number): string | null {
  if (!n) return null;
  const parts = [`${plural(n, "cell")} not scored, so never called doublets`];
  if (small) {
    parts.push(
      `${small.toLocaleString()} of them in channels under ${minCells.toLocaleString()} cells, too few for Scrublet ` +
        "(pick coarser channels, or all cells as one, in the doublet step to score them)",
    );
  }
  if (n > small) parts.push(`${(n - small).toLocaleString()} with under 3 genes or no channel`);
  return `${parts.join("; ")}.`;
}

/** The line that says the calls are per channel, on the doublet card. */
export function perChannelText(s: DoubletSummary): string | null {
  const rows = s.channels ?? [];
  if (!s.per_channel || !rows.length) return null;
  const from = s.channel_reason && REASON_WORDS[s.channel_reason]
    ? (s.channel_reason === "channel_column" || s.channel_reason === "requested"
        ? s.channel_column ?? ""
        : REASON_WORDS[s.channel_reason])
    : (s.channel_column ?? "");
  const scored = rows.filter((r) => r.status === "scored");
  return (
    `Scrublet ran in each of ${plural(rows.length, "capture channel")} alone (from ${from}), each with its own ` +
    `expected rate (${rateRange(scored.map((r) => r.expected_rate))}) and its own threshold: ` +
    "the default calls are per channel. A threshold applied here is one threshold over every channel's scores."
  );
}

/** A channel whose automatic threshold called almost nothing, in one sentence:
 *  "STIM: Scrublet's automatic threshold (0.54) called 9 of 7,263 cells (0.12%), where
 *  about 5.8% are expected." */
export function fewCalledText(r: DoubletChannelCalls): string {
  const called = r.scrublet_called ?? 0;
  const t = r.scrublet_threshold != null ? ` (${String(Number(r.scrublet_threshold.toFixed(4)))})` : "";
  const share = r.n_scored ? called / r.n_scored : 0;
  return (
    `${r.name}: Scrublet's automatic threshold${t} called ${called.toLocaleString()} of ` +
    `${r.n_scored.toLocaleString()} cells (${(100 * share).toFixed(2)}%), where about ${pct(r.expected_rate)} are expected.`
  );
}

/** The tooltip of that warning: what it means and what to do. */
export function fewCalledHelp(share: number, minCells: number): string {
  return (
    `The automatic threshold may have missed the doublets in this channel: it called under ${(100 * share).toLocaleString()}% ` +
    `of the cells in a channel of ${minCells.toLocaleString()} or more, where the rule of thumb expects several per cent. ` +
    "Scrublet places its threshold in the dip between the singlets' scores and the doublets'; when it finds no clear dip " +
    "it can land far out in the tail. Look at this channel's histogram: if its scores have a second peak or a long tail " +
    "below the threshold, set a threshold by hand (it applies to every channel's scores)."
  );
}

/** A channel's status for its table cell. */
export function statusText(r: DoubletChannelCalls, minCells: number): string | null {
  if (r.status === "too_small") return `under ${minCells} cells: not scored`;
  if (r.status === "failed") return "Scrublet failed: not scored";
  if (r.status === "scored" && r.scrublet_threshold == null) return "no threshold found: no calls";
  return null;
}
