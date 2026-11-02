/** Words and numbers for the motif card (components/unified/AtacMotifsCard.tsx). */
import type { MotifState } from "@/api/atacMotifs";
import { formatDuration } from "@/lib/hvgSensitivity";

export const motifLabel = (motif: string) => `${motif} chromVAR z`;

function size(bytes: number): string {
  return bytes >= 1e9 ? `${(bytes / 1e9).toFixed(1)} GB` : `${Math.max(1, Math.round(bytes / 1e6)).toLocaleString()} MB`;
}

/** What the first scan fetches, or null when both files are here. */
export function scanDownloadText(state: MotifState): string | null {
  const f = state.files;
  if (!f) return null;
  const parts: string[] = [];
  if (!f.genome.downloaded) parts.push(`the ${f.genome.source} sequence (${size(f.genome.size)})`);
  if (!f.jaspar.downloaded) parts.push(`the JASPAR 2024 motifs (${size(f.jaspar.size)})`);
  return parts.length ? `The first run downloads ${parts.join(" and ")}, once.` : null;
}

/** A switch being off stops the scan before it starts: say so instead of letting it fail. */
export function scanBlockedText(state: MotifState): string | null {
  const f = state.files;
  if (!f) return null;
  if (!f.genome.downloaded && !f.genome_auto) {
    return `The ${f.genome.source} sequence is not on this machine and GENOME_AUTO_DOWNLOAD is off: place it in the data volume's genome folder, or turn the switch on.`;
  }
  if (!f.jaspar.downloaded && !f.jaspar_auto) {
    return "The JASPAR 2024 motifs are not on this machine and TF_MOTIFS_AUTO_DOWNLOAD is off: place them in the data volume, or turn the switch on.";
  }
  return null;
}

export function scanEstimateText(state: MotifState): string | null {
  const s = state.scan;
  if (!s) return null;
  return `About ${formatDuration(s.estimate_seconds)} and ${size(s.estimate_bytes)} beyond the open dataset, for ${s.n_peaks.toLocaleString()} peaks (${(s.n_bases / 1e6).toFixed(0)} Mb) and 879 motifs; kept, so a dataset with the same peaks costs nothing.`;
}

export function chromvarEstimateText(state: MotifState): string | null {
  const e = state.chromvar_estimate;
  if (!e) return null;
  return `About ${formatDuration(e.estimate_seconds)}, ${size(e.estimate_bytes)} beyond the open dataset and ${size(e.disk_bytes)} on disk for ${e.n_cells.toLocaleString()} cells x 879 motifs.`;
}

/** The fold enrichment as text: null (0 in the background and none observed, or none in the background) is a dash. */
export function formatFold(fold: number | null | undefined): string {
  return fold === null || fold === undefined || !Number.isFinite(fold) ? "–" : fold.toFixed(2);
}

/** The TF's detection in the group: a share of cells, or a dash when it is not measured. */
export function formatPct(pct: number | null | undefined): string {
  return pct === null || pct === undefined ? "–" : `${Math.round(100 * pct)}%`;
}
