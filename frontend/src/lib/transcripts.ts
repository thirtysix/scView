/** Transcript dots on the tissue plot (backend core/transcripts.py). */

/** One colour per chip, in order: fixed, so they stay apart from the cluster palette under them. */
export const TRANSCRIPT_COLORS: readonly (readonly [number, number, number])[] = [
  [230, 25, 75],
  [60, 180, 75],
  [255, 225, 25],
  [0, 130, 200],
  [245, 130, 48],
  [145, 30, 180],
  [70, 240, 240],
  [240, 50, 230],
];

export const DOT_SIZE_RANGE: readonly [number, number] = [1, 6];
export const DEFAULT_DOT_SIZE = 1;
/** Over this many dots in a view a dark rim costs more than it shows (14 fps at 300,000 with one). */
export const MAX_RIMMED_DOTS = 30_000;

export function transcriptColorCss(index: number): string {
  const [r, g, b] = TRANSCRIPT_COLORS[index % TRANSCRIPT_COLORS.length]!;
  return `rgb(${r}, ${g}, ${b})`;
}

/** Points of a view: API coordinates turned into the tissue world (x, -y). */
export interface TranscriptPoints {
  n: number;
  /** World x, y per point */
  positions: Float32Array;
  /** RGBA per point */
  colors: Uint8Array;
}

/** `GET /datasets/{id}/spatial/transcripts`, little-endian: int32 n, uint8 gene[n], padding to 4, float32 x[n], y[n]. */
export function parseTranscripts(buffer: ArrayBuffer): TranscriptPoints {
  const n = new Int32Array(buffer, 0, 1)[0]!;
  const pad = (4 - ((4 + n) % 4)) % 4;
  const xAt = 4 + n + pad;
  const expected = xAt + 8 * n;
  if (buffer.byteLength !== expected) {
    throw new Error(`The transcripts answer is ${buffer.byteLength} bytes, not the ${expected} it announces.`);
  }
  const gene = new Uint8Array(buffer, 4, n);
  const x = new Float32Array(buffer, xAt, n);
  const y = new Float32Array(buffer, xAt + 4 * n, n);
  const positions = new Float32Array(n * 2);
  const colors = new Uint8Array(n * 4);
  for (let i = 0; i < n; i++) {
    positions[i * 2] = x[i]!;
    positions[i * 2 + 1] = -y[i]!;
    const [r, g, b] = TRANSCRIPT_COLORS[gene[i]! % TRANSCRIPT_COLORS.length]!;
    colors[i * 4] = r;
    colors[i * 4 + 1] = g;
    colors[i * 4 + 2] = b;
    colors[i * 4 + 3] = 255;
  }
  return { n, positions, colors };
}

/** The caption of the dots: how many of those in the view, when thinned. */
export function shownText(shown: number, total: number): string {
  const count = `${total.toLocaleString()} transcript${total === 1 ? "" : "s"} in view`;
  if (total <= shown) return `${count}, all shown.`;
  return `Showing ${shown.toLocaleString()} of ${count}, the same share of each gene, evenly: zoom in for every one.`;
}

export function sizeText(bytes: number): string {
  if (bytes >= 1024 ** 3) return `${(bytes / 1024 ** 3).toFixed(1)} GB`;
  return `${Math.max(1, Math.round(bytes / 1024 ** 2))} MB`;
}

export function timeText(seconds: number): string {
  return seconds < 90 ? `${Math.max(1, Math.round(seconds))} s` : `${Math.round(seconds / 60)} min`;
}
