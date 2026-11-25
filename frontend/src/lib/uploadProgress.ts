/**
 * Upload progress maths for the Data panel's progress card: percent, bytes sent of the total,
 * speed and time left. Speed comes from the last few seconds of data, and there is no time left
 * until a few seconds of data have gone (the first second of an upload fills buffers and says
 * nothing about the link).
 */

/** Progress over a whole upload (every file, every request): bytes of the files sent, and the
 *  file going now (1-based index). */
export interface UploadProgress {
  loaded: number;
  total: number;
  file: string;
  fileIndex: number;
  fileCount: number;
}

/** One reading of the bytes sent, at `t` ms. */
export interface Sample {
  t: number;
  loaded: number;
}

/** No speed or time left before this much time with data. */
export const RATE_AFTER_MS = 3000;
/** Speed over the last this many ms, so it follows a link that slows down or speeds up. */
export const RATE_WINDOW_MS = 8000;

/** Whole percent sent: 100 only once every byte is sent. */
export function percentSent(loaded: number, total: number): number {
  if (!(total > 0)) return 0;
  if (loaded >= total) return 100;
  return Math.min(99, Math.max(0, Math.floor((loaded / total) * 100)));
}

/** The index of the file being sent once `loaded` bytes of the set have gone (files in order). */
export function fileAt(sizes: number[], loaded: number): number {
  let end = 0;
  for (let i = 0; i < sizes.length; i++) {
    end += sizes[i]!;
    if (loaded < end) return i;
  }
  return Math.max(0, sizes.length - 1);
}

/** The samples with a new one added, older ones than the speed window needs dropped (the first
 *  sample is kept: it says how long data has been going). */
export function addSample(samples: Sample[], s: Sample): Sample[] {
  const next = [...samples, s];
  const cut = s.t - RATE_WINDOW_MS;
  // keep the newest sample at or before the window start, so the window is always full
  let start = 1;
  while (start + 1 < next.length && next[start + 1]!.t <= cut) start++;
  return next.length > 2 ? [next[0]!, ...next.slice(start)] : next;
}

/** Bytes per second over the speed window, or null before RATE_AFTER_MS of data. */
export function rateOf(samples: Sample[]): number | null {
  if (samples.length < 2) return null;
  const first = samples[0]!;
  const last = samples[samples.length - 1]!;
  if (last.t - first.t < RATE_AFTER_MS) return null;
  const cut = last.t - RATE_WINDOW_MS;
  let from = first;
  for (const s of samples) if (s.t <= cut) from = s;
  const seconds = (last.t - from.t) / 1000;
  if (seconds <= 0) return null;
  return Math.max(0, (last.loaded - from.loaded) / seconds);
}

/** Seconds left at the current speed, or null when the speed is not known (or zero). */
export function secondsLeft(loaded: number, total: number, rate: number | null): number | null {
  if (rate == null || rate <= 0) return null;
  return Math.max(0, total - loaded) / rate;
}

/** "a few seconds", "about 25 s", "about 4 min", "about 1 h 20 min": rounded up, coarser as it grows. */
export function formatTimeLeft(seconds: number): string {
  if (seconds < 10) return "a few seconds";
  if (seconds < 60) {
    const s = Math.ceil(seconds / 5) * 5;
    return s >= 60 ? "about 1 min" : `about ${s} s`;
  }
  if (seconds < 3600) {
    const minutes = Math.ceil(seconds / 60);
    return minutes >= 60 ? "about 1 h" : `about ${minutes} min`;
  }
  const minutes = Math.ceil(seconds / 300) * 5;
  const h = Math.floor(minutes / 60);
  const m = minutes % 60;
  return m === 0 ? `about ${h} h` : `about ${h} h ${m} min`;
}

/** A size unit that suits the total: [label, bytes per unit, decimals]. Decimal units, as file managers show sizes. */
function unitFor(total: number): [string, number, number] {
  if (total >= 1e9) return ["GB", 1e9, 2];
  if (total >= 1e8) return ["MB", 1e6, 0];
  if (total >= 1e6) return ["MB", 1e6, 1];
  return ["kB", 1e3, 0];
}

/** "152 of 331 MB": both in the total's unit. */
export function sentOfTotal(loaded: number, total: number): string {
  const [unit, scale, digits] = unitFor(total);
  const fmt = (n: number) => (n / scale).toFixed(digits);
  return `${fmt(Math.min(loaded, total))} of ${fmt(total)} ${unit}`;
}

/** "331 MB", "1.20 GB": a file size as the card shows it. */
export function formatSize(total: number): string {
  const [unit, scale, digits] = unitFor(total);
  return `${(total / scale).toFixed(digits)} ${unit}`;
}

/** "3.2 MB/s", "450 kB/s". */
export function formatRate(bytesPerSecond: number): string {
  if (bytesPerSecond >= 999.5e3) return `${(bytesPerSecond / 1e6).toFixed(1)} MB/s`; // never "1000 kB/s"
  return `${Math.round(bytesPerSecond / 1e3)} kB/s`;
}
