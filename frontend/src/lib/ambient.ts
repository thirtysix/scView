/**
 * Ambient RNA (backend core/ambient.py): how the card, the Markers table and the History
 * chips put the soup, rho and the ambient-looking markers into words.
 */
import type { AmbientCorrectionSample, AmbientMarker, AmbientResponse, AmbientSample } from "@/api/ambient";

/** A share as a percent: 1 decimal from 1% up, 2 below ("0.42%"). */
export function pct(v: number | null | undefined): string {
  if (v === null || v === undefined || !isFinite(v)) return "—";
  const p = v * 100;
  return `${p >= 1 ? p.toFixed(1) : p.toFixed(2)}%`;
}

/** "5.3% (3.1-7.4%)": rho and the range where its estimate is at least half its peak. */
export function rhoText(s: Pick<AmbientSample, "rho" | "rho_low" | "rho_high">): string {
  if (s.rho === null) return "—";
  if (s.rho_low === null || s.rho_high === null) return pct(s.rho);
  return `${pct(s.rho)} (${(s.rho_low * 100).toFixed(1)}-${(s.rho_high * 100).toFixed(1)}%)`;
}

const STATUS_TEXT: Record<string, string> = {
  no_soup: "Attach the raw matrix to estimate it",
  no_clusters: "Run clustering to estimate it",
  few_clusters: "Too few clusters in this sample",
  no_markers: "No marker genes to estimate from",
};

/** What the rho cell says when there is no estimate. */
export function rhoStatusText(s: AmbientSample): string {
  if (s.rho !== null) return rhoText(s);
  return STATUS_TEXT[s.rho_status] ?? "No estimate";
}

/** Group -> gene -> its flag, for marking the Markers table. */
export function flagIndex(resp: AmbientResponse | null | undefined): Map<string, Map<string, AmbientMarker>> {
  const out = new Map<string, Map<string, AmbientMarker>>();
  for (const [group, rows] of Object.entries(resp?.markers?.groups ?? {})) {
    out.set(group, new Map(rows.map((r) => [r.gene, r])));
  }
  return out;
}

/** The flag's tooltip: which soup rank, and how its signal outside the cluster compares. */
export function flagTitle(m: AmbientMarker, group: string, approximate: boolean): string {
  if (approximate || m.ratio === null) {
    return (
      `Possibly ambient: ${m.gene} is gene ${m.soup_rank} by abundance in its sample (no raw matrix to read ` +
      "the soup from), the kind of gene that spills into other clusters as ambient RNA."
    );
  }
  return (
    `Likely ambient outside ${group}: ${m.gene} is gene ${m.soup_rank} of the soup, and in the other clusters it is ` +
    `found at ${m.ratio.toFixed(1)}x what ambient RNA alone would put there (rho x soup). Its signal there is ` +
    "probably soup, which can inflate DE against them."
  );
}

/** The correction step's samples as one History chip. */
export function correctionText(rows: AmbientCorrectionSample[]): string {
  const parts = rows.map((r) => {
    const removed = r.removed_share !== undefined ? `, ${pct(r.removed_share)} removed` : "";
    return `${r.sample}: rho ${pct(r.rho)} (${r.rho_source === "set" ? "set" : "estimated"}${removed})`;
  });
  return `ambient RNA: ${parts.join("; ")}`;
}

/** Whether a recorded param is the correction step's per-sample list. */
export function isCorrectionSamples(key: string, value: unknown): value is AmbientCorrectionSample[] {
  return (
    key === "samples" &&
    Array.isArray(value) &&
    value.length > 0 &&
    value.every((r) => r && typeof r === "object" && "rho" in (r as object) && "sample" in (r as object))
  );
}
