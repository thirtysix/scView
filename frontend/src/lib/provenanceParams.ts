import { leidenIterationsText } from "@/lib/fastLeiden";

const LIST_SHOWN = 12;

// Recorded params a History re-run keeps: re-clustering at another resolution stays
// Louvain, or fast Leiden, when the step was.
const KEPT_ON_RERUN: Record<string, readonly string[]> = {
  clustering: ["clustering_method", "leiden_iterations"],
};

/** The recorded params of `step` that its re-run sends again (the edited one goes on top). */
export function keptRerunParams(step: { step: string; params?: Record<string, unknown> | null }): Record<string, unknown> {
  const out: Record<string, unknown> = {};
  for (const key of KEPT_ON_RERUN[step.step] ?? []) {
    const v = step.params?.[key];
    if (v !== undefined && v !== null) out[key] = v;
  }
  return out;
}

interface ChannelRun {
  n_cells?: number;
  expected_doublet_rate?: number;
  n_called?: number;
  status?: string;
}

/** The doublet step's channels (one object each, 133 for the 1.3M brain) in one chip. */
function channelsText(rows: ChannelRun[]): string {
  const sizes = rows.map((r) => r.n_cells ?? 0);
  const rates = rows.map((r) => 100 * (r.expected_doublet_rate ?? 0));
  const called = rows.reduce((t, r) => t + (r.n_called ?? 0), 0);
  const small = rows.filter((r) => r.status === "too_small").length;
  const range = (v: number[], d: number) => {
    const [lo, hi] = [Math.min(...v), Math.max(...v)];
    return lo.toFixed(d) === hi.toFixed(d) ? hi.toLocaleString(undefined, { maximumFractionDigits: d })
      : `${lo.toLocaleString(undefined, { maximumFractionDigits: d })}-${hi.toLocaleString(undefined, { maximumFractionDigits: d })}`;
  };
  return (
    `Scrublet per channel: ${rows.length.toLocaleString()} channel${rows.length === 1 ? "" : "s"}, ` +
    `${range(sizes, 0)} cells, expected ${range(rates, 1)}%, ${called.toLocaleString()} called` +
    (small ? `, ${small} too small to score` : "")
  );
}

/** A recorded step param as the History panel's chip shows it: lists joined (a long
 *  one cut to its first items and count), the filtering step's per-sample thresholds
 *  (an object per sample) and the doublet step's channels summarised, any other object
 *  as short JSON. */
export function paramText(key: string, value: unknown): string {
  if (key === "channels" && Array.isArray(value) && value.length && typeof value[0] === "object")
    return channelsText(value as ChannelRun[]);
  if (key === "leiden_iterations") return leidenIterationsText(value);
  // A long list (the gene programs step records its 2,000 genes): the first few and the count.
  if (Array.isArray(value) && value.length > LIST_SHOWN)
    return `${key}=${value.slice(0, LIST_SHOWN).join(", ")}, ... (${value.length.toLocaleString()})`;
  if (Array.isArray(value)) return `${key}=${value.join(", ")}`;
  if (value && typeof value === "object") {
    const v = value as { column?: string | null; samples?: unknown[] };
    if (key === "qc_thresholds") {
      const n = v.samples?.length ?? 0;
      return v.column ? `per-sample thresholds: ${v.column} (${n} samples)` : "thresholds: all cells pooled";
    }
    const json = JSON.stringify(value);
    return `${key}=${json.length > 80 ? `${json.slice(0, 77)}...` : json}`;
  }
  return `${key}=${String(value)}`;
}
