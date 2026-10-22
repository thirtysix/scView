const LIST_SHOWN = 12;

/** A recorded step param as the History panel's chip shows it: lists joined (a long
 *  one cut to its first items and count), the filtering step's per-sample thresholds
 *  (an object per sample) summarised, any other object as short JSON. */
export function paramText(key: string, value: unknown): string {
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
