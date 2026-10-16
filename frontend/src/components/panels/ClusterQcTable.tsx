import { useEffect, useMemo, useState } from "react";
import { AlertTriangle, ChevronDown, ChevronRight, Loader2, Sparkles } from "lucide-react";
import { getClusterQc, type ClusterFlag, type ClusterQc, type ClusterQcReport } from "@/api/qc";
import { useDatasetStore } from "@/stores/datasetStore";
import { useSelectionStore } from "@/stores/selectionStore";
import { useSettingsStore } from "@/stores/settingsStore";
import { useViewStore } from "@/stores/viewStore";

export const FLAG_LABELS: Record<ClusterFlag, string> = {
  low_quality: "Low quality",
  high_mito: "High mito",
  low_genes: "Few genes",
  doublet_rich: "Doublet-rich",
  small: "Very small",
};

const FLAG_STYLES: Record<ClusterFlag, string> = {
  low_quality: "border-red-200 bg-red-50 text-red-700",
  high_mito: "border-amber-200 bg-amber-50 text-amber-800",
  low_genes: "border-amber-200 bg-amber-50 text-amber-800",
  doublet_rich: "border-violet-200 bg-violet-50 text-violet-700",
  small: "border-slate-200 bg-slate-50 text-slate-600",
};

function fmt(v: number | null | undefined, kind: "n" | "pct" | "frac" = "n"): string {
  if (v === null || v === undefined || !isFinite(v)) return "—";
  if (kind === "pct") return `${v.toFixed(1)}%`;
  if (kind === "frac") return `${Math.round(v * 100)}%`;
  return Math.round(v).toLocaleString();
}

/** The question the ✦ button asks: the cluster's own numbers, so the answer is grounded. */
export function askQuestion(column: string, c: ClusterQc): string {
  const flags = c.flags.map((f) => FLAG_LABELS[f].toLowerCase()).join(", ");
  return `Cluster ${c.group} of ${column} is flagged as ${flags}. ${c.reasons.join(" ")} What could cause this, and should I remove it?`;
}

/** Per-cluster QC: each cluster's size and median QC values, with plain-language
 *  flags for clusters that stand out against the others (computed server-side,
 *  deterministically). Collapsed to one line when nothing stands out. */
export function ClusterQcTable({ datasetId, refreshKey = 0 }: { datasetId: string; refreshKey?: number }) {
  const dataset = useDatasetStore((s) => s.currentDataset);
  const askCopilot = useViewStore((s) => s.askCopilot);
  const [groupby, setGroupby] = useState<string>("");
  const [data, setData] = useState<ClusterQcReport | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [open, setOpen] = useState<boolean | null>(null); // null = follow the flags

  useEffect(() => {
    setGroupby("");
    setData(null);
  }, [datasetId]);

  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    setError(null);
    getClusterQc(datasetId, groupby || undefined)
      .then((d) => !cancelled && setData(d))
      .catch((e) => !cancelled && setError(String(e?.message ?? e)))
      .finally(() => !cancelled && setLoading(false));
    return () => {
      cancelled = true;
    };
  }, [datasetId, groupby, refreshKey]);

  const groupOptions = useMemo(
    () =>
      (dataset?.obs_columns ?? [])
        .filter(
          (c) =>
            (c.dtype === "category" || c.dtype === "object" || c.dtype === "bool") &&
            c.n_unique >= 2 &&
            c.n_unique <= 100,
        )
        .map((c) => c.name),
    [dataset],
  );

  const flagged = data?.clusters.filter((c) => c.flags.length > 0) ?? [];
  const expanded = open ?? flagged.length > 0;
  const has = (m: string) => data?.metrics.includes(m) ?? false;
  const kinds = [...new Set(flagged.flatMap((c) => c.flags))];

  const show = (c: ClusterQc) => {
    if (!data?.column) return;
    useSettingsStore.getState().setColorBy(data.column);
    useSelectionStore.getState().setHighlight({ column: data.column, value: c.group });
    useViewStore.getState().setPanel("unified");
  };

  if (error) {
    return (
      <div className="rounded-xl border border-slate-200 bg-white p-4 text-xs text-slate-400">
        Cluster QC unavailable: {error}
      </div>
    );
  }

  return (
    <div className="rounded-xl border border-slate-200 bg-white p-3">
      <div className="flex flex-wrap items-center gap-2">
        <button
          type="button"
          onClick={() => setOpen(!expanded)}
          aria-expanded={expanded}
          className="flex items-center gap-1 text-sm font-semibold text-slate-700"
        >
          {expanded ? <ChevronDown className="h-4 w-4" /> : <ChevronRight className="h-4 w-4" />}
          Cluster QC
        </button>
        {loading ? (
          <Loader2 className="h-3.5 w-3.5 animate-spin text-slate-400" />
        ) : data?.column ? (
          <span className={`text-xs ${flagged.length ? "text-amber-700" : "text-slate-500"}`}>
            {flagged.length
              ? `${flagged.length} of ${data.n_groups} clusters stand out`
              : `No cluster stands out (${data.n_groups} clusters)`}
          </span>
        ) : null}
        <span className="flex-1" />
        {groupOptions.length > 0 && (
          <select
            aria-label="Grouping for cluster QC"
            value={groupby || data?.column || ""}
            onChange={(e) => setGroupby(e.target.value)}
            className="rounded border border-slate-200 bg-white px-1.5 py-0.5 text-xs text-slate-600"
          >
            {!data?.column && <option value="">Choose a grouping…</option>}
            {groupOptions.map((c) => (
              <option key={c} value={c}>
                {c}
              </option>
            ))}
          </select>
        )}
      </div>

      {data && !data.column && <p className="mt-2 text-xs text-slate-500">{data.note}</p>}

      {expanded && data?.column && (
        <div className="mt-2 space-y-2">
          <div className="max-h-72 overflow-auto rounded border border-slate-100">
            <table className="w-full text-xs">
              <thead className="sticky top-0 bg-slate-50 text-left text-[11px] text-slate-500">
                <tr>
                  <th className="px-2 py-1 font-medium">Cluster</th>
                  <th className="px-2 py-1 text-right font-medium">Cells</th>
                  <th className="px-2 py-1 text-right font-medium" title="Median genes per cell">Genes</th>
                  <th className="px-2 py-1 text-right font-medium" title="Median counts per cell">Counts</th>
                  {has("pct_mt") && <th className="px-2 py-1 text-right font-medium" title="Median % mitochondrial">Mito</th>}
                  {has("pct_ribo") && <th className="px-2 py-1 text-right font-medium" title="Median % ribosomal">Ribo</th>}
                  {has("doublet_fraction") && <th className="px-2 py-1 text-right font-medium" title="Share of predicted doublets">Doublets</th>}
                  {!has("doublet_fraction") && has("doublet_score") && <th className="px-2 py-1 text-right font-medium" title="Median doublet score">Dbl score</th>}
                  <th className="px-2 py-1 font-medium">Flags</th>
                </tr>
              </thead>
              <tbody>
                <tr className="border-t border-slate-100 italic text-slate-400">
                  <td className="px-2 py-1">typical</td>
                  <td className="px-2 py-1" />
                  <td className="px-2 py-1 text-right">{fmt(data.typical.median_genes)}</td>
                  <td className="px-2 py-1 text-right">{fmt(data.typical.median_counts)}</td>
                  {has("pct_mt") && <td className="px-2 py-1 text-right">{fmt(data.typical.median_pct_mt, "pct")}</td>}
                  {has("pct_ribo") && <td className="px-2 py-1 text-right">{fmt(data.typical.median_pct_ribo, "pct")}</td>}
                  {has("doublet_fraction") && <td className="px-2 py-1 text-right">{fmt(data.typical.doublet_fraction, "frac")}</td>}
                  {!has("doublet_fraction") && has("doublet_score") && <td className="px-2 py-1 text-right">{data.typical.median_doublet_score?.toFixed(2) ?? "—"}</td>}
                  <td className="px-2 py-1" />
                </tr>
                {data.clusters.map((c) => (
                  <tr
                    key={c.group}
                    className={`border-t border-slate-100 ${c.flags.length ? "bg-amber-50/40" : ""}`}
                  >
                    <td className="px-2 py-1">
                      <button
                        type="button"
                        onClick={() => show(c)}
                        title="Show this cluster in the Unified View"
                        className="font-medium text-slate-700 hover:text-primary hover:underline"
                      >
                        {c.group}
                      </button>
                    </td>
                    <td className="px-2 py-1 text-right tabular-nums">{fmt(c.n_cells)}</td>
                    <td className="px-2 py-1 text-right tabular-nums">{fmt(c.median_genes)}</td>
                    <td className="px-2 py-1 text-right tabular-nums">{fmt(c.median_counts)}</td>
                    {has("pct_mt") && <td className="px-2 py-1 text-right tabular-nums">{fmt(c.median_pct_mt, "pct")}</td>}
                    {has("pct_ribo") && <td className="px-2 py-1 text-right tabular-nums">{fmt(c.median_pct_ribo, "pct")}</td>}
                    {has("doublet_fraction") && <td className="px-2 py-1 text-right tabular-nums">{fmt(c.doublet_fraction, "frac")}</td>}
                    {!has("doublet_fraction") && has("doublet_score") && <td className="px-2 py-1 text-right tabular-nums">{c.median_doublet_score?.toFixed(2) ?? "—"}</td>}
                    <td className="px-2 py-1">
                      <div className="flex flex-wrap items-center gap-1">
                        {c.flags.map((f, i) => (
                          <span
                            key={f}
                            title={c.reasons[i]}
                            className={`rounded border px-1 py-px text-[10px] font-medium ${FLAG_STYLES[f]}`}
                          >
                            {FLAG_LABELS[f]}
                          </span>
                        ))}
                        {c.flags.length > 0 && (
                          <button
                            type="button"
                            onClick={() => askCopilot(askQuestion(data.column!, c))}
                            title="Ask the co-pilot about this cluster"
                            aria-label={`Ask the co-pilot about cluster ${c.group}`}
                            className="rounded p-0.5 text-primary hover:bg-primary/10"
                          >
                            <Sparkles className="h-3 w-3" />
                          </button>
                        )}
                      </div>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          {flagged.length > 0 && (
            <ul className="space-y-1 text-xs text-slate-600">
              {flagged.map((c) => (
                <li key={c.group} className="flex items-start gap-1.5">
                  <AlertTriangle className="mt-0.5 h-3.5 w-3.5 flex-shrink-0 text-amber-500" />
                  <span>
                    <span className="font-medium text-slate-700">Cluster {c.group}:</span>{" "}
                    {c.reasons.join(" ")}
                  </span>
                </li>
              ))}
            </ul>
          )}
          {kinds.length > 0 && (
            <dl className="space-y-0.5 border-t border-slate-100 pt-1.5 text-[11px] text-slate-500">
              {kinds.map((k) => (
                <div key={k}>
                  <dt className="inline font-medium text-slate-600">{FLAG_LABELS[k]}:</dt>{" "}
                  <dd className="inline">{data.flag_help[k]}</dd>
                </div>
              ))}
            </dl>
          )}
          <p className="text-[11px] text-slate-400">
            Each cluster is compared with the typical cluster (the median across clusters,
            using robust spread), so one bad cluster can&apos;t hide by widening the range.
            {data.note ? ` ${data.note}` : ""}
          </p>
        </div>
      )}
    </div>
  );
}
