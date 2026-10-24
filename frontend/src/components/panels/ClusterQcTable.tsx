import { useEffect, useMemo, useState } from "react";
import { AlertTriangle, ChevronDown, ChevronRight, Info, Loader2, Sparkles } from "lucide-react";
import { getClusterQc, type ClusterFlag, type ClusterQc, type ClusterQcReport, type QcDriver } from "@/api/qc";
import { useDatasetStore } from "@/stores/datasetStore";
import { useSelectionStore } from "@/stores/selectionStore";
import { useSettingsStore } from "@/stores/settingsStore";
import { useViewStore } from "@/stores/viewStore";

export const FLAG_LABELS: Record<ClusterFlag, string> = {
  low_quality: "Low quality",
  high_mito: "High mito",
  low_genes: "Few genes",
  doublet_rich: "Doublet-rich",
  stress_driven: "Stress-driven",
  hb_ambient: "Ambient Hb",
  cycling: "Cycling",
  low_malat1: "Low MALAT1",
  small: "Very small",
};

const FLAG_STYLES: Record<ClusterFlag, string> = {
  low_quality: "border-red-200 bg-red-50 text-red-700",
  high_mito: "border-amber-200 bg-amber-50 text-amber-800",
  low_genes: "border-amber-200 bg-amber-50 text-amber-800",
  doublet_rich: "border-violet-200 bg-violet-50 text-violet-700",
  stress_driven: "border-orange-200 bg-orange-50 text-orange-800",
  hb_ambient: "border-rose-200 bg-rose-50 text-rose-700",
  cycling: "border-sky-200 bg-sky-50 text-sky-700",
  low_malat1: "border-teal-200 bg-teal-50 text-teal-800",
  small: "border-slate-200 bg-slate-50 text-slate-600",
};

const MALAT1_TITLE =
  "Median % of a cell's counts from MALAT1, a nuclear-retained RNA: a cluster far below the median cell may be " +
  "fragments or cytoplasm without its nucleus (Montserrat-Ayuso & Esteve-Codina 2024, doi:10.1186/s12864-024-11015-5). " +
  "Flagged below a third of the median cell's share; off when MALAT1 is barely captured (under 0.5%).";

const ERYTHROID_TITLE = "Red-cell lineage by its own markers: its haemoglobin is real expression, not ambient RNA.";

function fmt(v: number | null | undefined, kind: "n" | "pct" | "frac" | "hb" = "n"): string {
  if (v === null || v === undefined || !isFinite(v)) return "—";
  if (kind === "pct") return `${v.toFixed(1)}%`;
  if (kind === "frac") return `${Math.round(v * 100)}%`;
  if (kind === "hb") {
    // From red cells (~50%) down to ambient traces (0.01%).
    if (v >= 1) return `${v.toFixed(1)}%`;
    if (v >= 0.01) return `${v.toFixed(2)}%`;
    return v > 0 ? "<0.01%" : "0%";
  }
  return Math.round(v).toLocaleString();
}

/** Technical scores against the clustering's leading PCs: each driver's sentence,
 *  or which scores were looked at when none tracks a PC. */
function DriversNote({ report }: { report: ClusterQcReport }) {
  const drivers: QcDriver[] = report.drivers ?? [];
  const checked = report.drivers_checked ?? [];
  if (!report.drivers_n_pcs || (!drivers.length && !checked.length)) return null;
  const space = `first ${report.drivers_n_pcs} PCs of ${report.drivers_space ?? "the PCA"}`;
  return (
    <div className="space-y-1 border-t border-slate-100 pt-1.5 text-xs text-slate-600">
      <p className="text-[11px] font-medium text-slate-500">
        Technical scores vs the clustering&apos;s PCs ({space})
      </p>
      {drivers.length ? (
        <ul className="space-y-1">
          {drivers.map((d) => (
            <li key={d.score} className="flex items-start gap-1.5">
              {d.inside_clusters ? (
                <AlertTriangle className="mt-0.5 h-3.5 w-3.5 flex-shrink-0 text-amber-500" aria-label="Inside clusters too" />
              ) : (
                <Info className="mt-0.5 h-3.5 w-3.5 flex-shrink-0 text-slate-400" aria-label="Between clusters" />
              )}
              <span>{d.reason}</span>
            </li>
          ))}
        </ul>
      ) : (
        <p className="text-[11px] text-slate-500">
          None of {checked.join(", ")} correlates strongly (|r| ≥ 0.5) with a leading PC.
        </p>
      )}
    </div>
  );
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
  const noted = data?.clusters.filter((c) => (c.notes ?? []).length > 0) ?? [];
  const drivers = data?.drivers ?? [];
  // Open by default when something needs a look: a flag, ambient Hb, or a technical
  // score that moves along a PC inside clusters (between-cluster ones are often biology).
  const expanded = open ?? (flagged.length > 0 || !!data?.ambient_hb || drivers.some((d) => d.inside_clusters));
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
            {drivers.length > 0 &&
              ` · ${drivers.length} technical score${drivers.length > 1 ? "s" : ""} track${drivers.length > 1 ? "" : "s"} a PC`}
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
                  <th className="px-2 py-1 text-right font-medium" title="Cells in the cluster">Cells</th>
                  <th className="px-2 py-1 text-right font-medium" title="Median genes per cell">Genes</th>
                  <th className="px-2 py-1 text-right font-medium" title="Median counts per cell">Counts</th>
                  {has("pct_mt") && <th className="px-2 py-1 text-right font-medium" title="Median % mitochondrial">Mito</th>}
                  {has("pct_ribo") && <th className="px-2 py-1 text-right font-medium" title="Median % ribosomal">Ribo</th>}
                  {has("doublet_fraction") && <th className="px-2 py-1 text-right font-medium" title={`Share of the cluster's cells called doublets${data.doublet_source ? ` by ${data.doublet_source}` : ""} (cell hashing when it was run, else Scrublet)`}>Doublets</th>}
                  {!has("doublet_fraction") && has("doublet_score") && <th className="px-2 py-1 text-right font-medium" title="Median doublet score">Dbl score</th>}
                  {has("stress_score") && <th className="px-2 py-1 text-right font-medium" title="Median dissociation-stress score (immediate-early and heat-shock genes)">Stress</th>}
                  {has("pct_hb") && <th className="px-2 py-1 text-right font-medium" title="Mean % haemoglobin reads per cell (the mean: most cells have none); the typical value is a non-erythroid cluster's">Hb</th>}
                  {has("cycling_fraction") && <th className="px-2 py-1 text-right font-medium" title="Share of cells with a clear S or G2M cell-cycle score (above the noise band around 0)">Cycling</th>}
                  {has("pct_malat1") && <th className="px-2 py-1 text-right font-medium" title={MALAT1_TITLE}>MALAT1</th>}
                  {has("pct_intronic") && (
                    <th
                      className="px-2 py-1 text-right font-medium"
                      title={`Median intronic / unspliced share of a cell's reads, from the author's '${data.intronic_column}' column (shown as given, not computed by scView); nuclei and pre-mRNA-rich cells run higher`}
                    >
                      Intronic
                    </th>
                  )}
                  <th className="px-2 py-1 font-medium" title="Metrics on which the cluster sits far from the typical cluster (hover a flag for the numbers)">Flags</th>
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
                  {has("stress_score") && <td className="px-2 py-1 text-right">{data.typical.median_stress_score?.toFixed(2) ?? "—"}</td>}
                  {has("pct_hb") && <td className="px-2 py-1 text-right">{fmt(data.typical.mean_pct_hb, "hb")}</td>}
                  {has("cycling_fraction") && <td className="px-2 py-1 text-right">{fmt(data.typical.cycling_fraction, "frac")}</td>}
                  {has("pct_malat1") && (
                    <td className="px-2 py-1 text-right" title="The median cell's MALAT1 share (the low-MALAT1 flag compares with this)">
                      {fmt(data.dataset_median_pct_malat1, "hb")}
                    </td>
                  )}
                  {has("pct_intronic") && <td className="px-2 py-1 text-right">{fmt(data.typical.median_pct_intronic, "pct")}</td>}
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
                    {has("stress_score") && (
                      <td
                        className="px-2 py-1 text-right tabular-nums"
                        title={c.stress_markers?.length ? `Stress genes among its top markers: ${c.stress_markers.join(", ")}` : undefined}
                      >
                        {c.median_stress_score?.toFixed(2) ?? "—"}
                      </td>
                    )}
                    {has("pct_hb") && (
                      <td
                        className="px-2 py-1 text-right tabular-nums"
                        title={c.erythroid_markers?.length ? `Red-cell genes among its top markers: ${c.erythroid_markers.join(", ")}` : undefined}
                      >
                        {fmt(c.mean_pct_hb, "hb")}
                      </td>
                    )}
                    {has("cycling_fraction") && <td className="px-2 py-1 text-right tabular-nums">{fmt(c.cycling_fraction, "frac")}</td>}
                    {has("pct_malat1") && <td className="px-2 py-1 text-right tabular-nums">{fmt(c.median_pct_malat1, "hb")}</td>}
                    {has("pct_intronic") && <td className="px-2 py-1 text-right tabular-nums">{fmt(c.median_pct_intronic, "pct")}</td>}
                    <td className="px-2 py-1">
                      <div className="flex flex-wrap items-center gap-1">
                        {c.erythroid && (
                          <span
                            title={c.notes?.[0] ?? ERYTHROID_TITLE}
                            className="rounded border border-red-100 bg-white px-1 py-px text-[10px] font-medium text-red-700"
                          >
                            Erythroid
                          </span>
                        )}
                        {(c.ambient_markers ?? []).length > 0 && (
                          <span
                            title={`${data.ambient_markers_approximate ? "Possibly ambient" : "Ambient-looking"} markers: ${(c.ambient_markers ?? []).join(", ")}. ${data.ambient_markers_help ?? ""}`}
                            className="rounded border border-sky-100 bg-white px-1 py-px text-[10px] font-medium text-sky-700"
                            data-testid="cluster-ambient-markers"
                          >
                            Ambient markers: {c.ambient_markers?.length}
                          </span>
                        )}
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

          {data.ambient_hb && (
            <p className="flex items-start gap-1.5 text-xs text-slate-600">
              <AlertTriangle className="mt-0.5 h-3.5 w-3.5 flex-shrink-0 text-rose-500" />
              <span>{data.ambient_hb}</span>
            </p>
          )}
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
          {noted.length > 0 && (
            <ul className="space-y-1 text-xs text-slate-600">
              {noted.map((c) => (
                <li key={c.group} className="flex items-start gap-1.5">
                  <Info className="mt-0.5 h-3.5 w-3.5 flex-shrink-0 text-slate-400" />
                  <span>
                    <span className="font-medium text-slate-700">Cluster {c.group}:</span>{" "}
                    {(c.notes ?? []).join(" ")}
                  </span>
                </li>
              ))}
            </ul>
          )}
          <DriversNote report={data} />
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
