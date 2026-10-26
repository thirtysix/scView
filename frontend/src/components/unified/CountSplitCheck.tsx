import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { Loader2 } from "lucide-react";
import Plot from "@/components/plots/Plot";
import { Caption } from "@/components/common/Caption";
import { InfoTip } from "@/components/common/InfoTip";
import {
  getSplitMarkersInfo,
  keepSplitResult,
  runSplitMarkers,
  type SplitMarkersInfo,
  type SplitMarkersResult,
} from "@/api/splitMarkers";
import { prettyObsLabel } from "@/lib/formatting";
import { formatDuration } from "@/lib/hvgSensitivity";
import { plotConfig } from "@/lib/plotConfig";
import { comparisonPoints, groupsByConcern, keptText, summaryText, type PointKind } from "@/lib/splitMarkers";
import { errorMessage } from "@/lib/errorMessage";

// Blue / orange stay apart under every common colour-vision deficiency.
const KIND_STYLE: Record<PointKind, { color: string; label: string }> = {
  kept: { color: "#2563eb", label: "Marker, kept" },
  lost: { color: "#ea580c", label: "Marker, lost" },
  other: { color: "#94a3b8", label: "Other genes" },
};

/**
 * "Check with count splitting" in the Markers subtab (backend core/split_markers.py):
 * the counts are split into two halves, the clustering re-runs on one and each
 * cluster's markers are tested on the other, so the p-values do not reuse the data
 * that made the clusters. Shows per group the ordinary markers kept and lost, the
 * two p-values side by side, and hands the selected group's split p-values to the
 * Markers table.
 */
export function CountSplitCheck({
  datasetId,
  column,
  group,
  onGroupChange,
  showSplitP,
  onShowSplitPChange,
  onResult,
}: {
  datasetId: string;
  /** The Markers subtab's grouping column */
  column: string;
  group: string;
  onGroupChange: (group: string) => void;
  showSplitP: boolean;
  onShowSplitPChange: (on: boolean) => void;
  onResult: (r: SplitMarkersResult | null) => void;
}) {
  const [info, setInfo] = useState<SplitMarkersInfo | null>(null);
  const [running, setRunning] = useState(false);
  const [step, setStep] = useState<string | null>(null);
  const [estimate, setEstimate] = useState<number | null>(null);
  const [result, setResult] = useState<SplitMarkersResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [kept, setKept] = useState<string | null>(null);
  const [keeping, setKeeping] = useState(false);
  const abortRef = useRef<AbortController | null>(null);

  useEffect(() => {
    let cancelled = false;
    abortRef.current?.abort(); // a run for the column shown before
    setInfo(null);
    setResult(null);
    setError(null);
    setKept(null);
    onResult(null);
    getSplitMarkersInfo(datasetId, column)
      .then((i) => !cancelled && setInfo(i))
      .catch(
        (e) =>
          !cancelled &&
          setInfo({ available: false, reason: errorMessage(e) } as SplitMarkersInfo),
      );
    return () => {
      cancelled = true;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps -- onResult is a setter from the parent
  }, [datasetId, column]);

  useEffect(() => () => abortRef.current?.abort(), []); // leaving the tab stops a running check

  const run = useCallback(async () => {
    if (running) return;
    const ctrl = new AbortController();
    abortRef.current = ctrl;
    setRunning(true);
    setError(null);
    setStep(null);
    setKept(null);
    try {
      await runSplitMarkers(datasetId, column, {
        onStart: (s) => setEstimate(s.cached ? null : s.estimate_seconds),
        onProgress: (p) => setStep(p.label),
        onResult: (r) => {
          setResult(r);
          onResult(r);
        },
        signal: ctrl.signal,
      });
    } catch (e) {
      if (!ctrl.signal.aborted) setError(e instanceof Error ? e.message : "The count-split check failed.");
    } finally {
      if (abortRef.current === ctrl) abortRef.current = null;
      setRunning(false);
      setStep(null);
    }
  }, [datasetId, column, running, onResult]);

  const current = result?.groups.find((g) => g.group === group) ?? null;
  const ordered = useMemo(
    () => (result ? groupsByConcern(result.groups, result.thresholds.keeps_enough) : []),
    [result],
  );

  const keep = async () => {
    if (!current) return;
    setKeeping(true);
    setError(null);
    try {
      const r = await keepSplitResult(datasetId, column, current.group);
      setKept(r.label);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Could not keep the result.");
    } finally {
      setKeeping(false);
    }
  };

  return (
    <section aria-label="Count-split marker check" className="space-y-1.5 rounded-lg border border-slate-200 p-2">
      <div className="flex items-center gap-1">
        <span className="text-[11px] font-medium text-slate-600">Are these markers real?</span>
        <InfoTip label="About count splitting" width={300}>
          Marker p-values test genes on the same counts that made the clusters, so every cluster looks
          significant, even one cut out of a single population. Count splitting divides each cell&rsquo;s counts
          into two independent halves: the clustering re-runs on one half and each cluster is tested on the other.
          A marker that keeps adjusted p &lt; 0.05 there is supported by data the clustering never saw. The held-out
          half has half the molecules, so weak markers can also be lost for lack of power. Neufeld et al. 2024,
          Biostatistics (PMID 36511385); negative binomial splitting as in the merge check.
        </InfoTip>
        <span className="ml-auto" />
        {running ? (
          <button
            type="button"
            onClick={() => abortRef.current?.abort()}
            className="rounded border border-slate-300 px-2 py-0.5 text-xs text-slate-600 hover:bg-slate-50"
          >
            Stop
          </button>
        ) : (
          <button
            type="button"
            onClick={() => void run()}
            disabled={!info?.available}
            className="rounded bg-blue-500 px-2 py-0.5 text-xs font-medium text-white hover:bg-blue-600 disabled:opacity-50"
          >
            Check with count splitting
          </button>
        )}
      </div>
      {info && !info.available && <p className="text-[11px] text-amber-700">{info.reason}</p>}
      {running ? (
        <div className="flex items-center gap-1.5 text-[11px] text-slate-500" role="status">
          <Loader2 className="h-3.5 w-3.5 animate-spin" />
          {step ?? "Starting"}
          {estimate != null ? ` (about ${formatDuration(estimate)} in all)` : ""}…
        </div>
      ) : (
        info?.available &&
        !result && (
          <p className="text-[10px] text-slate-400">
            Re-tests the markers of {prettyObsLabel(info.reference ?? column)} ({info.n_groups} groups) on held-out
            counts
            {info.cached
              ? ": the last run is kept, no waiting."
              : info.estimate_seconds != null
                ? `: about ${formatDuration(info.estimate_seconds)} (the first run after the server starts adds ~${formatDuration(info.first_run_seconds)}).`
                : "."}
          </p>
        )
      )}

      {result && (
        <div className="space-y-1.5">
          <p className="text-[11px] text-slate-600">
            <span className="font-medium">{summaryText(result)}.</span> The training half made{" "}
            {result.train.n_clusters} clusters (ARI {result.train.ari.toFixed(2)} with these groups).
          </p>
          <div className="max-h-48 overflow-auto rounded border border-slate-100">
            <table className="w-full border-collapse text-[11px]">
              <thead className="sticky top-0 bg-slate-50">
                <tr className="border-b border-slate-200 text-left text-slate-500">
                  <th className="px-1.5 py-1 font-medium">Group</th>
                  <th className="px-1.5 py-1 text-right font-medium">Cells</th>
                  <th className="px-1.5 py-1 text-right font-medium" title="Ordinary markers: up, in 20% of the group, adjusted p < 0.05 on all counts">
                    Markers
                  </th>
                  <th className="px-1.5 py-1 text-right font-medium" title="Still adjusted p < 0.05 (and up) on the held-out half">
                    Kept
                  </th>
                  <th className="px-1.5 py-1 text-right font-medium" title="Not significant on the held-out half: double dipping, or too weak at half depth">
                    Lost
                  </th>
                  <th className="px-1.5 py-1 text-right font-medium" title="Jaccard overlap with the training cluster it was tested as">
                    Match
                  </th>
                </tr>
              </thead>
              <tbody>
                {ordered.map((g) => (
                  <tr
                    key={g.group}
                    title={g.reason ? `Not tested: ${g.reason}` : undefined}
                    className={`border-b border-slate-50 ${g.group === group ? "bg-blue-50" : ""} ${g.tested ? "" : "text-slate-400"}`}
                  >
                    <td className="px-1.5 py-0.5">
                      <button
                        type="button"
                        onClick={() => onGroupChange(g.group)}
                        aria-pressed={g.group === group}
                        className="font-medium text-slate-700 hover:underline"
                      >
                        {g.group}
                      </button>
                    </td>
                    <td className="px-1.5 py-0.5 text-right tabular-nums">{g.n_cells.toLocaleString()}</td>
                    <td className="px-1.5 py-0.5 text-right tabular-nums">{g.n_ordinary.toLocaleString()}</td>
                    <td
                      className={`px-1.5 py-0.5 text-right tabular-nums ${g.tested && g.n_kept < result.thresholds.keeps_enough ? "font-semibold text-orange-700" : ""}`}
                    >
                      {g.tested ? g.n_kept.toLocaleString() : "—"}
                    </td>
                    <td className="px-1.5 py-0.5 text-right tabular-nums">{g.tested ? g.n_lost.toLocaleString() : "—"}</td>
                    <td
                      className="px-1.5 py-0.5 text-right tabular-nums"
                      title={g.shared ? "Another group matches the same training cluster: tested together" : undefined}
                    >
                      {g.jaccard != null ? g.jaccard.toFixed(2) : "—"}
                      {g.shared ? "*" : ""}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <Caption n={{ value: result.n_groups, unit: "groups" }}>
            Groups keeping fewer than {result.thresholds.keeps_enough} markers first. * another group matches the same
            training cluster, so the two were tested as one.
          </Caption>

          {current && <GroupDetail result={result} groupName={current.group} />}

          <div className="flex flex-wrap items-center gap-2">
            <label className="flex items-center gap-1 text-[11px] text-slate-600">
              <input
                type="checkbox"
                checked={showSplitP}
                onChange={(e) => onShowSplitPChange(e.target.checked)}
              />
              Split p-values in the table
            </label>
            {current?.tested && (
              <button
                type="button"
                onClick={() => void keep()}
                disabled={keeping}
                title="Keep this group's held-out test with the DE results, for GSEA and the DE overlap"
                className="ml-auto flex items-center gap-1 rounded border border-slate-300 px-2 py-0.5 text-[11px] text-slate-700 hover:bg-slate-50 disabled:opacity-50"
              >
                {keeping && <Loader2 className="h-3 w-3 animate-spin" />}
                Keep for GSEA / overlap
              </button>
            )}
          </div>
          {kept && <p className="text-[11px] text-emerald-700">Kept with the DE results as &ldquo;{kept}&rdquo;.</p>}
          {result.notes.length > 0 && (
            <p className="text-[10px] leading-snug text-slate-400">{result.notes.join(" ")}</p>
          )}
          <p className="text-[10px] leading-snug text-slate-400">
            One split with one seed. The tested clusters are the training half&rsquo;s, standing for these groups as far
            as they match. A lost marker is not proof of an artefact: the held-out half has half the molecules. A kept
            marker is not proof of a cell type either: a real difference such as sequencing depth is shared by both
            halves and keeps its markers.
          </p>
        </div>
      )}
      {error && (
        <p role="alert" className="rounded bg-red-50 px-2 py-1.5 text-[11px] text-red-700">
          {error}
        </p>
      )}
    </section>
  );
}

/** The selected group: lost markers and the p-value comparison plot. */
function GroupDetail({ result, groupName }: { result: SplitMarkersResult; groupName: string }) {
  const g = result.groups.find((x) => x.group === groupName);
  const points = useMemo(() => comparisonPoints(g?.genes ?? []), [g]);
  const traces = useMemo(
    () =>
      (["other", "lost", "kept"] as PointKind[]).map((kind) => {
        const pts = points.filter((p) => p.kind === kind);
        return {
          type: "scatter" as const,
          mode: "markers" as const,
          name: `${KIND_STYLE[kind].label} (${pts.length})`,
          x: pts.map((p) => p.x),
          y: pts.map((p) => p.y),
          text: pts.map((p) => p.gene),
          hovertemplate: "<b>%{text}</b><br>ordinary %{x:.1f} · split %{y:.1f}<extra></extra>",
          marker: { color: KIND_STYLE[kind].color, size: 5, opacity: 0.8 },
        };
      }),
    [points],
  );
  const layout = useMemo(() => {
    const max = Math.max(2, ...points.map((p) => Math.max(p.x, p.y)));
    const cut = -Math.log10(result.thresholds.padj);
    return {
      xaxis: { title: { text: "ordinary −log10 adj. p", font: { size: 10 } }, tickfont: { size: 9 }, gridcolor: "#f1f5f9", rangemode: "tozero" as const },
      yaxis: { title: { text: "count-split −log10 adj. p", font: { size: 10 } }, tickfont: { size: 9 }, gridcolor: "#f1f5f9", rangemode: "tozero" as const },
      shapes: [
        { type: "line" as const, x0: 0, x1: max, y0: 0, y1: max, line: { color: "#cbd5e1", width: 1, dash: "dot" as const } },
        { type: "line" as const, x0: 0, x1: max, y0: cut, y1: cut, line: { color: "#cbd5e1", width: 1, dash: "dash" as const } },
      ],
      legend: { font: { size: 9 }, orientation: "h" as const, y: 1.18, x: 0 },
      margin: { t: 28, r: 10, b: 36, l: 44 },
      paper_bgcolor: "white",
      plot_bgcolor: "white",
      autosize: true,
      height: 230,
    };
  }, [points, result.thresholds.padj]);
  if (!g) return null;
  return (
    <div className="space-y-1 rounded border border-slate-100 p-1.5">
      <p className="text-[11px] text-slate-600">
        <span className="font-medium">{g.group}</span>: {keptText(g)}
        {g.train_cluster != null ? ` (tested as training cluster ${g.train_cluster})` : ""}.
        {g.lost.length > 0 && <> Lost, e.g. {g.lost.slice(0, 6).join(", ")}.</>}
        {g.n_split_only > 0 && <> {g.n_split_only} gene{g.n_split_only === 1 ? "" : "s"} pass only on the held-out half.</>}
      </p>
      {g.tested && points.length > 0 && (
        <>
          <Plot
            data={traces}
            layout={layout}
            config={plotConfig(`count_split_${result.reference}_${g.group}`)}
            useResizeHandler
            style={{ width: "100%" }}
          />
          <Caption n={{ value: points.length, unit: "genes" }}>
            Adjusted p of each of the group&rsquo;s top genes (ordinary Wilcoxon score), on all counts (x) and on the
            held-out half (y). Above the dashed line: adjusted p &lt; {result.thresholds.padj} on held-out counts; the
            dotted line is equal evidence.
          </Caption>
        </>
      )}
    </div>
  );
}
