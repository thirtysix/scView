import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { Loader2 } from "lucide-react";
import {
  getMergeCheckInfo,
  runMergeCheck,
  saveMergedClustering,
  type MergeCheckInfo,
  type MergePair,
  type MergeResult,
  type MergeTarget,
  type SweepRow,
} from "@/api/clustering";
import { useDatasetRefresh } from "@/hooks/useDatasetRefresh";
import { prettyObsLabel } from "@/lib/formatting";
import { formatDuration } from "@/lib/hvgSensitivity";
import { clustersAfter, defaultSelection, describePair, pairKey } from "@/lib/mergeCheck";
import { formatResolutions } from "@/lib/resolutionSweep";
import { useSettingsStore } from "@/stores/settingsStore";

const THRESHOLDS = [5, 10, 20, 30];
const EVIDENCE_ORDER = { held_out: 0, together: 1, untested: 2 } as const;

/**
 * "Should any clusters merge?" in the resolution sweep panel: count splitting
 * (backend core/merge_check.py) re-clusters half of every cell's counts and tests
 * neighbouring clusters on the other half, then saves the chosen merges.
 */
export function MergeCheck({
  datasetId,
  rows,
  method,
  selected,
}: {
  datasetId: string;
  rows: SweepRow[];
  method: "leiden" | "louvain";
  /** The resolution picked in the sweep table or tree: checked when chosen */
  selected: number | null;
}) {
  const refresh = useDatasetRefresh();
  const [target, setTarget] = useState<"active" | number>("active");
  const [maxGenes, setMaxGenes] = useState(10);
  const [info, setInfo] = useState<MergeCheckInfo | null>(null);
  const [running, setRunning] = useState(false);
  const [step, setStep] = useState<string | null>(null);
  const [estimate, setEstimate] = useState<number | null>(null);
  const [result, setResult] = useState<MergeResult | null>(null);
  const [ticked, setTicked] = useState<Set<string>>(new Set());
  const [showAll, setShowAll] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [saving, setSaving] = useState(false);
  const [saved, setSaved] = useState<string | null>(null);
  const abortRef = useRef<AbortController | null>(null);

  useEffect(() => {
    if (selected != null && !running) setTarget(selected);
    // eslint-disable-next-line react-hooks/exhaustive-deps -- follow the picked row, not the run state
  }, [selected]);

  const request: MergeTarget = useMemo(
    () => (target === "active" ? {} : { resolution: target, method }),
    [target, method],
  );

  useEffect(() => {
    let cancelled = false;
    setInfo(null);
    setResult(null); // it was for the clustering checked before
    getMergeCheckInfo(datasetId, request)
      .then((i) => !cancelled && setInfo(i))
      .catch((e) => !cancelled && setInfo({ available: false, reason: e instanceof Error ? e.message : String(e) } as MergeCheckInfo));
    return () => {
      cancelled = true;
    };
  }, [datasetId, request]);

  useEffect(() => () => abortRef.current?.abort(), []); // closing the panel stops a running check

  const run = useCallback(
    async (genes: number) => {
      if (running) return;
      const ctrl = new AbortController();
      abortRef.current = ctrl;
      setRunning(true);
      setError(null);
      setSaved(null);
      setStep(null);
      try {
        await runMergeCheck(
          datasetId,
          { ...request, max_genes: genes },
          {
            onStart: (s) => setEstimate(s.cached ? null : s.estimate_seconds),
            onProgress: (p) => setStep(p.label),
            onResult: (r) => {
              setResult(r);
              setTicked(defaultSelection(r.pairs));
            },
            signal: ctrl.signal,
          },
        );
      } catch (e) {
        if (!ctrl.signal.aborted) setError(e instanceof Error ? e.message : "The merge check failed.");
      } finally {
        if (abortRef.current === ctrl) abortRef.current = null;
        setRunning(false);
        setStep(null);
      }
    },
    [datasetId, request, running],
  );

  const changeThreshold = (n: number) => {
    setMaxGenes(n);
    if (result && !running) void run(n); // the server re-reads the same tests
  };

  const pairs = useMemo(
    () =>
      (result?.pairs ?? [])
        .filter((p) => showAll || p.candidate || p.evidence === "together")
        .sort((x, y) => Number(y.candidate) - Number(x.candidate) || EVIDENCE_ORDER[x.evidence] - EVIDENCE_ORDER[y.evidence]),
    [result, showAll],
  );
  const chosen = (result?.pairs ?? []).filter((p) => ticked.has(pairKey(p)));
  const after = result ? clustersAfter(result.n_reference_clusters, chosen) : null;

  const toggle = (p: MergePair) =>
    setTicked((prev) => {
      const next = new Set(prev);
      const k = pairKey(p);
      if (next.has(k)) next.delete(k);
      else next.add(k);
      return next;
    });

  const save = async () => {
    if (!result || chosen.length === 0) return;
    setSaving(true);
    setError(null);
    try {
      const res = await saveMergedClustering(datasetId, {
        ...(result.reference_exists ? { reference: result.reference } : { resolution: result.resolution, method: result.method }),
        pairs: chosen.map((p) => [p.a, p.b] as [string, string]),
      });
      await refresh(datasetId);
      useSettingsStore.getState().setColorBy(res.column);
      setSaved(res.column);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Could not save the merged clustering.");
    } finally {
      setSaving(false);
    }
  };

  const reference = result?.reference ?? info?.reference;

  return (
    <section aria-label="Merge check" className="space-y-1.5 border-t border-slate-200 pt-2">
      <div>
        <div className="text-[11px] font-medium text-slate-600">Should any clusters merge?</div>
        <p className="text-[10px] leading-snug text-slate-500">
          Splits every cell&rsquo;s counts into two independent halves (negative binomial count splitting), re-runs the
          pipeline on one half and tests neighbouring clusters on the other, so the test does not reuse the data that made
          the clusters.
        </p>
      </div>
      <div className="flex flex-wrap items-center gap-1.5">
        <select
          aria-label="Clustering to check"
          value={String(target)}
          disabled={running}
          onChange={(e) => setTarget(e.target.value === "active" ? "active" : Number(e.target.value))}
          className="rounded border border-slate-300 px-1 py-0.5 text-xs text-slate-700"
        >
          <option value="active">Active clustering</option>
          {rows.map((r) => (
            <option key={r.resolution} value={String(r.resolution)}>
              Resolution {formatResolutions([r.resolution])} ({r.n_clusters} clusters)
            </option>
          ))}
        </select>
        <label className="flex items-center gap-1 text-slate-500">
          merge below
          <select
            aria-label="Genes below which a pair is a merge candidate"
            value={maxGenes}
            disabled={running}
            onChange={(e) => changeThreshold(Number(e.target.value))}
            className="rounded border border-slate-300 px-1 py-0.5 text-xs text-slate-700"
          >
            {THRESHOLDS.map((n) => (
              <option key={n} value={n}>
                {n}
              </option>
            ))}
          </select>
          genes
        </label>
        {running ? (
          <button
            type="button"
            onClick={() => abortRef.current?.abort()}
            className="rounded border border-slate-300 px-3 py-0.5 text-xs text-slate-600 hover:bg-slate-50"
          >
            Stop
          </button>
        ) : (
          <button
            type="button"
            onClick={() => void run(maxGenes)}
            disabled={!info?.available}
            className="rounded bg-blue-500 px-3 py-0.5 text-xs font-medium text-white hover:bg-blue-600 disabled:opacity-50"
          >
            Check merges
          </button>
        )}
      </div>
      {info && !info.available && <p className="text-[11px] text-amber-700">{info.reason}</p>}
      {running ? (
        <div className="flex items-center gap-1.5 text-slate-500">
          <Loader2 className="h-3.5 w-3.5 animate-spin" />
          {step ?? "Starting"}
          {estimate != null ? ` (about ${formatDuration(estimate)} in all)` : ""}…
        </div>
      ) : (
        info?.available &&
        !result && (
          <p className="text-[10px] text-slate-400">
            Checks {prettyObsLabel(info.reference ?? "")}
            {info.estimate_seconds != null ? `: about ${formatDuration(info.estimate_seconds)}` : ""} (the first run after
            the server starts adds ~{formatDuration(info.first_run_seconds)}).
          </p>
        )
      )}

      {result && (
        <div className="space-y-1">
          <p className="text-[11px] text-slate-600">
            {prettyObsLabel(result.reference)}: the training half made {result.train.n_clusters} clusters at resolution{" "}
            {result.train_resolution} (ARI {result.train.ari.toFixed(2)} with these).{" "}
            <span className="font-medium">
              {result.n_candidates} merge candidate{result.n_candidates === 1 ? "" : "s"}
            </span>{" "}
            with fewer than {result.thresholds.merge_max_genes} genes at adjusted p &lt; {result.thresholds.padj} and two-fold
            on the held-out half.
          </p>
          {pairs.length > 0 && (
            <table className="w-full border-collapse text-[11px]">
              <thead>
                <tr className="border-b border-slate-200 text-left text-slate-500">
                  <th className="w-5 py-1" />
                  <th className="py-1 pr-2 font-medium">Pair</th>
                  <th className="py-1 pr-2 text-right font-medium">Cells</th>
                  <th className="py-1 pr-2 text-right font-medium" title="Genes distinguishing them on the held-out half">
                    Genes
                  </th>
                  <th className="py-1 pr-2 font-medium">Top genes</th>
                  <th className="py-1 text-right font-medium" title="Share of the smaller cluster's graph edges joining the two">
                    Adj.
                  </th>
                </tr>
              </thead>
              <tbody>
                {pairs.map((p) => (
                  <tr
                    key={pairKey(p)}
                    title={describePair(p, result.thresholds.merge_max_genes)}
                    className={`border-b border-slate-100 ${p.candidate ? "" : "text-slate-500"}`}
                  >
                    <td className="py-0.5">
                      <input
                        type="checkbox"
                        aria-label={`Merge ${p.a} and ${p.b}`}
                        checked={ticked.has(pairKey(p))}
                        onChange={() => toggle(p)}
                      />
                    </td>
                    <td className="py-0.5 pr-2 font-medium">
                      {p.a} + {p.b}
                    </td>
                    <td className="py-0.5 pr-2 text-right tabular-nums">
                      {p.n_cells_a.toLocaleString()} / {p.n_cells_b.toLocaleString()}
                    </td>
                    <td className={`py-0.5 pr-2 text-right tabular-nums ${p.candidate ? "font-semibold text-emerald-700" : ""}`}>
                      {p.n_genes ?? (p.evidence === "together" ? "not split" : "—")}
                    </td>
                    <td className="max-w-[10rem] truncate py-0.5 pr-2">{[...p.up.slice(0, 2), ...p.down.slice(0, 2)].join(", ")}</td>
                    <td className="py-0.5 text-right tabular-nums">{p.adjacency.toFixed(2)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
          <div className="flex flex-wrap items-center gap-2">
            <button
              type="button"
              onClick={() => setShowAll((s) => !s)}
              className="text-[11px] text-blue-600 hover:underline"
            >
              {showAll ? "Show candidates only" : `Show all ${result.pairs.length} neighbouring pairs`}
            </button>
            <button
              type="button"
              onClick={() => void save()}
              disabled={saving || chosen.length === 0}
              title={`Save as ${result.reference}_merged; the active clustering stays as it is`}
              className="ml-auto flex items-center gap-1 rounded border border-slate-300 px-2 py-0.5 text-slate-700 hover:bg-slate-50 disabled:opacity-50"
            >
              {saving && <Loader2 className="h-3 w-3 animate-spin" />}
              Save merged clustering
              {after != null && chosen.length > 0 ? ` (${result.n_reference_clusters} → ${after})` : ""}
            </button>
          </div>
          {saved && <p className="text-[11px] text-emerald-700">Saved as {prettyObsLabel(saved)}; the plot is coloured by it.</p>}
          <p className="text-[10px] leading-snug text-slate-400">
            The halves are independent only as far as each gene&rsquo;s overdispersion is right; the tested clusters are
            the training half&rsquo;s, standing for {reference ? prettyObsLabel(reference) : "these"}; one split, so a
            borderline pair can change on another. Few distinguishing genes supports a merge but does not prove one cell
            type. &ldquo;Not split&rdquo;: the training half kept both in one cluster, so nothing tested them.
          </p>
        </div>
      )}
      {error && <p role="alert" className="rounded bg-red-50 px-2 py-1.5 text-red-700">{error}</p>}
    </section>
  );
}
