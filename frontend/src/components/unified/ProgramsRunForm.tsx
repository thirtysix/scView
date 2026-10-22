import { useState } from "react";
import { Loader2, Play, Square } from "lucide-react";
import type { ProgramsDefaults, ProgramsRunnable } from "@/api/programs";
import { InfoTip } from "@/components/common/InfoTip";
import { formatDuration } from "@/lib/hvgSensitivity";
import { parseSweep } from "@/lib/programs";
import { useProgramsRunStore } from "@/stores/programsRunStore";

const inputClass = "w-14 rounded border border-slate-300 bg-white px-1.5 py-0.5 text-xs text-slate-700";

/**
 * The run form of the Programs subtab: k, seeds, an optional sweep over k, Run and
 * Cancel, and the run's progress (the run lives in useProgramsRunStore, so it goes
 * on when the tab changes).
 */
export function ProgramsRunForm({
  datasetId,
  runnable,
  defaults,
  hasResult,
}: {
  datasetId: string;
  runnable: ProgramsRunnable;
  defaults: ProgramsDefaults;
  hasResult: boolean;
}) {
  const [k, setK] = useState(defaults.k);
  const [seeds, setSeeds] = useState(defaults.seeds);
  const [doSweep, setDoSweep] = useState(false);
  const [sweepText, setSweepText] = useState(defaults.sweep.join(", "));
  const run = useProgramsRunStore((s) => s.run);
  const cancel = useProgramsRunStore((s) => s.cancel);
  const state = useProgramsRunStore();
  const mine = state.datasetId === datasetId;
  const running = state.running;
  const sweep = doSweep ? parseSweep(sweepText, defaults.min_k, defaults.max_k, defaults.max_sweep) : [];
  const sweepError = typeof sweep === "string" ? sweep : null;
  const kOk = Number.isInteger(k) && k >= defaults.min_k && k <= defaults.max_k;
  const seedsOk = Number.isInteger(seeds) && seeds >= 1 && seeds <= defaults.max_seeds;

  if (!runnable.ok) {
    return (
      <p role="alert" className="rounded-md bg-amber-50 px-3 py-2 text-xs text-amber-800">
        Gene programs can&apos;t run on this dataset: {runnable.reason}
      </p>
    );
  }

  const start = () => {
    if (!kOk || !seedsOk || sweepError) return;
    run(datasetId, { k, seeds, ...(Array.isArray(sweep) && sweep.length ? { sweep } : {}) });
  };

  const p = mine ? state.progress : null;
  const est = mine ? state.start?.estimate_seconds : undefined;
  return (
    <div className="space-y-2 rounded-lg border border-slate-200 p-2.5" data-testid="programs-run-form">
      <div className="flex flex-wrap items-center gap-x-3 gap-y-1.5 text-xs text-slate-600">
        <label className="flex items-center gap-1" title={`Programs to find (${defaults.min_k} to ${defaults.max_k})`}>
          k
          <input
            type="number"
            aria-label="Number of programs (k)"
            min={defaults.min_k}
            max={defaults.max_k}
            value={k}
            onChange={(e) => setK(Number(e.target.value))}
            className={inputClass}
          />
        </label>
        <label
          className="flex items-center gap-1"
          title="NMF runs from different random starts (seeds 0, 1, ...); a program is stable when every run finds it"
        >
          seeds
          <input
            type="number"
            aria-label="Number of seeds"
            min={1}
            max={defaults.max_seeds}
            value={seeds}
            onChange={(e) => setSeeds(Number(e.target.value))}
            className={inputClass}
          />
        </label>
        <label
          className="flex items-center gap-1"
          title="Also fit these values of k and compare their stability (silhouette) and reconstruction error"
        >
          <input type="checkbox" checked={doSweep} onChange={(e) => setDoSweep(e.target.checked)} />
          compare k
        </label>
        {doSweep && (
          <input
            aria-label="Values of k to compare"
            value={sweepText}
            onChange={(e) => setSweepText(e.target.value)}
            className="w-28 rounded border border-slate-300 bg-white px-1.5 py-0.5 text-xs text-slate-700"
          />
        )}
        <InfoTip label="About the run" align="right" width={280}>
          NMF on the {runnable.n_flagged?.toLocaleString()} variable genes ({runnable.source}), each
          divided by its standard deviation, from {seeds} random start{seeds === 1 ? "" : "s"}; the runs&apos;
          programs are matched by k-means and their median kept (consensus NMF, Kotliar et al. 2019).
          Compare k fits each value the same way, then saves the programs for k = {k}. A new run replaces
          the saved programs.
        </InfoTip>
      </div>
      {(sweepError || !kOk || !seedsOk) && (
        <p className="text-[11px] text-amber-700">
          {sweepError ??
            (!kOk
              ? `k must be a whole number from ${defaults.min_k} to ${defaults.max_k}.`
              : `Use 1 to ${defaults.max_seeds} seeds.`)}
        </p>
      )}
      <div className="flex items-center gap-2">
        {!running ? (
          <button
            onClick={start}
            disabled={!kOk || !seedsOk || !!sweepError}
            className="flex items-center gap-1 rounded-md bg-primary px-2.5 py-1 text-xs font-medium text-white disabled:opacity-50"
          >
            <Play className="h-3 w-3" />
            {hasResult ? "Run again" : "Find programs"}
          </button>
        ) : (
          <button
            onClick={cancel}
            disabled={!mine}
            className="flex items-center gap-1 rounded-md border border-slate-300 px-2.5 py-1 text-xs font-medium text-slate-700 hover:bg-slate-50"
            title="Stop after the current NMF run; nothing is saved"
          >
            <Square className="h-3 w-3" />
            Cancel
          </button>
        )}
        {running && mine && (
          <span className="flex items-center gap-1.5 text-[11px] text-slate-500" data-testid="programs-progress">
            <Loader2 className="h-3 w-3 animate-spin" />
            {p
              ? `${p.done} of ${p.total} NMF runs done (k = ${p.k}), ${formatDuration(p.elapsed_s)}`
              : "Reading the variable genes…"}
            {est !== undefined && ` · about ${formatDuration(est)} in all`}
          </span>
        )}
        {running && !mine && <span className="text-[11px] text-slate-500">A run is working on another dataset.</span>}
      </div>
      {mine && state.warnings.map((w, i) => (
        <p key={i} className="text-[11px] text-amber-700">
          {w}
        </p>
      ))}
      {mine && state.error && (
        <p role="alert" className="rounded-md bg-amber-50 px-2 py-1 text-xs text-amber-800">
          {state.error}
        </p>
      )}
    </div>
  );
}
