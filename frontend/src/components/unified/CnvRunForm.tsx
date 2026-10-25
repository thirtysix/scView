import { useEffect, useMemo, useState } from "react";
import { Loader2, Play, Square } from "lucide-react";
import type { CnvDefaults, CnvRunnable, CnvSummary } from "@/api/cnv";
import { InfoTip } from "@/components/common/InfoTip";
import { formatDuration } from "@/lib/hvgSensitivity";
import { progressText, referenceColumns, suggestReferenceColumn, suggestReferenceGroups } from "@/lib/cnv";
import { prettyObsLabel } from "@/lib/formatting";
import { useCnvRunStore } from "@/stores/cnvRunStore";
import type { ObsColumnInfo } from "@/api/types";

const inputClass = "w-14 rounded border border-slate-300 bg-white px-1.5 py-0.5 text-xs text-slate-700";

function gb(bytes: number): string {
  return bytes >= 1 << 30 ? `${(bytes / (1 << 30)).toFixed(1)} GB` : `${Math.max(1, Math.round(bytes / (1 << 20)))} MB`;
}

/**
 * The run form of the Copy number subtab: the reference (column + normal groups),
 * window, step, whether chromosome X counts, Run and Cancel, and the run's progress
 * (the run lives in useCnvRunStore, so it goes on when the tab changes).
 */
export function CnvRunForm({
  datasetId,
  columns,
  nCells,
  runnable,
  defaults,
  estimate,
  saved,
}: {
  datasetId: string;
  columns: ObsColumnInfo[];
  nCells: number;
  runnable: CnvRunnable;
  defaults: CnvDefaults;
  estimate?: CnvSummary["estimate"];
  saved: CnvSummary | null;
}) {
  const candidates = useMemo(() => referenceColumns(columns, nCells), [columns, nCells]);
  const [column, setColumn] = useState(() => saved?.reference_column ?? suggestReferenceColumn(candidates));
  const values = useMemo(() => columns.find((c) => c.name === column)?.values ?? [], [columns, column]);
  const [groups, setGroups] = useState<string[]>(() => saved?.reference_groups ?? []);
  const [windowSize, setWindowSize] = useState(saved?.params?.window ?? defaults.window);
  const [step, setStep] = useState(saved?.params?.step ?? defaults.step);
  const [dropX, setDropX] = useState(saved?.params?.exclude?.includes("chrX") ?? false);
  const run = useCnvRunStore((s) => s.run);
  const cancel = useCnvRunStore((s) => s.cancel);
  const state = useCnvRunStore();
  const mine = state.datasetId === datasetId;
  const running = state.running;

  // A new column: suggest its immune / stromal groups (kept when it is the saved one).
  useEffect(() => {
    if (saved?.reference_column === column && saved.reference_groups) setGroups(saved.reference_groups);
    else setGroups(suggestReferenceGroups(values));
  }, [column, values, saved?.reference_column, saved?.reference_groups]);

  if (!runnable.ok) {
    return (
      <p role="alert" className="rounded-md bg-amber-50 px-3 py-2 text-xs text-amber-800">
        Copy number can&apos;t be inferred on this dataset: {runnable.reason}
      </p>
    );
  }

  const windowOk = Number.isInteger(windowSize) && windowSize >= defaults.min_window && windowSize <= defaults.max_window;
  const stepOk = Number.isInteger(step) && step >= 1 && step <= Math.min(defaults.max_step, windowSize);
  const allRef = values.length > 0 && groups.length >= values.length;
  const problem = !column
    ? "Choose the column with the cell types."
    : groups.length === 0
      ? "Tick at least one group of normal cells."
      : allRef
        ? "Leave out the groups you want to test (they cannot all be the reference)."
        : !windowOk
          ? `The window must be ${defaults.min_window} to ${defaults.max_window} genes.`
          : !stepOk
            ? "The step must be at least 1 and at most the window."
            : null;
  const start = () => {
    if (problem) return;
    const exclude = dropX ? [...defaults.exclude, "chrX"] : defaults.exclude;
    run(datasetId, { reference_column: column, reference_groups: groups, window: windowSize, step, exclude_chromosomes: exclude });
  };
  const toggle = (g: string) => setGroups((cur) => (cur.includes(g) ? cur.filter((x) => x !== g) : [...cur, g]));
  const p = mine ? state.progress : null;
  const est = mine ? state.start?.estimate_seconds : estimate?.seconds;

  return (
    <div className="space-y-2 rounded-lg border border-slate-200 p-2.5" data-testid="cnv-run-form">
      <p className="text-[11px] leading-snug text-slate-600">
        Tumour cells often gain or lose whole chromosome arms, which nudges all the genes there up or down
        together. Comparing each cell with <strong>normal reference cells</strong> from the same samples (immune or
        stromal cells, which keep two copies) shows those changes. Pick the normal groups below.
      </p>
      <label className="block text-[11px] font-medium text-slate-500">
        Reference column
        <select
          aria-label="Reference column"
          value={column}
          onChange={(e) => setColumn(e.target.value)}
          className="mt-0.5 w-full rounded border border-slate-300 bg-white px-2 py-1.5 text-xs text-slate-700"
        >
          {candidates.map((c) => (
            <option key={c.name} value={c.name}>
              {prettyObsLabel(c.name)} ({c.n_unique} groups)
            </option>
          ))}
        </select>
      </label>
      {values.length > 0 && (
        <fieldset>
          <legend className="mb-0.5 text-[11px] font-medium text-slate-500">
            Normal reference groups ({groups.length} ticked)
          </legend>
          <div className="flex max-h-28 flex-wrap gap-x-3 gap-y-0.5 overflow-auto">
            {values.map((v) => (
              <label key={v} className="flex items-center gap-1 text-xs text-slate-700">
                <input type="checkbox" checked={groups.includes(v)} onChange={() => toggle(v)} />
                {v}
              </label>
            ))}
          </div>
        </fieldset>
      )}
      <div className="flex flex-wrap items-center gap-x-3 gap-y-1.5 text-xs text-slate-600">
        <label className="flex items-center gap-1" title="Genes averaged in each running window">
          window
          <input
            type="number"
            aria-label="Window (genes)"
            min={defaults.min_window}
            max={defaults.max_window}
            value={windowSize}
            onChange={(e) => setWindowSize(Number(e.target.value))}
            className={inputClass}
          />
        </label>
        <label className="flex items-center gap-1" title="Keep every n-th window (smaller result, faster)">
          step
          <input
            type="number"
            aria-label="Step (genes)"
            min={1}
            max={defaults.max_step}
            value={step}
            onChange={(e) => setStep(Number(e.target.value))}
            className={inputClass}
          />
        </label>
        <label className="flex items-center gap-1" title="Leave chromosome X out: it differs between sexes, so a tumour and reference from different donors can look like an X change">
          <input type="checkbox" checked={dropX} onChange={(e) => setDropX(e.target.checked)} />
          leave out X
        </label>
        <InfoTip label="About the method" align="right" width={300}>
          inferCNV-style (Tirosh et al. 2016; as infercnvpy): log expression of the {runnable.n_placed?.toLocaleString()}{" "}
          genes placed on {runnable.assembly} ({runnable.source}) minus the reference groups&apos; means, clipped to
          ±3, averaged over {windowSize} neighbouring genes per chromosome, centred per cell, small values set to 0. Y and
          mitochondrial genes are left out. A new run replaces the saved one.
        </InfoTip>
      </div>
      {problem && <p className="text-[11px] text-amber-700">{problem}</p>}
      <div className="flex flex-wrap items-center gap-2">
        {!running ? (
          <button
            onClick={start}
            disabled={!!problem}
            className="flex items-center gap-1 rounded-md bg-primary px-2.5 py-1 text-xs font-medium text-white disabled:opacity-50"
          >
            <Play className="h-3 w-3" />
            {saved?.available ? "Run again" : "Infer copy number"}
          </button>
        ) : (
          <button
            onClick={cancel}
            disabled={!mine}
            className="flex items-center gap-1 rounded-md border border-slate-300 px-2.5 py-1 text-xs font-medium text-slate-700 hover:bg-slate-50"
            title="Stop the run; nothing is saved"
          >
            <Square className="h-3 w-3" />
            Cancel
          </button>
        )}
        {running && mine && (
          <span className="flex items-center gap-1.5 text-[11px] text-slate-500" data-testid="cnv-progress">
            <Loader2 className="h-3 w-3 animate-spin" />
            {progressText(p)}
            {p && ` · ${formatDuration(p.elapsed_s)}`}
            {est !== undefined && ` · about ${formatDuration(est)} in all`}
          </span>
        )}
        {!running && estimate && (
          <span className="text-[11px] text-slate-500">
            About {formatDuration(estimate.seconds)} and {gb(estimate.need_bytes)} of memory
            {estimate.level === "wont_fit" ? ": more than is free" : estimate.level === "tight" ? ": tight" : ""}.
          </span>
        )}
        {running && !mine && <span className="text-[11px] text-slate-500">A run is working on another dataset.</span>}
      </div>
      {mine &&
        state.warnings.map((w, i) => (
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
