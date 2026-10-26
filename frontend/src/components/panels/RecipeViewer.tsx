import { useCallback, useEffect, useState } from "react";
import { ChevronDown, ChevronRight, FileCode2, FileJson, FileText, Loader2 } from "lucide-react";

import { getRecipe, type RecipeStep } from "@/api/provenance";
import { downloadRecipeFile, type RecipeFile } from "@/api/export";
import { paramText } from "@/lib/provenanceParams";
import { REPLAY_LABELS, replayCounts, replayStatus } from "@/lib/replayStatus";
import { useViewStore } from "@/stores/viewStore";
import { stepLabel } from "@/lib/stepLabel";

function fmtTime(iso?: string | null): string {
  if (!iso) return "time not recorded";
  const d = new Date(iso);
  return Number.isNaN(d.getTime()) ? iso : d.toLocaleString();
}

/**
 * "Reproduce this analysis" (C006): the recorded recipe step by step with its
 * parameters, time and scView version, how a replay script reproduces each
 * step, and downloads of that script and of a RUN_MANIFEST.
 */
export function RecipeViewer({ datasetId }: { datasetId: string }) {
  const [steps, setSteps] = useState<RecipeStep[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [open, setOpen] = useState(false);
  const [busy, setBusy] = useState<RecipeFile | null>(null);
  const [downloadError, setDownloadError] = useState<string | null>(null);
  const setCopilotOpen = useViewStore((s) => s.setCopilotOpen);

  useEffect(() => {
    let live = true;
    setSteps(null);
    setError(null);
    getRecipe(datasetId)
      .then((r) => live && setSteps(r.recipe))
      .catch((e) => live && setError(e instanceof Error ? e.message : "Could not load the recipe."));
    return () => {
      live = false;
    };
  }, [datasetId]);

  const download = useCallback(
    async (file: RecipeFile) => {
      setBusy(file);
      setDownloadError(null);
      try {
        await downloadRecipeFile(datasetId, file);
      } catch (e) {
        setDownloadError(e instanceof Error ? e.message : "Download failed.");
      } finally {
        setBusy(null);
      }
    },
    [datasetId],
  );

  if (error) return <div className="rounded-lg bg-red-50 p-3 text-sm text-red-700">{error}</div>;
  if (!steps) {
    return (
      <div className="flex items-center gap-2 text-sm text-slate-500">
        <Loader2 className="h-4 w-4 animate-spin text-primary" /> Loading recipe…
      </div>
    );
  }

  const counts = replayCounts(steps);
  const button =
    "flex items-center gap-1.5 rounded-lg border border-slate-200 px-3 py-1.5 text-xs font-medium text-slate-700 hover:bg-slate-50 disabled:opacity-50";

  return (
    <section className="rounded-xl border border-slate-200 bg-white p-4" aria-label="Reproduce this analysis">
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div className="min-w-0">
          <h3 className="text-sm font-semibold text-slate-700">Reproduce this analysis</h3>
          <p className="mt-0.5 text-xs text-slate-500">
            A Python script (scanpy) that replays the recorded steps from the input file, and a run
            manifest with the versions and the input file's checksum.
          </p>
          {counts.length > 0 && (
            <p className="mt-1 flex flex-wrap gap-1 text-xs" data-testid="replay-counts">
              {counts.map(([st, n]) => (
                <span key={st} className={`rounded px-1.5 py-0.5 ${REPLAY_LABELS[st].cls}`} title={REPLAY_LABELS[st].help}>
                  {n} {REPLAY_LABELS[st].label}
                </span>
              ))}
            </p>
          )}
        </div>
        <div className="flex flex-wrap gap-2">
          <button className={button} onClick={() => download("script")} disabled={busy !== null || steps.length === 0}>
            {busy === "script" ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <FileCode2 className="h-3.5 w-3.5" />}
            Replay script (.py)
          </button>
          <button className={button} onClick={() => download("manifest")} disabled={busy !== null}>
            {busy === "manifest" ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <FileJson className="h-3.5 w-3.5" />}
            Run manifest (.json)
          </button>
          <button
            className={button}
            onClick={() => setCopilotOpen(true)}
            title="Opens the AI Co-pilot: its Write methods button drafts a methods paragraph from this recipe"
          >
            <FileText className="h-3.5 w-3.5" /> Methods text (AI Co-pilot)
          </button>
        </div>
      </div>
      {downloadError && (
        <div role="alert" className="mt-2 rounded-lg bg-red-50 p-2 text-xs text-red-700">
          {downloadError}
        </div>
      )}

      {steps.length === 0 ? (
        <p className="mt-3 text-xs text-slate-500">No steps are recorded yet, so there is nothing to replay.</p>
      ) : (
        <>
          <button
            onClick={() => setOpen((o) => !o)}
            aria-expanded={open}
            className="mt-3 flex items-center gap-1 text-xs font-medium text-slate-600 hover:text-primary"
          >
            {open ? <ChevronDown className="h-3.5 w-3.5" /> : <ChevronRight className="h-3.5 w-3.5" />}
            Recipe ({steps.length} step{steps.length > 1 ? "s" : ""})
          </button>
          {open && (
            <ol className="mt-2 divide-y divide-slate-100 text-xs" aria-label="Recipe steps">
              {steps.map((s, i) => (
                <RecipeRow key={s.commit_id ?? i} index={i + 1} step={s} />
              ))}
            </ol>
          )}
        </>
      )}
    </section>
  );
}

function RecipeRow({ index, step }: { index: number; step: RecipeStep }) {
  const meta = REPLAY_LABELS[replayStatus(step)];
  const params = Object.entries(step.params ?? {}).filter(
    ([, v]) => v !== null && v !== "" && !(Array.isArray(v) && v.length === 0),
  );
  return (
    <li className="flex gap-3 py-2">
      <span className="w-5 shrink-0 text-right tabular-nums text-slate-400">{index}</span>
      <div className="min-w-0 flex-1">
        <div className="flex flex-wrap items-baseline gap-x-2 gap-y-0.5">
          <span className="font-medium text-slate-800">{stepLabel(step.step)}</span>
          <span className={`rounded px-1.5 py-0.5 ${meta.cls}`} title={meta.help}>
            {meta.label}
          </span>
          {step.tool && <span className="text-slate-400">{step.tool}</span>}
        </div>
        <div className="mt-0.5 text-slate-400">
          {fmtTime(step.timestamp)} · scView {step.scview_version ?? "?"}
          {step.effect && ` · ${step.effect.n_cells.toLocaleString()} cells × ${step.effect.n_genes.toLocaleString()} genes`}
        </div>
        {step.replay?.note && <div className="mt-0.5 text-slate-500">{step.replay.note}</div>}
        {params.length > 0 && (
          <div className="mt-1 flex flex-wrap gap-1">
            {params.map(([k, v]) => (
              <span key={k} className="rounded bg-slate-100 px-1.5 py-0.5 text-slate-600">
                {paramText(k, v)}
              </span>
            ))}
          </div>
        )}
      </div>
    </li>
  );
}
