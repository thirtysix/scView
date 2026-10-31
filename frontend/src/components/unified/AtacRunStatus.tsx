import { Loader2 } from "lucide-react";
import { RequestError } from "@/components/common/RequestError";
import { stageText } from "@/lib/atac";
import { useAtacRunStore, type AtacRunKind } from "@/stores/atacRunStore";

/** One run's progress with Cancel, its memory warnings and its error; nothing when it is not this card's. */
export function AtacRunStatus({ datasetId, kind }: { datasetId: string; kind: AtacRunKind }) {
  const run = useAtacRunStore();
  if (run.datasetId !== datasetId || run.kind !== kind) return null;
  return (
    <>
      {run.running && (
        <div className="space-y-1" data-testid={`atac-progress-${kind}`}>
          <p className="flex items-center gap-1.5 text-[11px] text-slate-600">
            <Loader2 className="h-3 w-3 animate-spin" />
            {kind === "differential" ? run.progress ?? "Starting…" : stageText(run.stage)}
          </p>
          {run.stage && (
            <div className="h-1 w-full rounded bg-slate-100">
              <div className="h-1 rounded bg-primary" style={{ width: `${Math.round(((run.stage.index - 1) / run.stage.total) * 100)}%` }} />
            </div>
          )}
          <button type="button" onClick={() => run.cancel()}
                  className="rounded border border-slate-300 px-2 py-0.5 text-[11px] text-slate-600 hover:border-slate-400">
            Cancel
          </button>
        </div>
      )}
      <RequestError message={run.error} />
      {run.warnings.map((w) => (
        <p key={w} className="text-[11px] text-amber-700">{w}</p>
      ))}
    </>
  );
}
