import { Loader2 } from "lucide-react";
import { RequestError } from "@/components/common/RequestError";
import { runShare, runText } from "@/lib/spatialDomains";
import { useDatasetStore } from "@/stores/datasetStore";
import { useSpatialDomainsRunStore, type RunKind } from "@/stores/spatialDomainsRunStore";

const OUTLINE =
  "rounded border border-slate-300 px-2 py-0.5 text-[11px] text-slate-700 hover:border-primary hover:text-primary disabled:opacity-50";

/** A Domains tab run of this kind: progress, Cancel, its error and memory warnings. */
export function SpatialRunStatus({ kind }: { kind: RunKind }) {
  const datasetId = useDatasetStore((s) => s.currentDatasetId);
  const run = useSpatialDomainsRunStore();
  if (run.datasetId !== datasetId || run.kind !== kind) return null;
  return (
    <>
      {run.running && (
        <div className="space-y-1" data-testid={`${kind}-progress`}>
          <p className="flex items-center gap-1.5 text-[11px] text-slate-600">
            <Loader2 className="h-3 w-3 animate-spin" />
            {runText(run.stage, run.progress)}
          </p>
          <div className="h-1 w-full rounded bg-slate-100">
            <div className="h-1 rounded bg-primary" style={{ width: `${Math.round(runShare(run.stage, run.progress) * 100)}%` }} />
          </div>
          <button type="button" onClick={() => run.cancel()} className={OUTLINE}>
            Cancel
          </button>
        </div>
      )}
      <RequestError message={run.error} />
      {run.warnings.map((w) => (
        <p key={w} className="text-[11px] text-amber-700">
          {w}
        </p>
      ))}
    </>
  );
}
