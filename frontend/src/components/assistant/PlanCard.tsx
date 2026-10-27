import { AlertTriangle, Check, Circle, Clock, ListChecks, Loader2, Minus, Play, Square, X } from "lucide-react";
import type { PlanTurn, StepStatus } from "@/lib/copilotPlan";

const STATUS_TEXT: Record<StepStatus, string> = {
  pending: "to do",
  running: "running",
  done: "done",
  failed: "failed",
  skipped: "not run",
};

function StepIcon({ status }: { status: StepStatus }) {
  const cls = "h-3.5 w-3.5 flex-shrink-0";
  if (status === "running") return <Loader2 className={`${cls} animate-spin text-primary`} />;
  if (status === "done") return <Check className={`${cls} text-emerald-600`} />;
  if (status === "failed") return <X className={`${cls} text-red-600`} />;
  if (status === "skipped") return <Minus className={`${cls} text-slate-400`} />;
  return <Circle className={`${cls} text-slate-300`} />;
}

/** A co-pilot plan as a numbered checklist with one Run plan confirm. Steps run one at
 *  a time; each is checked before the next (lib/copilotPlan.ts). */
export function PlanCard({
  plan,
  onRun,
  onStop,
  onDismiss,
  canRun = true,
}: {
  plan: PlanTurn;
  onRun: () => void;
  /** While running: stop after the current step. */
  onStop: () => void;
  /** Before running: drop the plan. */
  onDismiss: () => void;
  /** False while another plan or run is going. */
  canRun?: boolean;
}) {
  const running = plan.status === "running";
  return (
    <div className="mt-2 rounded-lg border border-amber-200 bg-amber-50 p-3 text-xs" data-testid="copilot-plan">
      <div className="mb-1.5 flex items-center gap-1.5 font-medium text-amber-900">
        <ListChecks className="h-3.5 w-3.5" />
        Plan: {plan.steps.length} step{plan.steps.length === 1 ? "" : "s"}, run in order
      </div>
      {plan.advisory && (
        <div className="mb-1.5 flex items-start gap-1.5 text-amber-800">
          <AlertTriangle className="mt-0.5 h-3.5 w-3.5 flex-shrink-0" />
          <span>{plan.advisory}</span>
        </div>
      )}
      <ol className="mb-2 space-y-1">
        {plan.steps.map((s, i) => {
          const st = plan.stepStatus[i] ?? "pending";
          return (
            <li key={i} className="flex items-start gap-1.5 text-slate-700">
              <span className="w-4 text-right text-slate-500">{i + 1}.</span>
              <span className="mt-px" title={STATUS_TEXT[st]} aria-label={STATUS_TEXT[st]}>
                <StepIcon status={st} />
              </span>
              <span className="flex-1">
                {s.label || s.type}
                {s.requires_confirm && s.estimate && (
                  <span className="ml-1 inline-flex items-center gap-0.5 text-amber-700">
                    <Clock className="h-3 w-3" />
                    {s.estimate}
                  </span>
                )}
                {s.requires_confirm && s.advisory && (
                  <span className="block text-[11px] text-slate-500">{s.advisory}</span>
                )}
              </span>
            </li>
          );
        })}
      </ol>
      {plan.message && (
        <p
          className={`mb-2 ${plan.status === "failed" ? "text-red-700" : plan.status === "done" ? "text-emerald-700" : "text-slate-600"}`}
          role="status"
        >
          {plan.message}
        </p>
      )}
      {plan.status === "ready" && (
        <div className="flex gap-2">
          <button
            type="button"
            disabled={!canRun}
            onClick={onRun}
            className="inline-flex items-center gap-1 rounded-md bg-primary px-3 py-1 font-medium text-white hover:bg-primary/90 disabled:opacity-50"
          >
            <Play className="h-3.5 w-3.5" />
            Run plan
          </button>
          <button
            type="button"
            onClick={onDismiss}
            className="inline-flex items-center gap-1 rounded-md border border-slate-300 px-3 py-1 font-medium text-slate-600 hover:bg-slate-50"
          >
            <X className="h-3.5 w-3.5" />
            Cancel
          </button>
        </div>
      )}
      {running && (
        <button
          type="button"
          onClick={onStop}
          className="inline-flex items-center gap-1 rounded-md border border-slate-300 px-3 py-1 font-medium text-slate-600 hover:bg-slate-50"
        >
          <Square className="h-3.5 w-3.5" />
          Cancel after this step
        </button>
      )}
    </div>
  );
}
