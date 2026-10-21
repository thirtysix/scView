import { AlertTriangle, CheckCircle2, X } from "lucide-react";

/** A restore's outcome (lib/sessionRestore.ts RestoreReport, or an error sentence). */
export interface RestoreOutcome {
  headline: string;
  notes: string[];
  level: "ok" | "warning" | "error";
}

/**
 * What a restore did, in plain words: the headline, then what was not
 * restored. "Open session…" (SessionMenu) and opening a share link
 * (ShareLinkBanner) both show it.
 */
export function RestoreOutcomeCard({
  outcome,
  label,
  onDismiss,
  className = "",
}: {
  outcome: RestoreOutcome;
  /** The card's accessible name */
  label: string;
  onDismiss: () => void;
  className?: string;
}) {
  return (
    <div
      role={outcome.level === "error" ? "alert" : "status"}
      aria-label={label}
      className={`rounded-lg border p-3 text-xs shadow-lg ${
        outcome.level === "ok"
          ? "border-emerald-200 bg-emerald-50 text-emerald-900"
          : outcome.level === "warning"
            ? "border-amber-200 bg-amber-50 text-amber-900"
            : "border-red-200 bg-red-50 text-red-800"
      } ${className}`}
    >
      <div className="flex items-start gap-2">
        {outcome.level === "ok" ? (
          <CheckCircle2 className="mt-0.5 h-4 w-4 flex-shrink-0 text-emerald-600" />
        ) : (
          <AlertTriangle className="mt-0.5 h-4 w-4 flex-shrink-0" />
        )}
        <div className="min-w-0 flex-1 space-y-1">
          <p className="font-medium">{outcome.headline}</p>
          {outcome.notes.length > 0 && (
            <>
              <p>Not restored:</p>
              <ul className="list-disc space-y-0.5 pl-4">
                {outcome.notes.map((n) => (
                  <li key={n}>{n}</li>
                ))}
              </ul>
            </>
          )}
        </div>
        <button onClick={onDismiss} aria-label="Dismiss" className="rounded p-0.5 hover:bg-black/5">
          <X className="h-3.5 w-3.5" />
        </button>
      </div>
    </div>
  );
}
