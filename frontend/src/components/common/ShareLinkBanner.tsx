import { Loader2 } from "lucide-react";
import { useShareLinkRestore } from "@/hooks/useShareLinkRestore";
import { RestoreOutcomeCard } from "./RestoreOutcomeCard";

/**
 * Opens a share link (`#view=...`) when the app loads or one is pasted into
 * the address bar, and says what was restored and what was not, until dismissed.
 */
export function ShareLinkBanner() {
  const { outcome, busy, dismiss } = useShareLinkRestore();
  if (busy) {
    return (
      <div
        role="status"
        aria-label="Opening share link"
        className="mb-4 flex items-center gap-2 rounded-lg border border-slate-200 bg-white p-3 text-xs text-slate-600 shadow-sm"
      >
        <Loader2 className="h-4 w-4 animate-spin" />
        Opening the view from the link…
      </div>
    );
  }
  if (!outcome) return null;
  return (
    <RestoreOutcomeCard
      outcome={outcome}
      label="Share link"
      onDismiss={dismiss}
      className="mb-4 shadow-sm"
    />
  );
}
