import { useEffect, useState } from "react";
import { Check, Loader2 } from "lucide-react";
import { Help } from "@/components/common/Help";
import {
  getStoragePlan,
  shortDiskText,
  storeByRows,
  storeByRowsLabel,
  type StoragePlan,
  type StoreByRowsProgress,
} from "@/api/storage";
import { errorMessage } from "@/lib/errorMessage";

/**
 * The one offered action for a dataset whose matrix is stored by columns, shown only where
 * that blocks something: a run that goes to large-dataset mode (the run preflight's
 * `mode.by_columns`, or the run's own refusal). A click stores the dataset by rows, with
 * progress, then `onDone` (the caller runs the refused action again, or offers it again).
 * Never rendered for a dataset where nothing is blocked, and never started without the click.
 */
export function StoreByRowsOffer({
  datasetId,
  plan: given,
  doneText = "Stored by rows.",
  onDone,
}: {
  datasetId: string;
  /** From the run preflight; without it the plan is asked for (the run was refused) */
  plan?: StoragePlan | null;
  /** What is said once it is done, e.g. "Stored by rows. Running the steps again." */
  doneText?: string;
  onDone?: () => void;
}) {
  const [asked, setAsked] = useState<StoragePlan | null>(null);
  const [progress, setProgress] = useState<StoreByRowsProgress | null>(null);
  const [state, setState] = useState<"idle" | "working" | "done">("idle");
  const [error, setError] = useState<string | null>(null);
  const plan = given ?? asked;

  useEffect(() => {
    if (given) return;
    const ctl = new AbortController();
    getStoragePlan(datasetId, ctl.signal)
      .then(setAsked)
      .catch(() => {
        /* the offer still names the action; its cost is unknown */
      });
    return () => ctl.abort();
  }, [datasetId, given]);

  const start = async () => {
    setState("working");
    setError(null);
    setProgress(null);
    try {
      await storeByRows(datasetId, setProgress);
      setState("done");
      onDone?.();
    } catch (e) {
      setError(errorMessage(e));
      setState("idle");
    }
  };

  const short = plan ? shortDiskText(plan) : null;
  const percent = progress && progress.total > 0 ? Math.floor((100 * progress.done) / progress.total) : 0;
  return (
    <div
      className="mt-1 rounded-md border border-amber-300 bg-amber-50 px-2.5 py-1.5 text-[11px] leading-snug text-amber-900"
      data-testid="store-by-rows-offer"
    >
      <p>
        <span className="font-medium">This dataset is stored by columns.</span> Large-dataset mode reads cells in
        blocks, which that layout cannot give, so this run cannot start as it is.
      </p>
      {state === "done" ? (
        <p className="mt-1 flex items-center gap-1 font-medium text-emerald-700" role="status">
          <Check className="h-3.5 w-3.5 shrink-0" aria-hidden />
          {doneText}
        </p>
      ) : state === "working" ? (
        <div className="mt-1.5" role="status">
          <p className="flex items-center gap-1.5">
            <Loader2 className="h-3.5 w-3.5 shrink-0 animate-spin" aria-hidden />
            Storing by rows: {percent}%. The dataset stays as it is until the new file is whole.
          </p>
          <div
            role="progressbar"
            aria-label="Storing by rows"
            aria-valuemin={0}
            aria-valuemax={100}
            aria-valuenow={percent}
            className="mt-1 h-1.5 w-full overflow-hidden rounded-full bg-amber-200"
          >
            <div className="h-full bg-amber-600 transition-all" style={{ width: `${percent}%` }} />
          </div>
        </div>
      ) : (
        <p className="mt-1.5">
          <Help id="storage.by-rows" side="below">
            <button
              type="button"
              onClick={start}
              disabled={short != null}
              className="rounded border border-amber-400 bg-white px-1.5 py-0.5 font-medium text-amber-900 hover:bg-amber-100 disabled:cursor-not-allowed disabled:opacity-60"
            >
              {plan && plan.files > 0 ? storeByRowsLabel(plan) : "Store this dataset by rows"}
            </button>
          </Help>
        </p>
      )}
      {short && state === "idle" && <p className="mt-1">{short}</p>}
      {error && (
        <p className="mt-1 text-red-700" role="alert">
          {error}
        </p>
      )}
    </div>
  );
}
