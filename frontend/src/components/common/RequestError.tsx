import { isLargeRefusal } from "@/api/client";

/** A request's error in a panel. Work the backend does not do on a large dataset (a 422
 *  refusal, backend core/large_mode.py) is shown as a plain note, not as an error. */
export function RequestError({ message, className = "" }: { message: string | null | undefined; className?: string }) {
  if (!message) return null;
  if (isLargeRefusal(message)) {
    return (
      <p role="note" className={`rounded border border-slate-200 bg-slate-50 px-3 py-2 text-xs text-slate-600 ${className}`}>
        {message}
      </p>
    );
  }
  return (
    <p role="alert" className={`rounded border border-red-200 bg-red-50 px-3 py-2 text-xs text-red-700 ${className}`}>
      {message}
    </p>
  );
}
