import { useEffect, useRef, useState } from "react";
import { Check, Link2, Loader2, X } from "lucide-react";
import { buildShareUrl, describeSharedView, readShareHash } from "@/lib/shareUrl";
import { captureSession } from "@/lib/sessionRestore";
import { useDatasetStore } from "@/stores/datasetStore";

interface Made {
  url: string;
  /** What opening it restores */
  restores: string[];
  /** Left out to keep it short */
  dropped: string[];
  copied: boolean;
}

/** Copy to the clipboard where the page may (https or localhost); false otherwise. */
async function copyText(text: string): Promise<boolean> {
  if (typeof window !== "undefined" && window.isSecureContext === false) return false;
  if (!navigator.clipboard?.writeText) return false;
  try {
    await navigator.clipboard.writeText(text);
    return true;
  } catch {
    return false;
  }
}

/**
 * Header "Copy link": the current view (dataset, tab, embedding, colouring,
 * overlay, highlight, camera, DE settings, colours, a small selection) as a
 * URL (lib/shareUrl.ts). The popover says what the link restores and what was
 * left out; where the browser won't copy (plain http over the network) the
 * link is shown selected, to copy by hand.
 */
export function ShareLinkButton() {
  const dataset = useDatasetStore((s) => s.currentDataset);
  const [busy, setBusy] = useState(false);
  const [made, setMade] = useState<Made | null>(null);
  const [error, setError] = useState<string | null>(null);
  const ref = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLInputElement>(null);
  const open = made != null || error != null;

  const close = () => {
    setMade(null);
    setError(null);
  };

  useEffect(() => {
    if (!open) return;
    const shut = () => {
      setMade(null);
      setError(null);
    };
    const onDown = (e: MouseEvent) => {
      if (ref.current && !ref.current.contains(e.target as Node)) shut();
    };
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") shut();
    };
    document.addEventListener("mousedown", onDown);
    document.addEventListener("keydown", onKey);
    return () => {
      document.removeEventListener("mousedown", onDown);
      document.removeEventListener("keydown", onKey);
    };
  }, [open]);

  // Not copied: the link selected, ready for Ctrl+C.
  useEffect(() => {
    if (made && !made.copied) inputRef.current?.select();
  }, [made]);

  const make = async () => {
    setBusy(true);
    setError(null);
    try {
      const { url, dropped } = await buildShareUrl(captureSession(), window.location);
      // What it restores, read back from the link itself.
      const shared = await readShareHash(new URL(url).hash);
      const copied = await copyText(url);
      setMade({ url, dropped, copied, restores: shared ? describeSharedView(shared) : [] });
    } catch (e) {
      setMade(null);
      setError(`The link could not be made: ${e instanceof Error ? e.message : String(e)}`);
    } finally {
      setBusy(false);
    }
  };

  return (
    <div ref={ref} className="relative inline-block">
      <span className="inline-flex" title={dataset ? "Copy a link that reopens this view" : "Open a dataset first"}>
        <button
          onClick={() => (open ? close() : void make())}
          disabled={!dataset || busy}
          aria-haspopup="dialog"
          aria-expanded={open}
          className="flex items-center gap-1.5 rounded-lg border border-slate-300 bg-white px-3 py-1.5 text-sm font-medium text-slate-700 shadow-sm transition-colors hover:bg-slate-50 disabled:cursor-not-allowed disabled:opacity-50"
        >
          {busy ? <Loader2 className="h-4 w-4 animate-spin" /> : <Link2 className="h-4 w-4" />}
          Copy link
        </button>
      </span>

      {open && (
        <div
          role="dialog"
          aria-label="Share link"
          className="absolute right-0 z-50 mt-1 w-[26rem] space-y-2 rounded-lg border border-slate-200 bg-white p-3 text-xs text-slate-700 shadow-lg"
        >
          <div className="flex items-start gap-2">
            <p className="min-w-0 flex-1 font-medium text-slate-900">
              {error ? (
                <span className="text-red-700">{error}</span>
              ) : made?.copied ? (
                <span className="inline-flex items-center gap-1 text-emerald-700">
                  <Check className="h-3.5 w-3.5" /> Link copied to the clipboard.
                </span>
              ) : (
                "Copy this link (Ctrl+C or ⌘C): the browser lets a page copy for you only over https or on localhost."
              )}
            </p>
            <button onClick={close} aria-label="Close" className="rounded p-0.5 text-slate-400 hover:bg-slate-100">
              <X className="h-3.5 w-3.5" />
            </button>
          </div>
          {made && (
            <>
              <input
                ref={inputRef}
                readOnly
                value={made.url}
                aria-label="Link to this view"
                onFocus={(e) => e.currentTarget.select()}
                className="w-full rounded border border-slate-300 bg-slate-50 px-2 py-1 font-mono text-[11px] text-slate-700"
              />
              <p className="text-[11px] text-slate-400" title="Links over about 6,000 characters leave out the selection, then the camera">
                {made.url.length.toLocaleString("en-US")} characters
              </p>
              {made.restores.length > 0 && (
                <div>
                  <p>Opening it restores:</p>
                  <ul className="list-disc space-y-0.5 pl-4">
                    {made.restores.map((r) => (
                      <li key={r}>{r}</li>
                    ))}
                  </ul>
                </div>
              )}
              {made.dropped.length > 0 && (
                <p className="rounded border border-amber-200 bg-amber-50 px-2 py-1 text-amber-900">
                  Left out to keep the link short: {made.dropped.join("; ")}. Save a session (Session menu) to keep
                  {made.dropped.length > 1 ? " them" : " it"}.
                </p>
              )}
              <p className="text-slate-500">
                Anyone opening it needs this dataset loaded in their own scView; the link carries the view, not the
                data. Gene lists and gates stay out of it.
              </p>
            </>
          )}
        </div>
      )}
    </div>
  );
}
