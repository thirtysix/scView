import { useEffect, useRef, useState } from "react";
import { AlertTriangle, CheckCircle2, ChevronDown, FolderOpen, Loader2, Save, X } from "lucide-react";
import { readSessionFile, sessionFilename, sessionJson } from "@/lib/session";
import { captureSession, restoreSession, type RestoreReport } from "@/lib/sessionRestore";
import { useDatasetStore } from "@/stores/datasetStore";

type Outcome = RestoreReport | { headline: string; notes: string[]; level: "error" };

/**
 * Header "Session" menu: "Save session" downloads the workspace as a small
 * JSON file (lib/session.ts); "Open session…" restores one and says plainly
 * what could not be restored (another dataset, a missing column...).
 */
export function SessionMenu() {
  const dataset = useDatasetStore((s) => s.currentDataset);
  const [open, setOpen] = useState(false);
  const [busy, setBusy] = useState(false);
  const [outcome, setOutcome] = useState<Outcome | null>(null);
  const ref = useRef<HTMLDivElement>(null);
  const fileRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    if (!open) return;
    const onDown = (e: MouseEvent) => {
      if (ref.current && !ref.current.contains(e.target as Node)) setOpen(false);
    };
    document.addEventListener("mousedown", onDown);
    return () => document.removeEventListener("mousedown", onDown);
  }, [open]);

  const save = () => {
    const now = new Date();
    const blob = new Blob([sessionJson(captureSession(now))], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = sessionFilename(dataset?.name, now);
    a.click();
    URL.revokeObjectURL(url);
    setOpen(false);
  };

  const restore = async (file: File) => {
    setOpen(false);
    setBusy(true);
    setOutcome(null);
    try {
      setOutcome(await restoreSession(await readSessionFile(file)));
    } catch (e) {
      // SessionError and restoreSession's errors are already plain sentences.
      const message = e instanceof Error ? e.message : `The session could not be opened: ${String(e)}`;
      setOutcome({ headline: message, notes: [], level: "error" });
    } finally {
      setBusy(false);
    }
  };

  const itemCls =
    "flex w-full items-start gap-2.5 px-4 py-2 text-left text-sm text-slate-700 transition-colors hover:bg-slate-50 disabled:cursor-not-allowed disabled:opacity-50";

  return (
    <div ref={ref} className="relative inline-block">
      <button
        onClick={() => setOpen((o) => !o)}
        aria-haspopup="menu"
        aria-expanded={open}
        className="flex items-center gap-1.5 rounded-lg border border-slate-300 bg-white px-3 py-1.5 text-sm font-medium text-slate-700 shadow-sm transition-colors hover:bg-slate-50"
      >
        {busy ? <Loader2 className="h-4 w-4 animate-spin" /> : <Save className="h-4 w-4" />}
        Session
        <ChevronDown className="h-3.5 w-3.5" />
      </button>
      <input
        ref={fileRef}
        type="file"
        accept=".json,application/json"
        aria-label="Session file"
        className="hidden"
        onChange={(e) => {
          const file = e.target.files?.[0];
          e.target.value = ""; // the same file can be opened again
          if (file) void restore(file);
        }}
      />

      {open && (
        <div role="menu" className="absolute right-0 z-50 mt-1 w-72 rounded-lg border border-slate-200 bg-white py-1 shadow-lg">
          <button role="menuitem" onClick={save} disabled={!dataset} className={itemCls}>
            <Save className="mt-0.5 h-4 w-4 flex-shrink-0 text-blue-500" />
            <span>
              <span className="block">Save session</span>
              <span className="block text-xs text-slate-400">
                {dataset
                  ? "Download this workspace (view, selection, settings) as a .json file"
                  : "Open a dataset first"}
              </span>
            </span>
          </button>
          <button role="menuitem" onClick={() => fileRef.current?.click()} disabled={busy} className={itemCls}>
            <FolderOpen className="mt-0.5 h-4 w-4 flex-shrink-0 text-violet-500" />
            <span>
              <span className="block">Open session…</span>
              <span className="block text-xs text-slate-400">Restore a saved .json session</span>
            </span>
          </button>
        </div>
      )}

      {outcome && (
        <div
          role={outcome.level === "error" ? "alert" : "status"}
          aria-label="Session restore"
          className={`absolute right-0 z-50 mt-1 w-96 rounded-lg border p-3 text-xs shadow-lg ${
            outcome.level === "ok"
              ? "border-emerald-200 bg-emerald-50 text-emerald-900"
              : outcome.level === "warning"
                ? "border-amber-200 bg-amber-50 text-amber-900"
                : "border-red-200 bg-red-50 text-red-800"
          }`}
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
            <button onClick={() => setOutcome(null)} aria-label="Dismiss" className="rounded p-0.5 hover:bg-black/5">
              <X className="h-3.5 w-3.5" />
            </button>
          </div>
        </div>
      )}
    </div>
  );
}
