import { useCallback, useEffect, useRef, useState } from "react";
import { ChevronDown, ChevronRight, FileText, Info, Loader2, RefreshCw, Sparkles } from "lucide-react";
import { getQcReport, postQcReport, streamQcReport, type QcReport } from "@/api/qc";
import { MarkdownLite } from "@/components/assistant/MarkdownLite";
import { useSelectionStore } from "@/stores/selectionStore";
import { useSettingsStore } from "@/stores/settingsStore";
import { useViewStore } from "@/stores/viewStore";

/** A cluster-QC chip (result:qc:<column>:<cluster>) jumps to that cluster. */
export function jumpToQcTag(tag: string): boolean {
  const m = /^result:qc:(.+):([^:]+)$/.exec(tag);
  if (!m) return false;
  useSettingsStore.getState().setColorBy(m[1]!);
  useSelectionStore.getState().setHighlight({ column: m[1]!, value: m[2]! });
  useViewStore.getState().setPanel("unified");
  return true;
}

function when(iso?: string | null): string {
  if (!iso) return "";
  const d = new Date(iso);
  return isNaN(d.getTime()) ? "" : d.toLocaleString();
}

/** "Explain this data": a short written QC report in four fixed sections, written
 *  by the LLM from the deterministic assessment facts only. Opening the tab shows
 *  the saved report for the current facts (no LLM call); "Regenerate" writes a new
 *  one. Without an API key the button lays the same facts out instead. */
export function QcReportCard({ datasetId, refreshKey = 0 }: { datasetId: string; refreshKey?: number }) {
  const [data, setData] = useState<QcReport | null>(null);
  const [text, setText] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [open, setOpen] = useState(true);
  // A stream still running for the previous dataset must not write into this one.
  const epoch = useRef(0);

  useEffect(() => {
    const mine = ++epoch.current;
    setData(null);
    setText("");
    setError(null);
    getQcReport(datasetId)
      .then((d) => {
        if (epoch.current !== mine) return;
        setData(d);
        setText(d.report);
      })
      .catch((e) => epoch.current === mine && setError(String(e?.message ?? e)));
  }, [datasetId, refreshKey]);

  const generate = useCallback(
    async (regenerate: boolean) => {
      const mine = epoch.current;
      const live = () => epoch.current === mine;
      setBusy(true);
      setError(null);
      setText("");
      setOpen(true);
      try {
        await streamQcReport(datasetId, regenerate, {
          onFacts: (ev) =>
            live() &&
            setData((d) => ({
              ...(d ?? { report: "", grounded: false, cached: false }),
              facts: ev.facts,
              facts_hash: ev.facts_hash,
              llm_available: ev.llm_available,
            })),
          onDelta: (t) => live() && setText((prev) => prev + t),
          onDone: (ev) =>
            live() &&
            setData((d) => (d ? { ...d, grounded: ev.grounded, cached: ev.cached, model: ev.model, generated_at: ev.generated_at } : d)),
          onError: (msg) => live() && setError(msg),
        });
      } catch {
        // Streaming unavailable: fall back to the one-shot endpoint.
        try {
          const res = await postQcReport(datasetId, regenerate);
          if (live()) {
            setData(res);
            setText(res.report);
          }
        } catch (e2) {
          if (live()) setError(e2 instanceof Error ? e2.message : String(e2));
        }
      } finally {
        if (live()) setBusy(false);
      }
    },
    [datasetId],
  );

  const hasReport = text.trim().length > 0;
  const llm = data?.llm_available ?? false;

  return (
    <div className="rounded-xl border border-slate-200 bg-white p-4">
      <div className="flex flex-wrap items-center gap-2">
        <button
          type="button"
          onClick={() => setOpen(!open)}
          aria-expanded={open}
          disabled={!hasReport}
          className="flex items-center gap-1 text-sm font-semibold text-slate-700 disabled:cursor-default"
        >
          {hasReport && (open ? <ChevronDown className="h-4 w-4" /> : <ChevronRight className="h-4 w-4" />)}
          <FileText className="h-4 w-4 text-primary" />
          QC report
        </button>
        <span className="flex-1" />
        {!hasReport || busy ? (
          <button
            type="button"
            onClick={() => generate(false)}
            disabled={busy}
            className="inline-flex items-center gap-1.5 rounded-lg border border-slate-200 bg-white px-3 py-1.5 text-sm font-medium text-slate-700 transition hover:bg-slate-50 disabled:cursor-not-allowed disabled:opacity-50"
          >
            {busy ? <Loader2 className="h-4 w-4 animate-spin" /> : <Sparkles className="h-4 w-4 text-amber-500" />}
            {busy ? "Writing…" : "Explain this data"}
          </button>
        ) : (
          <button
            type="button"
            onClick={() => generate(llm)}
            title={llm ? "Write a new report (a new language-model call)" : "Lay out the facts again"}
            className="inline-flex items-center gap-1 rounded px-2 py-1 text-xs text-slate-500 hover:bg-slate-50 hover:text-slate-700"
          >
            <RefreshCw className="h-3.5 w-3.5" />
            {llm ? "Regenerate" : "Refresh"}
          </button>
        )}
      </div>

      {!hasReport && !busy && (
        <p className="mt-1 text-xs text-slate-500">
          {data && !llm
            ? "No language model is configured, so this lays out the assessment facts in the report's sections."
            : "A short written report on this dataset's quality, from the assessment numbers only, with each claim cited."}
        </p>
      )}
      {error && <p className="mt-2 text-xs text-red-600">Could not write the report: {error}</p>}

      {open && (hasReport || busy) && (
        <div className="mt-3 border-t border-slate-100 pt-3">
          {hasReport ? (
            <MarkdownLite text={text} onCitation={jumpToQcTag} />
          ) : (
            <div className="flex items-center gap-2 text-sm text-slate-400">
              <Loader2 className="h-4 w-4 animate-spin" /> Writing the report…
            </div>
          )}
          {hasReport && !busy && data && (
            <div className="mt-3 space-y-1.5 border-t border-dashed border-slate-200 pt-2 text-[11px] text-slate-500">
              {data.grounded ? (
                <div className="flex flex-wrap items-center gap-1">
                  <Info className="h-3 w-3" />
                  AI-generated{data.model ? ` · ${data.model.split("/").pop()}` : ""} from the facts below — verify before use
                  {data.cached && data.generated_at && (
                    <span className="text-slate-400">· saved report from {when(data.generated_at)}, no new call</span>
                  )}
                </div>
              ) : (
                <div className="italic text-amber-600">Deterministic facts summary (no language model).</div>
              )}
              {data.facts.length > 0 && (
                <details>
                  <summary className="cursor-pointer hover:text-slate-700">
                    {data.facts.length} facts the report may use
                  </summary>
                  <ul className="mt-1 space-y-1 border-l-2 border-slate-100 pl-3">
                    {data.facts.map((s) => (
                      <li key={s.ref}>
                        <code className="rounded bg-slate-100 px-1 py-0.5 text-[10px] text-primary">{s.ref}</code>{" "}
                        {s.detail}
                      </li>
                    ))}
                  </ul>
                </details>
              )}
            </div>
          )}
        </div>
      )}
    </div>
  );
}
