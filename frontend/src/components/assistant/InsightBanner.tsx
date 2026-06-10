import { useEffect, useState } from "react";
import { Sparkles, X } from "lucide-react";
import { useDatasetStore } from "@/stores/datasetStore";
import { useViewStore } from "@/stores/viewStore";
import { getInsight, type DatasetInsight } from "@/api/assistant";

const dismissKey = (id: string) => `scview.insight.dismissed.${id}`;

/**
 * A proactive "I notice…" banner shown when a dataset opens. The insight is a
 * deterministic next-step nudge from the backend; the optional question is a
 * click-to-ask follow-up handed to the co-pilot. Dismissal is remembered per
 * dataset+insight, so it won't nag — but a changed analysis state (a new
 * insight) surfaces again.
 */
export function InsightBanner() {
  const datasetId = useDatasetStore((s) => s.currentDatasetId);
  const askCopilot = useViewStore((s) => s.askCopilot);
  const [insight, setInsight] = useState<DatasetInsight | null>(null);

  useEffect(() => {
    if (!datasetId) {
      setInsight(null);
      return;
    }
    let cancelled = false;
    getInsight(datasetId)
      .then((ins) => {
        if (cancelled) return;
        if (!ins.insight) {
          setInsight(null);
          return;
        }
        let dismissed = "";
        try {
          dismissed = localStorage.getItem(dismissKey(datasetId)) ?? "";
        } catch {
          /* ignore */
        }
        setInsight(dismissed === ins.insight ? null : ins);
      })
      .catch(() => {
        if (!cancelled) setInsight(null);
      });
    return () => {
      cancelled = true;
    };
  }, [datasetId]);

  if (!datasetId || !insight) return null;

  const dismiss = () => {
    try {
      localStorage.setItem(dismissKey(datasetId), insight.insight);
    } catch {
      /* ignore */
    }
    setInsight(null);
  };

  return (
    <div className="mb-4 flex items-start gap-3 rounded-xl border border-primary/20 bg-primary/5 px-4 py-3">
      <Sparkles className="mt-0.5 h-4 w-4 flex-shrink-0 text-primary" />
      <div className="min-w-0 flex-1">
        <p className="text-sm text-slate-700">{insight.insight}</p>
        {insight.question && (
          <button
            onClick={() => {
              askCopilot(insight.question!);
              dismiss();
            }}
            className="mt-1.5 inline-flex items-center gap-1 rounded-full border border-primary/30 bg-white px-2.5 py-1 text-xs font-medium text-primary hover:bg-primary/10"
          >
            <Sparkles className="h-3 w-3" />
            {insight.question}
          </button>
        )}
      </div>
      <button
        onClick={dismiss}
        title="Dismiss"
        className="flex-shrink-0 text-slate-400 hover:text-slate-600"
      >
        <X className="h-4 w-4" />
      </button>
    </div>
  );
}
