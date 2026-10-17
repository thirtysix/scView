import { ChevronDown, ChevronUp } from "lucide-react";
import { useUnifiedViewStore, type BottomView } from "@/stores/unifiedViewStore";
import { ViolinPlot } from "@/components/plots/ViolinPlot";
import { ClusterDistanceHeatmap } from "@/components/unified/ClusterDistanceHeatmap";

interface UnifiedBottomPanelProps {
  violinData: Record<string, number[]>;
  violinTitle?: string;
  violinGroupLabel?: string;
  /** Obs column the violin groups come from (for category colours). */
  violinGroupColumn?: string;
  isLoading?: boolean;
  /** For the cluster-distance view: the dataset, the column it opens on, and
   *  the categorical columns it offers. */
  datasetId?: string | null;
  distanceColumn?: string;
  distanceColumns?: readonly { name: string; n_unique: number }[];
}

export function UnifiedBottomPanel({
  violinData,
  violinTitle,
  violinGroupLabel,
  violinGroupColumn,
  isLoading = false,
  datasetId = null,
  distanceColumn = "",
  distanceColumns = [],
}: UnifiedBottomPanelProps) {
  const isOpen = useUnifiedViewStore((s) => s.bottomPanelOpen);
  const toggle = useUnifiedViewStore((s) => s.toggleBottomPanel);
  const view = useUnifiedViewStore((s) => s.bottomView);
  const setView = useUnifiedViewStore((s) => s.setBottomView);

  const hasData = Object.keys(violinData).length > 0;
  const canDistances = !!datasetId && !!distanceColumn;
  const shown: BottomView = view === "distances" && canDistances ? "distances" : "distribution";
  // A view's header opens it; the open view's header closes the panel.
  const pick = (v: BottomView) => {
    if (isOpen && shown === v) toggle();
    else {
      setView(v);
      if (!isOpen) toggle();
    }
  };
  const tab = (v: BottomView) =>
    `truncate rounded px-1.5 py-0.5 hover:bg-slate-100 ${
      isOpen && shown === v ? "font-semibold text-slate-700" : "text-slate-500"
    }`;
  const nGroups = Math.max(2, distanceColumns.find((c) => c.name === distanceColumn)?.n_unique ?? 10);

  return (
    <div className="flex-shrink-0 overflow-hidden rounded-lg border border-slate-200 bg-white shadow-sm">
      {/* Header */}
      <div className="flex w-full items-center gap-1 px-2 py-1 text-xs font-medium">
        <button onClick={() => pick("distribution")} className={`min-w-0 ${tab("distribution")}`}>
          Distribution {violinTitle ? `— ${violinTitle}` : ""}
        </button>
        {canDistances && (
          <button
            onClick={() => pick("distances")}
            title="Distances between groups in PCA space, which a UMAP or t-SNE plot does not show"
            className={`flex-shrink-0 ${tab("distances")}`}
          >
            Cluster distances
          </button>
        )}
        <button
          onClick={toggle}
          aria-label={isOpen ? "Collapse" : "Expand"}
          className="ml-auto flex flex-shrink-0 items-center gap-2 rounded px-1 py-0.5 text-slate-600 hover:bg-slate-100"
        >
          {shown === "distribution" &&
            (isLoading ? (
              <span className="text-slate-400">Loading…</span>
            ) : (
              !hasData && <span className="text-slate-400">No data</span>
            ))}
          {isOpen ? <ChevronDown className="h-3.5 w-3.5" /> : <ChevronUp className="h-3.5 w-3.5" />}
        </button>
      </div>

      {/* Content */}
      {isOpen && shown === "distances" && datasetId && (
        <div className="border-t border-slate-100 px-2 py-1">
          <ClusterDistanceHeatmap
            datasetId={datasetId}
            defaultColumn={distanceColumn}
            columns={distanceColumns}
            height={Math.min(420, Math.max(240, nGroups * 16 + 110))}
          />
        </div>
      )}
      {isOpen && shown === "distribution" && (
        <div className="border-t border-slate-100 px-2 py-1">
          {hasData ? (
            <ViolinPlot
              data={violinData}
              groupColumn={violinGroupColumn}
              title={violinTitle}
              xLabel={violinGroupLabel}
              yLabel="Value"
              height={220}
            />
          ) : (
            <div className="flex items-center justify-center py-8 text-xs text-slate-400">
              {isLoading
                ? "Loading distribution…"
                : "Select a gene or score a gene set to see its distribution"}
            </div>
          )}
        </div>
      )}
    </div>
  );
}
