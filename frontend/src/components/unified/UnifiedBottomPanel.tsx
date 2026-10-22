import { useState } from "react";
import { ChevronDown, ChevronUp } from "lucide-react";
import { useUnifiedViewStore, type BottomView } from "@/stores/unifiedViewStore";
import { ViolinPlot } from "@/components/plots/ViolinPlot";
import { Caption } from "@/components/common/Caption";
import { ClusterDistanceHeatmap } from "@/components/unified/ClusterDistanceHeatmap";
import { PerDonorExpression } from "@/components/unified/PerDonorExpression";
import { SelectionHistogram } from "@/components/unified/SelectionHistogram";

type DistributionMode = "cells" | "donor" | "histogram";

const MODES: { id: DistributionMode; label: string; title: string }[] = [
  { id: "cells", label: "Cells", title: "Every cell (violin)" },
  { id: "donor", label: "Per donor", title: "One pseudobulk value per donor in each group, split by condition" },
  {
    id: "histogram",
    label: "Histogram",
    title: "The selection (or a group) against the rest, as two overlaid normalised histograms",
  },
];

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
  const overlay = useUnifiedViewStore((s) => s.scatterOverlay);
  // The distribution as cells (violin), as one pseudobulk value per donor, or as
  // overlaid histograms of a selection or group against the rest (N18).
  const [mode, setMode] = useState<DistributionMode>("cells");
  const perDonor = mode === "donor";
  const gene = overlay?.type === "expression" && !overlay.layer ? overlay.gene : null;
  const noGeneReason =
    overlay?.type === "score"
      ? "Per-donor values are for one gene; a score is per cell. Pick a gene."
      : overlay?.type === "program"
        ? "Per-donor values are for one gene; a program's usage is per cell. Pick a gene."
        : overlay?.type === "expression"
          ? "Per-donor values are for genes; proteins are not supported yet."
          : undefined;

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
          {shown === "distribution" && mode !== "histogram" &&
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
          />
        </div>
      )}
      {isOpen && shown === "distribution" && (
        <div className="border-t border-slate-100 px-2 py-1">
          {datasetId && (
            <div className="mb-1 flex justify-end">
              <div className="flex overflow-hidden rounded border border-slate-300 text-[11px]" role="group" aria-label="Distribution of">
                {MODES.filter((m) => m.id !== "donor" || violinGroupColumn).map((m) => (
                  <button
                    key={m.id}
                    aria-pressed={mode === m.id}
                    onClick={() => setMode(m.id)}
                    title={m.title}
                    className={`px-1.5 py-0.5 ${mode === m.id ? "bg-primary/10 font-medium text-primary" : "bg-white text-slate-600 hover:bg-slate-50"}`}
                  >
                    {m.label}
                  </button>
                ))}
              </div>
            </div>
          )}
          {mode === "histogram" && datasetId ? (
            <SelectionHistogram datasetId={datasetId} groupColumn={violinGroupColumn} />
          ) : perDonor && datasetId && violinGroupColumn ? (
            <PerDonorExpression
              datasetId={datasetId}
              gene={gene}
              noGeneReason={noGeneReason}
              groupColumn={violinGroupColumn}
            />
          ) : hasData ? (
            <>
              <ViolinPlot
                data={violinData}
                groupColumn={violinGroupColumn}
                title={violinTitle}
                xLabel={violinGroupLabel}
                yLabel="Value"
                height={220}
              />
              <Caption
                testId="violin-caption"
                className="px-1"
                n={[
                  { value: Object.values(violinData).reduce((s, v) => s + v.length, 0), unit: "cells" },
                  { value: Object.keys(violinData).length, unit: "groups" },
                ]}
              >
                {violinTitle ?? "Values"} per {violinGroupLabel ?? violinGroupColumn}: one violin per group.
              </Caption>
            </>
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
