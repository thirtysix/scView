import { ChevronDown, ChevronUp } from "lucide-react";
import { useUnifiedViewStore } from "@/stores/unifiedViewStore";
import { ViolinPlot } from "@/components/plots/ViolinPlot";

interface UnifiedBottomPanelProps {
  violinData: Record<string, number[]>;
  violinTitle?: string;
  violinGroupLabel?: string;
}

export function UnifiedBottomPanel({
  violinData,
  violinTitle,
  violinGroupLabel,
}: UnifiedBottomPanelProps) {
  const isOpen = useUnifiedViewStore((s) => s.bottomPanelOpen);
  const toggle = useUnifiedViewStore((s) => s.toggleBottomPanel);

  const hasData = Object.keys(violinData).length > 0;

  return (
    <div className="flex-shrink-0 overflow-hidden rounded-lg border border-slate-200 bg-white shadow-sm">
      {/* Header */}
      <button
        onClick={toggle}
        className="flex w-full items-center justify-between px-3 py-1.5 text-xs font-medium text-slate-600 hover:bg-slate-50"
      >
        <span>Distribution {violinTitle ? `\u2014 ${violinTitle}` : ""}</span>
        <span className="flex items-center gap-2">
          {!hasData && <span className="text-slate-400">No data</span>}
          {isOpen ? (
            <ChevronDown className="h-3.5 w-3.5" />
          ) : (
            <ChevronUp className="h-3.5 w-3.5" />
          )}
        </span>
      </button>

      {/* Content */}
      {isOpen && (
        <div className="border-t border-slate-100 px-2 py-1">
          {hasData ? (
            <ViolinPlot
              data={violinData}
              title={violinTitle}
              xLabel={violinGroupLabel}
              yLabel="Value"
              height={220}
            />
          ) : (
            <div className="flex items-center justify-center py-8 text-xs text-slate-400">
              Select a gene or score a gene set to see its distribution
            </div>
          )}
        </div>
      )}
    </div>
  );
}
