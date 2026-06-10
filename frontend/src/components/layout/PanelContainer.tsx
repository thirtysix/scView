import { lazy, Suspense } from "react";
import { Loader2 } from "lucide-react";
import { useViewStore } from "@/stores/viewStore";
import { ErrorBoundary } from "@/components/common/ErrorBoundary";
import type { PanelId } from "@/lib/constants";

// Lazy-load every panel so heavy dependencies (Plotly, deck.gl, Arrow) ship in
// per-panel chunks loaded on demand, not in the initial bundle. The landing
// "load" panel pulls none of them. Named exports → map to default for React.lazy.
const DataPanel = lazy(() =>
  import("@/components/panels/DataPanel").then((m) => ({ default: m.DataPanel })),
);
const DataAssessmentPanel = lazy(() =>
  import("@/components/panels/DataAssessmentPanel").then((m) => ({ default: m.DataAssessmentPanel })),
);
const UnifiedViewPanel = lazy(() =>
  import("@/components/panels/UnifiedViewPanel").then((m) => ({ default: m.UnifiedViewPanel })),
);
const ObservationsPanel = lazy(() =>
  import("@/components/panels/ObservationsPanel").then((m) => ({ default: m.ObservationsPanel })),
);
const GeneExpressionPanel = lazy(() =>
  import("@/components/panels/GeneExpressionPanel").then((m) => ({ default: m.GeneExpressionPanel })),
);
const GeneSetPanel = lazy(() =>
  import("@/components/panels/GeneSetPanel").then((m) => ({ default: m.GeneSetPanel })),
);
const MarkerGenesPanel = lazy(() =>
  import("@/components/panels/MarkerGenesPanel").then((m) => ({ default: m.MarkerGenesPanel })),
);
const TrajectoryPanel = lazy(() =>
  import("@/components/panels/TrajectoryPanel").then((m) => ({ default: m.TrajectoryPanel })),
);
const ProvenancePanel = lazy(() =>
  import("@/components/panels/ProvenancePanel").then((m) => ({ default: m.ProvenancePanel })),
);
const AssistantPanel = lazy(() =>
  import("@/components/panels/AssistantPanel").then((m) => ({ default: m.AssistantPanel })),
);

const PANEL_COMPONENTS: Record<PanelId, React.ComponentType> = {
  load: DataPanel,
  assessment: DataAssessmentPanel,
  unified: UnifiedViewPanel,
  observations: ObservationsPanel,
  expression: GeneExpressionPanel,
  genesets: GeneSetPanel,
  markers: MarkerGenesPanel,
  trajectory: TrajectoryPanel,
  provenance: ProvenancePanel,
  assistant: AssistantPanel,
};

function PanelFallback() {
  return (
    <div className="flex h-full items-center justify-center py-24 text-slate-400">
      <Loader2 className="h-6 w-6 animate-spin" />
    </div>
  );
}

export function PanelContainer() {
  const activePanel = useViewStore((s) => s.activePanel);
  const Panel = PANEL_COMPONENTS[activePanel];
  return (
    <ErrorBoundary key={activePanel}>
      <Suspense fallback={<PanelFallback />}>
        <Panel />
      </Suspense>
    </ErrorBoundary>
  );
}
