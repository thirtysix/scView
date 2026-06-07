import { useViewStore } from "@/stores/viewStore";
import { ErrorBoundary } from "@/components/common/ErrorBoundary";
import { DataPanel } from "@/components/panels/DataPanel";
import { DataAssessmentPanel } from "@/components/panels/DataAssessmentPanel";
import { OverviewPanel } from "@/components/panels/OverviewPanel";
import { UnifiedViewPanel } from "@/components/panels/UnifiedViewPanel";
import { ObservationsPanel } from "@/components/panels/ObservationsPanel";
import { GeneExpressionPanel } from "@/components/panels/GeneExpressionPanel";
import { GeneSetPanel } from "@/components/panels/GeneSetPanel";
import { MarkerGenesPanel } from "@/components/panels/MarkerGenesPanel";
import { TrajectoryPanel } from "@/components/panels/TrajectoryPanel";
import { ProvenancePanel } from "@/components/panels/ProvenancePanel";
import { AssistantPanel } from "@/components/panels/AssistantPanel";
import type { PanelId } from "@/lib/constants";

const PANEL_COMPONENTS: Record<PanelId, React.ComponentType> = {
  load: DataPanel,
  assessment: DataAssessmentPanel,
  overview: OverviewPanel,
  unified: UnifiedViewPanel,
  observations: ObservationsPanel,
  expression: GeneExpressionPanel,
  genesets: GeneSetPanel,
  markers: MarkerGenesPanel,
  trajectory: TrajectoryPanel,
  provenance: ProvenancePanel,
  assistant: AssistantPanel,
};

export function PanelContainer() {
  const activePanel = useViewStore((s) => s.activePanel);
  const Panel = PANEL_COMPONENTS[activePanel];
  return (
    <ErrorBoundary key={activePanel}>
      <Panel />
    </ErrorBoundary>
  );
}
