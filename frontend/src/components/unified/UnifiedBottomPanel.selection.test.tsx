import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen } from "@testing-library/react";
import { useSelectionStore } from "@/stores/selectionStore";
import { useUnifiedViewStore } from "@/stores/unifiedViewStore";

const { violinProps } = vi.hoisted(() => ({ violinProps: { current: null as Record<string, unknown> | null } }));
vi.mock("@/components/plots/ViolinPlot", () => ({
  ViolinPlot: (p: Record<string, unknown>) => {
    violinProps.current = p;
    return <div data-testid="violin" />;
  },
}));
vi.mock("@/components/unified/ClusterDistanceHeatmap", () => ({ ClusterDistanceHeatmap: () => null }));
vi.mock("@/components/unified/PerDonorExpression", () => ({ PerDonorExpression: () => null }));
vi.mock("@/components/unified/SelectionHistogram", () => ({ SelectionHistogram: () => null }));

import { UnifiedBottomPanel } from "./UnifiedBottomPanel";

const renderPanel = () =>
  render(<UnifiedBottomPanel violinData={{ B: [1, 2], T: [0] }} violinTitle="CD79A expression" datasetId="kang" />);

describe("UnifiedBottomPanel violin with a selection (C035)", () => {
  beforeEach(() => {
    violinProps.current = null;
    useUnifiedViewStore.setState({
      bottomPanelOpen: true,
      bottomView: "distribution",
      scatterOverlay: { type: "expression", gene: "CD79A" },
      overlayValues: Float32Array.from([1, 2, 0, 7]),
      overlayDatasetId: "kang",
    });
    useSelectionStore.setState({ selectedCellIndices: null });
  });

  it("adds the selected cells' values (from the plot's overlay) as one more violin", () => {
    useSelectionStore.setState({ selectedCellIndices: new Set([1, 3]) });
    renderPanel();
    const extra = violinProps.current!.extra as { name: string; values: Float32Array };
    expect(extra.name).toBe("Selection (2 cells)");
    expect([...extra.values]).toEqual([2, 7]);
    expect(screen.getByTestId("violin-caption")).toHaveTextContent("the last (orange) is the 2 selected cells");
    expect(screen.getByTestId("selection-chip")).toBeInTheDocument();
  });

  it("has no extra violin without a selection, or when the overlay is another dataset's", () => {
    renderPanel();
    expect(violinProps.current!.extra).toBeNull();
    useSelectionStore.setState({ selectedCellIndices: new Set([1]) });
    useUnifiedViewStore.setState({ overlayDatasetId: "other" });
    renderPanel();
    expect(violinProps.current!.extra).toBeNull();
  });
});
