import { describe, it, expect, vi, beforeEach } from "vitest";
import { act, fireEvent, render, screen } from "@testing-library/react";
import { useUnifiedViewStore } from "@/stores/unifiedViewStore";

vi.mock("@/components/plots/ViolinPlot", () => ({ ViolinPlot: () => <div data-testid="violin" /> }));
vi.mock("@/components/unified/ClusterDistanceHeatmap", () => ({
  ClusterDistanceHeatmap: ({ defaultColumn }: { defaultColumn: string }) => (
    <div data-testid="distances">{defaultColumn}</div>
  ),
}));

import { UnifiedBottomPanel } from "./UnifiedBottomPanel";

const COLUMNS = [{ name: "cluster", n_unique: 13 }];
const renderPanel = (props: Partial<Parameters<typeof UnifiedBottomPanel>[0]> = {}) =>
  render(
    <UnifiedBottomPanel
      violinData={{ B: [1, 2] }}
      violinTitle="CD79A expression"
      datasetId="kang"
      distanceColumn="cluster"
      distanceColumns={COLUMNS}
      {...props}
    />,
  );

beforeEach(() => {
  useUnifiedViewStore.setState({ bottomPanelOpen: false, bottomView: "distribution" });
});

describe("UnifiedBottomPanel", () => {
  it("opens the violin or the cluster distances from their headers, and closes the open one", () => {
    renderPanel();
    expect(screen.queryByTestId("violin")).toBeNull();
    fireEvent.click(screen.getByRole("button", { name: /Distribution/ }));
    expect(screen.getByTestId("violin")).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "Cluster distances" }));
    expect(screen.queryByTestId("violin")).toBeNull();
    expect(screen.getByTestId("distances")).toHaveTextContent("cluster");
    fireEvent.click(screen.getByRole("button", { name: "Cluster distances" }));
    expect(screen.queryByTestId("distances")).toBeNull();
    expect(useUnifiedViewStore.getState().bottomPanelOpen).toBe(false);
  });

  it("shows the distances when the caveat asks, and the violin again for a new gene", () => {
    renderPanel();
    act(() => useUnifiedViewStore.getState().showDistances());
    expect(screen.getByTestId("distances")).toBeInTheDocument();
    act(() => useUnifiedViewStore.getState().openBottomPanel()); // a gene was clicked
    expect(screen.getByTestId("violin")).toBeInTheDocument();
  });

  it("offers no distances without a grouping column", () => {
    useUnifiedViewStore.setState({ bottomPanelOpen: true, bottomView: "distances" });
    renderPanel({ distanceColumn: "" });
    expect(screen.queryByRole("button", { name: "Cluster distances" })).toBeNull();
    expect(screen.getByTestId("violin")).toBeInTheDocument();
  });
});
