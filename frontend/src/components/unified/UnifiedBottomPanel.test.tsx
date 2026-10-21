import { describe, it, expect, vi, beforeEach } from "vitest";
import { act, fireEvent, render, screen } from "@testing-library/react";
import { useUnifiedViewStore } from "@/stores/unifiedViewStore";

vi.mock("@/components/plots/ViolinPlot", () => ({ ViolinPlot: () => <div data-testid="violin" /> }));
vi.mock("@/components/unified/ClusterDistanceHeatmap", () => ({
  ClusterDistanceHeatmap: ({ defaultColumn }: { defaultColumn: string }) => (
    <div data-testid="distances">{defaultColumn}</div>
  ),
}));

vi.mock("@/components/unified/PerDonorExpression", () => ({
  PerDonorExpression: ({ gene, groupColumn, noGeneReason }: { gene: string | null; groupColumn: string; noGeneReason?: string }) => (
    <div data-testid="per-donor">{`${gene ?? noGeneReason ?? "none"} by ${groupColumn}`}</div>
  ),
}));

vi.mock("@/components/unified/SelectionHistogram", () => ({
  SelectionHistogram: ({ groupColumn }: { groupColumn?: string }) => (
    <div data-testid="histogram">{groupColumn ?? "no grouping"}</div>
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

  it("switches the distribution to selection-vs-rest histograms, with or without violin data", () => {
    useUnifiedViewStore.setState({ bottomPanelOpen: true, bottomView: "distribution" });
    renderPanel({ violinData: {}, violinGroupColumn: undefined });
    // Per donor needs a grouping; the histogram does not (a selection is enough).
    expect(screen.queryByRole("button", { name: "Per donor" })).toBeNull();
    expect(screen.getByText("No data")).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "Histogram" }));
    expect(screen.getByTestId("histogram")).toHaveTextContent("no grouping");
    expect(screen.queryByText("No data")).toBeNull();
    fireEvent.click(screen.getByRole("button", { name: "Cells" }));
    expect(screen.queryByTestId("histogram")).toBeNull();
  });

  it("switches the distribution to one value per donor for the gene on the scatter", () => {
    useUnifiedViewStore.setState({
      bottomPanelOpen: true,
      bottomView: "distribution",
      scatterOverlay: { type: "expression", gene: "ISG15" },
    });
    renderPanel({ violinGroupColumn: "cluster" });
    expect(screen.getByTestId("violin")).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "Per donor" }));
    expect(screen.queryByTestId("violin")).toBeNull();
    expect(screen.getByTestId("per-donor")).toHaveTextContent("ISG15 by cluster");
    // A score is per cell: no gene, and a reason.
    act(() => useUnifiedViewStore.setState({ scatterOverlay: { type: "score", name: "IFN" } }));
    expect(screen.getByTestId("per-donor")).toHaveTextContent("a score is per cell");
    act(() => useUnifiedViewStore.setState({ scatterOverlay: { type: "program", index: 3, name: "P3" } }));
    expect(screen.getByTestId("per-donor")).toHaveTextContent("a program's usage is per cell");
    fireEvent.click(screen.getByRole("button", { name: "Cells" }));
    expect(screen.getByTestId("violin")).toBeInTheDocument();
  });
});

describe("UnifiedBottomPanel violin caption", () => {
  it("states what the violins show and the n of cells and groups", () => {
    renderPanel({ violinData: { B: [1, 2, 3], NK: [4, 5] }, violinGroupLabel: "Cell type" });
    fireEvent.click(screen.getByRole("button", { name: /Distribution/ }));
    expect(screen.getByTestId("violin-caption")).toHaveTextContent(
      "CD79A expression per Cell type: one violin per group. n = 5 cells; 2 groups.",
    );
  });
});
