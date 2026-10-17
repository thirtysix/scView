import { describe, it, expect, vi, beforeEach } from "vitest";
import { act, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import type { GroupDistances } from "@/api/clustering";
import { useSelectionStore } from "@/stores/selectionStore";

const { getGroupDistances, plotProps } = vi.hoisted(() => ({
  getGroupDistances: vi.fn(),
  plotProps: { current: null as Record<string, unknown> | null },
}));
vi.mock("@/api/clustering", () => ({ getGroupDistances }));
// Plotly needs a real canvas; what the heatmap is handed is what matters here.
vi.mock("@/components/plots/Plot", () => ({
  default: (props: Record<string, unknown>) => {
    plotProps.current = props;
    return <div data-testid="plot" />;
  },
}));

import { ClusterDistanceHeatmap } from "./ClusterDistanceHeatmap";

const KANG: GroupDistances = {
  column: "cluster",
  groups: ["B", "CD14 Mono", "NK", "CD16 Mono"],
  sizes: [100, 400, 60, 90],
  distances: [
    [0, 13.9, 15.7, 12.7],
    [13.9, 0, 17.6, 8.6],
    [15.7, 17.6, 0, 16.5],
    [12.7, 8.6, 16.5, 0],
  ],
  order: [2, 0, 1, 3],
  representation: "X_pca_harmony",
  n_pcs: 50,
  rep_source: "graph",
  linkage: "complete",
  metric: "euclidean",
  n_left_out: 3,
};

const COLUMNS = [
  { name: "cluster", n_unique: 4 },
  { name: "stim", n_unique: 2 },
];

type Trace = { x: string[]; y: string[]; z: (number | null)[][]; text: string[][]; texttemplate?: string };
const trace = () => (plotProps.current!.data as Trace[])[0]!;

function renderHeatmap(defaultColumn = "cluster") {
  return render(
    <QueryClientProvider client={new QueryClient()}>
      <ClusterDistanceHeatmap datasetId="kang" defaultColumn={defaultColumn} columns={COLUMNS} height={300} />
    </QueryClientProvider>,
  );
}

beforeEach(() => {
  getGroupDistances.mockReset().mockResolvedValue(KANG);
  plotProps.current = null;
  useSelectionStore.setState({ highlightedGroup: null });
});

describe("ClusterDistanceHeatmap", () => {
  it("draws the distances in the clustering's order and says what space they are in", async () => {
    renderHeatmap();
    await screen.findByTestId("plot");
    expect(getGroupDistances).toHaveBeenCalledWith("kang", "cluster");
    expect(trace().y).toEqual(["NK", "B", "CD14 Mono", "CD16 Mono"]);
    expect(trace().x).toEqual(trace().y);
    expect(trace().z[2]).toEqual([17.6, 13.9, null, 8.6]);
    expect(trace().text[2]![3]).toContain("<b>CD14 Mono</b> to <b>CD16 Mono</b><br>distance 8.60");
    expect(trace().texttemplate).toBe("%{z:.1f}"); // few groups: values written in
    expect(screen.getByText(/50 Harmony-corrected PCs, the space the neighbour graph was built in/)).toBeInTheDocument();
    expect(screen.getByText(/3 cells without a group are left out/)).toBeInTheDocument();
  });

  it("names the nearest groups of the highlighted one, and a click highlights a row's group", async () => {
    renderHeatmap();
    await screen.findByTestId("plot");
    expect(screen.queryByTestId("nearest-groups")).toBeNull();
    act(() => (plotProps.current!.onClick as (e: unknown) => void)({ points: [{ x: "NK", y: "CD14 Mono" }] }));
    expect(useSelectionStore.getState().highlightedGroup).toEqual({ column: "cluster", value: "CD14 Mono" });
    expect(screen.getByTestId("nearest-groups")).toHaveTextContent(
      "Nearest to CD14 Mono: CD16 Mono (8.6), B (13.9), NK (17.6)",
    );
    // A highlight on another column says nothing here.
    act(() => useSelectionStore.setState({ highlightedGroup: { column: "stim", value: "CD14 Mono" } }));
    expect(screen.queryByTestId("nearest-groups")).toBeNull();
  });

  it("asks again for another column, and shows the server's reason when it can't", async () => {
    renderHeatmap();
    await screen.findByTestId("plot");
    getGroupDistances.mockRejectedValueOnce(
      new Error('API error 400: {"detail":"No principal components (X_pca) in this dataset."}'),
    );
    fireEvent.change(screen.getByLabelText("Distances between"), { target: { value: "stim" } });
    await waitFor(() => expect(getGroupDistances).toHaveBeenLastCalledWith("kang", "stim"));
    expect(await screen.findByRole("alert")).toHaveTextContent("No principal components (X_pca) in this dataset.");
  });
});
