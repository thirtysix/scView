import { describe, it, expect, vi, beforeEach } from "vitest";
import { act, render, screen, waitFor } from "@testing-library/react";
import { useSelectionStore } from "@/stores/selectionStore";

const apiFetch = vi.fn();
vi.mock("@/api/client", () => ({ apiFetch: (...a: unknown[]) => apiFetch(...a) }));
const plotProps = vi.fn();
vi.mock("@/components/plots/Plot", () => ({
  default: (p: unknown) => {
    plotProps(p);
    return <div data-testid="plot" />;
  },
}));

import { MarkerPlot } from "./MarkerPlot";

interface Trace { type: string; x: string[]; y: string[]; z?: (number | null)[][] }
const last = () => plotProps.mock.calls.at(-1)![0] as { data: Trace[]; layout: { yaxis: { categoryarray: string[] } } };

const groups = {
  groupby: "leiden", layer: "raw", groups: ["A", "B"], genes: ["G1", "G2"],
  mean: [[0.1, 2], [1.5, 0]], mean_expressing: [[2, 2.5], [3, 0]], fraction: [[0.03, 0.8], [0.5, 0]],
  scale_range: [0, 5],
};
const selection = { layer: "raw", genes: ["G1", "G2"], n_cells: 3, mean: [1, 1.2], mean_expressing: [2, 2], fraction: [0.5, 0.6] };

function respond(_path: string, init?: RequestInit) {
  if (init?.method === "POST") return Promise.resolve({ groupby: "leiden", layer: "raw", groups: [], genes: selection.genes, selection });
  return Promise.resolve(groups);
}

const renderPlot = () =>
  render(<MarkerPlot datasetId="ds1" groupby="leiden" groups={["A", "B"]} markers={{}} kind="heatmap" perGroup={1} genes={["G1", "G2"]} />);

describe("MarkerPlot selection row (C035)", () => {
  beforeEach(() => {
    apiFetch.mockReset().mockImplementation(respond);
    plotProps.mockReset();
    useSelectionStore.setState({ selectedCellIndices: null, highlightedGroup: null });
  });

  it("has no extra row and asks nothing more without a selection", async () => {
    renderPlot();
    await screen.findByTestId("plot");
    expect(last().layout.yaxis.categoryarray).toEqual(["A", "B"]);
    expect(apiFetch).toHaveBeenCalledTimes(1);
    expect(screen.queryByTestId("selection-chip")).toBeNull();
  });

  it("adds the selection as the last row, from a POST with the selected cells", async () => {
    useSelectionStore.setState({ selectedCellIndices: new Set([5, 1, 3]) });
    renderPlot();
    await waitFor(() => expect(last().layout.yaxis.categoryarray).toHaveLength(3));
    const [url, init] = apiFetch.mock.calls.find(([, i]) => (i as RequestInit | undefined)?.method === "POST")!;
    expect(url).toBe("/datasets/ds1/expression/by-group");
    expect(JSON.parse((init as RequestInit).body as string)).toEqual({
      genes: ["G1", "G2"], cells: [1, 3, 5], include_groups: false,
    });
    expect(last().layout.yaxis.categoryarray[2]).toBe("Selection (3 cells)");
    // Scaled per gene over the three rows: G1 0.1 / 1.5 / 1.0 -> 0, 1, 0.64
    const z = last().data[0]!.z!;
    expect(z).toHaveLength(3);
    expect(z[2]![0]).toBeCloseTo(0.9 / 1.4, 5);
    expect(screen.getByTestId("selection-chip")).toHaveTextContent("Selection: 3 cells");
    expect(screen.getByText(/Last row: the selected cells/)).toBeInTheDocument();
  });

  it("drops the row when the selection is cleared", async () => {
    useSelectionStore.setState({ selectedCellIndices: new Set([1]) });
    renderPlot();
    await waitFor(() => expect(last().layout.yaxis.categoryarray).toHaveLength(3));
    act(() => useSelectionStore.getState().clearSelection());
    expect(last().layout.yaxis.categoryarray).toEqual(["A", "B"]);
  });
});
