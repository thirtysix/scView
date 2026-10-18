import { describe, it, expect, vi, beforeEach } from "vitest";
import { act, fireEvent, render, screen } from "@testing-library/react";

vi.mock("@/components/plots/Plot", () => ({
  default: ({ data, layout }: { data: unknown; layout: unknown }) => (
    <div data-testid="plot" data-traces={JSON.stringify(data)} data-layout={JSON.stringify(layout)} />
  ),
}));
const hooks = vi.hoisted(() => ({
  codes: null as Int32Array | null,
  obs: {} as Record<string, Float32Array>,
}));
vi.mock("@/hooks/useObsCodes", () => ({
  useObsCodes: (_d: string, _e: string, column: string | null) => ({ codes: column ? hooks.codes : null, error: false }),
}));
vi.mock("@/hooks/useObsValues", () => ({
  useObsValues: (_d: string, _e: string, column: string | null) => ({
    values: column ? hooks.obs[column] ?? null : null,
    isLoading: false,
    error: false,
  }),
}));

import { SelectionHistogram } from "./SelectionHistogram";
import { useDatasetStore } from "@/stores/datasetStore";
import { useSelectionStore } from "@/stores/selectionStore";
import { useUnifiedViewStore } from "@/stores/unifiedViewStore";
import type { DatasetInfo } from "@/api/types";

const GENE = Float32Array.from([0, 0, 1, 2, 3, 4, 5, 6]);
const DATASET = {
  id: "kang",
  name: "Kang",
  obs_columns: [
    { name: "cluster", dtype: "category", n_unique: 2, values: ["T", "B"] },
    { name: "total_counts", dtype: "float32", n_unique: 8 },
    { name: "stim", dtype: "category", n_unique: 2, values: ["CTRL", "STIM"] },
    { name: "constant", dtype: "int64", n_unique: 1 },
  ],
} as unknown as DatasetInfo;

const traces = () => JSON.parse(screen.getByTestId("plot").dataset.traces!);
const layout = () => JSON.parse(screen.getByTestId("plot").dataset.layout!);

beforeEach(() => {
  hooks.codes = Int32Array.from([0, 0, 0, 0, 1, 1, 1, 1]);
  hooks.obs = { total_counts: Float32Array.from([100, 200, 300, 400, 500, 600, 700, 800]) };
  useDatasetStore.setState({ currentDataset: DATASET });
  useSelectionStore.setState({ selectedCellIndices: null });
  useUnifiedViewStore.setState({
    scatterOverlay: { type: "expression", gene: "ISG15" },
    overlayValues: GENE,
    overlayLabel: "ISG15",
    overlayDatasetId: "kang",
  });
});

describe("SelectionHistogram", () => {
  it("compares the selection with the rest for the gene on the plot", () => {
    useSelectionStore.setState({ selectedCellIndices: new Set([4, 5, 6, 7]) });
    render(<SelectionHistogram datasetId="kang" groupColumn="cluster" />);
    expect(screen.getByRole("combobox", { name: "Compare" })).toHaveValue("selection");
    const [a, b] = traces();
    expect(a.name).toBe("selected (n = 4)");
    expect(b.name).toBe("other cells (n = 4)");
    expect(a.type).toBe("bar");
    const sum = (ys: number[]) => ys.reduce((s, v) => s + v, 0);
    expect(sum(a.y)).toBeCloseTo(1);
    expect(sum(b.y)).toBeCloseTo(1);
    expect(screen.getByText(/n = 4 · median 4.5 · mean 4.5 · 100% expressing/)).toBeInTheDocument();
    expect(screen.getByText(/n = 4 · median 0.5 · mean 0.75 · 50% expressing/)).toBeInTheDocument();
    expect(screen.getByText("Difference in medians (selected − other cells): +4")).toBeInTheDocument();
    expect(screen.getByText(/Mann-Whitney AUC 1.00/)).toBeInTheDocument();
    expect(screen.getByText(/not tests/)).toBeInTheDocument();
    // dashed median lines at each side's median
    expect(layout().shapes.map((s: { x0: number }) => s.x0)).toEqual([4.5, 0.5]);
  });

  it("without a selection, compares a group with the rest, then with another group", () => {
    render(<SelectionHistogram datasetId="kang" groupColumn="cluster" />);
    const compare = screen.getByRole("combobox", { name: "Compare" });
    expect(compare).toHaveValue("group_rest");
    expect(screen.getByRole("option", { name: /select cells first/ })).toBeDisabled();
    expect(traces()[0].name).toBe("T (n = 4)");
    expect(traces()[1].name).toBe("rest (n = 4)");
    fireEvent.change(compare, { target: { value: "group_group" } });
    fireEvent.change(screen.getByRole("combobox", { name: "Group" }), { target: { value: "1" } });
    // picking the other side's group swaps the other side
    expect(screen.getByRole("combobox", { name: "Other group" })).toHaveValue("0");
    expect(traces()[0].name).toBe("B (n = 4)");
    expect(traces()[1].name).toBe("T (n = 4)");
  });

  it("switches to a selection made while it is open", () => {
    render(<SelectionHistogram datasetId="kang" groupColumn="cluster" />);
    expect(screen.getByRole("combobox", { name: "Compare" })).toHaveValue("group_rest");
    act(() => useSelectionStore.setState({ selectedCellIndices: new Set([0, 1]) }));
    expect(screen.getByRole("combobox", { name: "Compare" })).toHaveValue("selection");
    expect(traces()[0].name).toBe("selected (n = 2)");
  });

  it("offers numeric metadata columns, and uses one when nothing is on the plot", () => {
    useUnifiedViewStore.setState({ scatterOverlay: null, overlayValues: null, overlayDatasetId: null });
    render(<SelectionHistogram datasetId="kang" groupColumn="cluster" />);
    const values = screen.getByRole("combobox", { name: "Values" });
    expect(values).toHaveValue("total_counts");
    const options = Array.from((values as HTMLSelectElement).options).map((o) => o.value);
    expect(options).toEqual(["total_counts"]); // no categorical or constant columns
    expect(screen.getByText(/n = 4 · median 250 · mean 250$/)).toBeInTheDocument(); // no "% expressing"
    expect(layout().xaxis.title.text).toMatch(/total/i);
  });

  it("draws densities and a log value axis on request", () => {
    useSelectionStore.setState({ selectedCellIndices: new Set([4, 5, 6, 7]) });
    render(<SelectionHistogram datasetId="kang" groupColumn="cluster" />);
    fireEvent.click(screen.getByRole("button", { name: "Density" }));
    expect(traces()[0].type).toBe("scatter");
    expect(layout().yaxis.title.text).toBe("Density");
    fireEvent.click(screen.getByRole("button", { name: "Log" }));
    expect(layout().xaxis.title.text).toBe("ISG15 (log scale)");
    expect(layout().xaxis.ticktext).toContain("1");
  });

  it("explains an empty side instead of drawing", () => {
    useSelectionStore.setState({ selectedCellIndices: new Set([0, 1, 2, 3, 4, 5, 6, 7]) });
    render(<SelectionHistogram datasetId="kang" groupColumn="cluster" />);
    expect(screen.queryByTestId("plot")).toBeNull();
    expect(screen.getByText(/No cells with a value on one side/)).toBeInTheDocument();
  });
});
