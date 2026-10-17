import { describe, it, expect, beforeEach, vi } from "vitest";
import { render, screen, fireEvent, act, waitFor } from "@testing-library/react";
import type { DEResponse } from "@/api/de";
import type { DatasetInfo } from "@/api/types";
import { DE_DEFAULTS, useDESettingsStore } from "@/stores/deSettingsStore";
import { useDatasetStore } from "@/stores/datasetStore";
import { useSelectionStore } from "@/stores/selectionStore";

const { computeDE, computeDECompare } = vi.hoisted(() => ({ computeDE: vi.fn(), computeDECompare: vi.fn() }));
vi.mock("@/api/de", () => ({ computeDE, computeDECompare }));
// The follow-up cards fetch on their own; not what is under test here.
vi.mock("@/components/unified/DEEnrichment", () => ({ DEEnrichment: () => null }));
vi.mock("@/components/unified/DEActivity", () => ({ DEActivity: () => null }));
type PlotProps = {
  data: { x: number[]; y: number[]; marker?: { color?: string[] } }[];
  layout: { shapes?: { x0: number; line: { dash: string } }[]; xaxis?: { ticktext?: string[] } };
};
/** The latest render of the detection histogram (the volcano is a Plot too). */
const histogram = () => plots.props.filter((p) => p.layout.xaxis?.ticktext?.includes("<0.1%")).at(-1);
const plots = vi.hoisted(() => ({ props: [] as PlotProps[] }));
vi.mock("@/components/plots/Plot", () => ({
  default: (props: PlotProps) => {
    plots.props.push(props);
    return <div data-testid="plot" />;
  },
}));

import { UnifiedDESubtab } from "./UnifiedDESubtab";

const ds = (id = "ds"): DatasetInfo => ({
  id,
  name: id,
  filename: `${id}.h5ad`,
  n_cells: 100,
  n_genes: 10,
  status: "ready",
  available_embeddings: ["X_umap"],
  obs_columns: [
    { name: "stim", dtype: "category", n_unique: 2, values: ["CTRL", "STIM"] },
    { name: "cell_type", dtype: "category", n_unique: 3, values: ["B", "NK", "T"] },
    { name: "ind", dtype: "category", n_unique: 3, values: ["d1", "d2", "d3"] },
  ],
  created_at: "",
});

const noop = () => {};
const renderDE = () => render(<UnifiedDESubtab onGeneClick={noop} onScoreGeneSet={noop} />);
const selects = () => screen.getAllByRole("combobox") as HTMLSelectElement[];

beforeEach(() => {
  useDESettingsStore.setState({ ...DE_DEFAULTS, scopedHighlight: null });
  useSelectionStore.setState({ selectedCellIndices: null, highlightedGroup: null });
  useDatasetStore.setState({ currentDataset: null, currentDatasetId: null });
  useDatasetStore.getState().setCurrentDataset(ds());
});

describe("UnifiedDESubtab settings", () => {
  it("keep their values across a tab switch (a remount)", () => {
    const first = renderDE();
    fireEvent.click(screen.getByRole("button", { name: "Between groups" }));
    // Compare stim, CTRL vs STIM, within cell_type = T
    fireEvent.change(selects()[1]!, { target: { value: "CTRL" } });
    fireEvent.change(selects()[2]!, { target: { value: "STIM" } });
    fireEvent.change(selects()[3]!, { target: { value: "cell_type" } });
    fireEvent.change(selects()[4]!, { target: { value: "T" } });
    first.unmount();

    renderDE();
    expect(screen.getByRole("button", { name: "Between groups" })).toHaveClass("bg-primary");
    expect(selects().map((s) => s.value).slice(0, 5)).toEqual(["stim", "CTRL", "STIM", "cell_type", "T"]);
  });

  it("take a highlighted group as the scope once, not over a scope picked since", () => {
    useDESettingsStore.getState().update({ mode: "groups" });
    const first = renderDE();
    act(() => useSelectionStore.getState().setHighlight({ column: "cell_type", value: "NK" }));
    expect(selects()[4]!.value).toBe("NK");
    fireEvent.change(selects()[4]!, { target: { value: "B" } });
    first.unmount();
    renderDE();
    expect(selects()[4]!.value).toBe("B");
  });

  it("start from the defaults on another dataset", () => {
    useDESettingsStore.getState().update({ mode: "groups", splitBy: "ind", fcThreshold: 3 });
    act(() => useDatasetStore.getState().setCurrentDataset(ds("other")));
    renderDE();
    expect(screen.getByRole("button", { name: "Selection vs rest" })).toHaveClass("bg-primary");
    expect(useDESettingsStore.getState().fcThreshold).toBe(1);
  });
});

/** A 0.01 % grid detection histogram: 5,930 genes under 10 %, 8,123 at or over. */
function detection(): number[] {
  const h = new Array<number>(10_001).fill(0);
  h[0] = 2000;
  h[300] = 2930; // 3 %
  h[700] = 1000; // 7 %
  h[2000] = 8123; // 20 %
  return h;
}

const RESULT: DEResponse = {
  n_selected: 40,
  n_rest: 60,
  label: "selection",
  genes: [{ gene: "CD14", logfoldchange: 3, pval: 1e-10, pval_adj: 1e-8, stat: 7 }],
  min_pct: 0.1,
  n_genes_tested: 8123,
  n_genes_skipped: 5930,
  detection: detection(),
};

describe("UnifiedDESubtab gene prefilter", () => {
  beforeEach(() => {
    plots.props = [];
    computeDE.mockReset().mockResolvedValue(RESULT);
    computeDECompare.mockReset().mockResolvedValue(RESULT);
  });

  it("sends the detection cut-off picked, 1% by default", async () => {
    act(() => useSelectionStore.getState().setSelection(new Set([1, 2, 3, 4])));
    renderDE();
    const pick = screen.getByRole("combobox", { name: "Genes tested" }) as HTMLSelectElement;
    expect(pick.value).toBe("0.01");
    fireEvent.change(pick, { target: { value: "0.1" } });
    expect(useDESettingsStore.getState().minPct).toBe(0.1);
    fireEvent.click(screen.getByRole("button", { name: /compute de/i }));
    await waitFor(() => expect(computeDE).toHaveBeenCalled());
    expect(computeDE.mock.calls[0]![3]).toBe(0.1);
  });

  it("says how many genes were tested and skipped, and marks the cut-off on the histogram", async () => {
    useDESettingsStore.getState().update({ minPct: 0.1 }); // what RESULT was computed with
    act(() => useSelectionStore.getState().setSelection(new Set([1, 2, 3, 4])));
    renderDE();
    fireEvent.click(screen.getByRole("button", { name: /compute de/i }));
    await screen.findByText(/8,123 genes tested; 5,930 detected in under 10%/);
    const h = histogram()!;
    expect(h.layout.shapes!.map((s) => [s.x0, s.line.dash])).toEqual([[-1, "dash"]]);
    // The under-0.1 %, 3 % and 7 % bars are grey (left out), the 20 % bar blue.
    const bars = h.data[0]!;
    // The 0.1-decade bar holding a fraction (bar 0 is the one under 0.1 %).
    const colour = (frac: number) =>
      bars.marker!.color![bars.x.findIndex((v, i) => i > 0 && Math.abs(v - Math.log10(frac)) <= 0.05)];
    expect([bars.marker!.color![0], colour(0.03), colour(0.07), colour(0.2)]).toEqual([
      "#cbd5e1", "#cbd5e1", "#cbd5e1", "#60a5fa",
    ]);

    // Another cut-off is previewed from the histogram before computing again.
    const pick = () => screen.getByRole("combobox", { name: "Genes tested" });
    fireEvent.change(pick(), { target: { value: "0.05" } });
    expect(screen.getByText(/At 5% \(dotted line\), 9,123 genes would be tested/)).toBeTruthy();
    expect(histogram()!.layout.shapes!.map((s) => s.line.dash)).toEqual(["dash", "dot"]);
    fireEvent.change(pick(), { target: { value: "0.01" } });
    expect(screen.getByText(/At 1% \(dotted line\), 12,053 genes would be tested/)).toBeTruthy();
    fireEvent.change(pick(), { target: { value: "0" } });
    expect(screen.getByText(/With no cut-off, 14,053 genes would be tested/)).toBeTruthy();
  });

  it("applies to the cell-level test between groups, not to pseudobulk", async () => {
    useDESettingsStore.getState().update({ mode: "groups", minPct: 0.25 });
    renderDE();
    fireEvent.click(screen.getByRole("button", { name: /compute de/i }));
    await waitFor(() => expect(computeDECompare).toHaveBeenCalled());
    expect(computeDECompare.mock.calls[0]![1]).toMatchObject({ min_pct: 0.25 });
    expect(computeDECompare.mock.calls[0]![1]).not.toHaveProperty("method");

    fireEvent.click(screen.getByRole("button", { name: "Pseudobulk (donor-aware)" }));
    expect(screen.queryByRole("combobox", { name: "Genes tested" })).toBeNull();
  });
});
