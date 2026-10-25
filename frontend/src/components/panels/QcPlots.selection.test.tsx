import { describe, it, expect, vi, beforeEach } from "vitest";
import { act, render, screen, waitFor } from "@testing-library/react";
import { tableFromArrays, tableToIPC } from "apache-arrow";
import type { DatasetInfo } from "@/api/types";

const { apiFetch, getQcByGroup, getFilterPreview, fetchEmbeddingBinary, plots } = vi.hoisted(() => ({
  apiFetch: vi.fn(),
  getQcByGroup: vi.fn(),
  getFilterPreview: vi.fn(),
  fetchEmbeddingBinary: vi.fn(),
  plots: [] as { data: { type: string; name?: string; x?: number[]; y?: number[] }[]; onSelected?: (e: unknown) => void }[],
}));
vi.mock("@/api/client", () => ({ apiFetch }));
vi.mock("@/api/qc", () => ({ getQcByGroup, getFilterPreview }));
vi.mock("@/api/embeddings", () => ({ fetchEmbeddingBinary }));
vi.mock("@/components/plots/Plot", () => ({
  default: (p: (typeof plots)[number]) => {
    plots.push(p);
    return <div data-testid="plot" />;
  },
}));

import { QcPlots } from "./QcPlots";
import { useDatasetStore } from "@/stores/datasetStore";
import { useSelectionStore } from "@/stores/selectionStore";
import { useSettingsStore } from "@/stores/settingsStore";

// Ten cells: total_counts 0..9, genes 10 x that.
const TOTAL = Float32Array.from({ length: 10 }, (_, i) => i);
const GENES = Float32Array.from({ length: 10 }, (_, i) => 10 * i);
const metric = { min: 0, max: 10, mean: 5, median: 5, q1: 3, q3: 7, hist: { bin_edges: [0, 5, 10], counts: [5, 5] } };

beforeEach(() => {
  plots.length = 0;
  apiFetch.mockReset().mockResolvedValue({
    n_cells: 10,
    computed_on_demand: false,
    has_mt: true,
    metrics: { total_counts: metric },
    scatter: { x: [0, 9], y: [0, 90], color: [], x_label: "total_counts", y_label: "n_genes_by_counts", color_label: "", n_shown: 2 },
  });
  getQcByGroup.mockReset().mockResolvedValue({
    column: null, reason: null, n_cells: 10, n_groups: 0, max_groups: 50, groups: [], metrics: {}, scatter: null,
  });
  fetchEmbeddingBinary.mockReset().mockImplementation(async (_d: string, _e: string, column: string) =>
    tableToIPC(
      tableFromArrays({ x: new Float32Array(10), y: new Float32Array(10), [column]: column === "total_counts" ? TOTAL : GENES }),
    ).slice().buffer,
  );
  useDatasetStore.setState({ currentDatasetId: "d1", currentDataset: { id: "d1", obs_columns: [] } as unknown as DatasetInfo });
  useSettingsStore.setState({ embedding: "X_umap" });
  useSelectionStore.setState({ selectedCellIndices: null, highlightedGroup: null });
});

const histogram = () => plots.filter((p) => p.data[0]?.type === "bar").at(-1)!;
const scatter = () => plots.filter((p) => p.data[0]?.type === "scattergl").at(-1)!;

describe("QcPlots linked selection (C035)", () => {
  it("fetches nothing per cell without a selection", async () => {
    render(<QcPlots datasetId="d1" />);
    await screen.findByText("QC Distributions");
    expect(fetchEmbeddingBinary).not.toHaveBeenCalled();
    expect(histogram().data.filter((t) => t.type === "bar")).toHaveLength(1);
  });

  it("overlays the selected cells in the histogram's bins and on the scatter", async () => {
    useSelectionStore.setState({ selectedCellIndices: new Set([1, 2, 7]) });
    render(<QcPlots datasetId="d1" />);
    await waitFor(() => expect(histogram().data.filter((t) => t.type === "bar")).toHaveLength(2));
    const sel = histogram().data[1]!;
    expect(sel.name).toBe("Selection (3 cells)");
    expect(sel.y).toEqual([2, 1]); // [0, 5): 1, 2; [5, 10]: 7
    await waitFor(() => expect(scatter().data).toHaveLength(2));
    expect(scatter().data[1]!.x).toEqual([1, 2, 7]);
    expect(scatter().data[1]!.y).toEqual([10, 20, 70]);
    expect(screen.getAllByTestId("selection-chip").length).toBeGreaterThan(0);
  });

  it("a box drawn on the counts-vs-genes scatter selects every cell inside it", async () => {
    render(<QcPlots datasetId="d1" />);
    await screen.findByText("QC Distributions");
    await waitFor(() => expect(scatter().onSelected).toBeDefined());
    act(() => scatter().onSelected!({ range: { x: [2.5, 6.5], y: [0, 100] } }));
    await waitFor(() => expect(useSelectionStore.getState().selectedCellIndices).not.toBeNull());
    expect([...useSelectionStore.getState().selectedCellIndices!].sort()).toEqual([3, 4, 5, 6]);
  });
});
