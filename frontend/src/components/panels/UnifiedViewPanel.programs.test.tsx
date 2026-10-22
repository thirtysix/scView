import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import type { DatasetInfo } from "@/api/types";
import { useDatasetStore } from "@/stores/datasetStore";
import { useSettingsStore } from "@/stores/settingsStore";
import { useUnifiedViewStore } from "@/stores/unifiedViewStore";
import { useSplitViewStore } from "@/stores/splitViewStore";

// The panel's scatter, data hook and side panels are stubbed, as in the reliability
// tests: these look at the gene program overlay the panel restores and draws.
const { embedding, scatterProps, compareProps, bottomProps, lassoProps, apiFetch, stub } = vi.hoisted(() => ({
  stub: (name: string) => () => ({ [name]: () => null }),
  embedding: {
    current: {
      positions: null as Float32Array | null,
      colors: null as Float32Array | Int32Array | null,
      numCells: 0,
      colorColumnName: null as string | null,
      dimensions: 2 as 2 | 3,
      isLoading: false,
      error: null as string | null,
    },
  },
  scatterProps: { current: null as Record<string, unknown> | null },
  compareProps: { current: null as Record<string, unknown> | null },
  bottomProps: { current: null as Record<string, unknown> | null },
  lassoProps: { current: null as Record<string, unknown> | null },
  apiFetch: vi.fn(),
}));
vi.mock("@/hooks/useEmbedding", () => ({ useEmbedding: () => embedding.current }));
vi.mock("@/components/plots/EmbeddingScatter", () => ({
  EmbeddingScatter: (props: Record<string, unknown>) => {
    scatterProps.current = props;
    return <div data-testid="scatter" />;
  },
}));
vi.mock("@/api/client", () => ({ apiFetch, apiFetchBinary: vi.fn() }));
vi.mock("@/components/unified/UnifiedToolbar", stub("UnifiedToolbar"));
vi.mock("@/components/unified/UnifiedBottomPanel", () => ({
  UnifiedBottomPanel: (props: Record<string, unknown>) => {
    bottomProps.current = props;
    return null;
  },
}));
vi.mock("@/components/unified/UnifiedMarkersSubtab", stub("UnifiedMarkersSubtab"));
vi.mock("@/components/unified/UnifiedExpressionSubtab", stub("UnifiedExpressionSubtab"));
vi.mock("@/components/unified/UnifiedGeneSetsSubtab", stub("UnifiedGeneSetsSubtab"));
vi.mock("@/components/unified/UnifiedEnrichmentSubtab", stub("UnifiedEnrichmentSubtab"));
vi.mock("@/components/unified/UnifiedDESubtab", stub("UnifiedDESubtab"));
vi.mock("@/components/unified/ClusterReference", stub("ClusterReference"));
vi.mock("@/components/unified/SplitScatterGrid", stub("SplitScatterGrid"));
vi.mock("@/components/plots/LassoSelector", () => ({
  LassoSelector: (props: Record<string, unknown>) => {
    lassoProps.current = props;
    return <div data-testid="lasso" />;
  },
}));
vi.mock("@/components/unified/CompareColoringsGrid", () => ({
  CompareColoringsGrid: (props: Record<string, unknown>) => {
    compareProps.current = props;
    return <div data-testid="compare-grid" />;
  },
}));

import { UnifiedViewPanel } from "./UnifiedViewPanel";

const DATASET: DatasetInfo = {
  id: "kang",
  name: "Kang",
  filename: "kang.h5ad",
  n_cells: 3,
  n_genes: 10,
  status: "ready",
  available_embeddings: ["X_umap"],
  obs_columns: [
    { name: "cell_type", dtype: "category", n_unique: 2, values: ["B", "NK"] },
    { name: "n_genes", dtype: "int64", n_unique: 3 },
  ],
  created_at: "",
};

const renderPanel = () =>
  render(
    <QueryClientProvider client={new QueryClient()}>
      <UnifiedViewPanel />
    </QueryClientProvider>,
  );

beforeEach(() => {
  useDatasetStore.setState({ currentDataset: DATASET, currentDatasetId: "kang" });
  useSettingsStore.setState({ colorBy: "cell_type", embedding: "X_umap" });
  useUnifiedViewStore.getState().clearOverlayState();
  useSplitViewStore.setState({ splitBy: null, compareColumns: null });
  compareProps.current = null;
  embedding.current = {
    positions: Float32Array.from([0, 0, 1, 1, 2, 2]),
    colors: Int32Array.from([0, 1, 1]),
    numCells: 3,
    colorColumnName: "cell_type",
    dimensions: 2,
    isLoading: false,
    error: null,
  };
  apiFetch.mockReset();
  apiFetch.mockResolvedValue({ cell_type: "NK", n_genes: 812, sample: "ctrl" });
});

describe("Unified View gene program overlay", () => {
  it("reads a session's program overlay again on open, with its violin, on the data's own range", async () => {
    useUnifiedViewStore.setState({ groupByColumn: "cell_type" });
    apiFetch.mockImplementation(async (path: string) =>
      path.includes("/programs/3/usage")
        ? { index: 3, name: "P3", raw: false, run_id: "r", top_genes: ["ISG15", "IFI6", "MX1", "RSAD2"],
            values: [0.1, 0.5, 0.9], violin: { B: [0.1], NK: [0.5, 0.9] } }
        : {},
    );
    useUnifiedViewStore.setState({ pendingOverlay: { datasetId: "kang", overlay: { type: "program", index: 3 } } });
    renderPanel();
    await vi.waitFor(() => expect(useUnifiedViewStore.getState().scatterOverlay).not.toBeNull());
    expect(apiFetch.mock.calls.map(([p]) => p)).toContain("/datasets/kang/programs/3/usage?groupby=cell_type");
    const u = useUnifiedViewStore.getState();
    expect(u.scatterOverlay).toEqual({ type: "program", index: 3, name: "P3" });
    expect(u.overlayLabel).toBe("P3 usage share (ISG15, IFI6, MX1)");
    expect(Array.from(u.overlayValues!)).toEqual([Float32Array.from([0.1])[0], 0.5, Float32Array.from([0.9])[0]]);
    expect(u.violinData).toEqual({ B: [0.1], NK: [0.5, 0.9] });
    expect(u.violinTitle).toBe("P3 usage share");
    expect(u.pendingOverlay).toBeNull();
    // The chip names the overlay a program, not a score.
    expect(await screen.findByText("program")).toBeTruthy();
    // A usage share is drawn as a continuous overlay, sequential (never negative).
    await vi.waitFor(() => expect(scatterProps.current!.colorType).toBe("continuous"));
    const scale = scatterProps.current!.colorScale as { colormap: string; min: number; max: number };
    expect(scale.colormap).toBe("viridis");
    expect(scale.min).toBeCloseTo(0.1, 5);
    expect(scale.max).toBeCloseTo(0.9, 5);
  });
});
