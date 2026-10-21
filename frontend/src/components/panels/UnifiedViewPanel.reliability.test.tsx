import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import type { DatasetInfo } from "@/api/types";
import { useDatasetStore } from "@/stores/datasetStore";
import { useSettingsStore } from "@/stores/settingsStore";
import { useUnifiedViewStore } from "@/stores/unifiedViewStore";
import { useSplitViewStore } from "@/stores/splitViewStore";

// The panel's scatter, data hook and side panels are stubbed: these tests look
// at what the panel itself draws over the plot (tooltip) and hands the scatter.
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

describe("Unified View plot reliability overlay", () => {
  const ref = { embedding: "X_umap", k: 15, chance: 15 / 13835, random: 0.00107 };

  it("draws it on a fixed 0-1 scale whatever the clipping, says so and gives chance", () => {
    useSettingsStore.setState({ colorClip: { preset: "p5" } });
    useUnifiedViewStore.setState({
      scatterOverlay: { type: "score", name: "Neighbour preservation (UMAP)", reliability: ref },
      overlayValues: Float32Array.from([0.2, 0.4, 0.6]),
      overlayLabel: "Neighbour preservation (UMAP)",
      overlayDatasetId: "kang",
    });
    renderPanel();
    expect(scatterProps.current!.colorScale).toEqual({ colormap: "viridis", min: 0, max: 1 });
    expect(screen.getByTestId("colour-scale-caption")).toHaveTextContent("Colour range 0 to 1, fixed (not clipped to the data).");
    expect(screen.getByTestId("colour-scale-note")).toHaveTextContent(
      "Fixed 0 to 1 scale, not clipped: 1 = all 15 graph neighbours are beside the cell on the plot. Chance 0.0011; random placement about 0.0011.",
    );
    expect(screen.queryByLabelText("Colour range clipping")).toBeNull();
    expect(scatterProps.current!.exportLegend).toMatchObject({ type: "continuous", min: 0, max: 1 });
  });

  it("an ordinary score keeps its data range and clipping", () => {
    useSettingsStore.setState({ colorClip: { preset: "p5" } });
    useUnifiedViewStore.setState({
      scatterOverlay: { type: "score", name: "Cluster stability" },
      overlayValues: Float32Array.from([0.2, 0.4, 0.6]),
      overlayLabel: "Cluster stability",
      overlayDatasetId: "kang",
    });
    renderPanel();
    const scale = scatterProps.current!.colorScale as { min: number; max: number };
    expect(scale.min).toBeCloseTo(0.22, 5); // the 5th percentile of 0.2, 0.4, 0.6
    expect(scale.max).toBeCloseTo(0.58, 5);
    expect(screen.queryByTestId("colour-scale-note")).toBeNull();
    expect(screen.getByLabelText("Colour range clipping")).toBeTruthy();
    useSettingsStore.setState({ colorClip: { preset: "none" } });
  });

  it("computes a session's reliability overlay again on open", async () => {
    apiFetch.mockImplementation(async (path: string) =>
      path.includes("/reliability")
        ? {
            embedding: "X_umap", dims: [0, 1], n_dims: 2, k: 15, chance: 0.0011,
            shuffled: { median: 0, mean: 0.001, seed: 1 }, scores: [0.5, null, 1],
          }
        : {},
    );
    useUnifiedViewStore.setState({
      pendingOverlay: { datasetId: "kang", overlay: { type: "reliability", embedding: "X_umap" } },
    });
    renderPanel();
    await vi.waitFor(() => expect(useUnifiedViewStore.getState().scatterOverlay).not.toBeNull());
    expect(apiFetch.mock.calls.map(([p]) => p)).toContain("/datasets/kang/embeddings/X_umap/reliability");
    const u = useUnifiedViewStore.getState();
    expect(u.scatterOverlay).toEqual({
      type: "score",
      name: "Neighbour preservation (UMAP)",
      reliability: { embedding: "X_umap", k: 15, chance: 0.0011, random: 0.001 },
    });
    expect(Number.isNaN(u.overlayValues![1]!)).toBe(true);
    expect(u.pendingOverlay).toBeNull();
  });
});
