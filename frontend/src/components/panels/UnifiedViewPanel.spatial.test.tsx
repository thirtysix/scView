import { describe, it, expect, vi, beforeEach } from "vitest";
import { fireEvent, render, screen } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import type { DatasetInfo } from "@/api/types";
import { useDatasetStore } from "@/stores/datasetStore";
import { useSettingsStore } from "@/stores/settingsStore";
import { useUnifiedViewStore } from "@/stores/unifiedViewStore";
import { useSplitViewStore } from "@/stores/splitViewStore";

// The panel's scatter, data hook and side panels are stubbed, as in the programs
// tests: these look at the Spatial genes tab (shown only with spatial data).
const { embedding, scatterProps, compareProps, bottomProps, lassoProps, apiFetch, apiFetchBinary, stub } = vi.hoisted(() => ({
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
  apiFetchBinary: vi.fn(),
}));
vi.mock("@/hooks/useEmbedding", () => ({ useEmbedding: () => embedding.current }));
vi.mock("@/components/plots/EmbeddingScatter", () => ({
  EmbeddingScatter: (props: Record<string, unknown>) => {
    scatterProps.current = props;
    return <div data-testid="scatter" />;
  },
}));
vi.mock("@/api/client", () => ({ apiFetch, apiFetchBinary }));
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
  id: "ln",
  name: "Lymph node",
  filename: "ln.h5",
  n_cells: 3,
  n_genes: 10,
  status: "ready",
  available_embeddings: ["X_umap"],
  obs_columns: [{ name: "cell_type", dtype: "category", n_unique: 2, values: ["B", "NK"] }],
  created_at: "",
};
const SPATIAL = {
  technology: "visium" as const, unit: "spot" as const, library_column: "library_id" as const,
  libraries: [{ id: "LN", n_obs: 3, images: {}, spot_diameter_fullres: 89, bounds: [0, 0, 2, 2] as [number, number, number, number] }],
};

const renderPanel = () =>
  render(
    <QueryClientProvider client={new QueryClient()}>
      <UnifiedViewPanel />
    </QueryClientProvider>,
  );

beforeEach(() => {
  useSettingsStore.setState({ colorBy: "cell_type", embedding: "X_umap" });
  useUnifiedViewStore.getState().clearOverlayState();
  useUnifiedViewStore.setState({ activeSubtab: "markers" });
  useSplitViewStore.setState({ splitBy: null, compareColumns: null });
  compareProps.current = null;
  scatterProps.current = null;
  bottomProps.current = null;
  lassoProps.current = null;
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
  apiFetchBinary.mockReset();
  apiFetch.mockImplementation(async (path: string) =>
    path.split("?")[0]!.endsWith("/spatial/autocorr")
      ? {
          result_id: "r1", method: "moran", n_obs: 3, n_genes: 1, neighbors: { kind: "grid", k: 6 }, libraries: ["LN"],
          genes: "all", matrix_source: "log-normalised X", unit: "spot", n_constant: 0, notes: [], rows_shown: 1,
          n_significant: 1, n_dispersed: 0, elapsed_seconds: 0.1, created_at: 0,
          rows: [{ gene: "CCL21", I: 0.7, expected_I: -0.5, var_I: 0.1, z: 3, pval: 1e-3, pval_adj: 1e-3 }],
        }
      : {},
  );
});

describe("Unified View Spatial genes tab", () => {
  it("is not offered without spatial data, and the panel leaves it", () => {
    useDatasetStore.setState({ currentDataset: { ...DATASET, spatial: null }, currentDatasetId: "ln" });
    useUnifiedViewStore.setState({ activeSubtab: "spatial" });
    renderPanel();
    expect(screen.queryByRole("tab", { name: "Spatial genes" })).toBeNull();
    expect(useUnifiedViewStore.getState().activeSubtab).toBe("markers");
  });

  it("is offered with spatial data; a gene click colours the plot by that gene", async () => {
    useDatasetStore.setState({ currentDataset: { ...DATASET, spatial: SPATIAL }, currentDatasetId: "ln" });
    renderPanel();
    fireEvent.click(screen.getByRole("tab", { name: "Spatial genes" }));
    fireEvent.click(await screen.findByRole("button", { name: "CCL21" }));
    await vi.waitFor(() =>
      expect(apiFetchBinary).toHaveBeenCalledWith(expect.stringContaining("/datasets/ln/expression?genes=CCL21")),
    );
  });
});
