import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import type { DatasetInfo } from "@/api/types";
import { useDatasetStore } from "@/stores/datasetStore";
import { useSettingsStore } from "@/stores/settingsStore";
import { useUnifiedViewStore } from "@/stores/unifiedViewStore";
import { useSplitViewStore } from "@/stores/splitViewStore";

// The panel on the tissue plot: what it hands the scatter and the cluster map.
const { embedding, scatterProps, refProps, splitProps, stub } = vi.hoisted(() => ({
  stub: (name: string) => () => ({ [name]: () => null }),
  embedding: {
    current: {
      positions: Float32Array.from([10, -10, 20, -20, 30, -30]) as Float32Array | null,
      colors: Int32Array.from([0, 1, 1]) as Int32Array | null,
      numCells: 3,
      colorColumnName: "leiden" as string | null,
      dimensions: 2 as 2 | 3,
      isLoading: false,
      error: null as string | null,
      embedding: "spatial" as string | null,
    },
  },
  scatterProps: { current: null as Record<string, unknown> | null },
  refProps: { current: null as Record<string, unknown> | null },
  splitProps: { current: null as Record<string, unknown> | null },
}));
vi.mock("@/hooks/useEmbedding", () => ({ useEmbedding: () => embedding.current }));
vi.mock("@/components/plots/EmbeddingScatter", () => ({
  EmbeddingScatter: (props: Record<string, unknown>) => {
    scatterProps.current = props;
    return <div data-testid="scatter" />;
  },
}));
vi.mock("@/components/unified/ClusterReference", () => ({
  ClusterReference: (props: Record<string, unknown>) => {
    refProps.current = props;
    return null;
  },
}));
vi.mock("@/components/unified/SplitScatterGrid", () => ({
  SplitScatterGrid: (props: Record<string, unknown>) => {
    splitProps.current = props;
    return null;
  },
}));
vi.mock("@/api/client", () => ({ apiFetch: vi.fn(() => Promise.resolve({})), apiFetchBinary: vi.fn() }));
vi.mock("@/components/unified/UnifiedToolbar", stub("UnifiedToolbar"));
vi.mock("@/components/unified/UnifiedBottomPanel", stub("UnifiedBottomPanel"));
vi.mock("@/components/unified/UnifiedMarkersSubtab", stub("UnifiedMarkersSubtab"));
vi.mock("@/components/unified/UnifiedExpressionSubtab", stub("UnifiedExpressionSubtab"));
vi.mock("@/components/unified/UnifiedGeneSetsSubtab", stub("UnifiedGeneSetsSubtab"));
vi.mock("@/components/unified/UnifiedEnrichmentSubtab", stub("UnifiedEnrichmentSubtab"));
vi.mock("@/components/unified/UnifiedDESubtab", stub("UnifiedDESubtab"));

import { UnifiedViewPanel } from "./UnifiedViewPanel";

const DATASET: DatasetInfo = {
  id: "vis",
  name: "Visium",
  filename: "v.h5",
  n_cells: 3,
  n_genes: 10,
  status: "ready",
  available_embeddings: ["X_umap"],
  obs_columns: [{ name: "leiden", dtype: "category", n_unique: 2, values: ["0", "1"] }],
  created_at: "",
  spatial: {
    technology: "visium",
    unit: "spot",
    library_column: "library_id",
    libraries: [
      {
        id: "A1",
        n_obs: 3,
        images: { lowres: { width: 600, height: 600, scalef: 0.02 } },
        spot_diameter_fullres: 90,
        bounds: [10, 10, 30, 30],
      },
    ],
  },
};

const renderPanel = () =>
  render(
    <QueryClientProvider client={new QueryClient()}>
      <UnifiedViewPanel />
    </QueryClientProvider>,
  );

beforeEach(() => {
  useDatasetStore.setState({ currentDataset: DATASET, currentDatasetId: "vis" });
  useSettingsStore.setState({ colorBy: "leiden", embedding: "spatial" });
  useUnifiedViewStore.getState().clearOverlayState();
  useUnifiedViewStore.setState({ groupByColumn: "leiden" });
  useSplitViewStore.setState({ splitBy: null, compareColumns: null, geneGrid: null });
  scatterProps.current = refProps.current = splitProps.current = null;
  embedding.current.embedding = "spatial";
});

describe("Unified View on the tissue", () => {
  it("keeps Tissue chosen and hands the scatter the section", () => {
    renderPanel();
    expect(useSettingsStore.getState().embedding).toBe("spatial");
    const tissue = scatterProps.current!.tissue as { library: { id: string }; unit: string; showImage: boolean };
    expect(tissue.library.id).toBe("A1");
    expect(tissue.unit).toBe("spot");
    expect(tissue.showImage).toBe(true);
    expect(screen.getByText("Spots")).toBeInTheDocument();
  });

  it("keeps the UMAP in the cluster map, on its own coordinates and camera", () => {
    renderPanel();
    expect(refProps.current!.embedding).toBe("X_umap");
    expect(refProps.current!.positions).toBeNull();
    expect(refProps.current!.viewState).toBeNull();
    expect(refProps.current!.onViewStateChange).toBeUndefined();
  });

  it("on the UMAP the map shares the main plot's coordinates, without the tissue", () => {
    useSettingsStore.setState({ embedding: "X_umap" });
    renderPanel();
    expect(refProps.current!.embedding).toBe("X_umap");
    expect(refProps.current!.positions).toBe(embedding.current.positions);
    expect(scatterProps.current!.tissue).toBeNull();
  });

  it("leaves a split off on the tissue, saying why", () => {
    useSplitViewStore.setState({ splitBy: "leiden" });
    renderPanel();
    expect(splitProps.current).toBeNull();
    expect(screen.getByRole("status")).toHaveTextContent(/not the tissue/);
  });

  it("draws no tissue until the tissue coordinates are in (not the UMAP's, kept while they load)", () => {
    embedding.current.embedding = "X_umap";
    embedding.current.isLoading = true;
    renderPanel();
    expect(scatterProps.current).toBeNull();
    embedding.current.isLoading = false;
    renderPanel();
    expect(scatterProps.current).toBeNull();
  });

  it("switches to the UMAP on a dataset without spatial data", () => {
    useDatasetStore.setState({ currentDataset: { ...DATASET, spatial: null } });
    renderPanel();
    expect(useSettingsStore.getState().embedding).toBe("X_umap");
  });
});
