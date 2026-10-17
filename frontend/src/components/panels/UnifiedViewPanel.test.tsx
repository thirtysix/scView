import { describe, it, expect, vi, beforeEach } from "vitest";
import { act, render, screen } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import type { DatasetInfo } from "@/api/types";
import { useDatasetStore } from "@/stores/datasetStore";
import { useSettingsStore } from "@/stores/settingsStore";
import { useUnifiedViewStore } from "@/stores/unifiedViewStore";
import { useSplitViewStore } from "@/stores/splitViewStore";
import { useSelectionStore } from "@/stores/selectionStore";

// The panel's scatter, data hook and side panels are stubbed: these tests look
// at what the panel itself draws over the plot (tooltip) and hands the scatter.
const { embedding, scatterProps, compareProps, bottomProps, apiFetch, stub } = vi.hoisted(() => ({
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
vi.mock("@/components/plots/LassoSelector", () => ({ LassoSelector: () => <div data-testid="lasso" /> }));
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

const hover = (index: number) =>
  act(() => (scatterProps.current!.onHover as (i: unknown) => void)({ index, x: 10, y: 10 }));

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

describe("Unified View hover tooltip", () => {
  it("leads with the colour-by column and the cell's category, then the rest of its metadata", async () => {
    renderPanel();
    hover(2);
    const line = screen.getByTestId("hover-colour-line");
    expect(line).toHaveTextContent("cell_type:NK");
    expect(line.querySelector("span[aria-hidden]")).not.toBeNull();
    // The metadata arrives later; the colour-by column is not listed twice.
    await screen.findByText("sample:");
    const tooltip = line.closest("div.absolute")! as HTMLElement;
    const listed = Array.from(tooltip.querySelectorAll("dt")).map((dt) => dt.textContent);
    expect(listed).toEqual(["n_genes:", "sample:"]);
  });

  it("shows the numeric value when coloured by a numeric column", () => {
    useSettingsStore.setState({ colorBy: "n_genes" });
    embedding.current = {
      ...embedding.current,
      colors: Float32Array.from([500, 812, 1200]),
      colorColumnName: "n_genes",
    };
    renderPanel();
    hover(1);
    expect(screen.getByTestId("hover-colour-line")).toHaveTextContent("n_genes:812");
  });

  it("shows the gene and its expression while a gene overlay is on", () => {
    useUnifiedViewStore.setState({
      scatterOverlay: { type: "expression", gene: "CD3E" },
      overlayValues: Float32Array.from([0, 0.25, 2.5]),
      overlayLabel: "CD3E",
      overlayDatasetId: "kang",
    });
    renderPanel();
    hover(2);
    expect(screen.getByTestId("hover-colour-line")).toHaveTextContent("CD3E expression:2.500");
  });

  it("shows the gene-set score while a score overlay is on", () => {
    useUnifiedViewStore.setState({
      scatterOverlay: { type: "score", name: "IFN" },
      overlayValues: Float32Array.from([0.1, 0.2, 0.3]),
      overlayLabel: "IFN",
      overlayDatasetId: "kang",
    });
    renderPanel();
    hover(0);
    expect(screen.getByTestId("hover-colour-line")).toHaveTextContent("IFN score:0.100");
  });
});

describe("Unified View compare colourings", () => {
  it("covers the plot with the compare grid, lasso and the single plot's legend off", () => {
    useSplitViewStore.setState({ compareColumns: ["cell_type", "sample"] });
    renderPanel();
    expect(screen.getByTestId("compare-grid")).toBeInTheDocument();
    expect(compareProps.current!.columns).toEqual(["cell_type", "sample"]);
    expect(screen.queryByTestId("lasso")).toBeNull();
    // The grid draws a legend per column; the single plot's legend steps aside.
    expect(screen.queryByText("B")).toBeNull();
  });

  it("leads the tooltip with the cell's value in every compared column", async () => {
    useSplitViewStore.setState({ compareColumns: ["cell_type", "sample"] });
    renderPanel();
    act(() =>
      (compareProps.current!.onHover as (i: unknown) => void)({
        index: 1,
        x: 5,
        y: 5,
        lines: [
          { label: "cell_type", value: "NK", color: [1, 2, 3] },
          { label: "sample", value: "ctrl", color: [4, 5, 6] },
        ],
      }),
    );
    const rows = screen.getAllByTestId("hover-colour-line").map((r) => r.textContent);
    expect(rows).toEqual(["cell_type:NK", "sample:ctrl"]);
    await screen.findByText("n_genes:");
    const tooltip = screen.getAllByTestId("hover-colour-line")[0]!.closest("div.absolute")! as HTMLElement;
    expect(Array.from(tooltip.querySelectorAll("dt")).map((dt) => dt.textContent)).toEqual(["n_genes:"]);
  });

  it("shows the lasso and the legend on the single plot", () => {
    renderPanel();
    expect(screen.getByTestId("lasso")).toBeInTheDocument();
    expect(screen.getByText("B")).toBeInTheDocument();
  });
});

describe("Unified View group labels", () => {
  beforeEach(() => {
    useUnifiedViewStore.setState({ centroidLabels: null });
    useSelectionStore.setState({ highlightedGroup: null });
  });

  it("asks the scatter for the colour-by column's labels, with the highlighted group", () => {
    useSelectionStore.setState({ highlightedGroup: { column: "cell_type", value: "NK" } });
    renderPanel();
    expect(scatterProps.current).toMatchObject({ showLabels: true, categories: ["B", "NK"], highlightCategory: "NK" });
  });

  it("leaves them off when toggled off, for a numeric colouring and under a gene overlay", () => {
    useUnifiedViewStore.setState({ centroidLabels: { key: "cell_type", show: false } });
    const { unmount } = renderPanel();
    expect(scatterProps.current!.showLabels).toBe(false);
    unmount();

    useUnifiedViewStore.setState({ centroidLabels: null });
    useSettingsStore.setState({ colorBy: "n_genes" });
    embedding.current = { ...embedding.current, colors: Float32Array.from([1, 2, 3]), colorColumnName: "n_genes" };
    const second = renderPanel();
    expect(scatterProps.current!.showLabels).toBe(false);
    second.unmount();

    useSettingsStore.setState({ colorBy: "cell_type" });
    embedding.current = { ...embedding.current, colors: Int32Array.from([0, 1, 1]), colorColumnName: "cell_type" };
    useUnifiedViewStore.setState({
      scatterOverlay: { type: "expression", gene: "CD3E" },
      overlayValues: Float32Array.from([0, 1, 2]),
      overlayLabel: "CD3E",
      overlayDatasetId: "kang",
    });
    renderPanel();
    expect(scatterProps.current!.showLabels).toBe(false);
  });
});

describe("Unified View cluster distances", () => {
  it("gives the scatter its embedding (for the UMAP caveat) and the distances the plot's grouping", () => {
    useUnifiedViewStore.setState({ groupByColumn: "cell_type" });
    renderPanel();
    expect(scatterProps.current!.embedding).toBe("X_umap");
    expect(bottomProps.current).toMatchObject({ datasetId: "kang", distanceColumn: "cell_type" });
    expect((bottomProps.current!.distanceColumns as { name: string }[]).map((c) => c.name)).toEqual(["cell_type"]);
  });
});

describe("Unified View and a restored session", () => {
  beforeEach(() => {
    useUnifiedViewStore.setState({ camera: null, pendingCamera: null, pendingOverlay: null });
  });

  it("shows the session's camera, then keeps the camera shown for the next save", () => {
    const camera = { target: [1, 2, 0], zoom: 3 };
    useUnifiedViewStore.setState({ pendingCamera: { datasetId: "kang", embedding: "X_umap", camera } });
    renderPanel();
    expect(scatterProps.current!.restoreViewState).toEqual(camera);
    act(() => (scatterProps.current!.onRestoreViewState as () => void)());
    expect(useUnifiedViewStore.getState().pendingCamera).toBeNull();

    const onCamera = scatterProps.current!.onCameraChange as (vs: Record<string, unknown>, fitted: boolean) => void;
    act(() => onCamera({ target: [4, 5, 0], zoom: 1.5, minZoom: -3, width: 400 }, false));
    expect(useUnifiedViewStore.getState().camera).toEqual({
      datasetId: "kang",
      embedding: "X_umap",
      camera: { target: [4, 5, 0], zoom: 1.5 },
    });
    act(() => onCamera({ target: [0, 0, 0], zoom: 1 }, true)); // fitted again: nothing to save
    expect(useUnifiedViewStore.getState().camera).toBeNull();
  });

  it("leaves a camera for another embedding or dataset alone", () => {
    useUnifiedViewStore.setState({
      pendingCamera: { datasetId: "kang", embedding: "X_pca", camera: { target: [1, 2], zoom: 3 } },
    });
    renderPanel();
    expect(scatterProps.current!.restoreViewState).toBeNull();
  });

  it("scores the session's gene set again and overlays it", async () => {
    apiFetch.mockImplementation(async (path: string) =>
      path.endsWith("/genesets/score") ? { scores: [0.5, 1, 1.5] } : {},
    );
    useUnifiedViewStore.setState({
      pendingOverlay: { datasetId: "kang", overlay: { type: "score", name: "IFN", genes: ["ISG15", "MX1"] } },
    });
    renderPanel();
    await vi.waitFor(() => expect(useUnifiedViewStore.getState().scatterOverlay).not.toBeNull());
    const [path, init] = apiFetch.mock.calls.find(([p]) => String(p).endsWith("/genesets/score"))!;
    expect(path).toBe("/datasets/kang/genesets/score");
    expect(JSON.parse(init.body)).toEqual({ gene_set: ["ISG15", "MX1"], score_name: "IFN" });
    const u = useUnifiedViewStore.getState();
    expect(u.scatterOverlay).toEqual({ type: "score", name: "IFN", genes: ["ISG15", "MX1"] });
    expect(Array.from(u.overlayValues!)).toEqual([0.5, 1, 1.5]);
    expect(u.pendingOverlay).toBeNull();
  });
});
