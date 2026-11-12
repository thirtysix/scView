import { describe, it, expect, vi, beforeEach } from "vitest";
import { act, render } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import type { DatasetInfo } from "@/api/types";
import { useDatasetStore } from "@/stores/datasetStore";
import { useSettingsStore } from "@/stores/settingsStore";
import { useUnifiedViewStore } from "@/stores/unifiedViewStore";
import { useSplitViewStore } from "@/stores/splitViewStore";
import { tableFromArrays, tableToIPC } from "apache-arrow";

// The panel's scatter, data hook and side panels are stubbed, as in the reliability
// tests: these check that changing "Group by (violin)" splits the violin on screen again.
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
  id: "kang",
  name: "Kang",
  filename: "kang.h5ad",
  n_cells: 3,
  n_genes: 10,
  status: "ready",
  available_embeddings: ["X_umap"],
  obs_columns: [
    { name: "cell_type", dtype: "category", n_unique: 2, values: ["B", "NK"] },
    { name: "sample", dtype: "category", n_unique: 3, values: ["s1", "s2", "s3"] },
  ],
  created_at: "",
};

const renderPanel = () =>
  render(
    <QueryClientProvider client={new QueryClient()}>
      <UnifiedViewPanel />
    </QueryClientProvider>,
  );

const BY_TYPE = { B: [0.1], NK: [0.5, 0.9] };
const BY_SAMPLE = { s1: [0.1], s2: [0.5], s3: [0.9] };

beforeEach(() => {
  useDatasetStore.setState({ currentDataset: DATASET, currentDatasetId: "kang" });
  useSettingsStore.setState({ colorBy: "cell_type", embedding: "X_umap" });
  useUnifiedViewStore.getState().clearOverlayState();
  useSplitViewStore.setState({ splitBy: null, compareColumns: null });
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
  apiFetch.mockResolvedValue({});
  useUnifiedViewStore.setState({
    groupByColumn: "cell_type",
    overlayValues: Float32Array.from([0.1, 0.5, 0.9]),
    overlayDatasetId: "kang",
    violinData: BY_TYPE,
  });
});

const violinCalls = () => apiFetch.mock.calls.map(([p]) => p as string).filter((p) => p.includes("/expression/violin"));

describe('Unified View "Group by (violin)"', () => {
  it("splits a gene's violin by the newly chosen column", async () => {
    useUnifiedViewStore.setState({
      scatterOverlay: { type: "expression", gene: "FOXO3" }, overlayLabel: "FOXO3", violinTitle: "FOXO3 expression",
    });
    apiFetch.mockImplementation(async (path: string) =>
      path.includes("groupby=sample") ? { gene: "FOXO3", groups: BY_SAMPLE } : {},
    );
    renderPanel();
    expect(violinCalls()).toEqual([]);
    act(() => useUnifiedViewStore.getState().setGroupByColumn("sample"));
    await vi.waitFor(() => expect(useUnifiedViewStore.getState().violinData).toEqual(BY_SAMPLE));
    expect(violinCalls()).toHaveLength(1);
    expect(violinCalls()[0]).toContain("gene=FOXO3&groupby=sample");
    expect(useUnifiedViewStore.getState().violinTitle).toBe("FOXO3 expression");
  });

  it("keeps the latest grouping when an earlier reply arrives last", async () => {
    useUnifiedViewStore.setState({ scatterOverlay: { type: "expression", gene: "FOXO3" }, overlayLabel: "FOXO3" });
    let releaseSample: (v: unknown) => void = () => {};
    apiFetch.mockImplementation((path: string) =>
      path.includes("groupby=sample")
        ? new Promise((resolve) => { releaseSample = resolve; })
        : Promise.resolve({ gene: "FOXO3", groups: BY_TYPE }),
    );
    renderPanel();
    act(() => useUnifiedViewStore.getState().setGroupByColumn("sample"));
    act(() => useUnifiedViewStore.getState().setGroupByColumn("cell_type"));
    await vi.waitFor(() => expect(violinCalls()).toHaveLength(2));
    await act(async () => releaseSample({ gene: "FOXO3", groups: BY_SAMPLE }));
    expect(useUnifiedViewStore.getState().violinData).toEqual(BY_TYPE);
  });

  it("asks a gene program's usage again by the new column, leaving the overlay as it is", async () => {
    const overlay = { type: "program" as const, index: 3, name: "P3" };
    useUnifiedViewStore.setState({ scatterOverlay: overlay, overlayLabel: "P3 usage share" });
    apiFetch.mockImplementation(async (path: string) =>
      path.includes("/programs/3/usage?groupby=sample")
        ? { index: 3, name: "P3", raw: false, run_id: "r", top_genes: [], values: [0.1, 0.5, 0.9], violin: BY_SAMPLE }
        : {},
    );
    renderPanel();
    act(() => useUnifiedViewStore.getState().setGroupByColumn("sample"));
    await vi.waitFor(() => expect(useUnifiedViewStore.getState().violinData).toEqual(BY_SAMPLE));
    expect(useUnifiedViewStore.getState().scatterOverlay).toEqual(overlay);
  });

  it("splits a gene's violin by the new column after the plot was coloured by a column", async () => {
    // Choosing a Color cleared the FOXO3 overlay; the violin is still FOXO3's.
    useUnifiedViewStore.setState({ scatterOverlay: null, violinGene: { gene: "FOXO3" }, violinTitle: "FOXO3 expression" });
    apiFetch.mockImplementation(async (path: string) =>
      path.includes("groupby=sample") ? { gene: "FOXO3", groups: BY_SAMPLE } : {},
    );
    renderPanel();
    act(() => useUnifiedViewStore.getState().setGroupByColumn("sample"));
    await vi.waitFor(() => expect(useUnifiedViewStore.getState().violinData).toEqual(BY_SAMPLE));
    expect(violinCalls()[0]).toContain("gene=FOXO3&groupby=sample");
    expect(useUnifiedViewStore.getState().violinGene).toEqual({ gene: "FOXO3" });
  });

  it("splits a saved score picked in Colour by, and again by a new column", async () => {
    const score = { name: "score_IFN", dtype: "float32", n_unique: 3,
      saved_score: { kind: "geneset" as const, name: "IFN", method: "aucell" } };
    useDatasetStore.setState({ currentDataset: { ...DATASET, obs_columns: [...DATASET.obs_columns, score] } });
    useSettingsStore.setState({ colorBy: "score_IFN" });
    useUnifiedViewStore.setState({ scatterOverlay: null, violinData: {}, violinTitle: "" });
    embedding.current = { ...embedding.current, colors: Float32Array.from([0.25, 0.5, 0.75]), colorColumnName: "score_IFN" };
    const codes: Record<string, Int32Array> = { cell_type: Int32Array.from([0, 1, 1]), sample: Int32Array.from([0, 1, 2]) };
    apiFetchBinary.mockImplementation(async (path: string) => {
      const column = new URLSearchParams(path.split("?")[1]).get("color_by")!;
      return tableToIPC(tableFromArrays({ x: Float32Array.from([0, 1, 2]), [column]: codes[column]! })).slice().buffer;
    });
    renderPanel();
    await vi.waitFor(() => expect(useUnifiedViewStore.getState().violinData).toEqual({ B: [0.25], NK: [0.5, 0.75] }));
    expect(useUnifiedViewStore.getState().violinTitle).toBe("IFN (Gene-set score (AUCell))");
    act(() => useUnifiedViewStore.getState().setGroupByColumn("sample"));
    await vi.waitFor(() => expect(useUnifiedViewStore.getState().violinData).toEqual({ s1: [0.25], s2: [0.5], s3: [0.75] }));
  });

  it("clears a violin left over from an overlay that is gone", async () => {
    useUnifiedViewStore.setState({ scatterOverlay: null });
    renderPanel();
    act(() => useUnifiedViewStore.getState().setGroupByColumn("sample"));
    await vi.waitFor(() => expect(useUnifiedViewStore.getState().violinData).toEqual({}));
    expect(violinCalls()).toEqual([]);
  });
});
