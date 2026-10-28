import { describe, it, expect, vi, beforeEach } from "vitest";
import { act, fireEvent, render, screen } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import type { DatasetInfo } from "@/api/types";
import type { GroupSelectSource } from "@/components/unified/SelectByGroup";
import { useDatasetStore } from "@/stores/datasetStore";
import { useSettingsStore } from "@/stores/settingsStore";
import { useUnifiedViewStore } from "@/stores/unifiedViewStore";
import { useSplitViewStore } from "@/stores/splitViewStore";
import { useSelectionStore } from "@/stores/selectionStore";

// The keyboard paths to a selection ("By group" in the toolbar, the scatter's
// rectangle) end in the same selection store as a lasso or a legend click.
const { scatterProps, toolbarProps, stub } = vi.hoisted(() => ({
  stub: (name: string) => () => ({ [name]: () => null }),
  scatterProps: { current: null as Record<string, unknown> | null },
  toolbarProps: { current: null as { groupSelect?: GroupSelectSource | null } | null },
}));
vi.mock("@/hooks/useEmbedding", async () => ({
  ...(await vi.importActual<object>("@/hooks/useEmbedding")),
  useEmbedding: () => ({
    positions: Float32Array.from([0, 0, 1, 1, 2, 2]),
    colors: Int32Array.from([0, 1, 1]),
    numCells: 3,
    colorColumnName: "cell_type",
    dimensions: 2,
    isLoading: false,
    error: null,
  }),
}));
vi.mock("@/components/plots/EmbeddingScatter", () => ({
  EmbeddingScatter: (props: Record<string, unknown>) => {
    scatterProps.current = props;
    return <div data-testid="scatter" />;
  },
}));
vi.mock("@/components/unified/UnifiedToolbar", () => ({
  UnifiedToolbar: (props: { groupSelect?: GroupSelectSource | null }) => {
    toolbarProps.current = props;
    return null;
  },
}));
vi.mock("@/api/client", () => ({ apiFetch: vi.fn(() => new Promise(() => {})), apiFetchBinary: vi.fn() }));
vi.mock("@/components/unified/UnifiedBottomPanel", stub("UnifiedBottomPanel"));
vi.mock("@/components/unified/UnifiedMarkersSubtab", stub("UnifiedMarkersSubtab"));
vi.mock("@/components/unified/UnifiedExpressionSubtab", stub("UnifiedExpressionSubtab"));
vi.mock("@/components/unified/UnifiedDESubtab", stub("UnifiedDESubtab"));
vi.mock("@/components/unified/ClusterReference", stub("ClusterReference"));
vi.mock("@/components/plots/LassoSelector", stub("LassoSelector"));

import { UnifiedViewPanel } from "./UnifiedViewPanel";

const DATASET: DatasetInfo = {
  id: "kang",
  name: "Kang",
  filename: "kang.h5ad",
  n_cells: 3,
  n_genes: 10,
  status: "ready",
  available_embeddings: ["X_umap"],
  obs_columns: [{ name: "cell_type", dtype: "category", n_unique: 2, values: ["B", "NK"] }],
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
  useUnifiedViewStore.setState({ activeSubtab: "markers" });
  useSplitViewStore.setState({ splitBy: null, compareColumns: null });
  useSelectionStore.getState().clearSelection();
  useSelectionStore.getState().setHighlight(null);
});

describe("Unified View keyboard selection", () => {
  it("hands the toolbar the colour-by groups with their cell counts", () => {
    renderPanel();
    const g = toolbarProps.current!.groupSelect!;
    expect(g.column).toBe("cell_type");
    expect(g.categories).toEqual(["B", "NK"]);
    expect(g.counts).toEqual([1, 2]);
  });

  it("By group selects the groups' cells; one group is also highlighted, as a legend click", () => {
    renderPanel();
    act(() => toolbarProps.current!.groupSelect!.onSelect(["NK"]));
    expect([...useSelectionStore.getState().selectedCellIndices!]).toEqual([1, 2]);
    expect(useSelectionStore.getState().highlightedGroup).toEqual({ column: "cell_type", value: "NK" });
    expect(screen.getByRole("button", { name: "NK", pressed: true })).toBeInTheDocument();

    act(() => toolbarProps.current!.groupSelect!.onSelect(["B", "NK"]));
    expect(useSelectionStore.getState().selectedCellIndices!.size).toBe(3);
    expect(useSelectionStore.getState().highlightedGroup).toBeNull();
  });

  it("the scatter's keyboard rectangle ends in the selection, like a lasso", () => {
    renderPanel();
    const select = scatterProps.current!.onKeyboardSelect as (s: Set<number>) => void;
    act(() => select(new Set([0, 2])));
    expect([...useSelectionStore.getState().selectedCellIndices!]).toEqual([0, 2]);
    act(() => select(new Set()));
    expect(useSelectionStore.getState().selectedCellIndices).toBeNull();
  });

  it("the analysis tabs switch by arrow key", () => {
    renderPanel();
    const markers = screen.getByRole("tab", { name: "Markers" });
    fireEvent.keyDown(markers, { key: "ArrowRight" });
    expect(useUnifiedViewStore.getState().activeSubtab).toBe("expression");
    expect(screen.getByRole("tab", { name: "Expression" })).toHaveFocus();
  });
});

describe("Unified View split pane", () => {
  it("a drag moves the guide only; the columns resize once, on release", () => {
    const { container } = renderPanel();
    const split = screen.getByTestId("split-guide").parentElement!;
    vi.spyOn(split, "getBoundingClientRect").mockReturnValue({ left: 0, width: 1000 } as DOMRect);
    const left = () => split.children[1] as HTMLElement;
    const guide = screen.getByTestId("split-guide");
    expect(left().style.width).toBe("60%");

    fireEvent.mouseDown(container.querySelector(".cursor-col-resize")!);
    fireEvent.mouseMove(window, { clientX: 450 });
    fireEvent.mouseMove(window, { clientX: 400 });
    expect(guide.style.display).toBe("block");
    expect(guide.style.left).toBe("40%");
    expect(left().style.width).toBe("60%");

    fireEvent.mouseUp(window);
    expect(guide.style.display).toBe("none");
    expect(left().style.width).toBe("40%");
  });
});
