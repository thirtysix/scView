import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { act, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { tableFromArrays, tableToIPC } from "apache-arrow";
import type { ReactNode } from "react";
import type { DatasetInfo } from "@/api/types";
import { useDatasetStore } from "@/stores/datasetStore";
import { useSplitViewStore } from "@/stores/splitViewStore";

// jsdom has no WebGL: capture what the grid hands deck.gl instead.
const { deckProps, fetchEmbeddingBinary } = vi.hoisted(() => ({
  deckProps: { current: null as Record<string, unknown> | null },
  fetchEmbeddingBinary: vi.fn(),
}));
vi.mock("@deck.gl/react", async () => {
  const { forwardRef } = await import("react");
  return {
    DeckGL: forwardRef<HTMLDivElement, Record<string, unknown>>((props, ref) => {
      deckProps.current = props;
      return <div ref={ref} data-testid="deck" />;
    }),
  };
});
vi.mock("@/api/embeddings", () => ({ fetchEmbeddingBinary }));

import { SplitScatterGrid } from "./SplitScatterGrid";
import { SplitByControl } from "./SplitByControl";

const dataset = (obs: DatasetInfo["obs_columns"]): DatasetInfo => ({
  id: "ds1",
  name: "kang",
  filename: "kang.h5ad",
  n_cells: 6,
  n_genes: 10,
  status: "ready",
  available_embeddings: ["X_umap"],
  obs_columns: obs,
  created_at: "",
});

const STIM = { name: "stim", dtype: "category", n_unique: 2, values: ["CTRL", "STIM"] };
const CELL_TYPE = {
  name: "cell_type",
  dtype: "category",
  n_unique: 8,
  values: ["B", "T", "NK", "Mono", "DC", "pDC", "Mk", "Eryth"],
};

// Lets the grid measure its plot area (jsdom lays nothing out).
class FakeResizeObserver {
  constructor(private cb: ResizeObserverCallback) {}
  observe() {
    this.cb([{ contentRect: { width: 800, height: 400 } } as ResizeObserverEntry], this as never);
  }
  disconnect() {}
  unobserve() {}
}

beforeEach(() => {
  vi.stubGlobal("ResizeObserver", FakeResizeObserver);
  deckProps.current = null;
  useSplitViewStore.setState({ splitBy: null, showContext: true });
});
afterEach(() => vi.unstubAllGlobals());

const withQuery = (ui: ReactNode) => (
  <QueryClientProvider client={new QueryClient({ defaultOptions: { queries: { retry: false } } })}>
    {ui}
  </QueryClientProvider>
);

describe("SplitByControl", () => {
  it("offers columns with 2–6 values, lists bigger ones disabled, and turns the split on and off", () => {
    useDatasetStore.setState({ currentDataset: dataset([STIM, CELL_TYPE]), currentDatasetId: "ds1" });
    render(<SplitByControl />);
    const select = screen.getByLabelText("Split the plot by") as HTMLSelectElement;
    expect(screen.getByRole("option", { name: "stim (2)" })).not.toBeDisabled();
    expect(screen.getByRole("option", { name: "cell_type (8)" })).toBeDisabled();

    fireEvent.change(select, { target: { value: "stim" } });
    expect(useSplitViewStore.getState().splitBy).toBe("stim");
    fireEvent.change(select, { target: { value: "" } });
    expect(useSplitViewStore.getState().splitBy).toBeNull();
  });

  it("drops a split the newly loaded dataset cannot do", () => {
    useDatasetStore.setState({ currentDataset: dataset([STIM]), currentDatasetId: "ds1" });
    useSplitViewStore.setState({ splitBy: "stim" });
    render(<SplitByControl />);
    expect(useSplitViewStore.getState().splitBy).toBe("stim");
    act(() => useDatasetStore.setState({ currentDataset: dataset([CELL_TYPE]), currentDatasetId: "ds2" }));
    expect(useSplitViewStore.getState().splitBy).toBeNull();
  });
});

describe("SplitScatterGrid", () => {
  // Six cells: CTRL = 0, 2, 4; STIM = 1, 3, 5. Expression rises with the index.
  const positions = Float32Array.from([0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5]);
  const expression = Float32Array.from([0, 1, 2, 3, 4, 5]);

  beforeEach(() => {
    useDatasetStore.setState({ currentDataset: dataset([STIM]), currentDatasetId: "ds1" });
    const table = tableFromArrays({
      x: Float32Array.from([0, 1, 2, 3, 4, 5]),
      y: Float32Array.from([0, 1, 2, 3, 4, 5]),
      stim: Int32Array.from([0, 1, 0, 1, 0, 1]),
    });
    fetchEmbeddingBinary.mockResolvedValue(tableToIPC(table).slice().buffer);
  });

  const renderGrid = (onHover = vi.fn()) =>
    render(
      withQuery(
        <SplitScatterGrid
          datasetId="ds1"
          embedding="X_umap"
          column="stim"
          positions={positions}
          dimensions={2}
          colorValues={expression}
          colorType="continuous"
          onHover={onHover}
        />,
      ),
    );

  it("draws one titled panel per value, each in its own view over the grey context", async () => {
    renderGrid();
    await screen.findByTestId("deck");
    expect(fetchEmbeddingBinary).toHaveBeenCalledWith("ds1", "X_umap", "stim");
    expect(screen.getByText("CTRL")).toBeInTheDocument();
    expect(screen.getByText("STIM")).toBeInTheDocument();
    expect(screen.getAllByText(`${(3).toLocaleString()} cells`)).toHaveLength(2);

    const props = deckProps.current!;
    const views = props.views as { id: string }[];
    expect(views.map((v) => v.id)).toEqual(["split-0", "split-1"]);
    const layers = props.layers as { id: string; props: { data: { length: number } } }[];
    expect(layers.map((l) => [l.id, l.props.data.length])).toEqual([
      ["split-context", 6],
      ["split-fg-0", 3],
      ["split-fg-1", 3],
    ]);
    // One camera: every view gets the same state object.
    const vs = props.viewState as Record<string, unknown>;
    expect(vs["split-0"]).toBe(vs["split-1"]);
  });

  it("notes in its header that a UMAP's cluster gaps aren't distances", async () => {
    renderGrid();
    await screen.findByTestId("deck");
    expect(screen.getByTestId("embedding-caveat")).toHaveTextContent("UMAP: gaps between clusters");
  });

  it("links the cameras: panning one panel moves them all", async () => {
    renderGrid();
    await screen.findByTestId("deck");
    const onChange = deckProps.current!.onViewStateChange as (p: { viewId: string; viewState: unknown }) => void;
    const moved = { target: [9, 9, 0], zoom: 3 };
    act(() => onChange({ viewId: "split-1", viewState: moved }));
    const vs = deckProps.current!.viewState as Record<string, unknown>;
    expect(vs["split-0"]).toEqual(moved);
    expect(vs["split-1"]).toEqual(moved);
  });

  it("reports the hovered cell's original index from any panel", async () => {
    const onHover = vi.fn();
    renderGrid(onHover);
    await screen.findByTestId("deck");
    const hover = deckProps.current!.onHover as (info: unknown) => void;
    // Panels draw their cells in a shuffled order: each slot reports the cell it
    // draws (cell c sits at x = c), and STIM's slots cover its cells 1, 3, 5.
    const stim = (deckProps.current!.layers as { id: string; props: { data: { attributes: { getPosition: { value: Float32Array } } } } }[])
      .find((l) => l.id === "split-fg-1")!.props.data.attributes.getPosition.value;
    const reported = [0, 1, 2].map((s) => {
      onHover.mockClear();
      hover({ picked: true, index: s, x: 10, y: 20, layer: { id: "split-fg-1" } });
      const { index } = onHover.mock.calls[0]![0] as { index: number };
      expect(index).toBe(stim[s * 2]);
      return index;
    });
    expect(reported.sort()).toEqual([1, 3, 5]);
    hover({ picked: false, index: -1, x: 0, y: 0, layer: null });
    expect(onHover).toHaveBeenLastCalledWith(null);
  });

  it("hides the grey context when 'Other cells' is unticked", async () => {
    renderGrid();
    await screen.findByTestId("deck");
    fireEvent.click(screen.getByLabelText("Other cells"));
    await waitFor(() => {
      const ids = (deckProps.current!.layers as { id: string }[]).map((l) => l.id);
      expect(ids).toEqual(["split-fg-0", "split-fg-1"]);
    });
    expect(useSplitViewStore.getState().showContext).toBe(false);
  });

  it("labels each panel's colour groups at their centre in that panel, skipping absent groups", async () => {
    // Coloured by cell type: B = cells 0, 1; NK = 2, 3; T = 4 (CTRL only) and 5 (STIM only).
    render(
      withQuery(
        <SplitScatterGrid
          datasetId="ds1"
          embedding="X_umap"
          column="stim"
          positions={positions}
          dimensions={2}
          colorValues={Int32Array.from([0, 0, 1, 1, 2, 3])}
          colorType="categorical"
          labelCategories={["B", "NK", "T", "Mk"]}
          showLabels
        />,
      ),
    );
    await screen.findByTestId("deck");
    type L = { id: string; props: { data: { label: string; position: number[] }[] } };
    const labels = (deckProps.current!.layers as L[]).filter((l) => l.id.startsWith("split-label"));
    // CTRL has cells 0, 2, 4 (B, NK, T); STIM has 1, 3, 5 (B, NK, Mk).
    expect(labels.map((l) => [l.id, l.props.data.map((d) => `${d.label}@${d.position.join(",")}`)])).toEqual([
      ["split-label-0", ["B@0,0", "NK@2,2", "T@4,4"]],
      ["split-label-1", ["B@1,1", "NK@3,3", "Mk@5,5"]],
    ]);
  });
});
