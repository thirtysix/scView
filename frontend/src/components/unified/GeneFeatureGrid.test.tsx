import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import type { ReactNode } from "react";
import type { DatasetInfo } from "@/api/types";
import { useDatasetStore } from "@/stores/datasetStore";
import { useGeneGridStore } from "@/stores/geneGridStore";
import { useSettingsStore } from "@/stores/settingsStore";
import { isGridActive, useSplitViewStore, type GeneGrid } from "@/stores/splitViewStore";

// jsdom has no WebGL: capture what the grid hands deck.gl instead.
const { deckProps, fetchGeneExpression } = vi.hoisted(() => ({
  deckProps: { current: null as Record<string, unknown> | null },
  fetchGeneExpression: vi.fn(),
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
vi.mock("@/api/expression", () => ({ fetchGeneExpression }));

import { GeneFeatureGrid } from "./GeneFeatureGrid";

class FakeResizeObserver {
  constructor(private cb: ResizeObserverCallback) {}
  observe() {
    this.cb([{ contentRect: { width: 900, height: 400 } } as ResizeObserverEntry], this as never);
  }
  disconnect() {}
  unobserve() {}
}

const dataset: DatasetInfo = {
  id: "ds1",
  name: "kang",
  filename: "kang.h5ad",
  n_cells: 6,
  n_genes: 10,
  status: "ready",
  available_embeddings: ["X_umap"],
  obs_columns: [],
  created_at: "",
};

// Six cells. CD3D in cells 0-2, MS4A1 in 3-4 (much higher), cell 5 neither.
const EXPR: Record<string, Float32Array> = {
  CD3D: Float32Array.from([1, 2, 3, 0, 0, 0]),
  MS4A1: Float32Array.from([0, 0, 0, 8, 10, 0]),
};
const positions = Float32Array.from([0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5]);

type Layer = {
  id: string;
  props: { data: { length: number; attributes: { getFillColor: { value: Uint8Array }; getPosition: { value: Float32Array } } } };
};
const layers = () => deckProps.current!.layers as Layer[];
/** Panel p's RGBA per original cell (the panels draw cells in their own order). */
function colorsByCell(p: number): number[][] {
  const l = layers().find((x) => x.id === `split-fg-${p}`)!;
  const pos = l.props.data.attributes.getPosition.value;
  const rgba = l.props.data.attributes.getFillColor.value;
  const out: number[][] = [];
  for (let s = 0; s < l.props.data.length; s++) out[pos[s * 2]!] = Array.from(rgba.slice(s * 4, s * 4 + 4));
  return out;
}
const drawOrder = (p: number) =>
  Array.from(layers().find((x) => x.id === `split-fg-${p}`)!.props.data.attributes.getPosition.value).filter(
    (_, i) => i % 2 === 0,
  );

const withQuery = (ui: ReactNode) => (
  <QueryClientProvider client={new QueryClient({ defaultOptions: { queries: { retry: false } } })}>{ui}</QueryClientProvider>
);
const renderGrid = (grid: GeneGrid, onHover = vi.fn()) =>
  render(
    withQuery(
      <GeneFeatureGrid datasetId="ds1" embedding="X_umap" grid={grid} positions={positions} dimensions={2} onHover={onHover} />,
    ),
  );

beforeEach(() => {
  vi.stubGlobal("ResizeObserver", FakeResizeObserver);
  deckProps.current = null;
  fetchGeneExpression.mockReset();
  fetchGeneExpression.mockImplementation(async (_ds: string, genes: string[]) =>
    Object.fromEntries(genes.filter((g) => g in EXPR).map((g) => [g, EXPR[g]!])),
  );
  useDatasetStore.setState({ currentDataset: dataset, currentDatasetId: "ds1" });
  useSettingsStore.setState({ expressionLayer: "X", opacity: 1, maxRenderedCells: 1_000_000 });
  useGeneGridStore.setState({ sharedScale: false, blendClip: 0.99 });
  useSplitViewStore.setState({ splitBy: null, compareColumns: null, geneGrid: null });
});
afterEach(() => vi.unstubAllGlobals());

describe("gene grid store", () => {
  it("is one grid at a time with split and compare, and clears on a dataset change", () => {
    const s = useSplitViewStore.getState();
    s.setCompareColumns(["a", "b"]);
    s.setGeneGrid({ datasetId: "ds1", kind: "features", genes: ["CD3D", "MS4A1"] });
    expect(useSplitViewStore.getState().compareColumns).toBeNull();
    expect(isGridActive(useSplitViewStore.getState())).toBe(true);
    useSplitViewStore.getState().setSplitBy("stim");
    expect(useSplitViewStore.getState().geneGrid).toBeNull();
    useSplitViewStore.getState().setGeneGrid({ datasetId: "ds1", kind: "features", genes: ["CD3D", "MS4A1"] });
    useDatasetStore.getState().setCurrentDataset({ ...dataset, id: "ds2" });
    expect(useSplitViewStore.getState().geneGrid).toBeNull();
  });
});

describe("GeneFeatureGrid", () => {
  it("draws one panel per gene from one request, each on its own scale, expressing cells on top", async () => {
    renderGrid({ datasetId: "ds1", kind: "features", genes: ["CD3D", "MS4A1"] });
    await screen.findByTestId("deck");
    expect(fetchGeneExpression).toHaveBeenCalledTimes(1);
    expect(fetchGeneExpression).toHaveBeenCalledWith("ds1", ["CD3D", "MS4A1"], "X");
    expect(layers().map((l) => l.id)).toEqual(["split-fg-0", "split-fg-1"]);
    expect(screen.getByText("50% expressing")).toBeInTheDocument();
    expect(screen.getByText("33% expressing")).toBeInTheDocument();
    // The highest cell of each gene is its last drawn, and gets the top colour.
    expect(drawOrder(0).slice(-3)).toEqual([0, 1, 2]);
    expect(drawOrder(1).slice(-2)).toEqual([3, 4]);
    expect(colorsByCell(0)[2]).toEqual(colorsByCell(1)[4]);
  });

  it("puts every gene on one scale when shared", async () => {
    renderGrid({ datasetId: "ds1", kind: "features", genes: ["CD3D", "MS4A1"] });
    await screen.findByTestId("deck");
    const own = colorsByCell(0)[2];
    fireEvent.click(screen.getByRole("checkbox", { name: /Shared scale/ }));
    await waitFor(() => expect(colorsByCell(0)[2]).not.toEqual(own));
    // CD3D's 3 is now low on the 0..10 scale MS4A1 tops.
    expect(colorsByCell(1)[4]).toEqual(own);
    expect(screen.getByText("Expression (all genes)")).toBeInTheDocument();
  });

  it("names genes the layer lacks and draws the rest", async () => {
    renderGrid({ datasetId: "ds1", kind: "features", genes: ["CD3D", "NOPE", "MS4A1"] });
    await screen.findByTestId("deck");
    expect(layers()).toHaveLength(2);
    expect(screen.getByText(/not in this layer: NOPE/)).toBeInTheDocument();
  });

  it("reports the hovered cell's value of every gene", async () => {
    const onHover = vi.fn();
    renderGrid({ datasetId: "ds1", kind: "features", genes: ["CD3D", "MS4A1"] }, onHover);
    await screen.findByTestId("deck");
    onHover.mockClear();
    const slot = drawOrder(1).indexOf(4);
    (deckProps.current!.onHover as (i: unknown) => void)({ picked: true, index: slot, x: 1, y: 2, layer: { id: "split-fg-1" } });
    const info = onHover.mock.calls[0]![0];
    expect(info.index).toBe(4);
    expect(info.lines.map((l: { label: string; value: string }) => `${l.label}=${l.value}`)).toEqual(["CD3D=0", "MS4A1=10"]);
  });

  it("keeps the tooltip when the parent re-renders (the hover handler stays the same)", async () => {
    const onHover = vi.fn();
    const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
    const grid: GeneGrid = { datasetId: "ds1", kind: "features", genes: ["CD3D", "MS4A1"] };
    const ui = () => (
      <QueryClientProvider client={client}>
        <GeneFeatureGrid datasetId="ds1" embedding="X_umap" grid={grid} positions={positions} dimensions={2} onHover={onHover} />
      </QueryClientProvider>
    );
    const { rerender } = render(ui());
    await screen.findByTestId("deck");
    onHover.mockClear();
    rerender(ui()); // as the tooltip's own state change re-renders UnifiedViewPanel
    expect(onHover).not.toHaveBeenCalledWith(null);
  });

  it("blends two genes in a third panel: A red, B green, neither grey, with the 2D key", async () => {
    renderGrid({ datasetId: "ds1", kind: "blend", genes: ["CD3D", "MS4A1"] });
    await screen.findByTestId("deck");
    expect(layers().map((l) => l.id)).toEqual(["split-fg-0", "split-fg-1", "split-fg-2"]);
    expect(screen.getByText("CD3D + MS4A1")).toBeInTheDocument();
    const blend = colorsByCell(2);
    // Top of each scale (clip 99th pct of the expressing cells) is pure red / green.
    expect(blend[2]!.slice(0, 3)).toEqual([255, 0, 0]);
    expect(blend[4]!.slice(0, 3)).toEqual([0, 255, 0]);
    expect(blend[5]!.slice(0, 3)).toEqual([211, 211, 211]);
    // Neither-expressing cells are drawn first, under the rest.
    expect(drawOrder(2)[0]).toBe(5);
    expect(screen.getByLabelText(/Blend key: CD3D red, MS4A1 green/)).toBeInTheDocument();
    // Scale top at the max instead: CD3D's 2 of 3 is no longer near-full red.
    const before = blend[1]!;
    fireEvent.change(screen.getByRole("combobox", { name: "Blend scale top" }), { target: { value: "1" } });
    await waitFor(() => expect(colorsByCell(2)[1]).not.toEqual(before));
  });

  it("closes from its header", async () => {
    useSplitViewStore.setState({ geneGrid: { datasetId: "ds1", kind: "features", genes: ["CD3D", "MS4A1"] } });
    renderGrid(useSplitViewStore.getState().geneGrid!);
    await screen.findByTestId("deck");
    fireEvent.click(screen.getByRole("button", { name: "Close the gene grid" }));
    expect(useSplitViewStore.getState().geneGrid).toBeNull();
  });
});
