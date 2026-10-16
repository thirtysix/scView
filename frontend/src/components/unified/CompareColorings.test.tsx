import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { act, fireEvent, render, screen } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { tableFromArrays, tableToIPC } from "apache-arrow";
import type { ReactNode } from "react";
import type { DatasetInfo } from "@/api/types";
import { useDatasetStore } from "@/stores/datasetStore";
import { useSettingsStore } from "@/stores/settingsStore";
import { isGridActive, useSplitViewStore } from "@/stores/splitViewStore";
import { useSelectionStore } from "@/stores/selectionStore";
import { useUnifiedViewStore } from "@/stores/unifiedViewStore";
import { compareLabelKey } from "@/lib/centroidLabels";

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

import { CompareColoringsControl } from "./CompareColoringsControl";
import { CompareColoringsGrid } from "./CompareColoringsGrid";

const STIM = { name: "stim", dtype: "category", n_unique: 2, values: ["CTRL", "STIM"] };
const CELL_TYPE = { name: "cell_type", dtype: "category", n_unique: 3, values: ["B", "NK", "T"] };
const LEIDEN = { name: "leiden", dtype: "category", n_unique: 2, values: ["0", "1"] };
const SAMPLE = { name: "sample", dtype: "category", n_unique: 2, values: ["s1", "s2"] };
const N_GENES = { name: "n_genes", dtype: "int64", n_unique: 6 };

const dataset = (obs: DatasetInfo["obs_columns"], id = "ds1"): DatasetInfo => ({
  id,
  name: "kang",
  filename: "kang.h5ad",
  n_cells: 6,
  n_genes: 10,
  status: "ready",
  available_embeddings: ["X_umap"],
  obs_columns: obs,
  created_at: "",
});

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
  useSplitViewStore.setState({ splitBy: null, compareColumns: null, showContext: true });
  useSettingsStore.setState({ colorBy: "stim" });
  useUnifiedViewStore.setState({ centroidLabels: null });
  useSelectionStore.setState({ highlightedGroup: null });
});
afterEach(() => vi.unstubAllGlobals());

describe("split and compare are one grid at a time", () => {
  it("turning one on turns the other off", () => {
    const s = useSplitViewStore.getState();
    s.setSplitBy("stim");
    s.setCompareColumns(["cell_type", "leiden"]);
    expect(useSplitViewStore.getState()).toMatchObject({ splitBy: null, compareColumns: ["cell_type", "leiden"] });
    expect(isGridActive(useSplitViewStore.getState())).toBe(true);
    useSplitViewStore.getState().setSplitBy("stim");
    expect(useSplitViewStore.getState()).toMatchObject({ splitBy: "stim", compareColumns: null });
    useSplitViewStore.getState().setSplitBy(null);
    expect(isGridActive(useSplitViewStore.getState())).toBe(false);
  });
});

describe("CompareColoringsControl", () => {
  beforeEach(() => {
    useDatasetStore.setState({
      currentDataset: dataset([STIM, N_GENES, CELL_TYPE, LEIDEN, SAMPLE]),
      currentDatasetId: "ds1",
    });
  });

  it("opens with the current colouring and the best other column ticked, and shows them", () => {
    render(<CompareColoringsControl />);
    fireEvent.click(screen.getByRole("button", { name: /Compare/ }));
    const box = (name: string) => screen.getByRole("checkbox", { name: new RegExp(`^${name}`) }) as HTMLInputElement;
    // Groupings only (no n_genes), best first.
    expect(screen.getAllByRole("checkbox").map((c) => c.closest("label")!.textContent)).toEqual([
      "cell_type3",
      "leiden2",
      "stim2",
      "sample2",
    ]);
    expect(box("stim").checked).toBe(true);
    expect(box("cell_type").checked).toBe(true);
    fireEvent.click(box("leiden"));
    fireEvent.click(screen.getByRole("button", { name: /side by side/ }));
    expect(useSplitViewStore.getState().compareColumns).toEqual(["stim", "cell_type", "leiden"]);
  });

  it("lists the dataset's hand-label columns first, and ticks one as the best other column", () => {
    const labelled = { ...dataset([STIM, CELL_TYPE, LEIDEN, SAMPLE]), user_annotation_columns: ["sample"] };
    useDatasetStore.setState({ currentDataset: labelled, currentDatasetId: "ds1" });
    render(<CompareColoringsControl />);
    fireEvent.click(screen.getByRole("button", { name: /Compare/ }));
    const ticked = screen.getAllByRole("checkbox").filter((c) => (c as HTMLInputElement).checked);
    expect(screen.getAllByRole("checkbox")[0]!.closest("label")!.textContent).toBe("sample2");
    expect(ticked.map((c) => c.closest("label")!.textContent)).toEqual(["sample2", "stim2"]);
  });

  it("allows at most four columns and needs two", () => {
    useDatasetStore.setState({
      currentDataset: dataset([STIM, CELL_TYPE, LEIDEN, SAMPLE, { ...SAMPLE, name: "donor" }]),
    });
    render(<CompareColoringsControl />);
    fireEvent.click(screen.getByRole("button", { name: /Compare/ }));
    const boxes = () => screen.getAllByRole("checkbox") as HTMLInputElement[];
    boxes().forEach((b) => !b.checked && !b.disabled && fireEvent.click(b));
    expect(boxes().filter((b) => b.checked)).toHaveLength(4);
    expect(boxes().filter((b) => b.disabled)).toHaveLength(1);
    boxes().forEach((b) => b.checked && fireEvent.click(b));
    fireEvent.click(boxes()[0]!);
    expect(screen.getByRole("button", { name: /side by side/ })).toBeDisabled();
  });

  it("keeps only the columns a newly loaded dataset has, and ends below two", () => {
    useSplitViewStore.setState({ compareColumns: ["cell_type", "leiden", "stim"] });
    render(<CompareColoringsControl />);
    act(() => useDatasetStore.setState({ currentDataset: dataset([CELL_TYPE, STIM], "ds2"), currentDatasetId: "ds2" }));
    expect(useSplitViewStore.getState().compareColumns).toEqual(["cell_type", "stim"]);
    act(() => useDatasetStore.setState({ currentDataset: dataset([CELL_TYPE], "ds3"), currentDatasetId: "ds3" }));
    expect(useSplitViewStore.getState().compareColumns).toBeNull();
  });
});

describe("CompareColoringsGrid", () => {
  // Six cells; cell_type B B NK NK T T, leiden 0 1 0 1 0 1.
  const positions = Float32Array.from([0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5]);
  const codes: Record<string, Int32Array> = {
    cell_type: Int32Array.from([0, 0, 1, 1, 2, 2]),
    leiden: Int32Array.from([0, 1, 0, 1, 0, 1]),
  };

  beforeEach(() => {
    useDatasetStore.setState({ currentDataset: dataset([CELL_TYPE, LEIDEN]), currentDatasetId: "ds1" });
    fetchEmbeddingBinary.mockImplementation(async (_ds: string, _emb: string, column: string) => {
      const table = tableFromArrays({
        x: Float32Array.from([0, 1, 2, 3, 4, 5]),
        y: Float32Array.from([0, 1, 2, 3, 4, 5]),
        [column]: codes[column]!,
      });
      return tableToIPC(table).slice().buffer;
    });
  });

  const withQuery = (ui: ReactNode) => (
    <QueryClientProvider client={new QueryClient({ defaultOptions: { queries: { retry: false } } })}>
      {ui}
    </QueryClientProvider>
  );
  const renderGrid = (onHover = vi.fn()) =>
    render(
      withQuery(
        <CompareColoringsGrid
          datasetId="ds1"
          embedding="X_umap"
          columns={["cell_type", "leiden"]}
          positions={positions}
          dimensions={2}
          onHover={onHover}
        />,
      ),
    );

  it("draws every cell once per column, each panel in its own view, with a legend per column", async () => {
    renderGrid();
    await screen.findByTestId("deck");
    expect(fetchEmbeddingBinary).toHaveBeenCalledWith("ds1", "X_umap", "cell_type");
    expect(fetchEmbeddingBinary).toHaveBeenCalledWith("ds1", "X_umap", "leiden");
    expect(screen.getByText("3 groups")).toBeInTheDocument();
    expect(screen.getByText("2 groups")).toBeInTheDocument();

    const props = deckProps.current!;
    expect((props.views as { id: string }[]).map((v) => v.id)).toEqual(["split-0", "split-1"]);
    const layers = props.layers as { id: string; props: { data: { length: number; label?: string }[] & { length: number } } }[];
    // No grey context: each panel already draws all six cells. Then each
    // panel's group labels (on by default for a few groups).
    expect(layers.map((l) => [l.id, l.props.data.length])).toEqual([
      ["split-fg-0", 6],
      ["split-fg-1", 6],
      ["split-label-0", 3],
      ["split-label-1", 2],
    ]);
    // Both legends, by their categories.
    expect(screen.getByText("NK")).toBeInTheDocument();
    expect(screen.getByText("1")).toBeInTheDocument();
  });

  it("keeps a long label at a panel's left edge inside the panel", async () => {
    // "CD4+ memory T cell" sits on the leftmost cells, ~26 px from the panel's
    // edge; its ~134 px of text would lose its first characters there.
    const long = { ...CELL_TYPE, values: ["CD4+ memory T cell", "NK", "T"] };
    useDatasetStore.setState({ currentDataset: dataset([long, LEIDEN]), currentDatasetId: "ds1" });
    renderGrid();
    await screen.findByTestId("deck");
    type L = { id: string; props: { data: { label: string; offset: number[] }[] } };
    const layer = (deckProps.current!.layers as L[]).find((l) => l.id === "split-label-0")!;
    const offsetX = (label: string) => layer.props.data.find((d) => d.label === label)!.offset[0]!;
    expect(offsetX("CD4+ memory T cell")).toBeGreaterThan(30);
    expect(offsetX("NK")).toBe(0);
  });

  it("drops the labels when toggled off for this set of columns, and fades all but a highlighted group", async () => {
    useUnifiedViewStore.setState({ centroidLabels: { key: compareLabelKey(["cell_type", "leiden"]), show: false } });
    const { unmount } = renderGrid();
    await screen.findByTestId("deck");
    const ids = () => (deckProps.current!.layers as { id: string }[]).map((l) => l.id);
    expect(ids()).toEqual(["split-fg-0", "split-fg-1"]);
    unmount();

    useUnifiedViewStore.setState({ centroidLabels: null });
    useSelectionStore.setState({ highlightedGroup: { column: "cell_type", value: "NK" } });
    renderGrid();
    await screen.findByTestId("deck");
    expect(ids()).toEqual(["split-fg-0", "split-fg-1", "split-label-0", "split-label-0-hl", "split-label-1"]);
  });

  it("reports the hovered cell's value in every compared column", async () => {
    const onHover = vi.fn();
    renderGrid(onHover);
    await screen.findByTestId("deck");
    onHover.mockClear();
    (deckProps.current!.onHover as (i: unknown) => void)({
      picked: true,
      index: 3,
      x: 10,
      y: 20,
      layer: { id: "split-fg-1" },
    });
    const info = onHover.mock.calls[0]![0];
    expect(info.index).toBe(3);
    expect(info.lines.map((l: { label: string; value: string }) => `${l.label}=${l.value}`)).toEqual([
      "cell_type=NK",
      "leiden=1",
    ]);
  });

  it("closes from its header", async () => {
    useSplitViewStore.setState({ compareColumns: ["cell_type", "leiden"] });
    renderGrid();
    await screen.findByTestId("deck");
    fireEvent.click(screen.getByRole("button", { name: "Stop comparing" }));
    expect(useSplitViewStore.getState().compareColumns).toBeNull();
  });
});
