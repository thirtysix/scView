import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { act, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { tableFromArrays, tableToIPC } from "apache-arrow";
import type { DatasetInfo } from "@/api/types";
import { splitGridLayout } from "@/lib/splitView";
import { useDatasetStore } from "@/stores/datasetStore";
import { useSelectionStore } from "@/stores/selectionStore";
import { useSplitViewStore } from "@/stores/splitViewStore";

// jsdom has no WebGL: capture what the grid hands deck.gl instead.
const { deckProps, fetchEmbeddingBinary, saved, sheetPng } = vi.hoisted(() => ({
  deckProps: { current: null as Record<string, unknown> | null },
  fetchEmbeddingBinary: vi.fn(),
  saved: { calls: [] as { format: string; svg: string; filename: string }[] },
  sheetPng: vi.fn(),
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
vi.mock("@/lib/pdfExport", () => ({
  saveFigure: vi.fn(async (format: string, figure: { svg: string }, filename: string) => {
    saved.calls.push({ format, svg: figure.svg, filename });
  }),
}));
vi.mock("@/lib/splitView", async (orig) => ({
  ...(await orig<typeof import("@/lib/splitView")>()),
  downloadSplitSheetPng: sheetPng,
}));

import { SplitScatterGrid } from "./SplitScatterGrid";

const W = 900;
const H = 600;
class FakeResizeObserver {
  constructor(private cb: ResizeObserverCallback) {}
  observe() {
    this.cb([{ contentRect: { width: W, height: H } } as ResizeObserverEntry], this as never);
  }
  disconnect() {}
  unobserve() {}
}

// Eight donors, two cells each: cell c sits at x = y = c and belongs to donor floor(c / 2).
const DONORS = Array.from({ length: 8 }, (_, i) => `D${i + 1}`);
const N = 16;
const positions = Float32Array.from(Array.from({ length: N }, (_, c) => [c, c]).flat());
const donorCodes = Int32Array.from(Array.from({ length: N }, (_, c) => Math.floor(c / 2)));

const dataset: DatasetInfo = {
  id: "ds1",
  name: "kang",
  filename: "kang.h5ad",
  n_cells: N,
  n_genes: 10,
  status: "ready",
  available_embeddings: ["X_umap"],
  obs_columns: [{ name: "donor", dtype: "category", n_unique: 8, values: DONORS }],
  created_at: "",
};

beforeEach(() => {
  vi.stubGlobal("ResizeObserver", FakeResizeObserver);
  deckProps.current = null;
  saved.calls = [];
  sheetPng.mockReset();
  useDatasetStore.setState({ currentDataset: dataset, currentDatasetId: "ds1" });
  useSplitViewStore.setState({ splitBy: "donor", showContext: true, lassoScope: "panel", panelsPerPage: 6, page: 0 });
  useSelectionStore.setState({ selectedCellIndices: null, selectionMode: "none", highlightedGroup: null });
  const table = tableFromArrays({
    x: Float32Array.from(positions.filter((_, i) => i % 2 === 0)),
    y: Float32Array.from(positions.filter((_, i) => i % 2 === 1)),
    donor: donorCodes,
  });
  fetchEmbeddingBinary.mockResolvedValue(tableToIPC(table).slice().buffer);
});
afterEach(() => vi.unstubAllGlobals());

function renderGrid(onLassoComplete?: (cells: Set<number>) => void, onHover = vi.fn()) {
  render(
    <QueryClientProvider client={new QueryClient({ defaultOptions: { queries: { retry: false } } })}>
      <SplitScatterGrid
        datasetId="ds1"
        embedding="X_umap"
        column="donor"
        positions={positions}
        dimensions={2}
        colorValues={null}
        colorType="categorical"
        onLassoComplete={onLassoComplete}
        onHover={onHover}
        exportLegend={{ type: "categorical", title: "donor", categories: DONORS, colors: DONORS.map(() => [1, 2, 3]) }}
      />
    </QueryClientProvider>,
  );
  return { onHover };
}

type L = { id: string; props: { data: { length: number; attributes?: { getPosition: { value: Float32Array } } } } };
const layers = () => deckProps.current!.layers as L[];
const viewIds = () => (deckProps.current!.views as { id: string }[]).map((v) => v.id);

describe("SplitScatterGrid paging", () => {
  it("splits by up to 24 values, six panels a page by default, drawing only the page", async () => {
    renderGrid();
    await screen.findByTestId("deck");
    expect(screen.getByText("Page 1 of 2")).toBeInTheDocument();
    expect(viewIds()).toEqual(["split-0", "split-1", "split-2", "split-3", "split-4", "split-5"]);
    expect(layers().map((l) => l.id)).toEqual(["split-context", ...[0, 1, 2, 3, 4, 5].map((i) => `split-fg-${i}`)]);
    expect(screen.getByText("D1")).toBeInTheDocument();
    expect(screen.queryByText("D7")).toBeNull();

    fireEvent.click(screen.getByRole("button", { name: "Next page" }));
    expect(screen.getByText("Page 2 of 2")).toBeInTheDocument();
    expect(useSplitViewStore.getState().page).toBe(1);
    // A short last page keeps the full page's layout: two views of the same size.
    expect(viewIds()).toEqual(["split-0", "split-1"]);
    const views = deckProps.current!.views as { props: { width: number } }[];
    expect(views[0]!.props.width).toBe(splitGridLayout(6, W, H, 1).cells[0]!.view.width);
    expect(screen.getByText("D7")).toBeInTheDocument();
    expect(screen.getByText("D8")).toBeInTheDocument();
    expect(screen.queryByText("D1")).toBeNull();
    // Slot 0 of page 2 draws donor D7's cells, 12 and 13; the context is still every cell.
    const d7 = layers().find((l) => l.id === "split-fg-0")!.props.data.attributes!.getPosition.value;
    expect([...d7].filter((_, i) => i % 2 === 0).sort((a, b) => a - b)).toEqual([12, 13]);
    expect(layers().find((l) => l.id === "split-context")!.props.data.length).toBe(N);
  });

  it("shares one camera across pages", async () => {
    renderGrid();
    await screen.findByTestId("deck");
    const onChange = deckProps.current!.onViewStateChange as (p: { viewState: unknown }) => void;
    const moved = { target: [3, 3, 0], zoom: 4 };
    act(() => onChange({ viewState: moved }));
    fireEvent.click(screen.getByRole("button", { name: "Next page" }));
    const vs = deckProps.current!.viewState as Record<string, unknown>;
    expect(vs["split-0"]).toEqual(moved);
    expect(vs["split-1"]).toEqual(moved);
  });

  it("shows every panel on one page with All, and 4 a page keeps the panels in view", async () => {
    renderGrid();
    await screen.findByTestId("deck");
    fireEvent.click(screen.getByRole("button", { name: "Next page" }));
    fireEvent.change(screen.getByLabelText("Panels per page"), { target: { value: "4" } });
    // Page 2 of 6-a-page started at D7: at 4 a page that is page 2 (D5-D8).
    expect(screen.getByText("Page 2 of 2")).toBeInTheDocument();
    expect(screen.getByText("D5")).toBeInTheDocument();
    fireEvent.change(screen.getByLabelText("Panels per page"), { target: { value: "all" } });
    expect(viewIds()).toHaveLength(8);
    expect(screen.queryByRole("button", { name: "Next page" })).toBeNull();
  });

  it("reports a hovered cell's panel as its index among all panels", async () => {
    const { onHover } = renderGrid();
    await screen.findByTestId("deck");
    fireEvent.click(screen.getByRole("button", { name: "Next page" }));
    const hover = deckProps.current!.onHover as (info: unknown) => void;
    hover({ picked: true, index: 0, x: 1, y: 1, layer: { id: "split-fg-1" } });
    const info = onHover.mock.calls.at(-1)![0] as { index: number; panel: number };
    expect(info.panel).toBe(7);
    expect([14, 15]).toContain(info.index);
  });

  it("a lasso on page 2 picks that page's panel's cells", async () => {
    useSelectionStore.setState({ selectionMode: "lasso" });
    const onLasso = vi.fn();
    renderGrid(onLasso);
    await screen.findByTestId("deck");
    fireEvent.click(screen.getByRole("button", { name: "Next page" }));
    // A box over the whole of slot 1 (donor D8: cells 14, 15).
    const v = splitGridLayout(6, W, H, 1).cells[1]!.view;
    const svg = screen.getByTestId("grid-lasso");
    const pts = [
      [v.x + 1, v.y + 1],
      [v.x + v.width - 1, v.y + 1],
      [v.x + v.width - 1, v.y + v.height - 1],
      [v.x + 1, v.y + v.height - 1],
    ];
    fireEvent.mouseDown(svg, { clientX: pts[0]![0], clientY: pts[0]![1] });
    for (const [x, y] of pts.slice(1)) fireEvent.mouseMove(svg, { clientX: x, clientY: y });
    fireEvent.mouseUp(svg);
    expect(onLasso).toHaveBeenCalledTimes(1);
    expect([...(onLasso.mock.calls[0]![0] as Set<number>)].sort((a, b) => a - b)).toEqual([14, 15]);
  });

  it("downloads this page as SVG: the page's panels, titles and the legend", async () => {
    renderGrid();
    await screen.findByTestId("deck");
    fireEvent.click(screen.getByRole("button", { name: "Next page" }));
    fireEvent.click(screen.getByRole("button", { name: "Download grid" }));
    await act(async () => {
      fireEvent.click(screen.getByRole("menuitem", { name: /^PDF \(vector\)/ }));
    });
    await waitFor(() => expect(saved.calls).toHaveLength(1));
    const { format, svg, filename } = saved.calls[0]!;
    expect(format).toBe("pdf");
    expect(filename).toBe("scview_kang_embedding_split_donor_page2");
    const doc = new DOMParser().parseFromString(svg, "image/svg+xml");
    expect(doc.querySelectorAll("g.panel")).toHaveLength(2);
    const texts = [...doc.querySelectorAll("text")].map((t) => t.textContent);
    expect(texts).toEqual(expect.arrayContaining(["D7", "D8", "2 cells", "donor"]));
    // D7 titles a panel and is in the legend; D1 is only in the legend.
    expect(texts.filter((t) => t === "D7")).toHaveLength(2);
    expect(texts.filter((t) => t === "D1")).toHaveLength(1);
  });

  it("downloads all panels on one sheet as SVG, whatever the page", async () => {
    renderGrid();
    await screen.findByTestId("deck");
    fireEvent.click(screen.getByRole("button", { name: "Download grid" }));
    await act(async () => {
      fireEvent.click(screen.getByRole("menuitem", { name: /All 8 panels: SVG \(vector\)/ }));
    });
    await waitFor(() => expect(saved.calls).toHaveLength(1));
    const { svg, filename } = saved.calls[0]!;
    expect(filename).toBe("scview_kang_embedding_split_donor_all");
    const doc = new DOMParser().parseFromString(svg, "image/svg+xml");
    const panels = [...doc.querySelectorAll("g.panel")];
    expect(panels).toHaveLength(8);
    // Each panel: all 16 cells in grey + its own 2, and one legend for the sheet.
    expect(panels.map((p) => p.querySelectorAll("g.points circle").length)).toEqual(Array(8).fill(18));
    expect(doc.querySelectorAll("g.legend")).toHaveLength(1);
    const texts = [...doc.querySelectorAll("text")].map((t) => t.textContent);
    expect(texts).toEqual(expect.arrayContaining(DONORS));
    // The live grid stays on its page.
    expect(viewIds()).toHaveLength(6);
  });

  it("downloads all panels on one PNG, painted off screen", async () => {
    renderGrid();
    await screen.findByTestId("deck");
    fireEvent.click(screen.getByRole("button", { name: "Download grid" }));
    await act(async () => {
      fireEvent.click(screen.getByRole("menuitem", { name: /All 8 panels: PNG/ }));
    });
    await waitFor(() => expect(sheetPng).toHaveBeenCalledTimes(1));
    const arg = sheetPng.mock.calls[0]![0] as {
      panels: { title: { label: string; subtitle: string } }[];
      legend: { title: string };
      filename: string;
      width: number;
      height: number;
    };
    expect(arg.panels.map((p) => p.title.label)).toEqual(DONORS);
    expect(arg.panels[0]!.title.subtitle).toBe("2 cells");
    expect(arg.legend.title).toBe("donor");
    expect(arg.filename).toBe("scview_kang_embedding_split_donor_all");
    expect(arg.width).toBeGreaterThan(0);
    expect(arg.height).toBeGreaterThan(0);
  });

  it("offers no all-panels items when every panel fits one page", async () => {
    useSplitViewStore.setState({ panelsPerPage: 12 });
    renderGrid();
    await screen.findByTestId("deck");
    fireEvent.click(screen.getByRole("button", { name: "Download grid" }));
    expect(screen.getAllByRole("menuitem")).toHaveLength(3);
  });
});
