import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { act, fireEvent, render, screen } from "@testing-library/react";
import type { SpatialLibrary } from "@/api/types";
import type { CellOutlines } from "@/lib/cellOutlines";

// jsdom has no WebGL: capture what the scatter hands deck.gl instead.
const { deckProps } = vi.hoisted(() => ({
  deckProps: { current: null as Record<string, unknown> | null },
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

// The SVG / PDF download: keep what would be saved; the outline limits can be lowered per test.
const saved = vi.hoisted(() => ({ calls: [] as { format: string; svg: string }[] }));
vi.mock("@/lib/pdfExport", () => ({
  saveFigure: vi.fn(async (format: string, figure: { svg: string }) => {
    saved.calls.push({ format, svg: figure.svg });
  }),
}));
const figureLimits = vi.hoisted(() => ({ current: null as { maxOutlines: number; maxVertices: number } | null }));
vi.mock("@/lib/outlineSvg", async (importOriginal) => {
  const real = await importOriginal<typeof import("@/lib/outlineSvg")>();
  return {
    ...real,
    figureOutlines: (...args: Parameters<typeof real.figureOutlines>) =>
      real.figureOutlines(args[0], args[1], args[2], args[3], args[4], figureLimits.current ?? undefined),
    omittedOutlinesNote: (f: Parameters<typeof real.omittedOutlinesNote>[0], kind: "cell" | "nucleus") =>
      real.omittedOutlinesNote(f, kind, figureLimits.current ?? undefined),
  };
});

import { EmbeddingScatter, type TissueMode } from "./EmbeddingScatter";

class FakeResizeObserver {
  constructor(private cb: ResizeObserverCallback) {}
  observe() {
    this.cb([{ contentRect: { width: 400, height: 300 } } as ResizeObserverEntry], this as never);
  }
  disconnect() {}
  unobserve() {}
}

const LIB: SpatialLibrary = { id: "r1", n_obs: 3, images: {}, spot_diameter_fullres: null, bounds: [0, 0, 100, 100] };

// Three cells with a square around them; cell 1 is a square of 2 around (50, -50).
const positions = Float32Array.from([10, -10, 50, -50, 90, -90]);
const square = (x: number, y: number) => [x - 1, y - 1, x + 1, y - 1, x + 1, y + 1, x - 1, y + 1];
const outlines: CellOutlines = {
  nCells: 3,
  offsets: Int32Array.from([0, 4, 8, 12]),
  xy: Float32Array.from([...square(10, 10), ...square(50, 50), ...square(90, 90)]),
};

const tissue = (over: Partial<TissueMode> = {}): TissueMode => ({
  datasetId: "ds1",
  library: LIB,
  showImage: false,
  imageOpacity: 1,
  unit: "cell",
  ...over,
});

type AnyLayer = { id: string; props: Record<string, unknown> };
const layers = () => deckProps.current!.layers as AnyLayer[];
const layer = (id: string) => layers().find((l) => l.id === id);

function renderTissue(t: TissueMode, extra: Record<string, unknown> = {}) {
  return render(
    <EmbeddingScatter
      positions={positions}
      colorValues={Int32Array.from([0, 1, 0])}
      colorType="categorical"
      categories={["A", "B"]}
      opacity={0.5}
      embedding="spatial"
      tissue={t}
      {...extra}
    />,
  );
}

const moveTo = (target: [number, number], zoom: number) =>
  act(() => {
    (deckProps.current!.onViewStateChange as (a: unknown) => void)({ viewState: { target: [...target, 0], zoom } });
  });
const settle = () => act(() => vi.advanceTimersByTime(200));

beforeEach(() => {
  vi.useFakeTimers();
  vi.stubGlobal("ResizeObserver", FakeResizeObserver);
  deckProps.current = null;
  saved.calls = [];
  figureLimits.current = null;
});
afterEach(() => {
  vi.useRealTimers();
  vi.unstubAllGlobals();
});

describe("EmbeddingScatter outlines", () => {
  it("draws the cells in view once the camera rests, each in its point's colour, its point hidden", () => {
    renderTissue(tissue({ outlines }));
    settle();
    expect(screen.getByTestId("deck")).toBeTruthy();
    moveTo([50, -50], 3); // 400 x 300 px at 8 px a unit: only cell 1, with a margin
    expect((layer("cell-outlines")!.props.data as { length: number }).length).toBe(3); // the fit's, while it moves
    settle();
    const fill = layer("cell-outlines")!;
    const data = fill.props.data as { length: number; startIndices: Uint32Array; attributes: Record<string, { value: Uint8Array }> };
    expect(data.length).toBe(1);
    expect([...data.startIndices]).toEqual([0, 4]);
    expect(fill.props._normalize).toBe(false); // deck.gl is handed triangles: nothing to tessellate
    expect(data.attributes.indices!.value.length).toBe(6);
    expect(data.attributes.getFillColor!.value[3]).toBe(Math.round(0.5 * 255));
    expect(fill.props.pickable).toBe(true);

    const points = layer("embedding-scatter")!;
    const colors = (points.props.data as { attributes: { getFillColor: { value: Uint8Array } } }).attributes.getFillColor.value;
    const alphas = [colors[3], colors[7], colors[11]];
    expect(alphas.filter((a) => a === 0).length).toBe(1); // cell 1's, wherever it is in the draw order
    expect(points.props.pickable).toBe(true);

    expect(screen.getByTestId("outline-note").textContent).toBe("Drawing the cell outlines in view (1).");
    const ids = layers().map((l) => l.id);
    expect(ids.indexOf("embedding-scatter")).toBeLessThan(ids.indexOf("cell-outlines"));
    expect(ids).toContain("cell-outlines-edges");
  });

  it("hover and click report the cell of a picked outline, not its polygon's number", () => {
    const onHover = vi.fn();
    const onClick = vi.fn();
    renderTissue(tissue({ outlines }), { onHover, onClick });
    settle();
    moveTo([50, -50], 3);
    settle();
    const info = { index: 0, picked: true, x: 1, y: 2, layer: { id: "cell-outlines" } };
    act(() => (deckProps.current!.onHover as (i: unknown) => void)(info));
    expect(onHover).toHaveBeenLastCalledWith({ index: 1, x: 1, y: 2 });
    act(() => (deckProps.current!.onClick as (i: unknown) => void)(info));
    expect(onClick).toHaveBeenLastCalledWith(1);
  });

  it("follows the colours: a selection dims the outline as it dims the point", () => {
    const { rerender } = renderTissue(tissue({ outlines }));
    settle();
    moveTo([50, -50], 3);
    settle();
    const alpha = () =>
      (layer("cell-outlines")!.props.data as { attributes: { getFillColor: { value: Uint8Array } } }).attributes.getFillColor.value[3];
    expect(alpha()).toBe(Math.round(0.5 * 255));
    rerender(
      <EmbeddingScatter
        positions={positions}
        colorValues={Int32Array.from([0, 1, 0])}
        colorType="categorical"
        categories={["A", "B"]}
        opacity={0.5}
        embedding="spatial"
        tissue={tissue({ outlines })}
        selectedIndices={new Set([0])}
      />,
    );
    expect(alpha()).toBe(Math.round(0.5 * 0.15 * 255));
  });

  it("a thin edge only when a cell is wide enough on screen", () => {
    renderTissue(tissue({ outlines }));
    settle();
    moveTo([50, -50], 0); // 1 px a unit: a cell is 2 px wide
    settle();
    expect(layer("cell-outlines")).toBeDefined();
    expect(layer("cell-outlines-edges")).toBeUndefined();
  });

  it("nothing without outlines: no layer, no note, the colours handed over as they are", () => {
    const { rerender } = renderTissue(tissue());
    settle();
    moveTo([50, -50], 3);
    settle();
    expect(layer("cell-outlines")).toBeUndefined();
    expect(screen.queryByTestId("outline-note")).toBeNull();
    const colors = (layer("embedding-scatter")!.props.data as { attributes: { getFillColor: { value: Uint8Array } } }).attributes.getFillColor.value;
    expect(colors[3]).toBe(Math.round(0.5 * 255));
    rerender(<div />);
  });

  it("tells the transcripts where the camera rests", () => {
    const onSettledView = vi.fn();
    renderTissue(tissue(), { onSettledView });
    settle();
    moveTo([50, -50], 1);
    settle();
    const rect = onSettledView.mock.calls.at(-1)![0];
    expect([rect.x0, rect.x1, rect.y0, rect.y1, rect.scale]).toEqual([-50, 150, -125, 25, 2]);
  });
});


describe("EmbeddingScatter outlines in the SVG / PDF", () => {
  async function saveSvg(): Promise<Document> {
    fireEvent.click(screen.getByRole("button", { name: "Download figure" }));
    await act(async () => {
      fireEvent.click(screen.getByRole("menuitem", { name: /SVG \(vector\)/ }));
      await vi.advanceTimersByTimeAsync(10);
    });
    expect(saved.calls).toHaveLength(1);
    return new DOMParser().parseFromString(saved.calls[0]!.svg, "image/svg+xml");
  }

  it("the outlines on screen are in the file, over the points, their cells not also dots", async () => {
    renderTissue(tissue({ outlines }));
    settle();
    moveTo([50, -50], 3);
    settle();
    fireEvent.click(screen.getByRole("button", { name: "Download figure" }));
    const hints = screen.getAllByRole("menuitem").map((b) => b.textContent ?? "").join(" | ");
    expect(hints).not.toMatch(/without the outlines/);
    fireEvent.click(screen.getByRole("button", { name: "Download figure" }));
    const doc = await saveSvg();
    const paths = doc.querySelectorAll("g.panel g.outlines path");
    expect(paths).toHaveLength(1); // cell 1 only: the margin's cells are out of the view
    expect(paths[0]!.getAttribute("fill-opacity")).toBe("0.502"); // the plot's opacity 0.5
    // 2 units at 8 px a unit: a 16 px square
    expect(paths[0]!.getAttribute("d")).toMatch(/^M[\d.]+ [\d.]+l16 0l0 -?16l-16 0z$/);
    const g = doc.querySelector("g.outlines")!;
    expect(g.getAttribute("stroke")).toBe("#1e293b"); // 16 px a cell: edged, as on screen
    expect(doc.querySelectorAll("g.points circle")).toHaveLength(0); // cell 1's dot is hidden under it
    const texts = [...doc.querySelectorAll("text")].map((t) => t.textContent);
    expect(texts.join(" ")).toContain("Drawing the cell outlines in view (1).");
  });

  it("zoomed out, every outline in view and no edge (cells under 4 px)", async () => {
    renderTissue(tissue({ outlines }));
    settle();
    moveTo([50, -50], 0);
    settle();
    const doc = await saveSvg();
    const g = doc.querySelector("g.outlines")!;
    expect(g.hasAttribute("stroke")).toBe(false);
    const rings = [...g.querySelectorAll("path")].map((p) => p.getAttribute("d")!.split("z").filter(Boolean).length);
    expect(rings.reduce((a, b) => a + b, 0)).toBe(3);
  });

  it("nothing changes without outlines: dots only, no outline group", async () => {
    renderTissue(tissue());
    settle();
    moveTo([50, -50], 0);
    settle();
    const doc = await saveSvg();
    expect(doc.querySelector("g.outlines")).toBeNull();
    expect(doc.querySelectorAll("g.points circle")).toHaveLength(3);
  });

  it("over the size limit: saved with the cells as dots, a note in the file and in the menu", async () => {
    figureLimits.current = { maxOutlines: 0, maxVertices: 1e9 };
    renderTissue(tissue({ outlines }));
    settle();
    moveTo([50, -50], 3);
    settle();
    const doc = await saveSvg();
    expect(doc.querySelector("g.outlines")).toBeNull();
    expect(doc.querySelectorAll("g.points circle")).toHaveLength(1); // cell 1 shown as its dot again
    const note = "Cell outlines left out (1 in view, over the 0 a figure holds): zoom in for them. The cells are drawn as points.";
    expect([...doc.querySelectorAll("text")].map((t) => t.textContent).join(" ")).toContain(note); // wrapped over lines
    expect(screen.getByRole("alert").textContent).toBe(`Saved. ${note}`);
  });
});
