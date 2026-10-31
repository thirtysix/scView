import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { act, render, screen } from "@testing-library/react";
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
