import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { act, render, screen } from "@testing-library/react";

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

import { EmbeddingScatter } from "./EmbeddingScatter";

class FakeResizeObserver {
  constructor(private cb: ResizeObserverCallback) {}
  observe() {
    this.cb([{ contentRect: { width: 400, height: 300 } } as ResizeObserverEntry], this as never);
  }
  disconnect() {}
  unobserve() {}
}

beforeEach(() => {
  vi.stubGlobal("ResizeObserver", FakeResizeObserver);
  deckProps.current = null;
});
afterEach(() => vi.unstubAllGlobals());

const positions = Float32Array.from([0, 0, 1, 1, 2, 2]);

type PointLayer = {
  id: string;
  props: { data: { length: number; attributes: { getPosition: { value: Float32Array }; getFillColor: { value: Uint8Array } } } };
};
const pointLayer = () => (deckProps.current!.layers as PointLayer[]).find((l) => l.id === "embedding-scatter")!;
const drawnPositions = () => pointLayer().props.data.attributes.getPosition.value;
/** The slot that draws the cell at x = `x` (cells are drawn in a shuffled order). */
const slotAtX = (x: number) => {
  const pos = drawnPositions();
  for (let s = 0; s < pos.length / 2; s++) if (pos[s * 2] === x) return s;
  throw new Error(`nothing drawn at x = ${x}`);
};
// Cell c of `positions` sits at (c, c).
const hover = (cell: number) =>
  act(() => (deckProps.current!.onHover as (i: unknown) => void)({ picked: true, index: slotAtX(cell), x: 5, y: 5 }));

describe("EmbeddingScatter built-in tooltip", () => {
  it("leads with the colour-by value: the gene and the cell's expression", async () => {
    render(
      <EmbeddingScatter
        positions={positions}
        colorValues={Float32Array.from([0, 1.5, 3])}
        colorType="continuous"
        colorLabel="CD3E expression"
      />,
    );
    await screen.findByTestId("deck");
    hover(1);
    const line = screen.getByTestId("hover-colour-line");
    expect(line).toHaveTextContent("CD3E expression:");
    expect(line).toHaveTextContent("1.500");
    expect(screen.getByText("Cell 1")).toBeInTheDocument();
  });

  it("names the category, with its swatch, when coloured by a column", async () => {
    render(
      <EmbeddingScatter
        positions={positions}
        colorValues={Int32Array.from([0, 1, 1])}
        colorType="categorical"
        colorLabel="Cell type"
        categories={["B", "NK"]}
      />,
    );
    await screen.findByTestId("deck");
    hover(2);
    const line = screen.getByTestId("hover-colour-line");
    expect(line).toHaveTextContent("Cell type:NK");
    expect(line.querySelector("span[aria-hidden]")).not.toBeNull();
  });
});

describe("EmbeddingScatter group labels", () => {
  type L = {
    id: string;
    props: { data: { label: string; position: number[]; offset: number[] }[]; opacity: number };
  };
  const labelLayers = () => (deckProps.current!.layers as L[]).filter((l) => l.id.startsWith("embedding-labels"));
  const codes = Int32Array.from([0, 1, 1]);

  it("draws each category's name among its cells when asked", async () => {
    render(
      <EmbeddingScatter positions={positions} colorValues={codes} colorType="categorical" categories={["B", "NK"]} showLabels />,
    );
    await screen.findByTestId("deck");
    const [layer] = labelLayers();
    expect(layer!.props.data.map((d) => [d.label, d.position, d.offset])).toEqual([
      ["B", [0, 0], [0, 0]],
      ["NK", [1, 1], [0, 0]],
    ]);
  });

  it("keeps a label near the plot's edge inside it", async () => {
    // A wide embedding: the fitted view puts x = 0 about 18 px from the left
    // edge of the 400 px plot, and x = 100 as far from the right one.
    render(
      <EmbeddingScatter
        positions={Float32Array.from([0, 0, 100, 1])}
        colorValues={Int32Array.from([0, 1])}
        colorType="categorical"
        categories={["CD4+ memory T cell", "Megakaryocyte"]}
        showLabels
      />,
    );
    await screen.findByTestId("deck");
    const [layer] = labelLayers();
    const [left, right] = layer!.props.data.map((d) => d.offset[0]!);
    expect(left).toBeGreaterThan(40);
    expect(right).toBeLessThan(-30);
  });

  it("keeps the highlighted group's label and fades the others", async () => {
    render(
      <EmbeddingScatter
        positions={positions}
        colorValues={codes}
        colorType="categorical"
        categories={["B", "NK"]}
        showLabels
        highlightCategory="NK"
      />,
    );
    await screen.findByTestId("deck");
    expect(labelLayers().map((l) => [l.id, l.props.opacity])).toEqual([
      ["embedding-labels", 0.3],
      ["embedding-labels-hl", 1],
    ]);
  });

  it("draws no labels when off, or for a gene's expression", async () => {
    const { unmount } = render(
      <EmbeddingScatter positions={positions} colorValues={codes} colorType="categorical" categories={["B", "NK"]} />,
    );
    await screen.findByTestId("deck");
    expect(labelLayers()).toHaveLength(0);
    unmount();
    render(
      <EmbeddingScatter
        positions={positions}
        colorValues={Float32Array.from([0, 1, 2])}
        colorType="continuous"
        categories={["B", "NK"]}
        showLabels
      />,
    );
    await screen.findByTestId("deck");
    expect(labelLayers()).toHaveLength(0);
  });
});

describe("EmbeddingScatter restored camera", () => {
  const viewState = () => deckProps.current!.viewState as Record<string, unknown>;
  const userMoves = (vs: Record<string, unknown>) =>
    act(() => (deckProps.current!.onViewStateChange as (e: unknown) => void)({ viewState: vs }));

  it("shows a restored camera over the fit, keeps it through a data reload, and drops it when the user moves", async () => {
    const onRestored = vi.fn();
    const onCamera = vi.fn();
    const restored = { target: [1.5, 1.5, 0], zoom: 6 };
    const props = { colorValues: null, colorType: "continuous" as const, onCameraChange: onCamera };
    const { rerender } = render(
      <EmbeddingScatter positions={positions} restoreViewState={restored} onRestoreViewState={onRestored} {...props} />,
    );
    await screen.findByTestId("deck");
    expect(viewState()).toMatchObject(restored);
    expect(typeof viewState().minZoom).toBe("number"); // the fit's limits stay
    expect(onRestored).toHaveBeenCalled();
    expect(onCamera).toHaveBeenLastCalledWith(expect.objectContaining(restored), false);

    // A recolour reloads the positions: the restored camera stays (the panel
    // has cleared restoreViewState by then).
    rerender(<EmbeddingScatter positions={Float32Array.from(positions)} restoreViewState={null} {...props} />);
    expect(viewState()).toMatchObject(restored);

    userMoves({ ...viewState(), target: [0, 0, 0] });
    expect(onCamera).toHaveBeenLastCalledWith(expect.objectContaining({ target: [0, 0, 0] }), false);
    rerender(<EmbeddingScatter positions={Float32Array.from(positions)} restoreViewState={null} {...props} />);
    expect(viewState().zoom).not.toBe(6); // fitted again, as before
    expect(onCamera).toHaveBeenLastCalledWith(expect.anything(), true);
  });

  it("drops a restored camera when another dataset or embedding is shown", async () => {
    const restored = { target: [1.5, 1.5, 0], zoom: 6 };
    const props = { colorValues: null, colorType: "continuous" as const };
    const { rerender } = render(
      <EmbeddingScatter positions={positions} restoreViewState={restored} cameraKey="kang:X_umap" {...props} />,
    );
    await screen.findByTestId("deck");
    expect(viewState()).toMatchObject(restored);
    rerender(<EmbeddingScatter positions={positions} restoreViewState={null} cameraKey="pbmc:X_umap" {...props} />);
    rerender(<EmbeddingScatter positions={Float32Array.from([5, 5, 9, 9])} restoreViewState={null} cameraKey="pbmc:X_umap" {...props} />);
    expect(viewState().zoom).not.toBe(6);
    expect(viewState().target).not.toEqual(restored.target);
  });
});

describe("EmbeddingScatter draw order", () => {
  // 60 cells on a line (cell i at x = i), stored cluster by cluster: 50 of A, then 10 of B.
  const n = 60;
  const line = Float32Array.from({ length: n * 2 }, (_, k) => (k % 2 === 0 ? k / 2 : 0));
  const codes = Int32Array.from({ length: n }, (_, i) => (i < 50 ? 0 : 1));
  const cellAt = (s: number) => drawnPositions()[s * 2]!;

  it("draws the cells shuffled, and a picked point still names its own cell", async () => {
    const onHover = vi.fn();
    const onClick = vi.fn();
    render(
      <EmbeddingScatter
        positions={line}
        colorValues={codes}
        colorType="categorical"
        onHover={onHover}
        onClick={onClick}
      />,
    );
    await screen.findByTestId("deck");
    const drawn = Array.from({ length: n }, (_, s) => cellAt(s));
    expect([...drawn].sort((a, b) => a - b)).toEqual(Array.from({ length: n }, (_, i) => i)); // every cell once
    expect(drawn.slice(50).every((c) => c >= 50)).toBe(false); // B is not simply painted last
    // Each slot's colour is its own cell's group colour.
    const fill = pointLayer().props.data.attributes.getFillColor.value;
    const colourOf = (s: number) => Array.from(fill.slice(s * 4, s * 4 + 3)).join(",");
    expect(colourOf(slotAtX(55))).toBe(colourOf(slotAtX(59)));
    expect(colourOf(slotAtX(55))).not.toBe(colourOf(slotAtX(3)));
    // Hover and click report the cell drawn in the picked slot.
    const s = slotAtX(55);
    act(() => (deckProps.current!.onHover as (i: unknown) => void)({ picked: true, index: s, x: 1, y: 2 }));
    expect(onHover).toHaveBeenLastCalledWith({ index: 55, x: 1, y: 2 });
    act(() => (deckProps.current!.onClick as (i: unknown) => void)({ picked: true, index: s }));
    expect(onClick).toHaveBeenLastCalledWith(55);
  });

  it("keeps the order and the very position buffer for another colouring", async () => {
    const { rerender } = render(<EmbeddingScatter positions={line} colorValues={codes} colorType="categorical" />);
    await screen.findByTestId("deck");
    const before = drawnPositions();
    rerender(
      <EmbeddingScatter
        positions={line}
        colorValues={Float32Array.from({ length: n }, (_, i) => i)}
        colorType="continuous"
      />,
    );
    expect(drawnPositions()).toBe(before); // same array: nothing re-uploaded, no reshuffle
  });

  it("draws a selection over the rest, and its cells still pick as themselves", async () => {
    const onClick = vi.fn();
    render(
      <EmbeddingScatter
        positions={line}
        colorValues={codes}
        colorType="categorical"
        selectedIndices={new Set([3, 7, 1000])}
        onClick={onClick}
      />,
    );
    await screen.findByTestId("deck");
    expect([cellAt(n - 2), cellAt(n - 1)].sort((a, b) => a - b)).toEqual([3, 7]);
    act(() => (deckProps.current!.onClick as (i: unknown) => void)({ picked: true, index: n - 1 }));
    expect(onClick).toHaveBeenLastCalledWith(cellAt(n - 1));
  });

  it("draws only the sample when there are too many cells, shuffled, mapping picks back", async () => {
    const onClick = vi.fn();
    render(
      <EmbeddingScatter positions={line} colorValues={codes} colorType="categorical" maxRenderedCells={20} onClick={onClick} />,
    );
    await screen.findByTestId("deck");
    expect(pointLayer().props.data.length).toBe(20);
    expect(screen.getByText(/cells \(sampled\)/)).toBeInTheDocument();
    act(() => (deckProps.current!.onClick as (i: unknown) => void)({ picked: true, index: 7 }));
    expect(onClick).toHaveBeenLastCalledWith(cellAt(7));
  });

  it("lets the camera zoom out past the fit (2D)", async () => {
    render(<EmbeddingScatter positions={line} colorValues={null} colorType="continuous" />);
    await screen.findByTestId("deck");
    const vs = deckProps.current!.viewState as { zoom: number; minZoom: number; maxZoom: number };
    expect(vs.minZoom).toBeCloseTo(vs.zoom - 4);
    expect(vs.maxZoom).toBeGreaterThan(vs.zoom);
  });
});

describe("EmbeddingScatter embedding caveat", () => {
  it("notes on a UMAP that cluster gaps aren't distances, but not on PCA or the reference map", async () => {
    const props = { positions, colorValues: null, colorType: "continuous" as const };
    const { rerender } = render(<EmbeddingScatter {...props} embedding="X_umap" />);
    await screen.findByTestId("deck");
    expect(screen.getByTestId("embedding-caveat")).toHaveTextContent("UMAP: gaps between clusters");
    rerender(<EmbeddingScatter {...props} embedding="X_pca" />);
    expect(screen.queryByTestId("embedding-caveat")).toBeNull();
    rerender(<EmbeddingScatter {...props} embedding="X_umap" minimal />);
    expect(screen.queryByTestId("embedding-caveat")).toBeNull();
  });
});

describe("EmbeddingScatter overlay layers", () => {
  it("draws a panel's overlay (the PAGA graph) over the points", async () => {
    const { ScatterplotLayer } = await import("@deck.gl/layers");
    const overlay = [new ScatterplotLayer({ id: "paga-nodes", data: [] })];
    render(<EmbeddingScatter positions={positions} colorValues={null} colorType="continuous" overlayLayers={overlay} />);
    await screen.findByTestId("deck");
    const ids = (deckProps.current!.layers as { id: string }[]).map((l) => l.id);
    expect(ids).toEqual(["embedding-scatter", "paga-nodes"]);
  });
});
