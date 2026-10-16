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
const hover = (index: number) =>
  act(() => (deckProps.current!.onHover as (i: unknown) => void)({ picked: true, index, x: 5, y: 5 }));

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
