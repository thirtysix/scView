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
