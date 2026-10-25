import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { act, fireEvent, render, screen } from "@testing-library/react";

// jsdom has no WebGL: the scatter's Deck is a stub.
vi.mock("@deck.gl/react", async () => {
  const { forwardRef } = await import("react");
  return {
    DeckGL: forwardRef<HTMLDivElement, Record<string, unknown>>((_props, ref) => <div ref={ref} data-testid="deck" />),
  };
});
const saved = vi.hoisted(() => ({ calls: [] as { format: string; svg: string; filename: string }[] }));
vi.mock("@/lib/pdfExport", () => ({
  saveFigure: vi.fn(async (format: string, figure: { svg: string }, filename: string) => {
    saved.calls.push({ format, svg: figure.svg, filename });
  }),
}));

import { EmbeddingScatter } from "./EmbeddingScatter";
import { HYBRID_THRESHOLD } from "@/lib/scatterSvg";

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
  saved.calls = [];
});
afterEach(() => vi.unstubAllGlobals());

const openMenu = () => fireEvent.click(screen.getByRole("button", { name: "Download figure" }));
const items = () => screen.getAllByRole("menuitem").map((b) => b.textContent ?? "");

describe("EmbeddingScatter export menu", () => {
  const positions = Float32Array.from([0, 0, 1, 1, 2, 2, 3, 3]);
  const codes = Int32Array.from([0, 0, 1, 1]);
  const legend = { type: "categorical" as const, title: "cell type", categories: ["B", "NK"], colors: [[1, 2, 3], [4, 5, 6]] as [number, number, number][] };

  it("offers PNG, SVG and PDF in one menu; vector first for a small plot", () => {
    render(
      <EmbeddingScatter positions={positions} colorValues={codes} colorType="categorical" categories={["B", "NK"]} exportLegend={legend} />,
    );
    openMenu();
    const labels = items();
    expect(labels[0]).toMatch(/^PNG/);
    expect(labels[1]).toMatch(/^SVG \(vector\)/);
    expect(labels[2]).toMatch(/^PDF \(vector\)/);
    expect(labels.some((l) => l.includes("points as image"))).toBe(false);
  });

  it("saves an SVG with one circle per cell, the legend, axis names and the labels shown", async () => {
    render(
      <EmbeddingScatter
        positions={positions}
        colorValues={codes}
        colorType="categorical"
        categories={["B", "NK"]}
        showLabels
        exportLegend={legend}
        embedding="X_umap"
        exportName="UMAP_cell_type"
      />,
    );
    openMenu();
    await act(async () => {
      fireEvent.click(screen.getByRole("menuitem", { name: /SVG \(vector\)/ }));
    });
    await vi.waitFor(() => expect(saved.calls).toHaveLength(1));
    const { format, svg, filename } = saved.calls[0]!;
    expect(format).toBe("svg");
    expect(filename).toBe("scview_UMAP_cell_type");
    const doc = new DOMParser().parseFromString(svg, "image/svg+xml");
    expect(doc.querySelectorAll("g.points circle")).toHaveLength(4);
    const texts = [...doc.querySelectorAll("text")].map((t) => t.textContent);
    expect(texts).toEqual(expect.arrayContaining(["UMAP1", "UMAP2", "cell type", "UMAP, coloured by cell type"]));
    // Group labels drawn on screen are in the figure too (halo + text each)
    expect(texts.filter((t) => t === "B")).toHaveLength(3);
  });

  it("puts points as an image first above the threshold", () => {
    const n = HYBRID_THRESHOLD + 10;
    const big = new Float32Array(n * 2).map((_, i) => i % 97);
    render(<EmbeddingScatter positions={big} colorValues={null} colorType="continuous" maxRenderedCells={n} />);
    openMenu();
    const labels = items();
    expect(labels[1]).toMatch(/^SVG, points as image.*Recommended/);
    expect(labels[2]).toMatch(/^PDF, points as image/);
    expect(labels[3]).toMatch(/^SVG, every point vector.*large file/);
  });
});
