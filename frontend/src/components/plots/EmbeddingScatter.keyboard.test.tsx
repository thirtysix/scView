import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { fireEvent, render, screen } from "@testing-library/react";
import { seriousViolations } from "@/test/axe";

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

// Cell c sits at (c, c).
const positions = Float32Array.from([0, 0, 1, 1, 2, 2]);
const camera = () => deckProps.current!.viewState as { target: number[]; zoom: number };
const plot = () => screen.getByRole("application");
const key = (k: string, opts: Record<string, unknown> = {}) => fireEvent.keyDown(plot(), { key: k, ...opts });

function renderScatter(onKeyboardSelect?: (s: Set<number>) => void) {
  return render(
    <EmbeddingScatter
      positions={positions}
      colorValues={Int32Array.from([0, 1, 1])}
      colorType="categorical"
      categories={["B", "NK"]}
      embedding="X_umap"
      exportLegend={{ type: "categorical", title: "cell_type", categories: ["B", "NK"], colors: [[0, 0, 0], [1, 1, 1]] }}
      onKeyboardSelect={onKeyboardSelect}
    />,
  );
}

describe("EmbeddingScatter by keyboard", () => {
  it("is a focusable, named plot that says what it shows and which keys it takes", async () => {
    const { container } = renderScatter(() => {});
    const el = plot();
    expect(el).toHaveAttribute("tabindex", "0");
    expect(el).toHaveAccessibleName("UMAP scatter plot, coloured by cell_type, 3 cells");
    expect(el).toHaveAccessibleDescription(/Arrow keys pan, plus and minus zoom, 0 fits the plot\. S starts a rectangle selection/);
    expect(await seriousViolations(container)).toEqual([]);
  });

  it("pans with the arrows, zooms with + and -, and 0 fits it again", () => {
    renderScatter();
    const fit = camera();
    key("ArrowRight");
    expect(camera().target[0]).toBeGreaterThan(fit.target[0]!);
    expect(camera().target[1]).toBe(fit.target[1]);
    key("ArrowUp");
    expect(camera().target[1]).toBeGreaterThan(fit.target[1]!);
    key("+");
    expect(camera().zoom).toBeCloseTo(fit.zoom + 0.5);
    key("-");
    key("-");
    expect(camera().zoom).toBeCloseTo(fit.zoom - 0.5);
    key("0");
    expect(camera().target).toEqual(fit.target);
    expect(camera().zoom).toBe(fit.zoom);
  });

  it("selects the cells under a rectangle drawn with the keyboard (S, arrows, Enter, Enter)", () => {
    const onSelect = vi.fn();
    renderScatter(onSelect);
    key("s");
    expect(screen.getByTestId("keyboard-rect")).toBeInTheDocument();
    expect(screen.getByText(/move the crosshair to one corner/)).toBeInTheDocument();
    // First corner just left of and below the centre (cell 1 sits at the centre).
    key("ArrowLeft");
    key("ArrowDown");
    key("Enter");
    expect(screen.getByText(/First corner set/)).toBeInTheDocument();
    // Opposite corner: top right of the plot (Shift moves further).
    for (let i = 0; i < 5; i++) {
      key("ArrowRight", { shiftKey: true });
      key("ArrowUp", { shiftKey: true });
    }
    key("Enter");
    expect(onSelect).toHaveBeenCalledTimes(1);
    expect([...onSelect.mock.calls[0]![0]].sort()).toEqual([1, 2]);
    expect(screen.queryByTestId("keyboard-rect")).toBeNull();
  });

  it("Escape cancels the rectangle without reaching the panel's Escape (clear selection)", () => {
    const onSelect = vi.fn();
    const outer = vi.fn((e: KeyboardEvent) => e.key);
    window.addEventListener("keydown", outer);
    renderScatter(onSelect);
    key("s");
    key("Escape");
    expect(screen.queryByTestId("keyboard-rect")).toBeNull();
    expect(screen.getByText("Rectangle selection cancelled.")).toBeInTheDocument();
    expect(outer.mock.results.map((r) => r.value)).toEqual(["s"]);
    // While drawing, arrows move the crosshair, not the camera; afterwards they pan again.
    const before = camera().target;
    key("ArrowRight");
    expect(camera().target).not.toEqual(before);
    window.removeEventListener("keydown", outer);
    expect(onSelect).not.toHaveBeenCalled();
  });

  it("offers no rectangle without a selection handler, and a compact map takes no focus", () => {
    const { unmount } = renderScatter();
    key("s");
    expect(screen.queryByTestId("keyboard-rect")).toBeNull();
    expect(plot()).toHaveAccessibleDescription(/0 fits the plot\.$/);
    unmount();
    render(<EmbeddingScatter positions={positions} colorValues={null} colorType="continuous" minimal />);
    expect(screen.queryByRole("application")).toBeNull();
  });
});
