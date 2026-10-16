import { describe, it, expect, vi, afterEach } from "vitest";
import {
  MAX_CAPTURE_DIM,
  captureScale,
  downloadDeckPng,
  legendLayout,
  type CapturableDeck,
  type ScatterLegend,
} from "./scatterExport";

// 7 px per character: enough to check widths without a real 2D canvas.
const measure = (text: string) => text.length * 7;

const cats = (n: number): Extract<ScatterLegend, { type: "categorical" }> => ({
  type: "categorical",
  title: "leiden",
  categories: Array.from({ length: n }, (_, i) => `cluster ${i}`),
  colors: Array.from({ length: n }, () => [10, 20, 30] as [number, number, number]),
});

afterEach(() => vi.restoreAllMocks());

describe("captureScale", () => {
  it("renders at 3x unless the image would exceed the GPU limit", () => {
    expect(captureScale(800, 600)).toBe(3);
    expect(captureScale(4000, 1000)).toBeCloseTo(MAX_CAPTURE_DIM / 4000, 1);
    expect(captureScale(4000, 1000) * 4000).toBeLessThanOrEqual(MAX_CAPTURE_DIM);
    expect(captureScale(20000, 100)).toBe(1); // never below on-screen resolution
  });
});

describe("legendLayout", () => {
  it("stacks categories in one column when they fit", () => {
    const l = legendLayout(cats(5), 600, measure);
    expect(l.entries).toHaveLength(5);
    expect(new Set(l.entries.map((e) => e.x)).size).toBe(1);
    expect(l.entries[1]!.y - l.entries[0]!.y).toBe(18);
    expect(l.height).toBeLessThan(600);
  });

  it("wraps into more columns on a short plot, keeping inside its height", () => {
    const l = legendLayout(cats(30), 200, measure);
    const xs = [...new Set(l.entries.map((e) => e.x))];
    expect(xs.length).toBeGreaterThan(1);
    expect(Math.max(...l.entries.map((e) => e.y)) + 18).toBeLessThanOrEqual(200);
    expect(l.width).toBeGreaterThan(xs[xs.length - 1]!);
  });

  it("truncates long labels and summarises categories past the cap", () => {
    const legend = cats(150);
    legend.categories[0] = "a".repeat(60);
    const l = legendLayout(legend, 10_000, measure);
    expect(l.entries[0]!.label).toHaveLength(32);
    expect(l.entries[0]!.label.endsWith("…")).toBe(true);
    const last = l.entries[l.entries.length - 1]!;
    expect(last).toMatchObject({ label: "+30 more", color: null });
    expect(l.entries).toHaveLength(121);
  });

  it("draws a colour bar with the value range for continuous colours", () => {
    const l = legendLayout({ type: "continuous", title: "CD3E", min: 0, max: 4.5 }, 600, measure);
    expect(l.bar).toMatchObject({ maxLabel: "4.50", minLabel: "0.00" });
    expect(l.bar!.h).toBe(240);
    expect(l.entries).toHaveLength(0);
  });
});

describe("downloadDeckPng", () => {
  it("redraws once at the capture scale, copies the canvas, then restores the live view", () => {
    const log: string[] = [];
    const source = document.createElement("canvas");
    Object.defineProperty(source, "clientWidth", { value: 400 });
    Object.defineProperty(source, "clientHeight", { value: 300 });
    const deck: CapturableDeck = {
      getCanvas: () => source,
      setProps: ({ useDevicePixels }) => {
        log.push(`dpr ${useDevicePixels}`);
        if (typeof useDevicePixels === "number") {
          source.width = 400 * useDevicePixels;
          source.height = 300 * useDevicePixels;
        }
      },
      redraw: (reason) => log.push(`redraw ${reason}`),
    };
    const ctx = {
      font: "",
      fillStyle: "",
      textBaseline: "",
      fillRect: vi.fn(),
      drawImage: vi.fn(() => log.push("copy")),
      scale: vi.fn(),
      translate: vi.fn(),
      fillText: vi.fn(),
      beginPath: vi.fn(),
      arc: vi.fn(),
      fill: vi.fn(),
      measureText: (t: string) => ({ width: t.length * 7 }),
      createLinearGradient: () => ({ addColorStop: vi.fn() }),
    };
    vi.spyOn(HTMLCanvasElement.prototype, "getContext").mockReturnValue(ctx as unknown as CanvasRenderingContext2D);
    const toBlob = vi.spyOn(HTMLCanvasElement.prototype, "toBlob").mockImplementation(function (this: HTMLCanvasElement) {
      log.push(`png ${this.width}x${this.height}`);
    });

    downloadDeckPng(deck, { legend: cats(3), colormap: "viridis", filename: "scview_x_umap" });

    expect(log.slice(0, 5)).toEqual(["dpr 3", "redraw png export", "copy", "dpr true", "redraw png export done"]);
    expect(toBlob).toHaveBeenCalledTimes(1);
    // 3x the plot, plus the legend to its right; white first, then the points.
    const [, size] = log[5]!.split(" ");
    const [w, h] = size!.split("x").map(Number);
    expect(h).toBe(900);
    expect(w).toBeGreaterThan(1200);
    expect(ctx.fillRect).toHaveBeenCalledWith(0, 0, w, h);
    expect(ctx.fillText).toHaveBeenCalledWith("cluster 2", expect.any(Number), expect.any(Number));
  });

  it("restores the live view even if compositing fails", () => {
    const source = document.createElement("canvas");
    const setProps = vi.fn();
    const deck: CapturableDeck = { getCanvas: () => source, setProps, redraw: vi.fn() };
    vi.spyOn(HTMLCanvasElement.prototype, "getContext").mockReturnValue(null);
    expect(() => downloadDeckPng(deck, { legend: null, colormap: "viridis", filename: "f" })).toThrow();
    expect(setProps).toHaveBeenLastCalledWith({ useDevicePixels: true });
  });
});
