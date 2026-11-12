import { afterEach, describe, expect, it, vi } from "vitest";
import {
  SPLIT_GAP,
  dataBounds,
  downloadSplitSheetPng,
  fitSplitViewState,
  rescaleViewState,
  splitGridLayout,
  splitSheetLayout,
} from "./splitView";
import { rasterizePanels, type SvgPanel } from "./scatterSvg";

afterEach(() => vi.restoreAllMocks());

/** A 2D context that records what is drawn (jsdom has no canvas). */
function fakeContext() {
  const calls: string[] = [];
  const rec = (name: string) => vi.fn((...a: unknown[]) => calls.push(`${name} ${a.join(",")}`));
  const ctx = {
    font: "",
    fillStyle: "",
    strokeStyle: "",
    lineWidth: 1,
    lineJoin: "",
    textAlign: "",
    textBaseline: "",
    globalAlpha: 1,
    fillRect: rec("fillRect"),
    strokeRect: rec("strokeRect"),
    drawImage: rec("drawImage"),
    scale: rec("scale"),
    translate: rec("translate"),
    save: rec("save"),
    restore: rec("restore"),
    rect: rec("rect"),
    clip: rec("clip"),
    fillText: rec("fillText"),
    strokeText: rec("strokeText"),
    beginPath: vi.fn(),
    arc: rec("arc"),
    fill: vi.fn(),
    measureText: (t: string) => ({ width: t.length * 7 }),
    createLinearGradient: () => ({ addColorStop: vi.fn() }),
  };
  vi.spyOn(HTMLCanvasElement.prototype, "getContext").mockReturnValue(ctx as unknown as CanvasRenderingContext2D);
  return { ctx, calls };
}

const panel = (x: number, label: string, labels: SvgPanel["labels"] = []): SvgPanel => ({
  view: { x, y: 22, width: 100, height: 80 },
  title: { box: { x, y: 0, width: 100, height: 22 }, label, subtitle: "3 cells" },
  frame: true,
  sets: [
    { xy: Float32Array.from([1, 1, 2, 2]), colors: [200, 200, 200, 100], radius: 2 },
    { xy: Float32Array.from([5, 5]), colors: Uint8Array.from([255, 0, 0, 255]), radius: 2 },
  ],
  labels,
});

describe("rescaleViewState", () => {
  const b = dataBounds(Float32Array.from([0, 0, 10, 5]), 2);

  it("frames the same part of the embedding in a view of another size", () => {
    const fit = fitSplitViewState(b, 2, 200, 100);
    const zoomedIn = { ...fit, zoom: (fit.zoom as number) + 1, target: [3, 2, 0] };
    const out = rescaleViewState(zoomedIn, b, 2, { width: 200, height: 100 }, { width: 400, height: 200 });
    // Twice the size: one zoom level more, the user's zoom-in and pan kept.
    expect(out.zoom).toBeCloseTo((zoomedIn.zoom as number) + 1);
    expect(out.minZoom).toBeCloseTo((fit.minZoom as number) + 1);
    expect(out.target).toEqual([3, 2, 0]);
  });

  it("leaves the camera alone at the same size", () => {
    const vs = { target: [1, 1, 0], zoom: 2 };
    expect(rescaleViewState(vs, b, 2, { width: 50, height: 50 }, { width: 50, height: 50 })).toEqual(vs);
  });
});

describe("splitSheetLayout", () => {
  it("puts every panel on one sheet at about the on-screen panel size", () => {
    const screen = splitGridLayout(6, 900, 600);
    const sheet = splitSheetLayout(24, 6, 900, 600);
    expect(sheet.grid.cells).toHaveLength(24);
    const onScreen = screen.cells[0]!.view;
    const onSheet = sheet.grid.cells[0]!.view;
    expect(onSheet.width / onScreen.width).toBeGreaterThan(0.7);
    expect(onSheet.width / onScreen.width).toBeLessThan(1.4);
    // Trimmed to the panels: the last panel ends at the sheet's edge.
    const last = sheet.grid.cells[23]!.view;
    expect(sheet.height).toBe(last.y + last.height);
    expect(sheet.width).toBe(sheet.grid.cols * onSheet.width + (sheet.grid.cols - 1) * SPLIT_GAP);
  });

  it("is empty with no panels", () => {
    expect(splitSheetLayout(0, 6, 900, 600)).toMatchObject({ width: 0, height: 0 });
  });
});

describe("rasterizePanels", () => {
  it("paints each panel's title, points clipped to its view, labels over a halo, and its frame", () => {
    const { ctx, calls } = fakeContext();
    rasterizePanels(
      [panel(0, "CTRL", [{ text: "B", x: 10, y: 10, faded: false }, { text: "T", x: 20, y: 20, faded: true }]), panel(108, "STIM")],
      208,
      102,
      2,
    );
    expect(ctx.scale).toHaveBeenCalledWith(2, 2);
    expect(ctx.fillText).toHaveBeenCalledWith("CTRL", 4, 11);
    expect(ctx.fillText).toHaveBeenCalledWith("STIM", 112, 11);
    expect(ctx.fillText).toHaveBeenCalledWith("3 cells", 4 + 4 * 7 + 8, 11);
    expect(ctx.clip).toHaveBeenCalledTimes(2);
    expect(ctx.arc).toHaveBeenCalledTimes(6); // 2 grey + 1 own, per panel
    // The faded label first, both with a halo.
    expect(calls.filter((c) => c.startsWith("strokeText")).map((c) => c.split(" ")[1]!.split(",")[0])).toEqual(["T", "B"]);
    expect(ctx.strokeRect).toHaveBeenCalledTimes(2);
  });
});

describe("downloadSplitSheetPng", () => {
  it("paints the sheet off screen, adds the legend and saves one PNG", () => {
    const { ctx } = fakeContext();
    const toBlob = vi.spyOn(HTMLCanvasElement.prototype, "toBlob").mockImplementation(() => {});
    downloadSplitSheetPng({
      width: 208,
      height: 102,
      panels: [panel(0, "CTRL"), panel(108, "STIM")],
      legend: { type: "categorical", title: "cell type", categories: ["B", "T"], colors: [[1, 2, 3], [4, 5, 6]] },
      colormap: "viridis",
      filename: "scview_kang_split_all",
    });
    expect(ctx.fillText).toHaveBeenCalledWith("CTRL", 4, 11);
    expect(ctx.fillText).toHaveBeenCalledWith("cell type", expect.any(Number), expect.any(Number));
    expect(ctx.drawImage).toHaveBeenCalledTimes(1);
    expect(toBlob).toHaveBeenCalledTimes(1);
  });
});
