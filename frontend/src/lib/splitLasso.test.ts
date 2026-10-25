import { describe, it, expect } from "vitest";
import { cellsInPanelLasso, panelAt, panelSelectionText } from "./splitLasso";

describe("splitLasso", () => {
  // Camera on (0, 0) at zoom 0 in a 100 x 100 view: pixel (50, 50) is data (0, 0), y up.
  const vs = { target: [0, 0, 0], zoom: 0 };
  const positions = Float32Array.from([-10, -10, 10, 10, 10, -10, 40, 40]);

  it("maps an outline in a panel's pixels to the cells under it", () => {
    // Right half of the view: data x > 0.
    const right: [number, number][] = [[50, 0], [100, 0], [100, 100], [50, 100]];
    expect([...cellsInPanelLasso(positions, 2, right, vs, 100, 100)].sort()).toEqual([1, 2, 3]);
    // Top-right quarter: data x > 0, y > 0 (screen y down).
    const topRight: [number, number][] = [[50, 0], [100, 0], [100, 50], [50, 50]];
    expect([...cellsInPanelLasso(positions, 2, topRight, vs, 100, 100)].sort()).toEqual([1, 3]);
    // ...limited to some cells (a panel's own).
    expect([...cellsInPanelLasso(positions, 2, right, vs, 100, 100, (i) => i !== 3)].sort()).toEqual([1, 2]);
    expect(cellsInPanelLasso(positions, 2, [[0, 0], [1, 1]], vs, 100, 100).size).toBe(0);
  });

  it("finds the panel under a point and words its count", () => {
    const cells = [
      { view: { x: 0, y: 20, width: 100, height: 100 } },
      { view: { x: 110, y: 20, width: 100, height: 100 } },
    ];
    expect(panelAt(cells, 50, 50)).toBe(0);
    expect(panelAt(cells, 150, 50)).toBe(1);
    expect(panelAt(cells, 105, 50)).toBe(-1);
    expect(panelAt(cells, 50, 10)).toBe(-1); // the title band
    expect(panelSelectionText(null, 4210)).toBe(`${(4210).toLocaleString()} cells`);
    expect(panelSelectionText(120, 4210)).toBe(`120 selected of ${(4210).toLocaleString()}`);
  });
});
