import { describe, it, expect, vi, afterEach } from "vitest";
import { fireEvent, render } from "@testing-library/react";
import { cellsInPolygon3D, LassoSelector, screenToData } from "./LassoSelector";

afterEach(() => vi.restoreAllMocks());

describe("screenToData", () => {
  it("maps the plot's centre to the camera target, with data y growing upwards", () => {
    const vs = { target: [10, 5, 0], zoom: 1 }; // 2 px per data unit
    expect(screenToData(100, 50, vs, 200, 100)).toEqual([10, 5]);
    expect(screenToData(120, 50, vs, 200, 100)).toEqual([20, 5]); // right
    expect(screenToData(100, 30, vs, 200, 100)).toEqual([10, 15]); // up on screen = up in data
  });
});

describe("LassoSelector", () => {
  // 100 x 100 px plot, camera on the origin at 1 px per unit: cell 0 sits 30
  // units above the centre (screen y 20), cell 1 30 below (screen y 80).
  const positions = Float32Array.from([0, 30, 0, -30]);
  const drawLasso = (getViewState: () => Record<string, unknown> | null) => {
    vi.spyOn(Element.prototype, "getBoundingClientRect").mockReturnValue(
      { left: 0, top: 0, width: 100, height: 100, right: 100, bottom: 100, x: 0, y: 0 } as DOMRect,
    );
    const onSelectionComplete = vi.fn();
    const { container } = render(
      <LassoSelector active positions={positions} getViewState={getViewState} onSelectionComplete={onSelectionComplete} />,
    );
    const svg = container.querySelector("svg")!;
    // A box around the top of the plot.
    fireEvent.mouseDown(svg, { clientX: 40, clientY: 10 });
    for (const [x, y] of [[60, 10], [60, 30], [40, 30]]) fireEvent.mouseMove(svg, { clientX: x, clientY: y });
    fireEvent.mouseUp(svg);
    return onSelectionComplete;
  };

  it("selects the cells drawn inside the lasso, under the camera shown when it closes", () => {
    const done = drawLasso(() => ({ target: [0, 0, 0], zoom: 0 }));
    expect(done).toHaveBeenCalledWith(new Set([0]));
  });

  it("follows the camera: panned 60 units up, the top of the plot holds the other cell", () => {
    const done = drawLasso(() => ({ target: [0, -60, 0], zoom: 0 }));
    expect(done).toHaveBeenCalledWith(new Set([1]));
  });

  it("selects nothing before there is a camera", () => {
    const done = drawLasso(() => null);
    expect(done).not.toHaveBeenCalled();
  });
});

describe("3D lasso", () => {
  // 100 x 100 px plot; OrbitView at zoom 0 draws 1 px per unit on the plane
  // through the target. Cells: 0 above the centre, 1 below, 2 above and 40
  // units nearer the camera, 3 right of the centre, 4 left of it.
  const cells = Float32Array.from([0, 30, 0, 0, -30, 0, 0, 30, 40, 30, 0, 0, -30, 0, 0]);
  const camera = (rotationOrbit = 0) => ({ target: [0, 0, 0], zoom: 0, rotationOrbit, rotationX: 0 });
  const topBox: [number, number][] = [[35, 0], [65, 0], [65, 35], [35, 35]];
  const rightBox: [number, number][] = [[65, 40], [100, 40], [100, 60], [65, 60]];

  it("selects every cell under the outline, at any depth", () => {
    expect(cellsInPolygon3D(cells, topBox, camera(), 100, 100)).toEqual(new Set([0, 2]));
  });

  it("follows the rotation: turned half way round, the right of the plot holds the other cell", () => {
    expect(cellsInPolygon3D(cells, rightBox, camera(0), 100, 100)).toEqual(new Set([3]));
    expect(cellsInPolygon3D(cells, rightBox, camera(180), 100, 100)).toEqual(new Set([4]));
  });

  it("draws on a 3D plot instead of leaving the drag to the camera", () => {
    vi.spyOn(Element.prototype, "getBoundingClientRect").mockReturnValue(
      { left: 0, top: 0, width: 100, height: 100, right: 100, bottom: 100, x: 0, y: 0 } as DOMRect,
    );
    const onSelectionComplete = vi.fn();
    const { container } = render(
      <LassoSelector active dimensions={3} positions={cells} getViewState={() => camera()} onSelectionComplete={onSelectionComplete} />,
    );
    const svg = container.querySelector("svg")!;
    fireEvent.mouseDown(svg, { clientX: 35, clientY: 0 });
    for (const [x, y] of [[65, 0], [65, 35], [35, 35]]) fireEvent.mouseMove(svg, { clientX: x, clientY: y });
    fireEvent.mouseUp(svg);
    expect(onSelectionComplete).toHaveBeenCalledWith(new Set([0, 2]));
  });
});
