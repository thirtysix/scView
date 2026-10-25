import { describe, it, expect } from "vitest";
import {
  cameraAfterKey,
  cellsInScreenRect,
  crosshairAfterKey,
  rectPolygon,
  scatterAriaLabel,
  scatterKeyHelp,
  ZOOM_STEP,
} from "./scatterKeyboard";

const size = { width: 400, height: 300 };

describe("cameraAfterKey", () => {
  const cam = { target: [10, 20, 0], zoom: 1, minZoom: -1, maxZoom: 1.5 };

  it("2D: arrows pan a tenth of the plot in data units, the way the arrow points", () => {
    // zoom 1: 2 pixels per data unit; a tenth of 400 px = 40 px = 20 units.
    expect(cameraAfterKey(cam, "ArrowRight", size, 2)!.target).toEqual([30, 20, 0]);
    expect(cameraAfterKey(cam, "ArrowLeft", size, 2)!.target).toEqual([-10, 20, 0]);
    // data y grows upwards: Up moves the view up.
    expect(cameraAfterKey(cam, "ArrowUp", size, 2)!.target).toEqual([10, 35, 0]);
    expect(cameraAfterKey(cam, "ArrowDown", size, 2)!.target).toEqual([10, 5, 0]);
  });

  it("+ and - zoom within the fit's limits; other keys do nothing", () => {
    expect(cameraAfterKey(cam, "+", size, 2)!.zoom).toBe(1.5);
    expect(cameraAfterKey({ ...cam, zoom: 1.5 }, "=", size, 2)!.zoom).toBe(1.5);
    expect(cameraAfterKey(cam, "-", size, 2)!.zoom).toBe(1 - ZOOM_STEP);
    expect(cameraAfterKey({ ...cam, zoom: -1 }, "-", size, 2)!.zoom).toBe(-1);
    expect(cameraAfterKey(cam, "x", size, 2)).toBeNull();
    expect(cameraAfterKey(cam, "0", size, 2)).toBeNull();
  });

  it("3D: arrows turn the plot (orbit and tilt, tilt kept within +-90 degrees)", () => {
    const orbit = { target: [0, 0, 0], zoom: 0, rotationOrbit: 0, rotationX: 85 };
    expect(cameraAfterKey(orbit, "ArrowRight", size, 3)).toMatchObject({ rotationOrbit: 10, rotationX: 85 });
    expect(cameraAfterKey(orbit, "ArrowUp", size, 3)).toMatchObject({ rotationX: 90 });
    expect(cameraAfterKey(orbit, "ArrowDown", size, 3)).toMatchObject({ rotationX: 75 });
  });
});

describe("crosshair and rectangle", () => {
  it("moves the crosshair 10 px (Shift: 50) and keeps it in the plot", () => {
    expect(crosshairAfterKey([200, 150], "ArrowRight", false, size)).toEqual([210, 150]);
    expect(crosshairAfterKey([200, 150], "ArrowUp", true, size)).toEqual([200, 100]);
    expect(crosshairAfterKey([395, 5], "ArrowRight", true, size)).toEqual([400, 5]);
    expect(crosshairAfterKey([395, 5], "ArrowUp", true, size)).toEqual([395, 0]);
    expect(crosshairAfterKey([1, 1], "Enter", false, size)).toBeNull();
  });

  it("a rectangle from any two opposite corners", () => {
    expect(rectPolygon([30, 40], [10, 5])).toEqual([[10, 5], [30, 5], [30, 40], [10, 40]]);
  });

  it("selects the 2D cells under the rectangle with the camera shown (the lasso's test)", () => {
    // Camera on (0, 0) at zoom 0: screen (200, 150) is data (0, 0), 1 px per unit, y up.
    const cam = { target: [0, 0, 0], zoom: 0 };
    const positions = Float32Array.from([0, 0, 50, 50, -50, -50, 100, 0]);
    // Screen x 190..260, y 90..160 = data x -10..60, y -10..60.
    const cells = cellsInScreenRect(positions, 2, [190, 160], [260, 90], cam, size);
    expect([...cells].sort()).toEqual([0, 1]);
  });
});

describe("what a screen reader hears", () => {
  it("names the embedding, the colouring and the cells (sampled or not)", () => {
    expect(scatterAriaLabel({ embedding: "X_umap", colorLabel: "cell_type", total: 13836, shown: 13836, dimensions: 2 })).toBe(
      "UMAP scatter plot, coloured by cell_type, 13,836 cells",
    );
    expect(scatterAriaLabel({ embedding: "X_umap", colorLabel: null, total: 300000, shown: 100000, dimensions: 3 })).toBe(
      "UMAP (3D) scatter plot, 100,000 of 300,000 cells shown (sampled)",
    );
  });

  it("lists the keys, with the rectangle only where it selects", () => {
    expect(scatterKeyHelp(2, false)).toBe("Arrow keys pan, plus and minus zoom, 0 fits the plot.");
    expect(scatterKeyHelp(3, true)).toMatch(/^Arrow keys turn the plot.*S starts a rectangle selection/);
  });
});
