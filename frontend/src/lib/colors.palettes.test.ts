import { describe, it, expect } from "vitest";
import {
  CATEGORICAL_COLORS,
  CATEGORICAL_PALETTES,
  getPalette,
  colormapColor,
  mapCategoryToColor,
  mapValueToColor,
  plotlyColorscale,
} from "./colors";
import type { CategoricalPaletteName, ColormapName } from "./colors";
import { useSettingsStore } from "@/stores/settingsStore";

// First/last stops of matplotlib 3.10.9 colormaps (sampled from the backend container).
const MPL_ENDPOINTS: Record<ColormapName, [number[], number[]]> = {
  viridis: [[68,1,84], [253,231,37]],
  cividis: [[0,34,78], [254,232,56]],
  magma: [[0,0,4], [252,253,191]],
  plasma: [[13,8,135], [240,249,33]],
};

const OLD_CLASSIC = [[0,117,220],[230,25,75],[60,180,75],[255,127,0],[145,30,180],[0,183,183],[240,163,10],[233,90,196],[128,128,0],[70,153,144],[0,51,128],[255,0,16],[43,206,72],[157,204,0],[194,0,136],[116,10,255],[153,63,0],[0,153,143],[255,164,5],[94,106,211]];

describe("settings defaults", () => {
  it("defaults to the colour-blind-safe palette and viridis", () => {
    const s = useSettingsStore.getState();
    expect(s.categoricalPalette).toBe("cvd");
    expect(s.continuousColormap).toBe("viridis");
  });
  it("the default palette is the one the CVD test checks", () => {
    expect(getPalette(useSettingsStore.getState().categoricalPalette)).toBe(CATEGORICAL_COLORS);
  });
  it("setters update the store", () => {
    useSettingsStore.getState().setCategoricalPalette("tableau20");
    useSettingsStore.getState().setContinuousColormap("magma");
    expect(useSettingsStore.getState().categoricalPalette).toBe("tableau20");
    expect(useSettingsStore.getState().continuousColormap).toBe("magma");
    useSettingsStore.getState().setCategoricalPalette("cvd");
    useSettingsStore.getState().setContinuousColormap("viridis");
  });
});

describe("categorical palettes", () => {
  for (const name of Object.keys(CATEGORICAL_PALETTES) as CategoricalPaletteName[]) {
    it(`${name} has 20 unique valid colours`, () => {
      const p = CATEGORICAL_PALETTES[name];
      expect(p.length).toBe(20);
      expect(new Set(p.map((c) => c.join(","))).size).toBe(20);
      expect(p.every((c) => c.length === 3 && c.every((v) => Number.isInteger(v) && v >= 0 && v <= 255))).toBe(true);
    });
  }
  it("classic equals the pre-CVD list (dd72971)", () => {
    expect(CATEGORICAL_PALETTES.classic.map((c) => [...c])).toEqual(OLD_CLASSIC);
  });
  it("mapCategoryToColor honours the palette", () => {
    expect(mapCategoryToColor(0, "classic").slice(0, 3)).toEqual([0, 117, 220]);
    expect(mapCategoryToColor(0).slice(0, 3)).toEqual([...CATEGORICAL_COLORS[0]!]);
  });
});

describe("continuous colormaps", () => {
  for (const name of Object.keys(MPL_ENDPOINTS) as ColormapName[]) {
    it(`${name} endpoints equal matplotlib's first/last stop`, () => {
      expect(colormapColor(name, 0)).toEqual(MPL_ENDPOINTS[name][0]);
      expect(colormapColor(name, 1)).toEqual(MPL_ENDPOINTS[name][1]);
    });
    it(`${name} plotly colorscale spans 0..1`, () => {
      const cs = plotlyColorscale(name);
      expect(cs[0]![0]).toBe(0);
      expect(cs[cs.length - 1]![0]).toBe(1);
    });
  }
  it("mapValueToColor uses the chosen colormap; NaN stays grey", () => {
    expect(mapValueToColor(10, 0, 10, "magma").slice(0, 3)).toEqual(MPL_ENDPOINTS.magma[1]);
    expect(mapValueToColor(NaN, 0, 10, "magma").slice(0, 3)).toEqual([200, 200, 200]);
  });
});
