import { describe, it, expect, beforeEach, vi } from "vitest";
import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import type { Data, Layout, PlotlyHTMLElement } from "plotly.js";
import {
  DARK_SLATE,
  DARK_SURFACE,
  THEME_STORAGE_KEY,
  applyTheme,
  currentTheme,
  darkNeutral,
  exportFigure,
  initTheme,
  plotTheme,
  readStoredPreference,
  rememberBaseFigure,
  resolveTheme,
  scatterBackground,
  themedData,
  themedLayout,
  useThemeStore,
} from "./theme";
import { useSettingsStore } from "@/stores/settingsStore";

beforeEach(() => {
  localStorage.clear();
  useThemeStore.setState({ preference: "system", systemDark: false });
  useSettingsStore.setState({ plotBackground: "white", plotBackgroundChosen: false });
  document.documentElement.classList.remove("dark");
});

describe("theme resolution", () => {
  it("System follows the OS setting, Light and Dark do not", () => {
    expect(resolveTheme("system", true)).toBe("dark");
    expect(resolveTheme("system", false)).toBe("light");
    expect(resolveTheme("light", true)).toBe("light");
    expect(resolveTheme("dark", false)).toBe("dark");
  });

  it("defaults to System, and ignores junk in storage", () => {
    expect(readStoredPreference()).toBe("system");
    localStorage.setItem(THEME_STORAGE_KEY, "purple");
    expect(readStoredPreference()).toBe("system");
  });

  it("persists the choice and reads it back", () => {
    useThemeStore.getState().setPreference("dark");
    expect(localStorage.getItem(THEME_STORAGE_KEY)).toBe("dark");
    expect(readStoredPreference()).toBe("dark");
    expect(currentTheme()).toBe("dark");
  });

  it("keeps working when storage throws", () => {
    const get = vi.spyOn(Storage.prototype, "getItem").mockImplementation(() => {
      throw new Error("blocked");
    });
    const set = vi.spyOn(Storage.prototype, "setItem").mockImplementation(() => {
      throw new Error("blocked");
    });
    expect(readStoredPreference()).toBe("system");
    expect(() => useThemeStore.getState().setPreference("light")).not.toThrow();
    expect(currentTheme()).toBe("light");
    get.mockRestore();
    set.mockRestore();
  });

  it("puts .dark and color-scheme on <html>, and follows store changes", () => {
    applyTheme("dark");
    expect(document.documentElement.classList.contains("dark")).toBe(true);
    expect(document.documentElement.style.colorScheme).toBe("dark");
    const stop = initTheme();
    expect(document.documentElement.classList.contains("dark")).toBe(false); // system, OS light
    useThemeStore.getState().setSystemDark(true);
    expect(document.documentElement.classList.contains("dark")).toBe(true);
    useThemeStore.getState().setPreference("light");
    expect(document.documentElement.classList.contains("dark")).toBe(false);
    stop();
  });
});

describe("scatter background", () => {
  it("follows the theme until the user picks one", () => {
    expect(scatterBackground(false, "white", "dark")).toBe("dark");
    expect(scatterBackground(false, "dark", "light")).toBe("white");
    expect(scatterBackground(true, "white", "dark")).toBe("white");
  });

  it("the toolbar choice is remembered, a theme change hands it back to the theme", () => {
    useSettingsStore.getState().setPlotBackground("white");
    expect(useSettingsStore.getState().plotBackgroundChosen).toBe(true);
    useThemeStore.getState().setPreference("dark");
    expect(useSettingsStore.getState().plotBackgroundChosen).toBe(false);
  });
});

describe("plotTheme", () => {
  it("gives dark paper, light text and dim grid in dark mode", () => {
    const d = plotTheme("dark");
    expect(d.paper).toBe(DARK_SURFACE);
    expect(d.text).toBe(DARK_SLATE[700]);
    expect(d.grid).toBe(DARK_SLATE[100]);
    expect(plotTheme("light").paper).toBe("#ffffff");
  });

  it("maps light neutrals to dark ones and leaves real colours alone", () => {
    expect(darkNeutral("#334155")).toBe(DARK_SLATE[700]);
    expect(darkNeutral("white")).toBe(DARK_SURFACE);
    expect(darkNeutral("#F1F5F9")).toBe(DARK_SLATE[100]);
    expect(darkNeutral("rgba(255,255,255,0.8)")).toBe("rgba(26, 36, 54, 0.8)");
    expect(darkNeutral("#d95f02")).toBe("#d95f02");
    expect(darkNeutral(undefined)).toBeUndefined();
  });
});

const LAYOUT: Partial<Layout> = {
  paper_bgcolor: "white",
  plot_bgcolor: "rgba(0,0,0,0)",
  font: { size: 11, color: "#334155" },
  xaxis: { gridcolor: "#f1f5f9", range: [0, 5], title: { text: "x" } },
  yaxis2: { tickfont: { color: "#64748b" } },
  legend: { bgcolor: "rgba(255,255,255,0.8)" },
  annotations: [{ text: "n = 5", font: { color: "#94a3b8" } }, { text: "hit", font: { color: "#dc2626" } }],
  shapes: [{ type: "line", line: { color: "#cbd5e1" } }],
};

describe("themedLayout", () => {
  it("returns the layout itself in light mode", () => {
    expect(themedLayout(LAYOUT, "light")).toBe(LAYOUT);
  });

  it("darkens backgrounds, text, grid, legend, annotations and shapes, without touching the input", () => {
    const before = JSON.stringify(LAYOUT);
    const d = themedLayout(LAYOUT, "dark");
    expect(JSON.stringify(LAYOUT)).toBe(before);
    expect(d.paper_bgcolor).toBe(DARK_SURFACE);
    expect(d.plot_bgcolor).toBe("rgba(0,0,0,0)"); // transparent stays transparent
    expect(d.font).toEqual({ size: 11, color: DARK_SLATE[700] });
    expect(d.xaxis!.gridcolor).toBe(plotTheme("dark").grid);
    expect(d.xaxis!.range).toEqual([0, 5]);
    expect(d.yaxis!.gridcolor).toBe(plotTheme("dark").grid); // default axes too
    expect(d.yaxis2!.tickfont!.color).toBe(DARK_SLATE[500]);
    expect(d.legend!.bgcolor).toBe("rgba(26, 36, 54, 0.8)");
    expect(d.annotations![0]!.font!.color).toBe(DARK_SLATE[400]);
    expect(d.annotations![1]!.font!.color).toBe("#dc2626");
    expect((d.shapes![0]!.line as { color: string }).color).toBe(DARK_SLATE[300]);
  });

  it("gives a plot with no layout a dark font", () => {
    expect(themedLayout({} as Partial<Layout>, "dark").font!.color).toBe(plotTheme("dark").text);
  });
});

describe("themedData", () => {
  const DATA = [
    { type: "violin", y: [1], line: { color: "#334155" }, marker: { color: "#1e293b" } },
    { type: "scatter", x: [1], marker: { color: "#cbd5e1", line: { color: "#ffffff" } }, textfont: { color: "#475569" } },
    { type: "bar", x: [1], marker: { color: "#d95f02" } },
  ] as Data[];

  it("is the identity in light mode", () => {
    expect(themedData(DATA, "light")).toBe(DATA);
  });

  it("flips dark neutral lines/markers and text, keeps light-grey markers and real colours", () => {
    const d = themedData(DATA, "dark") as unknown as Record<string, Record<string, unknown>>[];
    expect(d[0]!.line!.color).toBe(DARK_SLATE[700]);
    expect(d[0]!.marker!.color).toBe(DARK_SLATE[800]);
    expect(d[1]!.marker!.color).toBe("#cbd5e1");
    expect((d[1]!.marker!.line as { color: string }).color).toBe(DARK_SURFACE);
    expect(d[1]!.textfont!.color).toBe(DARK_SLATE[600]);
    expect(d[2]!.marker!.color).toBe("#d95f02");
    expect((DATA[0] as { line: { color: string } }).line.color).toBe("#334155");
  });
});

describe("exportFigure", () => {
  it("returns the graph div itself in light mode or when nothing was themed", () => {
    const gd = document.createElement("div") as unknown as PlotlyHTMLElement;
    expect(exportFigure(gd, "dark")).toBe(gd);
    rememberBaseFigure(gd as unknown as HTMLElement, [], LAYOUT);
    expect(exportFigure(gd, "light")).toBe(gd);
  });

  it("puts the light colours back on the plot's current state, at its on-screen size", () => {
    const data = [{ type: "violin", y: [1], line: { color: "#334155" } }] as Data[];
    const gd = document.createElement("div") as unknown as Record<string, unknown>;
    const dark = themedLayout(LAYOUT, "dark");
    // The user zoomed (Plotly writes the range into gd.layout) and hid a trace.
    gd.layout = { ...dark, xaxis: { ...dark.xaxis, range: [1, 2] } };
    gd.data = themedData(data, "dark").map((t) => ({ ...t, visible: "legendonly" }));
    gd._fullLayout = { width: 640, height: 300 };
    rememberBaseFigure(gd as unknown as HTMLElement, data, LAYOUT);
    const fig = exportFigure(gd as unknown as PlotlyHTMLElement, "dark") as { data: Record<string, unknown>[]; layout: Partial<Layout> };
    expect(fig.layout.paper_bgcolor).toBe("white");
    expect(fig.layout.font!.color).toBe("#334155");
    expect(fig.layout.xaxis!.gridcolor).toBe("#f1f5f9");
    expect(fig.layout.xaxis!.range).toEqual([1, 2]);
    expect(fig.layout.yaxis!.gridcolor).toBeUndefined(); // back to Plotly's default
    expect(fig.layout.annotations![0]!.font!.color).toBe("#94a3b8");
    expect(fig.layout.width).toBe(640);
    expect(fig.data[0]!.visible).toBe("legendonly");
    expect((fig.data[0]!.line as { color: string }).color).toBe("#334155");
  });
});

describe("index.css dark block", () => {
  it("uses the same white and slate values as lib/theme.ts", () => {
    const css = readFileSync(resolve(__dirname, "../index.css"), "utf8");
    const block = css.slice(css.indexOf(".dark {"));
    expect(block).toContain(`--color-white: ${DARK_SURFACE};`);
    for (const [shade, hex] of Object.entries(DARK_SLATE)) {
      expect(block).toContain(`--color-slate-${shade}: ${hex};`);
    }
  });
});

/** WCAG relative luminance contrast of two #rrggbb colours. */
function contrast(a: string, b: string): number {
  const lum = (h: string) => {
    const c = [1, 3, 5].map((i) => {
      const v = parseInt(h.slice(i, i + 2), 16) / 255;
      return v <= 0.03928 ? v / 12.92 : ((v + 0.055) / 1.055) ** 2.4;
    });
    return 0.2126 * c[0]! + 0.7152 * c[1]! + 0.0722 * c[2]!;
  };
  const [x, y] = [lum(a), lum(b)].sort((p, q) => q - p);
  return (x! + 0.05) / (y! + 0.05);
}

describe("dark theme contrast (WCAG AA 4.5:1)", () => {
  it("body text and captions on cards, page wells and hover rows", () => {
    for (const bg of [DARK_SURFACE, DARK_SLATE[50]!, DARK_SLATE[100]!]) {
      for (const shade of [400, 500, 600, 700, 800]) {
        expect(contrast(DARK_SLATE[shade]!, bg)).toBeGreaterThanOrEqual(4.5);
      }
    }
  });

  it("plot text and tick labels on the plot paper", () => {
    const t = plotTheme("dark");
    expect(contrast(t.text, t.paper)).toBeGreaterThanOrEqual(4.5);
    expect(contrast(t.muted, t.paper)).toBeGreaterThanOrEqual(4.5);
    expect(contrast(t.hoverText, t.hoverBg)).toBeGreaterThanOrEqual(4.5);
  });
});
