import { describe, expect, it } from "vitest";
import { readFileSync, readdirSync, statSync } from "node:fs";
import { join, relative, resolve } from "node:path";

/**
 * The app's Plotly bundle (lib/plotly.ts) registers bar, scatter, scattergl, box, violin and heatmap.
 * Any other trace type raises no error: Plotly draws something wrong. A "histogram" came out as one
 * horizontal bar per cell, which only a screenshot showed.
 */
const SRC = resolve(__dirname, "..");
const TRACE_RX = /\btype:\s*"(histogram\w*|contour\w*|pie|sunburst|treemap|sankey|scatter3d|surface|mesh3d|scattergeo|scatterpolar\w*|parcoords|funnel\w*|waterfall|candlestick|ohlc|image|table)"/g;

function sources(dir: string): string[] {
  return readdirSync(dir).flatMap((name) => {
    const p = join(dir, name);
    if (statSync(p).isDirectory()) return sources(p);
    return /\.tsx?$/.test(name) && !/\.test\.tsx?$/.test(name) ? [p] : [];
  });
}

describe("Plotly trace types", () => {
  it("uses only the types the bundle registers", () => {
    const found = sources(SRC).flatMap((p) => {
      const text = readFileSync(p, "utf8");
      return [...text.matchAll(TRACE_RX)]
        .filter((m) => !text.includes(`plotly.js/lib/${m[1]}`)) // registered by the file itself (the Sankey)
        .map((m) => `${relative(SRC, p)}: ${m[1]}`);
    });
    expect(found).toEqual([]);
  });
});
