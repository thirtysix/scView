import { describe, expect, it } from "vitest";
import { COUNT_W, UPSET_DEFAULTS, intersectionName, upsetLayout } from "./upset";

const SETS = [
  { name: "CD14 Mono", n: 1800 },
  { name: "CD4 T", n: 360 },
  { name: "B", n: 320 },
];
const COLS = [
  { sets: [0], n: 1400 },
  { sets: [0, 1, 2], n: 190 },
  { sets: [0, 2], n: 70 },
  { sets: [1], n: 0 },
];

describe("upsetLayout", () => {
  const L = upsetLayout(SETS, COLS);
  const o = UPSET_DEFAULTS;

  it("draws intersection bars in proportion to their size, on one baseline", () => {
    expect(L.bars.map((b) => b.h)).toEqual([o.barsH, (190 / 1400) * o.barsH, (70 / 1400) * o.barsH, 0]);
    for (const b of L.bars) expect(b.y + b.h).toBeCloseTo(L.barsBottom);
    // One column each, left to right, centred on the column.
    expect(L.bars.map((b) => b.x + b.w / 2)).toEqual(COLS.map((_, i) => L.matrixLeft + i * o.colW + o.colW / 2));
    expect(L.maxColumn).toBe(1400);
  });

  it("draws set-size bars in proportion, ending at the set names, with room for the count", () => {
    const full = o.setBarW - COUNT_W;
    expect(L.setBars.map((b) => b.w)).toEqual([full, (360 / 1800) * full, (320 / 1800) * full]);
    for (const b of L.setBars) expect(b.x + b.w).toBe(L.setBarsRight);
    expect(Math.min(...L.setBars.map((b) => b.x))).toBe(COUNT_W);
  });

  it("fills the dots of each intersection's sets, one row per set", () => {
    const filled = (col: number) => L.dots.filter((d) => d.column === col && d.filled).map((d) => d.set);
    expect(L.dots).toHaveLength(COLS.length * SETS.length);
    expect(filled(0)).toEqual([0]);
    expect(filled(1)).toEqual([0, 1, 2]);
    expect(filled(2)).toEqual([0, 2]);
    expect(L.rowY).toEqual(SETS.map((_, j) => L.matrixTop + j * o.rowH + o.rowH / 2));
    for (const d of L.dots) expect(d.cy).toBe(L.rowY[d.set]);
  });

  it("joins the filled dots of a multi-set intersection, top to bottom", () => {
    expect(L.lines.map((l) => l.column)).toEqual([1, 2]); // single sets get no line
    expect(L.lines[0]).toMatchObject({ y1: L.rowY[0], y2: L.rowY[2] });
    expect(L.lines[1]).toMatchObject({ y1: L.rowY[0], y2: L.rowY[2] });
  });

  it("sizes the drawing to its columns and rows", () => {
    expect(L.width).toBe(o.setBarW + o.nameW + COLS.length * o.colW + o.right);
    expect(L.height).toBe(o.top + o.barsH + o.gap + SETS.length * o.rowH + o.bottom);
  });

  it("copes with nothing to draw", () => {
    const E = upsetLayout(SETS, []);
    expect(E.bars).toEqual([]);
    expect(E.dots).toEqual([]);
    expect(upsetLayout([], []).setBars).toEqual([]);
  });
});

describe("intersectionName", () => {
  const names = ["A", "B", "C"];
  it("names one set, all sets, or the sets of the combination", () => {
    expect(intersectionName(names, [1])).toBe("only B");
    expect(intersectionName(names, [0, 1, 2])).toBe("all 3 sets");
    expect(intersectionName(names, [0, 2])).toBe("A and C only");
  });
});
