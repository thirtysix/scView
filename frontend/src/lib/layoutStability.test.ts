import { describe, expect, it } from "vitest";
import type { LayoutRelation, LayoutSummary } from "@/api/layouts";
import {
  contactTitle,
  countsSentence,
  drawThumbnail,
  fitBox,
  HALO,
  MUTED,
  plotSentence,
  relationSentence,
  relationVerdict,
  visibleRelations,
} from "./layoutStability";

const rel = (over: Partial<LayoutRelation>): LayoutRelation => ({
  a: "3",
  b: "7",
  n_a: 100,
  n_b: 80,
  layouts: [true, true, false, false, false, false, false, false, false],
  n_touch: 2,
  n_layouts: 9,
  share: 2 / 9,
  current: true,
  status: "some",
  flag: "rare_in_plot",
  contact_a: [],
  contact_b: [],
  current_contact: [6, 7],
  ...over,
});

describe("relation sentences", () => {
  it("names how often a contact in the plot appears", () => {
    expect(relationSentence(rel({}))).toBe("The contact between 3 and 7 in your plot appears in 2 of 9 layouts.");
    expect(relationVerdict(rel({}))).toBe("in your plot, rare");
  });

  it("says when the plot lacks a common contact, and the plain statuses", () => {
    expect(relationSentence(rel({ flag: "hidden_in_plot", current: false, n_touch: 7 }))).toBe(
      "3 and 7 touch in 7 of 9 layouts but are apart in your plot.",
    );
    expect(relationSentence(rel({ flag: null, status: "every", n_touch: 9 }))).toBe("3 and 7 touch in every layout.");
    expect(relationSentence(rel({ flag: null, status: "never", n_touch: 0 }))).toBe("3 and 7 touch in no layout.");
    expect(relationSentence(rel({ flag: null, n_touch: 5 }))).toBe("3 and 7 touch in 5 of 9 layouts.");
  });

  it("summarises the run and the plot", () => {
    const s = {
      n_pairs: 78,
      n_done: 9,
      current: "X_umap",
      counts: { every: 10, some: 8, never: 60, rare_in_plot: 1, hidden_in_plot: 3 },
    } as unknown as LayoutSummary;
    expect(countsSentence(s)).toBe("Of 78 pairs of groups, 10 touch in every layout, 8 in some and 60 in none.");
    expect(plotSentence(s)).toBe(
      "1 contact in your plot appears in fewer than half of the layouts; 3 pairs apart in your plot touch in more than half.",
    );
    expect(plotSentence({ ...s, counts: { every: 1 } })).toMatch(/^Every contact in your plot/);
    expect(plotSentence({ ...s, current: null })).toBeNull();
  });

  it("states the plot's contact cells as a share of each group", () => {
    expect(contactTitle(rel({}))).toBe(
      "Contact cells in your plot: 6 of 3's 100 cells (6.0%) and 7 of 7's 80 (8.8%)",
    );
    expect(contactTitle(rel({ current_contact: null }))).toBeNull();
  });

  it("lists the pairs that never touch only on request (unless they touch in the plot)", () => {
    const rows = [rel({}), rel({ a: "1", b: "2", status: "never", flag: null, current: false, n_touch: 0 })];
    expect(visibleRelations(rows, false).map((r) => r.a)).toEqual(["3"]);
    expect(visibleRelations(rows, true).map((r) => r.a)).toEqual(["3", "1"]);
  });
});

describe("thumbnails", () => {
  it("fits the data box into the canvas, aspect kept, y up", () => {
    const at = fitBox([0, 10], [0, 5], 110, 110, 5);
    expect(at(0, 0)).toEqual([5, 80]);
    expect(at(10, 5)).toEqual([105, 30]);
    expect(at(5, 2.5)).toEqual([55, 55]);
  });

  function fakeCtx() {
    const rects: { style: string; x: number; y: number; w: number }[] = [];
    let style = "";
    const ctx = {
      clearRect: () => undefined,
      fillRect: (x: number, y: number, w: number) => rects.push({ style, x, y, w }),
      set fillStyle(v: string) {
        style = v;
      },
      get fillStyle() {
        return style;
      },
    };
    return { ctx: ctx as unknown as CanvasRenderingContext2D, rects };
  }

  const thumb = { cells: [10, 11, 12, 13], x: [0, 1, 2, 3], y: [0, 1, 2, 3] };
  const codes = new Int32Array(20).fill(-1);
  codes.set([0, 1, 2, 1], 10);
  const colorOf = (c: number) => ([[255, 0, 0], [0, 255, 0], [0, 0, 255]][c] as [number, number, number]);

  it("draws each cell in its group's colour", () => {
    const { ctx, rects } = fakeCtx();
    const out = drawThumbnail(ctx, { thumb, codes, colorOf, width: 50, height: 50 });
    expect(out).toEqual({ drawn: 4, outlined: 0 });
    expect(rects.slice(1).map((r) => r.style)).toEqual([
      "rgba(255,0,0,0.85)",
      "rgba(0,255,0,0.85)",
      "rgba(0,0,255,0.85)",
      "rgba(0,255,0,0.85)",
    ]);
  });

  it("outlines the hovered groups on top of muted others", () => {
    const { ctx, rects } = fakeCtx();
    const out = drawThumbnail(ctx, { thumb, codes, colorOf, width: 50, height: 50, outline: new Set([1]) });
    expect(out).toEqual({ drawn: 4, outlined: 2 });
    const styles = rects.slice(1).map((r) => r.style);
    const muted = `rgba(${MUTED.join(",")},0.6)`;
    const halo = `rgb(${HALO.join(",")})`;
    // Two others muted, then a halo for each outlined cell, then their colour.
    expect(styles).toEqual([muted, muted, halo, halo, "rgb(0,255,0)", "rgb(0,255,0)"]);
    expect(rects[3]!.w).toBe(4); // the halo is 1 px wider each side
    expect(rects[5]!.w).toBe(2);
  });
});
