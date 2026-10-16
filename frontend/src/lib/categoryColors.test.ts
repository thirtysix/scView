import { describe, it, expect, beforeEach } from "vitest";
import {
  categoryColor,
  categorySlot,
  resetCategoryColors,
  rgbCss,
  sortByCategoryOrder,
} from "./categoryColors";
import { getPalette, mapCategoryToColor } from "./colors";

const ORDER = ["B cell", "Monocyte", "NK", "T cell"];

describe("category colours", () => {
  beforeEach(() => resetCategoryColors());

  it("matches the scatter: a category's colour is the palette entry at its code", () => {
    // The scatter colours code i (the category's index in the backend order)
    // with mapCategoryToColor(i); every panel must agree with it.
    for (const palette of ["cvd", "classic", "tableau20"] as const) {
      ORDER.forEach((cat, i) => {
        expect([...categoryColor("ds1", "cell_type", ORDER, cat, palette)]).toEqual(
          mapCategoryToColor(i, palette).slice(0, 3),
        );
      });
    }
  });

  it("does not depend on the order a panel lists the categories in", () => {
    const fromViolin = ["NK", "B cell"].map((c) => categoryColor("ds1", "cell_type", ORDER, c));
    const fromTable = ["T cell", "B cell", "NK"].map((c) => categoryColor("ds1", "cell_type", ORDER, c));
    expect(fromViolin[0]).toEqual(fromTable[2]); // NK
    expect(fromViolin[1]).toEqual(fromTable[1]); // B cell
  });

  it("keys the mapping by dataset and column", () => {
    const other = ["NK", "B cell"];
    expect(categorySlot("ds1", "cell_type", ORDER, "NK")).toBe(2);
    expect(categorySlot("ds2", "cell_type", other, "NK")).toBe(0);
    expect(categorySlot("ds1", "annot", other, "NK")).toBe(0);
    expect(categorySlot("ds1", "cell_type", ORDER, "NK")).toBe(2);
  });

  it("gives unlisted categories the next free slots, and keeps them", () => {
    expect(categorySlot("ds1", "cell_type", ORDER, "nan")).toBe(4);
    expect(categorySlot("ds1", "cell_type", ORDER, "Doublet")).toBe(5);
    expect(categorySlot("ds1", "cell_type", ORDER, "nan")).toBe(4);
    // A column with no listed order (over the backend's value cap) still agrees across panels.
    expect(categorySlot("ds1", "barcode", [], "AAAC")).toBe(0);
    expect(categorySlot("ds1", "barcode", [], "AAAG")).toBe(1);
    expect(categorySlot("ds1", "barcode", [], "AAAC")).toBe(0);
  });

  it("starts afresh when the column's order changes (a renamed category)", () => {
    expect(categorySlot("ds1", "cell_type", ORDER, "Mono")).toBe(4);
    const renamed = ["B cell", "Mono", "NK", "T cell"];
    expect(categorySlot("ds1", "cell_type", renamed, "Mono")).toBe(1);
  });

  it("wraps past the end of the palette", () => {
    const n = getPalette("cvd").length;
    const many = Array.from({ length: n + 1 }, (_, i) => `c${i}`);
    expect(categoryColor("ds1", "big", many, `c${n}`)).toEqual(categoryColor("ds1", "big", many, "c0"));
  });
});

describe("sortByCategoryOrder", () => {
  it("sorts into the column's order and keeps unlisted names last, in their order", () => {
    expect(sortByCategoryOrder(["T cell", "zz", "B cell", "aa", "NK"], ORDER)).toEqual([
      "B cell",
      "NK",
      "T cell",
      "zz",
      "aa",
    ]);
  });
});

describe("rgbCss", () => {
  it("formats rgb and rgba", () => {
    expect(rgbCss([1, 2, 3])).toBe("rgb(1,2,3)");
    expect(rgbCss([1, 2, 3], 0.5)).toBe("rgba(1,2,3,0.5)");
  });
});
