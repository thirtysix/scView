import { describe, expect, it } from "vitest";
import {
  DEFAULT_PANELS_PER_PAGE,
  clampPage,
  isPanelsPerPage,
  layoutCount,
  pageCount,
  pageLabel,
  pagePanelsText,
  pageRange,
  repage,
} from "./gridPaging";

describe("gridPaging", () => {
  it("defaults to six panels a page, the grids' look before paging", () => {
    expect(DEFAULT_PANELS_PER_PAGE).toBe(6);
  });

  it("counts pages, at least one", () => {
    expect(pageCount(24, 6)).toBe(4);
    expect(pageCount(13, 6)).toBe(3);
    expect(pageCount(6, 6)).toBe(1);
    expect(pageCount(0, 6)).toBe(1);
    expect(pageCount(24, "all")).toBe(1);
    expect(pageCount(24, 9)).toBe(3);
  });

  it("gives each page's panels, a short last page included", () => {
    expect(pageRange(13, 6, 0)).toEqual({ start: 0, end: 6 });
    expect(pageRange(13, 6, 1)).toEqual({ start: 6, end: 12 });
    expect(pageRange(13, 6, 2)).toEqual({ start: 12, end: 13 });
    expect(pageRange(24, "all", 3)).toEqual({ start: 0, end: 24 });
  });

  it("clamps a page past the end (a gene removed) or below 0", () => {
    expect(clampPage(5, 13, 6)).toBe(2);
    expect(clampPage(-1, 13, 6)).toBe(0);
    expect(clampPage(Number.NaN, 13, 6)).toBe(0);
    expect(pageRange(13, 6, 9)).toEqual({ start: 12, end: 13 });
  });

  it("lays out every page for a full page, or for all panels when fewer", () => {
    expect(layoutCount(13, 6)).toBe(6);
    expect(layoutCount(3, 6)).toBe(3);
    expect(layoutCount(24, "all")).toBe(24);
    expect(layoutCount(24, 12)).toBe(12);
  });

  it("keeps the current panels in view when the page size changes", () => {
    // Page 3 of 6 a page shows panels 13-18; at 4 a page panel 13 is on page 4 (index 3).
    expect(repage(2, 6, 4, 24)).toBe(3);
    // At 12 a page panels 13-18 sit on page 2 (index 1).
    expect(repage(2, 6, 12, 24)).toBe(1);
    expect(repage(3, 6, "all", 24)).toBe(0);
    expect(repage(0, "all", 6, 24)).toBe(0);
  });

  it("names the page and its panels for people (1-based)", () => {
    expect(pageLabel(1, 24, 6)).toBe("Page 2 of 4");
    expect(pagePanelsText(24, 6, 1)).toBe("Panels 7–12 of 24");
    expect(pagePanelsText(13, 6, 2)).toBe("Panel 13 of 13");
    expect(pagePanelsText(6, 6, 0)).toBe("");
  });

  it("accepts only the offered page sizes (session files)", () => {
    expect([4, 6, 9, 12, "all"].every(isPanelsPerPage)).toBe(true);
    expect([5, "6", null, 0, "some"].some(isPanelsPerPage)).toBe(false);
  });
});
