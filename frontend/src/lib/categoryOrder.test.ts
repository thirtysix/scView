import { describe, it, expect } from "vitest";
import { cellsInCategories, moveCategory, togglePicked } from "./categoryOrder";

describe("category order helpers", () => {
  it("moves one category, leaving the rest in order", () => {
    expect(moveCategory(["a", "b", "c", "d"], 3, 1)).toEqual(["a", "d", "b", "c"]);
    expect(moveCategory(["a", "b", "c", "d"], 0, 3)).toEqual(["b", "c", "d", "a"]);
    const same = ["a", "b"];
    expect(moveCategory(same, 1, 1)).toBe(same);
    expect(moveCategory(same, 0, 5)).toBe(same);
  });

  it("finds the cells of the picked categories by their colour codes", () => {
    const codes = Int32Array.from([0, 1, 2, 1, -1, 0]);
    expect([...cellsInCategories(codes, ["B", "NK", "T"], ["B", "T"])!].sort()).toEqual([0, 2, 5]);
    expect(cellsInCategories(codes, ["B", "NK", "T"], ["gone"])).toBeNull();
  });

  it("toggles a pick in or out", () => {
    expect(togglePicked([], "B")).toEqual(["B"]);
    expect(togglePicked(["B", "NK"], "B")).toEqual(["NK"]);
  });
});
