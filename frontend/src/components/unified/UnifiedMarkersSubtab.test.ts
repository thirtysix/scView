import { describe, it, expect } from "vitest";
import { highlightedMarkerGroup } from "./UnifiedMarkersSubtab";

describe("highlightedMarkerGroup", () => {
  const groups = ["0", "1", "NK"];

  it("returns the highlighted group when it belongs to the markers' column", () => {
    expect(highlightedMarkerGroup(groups, "leiden", { column: "leiden", value: "NK" })).toBe("NK");
  });

  it("ignores a highlight on another column", () => {
    expect(highlightedMarkerGroup(groups, "leiden", { column: "cell_type", value: "NK" })).toBeNull();
  });

  it("ignores a value the markers have no group for, and no highlight", () => {
    expect(highlightedMarkerGroup(groups, "leiden", { column: "leiden", value: "7" })).toBeNull();
    expect(highlightedMarkerGroup(groups, "leiden", null)).toBeNull();
  });
});
