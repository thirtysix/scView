import { describe, it, expect } from "vitest";
import { followHighlight, highlightedMarkerGroup } from "./UnifiedMarkersSubtab";

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

describe("followHighlight", () => {
  const groupable = ["leiden", "cell_type", "stim"];

  it("opens a group of the markers' column", () => {
    expect(followHighlight({ column: "leiden", value: "1" }, "leiden", groupable, ["0", "1"]))
      .toEqual({ group: "1" });
  });

  it("switches to the highlight's own column (plot coloured by cell type)", () => {
    expect(followHighlight({ column: "cell_type", value: "NK" }, "leiden", groupable, ["0", "1"]))
      .toEqual({ column: "cell_type" });
  });

  it("leaves a column it cannot group by alone (too many groups, not categorical)", () => {
    expect(followHighlight({ column: "barcode", value: "AAC" }, "leiden", groupable, ["0"])).toBeNull();
  });

  it("waits for markers that are still loading, and ignores no highlight", () => {
    expect(followHighlight({ column: "leiden", value: "1" }, "leiden", groupable, null)).toBeNull();
    expect(followHighlight(null, "leiden", groupable, ["0"])).toBeNull();
  });
});
